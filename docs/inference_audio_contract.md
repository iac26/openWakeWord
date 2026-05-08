# Audio input contract for the wake-word ONNX models

This document specifies exactly what the deployed inference engine must
feed into the openWakeWord pipeline. Anything that deviates from this
spec will produce wrong scores — typically scores collapsed near 0,
because the embedding model produces features outside the region the
wake-word head was trained on.

Use this as a checklist when reviewing a deployment pipeline.

## TL;DR

```
Mic capture
  -> 16 kHz, mono, signed 16-bit little-endian PCM
  -> NO upstream noise suppression, NO AGC, NO codec, NO resampling at the model boundary
  -> int16 numpy array, shape (N,) where N is a multiple of 1280 samples (= 80 ms)
  -> Model.predict(int16_array)
  -> per-frame score in [0, 1]
```

If your pipeline fails to satisfy any line of the spec below, fix that
before doing anything else.

---

## 1. Sample format (hard requirements)

| Property        | Required value                          | Why                                      |
| --------------- | --------------------------------------- | ---------------------------------------- |
| Sample rate     | **16 000 Hz**                           | Mel filterbank assumes 16 kHz            |
| Channels        | **1 (mono)**                            | Pipeline is mono throughout              |
| Sample type     | **signed 16-bit integer (`int16`)**     | Enforced by `_get_melspectrogram`        |
| Endianness      | **little-endian** (`S16_LE`)            | Native numpy int16 on x86/ARM            |
| Range           | **[-32768, 32767]**                     | Standard 16-bit PCM range                |
| Layout          | Contiguous 1-D array, shape `(N,)`      | A 2-D `(1, N)` is also accepted          |

The pipeline **enforces** int16 at the entry point:

```python
# openwakeword/utils.py:120
if x.dtype != np.int16:
    raise ValueError("Input data must be 16-bit integers (i.e., 16-bit PCM audio).")
```

If you pass `float32` in `[-1.0, 1.0]` (a common DSP convention) without
multiplying by 32767 and casting to `int16` first, the call will throw.
If you pass `float32` in `[-32768, 32767]` (numerically int16 but typed
float), it will also throw. The cast must happen on your side.

## 2. Frame size and cadence

The wake-word head consumes a sliding window of **16 embedding frames**.
One embedding frame is computed from approximately 80 ms of audio. The
recommended chunk size into `Model.predict()` is therefore:

```
chunk_size = 1280 samples = 80 ms at 16 kHz
```

Larger chunks are accepted as long as they are integer multiples of
1280 samples (160 ms = 2560 samples, 240 ms = 3840 samples, etc.).
Larger chunks improve throughput at the cost of detection latency.
**Sub-1280-sample chunks must be buffered by the caller** until 1280
samples are available — `predict()` returns the *previous* score when
fewer than 1280 new samples are buffered.

The internal flow per `predict()` call:

```
new int16 chunk (>= 1280 samples)
  -> melspec ONNX                       -> (n_mel_frames, 32) float32
  -> Google speech_embedding ONNX       -> (n_emb_frames, 96) float32
  -> push into 16-frame embedding ring buffer
  -> wake-word ONNX, input shape (1, 16, 96) float32
  -> sigmoid output (1,) in [0, 1]      = the per-frame score
```

The 16-frame buffer represents **1.28 seconds of context** — that's the
amount of audio the head sees per inference. The first 5 calls after
model construction return 0 by design (`Model.predict` zero-pads while
the ring buffer fills).

## 3. What must NOT be in the path

These are the most common deployment bugs that produce score ≈ 0 on
real wake-word attempts:

### 3.1 No upstream noise suppression at the model boundary

Speex NS, WebRTC NS, RNNoise, NN-based denoisers — any of these,
applied to the audio *before* it reaches `Model.predict()`, will eat
consonants and breathy onsets (`/h/`, `/r/`, `/s/`). The training set
does NOT include consonant-eaten speech, so the embeddings of denoised
speech sit in a region the head learned to reject.

If your platform requires NS for other reasons (e.g. ASR downstream),
**branch the audio**: send the raw mic stream to the wake-word model
and the NS'd stream to ASR.

(openWakeWord *does* expose `enable_speex_noise_suppression` on the
`Model` class, but that's a model-internal toggle that uses a specific
upstream Speex configuration that was tested at training time. It is
NOT equivalent to an arbitrary platform NS stage.)

### 3.2 No AGC / dynamic range compression

Hardware or software AGC that boosts gain on quiet input and ducks on
loud input changes the spectral envelope frame-to-frame. The melspec
is computed on a 76-frame window (~600 ms); AGC moving within that
window distorts what the embedding sees.

Disable AGC on the capture device. Set a fixed gain that places typical
voice at -10 to -20 dBFS peak.

### 3.3 No codec passes

Bluetooth SCO (8 kHz mu-law), G.711, low-bitrate Opus, GSM — anything
that band-limits or quantises before `Model.predict()`. Even if the
codec output is upsampled back to 16 kHz, the high-frequency content
is permanently lost and the embedding sees a different spectral
distribution than training.

If the mic comes in over Bluetooth A2DP / SCO, capture from a USB or
analogue path instead, or accept that you'll need to retrain on
codec-augmented data.

### 3.4 No double resampling

Common bug: capture at 48 kHz, resample to 16 kHz, then a downstream
stage resamples to 16 kHz *again* (e.g. a soundcard pipe set to
44.1 kHz, then a software resampler). Bad resamplers introduce
aliasing or low-pass artefacts. Capture at native 16 kHz when
possible; if you must resample, use a high-quality polyphase resampler
(e.g. `scipy.signal.resample_poly`, `soxr`) and do it exactly once.

### 3.5 No stereo→mono accidents

If the device is stereo, mix to mono **by averaging** (`(L + R) / 2`),
not by selecting one channel. Single-channel selection biases the
spatial pickup pattern; averaging reproduces the omni response the
training data assumes.

### 3.6 No DC offset, no clipping

DC offset shifts the melspec floor and pushes embeddings into a corner
of feature space the head has rarely seen. Clipping (peaks at ±32767)
distorts harmonics and produces broadband artefacts the embedding
cannot see through.

Set capture gain so peaks are below -3 dBFS. If the mic has a
significant DC bias, high-pass at ~80 Hz before passing to the
pipeline.

### 3.7 No frame reordering

The 16-frame buffer is **time-ordered, oldest first**. If your
inference engine implements its own sliding buffer (e.g. for batched
or parallel calls), confirm the order: index 0 = oldest frame, index
15 = most recent frame. Time-reversed audio is acoustically meaningless
to the head and will score near 0.

### 3.8 No int16 misinterpretation

Common foot-guns:

| Bug                                                       | Symptom                                  |
| --------------------------------------------------------- | ---------------------------------------- |
| Passing `float32` in `[-1, 1]`                            | `ValueError` at `_get_melspectrogram`    |
| Passing `float32` in `[-32768, 32767]` without dtype cast | Same `ValueError`                        |
| Passing `int32` (e.g. from a 24-bit-in-32-bit ALSA path)  | Embedding shifted; scores ~0             |
| Passing `uint8` (lossy conversion)                        | Scores ~0                                |
| Passing big-endian int16 on a little-endian host          | Sample values garbled; scores ~0         |
| Passing samples scaled to `[-1.0, 1.0]` then `astype(int16)` | All samples become 0 or ±1; total silence |

The correct conversion from float `[-1.0, 1.0]`:

```python
audio_int16 = (audio_float32 * 32767.0).clip(-32768, 32767).astype(np.int16)
```

## 4. The diagnostic that catches all the above

For any audio captured from your deployed pipeline, run:

```python
from openwakeword.model import Model
import numpy as np
import wave

m = Model(wakeword_models=["v5/hey-ari.onnx"])

with wave.open("captured_attempt.wav", "rb") as w:
    assert w.getframerate() == 16000, f"sample rate is {w.getframerate()}, must be 16000"
    assert w.getnchannels() == 1,     f"channels is {w.getnchannels()}, must be 1"
    assert w.getsampwidth() == 2,     f"sample width is {w.getsampwidth()*8}-bit, must be 16-bit"
    pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)

# Score in 80 ms increments
scores = []
for i in range(0, len(pcm), 1280):
    chunk = pcm[i:i+1280]
    if len(chunk) < 1280:
        break
    s = m.predict(chunk)
    scores.append(s["hey_ari"])

print(f"max={max(scores):.3f}  mean={np.mean(scores):.3f}")
```

- If the captured WAV scores **>= 0.5** at peak: the model itself is
  fine; the deployment chain between the mic and `Model.predict()` is
  the problem. Compare your live pipeline's int16 buffer against the
  WAV's bytes — they must be identical for the same utterance.
- If the captured WAV scores **< 0.3** at peak: the audio reaching the
  WAV writer is already degraded. Move the WAV capture earlier in the
  pipeline (closer to the mic) and repeat.

Walking the WAV-capture point progressively closer to the mic until
scores rise tells you exactly which stage is mangling the audio.

## 5. Reference implementation

The canonical reference is `openwakeword/model.py:Model.predict` and
`openwakeword/utils.py:AudioFeatures._get_melspectrogram`. If your
inference engine is in a different language (C++, Rust, JS), the
behaviour of these two functions is what you must reproduce — same
int16 input expectation, same `melspec_transform = x/10 + 2`, same
sliding 16-frame buffer over embeddings.

## 6. Additional fork-specific notes

- The wake-word ONNXs in this fork (`hey_ari.onnx`, `accept.onnx`,
  `decline.onnx`) were exported with a **dynamic batch axis**. For
  real-time inference the call site always uses batch=1
  (`shape (1, 16, 96)`). The dynamic axis just means the same file can
  also be used for batched offline evaluation; it does not change
  streaming behaviour.
- Inference temperature is baked into the exported ONNX as
  `s' = sigmoid(T · logit(s))` and is currently `T = 1.0` (no-op).
  If a future export uses `T != 1.0`, scores will be sharper or softer
  but the **input contract above is unchanged**.
- The default activation threshold is `0.5`, but the right deployment
  threshold should be picked from the training-time threshold sweep
  (see `docs/evaluation.md`) plus a real-audio score histogram
  collected via the diagnostic in section 4.
