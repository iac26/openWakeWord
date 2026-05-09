# Architecture, findings, and what we changed

This document captures *why* the training pipeline looks the way it
does. It's a record of what we tried, what broke, and what the final
shape of the model and training loop converged to. Read it before
changing the loss, the head, or `auto_train`'s phase logic.

## The starting point

We forked [dscripka/openWakeWord](https://github.com/dscripka/openWakeWord)
v0.6.0. Upstream's recipe:

- Backbone: Google speech-embedding ONNX (frozen) producing one 96-dim
  feature per 80 ms frame.
- Classifier head: `Flatten(16×96) → Linear → Sigmoid` (`dnn`).
- Training: `auto_train` with a 3-phase schedule (full / refinement at
  LR×0.1 / fine-tune at LR×0.01), Adam, plain BCE.
- Validation: synthetic held-out positives + a long negative stream
  (`validation_set_features.npy`, ~11.3 hr).
- Best-checkpoint selection: average top-N checkpoints by
  Stochastic Weight Averaging (SWA, Izmailov et al. 2018) across **all**
  phases, score the average against the FP target.

Our first runs of this on "hey ari" produced models that scored ~0.88
recall in training but didn't fire on real audio when deployed. That
gap is what the rest of this doc is about.

## What we changed, and why

### 1. Classifier head: conv_attention instead of flat DNN

`Flatten(16×96)→Linear` collapses the 16 frames of temporal structure
in the embedding window. Single-word commands like "accept" / "decline"
are separated from their phonetic neighbours (`accent`, `incline`)
mostly by **fine-grained timing**, which the flat head cannot see.

Replaced with `conv_attention`:

```
x: (B, 16, 96)
  -> Linear(96 -> 128)
  -> 2x Conv1D(128, kernel=3) + BatchNorm1d
  -> MultiheadAttention(128, 4 heads), residual + LayerNorm
  -> mean-pool over time -> (B, 128)
  -> Linear(128 -> 1) -> Sigmoid
```

Reference: livekit-wakeword (Apache-2.0). On their published "hey
livekit" benchmark, conv_attention delivers ~60× lower AUT and ~100×
fewer FP/hr vs the flat DNN.

`layer_size=128` with `n_heads=4` is a good default; must be divisible
by `n_heads`.

### 2. Loss + regularisation

Switched from plain BCE + Adam to **focal + AdamW + embedding mixup +
label smoothing**:

```yaml
loss_type: focal
focal_gamma: 2.0
embedding_mixup: true
mixup_alpha: 0.2
label_smoothing: 0.05
weight_decay: 0.01
```

Direct port of the livekit-wakeword recipe. We briefly tested γ=1 with
mixup off in an attempt to push trained scores past ~0.6 (real-audio
scores were sitting low). User feedback after that run: *"the result
is genuinely bad."* Reverted to the upstream recipe. Lesson: if you
want sharper scores at inference time, use **temperature scaling**, not
a weaker loss.

### 3. Custom negatives: near vs. far

Initial config followed the intuition "block phonetic neighbours" and
listed `hey carl`, `hey call`, etc. as `custom_negative_phrases`.

This is wrong. Source: openWakeWord upstream evaluations
(Neural Engineer / dscripka). Verbatim guidance: *"training on
phonetically similar phrases like 'hey call' or 'hey carl' actually
hurts performance, and only clearly different phrases like 'hello',
'hey siri', 'alexa' should be used."*

The reason: phonetic near-rhymes are already produced automatically by
`generate_adversarial_texts`. Adding them again as custom negatives
double-counts that pressure and over-trains rejection of near-misses,
which causes the model to under-fire on the real word. That's how
`hey jealous` ended up in the official `hey_jarvis` training set —
synthetic, not curated.

Final lists (see `configs/*.yml`):

- `hey_ari`: clearly different wake words (`hello`, `hey siri`,
  `alexa`, `ok google`, `hey jarvis`, ...).
- `accept` / `decline`: common short responses (`hello`, `yes`, `no`,
  `okay`, `cancel`, `stop`, `go back`) **plus** the *other* command
  word, so the two confirmation models don't fire on each other.

### 4. Negative pressure: 1500 → 750, FP target 0.2 → 1.0

`auto_train` ramps the negative-class loss weight from 1 to
`max_negative_weight` over phase 1, then doubles it each subsequent
phase if FP/hr is still above target. With upstream's 1500 + the
old-style adversarial generator (which included input words) the
gradient was so heavily skewed against negatives that the model
under-fired on the real phrase.

Loosened: `max_negative_weight: 750`, `target_false_positives_per_hour:
1.0`. Focal loss handles the tail.

### 5. Inference temperature scaling

Added post-hoc temperature scaling baked into the exported ONNX:

    s' = sigmoid(T · logit(s))

`T > 1` sharpens: 0.5 stays fixed, scores above 0.5 climb toward 1,
scores below 0.5 collapse toward 0.

We initially set `T=11.0` to make synthetic-positive scores
(median ~0.7) read as close-to-1 in logs. Real-world deployment
result: the model never fired, because real-audio "hey ari" scores
sit *below* 0.5 (synthetic train/test data is cleaner than real
microphone capture). Cranking T pushed them to 0 instead of 1.

Final default: `inference_temperature: 1.0` (no-op). Pick T from the
post-training threshold sweep + observed real-audio score
distribution, not blindly.

## Bugs found in upstream `auto_train`

These are what closed the 30–60× recall mismatch between training-time
"recall=0.88" and post-selection "recall=0.51 / model doesn't fire."

### A. torchmetrics state accumulated across batches

`torchmetrics.Recall` and `Accuracy` are *stateful accumulators*. The
upstream loop reused the same instances across the train loop, the val
loop, and the post-selection scoring loop without `.reset()`, so
"validation recall" was actually "recall over every example seen since
the start of training." Fix: `self.recall.reset()` /
`self.accuracy.reset()` before each call (commit 2b9c665).

### B. BatchNorm ran in train mode during validation

`Model.forward` was called inside the val block without first putting
the module in `eval()` mode. BatchNorm therefore used the **current
batch's** mean/variance (train mode) instead of the **running** stats.
Validation recall on a tiny batch was effectively meaningless. Fix:
`model.eval()` / `model.train()` toggle around the val block with
try/finally (commit 619208d).

### C. `_select_best_model` deepcopied checkpoints in train mode

After scoring the SWA average and per-checkpoint candidates, the
"final scoring" pass was hitting `forward()` on freshly-deepcopied
modules in train mode (same BN issue). Fix: explicit `model.eval()`
loop in `_select_best_model` before scoring any candidate (commit
a256c74).

### D. Gradient accumulation lost sub-batch gradients

The training inner loop accumulated predictions until `accumulated_samples
>= 128` and then ran `optimizer.step()`. But `loss.backward()` only ran
on the **last** sub-batch — earlier sub-batches' gradients were
silently discarded. Fix:

```python
loss.backward()                       # accumulate every sub-batch
if accumulated_samples < 128:
    accumulation_steps += 1
    accumulated_predictions = torch.cat(...)
else:
    if accumulation_steps > 1:
        for p in self.model.parameters():
            if p.grad is not None:
                p.grad.data.div_(accumulation_steps)
    self.optimizer.step()
    self.optimizer.zero_grad()
```

(commit f09e29a). After this fix, saved-checkpoint val recall and
post-selection val recall agreed exactly to 4 decimals on the smoke
test (0.5100 / 0.5100). Before the fix, the same run reported 0.8820 /
0.6420 — the difference between "model that fires on real audio" and
"model that never fires."

### E. `val_set_hrs` hardcoded to 11.3

`auto_train` had `val_set_hrs=11.3` as a default kwarg, encoding the
length of the upstream `validation_set_features.npy`. Any other FP
stream produced wrong FP/hr. Fix: compute dynamically from feature
count, `val_set_hrs = X_val_fp.shape[0] * 0.08 / 3600.0`, and require
the caller to pass it.

### F. Phase-1 checkpoints in the SWA pool

SWA only makes sense **within the same LR regime** (Izmailov et al.
2018). The upstream `_select_best_model` pooled checkpoints across all
3 phases for averaging, which means an average between a
high-LR full-phase checkpoint and a low-LR fine-tune checkpoint —
two fundamentally different points in weight space. Fix:
`save_checkpoints_to_pool=False` for phase 1, `True` for phases 2–3.
The averaged model now sits in the same LR neighbourhood.

### G. Lazy imports for non-training paths

`mit_rirs/` was loaded at startup even when only `--train_model` was
passed. Same for `piper-sample-generator`. Both moved into their
respective `--augment_clips` / `--generate_clips` branches so users
who already have computed `.npy` features can train on a host without
those datasets.

### H. Silent stop after fallback warning

When no checkpoint hit the FP target, `_select_best_model` printed
*"No checkpoints met FP/hr <= ..."* and then stopped emitting any
output. The training run looked dead but was actually still doing the
post-selection val sweep silently. Fix: explicit
`logging.basicConfig(level=INFO, force=True)`,
line-buffered stderr, tqdm bars on the post-selection loops, and an
explicit fallback to highest-recall checkpoint when the FP target is
unreachable.

## Verification: smoke test discipline

After every `auto_train` change, the smoke-test run must satisfy:

> saved val_recall (printed mid-train) **==**
> post-selection val_recall (printed at end)

to ~3 decimal places. If those two numbers disagree, one of bugs
A–D has regressed and the model will not behave on real audio.

## Result: v5

Final cloud retrain on 100 k positives + 200 k+ adversarial negatives,
3-phase auto_train, 100 k base-LR steps, conv_attention head:

```
recall:                0.8850
FP/hr:                 0.4674   (5 FPs over 11.30 hr)
positive median score: 0.97
fp_windows p90 score:  0.02
```

Production-quality. Confirmed firing on real "hey ari" audio with
threshold 0.5 once the deployment audio contract was correct (see
below).

## Lesson: "score = 0 on real audio" was a deployment-pipeline bug

After the cloud retrain, the deployed inference engine reported
near-zero scores (mostly 0, occasionally 0.15) on real "hey ari"
attempts, while v5 scored 0.97 median on synthetic test data. The
initial speculation was that this was an OOD shift between
synthetic-Piper training data and real microphone audio — i.e. the
model had overfit the synthetic distribution. We even sketched a
smaller-capacity variant (`layer_size=64`, `n_conv=1`) on that
hypothesis.

The actual cause: the inference engine was passing **`float32`** PCM
samples to `Model.predict()` instead of **`int16`**. The pipeline
enforces int16 at `openwakeword/utils.py:_get_melspectrogram` (raises
`ValueError` if the dtype is wrong), but the deployment path was
casting samples to a numerically-int16-compatible representation that
the type check accepted while the underlying buffer's range or scaling
was wrong, so the embedding model received numerically nonsensical
input and produced features in a region the head had been trained to
reject. Hence "confidently zero" — not "uncertain" — on real audio.

Two takeaways:

1. **Score ≈ 0 on a confident model is almost always a pre-embedding
   contract violation, not an OOD shift.** A genuine OOD shift
   produces *uncertain* scores (0.1–0.4 range), not saturated zeros.
   If the head is confidently rejecting real positives, the input it
   sees is not the input you think it sees.
2. **Write down the audio contract before debugging the model.**
   `docs/inference_audio_contract.md` exists for this reason. Hand it
   to whoever owns the inference engine. The diagnostic in §4 of that
   doc — capture a real attempt to a WAV, score it via
   `Model.predict_clip()`, and compare to live deployment scores on
   the same utterance — would have isolated the bug in 5 minutes
   instead of a day of model-side speculation.

## Things still open

- **Verifier stage.** `configs/{hey_ari,accept,decline}_verifier.yml`
  document the intended config for a second-stage verifier (small MLP
  over mean-pooled embeddings, trained on real positives + mined hard
  negatives, fused into the ONNX via an If-node like
  `hey_jarvis_v0.1.onnx`). Not yet runnable — the training and
  fusion scripts don't exist. Blocked on real-audio data collection.
- **Cross-pollution between `accept` and `decline`.** Each lists the
  other in `custom_negative_phrases`, but if both models are loaded
  simultaneously on-device, deployment-time hard-negative mining
  (capture clips where one model fires while the user said the other)
  will likely be more effective than synthetic separation alone.
- **Real-audio FP/hr.** Current FP/hr is measured on the upstream
  `validation_set_features.npy` (~11.3 hr of mixed speech/music).
  Deployment audio (target environments, target microphone) is a
  different distribution; expected FP rate there is unknown until we
  capture and score deployment audio in bulk.
