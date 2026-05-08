# openWakeWord — Ari fork

This is a fork of [dscripka/openWakeWord](https://github.com/dscripka/openWakeWord),
re-tuned and re-architected to train high-quality wake-word and command-word
models for the Ari voice assistant. We ship three models trained from this
tree:

| Model        | Phrase     | Role                                  |
| ------------ | ---------- | ------------------------------------- |
| `hey_ari`    | "hey ari"  | Primary wake word                     |
| `accept`     | "accept"   | Confirmation after wake fires         |
| `decline`    | "decline"  | Rejection after wake fires            |

Inference at runtime is unchanged from upstream: 16 kHz mono PCM in, a
score per 80 ms frame out, sliding 16-frame buffer over the embedding
stream. Real-time inference is always batch=1.

## What's different from upstream

- **Single-purpose training pipeline.** Notebooks are gone. The flow is
  three CLI steps: `install_training.sh`, `prepare_datasets.py`,
  `python -m openwakeword.train --training_config ...`.
- **conv_attention classifier head** (Conv1D + multi-head self-attention +
  mean pool over time) instead of upstream's flat
  `Flatten(16×96)→Linear` DNN. Reference: livekit-wakeword (Apache-2.0).
  ~60× lower AUT and ~100× fewer FP/hr on their published "hey livekit"
  benchmark.
- **Loss + regularisation:** focal loss (γ=2.0), embedding mixup
  (α=0.2), label smoothing (ε=0.05), AdamW with weight_decay=0.01.
- **Phase-aware checkpoint pool for SWA.** Phase-1 checkpoints are
  excluded from averaging because they sit in a different LR regime
  (averaging across LR regimes is meaningless).
- **Post-hoc temperature scaling** baked into the exported ONNX
  (`s' = sigmoid(T · logit(s))`) — sharpen scores without changing
  training dynamics. Default `T=1.0` (no-op).
- **Bug fixes** to upstream's training loop: gradient accumulation,
  validation-time BatchNorm mode, torchmetrics state reset between
  evals, `val_set_hrs` derived from feature count rather than hardcoded
  to 11.3. See [docs/architecture.md](docs/architecture.md) for the
  trail and the impact of each fix.
- **ONNX-only.** Upstream supports both ONNX and TFLite; this fork
  drops TFLite. Exported models use a dynamic batch axis, which means
  the same `.onnx` file can be used for batch=1 streaming and batched
  offline evaluation.

## Quick links

- [docs/training.md](docs/training.md) — how to train a new model from scratch
- [docs/evaluation.md](docs/evaluation.md) — `scripts/eval_onnx.py`, threshold sweep, real-time inference shape
- [docs/architecture.md](docs/architecture.md) — what we found, what we changed, and why

## Inference (unchanged from upstream)

```python
import openwakeword
from openwakeword.model import Model

model = Model(wakeword_models=["path/to/hey_ari.onnx"])

# 16 kHz, 16-bit mono PCM, framed at multiples of 80 ms.
frame = my_function_to_get_audio_frame()
prediction = model.predict(frame)
```

The model emits one score per 80 ms frame. The on-device loop is
effectively:

```
every 80 ms:
    new_audio_chunk -> embedding_model -> 1 new (96,) feature row
    push into a 16-frame ring buffer
    score = wake_word_model([buffer])   # shape (1, 16, 96), batch=1
    if score >= threshold: trigger
```

## Training a new model

```bash
./install_training.sh                          # one-time: venv + torch + datasets
python prepare_datasets.py                     # one-time: AudioSet/MIT-RIRs/FMA
./run_train.sh --training_config examples/hey_ari.yml \
               --generate_clips --augment_clips --train_model
```

Outputs `training_workspace/output/<model_name>.onnx`. See
[docs/training.md](docs/training.md) for the full walkthrough and what
each config knob does.

## Evaluation

```bash
python scripts/eval_onnx.py \
    --onnx training_workspace/output/hey_ari.onnx \
    --pos_features training_workspace/output/hey_ari/positive_features_test.npy \
    --neg_features training_workspace/output/hey_ari/negative_features_test.npy \
    --fp_features training_workspace/features/validation_set_features.npy
```

Reports recall, FP/hr, balanced accuracy at the chosen threshold, plus
a threshold sweep at the configured FP-per-hour target. See
[docs/evaluation.md](docs/evaluation.md).

## Repository layout

```
examples/
    hey_ari.yml, accept.yml, decline.yml          training configs (primary models)
    hey_ari_verifier.yml, ...                     verifier scaffolding (NOT runnable yet)
    detect_from_microphone.py, capture_activations.py, ...
openwakeword/
    train.py            auto_train pipeline (Model class, _select_best_model, export_model)
    model.py            inference-side Model class
    data.py             feature extraction, mixup, augment
    metrics.py          recall/accuracy/FP-counter
    custom_verifier_model.py
scripts/
    eval_onnx.py        local ONNX benchmark (recall/FP/hr/threshold sweep)
docs/
    training.md, evaluation.md, architecture.md
    custom_verifier_models.md, synthetic_data_generation.md   (legacy, kept)
install_training.sh     bootstrap venv + CUDA torch + piper-sample-generator
prepare_datasets.py     download MIT-RIRs / AudioSet / FMA
run_train.sh            wrapper exposing torch's bundled CUDA libs
```

## License

Code: Apache 2.0 (unchanged). Pre-trained models inherit the upstream
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)
license because they include data from sources with restrictive or
unknown licensing. Models you train from this tree on your own data are
yours to license.

## Upstream credit

The shared melspectrogram + speech-embedding backbone, the synthetic
data generation approach, the `auto_train` skeleton, and the original
README content are all from
[dscripka/openWakeWord](https://github.com/dscripka/openWakeWord). The
conv_attention head and focal+mixup recipe come from
[livekit-wakeword](https://github.com/livekit/livekit) (Apache-2.0).
