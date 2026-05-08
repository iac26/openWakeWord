# Training a wake-word model

End-to-end pipeline for training one of the Ari wake-word /
command-word models (`hey_ari`, `accept`, `decline`) or your own. Three
CLI steps, one YAML config per model.

## Prerequisites

- Linux x86_64, NVIDIA GPU with CUDA, ~80 GB free disk for datasets.
- Python 3.12 (older `>=3.10` works but the lockfile pins 3.12).
- ~100 GB scratch under `./training_workspace/` (datasets + features).

## Step 1 — Bootstrap the environment

```bash
./install_training.sh
```

What it does:
- creates `.venv-train/` with `torch<2.6` + CUDA libs (`cu124` by default),
- installs the `[training]` extras from `pyproject.toml`,
- clones `piper-sample-generator` into `training_workspace/piper-sample-generator/`,
- pre-downloads the openWakeWord melspectrogram + embedding ONNX models,
- pre-downloads the openWakeWord precomputed feature shards
  (`openwakeword_features_ACAV100M_2000_hrs_16bit.npy`,
  `validation_set_features.npy`).

Env knobs (override before invoking the script):

| var                  | default       | meaning                                              |
| -------------------- | ------------- | ---------------------------------------------------- |
| `PYTHON`             | `python3.12`  | interpreter to use                                   |
| `CUDA`               | `cu124`       | torch CUDA wheel suffix                              |
| `WORKDIR`            | `$PWD/training_workspace` | where features + clips land               |
| `DOWNLOAD_DATASETS`  | `0`           | also download AudioSet/FMA via `prepare_datasets.py` |

Everything `install_training.sh` does is idempotent — re-running is
safe.

## Step 2 — Prepare datasets

```bash
python prepare_datasets.py             # default: 1 AudioSet shard, 1 hr FMA
python prepare_datasets.py --help      # see all flags
```

Downloads and converts to 16 kHz wav under `training_workspace/`:

- `mit_rirs/` — MIT room-impulse-response set (used as RIR augmentation).
- `audioset_16k/` — AudioSet shards (background noise / music). One
  `bal_train09.tar` shard ≈ 5 hr; bump `AUDIOSET_SHARDS` for more.
- `fma/` — FMA "small" subset music clips. Skip with `--skip-fma`.

If `--skip-fma` is used, do not list `./training_workspace/fma` under
`background_paths` in your config, and leave
`background_paths_duplication_rate: [1]`.

The validation feature stream
(`training_workspace/features/validation_set_features.npy`) is what
`scripts/eval_onnx.py` and `auto_train`'s FP/hr metric measure
against. It is one continuous (~11 hr) negative-only feature stream;
one row = one 80 ms frame.

## Step 3 — Run training

```bash
./run_train.sh \
    --training_config examples/hey_ari.yml \
    --generate_clips \
    --augment_clips \
    --train_model
```

Each flag is independent and can be run separately:

| flag             | what it does                                                                |
| ---------------- | --------------------------------------------------------------------------- |
| `--generate_clips` | Synthesise `n_samples` positive WAVs with Piper TTS, plus adversarial      |
|                  | negatives via `generate_adversarial_texts`. Writes to                       |
|                  | `output_dir/<model_name>/{positive,negative}_{train,test}/`.                |
| `--augment_clips`  | Mix positives/negatives with RIRs and background noise, run them through   |
|                  | the openWakeWord embedding model, save windows as `.npy` features.          |
|                  | Skipped if features exist (use `--overwrite` to force).                     |
|                  | RIR/background paths are only loaded when this flag is set.                 |
| `--train_model`    | Run `auto_train` (3-phase schedule), select best checkpoint, export ONNX.  |

`run_train.sh` is just `python -m openwakeword.train` with `LD_LIBRARY_PATH`
pointed at torch's bundled CUDA libs. Use it instead of bare Python.

## Anatomy of a training config

The shipped configs (`examples/hey_ari.yml`, `accept.yml`, `decline.yml`)
are heavily commented; this section explains the *why* behind each
group of knobs.

### Phrase + adversarial negatives

```yaml
target_phrase:
  - "hey ari"

custom_negative_phrases:
  - "hello"
  - "hey siri"
  - ...
```

`target_phrase` is the list of phrases Piper synthesises positives for
(`n_samples` is split evenly across the list). For multi-variant wake
words use multiple entries; for `hey_ari` we ship one phrase only.

`custom_negative_phrases` should be **clearly different** from the
target phrase. Phonetic neighbours (`hey carl`, `incline` for
`decline`) are produced automatically by
`generate_adversarial_texts` — adding them here as well over-trains
rejection of near-misses and causes the model to under-fire on the real
word. Source: openWakeWord upstream guidance from dscripka. We
relearned this the hard way; see
[architecture.md](architecture.md#custom-negatives-near-vs-far).

### Sample counts

```yaml
n_samples: 100000        # synthetic positives (split across target_phrase)
n_samples_val: 2000      # synthetic positives held out for validation
```

100 k matches upstream's `alexa` / `hey_mycroft` per-phrase scale. Below
~30 k recall noticeably degrades.

### Batch sizes

```yaml
tts_batch_size: 256          # piper synthesis
augmentation_batch_size: 16  # MUST stay 16 — see below
```

`augmentation_batch_size` must stay at 16. `torch_audiomentations` uses
`mode="per_batch"` for its augment chain (one set of params per batch),
so larger batches reduce augmentation diversity. Per-clip CPU prep is
already pipelined via DataLoader workers, so 16 is not a speed
bottleneck.

### Per-step gradient batches

```yaml
batch_n_per_class:
  ACAV100M_sample: 1024       # negatives drawn from precomputed feature shards
  adversarial_negative: 50    # adversarial-text negatives
  positive: 50                # synthetic positives
```

These set the gradient batch shape per step. Changing them changes the
effective LR and convergence — leave at upstream defaults.

### Architecture

```yaml
model_type: "conv_attention"
layer_size: 128       # hidden dim, must be divisible by n_heads
n_heads: 4
n_conv: 2             # Conv1D(kernel=3) blocks
n_attn: 1             # MultiheadAttention blocks (residual + LayerNorm)
```

`conv_attention`: `Linear(96→128) → 2× Conv1D(128) → BN → MultiheadAttention(128, 4 heads) → mean pool → Linear(128→1) → Sigmoid`.

Replaces upstream's `dnn` head's `Flatten(16×96)→Linear`, which
discards the temporal structure of the embedding sequence. The mean
pool over time is what makes single-word commands like "accept" work
well — phonetic neighbours differ in fine-grained timing.

Other model types are kept for compatibility (`dnn`, `lstm`, `rnn`)
but not used by the shipped configs.

### Loss + regularisation

```yaml
loss_type: "focal"
focal_gamma: 2.0
embedding_mixup: true
mixup_alpha: 0.2
label_smoothing: 0.05
weight_decay: 0.01
```

Direct port of the livekit-wakeword recipe. We tested γ=1 with
mixup off to push trained scores past ~0.6; that hurt the model on
real audio enough to be net-negative. If you want sharper scores, use
`inference_temperature` instead of weakening the loss.

### Negative pressure

```yaml
max_negative_weight: 750
target_false_positives_per_hour: 1.0
```

`auto_train` ramps the negative-class loss weight from 1 to
`max_negative_weight` over phase 1, then doubles it each subsequent
phase if FP/hr is still above target. Upstream defaults
(`max_negative_weight: 1500`) plus old-style adversarial negatives
that included input words skewed gradients so heavily against
negatives that the model under-fired on the real phrase. 750 + the
relaxed 1.0 FP/hr target loosens that pressure; the focal loss + better
adversarial generation handle the tail.

### Inference temperature

```yaml
inference_temperature: 1.0
```

Post-hoc temperature scaling baked into the exported ONNX:

    s' = sigmoid(T · logit(s))

`T > 1` sharpens the distribution: 0.5 stays fixed, scores above 0.5
climb toward 1, scores below 0.5 collapse toward 0. Default 1.0
(no-op). Pick `T` based on the post-training threshold sweep, **not
blindly** — cranking T up on a model whose real-audio scores live
below 0.5 will make the model never fire (we did this; see
[architecture.md](architecture.md#inference-temperature)).

### Steps

```yaml
steps: 100000
```

100 k steps lets the conv_attention head fully converge with a 100 k
positive set. `auto_train` splits this across 3 phases:

1. **Full** at base LR until step `steps`.
2. **Refinement** at LR × 0.1.
3. **Fine-tune** at LR × 0.01.

Phase 1 checkpoints are excluded from the SWA pool because averaging
across LR regimes is meaningless (Izmailov et al. 2018).

## What gets produced

After `--train_model` completes, you'll find under
`training_workspace/output/<model_name>/`:

```
positive_features_train.npy   training positives (synthetic + augmented)
positive_features_test.npy    held-out positives
negative_features_train.npy   adversarial negatives
negative_features_test.npy    held-out negatives
checkpoints/                  per-phase checkpoint pool used for SWA selection
```

And at the parent `output_dir`:

```
<model_name>.onnx             final exported model with temperature baked in
```

## Smoke-test before a long run

The cloud runs are ~6 hr; verify the pipeline works end-to-end on
laptop GPU first by dropping `n_samples` to ~2000 and `steps` to
~5000. The smoke-test should produce a non-degenerate `.onnx` (i.e.
`scripts/eval_onnx.py` reports recall and FP/hr, even if both are bad).
This catches:

- missing piper / piper-sample-generator,
- missing RIR / background paths,
- BatchNorm / metric reset bugs (val recall and post-selection recall
  must agree exactly to ~3 decimal places),
- ONNX export failures.
