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
- runs `uv sync` against `pyproject.toml` + `uv.lock` to create `.venv`
  (or reuse `.venv-train` if it already exists). Inference and training
  deps are merged into a single dep list, no extras to select. Torch +
  torchaudio come from the pinned PyTorch CUDA 12.4 index via
  `[tool.uv.sources]`.
- clones `piper-sample-generator` into `training_workspace/piper-sample-generator/`,
- pre-downloads the openWakeWord melspectrogram + embedding + VAD ONNX models,
- pre-downloads the openWakeWord precomputed feature shards
  (`openwakeword_features_ACAV100M_2000_hrs_16bit.npy`,
  `validation_set_features.npy`),
- optionally invokes `prepare_datasets.py` for RIR/AudioSet/FMA
  (`DOWNLOAD_DATASETS=1` by default).

By default the script does **not** touch apt — on hosts with
`nvidia-dkms`, installing `build-essential` triggers a 5–10 min DKMS
rebuild. Set `INSTALL_SYSTEM_PKGS=1` only on a fresh box that's
missing `ffmpeg`, `sox`, `espeak-ng`, `libspeexdsp-dev`, or
`libsndfile1`.

Env knobs (override before invoking the script):

| var                   | default       | meaning                                                       |
| --------------------- | ------------- | ------------------------------------------------------------- |
| `WORKDIR`             | `$PWD/training_workspace` | where features + clips land                       |
| `INSTALL_SYSTEM_PKGS` | `0`           | `1` = run `apt-get install` for system deps                   |
| `DOWNLOAD_FEATURES`   | `1`           | `1` = download the ~7 GB precomputed feature shards           |
| `DOWNLOAD_PIPER`      | `1`           | `1` = clone piper-sample-generator + voice model              |
| `DOWNLOAD_DATASETS`   | `1`           | `1` = invoke `prepare_datasets.py` (RIR/AudioSet/FMA)         |
| `AUDIOSET_SUBSET`     | `balanced`    | AudioSet subset: `balanced` (~5 hr) / `unbalanced` / `eval`   |
| `AUDIOSET_CLIPS`      | (unset)       | optional cap on AudioSet clip count                           |
| `FMA_HOURS`           | `1`           | hours of FMA music to download                                |

Everything `install_training.sh` does is idempotent — re-running is
safe.

## Step 2 — Prepare datasets

```bash
python prepare_datasets.py             # default: 1 AudioSet shard, 1 hr FMA
python prepare_datasets.py --help      # see all flags
```

Downloads and converts to 16 kHz wav under `training_workspace/`:

- `mit_rirs/` — MIT room-impulse-response set (used as RIR augmentation).
- `audioset_16k/` — AudioSet background noise/music. The HuggingFace
  AudioSet repo migrated from `.tar` shards to parquet, so this step
  now streams rows directly via `datasets.load_dataset("agkphysics/AudioSet", subset, streaming=True)`.
  Default subset is `balanced` (~5 hr); use `--audioset-subset unbalanced`
  for more. Gated dataset — needs an HF token (`huggingface-cli login`
  or `HF_TOKEN=...`).
- `fma/` — FMA "small" subset music clips. Wrapped in try/except;
  failure to download is non-fatal. Skip with `--skip-fma`.

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
    --training_config configs/hey_ari.yml \
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

## Tier system

Each of the three models ships in four tiers, all using the same
`conv_attention` head and the same loss recipe — what changes is the
synthetic-positive count, the head width, the number of training
steps, and the negative-pressure schedule:

| Tier  | `n_samples` | `layer_size` | `n_conv` | `steps`  | ACAV batch | pos/adv batch | `max_negative_weight` | target FP/hr |
| ----- | ----------- | ------------ | -------- | -------- | ---------- | ------------- | --------------------- | ------------ |
| micro |        100  |       32     |    1     |    1 000 |     128    |       10      |          50           |     5.0      |
| tiny  |      1 000  |       32     |    1     |    5 000 |     256    |       20      |         100           |     5.0      |
| small |     30 000  |       64     |    1     |   15 000 |     512    |       50      |         150           |     5.0      |
| full  |    100 000  |      128     |    2     |  100 000 |    1 024   |       50      |         750           |     1.0      |

Promotion workflow (re-uses WAVs across tiers via symlink, so the
expensive Piper synthesis only happens once at the bottom tier):

```bash
./run_train.sh --training_config configs/hey_ari_micro.yml --generate_clips
# listen to a handful of clips; if they sound clean:
scripts/promote_clips.sh hey_ari_micro hey_ari_tiny
./run_train.sh --training_config configs/hey_ari_tiny.yml --generate_clips --augment_clips --train_model
# ... and so on up to full.
```

Clip generation is **incremental**: `train.py` checks how many WAVs
already live in `positive_train/` and only synthesises the delta to
reach `n_samples`. To force full regeneration, delete the clip dirs.
Feature `.npy` caching is also incremental — if you change `n_samples`
on an existing model dir, `rm <output_dir>/<model_name>/*_features_*.npy`
to regenerate features at the new count.

The `max_negative_weight` ladder is intentional: at the upstream
default of 750 on tiny/small data the gradient is so heavily skewed
against negatives that the score distribution collapses toward 0 on
real audio (same shape as the deployment-pipeline bug documented in
`inference_audio_contract.md`). The full tier still uses 750 because
it has the data to support that pressure.

## Anatomy of a training config

The shipped configs (`configs/hey_ari.yml`, `configs/accept.yml`, `configs/decline.yml`
and their tier variants) are heavily commented; this section explains
the *why* behind each group of knobs.

### Phrase + adversarial negatives

```yaml
target_phrase:
  - "hey ari ."
  - "hey ari !"
  - "hey ari ?"

custom_negative_phrases: []
```

`target_phrase` is the list of phrases Piper synthesises positives for
(`n_samples` is split evenly across the list). The shipped configs use
three intonation hints (`.`, `!`, `?`) per word so Piper produces the
same phrase with declarative / exclamative / interrogative prosody;
the trailing punctuation is stripped before the text is handed to
`generate_adversarial_texts` (which uses CMUDict and doesn't recognise
punctuation as words). We do **not** broaden the phrase with paraphrases
("yes accept", "i accept") — that was tried and rejected because it
trains the head to fire on the surrounding filler.

`custom_negative_phrases` is intentionally empty in all shipped
configs. `generate_adversarial_texts` auto-generates phonetic
near-misses for the target phrase (using DeepPhonemizer to handle
out-of-vocabulary words like "ari"), and ACAV100M's 2000 hr of real
audio covers the broad negative space. Phonetic neighbours added by
hand (`hey carl`, `incline` for `decline`) double-count the same
pressure and over-train rejection of near-misses, which causes the
model to under-fire on the real word. Source: openWakeWord upstream
guidance from dscripka. We relearned this the hard way; see
[architecture.md](architecture.md#custom-negatives-near-vs-far). If
production false positives reveal a class the auto adversarials miss,
add specific terms here later.

### Sample counts

```yaml
n_samples: 100000        # synthetic positives (split across target_phrase)
n_samples_val: 2000      # synthetic positives held out for validation
```

100 k (full tier) matches upstream's `alexa` / `hey_mycroft` per-phrase
scale. The small tier uses 30 k and trains a deployable-quality model
in 30–45 min on a laptop GPU. Below ~30 k recall noticeably degrades
on real audio; the tiny (1 k) and micro (100) tiers exist to exercise
the pipeline, not to produce usable models. See the tier table above.

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
max_negative_weight: 750            # full tier only
target_false_positives_per_hour: 1.0
```

`auto_train` ramps the negative-class loss weight from 1 to
`max_negative_weight` over phase 1, then doubles it each subsequent
phase if FP/hr is still above target. Upstream defaults
(`max_negative_weight: 1500`) plus old-style adversarial negatives
that included input words skewed gradients so heavily against
negatives that the model under-fired on the real phrase. 750 + the
relaxed 1.0 FP/hr target loosens that pressure on the full tier; the
focal loss + better adversarial generation handle the tail.

The smaller tiers use a graded ladder (50 / 100 / 150 for
micro / tiny / small) with a relaxed 5.0 FP/hr target — at smaller
data scales, a weight of 750 collapses the score distribution toward
0 on real audio (same failure mode as the deployment-pipeline bug
documented in `inference_audio_contract.md`). See the tier matrix
above.

### Piper synthesis settings

```yaml
noise_scales:    [0.667]   # acoustic noise (Piper's own default)
noise_scale_ws:  [0.8]     # phoneme-length stochasticity (hey_ari)
# or [0.4] for accept / decline
length_scales:   [1.1, 1.2, 1.3]   # accept / decline only
```

All three models override `noise_scales` to `0.667` (Piper's own
default), not upstream openwakeword's `0.98` — at 0.98 the short
command words ("accept", "decline") come out partially garbled. Per-
clip diversity comes from augmentation (RIRs + AudioSet background)
rather than sampling noise.

`hey_ari` keeps the Piper default `noise_scale_w=0.8` and lets
`train.py`'s default `length_scales` (a wide `[0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.15, 1.33]`
sweep) apply. `accept` and `decline` override
`noise_scale_w=0.4` and narrow `length_scales` to `[1.1, 1.2, 1.3]`:
the wider range produced ~200 ms "accept" clips that didn't sound
like real speech, and a higher `noise_scale_w` chopped the trailing
`/t/` off the consonant.

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

100 k steps (full tier) lets the conv_attention head fully converge
with a 100 k positive set. Smaller tiers use proportionally fewer
steps (see the tier matrix). `auto_train` splits this across 3 phases:

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

Full-tier cloud runs are ~6 hr; verify the pipeline works end-to-end
on laptop GPU first by running the **tiny** tier
(`configs/<model>_tiny.yml`, 1 k samples / 5 k steps, ~10 min). The
smoke-test should produce a non-degenerate `.onnx` — `scripts/eval_onnx.py`
should report recall and FP/hr, even if both are bad (a recall of
~1% on tiny is expected). This catches:

- missing piper / piper-sample-generator,
- missing RIR / background paths,
- BatchNorm / metric reset bugs (val recall and post-selection recall
  must agree exactly to ~3 decimal places),
- ONNX export failures.
