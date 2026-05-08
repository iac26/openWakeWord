# Evaluating a trained model

Once `auto_train` produces an `.onnx`, use `scripts/eval_onnx.py` to
measure it locally against the held-out test features and the long
negative-only validation stream. The script reproduces `auto_train`'s
end-of-training metrics exactly — same FP/hr definition, same
threshold-sweep logic — so the numbers it reports should match the
training log to ~3 decimal places.

## Quick start

```bash
python scripts/eval_onnx.py \
    --onnx training_workspace/output/hey_ari.onnx \
    --pos_features training_workspace/output/hey_ari/positive_features_test.npy \
    --neg_features training_workspace/output/hey_ari/negative_features_test.npy \
    --fp_features training_workspace/features/validation_set_features.npy
```

Output (truncated):

```
ONNX input: ['batch_size', 16, 96]
Positives: (2000, 16, 96)
Negatives: (10000, 16, 96)
FP source: (508500, 96) -> 11.300 hours
FP windows: (508484, 16, 96)

=== Metrics at threshold 0.50 ===
  recall (TP rate on pos):  0.8850
  FP/hr on long neg set:    0.4674  (5 FPs over 11.30 hours)
  balanced accuracy:        0.9420

=== Score distributions ===
  positives  : n=  2000  mean=0.91  median=0.97  p90=0.99  max=1.00
  negatives  : n= 10000  mean=0.04  median=0.01  p90=0.12  max=0.99
  fp_windows : n=508484  mean=0.01  median=0.00  p90=0.02  max=0.93

=== Optimal threshold (target FP/hr <= 1.0, min_recall 0.5) ===
  threshold:        0.50
  recall:           0.8850
  FP/hr:            0.4674
  balanced acc:     0.9420
```

## What the metrics mean

| Metric              | What it measures                                                |
| ------------------- | --------------------------------------------------------------- |
| recall              | Fraction of held-out positives whose score ≥ threshold.         |
| FP/hr               | False positives per hour on the long negative-only stream.     |
|                    | One row of `--fp_features` = 80 ms; the script slides a 16-frame |
|                    | window so FP count = number of windows with score ≥ threshold.  |
| balanced accuracy   | (recall + TNR_synthetic_neg) / 2 — sanity check, not the target.|
| score distributions | Where the model's outputs land. Bimodality is what you want:    |
|                    | positives near 1.0, fp_windows near 0.0. Collapse toward 0.5    |
|                    | means metrics are broken (we hit this; see architecture.md).    |

## Threshold sweep

The script also runs the same threshold sweep `auto_train` runs at
the end of training:

- Sweep `t ∈ [0.01, 0.99]` in steps of 0.01.
- Among thresholds with FP/hr ≤ `--target_fpph` (default 1.0) **and**
  recall ≥ `--min_recall` (default 0.5), pick the one with highest
  recall.
- If no threshold meets the FP target, fall back to the threshold
  with the best balanced accuracy.

Use this output to pick a deployment threshold — not just the
default 0.5.

## Real-time inference: batch=1

Models exported by this fork use a **dynamic batch axis**. The
`scripts/eval_onnx.py` benchmark exercises batched inference for speed
(default `batch_size=256`), but real-time deployment is always
batch=1: one window per ~80 ms audio chunk.

Effective inference loop:

```
every 80 ms:
    new_audio_chunk -> embedding_model -> 1 new (96,) feature row
    push into a 16-frame ring buffer (drop oldest)
    score = wake_word_model(buffer.unsqueeze(0))   # (1, 16, 96), batch=1
    if score >= threshold: trigger
```

Because the batch axis is dynamic, the **same** `.onnx` works for both
batch=1 streaming and batched offline evaluation — no re-export
needed.

### Older fixed-batch exports

Some early model snapshots in `WakeWordTraining/v1/` and `v2/` were
exported with a fixed `batch_size=1` (older `train.py`). They still
score correctly but can only be run one window at a time.
`scripts/eval_onnx.py` detects this case and pads the trailing partial
batch internally so eval still works:

```python
# scripts/eval_onnx.py
in_dim0 = session.get_inputs()[0].shape[0]
if isinstance(in_dim0, int) and in_dim0 > 0:
    batch_size = in_dim0  # respect the model's hardcoded batch
```

You don't need to re-export — but new training runs always produce
the dynamic-axis form.

## Comparing model versions

Run the same script against each `.onnx`:

```bash
for v in v1 v2 v3 v4 v5; do
    echo "=== $v ==="
    python scripts/eval_onnx.py \
        --onnx ../WakeWordTraining/$v/hey-ari.onnx \
        --pos_features training_workspace/output/hey_ari/positive_features_test.npy \
        --neg_features training_workspace/output/hey_ari/negative_features_test.npy \
        --fp_features training_workspace/features/validation_set_features.npy
done
```

The score distribution block is the most useful comparison signal:

- **Healthy model:** positive median > 0.9, fp_windows p90 < 0.1.
- **Broken metrics-era model (e.g. v3):** positive median ≈ 0.48,
  negative p90 ≈ 0.36 — bimodality has collapsed; recall reported in
  the training log was wrong.
- **Over-temperatured model:** positives still well-separated in
  *logit* space, but the post-T sigmoid pushes everything outside
  `[ε, 1-ε]`; real-audio scores hug 0 because the calibration was
  fit on synthetic-only positives.

## CLI reference

```
--onnx              path to trained .onnx
--pos_features      positive_features_test.npy from training output dir
--neg_features      negative_features_test.npy from training output dir
--fp_features       long negative-only stream (e.g. validation_set_features.npy)
--target_fpph       target FP/hr for threshold sweep (default 1.0)
--min_recall        recall floor for threshold sweep (default 0.5)
--threshold         threshold for the headline metrics (default 0.5)
```
