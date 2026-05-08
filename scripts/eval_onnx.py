"""Evaluate a trained openwakeword .onnx against the precomputed test features.

Usage:
    python scripts/eval_onnx.py \\
        --onnx training_workspace_local/output/hey_ari_small.onnx \\
        --pos_features training_workspace_local/output/hey_ari_small/positive_features_test.npy \\
        --neg_features training_workspace_local/output/hey_ari_small/negative_features_test.npy \\
        --fp_features training_workspace_local/features/validation_set_features.npy

Outputs:
  - recall / FP-per-hour / accuracy at the default 0.5 threshold
  - optimal-threshold sweep matching auto_train's _find_optimal_threshold

The numbers should match what auto_train reports at the end of training,
since this reuses the exact same metric definitions.
"""

import argparse
import sys

import numpy as np
import onnxruntime as ort


FRAME_SECONDS = 0.08  # one openwakeword embedding frame


def run_onnx(session: ort.InferenceSession, X: np.ndarray, batch_size: int = 256) -> np.ndarray:
    """Run an onnx session on (N, 16, 96) features in batches; return scores in (N,)."""
    if X.ndim != 3:
        raise ValueError(f"Expected (N, 16, 96), got {X.shape}")
    out_name = session.get_outputs()[0].name
    in_name = session.get_inputs()[0].name
    parts = []
    for i in range(0, X.shape[0], batch_size):
        chunk = X[i:i + batch_size].astype(np.float32, copy=False)
        scores = session.run([out_name], {in_name: chunk})[0]
        parts.append(np.asarray(scores).reshape(-1))
    return np.concatenate(parts)


def sliding_windows(features: np.ndarray, window: int = 16) -> np.ndarray:
    """Convert a flat (N, 96) feature stream into (N - window + 1, 16, 96) windows.
    Mirrors the X_val_fp construction in train.py.
    """
    if features.ndim != 2:
        raise ValueError(f"Expected (N, 96), got {features.shape}")
    n = features.shape[0] - window
    if n <= 0:
        raise ValueError(f"Feature stream too short ({features.shape[0]}) for window={window}")
    return np.stack([features[i:i + window] for i in range(0, n, 1)])


def metrics_at_threshold(pos_preds, neg_preds, fp_preds, fp_hours, t):
    """Compute recall, FP/hr, and balanced accuracy at threshold t."""
    recall = float(np.mean(pos_preds >= t)) if pos_preds.size else 0.0
    fp_count = int(np.sum(fp_preds >= t)) if fp_preds.size else 0
    fpph = fp_count / fp_hours if fp_hours > 0 else float("inf")
    tnr_short = float(np.mean(neg_preds < t)) if neg_preds.size else 0.0
    balanced_acc = (recall + tnr_short) / 2.0
    return recall, fpph, balanced_acc, fp_count


def find_best_threshold(pos_preds, neg_preds, fp_preds, fp_hours, target_fpph, min_recall=0.5):
    """Sweep thresholds in [0.01, 0.99]. Pick the one that maximizes recall while
    keeping FP/hr <= target_fpph. Falls back to best balanced accuracy otherwise.
    """
    thresholds = np.arange(0.01, 1.0, 0.01)
    best, best_fallback = None, None
    for t in thresholds:
        recall, fpph, bal_acc, _ = metrics_at_threshold(pos_preds, neg_preds, fp_preds, fp_hours, float(t))
        if recall < min_recall:
            continue
        entry = (float(t), recall, fpph, bal_acc)
        if fpph <= target_fpph and (best is None or recall > best[1]):
            best = entry
        if best_fallback is None or bal_acc > best_fallback[3]:
            best_fallback = entry
    return best if best is not None else best_fallback


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", required=True, help="Path to the trained .onnx model")
    ap.add_argument("--pos_features", required=True, help="positive_features_test.npy (synthetic pos windows)")
    ap.add_argument("--neg_features", required=True, help="negative_features_test.npy (synthetic neg windows)")
    ap.add_argument("--fp_features", required=True,
                    help="Long negative-only feature stream, e.g. validation_set_features.npy. "
                         "Used for FP/hr; one row = 80ms.")
    ap.add_argument("--target_fpph", type=float, default=1.0,
                    help="Target FP/hr for the threshold sweep (default 1.0)")
    ap.add_argument("--min_recall", type=float, default=0.5,
                    help="Floor for the threshold sweep (default 0.5)")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Threshold for the headline metrics (default 0.5)")
    args = ap.parse_args()

    session = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    in_shape = session.get_inputs()[0].shape
    print(f"ONNX input: {in_shape}", flush=True)

    pos = np.load(args.pos_features)
    neg = np.load(args.neg_features)
    fp_raw = np.load(args.fp_features)
    fp_hours = fp_raw.shape[0] * FRAME_SECONDS / 3600.0
    print(f"Positives: {pos.shape}", flush=True)
    print(f"Negatives: {neg.shape}", flush=True)
    print(f"FP source: {fp_raw.shape} -> {fp_hours:.3f} hours", flush=True)

    # FP source is a flat (N, 96) feature stream — slide a 16-frame window.
    if fp_raw.ndim == 2:
        fp_windows = sliding_windows(fp_raw, window=16)
    else:
        fp_windows = fp_raw
    print(f"FP windows: {fp_windows.shape}", flush=True)

    print("Running inference...", flush=True)
    pos_preds = run_onnx(session, pos)
    neg_preds = run_onnx(session, neg)
    fp_preds = run_onnx(session, fp_windows)

    # Headline metrics at the chosen threshold.
    t = args.threshold
    recall, fpph, bal_acc, fp_count = metrics_at_threshold(pos_preds, neg_preds, fp_preds, fp_hours, t)
    print()
    print(f"=== Metrics at threshold {t:.2f} ===")
    print(f"  recall (TP rate on pos):  {recall:.4f}")
    print(f"  FP/hr on long neg set:    {fpph:.4f}  ({fp_count} FPs over {fp_hours:.2f} hours)")
    print(f"  balanced accuracy:        {bal_acc:.4f}")

    # Score-distribution summary so you can see where the model's outputs land.
    print()
    print("=== Score distributions ===")
    for name, arr in [("positives", pos_preds), ("negatives", neg_preds), ("fp_windows", fp_preds)]:
        if arr.size:
            print(f"  {name:11s}: n={arr.size:6d}  "
                  f"mean={arr.mean():.4f}  median={np.median(arr):.4f}  "
                  f"p90={np.percentile(arr, 90):.4f}  max={arr.max():.4f}")

    # Optimal-threshold sweep.
    chosen = find_best_threshold(pos_preds, neg_preds, fp_preds, fp_hours,
                                 target_fpph=args.target_fpph, min_recall=args.min_recall)
    print()
    if chosen is None:
        print(f"=== Threshold sweep ===\n  no threshold met min_recall={args.min_recall}")
        return 1
    t_opt, recall_opt, fpph_opt, bal_opt = chosen
    print(f"=== Optimal threshold (target FP/hr <= {args.target_fpph}, min_recall {args.min_recall}) ===")
    print(f"  threshold:        {t_opt:.2f}")
    print(f"  recall:           {recall_opt:.4f}")
    print(f"  FP/hr:            {fpph_opt:.4f}")
    print(f"  balanced acc:     {bal_opt:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
