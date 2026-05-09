"""Slice (N, 16, 96) feature .npy files down to (N, 8, 96).

Used to retrain accept/decline with a shorter input window without
re-running --augment_clips. Synthesised clips have the word landing
near the end of the 1.28 s window (Piper tends to leave leading
silence), so the trailing 8 frames carry most of the acoustic signal.

Usage:
    python scripts/slice_features_to_8.py \\
        --src_dir training_workspace/output/accept \\
        --dst_dir training_workspace/output/accept_8frame
"""

import argparse
import os
import sys

import numpy as np


FILES = [
    "positive_features_train.npy",
    "positive_features_test.npy",
    "negative_features_train.npy",
    "negative_features_test.npy",
]


def slice_file(src: str, dst: str, n_frames: int, position: str) -> None:
    src_arr = np.load(src, mmap_mode="r")
    if src_arr.ndim != 3 or src_arr.shape[2] != 96:
        raise ValueError(f"Expected (N, T, 96), got {src_arr.shape} for {src}")
    if src_arr.shape[1] < n_frames:
        raise ValueError(
            f"Source has only {src_arr.shape[1]} frames, can't slice to {n_frames}"
        )
    if position == "trailing":
        sliced = np.ascontiguousarray(src_arr[:, -n_frames:, :])
    elif position == "center":
        start = (src_arr.shape[1] - n_frames) // 2
        sliced = np.ascontiguousarray(src_arr[:, start:start + n_frames, :])
    elif position == "peak":
        # Peak-centered: empirically the per-frame L2 norm peaks around
        # frame 11 of 16 for accept (Piper places the word slightly toward
        # the back). Frames 6-13 capture the word with one frame of lead-in
        # and one of trail-out.
        if src_arr.shape[1] == 16 and n_frames == 8:
            sliced = np.ascontiguousarray(src_arr[:, 6:14, :])
        else:
            raise ValueError(
                f"position=peak only defined for 16->8 slicing, got "
                f"{src_arr.shape[1]}->{n_frames}"
            )
    else:
        raise ValueError(f"Unknown position: {position}")
    np.save(dst, sliced)
    print(f"  {os.path.basename(src)}: {src_arr.shape} -> {sliced.shape}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src_dir", required=True, help="Dir containing 16-frame .npy files")
    ap.add_argument("--dst_dir", required=True, help="Dir for sliced .npy files")
    ap.add_argument("--n_frames", type=int, default=8, help="Output frame count (default 8)")
    ap.add_argument("--position", choices=["trailing", "center", "peak"], default="peak",
                    help="Which slice of the 16-frame window to keep. 'peak' is empirically "
                         "best for 16->8 (frames 6-13, centered on the per-frame energy peak "
                         "observed at frame 11). Default: peak.")
    args = ap.parse_args()

    os.makedirs(args.dst_dir, exist_ok=True)
    print(f"Slicing {args.src_dir} -> {args.dst_dir} ({args.n_frames} {args.position} frames)")
    for fname in FILES:
        src = os.path.join(args.src_dir, fname)
        dst = os.path.join(args.dst_dir, fname)
        if not os.path.exists(src):
            print(f"  WARN: {src} missing, skipping")
            continue
        slice_file(src, dst, args.n_frames, args.position)
    return 0


if __name__ == "__main__":
    sys.exit(main())
