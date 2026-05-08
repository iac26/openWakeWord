"""Download + prepare background-noise / RIR datasets for openWakeWord training.

Mirrors the dataset-prep cells of `notebooks/automatic_model_training.ipynb`,
but as a CLI you can run on a headless GPU server.

Outputs (under --workdir, default ./training_workspace):
    mit_rirs/                MIT environmental impulse responses (16 kHz wav)
    audioset/                downloaded raw AudioSet tar shards + extracted flacs
    audioset_16k/            AudioSet resampled to 16 kHz wav
    fma/                     FMA "small" clips resampled to 16 kHz wav

Run AFTER `install_training.sh` has installed the `[training]` extra
(this script needs `datasets`, `scipy`, `tqdm`, `numpy`).
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np
import scipy.io.wavfile
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("prepare_datasets")


def _write_wav(path: Path, sr: int, arr: np.ndarray) -> None:
    scipy.io.wavfile.write(path, sr, (np.asarray(arr) * 32767).astype(np.int16))


def download_mit_rirs(out_dir: Path) -> None:
    import datasets

    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("MIT RIRs -> %s (streaming from HuggingFace)", out_dir)
    ds = datasets.load_dataset(
        "davidscripka/MIT_environmental_impulse_responses",
        split="train",
        streaming=True,
    )
    for row in tqdm(ds, desc="mit_rirs"):
        name = Path(row["audio"]["path"]).name
        target = out_dir / name
        if target.exists():
            continue
        _write_wav(target, 16000, row["audio"]["array"])


def download_audioset(workdir: Path, shards: list[str]) -> None:
    """Download given AudioSet shards from HuggingFace, extract, resample to 16 kHz wav.

    Defaults to a single ~1 GB shard ("bal_train09.tar"). For real training pass
    multiple shards (or the full balanced+unbalanced set) — they cover ~50 GB+.
    Browse names at https://huggingface.co/datasets/agkphysics/AudioSet/tree/main/data.
    """
    import datasets

    raw_dir = workdir / "audioset"
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_dir = workdir / "audioset_16k"
    out_dir.mkdir(parents=True, exist_ok=True)

    base_url = "https://huggingface.co/datasets/agkphysics/AudioSet/resolve/main/data"
    for shard in shards:
        tar_path = raw_dir / shard
        if not tar_path.exists() or tar_path.stat().st_size < 1024:
            log.info("Downloading %s", shard)
            subprocess.check_call(
                ["wget", "-q", "--show-progress", "-O", str(tar_path), f"{base_url}/{shard}"]
            )
        else:
            log.info("[skip] %s already present", shard)

        log.info("Extracting %s", shard)
        with tarfile.open(tar_path) as tf:
            tf.extractall(raw_dir)

    flacs = sorted(str(p) for p in (raw_dir / "audio").rglob("*.flac"))
    if not flacs:
        log.warning("No FLAC files found in %s/audio after extraction.", raw_dir)
        return

    log.info("Resampling %d AudioSet clips to 16 kHz wav -> %s", len(flacs), out_dir)
    ds = datasets.Dataset.from_dict({"audio": flacs})
    ds = ds.cast_column("audio", datasets.Audio(sampling_rate=16000))
    for row in tqdm(ds, desc="audioset_16k"):
        name = Path(row["audio"]["path"]).name.replace(".flac", ".wav")
        target = out_dir / name
        if target.exists():
            continue
        _write_wav(target, 16000, row["audio"]["array"])


def download_fma(out_dir: Path, n_hours: float) -> None:
    """Download FMA "small" clips (HF streaming) and resample to 16 kHz wav.

    FMA "small" clips are 30 s each, so n_clips = n_hours * 3600 / 30.
    """
    import datasets

    out_dir.mkdir(parents=True, exist_ok=True)
    n_clips = int(n_hours * 3600 / 30)
    log.info("FMA -> %s (%d clips, ~%.1f hours)", out_dir, n_clips, n_hours)

    ds = datasets.load_dataset("rudraml/fma", name="small", split="train", streaming=True)
    ds = iter(ds.cast_column("audio", datasets.Audio(sampling_rate=16000)))

    for i in tqdm(range(n_clips), desc="fma"):
        try:
            row = next(ds)
        except StopIteration:
            log.warning("FMA stream exhausted after %d clips", i)
            break
        name = Path(row["audio"]["path"]).name.replace(".mp3", ".wav")
        target = out_dir / name
        if target.exists():
            continue
        _write_wav(target, 16000, row["audio"]["array"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workdir", type=Path, default=Path("training_workspace"),
                        help="Root directory for downloaded datasets (default: ./training_workspace)")
    parser.add_argument("--skip-rirs", action="store_true", help="Skip MIT RIR download")
    parser.add_argument("--skip-audioset", action="store_true", help="Skip AudioSet download")
    parser.add_argument("--skip-fma", action="store_true", help="Skip FMA download")
    parser.add_argument("--audioset-shards", nargs="+", default=["bal_train09.tar"],
                        help="AudioSet tar shard filenames to fetch from HuggingFace "
                             "(default: bal_train09.tar — one ~1 GB shard).")
    parser.add_argument("--fma-hours", type=float, default=1.0,
                        help="Hours of FMA audio to download (default: 1.0). "
                             "Bump to 50+ for real training runs.")
    args = parser.parse_args()

    args.workdir.mkdir(parents=True, exist_ok=True)
    log.info("workdir = %s", args.workdir.resolve())

    try:
        if not args.skip_rirs:
            download_mit_rirs(args.workdir / "mit_rirs")
        if not args.skip_audioset:
            download_audioset(args.workdir, args.audioset_shards)
        if not args.skip_fma:
            download_fma(args.workdir / "fma", args.fma_hours)
    except ImportError as e:
        log.error("Missing dependency: %s. Run `pip install -e .[training]` first.", e)
        return 2

    log.info("Done. Set in your training YAML:")
    log.info("  background_paths: ['%s', '%s']",
             (args.workdir / "audioset_16k").resolve(),
             (args.workdir / "fma").resolve())
    log.info("  rir_paths: ['%s']", (args.workdir / "mit_rirs").resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
