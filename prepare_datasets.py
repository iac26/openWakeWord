"""Download + prepare background-noise / RIR datasets for openWakeWord training.

Mirrors the dataset-prep cells of `notebooks/automatic_model_training.ipynb`,
runnable as a CLI on a headless GPU server.

Outputs (under --workdir, default ./training_workspace):
    mit_rirs/                MIT environmental impulse responses (16 kHz wav)
    audioset/                downloaded raw AudioSet tar shards + extracted flacs
    audioset_16k/            AudioSet resampled to 16 kHz wav
    fma/                     FMA "small" clips resampled to 16 kHz wav

Requires `datasets<3` (pinned in [training]); newer versions force
torchcodec, which is brittle on hosts whose ffmpeg ABI doesn't match.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import tarfile
from pathlib import Path

import numpy as np
import scipy.io.wavfile
from tqdm import tqdm

_BAR_FMT = "{l_bar}{bar:30}{r_bar}"


def _section(title: str) -> None:
    line = "=" * 70
    print(f"\n{line}\n  {title}\n{line}")

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("prepare_datasets")

# Quiet down chatty third-party loggers so the script's own progress is readable.
for _name in (
    "httpx", "httpcore", "urllib3", "requests",
    "huggingface_hub", "datasets", "filelock", "fsspec",
):
    logging.getLogger(_name).setLevel(logging.WARNING)
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "0")
os.environ.setdefault("DATASETS_VERBOSITY", "warning")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "warning")

TARGET_SR = 16000


def _write_wav16(path: Path, arr: np.ndarray) -> None:
    scipy.io.wavfile.write(path, TARGET_SR, (np.asarray(arr) * 32767).astype(np.int16))


def download_mit_rirs(out_dir: Path) -> None:
    import datasets

    out_dir.mkdir(parents=True, exist_ok=True)
    _section(f"MIT RIRs -> {out_dir}")
    ds = datasets.load_dataset(
        "davidscripka/MIT_environmental_impulse_responses",
        split="train",
        streaming=True,
    )
    for row in tqdm(ds, desc="mit_rirs", bar_format=_BAR_FMT, unit="file"):
        name = Path(row["audio"]["path"]).name
        target = out_dir / name
        if target.exists():
            continue
        _write_wav16(target, row["audio"]["array"])


def download_audioset(workdir: Path, subset: str = "balanced", max_clips: int | None = None) -> None:
    """Download AudioSet via the parquet-format HuggingFace dataset and
    resample each clip to 16 kHz wav.

    `agkphysics/AudioSet` is gated — `huggingface-cli login` (or set
    HF_TOKEN) before running this. The repo migrated from `.tar` shards
    to parquet sometime in 2024-2025; we now stream rows directly via
    `datasets.load_dataset`, which decodes the embedded audio per row.

    Subsets:
      "balanced"   (~22k clips, ~5 hr)   — default, enough for a POC
      "unbalanced" (~2M clips, ~5500 hr) — full set, several hundred GB
      "eval"       (~22k clips, ~5 hr)   — eval split, generally avoid
    """
    import datasets

    out_dir = workdir / "audioset_16k"
    out_dir.mkdir(parents=True, exist_ok=True)

    _section(f"AudioSet[{subset}] -> {out_dir}"
             + (f" (cap: {max_clips} clips)" if max_clips else ""))

    ds = datasets.load_dataset(
        "agkphysics/AudioSet",
        subset,
        split="train" if subset != "eval" else "test",
        streaming=True,
        trust_remote_code=False,
    )
    ds = ds.cast_column("audio", datasets.Audio(sampling_rate=TARGET_SR))

    n_written = 0
    n_skipped = 0
    pbar = tqdm(ds, desc=f"audioset_16k", bar_format=_BAR_FMT, unit="clip",
                total=max_clips)
    for row in pbar:
        if max_clips is not None and (n_written + n_skipped) >= max_clips:
            break
        # Path inside the parquet has the form "audio/<id>.flac" or similar
        src_path = Path(row["audio"]["path"]).name
        target = out_dir / src_path.replace(".flac", ".wav").replace(".mp3", ".wav")
        if target.suffix != ".wav":
            target = target.with_suffix(".wav")
        if target.exists():
            n_skipped += 1
            continue
        _write_wav16(target, row["audio"]["array"])
        n_written += 1
    pbar.close()
    log.info("AudioSet[%s]: wrote %d new clips, %d already on disk",
             subset, n_written, n_skipped)


def download_fma(out_dir: Path, n_hours: float) -> None:
    """Download FMA "small" subset clips, resample to 16 kHz wav.

    FMA small clips are ~30 s each, so n_clips = n_hours * 3600 / 30.
    """
    import datasets

    out_dir.mkdir(parents=True, exist_ok=True)
    n_clips = int(n_hours * 3600 / 30)
    _section(f"FMA -> {out_dir} ({n_clips} clips, ~{n_hours:.1f} h)")

    # rudraml/fma ships a loading script (custom code); explicit opt-in required
    # since `datasets` 2.16+. Inspect at https://hf.co/datasets/rudraml/fma.
    ds = datasets.load_dataset(
        "rudraml/fma", name="small", split="train", streaming=True, trust_remote_code=True,
    )
    ds = iter(ds.cast_column("audio", datasets.Audio(sampling_rate=TARGET_SR)))

    for i in tqdm(range(n_clips), desc="fma", bar_format=_BAR_FMT, unit="clip"):
        try:
            row = next(ds)
        except StopIteration:
            log.warning("FMA stream exhausted after %d clips", i)
            break
        name = Path(row["audio"]["path"]).name.replace(".mp3", ".wav")
        target = out_dir / name
        if target.exists():
            continue
        _write_wav16(target, row["audio"]["array"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workdir", type=Path, default=Path("training_workspace"),
                        help="Root directory for downloaded datasets (default: ./training_workspace)")
    parser.add_argument("--skip-rirs", action="store_true", help="Skip MIT RIR download")
    parser.add_argument("--skip-audioset", action="store_true", help="Skip AudioSet download")
    parser.add_argument("--skip-fma", action="store_true", help="Skip FMA download")
    parser.add_argument("--audioset-subset", default="balanced",
                        choices=["balanced", "unbalanced", "eval"],
                        help="AudioSet subset (default: balanced ~22k clips ~5 hr). "
                             "'unbalanced' is the full ~2M-clip set (several hundred GB).")
    parser.add_argument("--audioset-clips", type=int, default=None,
                        help="Cap AudioSet at this many clips (default: full subset). "
                             "Useful for quick POCs — try 5000 for ~1 hr of audio.")
    parser.add_argument("--fma-hours", type=float, default=1.0,
                        help="Hours of FMA audio to download (default: 1.0). "
                             "Bump to 50+ for real training runs.")
    args = parser.parse_args()

    args.workdir.mkdir(parents=True, exist_ok=True)
    print(f"workdir = {args.workdir.resolve()}")

    try:
        if not args.skip_rirs:
            download_mit_rirs(args.workdir / "mit_rirs")
        if not args.skip_audioset:
            try:
                download_audioset(args.workdir, args.audioset_subset, args.audioset_clips)
            except Exception as e:
                log.error("AudioSet download failed: %s", e)
                log.error("AudioSet is gated. Accept the license at "
                          "https://huggingface.co/datasets/agkphysics/AudioSet "
                          "and run `huggingface-cli login` (or export HF_TOKEN), then re-run.")
        if not args.skip_fma:
            # FMA via rudraml/fma's custom loader is fragile (streaming/ZIP
            # interaction with fsspec HTTP). If it fails, log and continue
            # so AudioSet (the primary noise source) still has a chance to
            # populate audioset_16k/.
            try:
                download_fma(args.workdir / "fma", args.fma_hours)
            except Exception as e:
                log.error("FMA download failed: %s", e)
                log.error("FMA is optional — AudioSet alone is sufficient as a "
                          "background-noise source. Continuing.")
    except ImportError as e:
        log.error("Missing dependency: %s. Run `pip install -e .[training]` first.", e)
        return 2

    _section("Done. Set in your training YAML:")
    print(f"  background_paths: ['{(args.workdir / 'audioset_16k').resolve()}',")
    print(f"                     '{(args.workdir / 'fma').resolve()}']")
    print(f"  rir_paths:        ['{(args.workdir / 'mit_rirs').resolve()}']")
    return 0


if __name__ == "__main__":
    sys.exit(main())
