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


def _resolve_audioset_shards(repo_id: str, requested: list[str]) -> list[str]:
    """Map requested shard names to actual paths in the repo, listing files via the HF API."""
    from huggingface_hub import HfApi
    from huggingface_hub.utils import GatedRepoError, HfHubHTTPError

    api = HfApi()
    try:
        all_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    except (GatedRepoError, HfHubHTTPError) as e:
        log.error(
            "Could not list files in %s: %s\n"
            "This dataset is gated. Accept the license at "
            "https://huggingface.co/datasets/%s then run "
            "`huggingface-cli login` (or export HF_TOKEN) and retry.",
            repo_id, e, repo_id,
        )
        raise

    tars = [f for f in all_files if f.endswith(".tar")]
    if not tars:
        log.error(
            "No .tar shards found in %s (repo may have been migrated to parquet). "
            "Available top-level files: %s",
            repo_id, sorted({f.split('/')[0] for f in all_files})[:20],
        )
        raise SystemExit(1)

    resolved: list[str] = []
    for shard in requested:
        # Match by exact path, basename, or substring (e.g. "bal_train09" or "bal_train09.tar")
        cand = [f for f in tars if f == shard or Path(f).name == shard or shard in f]
        if not cand:
            log.error(
                "Requested AudioSet shard %r not found in %s.\n"
                "Available .tar shards (first 20): %s",
                shard, repo_id, tars[:20],
            )
            raise SystemExit(1)
        if len(cand) > 1:
            log.warning("Multiple matches for %r, using %s", shard, cand[0])
        resolved.append(cand[0])
    return resolved


def download_audioset(workdir: Path, shards: list[str]) -> None:
    """Download AudioSet shards, extract, resample to 16 kHz wav.

    `agkphysics/AudioSet` is a gated HuggingFace dataset; you must
    `huggingface-cli login` (or set HF_TOKEN) before running this. We use
    `hf_hub_download` so the token is sent automatically, and we resolve the
    requested shard names against the actual repo file list (the layout has
    moved between `data/*.tar` and `data/{balanced,unbalanced}/*.tar` in the
    past).
    """
    import datasets
    from huggingface_hub import hf_hub_download

    raw_dir = workdir / "audioset"
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_dir = workdir / "audioset_16k"
    out_dir.mkdir(parents=True, exist_ok=True)

    repo_id = "agkphysics/AudioSet"
    _section(f"AudioSet ({len(shards)} shard(s)) -> {out_dir}")
    resolved = _resolve_audioset_shards(repo_id, shards)

    for shard_path in resolved:
        local_name = Path(shard_path).name
        tar_path = raw_dir / local_name
        if not tar_path.exists() or tar_path.stat().st_size < 1024:
            print(f"  download  {shard_path}")
            cached = hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=shard_path)
            try:
                os.symlink(cached, tar_path)
            except FileExistsError:
                pass
        else:
            print(f"  cached    {local_name}")

        print(f"  extract   {local_name}")
        with tarfile.open(tar_path) as tf:
            tf.extractall(raw_dir)

    flacs = sorted(str(p) for p in (raw_dir / "audio").rglob("*.flac"))
    if not flacs:
        log.warning("No FLAC files found in %s/audio after extraction.", raw_dir)
        return

    print(f"  resample  {len(flacs)} clips -> {out_dir}")
    ds = datasets.Dataset.from_dict({"audio": flacs})
    ds = ds.cast_column("audio", datasets.Audio(sampling_rate=TARGET_SR))
    for row in tqdm(ds, desc="audioset_16k", bar_format=_BAR_FMT, unit="clip"):
        name = Path(row["audio"]["path"]).name.replace(".flac", ".wav")
        target = out_dir / name
        if target.exists():
            continue
        _write_wav16(target, row["audio"]["array"])


def download_fma(out_dir: Path, n_hours: float) -> None:
    """Download FMA "small" subset clips, resample to 16 kHz wav.

    FMA small clips are ~30 s each, so n_clips = n_hours * 3600 / 30.
    """
    import datasets

    out_dir.mkdir(parents=True, exist_ok=True)
    n_clips = int(n_hours * 3600 / 30)
    _section(f"FMA -> {out_dir} ({n_clips} clips, ~{n_hours:.1f} h)")

    ds = datasets.load_dataset("rudraml/fma", name="small", split="train", streaming=True)
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
    parser.add_argument("--audioset-shards", nargs="+", default=["bal_train09.tar"],
                        help="AudioSet tar shard filenames to fetch from HuggingFace "
                             "(default: bal_train09.tar — one ~1 GB shard).")
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
            download_audioset(args.workdir, args.audioset_shards)
        if not args.skip_fma:
            download_fma(args.workdir / "fma", args.fma_hours)
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
