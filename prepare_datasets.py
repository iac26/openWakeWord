"""Download + prepare background-noise / RIR datasets for openWakeWord training.

Mirrors the dataset-prep cells of `notebooks/automatic_model_training.ipynb`,
but as a CLI you can run on a headless GPU server.

Outputs (under --workdir, default ./training_workspace):
    mit_rirs/                MIT environmental impulse responses (16 kHz wav)
    audioset/                downloaded raw AudioSet tar shards + extracted flacs
    audioset_16k/            AudioSet resampled to 16 kHz wav
    fma/                     FMA "small" clips resampled to 16 kHz wav

Avoids `datasets.Audio(...)` (which pulls in torchcodec / has fragile bytes
handling under streaming) and `datasets.load_dataset(streaming=True)`. Files
are pulled directly via `huggingface_hub.snapshot_download` and decoded with
`librosa.load` (which uses soundfile for wav/flac and the ffmpeg binary for
mp3).
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np
import scipy.io.wavfile
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("prepare_datasets")

TARGET_SR = 16000


def _resample_to_16k(path) -> np.ndarray:
    """Decode an audio file (path) to mono 16 kHz float32."""
    import librosa

    arr, _ = librosa.load(str(path), sr=TARGET_SR, mono=True)
    return arr


def _write_wav16(path: Path, arr: np.ndarray) -> None:
    scipy.io.wavfile.write(path, TARGET_SR, (np.asarray(arr) * 32767).astype(np.int16))


def download_mit_rirs(out_dir: Path) -> None:
    """Download MIT IR Survey from HuggingFace (already 16 kHz wav, just copy)."""
    from huggingface_hub import snapshot_download

    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("MIT RIRs -> %s (snapshot_download from HuggingFace)", out_dir)

    cache_dir = Path(snapshot_download(
        repo_id="davidscripka/MIT_environmental_impulse_responses",
        repo_type="dataset",
        allow_patterns=["16khz/*", "data/*"],
    ))

    # Repo layout has audio under 16khz/ (and sometimes data/)
    src_files: list[Path] = []
    for sub in ("16khz", "data"):
        src_files.extend((cache_dir / sub).rglob("*.wav") if (cache_dir / sub).exists() else [])
    if not src_files:
        log.warning("No .wav files found under %s", cache_dir)
        return

    log.info("Copying %d RIR files -> %s", len(src_files), out_dir)
    for src in tqdm(src_files, desc="mit_rirs"):
        target = out_dir / src.name
        if target.exists() and target.stat().st_size > 0:
            continue
        # Already 16 kHz mono wav per the dataset; copy as-is.
        shutil.copyfile(src, target)


def download_audioset(workdir: Path, shards: list[str]) -> None:
    """Download given AudioSet shards from HuggingFace, extract, resample to 16 kHz wav.

    Defaults to a single ~1 GB shard ("bal_train09.tar"). For real training pass
    multiple shards (or the full balanced+unbalanced set) — they cover ~50 GB+.
    Browse names at https://huggingface.co/datasets/agkphysics/AudioSet/tree/main/data.
    """
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

    flacs = sorted((raw_dir / "audio").rglob("*.flac"))
    if not flacs:
        log.warning("No FLAC files found in %s/audio after extraction.", raw_dir)
        return

    log.info("Resampling %d AudioSet clips to 16 kHz wav -> %s", len(flacs), out_dir)
    for flac_path in tqdm(flacs, desc="audioset_16k"):
        target = out_dir / (flac_path.stem + ".wav")
        if target.exists():
            continue
        arr = _resample_to_16k(flac_path)
        _write_wav16(target, arr)


def download_fma(out_dir: Path, n_hours: float) -> None:
    """Download FMA "small" subset and resample to 16 kHz wav.

    Pulls mp3 audio files from the HF mirror `rudraml/fma` (config "small").
    FMA small clips are ~30 s each, so n_clips = n_hours * 3600 / 30.
    """
    from huggingface_hub import snapshot_download

    out_dir.mkdir(parents=True, exist_ok=True)
    n_clips = int(n_hours * 3600 / 30)
    log.info("FMA -> %s (target %d clips, ~%.1f hours)", out_dir, n_clips, n_hours)

    log.info("Pulling FMA small subset from HuggingFace (this can be ~7 GB)")
    cache_dir = Path(snapshot_download(
        repo_id="rudraml/fma",
        repo_type="dataset",
        allow_patterns=["fma_small/*", "small/*", "*.mp3"],
    ))

    mp3s = sorted(cache_dir.rglob("*.mp3"))
    if not mp3s:
        log.warning("No .mp3 files found under %s — checking for parquet shards.", cache_dir)
        # Fall back: dataset may be stored as parquet with embedded audio
        _download_fma_via_parquet(cache_dir, out_dir, n_clips)
        return

    mp3s = mp3s[:n_clips]
    log.info("Resampling %d FMA mp3s -> %s", len(mp3s), out_dir)
    for mp3_path in tqdm(mp3s, desc="fma"):
        target = out_dir / (mp3_path.stem + ".wav")
        if target.exists():
            continue
        try:
            arr = _resample_to_16k(mp3_path)
        except Exception as e:  # corrupt mp3, skip
            log.warning("Skipping %s: %s", mp3_path.name, e)
            continue
        _write_wav16(target, arr)


def _download_fma_via_parquet(cache_dir: Path, out_dir: Path, n_clips: int) -> None:
    """Fallback path when FMA repo stores audio as parquet shards with bytes."""
    import io

    import pyarrow.parquet as pq

    parquets = sorted(cache_dir.rglob("*.parquet"))
    if not parquets:
        log.error("No mp3s and no parquets found in %s; cannot prepare FMA.", cache_dir)
        return

    written = 0
    pbar = tqdm(total=n_clips, desc="fma")
    for pq_path in parquets:
        if written >= n_clips:
            break
        table = pq.read_table(pq_path)
        # Expect an 'audio' column of struct<bytes: binary, path: string>
        col = table.column("audio").to_pylist()
        for entry in col:
            if written >= n_clips:
                break
            audio_bytes = entry.get("bytes") if isinstance(entry, dict) else None
            audio_path = entry.get("path") if isinstance(entry, dict) else None
            if not audio_bytes:
                continue
            name = (Path(audio_path).stem if audio_path else f"clip_{written}") + ".wav"
            target = out_dir / name
            if target.exists():
                written += 1
                pbar.update(1)
                continue
            try:
                arr = _resample_to_16k(io.BytesIO(audio_bytes))
            except Exception as e:
                log.warning("Skipping clip %d: %s", written, e)
                continue
            _write_wav16(target, arr)
            written += 1
            pbar.update(1)
    pbar.close()


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
