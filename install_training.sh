#!/usr/bin/env bash
# Bootstrap a Python training environment for openWakeWord (ONNX only).
#
# This script does TWO things:
#   1) `uv sync` to create .venv from pyproject.toml + uv.lock
#      (torch + torchaudio come from the pinned CUDA index).
#   2) Download the data files needed for clip generation, augmentation,
#      and training: melspec/embedding/VAD ONNXes, piper-sample-generator
#      + voice model, the precomputed feature shards, and (optionally)
#      MIT-RIRs / AudioSet / FMA via prepare_datasets.py.
#
# It does NOT touch system packages by default. On Pop!_OS / Ubuntu with
# nvidia-dkms, an `apt install build-essential` will pull in linux-headers
# and trigger a long nvidia-dkms rebuild — avoid unless you know you need
# missing system packages. Set INSTALL_SYSTEM_PKGS=1 to opt in.
#
# Override via env vars:
#
#   WORKDIR=$PWD/training_workspace
#   INSTALL_SYSTEM_PKGS=0      # 1 = run apt-get for system deps (fresh boxes only)
#   DOWNLOAD_FEATURES=1        # 1 = download ~7 GB of pre-computed negative features
#   DOWNLOAD_PIPER=1           # 1 = clone piper-sample-generator + voice model
#   DOWNLOAD_DATASETS=1        # 1 = fetch RIR/AudioSet/FMA via prepare_datasets.py
#   AUDIOSET_SHARDS="bal_train09.tar"  # space-separated AudioSet tar shard names
#   FMA_HOURS=1                # hours of FMA audio (bump for real training)
#
# After running:
#   ./run_train.sh --training_config <your.yml> \
#       --generate_clips --augment_clips --train_model

set -euo pipefail

WORKDIR="${WORKDIR:-$PWD/training_workspace}"
INSTALL_SYSTEM_PKGS="${INSTALL_SYSTEM_PKGS:-0}"
DOWNLOAD_FEATURES="${DOWNLOAD_FEATURES:-1}"
DOWNLOAD_PIPER="${DOWNLOAD_PIPER:-1}"
DOWNLOAD_DATASETS="${DOWNLOAD_DATASETS:-1}"
AUDIOSET_SHARDS="${AUDIOSET_SHARDS:-bal_train09.tar}"
FMA_HOURS="${FMA_HOURS:-1}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==> openWakeWord training bootstrap (ONNX only)"
echo "    repo:        $REPO_ROOT"
echo "    workdir:     $WORKDIR"

# 1. System packages (Linux only, opt-in) -----------------------------------
# By default, do NOT touch apt — assume dev tools are already present. On
# Pop!_OS / Ubuntu with nvidia-dkms, an `apt install build-essential` will
# pull in linux-headers and trigger a 5-10 min nvidia-dkms rebuild. Set
# INSTALL_SYSTEM_PKGS=1 only when bootstrapping a fresh machine that's
# missing build-essential/ffmpeg/sox/espeak-ng/libspeexdsp-dev/libsndfile1.
if [[ "$INSTALL_SYSTEM_PKGS" == "1" && "$(uname -s)" == "Linux" ]]; then
    echo "==> Installing system packages (sudo apt-get)"
    sudo apt-get update -qq
    sudo apt-get install -y --no-install-recommends \
        build-essential git wget curl ca-certificates \
        libspeexdsp-dev espeak-ng sox libsndfile1 ffmpeg
else
    echo "==> Skipping system package install (set INSTALL_SYSTEM_PKGS=1 if needed)"
fi

# 2. Python environment via uv sync -----------------------------------------
# pyproject.toml's [tool.uv.sources] pins torch + torchaudio to the
# PyTorch CUDA 12.4 index, so the wheels include the bundled NVIDIA libs
# that onnxruntime-gpu dlopens at runtime. Both inference and training
# deps live in the main `dependencies` table — no extras needed.
if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: 'uv' is not installed. Install it first:"
    echo "    curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi
echo "==> uv sync (creates/updates $REPO_ROOT/.venv)"
uv sync --project "$REPO_ROOT"

# 3. Workspace + base feature/VAD models (ONNX) -----------------------------
mkdir -p "$WORKDIR"
RES_DIR="$REPO_ROOT/openwakeword/resources/models"
mkdir -p "$RES_DIR"

declare -a BASE_MODELS=(
    "embedding_model.onnx"
    "melspectrogram.onnx"
    "silero_vad.onnx"
)
echo "==> Downloading base feature/VAD models (ONNX) -> $RES_DIR"
for f in "${BASE_MODELS[@]}"; do
    if [[ ! -s "$RES_DIR/$f" ]]; then
        wget -q --show-progress -O "$RES_DIR/$f" \
            "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/$f"
    else
        echo "    [skip] $f already present"
    fi
done

# 4. Piper sample generator (synthetic positive clips) ----------------------
if [[ "$DOWNLOAD_PIPER" == "1" ]]; then
    PIPER_DIR="$WORKDIR/piper-sample-generator"
    if [[ ! -d "$PIPER_DIR" ]]; then
        echo "==> Cloning piper-sample-generator"
        # dscripka's fork keeps the flat `generate_samples.py` layout that
        # openwakeword's train.py expects (rhasspy's was restructured).
        git clone --depth 1 https://github.com/dscripka/piper-sample-generator "$PIPER_DIR"
    fi
    # dscripka's fork expects the v1.0.0 LibriTTS model. The v2.0.0 medium
    # checkpoint from the rhasspy main repo doesn't load with this fork.
    PIPER_VOICE="$PIPER_DIR/models/en-us-libritts-high.pt"
    if [[ ! -s "$PIPER_VOICE" ]]; then
        echo "==> Downloading Piper voice model (~250 MB)"
        mkdir -p "$PIPER_DIR/models"
        wget -q --show-progress -O "$PIPER_VOICE" \
            "https://github.com/rhasspy/piper-sample-generator/releases/download/v1.0.0/en-us-libritts-high.pt"
    fi
fi

# 5. Pre-computed negative features (~7 GB) ---------------------------------
if [[ "$DOWNLOAD_FEATURES" == "1" ]]; then
    FEAT_DIR="$WORKDIR/features"
    mkdir -p "$FEAT_DIR"
    declare -a FEAT_FILES=(
        "openwakeword_features_ACAV100M_2000_hrs_16bit.npy"
        "validation_set_features.npy"
    )
    echo "==> Downloading pre-computed negative features -> $FEAT_DIR"
    for f in "${FEAT_FILES[@]}"; do
        if [[ ! -s "$FEAT_DIR/$f" ]]; then
            wget -q --show-progress -O "$FEAT_DIR/$f" \
                "https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/$f"
        else
            echo "    [skip] $f already present"
        fi
    done
fi

# 6. Background datasets (RIR + AudioSet + FMA) ------------------------------
if [[ "$DOWNLOAD_DATASETS" == "1" ]]; then
    echo "==> Preparing background datasets (RIR + AudioSet + FMA)"
    "$REPO_ROOT/.venv/bin/python" "$REPO_ROOT/prepare_datasets.py" \
        --workdir "$WORKDIR" \
        --audioset-shards $AUDIOSET_SHARDS \
        --fma-hours "$FMA_HOURS"
fi

# 7. GPU sanity check --------------------------------------------------------
echo "==> Verifying PyTorch CUDA"
"$REPO_ROOT/.venv/bin/python" - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY

cat <<EOF

==> Done.

Run training with:
    ./run_train.sh --training_config configs/<your_model>.yml \\
        --generate_clips --augment_clips --train_model

run_train.sh prefers .venv-train if it exists, otherwise .venv (the uv
default created by this script).
EOF
