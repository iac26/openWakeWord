#!/usr/bin/env bash
# Bootstrap a Python 3.12 GPU training environment for openWakeWord (ONNX only).
#
# Defaults assume a Linux host with an NVIDIA GPU. Override via env vars:
#
#   PYTHON=python3.12          # which interpreter to seed the venv from
#   VENV=.venv                 # venv directory
#   CUDA=cu124                 # PyTorch CUDA build (cu121 / cu124 / cpu)
#   WORKDIR=$PWD/training_workspace
#   DOWNLOAD_FEATURES=1        # 1 = download ~7 GB of pre-computed negative features
#   DOWNLOAD_PIPER=1           # 1 = clone piper-sample-generator + voice model
#   DOWNLOAD_DATASETS=0        # 1 = also fetch RIR/AudioSet/FMA via prepare_datasets.py
#   AUDIOSET_SHARDS="bal_train09.tar"  # space-separated AudioSet tar shard names
#   FMA_HOURS=1                # hours of FMA audio (bump for real training)
#
# After running:
#   source $VENV/bin/activate
#   python -m openwakeword.train --training_config <your.yml> \
#       --generate_clips --augment_clips --train_model

set -euo pipefail

PYTHON="${PYTHON:-python3.12}"
VENV="${VENV:-.venv}"
CUDA="${CUDA:-cu124}"
WORKDIR="${WORKDIR:-$PWD/training_workspace}"
DOWNLOAD_FEATURES="${DOWNLOAD_FEATURES:-1}"
DOWNLOAD_PIPER="${DOWNLOAD_PIPER:-1}"
DOWNLOAD_DATASETS="${DOWNLOAD_DATASETS:-0}"
AUDIOSET_SHARDS="${AUDIOSET_SHARDS:-bal_train09.tar}"
FMA_HOURS="${FMA_HOURS:-1}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==> openWakeWord training bootstrap (ONNX only)"
echo "    repo:        $REPO_ROOT"
echo "    python:      $PYTHON"
echo "    venv:        $VENV"
echo "    cuda build:  $CUDA"
echo "    workdir:     $WORKDIR"

# 1. System packages (Linux only) -------------------------------------------
if [[ "$(uname -s)" == "Linux" ]]; then
    echo "==> Installing system packages (sudo apt-get)"
    sudo apt-get update -qq
    sudo apt-get install -y --no-install-recommends \
        build-essential git wget curl ca-certificates \
        libspeexdsp-dev espeak-ng sox libsndfile1 ffmpeg
fi

# 2. Python venv -------------------------------------------------------------
if [[ ! -x "$VENV/bin/python" ]]; then
    echo "==> Creating venv with $PYTHON"
    if command -v uv >/dev/null 2>&1; then
        uv venv --python "$PYTHON" "$VENV"
        PIP=("uv" "pip" "install" "--python" "$VENV/bin/python")
    else
        "$PYTHON" -m venv "$VENV"
        "$VENV/bin/python" -m pip install --upgrade pip
        PIP=("$VENV/bin/python" "-m" "pip" "install")
    fi
else
    echo "==> Reusing existing venv at $VENV"
    if command -v uv >/dev/null 2>&1; then
        PIP=("uv" "pip" "install" "--python" "$VENV/bin/python")
    else
        PIP=("$VENV/bin/python" "-m" "pip" "install")
    fi
fi

# 3. PyTorch (must be installed before extras to pick the right CUDA wheel) --
echo "==> Installing PyTorch ($CUDA)"
if [[ "$CUDA" == "cpu" ]]; then
    "${PIP[@]}" --index-url https://download.pytorch.org/whl/cpu torch torchaudio
else
    "${PIP[@]}" --index-url "https://download.pytorch.org/whl/$CUDA" torch torchaudio
fi

# 4. openWakeWord + training extras -----------------------------------------
echo "==> Installing openwakeword[training]"
"${PIP[@]}" -e "$REPO_ROOT[training]"

# 5. Workspace + base feature models (ONNX only) ----------------------------
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

# 6. Piper sample generator (synthetic positive clips) ----------------------
if [[ "$DOWNLOAD_PIPER" == "1" ]]; then
    PIPER_DIR="$WORKDIR/piper-sample-generator"
    if [[ ! -d "$PIPER_DIR" ]]; then
        echo "==> Cloning piper-sample-generator"
        # dscripka's fork keeps the flat `generate_samples.py` layout that
        # openwakeword's train.py expects (rhasspy's was restructured).
        git clone --depth 1 https://github.com/dscripka/piper-sample-generator "$PIPER_DIR"
    fi
    PIPER_VOICE="$PIPER_DIR/models/en_US-libritts_r-medium.pt"
    if [[ ! -s "$PIPER_VOICE" ]]; then
        echo "==> Downloading Piper voice model (~600 MB)"
        mkdir -p "$PIPER_DIR/models"
        wget -q --show-progress -O "$PIPER_VOICE" \
            "https://github.com/rhasspy/piper-sample-generator/releases/download/v2.0.0/en_US-libritts_r-medium.pt"
    fi
fi

# 7. Pre-computed negative features (~7 GB) ---------------------------------
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

# 8. Background datasets (RIR + AudioSet + FMA) ------------------------------
if [[ "$DOWNLOAD_DATASETS" == "1" ]]; then
    echo "==> Preparing background datasets (RIR + AudioSet + FMA)"
    "$VENV/bin/python" "$REPO_ROOT/prepare_datasets.py" \
        --workdir "$WORKDIR" \
        --audioset-shards $AUDIOSET_SHARDS \
        --fma-hours "$FMA_HOURS"
fi

# 9. GPU sanity check --------------------------------------------------------
echo "==> Verifying PyTorch CUDA"
"$VENV/bin/python" - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY

cat <<EOF

==> Done.

Next steps:
  1. source $VENV/bin/activate
  2. Edit a copy of examples/custom_model.yml, pointing
     - 'piper_sample_generator_path' at $WORKDIR/piper-sample-generator
     - 'feature_data_files' at $WORKDIR/features/openwakeword_features_ACAV100M_2000_hrs_16bit.npy
     - 'false_positive_validation_data_path' at $WORKDIR/features/validation_set_features.npy
     - 'background_paths' at directories of 16 kHz WAVs (AudioSet/FSD50k/FMA)
  3. Run:
       python -m openwakeword.train --training_config your_config.yml \\
         --generate_clips --augment_clips --train_model

You still need to provide background-noise audio yourself. Suggested sources:
  - AudioSet: https://huggingface.co/datasets/agkphysics/AudioSet
  - FSD50k:   https://zenodo.org/record/4060432
  - FMA:      https://github.com/mdeff/fma
EOF
