#!/usr/bin/env bash
# Symlink generated WAVs from a smaller tier into a larger tier so the
# bigger model reuses (not regenerates) the smaller tier's clips.
#
# Layout assumed:
#   training_workspace/output/<from_model_name>/{positive,negative}_{train,test}/*.wav
#   training_workspace/output/<to_model_name>/{positive,negative}_{train,test}/      <-- empty or missing
#
# After this script runs, the destination dirs contain symlinks to all
# of the source WAVs. When you then run:
#   ./run_train.sh --training_config configs/<to>.yml --generate_clips ...
# train.py sees the symlinked clips, sees that count < n_samples, and
# generates only the *additional* WAVs needed to reach the larger tier's
# n_samples. Augmentation + feature extraction then runs over the union.
#
# Usage:
#   scripts/promote_clips.sh <from_model_name> <to_model_name>
#
# Example:
#   scripts/promote_clips.sh hey_ari_micro hey_ari_tiny

set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <from_model_name> <to_model_name>"
    exit 1
fi

FROM="$1"
TO="$2"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_BASE="$ROOT/training_workspace/output/$FROM"
DST_BASE="$ROOT/training_workspace/output/$TO"

if [[ ! -d "$SRC_BASE" ]]; then
    echo "ERROR: source dir does not exist: $SRC_BASE"
    exit 1
fi

for sub in positive_train positive_test negative_train negative_test; do
    src="$SRC_BASE/$sub"
    dst="$DST_BASE/$sub"
    if [[ ! -d "$src" ]]; then
        echo "  [skip] $src (missing)"
        continue
    fi
    mkdir -p "$dst"
    n=0
    for f in "$src"/*.wav; do
        [[ -e "$f" ]] || continue
        ln -sf "$f" "$dst/$(basename "$f")"
        n=$((n + 1))
    done
    echo "  $sub: linked $n clips"
done

echo "==> Done. Now run:"
echo "    ./run_train.sh --training_config configs/$TO.yml --generate_clips --augment_clips --train_model"
