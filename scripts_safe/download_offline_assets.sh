#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
. "$SCRIPT_DIR/common/hf_env.sh"

cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
CLIP_LOCAL_DIR="${CLIP_LOCAL_DIR:-$ROOT_DIR/pretrained_models/clip-vit-base-patch32}"
YOLOWORLD_URL="${YOLOWORLD_URL:-$HF_ENDPOINT/wondervictor/YOLO-World-V2.1/resolve/main/x_stage1-62b674ad.pth}"
YOLOWORLD_DST="${YOLOWORLD_DST:-$ROOT_DIR/weights/x_stage1-62b674ad.pth}"

if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "Please run this script inside an activated conda environment." >&2
  echo "Example: conda activate yolo-iod" >&2
  exit 1
fi

mkdir -p "$ROOT_DIR/weights" "$ROOT_DIR/pretrained_models"

if ! "$PYTHON_BIN" -c "import huggingface_hub" >/dev/null 2>&1; then
  "$PYTHON_BIN" -m pip install --upgrade pip
  "$PYTHON_BIN" -m pip install "huggingface_hub>=0.23.0,<1.0"
fi

if [[ ! -f "$YOLOWORLD_DST" ]]; then
  if command -v wget >/dev/null 2>&1; then
    wget -4 -O "$YOLOWORLD_DST" "$YOLOWORLD_URL"
  elif command -v curl >/dev/null 2>&1; then
    curl -4 -L "$YOLOWORLD_URL" -o "$YOLOWORLD_DST"
  else
    python -c "import sys, urllib.request; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])" "$YOLOWORLD_URL" "$YOLOWORLD_DST"
  fi
fi

"$PYTHON_BIN" -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='openai/clip-vit-base-patch32', local_dir=r'$CLIP_LOCAL_DIR', local_dir_use_symlinks=False)"

echo "Offline assets are ready:"
echo "  - $YOLOWORLD_DST"
echo "  - $CLIP_LOCAL_DIR"
