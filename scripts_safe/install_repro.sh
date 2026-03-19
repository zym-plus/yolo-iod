#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python executable not found: $PYTHON_BIN" >&2
  exit 1
fi

if ! "$PYTHON_BIN" -c "import torch" >/dev/null 2>&1; then
  cat <<'EOF' >&2
PyTorch is not installed in the current environment.
Install a torch/torchvision build matching your CUDA driver first.
Example for CUDA 11.8:
  python -m pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
EOF
  exit 1
fi

"$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel
"$PYTHON_BIN" -m pip install openmim
"$PYTHON_BIN" -m mim install "mmengine>=0.10.0,<0.11.0" "mmcv>=2.0.0rc4,<2.1.0" "mmdet>=3.0.0,<4.0.0"
"$PYTHON_BIN" -m pip install -r requirements/repro_requirements.txt
"$PYTHON_BIN" -m pip install -v -e .
"$PYTHON_BIN" -m pip install -v -e third_party/mmyolo

echo "Reproduction dependencies installed successfully."
