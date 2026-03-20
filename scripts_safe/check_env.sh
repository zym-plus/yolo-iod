#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "==== Python ===="
python -V

echo "==== Torch/CUDA ===="
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"

echo "==== GPU ===="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  echo "nvidia-smi not found"
fi

echo "==== Key Files ===="
for path in \
  "weights/x_stage1-62b674ad.pth" \
  "data/texts/coco_class_unknown.json" \
  "data/texts/unknown_class_texts.json" \
  "data/coco/annotations/instances_train2017.json" \
  "data/coco/annotations/instances_val2017.json"; do
  if [[ -e "$path" ]]; then
    echo "[OK] $path"
  else
    echo "[MISSING] $path"
  fi
done

echo "==== data/coco/annotations ===="
ls data/coco/annotations || true
