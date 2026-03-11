#!/usr/bin/env bash
set -e
echo "==== Python ===="
python -V
echo "==== Torch/CUDA ===="
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
echo "==== GPU ===="
nvidia-smi
echo "==== weights ===="
ls weights || true
echo "==== data/coco/annotations ===="
ls data/coco/annotations || true
