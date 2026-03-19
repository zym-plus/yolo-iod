#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

bash scripts_safe/prepare_repro.sh

python tools/train_gps.py configs/40_40/yolo_iod_coco_40_40_task1.py \
  --amp \
  --cfg-options train_dataloader.batch_size=6 \
                optim_wrapper.optimizer.batch_size_per_gpu=6
