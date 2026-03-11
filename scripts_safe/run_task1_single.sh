#!/usr/bin/env bash
set -e
python tools/train_gps.py configs/40_40/yolo_iod_coco_40_40_task1.py \
  --amp \
  --cfg-options train_dataloader.batch_size=6 \
                optim_wrapper.optimizer.batch_size_per_gpu=6
