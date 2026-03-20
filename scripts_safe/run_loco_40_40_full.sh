#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
. "$SCRIPT_DIR/common/hf_env.sh"

cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

GPUS="${GPUS:-1}"
TASK0_BS="${TASK0_BS:-8}"
STAGE1_BS="${STAGE1_BS:-8}"
TASK1_BS="${TASK1_BS:-6}"
NUM_CLUSTERS="${NUM_CLUSTERS:-30}"

bash scripts_safe/prepare_repro.sh

run_train() {
  local config_path="$1"
  local batch_size="$2"

  if [[ "$GPUS" -gt 1 ]]; then
    bash tools/dist_train_gps.sh "$config_path" "$GPUS" --amp \
      --cfg-options "train_dataloader.batch_size=$batch_size" "optim_wrapper.optimizer.batch_size_per_gpu=$batch_size"
  else
    python tools/train_gps.py "$config_path" --amp \
      --cfg-options "train_dataloader.batch_size=$batch_size" "optim_wrapper.optimizer.batch_size_per_gpu=$batch_size"
  fi
}

python script/cpr_unknown_pseudo_label.py --setting LOCO_COCO --task 40+40 --stage 0 --num_clusters "$NUM_CLUSTERS"
run_train configs/loco_40_40/yolo_iod_loco_coco_40_40_task0.py "$TASK0_BS"

python script/pseudo_label_sc.py --setting LOCO_COCO --task 40+40 --stage 1
run_train configs/loco_40_40/yolo_iod_loco_coco_40_40_stage1.py "$STAGE1_BS"
run_train configs/loco_40_40/yolo_iod_loco_coco_40_40_task1.py "$TASK1_BS"

FINAL_CKPT="work_dirs/yolo_iod_loco_coco_40_40_task1/epoch_20.pth"
if [[ -f "$FINAL_CKPT" ]]; then
  python tools/test.py configs/loco_40_40/yolo_iod_loco_coco_40_40_task1.py "$FINAL_CKPT"
fi
