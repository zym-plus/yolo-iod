#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
. "$SCRIPT_DIR/common/hf_env.sh"

cd "$ROOT_DIR"

TASK_SPLIT="${TASK_SPLIT:-40+40}"
COCO_ROOT="${COCO_ROOT:-$ROOT_DIR/data/coco}"
WEIGHTS_DIR="${WEIGHTS_DIR:-$ROOT_DIR/weights}"
PRETRAIN_PATH="${PRETRAIN_PATH:-$WEIGHTS_DIR/x_stage1-62b674ad.pth}"
PRETRAIN_URL="${PRETRAIN_URL:-$HF_ENDPOINT/wondervictor/YOLO-World-V2.1/resolve/main/x_stage1-62b674ad.pth}"
FORCE_REGEN_SPLITS="${FORCE_REGEN_SPLITS:-0}"
SKIP_WEIGHT_DOWNLOAD="${SKIP_WEIGHT_DOWNLOAD:-0}"

mkdir -p "$WEIGHTS_DIR" "$ROOT_DIR/work_dirs" "$ROOT_DIR/data" "$ROOT_DIR/data/texts" "$COCO_ROOT" "$COCO_ROOT/annotations"

if [[ -f "$ROOT_DIR/data/texts/coco_class_unknown.json" && ! -f "$ROOT_DIR/data/texts/unknown_class_texts.json" ]]; then
  cp "$ROOT_DIR/data/texts/coco_class_unknown.json" "$ROOT_DIR/data/texts/unknown_class_texts.json"
fi

if [[ -d "$ROOT_DIR/datasets/coco/loco_annotations" && ! -d "$COCO_ROOT/loco_annotations" ]]; then
  mkdir -p "$COCO_ROOT"
  cp -rn "$ROOT_DIR/datasets/coco/loco_annotations" "$COCO_ROOT/"
fi

download_file() {
  local url="$1"
  local output="$2"

  if command -v wget >/dev/null 2>&1; then
    wget -O "$output" "$url"
    return
  fi

  if command -v curl >/dev/null 2>&1; then
    curl -L "$url" -o "$output"
    return
  fi

  python -c "import sys, urllib.request; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])" "$url" "$output"
}

if [[ ! -f "$PRETRAIN_PATH" ]]; then
  if [[ "$SKIP_WEIGHT_DOWNLOAD" == "1" ]]; then
    echo "Missing pretrained checkpoint: $PRETRAIN_PATH" >&2
    exit 1
  fi
  download_file "$PRETRAIN_URL" "$PRETRAIN_PATH"
fi

for required_path in \
  "$COCO_ROOT/train2017" \
  "$COCO_ROOT/val2017" \
  "$COCO_ROOT/annotations/instances_train2017.json" \
  "$COCO_ROOT/annotations/instances_val2017.json"; do
  if [[ ! -e "$required_path" ]]; then
    echo "Missing required COCO path: $required_path" >&2
    echo "Please place the original COCO2017 dataset under $COCO_ROOT before running experiments." >&2
    exit 1
  fi
done

COCO_SPLIT_DIR="$COCO_ROOT/annotations/${TASK_SPLIT}(order)"
if [[ "$FORCE_REGEN_SPLITS" == "1" || ! -f "$COCO_SPLIT_DIR/instances_train2017_part0.json" || ! -f "$COCO_SPLIT_DIR/instances_train2017_part1.json" || ! -f "$COCO_SPLIT_DIR/instances_val2017_part0.json" || ! -f "$COCO_SPLIT_DIR/instances_val2017_part1.json" || ! -f "$COCO_SPLIT_DIR/coco_class_texts_stage1.json" ]]; then
  python script/coco2017_split.py --pattern "$TASK_SPLIT"
fi

LOCO_SPLIT_DIR="$COCO_ROOT/loco_annotations/${TASK_SPLIT}(order)"
if [[ "$FORCE_REGEN_SPLITS" == "1" || ! -f "$LOCO_SPLIT_DIR/instances_train2017_part0.json" || ! -f "$LOCO_SPLIT_DIR/instances_train2017_part1.json" || ! -f "$LOCO_SPLIT_DIR/instances_val2017_part0.json" || ! -f "$LOCO_SPLIT_DIR/instances_val2017_part1.json" || ! -f "$LOCO_SPLIT_DIR/loco_class_texts_stage1.json" ]]; then
  python script/loco_coco_split.py --pattern "$TASK_SPLIT"
fi

echo "Reproduction assets are ready."
