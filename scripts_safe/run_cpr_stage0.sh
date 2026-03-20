#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

bash scripts_safe/prepare_repro.sh

python script/cpr_unknown_pseudo_label.py \
  --setting COCO \
  --task 40+40 \
  --stage 0 \
  --num_clusters 30
