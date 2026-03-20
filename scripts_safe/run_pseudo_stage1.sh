#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

bash scripts_safe/prepare_repro.sh

python script/pseudo_label_sc.py --setting COCO --task 40+40 --stage 1
