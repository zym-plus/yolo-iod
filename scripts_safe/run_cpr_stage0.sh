#!/usr/bin/env bash
set -e
python script/cpr_unknown_pseudo_label.py \
  --setting COCO \
  --task 40+40 \
  --stage 0 \
  --num_clusters 30
