#!/usr/bin/env bash

if [[ "${HF_MIRROR_ENV_INITIALIZED:-0}" == "1" ]]; then
  return 0
fi

: "${HF_ENDPOINT:=https://hf-mirror.com}"
: "${HF_HOME:=$HOME/.cache/huggingface}"
: "${HF_HUB_CACHE:=$HF_HOME/hub}"
: "${HUGGINGFACE_HUB_CACHE:=$HF_HUB_CACHE}"
: "${TRANSFORMERS_CACHE:=$HF_HUB_CACHE}"
: "${HUGGINGFACE_HUB_ENDPOINT:=$HF_ENDPOINT}"

export HF_ENDPOINT
export HF_HOME
export HF_HUB_CACHE
export HUGGINGFACE_HUB_CACHE
export TRANSFORMERS_CACHE
export HUGGINGFACE_HUB_ENDPOINT
export HF_MIRROR_ENV_INITIALIZED=1

echo "[HF Mirror] HF_ENDPOINT=$HF_ENDPOINT"
echo "[HF Mirror] HF_HOME=$HF_HOME"
echo "[HF Mirror] HF_HUB_CACHE=$HF_HUB_CACHE"
