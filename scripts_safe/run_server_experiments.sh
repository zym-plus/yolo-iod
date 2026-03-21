#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
. "$SCRIPT_DIR/common/hf_env.sh"

cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

RUN_NAME="${RUN_NAME:-$(date '+%Y%m%d_%H%M%S')}"
LOG_ROOT="${LOG_ROOT:-$ROOT_DIR/work_dirs/server_runs/$RUN_NAME}"
EXPERIMENTS="${EXPERIMENTS:-coco,loco}"
SKIP_ENV_CHECK="${SKIP_ENV_CHECK:-0}"

mkdir -p "$LOG_ROOT"

run_step() {
  local step_name="$1"
  shift

  local log_file="$LOG_ROOT/${step_name}.log"
  echo "==== [$step_name] START $(date '+%F %T') ===="
  echo "Log: $log_file"
  "$@" 2>&1 | tee "$log_file"
  echo "==== [$step_name] END $(date '+%F %T') ===="
}

should_run() {
  local key="$1"
  [[ ",$EXPERIMENTS," == *",$key,"* ]]
}

echo "Run name: $RUN_NAME"
echo "Logs dir: $LOG_ROOT"
echo "Experiments: $EXPERIMENTS"
echo "GPUS: ${GPUS:-1}"

if [[ "$SKIP_ENV_CHECK" != "1" ]]; then
  run_step check_env bash scripts_safe/check_env.sh
fi

if should_run "coco"; then
  run_step coco_40_40 bash scripts_safe/run_coco_40_40_full.sh
fi

if should_run "loco"; then
  run_step loco_40_40 bash scripts_safe/run_loco_40_40_full.sh
fi

echo "All requested experiments finished successfully."
