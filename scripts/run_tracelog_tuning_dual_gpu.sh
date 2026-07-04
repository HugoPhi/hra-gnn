#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

SEARCH="configs/tuning/tracelog_core_grid.yaml"
LOG_DIR="artifacts/logs/hyperparameter_search"
mkdir -p "${LOG_DIR}"

CUDA_VISIBLE_DEVICES=0 .venv/bin/python run.py tune \
  --search "${SEARCH}" --shard-index 0 --num-shards 2 \
  >"${LOG_DIR}/tracelog_core_grid_gpu0.log" 2>&1 &
PID0=$!

CUDA_VISIBLE_DEVICES=1 .venv/bin/python run.py tune \
  --search "${SEARCH}" --shard-index 1 --num-shards 2 \
  >"${LOG_DIR}/tracelog_core_grid_gpu1.log" 2>&1 &
PID1=$!

echo "GPU 0 PID: ${PID0}"
echo "GPU 1 PID: ${PID1}"
wait "${PID0}"
wait "${PID1}"

.venv/bin/python run.py tune-merge --search "${SEARCH}"
