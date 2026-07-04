#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

SEARCH="configs/tuning/tracelog_core_grid.yaml"
LOG_DIR="artifacts/logs/hyperparameter_search"
mkdir -p "${LOG_DIR}"

GPU_COUNT="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')"
if [[ "${GPU_COUNT}" -ge 2 ]]; then
  exec scripts/run_tracelog_tuning_dual_gpu.sh
fi

CUDA_VISIBLE_DEVICES=0 .venv/bin/python run.py tune \
  --search "${SEARCH}" \
  >"${LOG_DIR}/tracelog_core_grid_gpu0.log" 2>&1

.venv/bin/python run.py tune-merge --search "${SEARCH}"
