#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

LOG_DIR="artifacts/logs/hyperparameter_search"
mkdir -p "${LOG_DIR}"

run_search() {
  local search="$1"
  local name="$2"
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u run.py tune \
    --search "${search}" \
    >"${LOG_DIR}/${name}.log" 2>&1
  .venv/bin/python run.py tune-merge --search "${search}"
}

run_search configs/tuning/tracelog_lr_boundary.yaml tracelog_lr_boundary
run_search configs/tuning/tracelog_positive_structure.yaml tracelog_positive_structure
