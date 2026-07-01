#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"

"$PYTHON" -m pytest
"$PYTHON" run.py data-info --config configs/flowgraph.yaml
"$PYTHON" run.py data-info --config configs/tracelog.yaml
"$PYTHON" run.py experiment --suite configs/experiments/smoke.yaml
"$PYTHON" run.py plot \
  --kind ablation \
  --input artifacts/results/suites/smoke/variant_summary.csv \
  --output artifacts/figures/smoke/variant_summary.png
