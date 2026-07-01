#!/usr/bin/env bash
set -euo pipefail

PYTHON="${PYTHON:-.venv/bin/python}"

for dataset in tracelog flowgraph; do
  "$PYTHON" run.py experiment \
    --suite "configs/experiments/baselines_${dataset}.yaml"
  "$PYTHON" run.py experiment \
    --suite "configs/experiments/paper_${dataset}.yaml"

  "$PYTHON" run.py plot \
    --kind comparison \
    --input "artifacts/results/suites/baselines_${dataset}/variant_summary.csv" \
    --output "artifacts/figures/paper/${dataset}_comparison.png"
  "$PYTHON" run.py plot \
    --kind ablation \
    --input "artifacts/results/suites/paper_${dataset}/variant_summary.csv" \
    --output "artifacts/figures/paper/${dataset}_ablation.png"
  "$PYTHON" run.py plot \
    --kind sensitivity \
    --input "artifacts/results/suites/paper_${dataset}/sweep_summary.csv" \
    --output "artifacts/figures/paper/${dataset}_sensitivity.png"
done

"$PYTHON" run.py diagnose --config configs/flowgraph.yaml
"$PYTHON" run.py plot \
  --kind diagnostics \
  --input artifacts/results/diagnostics/FlowGraph/graph_statistics.csv \
  --output artifacts/figures/diagnostics/flowgraph_size_distribution.png
