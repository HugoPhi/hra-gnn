from pathlib import Path

import pandas as pd

from hra_gnn.reporting import summarize_runs, write_latex_table


def test_latex_table_contains_models_datasets_and_metrics(tmp_path: Path) -> None:
    source = tmp_path / "runs.csv"
    pd.DataFrame(
        [
            {"dataset": "HDFS", "variant": "HRA-GNN", "auc": 0.9, "ap": 0.8},
            {"dataset": "HDFS", "variant": "HRA-GNN", "auc": 0.8, "ap": 0.7},
            {"dataset": "ADFA-LD", "variant": "HRGCN", "auc": 0.7, "ap": 0.6},
        ]
    ).to_csv(source, index=False)
    summary = summarize_runs([source], ["auc", "ap"])
    output = write_latex_table(
        summary, tmp_path / "results.tex", metrics=["auc", "ap"]
    )
    text = output.read_text(encoding="utf-8")
    assert "HRA-GNN" in text
    assert "ADFA-LD" in text
    assert "AUROC" in text
    assert "\\pm" in text
