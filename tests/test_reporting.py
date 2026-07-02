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


def test_best_table_keeps_metrics_from_the_auc_selected_run(tmp_path: Path) -> None:
    source = tmp_path / "runs.csv"
    pd.DataFrame(
        [
            {
                "dataset": "HDFS",
                "variant": "SIGNET-fair",
                "seed": 11,
                "auc": 0.90,
                "ap": 0.60,
            },
            {
                "dataset": "HDFS",
                "variant": "SIGNET-fair",
                "seed": 22,
                "auc": 0.80,
                "ap": 0.95,
            },
        ]
    ).to_csv(source, index=False)

    summary = summarize_runs(
        [source],
        ["auc", "ap"],
        aggregation="best",
        selection_metric="auc",
    )

    assert summary.loc[0, "selected_seed"] == 11
    assert summary.loc[0, "auc_best"] == 0.90
    assert summary.loc[0, "ap_best"] == 0.60
    output = write_latex_table(
        summary, tmp_path / "best.tex", metrics=["auc", "ap"]
    )
    text = output.read_text(encoding="utf-8")
    assert "0.9000" in text
    assert "0.6000" in text
    assert "\\pm" not in text
