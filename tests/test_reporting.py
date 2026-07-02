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
    assert "数据集 & 模型 & AUROC & AP" in text
    assert "\\multirow{1}{*}{HDFS}" in text
    assert "\\begin{sidewaystable" not in text


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
    assert summary.loc[0, "method"] == "SIGNET"
    assert summary.loc[0, "auc_best"] == 0.90
    assert summary.loc[0, "ap_best"] == 0.60
    output = write_latex_table(
        summary, tmp_path / "best.tex", metrics=["auc", "ap"]
    )
    text = output.read_text(encoding="utf-8")
    assert "0.9000" in text
    assert "0.6000" in text
    assert "\\pm" not in text


def test_reimplemented_method_aliases_are_merged(tmp_path: Path) -> None:
    source = tmp_path / "runs.csv"
    pd.DataFrame(
        [
            {
                "dataset": "TraceLog",
                "variant": "DeepTraLog",
                "seed": 11,
                "auc": 0.70,
            },
            {
                "dataset": "HDFS",
                "variant": "DeepTraLog-adapted",
                "seed": 22,
                "auc": 0.80,
            },
        ]
    ).to_csv(source, index=False)

    summary = summarize_runs([source], ["auc"], aggregation="best")

    assert set(summary["method"]) == {"DeepTraLog"}
    assert set(summary["dataset"]) == {"TraceLog", "HDFS"}


def test_display_names_drop_protocol_suffixes_and_note_adaptations(
    tmp_path: Path,
) -> None:
    source = tmp_path / "runs.csv"
    pd.DataFrame(
        [
            {
                "dataset": "HDFS",
                "variant": "GLADMamba-fair",
                "seed": 11,
                "auc": 0.80,
            },
            {
                "dataset": "HDFS",
                "variant": "HGT-reimplemented",
                "seed": 11,
                "auc": 0.70,
            },
        ]
    ).to_csv(source, index=False)
    summary = summarize_runs([source], ["auc"], aggregation="best")
    output = write_latex_table(summary, tmp_path / "results.tex", metrics=["auc"])
    text = output.read_text(encoding="utf-8")

    assert set(summary["method"]) == {"GLADMamba", "HGT"}
    assert "-fair" not in text
    assert "-reimplemented" not in text
    assert "统一 GraphSample 版本" in text
    assert "MUSE 因稠密邻接复杂度未在 FlowGraph 上运行" in text
