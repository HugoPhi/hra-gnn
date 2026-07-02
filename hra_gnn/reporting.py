from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_METRICS = (
    "auc",
    "ap",
    "precision_at_budget",
    "recall_at_budget",
    "tpr_at_fpr",
    "f1",
    "mcc",
)

DISPLAY_NAMES = {
    "auc": "AUROC",
    "ap": "AP",
    "precision_at_budget": r"P@1\%",
    "recall_at_budget": r"R@1\%",
    "tpr_at_fpr": r"TPR@1\%FPR",
    "f1": "F1",
    "mcc": "MCC",
}


def _escape(value: object) -> str:
    return (
        str(value)
        .replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
        .replace("#", r"\#")
    )


def summarize_runs(
    inputs: Iterable[str | Path],
    metrics: Iterable[str] = DEFAULT_METRICS,
) -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in inputs]
    if not frames:
        raise ValueError("At least one result CSV is required")
    runs = pd.concat(frames, ignore_index=True)
    method_column = "variant" if "variant" in runs else "run_name"
    missing = {"dataset", method_column} - set(runs.columns)
    if missing:
        raise ValueError(f"Result CSV is missing columns: {sorted(missing)}")
    available = [metric for metric in metrics if metric in runs]
    if not available:
        raise ValueError("None of the requested metrics are present")
    runs = runs.rename(columns={method_column: "method"})
    grouped = runs.groupby(["method", "dataset"], dropna=False)[available].agg(
        ["mean", "std", "count"]
    )
    grouped.columns = [f"{metric}_{stat}" for metric, stat in grouped.columns]
    return grouped.reset_index()


def write_latex_table(
    summary: pd.DataFrame,
    output: str | Path,
    *,
    metrics: Iterable[str] = DEFAULT_METRICS,
    caption: str = "不同模型在多个数据集上的异常检测结果",
    label: str = "tab:multi_dataset_results",
) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    datasets = summary["dataset"].drop_duplicates().tolist()
    metrics = [
        metric
        for metric in metrics
        if f"{metric}_mean" in summary and f"{metric}_std" in summary
    ]
    columns = "l" + "c" * (len(datasets) * len(metrics))
    lines = [
        r"% 需要 \usepackage{booktabs,graphicx,rotating}",
        r"\begin{sidewaystable*}[t]",
        r"\centering",
        r"\small",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{{columns}}}",
        r"\toprule",
    ]
    first_header = ["模型"]
    for dataset in datasets:
        first_header.append(
            rf"\multicolumn{{{len(metrics)}}}{{c}}{{{_escape(dataset)}}}"
        )
    lines.append(" & ".join(first_header) + r" \\")
    cmidrules = []
    start = 2
    for _ in datasets:
        end = start + len(metrics) - 1
        cmidrules.append(rf"\cmidrule(lr){{{start}-{end}}}")
        start = end + 1
    lines.append(" ".join(cmidrules))
    second_header = [""] + [
        DISPLAY_NAMES.get(metric, _escape(metric))
        for _dataset in datasets
        for metric in metrics
    ]
    lines.append(" & ".join(second_header) + r" \\")
    lines.append(r"\midrule")

    methods = summary["method"].drop_duplicates().tolist()
    indexed = summary.set_index(["method", "dataset"])
    for method in methods:
        row = [_escape(method)]
        for dataset in datasets:
            if (method, dataset) not in indexed.index:
                row.extend(["--"] * len(metrics))
                continue
            values = indexed.loc[(method, dataset)]
            for metric in metrics:
                mean = values[f"{metric}_mean"]
                std = values[f"{metric}_std"]
                if pd.isna(mean):
                    row.append("--")
                elif pd.isna(std):
                    row.append(f"{mean:.4f}")
                else:
                    row.append(rf"${mean:.4f}\pm{std:.4f}$")
        lines.append(" & ".join(row) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{sidewaystable*}",
            "",
        ]
    )
    output.write_text("\n".join(lines), encoding="utf-8")
    return output
