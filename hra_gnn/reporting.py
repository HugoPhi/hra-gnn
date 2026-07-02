from __future__ import annotations

import math
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

METHOD_ALIASES = {
    "DeepTraLog-adapted": "DeepTraLog",
    "DeepTraLog-reimplemented": "DeepTraLog",
    "GLocalKD-adapted": "GLocalKD",
    "GLocalKD-reimplemented": "GLocalKD",
    "HGT-reimplemented": "HGT",
    "OCHetGCN-reimplemented": "OCHetGCN",
    "SIGNET-fair": "SIGNET",
    "CVTGAD-fair": "CVTGAD",
    "MUSE-fair": "MUSE",
    "GLADMamba-fair": "GLADMamba",
}

PREFERRED_DATASET_ORDER = ("TraceLog", "FlowGraph", "HDFS", "ADFA-LD")

TABLE_ADAPTATION_NOTE = (
    "说明：所有方法沿用本项目预定义的数据划分、节点特征和评测指标；仅使用正常"
    "训练图进行单类学习，F1/MCC 的阈值由正常训练分数的 99% 分位数确定。"
    "SIGNET、CVTGAD、MUSE 和 GLADMamba 基于官方实现接入统一划分与固定正常"
    "参考评分；DeepTraLog、GLocalKD、HGT 和 OCHetGCN 为依据论文机制实现的"
    "统一 GraphSample 版本。ADFA-LD 使用 edge-only 关系模式控制高基数系统"
    "调用类型带来的关系规模。MUSE 因稠密邻接复杂度未在 FlowGraph 上运行。"
    "表中按 AUROC 选择最佳运行，其他指标均取自同一随机种子。本表汇总第一轮"
    "阶段性实验，旧直接基线与近期方法的受控采样上限尚未完全统一，最终公平主表"
    "需在统一预算下重跑。"
)


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
    *,
    aggregation: str = "mean_std",
    selection_metric: str = "auc",
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
    runs["method"] = runs["method"].replace(METHOD_ALIASES)
    if "status" in runs:
        runs = runs[runs["status"].fillna("complete") == "complete"]
    if aggregation == "best":
        if selection_metric not in runs:
            raise ValueError(
                f"Selection metric is missing from result CSV: {selection_metric}"
            )
        eligible = runs.dropna(subset=[selection_metric])
        if eligible.empty:
            raise ValueError(f"No finite {selection_metric} values are available")
        selected = eligible.loc[
            eligible.groupby(["method", "dataset"], dropna=False)[selection_metric]
            .idxmax()
            .tolist()
        ].copy()
        columns = ["method", "dataset"]
        if "seed" in selected:
            selected = selected.rename(columns={"seed": "selected_seed"})
            columns.append("selected_seed")
        for metric in available:
            selected = selected.rename(columns={metric: f"{metric}_best"})
            columns.append(f"{metric}_best")
        selected["selection_metric"] = selection_metric
        columns.append("selection_metric")
        return selected[columns].reset_index(drop=True)
    if aggregation != "mean_std":
        raise ValueError(f"Unsupported aggregation mode: {aggregation}")
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
    highlight_ranks: bool = False,
) -> Path:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    observed_datasets = summary["dataset"].drop_duplicates().tolist()
    datasets = [
        dataset for dataset in PREFERRED_DATASET_ORDER if dataset in observed_datasets
    ]
    datasets.extend(
        sorted(dataset for dataset in observed_datasets if dataset not in datasets)
    )
    best_mode = any(f"{metric}_best" in summary for metric in metrics)
    metrics = [
        metric
        for metric in metrics
        if (
            f"{metric}_best" in summary
            if best_mode
            else f"{metric}_mean" in summary and f"{metric}_std" in summary
        )
    ]
    columns = "ll" + "c" * len(metrics)
    lines = [
        r"% 需要 \usepackage{booktabs,graphicx,multirow}",
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\renewcommand{\arraystretch}{0.92}",
        r"\setlength{\tabcolsep}{3pt}",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{{columns}}}",
        r"\toprule",
    ]
    header = ["数据集", "模型"] + [
        DISPLAY_NAMES.get(metric, _escape(metric)) for metric in metrics
    ]
    lines.append(" & ".join(header) + r" \\")
    lines.append(r"\midrule")

    for dataset_index, dataset in enumerate(datasets):
        subset = summary[summary["dataset"] == dataset].sort_values("method")
        methods = subset["method"].tolist()
        indexed = subset.set_index("method")
        rank_values = {}
        if highlight_ranks:
            for metric in metrics:
                suffix = "best" if best_mode else "mean"
                values = subset[f"{metric}_{suffix}"].dropna().astype(float)
                rank_values[metric] = sorted(set(values.tolist()), reverse=True)[:2]
        for method_index, method in enumerate(methods):
            dataset_cell = (
                rf"\multirow{{{len(methods)}}}{{*}}{{{_escape(dataset)}}}"
                if method_index == 0
                else ""
            )
            row = [dataset_cell, _escape(method)]
            values = indexed.loc[method]
            for metric in metrics:
                if best_mode:
                    best = values[f"{metric}_best"]
                    formatted = "--" if pd.isna(best) else f"{best:.4f}"
                    if not pd.isna(best) and highlight_ranks:
                        ranks = rank_values[metric]
                        if ranks and math.isclose(
                            float(best), ranks[0], rel_tol=0.0, abs_tol=1e-12
                        ):
                            formatted = rf"\textbf{{{formatted}}}"
                        elif len(ranks) > 1 and math.isclose(
                            float(best), ranks[1], rel_tol=0.0, abs_tol=1e-12
                        ):
                            formatted = rf"\underline{{{formatted}}}"
                    row.append(formatted)
                    continue
                mean = values[f"{metric}_mean"]
                std = values[f"{metric}_std"]
                if pd.isna(mean):
                    row.append("--")
                elif pd.isna(std):
                    row.append(f"{mean:.4f}")
                else:
                    row.append(rf"${mean:.4f}\pm{std:.4f}$")
            lines.append(" & ".join(row) + r" \\")
        if dataset_index < len(datasets) - 1:
            lines.append(r"\midrule")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\begin{minipage}{\textwidth}",
            r"\scriptsize",
            _escape(TABLE_ADAPTATION_NOTE),
            r"\end{minipage}",
            r"\end{table*}",
            "",
        ]
    )
    output.write_text("\n".join(lines), encoding="utf-8")
    return output
