from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DATASETS = ["TraceLog", "FlowGraph", "ADFA-LD"]
METHOD_ORDER = [
    "DeepTraLog",
    "GLocalKD",
    "HGT",
    "OCHetGCN",
    "SIGNET",
    "CVTGAD",
    "MUSE",
    "GLADMamba",
    "HRGCN",
    "HRA-GNN",
]


def format_value(value: float | None, *, bold: bool = False, underline: bool = False) -> str:
    if value is None or pd.isna(value):
        return "--"
    text = f"{float(value):.4f}"
    if bold:
        return rf"\textbf{{{text}}}"
    if underline:
        return rf"\underline{{{text}}}"
    return text


def rank_values(frame: pd.DataFrame, dataset: str, metric: str) -> list[float]:
    values = (
        frame[frame["dataset"] == dataset][metric]
        .dropna()
        .astype(float)
        .drop_duplicates()
        .sort_values(ascending=False)
        .tolist()
    )
    return values[:2]


def write_table(frame: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    frame = frame[frame["dataset"].isin(DATASETS)].copy()
    indexed = frame.set_index(["method", "dataset"])
    ranks = {
        (dataset, metric): rank_values(frame, dataset, metric)
        for dataset in DATASETS
        for metric in ("auc_best", "ap_best")
    }
    lines = [
        r"% 需要 \usepackage{booktabs,graphicx,multirow}",
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\renewcommand{\arraystretch}{0.92}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{去除 HDFS 后的图异常检测主表结果（最佳运行）}",
        r"\label{tab:final_auroc_ap_results_no_hdfs}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"\multicolumn{1}{c}{\multirow{2}{*}{方法名称}} & \multicolumn{2}{c}{TraceLog} & \multicolumn{2}{c}{FlowGraph} & \multicolumn{2}{c}{ADFA-LD} \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}",
        r" & AUROC & AP & AUROC & AP & AUROC & AP \\",
        r"\midrule",
    ]
    for method in METHOD_ORDER:
        row = [rf"\textbf{{{method}}}" if method == "HRA-GNN" else method]
        for dataset in DATASETS:
            if (method, dataset) not in indexed.index:
                row.extend(["--", "--"])
                continue
            values = indexed.loc[(method, dataset)]
            for metric in ("auc_best", "ap_best"):
                value = float(values[metric])
                top = ranks[(dataset, metric)]
                row.append(
                    format_value(
                        value,
                        bold=bool(top) and abs(value - top[0]) < 1e-12,
                        underline=len(top) > 1 and abs(value - top[1]) < 1e-12,
                    )
                )
        lines.append(" & ".join(row) + r" \\")
        if method != METHOD_ORDER[-1]:
            lines.append(r"\addlinespace")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"\par\vspace{2pt}",
            r"\begin{minipage}{\textwidth}",
            r"\scriptsize",
            "说明：本表由四数据集主表删除 HDFS 列得到，所有数值与主表 CSV 完全同源。"
            "粗体和下划线分别表示同一数据集、同一指标的最佳和次佳结果，并列最佳同时加粗。",
            r"\end{minipage}",
            r"\end{table*}",
            "",
        ]
    )
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="reference_results/final_paper_auroc_ap_best.csv")
    parser.add_argument("--output", default="reference_results/final_paper_auroc_ap_best_no_hdfs.tex")
    args = parser.parse_args()

    frame = pd.read_csv(args.input)
    write_table(frame, Path(args.output))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
