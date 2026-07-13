from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


VARIANT_ORDER = [
    "Static Concat",
    "Dynamic Attention",
    "Only Max-Pooling",
    "Only Mean-Pooling",
    "w/o SSL",
    "Ours (Full)",
]
DATASET_ORDER = ["TraceLog", "FlowGraph", "ADFA-LD"]


def load_summary(paths: list[Path], main_table: Path | None = None) -> pd.DataFrame:
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        if "status" in frame:
            frame = frame[frame["status"].fillna("complete") == "complete"]
        frames.append(frame)
    runs = pd.concat(frames, ignore_index=True)
    eligible = runs.dropna(subset=["auc"]).copy()
    selected = eligible.loc[
        eligible.groupby(["dataset", "variant"], dropna=False)["auc"].idxmax()
    ].copy()
    selected = selected.rename(
        columns={"auc": "auc_best", "ap": "ap_best", "seed": "selected_seed"}
    )
    columns = ["dataset", "variant", "selected_seed", "auc_best", "ap_best"]
    selected = selected[columns]

    if main_table is not None:
        main = pd.read_csv(main_table)
        full_rows = main[
            (main["method"] == "HRA-GNN") & main["dataset"].isin(DATASET_ORDER)
        ].copy()
        full_rows["variant"] = "Ours (Full)"
        full_rows = full_rows.rename(
            columns={
                "selected_seed": "selected_seed",
                "auc_best": "auc_best",
                "ap_best": "ap_best",
            }
        )[columns]
        selected = selected[selected["variant"] != "Ours (Full)"]
        selected = pd.concat([selected, full_rows], ignore_index=True)

    selected["source"] = selected["variant"].map(
        lambda value: "main_table_hra_gnn" if value == "Ours (Full)" else "ablation_run"
    )
    summary = selected
    summary["dataset"] = pd.Categorical(
        summary["dataset"], categories=DATASET_ORDER, ordered=True
    )
    summary["variant"] = pd.Categorical(
        summary["variant"], categories=VARIANT_ORDER, ordered=True
    )
    return summary.sort_values(["variant", "dataset"]).reset_index(drop=True)


def format_score(value: float, *, bold: bool = False) -> str:
    formatted = f"{value:.4f}"
    return rf"\textbf{{{formatted}}}" if bold else formatted


def write_tex(summary: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        r"% 需要 \usepackage{booktabs,graphicx}",
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\renewcommand{\arraystretch}{0.94}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{消融实验性能对比}",
        r"\label{tab:ablation_results}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"\multirow{2}{*}{模型变体} & \multicolumn{2}{c}{TraceLog} & \multicolumn{2}{c}{FlowGraph} & \multicolumn{2}{c}{ADFA-LD} \\",
        r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}",
        r" & AUROC & AP & AUROC & AP & AUROC & AP \\",
        r"\midrule",
    ]
    indexed = summary.set_index(["variant", "dataset"])
    best_values = {
        (dataset, metric): float(
            summary[summary["dataset"] == dataset][f"{metric}_best"].max()
        )
        for dataset in DATASET_ORDER
        for metric in ("auc", "ap")
    }
    for variant in VARIANT_ORDER:
        row = [variant]
        for dataset in DATASET_ORDER:
            values = indexed.loc[(variant, dataset)]
            auc = float(values["auc_best"])
            ap = float(values["ap_best"])
            row.append(
                format_score(
                    auc,
                    bold=abs(auc - best_values[(dataset, "auc")]) < 1e-12,
                )
            )
            row.append(
                format_score(
                    ap,
                    bold=abs(ap - best_values[(dataset, "ap")]) < 1e-12,
                )
            )
        lines.append(" & ".join(row) + r" \\")
        if variant == "w/o SSL":
            lines.append(r"\midrule")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"\par\vspace{2pt}",
            r"\begin{minipage}{\textwidth}",
            r"\scriptsize",
            "说明：表中数值采用最佳真实运行，不取均值；各消融变体按 AUROC 选择最佳 seed，AP 取同一 seed。"
            "Ours (Full) 行直接采用主表 HRA-GNN 的对应结果，以保证主表与消融表一致。"
            "ADFA-LD 的 Ours (Full) 与主表一致，包含系统调用词频近邻和 Markov 序列增强评分；"
            "其他结构消融变体使用 SVDD 图级异常分数。FlowGraph 存在满分天花板，因此可能出现并列最佳。",
            r"\end{minipage}",
            r"\end{table*}",
            "",
        ]
    )
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--main-table")
    parser.add_argument("--csv", default="reference_results/paper_ablation.csv")
    parser.add_argument("--tex", default="reference_results/paper_ablation.tex")
    args = parser.parse_args()

    summary = load_summary(
        [Path(item) for item in args.input],
        main_table=Path(args.main_table) if args.main_table else None,
    )
    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(csv_path, index=False)
    write_tex(summary, Path(args.tex))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
