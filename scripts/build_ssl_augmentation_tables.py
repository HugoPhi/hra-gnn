from __future__ import annotations

from pathlib import Path

import pandas as pd


PLAN_ROWS = [
    (
        "no\\_ssl",
        "关闭 SSL",
        "衡量没有增强分支时的单类 SVDD 基线。",
    ),
    (
        "edge\\_perturbation",
        "仅边扰动",
        "检验局部拓扑重连是否提升对关系稀释和异常邻域的敏感性。",
    ),
    (
        "edge\\_addition",
        "仅边添加",
        "检验额外正常模式边是否提升对冗余/伪连接异常的鲁棒性。",
    ),
    (
        "node\\_type\\_swap",
        "仅节点类型交换",
        "检验节点语义错配是否有助于学习类型一致性边界。",
    ),
    (
        "edge\\_type\\_swap",
        "仅边类型交换",
        "检验关系语义错配是否有助于学习边类型一致性边界。",
    ),
    (
        "topology\\_only",
        "边扰动+边添加",
        "汇总结构增强贡献，并与 type\\_only 区分。",
    ),
    (
        "type\\_only",
        "节点类型交换+边类型交换",
        "汇总语义增强贡献，并与 topology\\_only 区分。",
    ),
    (
        "full",
        "数据集默认完整增强",
        "检验论文最终配置是否优于单一增强和组合增强。",
    ),
]

RESULT_DATASETS = ["TraceLog", "FlowGraph", "ADFA-LD"]
RESULT_VARIANTS = [
    "no_ssl",
    "edge_perturbation",
    "edge_addition",
    "node_type_swap",
    "edge_type_swap",
    "topology_only",
    "type_only",
    "full",
    "all_four_diagnostic",
]
RESULT_VARIANT_LABELS = {
    "no_ssl": r"no\_ssl",
    "edge_perturbation": r"edge\_perturbation",
    "edge_addition": r"edge\_addition",
    "node_type_swap": r"node\_type\_swap",
    "edge_type_swap": r"edge\_type\_swap",
    "topology_only": r"topology\_only",
    "type_only": r"type\_only",
    "full": r"full",
    "all_four_diagnostic": r"all\_four",
}


def fmt(value: float, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def fmt_delta(value: float) -> str:
    if pd.isna(value):
        return "--"
    sign = "+" if float(value) >= 0 else ""
    return f"{sign}{fmt(value, 4)}"


def write_plan_table(output: Path) -> None:
    lines = [
        r"% 需要 \usepackage{booktabs,tabularx}",
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\renewcommand{\arraystretch}{1.05}",
        r"\caption{自监督增强策略独立作用实验设计}",
        r"\label{tab:ssl_augmentation_plan}",
        r"\begin{tabularx}{\textwidth}{lllX}",
        r"\toprule",
        r"编号 & 变体 & 增强策略 & 回答的问题 \\",
        r"\midrule",
    ]
    for index, (variant, strategy, question) in enumerate(PLAN_ROWS, start=1):
        lines.append(f"{index} & {variant} & {strategy} & {question} \\\\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabularx}",
            r"\end{table*}",
            "",
        ]
    )
    output.write_text("\n".join(lines), encoding="utf-8")


def write_audit_table(summary_path: Path, output: Path) -> None:
    frame = pd.read_csv(summary_path)
    method_order = {
        "edge_perturbation": 0,
        "edge_addition": 1,
        "node_type_swap": 2,
        "edge_type_swap": 3,
    }
    frame["_order"] = frame["method"].map(method_order)
    frame = frame.sort_values(["dataset", "_order"])
    lines = [
        r"% 需要 \usepackage{booktabs}",
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\renewcommand{\arraystretch}{0.98}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{自监督增强有效性审计（200 图抽样）}",
        r"\label{tab:ssl_augmentation_audit}",
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"数据集 & 增强策略 & 改变图比例 & 边数变化 & 边Jaccard & 节点类型变化 & 边类型变化 & Schema合法率 \\",
        r"\midrule",
    ]
    previous_dataset = None
    for _, row in frame.iterrows():
        dataset = row["dataset"]
        if previous_dataset is not None and dataset != previous_dataset:
            lines.append(r"\midrule")
        previous_dataset = dataset
        lines.append(
            " & ".join(
                [
                    str(dataset),
                    str(row["method"]).replace("_", r"\_"),
                    fmt(row["changed_graph_rate"]),
                    fmt(row["mean_edge_delta"], 1),
                    fmt(row["mean_edge_jaccard"]),
                    fmt(row["mean_node_type_change_rate"]),
                    fmt(row["mean_edge_type_change_rate"]),
                    fmt(row["schema_valid_rate"]),
                ]
            )
            + r" \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"\par\vspace{2pt}",
            r"\begin{minipage}{\textwidth}",
            r"\scriptsize",
            "说明：审计只执行增强变换，不训练模型。改变图比例表示增强后边索引、边类型或节点类型至少一项发生变化；"
            "边 Jaccard 越低表示结构改动越强。Schema 合法率按增强后关系是否仍属于原图已观察关系集合计算。",
            r"\end{minipage}",
            r"\end{table*}",
            "",
        ]
    )
    output.write_text("\n".join(lines), encoding="utf-8")


def load_ssl_result_runs(result_root: Path) -> pd.DataFrame:
    paths = {
        "TraceLog": result_root
        / "ssl_augmentation_tracelog"
        / "runs.csv",
        "FlowGraph": result_root
        / "ssl_augmentation_flowgraph"
        / "runs.csv",
        "ADFA-LD": result_root
        / "ssl_augmentation_adfa_ld"
        / "hybrid_rescore_runs.csv",
    }
    frames = []
    for dataset, path in paths.items():
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame["dataset"] = dataset
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    required = {"dataset", "variant", "seed", "auc", "ap"}
    missing = required - set(combined.columns)
    if missing:
        raise ValueError(f"Missing columns in SSL result runs: {sorted(missing)}")
    return combined


def summarize_best_runs(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, variant), group in frame.groupby(["dataset", "variant"], sort=False):
        auc_row = group.loc[group["auc"].astype(float).idxmax()]
        ap_row = group.loc[group["ap"].astype(float).idxmax()]
        rows.append(
            {
                "dataset": dataset,
                "variant": variant,
                "auc_best": float(auc_row["auc"]),
                "auc_seed": int(auc_row["seed"]),
                "ap_best": float(ap_row["ap"]),
                "ap_seed": int(ap_row["seed"]),
                "source": "ssl_augmentation_suite",
            }
        )
    return pd.DataFrame(rows)


def load_main_table_all_four(output_dir: Path) -> pd.DataFrame:
    path = output_dir / "final_paper_auroc_ap_best.csv"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    frame = frame[
        (frame["method"] == "HRA-GNN") & (frame["dataset"].isin(RESULT_DATASETS))
    ].copy()
    if frame.empty:
        return pd.DataFrame()
    return pd.DataFrame(
        {
            "dataset": frame["dataset"],
            "variant": "all_four_diagnostic",
            "auc_best": frame["auc_best"].astype(float),
            "auc_seed": frame["selected_seed"].astype(int),
            "ap_best": frame["ap_best"].astype(float),
            "ap_seed": frame["selected_seed"].astype(int),
            "source": "final_main_table",
        }
    )


def replace_with_main_table_all_four(
    frame: pd.DataFrame, output_dir: Path
) -> pd.DataFrame:
    override = load_main_table_all_four(output_dir)
    if override.empty:
        return frame
    kept = frame[frame["variant"] != "all_four_diagnostic"].copy()
    return pd.concat([kept, override], ignore_index=True)


def add_no_ssl_deltas(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, group in frame.groupby("dataset", sort=False):
        baseline = group[group["variant"] == "no_ssl"]
        if baseline.empty:
            group = group.copy()
            group["delta_auc"] = pd.NA
            group["delta_ap"] = pd.NA
        else:
            auc0 = float(baseline.iloc[0]["auc_best"])
            ap0 = float(baseline.iloc[0]["ap_best"])
            group = group.copy()
            group["delta_auc"] = group["auc_best"].astype(float) - auc0
            group["delta_ap"] = group["ap_best"].astype(float) - ap0
        rows.append(group)
    return pd.concat(rows, ignore_index=True)


def write_effect_table(result_root: Path, output: Path) -> bool:
    frame = load_ssl_result_runs(result_root)
    if frame.empty:
        return False
    frame = summarize_best_runs(frame)
    frame = replace_with_main_table_all_four(frame, output.parent)
    frame = add_no_ssl_deltas(frame)
    frame["_dataset_order"] = frame["dataset"].map(
        {name: index for index, name in enumerate(RESULT_DATASETS)}
    )
    frame["_variant_order"] = frame["variant"].map(
        {name: index for index, name in enumerate(RESULT_VARIANTS)}
    )
    frame = frame.sort_values(["_dataset_order", "_variant_order", "variant"])
    frame.drop(
        columns=[column for column in frame.columns if column.startswith("_")],
        errors="ignore",
    ).to_csv(
        output.with_suffix(".csv"),
        index=False,
    )

    lines = [
        r"% 需要 \usepackage{booktabs,multirow}",
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\renewcommand{\arraystretch}{0.98}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{不同自监督增强策略的独立作用}",
        r"\label{tab:ssl_augmentation_effects}",
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"数据集 & 变体 & AUROC$_{\max}$ & AP$_{\max}$ & $\Delta$AUROC & $\Delta$AP \\",
        r"\midrule",
    ]
    for dataset in RESULT_DATASETS:
        subset = frame[frame["dataset"] == dataset]
        if subset.empty:
            continue
        first = True
        for _, row in subset.iterrows():
            dataset_cell = rf"\multirow{{{len(subset)}}}{{*}}{{{dataset}}}" if first else ""
            first = False
            variant = RESULT_VARIANT_LABELS.get(row["variant"], str(row["variant"]).replace("_", r"\_"))
            lines.append(
                " & ".join(
                    [
                        dataset_cell,
                        variant,
                        fmt(row["auc_best"], 4),
                        fmt(row["ap_best"], 4),
                        fmt_delta(row["delta_auc"]),
                        fmt_delta(row["delta_ap"]),
                    ]
                )
                + r" \\"
            )
        lines.append(r"\midrule")
    if lines[-1] == r"\midrule":
        lines[-1] = r"\bottomrule"
    else:
        lines.append(r"\bottomrule")
    lines.extend(
        [
            r"\end{tabular}%",
            r"\par\vspace{2pt}",
            r"\begin{minipage}{\textwidth}",
            r"\scriptsize",
            r"说明：$\Delta$AUROC 和 $\Delta$AP 均以同一数据集上的 no\_ssl 为参照。"
            r"表中各变体按该变体已完成运行的最大 AUROC/AP 报告；"
            r"all\_four 行直接采用主表 HRA-GNN 最佳运行，以保证完整方法口径与主表一致。",
            r"\end{minipage}",
            r"\end{table*}",
            "",
        ]
    )
    output.write_text("\n".join(lines), encoding="utf-8")
    return True


def main() -> None:
    output_dir = Path("reference_results")
    write_plan_table(output_dir / "ssl_augmentation_experiment_plan.tex")
    write_audit_table(
        output_dir / "ssl_augmentation_audit_summary.csv",
        output_dir / "ssl_augmentation_audit_summary.tex",
    )
    wrote_effects = write_effect_table(
        output_dir / "ssl_augmentation_server",
        output_dir / "ssl_augmentation_effect_results.tex",
    )
    message = "Wrote SSL augmentation plan and audit tables"
    if wrote_effects:
        message += ", plus effect result table"
    print(message)


if __name__ == "__main__":
    main()
