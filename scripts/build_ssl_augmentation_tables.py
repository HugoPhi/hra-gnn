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


def fmt(value: float, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


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


def main() -> None:
    output_dir = Path("reference_results")
    write_plan_table(output_dir / "ssl_augmentation_experiment_plan.tex")
    write_audit_table(
        output_dir / "ssl_augmentation_audit_summary.csv",
        output_dir / "ssl_augmentation_audit_summary.tex",
    )
    print("Wrote SSL augmentation plan and audit tables")


if __name__ == "__main__":
    main()
