from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


PALETTE = ["#247BA0", "#F25F5C", "#70C1B3", "#50514F", "#FFB000", "#6A4C93"]


def _prepare_output(output: str | Path) -> Path:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def plot_main_comparison(input_csv: str | Path, output: str | Path) -> None:
    data = pd.read_csv(input_csv)
    if "variant" in data.columns:
        data = data.rename(
            columns={"variant": "method", "auc_mean": "auc", "ap_mean": "ap"}
        )
    required = {"method", "dataset", "auc", "ap"}
    if not required.issubset(data.columns):
        raise ValueError(
            "Comparison CSV must contain method/dataset/auc/ap or the "
            "variant-summary equivalents"
        )
    long = data.melt(
        id_vars=["method", "dataset"],
        value_vars=["auc", "ap"],
        var_name="metric",
        value_name="score",
    )
    sns.set_theme(style="whitegrid", font_scale=1.05)
    datasets = data["dataset"].drop_duplicates().tolist()
    figure, axes = plt.subplots(
        1,
        len(datasets),
        figsize=(6.5 * len(datasets), 4.8),
        sharey=True,
        squeeze=False,
    )
    for axis, dataset in zip(axes.flat, datasets):
        subset = data[data["dataset"] == dataset].reset_index(drop=True)
        if {"auc_std", "ap_std"}.issubset(subset.columns):
            positions = np.arange(len(subset))
            width = 0.36
            auc_error = np.vstack(
                [
                    np.minimum(subset["auc_std"], subset["auc"]),
                    np.minimum(subset["auc_std"], 1.0 - subset["auc"]),
                ]
            )
            ap_error = np.vstack(
                [
                    np.minimum(subset["ap_std"], subset["ap"]),
                    np.minimum(subset["ap_std"], 1.0 - subset["ap"]),
                ]
            )
            axis.bar(
                positions - width / 2,
                subset["auc"],
                width,
                yerr=auc_error,
                capsize=4,
                label="auc",
                color=PALETTE[0],
            )
            axis.bar(
                positions + width / 2,
                subset["ap"],
                width,
                yerr=ap_error,
                capsize=4,
                label="ap",
                color=PALETTE[1],
            )
            axis.set_xticks(positions, subset["method"])
            axis.legend(title="metric")
        else:
            long_subset = long[long["dataset"] == dataset]
            sns.barplot(
                data=long_subset,
                x="method",
                y="score",
                hue="metric",
                palette=PALETTE[:2],
                ax=axis,
            )
        axis.set_title(dataset)
        axis.set_xlabel("")
        axis.set_ylabel("Score")
        axis.set_ylim(0, 1.05)
        axis.tick_params(axis="x", rotation=35)
    figure.tight_layout()
    figure.savefig(_prepare_output(output), dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_ablation(input_csv: str | Path, output: str | Path) -> None:
    data = pd.read_csv(input_csv)
    long = data.melt(
        id_vars=["dataset", "variant"],
        value_vars=["auc_mean", "ap_mean"],
        var_name="metric",
        value_name="score",
    )
    long["metric"] = long["metric"].str.replace("_mean", "", regex=False).str.upper()
    sns.set_theme(style="whitegrid", font_scale=1.0)
    grid = sns.catplot(
        data=long,
        x="variant",
        y="score",
        hue="metric",
        col="dataset",
        kind="bar",
        palette=PALETTE[:2],
        height=4.6,
        aspect=1.25,
        sharey=True,
    )
    grid.set_axis_labels("", "Score")
    grid.set(ylim=(0, 1.05))
    for axis in grid.axes.flat:
        axis.tick_params(axis="x", rotation=35)
    grid.figure.tight_layout()
    grid.figure.savefig(_prepare_output(output), dpi=220, bbox_inches="tight")
    plt.close(grid.figure)


def plot_sensitivity(input_csv: str | Path, output: str | Path) -> None:
    data = pd.read_csv(input_csv)
    long = data.melt(
        id_vars=["dataset", "sweep", "sweep_value"],
        value_vars=["auc_mean", "ap_mean"],
        var_name="metric",
        value_name="score",
    )
    long["metric"] = long["metric"].str.replace("_mean", "", regex=False).str.upper()
    sns.set_theme(style="whitegrid", font_scale=1.0)
    grid = sns.relplot(
        data=long,
        x="sweep_value",
        y="score",
        hue="metric",
        col="dataset",
        row="sweep",
        kind="line",
        marker="o",
        palette=PALETTE[:2],
        height=3.8,
        aspect=1.3,
        facet_kws={"sharex": False},
    )
    grid.set_axis_labels("Parameter value", "Score")
    grid.set(ylim=(0, 1.05))
    grid.figure.tight_layout()
    grid.figure.savefig(_prepare_output(output), dpi=220, bbox_inches="tight")
    plt.close(grid.figure)


def plot_tuning_marginals(input_csv: str | Path, output: str | Path) -> None:
    data = pd.read_csv(input_csv)
    parameters = [column for column in data if "." in column]
    if not parameters or not {"auc", "ap"}.issubset(data.columns):
        raise ValueError(
            "Tuning CSV must contain dotted parameter columns plus auc and ap"
        )
    sns.set_theme(style="whitegrid", font_scale=1.0)
    figure, axes = plt.subplots(
        1,
        len(parameters),
        figsize=(5.2 * len(parameters), 4.4),
        squeeze=False,
    )
    for axis, parameter in zip(axes.flat, parameters):
        grouped = data.groupby(parameter, as_index=False)[["auc", "ap"]].mean()
        long = grouped.melt(
            id_vars=[parameter],
            value_vars=["auc", "ap"],
            var_name="metric",
            value_name="score",
        )
        sns.lineplot(
            data=long,
            x=parameter,
            y="score",
            hue="metric",
            marker="o",
            palette=PALETTE[:2],
            ax=axis,
        )
        axis.set_xlabel(parameter)
        axis.set_ylabel("Validation score")
        axis.set_ylim(0, 1.0)
        axis.legend(title="")
    figure.tight_layout()
    figure.savefig(_prepare_output(output), dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_training_history(input_csv: str | Path, output: str | Path) -> None:
    data = pd.read_csv(input_csv)
    columns = [
        column for column in ("loss", "svdd_loss", "ssl_loss") if column in data.columns
    ]
    long = data.melt(
        id_vars=["epoch"], value_vars=columns, var_name="loss", value_name="value"
    )
    sns.set_theme(style="whitegrid")
    figure, axis = plt.subplots(figsize=(7.5, 4.5))
    sns.lineplot(
        data=long,
        x="epoch",
        y="value",
        hue="loss",
        palette=PALETTE[: len(columns)],
        ax=axis,
    )
    figure.tight_layout()
    figure.savefig(_prepare_output(output), dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_relation_diagnostics(input_csv: str | Path, output: str | Path) -> None:
    data = pd.read_csv(input_csv)
    grouped = (
        data.groupby(["label", "relation_id"])[["mean_attention", "mean_deviation"]]
        .mean()
        .reset_index()
    )
    top_relations = (
        grouped.groupby("relation_id")["mean_deviation"].max().nlargest(15).index
    )
    grouped = grouped[grouped["relation_id"].isin(top_relations)]
    grouped["class"] = grouped["label"].map({0: "Normal", 1: "Anomaly"})
    sns.set_theme(style="whitegrid")
    figure, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for axis, metric in zip(axes, ("mean_attention", "mean_deviation")):
        sns.barplot(
            data=grouped,
            x="relation_id",
            y=metric,
            hue="class",
            palette=PALETTE[:2],
            ax=axis,
        )
        axis.set_ylabel(metric.replace("_", " ").title())
    axes[-1].set_xlabel("Canonical relation ID")
    figure.tight_layout()
    figure.savefig(_prepare_output(output), dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_data_diagnostics(input_csv: str | Path, output: str | Path) -> None:
    data = pd.read_csv(input_csv)
    data["class"] = data["label"].map({0: "Normal", 1: "Anomaly"})
    long = data.melt(
        id_vars=["class"],
        value_vars=["num_nodes", "num_edges"],
        var_name="statistic",
        value_name="value",
    )
    sns.set_theme(style="whitegrid")
    grid = sns.catplot(
        data=long,
        x="class",
        y="value",
        hue="class",
        col="statistic",
        kind="box",
        palette=PALETTE[:2],
        legend=False,
        height=4.5,
        aspect=1.1,
        sharey=False,
    )
    grid.set_axis_labels("", "Count")
    grid.figure.tight_layout()
    grid.figure.savefig(_prepare_output(output), dpi=220, bbox_inches="tight")
    plt.close(grid.figure)
