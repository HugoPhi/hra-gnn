from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import apply_overrides, load_config, validate_config
from .data import CSVGraphDataset, load_dataset
from .diagnostics import run_diagnostics
from .experiments import run_experiment_suite
from .interop import export_tu_dataset
from .plotting import (
    plot_ablation,
    plot_main_comparison,
    plot_relation_diagnostics,
    plot_data_diagnostics,
    plot_sensitivity,
    plot_training_history,
)
from .preprocessing import prepare_adfa_ld, prepare_hdfs
from .reporting import DEFAULT_METRICS, summarize_runs, write_latex_table
from .trainer import Trainer, evaluate_checkpoint


def _config(arguments: argparse.Namespace) -> dict:
    config = apply_overrides(load_config(arguments.config), arguments.set)
    validate_config(config)
    return config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unified HRA-GNN reproduction entry point"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in ("train", "evaluate", "test", "data-info", "diagnose"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--config", required=True)
        subparser.add_argument(
            "--set", action="append", default=[], help="YAML override: key=value"
        )
        if command in {"evaluate", "test"}:
            subparser.add_argument("--checkpoint", required=True)
            subparser.add_argument("--split", default="test")

    experiment = subparsers.add_parser("experiment")
    experiment.add_argument("--suite", required=True)
    experiment.add_argument(
        "--force", action="store_true", help="rerun completed suite jobs"
    )

    table = subparsers.add_parser("table")
    table.add_argument("--input", action="append", required=True)
    table.add_argument("--output", required=True)
    table.add_argument("--summary-csv")
    table.add_argument("--metrics", nargs="+", default=list(DEFAULT_METRICS))

    prepare = subparsers.add_parser("prepare-data")
    prepare.add_argument("--kind", required=True, choices=("hdfs", "adfa-ld"))
    prepare.add_argument("--input", required=True)
    prepare.add_argument("--labels")
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--max-graphs", type=int)

    export = subparsers.add_parser("export-tu")
    export.add_argument("--config", required=True)
    export.add_argument("--output", required=True)
    export.add_argument("--name", required=True)
    export.add_argument(
        "--set", action="append", default=[], help="YAML override: key=value"
    )

    plot = subparsers.add_parser("plot")
    plot.add_argument(
        "--kind",
        required=True,
        choices=(
            "comparison",
            "ablation",
            "sensitivity",
            "training",
            "relations",
            "diagnostics",
        ),
    )
    plot.add_argument("--input", required=True)
    plot.add_argument("--output", required=True)
    return parser


def main() -> None:
    parser = build_parser()
    arguments = parser.parse_args()

    if arguments.command == "train":
        summary = Trainer(_config(arguments)).train()
        print(json.dumps(summary, indent=2))
    elif arguments.command in {"evaluate", "test"}:
        summary = evaluate_checkpoint(
            _config(arguments), arguments.checkpoint, arguments.split
        )
        print(json.dumps(summary, indent=2))
    elif arguments.command == "data-info":
        config = _config(arguments)
        dataset = load_dataset(config["dataset"])
        if isinstance(dataset, CSVGraphDataset):
            labels = [dataset.labels[graph_id] for graph_id in dataset.graph_ids]
        else:
            labels = [dataset[index].label for index in range(len(dataset))]
        first = dataset[0]
        print(
            json.dumps(
                {
                    "graphs": len(dataset),
                    "normal_graphs": labels.count(0),
                    "anomaly_graphs": labels.count(1),
                    "feature_dim": first.x.shape[1],
                    "first_graph_nodes": first.num_nodes,
                    "first_graph_edges": first.num_edges,
                },
                indent=2,
            )
        )
    elif arguments.command == "diagnose":
        root = run_diagnostics(_config(arguments))
        print(f"Wrote diagnostics to {root.resolve()}")
    elif arguments.command == "experiment":
        root, rows = run_experiment_suite(arguments.suite, resume=not arguments.force)
        print(f"Wrote {len(rows)} runs to {root}")
    elif arguments.command == "table":
        summary = summarize_runs(arguments.input, arguments.metrics)
        if arguments.summary_csv:
            summary_path = Path(arguments.summary_csv)
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary.to_csv(summary_path, index=False)
        output = write_latex_table(
            summary,
            arguments.output,
            metrics=arguments.metrics,
        )
        print(f"Wrote {output.resolve()}")
    elif arguments.command == "prepare-data":
        if arguments.kind == "hdfs":
            if not arguments.labels:
                parser.error("prepare-data --kind hdfs requires --labels")
            output = prepare_hdfs(
                arguments.input,
                arguments.labels,
                arguments.output,
                seed=arguments.seed,
                max_graphs=arguments.max_graphs,
            )
        else:
            output = prepare_adfa_ld(
                arguments.input,
                arguments.output,
                seed=arguments.seed,
                max_graphs=arguments.max_graphs,
            )
        print(f"Wrote prepared dataset to {output.resolve()}")
    elif arguments.command == "export-tu":
        output = export_tu_dataset(
            _config(arguments),
            arguments.output,
            arguments.name,
        )
        print(f"Wrote TU Dataset to {output.resolve()}")
    elif arguments.command == "plot":
        functions = {
            "comparison": plot_main_comparison,
            "ablation": plot_ablation,
            "sensitivity": plot_sensitivity,
            "training": plot_training_history,
            "relations": plot_relation_diagnostics,
            "diagnostics": plot_data_diagnostics,
        }
        functions[arguments.kind](arguments.input, arguments.output)
        print(f"Wrote {Path(arguments.output).resolve()}")
