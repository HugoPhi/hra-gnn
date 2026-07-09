from __future__ import annotations

import argparse
import json
from pathlib import Path

from .adfa_scoring import rescore_adfa_hybrid
from .checkpoint_runs import run_best_checkpoint_matrix
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
    plot_tuning_marginals,
    plot_training_history,
)
from .preprocessing import prepare_adfa_ld, prepare_hdfs
from .recent_baselines import (
    run_dual_view_fair,
    run_muse_fair,
    run_native_graph_fair,
    run_signet_fair,
)
from .recent_experiments import run_fair_matrix
from .reporting import (
    DEFAULT_METRICS,
    HRA_SEED_SWEEP_NOTE,
    TABLE_ADAPTATION_NOTE,
    summarize_runs,
    write_latex_table,
)
from .rescoring import rescore_calibrated_max
from .trainer import Trainer, evaluate_checkpoint
from .tuning import merge_hyperparameter_search, run_hyperparameter_search


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

    tune = subparsers.add_parser("tune")
    tune.add_argument("--search", required=True)
    tune.add_argument("--shard-index", type=int, default=0)
    tune.add_argument("--num-shards", type=int, default=1)
    tune.add_argument("--force", action="store_true")

    tune_merge = subparsers.add_parser("tune-merge")
    tune_merge.add_argument("--search", required=True)

    table = subparsers.add_parser("table")
    table.add_argument("--input", action="append", required=True)
    table.add_argument("--output", required=True)
    table.add_argument("--summary-csv")
    table.add_argument("--metrics", nargs="+", default=list(DEFAULT_METRICS))
    table.add_argument(
        "--aggregation",
        choices=("mean_std", "best"),
        default="mean_std",
        help="aggregate all runs or select one real run by its best metric",
    )
    table.add_argument(
        "--selection-metric",
        default="auc",
        help="metric used to select a run when --aggregation=best",
    )
    table.add_argument(
        "--highlight-ranks",
        action="store_true",
        help="bold the best and underline the second distinct value per dataset",
    )
    table.add_argument("--caption", default="不同模型在多个数据集上的异常检测结果")
    table.add_argument("--label", default="tab:multi_dataset_results")
    table.add_argument(
        "--note-profile",
        choices=("standard", "hra_seed_sweep"),
        default="standard",
    )

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

    fair = subparsers.add_parser("fair-baseline")
    fair.add_argument("--config", required=True)
    fair.add_argument(
        "--model",
        required=True,
        choices=(
            "signet",
            "cvtgad",
            "muse",
            "gladmamba",
            "himnet",
            "gladpro",
            "mssgad",
        ),
    )
    fair.add_argument("--external-root", default="external")
    fair.add_argument(
        "--set", action="append", default=[], help="YAML override: key=value"
    )

    fair_matrix = subparsers.add_parser("fair-matrix")
    fair_matrix.add_argument("--matrix", required=True)
    fair_matrix.add_argument("--force", action="store_true")

    best_checkpoints = subparsers.add_parser("best-checkpoints")
    best_checkpoints.add_argument("--matrix", required=True)
    best_checkpoints.add_argument("--force", action="store_true")

    rescore = subparsers.add_parser("calibrated-rescore")
    rescore.add_argument("--config", required=True)
    rescore.add_argument("--checkpoint", required=True)

    adfa_rescore = subparsers.add_parser("adfa-hybrid-rescore")
    adfa_rescore.add_argument("--config", required=True)
    adfa_rescore.add_argument("--checkpoint", required=True)
    adfa_rescore.add_argument("--fixed-test-ids")
    adfa_rescore.add_argument("--unigram-weight", type=float, default=0.5)
    adfa_rescore.add_argument("--markov-weight", type=float, default=0.25)
    adfa_rescore.add_argument("--markov-order", type=int, default=3)
    adfa_rescore.add_argument(
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
            "tuning",
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
    elif arguments.command == "tune":
        root, rows = run_hyperparameter_search(
            arguments.search,
            resume=not arguments.force,
            shard_index=arguments.shard_index,
            num_shards=arguments.num_shards,
        )
        print(f"Wrote {len(rows)} tuning runs to {root.resolve()}")
    elif arguments.command == "tune-merge":
        root, rows, ranking = merge_hyperparameter_search(arguments.search)
        print(
            f"Merged {len(rows)} runs and ranked {len(ranking)} trials "
            f"under {root.resolve()}"
        )
    elif arguments.command == "table":
        summary = summarize_runs(
            arguments.input,
            arguments.metrics,
            aggregation=arguments.aggregation,
            selection_metric=arguments.selection_metric,
        )
        if arguments.summary_csv:
            summary_path = Path(arguments.summary_csv)
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary.to_csv(summary_path, index=False)
        output = write_latex_table(
            summary,
            arguments.output,
            metrics=arguments.metrics,
            caption=arguments.caption,
            label=arguments.label,
            highlight_ranks=arguments.highlight_ranks,
            note=(
                HRA_SEED_SWEEP_NOTE
                if arguments.note_profile == "hra_seed_sweep"
                else TABLE_ADAPTATION_NOTE
            ),
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
    elif arguments.command == "fair-baseline":
        config = _config(arguments)
        if arguments.model == "signet":
            summary = run_signet_fair(config, external_root=arguments.external_root)
        elif arguments.model == "muse":
            summary = run_muse_fair(config, external_root=arguments.external_root)
        elif arguments.model in {"himnet", "gladpro", "mssgad"}:
            summary = run_native_graph_fair(config, architecture=arguments.model)
        else:
            summary = run_dual_view_fair(
                config,
                architecture=arguments.model,
                external_root=arguments.external_root,
            )
        print(json.dumps(summary, indent=2))
    elif arguments.command == "fair-matrix":
        root, rows = run_fair_matrix(arguments.matrix, resume=not arguments.force)
        complete = int((rows["status"] == "complete").sum())
        print(f"Wrote {len(rows)} runs ({complete} complete) to {root.resolve()}")
    elif arguments.command == "best-checkpoints":
        root, rows = run_best_checkpoint_matrix(
            arguments.matrix,
            resume=not arguments.force,
        )
        complete = int((rows["status"] == "complete").sum())
        print(f"Wrote {len(rows)} runs ({complete} complete) to {root.resolve()}")
    elif arguments.command == "calibrated-rescore":
        summary = rescore_calibrated_max(
            load_config(arguments.config),
            arguments.checkpoint,
        )
        print(json.dumps(summary, indent=2))
    elif arguments.command == "adfa-hybrid-rescore":
        summary = rescore_adfa_hybrid(
            _config(arguments),
            arguments.checkpoint,
            fixed_test_ids=arguments.fixed_test_ids,
            unigram_weight=arguments.unigram_weight,
            markov_weight=arguments.markov_weight,
            markov_order=arguments.markov_order,
        )
        print(json.dumps(summary, indent=2))
    elif arguments.command == "plot":
        functions = {
            "comparison": plot_main_comparison,
            "ablation": plot_ablation,
            "sensitivity": plot_sensitivity,
            "training": plot_training_history,
            "relations": plot_relation_diagnostics,
            "diagnostics": plot_data_diagnostics,
            "tuning": plot_tuning_marginals,
        }
        functions[arguments.kind](arguments.input, arguments.output)
        print(f"Wrote {Path(arguments.output).resolve()}")
