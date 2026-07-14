from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hra_gnn.augment import HeterogeneousAugmentor  # noqa: E402
from hra_gnn.config import load_config  # noqa: E402
from hra_gnn.data import load_dataset  # noqa: E402
from hra_gnn.graph import GraphSample  # noqa: E402


METHODS = ("edge_perturbation", "edge_addition", "node_type_swap", "edge_type_swap")


def edge_set(graph: GraphSample) -> set[tuple[int, int, int]]:
    source, target = graph.edge_index.cpu()
    edge_type = graph.edge_type.cpu()
    return {
        (int(source[index]), int(target[index]), int(edge_type[index]))
        for index in range(graph.num_edges)
    }


def duplicate_edge_fraction(graph: GraphSample) -> float:
    if graph.num_edges == 0:
        return 0.0
    return 1.0 - len(edge_set(graph)) / graph.num_edges


def self_loop_fraction(graph: GraphSample) -> float:
    if graph.num_edges == 0:
        return 0.0
    source, target = graph.edge_index
    return float((source == target).float().mean().item())


def tensor_changed(left: torch.Tensor, right: torch.Tensor) -> bool:
    return left.shape != right.shape or bool((left != right).any().item())


def audit_graph(
    graph: GraphSample,
    augmented: GraphSample,
    *,
    method: str,
    allowed_schema: set[int],
    num_node_types: int,
    num_edge_types: int,
    relation_schema: str,
) -> dict[str, float | int | str]:
    original_edges = edge_set(graph)
    augmented_edges = edge_set(augmented)
    union = original_edges | augmented_edges
    intersection = original_edges & augmented_edges
    schema = set(
        augmented.canonical_relation_ids(
            num_node_types,
            num_edge_types,
            relation_schema=relation_schema,
        ).tolist()
    )
    changed = (
        tensor_changed(graph.edge_index, augmented.edge_index)
        or tensor_changed(graph.edge_type, augmented.edge_type)
        or tensor_changed(graph.node_type, augmented.node_type)
    )
    node_type_change_rate = 0.0
    if graph.node_type.shape == augmented.node_type.shape and graph.num_nodes > 0:
        node_type_change_rate = float(
            (graph.node_type != augmented.node_type).float().mean().item()
        )
    comparable_edge_types = min(graph.edge_type.numel(), augmented.edge_type.numel())
    edge_type_change_rate = 0.0
    if comparable_edge_types > 0:
        edge_type_change_rate = float(
            (
                graph.edge_type[:comparable_edge_types]
                != augmented.edge_type[:comparable_edge_types]
            )
            .float()
            .mean()
            .item()
        )
    return {
        "method": method,
        "changed": int(changed),
        "node_count": graph.num_nodes,
        "edge_count": graph.num_edges,
        "edge_count_delta": augmented.num_edges - graph.num_edges,
        "edge_jaccard": len(intersection) / len(union) if union else 1.0,
        "node_type_change_rate": node_type_change_rate,
        "edge_type_change_rate": edge_type_change_rate,
        "schema_valid": int(schema.issubset(allowed_schema)),
        "duplicate_edge_fraction": duplicate_edge_fraction(augmented),
        "self_loop_fraction": self_loop_fraction(augmented),
    }


def audit_config(config_path: Path, *, sample_size: int, seed: int) -> pd.DataFrame:
    config = load_config(config_path)
    dataset_config = config["dataset"]
    dataset = load_dataset(dataset_config)
    sample_size = min(sample_size, len(dataset))
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:sample_size].tolist()
    relation_schema = config["model"].get("relation_schema", "canonical")
    augmentor = HeterogeneousAugmentor(
        num_node_types=int(dataset_config["num_node_types"]),
        num_edge_types=int(dataset_config["num_edge_types"]),
        edge_perturbation_rate=float(
            config["augmentation"].get("edge_perturbation_rate", 0.1)
        ),
        edge_addition_rate=float(config["augmentation"].get("edge_addition_rate", 0.1)),
        node_type_swap_rate=float(
            config["augmentation"].get("node_type_swap_rate", 0.1)
        ),
        edge_type_swap_rate=float(config["augmentation"].get("edge_type_swap_rate", 0.1)),
        methods=list(METHODS),
        preserve_observed_schema=bool(
            config["augmentation"].get("preserve_observed_schema", True)
        ),
        seed=seed,
    )
    rows: list[dict[str, float | int | str]] = []
    for index in indices:
        graph = dataset[int(index)]
        allowed_schema = set(
            graph.canonical_relation_ids(
                int(dataset_config["num_node_types"]),
                int(dataset_config["num_edge_types"]),
                relation_schema=relation_schema,
            ).tolist()
        )
        for method in METHODS:
            augmented = augmentor.augment(graph, method).graph
            rows.append(
                {
                    "dataset": dataset_config["name"],
                    "graph_id": graph.graph_id,
                    **audit_graph(
                        graph,
                        augmented,
                        method=method,
                        allowed_schema=allowed_schema,
                        num_node_types=int(dataset_config["num_node_types"]),
                        num_edge_types=int(dataset_config["num_edge_types"]),
                        relation_schema=relation_schema,
                    ),
                }
            )
    return pd.DataFrame(rows)


def write_summary(raw: pd.DataFrame, output: Path) -> pd.DataFrame:
    summary = (
        raw.groupby(["dataset", "method"], as_index=False)
        .agg(
            graphs=("graph_id", "nunique"),
            changed_graph_rate=("changed", "mean"),
            mean_edge_delta=("edge_count_delta", "mean"),
            mean_edge_jaccard=("edge_jaccard", "mean"),
            mean_node_type_change_rate=("node_type_change_rate", "mean"),
            mean_edge_type_change_rate=("edge_type_change_rate", "mean"),
            schema_valid_rate=("schema_valid", "mean"),
            duplicate_edge_fraction=("duplicate_edge_fraction", "mean"),
            self_loop_fraction=("self_loop_fraction", "mean"),
        )
        .sort_values(["dataset", "method"])
    )
    summary.to_csv(output, index=False)
    return summary


def plot_summary(summary: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    datasets = summary["dataset"].drop_duplicates().tolist()
    fig, axes = plt.subplots(len(datasets), 1, figsize=(9, 2.5 * len(datasets)))
    if len(datasets) == 1:
        axes = [axes]
    for axis, dataset in zip(axes, datasets, strict=True):
        frame = summary[summary["dataset"] == dataset]
        axis.bar(frame["method"], frame["changed_graph_rate"], color="#4C78A8")
        axis.set_ylim(0, 1.05)
        axis.set_title(dataset)
        axis.set_ylabel("changed graph rate")
        axis.tick_params(axis="x", rotation=20)
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        action="append",
        default=[
            "configs/tracelog.yaml",
            "configs/flowgraph.yaml",
            "configs/adfa_ld.yaml",
        ],
    )
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument(
        "--raw-output",
        default="reference_results/ssl_augmentation_audit_raw.csv",
    )
    parser.add_argument(
        "--summary-output",
        default="reference_results/ssl_augmentation_audit_summary.csv",
    )
    parser.add_argument(
        "--figure-output",
        default="doc/assets/ssl_augmentation/ssl_augmentation_changed_rate.svg",
    )
    args = parser.parse_args()

    frames = [
        audit_config(Path(path), sample_size=args.sample_size, seed=args.seed)
        for path in args.config
    ]
    raw = pd.concat(frames, ignore_index=True)
    raw_path = Path(args.raw_output)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(raw_path, index=False)
    summary = write_summary(raw, Path(args.summary_output))
    plot_summary(summary, Path(args.figure_output))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
