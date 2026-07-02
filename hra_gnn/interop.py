from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .data import load_dataset, load_provided_splits, make_splits


def _write_lines(path: Path, values: list[str]) -> None:
    path.write_text("\n".join(values) + "\n", encoding="utf-8")


def export_tu_dataset(config: dict, output: str | Path, name: str) -> Path:
    """Export the unified graph representation as a TU Dataset raw directory."""
    dataset = load_dataset(config["dataset"])
    split_config = config["dataset"].get("split", {})
    if split_config.get("mode", "generated") == "provided":
        splits = load_provided_splits(
            dataset,
            config["dataset"]["root"],
            train_file=split_config.get("train_file", "train_ids.txt"),
            validation_file=split_config.get("validation_file"),
            test_file=split_config.get("test_file", "test_ids.txt"),
        )
    else:
        splits = make_splits(
            dataset,
            train_normal_ratio=split_config.get("train_normal_ratio", 0.6),
            validation_normal_ratio=split_config.get(
                "validation_normal_ratio", 0.2
            ),
            validation_anomaly_ratio=split_config.get(
                "validation_anomaly_ratio", 0.0
            ),
            seed=int(config["training"].get("seed", 42)),
        )
    split_by_index = {
        index: split for split, indices in splits.items() for index in indices
    }

    raw = Path(output) / name / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    graph_indicator: list[str] = []
    graph_labels: list[str] = []
    node_labels: list[str] = []
    node_attributes: list[str] = []
    edge_rows: list[str] = []
    edge_labels: list[str] = []
    mapping: list[dict[str, object]] = []
    node_offset = 0

    for index in range(len(dataset)):
        graph = dataset[index]
        tu_graph_index = index + 1
        graph_indicator.extend([str(tu_graph_index)] * graph.num_nodes)
        graph_labels.append("1" if graph.label == 0 else "-1")
        node_labels.extend(str(int(value) + 1) for value in graph.node_type.tolist())
        node_attributes.extend(
            ",".join(f"{float(value):.9g}" for value in row)
            for row in graph.x.tolist()
        )
        for edge_index in range(graph.num_edges):
            source = int(graph.edge_index[0, edge_index]) + node_offset + 1
            destination = int(graph.edge_index[1, edge_index]) + node_offset + 1
            edge_rows.append(f"{source}, {destination}")
            edge_labels.append(str(int(graph.edge_type[edge_index]) + 1))
        mapping.append(
            {
                "tu_graph_index": tu_graph_index,
                "dataset_index": index,
                "graph_id": graph.graph_id,
                "label": graph.label,
                "split": split_by_index.get(index, "unused"),
            }
        )
        node_offset += graph.num_nodes

    prefix = raw / name
    _write_lines(prefix.with_name(f"{name}_A.txt"), edge_rows)
    _write_lines(
        prefix.with_name(f"{name}_graph_indicator.txt"), graph_indicator
    )
    _write_lines(prefix.with_name(f"{name}_graph_labels.txt"), graph_labels)
    _write_lines(prefix.with_name(f"{name}_node_labels.txt"), node_labels)
    _write_lines(
        prefix.with_name(f"{name}_node_attributes.txt"), node_attributes
    )
    _write_lines(prefix.with_name(f"{name}_edge_labels.txt"), edge_labels)
    pd.DataFrame(mapping).to_csv(raw.parent / "split_mapping.csv", index=False)
    (raw.parent / "export_metadata.json").write_text(
        json.dumps(
            {
                "name": name,
                "num_graphs": len(dataset),
                "num_nodes": len(graph_indicator),
                "num_edges": len(edge_rows),
                "graph_label_semantics": {"1": "normal", "-1": "anomaly"},
                "node_and_edge_indices": "one_based",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return raw.parent
