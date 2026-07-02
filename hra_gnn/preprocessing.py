from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .graph import GraphSample


BLOCK_PATTERN = re.compile(r"blk_-?\d+")


def _split_normal_ids(
    normal_ids: list[int], anomaly_ids: list[int], seed: int
) -> dict[str, list[int]]:
    rng = np.random.default_rng(seed)
    normal = np.asarray(normal_ids, dtype=np.int64)
    rng.shuffle(normal)
    train_end = int(0.6 * len(normal))
    validation_end = train_end + int(0.2 * len(normal))
    return {
        "train": normal[:train_end].tolist(),
        "validation": normal[train_end:validation_end].tolist(),
        "test": normal[validation_end:].tolist() + list(anomaly_ids),
    }


def write_packed_dataset(
    graphs: Iterable[GraphSample],
    output: str | Path,
    *,
    splits: dict[str, list[int]],
    metadata: dict,
    sample_metadata: list[dict] | None = None,
) -> Path:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    graph_list = list(graphs)
    if not graph_list:
        raise ValueError("Cannot write an empty graph dataset")

    node_counts = np.asarray([graph.num_nodes for graph in graph_list], dtype=np.int64)
    edge_counts = np.asarray([graph.num_edges for graph in graph_list], dtype=np.int64)
    graph_ptr = np.concatenate([[0], np.cumsum(node_counts)])
    edge_ptr = np.concatenate([[0], np.cumsum(edge_counts)])
    x = np.concatenate([graph.x.numpy() for graph in graph_list]).astype(np.float32)
    node_type = np.concatenate(
        [graph.node_type.numpy() for graph in graph_list]
    ).astype(np.int64)
    edge_type = np.concatenate(
        [graph.edge_type.numpy() for graph in graph_list]
    ).astype(np.int64)
    edge_index_parts = []
    for graph, offset in zip(graph_list, graph_ptr[:-1]):
        edge_index_parts.append(graph.edge_index.numpy() + int(offset))
    edge_index = np.concatenate(edge_index_parts, axis=1).astype(np.int64)

    arrays = {
        "x.npy": x,
        "node_type.npy": node_type,
        "edge_index.npy": edge_index,
        "edge_type.npy": edge_type,
        "graph_ptr.npy": graph_ptr,
        "edge_ptr.npy": edge_ptr,
        "labels.npy": np.asarray([graph.label for graph in graph_list], dtype=np.int64),
        "graph_ids.npy": np.asarray(
            [graph.graph_id for graph in graph_list], dtype=np.int64
        ),
    }
    for filename, values in arrays.items():
        np.save(output / filename, values, allow_pickle=False)
    if any(graph.edge_weight is not None for graph in graph_list):
        weights = np.concatenate(
            [
                graph.edge_weight.numpy()
                if graph.edge_weight is not None
                else np.ones(graph.num_edges, dtype=np.float32)
                for graph in graph_list
            ]
        ).astype(np.float32)
        np.save(output / "edge_weight.npy", weights, allow_pickle=False)

    complete_metadata = {
        **metadata,
        "num_graphs": len(graph_list),
        "feature_dim": int(x.shape[1]),
        "num_node_types": int(node_type.max()) + 1,
        "num_edge_types": int(edge_type.max()) + 1,
        "normal_graphs": int(sum(graph.label == 0 for graph in graph_list)),
        "anomaly_graphs": int(sum(graph.label == 1 for graph in graph_list)),
    }
    (output / "metadata.json").write_text(
        json.dumps(complete_metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    for split, graph_ids in splits.items():
        (output / f"{split}_ids.txt").write_text(
            "\n".join(map(str, graph_ids)) + "\n", encoding="utf-8"
        )
    if sample_metadata:
        pd.DataFrame(sample_metadata).to_csv(output / "samples.csv", index=False)
    return output


def _temporal_graph(
    graph_id: int,
    token_ids: list[int],
    *,
    label: int,
    auxiliary_ids: list[int] | None = None,
    feature_dim: int = 8,
) -> GraphSample:
    import torch

    count = len(token_ids)
    if count == 0:
        raise ValueError("A sequence graph must contain at least one token")
    auxiliary_ids = auxiliary_ids or [0] * count
    total_frequency: dict[int, int] = defaultdict(int)
    for token in token_ids:
        total_frequency[token] += 1
    seen: dict[int, int] = defaultdict(int)
    features = np.zeros((count, feature_dim), dtype=np.float32)
    denominator = max(count - 1, 1)
    for index, (token, auxiliary) in enumerate(zip(token_ids, auxiliary_ids)):
        seen[token] += 1
        base = [
            index / denominator,
            np.log1p(count) / 16.0,
            seen[token] / total_frequency[token],
            total_frequency[token] / count,
            float(index == 0),
            float(index == count - 1),
            (auxiliary % 997) / 997.0,
            np.log1p(token) / 16.0,
        ]
        features[index, : min(feature_dim, len(base))] = base[:feature_dim]

    sources: list[int] = []
    destinations: list[int] = []
    edge_types: list[int] = []
    for index in range(1, count):
        sources.append(index - 1)
        destinations.append(index)
        edge_types.append(0)
    previous_token: dict[int, int] = {}
    previous_auxiliary: dict[int, int] = {}
    for index, (token, auxiliary) in enumerate(zip(token_ids, auxiliary_ids)):
        if token in previous_token:
            sources.append(previous_token[token])
            destinations.append(index)
            edge_types.append(1)
        previous_token[token] = index
        if auxiliary in previous_auxiliary:
            sources.append(previous_auxiliary[auxiliary])
            destinations.append(index)
            edge_types.append(2)
        previous_auxiliary[auxiliary] = index
    if not sources:
        sources = [0]
        destinations = [0]
        edge_types = [0]
    return GraphSample(
        graph_id=graph_id,
        x=torch.from_numpy(features),
        edge_index=torch.tensor([sources, destinations], dtype=torch.long),
        node_type=torch.tensor(token_ids, dtype=torch.long),
        edge_type=torch.tensor(edge_types, dtype=torch.long),
        label=label,
    )


def prepare_hdfs(
    structured_csv: str | Path,
    labels_csv: str | Path,
    output: str | Path,
    *,
    seed: int = 42,
    max_graphs: int | None = None,
) -> Path:
    events = pd.read_csv(structured_csv)
    required = {"Content", "EventId"}
    if not required.issubset(events):
        raise ValueError(f"HDFS structured CSV must contain {sorted(required)}")
    events = events.copy()
    events["BlockId"] = events["Content"].astype(str).str.extract(
        f"({BLOCK_PATTERN.pattern})", expand=False
    )
    events = events.dropna(subset=["BlockId"])
    labels_frame = pd.read_csv(labels_csv)
    label_column = "Label" if "Label" in labels_frame else labels_frame.columns[-1]
    block_column = (
        "BlockId" if "BlockId" in labels_frame else labels_frame.columns[0]
    )
    label_by_block = {
        str(row[block_column]): int(str(row[label_column]).strip().lower() != "normal")
        for _, row in labels_frame.iterrows()
    }
    event_vocab = {
        value: index
        for index, value in enumerate(sorted(events["EventId"].astype(str).unique()))
    }
    component_values = (
        events["Component"].astype(str)
        if "Component" in events
        else pd.Series(["unknown"] * len(events), index=events.index)
    )
    component_vocab = {
        value: index
        for index, value in enumerate(sorted(component_values.unique()))
    }
    events["_component"] = component_values

    graphs: list[GraphSample] = []
    sample_metadata: list[dict] = []
    grouped = events.groupby("BlockId", sort=True)
    for graph_id, (block_id, frame) in enumerate(grouped):
        if max_graphs is not None and graph_id >= max_graphs:
            break
        token_ids = [event_vocab[value] for value in frame["EventId"].astype(str)]
        auxiliary = [
            component_vocab[value] for value in frame["_component"].astype(str)
        ]
        label = label_by_block.get(str(block_id), 0)
        graphs.append(
            _temporal_graph(
                graph_id, token_ids, label=label, auxiliary_ids=auxiliary
            )
        )
        sample_metadata.append(
            {
                "graph_id": graph_id,
                "source_id": block_id,
                "label": label,
                "sequence_length": len(token_ids),
            }
        )
    normal = [graph.graph_id for graph in graphs if graph.label == 0]
    anomaly = [graph.graph_id for graph in graphs if graph.label == 1]
    return write_packed_dataset(
        graphs,
        output,
        splits=_split_normal_ids(normal, anomaly, seed),
        metadata={
            "name": "HDFS",
            "source": "Loghub HDFS_v1 structured log",
            "construction": "event-occurrence temporal graph",
            "seed": seed,
        },
        sample_metadata=sample_metadata,
    )


def _read_syscall_trace(path: Path) -> list[int]:
    values = path.read_text(encoding="utf-8", errors="ignore").split()
    return [int(value) for value in values if value.lstrip("-").isdigit()]


def prepare_adfa_ld(
    root: str | Path,
    output: str | Path,
    *,
    seed: int = 42,
    max_graphs: int | None = None,
) -> Path:
    root = Path(root)
    training = sorted((root / "Training_Data_Master").rglob("*.txt"))
    validation = sorted((root / "Validation_Data_Master").rglob("*.txt"))
    attacks = sorted((root / "Attack_Data_Master").rglob("*.txt"))
    if not training or not validation or not attacks:
        raise FileNotFoundError(
            "ADFA-LD requires Training_Data_Master, Validation_Data_Master "
            "and Attack_Data_Master with trace text files"
        )
    train_sequences = [_read_syscall_trace(path) for path in training]
    vocabulary = sorted({token for sequence in train_sequences for token in sequence})
    token_map = {token: index + 1 for index, token in enumerate(vocabulary)}

    records = (
        [(path, sequence, 0, "train") for path, sequence in zip(training, train_sequences)]
        + [(path, _read_syscall_trace(path), 0, "validation") for path in validation]
        + [(path, _read_syscall_trace(path), 1, "attack") for path in attacks]
    )
    records = [record for record in records if record[1]]
    if max_graphs is not None:
        records = records[:max_graphs]
    graphs: list[GraphSample] = []
    sample_metadata: list[dict] = []
    split_ids = {"train": [], "validation": [], "test": []}
    rng = np.random.default_rng(seed)
    validation_ids = [
        index for index, record in enumerate(records) if record[3] == "validation"
    ]
    rng.shuffle(validation_ids)
    validation_cut = len(validation_ids) // 2
    validation_set = set(validation_ids[:validation_cut])
    for graph_id, (path, sequence, label, source_split) in enumerate(records):
        mapped = [token_map.get(token, 0) for token in sequence]
        graphs.append(_temporal_graph(graph_id, mapped, label=label))
        if source_split == "train":
            split_ids["train"].append(graph_id)
        elif source_split == "validation" and graph_id in validation_set:
            split_ids["validation"].append(graph_id)
        else:
            split_ids["test"].append(graph_id)
        sample_metadata.append(
            {
                "graph_id": graph_id,
                "source_file": str(path.relative_to(root)),
                "source_split": source_split,
                "attack_type": path.parent.name if label else "",
                "label": label,
                "sequence_length": len(sequence),
            }
        )
    return write_packed_dataset(
        graphs,
        output,
        splits=split_ids,
        metadata={
            "name": "ADFA-LD",
            "source": "UNSW ADFA-LD",
            "construction": "syscall-occurrence temporal graph",
            "unknown_node_type": 0,
            "seed": seed,
        },
        sample_metadata=sample_metadata,
    )
