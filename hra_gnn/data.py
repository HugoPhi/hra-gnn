from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .graph import GraphSample


def _boolean_label(value: object) -> int:
    if isinstance(value, str):
        return 0 if value.strip().lower() in {"true", "1", "normal", "benign"} else 1
    return 0 if bool(value) else 1


class CSVGraphDataset(Dataset[GraphSample]):
    """Loader for the processed CSV format released with HRGCN."""

    def __init__(
        self,
        root: str | Path,
        *,
        feature_file: str = "node_feature_norm.csv",
        edge_file: str = "edge_index.csv",
        node_type_file: str = "node_types.txt",
        trace_info_file: str = "trace_info.csv",
        graph_id_column: str = "trace_id",
        label_column: str = "trace_bool",
        cache_graphs: bool = False,
    ) -> None:
        self.root = Path(root)
        self.graph_id_column = graph_id_column
        self.cache_graphs = cache_graphs
        self._cache: dict[int, GraphSample] = {}

        required = [
            self.root / feature_file,
            self.root / edge_file,
            self.root / node_type_file,
            self.root / trace_info_file,
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Processed HRGCN data are incomplete. Missing:\n- "
                + "\n- ".join(missing)
            )

        features = pd.read_csv(required[0]).sort_values(
            [graph_id_column, "node_id"], kind="stable"
        )
        edges = pd.read_csv(required[1]).sort_values([graph_id_column], kind="stable")
        self.trace_info = pd.read_csv(required[3])

        self.node_types: list[list[list[int]]] = []
        with required[2].open("r", encoding="utf-8") as handle:
            for line in handle:
                self.node_types.append(json.loads(line))

        self.graph_ids = self.trace_info[graph_id_column].astype(int).tolist()
        self.labels = {
            int(row[graph_id_column]): _boolean_label(row[label_column])
            for _, row in self.trace_info.iterrows()
        }

        ignored = {graph_id_column, "node_id"}
        self.feature_columns = [
            column for column in features.columns if column not in ignored
        ]
        self._feature_values = np.ascontiguousarray(
            features[self.feature_columns].to_numpy(dtype=np.float32)
        ).copy()
        self._edge_index_values = np.ascontiguousarray(
            edges[["src_id", "dst_id"]].to_numpy(dtype=np.int64)
        ).copy()
        self._edge_type_values = (
            edges["edge_type"].to_numpy(dtype=np.int64).copy()
            if "edge_type" in edges
            else np.zeros(len(edges), dtype=np.int64)
        )
        self._edge_weight_values = (
            edges["weight"].to_numpy(dtype=np.float32).copy()
            if "weight" in edges
            else None
        )
        self._feature_slices = self._group_slices(
            features[graph_id_column].to_numpy(dtype=np.int64)
        )
        self._edge_slices = self._group_slices(
            edges[graph_id_column].to_numpy(dtype=np.int64)
        )

    @staticmethod
    def _group_slices(graph_ids: np.ndarray) -> dict[int, slice]:
        unique, starts, counts = np.unique(
            graph_ids, return_index=True, return_counts=True
        )
        return {
            int(graph_id): slice(int(start), int(start + count))
            for graph_id, start, count in zip(unique, starts, counts)
        }

    def __len__(self) -> int:
        return len(self.graph_ids)

    def graph_at_id(self, graph_id: int) -> GraphSample:
        if graph_id in self._cache:
            return self._cache[graph_id]

        feature_slice = self._feature_slices[graph_id]
        edge_slice = self._edge_slices[graph_id]
        x = torch.from_numpy(self._feature_values[feature_slice])
        edge_index = torch.from_numpy(self._edge_index_values[edge_slice]).T
        edge_type = torch.from_numpy(self._edge_type_values[edge_slice])
        edge_weight = None
        if self._edge_weight_values is not None:
            edge_weight = torch.from_numpy(self._edge_weight_values[edge_slice])

        type_lists = self.node_types[graph_id]
        node_type = torch.empty(x.shape[0], dtype=torch.long)
        for type_id, nodes in enumerate(type_lists):
            if nodes:
                node_type[torch.tensor(nodes, dtype=torch.long)] = type_id

        graph = GraphSample(
            graph_id=graph_id,
            x=x,
            edge_index=edge_index,
            node_type=node_type,
            edge_type=edge_type,
            edge_weight=edge_weight,
            label=self.labels[graph_id],
        )
        if self.cache_graphs:
            self._cache[graph_id] = graph
        return graph

    def __getitem__(self, index: int) -> GraphSample:
        return self.graph_at_id(self.graph_ids[index])

    def indices_for_ids(self, ids: Sequence[int]) -> list[int]:
        index_by_id = {graph_id: index for index, graph_id in enumerate(self.graph_ids)}
        return [index_by_id[int(graph_id)] for graph_id in ids]


class SyntheticGraphDataset(Dataset[GraphSample]):
    """Small deterministic dataset used for smoke tests and CI."""

    def __init__(
        self,
        num_graphs: int = 48,
        feature_dim: int = 7,
        num_node_types: int = 3,
        num_edge_types: int = 2,
        anomaly_ratio: float = 0.25,
        seed: int = 7,
    ) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.graphs: list[GraphSample] = []
        anomaly_start = int(num_graphs * (1 - anomaly_ratio))

        for graph_id in range(num_graphs):
            anomalous = graph_id >= anomaly_start
            num_nodes = 14 + graph_id % 5
            node_type = torch.arange(num_nodes) % num_node_types
            x = torch.randn(num_nodes, feature_dim, generator=generator) * 0.2
            x += node_type.unsqueeze(1).float() * 0.25

            src = torch.arange(num_nodes)
            dst = (src + 1) % num_nodes
            edge_type = src % num_edge_types
            if anomalous:
                x[0, 0] += 4.0
                src = torch.cat([src, torch.tensor([0, 0, 1])])
                dst = torch.cat(
                    [dst, torch.tensor([num_nodes - 1, num_nodes - 2, num_nodes - 1])]
                )
                edge_type = torch.cat([edge_type, torch.ones(3, dtype=torch.long)])

            self.graphs.append(
                GraphSample(
                    graph_id=graph_id,
                    x=x,
                    edge_index=torch.stack([src, dst]),
                    node_type=node_type.long(),
                    edge_type=edge_type.long(),
                    label=int(anomalous),
                )
            )

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, index: int) -> GraphSample:
        return self.graphs[index]


def load_dataset(config: dict) -> Dataset[GraphSample]:
    kind = config.get("kind", "csv")
    if kind == "synthetic":
        return SyntheticGraphDataset(
            num_graphs=config.get("num_graphs", 48),
            feature_dim=config["feature_dim"],
            num_node_types=config["num_node_types"],
            num_edge_types=config["num_edge_types"],
            anomaly_ratio=config.get("anomaly_ratio", 0.25),
            seed=config.get("seed", 7),
        )
    if kind == "csv":
        return CSVGraphDataset(
            config["root"],
            feature_file=config.get("feature_file", "node_feature_norm.csv"),
            edge_file=config.get("edge_file", "edge_index.csv"),
            node_type_file=config.get("node_type_file", "node_types.txt"),
            trace_info_file=config.get("trace_info_file", "trace_info.csv"),
            cache_graphs=config.get("cache_graphs", False),
        )
    raise ValueError(f"Unsupported dataset kind: {kind}")


def make_splits(
    dataset: Dataset[GraphSample],
    *,
    train_normal_ratio: float,
    validation_normal_ratio: float,
    validation_anomaly_ratio: float,
    seed: int,
) -> dict[str, list[int]]:
    rng = np.random.default_rng(seed)
    normal = np.array(
        [index for index in range(len(dataset)) if dataset[index].label == 0]
    )
    anomaly = np.array(
        [index for index in range(len(dataset)) if dataset[index].label == 1]
    )
    rng.shuffle(normal)
    rng.shuffle(anomaly)

    train_end = int(len(normal) * train_normal_ratio)
    validation_end = train_end + int(len(normal) * validation_normal_ratio)
    anomaly_validation_end = int(len(anomaly) * validation_anomaly_ratio)

    return {
        "train": normal[:train_end].tolist(),
        "validation": np.concatenate(
            [normal[train_end:validation_end], anomaly[:anomaly_validation_end]]
        ).tolist(),
        "test": np.concatenate(
            [normal[validation_end:], anomaly[anomaly_validation_end:]]
        ).tolist(),
    }


def load_provided_splits(
    dataset: Dataset[GraphSample],
    root: str | Path,
    *,
    train_file: str,
    validation_file: str | None,
    test_file: str,
) -> dict[str, list[int]]:
    if not isinstance(dataset, CSVGraphDataset):
        raise TypeError("Provided graph-ID splits require CSVGraphDataset")

    def read_ids(filename: str | None) -> list[int]:
        if not filename:
            return []
        path = Path(root) / filename
        if not path.exists():
            raise FileNotFoundError(f"Split file does not exist: {path}")
        ids = [int(value) for value in path.read_text(encoding="utf-8").split()]
        return dataset.indices_for_ids(ids)

    return {
        "train": read_ids(train_file),
        "validation": read_ids(validation_file),
        "test": read_ids(test_file),
    }
