from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class GraphSample:
    """A single attributed heterogeneous graph."""

    graph_id: int
    x: torch.Tensor
    edge_index: torch.Tensor
    node_type: torch.Tensor
    edge_type: torch.Tensor
    label: int = 0
    edge_weight: torch.Tensor | None = None
    batch: torch.Tensor | None = None
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.x.ndim != 2:
            raise ValueError("x must have shape [num_nodes, num_features]")
        if self.edge_index.ndim != 2 or self.edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, num_edges]")
        if self.node_type.shape != (self.x.shape[0],):
            raise ValueError("node_type must contain one entry per node")
        if self.edge_type.shape != (self.edge_index.shape[1],):
            raise ValueError("edge_type must contain one entry per edge")
        if (
            self.edge_weight is not None
            and self.edge_weight.shape != self.edge_type.shape
        ):
            raise ValueError("edge_weight must contain one entry per edge")
        if self.batch is not None and self.batch.shape != (self.x.shape[0],):
            raise ValueError("batch must contain one graph index per node")

    @property
    def num_nodes(self) -> int:
        return int(self.x.shape[0])

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])

    @property
    def num_graphs(self) -> int:
        if self.batch is None or self.batch.numel() == 0:
            return 1
        return int(self.batch.max().item()) + 1

    def to(self, device: torch.device | str) -> "GraphSample":
        return GraphSample(
            graph_id=self.graph_id,
            x=self.x.to(device),
            edge_index=self.edge_index.to(device),
            node_type=self.node_type.to(device),
            edge_type=self.edge_type.to(device),
            label=self.label,
            edge_weight=None
            if self.edge_weight is None
            else self.edge_weight.to(device),
            batch=None if self.batch is None else self.batch.to(device),
            metadata=self.metadata,
        )

    def clone(self) -> "GraphSample":
        return GraphSample(
            graph_id=self.graph_id,
            x=self.x.clone(),
            edge_index=self.edge_index.clone(),
            node_type=self.node_type.clone(),
            edge_type=self.edge_type.clone(),
            label=self.label,
            edge_weight=None if self.edge_weight is None else self.edge_weight.clone(),
            batch=None if self.batch is None else self.batch.clone(),
            metadata=None if self.metadata is None else dict(self.metadata),
        )

    def canonical_relation_ids(
        self,
        num_node_types: int,
        num_edge_types: int,
        relation_schema: str = "canonical",
    ) -> torch.Tensor:
        if relation_schema == "edge_only":
            return self.edge_type.long()
        if relation_schema != "canonical":
            raise ValueError("relation_schema must be canonical or edge_only")
        src, dst = self.edge_index
        return (
            self.node_type[src] * num_edge_types + self.edge_type.long()
        ) * num_node_types + self.node_type[dst]


def batch_graphs(graphs: list[GraphSample]) -> GraphSample:
    """Combine independent graphs into one disconnected graph."""
    if not graphs:
        raise ValueError("Cannot batch an empty graph list")

    offsets: list[int] = []
    current = 0
    for graph in graphs:
        offsets.append(current)
        current += graph.num_nodes

    edge_indices = [graph.edge_index + offset for graph, offset in zip(graphs, offsets)]
    has_edge_weights = any(graph.edge_weight is not None for graph in graphs)
    edge_weights = None
    if has_edge_weights:
        edge_weights = torch.cat(
            [
                graph.edge_weight
                if graph.edge_weight is not None
                else torch.ones(
                    graph.num_edges, dtype=graph.x.dtype, device=graph.x.device
                )
                for graph in graphs
            ]
        )

    return GraphSample(
        graph_id=-1,
        x=torch.cat([graph.x for graph in graphs]),
        edge_index=torch.cat(edge_indices, dim=1),
        node_type=torch.cat([graph.node_type for graph in graphs]),
        edge_type=torch.cat([graph.edge_type for graph in graphs]),
        edge_weight=edge_weights,
        batch=torch.cat(
            [
                torch.full(
                    (graph.num_nodes,),
                    graph_index,
                    dtype=torch.long,
                    device=graph.x.device,
                )
                for graph_index, graph in enumerate(graphs)
            ]
        ),
        metadata={
            "graph_ids": [graph.graph_id for graph in graphs],
            "labels": [graph.label for graph in graphs],
        },
    )
