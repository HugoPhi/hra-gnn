from __future__ import annotations

import random
from dataclasses import dataclass

import torch

from .graph import GraphSample


@dataclass
class AugmentationResult:
    graph: GraphSample
    method: str


class HeterogeneousAugmentor:
    """Schema-aware sparse graph augmentations used by the SSL branch."""

    METHODS = ("edge_perturbation", "edge_addition", "node_type_swap", "edge_type_swap")

    def __init__(
        self,
        *,
        num_node_types: int,
        num_edge_types: int,
        edge_perturbation_rate: float,
        edge_addition_rate: float,
        node_type_swap_rate: float,
        edge_type_swap_rate: float,
        methods: list[str] | None = None,
        preserve_observed_schema: bool = True,
        seed: int = 0,
    ) -> None:
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types
        self.edge_perturbation_rate = edge_perturbation_rate
        self.edge_addition_rate = edge_addition_rate
        self.node_type_swap_rate = node_type_swap_rate
        self.edge_type_swap_rate = edge_type_swap_rate
        self.methods = methods or list(self.METHODS)
        unknown = set(self.methods) - set(self.METHODS)
        if unknown:
            raise ValueError(f"Unknown augmentations: {sorted(unknown)}")
        self.preserve_observed_schema = preserve_observed_schema
        self.random = random.Random(seed)

    def _observed_schema(self, graph: GraphSample) -> set[int]:
        return set(
            graph.canonical_relation_ids(
                self.num_node_types, self.num_edge_types
            ).tolist()
        )

    def _schema_valid(self, graph: GraphSample, allowed: set[int]) -> bool:
        if not self.preserve_observed_schema:
            return True
        current = self._observed_schema(graph)
        return current.issubset(allowed)

    def augment(
        self, graph: GraphSample, method: str | None = None
    ) -> AugmentationResult:
        method = method or self.random.choice(self.methods)
        if method == "edge_perturbation":
            augmented = self.edge_perturbation(graph)
        elif method == "edge_addition":
            augmented = self.edge_addition(graph)
        elif method == "node_type_swap":
            augmented = self.node_type_swap(graph)
        elif method == "edge_type_swap":
            augmented = self.edge_type_swap(graph)
        else:
            raise ValueError(f"Unknown augmentation: {method}")
        return AugmentationResult(augmented, method)

    def edge_perturbation(self, graph: GraphSample) -> GraphSample:
        result = graph.clone()
        if graph.num_edges == 0:
            return result
        count = max(1, round(graph.num_edges * self.edge_perturbation_rate))
        selected = self.random.sample(
            range(graph.num_edges), min(count, graph.num_edges)
        )
        selected_tensor = torch.tensor(
            selected, dtype=torch.long, device=graph.edge_index.device
        )
        _, destination = result.edge_index
        destination_types = result.node_type[destination[selected_tensor]]
        for destination_type_raw in torch.unique(destination_types).tolist():
            destination_type = int(destination_type_raw)
            candidates = torch.nonzero(
                result.node_type == destination_type, as_tuple=False
            ).flatten()
            if candidates.numel() > 1:
                positions = selected_tensor[destination_types == destination_type]
                choices = torch.tensor(
                    [
                        self.random.randrange(candidates.numel())
                        for _ in range(positions.numel())
                    ],
                    dtype=torch.long,
                    device=candidates.device,
                )
                destination[positions] = candidates[choices]
        result.edge_weight = None
        return result

    def edge_addition(self, graph: GraphSample) -> GraphSample:
        result = graph.clone()
        if graph.num_edges == 0:
            return result
        count = max(1, round(graph.num_edges * self.edge_addition_rate))
        source, destination = graph.edge_index
        templates = torch.tensor(
            [self.random.randrange(graph.num_edges) for _ in range(count)],
            dtype=torch.long,
            device=graph.edge_index.device,
        )
        source_types = graph.node_type[source[templates]]
        destination_types = graph.node_type[destination[templates]]
        new_source = torch.empty(
            count, dtype=torch.long, device=graph.edge_index.device
        )
        new_destination = torch.empty_like(new_source)

        for type_id_raw in torch.unique(source_types).tolist():
            type_id = int(type_id_raw)
            positions = torch.nonzero(source_types == type_id, as_tuple=False).flatten()
            candidates = torch.nonzero(
                graph.node_type == type_id, as_tuple=False
            ).flatten()
            choices = torch.tensor(
                [
                    self.random.randrange(candidates.numel())
                    for _ in range(positions.numel())
                ],
                dtype=torch.long,
                device=candidates.device,
            )
            new_source[positions] = candidates[choices]

        for type_id_raw in torch.unique(destination_types).tolist():
            type_id = int(type_id_raw)
            positions = torch.nonzero(
                destination_types == type_id, as_tuple=False
            ).flatten()
            candidates = torch.nonzero(
                graph.node_type == type_id, as_tuple=False
            ).flatten()
            choices = torch.tensor(
                [
                    self.random.randrange(candidates.numel())
                    for _ in range(positions.numel())
                ],
                dtype=torch.long,
                device=candidates.device,
            )
            new_destination[positions] = candidates[choices]

        added_index = torch.stack([new_source, new_destination])
        result.edge_index = torch.cat([result.edge_index, added_index], dim=1)
        result.edge_type = torch.cat([result.edge_type, graph.edge_type[templates]])
        result.edge_weight = None
        return result

    def node_type_swap(self, graph: GraphSample) -> GraphSample:
        result = graph.clone()
        if graph.num_nodes < 2 or self.node_type_swap_rate <= 0:
            return result
        allowed = self._observed_schema(graph)
        count = max(2, round(graph.num_nodes * self.node_type_swap_rate))
        count -= count % 2
        count = min(count, graph.num_nodes - graph.num_nodes % 2)
        candidates = list(range(graph.num_nodes))

        for _ in range(20):
            selected = self.random.sample(candidates, count)
            selected_tensor = torch.tensor(
                selected, dtype=torch.long, device=graph.node_type.device
            )
            swapped_tensor = selected_tensor.reshape(-1, 2).flip(1).flatten()
            proposal = graph.clone()
            proposal.node_type[selected_tensor] = graph.node_type[swapped_tensor]
            if self._schema_valid(proposal, allowed):
                return proposal
        return result

    def edge_type_swap(self, graph: GraphSample) -> GraphSample:
        result = graph.clone()
        if graph.num_edges < 2 or self.num_edge_types < 2:
            return result
        allowed = self._observed_schema(graph)
        count = max(2, round(graph.num_edges * self.edge_type_swap_rate))
        count -= count % 2
        count = min(count, graph.num_edges - graph.num_edges % 2)
        candidates = list(range(graph.num_edges))

        for _ in range(20):
            selected = self.random.sample(candidates, count)
            selected_tensor = torch.tensor(
                selected, dtype=torch.long, device=graph.edge_type.device
            )
            swapped_tensor = selected_tensor.reshape(-1, 2).flip(1).flatten()
            proposal = graph.clone()
            proposal.edge_type[selected_tensor] = graph.edge_type[swapped_tensor]
            if self._schema_valid(proposal, allowed):
                proposal.edge_weight = None
                return proposal
        return result
