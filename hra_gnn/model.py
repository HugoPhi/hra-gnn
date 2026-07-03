from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn

from .graph import GraphSample


@dataclass
class ModelOutput:
    embedding: torch.Tensor
    ssl_logit: torch.Tensor
    node_embeddings: torch.Tensor
    gate: torch.Tensor
    relation_diagnostics: dict[int, dict[str, float]]
    auxiliary: dict[str, torch.Tensor] | None = None


def segment_softmax(
    scores: torch.Tensor, segment_ids: torch.Tensor, num_segments: int
) -> torch.Tensor:
    if scores.numel() == 0:
        return scores
    maximum = torch.full(
        (num_segments,), -torch.inf, dtype=scores.dtype, device=scores.device
    )
    maximum.scatter_reduce_(0, segment_ids, scores, reduce="amax", include_self=True)
    stabilized = scores - maximum[segment_ids]
    numerator = stabilized.exp()
    denominator = torch.zeros(num_segments, dtype=scores.dtype, device=scores.device)
    denominator.index_add_(0, segment_ids, numerator)
    return numerator / denominator[segment_ids].clamp_min(1e-12)


def segment_mean(
    values: torch.Tensor, segment_ids: torch.Tensor, num_segments: int
) -> torch.Tensor:
    result = torch.zeros(
        num_segments, values.shape[1], dtype=values.dtype, device=values.device
    )
    result.index_add_(0, segment_ids, values)
    counts = torch.zeros(num_segments, dtype=values.dtype, device=values.device)
    counts.index_add_(0, segment_ids, torch.ones_like(segment_ids, dtype=values.dtype))
    return result / counts.clamp_min(1).unsqueeze(1)


def segment_max(
    values: torch.Tensor, segment_ids: torch.Tensor, num_segments: int
) -> torch.Tensor:
    result = torch.full(
        (num_segments, values.shape[1]),
        -torch.inf,
        dtype=values.dtype,
        device=values.device,
    )
    result.scatter_reduce_(
        0,
        segment_ids.unsqueeze(1).expand_as(values),
        values,
        reduce="amax",
        include_self=True,
    )
    return result


class RelationDeviationLayer(nn.Module):
    """Sparse relation-specific message passing with prototype modulation."""

    def __init__(
        self,
        hidden_dim: int,
        num_relations: int,
        *,
        fusion: str,
        deviation_weight: float,
        temperature: float,
        prototype_momentum: float,
        prototype_min_scale: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_relations = num_relations
        self.fusion = fusion
        self.deviation_weight = deviation_weight
        self.temperature = temperature
        self.prototype_momentum = prototype_momentum
        self.prototype_min_scale = prototype_min_scale

        self.relation_weight = nn.Parameter(
            torch.empty(num_relations, hidden_dim, hidden_dim)
        )
        if fusion == "static_concat":
            self.concat_projection = nn.Parameter(
                torch.empty(num_relations, hidden_dim, hidden_dim)
            )
        else:
            self.register_parameter("concat_projection", None)
        self.root = nn.Linear(hidden_dim, hidden_dim)
        self.semantic_score = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1, bias=False),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.activation = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            "prototype", torch.zeros(num_relations, hidden_dim), persistent=True
        )
        self.register_buffer(
            "prototype_scale",
            torch.ones(num_relations, hidden_dim),
            persistent=True,
        )
        self.register_buffer(
            "prototype_initialized",
            torch.zeros(num_relations, dtype=torch.bool),
            persistent=True,
        )
        self.register_buffer(
            "prototype_updates",
            torch.zeros(num_relations, dtype=torch.long),
            persistent=True,
        )
        self._prototype_batch: dict[int, list[torch.Tensor]] | None = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.relation_weight)
        if self.concat_projection is not None:
            nn.init.xavier_uniform_(self.concat_projection)
        self.root.reset_parameters()
        for module in self.semantic_score:
            if isinstance(module, nn.Linear):
                module.reset_parameters()
        self.norm.reset_parameters()

    @torch.no_grad()
    def _update_prototype(self, relation_id: int, values: torch.Tensor) -> None:
        batch_mean = values.detach().mean(dim=0)
        batch_scale = (
            values.detach()
            .std(dim=0, unbiased=False)
            .clamp_min(self.prototype_min_scale)
        )
        if not bool(self.prototype_initialized[relation_id]):
            self.prototype[relation_id].copy_(batch_mean)
            self.prototype_scale[relation_id].copy_(batch_scale)
            self.prototype_initialized[relation_id] = True
        else:
            momentum = self.prototype_momentum
            self.prototype[relation_id].mul_(momentum).add_(
                batch_mean, alpha=1.0 - momentum
            )
            self.prototype_scale[relation_id].mul_(momentum).add_(
                batch_scale, alpha=1.0 - momentum
            )
        self.prototype_updates[relation_id] += 1

    def begin_prototype_batch(self) -> None:
        self._prototype_batch = {}

    @torch.no_grad()
    def commit_prototype_batch(self) -> None:
        if self._prototype_batch is None:
            return
        for relation_id, values in self._prototype_batch.items():
            self._update_prototype(relation_id, torch.cat(values, dim=0))
        self._prototype_batch = None

    def _collect_or_update_prototype(
        self, relation_id: int, values: torch.Tensor
    ) -> None:
        if self._prototype_batch is None:
            self._update_prototype(relation_id, values)
        else:
            self._prototype_batch.setdefault(relation_id, []).append(
                values.detach().clone()
            )

    def _deviation(self, relation_id: int, values: torch.Tensor) -> torch.Tensor:
        if not bool(self.prototype_initialized[relation_id]):
            return torch.zeros(
                values.shape[0], dtype=values.dtype, device=values.device
            )
        prototype = self.prototype[relation_id].detach().clone()
        scale = self.prototype_scale[relation_id].detach().clone()
        standardized = (values - prototype) / scale.clamp_min(self.prototype_min_scale)
        return (standardized.square().mean(dim=1) + 1e-8).sqrt()

    def forward(
        self,
        h: torch.Tensor,
        graph: GraphSample,
        relation_ids: torch.Tensor,
        *,
        update_prototypes: bool,
        collect_diagnostics: bool,
    ) -> tuple[torch.Tensor, dict[int, dict[str, float]], torch.Tensor]:
        num_nodes = h.shape[0]
        source, destination = graph.edge_index
        base = self.root(h)

        diagnostics: dict[int, dict[str, float]] = {}
        node_deviation = torch.zeros(num_nodes, dtype=h.dtype, device=h.device)
        if relation_ids.numel() == 0:
            fused = base
        else:
            transformed = torch.bmm(
                h[source].unsqueeze(1), self.relation_weight[relation_ids]
            ).squeeze(1)
            if graph.edge_weight is not None:
                transformed = transformed * graph.edge_weight.unsqueeze(1)

            relation_node_ids = relation_ids * num_nodes + destination
            unique_pairs, inverse = torch.unique(
                relation_node_ids, sorted=False, return_inverse=True
            )
            relation_values = torch.zeros(
                unique_pairs.shape[0],
                self.hidden_dim,
                dtype=h.dtype,
                device=h.device,
            )
            relation_values.index_add_(0, inverse, transformed)
            degree = torch.zeros(unique_pairs.shape[0], dtype=h.dtype, device=h.device)
            degree.index_add_(0, inverse, torch.ones_like(inverse, dtype=h.dtype))
            relation_values = relation_values / degree.unsqueeze(1)
            active_nodes = unique_pairs.remainder(num_nodes)
            active_relations = torch.div(unique_pairs, num_nodes, rounding_mode="floor")

            initialized = self.prototype_initialized[active_relations]
            prototype = self.prototype[active_relations].detach()
            scale = self.prototype_scale[active_relations].detach()
            standardized = (relation_values - prototype) / scale.clamp_min(
                self.prototype_min_scale
            )
            deviations = (standardized.square().mean(dim=1) + 1e-8).sqrt()
            deviations = torch.where(
                initialized, deviations, torch.zeros_like(deviations)
            )
            node_deviation.scatter_reduce_(
                0,
                active_nodes,
                deviations,
                reduce="amax",
                include_self=True,
            )

            if self.training and update_prototypes:
                for relation_id_raw in torch.unique(active_relations).tolist():
                    relation_id = int(relation_id_raw)
                    self._collect_or_update_prototype(
                        relation_id,
                        relation_values[active_relations == relation_id],
                    )

            if self.fusion == "static_concat":
                assert self.concat_projection is not None
                projected = torch.bmm(
                    relation_values.unsqueeze(1),
                    self.concat_projection[active_relations],
                ).squeeze(1)
                static_fused = torch.zeros_like(h)
                static_fused.index_add_(0, active_nodes, projected)
                fused = base + static_fused
                attention = torch.zeros_like(deviations)
            else:
                semantic = self.semantic_score(
                    torch.cat([h[active_nodes], relation_values], dim=1)
                ).squeeze(1)
                scores = semantic
                if self.fusion == "deviation_attention":
                    scores = scores + self.deviation_weight * deviations
                attention = segment_softmax(
                    scores / self.temperature, active_nodes, num_nodes
                )
                fused_messages = torch.zeros_like(h)
                fused_messages.index_add_(
                    0, active_nodes, attention.unsqueeze(1) * relation_values
                )
                fused = base + fused_messages

            if collect_diagnostics:
                for relation_id_raw in torch.unique(active_relations).tolist():
                    relation_id = int(relation_id_raw)
                    mask = active_relations == relation_id
                    diagnostics[relation_id] = {
                        "mean_attention": float(attention[mask].detach().mean().cpu()),
                        "mean_deviation": float(deviations[mask].detach().mean().cpu()),
                        "prototype_updates": float(
                            self.prototype_updates[relation_id].item()
                        ),
                    }

        return (
            self.norm(h + self.dropout(self.activation(fused))),
            diagnostics,
            node_deviation,
        )


class HRAGNN(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_node_types: int,
        num_edge_types: int,
        relation_schema: str = "canonical",
        num_layers: int = 2,
        relation_fusion: str = "deviation_attention",
        deviation_weight: float = 1.0,
        attention_temperature: float = 1.0,
        prototype_momentum: float = 0.9,
        prototype_min_scale: float = 1e-3,
        readout: str = "hybrid",
        dropout: float = 0.1,
        score_ssl_weight: float = 1.0,
        score_mode: str = "paper_product",
        deviation_score_pool: str = "topk",
        deviation_score_topk_fraction: float = 0.05,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types
        self.relation_schema = relation_schema
        if relation_schema == "edge_only":
            self.num_relations = num_edge_types
        elif relation_schema == "canonical":
            self.num_relations = num_node_types * num_edge_types * num_node_types
        else:
            raise ValueError("relation_schema must be canonical or edge_only")
        self.readout = readout
        self.score_ssl_weight = score_ssl_weight
        self.score_mode = score_mode
        self.deviation_score_pool = deviation_score_pool
        self.deviation_score_topk_fraction = deviation_score_topk_fraction

        self.type_projection = nn.ModuleList(
            [nn.Linear(input_dim, hidden_dim) for _ in range(num_node_types)]
        )
        self.layers = nn.ModuleList(
            [
                RelationDeviationLayer(
                    hidden_dim,
                    self.num_relations,
                    fusion=relation_fusion,
                    deviation_weight=deviation_weight,
                    temperature=attention_temperature,
                    prototype_momentum=prototype_momentum,
                    prototype_min_scale=prototype_min_scale,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.readout_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.graph_projection = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LeakyReLU(0.2),
        )
        self.ssl_head = nn.Sequential(
            nn.Linear(output_dim, max(output_dim // 2, 1)),
            nn.LeakyReLU(0.2),
            nn.Linear(max(output_dim // 2, 1), 1),
        )
        self.register_buffer("svdd_center", torch.zeros(output_dim), persistent=True)
        self.register_buffer(
            "svdd_center_initialized", torch.tensor(False), persistent=True
        )

    def reset_prototypes(self) -> None:
        for layer in self.layers:
            layer.prototype.zero_()
            layer.prototype_scale.fill_(1.0)
            layer.prototype_initialized.zero_()
            layer.prototype_updates.zero_()

    def begin_prototype_batch(self) -> None:
        for layer in self.layers:
            layer.begin_prototype_batch()

    def commit_prototype_batch(self) -> None:
        for layer in self.layers:
            layer.commit_prototype_batch()

    @torch.no_grad()
    def initialize_svdd_center(self, embeddings: torch.Tensor) -> None:
        self.svdd_center.copy_(embeddings.detach().mean(dim=0))
        near_zero = self.svdd_center.abs() < 1e-4
        self.svdd_center[near_zero] = 1e-4
        self.svdd_center_initialized.fill_(True)

    def encode_nodes(
        self,
        graph: GraphSample,
        *,
        update_prototypes: bool = False,
        collect_diagnostics: bool = False,
    ) -> tuple[torch.Tensor, dict[int, dict[str, float]], torch.Tensor]:
        h = torch.zeros(
            graph.num_nodes,
            self.hidden_dim,
            dtype=graph.x.dtype,
            device=graph.x.device,
        )
        for type_id, projection in enumerate(self.type_projection):
            mask = graph.node_type == type_id
            if bool(mask.any()):
                h[mask] = projection(graph.x[mask])
        h = torch.nn.functional.leaky_relu(h, 0.2)

        relation_ids = graph.canonical_relation_ids(
            self.num_node_types,
            self.num_edge_types,
            self.relation_schema,
        )
        diagnostics: dict[int, dict[str, float]] = {}
        node_deviation = torch.zeros(graph.num_nodes, dtype=h.dtype, device=h.device)
        for layer_index, layer in enumerate(self.layers):
            h, layer_diagnostics, layer_node_deviation = layer(
                h,
                graph,
                relation_ids,
                update_prototypes=update_prototypes,
                collect_diagnostics=collect_diagnostics,
            )
            if collect_diagnostics and layer_index == len(self.layers) - 1:
                diagnostics = layer_diagnostics
            if layer_index == len(self.layers) - 1:
                node_deviation = layer_node_deviation
        return h, diagnostics, node_deviation

    def _pool_relation_deviation(
        self,
        node_deviation: torch.Tensor,
        batch: torch.Tensor,
        num_graphs: int,
    ) -> torch.Tensor:
        if self.deviation_score_pool == "max":
            return segment_max(node_deviation.unsqueeze(1), batch, num_graphs).squeeze(
                1
            )
        if self.deviation_score_pool == "mean":
            return segment_mean(node_deviation.unsqueeze(1), batch, num_graphs).squeeze(
                1
            )
        values = []
        fraction = max(min(float(self.deviation_score_topk_fraction), 1.0), 0.0)
        for graph_id in range(num_graphs):
            graph_values = node_deviation[batch == graph_id]
            count = max(1, int(math.ceil(fraction * graph_values.numel())))
            values.append(torch.topk(graph_values, count).values.mean())
        return torch.stack(values)

    def forward(
        self,
        graph: GraphSample,
        *,
        update_prototypes: bool = False,
        collect_diagnostics: bool = False,
    ) -> ModelOutput:
        node_embeddings, diagnostics, node_deviation = self.encode_nodes(
            graph,
            update_prototypes=update_prototypes,
            collect_diagnostics=collect_diagnostics,
        )
        if graph.batch is None:
            batch = torch.zeros(
                graph.num_nodes, dtype=torch.long, device=node_embeddings.device
            )
        else:
            batch = graph.batch
        relation_deviation = None
        if self.score_mode == "relation_deviation" or collect_diagnostics:
            relation_deviation = self._pool_relation_deviation(
                node_deviation, batch, graph.num_graphs
            )
        maximum = segment_max(node_embeddings, batch, graph.num_graphs)
        mean = segment_mean(node_embeddings, batch, graph.num_graphs)

        if self.readout == "max":
            pooled = maximum
            gate = torch.ones_like(maximum)
        elif self.readout == "mean":
            pooled = mean
            gate = torch.zeros_like(mean)
        else:
            gate = self.readout_gate(torch.cat([maximum, mean], dim=1))
            pooled = gate * maximum + (1.0 - gate) * mean

        embedding = self.graph_projection(pooled)
        if graph.batch is None:
            embedding = embedding.squeeze(0)
            gate = gate.squeeze(0)
            if relation_deviation is not None:
                relation_deviation = relation_deviation.squeeze(0)
        ssl_logit = self.ssl_head(embedding).squeeze(-1)
        return ModelOutput(
            embedding=embedding,
            ssl_logit=ssl_logit,
            node_embeddings=node_embeddings,
            gate=gate,
            relation_diagnostics=diagnostics,
            auxiliary={
                "node_relation_deviation": node_deviation,
                **(
                    {"relation_deviation": relation_deviation}
                    if relation_deviation is not None
                    else {}
                ),
            },
        )

    def anomaly_score(self, output: ModelOutput) -> torch.Tensor:
        if self.score_mode == "relation_deviation":
            assert output.auxiliary is not None
            return output.auxiliary["relation_deviation"]
        if not bool(self.svdd_center_initialized):
            raise RuntimeError("SVDD center has not been initialized")
        svdd = (output.embedding - self.svdd_center).square().mean(dim=-1)
        ssl_anomaly = 1.0 - output.ssl_logit.sigmoid()
        if self.score_mode == "svdd":
            return svdd
        if self.score_mode == "ssl":
            return ssl_anomaly
        if self.score_mode == "product":
            return svdd * ssl_anomaly
        if self.score_mode == "paper_product":
            return svdd * (1.0 + ssl_anomaly)
        return svdd + self.score_ssl_weight * ssl_anomaly
