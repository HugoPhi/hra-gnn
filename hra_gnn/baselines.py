from __future__ import annotations

import math

import torch
from torch import nn

from .graph import GraphSample
from .model import HRAGNN, ModelOutput, segment_max, segment_mean, segment_softmax


class OneClassGraphModel(nn.Module):
    def __init__(
        self,
        output_dim: int,
        score_ssl_weight: float = 0.0,
        score_mode: str = "svdd",
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.score_ssl_weight = score_ssl_weight
        self.score_mode = score_mode
        self.register_buffer("svdd_center", torch.zeros(output_dim), persistent=True)
        self.register_buffer(
            "svdd_center_initialized", torch.tensor(False), persistent=True
        )

    @torch.no_grad()
    def initialize_svdd_center(self, embeddings: torch.Tensor) -> None:
        self.svdd_center.copy_(embeddings.detach().mean(dim=0))
        near_zero = self.svdd_center.abs() < 1e-4
        self.svdd_center[near_zero] = 1e-4
        self.svdd_center_initialized.fill_(True)

    def anomaly_score(self, output: ModelOutput) -> torch.Tensor:
        if not bool(self.svdd_center_initialized):
            raise RuntimeError("SVDD center has not been initialized")
        svdd = (output.embedding - self.svdd_center).square().mean(dim=-1)
        ssl_anomaly = 1.0 - output.ssl_logit.sigmoid()
        if self.score_mode == "ssl":
            return ssl_anomaly
        if self.score_mode == "product":
            return svdd * ssl_anomaly
        if self.score_mode == "paper_product":
            return svdd * (1.0 + ssl_anomaly)
        if self.score_mode == "weighted_sum":
            return svdd + self.score_ssl_weight * ssl_anomaly
        return svdd


class HGTLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_relations: int, dropout: float) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.relation_key = nn.Parameter(
            torch.empty(num_relations, hidden_dim, hidden_dim)
        )
        self.relation_value = nn.Parameter(
            torch.empty(num_relations, hidden_dim, hidden_dim)
        )
        self.root = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.relation_key)
        nn.init.xavier_uniform_(self.relation_value)

    def forward(
        self, h: torch.Tensor, graph: GraphSample, relation_ids: torch.Tensor
    ) -> torch.Tensor:
        source, destination = graph.edge_index
        queries = self.query(h[destination])
        keys = self.key(h[source])
        values = self.value(h[source])
        transformed_keys = torch.bmm(
            keys.unsqueeze(1), self.relation_key[relation_ids]
        ).squeeze(1)
        transformed_values = torch.bmm(
            values.unsqueeze(1), self.relation_value[relation_ids]
        ).squeeze(1)
        scores = (queries * transformed_keys).sum(dim=1) / math.sqrt(self.hidden_dim)
        attention = segment_softmax(scores, destination, h.shape[0])
        messages = torch.zeros_like(h)
        messages.index_add_(0, destination, attention.unsqueeze(1) * transformed_values)
        return self.norm(
            h + self.dropout(torch.nn.functional.gelu(self.root(h) + messages))
        )


class HGTBaseline(OneClassGraphModel):
    """HGT-style typed transformer with max readout and DeepSVDD."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_node_types: int,
        num_edge_types: int,
        num_layers: int,
        dropout: float,
        relation_schema: str = "canonical",
    ) -> None:
        super().__init__(output_dim)
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types
        self.relation_schema = relation_schema
        num_relations = (
            num_edge_types
            if relation_schema == "edge_only"
            else num_node_types * num_edge_types * num_node_types
        )
        self.type_projection = nn.ModuleList(
            [nn.Linear(input_dim, hidden_dim) for _ in range(num_node_types)]
        )
        self.layers = nn.ModuleList(
            [HGTLayer(hidden_dim, num_relations, dropout) for _ in range(num_layers)]
        )
        self.graph_projection = nn.Linear(hidden_dim, output_dim)
        self.ssl_head = nn.Linear(output_dim, 1)

    def forward(
        self,
        graph: GraphSample,
        *,
        update_prototypes: bool = False,
        collect_diagnostics: bool = False,
    ) -> ModelOutput:
        del update_prototypes, collect_diagnostics
        h = torch.zeros(
            graph.num_nodes,
            self.layers[0].hidden_dim,
            dtype=graph.x.dtype,
            device=graph.x.device,
        )
        for type_id, projection in enumerate(self.type_projection):
            mask = graph.node_type == type_id
            if bool(mask.any()):
                h[mask] = projection(graph.x[mask])
        relation_ids = graph.canonical_relation_ids(
            self.num_node_types,
            self.num_edge_types,
            self.relation_schema,
        )
        for layer in self.layers:
            h = layer(h, graph, relation_ids)
        if graph.batch is None:
            batch = torch.zeros(graph.num_nodes, dtype=torch.long, device=h.device)
        else:
            batch = graph.batch
        embedding = torch.nn.functional.leaky_relu(
            self.graph_projection(segment_max(h, batch, graph.num_graphs)), 0.2
        )
        gate = torch.ones(graph.num_graphs, h.shape[1], dtype=h.dtype, device=h.device)
        if graph.batch is None:
            embedding = embedding.squeeze(0)
            gate = gate.squeeze(0)
        return ModelOutput(
            embedding=embedding,
            ssl_logit=self.ssl_head(embedding).squeeze(-1),
            node_embeddings=h,
            gate=gate,
            relation_diagnostics={},
        )


class DeepTraLogBaseline(OneClassGraphModel):
    """Topology-only GGNN/attention-readout approximation used in the manuscript."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_node_types: int,
        num_layers: int,
    ) -> None:
        super().__init__(output_dim)
        self.type_projection = nn.ModuleList(
            [nn.Linear(input_dim, hidden_dim) for _ in range(num_node_types)]
        )
        self.gru = nn.ModuleList(
            [nn.GRUCell(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )
        self.attention = nn.Linear(hidden_dim, 1)
        self.graph_projection = nn.Linear(hidden_dim, output_dim)
        self.ssl_head = nn.Linear(output_dim, 1)

    def forward(
        self,
        graph: GraphSample,
        *,
        update_prototypes: bool = False,
        collect_diagnostics: bool = False,
    ) -> ModelOutput:
        del update_prototypes, collect_diagnostics
        hidden_dim = self.gru[0].hidden_size
        h = torch.zeros(
            graph.num_nodes,
            hidden_dim,
            dtype=graph.x.dtype,
            device=graph.x.device,
        )
        for type_id, projection in enumerate(self.type_projection):
            mask = graph.node_type == type_id
            if bool(mask.any()):
                h[mask] = projection(graph.x[mask])
        source, destination = graph.edge_index
        for cell in self.gru:
            aggregated = torch.zeros_like(h)
            aggregated.index_add_(0, destination, h[source])
            degree = torch.zeros(graph.num_nodes, dtype=h.dtype, device=h.device)
            degree.index_add_(
                0, destination, torch.ones_like(destination, dtype=h.dtype)
            )
            aggregated = aggregated / degree.clamp_min(1).unsqueeze(1)
            h = cell(aggregated, h)
        if graph.batch is None:
            batch = torch.zeros(graph.num_nodes, dtype=torch.long, device=h.device)
        else:
            batch = graph.batch
        attention = segment_softmax(
            self.attention(h).squeeze(1), batch, graph.num_graphs
        )
        pooled = torch.zeros(
            graph.num_graphs, h.shape[1], dtype=h.dtype, device=h.device
        )
        pooled.index_add_(0, batch, attention.unsqueeze(1) * h)
        embedding = torch.nn.functional.leaky_relu(self.graph_projection(pooled), 0.2)
        gate = segment_mean(attention.unsqueeze(1), batch, graph.num_graphs)
        if graph.batch is None:
            embedding = embedding.squeeze(0)
            gate = gate.squeeze(0)
        return ModelOutput(
            embedding=embedding,
            ssl_logit=self.ssl_head(embedding).squeeze(-1),
            node_embeddings=h,
            gate=gate,
            relation_diagnostics={},
        )


class GLocalKDBaseline(nn.Module):
    """Random teacher/student glocal knowledge-distillation baseline."""

    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_node_types: int,
        num_edge_types: int,
        num_layers: int,
        dropout: float,
        relation_schema: str = "canonical",
    ) -> None:
        super().__init__()
        parameters = dict(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_node_types=num_node_types,
            num_edge_types=num_edge_types,
            num_layers=num_layers,
            relation_fusion="static_concat",
            relation_schema=relation_schema,
            readout="mean",
            dropout=dropout,
        )
        self.student = HRAGNN(**parameters)
        self.teacher = HRAGNN(**parameters)
        for parameter in self.teacher.parameters():
            parameter.requires_grad_(False)
        self.teacher.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self.teacher.eval()
        return self

    def forward(
        self,
        graph: GraphSample,
        *,
        update_prototypes: bool = False,
        collect_diagnostics: bool = False,
    ) -> ModelOutput:
        del update_prototypes, collect_diagnostics
        student = self.student(graph)
        with torch.no_grad():
            teacher = self.teacher(graph)
        return ModelOutput(
            embedding=student.embedding,
            ssl_logit=student.ssl_logit,
            node_embeddings=student.node_embeddings,
            gate=student.gate,
            relation_diagnostics={},
            auxiliary={
                "teacher_embedding": teacher.embedding,
                "teacher_nodes": teacher.node_embeddings,
                "batch": (
                    torch.zeros(
                        graph.num_nodes,
                        dtype=torch.long,
                        device=graph.x.device,
                    )
                    if graph.batch is None
                    else graph.batch
                ),
            },
        )

    def distillation_scores(self, output: ModelOutput) -> torch.Tensor:
        assert output.auxiliary is not None
        global_scores = (
            (output.embedding - output.auxiliary["teacher_embedding"])
            .square()
            .mean(dim=-1)
        )
        local_values = (
            (output.node_embeddings - output.auxiliary["teacher_nodes"])
            .square()
            .mean(dim=1, keepdim=True)
        )
        batch = output.auxiliary["batch"]
        num_graphs = 1 if global_scores.ndim == 0 else global_scores.shape[0]
        local_scores = segment_mean(local_values, batch, num_graphs).squeeze(-1)
        if global_scores.ndim == 0:
            local_scores = local_scores.squeeze(0)
        return global_scores + local_scores

    def distillation_loss(self, output: ModelOutput) -> torch.Tensor:
        return self.distillation_scores(output).mean()

    def anomaly_score(self, output: ModelOutput) -> torch.Tensor:
        return self.distillation_scores(output)
