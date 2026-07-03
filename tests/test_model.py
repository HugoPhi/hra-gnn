import pytest
import torch

from hra_gnn.data import SyntheticGraphDataset
from hra_gnn.graph import GraphSample, batch_graphs
from hra_gnn.model import HRAGNN


@pytest.mark.parametrize(
    "fusion", ["deviation_attention", "semantic_attention", "static_concat"]
)
@pytest.mark.parametrize("readout", ["hybrid", "max", "mean"])
def test_model_variants_forward_and_backward(fusion: str, readout: str) -> None:
    graph = SyntheticGraphDataset(num_graphs=8)[0]
    model = HRAGNN(
        input_dim=7,
        hidden_dim=8,
        output_dim=8,
        num_node_types=3,
        num_edge_types=2,
        relation_fusion=fusion,
        readout=readout,
        num_layers=2,
    )
    model.train()
    output = model(graph, update_prototypes=True, collect_diagnostics=True)
    model.initialize_svdd_center(output.embedding.unsqueeze(0))
    loss = model.anomaly_score(output)
    loss.backward()
    assert output.embedding.shape == (8,)
    assert output.node_embeddings.shape == (graph.num_nodes, 8)
    assert output.auxiliary is not None
    assert torch.isfinite(output.auxiliary["relation_deviation"])
    assert torch.isfinite(loss)


def test_prototypes_update_only_when_requested() -> None:
    graph = SyntheticGraphDataset(num_graphs=8)[0]
    model = HRAGNN(
        input_dim=7,
        hidden_dim=8,
        output_dim=8,
        num_node_types=3,
        num_edge_types=2,
    )
    model.train()
    model(graph, update_prototypes=False)
    assert not bool(model.layers[0].prototype_initialized.any())
    model(graph, update_prototypes=True)
    assert bool(model.layers[0].prototype_initialized.any())


def test_prototypes_commit_once_per_batch() -> None:
    dataset = SyntheticGraphDataset(num_graphs=8)
    model = HRAGNN(
        input_dim=7,
        hidden_dim=8,
        output_dim=8,
        num_node_types=3,
        num_edge_types=2,
    )
    model.train()
    model.begin_prototype_batch()
    model(dataset[0], update_prototypes=True)
    model(dataset[1], update_prototypes=True)
    assert not bool(model.layers[0].prototype_initialized.any())
    model.commit_prototype_batch()
    assert bool(model.layers[0].prototype_initialized.any())
    assert int(model.layers[0].prototype_updates.max()) == 1


def test_batched_embeddings_match_individual_graphs() -> None:
    dataset = SyntheticGraphDataset(num_graphs=8)
    graphs = [dataset[index] for index in range(4)]
    model = HRAGNN(
        input_dim=7,
        hidden_dim=8,
        output_dim=8,
        num_node_types=3,
        num_edge_types=2,
        dropout=0.0,
    )
    model.eval()
    individual = torch.stack([model(graph).embedding for graph in graphs])
    batched = model(batch_graphs(graphs))
    assert batched.embedding.shape == (4, 8)
    assert batched.ssl_logit.shape == (4,)
    torch.testing.assert_close(batched.embedding, individual, rtol=1e-5, atol=1e-6)


def test_paper_product_score_matches_recovered_equation() -> None:
    graph = SyntheticGraphDataset(num_graphs=8)[0]
    model = HRAGNN(
        input_dim=7,
        hidden_dim=8,
        output_dim=8,
        num_node_types=3,
        num_edge_types=2,
        score_mode="paper_product",
    )
    output = model(graph)
    model.initialize_svdd_center(torch.zeros(1, 8))
    distance = (output.embedding - model.svdd_center).square().mean()
    ssl_anomaly = 1.0 - output.ssl_logit.sigmoid()
    torch.testing.assert_close(
        model.anomaly_score(output), distance * (1.0 + ssl_anomaly)
    )


def test_edge_only_schema_avoids_quadratic_relation_expansion() -> None:
    graph = SyntheticGraphDataset(num_graphs=8)[0]
    model = HRAGNN(
        input_dim=7,
        hidden_dim=8,
        output_dim=8,
        num_node_types=151,
        num_edge_types=3,
        relation_schema="edge_only",
    )

    output = model(graph)

    assert model.num_relations == 3
    assert model.layers[0].relation_weight.shape == (3, 8, 8)
    assert output.embedding.shape == (8,)


@pytest.mark.parametrize("pool", ["topk", "max", "mean"])
def test_relation_deviation_is_a_finite_graph_score(pool: str) -> None:
    dataset = SyntheticGraphDataset(num_graphs=8)
    graph = batch_graphs([dataset[0], dataset[1]])
    model = HRAGNN(
        input_dim=7,
        hidden_dim=8,
        output_dim=8,
        num_node_types=3,
        num_edge_types=2,
        score_mode="relation_deviation",
        deviation_score_pool=pool,
    )
    model.train()
    model(graph, update_prototypes=True)
    model.eval()

    output = model(graph)
    score = model.anomaly_score(output)

    assert score.shape == (2,)
    assert torch.isfinite(score).all()


def test_single_node_relation_diagnostics_keep_node_axis() -> None:
    graph = GraphSample(
        x=torch.ones((1, 3)),
        node_type=torch.zeros(1, dtype=torch.long),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_type=torch.empty(0, dtype=torch.long),
        label=0,
        graph_id=0,
    )
    model = HRAGNN(
        input_dim=3,
        hidden_dim=4,
        output_dim=4,
        num_node_types=1,
        num_edge_types=1,
        num_layers=1,
    )

    output = model(graph)

    assert output.auxiliary is not None
    assert output.auxiliary["node_relation_deviation"].shape == (1,)
    assert "relation_deviation" not in output.auxiliary
