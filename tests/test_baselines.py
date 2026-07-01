import pytest
import torch

from hra_gnn.config import load_config, merge_config
from hra_gnn.data import SyntheticGraphDataset
from hra_gnn.graph import batch_graphs
from hra_gnn.trainer import build_model


@pytest.mark.parametrize(
    "architecture", ["ochetgcn", "hrgcn", "hgt", "deeptralog", "glocalkd"]
)
def test_baseline_forward(architecture: str) -> None:
    config = merge_config(
        load_config("configs/synthetic.yaml"),
        {"model": {"architecture": architecture}},
    )
    model = build_model(config)
    graph = SyntheticGraphDataset(num_graphs=8)[0]
    output = model(graph)
    assert output.embedding.shape == (16,)
    assert torch.isfinite(output.embedding).all()
    if architecture == "glocalkd":
        score = model.anomaly_score(output)
    else:
        model.initialize_svdd_center(output.embedding.unsqueeze(0))
        score = model.anomaly_score(output)
    assert torch.isfinite(score)


@pytest.mark.parametrize(
    "architecture", ["ochetgcn", "hrgcn", "hgt", "deeptralog", "glocalkd"]
)
def test_baseline_batched_forward(architecture: str) -> None:
    config = merge_config(
        load_config("configs/synthetic.yaml"),
        {"model": {"architecture": architecture}},
    )
    model = build_model(config)
    dataset = SyntheticGraphDataset(num_graphs=8)
    output = model(batch_graphs([dataset[index] for index in range(3)]))
    assert output.embedding.shape == (3, 16)
    if architecture != "glocalkd":
        model.initialize_svdd_center(output.embedding)
    score = model.anomaly_score(output)
    assert score.shape == (3,)
    assert torch.isfinite(score).all()
