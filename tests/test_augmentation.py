import pytest

from hra_gnn.augment import HeterogeneousAugmentor
from hra_gnn.data import SyntheticGraphDataset


@pytest.fixture
def graph():
    return SyntheticGraphDataset(num_graphs=8)[0]


@pytest.fixture
def augmentor():
    return HeterogeneousAugmentor(
        num_node_types=3,
        num_edge_types=2,
        edge_perturbation_rate=0.2,
        edge_addition_rate=0.2,
        node_type_swap_rate=0.2,
        edge_type_swap_rate=0.2,
        preserve_observed_schema=True,
        seed=3,
    )


@pytest.mark.parametrize("method", HeterogeneousAugmentor.METHODS)
def test_augmentation_returns_valid_graph(graph, augmentor, method) -> None:
    result = augmentor.augment(graph, method)
    assert result.method == method
    assert result.graph.x.shape[0] == graph.x.shape[0]
    assert result.graph.edge_index.shape[0] == 2
    assert result.graph.edge_type.shape[0] == result.graph.edge_index.shape[1]
