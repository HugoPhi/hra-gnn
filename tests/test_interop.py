from pathlib import Path

import pandas as pd

from hra_gnn.config import load_config, merge_config
from hra_gnn.interop import export_tu_dataset


def test_export_tu_dataset_preserves_graphs_and_splits(tmp_path: Path) -> None:
    config = merge_config(
        load_config("configs/synthetic.yaml"),
        {
            "dataset": {"num_graphs": 12},
            "training": {"seed": 5},
        },
    )
    root = export_tu_dataset(config, tmp_path, "SyntheticFair")
    raw = root / "raw"
    assert (raw / "SyntheticFair_A.txt").exists()
    assert (raw / "SyntheticFair_node_attributes.txt").exists()
    labels = (raw / "SyntheticFair_graph_labels.txt").read_text().splitlines()
    assert len(labels) == 12
    assert set(labels) == {"1", "-1"}
    mapping = pd.read_csv(root / "split_mapping.csv")
    assert set(mapping["split"]) == {"train", "validation", "test"}
    assert mapping["graph_id"].is_unique
