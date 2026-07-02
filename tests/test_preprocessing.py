from pathlib import Path

import pandas as pd

from hra_gnn.data import PackedGraphDataset
from hra_gnn.preprocessing import prepare_adfa_ld, prepare_hdfs


def test_prepare_hdfs_writes_loadable_packed_graphs(tmp_path: Path) -> None:
    structured = tmp_path / "HDFS.log_structured.csv"
    labels = tmp_path / "anomaly_label.csv"
    pd.DataFrame(
        [
            {
                "Content": "Receiving blk_1",
                "EventId": "E1",
                "Component": "DataNode",
            },
            {
                "Content": "Stored blk_1",
                "EventId": "E2",
                "Component": "DataNode",
            },
            {
                "Content": "Receiving blk_2",
                "EventId": "E1",
                "Component": "DataNode",
            },
            {
                "Content": "Failed blk_2",
                "EventId": "E3",
                "Component": "NameNode",
            },
        ]
    ).to_csv(structured, index=False)
    pd.DataFrame(
        [{"BlockId": "blk_1", "Label": "Normal"}, {"BlockId": "blk_2", "Label": "Anomaly"}]
    ).to_csv(labels, index=False)
    output = prepare_hdfs(structured, labels, tmp_path / "packed")
    dataset = PackedGraphDataset(output)
    assert len(dataset) == 2
    assert dataset[0].x.shape[1] == 8
    assert dataset[1].label == 1
    assert dataset[0].edge_type.max().item() < dataset.metadata["num_edge_types"]


def test_prepare_adfa_ld_preserves_official_source_splits(tmp_path: Path) -> None:
    root = tmp_path / "ADFA-LD"
    directories = [
        root / "Training_Data_Master",
        root / "Validation_Data_Master",
        root / "Attack_Data_Master" / "Hydra_FTP",
    ]
    for directory in directories:
        directory.mkdir(parents=True)
    (directories[0] / "train.txt").write_text("1 2 3 2", encoding="utf-8")
    (directories[1] / "val1.txt").write_text("1 3 2", encoding="utf-8")
    (directories[1] / "val2.txt").write_text("2 3 1", encoding="utf-8")
    (directories[2] / "attack.txt").write_text("9 9 2", encoding="utf-8")
    output = prepare_adfa_ld(root, tmp_path / "packed", seed=7)
    dataset = PackedGraphDataset(output)
    assert len(dataset) == 4
    assert sum(dataset[index].label for index in range(len(dataset))) == 1
    assert (output / "train_ids.txt").read_text().strip()
    assert (output / "validation_ids.txt").read_text().strip()
    assert (output / "test_ids.txt").read_text().strip()
    assert dataset.metadata["unknown_node_type"] == 0
