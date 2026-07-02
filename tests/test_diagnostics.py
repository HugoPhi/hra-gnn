from pathlib import Path

from hra_gnn.config import load_config, merge_config
from hra_gnn.diagnostics import graph_statistics
from hra_gnn.preprocessing import prepare_adfa_ld


def test_graph_statistics_uses_packed_dataset_metadata(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "raw"
    for directory, content in (
        ("Training_Data_Master", "1 2 3"),
        ("Validation_Data_Master", "2 3 4"),
        ("Attack_Data_Master/Adduser_1", "4 5 6"),
    ):
        path = raw / directory / "trace.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    packed = prepare_adfa_ld(raw, tmp_path / "packed")
    config = merge_config(
        load_config("configs/adfa_ld.yaml"),
        {
            "dataset": {
                "root": str(packed),
                "num_node_types": 1,
            }
        },
    )

    statistics, _ = graph_statistics(config)

    assert "node_type_3" in statistics
