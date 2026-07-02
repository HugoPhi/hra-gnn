from pathlib import Path

import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from hra_gnn.config import load_config, merge_config
from hra_gnn.trainer import Trainer


def test_smoke_training_writes_separate_results(tmp_path: Path) -> None:
    config = load_config("configs/synthetic.yaml")
    config = merge_config(
        config,
        {
            "training": {"epochs": 1, "seed": 9},
            "evaluation": {"collect_diagnostics": True},
            "output": {
                "results_root": str(tmp_path / "results"),
                "figures_root": str(tmp_path / "figures"),
                "run_name": "pytest",
            },
        },
    )
    summary = Trainer(config).train()
    run_dir = tmp_path / "results" / "Synthetic" / "pytest" / "seed_9"
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "history.csv").exists()
    assert (run_dir / "test_predictions.csv").exists()
    assert (run_dir / "test_relations.csv").exists()
    assert not (tmp_path / "figures").exists()
    assert 0.0 <= summary["auc"] <= 1.0


def test_tensorboard_monitoring_writes_two_split_metrics(tmp_path: Path) -> None:
    config = load_config("configs/synthetic.yaml")
    config = merge_config(
        config,
        {
            "training": {"epochs": 1, "seed": 10},
            "monitoring": {
                "enabled": True,
                "log_dir": str(tmp_path / "tensorboard"),
            },
            "output": {
                "results_root": str(tmp_path / "results"),
                "figures_root": str(tmp_path / "figures"),
                "run_name": "monitoring",
            },
        },
    )
    Trainer(config).train()
    history = pd.read_csv(
        tmp_path / "results" / "Synthetic" / "monitoring" / "seed_10" / "history.csv"
    )
    assert "monitor_validation_auc" in history
    assert "monitor_test_auc" in history
    assert "monitor_validation_ap" in history
    assert "monitor_test_ap" in history
    event_files = list((tmp_path / "tensorboard").rglob("events.out.tfevents.*"))
    assert event_files
    accumulator = EventAccumulator(str(event_files[0]))
    accumulator.Reload()
    assert "Synthetic/AP/validation" in accumulator.Tags()["scalars"]
    assert "Synthetic/AP/test" in accumulator.Tags()["scalars"]


def test_evaluation_sampling_is_seeded_and_preserves_labels(
    tmp_path: Path,
) -> None:
    base = load_config("configs/synthetic.yaml")
    update = {
        "training": {"seed": 17},
        "evaluation": {
            "max_graphs": 4,
            "sampling": "seeded_stratified",
        },
        "output": {"results_root": str(tmp_path / "first")},
    }
    first = Trainer(merge_config(base, update))
    update["output"]["results_root"] = str(tmp_path / "second")
    second = Trainer(merge_config(base, update))

    assert first.evaluation_splits["test"] == second.evaluation_splits["test"]
    assert len(first.evaluation_splits["test"]) == 4
    original_labels = [first._label_for_index(index) for index in first.splits["test"]]
    sampled_labels = [
        first._label_for_index(index) for index in first.evaluation_splits["test"]
    ]
    assert set(sampled_labels) == set(original_labels)
