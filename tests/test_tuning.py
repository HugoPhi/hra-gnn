from pathlib import Path

import pytest
import yaml

from hra_gnn.config import load_config, save_resolved_config
from hra_gnn.plotting import plot_tuning_marginals
from hra_gnn.tuning import (
    _combinations,
    _validate_parameter_types,
    merge_hyperparameter_search,
    run_hyperparameter_search,
)


def test_tuning_preserves_types_and_ranks_validation_trials(
    tmp_path: Path,
) -> None:
    base_path = tmp_path / "base.yaml"
    save_resolved_config(load_config("configs/synthetic.yaml"), base_path)
    search_path = tmp_path / "search.yaml"
    search_path.write_text(
        yaml.safe_dump(
            {
                "name": "pytest_tuning",
                "base_config": "base.yaml",
                "results_root": str(tmp_path / "search_results"),
                "seeds": [3],
                "common": {
                    "experiment": {
                        "stage": "pytest",
                        "final_evaluation_split": "validation",
                    },
                    "training": {"epochs": 1},
                },
                "parameters": {
                    "training.learning_rate": [0.00003, 0.0001],
                    "model.deviation_weight": [0.5],
                    "ssl.loss_weight": [0.01],
                },
                "selection": {
                    "split": "validation",
                    "metric": "auc_ap_mean",
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    root, runs = run_hyperparameter_search(search_path)
    assert len(runs) == 2
    assert set(runs["evaluation_split"]) == {"validation"}
    assert not list((root / "model_runs").rglob("checkpoints"))

    resolved = yaml.safe_load(
        next((root / "model_runs").rglob("config.yaml")).read_text(encoding="utf-8")
    )
    assert isinstance(resolved["training"]["learning_rate"], float)

    _, merged, ranking = merge_hyperparameter_search(search_path)
    assert len(merged) == 2
    assert list(ranking["rank"]) == [1, 2]
    assert (root / "best_parameters.yaml").exists()

    figure = tmp_path / "tuning.svg"
    plot_tuning_marginals(root / "runs.csv", figure)
    assert figure.exists()


def test_explicit_trials_keep_correlated_parameters() -> None:
    trials = [
        {"augmentation.edge_addition_rate": 0.0, "ssl.loss_weight": 0.0},
        {"augmentation.edge_addition_rate": 0.2, "ssl.loss_weight": 0.01},
    ]
    assert _combinations({"trials": trials}) == trials


def test_numeric_search_parameter_rejects_string() -> None:
    with pytest.raises(TypeError, match="training.learning_rate"):
        _validate_parameter_types(
            {"training": {"learning_rate": 0.001}},
            [{"training.learning_rate": "3e-5"}],
        )
