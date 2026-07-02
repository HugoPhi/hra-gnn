from types import SimpleNamespace

import torch

from hra_gnn.recent_baselines import (
    _anchor_graph_loss,
    _limited_splits,
    _score_audit,
)


def test_limited_splits_are_seeded_and_stratified() -> None:
    dataset = [SimpleNamespace(label=0 if index < 20 else 1) for index in range(40)]
    splits = {"test": list(range(40))}

    first = _limited_splits(dataset, splits, 10, seed=11)["test"]
    repeated = _limited_splits(dataset, splits, 10, seed=11)["test"]
    second = _limited_splits(dataset, splits, 10, seed=22)["test"]

    assert first == repeated
    assert first != second
    assert sum(dataset[index].label == 0 for index in first) == 5
    assert sum(dataset[index].label == 1 for index in first) == 5


def test_fixed_reference_score_does_not_depend_on_target_batch() -> None:
    generator = torch.Generator().manual_seed(7)
    left = torch.randn(3, 8, generator=generator)
    right = torch.randn(3, 8, generator=generator)
    anchor_left = torch.randn(5, 8, generator=generator)
    anchor_right = torch.randn(5, 8, generator=generator)

    batched = _anchor_graph_loss(left, right, anchor_left, anchor_right)
    separate = torch.cat(
        [
            _anchor_graph_loss(
                left[index : index + 1],
                right[index : index + 1],
                anchor_left,
                anchor_right,
            )
            for index in range(3)
        ]
    )

    assert torch.allclose(batched, separate, atol=1e-6)


def test_score_audit_reports_inverse_direction_without_flipping() -> None:
    audit = _score_audit(
        labels=[0, 0, 1, 1],
        scores=[4.0, 3.0, 2.0, 1.0],
        normal_scores=[2.5, 3.5],
    )

    assert audit["score_direction"] == "higher_is_more_anomalous"
    assert audit["inverse_auc_diagnostic"] == 1.0
    assert audit["test_normal_score_median"] == 3.5
    assert audit["test_anomaly_score_median"] == 1.5
