import math

import pytest

from hra_gnn.metrics import anomaly_metrics, normal_score_threshold


def test_ranking_budget_and_threshold_metrics() -> None:
    labels = [0, 0, 0, 0, 1, 1]
    scores = [0.1, 0.2, 0.3, 0.4, 0.8, 0.9]
    metrics = anomaly_metrics(
        labels,
        scores,
        threshold=0.7,
        alert_fraction=0.34,
        target_fpr=0.01,
    )
    assert metrics["auc"] == pytest.approx(1.0)
    assert metrics["ap"] == pytest.approx(1.0)
    assert metrics["precision_at_budget"] == pytest.approx(2 / 3)
    assert metrics["recall_at_budget"] == pytest.approx(1.0)
    assert metrics["f1"] == pytest.approx(1.0)
    assert metrics["mcc"] == pytest.approx(1.0)
    assert metrics["observed_fpr"] == pytest.approx(0.0)


def test_single_class_metrics_are_defined_where_possible() -> None:
    metrics = anomaly_metrics([0, 0, 0], [0.1, 0.2, 0.3])
    assert math.isnan(metrics["auc"])
    assert metrics["alert_budget"] == 1
    assert metrics["precision_at_budget"] == 0.0


def test_threshold_uses_requested_normal_quantile() -> None:
    assert normal_score_threshold([0.0, 1.0, 2.0], 0.5) == pytest.approx(1.0)
