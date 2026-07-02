import numpy as np

from hra_gnn.rescoring import (
    calibrated_max_scores,
    empirical_normal_percentile,
)


def test_empirical_percentile_uses_only_normal_reference() -> None:
    values = empirical_normal_percentile(
        [1.0, 2.0, 3.0],
        [0.0, 2.0, 4.0],
    )
    assert np.allclose(values, [0.0, 0.5, 0.75])


def test_calibrated_max_can_be_high_when_either_component_is_high() -> None:
    scores = calibrated_max_scores(
        normal_svdd=[1.0, 2.0, 3.0],
        normal_ssl=[10.0, 20.0, 30.0],
        svdd=[4.0, 0.0],
        ssl=[0.0, 40.0],
    )
    assert np.allclose(scores, [0.75, 0.75])
