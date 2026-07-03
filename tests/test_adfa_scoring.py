import numpy as np

from hra_gnn.adfa_scoring import fit_markov_counts, markov_nll_scores


def test_markov_nll_prefers_seen_normal_sequence() -> None:
    normal = [[1, 2, 3, 1], [1, 2, 3, 2], [1, 2, 3, 1]]
    counts, contexts = fit_markov_counts(normal, order=3)

    scores = markov_nll_scores(
        [[1, 2, 3, 1], [3, 3, 3, 3]],
        counts,
        contexts,
        order=3,
        vocabulary_size=4,
    )

    assert scores.shape == (2,)
    assert np.isfinite(scores).all()
    assert scores[0] < scores[1]
