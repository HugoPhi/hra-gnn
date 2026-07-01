from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def anomaly_metrics(labels: list[int], scores: list[float]) -> dict[str, float]:
    labels_array = np.asarray(labels, dtype=np.int64)
    scores_array = np.asarray(scores, dtype=np.float64)
    if np.unique(labels_array).size < 2:
        return {"auc": math.nan, "ap": math.nan}
    return {
        "auc": float(roc_auc_score(labels_array, scores_array)),
        "ap": float(average_precision_score(labels_array, scores_array)),
    }
