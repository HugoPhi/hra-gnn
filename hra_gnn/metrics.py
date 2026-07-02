from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)


def normal_score_threshold(scores: list[float], quantile: float = 0.99) -> float:
    if not scores:
        return math.nan
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be between zero and one")
    return float(np.quantile(np.asarray(scores, dtype=np.float64), quantile))


def anomaly_metrics(
    labels: list[int],
    scores: list[float],
    *,
    threshold: float | None = None,
    alert_fraction: float = 0.01,
    target_fpr: float = 0.01,
) -> dict[str, float]:
    labels_array = np.asarray(labels, dtype=np.int64)
    scores_array = np.asarray(scores, dtype=np.float64)
    if np.unique(labels_array).size < 2:
        auc = math.nan
        ap = math.nan
        tpr_at_fpr = math.nan
    else:
        auc = float(roc_auc_score(labels_array, scores_array))
        ap = float(average_precision_score(labels_array, scores_array))
        fpr_values, tpr_values, _ = roc_curve(labels_array, scores_array)
        eligible = tpr_values[fpr_values <= target_fpr + 1e-12]
        tpr_at_fpr = float(eligible.max()) if eligible.size else 0.0

    count = len(labels_array)
    budget = max(1, min(count, int(math.ceil(count * alert_fraction)))) if count else 0
    order = np.argsort(-scores_array, kind="stable")[:budget]
    positives = int(labels_array.sum())
    true_at_budget = int(labels_array[order].sum()) if budget else 0
    result = {
        "prevalence": float(labels_array.mean()) if count else math.nan,
        "auc": auc,
        "ap": ap,
        "alert_fraction": float(alert_fraction),
        "alert_budget": float(budget),
        "precision_at_budget": true_at_budget / budget if budget else math.nan,
        "recall_at_budget": true_at_budget / positives if positives else math.nan,
        "target_fpr": float(target_fpr),
        "tpr_at_fpr": tpr_at_fpr,
    }
    if threshold is None or math.isnan(threshold):
        result.update(
            {
                "threshold": math.nan,
                "precision": math.nan,
                "recall": math.nan,
                "f1": math.nan,
                "mcc": math.nan,
                "observed_fpr": math.nan,
            }
        )
        return result

    predictions = (scores_array >= threshold).astype(np.int64)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels_array,
        predictions,
        average="binary",
        zero_division=0,
    )
    tn, fp, _, _ = confusion_matrix(
        labels_array, predictions, labels=[0, 1]
    ).ravel()
    result.update(
        {
            "threshold": float(threshold),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "mcc": float(matthews_corrcoef(labels_array, predictions)),
            "observed_fpr": float(fp / (fp + tn)) if fp + tn else math.nan,
        }
    )
    return result
