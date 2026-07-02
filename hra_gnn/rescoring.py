from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .metrics import anomaly_metrics, normal_score_threshold
from .trainer import Trainer


def empirical_normal_percentile(
    normal_scores: list[float], scores: list[float]
) -> np.ndarray:
    reference = np.sort(np.asarray(normal_scores, dtype=np.float64))
    if reference.size == 0:
        raise ValueError("Normal calibration scores must not be empty")
    values = np.asarray(scores, dtype=np.float64)
    return np.searchsorted(reference, values, side="right") / (reference.size + 1)


def calibrated_max_scores(
    normal_svdd: list[float],
    normal_ssl: list[float],
    svdd: list[float],
    ssl: list[float],
) -> np.ndarray:
    svdd_percentile = empirical_normal_percentile(normal_svdd, svdd)
    ssl_percentile = empirical_normal_percentile(normal_ssl, ssl)
    return np.maximum(svdd_percentile, ssl_percentile)


def rescore_calibrated_max(
    config: dict[str, Any],
    checkpoint: str | Path,
) -> dict[str, Any]:
    trainer = Trainer(config)
    trainer.load_checkpoint(checkpoint)
    normal = trainer.evaluate("train", return_details=True)
    test = trainer.evaluate("test", return_details=True)

    normal_scores = calibrated_max_scores(
        normal["_svdd_scores"],
        normal["_ssl_scores"],
        normal["_svdd_scores"],
        normal["_ssl_scores"],
    )
    test_scores = calibrated_max_scores(
        normal["_svdd_scores"],
        normal["_ssl_scores"],
        test["_svdd_scores"],
        test["_ssl_scores"],
    )
    threshold = normal_score_threshold(
        normal_scores.tolist(),
        float(config["evaluation"].get("threshold_quantile", 0.99)),
    )
    metrics = anomaly_metrics(
        test["_labels"],
        test_scores.tolist(),
        threshold=threshold,
        alert_fraction=float(config["evaluation"].get("alert_fraction", 0.01)),
        target_fpr=float(config["evaluation"].get("target_fpr", 0.01)),
    )
    output = trainer.output_dir
    pd.DataFrame(
        {
            "graph_id": test["_graph_ids"],
            "label": test["_labels"],
            "svdd_score": test["_svdd_scores"],
            "ssl_anomaly_score": test["_ssl_scores"],
            "calibrated_max_score": test_scores,
        }
    ).to_csv(output / "calibrated_max_predictions.csv", index=False)
    summary = {
        **metrics,
        "dataset": config["dataset"]["name"],
        "seed": trainer.seed,
        "score_mode": "normal_ecdf_max",
        "calibration_source": "normal_train_only",
        "checkpoint": str(checkpoint),
        "experimental_stage": "diagnostic_not_final",
    }
    (output / "calibrated_max_metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary
