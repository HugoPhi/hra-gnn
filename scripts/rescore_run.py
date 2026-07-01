from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recompute saved prediction scores without retraining"
    )
    parser.add_argument("run_directory")
    parser.add_argument(
        "--mode",
        choices=("paper_product", "svdd", "ssl", "product"),
        default="paper_product",
    )
    arguments = parser.parse_args()

    run_directory = Path(arguments.run_directory)
    predictions_path = run_directory / "test_predictions.csv"
    metrics_path = run_directory / "metrics.json"
    predictions = pd.read_csv(predictions_path)
    svdd = predictions["svdd_score"]
    ssl = predictions["ssl_anomaly_score"]
    if arguments.mode == "svdd":
        score = svdd
    elif arguments.mode == "ssl":
        score = ssl
    elif arguments.mode == "product":
        score = svdd * ssl
    else:
        score = svdd * (1.0 + ssl)

    predictions["score"] = score
    predictions.to_csv(predictions_path, index=False)
    labels = predictions["label"].tolist()
    values = score.tolist()
    metrics = {
        "auc": float(roc_auc_score(labels, values)),
        "ap": float(average_precision_score(labels, values)),
    }
    if metrics_path.exists():
        saved = json.loads(metrics_path.read_text(encoding="utf-8"))
        saved.update(metrics)
    else:
        saved = metrics
    saved["score_mode"] = arguments.mode
    metrics_path.write_text(
        json.dumps(saved, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
