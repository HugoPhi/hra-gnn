from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hra_gnn.metrics import anomaly_metrics


def _score_columns(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    svdd = frame["svdd_score"].to_numpy(dtype=np.float64)
    ssl_anomaly = frame["ssl_anomaly_score"].to_numpy(dtype=np.float64)
    ssl_probability = frame["ssl_probability"].to_numpy(dtype=np.float64)
    return {
        "svdd": svdd,
        "ssl_anomaly": ssl_anomaly,
        "product_svdd_x_ssl_anomaly": svdd * ssl_anomaly,
        "paper_product_svdd_x_1_plus_ssl_anomaly": svdd * (1.0 + ssl_anomaly),
        "wrong_product_svdd_x_ssl_probability": svdd * ssl_probability,
    }


def _dataset_from_path(path: Path) -> str:
    parts = path.parts
    if "results" in parts:
        index = parts.index("results")
        if index + 1 < len(parts):
            return parts[index + 1]
    return "unknown"


def _seed_from_path(path: Path) -> int | None:
    for part in reversed(path.parts):
        if part.startswith("seed_"):
            return int(part.split("_", maxsplit=1)[1])
    return None


def rescore(paths: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, float | int | str | None]] = []
    for path in paths:
        frame = pd.read_csv(path)
        required = {"label", "svdd_score", "ssl_anomaly_score", "ssl_probability"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        labels = frame["label"].astype(int).tolist()
        for mode, scores in _score_columns(frame).items():
            metrics = anomaly_metrics(labels, scores.tolist())
            rows.append(
                {
                    "dataset": _dataset_from_path(path),
                    "seed": _seed_from_path(path),
                    "path": str(path),
                    "mode": mode,
                    "auc": metrics["auc"],
                    "ap": metrics["ap"],
                    "num_graphs": len(frame),
                    "prevalence": metrics["prevalence"],
                }
            )
    runs = pd.DataFrame(rows)
    summary = (
        runs.groupby(["dataset", "mode"], dropna=False)
        .agg(
            auc_mean=("auc", "mean"),
            auc_std=("auc", "std"),
            ap_mean=("ap", "mean"),
            ap_std=("ap", "std"),
            n=("seed", "count"),
        )
        .reset_index()
    )
    return runs, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    arguments = parser.parse_args()

    paths: list[Path] = []
    for pattern in arguments.input:
        matches = sorted(Path().glob(pattern))
        if not matches:
            raise FileNotFoundError(f"No files matched: {pattern}")
        paths.extend(matches)
    output_dir = Path(arguments.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs, summary = rescore(paths)
    runs.to_csv(output_dir / "offline_rescore_runs.csv", index=False)
    summary.to_csv(output_dir / "offline_rescore_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
