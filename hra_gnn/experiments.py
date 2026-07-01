from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .config import apply_overrides, load_config, merge_config
from .trainer import Trainer, result_directory


def _set_seed(config: dict[str, Any], seed: int) -> dict[str, Any]:
    return merge_config(config, {"training": {"seed": seed}})


def _summarize(rows: pd.DataFrame, grouping: list[str]) -> pd.DataFrame:
    metrics = [
        "auc",
        "ap",
        "svdd_auc",
        "svdd_ap",
        "ssl_auc",
        "ssl_ap",
        "training_seconds",
        "mean_inference_seconds",
        "parameters",
    ]
    available = [metric for metric in metrics if metric in rows]
    summary = rows.groupby(grouping, dropna=False)[available].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    return summary.reset_index()


def _run_or_resume(config: dict[str, Any], resume: bool) -> dict[str, Any]:
    metrics_path = (
        result_directory(config, int(config["training"].get("seed", 42)))
        / "metrics.json"
    )
    if resume and metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    return Trainer(config).train()


def run_experiment_suite(
    path: str | Path, *, resume: bool = True
) -> tuple[Path, pd.DataFrame]:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        suite = yaml.safe_load(handle)
    base_config = load_config((path.parent / suite["base_config"]).resolve())
    suite_name = suite["name"]
    root = (
        Path(
            suite.get(
                "results_root",
                base_config["output"].get("results_root", "artifacts/results"),
            )
        )
        / "suites"
        / suite_name
    )
    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for variant_name, variant_update in suite.get("variants", {}).items():
        for seed in suite.get("seeds", [base_config["training"].get("seed", 42)]):
            config = merge_config(base_config, variant_update or {})
            config = merge_config(
                _set_seed(config, int(seed)),
                {"output": {"run_name": f"{suite_name}/{variant_name}"}},
            )
            summary = _run_or_resume(config, resume)
            rows.append(
                {
                    **summary,
                    "suite": suite_name,
                    "variant": variant_name,
                    "sweep": None,
                    "sweep_parameter": None,
                    "sweep_value": None,
                }
            )

    for sweep in suite.get("sweeps", []):
        for value in sweep["values"]:
            for seed in suite.get("seeds", [base_config["training"].get("seed", 42)]):
                config = apply_overrides(
                    _set_seed(base_config, int(seed)),
                    [f"{sweep['parameter']}={value}"],
                )
                run_name = f"{suite_name}/{sweep['name']}/{value}"
                config = merge_config(config, {"output": {"run_name": run_name}})
                summary = _run_or_resume(config, resume)
                rows.append(
                    {
                        **summary,
                        "suite": suite_name,
                        "variant": None,
                        "sweep": sweep["name"],
                        "sweep_parameter": sweep["parameter"],
                        "sweep_value": value,
                    }
                )

    raw = pd.DataFrame(rows)
    raw.to_csv(root / "runs.csv", index=False)
    variant_rows = raw[raw["variant"].notna()]
    if not variant_rows.empty:
        _summarize(variant_rows, ["dataset", "variant"]).to_csv(
            root / "variant_summary.csv", index=False
        )
    sweep_rows = raw[raw["sweep"].notna()]
    if not sweep_rows.empty:
        _summarize(
            sweep_rows,
            ["dataset", "sweep", "sweep_parameter", "sweep_value"],
        ).to_csv(root / "sweep_summary.csv", index=False)
    return root, raw
