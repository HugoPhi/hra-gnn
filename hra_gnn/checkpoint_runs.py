from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml

from .config import apply_overrides, load_config
from .recent_experiments import MODEL_NAMES, _run_model
from .trainer import Trainer, result_directory


def _trainer_metrics_path(config: dict[str, Any]) -> Path:
    return result_directory(config, int(config["training"].get("seed", 42))) / "metrics.json"


def _fair_metrics_path(config: dict[str, Any], model: str) -> Path:
    seed = int(config["training"].get("seed", 42))
    return (
        Path(config["output"].get("results_root", "artifacts/results"))
        / config["dataset"]["name"]
        / MODEL_NAMES[model]
        / f"seed_{seed}"
        / "metrics.json"
    )


def run_best_checkpoint_matrix(
    path: str | Path, *, resume: bool = True
) -> tuple[Path, pd.DataFrame]:
    path = Path(path).resolve()
    matrix = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = Path(matrix.get("results_root", "artifacts/results/best_run_checkpoints"))
    root.mkdir(parents=True, exist_ok=True)
    external_root = matrix.get("external_root", "external")
    rows: list[dict[str, Any]] = []

    for job in matrix["jobs"]:
        if not job.get("enabled", True):
            continue
        config_path = (path.parent / job["config"]).resolve()
        config = load_config(config_path)
        overrides = list(job.get("overrides", []))
        seed = int(job["seed"])
        method = job["method"]
        runner = job.get("runner", "trainer")
        overrides.extend(
            [
                f"training.seed={seed}",
                f"output.results_root={root}",
                f"experiment.stage={matrix.get('stage', 'best_run_checkpoint_rerun')}",
            ]
        )
        if runner == "trainer":
            overrides.append(f"output.run_name={method}")
        else:
            overrides.append(
                f"recent_baseline.experimental_stage={matrix.get('stage', 'best_run_checkpoint_rerun')}"
            )
        config = apply_overrides(config, overrides)
        try:
            if runner == "trainer":
                metrics_path = _trainer_metrics_path(config)
                if resume and metrics_path.exists():
                    summary = json.loads(metrics_path.read_text(encoding="utf-8"))
                else:
                    summary = Trainer(config).train()
            elif runner == "fair":
                fair_model = job["fair_model"]
                metrics_path = _fair_metrics_path(config, fair_model)
                if resume and metrics_path.exists():
                    summary = json.loads(metrics_path.read_text(encoding="utf-8"))
                else:
                    summary = _run_model(config, fair_model, external_root)
            else:
                raise ValueError(f"Unsupported runner: {runner}")
            rows.append(
                {
                    **summary,
                    "method": method,
                    "requested_seed": seed,
                    "runner": runner,
                    "status": "complete",
                    "error": "",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "method": method,
                    "dataset": config["dataset"]["name"],
                    "requested_seed": seed,
                    "runner": runner,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        pd.DataFrame(rows).to_csv(root / "runs.csv", index=False)
    return root, pd.DataFrame(rows)
