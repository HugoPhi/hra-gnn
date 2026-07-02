from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml

from .config import apply_overrides, load_config
from .recent_baselines import (
    run_dual_view_fair,
    run_muse_fair,
    run_signet_fair,
)

MODEL_NAMES = {
    "signet": "SIGNET-fair",
    "cvtgad": "CVTGAD-fair",
    "muse": "MUSE-fair",
    "gladmamba": "GLADMamba-fair",
}


def _run_model(config: dict[str, Any], model: str, external_root: str | Path):
    if model == "signet":
        return run_signet_fair(config, external_root=external_root)
    if model == "muse":
        return run_muse_fair(config, external_root=external_root)
    return run_dual_view_fair(
        config, architecture=model, external_root=external_root
    )


def _metrics_path(config: dict[str, Any], model: str, seed: int) -> Path:
    return (
        Path(config["output"].get("results_root", "artifacts/results"))
        / config["dataset"]["name"]
        / MODEL_NAMES[model]
        / f"seed_{seed}"
        / "metrics.json"
    )


def run_fair_matrix(
    path: str | Path, *, resume: bool = True
) -> tuple[Path, pd.DataFrame]:
    path = Path(path).resolve()
    matrix = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = Path(matrix.get("results_root", "artifacts/results/fair_matrix"))
    root = root / matrix["name"]
    root.mkdir(parents=True, exist_ok=True)
    external_root = matrix.get("external_root", "external")
    seeds = [int(seed) for seed in matrix.get("seeds", [11, 22, 33, 44, 55])]
    rows: list[dict[str, Any]] = []

    for job in matrix["jobs"]:
        if not job.get("enabled", True):
            continue
        config_path = (path.parent / job["config"]).resolve()
        base = load_config(config_path)
        model = job["model"]
        for seed in seeds:
            overrides = list(job.get("overrides", []))
            overrides.append(f"training.seed={seed}")
            config = apply_overrides(base, overrides)
            metrics_path = _metrics_path(config, model, seed)
            try:
                if resume and metrics_path.exists():
                    summary = json.loads(metrics_path.read_text(encoding="utf-8"))
                else:
                    summary = _run_model(config, model, external_root)
                rows.append({**summary, "status": "complete", "error": ""})
            except Exception as exc:
                rows.append(
                    {
                        "dataset": config["dataset"]["name"],
                        "variant": MODEL_NAMES[model],
                        "seed": seed,
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
