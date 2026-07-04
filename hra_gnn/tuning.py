from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
import shutil
from numbers import Number
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .config import load_config, merge_config
from .data import load_dataset
from .trainer import Trainer, result_directory


def _load_search(path: str | Path) -> tuple[Path, dict[str, Any]]:
    search_path = Path(path).resolve()
    search = yaml.safe_load(search_path.read_text(encoding="utf-8"))
    return search_path, search


def _combinations(search: dict[str, Any]) -> list[dict[str, Any]]:
    if "trials" in search:
        trials = search["trials"]
        if not isinstance(trials, list) or not trials:
            raise ValueError("Search trials must be a non-empty list")
        if not all(isinstance(trial, dict) for trial in trials):
            raise ValueError("Every explicit search trial must be a mapping")
        expected = set(trials[0])
        if any(set(trial) != expected for trial in trials[1:]):
            raise ValueError("Explicit search trials must use the same parameter keys")
        return [copy.deepcopy(trial) for trial in trials]
    parameters = search["parameters"]
    names = list(parameters)
    values = [parameters[name] for name in names]
    return [
        dict(zip(names, combination, strict=True))
        for combination in itertools.product(*values)
    ]


def _parameter_names(search: dict[str, Any]) -> list[str]:
    if "parameters" in search:
        return list(search["parameters"])
    names: list[str] = []
    for trial in search["trials"]:
        for name in trial:
            if name not in names:
                names.append(name)
    return names


def _dotted_value(config: dict[str, Any], dotted_key: str) -> Any:
    value: Any = config
    for part in dotted_key.split("."):
        value = value[part]
    return value


def _validate_parameter_types(
    base: dict[str, Any], combinations: list[dict[str, Any]]
) -> None:
    for trial_index, parameters in enumerate(combinations):
        for name, value in parameters.items():
            original = _dotted_value(base, name)
            if isinstance(original, Number) and not isinstance(value, Number):
                raise TypeError(
                    f"Trial {trial_index} parameter {name} must be numeric, "
                    f"got {type(value).__name__}: {value!r}"
                )


def _trial_id(index: int, parameters: dict[str, Any]) -> str:
    payload = json.dumps(parameters, sort_keys=True, ensure_ascii=True)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:8]
    return f"trial_{index:04d}_{digest}"


def _apply_parameters(
    config: dict[str, Any], parameters: dict[str, Any]
) -> dict[str, Any]:
    updated = copy.deepcopy(config)
    for dotted_key, value in parameters.items():
        cursor = updated
        parts = dotted_key.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = copy.deepcopy(value)
    return updated


def _selection_score(rows: pd.DataFrame, metric: str) -> pd.Series:
    if metric == "auc_ap_mean":
        return (rows["auc"] + rows["ap"]) / 2.0
    if metric == "negative_mean_svdd_loss":
        return -rows["mean_svdd_loss"]
    if metric not in rows:
        raise ValueError(f"Selection metric is unavailable: {metric}")
    return rows[metric]


def _result_root(search_path: Path, search: dict[str, Any]) -> Path:
    return (
        Path(search.get("results_root", "artifacts/results/hyperparameter_search"))
        / search["name"]
    )


def run_hyperparameter_search(
    path: str | Path,
    *,
    resume: bool = True,
    shard_index: int = 0,
    num_shards: int = 1,
) -> tuple[Path, pd.DataFrame]:
    search_path, search = _load_search(path)
    if not 0 <= shard_index < num_shards:
        raise ValueError("shard_index must be in [0, num_shards)")

    base = load_config((search_path.parent / search["base_config"]).resolve())
    base = merge_config(base, search.get("common", {}))
    combinations = _combinations(search)
    _validate_parameter_types(base, combinations)
    dataset = load_dataset(base["dataset"])
    retain_checkpoints = bool(search.get("retain_checkpoints", False))
    root = _result_root(search_path, search)
    root.mkdir(parents=True, exist_ok=True)
    model_results_root = root / "model_runs"
    seeds = [int(seed) for seed in search.get("seeds", [11])]
    rows: list[dict[str, Any]] = []

    for trial_index, parameters in enumerate(combinations):
        if trial_index % num_shards != shard_index:
            continue
        trial = _trial_id(trial_index, parameters)
        for seed in seeds:
            print(
                f"[tuning] trial={trial_index + 1}/{len(combinations)} "
                f"id={trial} seed={seed} parameters={parameters}",
                flush=True,
            )
            config = _apply_parameters(base, parameters)
            config = merge_config(
                config,
                {
                    "training": {"seed": seed},
                    "output": {
                        "results_root": str(model_results_root),
                        "run_name": trial,
                    },
                },
            )
            run_directory = result_directory(config, seed)
            metrics_path = run_directory / "metrics.json"
            if resume and metrics_path.exists():
                summary = json.loads(metrics_path.read_text(encoding="utf-8"))
            else:
                if not retain_checkpoints:
                    shutil.rmtree(run_directory / "checkpoints", ignore_errors=True)
                summary = Trainer(config, dataset=dataset).train()
            row = {
                **summary,
                "search": search["name"],
                "trial_index": trial_index,
                "trial_id": trial,
                **parameters,
            }
            print(
                f"[tuning] completed id={trial} seed={seed} "
                f"auc={float(summary['auc']):.6f} "
                f"ap={float(summary['ap']):.6f}",
                flush=True,
            )
            rows.append(row)
            pd.DataFrame(rows).to_csv(
                root / f"runs_shard_{shard_index}.csv", index=False
            )
            if not retain_checkpoints:
                shutil.rmtree(run_directory / "checkpoints", ignore_errors=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(root / f"runs_shard_{shard_index}.csv", index=False)
    return root, frame


def merge_hyperparameter_search(
    path: str | Path,
) -> tuple[Path, pd.DataFrame, pd.DataFrame]:
    search_path, search = _load_search(path)
    root = _result_root(search_path, search)
    shard_paths = sorted(root.glob("runs_shard_*.csv"))
    if not shard_paths:
        raise FileNotFoundError(f"No shard results found under {root}")
    raw = pd.concat([pd.read_csv(path) for path in shard_paths], ignore_index=True)
    raw = raw.drop_duplicates(["trial_id", "seed"], keep="last")
    raw.to_csv(root / "runs.csv", index=False)

    metric = search["selection"]["metric"]
    raw["selection_score"] = _selection_score(raw, metric)
    parameter_names = _parameter_names(search)
    aggregations: dict[str, list[str]] = {
        "selection_score": ["mean", "std"],
        "auc": ["mean", "std"],
        "ap": ["mean", "std"],
        "mean_svdd_loss": ["mean", "std"],
        "epochs_completed": ["mean"],
        "training_seconds": ["mean"],
    }
    available = {
        column: functions
        for column, functions in aggregations.items()
        if column in raw.columns
    }
    ranking = raw.groupby(
        ["trial_index", "trial_id", *parameter_names], dropna=False
    ).agg(available)
    ranking.columns = [f"{column}_{stat}" for column, stat in ranking.columns]
    ranking = ranking.reset_index()
    ranking = ranking.sort_values(
        ["selection_score_mean", "auc_mean", "ap_mean"],
        ascending=False,
        na_position="last",
    )
    ranking.insert(0, "rank", range(1, len(ranking) + 1))
    ranking.to_csv(root / "ranking.csv", index=False)

    best = ranking.iloc[0]
    best_parameters = {
        name: best[name].item() if hasattr(best[name], "item") else best[name]
        for name in parameter_names
    }
    best_record = {
        "search": search["name"],
        "selection_split": search.get("common", {})
        .get("experiment", {})
        .get("final_evaluation_split", "test"),
        "selection_metric": metric,
        "selection_score": float(best["selection_score_mean"]),
        "parameters": best_parameters,
    }
    (root / "best_parameters.yaml").write_text(
        yaml.safe_dump(best_record, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    if math.isnan(float(best["selection_score_mean"])):
        raise RuntimeError("All hyperparameter selection scores are NaN")
    return root, raw, ranking
