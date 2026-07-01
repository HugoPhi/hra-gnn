from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def merge_config(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    return _deep_merge(base, update)


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    parent = config.pop("extends", None)
    if parent:
        parent_path = (path.parent / parent).resolve()
        config = _deep_merge(load_config(parent_path), config)

    config["_config_path"] = str(path)
    return config


def apply_overrides(
    config: dict[str, Any], overrides: list[str] | None
) -> dict[str, Any]:
    updated = copy.deepcopy(config)
    for override in overrides or []:
        if "=" not in override:
            raise ValueError(f"Override must be key=value, got: {override}")
        dotted_key, raw_value = override.split("=", 1)
        value = yaml.safe_load(raw_value)
        cursor = updated
        parts = dotted_key.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value
    return updated


def save_resolved_config(config: dict[str, Any], path: str | Path) -> None:
    serializable = {
        key: value for key, value in config.items() if not key.startswith("_")
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(serializable, handle, sort_keys=False, allow_unicode=True)


def validate_config(config: dict[str, Any]) -> None:
    required = ("dataset", "model", "training", "evaluation", "output")
    missing = [section for section in required if section not in config]
    if missing:
        raise ValueError(f"Missing config sections: {', '.join(missing)}")

    fusion = config["model"].get("relation_fusion", "deviation_attention")
    if fusion not in {"deviation_attention", "semantic_attention", "static_concat"}:
        raise ValueError(f"Unsupported relation_fusion: {fusion}")

    readout = config["model"].get("readout", "hybrid")
    if readout not in {"hybrid", "max", "mean"}:
        raise ValueError(f"Unsupported readout: {readout}")
    architecture = config["model"].get("architecture", "hra")
    if architecture not in {
        "hra",
        "ochetgcn",
        "hrgcn",
        "hgt",
        "deeptralog",
        "glocalkd",
    }:
        raise ValueError(f"Unsupported architecture: {architecture}")
    score_mode = config["evaluation"].get("score_mode", "paper_product")
    if score_mode not in {
        "svdd",
        "ssl",
        "weighted_sum",
        "product",
        "paper_product",
    }:
        raise ValueError(f"Unsupported score_mode: {score_mode}")
