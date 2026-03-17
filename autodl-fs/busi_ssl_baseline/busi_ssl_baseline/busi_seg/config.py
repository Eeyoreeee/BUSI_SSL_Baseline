"""Minimal YAML config loader for the BUSI SSL baseline.

Design constraints for stage 1:
- YAML only
- no Hydra
- only one level of base inheritance
- base configs cannot inherit another base
- merge only known top-level sections with shallow dict updates
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


TOP_LEVEL_MERGE_SECTIONS = {
    "experiment",
    "data",
    "model",
    "loss",
    "optimizer",
    "train",
    "ssl",
    "analysis",
    "eval",
    "logging",
}


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Config file does not exist: {path}")

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, dict):
        raise TypeError(f"Top-level YAML object must be a mapping: {path}")

    return data


def _resolve_base_path(config_path: Path, base_value: str) -> Path:
    base_path = (config_path.parent / base_value).resolve()
    if not base_path.is_file():
        raise FileNotFoundError(
            f"Base config referenced by {config_path} does not exist: {base_path}"
        )
    return base_path


def _merge_known_sections(
    base_config: dict[str, Any], child_config: dict[str, Any]
) -> dict[str, Any]:
    merged = deepcopy(base_config)

    for key, value in child_config.items():
        if key == "base":
            continue

        if (
            key in TOP_LEVEL_MERGE_SECTIONS
            and isinstance(merged.get(key), dict)
            and isinstance(value, dict)
        ):
            merged[key] = {**merged[key], **value}
            continue

        merged[key] = deepcopy(value)

    return merged


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a config with at most one level of base inheritance."""

    path = Path(config_path).resolve()
    config = _read_yaml(path)

    base_value = config.get("base")
    if base_value is None:
        config["_config_path"] = str(path)
        config["_base_path"] = None
        return config

    if not isinstance(base_value, str):
        raise TypeError(f"Config 'base' must be a string path: {path}")

    base_path = _resolve_base_path(path, base_value)
    base_config = _read_yaml(base_path)
    if "base" in base_config:
        raise ValueError(
            "Base configs cannot inherit another base. "
            f"Found nested base in {base_path}."
        )

    merged = _merge_known_sections(base_config=base_config, child_config=config)
    merged["_config_path"] = str(path)
    merged["_base_path"] = str(base_path)
    return merged


def dump_config(config: dict[str, Any], output_path: str | Path) -> None:
    """Write a resolved config to disk for experiment reproducibility."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
