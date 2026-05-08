"""Runtime loader for Study 2 YAML defaults."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

from eeg_pipeline.utils.config.loader import resolve_config_paths


STUDY2_CONFIG_ENV_VAR = "PAIN_STUDY_STUDY2_CONFIG"


def _resolve_study2_config_path(config_path: str | Path | None = None) -> Path:
    if config_path is not None:
        return Path(config_path).expanduser().resolve()

    env_path = os.getenv(STUDY2_CONFIG_ENV_VAR)
    if env_path:
        return Path(env_path).expanduser().resolve()

    return (Path(__file__).parent / "study2_config.yaml").resolve()


def load_study2_config(config_path: str | Path | None = None) -> dict[str, Any]:
    resolved_path = _resolve_study2_config_path(config_path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Study 2 config file not found: {resolved_path}")

    with open(resolved_path, "r", encoding="utf-8") as handle:
        parsed = yaml.safe_load(handle) or {}

    if not isinstance(parsed, dict):
        raise ValueError(f"Study 2 config must be a YAML mapping: {resolved_path}")

    return resolve_config_paths(parsed, resolved_path)


def _merge_non_null(base: dict[str, Any], extra: dict[str, Any]) -> None:
    for key, value in extra.items():
        if value is None:
            continue
        if isinstance(value, dict):
            existing = base.get(key)
            if not isinstance(existing, dict):
                existing = {}
                base[key] = existing
            _merge_non_null(existing, value)
            continue
        base[key] = copy.deepcopy(value)


def apply_study2_config_defaults(
    config: dict[str, Any],
    config_path: str | Path | None = None,
) -> None:
    defaults = load_study2_config(config_path=config_path)
    _merge_non_null(config, defaults)


__all__ = [
    "STUDY2_CONFIG_ENV_VAR",
    "apply_study2_config_defaults",
    "load_study2_config",
]

