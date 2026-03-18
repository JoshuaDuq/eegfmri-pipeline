"""Runtime loader for behavior-specific YAML defaults."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from eeg_pipeline.utils.config.loader import ConfigDict, resolve_config_paths

BEHAVIOR_CONFIG_ENV_VAR = "EEG_PIPELINE_BEHAVIOR_CONFIG"


def _resolve_behavior_config_path(
    config_path: Optional[str | Path] = None,
) -> Path:
    if config_path is not None:
        return Path(config_path).expanduser().resolve()

    env_path = os.getenv(BEHAVIOR_CONFIG_ENV_VAR)
    if env_path:
        return Path(env_path).expanduser().resolve()

    return (Path(__file__).parent / "behavior_config.yaml").resolve()


def load_behavior_config(
    config_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Load the behavior YAML config as a resolved dictionary."""
    resolved_path = _resolve_behavior_config_path(config_path)
    if not resolved_path.exists():
        return {}

    with open(resolved_path, "r", encoding="utf-8") as handle:
        parsed = yaml.safe_load(handle) or {}

    if not isinstance(parsed, dict):
        return {}

    return resolve_config_paths(parsed, resolved_path)


def _merge_non_null(base: Dict[str, Any], extra: Dict[str, Any]) -> None:
    for key, value in extra.items():
        if value is None:
            continue
        if isinstance(value, dict):
            existing = base.get(key)
            if isinstance(existing, dict):
                _merge_non_null(existing, value)
            elif key not in base or base.get(key) is None:
                existing = {}
                base[key] = existing
                _merge_non_null(existing, value)
            continue
        if key not in base or base.get(key) is None:
            base[key] = copy.deepcopy(value)


def apply_behavior_config_defaults(
    config: Dict[str, Any],
    config_path: Optional[str | Path] = None,
) -> None:
    """Merge behavior YAML values into runtime config, skipping null entries."""
    behavior_defaults = load_behavior_config(config_path=config_path)
    if not behavior_defaults:
        return
    _merge_non_null(config, behavior_defaults)


def ensure_behavior_config(
    config: Optional[Dict[str, Any]] = None,
    config_path: Optional[str | Path] = None,
) -> ConfigDict:
    """Return a copy of config with behavior defaults merged in."""
    merged: Dict[str, Any] = copy.deepcopy(config or {})
    apply_behavior_config_defaults(merged, config_path=config_path)
    return ConfigDict(merged)
