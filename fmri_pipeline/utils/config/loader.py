"""Runtime loader for fMRI-specific YAML defaults."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from eeg_pipeline.utils.config.loader import resolve_config_paths

FMRI_CONFIG_ENV_VAR = "EEG_PIPELINE_FMRI_CONFIG"


def _resolve_fmri_config_path(config_path: Optional[str | Path] = None) -> Path:
    if config_path is not None:
        return Path(config_path).expanduser().resolve()

    env_path = os.getenv(FMRI_CONFIG_ENV_VAR)
    if env_path:
        return Path(env_path).expanduser().resolve()

    return (Path(__file__).parent / "fmri_config.yaml").resolve()


def load_fmri_config(config_path: Optional[str | Path] = None) -> Dict[str, Any]:
    """Load the fMRI YAML config as a resolved dictionary."""
    resolved_path = _resolve_fmri_config_path(config_path)
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
            if not isinstance(existing, dict):
                existing = {}
                base[key] = existing
            _merge_non_null(existing, value)
            continue
        base[key] = copy.deepcopy(value)


def apply_fmri_config_defaults(
    config: Dict[str, Any],
    config_path: Optional[str | Path] = None,
) -> None:
    """Merge fMRI YAML values into runtime config, skipping null entries."""
    fmri_defaults = load_fmri_config(config_path=config_path)
    if not fmri_defaults:
        return
    _merge_non_null(config, fmri_defaults)

