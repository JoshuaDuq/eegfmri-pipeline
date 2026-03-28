"""Runtime loader for EEG-BOLD coupling-specific YAML defaults."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from eeg_pipeline.utils.config.loader import resolve_config_paths


EEG_BOLD_COUPLING_CONFIG_ENV_VAR = "EEG_BOLD_COUPLING_CONFIG"


def _resolve_eeg_bold_coupling_config_path(
    config_path: Optional[str | Path] = None,
) -> Path:
    if config_path is not None:
        return Path(config_path).expanduser().resolve()

    env_path = os.getenv(EEG_BOLD_COUPLING_CONFIG_ENV_VAR)
    if env_path:
        return Path(env_path).expanduser().resolve()

    return (Path(__file__).parent / "eeg_bold_coupling_config.yaml").resolve()


def load_eeg_bold_coupling_config(
    config_path: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """Load the EEG-BOLD coupling YAML config as a resolved dictionary."""
    resolved_path = _resolve_eeg_bold_coupling_config_path(config_path)
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


def apply_eeg_bold_coupling_config_defaults(
    config: Dict[str, Any],
    config_path: Optional[str | Path] = None,
) -> None:
    """Merge EEG-BOLD coupling YAML values into runtime config, skipping null entries."""
    coupling_defaults = load_eeg_bold_coupling_config(config_path=config_path)
    if not coupling_defaults:
        return
    _merge_non_null(config, coupling_defaults)


__all__ = [
    "EEG_BOLD_COUPLING_CONFIG_ENV_VAR",
    "apply_eeg_bold_coupling_config_defaults",
    "load_eeg_bold_coupling_config",
]
