"""Runtime loader for Study 1 YAML defaults."""

from __future__ import annotations

import copy
import math
import os
from pathlib import Path
from typing import Any, Optional

import yaml

from eeg_pipeline.utils.config.loader import resolve_config_paths


STUDY1_CONFIG_ENV_VAR = "PAIN_STUDY_STUDY1_CONFIG"
TEMPORAL_NEGATIVE_CONTROL_TRANSFORM = "raw_log_power"


def _resolve_study1_config_path(config_path: Optional[str | Path] = None) -> Path:
    if config_path is not None:
        return Path(config_path).expanduser().resolve()

    env_path = os.getenv(STUDY1_CONFIG_ENV_VAR)
    if env_path:
        return Path(env_path).expanduser().resolve()

    return (Path(__file__).parent / "study1_config.yaml").resolve()


def load_study1_config(config_path: Optional[str | Path] = None) -> dict[str, Any]:
    resolved_path = _resolve_study1_config_path(config_path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Study 1 config file not found: {resolved_path}")

    with open(resolved_path, "r", encoding="utf-8") as handle:
        parsed = yaml.safe_load(handle) or {}

    if not isinstance(parsed, dict):
        raise ValueError(f"Study 1 config must be a YAML mapping: {resolved_path}")

    resolved = resolve_config_paths(parsed, resolved_path)
    _validate_temporal_negative_controls(resolved)
    return resolved


def _validate_temporal_negative_controls(config: dict[str, Any]) -> None:
    study1_config = config.get("study1")
    if not isinstance(study1_config, dict):
        return
    if "temporal_negative_controls" not in study1_config:
        raise ValueError("study1.temporal_negative_controls must be configured.")
    temporal_config = study1_config["temporal_negative_controls"]
    if not isinstance(temporal_config, dict):
        raise ValueError("study1.temporal_negative_controls must be a mapping.")

    transform = str(temporal_config.get("feature_transform", "")).strip()
    if transform != TEMPORAL_NEGATIVE_CONTROL_TRANSFORM:
        raise ValueError(
            "study1.temporal_negative_controls.feature_transform must be "
            f"{TEMPORAL_NEGATIVE_CONTROL_TRANSFORM!r} so temporal negative controls use "
            "raw, unbaselined log-power features."
        )
    if temporal_config.get("feature_baseline_window") is not None:
        raise ValueError(
            "study1.temporal_negative_controls.feature_baseline_window must be null; "
            "temporal negative controls cannot reuse the active-window TFR baseline."
        )

    windows = temporal_config.get("windows")
    if not isinstance(windows, dict) or not windows:
        raise ValueError("study1.temporal_negative_controls.windows must be a non-empty mapping.")
    for name, window in windows.items():
        _validate_prestimulus_window(
            window,
            field_name=f"study1.temporal_negative_controls.windows.{name}",
        )


def _validate_prestimulus_window(value: Any, *, field_name: str) -> None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{field_name} must be a two-value [start, end] window.")
    try:
        start = float(value[0])
        end = float(value[1])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must contain numeric seconds.") from exc
    if not math.isfinite(start) or not math.isfinite(end):
        raise ValueError(f"{field_name} must contain finite seconds.")
    if start >= end:
        raise ValueError(f"{field_name} must have start < end.")
    if end > 0.0:
        raise ValueError(f"{field_name} must be a pre-stimulus window ending at or before 0 s.")


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


def apply_study1_config_defaults(
    config: dict[str, Any],
    config_path: Optional[str | Path] = None,
) -> None:
    defaults = load_study1_config(config_path=config_path)
    _merge_non_null(config, defaults)


__all__ = [
    "STUDY1_CONFIG_ENV_VAR",
    "TEMPORAL_NEGATIVE_CONTROL_TRANSFORM",
    "apply_study1_config_defaults",
    "load_study1_config",
]
