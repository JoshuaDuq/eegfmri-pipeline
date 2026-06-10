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

    resolved = resolve_config_paths(copy.deepcopy(parsed), resolved_path)
    _preserve_signature_reference_paths(resolved, parsed)
    _validate_temporal_negative_controls(resolved)
    _validate_reference_power(resolved)
    _validate_feature_benchmark(resolved)
    return resolved


def _preserve_signature_reference_paths(
    resolved: dict[str, Any],
    raw: dict[str, Any],
) -> None:
    raw_targets = raw.get("study1", {}).get("targets", {})
    resolved_targets = resolved.get("study1", {}).get("targets", {})
    if not isinstance(raw_targets, dict) or not isinstance(resolved_targets, dict):
        return

    if "signature_manifest_path" in raw_targets:
        resolved_targets["signature_manifest_path"] = copy.deepcopy(
            raw_targets["signature_manifest_path"]
        )

    raw_provenance = raw_targets.get("signature_provenance")
    resolved_provenance = resolved_targets.get("signature_provenance")
    if not isinstance(raw_provenance, dict) or not isinstance(resolved_provenance, dict):
        return

    for signature_name, raw_spec in raw_provenance.items():
        resolved_spec = resolved_provenance.get(signature_name)
        if isinstance(raw_spec, dict) and isinstance(resolved_spec, dict) and "path" in raw_spec:
            resolved_spec["path"] = copy.deepcopy(raw_spec["path"])


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
    if (
        "feature_baseline_window" not in temporal_config
        or temporal_config.get("feature_baseline_window") is not None
    ):
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

    wrong_lag_windows = temporal_config.get("wrong_lag_windows")
    if not isinstance(wrong_lag_windows, dict) or not wrong_lag_windows:
        raise ValueError(
            "study1.temporal_negative_controls.wrong_lag_windows must be a non-empty mapping."
        )

    plateau_start, plateau_end = _active_plateau_window(config)
    for name, window in wrong_lag_windows.items():
        _validate_pre_plateau_window(
            window,
            field_name=f"study1.temporal_negative_controls.wrong_lag_windows.{name}",
            plateau_start=plateau_start,
        )

    plateau_windows = temporal_config.get("plateau_windows")
    if not isinstance(plateau_windows, dict) or not plateau_windows:
        raise ValueError(
            "study1.temporal_negative_controls.plateau_windows must be a non-empty mapping."
        )
    for name, window in plateau_windows.items():
        _validate_plateau_window(
            window,
            field_name=f"study1.temporal_negative_controls.plateau_windows.{name}",
            plateau_start=plateau_start,
            plateau_end=plateau_end,
        )


def _validate_reference_power(config: dict[str, Any]) -> None:
    study1_config = config.get("study1")
    if not isinstance(study1_config, dict):
        return
    if "reference_power" not in study1_config:
        raise ValueError("study1.reference_power must be configured.")

    reference_config = study1_config["reference_power"]
    if not isinstance(reference_config, dict):
        raise ValueError("study1.reference_power must be a mapping.")
    time_frequency = config.get("time_frequency_analysis")
    if not isinstance(time_frequency, dict):
        raise ValueError("time_frequency_analysis must be a mapping.")

    primary_window = _validate_prestimulus_window(
        reference_config.get("primary_window"),
        field_name="study1.reference_power.primary_window",
    )
    configured_baseline = _validate_prestimulus_window(
        time_frequency.get("baseline_window"),
        field_name="time_frequency_analysis.baseline_window",
    )
    if configured_baseline != primary_window:
        raise ValueError(
            "time_frequency_analysis.baseline_window must match "
            "study1.reference_power.primary_window."
        )

    sensitivity_windows = reference_config.get("sensitivity_windows")
    if not isinstance(sensitivity_windows, dict) or not sensitivity_windows:
        raise ValueError("study1.reference_power.sensitivity_windows must be a non-empty mapping.")
    for name, window in sensitivity_windows.items():
        _validate_prestimulus_window(
            window,
            field_name=f"study1.reference_power.sensitivity_windows.{name}",
        )

    unnormalized = reference_config.get("unnormalized_active_power")
    if not isinstance(unnormalized, dict):
        raise ValueError("study1.reference_power.unnormalized_active_power must be a mapping.")
    transform = str(unnormalized.get("feature_transform", "")).strip()
    if transform != TEMPORAL_NEGATIVE_CONTROL_TRANSFORM:
        raise ValueError(
            "study1.reference_power.unnormalized_active_power.feature_transform must be "
            f"{TEMPORAL_NEGATIVE_CONTROL_TRANSFORM!r}."
        )
    if (
        "feature_baseline_window" not in unnormalized
        or unnormalized.get("feature_baseline_window") is not None
    ):
        raise ValueError(
            "study1.reference_power.unnormalized_active_power.feature_baseline_window "
            "must be null."
        )
    active_window = _validate_time_window(
        unnormalized.get("active_window"),
        field_name="study1.reference_power.unnormalized_active_power.active_window",
    )
    configured_active = _validate_time_window(
        time_frequency.get("active_window"),
        field_name="time_frequency_analysis.active_window",
    )
    if active_window != configured_active:
        raise ValueError(
            "study1.reference_power.unnormalized_active_power.active_window must match "
            "time_frequency_analysis.active_window."
        )
    reference_window = _validate_prestimulus_window(
        unnormalized.get("reference_window"),
        field_name="study1.reference_power.unnormalized_active_power.reference_window",
    )
    if reference_window != primary_window:
        raise ValueError(
            "study1.reference_power.unnormalized_active_power.reference_window must match "
            "study1.reference_power.primary_window."
        )
    if unnormalized.get("reference_power_covariate") is not True:
        raise ValueError(
            "study1.reference_power.unnormalized_active_power.reference_power_covariate "
            "must be true."
        )


def _validate_time_window(value: Any, *, field_name: str) -> tuple[float, float]:
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
    return start, end


def _validate_prestimulus_window(value: Any, *, field_name: str) -> tuple[float, float]:
    start, end = _validate_time_window(value, field_name=field_name)
    if end > 0.0:
        raise ValueError(f"{field_name} must be a pre-stimulus window ending at or before 0 s.")
    return start, end


def _active_plateau_window(config: dict[str, Any]) -> tuple[float, float]:
    time_frequency = config.get("time_frequency_analysis")
    if not isinstance(time_frequency, dict):
        raise ValueError("time_frequency_analysis must be a mapping.")
    return _validate_time_window(
        time_frequency.get("active_window"),
        field_name="time_frequency_analysis.active_window",
    )


def _validate_pre_plateau_window(
    value: Any,
    *,
    field_name: str,
    plateau_start: float,
) -> tuple[float, float]:
    start, end = _validate_time_window(value, field_name=field_name)
    if start < 0.0 or end > plateau_start:
        raise ValueError(
            f"{field_name} must start at or after stimulus onset and end before the plateau starts."
        )
    return start, end


def _validate_plateau_window(
    value: Any,
    *,
    field_name: str,
    plateau_start: float,
    plateau_end: float,
) -> tuple[float, float]:
    start, end = _validate_time_window(value, field_name=field_name)
    if start < plateau_start or end > plateau_end:
        raise ValueError(f"{field_name} must be contained within the plateau.")
    return start, end


def _validate_feature_benchmark(config: dict[str, Any]) -> None:
    study1_config = config.get("study1")
    if not isinstance(study1_config, dict):
        return
    feature_config = study1_config.get("feature_benchmark")
    if not isinstance(feature_config, dict):
        raise ValueError("study1.feature_benchmark must be a mapping.")

    _validate_positive_int(
        feature_config.get("n_perm"),
        field_name="study1.feature_benchmark.n_perm",
    )
    permutation_scheme = str(feature_config.get("permutation_scheme", "")).strip()
    if not permutation_scheme:
        raise ValueError("study1.feature_benchmark.permutation_scheme must be configured.")

    _validate_fraction(
        feature_config.get("max_invalid_permutation_fraction"),
        field_name="study1.feature_benchmark.max_invalid_permutation_fraction",
    )
    if permutation_scheme == "circular_shift_within_run":
        circular_shift = feature_config.get("circular_shift")
        if not isinstance(circular_shift, dict):
            raise ValueError("study1.feature_benchmark.circular_shift must be a mapping.")
        for key in ("min_valid_blocks_per_subject", "min_retained_trials_per_subject"):
            _validate_positive_int(
                circular_shift.get(key),
                field_name=f"study1.feature_benchmark.circular_shift.{key}",
            )


def _validate_positive_int(value: Any, *, field_name: str) -> int:
    try:
        numeric = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a positive integer.") from exc
    if numeric <= 0:
        raise ValueError(f"{field_name} must be a positive integer.")
    return numeric


def _validate_fraction(value: Any, *, field_name: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a fraction in [0, 1).") from exc
    if not math.isfinite(numeric) or not 0.0 <= numeric < 1.0:
        raise ValueError(f"{field_name} must be a fraction in [0, 1).")
    return numeric


def _merge_non_null(base: dict[str, Any], extra: dict[str, Any]) -> None:
    for key, value in extra.items():
        if value is None:
            if key not in base:
                base[key] = None
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
