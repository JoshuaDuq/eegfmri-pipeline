"""Temporal-control definitions for Study 1."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from eeg_pipeline.utils.config.loader import get_config_value

TEMPORAL_CONTROL_BANDS = ("alpha", "beta", "gamma")
TEMPORAL_CONTROL_FEATURE_PREFIX = "temporal_"
TEMPORAL_CONTROL_FEATURE_STATS = ("log10raw",)
TEMPORAL_CONTROL_PARTITION = "temporal_control"
TEMPORAL_CONTROL_TRANSFORM = "raw_log_power"


@dataclass(frozen=True)
class TemporalControlWindow:
    name: str
    kind: str
    start: float
    end: float

    @property
    def feature_spec(self) -> str:
        return f"{TEMPORAL_CONTROL_FEATURE_PREFIX}{self.name}"

    def as_time_range(self) -> dict[str, float | str]:
        return {"name": self.name, "tmin": self.start, "tmax": self.end}


def resolve_temporal_control_windows(config: Any) -> tuple[TemporalControlWindow, ...]:
    temporal_config = get_config_value(config, "study1.temporal_negative_controls", None)
    if temporal_config is None:
        return tuple()
    if not isinstance(temporal_config, dict):
        raise ValueError("study1.temporal_negative_controls must be a mapping.")

    transform = str(temporal_config.get("feature_transform", "")).strip()
    if transform != TEMPORAL_CONTROL_TRANSFORM:
        raise ValueError(
            "study1.temporal_negative_controls.feature_transform must be "
            f"{TEMPORAL_CONTROL_TRANSFORM!r}."
        )
    if (
        "feature_baseline_window" not in temporal_config
        or temporal_config.get("feature_baseline_window") is not None
    ):
        raise ValueError(
            "study1.temporal_negative_controls.feature_baseline_window must be null."
        )

    windows: list[TemporalControlWindow] = []
    seen_names: set[str] = set()
    for kind, key in (
        ("prestimulus", "windows"),
        ("wrong_lag", "wrong_lag_windows"),
    ):
        raw_windows = temporal_config.get(key, {})
        if raw_windows is None:
            raw_windows = {}
        if not isinstance(raw_windows, dict):
            raise ValueError(f"study1.temporal_negative_controls.{key} must be a mapping.")
        for raw_name, raw_window in raw_windows.items():
            name = str(raw_name).strip()
            if not name:
                raise ValueError(f"study1.temporal_negative_controls.{key} contains an empty name.")
            if name in seen_names:
                raise ValueError(f"Duplicate Study 1 temporal-control window name: {name!r}.")
            start, end = _time_window(raw_window, field_name=f"{key}.{name}")
            if kind == "prestimulus" and end > 0.0:
                raise ValueError(f"Pre-stimulus temporal-control window {name!r} must end at 0 s.")
            windows.append(TemporalControlWindow(name=name, kind=kind, start=start, end=end))
            seen_names.add(name)
    return tuple(windows)


def temporal_control_time_ranges(config: Any) -> list[dict[str, float | str]]:
    return [window.as_time_range() for window in resolve_temporal_control_windows(config)]


def temporal_control_feature_spec(window_name: str) -> str:
    name = str(window_name).strip()
    if not name:
        raise ValueError("Temporal-control window names must be non-empty.")
    return f"{TEMPORAL_CONTROL_FEATURE_PREFIX}{name}"


def temporal_control_window_for_feature_spec(
    config: Any,
    feature_spec: str,
) -> TemporalControlWindow | None:
    spec = str(feature_spec).strip()
    if not spec.startswith(TEMPORAL_CONTROL_FEATURE_PREFIX):
        return None
    window_name = spec.removeprefix(TEMPORAL_CONTROL_FEATURE_PREFIX)
    for window in resolve_temporal_control_windows(config):
        if window.name == window_name:
            return window
    return None


def _time_window(value: Any, *, field_name: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"Temporal-control window {field_name} must be [start, end].")
    try:
        start = float(value[0])
        end = float(value[1])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Temporal-control window {field_name} must contain seconds.") from exc
    if not math.isfinite(start) or not math.isfinite(end):
        raise ValueError(f"Temporal-control window {field_name} must contain finite seconds.")
    if start >= end:
        raise ValueError(f"Temporal-control window {field_name} must satisfy start < end.")
    return start, end


__all__ = [
    "TEMPORAL_CONTROL_BANDS",
    "TEMPORAL_CONTROL_FEATURE_STATS",
    "TEMPORAL_CONTROL_PARTITION",
    "TemporalControlWindow",
    "resolve_temporal_control_windows",
    "temporal_control_feature_spec",
    "temporal_control_time_ranges",
    "temporal_control_window_for_feature_spec",
]
