"""Strict validation helpers for Study 2 configuration and metric inputs."""

from __future__ import annotations

from math import isfinite
from typing import Any, Mapping

from eeg_pipeline.utils.config.loader import get_config_value


_MISSING = object()


def require_config_bool(config: Any, key: str) -> bool:
    value = get_config_value(config, key, _MISSING)
    if value is _MISSING:
        raise ValueError(f"Study 2 config is missing required value: {key}.")
    if not isinstance(value, bool):
        raise TypeError(f"Study 2 config value must be boolean: {key}.")
    return value


def require_config_float(config: Any, key: str) -> float:
    value = get_config_value(config, key, _MISSING)
    if value is _MISSING:
        raise ValueError(f"Study 2 config is missing required value: {key}.")
    return finite_number(value, key)


def require_config_int(config: Any, key: str) -> int:
    value = get_config_value(config, key, _MISSING)
    if value is _MISSING:
        raise ValueError(f"Study 2 config is missing required value: {key}.")
    if isinstance(value, bool):
        raise TypeError(f"Study 2 config value must be an integer: {key}.")
    number = finite_number(value, key)
    if not number.is_integer():
        raise ValueError(f"Study 2 config value must be an integer: {key}.")
    return int(number)


def require_config_probability(config: Any, key: str) -> float:
    value = require_config_float(config, key)
    if value <= 0.0 or value > 1.0:
        raise ValueError(f"Study 2 config value must be in (0, 1]: {key}.")
    return value


def require_config_string(config: Any, key: str) -> str:
    value = get_config_value(config, key, _MISSING)
    if value is _MISSING:
        raise ValueError(f"Study 2 config is missing required value: {key}.")
    text = str(value).strip() if value is not None else ""
    if not text:
        raise ValueError(f"Study 2 config value must be a non-empty string: {key}.")
    return text


def require_config_value(config: Any, key: str) -> Any:
    value = get_config_value(config, key, _MISSING)
    if value is _MISSING:
        raise ValueError(f"Study 2 config is missing required value: {key}.")
    return value


def require_bool_metric(metrics: Mapping[str, object], key: str) -> bool:
    if key not in metrics:
        raise ValueError(f"Study 2 metrics missing required metric: {key}.")

    value = metrics[key]
    if not isinstance(value, bool):
        raise TypeError(f"Study 2 metric must be boolean: {key}.")
    return value


def require_finite_metric(metrics: Mapping[str, object], key: str) -> float:
    if key not in metrics:
        raise ValueError(f"Study 2 metrics missing required metric: {key}.")
    return finite_number(metrics[key], key)


def require_probability_metric(metrics: Mapping[str, object], key: str) -> float:
    value = require_finite_metric(metrics, key)
    if value < 0.0 or value > 1.0:
        raise ValueError(f"Study 2 metric must be in [0, 1]: {key}.")
    return value


def finite_number(value: object, key: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"Study 2 numeric value must not be boolean: {key}.")

    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"Study 2 numeric value must be finite: {key}.") from exc

    if not isfinite(number):
        raise ValueError(f"Study 2 numeric value must be finite: {key}.")
    return number


__all__ = [
    "finite_number",
    "require_bool_metric",
    "require_config_bool",
    "require_config_float",
    "require_config_int",
    "require_config_probability",
    "require_config_string",
    "require_config_value",
    "require_finite_metric",
    "require_probability_metric",
]
