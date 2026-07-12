"""Small statistical helpers shared by Study 2 analysis-core modules."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from studies.pain_study.study2.validation import finite_number


def as_finite_1d(values: object, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"Study 2 {name} must be 1D, got shape {array.shape}.")
    if array.size == 0:
        raise ValueError(f"Study 2 {name} must not be empty.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"Study 2 {name} contains non-finite values.")
    return array


def pearson_r(x: object, y: object, *, name: str) -> float:
    x_arr = as_finite_1d(x, name=f"{name} x")
    y_arr = as_finite_1d(y, name=f"{name} y")
    if x_arr.shape != y_arr.shape:
        raise ValueError(f"Study 2 {name} vectors must have the same shape.")

    x_centered = x_arr - float(np.mean(x_arr))
    y_centered = y_arr - float(np.mean(y_arr))
    denominator = float(np.linalg.norm(x_centered) * np.linalg.norm(y_centered))
    if denominator <= 0.0:
        raise ValueError(f"Study 2 {name} correlation requires nonzero variance.")
    return float(np.clip((x_centered @ y_centered) / denominator, -1.0, 1.0))


def plus_one_p_value(observed: float, null_values: object) -> float:
    observed_value = float(observed)
    null_arr = as_finite_1d(null_values, name="null distribution")
    exceedances = int(np.sum(null_arr >= observed_value))
    return float((exceedances + 1) / (null_arr.size + 1))


def holm_adjusted_p_values(p_values: Mapping[str, object]) -> dict[str, float]:
    if not p_values:
        return {}

    parsed: list[tuple[str, float]] = []
    for name, value in p_values.items():
        p_value = finite_number(value, f"p-value {name}")
        if p_value < 0.0 or p_value > 1.0:
            raise ValueError(f"Study 2 p-value must be in [0, 1]: {name}.")
        parsed.append((str(name), p_value))

    n_tests = len(parsed)
    running_max = 0.0
    adjusted: dict[str, float] = {}
    for rank, (name, p_value) in enumerate(sorted(parsed, key=lambda item: item[1])):
        running_max = max(running_max, (n_tests - rank) * p_value)
        adjusted[name] = float(min(running_max, 1.0))
    return adjusted


def residualize_vector(values: object, design: object, *, name: str) -> np.ndarray:
    value_arr = as_finite_1d(values, name=name)
    design_arr = np.asarray(design, dtype=float)
    if design_arr.ndim != 2:
        raise ValueError(f"Study 2 {name} design must be 2D, got shape {design_arr.shape}.")
    if design_arr.shape[0] != value_arr.size:
        raise ValueError(f"Study 2 {name} design rows must match values.")
    if not np.all(np.isfinite(design_arr)):
        raise ValueError(f"Study 2 {name} design contains non-finite values.")

    predictors = np.column_stack([np.ones(value_arr.size, dtype=float), design_arr])
    if np.linalg.matrix_rank(predictors) < predictors.shape[1]:
        raise ValueError(f"Study 2 {name} design must have full column rank.")
    coefficients, *_ = np.linalg.lstsq(predictors, value_arr, rcond=None)
    return value_arr - predictors @ coefficients


def standardized_residual(values: object, design: object, *, name: str) -> np.ndarray:
    residual = residualize_vector(values, design, name=name)
    residual_std = float(np.std(residual, ddof=0))
    if residual_std <= 1.0e-12:
        raise ValueError(f"Study 2 {name} has zero residual variance.")
    return (residual - float(np.mean(residual))) / residual_std


__all__ = [
    "as_finite_1d",
    "holm_adjusted_p_values",
    "pearson_r",
    "plus_one_p_value",
    "residualize_vector",
    "standardized_residual",
]
