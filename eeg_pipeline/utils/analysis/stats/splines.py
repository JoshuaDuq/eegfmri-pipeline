"""
Spline Helpers (Lightweight, Subject-Level)
==========================================

Implements a minimal restricted cubic spline (RCS) basis generator for
nonlinear temperature control without depending on statsmodels/patsy.

This is intended for covariate control (not interpretability of spline terms).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .base import safe_get_config_value as _get_config_value


# Constants
MIN_KNOTS_FOR_NONLINEAR = 4
MIN_SAMPLES_DEFAULT = 12
N_KNOTS_DEFAULT = 4
QUANTILE_LOW_DEFAULT = 0.05
QUANTILE_HIGH_DEFAULT = 0.95
QUANTILE_LOW_BOUND = 0.49
QUANTILE_HIGH_BOUND = 0.51
NUMERICAL_TOLERANCE = 1e-12


def truncated_power_cube(x: np.ndarray) -> np.ndarray:
    """Compute truncated power function of degree 3: max(x, 0)^3."""
    x = np.asarray(x, dtype=float)
    return np.maximum(x, 0.0) ** 3


def _validate_temperature_data(
    temperature_values: np.ndarray,
    finite_mask: np.ndarray,
    min_samples: int,
) -> Optional[str]:
    """Validate temperature data for spline construction."""
    n_valid = int(finite_mask.sum())
    if n_valid < min_samples:
        return "skipped_insufficient_samples"

    valid_temperature_values = temperature_values[finite_mask]
    if np.nanstd(valid_temperature_values, ddof=1) <= NUMERICAL_TOLERANCE:
        return "skipped_constant_temperature"

    return None


def _compute_knot_quantiles(
    quantile_low: float,
    quantile_high: float,
    n_knots: int,
) -> Tuple[float, float]:
    """Compute and validate quantile bounds for knot placement."""
    quantile_low = min(max(quantile_low, 0.0), QUANTILE_LOW_BOUND)
    quantile_high = max(min(quantile_high, 1.0), QUANTILE_HIGH_BOUND)

    if quantile_high <= quantile_low:
        quantile_low = QUANTILE_LOW_DEFAULT
        quantile_high = QUANTILE_HIGH_DEFAULT

    return quantile_low, quantile_high


def _compute_knots(
    valid_temperature_values: np.ndarray,
    quantile_low: float,
    quantile_high: float,
    n_knots: int,
) -> np.ndarray:
    """Compute unique, sorted knots from temperature quantiles."""
    quantile_low, quantile_high = _compute_knot_quantiles(
        quantile_low, quantile_high, n_knots
    )

    quantile_values = np.linspace(
        quantile_low, quantile_high, num=max(n_knots, MIN_KNOTS_FOR_NONLINEAR)
    )
    knots = np.quantile(valid_temperature_values, quantile_values)
    knots = np.unique(knots)

    return np.sort(knots)


def _validate_knots(knots: np.ndarray) -> Optional[str]:
    """Validate knots for spline construction."""
    if knots.size < MIN_KNOTS_FOR_NONLINEAR:
        return "skipped_insufficient_unique_knots"

    second_to_last_knot = float(knots[-2])
    last_knot = float(knots[-1])
    boundary_knot_difference = last_knot - second_to_last_knot

    if not np.isfinite(boundary_knot_difference):
        return "skipped_degenerate_boundary_knots"

    if abs(boundary_knot_difference) <= NUMERICAL_TOLERANCE:
        return "skipped_degenerate_boundary_knots"

    return None


def _compute_spline_basis_term(
    temperature_all: np.ndarray,
    interior_knot: float,
    second_to_last_knot: float,
    last_knot: float,
    boundary_knot_difference: float,
) -> np.ndarray:
    """Compute a single restricted cubic spline basis term."""
    term_at_knot = truncated_power_cube(temperature_all - interior_knot)
    term_at_second_to_last = truncated_power_cube(
        temperature_all - second_to_last_knot
    )
    term_at_last = truncated_power_cube(temperature_all - last_knot)

    coefficient_second_to_last = (last_knot - interior_knot) / boundary_knot_difference
    coefficient_last = (second_to_last_knot - interior_knot) / boundary_knot_difference

    spline_term = (
        term_at_knot
        - term_at_second_to_last * coefficient_second_to_last
        + term_at_last * coefficient_last
    )

    return spline_term


def build_temperature_rcs_design(
    temperature: pd.Series,
    *,
    config: Optional[Any] = None,
    key_prefix: str = "behavior_analysis.regression.temperature_spline",
    name_prefix: str = "temperature_rcs",
) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
    """
    Build restricted cubic spline covariate columns for temperature.

    Returns
    -------
    df_cols : pd.DataFrame
        Columns to join into the design matrix. Includes a linear `temperature`
        term plus K-2 nonlinear spline columns when possible.
    covariate_names : list[str]
        Names of columns in `df_cols` in order.
    meta : dict
        Diagnostics about knots and fallbacks.
    """
    temperature_numeric = pd.to_numeric(temperature, errors="coerce")
    temperature_values = temperature_numeric.to_numpy(dtype=float)
    finite_mask = np.isfinite(temperature_values)

    min_samples = int(
        _get_config_value(config, f"{key_prefix}.min_samples", MIN_SAMPLES_DEFAULT)
    )
    n_knots = int(
        _get_config_value(config, f"{key_prefix}.n_knots", N_KNOTS_DEFAULT)
    )
    quantile_low = float(
        _get_config_value(config, f"{key_prefix}.quantile_low", QUANTILE_LOW_DEFAULT)
    )
    quantile_high = float(
        _get_config_value(
            config, f"{key_prefix}.quantile_high", QUANTILE_HIGH_DEFAULT
        )
    )

    meta: Dict[str, Any] = {
        "n_valid": int(finite_mask.sum()),
        "min_samples": min_samples,
        "n_knots": n_knots,
        "quantile_low": quantile_low,
        "quantile_high": quantile_high,
    }

    output_dataframe = pd.DataFrame(index=temperature_numeric.index)
    output_dataframe["temperature"] = temperature_numeric

    validation_error = _validate_temperature_data(
        temperature_values, finite_mask, min_samples
    )
    if validation_error:
        meta["status"] = validation_error
        return output_dataframe, ["temperature"], meta

    if n_knots < MIN_KNOTS_FOR_NONLINEAR:
        meta["status"] = "skipped_n_knots_lt_4"
        return output_dataframe, ["temperature"], meta

    valid_temperature_values = temperature_values[finite_mask]
    knots = _compute_knots(valid_temperature_values, quantile_low, quantile_high, n_knots)

    knot_validation_error = _validate_knots(knots)
    if knot_validation_error:
        meta["status"] = knot_validation_error
        if knot_validation_error == "skipped_insufficient_unique_knots":
            meta["knots_unique"] = knots.size
        return output_dataframe, ["temperature"], meta

    second_to_last_knot = float(knots[-2])
    last_knot = float(knots[-1])
    boundary_knot_difference = last_knot - second_to_last_knot

    interior_knots = knots[1:-1]
    n_spline_terms = len(interior_knots) - 1

    if n_spline_terms <= 0:
        meta["status"] = "ok_linear_only"
        meta["knots"] = knots.tolist()
        return output_dataframe, ["temperature"], meta

    covariate_names: List[str] = ["temperature"]

    for term_index, interior_knot in enumerate(interior_knots[:-1]):
        spline_term = _compute_spline_basis_term(
            temperature_values,
            float(interior_knot),
            second_to_last_knot,
            last_knot,
            boundary_knot_difference,
        )
        spline_term[~finite_mask] = np.nan

        column_name = f"{name_prefix}_{term_index + 1}"
        output_dataframe[column_name] = spline_term
        covariate_names.append(column_name)

    meta["status"] = "ok"
    meta["knots"] = knots.tolist()
    meta["n_spline_terms"] = len(covariate_names) - 1
    return output_dataframe, covariate_names, meta


__all__ = ["build_temperature_rcs_design"]

