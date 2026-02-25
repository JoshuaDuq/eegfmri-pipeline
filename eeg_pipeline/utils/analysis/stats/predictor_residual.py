"""
Predictor Residual (Subject-Level)
===================================

Defines a subject-level predictor residual:

    predictor_residual = outcome - f(predictor)

where f(·) is a flexible dose-response curve fitted to outcome ~ predictor.
This targets "response beyond stimulus intensity" for downstream feature
associations. Only valid when the predictor is continuous and ordinal.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .base import get_config_value as _get_config_value
from .validation import assert_continuous_predictor


def _prepare_data(
    predictor: pd.Series,
    outcome: pd.Series,
) -> Tuple[pd.Series, pd.Series, pd.Index]:
    """Align and validate predictor and outcome series."""
    predictor_values = pd.to_numeric(predictor, errors="coerce")
    outcome_values = pd.to_numeric(outcome, errors="coerce")
    common_index = predictor_values.index.intersection(outcome_values.index)
    predictor_aligned = predictor_values.loc[common_index]
    outcome_aligned = outcome_values.loc[common_index]
    return predictor_aligned, outcome_aligned, common_index


def _compute_residuals(
    outcome_values: pd.Series,
    predicted_values: pd.Series,
) -> pd.Series:
    """Compute residuals as outcome - predicted."""
    return outcome_values - predicted_values


def _fit_spline_model(
    predictor_values: pd.Series,
    outcome_values: pd.Series,
    config: Optional[Any],
) -> Optional[Tuple[pd.Series, Dict[str, Any]]]:
    """Fit spline model using statsmodels if available."""
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        return None

    degrees_of_freedom_candidates = _get_config_value(
        config, "behavior_analysis.predictor_residual.spline_df_candidates", [3, 4, 5]
    )

    model_data = pd.DataFrame({
        "pred": predictor_values.to_numpy(dtype=float),
        "outcome": outcome_values.to_numpy(dtype=float),
    })

    best_model = None
    best_metadata = None
    best_aic = np.inf

    for candidate_df in degrees_of_freedom_candidates:
        try:
            degrees_of_freedom = int(candidate_df)
        except (ValueError, TypeError):
            continue

        if degrees_of_freedom < 3:
            continue

        formula = f"outcome ~ bs(pred, df={degrees_of_freedom}, degree=3)"
        try:
            model = smf.ols(formula, data=model_data).fit()
        except (ValueError, TypeError, AttributeError):
            continue

        if not np.isfinite(model.aic):
            continue

        if model.aic < best_aic:
            best_model = model
            best_aic = model.aic
            best_metadata = {
                "df": degrees_of_freedom,
                "aic": float(model.aic),
                "bic": float(model.bic),
                "formula": formula,
            }

    if best_model is None:
        return None

    try:
        prediction_data = pd.DataFrame({
            "pred": predictor_values.to_numpy(dtype=float),
        })
        predicted_values = best_model.predict(prediction_data)
        predicted_series = pd.Series(
            np.asarray(predicted_values, dtype=float),
            index=predictor_values.index,
            dtype=float,
        )
    except (ValueError, AttributeError, TypeError):
        return None

    metadata = {
        "model": "spline",
        "status": "ok",
    }
    if best_metadata:
        metadata.update(best_metadata)
    try:
        metadata["r2"] = float(best_model.rsquared)
    except (AttributeError, ValueError, TypeError):
        pass

    return predicted_series, metadata


def _fit_polynomial_model(
    predictor_values: pd.Series,
    outcome_values: pd.Series,
    config: Optional[Any],
) -> Tuple[pd.Series, Dict[str, Any]]:
    """Fit polynomial model as fallback."""
    degree = int(_get_config_value(config, "behavior_analysis.predictor_residual.poly_degree", 2))
    degree = max(1, min(degree, 5))

    metadata = {
        "model": "poly",
        "status": "failed",
    }

    try:
        coefficients = np.polyfit(
            predictor_values.to_numpy(dtype=float),
            outcome_values.to_numpy(dtype=float),
            deg=degree,
        )
        polynomial = np.poly1d(coefficients)
        predicted_values = polynomial(predictor_values.to_numpy(dtype=float))
        predicted_series = pd.Series(
            predicted_values,
            index=predictor_values.index,
            dtype=float,
        )
        metadata["status"] = "ok"
        metadata["poly_degree"] = int(degree)
    except (ValueError, TypeError, np.linalg.LinAlgError) as error:
        metadata["error"] = str(error)

    return predicted_series, metadata


def fit_predictor_outcome_curve(
    predictor: pd.Series,
    outcome: pd.Series,
    *,
    config: Optional[Any] = None,
) -> Tuple[pd.Series, pd.Series, Dict[str, Any]]:
    """Fit outcome ~ f(predictor) and return (predicted, residual, metadata).

    Uses a spline model when statsmodels is available; otherwise falls back
    to a low-order polynomial. Requires a continuous predictor with sufficient
    unique values for stable curve fitting.

    Raises
    ------
    ValueError
        If predictor_type is not 'continuous' or has < 5 unique values.
    """
    assert_continuous_predictor(predictor, config, context="predictor_residual")
    predictor_aligned, outcome_aligned, common_index = _prepare_data(predictor, outcome)

    metadata: Dict[str, Any] = {
        "model": None,
        "status": "init",
        "n_total": int(len(common_index)),
    }

    valid_mask = predictor_aligned.notna() & outcome_aligned.notna()
    n_valid = int(valid_mask.sum())
    metadata["n_valid"] = n_valid

    min_samples = int(_get_config_value(config, "behavior_analysis.predictor_residual.min_samples", 10))
    if n_valid < min_samples:
        metadata["status"] = "skipped_insufficient_samples"
        predicted = pd.Series(np.nan, index=common_index, dtype=float)
        residual = pd.Series(np.nan, index=common_index, dtype=float)
        return predicted, residual, metadata

    predictor_valid = predictor_aligned[valid_mask]
    outcome_valid = outcome_aligned[valid_mask]

    method = str(_get_config_value(config, "behavior_analysis.predictor_residual.method", "spline")).lower()

    predicted_full = pd.Series(np.nan, index=common_index, dtype=float)
    residual_full = pd.Series(np.nan, index=common_index, dtype=float)

    if method == "spline":
        spline_result = _fit_spline_model(predictor_valid, outcome_valid, config)
        if spline_result is not None:
            predicted_valid, spline_metadata = spline_result
            predicted_full.loc[predictor_valid.index] = predicted_valid
            residual_valid = _compute_residuals(outcome_valid, predicted_valid)
            residual_full.loc[outcome_valid.index] = residual_valid
            metadata.update(spline_metadata)
            return predicted_full, residual_full, metadata

    predicted_valid, polynomial_metadata = _fit_polynomial_model(
        predictor_valid, outcome_valid, config
    )
    predicted_full.loc[predictor_valid.index] = predicted_valid
    residual_valid = _compute_residuals(outcome_valid, predicted_valid)
    residual_full.loc[outcome_valid.index] = residual_valid
    metadata.update(polynomial_metadata)

    return predicted_full, residual_full, metadata


__all__ = ["fit_predictor_outcome_curve"]
