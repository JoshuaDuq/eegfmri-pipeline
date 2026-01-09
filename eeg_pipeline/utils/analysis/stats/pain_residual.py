"""
Pain Residual (Subject-Level)
=============================

Defines a subject-level pain residual:

    pain_residual = rating - f(temperature)

where f(·) is a flexible (but stable) dose-response curve. This targets
"pain beyond stimulus intensity" for downstream feature associations.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd


def _get_config_value(config: Any, key: str, default: Any) -> Any:
    """Extract configuration value with safe fallback."""
    if config is None:
        return default
    if hasattr(config, "get"):
        try:
            return config.get(key, default)
        except (AttributeError, KeyError, TypeError):
            return default
    return default


def _prepare_data(
    temperature: pd.Series,
    rating: pd.Series,
) -> Tuple[pd.Series, pd.Series, pd.Index]:
    """Align and validate temperature and rating series."""
    temperature_values = pd.to_numeric(temperature, errors="coerce")
    rating_values = pd.to_numeric(rating, errors="coerce")
    common_index = temperature_values.index.intersection(rating_values.index)
    temperature_aligned = temperature_values.loc[common_index]
    rating_aligned = rating_values.loc[common_index]
    return temperature_aligned, rating_aligned, common_index


def _compute_residuals(
    rating_values: pd.Series,
    predicted_values: pd.Series,
) -> pd.Series:
    """Compute residuals as rating - predicted."""
    return rating_values.to_numpy(dtype=float) - predicted_values.to_numpy(dtype=float)


def _fit_spline_model(
    temperature_values: pd.Series,
    rating_values: pd.Series,
    config: Optional[Any],
) -> Optional[Tuple[pd.Series, Dict[str, Any]]]:
    """Fit spline model using statsmodels if available."""
    try:
        import statsmodels.formula.api as smf
        import pandas as pd_statsmodels
    except ImportError:
        return None

    degrees_of_freedom_candidates = _get_config_value(
        config, "behavior_analysis.pain_residual.spline_df_candidates", [3, 4, 5]
    )
    if not isinstance(degrees_of_freedom_candidates, (list, tuple)) or not degrees_of_freedom_candidates:
        degrees_of_freedom_candidates = [3, 4, 5]

    model_data = pd_statsmodels.DataFrame({
        "temp": temperature_values.to_numpy(dtype=float),
        "rating": rating_values.to_numpy(dtype=float),
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

        formula = f"rating ~ bs(temp, df={degrees_of_freedom}, degree=3)"
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
        prediction_data = pd_statsmodels.DataFrame({
            "temp": temperature_values.to_numpy(dtype=float),
        })
        predicted_values = best_model.predict(prediction_data)
        predicted_series = pd.Series(
            np.asarray(predicted_values, dtype=float),
            index=temperature_values.index,
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
    temperature_values: pd.Series,
    rating_values: pd.Series,
    config: Optional[Any],
) -> Tuple[pd.Series, Dict[str, Any]]:
    """Fit polynomial model as fallback."""
    degree = int(_get_config_value(config, "behavior_analysis.pain_residual.poly_degree", 2))
    degree = max(1, min(degree, 5))

    metadata = {
        "model": "poly",
        "status": "failed",
    }

    try:
        coefficients = np.polyfit(
            temperature_values.to_numpy(dtype=float),
            rating_values.to_numpy(dtype=float),
            deg=degree,
        )
        polynomial = np.poly1d(coefficients)
        predicted_values = polynomial(temperature_values.to_numpy(dtype=float))
        predicted_series = pd.Series(
            predicted_values,
            index=temperature_values.index,
            dtype=float,
        )
        metadata["status"] = "ok"
        metadata["poly_degree"] = int(degree)
    except (ValueError, TypeError, np.linalg.LinAlgError) as error:
        metadata["error"] = str(error)

    return predicted_series, metadata


def fit_temperature_rating_curve(
    temperature: pd.Series,
    rating: pd.Series,
    *,
    config: Optional[Any] = None,
) -> Tuple[pd.Series, pd.Series, Dict[str, Any]]:
    """
    Fit rating ~ f(temperature) and return (predicted, residual, metadata).

    Uses a spline model when statsmodels is available; otherwise falls back
    to a low-order polynomial.
    """
    temperature_aligned, rating_aligned, common_index = _prepare_data(temperature, rating)

    metadata: Dict[str, Any] = {
        "model": None,
        "status": "init",
        "n_total": int(len(common_index)),
    }

    valid_mask = temperature_aligned.notna() & rating_aligned.notna()
    n_valid = int(valid_mask.sum())
    metadata["n_valid"] = n_valid

    min_samples = int(_get_config_value(config, "behavior_analysis.pain_residual.min_samples", 10))
    if n_valid < min_samples:
        metadata["status"] = "skipped_insufficient_samples"
        predicted = pd.Series(np.nan, index=common_index, dtype=float)
        residual = pd.Series(np.nan, index=common_index, dtype=float)
        return predicted, residual, metadata

    temperature_valid = temperature_aligned[valid_mask]
    rating_valid = rating_aligned[valid_mask]

    method = str(_get_config_value(config, "behavior_analysis.pain_residual.method", "spline")).strip().lower()

    predicted_full = pd.Series(np.nan, index=common_index, dtype=float)
    residual_full = pd.Series(np.nan, index=common_index, dtype=float)

    if method == "spline":
        spline_result = _fit_spline_model(temperature_valid, rating_valid, config)
        if spline_result is not None:
            predicted_valid, spline_metadata = spline_result
            predicted_full.loc[temperature_valid.index] = predicted_valid
            residual_valid = _compute_residuals(rating_valid, predicted_valid)
            residual_full.loc[rating_valid.index] = residual_valid
            metadata.update(spline_metadata)
            return predicted_full, residual_full, metadata

    predicted_valid, polynomial_metadata = _fit_polynomial_model(
        temperature_valid, rating_valid, config
    )
    predicted_full.loc[temperature_valid.index] = predicted_valid
    residual_valid = _compute_residuals(rating_valid, predicted_valid)
    residual_full.loc[rating_valid.index] = residual_valid
    metadata.update(polynomial_metadata)

    return predicted_full, residual_full, metadata


__all__ = ["fit_temperature_rating_curve"]

