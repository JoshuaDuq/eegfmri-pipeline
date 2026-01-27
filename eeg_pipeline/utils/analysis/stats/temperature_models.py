"""
Temperature→Rating Model Comparisons (Subject-Level)
====================================================

Pain studies often exhibit non-linear dose-response curves. This module provides
lightweight, subject-level comparisons of model families for `rating ~ f(temperature)`,
plus an optional single-breakpoint ("hinge") test.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .base import get_config_value as _get_config_value


def _calculate_rmse(observed: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate root mean squared error between observed and predicted values."""
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if observed.size == 0:
        return np.nan
    squared_errors = (observed - predicted) ** 2
    mean_squared_error = np.mean(squared_errors)
    return float(np.sqrt(mean_squared_error))


def _validate_and_prepare_data(
    temperature: pd.Series, rating: pd.Series, min_samples: int
) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    """Validate inputs and prepare aligned, clean data for modeling."""
    temperature_numeric = pd.to_numeric(temperature, errors="coerce")
    rating_numeric = pd.to_numeric(rating, errors="coerce")
    
    common_index = temperature_numeric.index.intersection(rating_numeric.index)
    temperature_aligned = temperature_numeric.loc[common_index]
    rating_aligned = rating_numeric.loc[common_index]
    
    is_valid = temperature_aligned.notna() & rating_aligned.notna()
    n_valid = int(is_valid.sum())
    
    meta = {"n_valid": n_valid, "min_samples": min_samples}
    if n_valid < min_samples:
        return None, {**meta, "status": "skipped_insufficient_samples"}
    
    data = pd.DataFrame(
        {
            "temp": temperature_aligned[is_valid].to_numpy(dtype=float),
            "rating": rating_aligned[is_valid].to_numpy(dtype=float),
        }
    )
    return data, meta


def _check_statsmodels_module(module_name: str) -> Tuple[bool, Any]:
    """Check if a statsmodels module is available."""
    try:
        if module_name == "formula":
            import statsmodels.formula.api as module
        elif module_name == "api":
            import statsmodels.api as module
        else:
            return False, None
        return True, module
    except ImportError:
        return False, None


def _extract_model_metrics(model: Any, n_samples: int, data: pd.DataFrame, predictions: np.ndarray) -> Dict[str, Any]:
    """Extract model fit metrics from a fitted statsmodels model."""
    df_model = getattr(model, "df_model", None)
    n_params = int(df_model) + 1 if df_model is not None else np.nan
    
    aic = float(model.aic) if np.isfinite(model.aic) else np.nan
    bic = float(model.bic) if np.isfinite(model.bic) else np.nan
    r_squared = float(model.rsquared) if hasattr(model, "rsquared") else np.nan
    
    observed = data["rating"].to_numpy()
    rmse = _calculate_rmse(observed, predictions) if np.isfinite(predictions).all() else np.nan
    
    return {
        "n": n_samples,
        "k_params": n_params,
        "aic": aic,
        "bic": bic,
        "r2": r_squared,
        "rmse": rmse,
    }


def _fit_single_model(smf: Any, model_name: str, formula: str, data: pd.DataFrame, n_samples: int) -> Optional[Dict[str, Any]]:
    """Fit a single model and return its metrics, or None if fitting fails."""
    try:
        model = smf.ols(formula, data=data).fit()
        predictions = np.asarray(model.predict(data), dtype=float)
    except (ValueError, AttributeError, TypeError):
        return None
    
    metrics = _extract_model_metrics(model, n_samples, data, predictions)
    return {
        "model": model_name,
        "formula": formula,
        **metrics,
    }


def _build_polynomial_terms(degree: int) -> str:
    """Build polynomial terms string for given degree."""
    if degree < 2:
        return ""
    terms = [f"I(temp**{k})" for k in range(2, degree + 1)]
    return " + ".join(terms)


def _get_polynomial_degrees(config: Any) -> List[int]:
    """Extract and validate polynomial degrees from config."""
    degrees_raw = _get_config_value(config, "behavior_analysis.pain_residual.model_comparison.poly_degrees", [2, 3])
    if not isinstance(degrees_raw, (list, tuple)) or not degrees_raw:
        return [2, 3]
    
    valid_degrees = [int(d) for d in degrees_raw if isinstance(d, (int, float)) and 2 <= int(d) <= 5]
    return valid_degrees if valid_degrees else [2, 3]


def _get_spline_df_candidates(config: Any) -> List[int]:
    """Extract and validate spline degrees of freedom from config."""
    df_candidates_raw = _get_config_value(config, "behavior_analysis.pain_residual.spline_df_candidates", [3, 4, 5])
    if not isinstance(df_candidates_raw, (list, tuple)) or not df_candidates_raw:
        return [3, 4, 5]
    
    valid_df = [int(df) for df in df_candidates_raw if isinstance(df, (int, float)) and int(df) >= 3]
    return valid_df if valid_df else [3, 4, 5]


def _fit_all_models(smf: Any, data: pd.DataFrame, config: Any) -> List[Dict[str, Any]]:
    """Fit all candidate models (linear, polynomial, spline)."""
    n_samples = len(data)
    model_results = []
    
    linear_result = _fit_single_model(smf, "linear", "rating ~ temp", data, n_samples)
    if linear_result is not None:
        model_results.append(linear_result)
    
    polynomial_degrees = _get_polynomial_degrees(config)
    for degree in polynomial_degrees:
        terms = _build_polynomial_terms(degree)
        formula = f"rating ~ temp + {terms}"
        result = _fit_single_model(smf, f"poly{degree}", formula, data, n_samples)
        if result is not None:
            model_results.append(result)
    
    spline_df_candidates = _get_spline_df_candidates(config)
    for df in spline_df_candidates:
        formula = f"rating ~ bs(temp, df={df}, degree=3)"
        result = _fit_single_model(smf, f"spline_df{df}", formula, data, n_samples)
        if result is not None:
            model_results.append(result)
    
    return model_results


def compare_temperature_rating_models(
    temperature: pd.Series,
    rating: pd.Series,
    *,
    config: Optional[Any] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Compare linear / polynomial / spline models for rating ~ f(temperature)."""
    min_samples = int(_get_config_value(config, "behavior_analysis.pain_residual.model_comparison.min_samples", 10))
    data, meta = _validate_and_prepare_data(temperature, rating, min_samples)
    if data is None:
        return pd.DataFrame(), meta
    
    has_statsmodels, smf = _check_statsmodels_module("formula")
    meta["has_statsmodels"] = has_statsmodels
    if not has_statsmodels:
        return pd.DataFrame(), {**meta, "status": "missing_statsmodels"}
    
    model_results = _fit_all_models(smf, data, config)
    if not model_results:
        return pd.DataFrame(), {**meta, "status": "empty"}
    
    results_df = pd.DataFrame(model_results).sort_values("aic", ascending=True)
    best_model = results_df.iloc[0].to_dict()
    
    best_aic = best_model.get("aic")
    meta.update(
        {
            "status": "ok",
            "best_model": best_model.get("model"),
            "best_aic": best_aic,
        }
    )
    
    if best_aic is not None and np.isfinite(best_aic):
        results_df["delta_aic"] = results_df["aic"] - float(best_aic)
    else:
        results_df["delta_aic"] = np.nan
    
    return results_df, meta


def _fit_linear_baseline(sm: Any, temperatures: np.ndarray, ratings: np.ndarray) -> Tuple[Any, Dict[str, Any]]:
    """Fit baseline linear model and return model with error info if it fails."""
    design_matrix = sm.add_constant(temperatures, has_constant="add")
    try:
        model = sm.OLS(ratings, design_matrix).fit()
        return model, {}
    except (ValueError, np.linalg.LinAlgError) as exc:
        return None, {"status": "failed_linear", "error": str(exc)}


def _get_breakpoint_search_range(temperatures: np.ndarray, config: Any) -> Tuple[Optional[float], Optional[float], Dict[str, Any]]:
    """Determine valid temperature range for breakpoint search."""
    quantile_low = float(_get_config_value(config, "behavior_analysis.pain_residual.breakpoint_test.quantile_low", 0.15))
    quantile_high = float(_get_config_value(config, "behavior_analysis.pain_residual.breakpoint_test.quantile_high", 0.85))
    
    temperature_low = float(np.quantile(temperatures, quantile_low))
    temperature_high = float(np.quantile(temperatures, quantile_high))
    
    if not np.isfinite(temperature_low) or not np.isfinite(temperature_high):
        return None, None, {"status": "skipped_invalid_temperature_range"}
    
    if temperature_high <= temperature_low:
        return None, None, {"status": "skipped_invalid_temperature_range"}
    
    return temperature_low, temperature_high, {}


def _generate_breakpoint_candidates(temperature_low: float, temperature_high: float, config: Any) -> np.ndarray:
    """Generate candidate breakpoint values for search."""
    n_candidates = max(int(_get_config_value(config, "behavior_analysis.pain_residual.breakpoint_test.n_candidates", 15)), 5)
    return np.linspace(temperature_low, temperature_high, num=n_candidates)


def _create_hinge_feature(temperatures: np.ndarray, breakpoint: float) -> np.ndarray:
    """Create hinge feature: max(0, temperature - breakpoint)."""
    return np.maximum(0.0, temperatures - breakpoint)


def _build_hinge_design_matrix(temperatures: np.ndarray, hinge: np.ndarray) -> np.ndarray:
    """Build design matrix for hinge model: [intercept, temperature, hinge]."""
    intercept = np.ones(len(temperatures))
    return np.column_stack([intercept, temperatures, hinge])


def _extract_hinge_model_metrics(model: Any, breakpoint: float) -> Dict[str, Any]:
    """Extract metrics from a fitted hinge model."""
    aic = float(model.aic) if np.isfinite(model.aic) else np.nan
    bic = float(model.bic) if np.isfinite(model.bic) else np.nan
    r_squared = float(model.rsquared) if hasattr(model, "rsquared") else np.nan
    
    params = model.params
    beta_temp = float(params[1]) if len(params) > 1 else np.nan
    beta_hinge = float(params[2]) if len(params) > 2 else np.nan
    
    pvalues = getattr(model, "pvalues", None)
    p_hinge = float(pvalues[2]) if pvalues is not None and len(pvalues) > 2 else np.nan
    
    return {
        "breakpoint_c": breakpoint,
        "aic": aic,
        "bic": bic,
        "r2": r_squared,
        "beta_temp": beta_temp,
        "beta_hinge": beta_hinge,
        "p_hinge": p_hinge,
    }


def _fit_hinge_model(sm: Any, temperatures: np.ndarray, ratings: np.ndarray, breakpoint: float) -> Optional[Dict[str, Any]]:
    """Fit a single hinge model at given breakpoint."""
    hinge = _create_hinge_feature(temperatures, breakpoint)
    design_matrix = _build_hinge_design_matrix(temperatures, hinge)
    
    try:
        model = sm.OLS(ratings, design_matrix).fit()
    except (ValueError, np.linalg.LinAlgError):
        return None
    
    return _extract_hinge_model_metrics(model, breakpoint)


def _find_best_breakpoint(sm: Any, temperatures: np.ndarray, ratings: np.ndarray, candidates: np.ndarray) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Search for best breakpoint by fitting hinge models at all candidates."""
    results = []
    best_result = None
    
    for breakpoint in candidates:
        result = _fit_hinge_model(sm, temperatures, ratings, breakpoint)
        if result is None:
            continue
        
        results.append(result)
        if best_result is None:
            best_result = result
        elif np.isfinite(result["aic"]) and result["aic"] < best_result["aic"]:
            best_result = result
    
    return results, best_result


def _calculate_f_test(linear_model: Any, hinge_model: Any, n_observations: int) -> Tuple[float, float]:
    """Calculate F-test comparing linear vs hinge model."""
    from scipy.stats import f as f_distribution
    
    residual_sum_squares_linear = float(np.sum(linear_model.resid ** 2))
    residual_sum_squares_hinge = float(np.sum(hinge_model.resid ** 2))
    
    df_linear = int(getattr(linear_model, "df_model", 1)) + 1
    df_hinge = int(getattr(hinge_model, "df_model", 2)) + 1
    
    df_numerator = max(df_hinge - df_linear, 1)
    df_denominator = max(n_observations - df_hinge, 1)
    
    if (n_observations <= df_hinge 
        or residual_sum_squares_hinge <= 0 
        or residual_sum_squares_linear < residual_sum_squares_hinge):
        return np.nan, np.nan
    
    numerator = (residual_sum_squares_linear - residual_sum_squares_hinge) / df_numerator
    denominator = residual_sum_squares_hinge / df_denominator
    
    if denominator <= 0:
        return np.nan, np.nan
    
    f_statistic = numerator / denominator
    if not np.isfinite(f_statistic):
        return np.nan, np.nan
    
    p_value = float(f_distribution.sf(f_statistic, dfn=df_numerator, dfd=df_denominator))
    return f_statistic, p_value


def fit_temperature_breakpoint_test(
    temperature: pd.Series,
    rating: pd.Series,
    *,
    config: Optional[Any] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Fit a single-breakpoint hinge model and compare to linear baseline.

    Model: rating ~ temp + max(0, temp - c)
    """
    min_samples = int(_get_config_value(config, "behavior_analysis.pain_residual.breakpoint_test.min_samples", 12))
    data, meta = _validate_and_prepare_data(temperature, rating, min_samples)
    if data is None:
        return pd.DataFrame(), meta
    
    has_statsmodels, sm = _check_statsmodels_module("api")
    meta["has_statsmodels"] = has_statsmodels
    if not has_statsmodels:
        return pd.DataFrame(), {**meta, "status": "missing_statsmodels"}
    
    temperatures = data["temp"].to_numpy(dtype=float)
    ratings = data["rating"].to_numpy(dtype=float)
    
    linear_model, error_meta = _fit_linear_baseline(sm, temperatures, ratings)
    if linear_model is None:
        return pd.DataFrame(), {**meta, **error_meta}
    
    temperature_low, temperature_high, range_error_meta = _get_breakpoint_search_range(temperatures, config)
    if temperature_low is None or temperature_high is None:
        return pd.DataFrame(), {**meta, **range_error_meta}
    
    breakpoint_candidates = _generate_breakpoint_candidates(temperature_low, temperature_high, config)
    breakpoint_results, best_result = _find_best_breakpoint(sm, temperatures, ratings, breakpoint_candidates)
    
    if not breakpoint_results or best_result is None:
        return pd.DataFrame(), {**meta, "status": "empty"}
    
    results_df = pd.DataFrame(breakpoint_results).sort_values("aic", ascending=True)
    
    best_breakpoint = float(best_result["breakpoint_c"])
    hinge_best = _create_hinge_feature(temperatures, best_breakpoint)
    design_matrix_best = _build_hinge_design_matrix(temperatures, hinge_best)
    
    try:
        hinge_model_best = sm.OLS(ratings, design_matrix_best).fit()
    except (ValueError, np.linalg.LinAlgError) as exc:
        return results_df, {**meta, "status": "ok", "best_breakpoint": best_breakpoint, "f_test_error": str(exc)}
    
    f_statistic, f_p_value = _calculate_f_test(linear_model, hinge_model_best, len(temperatures))
    
    best_aic = float(best_result["aic"])
    linear_aic = float(linear_model.aic) if np.isfinite(linear_model.aic) else np.nan
    delta_aic = float(best_aic - linear_aic) if np.isfinite(best_aic) and np.isfinite(linear_aic) else np.nan
    
    meta.update(
        {
            "status": "ok",
            "best_breakpoint": best_breakpoint,
            "best_aic": best_aic,
            "delta_aic_vs_linear": delta_aic,
            "f_test_stat": f_statistic,
            "f_test_p": f_p_value,
        }
    )
    return results_df, meta


__all__ = ["compare_temperature_rating_models", "fit_temperature_breakpoint_test"]

