"""
Partial Correlation
===================

Partial correlation and covariate-adjusted analysis.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .base import get_statistics_constants
from .correlation import compute_correlation


def _build_design_matrix(
    df: pd.DataFrame,
    covariate_columns: List[str],
    method: str,
) -> Optional[np.ndarray]:
    """Build design matrix for partial correlation."""
    intercept = np.ones(len(df))
    
    if method == "spearman":
        ranked = [stats.rankdata(df[c].to_numpy()) for c in covariate_columns]
        design = np.column_stack([intercept] + ranked) if ranked else intercept.reshape(-1, 1)
    else:
        cov_data = df[covariate_columns].to_numpy() if covariate_columns else np.empty((len(df), 0))
        design = np.column_stack([intercept, cov_data])
    
    if design.shape[1] > len(df) or np.linalg.matrix_rank(design) < design.shape[1]:
        return None
    
    return design


def _compute_partial_residuals(
    df: pd.DataFrame,
    method: str,
    config: Optional[Any] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray, int]]:
    """Compute residuals after regressing out covariates."""
    constants = get_statistics_constants(config)
    min_samples = constants.get("min_samples_for_correlation", 5)
    
    if len(df) < min_samples or df["y"].nunique() <= 1:
        return None
    
    cov_cols = [c for c in df.columns if c not in ("x", "y")]
    design = _build_design_matrix(df, cov_cols, method)
    if design is None:
        return None
    
    if method == "spearman":
        x_data = stats.rankdata(df["x"].to_numpy())
        y_data = stats.rankdata(df["y"].to_numpy())
    else:
        x_data = df["x"].to_numpy()
        y_data = df["y"].to_numpy()
    
    x_coef = np.linalg.lstsq(design, x_data, rcond=None)[0]
    y_coef = np.linalg.lstsq(design, y_data, rcond=None)[0]
    
    return x_data - design @ x_coef, y_data - design @ y_coef, len(df)


def partial_residuals_xy_given_Z(
    x: pd.Series,
    y: pd.Series,
    Z: pd.DataFrame,
    method: str,
    config: Optional[Any] = None,
) -> Tuple[pd.Series, pd.Series, int]:
    """Compute partial residuals of x and y given Z."""
    df = pd.concat([x.rename("x"), y.rename("y"), Z], axis=1).dropna()
    result = _compute_partial_residuals(df, method, config)
    if result is None:
        return pd.Series(dtype=float), pd.Series(dtype=float), 0
    
    x_res, y_res, n = result
    return pd.Series(x_res, index=df.index), pd.Series(y_res, index=df.index), n


def compute_partial_residuals(
    x: pd.Series,
    y: pd.Series,
    Z: Optional[pd.DataFrame],
    method: str,
    *,
    logger: Optional[logging.Logger] = None,
    context: str = "",
    config: Optional[Any] = None,
) -> Tuple[pd.Series, pd.Series, int]:
    """Compute partial residuals with logging."""
    if Z is None or Z.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float), 0
    
    data = pd.concat([x.rename("x"), y.rename("y"), Z], axis=1)
    clean = data.dropna()
    
    if logger and len(data) > len(clean):
        logger.warning(f"{context + ': ' if context else ''}dropped {len(data) - len(clean)} rows")
    
    constants = get_statistics_constants(config)
    min_samples = constants.get("min_samples_for_correlation", 5)
    
    if len(clean) < min_samples or clean["y"].nunique() <= 1:
        return pd.Series(dtype=float), pd.Series(dtype=float), 0
    
    return partial_residuals_xy_given_Z(clean["x"], clean["y"], clean[Z.columns], method, config)


def compute_partial_correlation_with_covariates(
    roi_values: pd.Series,
    target_values: pd.Series,
    covariates_df: pd.DataFrame,
    method: str,
    context: str,
    logger: Optional[logging.Logger] = None,
    min_samples: Optional[int] = None,
    config: Optional[Any] = None,
) -> Tuple[float, float, int]:
    """Compute partial correlation controlling for covariates."""
    from .correlation import partial_corr_xy_given_Z
    
    if min_samples is None:
        constants = get_statistics_constants(config)
        min_samples = constants.get("min_samples_for_correlation", 5)
    
    # Align data
    combined = pd.concat([roi_values.rename("x"), target_values.rename("y"), covariates_df], axis=1).dropna()
    
    if len(combined) < min_samples:
        return np.nan, np.nan, 0
    
    return partial_corr_xy_given_Z(combined["x"], combined["y"], combined[covariates_df.columns], method, config)


def compute_partial_correlations(
    roi_values: pd.Series,
    target_values: pd.Series,
    covariates_df: Optional[pd.DataFrame],
    temperature_series: Optional[pd.Series],
    method: str,
    context: str,
    logger: Optional[logging.Logger] = None,
    min_samples: Optional[int] = None,
    config: Optional[Any] = None,
) -> Tuple[float, float, int, float, float, int]:
    """Compute partial correlations with covariates and temperature."""
    r_partial = p_partial = np.nan
    n_partial = 0
    r_temp = p_temp = np.nan
    n_temp = 0
    
    if covariates_df is not None and not covariates_df.empty:
        r_partial, p_partial, n_partial = compute_partial_correlation_with_covariates(
            roi_values, target_values, covariates_df, method, f"{context} partial", logger, min_samples, config
        )
    
    if temperature_series is not None and not temperature_series.empty:
        temp_cov = pd.DataFrame({"temp": temperature_series})
        r_temp, p_temp, n_temp = compute_partial_correlation_with_covariates(
            roi_values, target_values, temp_cov, method, f"{context} rating|temp", logger, min_samples, config
        )
    
    return r_partial, p_partial, n_partial, r_temp, p_temp, n_temp


def compute_partial_correlation_for_roi_pair(
    x_masked: pd.Series,
    y_masked: pd.Series,
    covariates_df: Optional[pd.DataFrame],
    mask: pd.Series,
    method: str,
) -> Tuple[float, float, int]:
    """Compute partial correlation for ROI pair."""
    from .correlation import partial_corr_xy_given_Z
    
    if covariates_df is None or covariates_df.empty:
        return np.nan, np.nan, 0
    
    cov_valid = covariates_df.iloc[mask] if hasattr(mask, '__iter__') else covariates_df[mask]
    if cov_valid.empty:
        return np.nan, np.nan, 0
    
    return partial_corr_xy_given_Z(x_masked, y_masked, cov_valid, method)


def prepare_aligned_data(
    x: pd.Series,
    y: pd.Series,
    Z: Optional[pd.DataFrame] = None,
) -> Tuple[pd.Series, pd.Series, Optional[pd.DataFrame], int, int]:
    """Align x, y, and covariates, removing NaN rows."""
    x_series = pd.Series(x) if not isinstance(x, pd.Series) else x
    y_series = pd.Series(y) if not isinstance(y, pd.Series) else y
    
    frames = [x_series.rename("__x__"), y_series.rename("__y__")]
    has_z = False
    if Z is not None:
        if isinstance(Z, pd.DataFrame):
            if len(Z) > 0 and len(Z.columns) > 0:
                frames.append(Z)
                has_z = True
        else:
            try:
                Z_df = pd.DataFrame(Z)
                if len(Z_df) > 0 and len(Z_df.columns) > 0:
                    frames.append(Z_df)
                    has_z = True
            except (ValueError, TypeError):
                pass

    data = pd.concat(frames, axis=1)
    n_total = len(data)
    data_clean = data.dropna()
    n_kept = len(data_clean)

    if n_kept == 0:
        return pd.Series(dtype=float), pd.Series(dtype=float), None, n_total, n_kept

    x_clean = data_clean.pop("__x__").rename(x_series.name if hasattr(x_series, 'name') else "x")
    y_clean = data_clean.pop("__y__").rename(y_series.name if hasattr(y_series, 'name') else "y")
    Z_clean = data_clean if has_z else None

    return x_clean, y_clean, Z_clean, n_total, n_kept


