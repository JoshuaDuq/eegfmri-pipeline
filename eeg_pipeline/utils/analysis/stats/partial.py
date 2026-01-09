"""
Partial Correlation
===================

Partial correlation and covariate-adjusted analysis.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .base import get_statistics_constants


# Constants
_ILL_CONDITIONED_THRESHOLD = 1e10
_DEFAULT_COLLINEARITY_THRESHOLD = 0.9


def check_collinearity(
    design_matrix: np.ndarray,
    threshold: float = _DEFAULT_COLLINEARITY_THRESHOLD,
) -> Tuple[bool, float]:
    """Check design matrix for collinearity.
    
    Returns (has_collinearity, max_off_diagonal_correlation).
    """
    n_columns = design_matrix.shape[1]
    if n_columns <= 1:
        return False, 0.0
    
    try:
        correlation_matrix = np.corrcoef(design_matrix.T)
        off_diagonal_mask = ~np.eye(correlation_matrix.shape[0], dtype=bool)
        max_correlation = float(np.max(np.abs(correlation_matrix[off_diagonal_mask])))
        has_collinearity = max_correlation > threshold
        return has_collinearity, max_correlation
    except (ValueError, np.linalg.LinAlgError):
        return False, 0.0


def _build_design_matrix(
    df: pd.DataFrame,
    covariate_columns: List[str],
    method: str,
    logger: Optional[logging.Logger] = None,
    collinearity_threshold: float = _DEFAULT_COLLINEARITY_THRESHOLD,
) -> Optional[np.ndarray]:
    """Build design matrix for partial correlation with collinearity check."""
    n_samples = len(df)
    intercept = np.ones(n_samples)
    
    if method == "spearman":
        ranked_covariates = [stats.rankdata(df[col].to_numpy()) for col in covariate_columns]
        design = (
            np.column_stack([intercept] + ranked_covariates)
            if ranked_covariates
            else intercept.reshape(-1, 1)
        )
    else:
        covariate_data = (
            df[covariate_columns].to_numpy()
            if covariate_columns
            else np.empty((n_samples, 0))
        )
        design = np.column_stack([intercept, covariate_data])
    
    n_predictors = design.shape[1]
    design_rank = np.linalg.matrix_rank(design)
    is_rank_deficient = n_predictors > n_samples or design_rank < n_predictors
    
    if is_rank_deficient:
        if logger:
            logger.warning("Design matrix is rank-deficient, partial correlation skipped")
        return None
    
    has_collinearity, max_correlation = check_collinearity(design, collinearity_threshold)
    if has_collinearity:
        if logger:
            logger.warning(
                f"High collinearity detected in covariates (max r={max_correlation:.3f}), "
                "partial correlation estimates may be unstable"
            )
    
    try:
        condition_number = np.linalg.cond(design)
        is_ill_conditioned = condition_number > _ILL_CONDITIONED_THRESHOLD
        if is_ill_conditioned:
            if logger:
                logger.warning(
                    f"Design matrix ill-conditioned (cond={condition_number:.1e})"
                )
            return None
    except (ValueError, np.linalg.LinAlgError):
        pass
    
    return design


def _compute_partial_residuals(
    df: pd.DataFrame,
    method: str,
    config: Optional[Any] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray, int]]:
    """Compute residuals after regressing out covariates."""
    constants = get_statistics_constants(config)
    min_samples = constants.get("min_samples_for_correlation", 5)
    
    n_samples = len(df)
    y_unique_count = df["y"].nunique()
    has_insufficient_samples = n_samples < min_samples
    has_constant_y = y_unique_count <= 1
    
    if has_insufficient_samples or has_constant_y:
        return None
    
    covariate_columns = [col for col in df.columns if col not in ("x", "y")]
    design = _build_design_matrix(df, covariate_columns, method)
    if design is None:
        return None
    
    if method == "spearman":
        x_data = stats.rankdata(df["x"].to_numpy())
        y_data = stats.rankdata(df["y"].to_numpy())
    else:
        x_data = df["x"].to_numpy()
        y_data = df["y"].to_numpy()
    
    x_coefficients = np.linalg.lstsq(design, x_data, rcond=None)[0]
    y_coefficients = np.linalg.lstsq(design, y_data, rcond=None)[0]
    
    x_residuals = x_data - design @ x_coefficients
    y_residuals = y_data - design @ y_coefficients
    
    return x_residuals, y_residuals, n_samples


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
    clean_data = data.dropna()
    
    n_dropped = len(data) - len(clean_data)
    if logger and n_dropped > 0:
        context_prefix = f"{context}: " if context else ""
        logger.warning(f"{context_prefix}dropped {n_dropped} rows")
    
    constants = get_statistics_constants(config)
    min_samples = constants.get("min_samples_for_correlation", 5)
    
    n_clean = len(clean_data)
    y_unique_count = clean_data["y"].nunique()
    has_insufficient_samples = n_clean < min_samples
    has_constant_y = y_unique_count <= 1
    
    if has_insufficient_samples or has_constant_y:
        return pd.Series(dtype=float), pd.Series(dtype=float), 0
    
    return partial_residuals_xy_given_Z(
        clean_data["x"], clean_data["y"], clean_data[Z.columns], method, config
    )


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
    if min_samples is None:
        constants = get_statistics_constants(config)
        min_samples = constants.get("min_samples_for_correlation", 5)
    
    aligned_data = pd.concat(
        [roi_values.rename("x"), target_values.rename("y"), covariates_df], axis=1
    ).dropna()
    
    n_samples = len(aligned_data)
    if n_samples < min_samples:
        return np.nan, np.nan, 0
    
    # Local import to avoid circular dependency
    from .correlation import partial_corr_xy_given_Z
    return partial_corr_xy_given_Z(
        aligned_data["x"],
        aligned_data["y"],
        aligned_data[covariates_df.columns],
        method,
        config,
    )


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


def compute_partial_correlations_with_cov_temp(
    roi_values: pd.Series,
    target_values: pd.Series,
    covariates_df: Optional[pd.DataFrame],
    temperature_series: Optional[pd.Series],
    method: str,
    context: str,
    logger: Optional[logging.Logger] = None,
    min_samples: Optional[int] = None,
    config: Optional[Any] = None,
) -> Tuple[float, float, int, float, float, int, float, float, int]:
    """Compute partial correlations for covariates-only, temp-only, and covariates+temp."""
    r_cov, p_cov, n_cov, r_temp, p_temp, n_temp = compute_partial_correlations(
        roi_values=roi_values,
        target_values=target_values,
        covariates_df=covariates_df,
        temperature_series=temperature_series,
        method=method,
        context=context,
        logger=logger,
        min_samples=min_samples,
        config=config,
    )

    r_cov_temp = p_cov_temp = np.nan
    n_cov_temp = 0

    has_covariates = covariates_df is not None and not covariates_df.empty
    has_temperature = temperature_series is not None and not temperature_series.empty
    
    if has_covariates and has_temperature:
        covariates_with_temp = covariates_df.copy()
        covariates_with_temp["temp"] = temperature_series
        try:
            r_cov_temp, p_cov_temp, n_cov_temp = compute_partial_correlation_with_covariates(
                roi_values,
                target_values,
                covariates_with_temp,
                method,
                f"{context} rating|cov+temp",
                logger,
                min_samples,
                config,
            )
        except (ValueError, np.linalg.LinAlgError, KeyError):
            r_cov_temp, p_cov_temp, n_cov_temp = np.nan, np.nan, 0

    return (
        r_cov,
        p_cov,
        n_cov,
        r_temp,
        p_temp,
        n_temp,
        r_cov_temp,
        p_cov_temp,
        n_cov_temp,
    )


def compute_partial_correlation_for_roi_pair(
    x_masked: pd.Series,
    y_masked: pd.Series,
    covariates_df: Optional[pd.DataFrame],
    mask: pd.Series,
    method: str,
) -> Tuple[float, float, int]:
    """Compute partial correlation for ROI pair."""
    if covariates_df is None or covariates_df.empty:
        return np.nan, np.nan, 0
    
    is_series_mask = isinstance(mask, pd.Series)
    masked_covariates = covariates_df.iloc[mask] if is_series_mask else covariates_df[mask]
    
    if masked_covariates.empty:
        return np.nan, np.nan, 0
    
    # Local import to avoid circular dependency
    from .correlation import partial_corr_xy_given_Z
    return partial_corr_xy_given_Z(x_masked, y_masked, masked_covariates, method)


def prepare_aligned_data(
    x: pd.Series,
    y: pd.Series,
    Z: Optional[pd.DataFrame] = None,
) -> Tuple[pd.Series, pd.Series, Optional[pd.DataFrame], int, int]:
    """Align x, y, and covariates, removing NaN rows."""
    x_series = x if isinstance(x, pd.Series) else pd.Series(x)
    y_series = y if isinstance(y, pd.Series) else pd.Series(y)
    
    frames = [x_series.rename("__x__"), y_series.rename("__y__")]
    has_covariates = False
    
    if Z is not None:
        if isinstance(Z, pd.DataFrame):
            has_data = len(Z) > 0 and len(Z.columns) > 0
            if has_data:
                frames.append(Z)
                has_covariates = True
        else:
            try:
                Z_dataframe = pd.DataFrame(Z)
                has_data = len(Z_dataframe) > 0 and len(Z_dataframe.columns) > 0
                if has_data:
                    frames.append(Z_dataframe)
                    has_covariates = True
            except (ValueError, TypeError):
                pass

    combined_data = pd.concat(frames, axis=1)
    n_total = len(combined_data)
    clean_data = combined_data.dropna()
    n_kept = len(clean_data)

    if n_kept == 0:
        return pd.Series(dtype=float), pd.Series(dtype=float), None, n_total, n_kept

    x_name = x_series.name if x_series.name is not None else "x"
    y_name = y_series.name if y_series.name is not None else "y"
    
    x_clean = clean_data.pop("__x__").rename(x_name)
    y_clean = clean_data.pop("__y__").rename(y_name)
    Z_clean = clean_data if has_covariates else None

    return x_clean, y_clean, Z_clean, n_total, n_kept


