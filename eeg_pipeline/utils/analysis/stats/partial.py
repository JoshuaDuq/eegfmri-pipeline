"""
Partial Correlation
===================

Partial correlation and covariate-adjusted analysis.

WARNING: Predictor Control Limitations
---------------------------------------
Linear partial correlation assumes a linear relationship between predictor and outcome.
Many experimental paradigms produce nonlinear dose-response curves (e.g., sigmoid,
threshold effects). For more valid predictor control, consider:

1. Using ``predictor_residual`` to compute subject-specific residuals (spline-based).
2. Using spline terms for the predictor in regression models.
3. Stratifying analysis by predictor level rather than linear adjustment.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import lstsq

from .base import get_statistics_constants, get_config_value
from .correlation import compute_correlation
from .splines import build_predictor_rcs_design


# Constants
_ILL_CONDITIONED_THRESHOLD = 1e10
_DEFAULT_COLLINEARITY_THRESHOLD = 0.9
_MIN_SAMPLES_CORRELATION = 3


def _get_predictor_control_mode(config: Optional[Any]) -> str:
    """Resolve predictor control mode for correlation statistics."""
    if config is None:
        return "spline"
    mode = str(
        get_config_value(
            config,
            "behavior_analysis.statistics.predictor_control",
            get_config_value(config, "behavior_analysis.regression.predictor_control", "spline"),
        )
    ).strip().lower()
    if mode in {"none", "spline", "linear"}:
        return mode
    raise ValueError(
        "Invalid behavior_analysis.statistics.predictor_control value: "
        f"{mode!r}. Expected one of: 'none', 'linear', 'spline'."
    )


def _build_predictor_covariates(
    predictor_series: pd.Series,
    *,
    config: Optional[Any],
) -> pd.DataFrame:
    """Build predictor covariates (linear or restricted cubic spline) for control."""
    mode = _get_predictor_control_mode(config)
    if mode == "none":
        return pd.DataFrame(index=predictor_series.index)
    if mode == "linear":
        return pd.DataFrame({"predictor": predictor_series})

    from .splines import build_predictor_rcs_design

    df_cols, covariate_names, _meta = build_predictor_rcs_design(
        predictor_series,
        config=config,
        key_prefix="behavior_analysis.regression.predictor_spline",
        name_prefix="predictor_rcs",
    )

    rename_map = {"predictor": "predictor"}
    for name in covariate_names:
        if name.startswith("predictor_rcs_"):
            rename_map[name] = name
    return df_cols.rename(columns=rename_map)


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
        column_std = np.nanstd(design_matrix, axis=0)
        variable_mask = np.isfinite(column_std) & (column_std > 0.0)
        variable_design = design_matrix[:, variable_mask]
        if variable_design.shape[1] <= 1:
            return False, 0.0

        correlation_matrix = np.corrcoef(variable_design, rowvar=False)
        off_diagonal_mask = ~np.eye(correlation_matrix.shape[0], dtype=bool)
        off_diagonal_values = np.abs(correlation_matrix[off_diagonal_mask])
        finite_off_diagonal = off_diagonal_values[np.isfinite(off_diagonal_values)]
        if finite_off_diagonal.size == 0:
            return False, 0.0

        max_correlation = float(np.max(finite_off_diagonal))
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
    if has_collinearity and logger:
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
    logger: Optional[logging.Logger] = None,
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
    design = _build_design_matrix(df, covariate_columns, method, logger)
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


def partial_corr_xy_given_Z(
    x: pd.Series,
    y: pd.Series,
    Z: pd.DataFrame,
    method: str,
    logger: Optional[logging.Logger] = None,
    config: Optional[Any] = None,
) -> Tuple[float, float, int]:
    """
    Partial correlation of x,y controlling for Z.
    
    For Spearman partial correlation, rank-transforms all variables before
    residualization (the standard approach). For Pearson, uses raw values.
    
    Returns (r_partial, p_value, n).
    """
    df = pd.concat([x.rename("x"), y.rename("y"), Z], axis=1).dropna()
    min_required = Z.shape[1] + _MIN_SAMPLES_CORRELATION
    if len(df) < min_required:
        return np.nan, np.nan, 0

    x_vals = df["x"].values
    y_vals = df["y"].values
    z_vals = df[Z.columns].values
    
    if method.lower() == "spearman":
        x_vals = stats.rankdata(x_vals)
        y_vals = stats.rankdata(y_vals)
        z_vals = np.column_stack([stats.rankdata(z_vals[:, i]) for i in range(z_vals.shape[1])])
    
    design_dataframe = pd.DataFrame(z_vals, columns=list(Z.columns), index=df.index)
    design_matrix = _build_design_matrix(
        design_dataframe,
        list(Z.columns),
        method,
        logger=logger,
    )
    if design_matrix is None:
        return np.nan, np.nan, 0

    try:
        beta_x, *_ = lstsq(design_matrix, x_vals)
        beta_y, *_ = lstsq(design_matrix, y_vals)
    except (np.linalg.LinAlgError, ValueError):
        return np.nan, np.nan, 0

    residuals_x = x_vals - design_matrix @ beta_x
    residuals_y = y_vals - design_matrix @ beta_y

    residual_std_x = float(np.nanstd(residuals_x))
    residual_std_y = float(np.nanstd(residuals_y))
    if residual_std_x <= 1e-12 or residual_std_y <= 1e-12:
        if np.allclose(residuals_x, residuals_y, atol=1e-12, rtol=1e-6):
            return 1.0, np.nan, int(len(df))
        if np.allclose(residuals_x, -residuals_y, atol=1e-12, rtol=1e-6):
            return -1.0, np.nan, int(len(df))
        return np.nan, np.nan, int(len(df))

    # Partial correlation: correlation of residuals is valid, but the p-value must
    # use dof = n - k - 2 (k covariates), not the default dof = n - 2.
    corr_method = "pearson" if method.lower() == "spearman" else method
    r, _p_unused = compute_correlation(residuals_x, residuals_y, corr_method)
    r = float(r)

    n = int(len(df))
    k = int(Z.shape[1])
    dof = n - k - 2
    if not np.isfinite(r) or dof <= 0 or abs(r) >= 1.0:
        return (np.nan if not np.isfinite(r) else r), np.nan, n

    denom = max(1e-12, 1.0 - (r * r))
    t_stat = r * np.sqrt(float(dof) / denom)
    p = float(2.0 * stats.t.sf(np.abs(t_stat), df=float(dof)))
    return r, p, n


def compute_partial_corr(
    x: pd.Series,
    y: pd.Series,
    Z: Optional[pd.DataFrame],
    method: str,
) -> Tuple[float, float, int]:
    """
    Compute partial correlation, handling edge cases.

    If Z is None or empty, returns simple correlation.

    Parameters
    ----------
    x, y : pd.Series
        Input series
    Z : Optional[pd.DataFrame]
        Covariates to control for. If None or empty, computes simple correlation.
    method : str
        Correlation method ('pearson' or 'spearman')

    Returns
    -------
    Tuple[float, float, int]
        (r, p_value, n_samples)
    """
    if Z is None or Z.empty:
        valid = np.isfinite(x.values) & np.isfinite(y.values)
        if np.sum(valid) < 3:
            return np.nan, np.nan, 0
        r, p = compute_correlation(x.values[valid], y.values[valid], method)
        return r, p, int(np.sum(valid))

    return partial_corr_xy_given_Z(x, y, Z, method)


def partial_residuals_xy_given_Z(
    x: pd.Series,
    y: pd.Series,
    Z: pd.DataFrame,
    method: str,
    config: Optional[Any] = None,
    logger: Optional[logging.Logger] = None,
) -> Tuple[pd.Series, pd.Series, int]:
    """Compute partial residuals of x and y given Z."""
    df = pd.concat([x.rename("x"), y.rename("y"), Z], axis=1).dropna()
    result = _compute_partial_residuals(df, method, config, logger)
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
        clean_data["x"], clean_data["y"], clean_data[Z.columns], method, config, logger
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
    
    return partial_corr_xy_given_Z(
        aligned_data["x"],
        aligned_data["y"],
        aligned_data[covariates_df.columns],
        method,
        logger=logger,
        config=config,
    )


def compute_partial_correlations(
    roi_values: pd.Series,
    target_values: pd.Series,
    covariates_df: Optional[pd.DataFrame],
    predictor_series: Optional[pd.Series],
    method: str,
    context: str,
    logger: Optional[logging.Logger] = None,
    min_samples: Optional[int] = None,
    config: Optional[Any] = None,
) -> Tuple[float, float, int, float, float, int]:
    """Compute partial correlations with covariates and predictor control."""
    r_partial = p_partial = np.nan
    n_partial = 0
    r_temp = p_temp = np.nan
    n_temp = 0

    if covariates_df is not None and not covariates_df.empty:
        r_partial, p_partial, n_partial = compute_partial_correlation_with_covariates(
            roi_values, target_values, covariates_df, method, f"{context} partial", logger, min_samples, config
        )

    if predictor_series is not None and not predictor_series.empty:
        predictor_cov = _build_predictor_covariates(predictor_series, config=config)
        r_temp, p_temp, n_temp = compute_partial_correlation_with_covariates(
            roi_values, target_values, predictor_cov, method, f"{context} outcome|predictor", logger, min_samples, config
        )

    return r_partial, p_partial, n_partial, r_temp, p_temp, n_temp


def compute_partial_correlations_with_cov_predictor(
    roi_values: pd.Series,
    target_values: pd.Series,
    covariates_df: Optional[pd.DataFrame],
    predictor_series: Optional[pd.Series],
    method: str,
    context: str,
    logger: Optional[logging.Logger] = None,
    min_samples: Optional[int] = None,
    config: Optional[Any] = None,
) -> Tuple[float, float, int, float, float, int, float, float, int]:
    """Compute partial correlations for covariates-only, predictor-only, and covariates+predictor."""
    r_cov, p_cov, n_cov, r_temp, p_temp, n_temp = compute_partial_correlations(
        roi_values=roi_values,
        target_values=target_values,
        covariates_df=covariates_df,
        predictor_series=predictor_series,
        method=method,
        context=context,
        logger=logger,
        min_samples=min_samples,
        config=config,
    )

    r_cov_predictor = p_cov_predictor = np.nan
    n_cov_predictor = 0

    has_covariates = covariates_df is not None and not covariates_df.empty
    has_predictor = predictor_series is not None and not predictor_series.empty

    if has_covariates and has_predictor:
        covariates_with_predictor = covariates_df.copy()
        predictor_cov = _build_predictor_covariates(predictor_series, config=config)
        overlap = [c for c in predictor_cov.columns if c in covariates_with_predictor.columns]
        if overlap:
            covariates_with_predictor = covariates_with_predictor.drop(columns=overlap, errors="ignore")
        covariates_with_predictor_control = pd.concat([covariates_with_predictor, predictor_cov], axis=1)
        try:
            r_cov_predictor, p_cov_predictor, n_cov_predictor = compute_partial_correlation_with_covariates(
                roi_values,
                target_values,
                covariates_with_predictor_control,
                method,
                f"{context} outcome|cov+predictor",
                logger,
                min_samples,
                config,
            )
        except (ValueError, np.linalg.LinAlgError, KeyError):
            r_cov_predictor, p_cov_predictor, n_cov_predictor = np.nan, np.nan, 0

    return (
        r_cov,
        p_cov,
        n_cov,
        r_temp,
        p_temp,
        n_temp,
        r_cov_predictor,
        p_cov_predictor,
        n_cov_predictor,
    )
