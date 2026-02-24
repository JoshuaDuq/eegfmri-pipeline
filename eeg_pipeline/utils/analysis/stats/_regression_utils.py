"""
Internal Regression Utilities
==============================

Shared internal utilities for OLS regression, HC3 standard errors, and R² computation.
These are private helpers used by feature_models, trialwise_regression, moderation, and mediation modules.
"""

from __future__ import annotations

from typing import Optional, Tuple, List, Union, Dict, Any

import numpy as np
import pandas as pd


# Numerical stability threshold
_NUMERICAL_TOLERANCE = 1e-12


def _ols_fit(X: np.ndarray, y: np.ndarray) -> Optional[np.ndarray]:
    """Fit OLS regression coefficients.
    
    Parameters
    ----------
    X : np.ndarray
        Design matrix (n_samples, n_features)
    y : np.ndarray
        Target vector (n_samples,)
        
    Returns
    -------
    Optional[np.ndarray]
        Coefficient vector or None if fit fails
    """
    try:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None
    return beta


def _ols_regression(
    y: np.ndarray,
    X: np.ndarray,
    compute_r2: bool = False,
) -> Tuple[np.ndarray, np.ndarray, float, Optional[float]]:
    """Complete OLS regression with standard errors and optional R².
    
    Consolidated implementation used by moderation and mediation modules.
    
    Parameters
    ----------
    y : np.ndarray
        Dependent variable (n_samples,)
    X : np.ndarray
        Design matrix including intercept (n_samples, n_features)
    compute_r2 : bool
        Whether to compute R² (default: False)
        
    Returns
    -------
    beta : np.ndarray
        Coefficients (n_features,)
    se : np.ndarray
        Standard errors (n_features,)
    sigma_squared : float
        Residual variance
    r_squared : Optional[float]
        R² value if compute_r2=True, otherwise None
    """
    n, p = X.shape
    
    try:
        XtX_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        nan_array = np.full(p, np.nan)
        r_squared = np.nan if compute_r2 else None
        return nan_array, nan_array, np.nan, r_squared
    
    beta = XtX_inv @ X.T @ y
    residuals = y - X @ beta
    
    df = n - p
    if df <= 0:
        r_squared = np.nan if compute_r2 else None
        return beta, np.full(p, np.nan), np.nan, r_squared
    
    sigma_squared = np.sum(residuals**2) / df
    var_beta = sigma_squared * np.diag(XtX_inv)
    se = np.sqrt(var_beta)
    
    if compute_r2:
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
    else:
        r_squared = None
    
    return beta, se, sigma_squared, r_squared


def _hc3_se(X: np.ndarray, y: np.ndarray, beta: np.ndarray, min_denominator: float = _NUMERICAL_TOLERANCE) -> np.ndarray:
    """Compute HC3 heteroscedasticity-consistent standard errors.
    
    Parameters
    ----------
    X : np.ndarray
        Design matrix (n_samples, n_features)
    y : np.ndarray
        Target vector (n_samples,)
    beta : np.ndarray
        OLS coefficient vector (n_features,)
    min_denominator : float
        Minimum threshold for denominator to avoid division by zero
        
    Returns
    -------
    np.ndarray
        Standard errors for each coefficient
    """
    n, p = X.shape
    residuals = y - X @ beta
    try:
        XtX_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        return np.full(p, np.nan)
    
    # Leverage: h_i = x_i^T (X'X)^-1 x_i
    leverage = np.einsum("ij,jk,ik->i", X, XtX_inv, X)
    leverage_complement = 1.0 - leverage
    is_valid = np.isfinite(leverage_complement) & (np.abs(leverage_complement) > min_denominator)
    leverage_complement = np.where(is_valid, leverage_complement, np.nan)
    
    # HC3 weights: w_i = (r_i^2) / (1 - h_i)^2
    weights = (residuals**2) / (leverage_complement**2)
    weights = np.where(np.isfinite(weights), weights, 0.0)
    
    # HC3 covariance matrix
    middle_matrix = X.T @ (X * weights[:, None])
    covariance = XtX_inv @ middle_matrix @ XtX_inv
    return np.sqrt(np.diag(covariance))


def _r2(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Calculate R-squared coefficient of determination.
    
    Parameters
    ----------
    y : np.ndarray
        Observed values
    y_hat : np.ndarray
        Predicted values
        
    Returns
    -------
    float
        R-squared value (NaN if total sum of squares is zero)
    """
    y = np.asarray(y, dtype=float)
    y_hat = np.asarray(y_hat, dtype=float)
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot <= 0:
        return np.nan
    return 1.0 - ss_res / ss_tot


def _build_covariate_design(
    df: pd.DataFrame,
    covariate_cols: List[str],
    *,
    add_intercept: bool = True,
    max_dummies: int = 20,
    return_design_df: bool = False,
) -> Union[Tuple[np.ndarray, List[str]], Tuple[np.ndarray, List[str], pd.DataFrame]]:
    """Build numeric covariate design matrix with categorical dummies.
    
    Consolidated implementation for building design matrices from covariate columns.
    Handles categorical variables by creating dummy variables, and optionally adds
    an intercept term.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with covariate columns
    covariate_cols : List[str]
        List of column names to include as covariates
    add_intercept : bool
        Whether to add an intercept column (default: True)
    max_dummies : int
        Maximum number of levels for categorical variables to create dummies (default: 20)
    return_design_df : bool
        If True, also return the design DataFrame (default: False)
        
    Returns
    -------
    Union[Tuple[np.ndarray, List[str]], Tuple[np.ndarray, List[str], pd.DataFrame]]
        Design matrix, column names, and optionally the design DataFrame
    """
    use = df[covariate_cols].copy() if covariate_cols else pd.DataFrame(index=df.index)
    parts = []
    names: List[str] = []
    
    if add_intercept:
        parts.append(pd.Series(1.0, index=df.index, name="intercept"))
        names.append("intercept")
    
    for col in covariate_cols:
        s = use[col]
        is_categorical = (
            pd.api.types.is_categorical_dtype(s) or 
            pd.api.types.is_object_dtype(s)
        )
        
        if is_categorical:
            n_levels = int(s.nunique(dropna=True))
            if n_levels <= 1 or n_levels > max_dummies:
                continue
            dummies = pd.get_dummies(s.astype("category"), prefix=str(col), drop_first=True)
            for c in dummies.columns:
                parts.append(pd.to_numeric(dummies[c], errors="coerce").fillna(0.0))
                names.append(str(c))
        else:
            parts.append(pd.to_numeric(s, errors="coerce"))
            names.append(str(col))
    
    if parts:
        design_df = pd.concat(parts, axis=1)
    else:
        design_df = pd.DataFrame(index=df.index)
    
    X = design_df.to_numpy(dtype=float)
    
    if return_design_df:
        return X, names, design_df
    return X, names


def _build_predictor_covariates(
    trial_df: pd.DataFrame,
    outcome: str,
    predictor_control: str,
    include_predictor: bool,
    config: Optional[Any] = None,
    *,
    predictor_col: str = "predictor",
    key_prefix: str = "behavior_analysis.regression.predictor_spline",
    exclude_outcomes: Tuple[str, ...] = ("predictor_residual",),
) -> Tuple[List[str], Optional[pd.DataFrame], Dict[str, Any]]:
    """Build predictor-related covariates based on control strategy.

    Used by trialwise_regression, feature_models, and influence modules.

    Parameters
    ----------
    trial_df : pd.DataFrame
        Trial-level dataframe with predictor and outcome columns.
    outcome : str
        Name of the outcome variable column.
    predictor_control : str
        Control strategy: "linear", "spline"/"rcs"/"restricted_cubic", "outcome_hat"/"nonlinear".
    include_predictor : bool
        Whether to include predictor control.
    config : Optional[Any]
        Configuration object for spline parameters.
    predictor_col : str
        Name of the predictor column in trial_df.
    key_prefix : str
        Config key prefix for spline configuration.
    exclude_outcomes : Tuple[str, ...]
        Outcomes that should not use outcome_hat fallback.

    Returns
    -------
    Tuple[List[str], Optional[pd.DataFrame], Dict[str, Any]]
        Covariate column names, optional spline design DataFrame, metadata dictionary.
    """
    covariates: List[str] = []
    design_df: Optional[pd.DataFrame] = None
    meta: Dict[str, Any] = {"predictor_control_requested": predictor_control}

    if not include_predictor or outcome == predictor_col:
        return covariates, design_df, meta

    ctrl = str(predictor_control or "linear").strip().lower()

    # Rating hat / nonlinear control
    if ctrl in ("outcome_hat", "outcome_hat_from_predictor", "nonlinear"):
        if outcome not in exclude_outcomes and "outcome_hat_from_predictor" in trial_df.columns:
            covariates.append("outcome_hat_from_predictor")
            meta.update({"predictor_control_used": "outcome_hat", "predictor_control_column": "outcome_hat_from_predictor"})
        elif predictor_col in trial_df.columns:
            covariates.append(predictor_col)
            meta.update({"predictor_control_used": "linear", "predictor_control_fallback": predictor_col})

    # Spline / RCS control
    elif ctrl in ("spline", "rcs", "restricted_cubic"):
        if predictor_col in trial_df.columns:
            from eeg_pipeline.utils.analysis.stats.splines import build_predictor_rcs_design

            design_df, spline_cols, spline_meta = build_predictor_rcs_design(
                trial_df[predictor_col],
                config=config,
                key_prefix=key_prefix,
                name_prefix="predictor_rcs",
            )
            for col in spline_cols:
                if col not in covariates:
                    covariates.append(col)
            meta.update({
                "predictor_control_used": (
                    "spline" if spline_meta.get("status") in ("ok", "ok_linear_only") else "linear"
                ),
                "predictor_control_column": predictor_col,
                "predictor_spline": spline_meta,
            })
        elif outcome not in exclude_outcomes and "outcome_hat_from_predictor" in trial_df.columns:
            covariates.append("outcome_hat_from_predictor")
            meta.update({"predictor_control_used": "outcome_hat_fallback", "predictor_control_column": "outcome_hat_from_predictor"})

    # Linear control (default)
    elif predictor_col in trial_df.columns:
        covariates.append(predictor_col)
        meta.update({"predictor_control_used": "linear", "predictor_control_column": predictor_col})

    return covariates, design_df, meta
