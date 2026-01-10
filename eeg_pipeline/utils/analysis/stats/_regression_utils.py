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
        Whether to compute R² (default: False for backward compatibility)
        
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
        return nan_array, nan_array, np.nan, None if not compute_r2 else np.nan
    
    beta = XtX_inv @ X.T @ y
    residuals = y - X @ beta
    
    df = n - p
    if df <= 0:
        return beta, np.full(p, np.nan), np.nan, None if not compute_r2 else np.nan
    
    sigma_squared = np.sum(residuals**2) / df
    var_beta = sigma_squared * np.diag(XtX_inv)
    se = np.sqrt(var_beta)
    
    r_squared = None
    if compute_r2:
        y_mean = np.mean(y)
        ss_tot = np.sum((y - y_mean) ** 2)
        if ss_tot > 0:
            r_squared = 1.0 - (np.sum(residuals**2) / ss_tot)
        else:
            r_squared = np.nan
    
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
        is_categorical = s.dtype == object or str(s.dtype).startswith("category")
        
        if is_categorical:
            n_levels = int(pd.Series(s).nunique(dropna=True))
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


def _build_temperature_covariates(
    trial_df: pd.DataFrame,
    outcome: str,
    temperature_control: str,
    include_temperature: bool,
    config: Optional[Any] = None,
    *,
    key_prefix: str = "behavior_analysis.regression.temperature_spline",
    exclude_outcomes: Optional[Tuple[str, ...]] = None,
) -> Tuple[List[str], Optional[pd.DataFrame], Dict[str, Any]]:
    """Build temperature-related covariates based on control strategy.
    
    Consolidated implementation used by trialwise_regression, feature_models, and influence modules.
    
    Parameters
    ----------
    trial_df : pd.DataFrame
        Trial-level dataframe with temperature and rating columns
    outcome : str
        Name of the outcome variable (e.g., "rating", "temperature", "pain_residual")
    temperature_control : str
        Control strategy: "linear", "spline"/"rcs"/"restricted_cubic", "rating_hat"/"nonlinear"
    include_temperature : bool
        Whether to include temperature control
    config : Optional[Any]
        Configuration object for spline parameters
    key_prefix : str
        Config key prefix for spline configuration (default: "behavior_analysis.regression.temperature_spline")
    exclude_outcomes : Optional[Tuple[str, ...]]
        Outcomes that should not use rating_hat fallback (default: ("pain_residual",))
        
    Returns
    -------
    Tuple[List[str], Optional[pd.DataFrame], Dict[str, Any]]
        Covariate column names, optional spline design DataFrame, metadata dictionary
    """
    covariates: List[str] = []
    temp_design_df: Optional[pd.DataFrame] = None
    meta: Dict[str, Any] = {"temperature_control_requested": temperature_control}
    
    if exclude_outcomes is None:
        exclude_outcomes = ("pain_residual",)
    
    if not include_temperature or outcome == "temperature":
        return covariates, temp_design_df, meta
    
    temp_ctrl = str(temperature_control or "linear").strip().lower()
    
    # Rating hat / nonlinear control
    if temp_ctrl in ("rating_hat", "rating_hat_from_temp", "nonlinear"):
        if outcome not in exclude_outcomes and "rating_hat_from_temp" in trial_df.columns:
            covariates.append("rating_hat_from_temp")
            meta.update({
                "temperature_control_used": "rating_hat",
                "temperature_control_column": "rating_hat_from_temp"
            })
        elif "temperature" in trial_df.columns:
            covariates.append("temperature")
            meta.update({
                "temperature_control_used": "linear",
                "temperature_control_fallback": "temperature"
            })
    
    # Spline / RCS control
    elif temp_ctrl in ("spline", "rcs", "restricted_cubic"):
        if "temperature" in trial_df.columns:
            from eeg_pipeline.utils.analysis.stats.splines import build_temperature_rcs_design
            
            temp_design_df, spline_cols, spline_meta = build_temperature_rcs_design(
                trial_df["temperature"],
                config=config,
                key_prefix=key_prefix,
                name_prefix="temperature_rcs",
            )
            for col in spline_cols:
                if col not in covariates:
                    covariates.append(col)
            meta.update({
                "temperature_control_used": (
                    "spline" if spline_meta.get("status") in ("ok", "ok_linear_only") else "linear"
                ),
                "temperature_control_column": "temperature",
                "temperature_spline": spline_meta
            })
        elif outcome not in exclude_outcomes and "rating_hat_from_temp" in trial_df.columns:
            covariates.append("rating_hat_from_temp")
            meta.update({
                "temperature_control_used": "rating_hat_fallback",
                "temperature_control_column": "rating_hat_from_temp"
            })
    
    # Linear control (default)
    elif "temperature" in trial_df.columns:
        covariates.append("temperature")
        meta.update({
            "temperature_control_used": "linear",
            "temperature_control_column": "temperature"
        })
    
    return covariates, temp_design_df, meta
