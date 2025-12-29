"""
Regression and Model Diagnostics
================================

Statistical diagnostics for linear models including:
- Multicollinearity (Variance Inflation Factor)
- Influence (Cook's Distance, Leverage)
- Distributional checks (Normality summary)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from typing import Any, Dict, Tuple


def compute_vif(X: pd.DataFrame) -> pd.Series:
    """Compute Variance Inflation Factor for each predictor.
    
    VIF = 1 / (1 - R²) where R² is from regressing each predictor on others.
    VIF > 5: Moderate multicollinearity
    VIF > 10: High multicollinearity
    
    Parameters
    ----------
    X : pd.DataFrame
        Covariate matrix (each column is a predictor)
        
    Returns
    -------
    pd.Series
        VIF for each column
    """
    if X.empty:
        return pd.Series(dtype=float)
        
    n_cols = X.shape[1]
    vif_values = {}
    
    for i, col in enumerate(X.columns):
        if n_cols == 1:
            vif_values[col] = 1.0
            continue
            
        # Regress this column on all others
        y_temp = pd.to_numeric(X[col], errors="coerce").values
        X_other_df = X.drop(columns=[col])
        X_other = X_other_df.apply(pd.to_numeric, errors="coerce").values
        
        mask = np.isfinite(y_temp) & np.all(np.isfinite(X_other), axis=1)
        if mask.sum() < X_other.shape[1] + 2:
            vif_values[col] = np.nan
            continue
            
        y_v = y_temp[mask]
        X_v = X_other[mask]
        
        # Add intercept
        X_design = np.column_stack([np.ones(len(y_v)), X_v])
        
        # Compute R²
        try:
            beta = np.linalg.lstsq(X_design, y_v, rcond=None)[0]
            y_pred = X_design @ beta
            ss_res = np.sum((y_v - y_pred)**2)
            ss_tot = np.sum((y_v - np.mean(y_v))**2)
            
            if ss_tot < 1e-10:
                r_squared = 0.0
            else:
                r_squared = 1 - (ss_res / ss_tot)
            
            r_squared = np.clip(r_squared, 0, 0.9999)
            vif = 1.0 / (1.0 - r_squared)
            vif_values[col] = float(vif)
        except (np.linalg.LinAlgError, ValueError):
            vif_values[col] = np.nan
    
    return pd.Series(vif_values)


def compute_leverage_and_cooks(
    x: np.ndarray,
    y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Compute leverage (hat values) and Cook's distance for simple regression.
    
    Parameters
    ----------
    x : np.ndarray
        Predictor variable
    y : np.ndarray
        Response variable
        
    Returns
    -------
    leverage : np.ndarray
        Hat values (diagonal of hat matrix)
    cooks_d : np.ndarray
        Cook's distance for each observation
    residuals : np.ndarray
        Studentized residuals
    cooks_threshold : float
        Threshold for influential points (4/n)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    n = int(mask.sum())
    
    if n < 4:
        return np.full(len(x), np.nan), np.full(len(x), np.nan), np.full(len(x), np.nan), np.nan
    
    x_v = x[mask]
    y_v = y[mask]
    
    # Design matrix [1, x]
    X = np.column_stack([np.ones(n), x_v])
    
    # Hat matrix diagonal
    try:
        XtX_inv = np.linalg.inv(X.T @ X)
        h_v = np.diag(X @ XtX_inv @ X.T)
    except np.linalg.LinAlgError:
        return np.full(len(x), np.nan), np.full(len(x), np.nan), np.full(len(x), np.nan), np.nan
    
    # Fitted values and residuals
    beta = XtX_inv @ X.T @ y_v
    y_hat = X @ beta
    res_v = y_v - y_hat
    
    # MSE
    p = 2  # intercept + slope
    mse = np.sum(res_v**2) / (n - p) if n > p else 0
    
    # Studentized residuals
    with np.errstate(divide='ignore', invalid='ignore'):
        student_v = res_v / np.sqrt(mse * (1 - h_v) + 1e-12)
    
    # Cook's distance
    with np.errstate(divide='ignore', invalid='ignore'):
        cooks_v = (student_v**2 / p) * (h_v / (1 - h_v + 1e-12))
    
    # Threshold: 4/n
    cooks_threshold = 4.0 / n
    
    # Map back to original indices
    leverage = np.full(len(x), np.nan)
    cooks_d = np.full(len(x), np.nan)
    residuals = np.full(len(x), np.nan)
    
    leverage[mask] = h_v
    cooks_d[mask] = cooks_v
    residuals[mask] = student_v
    
    return leverage, cooks_d, residuals, cooks_threshold


def compute_normality_summary(data: np.ndarray, alpha: float = 0.05) -> Dict[str, Any]:
    """Compute multiple normality tests and provide a summary.
    
    Returns
    -------
    dict with keys:
        shapiro_stat, shapiro_p: Shapiro-Wilk test
        dagostino_stat, dagostino_p: D'Agostino K² test
        skewness, kurtosis: Distribution moments
        is_normal: Boolean (p > alpha for both tests)
        n: sample size
    """
    from eeg_pipeline.utils.analysis.stats.validation import (
        check_normality_shapiro,
        check_normality_dagostino
    )
    
    clean = np.asarray(data)
    clean = clean[np.isfinite(clean)]
    n = len(clean)
    
    res_s = check_normality_shapiro(clean, alpha=alpha)
    res_d = check_normality_dagostino(clean, alpha=alpha)
    
    return {
        "shapiro_stat": float(res_s.statistic),
        "shapiro_p": float(res_s.p_value),
        "dagostino_stat": float(res_d.statistic),
        "dagostino_p": float(res_d.p_value),
        "skewness": float(stats.skew(clean)) if n > 8 else np.nan,
        "kurtosis": float(stats.kurtosis(clean)) if n > 8 else np.nan,
        "is_normal": res_s.passed and (res_d.passed or n < 20),
        "n": n,
    }


__all__ = [
    "compute_vif",
    "compute_leverage_and_cooks",
    "compute_normality_summary",
]
