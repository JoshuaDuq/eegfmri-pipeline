"""
Mediation Analysis
===================

Test whether EEG features mediate stimulus-behavior relationships:
- Baron & Kenny approach
- Sobel test
- Bootstrap confidence intervals for indirect effect
"""

from __future__ import annotations

from typing import Optional, List, Dict, Tuple, Any
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class MediationResult:
    """Result container for mediation analysis."""
    # Paths
    a_path: float  # X -> M
    a_se: float
    a_p: float
    b_path: float  # M -> Y (controlling for X)
    b_se: float
    b_p: float
    c_path: float  # Total effect: X -> Y
    c_se: float
    c_p: float
    c_prime: float  # Direct effect: X -> Y (controlling for M)
    c_prime_se: float
    c_prime_p: float
    
    # Indirect effect
    indirect_effect: float  # a * b
    indirect_se: float
    indirect_ci_low: float
    indirect_ci_high: float
    indirect_p: float
    
    # Summary
    proportion_mediated: float
    sobel_z: float
    sobel_p: float
    n: int
    significant: bool


def _ols_regression(y: np.ndarray, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Simple OLS regression. Returns (coefficients, standard errors, r_squared)."""
    n, k = X.shape
    
    try:
        XtX_inv = np.linalg.inv(X.T @ X)
        beta = XtX_inv @ X.T @ y
        
        residuals = y - X @ beta
        mse = np.sum(residuals ** 2) / (n - k)
        se = np.sqrt(np.diag(XtX_inv) * mse)
        
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - ss_res / (ss_tot + 1e-12)
        
        return beta, se, r_squared
    except np.linalg.LinAlgError:
        return np.full(k, np.nan), np.full(k, np.nan), np.nan


def test_mediation(
    X: np.ndarray,
    M: np.ndarray,
    Y: np.ndarray,
    covariates: np.ndarray = None,
    n_bootstrap: int = 5000,
    ci_level: float = 0.95,
) -> MediationResult:
    """Test mediation: Does M mediate the X -> Y relationship?
    
    Parameters
    ----------
    X : array
        Independent variable (e.g., stimulus intensity)
    M : array
        Mediator (e.g., alpha power)
    Y : array
        Dependent variable (e.g., pain rating)
    covariates : array, optional
        Additional covariates to control for
    n_bootstrap : int
        Number of bootstrap samples for indirect effect CI
    ci_level : float
        Confidence interval level (default 0.95)
    """
    X = np.asarray(X).flatten()
    M = np.asarray(M).flatten()
    Y = np.asarray(Y).flatten()
    
    # Handle missing values
    valid = np.isfinite(X) & np.isfinite(M) & np.isfinite(Y)
    if covariates is not None:
        covariates = np.atleast_2d(covariates)
        if covariates.shape[0] == len(X):
            covariates = covariates.T
        valid &= np.all(np.isfinite(covariates), axis=0)
    
    n = np.sum(valid)
    if n < 10:
        return MediationResult(
            a_path=np.nan, a_se=np.nan, a_p=np.nan,
            b_path=np.nan, b_se=np.nan, b_p=np.nan,
            c_path=np.nan, c_se=np.nan, c_p=np.nan,
            c_prime=np.nan, c_prime_se=np.nan, c_prime_p=np.nan,
            indirect_effect=np.nan, indirect_se=np.nan,
            indirect_ci_low=np.nan, indirect_ci_high=np.nan, indirect_p=np.nan,
            proportion_mediated=np.nan, sobel_z=np.nan, sobel_p=np.nan,
            n=n, significant=False
        )
    
    X, M, Y = X[valid], M[valid], Y[valid]
    if covariates is not None:
        covariates = covariates[:, valid].T
    
    # Build design matrices
    ones = np.ones((n, 1))
    X_col = X.reshape(-1, 1)
    M_col = M.reshape(-1, 1)
    
    if covariates is not None:
        base_covs = np.column_stack([ones, covariates])
    else:
        base_covs = ones
    
    # Path a: X -> M
    X_a = np.column_stack([base_covs, X_col])
    beta_a, se_a, _ = _ols_regression(M, X_a)
    a = beta_a[-1]
    a_se = se_a[-1]
    df_a = n - X_a.shape[1]
    a_t = a / (a_se + 1e-12)
    a_p = 2 * (1 - stats.t.cdf(abs(a_t), df_a))
    
    # Path c (total): X -> Y
    X_c = np.column_stack([base_covs, X_col])
    beta_c, se_c, _ = _ols_regression(Y, X_c)
    c = beta_c[-1]
    c_se = se_c[-1]
    df_c = n - X_c.shape[1]
    c_t = c / (c_se + 1e-12)
    c_p = 2 * (1 - stats.t.cdf(abs(c_t), df_c))
    
    # Path b and c' (direct): M -> Y and X -> Y controlling for M
    X_bc = np.column_stack([base_covs, X_col, M_col])
    beta_bc, se_bc, _ = _ols_regression(Y, X_bc)
    c_prime = beta_bc[-2]  # X coefficient
    c_prime_se = se_bc[-2]
    b = beta_bc[-1]  # M coefficient
    b_se = se_bc[-1]
    df_bc = n - X_bc.shape[1]
    
    c_prime_t = c_prime / (c_prime_se + 1e-12)
    c_prime_p = 2 * (1 - stats.t.cdf(abs(c_prime_t), df_bc))
    b_t = b / (b_se + 1e-12)
    b_p = 2 * (1 - stats.t.cdf(abs(b_t), df_bc))
    
    # Indirect effect: a * b
    indirect = a * b
    
    # Sobel test for indirect effect
    sobel_se = np.sqrt(a**2 * b_se**2 + b**2 * a_se**2)
    sobel_z = indirect / (sobel_se + 1e-12)
    sobel_p = 2 * (1 - stats.norm.cdf(abs(sobel_z)))
    
    # Bootstrap CI for indirect effect
    rng = np.random.default_rng(42)
    indirect_boot = np.zeros(n_bootstrap)
    
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        X_b, M_b, Y_b = X[idx], M[idx], Y[idx]
        
        if covariates is not None:
            cov_b = covariates[idx]
            base_b = np.column_stack([np.ones((n, 1)), cov_b])
        else:
            base_b = np.ones((n, 1))
        
        # a path
        X_a_b = np.column_stack([base_b, X_b.reshape(-1, 1)])
        beta_a_b, _, _ = _ols_regression(M_b, X_a_b)
        a_b = beta_a_b[-1]
        
        # b path
        X_bc_b = np.column_stack([base_b, X_b.reshape(-1, 1), M_b.reshape(-1, 1)])
        beta_bc_b, _, _ = _ols_regression(Y_b, X_bc_b)
        b_b = beta_bc_b[-1]
        
        indirect_boot[i] = a_b * b_b
    
    # BCa confidence interval
    alpha = 1 - ci_level
    z0 = stats.norm.ppf(np.mean(indirect_boot < indirect))
    
    # Jackknife for acceleration
    jack_indirect = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        X_j, M_j, Y_j = X[mask], M[mask], Y[mask]
        
        if covariates is not None:
            cov_j = covariates[mask]
            base_j = np.column_stack([np.ones((n-1, 1)), cov_j])
        else:
            base_j = np.ones((n-1, 1))
        
        X_a_j = np.column_stack([base_j, X_j.reshape(-1, 1)])
        beta_a_j, _, _ = _ols_regression(M_j, X_a_j)
        
        X_bc_j = np.column_stack([base_j, X_j.reshape(-1, 1), M_j.reshape(-1, 1)])
        beta_bc_j, _, _ = _ols_regression(Y_j, X_bc_j)
        
        jack_indirect[i] = beta_a_j[-1] * beta_bc_j[-1]
    
    jack_mean = np.mean(jack_indirect)
    num = np.sum((jack_mean - jack_indirect) ** 3)
    denom = 6 * (np.sum((jack_mean - jack_indirect) ** 2) ** 1.5)
    accel = num / (denom + 1e-12)
    
    z_alpha_low = stats.norm.ppf(alpha / 2)
    z_alpha_high = stats.norm.ppf(1 - alpha / 2)
    
    p_low = stats.norm.cdf(z0 + (z0 + z_alpha_low) / (1 - accel * (z0 + z_alpha_low)))
    p_high = stats.norm.cdf(z0 + (z0 + z_alpha_high) / (1 - accel * (z0 + z_alpha_high)))
    
    ci_low = np.nanpercentile(indirect_boot, max(0, min(100, p_low * 100)))
    ci_high = np.nanpercentile(indirect_boot, max(0, min(100, p_high * 100)))
    
    # Proportion mediated
    if abs(c) > 1e-12:
        prop_mediated = indirect / c
    else:
        prop_mediated = np.nan
    
    # Significance: CI doesn't include zero
    significant = (ci_low > 0 and ci_high > 0) or (ci_low < 0 and ci_high < 0)
    
    return MediationResult(
        a_path=float(a), a_se=float(a_se), a_p=float(a_p),
        b_path=float(b), b_se=float(b_se), b_p=float(b_p),
        c_path=float(c), c_se=float(c_se), c_p=float(c_p),
        c_prime=float(c_prime), c_prime_se=float(c_prime_se), c_prime_p=float(c_prime_p),
        indirect_effect=float(indirect), indirect_se=float(sobel_se),
        indirect_ci_low=float(ci_low), indirect_ci_high=float(ci_high),
        indirect_p=float(sobel_p),
        proportion_mediated=float(prop_mediated),
        sobel_z=float(sobel_z), sobel_p=float(sobel_p),
        n=n, significant=significant
    )


def run_mediation_analysis(
    df: pd.DataFrame,
    x_col: str,
    mediator_cols: List[str],
    y_col: str,
    covariates: List[str] = None,
    n_bootstrap: int = 5000,
) -> pd.DataFrame:
    """Run mediation analysis for multiple potential mediators."""
    X = df[x_col].values
    Y = df[y_col].values
    
    if covariates:
        cov_data = df[covariates].values
    else:
        cov_data = None
    
    results = []
    
    for mediator in mediator_cols:
        if mediator not in df.columns:
            continue
        
        M = df[mediator].values
        result = test_mediation(X, M, Y, covariates=cov_data, n_bootstrap=n_bootstrap)
        
        results.append({
            "mediator": mediator,
            "x_variable": x_col,
            "y_variable": y_col,
            "a_path": result.a_path,
            "a_p": result.a_p,
            "b_path": result.b_path,
            "b_p": result.b_p,
            "c_path": result.c_path,
            "c_p": result.c_p,
            "c_prime": result.c_prime,
            "c_prime_p": result.c_prime_p,
            "indirect_effect": result.indirect_effect,
            "indirect_ci_low": result.indirect_ci_low,
            "indirect_ci_high": result.indirect_ci_high,
            "indirect_p": result.indirect_p,
            "proportion_mediated": result.proportion_mediated,
            "sobel_z": result.sobel_z,
            "sobel_p": result.sobel_p,
            "n": result.n,
            "significant": result.significant,
        })
    
    return pd.DataFrame(results)

