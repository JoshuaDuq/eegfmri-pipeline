"""
Mediation Analysis
==================

Statistical mediation analysis for the Temperature → Neural Feature → Pain pathway.

Implements Baron & Kenny (1986) approach with modern extensions:
- Bootstrap confidence intervals for indirect effect
- Sobel test for indirect effect significance
- Proportion mediated

Theoretical Framework:
    X = Temperature (stimulus intensity)
    M = Neural feature (e.g., alpha power)
    Y = Pain rating (subjective report)

Path Model:
    X → M (path a)
    M → Y controlling for X (path b)  
    X → Y (path c, total effect)
    X → Y controlling for M (path c', direct effect)
    
    Indirect effect = a × b
    Total effect c = c' + ab
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class MediationResult:
    """Container for mediation analysis results."""
    
    # Path coefficients
    a: float = np.nan          # X → M
    se_a: float = np.nan
    p_a: float = np.nan
    
    b: float = np.nan          # M → Y | X
    se_b: float = np.nan
    p_b: float = np.nan
    
    c: float = np.nan          # X → Y (total)
    se_c: float = np.nan
    p_c: float = np.nan
    
    c_prime: float = np.nan    # X → Y | M (direct)
    se_c_prime: float = np.nan
    p_c_prime: float = np.nan
    
    # Indirect effect
    ab: float = np.nan         # a × b
    se_ab: float = np.nan      # Sobel SE
    p_ab: float = np.nan       # Sobel test p-value
    ci_ab_low: float = np.nan  # Bootstrap CI
    ci_ab_high: float = np.nan
    
    # Effect proportions
    proportion_mediated: float = np.nan  # ab / c
    
    # Sample info
    n: int = 0
    
    # Variable labels
    x_label: str = "X"
    m_label: str = "M"
    y_label: str = "Y"
    
    def is_significant_mediation(self, alpha: float = 0.05) -> bool:
        """Check if there is significant mediation."""
        # All paths should be significant, and indirect effect CI shouldn't contain 0
        paths_sig = (self.p_a < alpha and self.p_b < alpha)
        ci_excludes_zero = (self.ci_ab_low > 0) or (self.ci_ab_high < 0)
        return paths_sig and ci_excludes_zero
    
    def summary_dict(self) -> Dict[str, Any]:
        """Return summary as dictionary."""
        return {
            "path_a": self.a,
            "path_b": self.b,
            "path_c_total": self.c,
            "path_c_prime_direct": self.c_prime,
            "indirect_effect_ab": self.ab,
            "sobel_p": self.p_ab,
            "ci_ab_low": self.ci_ab_low,
            "ci_ab_high": self.ci_ab_high,
            "proportion_mediated": self.proportion_mediated,
            "n": self.n,
            "significant": self.is_significant_mediation(),
        }


###################################################################
# Core Computation
###################################################################


def _ols_regression(y: np.ndarray, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Simple OLS regression returning coefficients, standard errors, and residual variance.
    
    Parameters
    ----------
    y : np.ndarray
        Dependent variable (n,)
    X : np.ndarray  
        Design matrix including intercept (n, p)
        
    Returns
    -------
    beta : np.ndarray
        Coefficients (p,)
    se : np.ndarray
        Standard errors (p,)
    sigma_squared : float
        Residual variance
    """
    n, p = X.shape
    
    try:
        XtX_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        return np.full(p, np.nan), np.full(p, np.nan), np.nan
    
    beta = XtX_inv @ X.T @ y
    residuals = y - X @ beta
    
    df = n - p
    if df <= 0:
        return beta, np.full(p, np.nan), np.nan
    
    sigma_squared = np.sum(residuals**2) / df
    var_beta = sigma_squared * np.diag(XtX_inv)
    se = np.sqrt(var_beta)
    
    return beta, se, sigma_squared


def compute_mediation_paths(
    X: np.ndarray,
    M: np.ndarray,
    Y: np.ndarray,
) -> MediationResult:
    """Compute mediation path coefficients using OLS.
    
    Parameters
    ----------
    X : np.ndarray
        Independent variable (temperature)
    M : np.ndarray
        Mediator (neural feature)
    Y : np.ndarray
        Dependent variable (pain rating)
        
    Returns
    -------
    MediationResult
        Path coefficients and statistics
    """
    # Clean data
    mask = np.isfinite(X) & np.isfinite(M) & np.isfinite(Y)
    n = np.sum(mask)
    
    result = MediationResult(n=n)
    
    if n < 10:
        return result
    
    X_c = X[mask]
    M_c = M[mask]
    Y_c = Y[mask]
    
    # Path a: X → M
    X_design = np.column_stack([np.ones(n), X_c])
    beta_a, se_a, _ = _ols_regression(M_c, X_design)
    result.a = beta_a[1]
    result.se_a = se_a[1]
    t_a = result.a / result.se_a if result.se_a > 0 else np.nan
    result.p_a = 2 * (1 - stats.t.cdf(np.abs(t_a), df=n-2)) if np.isfinite(t_a) else np.nan
    
    # Path c: X → Y (total effect)
    beta_c, se_c, _ = _ols_regression(Y_c, X_design)
    result.c = beta_c[1]
    result.se_c = se_c[1]
    t_c = result.c / result.se_c if result.se_c > 0 else np.nan
    result.p_c = 2 * (1 - stats.t.cdf(np.abs(t_c), df=n-2)) if np.isfinite(t_c) else np.nan
    
    # Paths b and c': Y ~ X + M
    XM_design = np.column_stack([np.ones(n), X_c, M_c])
    beta_bc, se_bc, _ = _ols_regression(Y_c, XM_design)
    
    result.c_prime = beta_bc[1]  # Direct effect
    result.se_c_prime = se_bc[1]
    t_cp = result.c_prime / result.se_c_prime if result.se_c_prime > 0 else np.nan
    result.p_c_prime = 2 * (1 - stats.t.cdf(np.abs(t_cp), df=n-3)) if np.isfinite(t_cp) else np.nan
    
    result.b = beta_bc[2]  # M → Y | X
    result.se_b = se_bc[2]
    t_b = result.b / result.se_b if result.se_b > 0 else np.nan
    result.p_b = 2 * (1 - stats.t.cdf(np.abs(t_b), df=n-3)) if np.isfinite(t_b) else np.nan
    
    # Indirect effect
    result.ab = result.a * result.b
    
    # Sobel test
    if np.isfinite(result.a) and np.isfinite(result.b):
        var_ab = (result.a**2 * result.se_b**2 + 
                  result.b**2 * result.se_a**2)
        result.se_ab = np.sqrt(var_ab)
        z_sobel = result.ab / result.se_ab if result.se_ab > 0 else np.nan
        result.p_ab = 2 * (1 - stats.norm.cdf(np.abs(z_sobel))) if np.isfinite(z_sobel) else np.nan
    
    # Proportion mediated
    if np.isfinite(result.c) and abs(result.c) > 1e-10:
        result.proportion_mediated = result.ab / result.c
    
    return result


def bootstrap_indirect_effect(
    X: np.ndarray,
    M: np.ndarray,
    Y: np.ndarray,
    n_boot: int = 5000,
    ci_level: float = 0.95,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float, np.ndarray]:
    """Bootstrap confidence interval for indirect effect (a×b).
    
    Uses percentile method which is more robust than Sobel for small samples.
    
    Returns
    -------
    ci_low : float
    ci_high : float
    boot_distribution : np.ndarray
    """
    if rng is None:
        rng = np.random.default_rng()
    
    mask = np.isfinite(X) & np.isfinite(M) & np.isfinite(Y)
    n = np.sum(mask)
    
    if n < 10:
        return np.nan, np.nan, np.array([])
    
    X_c = X[mask]
    M_c = M[mask]
    Y_c = Y[mask]
    
    boot_ab = []
    
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        X_b, M_b, Y_b = X_c[idx], M_c[idx], Y_c[idx]
        
        result = compute_mediation_paths(X_b, M_b, Y_b)
        if np.isfinite(result.ab):
            boot_ab.append(result.ab)
    
    if len(boot_ab) < 100:
        return np.nan, np.nan, np.array([])
    
    boot_ab = np.array(boot_ab)
    alpha = 1 - ci_level
    ci_low = np.percentile(boot_ab, 100 * alpha / 2)
    ci_high = np.percentile(boot_ab, 100 * (1 - alpha / 2))
    
    return float(ci_low), float(ci_high), boot_ab


def run_full_mediation_analysis(
    X: np.ndarray,
    M: np.ndarray,
    Y: np.ndarray,
    n_boot: int = 5000,
    x_label: str = "Temperature",
    m_label: str = "Power",
    y_label: str = "Pain Rating",
    rng: Optional[np.random.Generator] = None,
) -> MediationResult:
    """Run complete mediation analysis with bootstrap CIs.
    
    Parameters
    ----------
    X : np.ndarray
        Independent variable (temperature)
    M : np.ndarray
        Mediator (neural feature)  
    Y : np.ndarray
        Dependent variable (pain rating)
    n_boot : int
        Number of bootstrap iterations
    x_label, m_label, y_label : str
        Labels for variables
    rng : np.random.Generator, optional
        Random generator for reproducibility
        
    Returns
    -------
    MediationResult
        Complete mediation results
    """
    # Compute paths
    result = compute_mediation_paths(X, M, Y)
    result.x_label = x_label
    result.m_label = m_label
    result.y_label = y_label
    
    # Bootstrap CI for indirect effect
    ci_low, ci_high, _ = bootstrap_indirect_effect(
        X, M, Y, n_boot=n_boot, rng=rng
    )
    result.ci_ab_low = ci_low
    result.ci_ab_high = ci_high
    
    return result


###################################################################
# Batch Processing
###################################################################


def analyze_mediation_for_features(
    X: np.ndarray,
    feature_df: pd.DataFrame,
    Y: np.ndarray,
    n_boot: int = 2000,
    min_effect_size: float = 0.1,
    x_label: str = "Temperature",
    y_label: str = "Pain Rating",
    logger: Optional[logging.Logger] = None,
    rng: Optional[np.random.Generator] = None,
) -> List[MediationResult]:
    """Run mediation analysis for multiple neural features.
    
    Tests whether each feature mediates the X → Y relationship.
    
    Parameters
    ----------
    X : np.ndarray
        Independent variable
    feature_df : pd.DataFrame
        DataFrame where each column is a potential mediator
    Y : np.ndarray
        Dependent variable
    n_boot : int
        Bootstrap iterations per feature
    min_effect_size : float
        Minimum |r| between X-M or M-Y to consider testing
    x_label, y_label : str
        Variable labels
    logger : Logger, optional
    rng : np.random.Generator, optional
        
    Returns
    -------
    List[MediationResult]
        Results for features showing mediation potential
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    if rng is None:
        rng = np.random.default_rng()
    
    results = []
    
    for col in feature_df.columns:
        M = feature_df[col].values
        
        # Quick screening: check if feature is related to both X and Y
        mask = np.isfinite(X) & np.isfinite(M) & np.isfinite(Y)
        if mask.sum() < 20:
            continue
        
        r_xm, _ = stats.spearmanr(X[mask], M[mask])
        r_my, _ = stats.spearmanr(M[mask], Y[mask])
        
        if abs(r_xm) < min_effect_size or abs(r_my) < min_effect_size:
            continue
        
        # Full analysis
        result = run_full_mediation_analysis(
            X, M, Y,
            n_boot=n_boot,
            x_label=x_label,
            m_label=col,
            y_label=y_label,
            rng=rng,
        )
        
        if result.is_significant_mediation():
            results.append(result)
            logger.info(f"Significant mediation: {col} (ab={result.ab:.3f}, "
                       f"CI=[{result.ci_ab_low:.3f}, {result.ci_ab_high:.3f}])")
    
    logger.info(f"Found {len(results)} significant mediators out of {len(feature_df.columns)} tested")
    return results


__all__ = [
    "MediationResult",
    "compute_mediation_paths",
    "bootstrap_indirect_effect",
    "run_full_mediation_analysis",
    "analyze_mediation_for_features",
]
