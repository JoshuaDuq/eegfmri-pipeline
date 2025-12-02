"""
Moderation Analysis
===================

Statistical moderation analysis to test whether the relationship between
X and Y depends on a third variable (moderator W).

Moderation Model:
    Y = b0 + b1*X + b2*W + b3*(X*W) + e
    
    b3 is the interaction term - if significant, W moderates the X→Y relationship.
    
Simple Slopes Analysis:
    Effect of X on Y at different levels of W:
    - Low W (mean - 1 SD)
    - Mean W
    - High W (mean + 1 SD)
    
Johnson-Neyman Interval:
    The range of W values where the effect of X on Y is significant.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class ModerationResult:
    """Container for moderation analysis results."""
    
    # Main effects
    b0: float = np.nan           # Intercept
    b1: float = np.nan           # X main effect
    b2: float = np.nan           # W (moderator) main effect
    b3: float = np.nan           # X*W interaction
    
    # Standard errors
    se_b1: float = np.nan
    se_b2: float = np.nan
    se_b3: float = np.nan
    
    # P-values
    p_b1: float = np.nan
    p_b2: float = np.nan
    p_b3: float = np.nan         # Key: is moderation significant?
    
    # Simple slopes at W levels
    slope_low_w: float = np.nan  # Effect at W = mean - 1SD
    slope_mean_w: float = np.nan # Effect at W = mean
    slope_high_w: float = np.nan # Effect at W = mean + 1SD
    
    p_slope_low: float = np.nan
    p_slope_mean: float = np.nan
    p_slope_high: float = np.nan
    
    # Johnson-Neyman interval (if applicable)
    jn_low: float = np.nan       # Lower bound of significance region
    jn_high: float = np.nan      # Upper bound of significance region
    jn_type: str = "none"        # "inside", "outside", "always", "never"
    
    # Model fit
    r_squared: float = np.nan
    r_squared_change: float = np.nan  # R² change due to interaction
    f_interaction: float = np.nan
    p_f_interaction: float = np.nan
    
    # Sample info
    n: int = 0
    
    # Labels
    x_label: str = "X"
    w_label: str = "W"
    y_label: str = "Y"
    
    def is_significant_moderation(self, alpha: float = 0.05) -> bool:
        """Check if the interaction term is significant."""
        return np.isfinite(self.p_b3) and self.p_b3 < alpha
    
    def summary_dict(self) -> Dict[str, Any]:
        """Return summary as dictionary."""
        return {
            "b1_x_effect": self.b1,
            "b2_w_effect": self.b2,
            "b3_interaction": self.b3,
            "p_interaction": self.p_b3,
            "slope_low_w": self.slope_low_w,
            "slope_mean_w": self.slope_mean_w,
            "slope_high_w": self.slope_high_w,
            "r_squared": self.r_squared,
            "r_squared_change": self.r_squared_change,
            "jn_interval": (self.jn_low, self.jn_high),
            "jn_type": self.jn_type,
            "n": self.n,
            "significant": self.is_significant_moderation(),
        }


###################################################################
# Core Computation
###################################################################


def _ols_regression_full(y: np.ndarray, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """OLS regression returning coefficients, SEs, R², and residual variance."""
    n, p = X.shape
    
    try:
        XtX_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        return np.full(p, np.nan), np.full(p, np.nan), np.nan, np.nan
    
    beta = XtX_inv @ X.T @ y
    residuals = y - X @ beta
    
    df = n - p
    if df <= 0:
        return beta, np.full(p, np.nan), np.nan, np.nan
    
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
    
    sigma_squared = ss_res / df
    var_beta = sigma_squared * np.diag(XtX_inv)
    se = np.sqrt(var_beta)
    
    return beta, se, r_squared, sigma_squared


def compute_moderation_effect(
    X: np.ndarray,
    W: np.ndarray,
    Y: np.ndarray,
    center_predictors: bool = True,
) -> ModerationResult:
    """Compute moderation analysis with interaction term.
    
    Parameters
    ----------
    X : np.ndarray
        Predictor variable (e.g., neural feature)
    W : np.ndarray
        Moderator variable (e.g., connectivity strength)
    Y : np.ndarray
        Outcome variable (e.g., pain rating)
    center_predictors : bool
        Whether to mean-center X and W (recommended for interpretation)
        
    Returns
    -------
    ModerationResult
        Full moderation analysis results
    """
    # Clean data
    mask = np.isfinite(X) & np.isfinite(W) & np.isfinite(Y)
    n = np.sum(mask)
    
    result = ModerationResult(n=n)
    
    if n < 15:  # Need sufficient df for interaction
        return result
    
    X_c = X[mask].copy()
    W_c = W[mask].copy()
    Y_c = Y[mask].copy()
    
    # Mean center predictors
    X_mean, W_mean = np.mean(X_c), np.mean(W_c)
    W_std = np.std(W_c)
    
    if center_predictors:
        X_c = X_c - X_mean
        W_c = W_c - W_mean
    
    # Create interaction term
    XW = X_c * W_c
    
    # Full model: Y ~ 1 + X + W + X*W
    X_design_full = np.column_stack([np.ones(n), X_c, W_c, XW])
    beta_full, se_full, r2_full, _ = _ols_regression_full(Y_c, X_design_full)
    
    # Reduced model: Y ~ 1 + X + W (for R² change)
    X_design_reduced = np.column_stack([np.ones(n), X_c, W_c])
    _, _, r2_reduced, _ = _ols_regression_full(Y_c, X_design_reduced)
    
    result.b0 = beta_full[0]
    result.b1 = beta_full[1]
    result.b2 = beta_full[2]
    result.b3 = beta_full[3]
    
    result.se_b1 = se_full[1]
    result.se_b2 = se_full[2]
    result.se_b3 = se_full[3]
    
    # P-values
    df = n - 4
    for attr, b, se in [('p_b1', result.b1, result.se_b1),
                        ('p_b2', result.b2, result.se_b2),
                        ('p_b3', result.b3, result.se_b3)]:
        if np.isfinite(b) and np.isfinite(se) and se > 0:
            t_stat = b / se
            p_val = 2 * (1 - stats.t.cdf(np.abs(t_stat), df))
            setattr(result, attr, p_val)
    
    result.r_squared = r2_full
    result.r_squared_change = r2_full - r2_reduced if np.isfinite(r2_reduced) else np.nan
    
    # F-test for interaction term
    if np.isfinite(r2_full) and np.isfinite(r2_reduced):
        num = (r2_full - r2_reduced) / 1  # 1 df for interaction
        denom = (1 - r2_full) / (n - 4)
        if denom > 0:
            result.f_interaction = num / denom
            result.p_f_interaction = 1 - stats.f.cdf(result.f_interaction, 1, n - 4)
    
    # Simple slopes analysis
    result = _compute_simple_slopes(result, X_c, W_c, Y_c, W_std)
    
    # Johnson-Neyman interval
    result = _compute_johnson_neyman(result, n, W_mean, W_std)
    
    return result


def _compute_simple_slopes(
    result: ModerationResult,
    X_c: np.ndarray,
    W_c: np.ndarray,
    Y_c: np.ndarray,
    W_std: float,
) -> ModerationResult:
    """Compute simple slopes at W = mean ± 1SD."""
    n = len(X_c)
    
    # W levels (in centered units)
    w_levels = [-W_std, 0, W_std]  # low, mean, high
    
    for i, w_val in enumerate(w_levels):
        # Effect of X at this W level: b1 + b3 * w_val
        slope = result.b1 + result.b3 * w_val
        
        # SE of conditional slope
        # Var(b1 + b3*w) = Var(b1) + w²*Var(b3) + 2*w*Cov(b1,b3)
        # Approximation: assume Cov ≈ 0
        se_slope = np.sqrt(result.se_b1**2 + (w_val**2) * result.se_b3**2)
        
        t_stat = slope / se_slope if se_slope > 0 else np.nan
        p_val = 2 * (1 - stats.t.cdf(np.abs(t_stat), n - 4)) if np.isfinite(t_stat) else np.nan
        
        if i == 0:
            result.slope_low_w = slope
            result.p_slope_low = p_val
        elif i == 1:
            result.slope_mean_w = slope
            result.p_slope_mean = p_val
        else:
            result.slope_high_w = slope
            result.p_slope_high = p_val
    
    return result


def _compute_johnson_neyman(
    result: ModerationResult,
    n: int,
    W_mean: float,
    W_std: float,
) -> ModerationResult:
    """Compute Johnson-Neyman regions of significance."""
    if not np.isfinite(result.b3) or not np.isfinite(result.se_b3):
        return result
    
    # Critical t-value
    alpha = 0.05
    df = n - 4
    t_crit = stats.t.ppf(1 - alpha/2, df)
    
    # The conditional effect is: slope(W) = b1 + b3*W
    # SE(slope) = sqrt(Var(b1) + W²*Var(b3) + 2*W*Cov(b1,b3))
    # Assuming Cov(b1,b3) ≈ 0:
    # SE(slope) = sqrt(se_b1² + W²*se_b3²)
    
    # significance when: |b1 + b3*W| / sqrt(se_b1² + W²*se_b3²) = t_crit
    # This is a quadratic in W
    
    a = result.b3**2 - t_crit**2 * result.se_b3**2
    b = 2 * result.b1 * result.b3
    c = result.b1**2 - t_crit**2 * result.se_b1**2
    
    discriminant = b**2 - 4*a*c
    
    if discriminant < 0:
        # No real roots - effect is either always or never significant
        # Check at W = 0
        t_at_mean = abs(result.b1) / result.se_b1 if result.se_b1 > 0 else 0
        result.jn_type = "always" if t_at_mean > t_crit else "never"
    elif abs(a) < 1e-10:
        # Linear case (very small interaction)
        result.jn_type = "never"
    else:
        # Two roots
        root1 = (-b - np.sqrt(discriminant)) / (2 * a)
        root2 = (-b + np.sqrt(discriminant)) / (2 * a)
        
        # Convert back to original scale
        jn1 = root1 + W_mean
        jn2 = root2 + W_mean
        
        result.jn_low = min(jn1, jn2)
        result.jn_high = max(jn1, jn2)
        
        # Determine type based on a
        if a > 0:
            result.jn_type = "outside"  # Significant outside the interval
        else:
            result.jn_type = "inside"   # Significant inside the interval
    
    return result


def run_moderation_analysis(
    X: np.ndarray,
    W: np.ndarray,
    Y: np.ndarray,
    x_label: str = "Predictor",
    w_label: str = "Moderator",
    y_label: str = "Outcome",
    center_predictors: bool = True,
) -> ModerationResult:
    """Run complete moderation analysis.
    
    Parameters
    ----------
    X : np.ndarray
        Predictor variable
    W : np.ndarray
        Moderator variable
    Y : np.ndarray
        Outcome variable
    x_label, w_label, y_label : str
        Labels for variables
    center_predictors : bool
        Whether to mean-center X and W
        
    Returns
    -------
    ModerationResult
        Complete moderation analysis results
    """
    result = compute_moderation_effect(X, W, Y, center_predictors=center_predictors)
    result.x_label = x_label
    result.w_label = w_label
    result.y_label = y_label
    return result


__all__ = [
    "ModerationResult",
    "compute_moderation_effect",
    "run_moderation_analysis",
]
