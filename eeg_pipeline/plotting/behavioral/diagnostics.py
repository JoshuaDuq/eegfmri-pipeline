"""
Comprehensive Regression Diagnostics
=====================================

Extended diagnostic visualizations for behavioral correlations including:
- Cook's distance / leverage plots
- Normality tests with Shapiro-Wilk
- Multicollinearity diagnostics (VIF)
- Scale-location plots
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from eeg_pipeline.plotting.config import get_plot_config, PlotConfig
from eeg_pipeline.plotting.io.figures import (
    save_fig,
    get_behavior_footer as _get_behavior_footer,
    get_default_config as _get_default_config,
)
from eeg_pipeline.infra.logging import get_default_logger as _get_default_logger
from eeg_pipeline.infra.paths import ensure_dir


###################################################################
# Cook's Distance and Leverage
###################################################################


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
    n = len(x)
    if n < 4:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan), np.nan
    
    # Design matrix [1, x]
    X = np.column_stack([np.ones(n), x])
    
    # Hat matrix diagonal
    try:
        XtX_inv = np.linalg.inv(X.T @ X)
        hat_diag = np.diag(X @ XtX_inv @ X.T)
    except np.linalg.LinAlgError:
        return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan), np.nan
    
    # Fitted values and residuals
    beta = XtX_inv @ X.T @ y
    y_hat = X @ beta
    residuals = y - y_hat
    
    # MSE
    p = 2  # intercept + slope
    mse = np.sum(residuals**2) / (n - p)
    
    # Studentized residuals
    with np.errstate(divide='ignore', invalid='ignore'):
        student_resid = residuals / np.sqrt(mse * (1 - hat_diag))
    
    # Cook's distance
    with np.errstate(divide='ignore', invalid='ignore'):
        cooks_d = (student_resid**2 / p) * (hat_diag / (1 - hat_diag))
    
    # Threshold: 4/n is common rule of thumb
    cooks_threshold = 4.0 / n
    
    return hat_diag, cooks_d, student_resid, cooks_threshold


def _plot_cooks_distance(
    ax: plt.Axes,
    cooks_d: np.ndarray,
    threshold: float,
    band_color: str,
    plot_cfg: PlotConfig,
) -> None:
    """Plot Cook's distance bar chart with threshold line."""
    n = len(cooks_d)
    indices = np.arange(n)
    
    # Identify influential points
    influential = cooks_d > threshold
    
    # Bar colors
    colors = [plot_cfg.get_color("significant", plot_type="behavioral") if inf else band_color 
              for inf in influential]
    
    ax.bar(indices, cooks_d, color=colors, alpha=0.7, edgecolor='none')
    ax.axhline(threshold, color='red', linestyle='--', linewidth=1.5, 
               label=f'Threshold (4/n = {threshold:.3f})')
    
    # Label influential points
    for i, (cd, inf) in enumerate(zip(cooks_d, influential)):
        if inf and np.isfinite(cd):
            ax.text(i, cd, str(i), ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel("Observation Index", fontsize=plot_cfg.font.label)
    ax.set_ylabel("Cook's Distance", fontsize=plot_cfg.font.label)
    ax.set_title("Cook's Distance", fontsize=plot_cfg.font.title, fontweight="bold")
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)


def _plot_leverage_residuals(
    ax: plt.Axes,
    leverage: np.ndarray,
    student_resid: np.ndarray,
    cooks_d: np.ndarray,
    threshold: float,
    band_color: str,
    plot_cfg: PlotConfig,
) -> None:
    """Plot studentized residuals vs leverage with Cook's distance contours."""
    influential = cooks_d > threshold
    
    # Scatter points
    marker_size = plot_cfg.get_scatter_marker_size(plot_type="behavioral")
    colors = [plot_cfg.get_color("significant", plot_type="behavioral") if inf else band_color 
              for inf in influential]
    
    ax.scatter(leverage, student_resid, s=marker_size, c=colors, 
               alpha=plot_cfg.style.scatter.alpha, edgecolor='white', linewidth=0.5)
    
    # Add reference lines
    ax.axhline(0, color='gray', linestyle='-', linewidth=0.8)
    ax.axhline(2, color='red', linestyle=':', linewidth=0.8, alpha=0.7)
    ax.axhline(-2, color='red', linestyle=':', linewidth=0.8, alpha=0.7)
    
    # Typical leverage threshold: 2(p+1)/n where p=1 for simple regression
    n = len(leverage)
    lev_threshold = 2 * 2 / n  # 2(p+1)/n
    ax.axvline(lev_threshold, color='orange', linestyle='--', linewidth=1, 
               label=f'High leverage ({lev_threshold:.3f})')
    
    ax.set_xlabel("Leverage", fontsize=plot_cfg.font.label)
    ax.set_ylabel("Studentized Residuals", fontsize=plot_cfg.font.label)
    ax.set_title("Residuals vs Leverage", fontsize=plot_cfg.font.title, fontweight="bold")
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)


###################################################################
# Normality Tests
###################################################################


def compute_normality_tests(residuals: np.ndarray) -> Dict[str, Any]:
    """Compute normality tests on residuals.
    
    Returns
    -------
    dict with keys:
        shapiro_stat, shapiro_p: Shapiro-Wilk test
        dagostino_stat, dagostino_p: D'Agostino K² test
        skewness, kurtosis: Distribution moments
        is_normal: Boolean (p > 0.05 for both tests)
    """
    clean = residuals[np.isfinite(residuals)]
    n = len(clean)
    
    result = {
        "shapiro_stat": np.nan,
        "shapiro_p": np.nan,
        "dagostino_stat": np.nan,
        "dagostino_p": np.nan,
        "skewness": np.nan,
        "kurtosis": np.nan,
        "is_normal": False,
        "n": n,
    }
    
    if n < 8:
        return result
    
    # Shapiro-Wilk (best for n < 5000)
    if n <= 5000:
        stat, p = stats.shapiro(clean)
        result["shapiro_stat"] = float(stat)
        result["shapiro_p"] = float(p)
    
    # D'Agostino K² (better for larger samples)
    if n >= 20:
        stat, p = stats.normaltest(clean)
        result["dagostino_stat"] = float(stat)
        result["dagostino_p"] = float(p)
    
    # Moments
    result["skewness"] = float(stats.skew(clean))
    result["kurtosis"] = float(stats.kurtosis(clean))
    
    # Determine if approximately normal
    shapiro_ok = result["shapiro_p"] > 0.05 if np.isfinite(result["shapiro_p"]) else True
    dagostino_ok = result["dagostino_p"] > 0.05 if np.isfinite(result["dagostino_p"]) else True
    result["is_normal"] = shapiro_ok and dagostino_ok
    
    return result


def _plot_normality_diagnostics(
    ax: plt.Axes,
    residuals: np.ndarray,
    band_color: str,
    plot_cfg: PlotConfig,
) -> None:
    """Plot residual histogram with normal overlay and test results."""
    clean = residuals[np.isfinite(residuals)]
    
    # Histogram
    ax.hist(clean, bins=20, density=True, color=band_color, alpha=0.7, 
            edgecolor='white', linewidth=0.5, label='Residuals')
    
    # Overlay normal distribution
    mu, sigma = np.mean(clean), np.std(clean)
    x_range = np.linspace(mu - 4*sigma, mu + 4*sigma, 100)
    normal_pdf = stats.norm.pdf(x_range, mu, sigma)
    ax.plot(x_range, normal_pdf, 'k-', linewidth=2, label='Normal fit')
    
    # Compute tests
    tests = compute_normality_tests(residuals)
    
    # Annotation
    annotation = f"Shapiro-Wilk: W={tests['shapiro_stat']:.3f}, p={tests['shapiro_p']:.3f}\n"
    if np.isfinite(tests['dagostino_p']):
        annotation += f"D'Agostino: K²={tests['dagostino_stat']:.1f}, p={tests['dagostino_p']:.3f}\n"
    annotation += f"Skew={tests['skewness']:.2f}, Kurt={tests['kurtosis']:.2f}"
    
    # Color based on normality
    text_color = 'green' if tests['is_normal'] else 'red'
    ax.text(0.98, 0.98, annotation, transform=ax.transAxes, 
            fontsize=9, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            color=text_color)
    
    ax.set_xlabel("Residual", fontsize=plot_cfg.font.label)
    ax.set_ylabel("Density", fontsize=plot_cfg.font.label)
    ax.set_title("Normality Check", fontsize=plot_cfg.font.title, fontweight="bold")
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)


###################################################################
# Scale-Location Plot
###################################################################


def _plot_scale_location(
    ax: plt.Axes,
    fitted: np.ndarray,
    residuals: np.ndarray,
    band_color: str,
    plot_cfg: PlotConfig,
) -> None:
    """Scale-Location plot: sqrt(|standardized residuals|) vs fitted values."""
    # Standardize residuals
    std_resid = residuals / np.std(residuals)
    sqrt_abs_resid = np.sqrt(np.abs(std_resid))
    
    marker_size = plot_cfg.get_scatter_marker_size(plot_type="behavioral")
    ax.scatter(fitted, sqrt_abs_resid, s=marker_size, c=band_color,
               alpha=plot_cfg.style.scatter.alpha, edgecolor='white', linewidth=0.5)
    
    # Add lowess smoother approximation
    # Sort by fitted values for smooth line
    sort_idx = np.argsort(fitted)
    fitted_sorted = fitted[sort_idx]
    resid_sorted = sqrt_abs_resid[sort_idx]
    
    # Moving average as simple smoother
    window = max(5, len(fitted) // 10)
    smoothed = np.convolve(resid_sorted, np.ones(window)/window, mode='valid')
    x_smooth = fitted_sorted[(window-1)//2:-(window//2)] if window > 1 else fitted_sorted
    
    if len(x_smooth) == len(smoothed):
        ax.plot(x_smooth, smoothed, 'r-', linewidth=2, alpha=0.8, label='Smooth')
    
    ax.set_xlabel("Fitted Values", fontsize=plot_cfg.font.label)
    ax.set_ylabel("√|Standardized Residuals|", fontsize=plot_cfg.font.label)
    ax.set_title("Scale-Location", fontsize=plot_cfg.font.title, fontweight="bold")
    ax.grid(True, alpha=0.3)


###################################################################
# Multicollinearity (VIF)
###################################################################


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
    n_cols = X.shape[1]
    vif_values = {}
    
    for i, col in enumerate(X.columns):
        if n_cols == 1:
            vif_values[col] = 1.0
            continue
            
        # Regress this column on all others
        y_temp = X[col].values
        X_other = X.drop(columns=[col]).values
        
        if X_other.shape[1] == 0:
            vif_values[col] = 1.0
            continue
        
        # Add intercept
        X_design = np.column_stack([np.ones(len(y_temp)), X_other])
        
        # Compute R²
        beta = np.linalg.lstsq(X_design, y_temp, rcond=None)[0]
        y_pred = X_design @ beta
        ss_res = np.sum((y_temp - y_pred)**2)
        ss_tot = np.sum((y_temp - np.mean(y_temp))**2)
        
        if ss_tot < 1e-10:
            r_squared = 0.0
        else:
            r_squared = 1 - (ss_res / ss_tot)
        
        r_squared = np.clip(r_squared, 0, 0.9999)
        vif = 1.0 / (1.0 - r_squared)
        vif_values[col] = float(vif)
    
    return pd.Series(vif_values)


def _plot_vif_diagnostics(
    ax: plt.Axes,
    vif_values: pd.Series,
    plot_cfg: PlotConfig,
) -> None:
    """Plot VIF bar chart with threshold lines."""
    names = vif_values.index.tolist()
    values = vif_values.values
    
    # Color by severity
    colors = []
    for v in values:
        if np.isnan(v):
            colors.append('gray')
        elif v > 10:
            colors.append('red')
        elif v > 5:
            colors.append('orange')
        else:
            colors.append('green')
    
    ax.barh(names, values, color=colors, alpha=0.8, edgecolor='white')
    
    # Threshold lines
    ax.axvline(5, color='orange', linestyle='--', linewidth=1.5, label='Moderate (5)')
    ax.axvline(10, color='red', linestyle='--', linewidth=1.5, label='High (10)')
    
    ax.set_xlabel("VIF", fontsize=plot_cfg.font.label)
    ax.set_ylabel("Covariate", fontsize=plot_cfg.font.label)
    ax.set_title("Multicollinearity (VIF)", fontsize=plot_cfg.font.title, fontweight="bold")
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3, axis='x')


###################################################################
# Comprehensive Diagnostics Plot
###################################################################


def plot_comprehensive_diagnostics(
    x_data: pd.Series,
    y_data: pd.Series,
    *,
    title_prefix: str,
    output_path: Path,
    band_color: str,
    Z_covars: Optional[pd.DataFrame] = None,
    logger: Optional[logging.Logger] = None,
    config=None,
) -> None:
    """Generate comprehensive 6-panel diagnostic plot.
    
    Panels:
    1. Residuals vs Fitted
    2. Normal Q-Q
    3. Scale-Location
    4. Cook's Distance
    5. Residuals vs Leverage
    6. Normality Test / VIF (if covariates)
    """
    config = config or _get_default_config()
    logger = logger or _get_default_logger()
    plot_cfg = get_plot_config(config)
    
    # Prepare data
    x_arr = pd.to_numeric(x_data, errors="coerce").values
    y_arr = pd.to_numeric(y_data, errors="coerce").values
    
    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if mask.sum() < 5:
        logger.warning("Insufficient data for comprehensive diagnostics")
        return
    
    x_clean = x_arr[mask]
    y_clean = y_arr[mask]
    
    # Compute regression
    from eeg_pipeline.utils.analysis.stats import compute_linear_residuals
    fitted, residuals, _ = compute_linear_residuals(pd.Series(x_clean), pd.Series(y_clean))
    
    # Compute diagnostics
    leverage, cooks_d, student_resid, cooks_threshold = compute_leverage_and_cooks(x_clean, y_clean)
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Panel 1: Residuals vs Fitted
    from eeg_pipeline.plotting.behavioral.builders import _plot_residuals_vs_fitted
    _plot_residuals_vs_fitted(axes[0, 0], fitted, residuals, band_color, plot_cfg)
    
    # Panel 2: Q-Q Plot
    from eeg_pipeline.plotting.behavioral.builders import _plot_qq_plot
    _plot_qq_plot(axes[0, 1], residuals, band_color, plot_cfg)
    
    # Panel 3: Scale-Location
    _plot_scale_location(axes[0, 2], fitted, residuals, band_color, plot_cfg)
    
    # Panel 4: Cook's Distance
    _plot_cooks_distance(axes[1, 0], cooks_d, cooks_threshold, band_color, plot_cfg)
    
    # Panel 5: Residuals vs Leverage
    _plot_leverage_residuals(axes[1, 1], leverage, student_resid, cooks_d, 
                             cooks_threshold, band_color, plot_cfg)
    
    # Panel 6: Normality or VIF
    if Z_covars is not None and not Z_covars.empty and Z_covars.shape[1] > 1:
        vif = compute_vif(Z_covars)
        _plot_vif_diagnostics(axes[1, 2], vif, plot_cfg)
    else:
        _plot_normality_diagnostics(axes[1, 2], residuals, band_color, plot_cfg)
    
    fig.suptitle(f"{title_prefix} — Comprehensive Diagnostics", 
                 fontsize=plot_cfg.font.figure_title, fontweight="bold", y=1.02)
    fig.tight_layout()
    
    save_fig(
        fig,
        output_path,
        formats=plot_cfg.formats,
        dpi=plot_cfg.dpi,
        bbox_inches=plot_cfg.bbox_inches,
        pad_inches=plot_cfg.pad_inches,
        footer=_get_behavior_footer(config),
        logger=logger,
    )
    plt.close(fig)
    logger.info(f"Comprehensive diagnostics saved to {output_path}")


__all__ = [
    "compute_leverage_and_cooks",
    "compute_normality_tests",
    "compute_vif",
    "plot_comprehensive_diagnostics",
]
