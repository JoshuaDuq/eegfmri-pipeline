"""
EEG-Specific Statistics
=======================

Domain-specific statistical functions for EEG analysis.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import label

if TYPE_CHECKING:
    import mne

from .base import get_config_value, ensure_config, get_statistics_constants, get_fdr_alpha
from .correlation import compute_correlation, fisher_z, inverse_fisher_z


def align_epochs_to_pivot_chs(
    epochs: "mne.Epochs",
    pivot_channels: List[str],
    logger: Optional[logging.Logger] = None,
) -> "mne.Epochs":
    """Align epochs to subset of pivot channels."""
    import mne
    
    available = set(epochs.ch_names)
    valid_pivots = [ch for ch in pivot_channels if ch in available]
    
    if not valid_pivots:
        if logger:
            logger.warning("No pivot channels found in epochs")
        return epochs
    
    return epochs.pick_channels(valid_pivots)


def compute_correlation_for_metric_state(
    metric_values: np.ndarray,
    target_values: np.ndarray,
    state_idx: int,
    method: str = "spearman",
) -> Tuple[float, float]:
    """Compute correlation for a specific microstate metric."""
    valid = np.isfinite(metric_values) & np.isfinite(target_values)
    if np.sum(valid) < 3:
        return np.nan, np.nan
    
    return compute_correlation(metric_values[valid], target_values[valid], method)


def compute_channel_rating_correlations(
    channel_data: np.ndarray,
    ratings: np.ndarray,
    ch_names: List[str],
    method: str = "spearman",
) -> pd.DataFrame:
    """Compute correlation between each channel and ratings."""
    results = []
    for i, ch in enumerate(ch_names):
        ch_vals = channel_data[:, i] if channel_data.ndim == 2 else channel_data
        r, p = compute_correlation(ch_vals, ratings, method)
        results.append({"channel": ch, "r": r, "p": p})
    return pd.DataFrame(results)


def prepare_aligned_data(
    features_df: pd.DataFrame,
    targets: pd.Series,
    feature_cols: List[str],
) -> Tuple[pd.DataFrame, pd.Series]:
    """Align features and targets, removing NaN."""
    combined = features_df[feature_cols].copy()
    combined["_target"] = targets.values
    combined = combined.dropna()
    
    aligned_features = combined[feature_cols]
    aligned_targets = combined["_target"]
    
    return aligned_features, aligned_targets


def compute_residuals(
    log_freqs: np.ndarray,
    log_psd: np.ndarray,
    offsets: np.ndarray,
    slopes: np.ndarray,
) -> np.ndarray:
    """
    Compute residuals from an aperiodic fit.

    Supports vector (freq,) inputs as well as epoch/channel grids such as
    (epochs, channels, freqs) by broadcasting the offsets/slopes over the
    frequency axis.
    """
    log_freqs_arr = np.asarray(log_freqs)
    log_psd_arr = np.asarray(log_psd)
    offsets_arr = np.asarray(offsets)
    slopes_arr = np.asarray(slopes)

    if log_psd_arr.shape[-1] != log_freqs_arr.shape[-1]:
        raise ValueError(
            f"log_psd last dimension ({log_psd_arr.shape[-1]}) does not match "
            f"log_freqs length ({log_freqs_arr.shape[-1]})."
        )

    predicted = offsets_arr[..., None] + slopes_arr[..., None] * log_freqs_arr
    return log_psd_arr - predicted


def compute_robust_correlation(
    x: np.ndarray,
    y: np.ndarray,
    method: str = "percentage_bend",
) -> Tuple[float, float]:
    """Compute robust correlation (percentage bend or biweight midcorrelation)."""
    valid = np.isfinite(x) & np.isfinite(y)
    if np.sum(valid) < 5:
        return np.nan, np.nan
    
    x_v, y_v = x[valid], y[valid]
    
    if method == "percentage_bend":
        # Percentage bend correlation
        beta = 0.2
        n = len(x_v)
        
        # Compute psi values
        omega_x = np.sort(np.abs(x_v - np.median(x_v)))
        omega_y = np.sort(np.abs(y_v - np.median(y_v)))
        
        m = int(np.floor(beta * n))
        omega_x_m = omega_x[m] if m < n else omega_x[-1]
        omega_y_m = omega_y[m] if m < n else omega_y[-1]
        
        if omega_x_m < 1e-12 or omega_y_m < 1e-12:
            return compute_correlation(x_v, y_v, "spearman")
        
        psi_x = np.clip((x_v - np.median(x_v)) / omega_x_m, -1, 1)
        psi_y = np.clip((y_v - np.median(y_v)) / omega_y_m, -1, 1)
        
        r = np.sum(psi_x * psi_y) / np.sqrt(np.sum(psi_x**2) * np.sum(psi_y**2))
        
        # Approximate p-value using t-distribution
        t_stat = r * np.sqrt((n - 2) / (1 - r**2 + 1e-12))
        p = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=n - 2))
        
        return float(r), float(p)
    else:
        # Fallback to Spearman
        return compute_correlation(x_v, y_v, "spearman")


def compute_cluster_correction_2d(
    p_values: np.ndarray,
    t_values: np.ndarray,
    alpha: Optional[float] = None,
    n_permutations: int = 1000,
    rng: Optional[np.random.Generator] = None,
    config: Optional[Any] = None,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    2D cluster-based multiple comparison correction.
    
    Returns (significant_mask, cluster_info).
    """
    if alpha is None:
        alpha = get_fdr_alpha(config)
    if rng is None:
        rng = np.random.default_rng()
    
    # Get cluster structure
    try:
        constants = get_statistics_constants(config)
        structure = constants.get("cluster_structure_2d", np.ones((3, 3)))
    except:
        structure = np.ones((3, 3))
    
    # Initial threshold
    thresh_mask = p_values < alpha
    
    # Find clusters
    labeled, n_clusters = label(thresh_mask, structure=structure)
    
    if n_clusters == 0:
        return np.zeros_like(p_values, dtype=bool), []
    
    # Compute observed cluster masses
    obs_masses = []
    for i in range(1, n_clusters + 1):
        cluster_mask = labeled == i
        mass = np.abs(t_values[cluster_mask]).sum()
        obs_masses.append((i, mass, np.sum(cluster_mask)))
    
    # Permutation test for max cluster mass
    perm_max_masses = []
    for _ in range(n_permutations):
        # Random sign flipping
        signs = rng.choice([-1, 1], size=t_values.shape)
        perm_t = t_values * signs
        perm_p = 2 * (1 - stats.norm.cdf(np.abs(perm_t)))
        
        perm_thresh = perm_p < alpha
        perm_labeled, perm_n = label(perm_thresh, structure=structure)
        
        if perm_n > 0:
            max_mass = max(
                np.abs(perm_t[perm_labeled == i]).sum()
                for i in range(1, perm_n + 1)
            )
            perm_max_masses.append(max_mass)
        else:
            perm_max_masses.append(0)
    
    perm_max_masses = np.array(perm_max_masses)
    
    # Determine significant clusters
    sig_mask = np.zeros_like(p_values, dtype=bool)
    cluster_info = []
    
    for cluster_id, mass, size in obs_masses:
        p_cluster = (np.sum(perm_max_masses >= mass) + 1) / (n_permutations + 1)
        cluster_info.append({
            "cluster_id": cluster_id,
            "mass": float(mass),
            "size": int(size),
            "p_value": float(p_cluster),
        })
        if p_cluster <= alpha:
            sig_mask |= (labeled == cluster_id)
    
    return sig_mask, cluster_info


def get_fdr_alpha_from_config(config: Any) -> float:
    """Get FDR alpha from config (alias for get_fdr_alpha)."""
    return get_fdr_alpha(config)


def compute_bootstrap_ci(
    x: np.ndarray,
    y: np.ndarray,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    method: str = "spearman",
    rng: Optional[np.random.Generator] = None,
) -> Tuple[float, float]:
    """Compute bootstrap confidence interval for correlation."""
    if rng is None:
        rng = np.random.default_rng()
    
    valid = np.isfinite(x) & np.isfinite(y)
    n_valid = np.sum(valid)
    
    if n_valid < 4:
        return np.nan, np.nan
    
    x_v, y_v = x[valid], y[valid]
    
    boot_rs = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n_valid, size=n_valid)
        r, _ = compute_correlation(x_v[idx], y_v[idx], method)
        if np.isfinite(r):
            boot_rs.append(r)
    
    if len(boot_rs) < 10:
        return np.nan, np.nan
    
    alpha = 1 - ci_level
    lo = np.percentile(boot_rs, 100 * alpha / 2)
    hi = np.percentile(boot_rs, 100 * (1 - alpha / 2))
    
    return float(lo), float(hi)
