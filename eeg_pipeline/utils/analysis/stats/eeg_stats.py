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

from .base import (
    get_epsilon_std,
    get_fdr_alpha,
    get_statistics_constants,
)
from .correlation import compute_correlation


# Constants
_MIN_SAMPLES_CORRELATION = 3
_MIN_SAMPLES_ROBUST_CORRELATION = 5
_MIN_SAMPLES_BOOTSTRAP = 4
_MIN_BOOTSTRAP_RESULTS = 10
_PERCENTAGE_BEND_BETA = 0.2
_DEFAULT_CLUSTER_STRUCTURE_SIZE = 3


def align_epochs_to_pivot_chs(
    epochs: "mne.Epochs",
    pivot_channels: List[str],
    logger: Optional[logging.Logger] = None,
) -> "mne.Epochs":
    """Align epochs to subset of pivot channels."""
    import mne
    
    available_channels = set(epochs.ch_names)
    valid_pivot_channels = [
        channel for channel in pivot_channels if channel in available_channels
    ]
    
    if not valid_pivot_channels:
        if logger:
            logger.warning("No pivot channels found in epochs")
        return epochs
    
    return epochs.pick_channels(valid_pivot_channels)


def compute_correlation_for_metric_state(
    metric_values: np.ndarray,
    target_values: np.ndarray,
    state_idx: int,
    method: str = "spearman",
) -> Tuple[float, float]:
    """Compute correlation for a specific microstate metric."""
    valid_mask = np.isfinite(metric_values) & np.isfinite(target_values)
    n_valid = np.sum(valid_mask)
    
    if n_valid < _MIN_SAMPLES_CORRELATION:
        return np.nan, np.nan
    
    valid_metric_values = metric_values[valid_mask]
    valid_target_values = target_values[valid_mask]
    return compute_correlation(valid_metric_values, valid_target_values, method)


def compute_channel_rating_correlations(
    channel_data: np.ndarray,
    ratings: np.ndarray,
    ch_names: List[str],
    method: str = "spearman",
) -> pd.DataFrame:
    """Compute correlation between each channel and ratings."""
    results = []
    for channel_idx, channel_name in enumerate(ch_names):
        if channel_data.ndim == 2:
            channel_values = channel_data[:, channel_idx]
        else:
            channel_values = channel_data
        
        correlation, p_value = compute_correlation(channel_values, ratings, method)
        results.append({
            "channel": channel_name,
            "r": correlation,
            "p": p_value,
        })
    return pd.DataFrame(results)


def prepare_aligned_data(
    features_df: pd.DataFrame,
    targets: pd.Series,
    feature_cols: List[str],
) -> Tuple[pd.DataFrame, pd.Series]:
    """Align features and targets, removing NaN."""
    combined_data = features_df[feature_cols].copy()
    combined_data["_target"] = targets.values
    combined_data = combined_data.dropna()
    
    aligned_features = combined_data[feature_cols]
    aligned_targets = combined_data["_target"]
    
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
    log_freqs_array = np.asarray(log_freqs)
    log_psd_array = np.asarray(log_psd)
    offsets_array = np.asarray(offsets)
    slopes_array = np.asarray(slopes)

    n_frequencies_psd = log_psd_array.shape[-1]
    n_frequencies_freqs = log_freqs_array.shape[-1]
    
    if n_frequencies_psd != n_frequencies_freqs:
        raise ValueError(
            f"log_psd last dimension ({n_frequencies_psd}) does not match "
            f"log_freqs length ({n_frequencies_freqs})."
        )

    predicted_psd = offsets_array[..., None] + slopes_array[..., None] * log_freqs_array
    residuals = log_psd_array - predicted_psd
    return residuals


def _compute_percentage_bend_correlation(
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    config: Optional[Any] = None,
) -> Tuple[float, float]:
    """Compute percentage bend correlation for valid data."""
    epsilon = get_epsilon_std(config)
    n_samples = len(x_valid)
    
    median_x = np.median(x_valid)
    median_y = np.median(y_valid)
    
    deviations_x = np.abs(x_valid - median_x)
    deviations_y = np.abs(y_valid - median_y)
    
    sorted_deviations_x = np.sort(deviations_x)
    sorted_deviations_y = np.sort(deviations_y)
    
    beta_index = int(np.floor(_PERCENTAGE_BEND_BETA * n_samples))
    omega_x = sorted_deviations_x[beta_index] if beta_index < n_samples else sorted_deviations_x[-1]
    omega_y = sorted_deviations_y[beta_index] if beta_index < n_samples else sorted_deviations_y[-1]
    
    if omega_x < epsilon or omega_y < epsilon:
        return compute_correlation(x_valid, y_valid, "spearman")
    
    psi_x = np.clip((x_valid - median_x) / omega_x, -1, 1)
    psi_y = np.clip((y_valid - median_y) / omega_y, -1, 1)
    
    numerator = np.sum(psi_x * psi_y)
    denominator = np.sqrt(np.sum(psi_x**2) * np.sum(psi_y**2))
    correlation = numerator / denominator
    
    degrees_of_freedom = n_samples - 2
    t_statistic = correlation * np.sqrt(degrees_of_freedom / (1 - correlation**2 + epsilon))
    p_value = 2 * (1 - stats.t.cdf(np.abs(t_statistic), df=degrees_of_freedom))
    
    return float(correlation), float(p_value)


def compute_robust_correlation(
    x: np.ndarray,
    y: np.ndarray,
    method: str = "percentage_bend",
    config: Optional[Any] = None,
) -> Tuple[float, float]:
    """Compute robust correlation (percentage bend or biweight midcorrelation)."""
    valid_mask = np.isfinite(x) & np.isfinite(y)
    n_valid = np.sum(valid_mask)
    
    if n_valid < _MIN_SAMPLES_ROBUST_CORRELATION:
        return np.nan, np.nan
    
    x_valid = x[valid_mask]
    y_valid = y[valid_mask]
    
    if method == "percentage_bend":
        return _compute_percentage_bend_correlation(x_valid, y_valid, config)
    else:
        return compute_correlation(x_valid, y_valid, "spearman")


def _get_cluster_structure(config: Optional[Any] = None) -> np.ndarray:
    """Get cluster structure for 2D labeling."""
    constants = get_statistics_constants(config)
    structure = constants.get("cluster_structure_2d", None)
    
    if structure is None:
        structure = np.ones((_DEFAULT_CLUSTER_STRUCTURE_SIZE, _DEFAULT_CLUSTER_STRUCTURE_SIZE))
    
    return structure


def _compute_observed_cluster_masses(
    labeled_clusters: np.ndarray,
    n_clusters: int,
    t_values: np.ndarray,
) -> List[Tuple[int, float, int]]:
    """Compute observed cluster masses from labeled clusters."""
    cluster_masses = []
    for cluster_id in range(1, n_clusters + 1):
        cluster_mask = labeled_clusters == cluster_id
        cluster_mass = np.abs(t_values[cluster_mask]).sum()
        cluster_size = np.sum(cluster_mask)
        cluster_masses.append((cluster_id, cluster_mass, cluster_size))
    return cluster_masses


def _compute_permutation_max_masses(
    t_values: np.ndarray,
    alpha: float,
    n_permutations: int,
    cluster_structure: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Compute maximum cluster masses from permutation tests."""
    permutation_max_masses = []
    
    for _ in range(n_permutations):
        sign_flips = rng.choice([-1, 1], size=t_values.shape)
        permuted_t_values = t_values * sign_flips
        permuted_p_values = 2 * (1 - stats.norm.cdf(np.abs(permuted_t_values)))
        
        threshold_mask = permuted_p_values < alpha
        permuted_labeled, n_permuted_clusters = label(threshold_mask, structure=cluster_structure)
        
        if n_permuted_clusters > 0:
            max_cluster_mass = max(
                np.abs(permuted_t_values[permuted_labeled == cluster_id]).sum()
                for cluster_id in range(1, n_permuted_clusters + 1)
            )
            permutation_max_masses.append(max_cluster_mass)
        else:
            permutation_max_masses.append(0.0)
    
    return np.array(permutation_max_masses)


def _determine_significant_clusters(
    labeled_clusters: np.ndarray,
    observed_masses: List[Tuple[int, float, int]],
    permutation_max_masses: np.ndarray,
    n_permutations: int,
    alpha: float,
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """Determine which clusters are significant based on permutation test."""
    significant_mask = np.zeros_like(labeled_clusters, dtype=bool)
    cluster_info = []
    
    for cluster_id, cluster_mass, cluster_size in observed_masses:
        n_exceeding = np.sum(permutation_max_masses >= cluster_mass)
        cluster_p_value = (n_exceeding + 1) / (n_permutations + 1)
        
        cluster_info.append({
            "cluster_id": cluster_id,
            "mass": float(cluster_mass),
            "size": int(cluster_size),
            "p_value": float(cluster_p_value),
        })
        
        if cluster_p_value <= alpha:
            significant_mask |= (labeled_clusters == cluster_id)
    
    return significant_mask, cluster_info


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
    
    cluster_structure = _get_cluster_structure(config)
    threshold_mask = p_values < alpha
    
    labeled_clusters, n_clusters = label(threshold_mask, structure=cluster_structure)
    
    if n_clusters == 0:
        return np.zeros_like(p_values, dtype=bool), []
    
    observed_masses = _compute_observed_cluster_masses(labeled_clusters, n_clusters, t_values)
    permutation_max_masses = _compute_permutation_max_masses(
        t_values, alpha, n_permutations, cluster_structure, rng
    )
    
    significant_mask, cluster_info = _determine_significant_clusters(
        labeled_clusters, observed_masses, permutation_max_masses, n_permutations, alpha
    )
    
    return significant_mask, cluster_info


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
    
    valid_mask = np.isfinite(x) & np.isfinite(y)
    n_valid = np.sum(valid_mask)
    
    if n_valid < _MIN_SAMPLES_BOOTSTRAP:
        return np.nan, np.nan
    
    x_valid = x[valid_mask]
    y_valid = y[valid_mask]
    
    bootstrap_correlations = []
    for _ in range(n_bootstrap):
        bootstrap_indices = rng.integers(0, n_valid, size=n_valid)
        correlation, _ = compute_correlation(
            x_valid[bootstrap_indices], y_valid[bootstrap_indices], method
        )
        if np.isfinite(correlation):
            bootstrap_correlations.append(correlation)
    
    if len(bootstrap_correlations) < _MIN_BOOTSTRAP_RESULTS:
        return np.nan, np.nan
    
    alpha_level = 1 - ci_level
    lower_percentile = 100 * alpha_level / 2
    upper_percentile = 100 * (1 - alpha_level / 2)
    
    ci_lower = np.percentile(bootstrap_correlations, lower_percentile)
    ci_upper = np.percentile(bootstrap_correlations, upper_percentile)
    
    return float(ci_lower), float(ci_upper)
