"""
Data Transforms and Utilities
=============================

Consolidated module containing:
- Data transformation (centering, z-scoring, pooling)
- Regression utilities (linear residuals, binned statistics)
- Aperiodic fitting (1/f component extraction, residual computation)
- Coupling statistics (inter-band, group power)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


###################################################################
# Data Transformation
###################################################################


def zscore_array(arr: np.ndarray) -> np.ndarray:
    """Z-score normalize numpy array.
    
    Standardizes array to zero mean and unit variance.
    
    Parameters
    ----------
    arr : np.ndarray
        Input array
        
    Returns
    -------
    np.ndarray
        Z-scored array
    """
    mean = np.nanmean(arr)
    std_val = np.nanstd(arr, ddof=1)
    if std_val <= 0:
        return np.full_like(arr, np.nan)
    return (arr - mean) / std_val


def prepare_data_for_plotting(
    x_data: pd.Series,
    y_data: pd.Series,
) -> Tuple[pd.Series, pd.Series, int]:
    """Prepare data for plotting by removing NaN values.
    
    Parameters
    ----------
    x_data : pd.Series
        First input series
    y_data : pd.Series
        Second input series
        
    Returns
    -------
    Tuple[pd.Series, pd.Series, int]
        (x_clean, y_clean, n_valid)
    """
    mask = x_data.notna() & y_data.notna()
    return x_data[mask], y_data[mask], int(mask.sum())


###################################################################
# Regression Utilities
###################################################################


def fit_linear_regression(
    x: np.ndarray,
    y: np.ndarray,
    x_range: np.ndarray,
    min_samples: int = 3,
) -> np.ndarray:
    """Fit linear regression and return predictions over specified range.
    
    Parameters
    ----------
    x : np.ndarray
        Predictor values
    y : np.ndarray
        Outcome values
    x_range : np.ndarray
        Range of x values for prediction
    min_samples : int
        Minimum samples required for fitting
        
    Returns
    -------
    np.ndarray
        Predicted y values (NaN-filled if insufficient samples)
    """
    if len(x) < min_samples:
        return np.full_like(x_range, np.nan)
    coefficients = np.polyfit(x, y, 1)
    polynomial = np.poly1d(coefficients)
    return polynomial(x_range)


def _create_bin_mask(
    y_pred: np.ndarray,
    bin_edges: np.ndarray,
    bin_index: int,
    is_last_bin: bool,
) -> np.ndarray:
    """Create boolean mask for values in specified bin.
    
    Parameters
    ----------
    y_pred : np.ndarray
        Predicted values
    bin_edges : np.ndarray
        Bin edge values
    bin_index : int
        Index of current bin
    is_last_bin : bool
        Whether this is the last bin (inclusive upper bound)
        
    Returns
    -------
    np.ndarray
        Boolean mask for values in bin
    """
    lower_bound = bin_edges[bin_index]
    upper_bound = bin_edges[bin_index + 1]
    
    if is_last_bin:
        return (y_pred >= lower_bound) & (y_pred <= upper_bound)
    return (y_pred >= lower_bound) & (y_pred < upper_bound)


def compute_binned_statistics(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    n_bins: int,
) -> Tuple[List[float], List[float], List[float]]:
    """Compute binned means and standard errors for calibration plots.
    
    Parameters
    ----------
    y_pred : np.ndarray
        Predicted values (used for binning)
    y_true : np.ndarray
        True values (used for statistics)
    n_bins : int
        Number of bins
        
    Returns
    -------
    Tuple[List[float], List[float], List[float]]
        (bin_centers, bin_means, bin_stds)
    """
    y_min = y_pred.min()
    y_max = y_pred.max()
    bin_edges = np.linspace(y_min, y_max, n_bins + 1)
    
    bin_centers, bin_means, bin_stds = [], [], []
    
    for bin_idx in range(n_bins):
        is_last_bin = bin_idx == n_bins - 1
        bin_mask = _create_bin_mask(y_pred, bin_edges, bin_idx, is_last_bin)
        n_samples_in_bin = bin_mask.sum()
        
        if n_samples_in_bin > 0:
            bin_center = (bin_edges[bin_idx] + bin_edges[bin_idx + 1]) / 2
            y_true_in_bin = y_true[bin_mask]
            bin_mean = np.mean(y_true_in_bin)
            bin_std = np.std(y_true_in_bin) / np.sqrt(n_samples_in_bin)
            
            bin_centers.append(bin_center)
            bin_means.append(bin_mean)
            bin_stds.append(bin_std)
    
    return bin_centers, bin_means, bin_stds


def compute_residuals(
    log_freqs: np.ndarray,
    log_psd: np.ndarray,
    offsets: np.ndarray,
    slopes: np.ndarray,
) -> np.ndarray:
    """Compute residuals from an aperiodic fit.

    Supports vector (freq,) inputs as well as epoch/channel grids such as
    (epochs, channels, freqs) by broadcasting the offsets/slopes over the
    frequency axis.
    
    Parameters
    ----------
    log_freqs : np.ndarray
        Log-transformed frequencies
    log_psd : np.ndarray
        Log-transformed PSD values
    offsets : np.ndarray
        Aperiodic offset values
    slopes : np.ndarray
        Aperiodic slope values
        
    Returns
    -------
    np.ndarray
        Residuals after subtracting aperiodic component
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


###################################################################
# Feature Transformation
###################################################################


def compute_change_features(
    features_df: pd.DataFrame,
    window_pairs: Optional[List[Tuple[str, str]]] = None,
    transform: str = "difference",
    config: Optional[Any] = None,
) -> pd.DataFrame:
    """Compute change scores between matching feature pairs across time windows.
    
    Parameters
    ----------
    features_df : pd.DataFrame
        DataFrame with feature columns containing window name segments
    window_pairs : Optional[List[Tuple[str, str]]]
        List of (reference_window, target_window) pairs. If None, uses config or
        defaults to [("baseline", "active")].
    transform : str
        Transform type: "difference" (target - ref), "percent" ((target - ref) / ref * 100),
        or "log_ratio" (log10(target / ref)).
    config : Optional[Any]
        Config object to read window_pairs and transform from if not provided.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with change columns for each window pair
    """
    # Get window pairs from config if not provided
    if window_pairs is None and config is not None:
        cfg_pairs = config.get("feature_engineering.change_scores.window_pairs", None)
        if cfg_pairs and isinstance(cfg_pairs, list):
            window_pairs = [(str(p[0]), str(p[1])) for p in cfg_pairs if len(p) >= 2]
        cfg_transform = config.get("feature_engineering.change_scores.transform", None)
        if cfg_transform:
            transform = str(cfg_transform).strip().lower()
    
    # Default to baseline/active if nothing specified
    if not window_pairs:
        window_pairs = [("baseline", "active")]
    
    transform = transform.lower()
    if transform not in {"difference", "percent", "log_ratio"}:
        transform = "difference"
    
    from eeg_pipeline.domain.features.naming import NamingSchema

    suffix = "change" if transform == "difference" else "pct_change" if transform == "percent" else "log_ratio"

    # Build a lookup of parsed feature columns so pairing never relies on substring replacement.
    # Key ignores segment so we can match baseline vs active cleanly.
    by_segment: Dict[Tuple[str, str, str, str, str, str], str] = {}
    for col in features_df.columns:
        col_str = str(col)
        parsed = NamingSchema.parse(col_str)
        if not parsed.get("valid"):
            continue
        key = (
            str(parsed.get("group") or "").strip().lower(),
            str(parsed.get("segment") or "").strip().lower(),
            str(parsed.get("band") or "").strip().lower(),
            str(parsed.get("scope") or "").strip().lower(),
            str(parsed.get("identifier") or "").strip(),
            str(parsed.get("stat") or "").strip().lower(),
        )
        by_segment[key] = col_str

    change_data: Dict[str, np.ndarray] = {}
    for ref_window, target_window in window_pairs:
        ref_seg = str(ref_window).strip().lower()
        tgt_seg = str(target_window).strip().lower()

        ref_keys = [k for k in by_segment.keys() if k[1] == ref_seg]
        for group, _seg, band, scope, identifier, stat in ref_keys:
            ref_col = by_segment[(group, ref_seg, band, scope, identifier, stat)]
            target_col = by_segment.get((group, tgt_seg, band, scope, identifier, stat))
            if target_col is None:
                continue

            ref_vals = pd.to_numeric(features_df[ref_col], errors="coerce").to_numpy(dtype=float)
            target_vals = pd.to_numeric(features_df[target_col], errors="coerce").to_numpy(dtype=float)

            if transform == "difference":
                change_vals = target_vals - ref_vals
            elif transform == "percent":
                with np.errstate(divide="ignore", invalid="ignore"):
                    change_vals = np.where(
                        np.abs(ref_vals) > 1e-10,
                        (target_vals - ref_vals) / ref_vals * 100,
                        np.nan,
                    )
            else:  # log_ratio
                with np.errstate(divide="ignore", invalid="ignore"):
                    change_vals = np.where(
                        (ref_vals > 0) & (target_vals > 0),
                        np.log10(target_vals / ref_vals),
                        np.nan,
                    )

            build_kwargs: Dict[str, Any] = {}
            if scope in {"ch", "roi"}:
                build_kwargs["channel"] = identifier
            elif scope == "chpair":
                build_kwargs["channel_pair"] = identifier

            change_col = NamingSchema.build(group, suffix, band, scope, stat, **build_kwargs)
            change_data[change_col] = change_vals
    
    if not change_data:
        return pd.DataFrame(index=features_df.index)
    
    return pd.DataFrame(change_data, index=features_df.index)


__all__ = [
    # Transform
    "zscore_array",
    "prepare_data_for_plotting",
    # Feature Transformation
    "compute_change_features",
    # Regression
    "fit_linear_regression",
    "compute_binned_statistics",
    # Aperiodic
    "compute_residuals",
]
