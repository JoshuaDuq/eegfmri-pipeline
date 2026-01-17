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
from scipy import stats

from eeg_pipeline.utils.analysis.stats._regression_utils import _ols_fit


###################################################################
# Data Transformation
###################################################################


def center_series(series: pd.Series) -> pd.Series:
    """Center series by subtracting mean.
    
    Parameters
    ----------
    series : pd.Series
        Input series
        
    Returns
    -------
    pd.Series
        Centered series (zero mean)
    """
    return series - series.mean()


def zscore_array(x: np.ndarray) -> np.ndarray:
    """Z-score normalize numpy array.
    
    Standardizes array to zero mean and unit variance.
    Returns NaN-filled array if variance is zero or invalid.
    
    Parameters
    ----------
    x : np.ndarray
        Input array
        
    Returns
    -------
    np.ndarray
        Z-scored array (NaN if variance is zero or invalid)
    """
    x = np.asarray(x, dtype=float)
    mu = np.nanmean(x)
    sd = np.nanstd(x, ddof=1)
    if not np.isfinite(sd) or sd <= 0:
        return np.full_like(x, np.nan)
    return (x - mu) / sd


def zscore_series(series: pd.Series) -> pd.Series:
    """Z-score normalize pandas Series.
    
    Standardizes series to zero mean and unit variance.
    Returns empty series if variance is zero or invalid.
    
    Parameters
    ----------
    series : pd.Series
        Input series
        
    Returns
    -------
    pd.Series
        Z-scored series (empty if variance is zero or invalid)
    """
    mean = series.mean()
    std_val = series.std(ddof=1)
    if std_val <= 0:
        return pd.Series(dtype=float)
    return (series - mean) / std_val


def apply_pooling_strategy(
    x: pd.Series,
    y: pd.Series,
    pooling_strategy: str,
) -> Tuple[pd.Series, pd.Series]:
    """Apply pooling strategy for correlation analysis.
    
    Parameters
    ----------
    x : pd.Series
        First variable
    y : pd.Series
        Second variable
    pooling_strategy : str
        Strategy: "within_subject_centered", "within_subject_zscored", or "none"
        
    Returns
    -------
    Tuple[pd.Series, pd.Series]
        Transformed (x, y) series
    """
    if pooling_strategy == "within_subject_centered":
        return center_series(x), center_series(y)
    if pooling_strategy == "within_subject_zscored":
        return zscore_series(x), zscore_series(y)
    return x, y


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


def prepare_data_without_validation(
    x_data: pd.Series,
    y_data: pd.Series,
) -> Tuple[pd.Series, pd.Series, int]:
    """Return data without NaN filtering.
    
    Used when data has already been validated (e.g., partial residuals
    where NaNs were handled during regression).
    
    Parameters
    ----------
    x_data : pd.Series
        First input series
    y_data : pd.Series
        Second input series
        
    Returns
    -------
    Tuple[pd.Series, pd.Series, int]
        (x_data, y_data, length)
    """
    return x_data, y_data, len(x_data)


def _align_and_validate_series(
    x_array: np.ndarray,
    y_array: np.ndarray,
) -> Tuple[pd.Series, pd.Series]:
    """Align arrays to same length and remove NaNs.
    
    Parameters
    ----------
    x_array : np.ndarray
        First input array
    y_array : np.ndarray
        Second input array
        
    Returns
    -------
    Tuple[pd.Series, pd.Series]
        Aligned and validated series
    """
    x_series = pd.Series(np.asarray(x_array))
    y_series = pd.Series(np.asarray(y_array))
    
    min_length = min(len(x_series), len(y_series))
    x_series = x_series.iloc[:min_length]
    y_series = y_series.iloc[:min_length]
    
    valid_mask = x_series.notna() & y_series.notna()
    return x_series[valid_mask], y_series[valid_mask]


def _process_subject_data(
    x_array: np.ndarray,
    y_array: np.ndarray,
    subject_id: str,
    pooling_strategy: str,
) -> Tuple[pd.Series, pd.Series, List[str]]:
    """Process single subject's data and return normalized series with IDs.
    
    Parameters
    ----------
    x_array : np.ndarray
        First variable array
    y_array : np.ndarray
        Second variable array
    subject_id : str
        Subject identifier
    pooling_strategy : str
        Pooling strategy to apply
        
    Returns
    -------
    Tuple[pd.Series, pd.Series, List[str]]
        (x_normalized, y_normalized, subject_ids)
    """
    x_series, y_series = _align_and_validate_series(x_array, y_array)
    
    if x_series.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float), []
    
    x_normalized, y_normalized = apply_pooling_strategy(
        x_series, y_series, pooling_strategy
    )
    
    if x_normalized.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float), []
    
    n_samples = len(x_normalized)
    subject_ids = [subject_id] * n_samples
    x_reset = x_normalized.reset_index(drop=True)
    y_reset = y_normalized.reset_index(drop=True)
    
    return x_reset, y_reset, subject_ids


def prepare_group_data(
    x_lists: List[np.ndarray],
    y_lists: List[np.ndarray],
    subj_order: List[str],
    pooling_strategy: str,
) -> Tuple[pd.Series, pd.Series, List[str]]:
    """Prepare group data for correlation analysis.
    
    Parameters
    ----------
    x_lists : List[np.ndarray]
        List of first variable arrays (one per subject)
    y_lists : List[np.ndarray]
        List of second variable arrays (one per subject)
    subj_order : List[str]
        Subject identifiers in order
    pooling_strategy : str
        Pooling strategy to apply per subject
        
    Returns
    -------
    Tuple[pd.Series, pd.Series, List[str]]
        (x_concatenated, y_concatenated, subject_ids)
    """
    x_series_list, y_series_list, subject_ids = [], [], []

    for idx, (x_array, y_array) in enumerate(zip(x_lists, y_lists)):
        subject_id = (
            subj_order[idx] if idx < len(subj_order) else str(idx)
        )
        x_normalized, y_normalized, ids = _process_subject_data(
            x_array, y_array, subject_id, pooling_strategy
        )
        
        if x_normalized.empty:
            continue
        
        x_series_list.append(x_normalized)
        y_series_list.append(y_normalized)
        subject_ids.extend(ids)

    if not x_series_list:
        return pd.Series(dtype=float), pd.Series(dtype=float), []
    
    x_concatenated = pd.concat(x_series_list, ignore_index=True)
    y_concatenated = pd.concat(y_series_list, ignore_index=True)
    
    return x_concatenated, y_concatenated, subject_ids


###################################################################
# Regression Utilities
###################################################################


def compute_linear_residuals(
    x_data: pd.Series,
    y_data: pd.Series,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute linear regression residuals for plotting/visualization.
    
    Parameters
    ----------
    x_data : pd.Series
        Predictor variable
    y_data : pd.Series
        Outcome variable
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        (fitted_values, residuals, x_valid)
    """
    x_numeric = pd.to_numeric(x_data, errors="coerce")
    y_numeric = pd.to_numeric(y_data, errors="coerce")
    valid_mask = x_numeric.notna() & y_numeric.notna()
    
    x_valid = x_numeric[valid_mask].to_numpy(dtype=float)
    y_valid = y_numeric[valid_mask].to_numpy(dtype=float)
    
    if len(x_valid) < 2:
        return np.array([]), np.array([]), np.array([])
    
    design_matrix = np.column_stack([np.ones(len(x_valid)), x_valid])
    beta = _ols_fit(design_matrix, y_valid)
    
    if beta is None:
        return np.array([]), np.array([]), np.array([])
    
    fitted_values = design_matrix @ beta
    residuals = y_valid - fitted_values
    
    return fitted_values, residuals, x_valid


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


###################################################################
# Aperiodic Fitting
###################################################################


def _reject_peaks(
    frequencies: np.ndarray,
    psd_values: np.ndarray,
    peak_rejection_z: float,
    min_points: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Reject spectral peaks using robust outlier detection.
    
    Parameters
    ----------
    frequencies : np.ndarray
        Frequency values
    psd_values : np.ndarray
        Power spectral density values
    peak_rejection_z : float
        Z-score threshold for peak rejection
    min_points : int
        Minimum points required after rejection
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (frequencies, psd_values) with peaks removed
    """
    mad = stats.median_abs_deviation(
        psd_values, scale="normal", nan_policy="omit"
    )
    median_psd = np.median(psd_values) if np.isfinite(psd_values).any() else np.nan
    
    mad_threshold = 1e-12
    is_mad_valid = np.isfinite(mad) and mad > mad_threshold
    is_median_valid = np.isfinite(median_psd)
    
    if not (is_mad_valid and is_median_valid):
        return frequencies, psd_values
    
    rejection_threshold = median_psd + peak_rejection_z * mad
    keep_mask = psd_values <= rejection_threshold
    
    if keep_mask.sum() >= min_points:
        return frequencies[keep_mask], psd_values[keep_mask]
    
    return frequencies, psd_values


def fit_aperiodic(
    log_freqs: np.ndarray,
    log_psd: np.ndarray,
    peak_rejection_z: float = 3.5,
    min_points: int = 5,
) -> Tuple[float, float]:
    """Fit aperiodic (1/f) component to log-log PSD.
    
    Parameters
    ----------
    log_freqs : np.ndarray
        Log-transformed frequencies
    log_psd : np.ndarray
        Log-transformed power spectral density
    peak_rejection_z : float
        Z-score threshold for peak rejection
    min_points : int
        Minimum points required for fitting
        
    Returns
    -------
    Tuple[float, float]
        (intercept, slope) or (np.nan, np.nan) if fitting fails
    """
    finite_mask = np.isfinite(log_freqs) & np.isfinite(log_psd)
    frequencies = log_freqs[finite_mask]
    psd_values = log_psd[finite_mask]
    
    if frequencies.size < min_points:
        return np.nan, np.nan
    
    frequencies, psd_values = _reject_peaks(
        frequencies, psd_values, peak_rejection_z, min_points
    )
    
    if frequencies.size < min_points:
        return np.nan, np.nan
    
    try:
        slope, intercept = np.polyfit(frequencies, psd_values, 1)
        return float(intercept), float(slope)
    except (ValueError, np.linalg.LinAlgError):
        return np.nan, np.nan


def fit_aperiodic_to_all_epochs(
    log_freqs: np.ndarray,
    log_psd: np.ndarray,
    peak_rejection_z: float = 3.5,
    min_points: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit aperiodic component to all epochs and channels.
    
    Parameters
    ----------
    log_freqs : np.ndarray
        Log-transformed frequencies (1D array)
    log_psd : np.ndarray
        Log-transformed PSD with shape (n_epochs, n_channels, n_freqs)
    peak_rejection_z : float
        Z-score threshold for peak rejection
    min_points : int
        Minimum points required for fitting
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (offsets, slopes) with shape (n_epochs, n_channels)
    """
    n_epochs, n_channels, _ = log_psd.shape
    offsets = np.full((n_epochs, n_channels), np.nan)
    slopes = np.full((n_epochs, n_channels), np.nan)
    
    for epoch_idx in range(n_epochs):
        for channel_idx in range(n_channels):
            intercept, slope = fit_aperiodic(
                log_freqs, log_psd[epoch_idx, channel_idx, :],
                peak_rejection_z=peak_rejection_z, min_points=min_points,
            )
            offsets[epoch_idx, channel_idx] = intercept
            slopes[epoch_idx, channel_idx] = slope
    
    return offsets, slopes


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


###################################################################
# Data Alignment Utilities
###################################################################


def prepare_aligned_data(
    x: pd.Series,
    y: pd.Series,
    Z: Optional[pd.DataFrame] = None,
) -> Tuple[pd.Series, pd.Series, Optional[pd.DataFrame], int, int]:
    """Align x, y, and covariates, removing NaN rows.
    
    General utility for aligning multiple series/dataframes and removing
    rows with missing values. Used by correlation and partial correlation
    functions.
    
    Parameters
    ----------
    x : pd.Series
        First input series
    y : pd.Series
        Second input series
    Z : Optional[pd.DataFrame]
        Optional covariates dataframe
        
    Returns
    -------
    Tuple[pd.Series, pd.Series, Optional[pd.DataFrame], int, int]
        (x_clean, y_clean, Z_clean, n_total, n_kept)
    """
    x_series = x if isinstance(x, pd.Series) else pd.Series(x)
    y_series = y if isinstance(y, pd.Series) else pd.Series(y)
    
    frames = [x_series.rename("__x__"), y_series.rename("__y__")]
    has_covariates = False
    
    if Z is not None:
        if isinstance(Z, pd.DataFrame):
            has_data = len(Z) > 0 and len(Z.columns) > 0
            if has_data:
                frames.append(Z)
                has_covariates = True
        else:
            try:
                Z_dataframe = pd.DataFrame(Z)
                has_data = len(Z_dataframe) > 0 and len(Z_dataframe.columns) > 0
                if has_data:
                    frames.append(Z_dataframe)
                    has_covariates = True
            except (ValueError, TypeError):
                pass

    combined_data = pd.concat(frames, axis=1)
    n_total = len(combined_data)
    clean_data = combined_data.dropna()
    n_kept = len(clean_data)

    if n_kept == 0:
        return pd.Series(dtype=float), pd.Series(dtype=float), None, n_total, n_kept

    x_name = x_series.name if x_series.name is not None else "x"
    y_name = y_series.name if y_series.name is not None else "y"
    
    x_clean = clean_data.pop("__x__").rename(x_name)
    y_clean = clean_data.pop("__y__").rename(y_name)
    Z_clean = clean_data if has_covariates else None

    return x_clean, y_clean, Z_clean, n_total, n_kept


__all__ = [
    # Transform
    "center_series",
    "zscore_series",
    "apply_pooling_strategy",
    "prepare_data_for_plotting",
    "prepare_data_without_validation",
    "prepare_group_data",
    "prepare_aligned_data",
    # Feature Transformation
    "compute_change_features",
    # Regression
    "compute_linear_residuals",
    "fit_linear_regression",
    "compute_binned_statistics",
    # Aperiodic
    "fit_aperiodic",
    "fit_aperiodic_to_all_epochs",
    "compute_residuals",
]
