"""
Cross-Validation Hygiene Utilities
===================================

Functions to ensure scientific validity in cross-validation workflows by
computing fold-specific parameters that would otherwise cause data leakage.

Key principle: Any unsupervised "fit" step (even if label-free) should be done
within training folds only, otherwise you get optimistic bias.

Parameters that should be computed fold-specifically:
- IAF (Individual Alpha Frequency) band overrides
- Global/broadcast features (e.g., global ITPC)
- Feature scaling (mean/std)
- Feature selection thresholds
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np


@dataclass
class FoldSpecificParams:
    """Container for fold-specific parameters computed on training data only."""
    
    fold_idx: int
    train_indices: np.ndarray
    test_indices: np.ndarray
    
    iaf_hz: Optional[float] = None
    frequency_bands: Optional[Dict[str, Tuple[float, float]]] = None
    
    feature_means: Optional[Dict[str, float]] = None
    feature_stds: Optional[Dict[str, float]] = None
    
    global_itpc: Optional[Dict[str, np.ndarray]] = None
    
    metadata: Dict[str, Any] = field(default_factory=dict)


def compute_iaf_for_fold(
    epochs_data: np.ndarray,
    sfreq: float,
    train_mask: np.ndarray,
    config: Any,
    logger: Any = None,
) -> Tuple[Optional[float], Optional[Dict[str, Tuple[float, float]]]]:
    """
    Compute Individual Alpha Frequency (IAF) using ONLY training fold trials.
    
    This prevents leakage from test trials into band definition.
    
    Parameters
    ----------
    epochs_data : np.ndarray
        Full epochs data (n_epochs, n_channels, n_times)
    sfreq : float
        Sampling frequency
    train_mask : np.ndarray
        Boolean mask indicating training trials
    config : Any
        Configuration object
    logger : Any
        Logger instance
        
    Returns
    -------
    iaf_hz : float or None
        Estimated IAF in Hz (None if estimation failed)
    frequency_bands : dict or None
        IAF-adjusted frequency band definitions
    """
    import mne
    from scipy.signal import find_peaks
    
    if not np.any(train_mask):
        if logger:
            logger.warning("CV hygiene: No training trials for IAF estimation")
        return None, None
    
    train_data = epochs_data[train_mask]
    if train_data.shape[0] < 5:
        if logger:
            logger.warning("CV hygiene: Too few training trials (%d) for reliable IAF estimation", train_data.shape[0])
        return None, None
    
    iaf_cfg = config.get("feature_engineering.bands", {}) if hasattr(config, "get") else {}
    alpha_range = iaf_cfg.get("iaf_search_range_hz", [7.0, 13.0])
    alpha_fmin, alpha_fmax = float(alpha_range[0]), float(alpha_range[1])
    prom = float(iaf_cfg.get("iaf_min_prominence", 0.05))
    
    try:
        psds, freqs = mne.time_frequency.psd_array_multitaper(
            train_data,
            sfreq=sfreq,
            fmin=max(1.0, alpha_fmin - 4.0),
            fmax=min(40.0, sfreq / 2.0 - 0.5),
            adaptive=True,
            normalization="full",
            verbose=False,
        )
    except Exception as exc:
        if logger:
            logger.warning("CV hygiene: PSD computation failed for IAF: %s", exc)
        return None, None
    
    psds = np.asarray(psds, dtype=float)
    freqs = np.asarray(freqs, dtype=float)
    
    if psds.ndim != 3 or freqs.size == 0:
        return None, None
    
    mean_psd = np.nanmean(psds, axis=(0, 1))
    
    log_f = np.log10(np.maximum(freqs, 1e-6))
    log_p = np.log10(np.maximum(mean_psd, 1e-20))
    
    fit_mask = (freqs >= 2.0) & (freqs <= 40.0) & np.isfinite(log_p)
    if np.sum(fit_mask) < 10:
        return None, None
    
    try:
        slope, intercept = np.polyfit(log_f[fit_mask], log_p[fit_mask], 1)
        resid = log_p - (intercept + slope * log_f)
    except (np.linalg.LinAlgError, ValueError):
        resid = log_p
    
    a_mask = (freqs >= alpha_fmin) & (freqs <= alpha_fmax) & np.isfinite(resid)
    if not np.any(a_mask):
        return None, None
    
    y = resid[a_mask]
    peaks, props = find_peaks(y, prominence=prom)
    
    iaf = np.nan
    if peaks.size:
        best = int(peaks[np.argmax(props.get("prominences", np.ones_like(peaks)))])
        iaf = float(freqs[a_mask][best])
    else:
        y_pos = np.maximum(y, 0.0)
        denom = float(np.sum(y_pos))
        if denom > 0:
            iaf = float(np.sum(freqs[a_mask] * y_pos) / denom)
    
    if not np.isfinite(iaf):
        return None, None
    
    width = float(iaf_cfg.get("alpha_width_hz", 2.0))
    alpha_min = max(6.0, iaf - width)
    alpha_max = min(14.0, iaf + width)
    
    from eeg_pipeline.utils.config.loader import get_frequency_bands
    base_bands = dict(get_frequency_bands(config))
    
    freq_bands = {
        "delta": base_bands.get("delta", [1.0, 3.9]),
        "theta": [max(3.0, iaf - 6.0), max(4.0, alpha_min)],
        "alpha": [alpha_min, alpha_max],
        "beta": [max(13.0, alpha_max), base_bands.get("beta", [13.0, 30.0])[1]],
        "gamma": base_bands.get("gamma", [30.1, 80.0]),
    }
    
    if logger:
        logger.info("CV hygiene: Estimated fold-specific IAF=%.2f Hz from %d training trials", iaf, train_data.shape[0])
    
    return iaf, freq_bands


def compute_global_itpc_for_fold(
    complex_tfr_data: np.ndarray,
    train_mask: np.ndarray,
    logger: Any = None,
) -> np.ndarray:
    """
    Compute global ITPC using ONLY training fold trials.
    
    ITPC is inherently a cross-trial measure, so it must be computed
    within folds to avoid leakage.
    
    Parameters
    ----------
    complex_tfr_data : np.ndarray
        Complex TFR data (n_epochs, n_channels, n_freqs, n_times)
    train_mask : np.ndarray
        Boolean mask indicating training trials
    logger : Any
        Logger instance
        
    Returns
    -------
    itpc_map : np.ndarray
        ITPC values (n_channels, n_freqs, n_times) computed on training trials only
    """
    if not np.any(train_mask):
        if logger:
            logger.warning("CV hygiene: No training trials for ITPC computation")
        return np.full(complex_tfr_data.shape[1:], np.nan)
    
    train_data = complex_tfr_data[train_mask]
    
    eps = 1e-12
    unit = train_data / (np.abs(train_data) + eps)
    itpc_map = np.abs(np.mean(unit, axis=0))
    
    if logger:
        logger.debug("CV hygiene: Computed fold-specific ITPC from %d training trials", train_data.shape[0])
    
    return itpc_map


def compute_feature_scaling_for_fold(
    features_df: "pd.DataFrame",
    train_mask: np.ndarray,
    logger: Any = None,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Compute feature scaling parameters (mean, std) using ONLY training fold trials.
    
    Parameters
    ----------
    features_df : pd.DataFrame
        Feature DataFrame (n_trials x n_features)
    train_mask : np.ndarray
        Boolean mask indicating training trials
    logger : Any
        Logger instance
        
    Returns
    -------
    means : dict
        Mean values per feature (computed on training data)
    stds : dict
        Standard deviation per feature (computed on training data)
    """
    import pandas as pd
    
    if not np.any(train_mask):
        if logger:
            logger.warning("CV hygiene: No training trials for feature scaling")
        return {}, {}
    
    train_df = features_df.iloc[train_mask]
    
    means = {}
    stds = {}
    
    for col in train_df.columns:
        series = pd.to_numeric(train_df[col], errors="coerce")
        means[col] = float(series.mean())
        stds[col] = float(series.std())
        if stds[col] == 0 or not np.isfinite(stds[col]):
            stds[col] = 1.0
    
    if logger:
        logger.debug("CV hygiene: Computed fold-specific scaling from %d training trials", int(np.sum(train_mask)))
    
    return means, stds


def apply_fold_specific_scaling(
    features_df: "pd.DataFrame",
    means: Dict[str, float],
    stds: Dict[str, float],
) -> "pd.DataFrame":
    """
    Apply fold-specific scaling parameters to a feature DataFrame.
    
    Parameters
    ----------
    features_df : pd.DataFrame
        Feature DataFrame to scale
    means : dict
        Mean values per feature
    stds : dict
        Standard deviation per feature
        
    Returns
    -------
    scaled_df : pd.DataFrame
        Scaled feature DataFrame
    """
    import pandas as pd
    
    scaled = features_df.copy()
    
    for col in scaled.columns:
        if col in means and col in stds:
            series = pd.to_numeric(scaled[col], errors="coerce")
            scaled[col] = (series - means[col]) / stds[col]
    
    return scaled


def create_fold_specific_context(
    epochs,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    fold_idx: int,
    config: Any,
    logger: Any = None,
    compute_iaf: bool = True,
) -> FoldSpecificParams:
    """
    Create a FoldSpecificParams object with all fold-specific parameters.
    
    This is the main entry point for CV hygiene - call this at the start
    of each fold to get properly computed parameters.
    
    Parameters
    ----------
    epochs : mne.Epochs
        Full epochs object
    train_indices : np.ndarray
        Indices of training trials
    test_indices : np.ndarray
        Indices of test trials
    fold_idx : int
        Fold index
    config : Any
        Configuration object
    logger : Any
        Logger instance
    compute_iaf : bool
        Whether to compute fold-specific IAF
        
    Returns
    -------
    FoldSpecificParams
        Container with all fold-specific parameters
    """
    n_epochs = len(epochs)
    train_mask = np.zeros(n_epochs, dtype=bool)
    train_mask[train_indices] = True
    
    params = FoldSpecificParams(
        fold_idx=fold_idx,
        train_indices=train_indices,
        test_indices=test_indices,
    )
    
    use_iaf = bool(config.get("feature_engineering.bands.use_iaf", False)) if hasattr(config, "get") else False
    
    if compute_iaf and use_iaf:
        import mne
        picks = mne.pick_types(epochs.info, eeg=True, meg=False, eog=False, stim=False, exclude="bads")
        data = epochs.get_data(picks=picks)
        sfreq = float(epochs.info["sfreq"])
        
        iaf, freq_bands = compute_iaf_for_fold(data, sfreq, train_mask, config, logger)
        params.iaf_hz = iaf
        params.frequency_bands = freq_bands
    
    params.metadata["n_train"] = int(np.sum(train_mask))
    params.metadata["n_test"] = len(test_indices)
    
    if logger:
        logger.info(
            "CV hygiene: Created fold %d context (train=%d, test=%d, IAF=%.2f Hz)",
            fold_idx,
            params.metadata["n_train"],
            params.metadata["n_test"],
            params.iaf_hz if params.iaf_hz is not None else np.nan,
        )
    
    return params


__all__ = [
    "FoldSpecificParams",
    "compute_iaf_for_fold",
    "compute_global_itpc_for_fold",
    "compute_feature_scaling_for_fold",
    "apply_fold_specific_scaling",
    "create_fold_specific_context",
]
