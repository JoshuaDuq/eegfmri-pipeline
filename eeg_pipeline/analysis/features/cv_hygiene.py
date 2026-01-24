"""
Cross-Validation Hygiene Utilities
===================================

Functions to ensure scientific validity in cross-validation workflows by
computing fold-specific parameters that would otherwise cause data leakage.

Key principle: Any unsupervised "fit" step (even if label-free) should be done
within training folds only, otherwise you get optimistic bias.

Parameters that should be computed fold-specifically:
- IAF (Individual Alpha Frequency) band overrides
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
from dataclasses import dataclass, field

import mne
import numpy as np
from scipy.signal import find_peaks


# Constants
MIN_TRIALS_FOR_IAF = 5
MIN_FREQ_POINTS_FOR_FIT = 10
MIN_FREQ_HZ = 1.0
MAX_FREQ_HZ = 40.0
PSD_FMIN_OFFSET_HZ = 4.0
PSD_FMAX_SAFETY_MARGIN_HZ = 0.5
FIT_FMIN_HZ = 2.0
FIT_FMAX_HZ = 40.0
EPSILON_FREQ = 1e-6
EPSILON_PSD = 1e-20
DEFAULT_ALPHA_MIN_HZ = 6.0
DEFAULT_ALPHA_MAX_HZ = 14.0
DEFAULT_ALPHA_WIDTH_HZ = 2.0
DEFAULT_ALPHA_RANGE_HZ = (7.0, 13.0)
DEFAULT_IAF_PROMINENCE = 0.05


@dataclass
class FoldSpecificParams:
    """Container for fold-specific parameters computed on training data only."""

    fold_idx: int
    train_indices: np.ndarray
    test_indices: np.ndarray

    iaf_hz: Optional[float] = None
    frequency_bands: Optional[Dict[str, Tuple[float, float]]] = None

    metadata: Dict[str, Any] = field(default_factory=dict)


def _log_warning(logger: Any, message: str, *args: Any) -> None:
    """Log warning if logger is available."""
    if logger:
        logger.warning(message, *args)


def _log_info(logger: Any, message: str, *args: Any) -> None:
    """Log info if logger is available."""
    if logger:
        logger.info(message, *args)


def _log_debug(logger: Any, message: str, *args: Any) -> None:
    """Log debug if logger is available."""
    if logger:
        logger.debug(message, *args)


def _extract_iaf_config(config: Any) -> Dict[str, Any]:
    """Extract IAF configuration from config object."""
    if hasattr(config, "get"):
        return config.get("feature_engineering.bands", {})
    return {}


def _validate_train_mask(train_mask: np.ndarray, min_trials: int = MIN_TRIALS_FOR_IAF) -> bool:
    """Validate that training mask has sufficient trials."""
    return np.any(train_mask) and np.sum(train_mask) >= min_trials


def _compute_power_spectral_density(
    train_data: np.ndarray,
    sfreq: float,
    alpha_fmin: float,
    logger: Any,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Compute power spectral density for IAF estimation."""
    fmin = max(MIN_FREQ_HZ, alpha_fmin - PSD_FMIN_OFFSET_HZ)
    fmax = min(MAX_FREQ_HZ, sfreq / 2.0 - PSD_FMAX_SAFETY_MARGIN_HZ)

    psds, freqs = mne.time_frequency.psd_array_multitaper(
        train_data,
        sfreq=sfreq,
        fmin=fmin,
        fmax=fmax,
        adaptive=True,
        normalization="full",
        verbose=False,
    )
    return np.asarray(psds, dtype=float), np.asarray(freqs, dtype=float)


def _compute_aperiodic_residual(
    freqs: np.ndarray,
    mean_psd: np.ndarray,
) -> Optional[np.ndarray]:
    """Compute aperiodic residual by fitting and subtracting 1/f slope."""
    log_freqs = np.log10(np.maximum(freqs, EPSILON_FREQ))
    log_power = np.log10(np.maximum(mean_psd, EPSILON_PSD))

    fit_mask = (freqs >= FIT_FMIN_HZ) & (freqs <= FIT_FMAX_HZ) & np.isfinite(log_power)
    if np.sum(fit_mask) < MIN_FREQ_POINTS_FOR_FIT:
        return None

    slope, intercept = np.polyfit(log_freqs[fit_mask], log_power[fit_mask], 1)
    residual = log_power - (intercept + slope * log_freqs)
    return residual


def _estimate_iaf_from_residual(
    freqs: np.ndarray,
    residual: np.ndarray,
    alpha_fmin: float,
    alpha_fmax: float,
    prominence: float,
) -> Optional[float]:
    """Estimate IAF from residual spectrum using peak detection or weighted mean."""
    alpha_mask = (freqs >= alpha_fmin) & (freqs <= alpha_fmax) & np.isfinite(residual)
    if not np.any(alpha_mask):
        return None

    residual_in_alpha = residual[alpha_mask]
    peaks, peak_properties = find_peaks(residual_in_alpha, prominence=prominence)

    if peaks.size > 0:
        prominences = peak_properties.get("prominences", np.ones_like(peaks))
        best_peak_idx = int(peaks[np.argmax(prominences)])
        iaf_hz = float(freqs[alpha_mask][best_peak_idx])
    else:
        positive_residual = np.maximum(residual_in_alpha, 0.0)
        residual_sum = float(np.sum(positive_residual))
        if residual_sum > 0:
            weighted_freqs = freqs[alpha_mask] * positive_residual
            iaf_hz = float(np.sum(weighted_freqs) / residual_sum)
        else:
            return None

    if np.isfinite(iaf_hz):
        return iaf_hz
    return None


def _build_frequency_bands_from_iaf(
    iaf_hz: float,
    config: Any,
    alpha_width_hz: float,
) -> Dict[str, Tuple[float, float]]:
    """Build frequency band definitions adjusted for individual alpha frequency."""
    from eeg_pipeline.utils.config.loader import get_frequency_bands

    base_bands = dict(get_frequency_bands(config))
    alpha_min = max(DEFAULT_ALPHA_MIN_HZ, iaf_hz - alpha_width_hz)
    alpha_max = min(DEFAULT_ALPHA_MAX_HZ, iaf_hz + alpha_width_hz)

    return {
        "delta": base_bands.get("delta", (1.0, 3.9)),
        "theta": (max(3.0, iaf_hz - 6.0), max(4.0, alpha_min)),
        "alpha": (alpha_min, alpha_max),
        "beta": (max(13.0, alpha_max), base_bands.get("beta", (13.0, 30.0))[1]),
        "gamma": base_bands.get("gamma", (30.1, 80.0)),
    }


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
    if not _validate_train_mask(train_mask):
        n_trials = int(np.sum(train_mask)) if np.any(train_mask) else 0
        _log_warning(
            logger,
            "CV hygiene: Too few training trials (%d) for reliable IAF estimation",
            n_trials,
        )
        return None, None

    train_data = epochs_data[train_mask]
    iaf_config = _extract_iaf_config(config)

    alpha_range = iaf_config.get("iaf_search_range_hz", list(DEFAULT_ALPHA_RANGE_HZ))
    alpha_fmin = float(alpha_range[0])
    alpha_fmax = float(alpha_range[1])
    prominence = float(iaf_config.get("iaf_min_prominence", DEFAULT_IAF_PROMINENCE))
    alpha_width_hz = float(iaf_config.get("alpha_width_hz", DEFAULT_ALPHA_WIDTH_HZ))

    psds, freqs = _compute_power_spectral_density(train_data, sfreq, alpha_fmin, logger)
    if psds is None or freqs is None:
        return None, None

    if psds.ndim != 3 or freqs.size == 0:
        return None, None

    mean_psd = np.nanmean(psds, axis=(0, 1))
    residual = _compute_aperiodic_residual(freqs, mean_psd)
    if residual is None:
        return None, None

    iaf_hz = _estimate_iaf_from_residual(
        freqs, residual, alpha_fmin, alpha_fmax, prominence
    )
    if iaf_hz is None:
        return None, None

    frequency_bands = _build_frequency_bands_from_iaf(iaf_hz, config, alpha_width_hz)

    _log_info(
        logger,
        "CV hygiene: Estimated fold-specific IAF=%.2f Hz from %d training trials",
        iaf_hz,
        train_data.shape[0],
    )

    return iaf_hz, frequency_bands


def _should_compute_iaf(config: Any) -> bool:
    """Determine if IAF should be computed based on configuration."""
    if hasattr(config, "get"):
        return bool(config.get("feature_engineering.bands.use_iaf", False))
    return False


def create_fold_specific_context(
    epochs: Any,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    fold_idx: int,
    config: Any,
    logger: Any = None,
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

    if _should_compute_iaf(config):
        picks = mne.pick_types(
            epochs.info, eeg=True, meg=False, eog=False, stim=False, exclude="bads"
        )
        epochs_data = epochs.get_data(picks=picks)
        sfreq = float(epochs.info["sfreq"])

        iaf_hz, frequency_bands = compute_iaf_for_fold(
            epochs_data, sfreq, train_mask, config, logger
        )
        params.iaf_hz = iaf_hz
        params.frequency_bands = frequency_bands

    params.metadata["n_train"] = int(np.sum(train_mask))
    params.metadata["n_test"] = len(test_indices)

    iaf_display = params.iaf_hz if params.iaf_hz is not None else np.nan
    _log_info(
        logger,
        "CV hygiene: Created fold %d context (train=%d, test=%d, IAF=%.2f Hz)",
        fold_idx,
        params.metadata["n_train"],
        params.metadata["n_test"],
        iaf_display,
    )

    return params


__all__ = [
    "FoldSpecificParams",
    "compute_iaf_for_fold",
    "create_fold_specific_context",
]
