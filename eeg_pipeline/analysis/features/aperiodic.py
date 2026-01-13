"""
Aperiodic (1/f) Feature Extraction
===================================

Extracts aperiodic spectral features using FOOOF-like fitting:
- Slope: 1/f exponent (related to E/I balance)
- Offset: Broadband power level
- Power-corrected band power: Ratio of observed to 1/f background power within band

These features separate oscillatory from aperiodic neural activity.
"""

from __future__ import annotations

from typing import Optional, List, Dict, Tuple, Any, NamedTuple
import numpy as np
import pandas as pd
import mne
import warnings
from scipy import stats
from scipy.optimize import curve_fit
from joblib import Parallel, delayed

from eeg_pipeline.utils.analysis.channels import pick_eeg_channels
from eeg_pipeline.domain.features.naming import NamingSchema
from eeg_pipeline.domain.features.constants import get_segment_mask, validate_extractor_inputs
from eeg_pipeline.utils.config.loader import get_frequency_bands_for_aperiodic
from eeg_pipeline.utils.analysis.stats import compute_residuals
from eeg_pipeline.utils.parallel import get_n_jobs


# Constants
_MIN_POWER_LOG10 = 1e-20
_MIN_MAD_THRESHOLD = 1e-12
_MIN_EXPONENT = 0.01
_DEFAULT_MIN_FIT_POINTS = 5
_DEFAULT_PEAK_REJECTION_Z = 3.5
_DEFAULT_MIN_SEGMENT_SEC = 2.0
_DEFAULT_FMIN = 2.0
_DEFAULT_FMAX = 40.0
_DEFAULT_LINE_FREQ = 50.0
_DEFAULT_LINE_WIDTH = 1.0
_DEFAULT_LINE_HARMONICS = 3
_KNEEMODEL_MAX_ITERATIONS = 8000
_KNEEMODEL_MAX_EXPONENT = 5.0


# Data structures for grouping related parameters
class FitParameters(NamedTuple):
    """Parameters for aperiodic fitting."""
    peak_rejection_z: float
    min_fit_points: int
    model: str


class LineNoiseConfig(NamedTuple):
    """Configuration for line noise exclusion."""
    exclude: bool
    frequencies: List[float]
    width_hz: float
    n_harmonics: int


class PSDFitResult(NamedTuple):
    """Result from fitting aperiodic model to single epoch/channel."""
    epoch_idx: int
    channel_idx: int
    offset: float
    slope: float
    valid_bins: int
    kept_bins: int
    peak_rejected: bool
    fit_indices: np.ndarray
    status: int


class KneeFitResult(NamedTuple):
    """Result from fitting knee model to single epoch/channel."""
    epoch_idx: int
    channel_idx: int
    offset: float
    exponent: float
    knee: float
    valid_bins: int
    kept_bins: int
    peak_rejected: bool
    fit_indices: np.ndarray
    status: int


# Validation functions
def _validate_fit_parameters(
    peak_rejection_z: float,
    min_fit_points: int,
    model: str,
) -> FitParameters:
    """Validate and normalize fit parameters."""
    if not np.isfinite(peak_rejection_z) or peak_rejection_z <= 0:
        peak_rejection_z = _DEFAULT_PEAK_REJECTION_Z
    
    if min_fit_points < 1:
        min_fit_points = _DEFAULT_MIN_FIT_POINTS
    
    model = str(model).strip().lower()
    if model not in {"fixed", "knee"}:
        model = "fixed"
    
    return FitParameters(peak_rejection_z, min_fit_points, model)


def _validate_frequencies(freqs: np.ndarray) -> np.ndarray:
    """Validate frequency array."""
    if not isinstance(freqs, np.ndarray):
        freqs = np.asarray(freqs, dtype=float)
    if freqs.size == 0:
        raise ValueError("Frequency array cannot be empty")
    return freqs


def _validate_psd_data(psd: np.ndarray) -> np.ndarray:
    """Validate and normalize PSD data dimensions."""
    if psd.ndim == 4:
        psd = np.mean(psd, axis=-1)
    if psd.ndim != 3:
        raise ValueError(f"PSD data must be 3D (epochs, channels, frequencies), got {psd.ndim}D")
    return psd


# Peak rejection logic (shared between fixed and knee models)
def _apply_residual_based_peak_rejection(
    log_freqs: np.ndarray,
    log_psd: np.ndarray,
    peak_rejection_z: float,
    min_fit_points: int,
    model: str = "fixed",
    freqs_hz: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, bool]:
    """Apply iterative residual-based peak rejection.
    
    The scientifically correct approach for aperiodic fitting:
    1. Fit initial aperiodic model on all valid points
    2. Compute residuals (log_psd - aperiodic_fit)
    3. Reject points with positive residual outliers (oscillatory peaks)
    4. Refit on remaining points
    5. Repeat until convergence or max iterations
    
    This avoids the bias of raw MAD thresholding, which preferentially removes
    low-frequency bins (where 1/f power is highest) rather than true peaks.
    
    Args:
        log_freqs: Log10 of frequency values
        log_psd: Log10 of PSD values
        peak_rejection_z: Z-score threshold for outlier rejection
        min_fit_points: Minimum points required for valid fit
        model: "fixed" or "knee"
        freqs_hz: Linear frequency values (required for knee model)
    
    Returns:
        Tuple of (keep_mask, peak_rejected_flag)
    """
    n_values = log_psd.size
    keep_mask = np.ones(n_values, dtype=bool)
    peak_rejected = False
    
    finite_mask = np.isfinite(log_freqs) & np.isfinite(log_psd)
    if np.sum(finite_mask) < min_fit_points:
        return keep_mask, peak_rejected
    
    keep_mask = finite_mask.copy()
    max_iterations = 3
    
    for iteration in range(max_iterations):
        kept_indices = np.flatnonzero(keep_mask)
        if len(kept_indices) < min_fit_points:
            break
        
        # Fit aperiodic model on current kept points
        try:
            if model == "knee" and freqs_hz is not None:
                f_kept = freqs_hz[kept_indices]
                y_kept = log_psd[kept_indices]
                initial_offset = float(np.nanmedian(y_kept))
                popt, _ = curve_fit(
                    _knee_model_function, f_kept, y_kept,
                    p0=(initial_offset, 1.0, 1.0),
                    bounds=([-np.inf, 0.0, _MIN_EXPONENT], [np.inf, np.inf, _KNEEMODEL_MAX_EXPONENT]),
                    maxfev=_KNEEMODEL_MAX_ITERATIONS
                )
                predicted = _knee_model_function(freqs_hz, *popt)
            else:
                slope, intercept = np.polyfit(log_freqs[kept_indices], log_psd[kept_indices], 1)
                predicted = intercept + slope * log_freqs
        except Exception:
            break
        
        # Compute residuals (positive = above aperiodic fit = potential peak)
        residuals = log_psd - predicted
        
        # Only consider positive residuals for peak detection
        positive_residuals = np.where(residuals > 0, residuals, 0.0)
        kept_positive = positive_residuals[keep_mask]
        
        if len(kept_positive) == 0 or np.all(kept_positive == 0):
            break
        
        # MAD of positive residuals for robust threshold
        mad = stats.median_abs_deviation(kept_positive[kept_positive > 0], scale="normal", nan_policy="omit")
        if not np.isfinite(mad) or mad < _MIN_MAD_THRESHOLD:
            break
        
        threshold = peak_rejection_z * mad
        
        # Reject points with large positive residuals (oscillatory peaks)
        new_keep = keep_mask & (residuals <= threshold)
        n_kept = int(np.sum(new_keep))
        
        if n_kept < min_fit_points:
            break
        
        if np.array_equal(new_keep, keep_mask):
            break
        
        n_rejected = int(np.sum(keep_mask)) - n_kept
        if n_rejected > 0:
            peak_rejected = True
        
        keep_mask = new_keep
    
    return keep_mask, peak_rejected


# Fixed model fitting
def _fit_fixed_model(
    log_freqs: np.ndarray,
    log_psd: np.ndarray,
    fit_indices: np.ndarray,
) -> Tuple[float, float]:
    """Fit fixed (linear) aperiodic model: log10(P) = offset + slope * log10(f)."""
    freq_fit = log_freqs[fit_indices]
    psd_fit = log_psd[fit_indices]
    slope, intercept = np.polyfit(freq_fit, psd_fit, 1)
    return float(intercept), float(slope)


def _fit_single_epoch_channel(
    epoch_idx: int,
    channel_idx: int,
    log_freqs: np.ndarray,
    psd_vals: np.ndarray,
    fit_params: FitParameters,
) -> PSDFitResult:
    """Fit fixed aperiodic model to single epoch/channel using residual-based peak rejection."""
    finite_mask = np.isfinite(log_freqs) & np.isfinite(psd_vals)
    valid_bins = int(np.sum(finite_mask))
    
    if valid_bins < fit_params.min_fit_points:
        return PSDFitResult(
            epoch_idx, channel_idx, np.nan, np.nan,
            valid_bins, 0, False, np.array([], dtype=int), 1
        )
    
    # Use residual-based peak rejection (scientifically correct approach)
    keep_mask, peak_rejected = _apply_residual_based_peak_rejection(
        log_freqs, psd_vals,
        fit_params.peak_rejection_z, fit_params.min_fit_points,
        model="fixed"
    )
    
    kept_indices = np.flatnonzero(keep_mask)
    kept_bins = int(kept_indices.size)
    
    if kept_bins < fit_params.min_fit_points:
        return PSDFitResult(
            epoch_idx, channel_idx, np.nan, np.nan,
            valid_bins, kept_bins, peak_rejected, np.array([], dtype=int), 2
        )
    
    try:
        offset, slope = _fit_fixed_model(log_freqs, psd_vals, kept_indices)
        return PSDFitResult(
            epoch_idx, channel_idx, offset, slope,
            valid_bins, kept_bins, peak_rejected, kept_indices.astype(int), 0
        )
    except Exception:
        return PSDFitResult(
            epoch_idx, channel_idx, np.nan, np.nan,
            valid_bins, kept_bins, peak_rejected, np.array([], dtype=int), 3
        )


# Knee model fitting
def _knee_model_function(
    f_hz: np.ndarray,
    offset: float,
    knee: float,
    exponent: float,
) -> np.ndarray:
    """Knee model: log10(P) = offset - log10(knee + f^exponent)."""
    knee = np.maximum(knee, 0.0)
    exponent = np.maximum(exponent, _MIN_EXPONENT)
    return offset - np.log10(knee + np.power(f_hz, exponent))


def _fit_knee_model(
    freqs_hz: np.ndarray,
    log_psd: np.ndarray,
    fit_indices: np.ndarray,
) -> Tuple[float, float, float]:
    """Fit knee aperiodic model to data."""
    f = np.asarray(freqs_hz[fit_indices], dtype=float)
    yfit = np.asarray(log_psd[fit_indices], dtype=float)
    
    initial_offset = float(np.nanmedian(yfit))
    initial_guess = (initial_offset, 1.0, 1.0)
    bounds = ([-np.inf, 0.0, _MIN_EXPONENT], [np.inf, np.inf, _KNEEMODEL_MAX_EXPONENT])
    
    popt, _ = curve_fit(
        _knee_model_function, f, yfit,
        p0=initial_guess, bounds=bounds, maxfev=_KNEEMODEL_MAX_ITERATIONS
    )
    
    offset = float(popt[0])
    knee = float(popt[1])
    exponent = float(popt[2])
    return offset, knee, exponent


def _fit_single_epoch_channel_knee(
    epoch_idx: int,
    channel_idx: int,
    freqs_hz: np.ndarray,
    log_psd_vals: np.ndarray,
    fit_params: FitParameters,
) -> KneeFitResult:
    """Fit knee aperiodic model to single epoch/channel using residual-based peak rejection."""
    finite_mask = np.isfinite(freqs_hz) & np.isfinite(log_psd_vals)
    valid_bins = int(np.sum(finite_mask))
    
    if valid_bins < fit_params.min_fit_points:
        return KneeFitResult(
            epoch_idx, channel_idx, np.nan, np.nan, np.nan,
            valid_bins, 0, False, np.array([], dtype=int), 1
        )
    
    # Use residual-based peak rejection with knee model
    log_freqs = np.log10(np.maximum(freqs_hz, 1e-6))
    keep_mask, peak_rejected = _apply_residual_based_peak_rejection(
        log_freqs, log_psd_vals,
        fit_params.peak_rejection_z, fit_params.min_fit_points,
        model="knee", freqs_hz=freqs_hz
    )
    
    kept_indices = np.flatnonzero(keep_mask)
    kept_bins = int(kept_indices.size)
    
    if kept_bins < fit_params.min_fit_points:
        return KneeFitResult(
            epoch_idx, channel_idx, np.nan, np.nan, np.nan,
            valid_bins, kept_bins, peak_rejected, np.array([], dtype=int), 2
        )
    
    try:
        offset, knee, exponent = _fit_knee_model(freqs_hz, log_psd_vals, kept_indices)
        return KneeFitResult(
            epoch_idx, channel_idx, offset, exponent, knee,
            valid_bins, kept_bins, peak_rejected, kept_indices.astype(int), 0
        )
    except Exception:
        return KneeFitResult(
            epoch_idx, channel_idx, np.nan, np.nan, np.nan,
            valid_bins, kept_bins, peak_rejected, np.array([], dtype=int), 3
        )


# Line noise exclusion
def _build_line_noise_mask(
    freqs: np.ndarray,
    line_config: LineNoiseConfig,
) -> np.ndarray:
    """Build mask to exclude line-noise frequencies and harmonics.
    
    Returns:
        Boolean array where True indicates frequencies to KEEP (not line noise).
    """
    keep = np.ones_like(freqs, dtype=bool)
    
    should_exclude = (
        line_config.exclude and
        line_config.frequencies and
        line_config.width_hz > 0 and
        line_config.n_harmonics >= 1
    )
    
    if not should_exclude:
        return keep
    
    for base_freq in line_config.frequencies:
        if not np.isfinite(base_freq) or base_freq <= 0:
            continue
        
        for harmonic in range(1, line_config.n_harmonics + 1):
            harmonic_freq = base_freq * harmonic
            lower_bound = harmonic_freq - line_config.width_hz
            upper_bound = harmonic_freq + line_config.width_hz
            is_line_noise = (freqs >= lower_bound) & (freqs <= upper_bound)
            keep &= ~is_line_noise
    
    return keep


def _parse_line_noise_config(config: Any) -> LineNoiseConfig:
    """Parse line noise exclusion configuration from config object."""
    aperiodic_cfg = config.get("feature_engineering.aperiodic", {}) if hasattr(config, "get") else {}
    
    exclude = bool(aperiodic_cfg.get("exclude_line_noise", True))
    
    line_freqs_raw = aperiodic_cfg.get("line_noise_freqs", [_DEFAULT_LINE_FREQ])
    try:
        line_freqs = [float(f) for f in (line_freqs_raw or [])]
    except Exception:
        line_freqs = [_DEFAULT_LINE_FREQ]
    
    line_width = float(aperiodic_cfg.get("line_noise_width_hz", _DEFAULT_LINE_WIDTH))
    n_harm = int(aperiodic_cfg.get("line_noise_harmonics", _DEFAULT_LINE_HARMONICS))
    
    return LineNoiseConfig(exclude, line_freqs, line_width, n_harm)


# Parallel fitting driver
def _fit_aperiodic_with_qc(
    log_freqs: np.ndarray,
    log_psd: np.ndarray,
    fit_params: FitParameters,
    logger: Any,
    *,
    n_jobs: int = 1,
    line_noise_mask: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Fit aperiodic model across all epochs and channels with quality control."""
    n_epochs, n_channels, n_freqs = log_psd.shape
    
    offsets = np.full((n_epochs, n_channels), np.nan)
    slopes = np.full((n_epochs, n_channels), np.nan)
    valid_bins = np.zeros((n_epochs, n_channels), dtype=int)
    kept_bins = np.zeros((n_epochs, n_channels), dtype=int)
    peak_rejected = np.zeros((n_epochs, n_channels), dtype=bool)
    fit_masks = np.zeros((n_epochs, n_channels, n_freqs), dtype=bool)
    knees = np.full((n_epochs, n_channels), np.nan)
    statuses = np.zeros((n_epochs, n_channels), dtype=int)
    
    tasks = [(ep_idx, ch_idx) for ep_idx in range(n_epochs) for ch_idx in range(n_channels)]
    
    # Apply line-noise mask to exclude those bins from fitting
    if line_noise_mask is not None and line_noise_mask.shape[0] == n_freqs:
        log_freqs_fit = log_freqs[line_noise_mask]
        log_psd_fit = log_psd[:, :, line_noise_mask]
    else:
        log_freqs_fit = log_freqs
        log_psd_fit = log_psd
    
    # Execute fitting (parallel or serial)
    if fit_params.model == "knee":
        freqs_hz_fit = np.power(10.0, np.asarray(log_freqs_fit, dtype=float))
        
        if n_jobs != 1:
            results = Parallel(n_jobs=n_jobs)(
                delayed(_fit_single_epoch_channel_knee)(
                    ep_idx, ch_idx, freqs_hz_fit,
                    log_psd_fit[ep_idx, ch_idx, :], fit_params
                )
                for ep_idx, ch_idx in tasks
            )
        else:
            results = [
                _fit_single_epoch_channel_knee(
                    ep_idx, ch_idx, freqs_hz_fit,
                    log_psd_fit[ep_idx, ch_idx, :], fit_params
                )
                for ep_idx, ch_idx in tasks
            ]
        
        # Process knee model results
        for res in results:
            offsets[res.epoch_idx, res.channel_idx] = res.offset
            exponent = res.exponent
            slopes[res.epoch_idx, res.channel_idx] = -float(exponent) if np.isfinite(exponent) else np.nan
            knees[res.epoch_idx, res.channel_idx] = res.knee
            valid_bins[res.epoch_idx, res.channel_idx] = res.valid_bins
            kept_bins[res.epoch_idx, res.channel_idx] = res.kept_bins
            peak_rejected[res.epoch_idx, res.channel_idx] = res.peak_rejected
            statuses[res.epoch_idx, res.channel_idx] = int(res.status)
            if res.fit_indices.size > 0:
                fit_masks[res.epoch_idx, res.channel_idx, res.fit_indices] = True
    else:
        if n_jobs != 1:
            results = Parallel(n_jobs=n_jobs)(
                delayed(_fit_single_epoch_channel)(
                    ep_idx, ch_idx, log_freqs_fit,
                    log_psd_fit[ep_idx, ch_idx, :], fit_params
                )
                for ep_idx, ch_idx in tasks
            )
        else:
            results = [
                _fit_single_epoch_channel(
                    ep_idx, ch_idx, log_freqs_fit,
                    log_psd_fit[ep_idx, ch_idx, :], fit_params
                )
                for ep_idx, ch_idx in tasks
            ]
        
        # Process fixed model results
        for res in results:
            offsets[res.epoch_idx, res.channel_idx] = res.offset
            slopes[res.epoch_idx, res.channel_idx] = res.slope
            valid_bins[res.epoch_idx, res.channel_idx] = res.valid_bins
            kept_bins[res.epoch_idx, res.channel_idx] = res.kept_bins
            peak_rejected[res.epoch_idx, res.channel_idx] = res.peak_rejected
            statuses[res.epoch_idx, res.channel_idx] = int(res.status)
            if res.fit_indices.size > 0:
                fit_masks[res.epoch_idx, res.channel_idx, res.fit_indices] = True
    
    qc_dict = {
        "model": fit_params.model,
        "knees": knees,
        "status": statuses,
    }
    
    return offsets, slopes, valid_bins, kept_bins, peak_rejected, fit_masks, qc_dict


# Fit quality metrics
def _compute_fit_r2_and_rms(
    log_freqs: np.ndarray,
    log_psd: np.ndarray,
    offsets: np.ndarray,
    slopes: np.ndarray,
    fit_masks: np.ndarray,
    model: str = "fixed",
    knees: Optional[np.ndarray] = None,
    freqs_hz: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute R² and RMS error for aperiodic fits using the correct model.
    
    For fixed model: y_pred = offset + slope * log10(f)
    For knee model: y_pred = offset - log10(knee + f^exponent)
    
    Args:
        log_freqs: Log10 of frequency values
        log_psd: Log10 of PSD values (epochs, channels, freqs)
        offsets: Fitted offset values (epochs, channels)
        slopes: Fitted slope values (epochs, channels) - for knee model, this is -exponent
        fit_masks: Boolean masks of fitted points (epochs, channels, freqs)
        model: "fixed" or "knee"
        knees: Fitted knee values (required for knee model)
        freqs_hz: Linear frequency values (required for knee model)
    """
    n_epochs, n_channels, _ = log_psd.shape
    r2 = np.full((n_epochs, n_channels), np.nan)
    rms = np.full((n_epochs, n_channels), np.nan)
    
    use_knee = model == "knee" and knees is not None and freqs_hz is not None
    
    for epoch_idx in range(n_epochs):
        for channel_idx in range(n_channels):
            mask = fit_masks[epoch_idx, channel_idx, :]
            offset = offsets[epoch_idx, channel_idx]
            slope = slopes[epoch_idx, channel_idx]
            
            if not np.any(mask) or not np.isfinite(offset) or not np.isfinite(slope):
                continue
            
            y_true = log_psd[epoch_idx, channel_idx, mask]
            
            if use_knee:
                knee = knees[epoch_idx, channel_idx]
                if not np.isfinite(knee):
                    knee = 0.0
                exponent = max(-slope, _MIN_EXPONENT)
                f_masked = freqs_hz[mask]
                y_pred = _knee_model_function(f_masked, offset, knee, exponent)
            else:
                y_pred = offset + slope * log_freqs[mask]
            
            residuals = y_true - y_pred
            
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            
            if ss_tot > 0:
                r2[epoch_idx, channel_idx] = 1.0 - (ss_res / ss_tot)
            
            rms[epoch_idx, channel_idx] = np.sqrt(np.mean(residuals ** 2))
    
    return r2, rms


# Residual computation
def _compute_knee_residuals(
    freqs: np.ndarray,
    log_psd: np.ndarray,
    offsets: np.ndarray,
    slopes: np.ndarray,
    knees: np.ndarray,
) -> np.ndarray:
    """Compute residuals for knee model."""
    residuals = np.full_like(log_psd, np.nan, dtype=float)
    freqs_linear = freqs.astype(float)
    
    n_epochs, n_channels, _ = log_psd.shape
    for epoch_idx in range(n_epochs):
        for channel_idx in range(n_channels):
            offset = offsets[epoch_idx, channel_idx]
            slope = slopes[epoch_idx, channel_idx]
            
            if not np.isfinite(offset) or not np.isfinite(slope):
                continue
            
            knee = float(knees[epoch_idx, channel_idx]) if np.isfinite(knees[epoch_idx, channel_idx]) else 0.0
            exponent = max(float(-slope), _MIN_EXPONENT)
            background = float(offset) - np.log10(np.maximum(knee, 0.0) + np.power(freqs_linear, exponent))
            residuals[epoch_idx, channel_idx, :] = log_psd[epoch_idx, channel_idx, :] - background
    
    return residuals


# PSD computation
def _compute_psd(
    epochs: mne.Epochs,
    picks: np.ndarray,
    start_t: float,
    end_t: float,
    fmin: float,
    fmax: float,
    psd_method: str,
    psd_kwargs: Dict[str, Any],
    logger: Any,
    segment_name: str,
    *,
    subtract_evoked: bool = False,
    condition_labels: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute power spectral density with fallback handling.
    
    Parameters
    ----------
    subtract_evoked : bool
        If True, subtract evoked response before PSD computation to isolate
        induced activity. Recommended for pain paradigms where ERPs can bias
        low-frequency slope estimates.
    condition_labels : Optional[np.ndarray]
        If provided with subtract_evoked=True, subtract condition-specific
        evoked responses instead of grand average.
    """
    # Get data for the segment
    data = epochs.get_data(picks=picks, tmin=start_t, tmax=end_t)
    
    if subtract_evoked:
        from eeg_pipeline.utils.analysis.spectral import subtract_evoked as _subtract_evoked
        data = _subtract_evoked(data, condition_labels)
        logger.info(
            "Aperiodic: Computing induced spectra (evoked subtracted) for %s",
            segment_name
        )
    
    # Compute PSD on the (possibly evoked-subtracted) data
    sfreq = epochs.info["sfreq"]
    
    try:
        if psd_method == "multitaper":
            from mne.time_frequency import psd_array_multitaper
            psds, freqs = psd_array_multitaper(
                data,
                sfreq=sfreq,
                fmin=fmin,
                fmax=fmax,
                adaptive=psd_kwargs.get("adaptive", True),
                normalization=psd_kwargs.get("normalization", "full"),
                bandwidth=psd_kwargs.get("bandwidth"),
                verbose=False,
            )
        else:
            from mne.time_frequency import psd_array_welch
            n_times = data.shape[-1]
            n_per_seg = min(int(sfreq * 2.0), n_times)
            n_overlap = n_per_seg // 2
            psds, freqs = psd_array_welch(
                data,
                sfreq=sfreq,
                fmin=fmin,
                fmax=fmax,
                n_fft=max(256, n_per_seg),
                n_per_seg=n_per_seg,
                n_overlap=n_overlap,
                verbose=False,
            )
    except Exception as e:
        logger.warning(f"PSD computation failed for {segment_name}: {e}")
        raise
    
    psds = _validate_psd_data(psds)
    freqs = _validate_frequencies(freqs)
    
    return psds, freqs


def _parse_psd_config(config: Any) -> Tuple[str, Dict[str, Any], float, float]:
    """Parse PSD computation configuration."""
    aperiodic_cfg = config.get("feature_engineering.aperiodic", {}) if hasattr(config, "get") else {}
    
    fmin = float(aperiodic_cfg.get("fmin", config.get("feature_engineering.constants.aperiodic_fmin", _DEFAULT_FMIN)))
    fmax = float(aperiodic_cfg.get("fmax", config.get("feature_engineering.constants.aperiodic_fmax", _DEFAULT_FMAX)))
    
    psd_method = str(aperiodic_cfg.get("psd_method", "multitaper")).strip().lower()
    if psd_method not in {"multitaper", "welch"}:
        psd_method = "multitaper"
    
    psd_kwargs: Dict[str, Any] = {}
    if psd_method == "multitaper":
        bandwidth = aperiodic_cfg.get("psd_bandwidth", None)
        if bandwidth is not None:
            try:
                bandwidth = float(bandwidth)
            except Exception:
                bandwidth = None
        
        if bandwidth is not None and np.isfinite(bandwidth) and bandwidth > 0:
            psd_kwargs["bandwidth"] = float(bandwidth)
        
        psd_kwargs.setdefault("adaptive", True)
        psd_kwargs.setdefault("normalization", "full")
    
    return psd_method, psd_kwargs, fmin, fmax


# Feature computation
def _compute_alpha_peak_frequency(
    freqs: np.ndarray,
    residuals: np.ndarray,
    alpha_range: Tuple[float, float],
    fit_ok: np.ndarray,
) -> np.ndarray:
    """Compute alpha peak frequency using center-of-gravity method."""
    n_epochs, n_channels, _ = residuals.shape
    apf_matrix = np.full((n_epochs, n_channels), np.nan)
    
    alpha_mask = (freqs >= alpha_range[0]) & (freqs <= alpha_range[1])
    if not np.any(alpha_mask):
        return apf_matrix
    
    relative_power = np.power(10.0, residuals)
    
    for channel_idx in range(n_channels):
        if not np.any(fit_ok[:, channel_idx]):
            continue
        
        alpha_rel = np.maximum(relative_power[:, channel_idx, alpha_mask], 0.0)
        total_power = np.sum(alpha_rel, axis=1)
        
        with np.errstate(invalid="ignore", divide="ignore"):
            apf_matrix[:, channel_idx] = np.where(
                total_power > 0,
                np.sum(freqs[alpha_mask] * alpha_rel, axis=1) / total_power,
                np.nan,
            )
        
        apf_matrix[~fit_ok[:, channel_idx], channel_idx] = np.nan
    
    return apf_matrix


def _compute_theta_beta_ratio(
    freqs: np.ndarray,
    residuals: np.ndarray,
    theta_range: Tuple[float, float],
    beta_range: Tuple[float, float],
    fit_ok: np.ndarray,
) -> np.ndarray:
    """Compute theta/beta ratio from aperiodic-adjusted residuals."""
    n_epochs, n_channels, _ = residuals.shape
    tbr_matrix = np.full((n_epochs, n_channels), np.nan)
    
    theta_mask = (freqs >= theta_range[0]) & (freqs <= theta_range[1])
    beta_mask = (freqs >= beta_range[0]) & (freqs <= beta_range[1])
    
    if not (np.any(theta_mask) and np.any(beta_mask)):
        return tbr_matrix
    
    relative_power = np.power(10.0, residuals)
    
    for channel_idx in range(n_channels):
        if not np.any(fit_ok[:, channel_idx]):
            continue
        
        theta_rel = np.nanmean(relative_power[:, channel_idx, theta_mask], axis=1)
        beta_rel = np.nanmean(relative_power[:, channel_idx, beta_mask], axis=1)
        
        with np.errstate(invalid="ignore", divide="ignore"):
            tbr_matrix[:, channel_idx] = np.where(beta_rel > 0, theta_rel / beta_rel, np.nan)
        
        tbr_matrix[~fit_ok[:, channel_idx], channel_idx] = np.nan
    
    return tbr_matrix


def _validate_band_coverage(
    freqs: np.ndarray,
    band_name: str,
    band_range: Tuple[float, float],
    logger: Any,
) -> Tuple[bool, float]:
    """Validate that frequency array covers the band range.
    
    Returns:
        Tuple of (is_valid, coverage_fraction)
        - is_valid: True if at least some frequencies fall within band
        - coverage_fraction: Fraction of band range covered by freqs
    """
    fmin, fmax = float(freqs.min()), float(freqs.max())
    band_lo, band_hi = float(band_range[0]), float(band_range[1])
    
    # Check if band is completely outside PSD range
    if band_hi < fmin or band_lo > fmax:
        logger.warning(
            "Band '%s' [%.1f-%.1f Hz] is completely outside aperiodic PSD range [%.1f-%.1f Hz]. "
            "Features for this band will be NaN. Adjust aperiodic.fmin/fmax in config.",
            band_name, band_lo, band_hi, fmin, fmax
        )
        return False, 0.0
    
    # Check for partial coverage
    effective_lo = max(band_lo, fmin)
    effective_hi = min(band_hi, fmax)
    band_width = band_hi - band_lo
    covered_width = effective_hi - effective_lo
    coverage_fraction = covered_width / band_width if band_width > 0 else 0.0
    
    if coverage_fraction < 1.0:
        logger.warning(
            "Band '%s' [%.1f-%.1f Hz] is only %.0f%% covered by aperiodic PSD range [%.1f-%.1f Hz]. "
            "Effective range: [%.1f-%.1f Hz]. Consider adjusting aperiodic.fmin/fmax.",
            band_name, band_lo, band_hi, coverage_fraction * 100, fmin, fmax,
            effective_lo, effective_hi
        )
    
    return True, coverage_fraction


def _compute_frequency_weights(freqs: np.ndarray) -> np.ndarray:
    """Compute frequency bin widths for weighted integration.
    
    For uniform grids, all weights are equal. For non-uniform grids (log-spaced,
    adaptive), weights reflect the frequency range each bin represents.
    
    Uses trapezoidal rule: each bin's weight is half the distance to neighbors.
    """
    n_freqs = len(freqs)
    if n_freqs <= 1:
        return np.ones(n_freqs)
    
    weights = np.zeros(n_freqs)
    
    # First bin: half distance to next
    weights[0] = (freqs[1] - freqs[0]) / 2.0
    
    # Middle bins: half distance to both neighbors
    for i in range(1, n_freqs - 1):
        weights[i] = (freqs[i + 1] - freqs[i - 1]) / 2.0
    
    # Last bin: half distance to previous
    weights[-1] = (freqs[-1] - freqs[-2]) / 2.0
    
    return weights


def _compute_power_corrected_band_power(
    freqs: np.ndarray,
    residuals: np.ndarray,
    band_name: str,
    band_range: Tuple[float, float],
    fit_ok: np.ndarray,
) -> np.ndarray:
    """Compute power-corrected band power from residuals.
    
    Uses frequency-weighted integration to handle non-uniform frequency grids
    correctly. For uniform grids, this is equivalent to simple averaging.
    """
    n_epochs, n_channels, _ = residuals.shape
    pc_matrix = np.full((n_epochs, n_channels), np.nan)
    
    band_mask = (freqs >= band_range[0]) & (freqs <= band_range[1])
    if not np.any(band_mask):
        return pc_matrix
    
    # Compute frequency weights for proper integration
    all_weights = _compute_frequency_weights(freqs)
    band_weights = all_weights[band_mask]
    
    # Normalize weights to sum to 1 for weighted average
    weight_sum = band_weights.sum()
    if weight_sum > 0:
        band_weights = band_weights / weight_sum
    else:
        band_weights = np.ones_like(band_weights) / len(band_weights)
    
    for channel_idx in range(n_channels):
        res_band = residuals[:, channel_idx, band_mask]
        ratio = np.power(10.0, res_band)
        # Weighted mean instead of simple mean
        pc_matrix[:, channel_idx] = np.sum(ratio * band_weights, axis=1)
        pc_matrix[~fit_ok[:, channel_idx], channel_idx] = np.nan
    
    return pc_matrix


# Spatial aggregation
def _build_roi_map(ch_names: List[str], config: Any) -> Dict[str, List[int]]:
    """Build ROI mapping from channel names to indices."""
    from eeg_pipeline.utils.analysis.spatial import get_roi_definitions
    from eeg_pipeline.utils.analysis.channels import build_roi_map
    
    roi_defs = get_roi_definitions(config)
    if not roi_defs:
        return {}
    
    return build_roi_map(ch_names, roi_defs)


_DEFAULT_MIN_VALID_CHANNELS_ROI = 2
_DEFAULT_MIN_VALID_CHANNELS_GLOBAL = 3
_DEFAULT_MIN_VALID_FRACTION_ROI = 0.5


def _aggregate_features_by_spatial_mode(
    metrics: Dict[str, Tuple[str, str, np.ndarray]],
    ch_names: List[str],
    segment_name: str,
    spatial_modes: List[str],
    roi_map: Dict[str, List[int]],
    min_valid_channels_roi: int = _DEFAULT_MIN_VALID_CHANNELS_ROI,
    min_valid_channels_global: int = _DEFAULT_MIN_VALID_CHANNELS_GLOBAL,
    min_valid_fraction_roi: float = _DEFAULT_MIN_VALID_FRACTION_ROI,
) -> Dict[str, np.ndarray]:
    """Aggregate features according to spatial modes (channels, ROI, global).
    
    Only computes spatial modes that are explicitly requested.
    Applies minimum valid channel rules to prevent spurious estimates from
    trials where most channels failed QC.
    
    Args:
        metrics: Dict mapping metric name to (band, stat, matrix) tuples
        ch_names: List of channel names
        segment_name: Name of time segment
        spatial_modes: List of modes to compute ('channels', 'roi', 'global')
        roi_map: Dict mapping ROI names to channel indices
        min_valid_channels_roi: Minimum valid channels required per ROI per trial
        min_valid_channels_global: Minimum valid channels required for global average
        min_valid_fraction_roi: Minimum fraction of ROI channels that must be valid
    """
    data_dict: Dict[str, np.ndarray] = {}
    
    for metric_name, (band, stat, matrix) in metrics.items():
        n_epochs = matrix.shape[0]
        
        # Per-channel aggregation (only if requested)
        if 'channels' in spatial_modes:
            for channel_idx, channel_name in enumerate(ch_names):
                column_name = NamingSchema.build(
                    "aperiodic", segment_name, band, "ch", stat, channel=channel_name
                )
                data_dict[column_name] = matrix[:, channel_idx]
        
        # ROI aggregation with minimum valid channel rules
        if 'roi' in spatial_modes and roi_map:
            for roi_name, channel_indices in roi_map.items():
                if not channel_indices or len(channel_indices) == 0:
                    continue
                
                roi_matrix = matrix[:, channel_indices]
                roi_values = np.full(n_epochs, np.nan)
                
                for trial_idx in range(n_epochs):
                    trial_vals = roi_matrix[trial_idx, :]
                    n_valid = int(np.sum(np.isfinite(trial_vals)))
                    n_total = len(channel_indices)
                    valid_fraction = n_valid / n_total if n_total > 0 else 0.0
                    
                    # Apply minimum valid channel rules
                    if n_valid >= min_valid_channels_roi and valid_fraction >= min_valid_fraction_roi:
                        roi_values[trial_idx] = np.nanmean(trial_vals)
                
                column_name = NamingSchema.build(
                    "aperiodic", segment_name, band, "roi", stat, channel=roi_name
                )
                data_dict[column_name] = roi_values
        
        # Global aggregation with minimum valid channel rules
        if 'global' in spatial_modes:
            global_values = np.full(n_epochs, np.nan)
            
            for trial_idx in range(n_epochs):
                trial_vals = matrix[trial_idx, :]
                n_valid = int(np.sum(np.isfinite(trial_vals)))
                
                if n_valid >= min_valid_channels_global:
                    global_values[trial_idx] = np.nanmean(trial_vals)
            
            column_name = NamingSchema.build("aperiodic", segment_name, band, "global", stat)
            data_dict[column_name] = global_values
    
    return data_dict


# Main extraction function
def _extract_aperiodic_for_segment(
    epochs: mne.Epochs,
    picks: np.ndarray,
    ch_names: List[str],
    segment_name: str,
    start_t: float,
    end_t: float,
    bands: List[str],
    config: Any,
    logger: Any,
    spatial_modes: Optional[List[str]] = None,
    frequency_bands_override: Optional[Dict[str, List[float]]] = None,
    condition_labels: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Extract aperiodic and spectral features for a single segment.
    
    Features:
    - Aperiodic slope/offset (1/f)
    - Power-corrected band power
    - Alpha Peak Frequency (APF)
    - Theta/Beta Ratio (TBR)
    
    Outputs respect spatial_modes: 'channels', 'roi', 'global'
    
    Parameters
    ----------
    condition_labels : Optional[np.ndarray]
        If provided and subtract_evoked is enabled, subtract condition-specific
        evoked responses instead of grand average.
    """
    if spatial_modes is None:
        spatial_modes = ['roi', 'global']
    
    # Build ROI map if needed
    roi_map = {}
    if 'roi' in spatial_modes:
        roi_map = _build_roi_map(ch_names, config)
    
    # Parse configuration
    psd_method, psd_kwargs, fmin, fmax = _parse_psd_config(config)
    line_config = _parse_line_noise_config(config)
    
    # Check if induced spectra (evoked subtraction) is requested
    aperiodic_cfg = config.get("feature_engineering.aperiodic", {}) if hasattr(config, "get") else {}
    subtract_evoked = bool(aperiodic_cfg.get("subtract_evoked", False))
    
    # Compute PSD (with optional evoked subtraction for induced spectra)
    psds, freqs = _compute_psd(
        epochs, picks, start_t, end_t, fmin, fmax,
        psd_method, psd_kwargs, logger, segment_name,
        subtract_evoked=subtract_evoked,
        condition_labels=condition_labels,
    )
    
    # Apply line noise exclusion to PSD data (if configured)
    n_removed = 0
    if line_config.exclude and freqs.size > 0:
        line_noise_mask_psd = _build_line_noise_mask(freqs, line_config)
        n_removed = int(np.sum(~line_noise_mask_psd))
        
        if n_removed > 0 and np.any(line_noise_mask_psd):
            freqs = freqs[line_noise_mask_psd]
            psds = psds[..., line_noise_mask_psd]
    
    # Transform to log space
    log_freqs = np.log10(freqs)
    log_psd = np.log10(np.maximum(psds, _MIN_POWER_LOG10))
    
    # Parse fit parameters
    model = str(config.get("feature_engineering.aperiodic.model", "fixed")).strip().lower()
    if model not in {"fixed", "knee"}:
        model = "fixed"
    
    peak_z = float(config.get("feature_engineering.aperiodic.peak_rejection_z", _DEFAULT_PEAK_REJECTION_Z))
    min_pts = int(config.get("feature_engineering.aperiodic.min_fit_points", _DEFAULT_MIN_FIT_POINTS))
    fit_params = _validate_fit_parameters(peak_z, min_pts, model)
    
    n_jobs = get_n_jobs(config, default=-1, config_path="feature_engineering.parallel.n_jobs_aperiodic")
    
    # Build line-noise exclusion mask for aperiodic fitting
    line_noise_mask = None
    if line_config.exclude:
        line_noise_mask = _build_line_noise_mask(freqs, line_config)
        if logger and np.any(~line_noise_mask):
            n_excluded = int(np.sum(~line_noise_mask))
            logger.debug("Aperiodic: excluding %d line-noise bins from fitting", n_excluded)
    
    # Fit aperiodic model
    offsets, slopes, valid_bins, kept_bins, peak_rej, fit_masks, fit_qc = _fit_aperiodic_with_qc(
        log_freqs, log_psd, fit_params, logger,
        n_jobs=n_jobs, line_noise_mask=line_noise_mask
    )
    
    # Compute residuals
    knees = None
    if fit_params.model == "knee":
        knees = fit_qc.get("knees") if isinstance(fit_qc, dict) else None
        if knees is None:
            knees = np.full_like(offsets, np.nan)
        residuals = _compute_knee_residuals(freqs, log_psd, offsets, slopes, knees)
    else:
        residuals = compute_residuals(log_freqs, log_psd, offsets, slopes)
    
    # Compute fit quality metrics using the correct model
    r2, rms = _compute_fit_r2_and_rms(
        log_freqs, log_psd, offsets, slopes, fit_masks,
        model=fit_params.model, knees=knees, freqs_hz=freqs
    )
    
    # Apply quality filters
    min_r2 = float(config.get("feature_engineering.aperiodic.min_r2", 0.0))
    if not np.isfinite(min_r2):
        min_r2 = 0.0
    
    max_rms = config.get("feature_engineering.aperiodic.max_rms", None)
    if max_rms is not None:
        try:
            max_rms = float(max_rms)
        except Exception:
            max_rms = None
    if max_rms is not None and not np.isfinite(max_rms):
        max_rms = None
    
    fit_ok = np.isfinite(r2)
    if min_r2 > 0:
        fit_ok &= (r2 >= min_r2)
    if max_rms is not None:
        fit_ok &= (np.isfinite(rms) & (rms <= max_rms))
    
    # Prepare feature matrices
    n_epochs, n_channels = psds.shape[:2]
    
    metrics: Dict[str, Tuple[str, str, np.ndarray]] = {}
    
    if fit_params.model == "knee":
        exponent = (-slopes).copy()
        metrics["exponent"] = ("broadband", "exponent", exponent)
        knees = fit_qc.get("knees") if isinstance(fit_qc, dict) else None
        if isinstance(knees, np.ndarray) and knees.shape == exponent.shape:
            metrics["knee"] = ("broadband", "knee", knees.copy())
        metrics["slope"] = ("broadband", "slope", np.full_like(exponent, np.nan))
    else:
        metrics["slope"] = ("broadband", "slope", slopes.copy())
    
    metrics["offset"] = ("broadband", "offset", offsets.copy())
    metrics["r2"] = ("broadband", "r2", r2.copy())
    metrics["rms"] = ("broadband", "rms", rms.copy())
    
    # Apply fit_ok mask to all aperiodic parameters
    mask_metrics = ["slope", "offset"]
    if fit_params.model == "knee":
        mask_metrics.extend(["exponent", "knee"])
    for metric_name in mask_metrics:
        if metric_name in metrics:
            matrix = metrics[metric_name][2]
            matrix[~fit_ok] = np.nan
    
    # Get frequency bands for metrics
    freq_bands = frequency_bands_override or get_frequency_bands_for_aperiodic(config)
    theta_range = freq_bands.get("theta", (4.0, 8.0))
    beta_range = freq_bands.get("beta", (13.0, 30.0))
    alpha_range = freq_bands.get("alpha", (8.0, 13.0))
    
    # Compute APF and TBR
    apf_matrix = _compute_alpha_peak_frequency(freqs, residuals, alpha_range, fit_ok)
    tbr_matrix = _compute_theta_beta_ratio(freqs, residuals, theta_range, beta_range, fit_ok)
    
    metrics["peakfreq"] = ("alpha", "peakfreq", apf_matrix)
    metrics["tbr"] = ("broadband", "tbr", tbr_matrix)
    
    # Compute power-corrected band power per band with coverage validation
    band_coverage_info: Dict[str, float] = {}
    for band_name in bands:
        if band_name not in freq_bands:
            continue
        
        band_range = freq_bands[band_name]
        
        # Validate band coverage before computing features
        is_valid, coverage = _validate_band_coverage(freqs, band_name, band_range, logger)
        band_coverage_info[band_name] = coverage
        
        if not is_valid:
            # Band completely outside PSD range - skip with NaN matrix
            pc_matrix = np.full((n_epochs, n_channels), np.nan)
        else:
            pc_matrix = _compute_power_corrected_band_power(freqs, residuals, band_name, band_range, fit_ok)
        
        metrics[f"{band_name}_powcorr"] = (band_name, "powcorr", pc_matrix)
    
    # Aggregate features by spatial mode
    data_dict = _aggregate_features_by_spatial_mode(
        metrics, ch_names, segment_name, spatial_modes, roi_map
    )
    
    # Add QC metadata with proper per-trial structure
    data_dict["__qc__"] = {
        "segment": segment_name,
        "freqs": freqs,
        "log_freqs": log_freqs,
        "psd_fmin": float(freqs.min()) if freqs.size > 0 else np.nan,
        "psd_fmax": float(freqs.max()) if freqs.size > 0 else np.nan,
        "line_noise_excluded": bool(line_config.exclude),
        "line_noise_bins_removed": int(n_removed),
        "slopes": slopes,
        "offsets": offsets,
        "r2": r2,
        "rms": rms,
        "fit_ok": fit_ok,
        "min_r2": float(min_r2),
        "max_rms": float(max_rms) if max_rms is not None else None,
        "fit_ok_fraction": float(np.nanmean(fit_ok)) if fit_ok.size else np.nan,
        "n_valid_per_trial": np.sum(fit_ok, axis=1) if fit_ok.ndim == 2 else None,
        "valid_bins": valid_bins,
        "kept_bins": kept_bins,
        "peak_rejected": peak_rej,
        "channel_names": ch_names,
        "band_coverage": band_coverage_info,
    }
    
    return data_dict


# Public API functions
def extract_aperiodic_features(
    ctx: Any,
    bands: List[str],
) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
    """Extract aperiodic features from FeatureContext."""
    valid, err = validate_extractor_inputs(ctx, "Aperiodic", min_epochs=2)
    if not valid:
        ctx.logger.warning(err)
        return pd.DataFrame(), [], {}
    
    epochs = ctx.epochs
    picks, ch_names = pick_eeg_channels(epochs)
    if len(picks) == 0:
        ctx.logger.warning("Aperiodic: No EEG channels available; skipping extraction.")
        return pd.DataFrame(), [], {}
    
    config = ctx.config
    logger = ctx.logger
    freq_bands_override = getattr(ctx, "frequency_bands", None)
    sfreq = epochs.info["sfreq"]
    times = epochs.times
    min_samples = int(sfreq)
    
    # Scientific validity: aperiodic fits are unstable on short segments
    aperiodic_cfg = config.get("feature_engineering.aperiodic", {}) if hasattr(config, "get") else {}
    min_segment_sec = float(aperiodic_cfg.get("min_segment_sec", _DEFAULT_MIN_SEGMENT_SEC))
    if not np.isfinite(min_segment_sec) or min_segment_sec < 0:
        min_segment_sec = _DEFAULT_MIN_SEGMENT_SEC
    
    all_data: Dict[str, Any] = {}
    qc_payload: Dict[str, Any] = {
        "segments": {},
        "channel_names": ch_names,
        "min_segment_sec": min_segment_sec,
    }
    
    from eeg_pipeline.utils.analysis.windowing import get_segment_masks
    
    windows = ctx.windows
    target_name = getattr(ctx, "name", None)
    
    # Always derive mask from windows - never use np.ones() blindly
    if target_name and windows is not None:
        mask = windows.get_mask(target_name)
        if mask is not None and np.any(mask):
            segments = {target_name: mask}
        else:
            logger.warning(
                "Aperiodic: targeted window '%s' has no valid mask; using full epoch.",
                target_name,
            )
            segments = {target_name: np.ones_like(times, dtype=bool)}
    else:
        segments = get_segment_masks(times, windows, config)
    
    for seg_name, mask in segments.items():
        if mask is None or np.sum(mask) < min_samples:
            continue
        
        t_seg = times[mask]
        seg_duration_sec = float(t_seg[-1] - t_seg[0]) if len(t_seg) > 1 else 0.0
        
        # Validate segment duration for stable aperiodic fits
        if seg_duration_sec < min_segment_sec:
            logger.warning(
                "Aperiodic: segment '%s' duration (%.2fs) is shorter than min_segment_sec (%.2fs); "
                "skipping to avoid unstable slope/offset estimates.",
                seg_name, seg_duration_sec, min_segment_sec
            )
            continue
        
        spatial_modes = getattr(ctx, 'spatial_modes', ['roi', 'global'])
        seg_data = _extract_aperiodic_for_segment(
            epochs, picks, ch_names, seg_name,
            t_seg[0], t_seg[-1], bands, config, logger,
            spatial_modes=spatial_modes,
            frequency_bands_override=freq_bands_override,
        )
        qc_payload["segments"][seg_name] = seg_data.get("__qc__")
        seg_data.pop("__qc__", None)
        all_data.update(seg_data)
        logger.info(f"Computed Aperiodic for {seg_name}: [{t_seg[0]:.2f}, {t_seg[-1]:.2f}] ({seg_duration_sec:.2f}s)")
    
    if not all_data:
        logger.warning("No valid segments for Aperiodic; returning empty result.")
        return pd.DataFrame(), [], {}
    
    df = pd.DataFrame(all_data)
    
    segments_done = sorted([k for k, v in qc_payload.get("segments", {}).items() if v])
    qc_payload["segments_computed"] = segments_done
    
    # Pick first available segment for shared QC fields
    chosen_name = segments_done[0] if segments_done else None
    chosen = qc_payload.get("segments", {}).get(chosen_name) if chosen_name else None
    
    if chosen:
        qc_payload["freqs"] = chosen.get("freqs")
        qc_payload["residual_mean"] = chosen.get("residual_mean")
        qc_payload["r2"] = chosen.get("r2")
        qc_payload["slopes"] = chosen.get("slopes")
        qc_payload["offsets"] = chosen.get("offsets")
        qc_payload["channel_names"] = chosen.get("channel_names")
    
    return df, list(df.columns), qc_payload



