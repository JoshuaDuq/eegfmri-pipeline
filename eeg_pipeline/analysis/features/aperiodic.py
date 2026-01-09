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
def _apply_peak_rejection(
    values: np.ndarray,
    peak_rejection_z: float,
    min_fit_points: int,
) -> Tuple[np.ndarray, bool]:
    """Apply robust peak rejection using MAD-based thresholding.
    
    Returns:
        Tuple of (keep_mask, peak_rejected_flag)
    """
    n_values = values.size
    keep_mask = np.ones(n_values, dtype=bool)
    peak_rejected = False
    
    mad = stats.median_abs_deviation(values, scale="normal", nan_policy="omit")
    median = np.median(values) if np.isfinite(values).any() else np.nan
    
    has_valid_stats = np.isfinite(mad) and mad > _MIN_MAD_THRESHOLD and np.isfinite(median)
    if has_valid_stats:
        threshold = median + peak_rejection_z * mad
        candidate_keep = values <= threshold
        n_kept = int(np.sum(candidate_keep))
        
        if n_kept >= min_fit_points:
            keep_mask = candidate_keep
            peak_rejected = bool(np.any(~candidate_keep))
    
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
    """Fit fixed aperiodic model to single epoch/channel."""
    finite_mask = np.isfinite(log_freqs) & np.isfinite(psd_vals)
    finite_indices = np.flatnonzero(finite_mask)
    valid_bins = int(finite_indices.size)
    
    if valid_bins < fit_params.min_fit_points:
        return PSDFitResult(
            epoch_idx, channel_idx, np.nan, np.nan,
            valid_bins, 0, False, np.array([], dtype=int), 1
        )
    
    psd_finite = psd_vals[finite_indices]
    keep_mask, peak_rejected = _apply_peak_rejection(
        psd_finite, fit_params.peak_rejection_z, fit_params.min_fit_points
    )
    
    kept_indices = finite_indices[keep_mask]
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
    """Fit knee aperiodic model to single epoch/channel."""
    finite_mask = np.isfinite(freqs_hz) & np.isfinite(log_psd_vals)
    finite_indices = np.flatnonzero(finite_mask)
    valid_bins = int(finite_indices.size)
    
    if valid_bins < fit_params.min_fit_points:
        return KneeFitResult(
            epoch_idx, channel_idx, np.nan, np.nan, np.nan,
            valid_bins, 0, False, np.array([], dtype=int), 1
        )
    
    log_psd_finite = log_psd_vals[finite_indices]
    keep_mask, peak_rejected = _apply_peak_rejection(
        log_psd_finite, fit_params.peak_rejection_z, fit_params.min_fit_points
    )
    
    kept_indices = finite_indices[keep_mask]
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
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute R² and RMS error for aperiodic fits."""
    n_epochs, n_channels, _ = log_psd.shape
    r2 = np.full((n_epochs, n_channels), np.nan)
    rms = np.full((n_epochs, n_channels), np.nan)
    
    for epoch_idx in range(n_epochs):
        for channel_idx in range(n_channels):
            mask = fit_masks[epoch_idx, channel_idx, :]
            offset = offsets[epoch_idx, channel_idx]
            
            if not np.any(mask) or not np.isfinite(offset):
                continue
            
            y_true = log_psd[epoch_idx, channel_idx, mask]
            y_pred = offset + slopes[epoch_idx, channel_idx] * log_freqs[mask]
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
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute power spectral density with fallback handling."""
    try:
        spectrum = epochs.compute_psd(
            method=psd_method,
            fmin=fmin,
            fmax=fmax,
            tmin=start_t,
            tmax=end_t,
            picks=picks,
            average=False,
            verbose=False,
            **psd_kwargs,
        )
    except Exception:
        # Fallback to Welch for broader compatibility across MNE versions
        try:
            spectrum = epochs.compute_psd(
                method="welch",
                fmin=fmin,
                fmax=fmax,
                tmin=start_t,
                tmax=end_t,
                picks=picks,
                average=False,
                verbose=False,
            )
        except Exception as e2:
            logger.warning(f"PSD computation failed for {segment_name}: {e2}")
            raise
    
    psds, freqs = spectrum.get_data(return_freqs=True)
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


def _compute_power_corrected_band_power(
    freqs: np.ndarray,
    residuals: np.ndarray,
    band_name: str,
    band_range: Tuple[float, float],
    fit_ok: np.ndarray,
) -> np.ndarray:
    """Compute power-corrected band power from residuals."""
    n_epochs, n_channels, _ = residuals.shape
    pc_matrix = np.full((n_epochs, n_channels), np.nan)
    
    band_mask = (freqs >= band_range[0]) & (freqs <= band_range[1])
    if not np.any(band_mask):
        return pc_matrix
    
    for channel_idx in range(n_channels):
        res_band = residuals[:, channel_idx, band_mask]
        ratio = np.power(10.0, res_band)
        pc_matrix[:, channel_idx] = np.mean(ratio, axis=1)
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


def _aggregate_features_by_spatial_mode(
    metrics: Dict[str, Tuple[str, str, np.ndarray]],
    ch_names: List[str],
    segment_name: str,
    spatial_modes: List[str],
    roi_map: Dict[str, List[int]],
) -> Dict[str, np.ndarray]:
    """Aggregate features according to spatial modes (channels, ROI, global)."""
    data_dict: Dict[str, np.ndarray] = {}
    
    for metric_name, (band, stat, matrix) in metrics.items():
        # Per-channel aggregation
        if 'channels' in spatial_modes:
            for channel_idx, channel_name in enumerate(ch_names):
                column_name = NamingSchema.build(
                    "aperiodic", segment_name, band, "ch", stat, channel=channel_name
                )
                data_dict[column_name] = matrix[:, channel_idx]
        
        # ROI aggregation
        if 'roi' in spatial_modes and roi_map:
            for roi_name, channel_indices in roi_map.items():
                if channel_indices and len(channel_indices) > 0:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        roi_values = np.nanmean(matrix[:, channel_indices], axis=1)
                    column_name = NamingSchema.build(
                        "aperiodic", segment_name, band, "roi", stat, channel=roi_name
                    )
                    data_dict[column_name] = roi_values
        
        # Global aggregation
        if 'global' in spatial_modes:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                global_values = np.nanmean(matrix, axis=1)
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
) -> Dict[str, np.ndarray]:
    """Extract aperiodic and spectral features for a single segment.
    
    Features:
    - Aperiodic slope/offset (1/f)
    - Power-corrected band power
    - Alpha Peak Frequency (APF)
    - Theta/Beta Ratio (TBR)
    
    Outputs respect spatial_modes: 'channels', 'roi', 'global'
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
    
    # Compute PSD
    psds, freqs = _compute_psd(
        epochs, picks, start_t, end_t, fmin, fmax,
        psd_method, psd_kwargs, logger, segment_name
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
    if fit_params.model == "knee":
        knees = fit_qc.get("knees") if isinstance(fit_qc, dict) else None
        if knees is None:
            knees = np.full_like(offsets, np.nan)
        residuals = _compute_knee_residuals(freqs, log_psd, offsets, slopes, knees)
    else:
        residuals = compute_residuals(log_freqs, log_psd, offsets, slopes)
    
    # Compute fit quality metrics
    r2, rms = _compute_fit_r2_and_rms(log_freqs, log_psd, offsets, slopes, fit_masks)
    
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
    
    # Apply fit_ok mask to slope and offset
    for metric_name in ["slope", "offset"]:
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
    
    # Compute power-corrected band power per band
    for band_name in bands:
        if band_name not in freq_bands:
            continue
        
        band_range = freq_bands[band_name]
        pc_matrix = _compute_power_corrected_band_power(freqs, residuals, band_name, band_range, fit_ok)
        metrics[f"{band_name}_powcorr"] = (band_name, "powcorr", pc_matrix)
    
    # Aggregate features by spatial mode
    data_dict = _aggregate_features_by_spatial_mode(
        metrics, ch_names, segment_name, spatial_modes, roi_map
    )
    
    # Add QC metadata
    data_dict["__qc__"] = {
        "segment": segment_name,
        "freqs": freqs,
        "log_freqs": log_freqs,
        "line_noise_excluded": bool(line_config.exclude),
        "line_noise_bins_removed": int(n_removed),
        "slopes": slopes,
        "offsets": offsets,
        "r2": r2,
        "rms": rms,
        "min_r2": float(min_r2),
        "max_rms": float(max_rms) if max_rms is not None else None,
        "fit_ok_fraction": float(np.nanmean(fit_ok)) if fit_ok.size else np.nan,
        "residual_mean": np.nanmean(residuals, axis=2) if residuals.ndim == 3 else None,
        "valid_bins": valid_bins,
        "kept_bins": kept_bins,
        "peak_rejected": peak_rej,
        "channel_names": ch_names,
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
    segments = get_segment_masks(times, ctx.windows, config)
    
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
    active_qc = qc_payload.get("segments", {}).get("active")
    baseline_qc = qc_payload.get("segments", {}).get("baseline")
    chosen = active_qc or baseline_qc
    if chosen:
        qc_payload["freqs"] = chosen.get("freqs")
        qc_payload["residual_mean"] = chosen.get("residual_mean")
        qc_payload["r2"] = chosen.get("r2")
        qc_payload["slopes"] = chosen.get("slopes")
        qc_payload["offsets"] = chosen.get("offsets")
        qc_payload["channel_names"] = chosen.get("channel_names")
    
    return df, list(df.columns), qc_payload


def extract_aperiodic_features_from_epochs(
    epochs: mne.Epochs,
    baseline_window: Tuple[float, float],
    bands: List[str],
    config: Any,
    logger: Any,
    *,
    events_df: Optional[pd.DataFrame] = None,
    frequency_bands_override: Optional[Dict[str, List[float]]] = None,
) -> Tuple[pd.DataFrame, List[str], Dict[str, Any]]:
    """Extract aperiodic features directly from epochs."""
    picks, ch_names = pick_eeg_channels(epochs)
    if len(picks) == 0:
        logger.warning("Aperiodic: No EEG channels available; skipping extraction.")
        return pd.DataFrame(), [], {}
    
    times = epochs.times
    sfreq = float(epochs.info["sfreq"])
    min_samples = int(sfreq)
    
    def _clamp_window(window: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        """Clamp window to valid time range."""
        if times.size == 0:
            return None
        start, end = float(window[0]), float(window[1])
        start = max(start, float(times[0]))
        end = min(end, float(times[-1]))
        if end <= start:
            return None
        return (start, end)
    
    baseline = _clamp_window(baseline_window)
    if baseline is None:
        return pd.DataFrame(), [], {"skipped_reason": "invalid_baseline_window"}
    
    from eeg_pipeline.utils.config.loader import get_config_value
    
    ramp_end = float(get_config_value(config, "feature_engineering.features.ramp_end", 3.0))
    active_window = get_config_value(config, "time_frequency_analysis.active_window", [3.0, 10.5])
    active = _clamp_window((float(active_window[0]), float(active_window[1])))
    ramp = _clamp_window((0.0, ramp_end))
    
    all_data: Dict[str, np.ndarray] = {}
    segments_done: List[str] = []
    
    baseline_data = _extract_aperiodic_for_segment(
        epochs, picks, ch_names, "baseline",
        baseline[0], baseline[1], bands, config, logger,
        frequency_bands_override=frequency_bands_override,
    )
    if baseline_data:
        all_data.update(baseline_data)
        segments_done.append("baseline")
    
    if ramp is not None:
        ramp_mask = (times >= ramp[0]) & (times <= ramp[1])
        if int(np.sum(ramp_mask)) >= min_samples:
            ramp_data = _extract_aperiodic_for_segment(
                epochs, picks, ch_names, "ramp",
                ramp[0], ramp[1], bands, config, logger,
                frequency_bands_override=frequency_bands_override,
            )
            if ramp_data:
                all_data.update(ramp_data)
                segments_done.append("ramp")
    
    if active is not None:
        active_mask = (times >= active[0]) & (times <= active[1])
        if int(np.sum(active_mask)) >= min_samples:
            active_data = _extract_aperiodic_for_segment(
                epochs, picks, ch_names, "active",
                active[0], active[1], bands, config, logger,
                frequency_bands_override=frequency_bands_override,
            )
            if active_data:
                all_data.update(active_data)
                segments_done.append("active")
    
    if not all_data:
        return pd.DataFrame(), [], {"skipped_reason": "empty_result"}
    
    df = pd.DataFrame(all_data)
    qc_payload = {
        "segments_computed": sorted(set(segments_done)),
        "baseline_window": (float(baseline[0]), float(baseline[1])),
        "active_window": (float(active[0]), float(active[1])) if active is not None else None,
    }
    return df, list(df.columns), qc_payload
