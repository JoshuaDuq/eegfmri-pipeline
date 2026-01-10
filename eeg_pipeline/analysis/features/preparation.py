"""
Precomputation Module
=====================

Functions for precomputing expensive intermediate data (bands, PSD, GFP)
shared across multiple feature extractors.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import numpy as np
import mne

from eeg_pipeline.types import PrecomputedData, PrecomputedQC, TimeWindows
from eeg_pipeline.utils.analysis.windowing import TimeWindowSpec, time_windows_from_spec
from eeg_pipeline.utils.analysis.spectral import compute_band_data, compute_psd
from eeg_pipeline.utils.analysis.signal_metrics import compute_gfp
from eeg_pipeline.utils.config.loader import get_frequency_bands


###################################################################
# Constants
###################################################################

DEFAULT_CSD_LAMBDA2 = 1e-5
DEFAULT_CSD_STIFFNESS = 4.0
DEFAULT_ALPHA_FMIN = 7.0
DEFAULT_ALPHA_FMAX = 13.0
DEFAULT_ALPHA_WIDTH_HZ = 2.0
DEFAULT_IAF_MIN_PROMINENCE = 0.05
DEFAULT_PSD_FMIN_OFFSET = 4.0
DEFAULT_PSD_FMAX_OFFSET = 0.5
MIN_FREQ_FOR_PSD = 1.0
MAX_FREQ_FOR_PSD = 40.0
MIN_FIT_POINTS = 10
MIN_ALPHA_FMIN = 6.0
MAX_ALPHA_FMAX = 14.0
MIN_THETA_FMIN = 3.0
MIN_THETA_FMAX = 4.0
DEFAULT_BETA_FMIN = 13.0
DEFAULT_BETA_FMAX = 30.0


###################################################################
# Helper Functions
###################################################################


def _compute_single_band(
    data: np.ndarray,
    sfreq: float,
    band_name: str,
    fmin: float,
    fmax: float,
    *,
    pad_sec: float,
    pad_cycles: float,
) -> Optional[Tuple[str, Any, np.ndarray, dict]]:
    """
    Compute band data for a single band (parallel worker).
    
    Returns None on failure, or (band_name, band_data, gfp_band, qc_entry) on success.
    """
    try:
        band_data = compute_band_data(
            data,
            sfreq,
            band_name,
            fmin,
            fmax,
            logger=None,
            pad_sec=pad_sec,
            pad_cycles=pad_cycles,
        )
        if band_data is None:
            return None
        
        gfp_band = compute_gfp(band_data.filtered)
        band_power = band_data.power
        
        qc_entry = _create_band_qc_entry(band_power, fmin, fmax)
        return band_name, band_data, gfp_band, qc_entry
    except Exception as exc:
        return None


def _create_band_qc_entry(band_power: np.ndarray, fmin: float, fmax: float) -> dict:
    """Create QC entry for a frequency band."""
    if band_power.size == 0:
        return {}
    
    finite_count = np.isfinite(band_power).sum()
    finite_fraction = float(finite_count / band_power.size)
    median_power = float(np.nanmedian(band_power))
    
    return {
        "finite_fraction": finite_fraction,
        "median_power": median_power,
        "fmin": fmin,
        "fmax": fmax,
    }


def _apply_spatial_transform(
    epochs: mne.Epochs,
    transform_type: str,
    config: Any,
    logger: Any,
) -> mne.Epochs:
    """
    Apply spatial transform (CSD/Laplacian) to epochs.
    
    Returns epochs unchanged if transform is 'none' or fails.
    """
    if transform_type not in {"csd", "laplacian"}:
        return epochs
    
    try:
        lambda2 = float(config.get("feature_engineering.spatial_transform_params.lambda2", DEFAULT_CSD_LAMBDA2))
        stiffness = float(config.get("feature_engineering.spatial_transform_params.stiffness", DEFAULT_CSD_STIFFNESS))
    except (ValueError, TypeError, AttributeError):
        lambda2 = DEFAULT_CSD_LAMBDA2
        stiffness = DEFAULT_CSD_STIFFNESS
    
    try:
        transformed = mne.preprocessing.compute_current_source_density(
            epochs,
            lambda2=lambda2,
            stiffness=stiffness,
            verbose=False,
        )
        if logger:
            logger.info(
                "Applied spatial transform=%s (CSD) to epochs for feature precomputation (lambda2=%s, stiffness=%s).",
                transform_type,
                lambda2,
                stiffness,
            )
        return transformed
    except Exception as exc:
        if logger:
            logger.warning(
                "Failed to apply spatial transform=%s; proceeding without it (%s).",
                transform_type,
                exc,
            )
        return epochs


def _get_spatial_transform_type(config: Any) -> str:
    """Extract and validate spatial transform type from config."""
    transform = str(config.get("feature_engineering.spatial_transform", "none")).strip().lower()
    if transform not in {"none", "csd", "laplacian"}:
        return "none"
    return transform


def _create_empty_precomputed_data(
    epochs: mne.Epochs,
    config: Any,
    logger: Any,
) -> PrecomputedData:
    """Create empty PrecomputedData when no EEG channels are available."""
    return PrecomputedData(
        data=np.array([]),
        times=epochs.times,
        sfreq=float(epochs.info["sfreq"]),
        ch_names=[],
        picks=np.array([], dtype=int),
        config=config,
        logger=logger,
    )


def _populate_basic_qc(precomputed: PrecomputedData, data: np.ndarray, sfreq: float) -> None:
    """Populate basic QC metrics for precomputed data."""
    if data.size == 0:
        return
    
    precomputed.qc.data_finite_fraction = float(np.isfinite(data).sum() / data.size)
    precomputed.qc.n_epochs = data.shape[0]
    precomputed.qc.n_channels = data.shape[1]
    precomputed.qc.n_times = data.shape[2]
    precomputed.qc.sfreq = sfreq


def _compute_time_windows(
    times: np.ndarray,
    sfreq: float,
    config: Any,
    logger: Any,
    windows_spec: Any,
) -> Tuple[Optional[TimeWindows], dict]:
    """
    Compute time windows for feature extraction.
    
    Returns (windows, qc_dict). windows is None on failure.
    """
    try:
        if isinstance(windows_spec, TimeWindows):
            windows = windows_spec
        else:
            if windows_spec is None:
                windows_spec = TimeWindowSpec(
                    times=times,
                    config=config,
                    sampling_rate=sfreq,
                    logger=logger,
                )
            windows = time_windows_from_spec(
                windows_spec,
                logger=logger,
                strict=True,
            )
        
        qc_dict = {
            "window_names": list(windows.masks.keys()),
            "clamped": getattr(windows, "clamped", False),
            "errors": list(getattr(windows, "errors", [])),
        }
        
        # Add sample counts for each window to QC
        for name, mask in windows.masks.items():
            qc_dict[f"{name}_samples"] = int(np.sum(mask))
            if name in windows.ranges:
                qc_dict[f"{name}_range"] = windows.ranges[name]

        if logger and windows:
            # Determine suitable descriptive names for logging
            target_name = getattr(windows, "name", None)
            available = ", ".join(windows.masks.keys())
            logger.info(
                "Computed time windows (target=%s). Available windows: %s (clamped=%s)",
                target_name or "none",
                available,
                getattr(windows, "clamped", False),
            )
        
        return windows, qc_dict
    except ValueError as exc:
        qc_dict = {
            "baseline_samples": 0,
            "active_samples": 0,
            "baseline_range": (np.nan, np.nan),
            "active_range": (np.nan, np.nan),
            "clamped": False,
            "errors": [str(exc)],
        }
        return None, qc_dict


def _compute_gfp_with_qc(data: np.ndarray) -> Tuple[np.ndarray, dict]:
    """Compute GFP and associated QC metrics."""
    gfp = compute_gfp(data)
    
    if gfp.size == 0:
        return gfp, {}
    
    finite_count = np.isfinite(gfp).sum()
    finite_fraction = float(finite_count / gfp.size)
    median_gfp = float(np.nanmedian(gfp))
    
    qc_dict = {
        "finite_fraction": finite_fraction,
        "median": median_gfp,
    }
    
    return gfp, qc_dict


def _estimate_individual_alpha_frequency(
    data: np.ndarray,
    sfreq: float,
    ch_names: List[str],
    baseline_mask: Optional[np.ndarray],
    config: Any,
) -> Optional[float]:
    """
    Estimate individual alpha frequency (IAF) from baseline PSD.
    
    Returns IAF in Hz, or None if estimation fails.
    """
    from scipy.signal import find_peaks
    from eeg_pipeline.utils.analysis.spatial import get_roi_definitions
    from eeg_pipeline.utils.analysis.channels import build_roi_map
    
    if not hasattr(config, "get"):
        return None
    
    iaf_config = config.get("feature_engineering.bands", {})
    alpha_range = iaf_config.get("iaf_search_range_hz", [DEFAULT_ALPHA_FMIN, DEFAULT_ALPHA_FMAX])
    alpha_fmin = float(alpha_range[0])
    alpha_fmax = float(alpha_range[1])
    min_prominence = float(iaf_config.get("iaf_min_prominence", DEFAULT_IAF_MIN_PROMINENCE))
    
    roi_names = iaf_config.get(
        "iaf_rois",
        ["ParOccipital_Midline", "ParOccipital_Ipsi_L", "ParOccipital_Contra_R"]
    )
    roi_definitions = get_roi_definitions(config)
    roi_map = build_roi_map(ch_names, roi_definitions) if roi_definitions else {}
    
    roi_indices = _get_roi_channel_indices(roi_names, roi_map)
    
    baseline_data = _extract_baseline_data(data, baseline_mask)
    
    psd_fmin = max(MIN_FREQ_FOR_PSD, alpha_fmin - DEFAULT_PSD_FMIN_OFFSET)
    psd_fmax = min(MAX_FREQ_FOR_PSD, sfreq / 2.0 - DEFAULT_PSD_FMAX_OFFSET)
    
    psds, freqs = mne.time_frequency.psd_array_multitaper(
        baseline_data,
        sfreq=sfreq,
        fmin=psd_fmin,
        fmax=psd_fmax,
        adaptive=True,
        normalization="full",
        verbose=False,
    )
    psds = np.asarray(psds, dtype=float)
    freqs = np.asarray(freqs, dtype=float)
    
    if psds.ndim != 3 or freqs.size == 0:
        return None
    
    channel_indices = roi_indices if roi_indices else list(range(psds.shape[1]))
    mean_psd = np.nanmean(psds[:, channel_indices, :], axis=(0, 1))
    
    residual_psd = _remove_one_over_f_trend(freqs, mean_psd)
    iaf = _find_alpha_peak(freqs, residual_psd, alpha_fmin, alpha_fmax, min_prominence)
    
    return iaf if np.isfinite(iaf) else None


def _get_roi_channel_indices(roi_names: List[str], roi_map: dict) -> List[int]:
    """Extract channel indices for specified ROI names."""
    if not isinstance(roi_names, (list, tuple)) or not roi_map:
        return []
    
    indices = []
    for roi_name in roi_names:
        channel_indices = roi_map.get(str(roi_name), [])
        indices.extend(channel_indices)
    
    unique_indices = sorted(set(int(idx) for idx in indices if idx is not None))
    return unique_indices


def _extract_baseline_data(data: np.ndarray, baseline_mask: Optional[np.ndarray]) -> np.ndarray:
    """Extract baseline data using mask, or return full data if mask unavailable."""
    if baseline_mask is not None and np.any(baseline_mask):
        return data[:, :, baseline_mask]
    return data


def _remove_one_over_f_trend(freqs: np.ndarray, psd: np.ndarray) -> np.ndarray:
    """
    Remove 1/f trend from PSD using robust linear fit.
    
    Fits log-log space in 2-40 Hz range.
    """
    log_freqs = np.log10(np.maximum(freqs, 1e-6))
    log_psd = np.log10(np.maximum(psd, 1e-20))
    
    fit_mask = (freqs >= 2.0) & (freqs <= MAX_FREQ_FOR_PSD) & np.isfinite(log_psd)
    
    if np.sum(fit_mask) < MIN_FIT_POINTS:
        return log_psd
    
    slope, intercept = np.polyfit(log_freqs[fit_mask], log_psd[fit_mask], 1)
    residual = log_psd - (intercept + slope * log_freqs)
    
    return residual


def _find_alpha_peak(
    freqs: np.ndarray,
    residual_psd: np.ndarray,
    alpha_fmin: float,
    alpha_fmax: float,
    min_prominence: float,
) -> float:
    """
    Find alpha peak frequency in residual PSD.
    
    Uses peak detection first, falls back to weighted mean if no peaks found.
    """
    from scipy.signal import find_peaks
    
    alpha_mask = (freqs >= alpha_fmin) & (freqs <= alpha_fmax) & np.isfinite(residual_psd)
    
    if not np.any(alpha_mask):
        return np.nan
    
    alpha_residual = residual_psd[alpha_mask]
    peaks, properties = find_peaks(alpha_residual, prominence=min_prominence)
    
    if peaks.size > 0:
        prominences = properties.get("prominences", np.ones_like(peaks))
        best_peak_idx = int(peaks[np.argmax(prominences)])
        alpha_freqs = freqs[alpha_mask]
        return float(alpha_freqs[best_peak_idx])
    
    positive_residual = np.maximum(alpha_residual, 0.0)
    residual_sum = float(np.sum(positive_residual))
    
    if residual_sum > 0:
        alpha_freqs = freqs[alpha_mask]
        weighted_mean = float(np.sum(alpha_freqs * positive_residual) / residual_sum)
        return weighted_mean
    
    return np.nan


def _adjust_frequency_bands_for_iaf(
    frequency_bands: dict,
    iaf: float,
    config: Any,
) -> dict:
    """
    Adjust frequency band definitions based on estimated IAF.
    
    Modifies alpha, theta, and beta bands relative to IAF.
    """
    adjusted_bands = dict(frequency_bands)
    
    if not hasattr(config, "get"):
        return adjusted_bands
    
    iaf_config = config.get("feature_engineering.bands", {})
    alpha_width = float(iaf_config.get("alpha_width_hz", DEFAULT_ALPHA_WIDTH_HZ))
    
    alpha_min = max(MIN_ALPHA_FMIN, iaf - alpha_width)
    alpha_max = min(MAX_ALPHA_FMAX, iaf + alpha_width)
    adjusted_bands["alpha"] = [alpha_min, alpha_max]
    
    adjusted_bands.setdefault("theta", [4.0, 8.0])
    adjusted_bands.setdefault("beta", [DEFAULT_BETA_FMIN, DEFAULT_BETA_FMAX])
    
    theta_max = max(MIN_THETA_FMAX, alpha_min)
    theta_min = max(MIN_THETA_FMIN, iaf - 6.0)
    adjusted_bands["theta"] = [theta_min, theta_max]
    
    beta_min_default = float(adjusted_bands["beta"][0]) if len(adjusted_bands["beta"]) >= 2 else DEFAULT_BETA_FMIN
    beta_max_default = float(adjusted_bands["beta"][1]) if len(adjusted_bands["beta"]) >= 2 else DEFAULT_BETA_FMAX
    beta_min = max(beta_min_default, alpha_max)
    adjusted_bands["beta"] = [beta_min, beta_max_default]
    
    return adjusted_bands


def _compute_bands_parallel_or_sequential(
    data: np.ndarray,
    sfreq: float,
    band_definitions: List[Tuple[str, Tuple[float, float]]],
    config: Any,
    logger: Any,
) -> List[Optional[Tuple[str, Any, np.ndarray, dict]]]:
    """
    Compute band data in parallel or sequential mode based on config.
    
    Returns list of results (band_name, band_data, gfp_band, qc_entry) or None.
    """
    n_jobs = int(config.get("feature_engineering.parallel.n_jobs_bands", -1))
    pad_sec = float(config.get("feature_engineering.band_envelope.pad_sec", 0.5))
    pad_cycles = float(config.get("feature_engineering.band_envelope.pad_cycles", 3.0))
    
    if n_jobs == 1:
        return _compute_bands_sequential(data, sfreq, band_definitions, pad_sec, pad_cycles)
    
    return _compute_bands_parallel(data, sfreq, band_definitions, pad_sec, pad_cycles, n_jobs, logger)


def _compute_bands_parallel(
    data: np.ndarray,
    sfreq: float,
    band_definitions: List[Tuple[str, Tuple[float, float]]],
    pad_sec: float,
    pad_cycles: float,
    n_jobs: int,
    logger: Any,
) -> List[Optional[Tuple[str, Any, np.ndarray, dict]]]:
    """Compute bands in parallel using joblib."""
    try:
        from joblib import Parallel, delayed
        
        results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_compute_single_band)(
                data,
                sfreq,
                band_name,
                fmin,
                fmax,
                pad_sec=pad_sec,
                pad_cycles=pad_cycles,
            )
            for band_name, (fmin, fmax) in band_definitions
        )
        return results
    except Exception as exc:
        if logger:
            logger.warning(
                "Parallel band computation failed (%s); falling back to sequential.", exc
            )
        return _compute_bands_sequential(data, sfreq, band_definitions, pad_sec, pad_cycles)


def _compute_bands_sequential(
    data: np.ndarray,
    sfreq: float,
    band_definitions: List[Tuple[str, Tuple[float, float]]],
    pad_sec: float,
    pad_cycles: float,
) -> List[Optional[Tuple[str, Any, np.ndarray, dict]]]:
    """Compute bands sequentially."""
    return [
        _compute_single_band(
            data,
            sfreq,
            band_name,
            fmin,
            fmax,
            pad_sec=pad_sec,
            pad_cycles=pad_cycles,
        )
        for band_name, (fmin, fmax) in band_definitions
    ]


def _process_band_results(
    results: List[Optional[Tuple[str, Any, np.ndarray, dict]]],
    precomputed: PrecomputedData,
    logger: Any,
) -> None:
    """Process band computation results and populate precomputed data."""
    for result in results:
        if result is None:
            continue
        
        if len(result) == 2 and result[0] is None:
            if logger:
                logger.warning(f"Band computation failed: {result[1]}")
            continue
        
        band_name, band_data, gfp_band, qc_entry = result
        precomputed.band_data[band_name] = band_data
        precomputed.gfp_band[band_name] = gfp_band
        if qc_entry:
            precomputed.qc.bands[band_name] = qc_entry


def _compute_psd_with_qc(
    data: np.ndarray,
    sfreq: float,
    baseline_mask: Optional[np.ndarray],
    config: Any,
    logger: Any,
) -> Tuple[Optional[Any], dict]:
    """
    Compute PSD and associated QC metrics.
    
    Returns (psd_data, qc_dict). psd_data is None on failure.
    """
    psd_input = data
    window_type = "full"
    
    # Try to use targeted window if provided, else baseline
    if baseline_mask is not None and isinstance(baseline_mask, np.ndarray) and np.any(baseline_mask):
        psd_input = data[:, :, baseline_mask]
        
        # Identify the name for tagging purposes
        if hasattr(config, "get") and config.get("name"):
            window_type = config.get("name")
        else:
            window_type = "baseline"
    
    psd_data = compute_psd(psd_input, sfreq, config=config, logger=logger)
    
    if psd_data is None:
        return None, {}
    
    psd_array = psd_data.psd
    finite_count = np.isfinite(psd_array).sum()
    finite_fraction = float(finite_count / psd_array.size)
    
    freq_range = (np.nan, np.nan)
    if len(psd_data.freqs) > 1:
        freq_range = (float(psd_data.freqs[0]), float(psd_data.freqs[-1]))
    
    qc_dict = {
        "n_freq_bins": int(len(psd_data.freqs)),
        "finite_fraction": finite_fraction,
        "window": window_type,
        "freq_range": freq_range,
    }
    
    return psd_data, qc_dict


###################################################################
# Main Function
###################################################################


def precompute_data(
    epochs: mne.Epochs,
    bands: List[str],
    config: Any,
    logger: Any,
    *,
    compute_bands: bool = True,
    compute_psd_data: bool = True,
    windows_spec: Any = None,
    frequency_bands_override: Any = None,
) -> PrecomputedData:
    """
    Precompute all intermediate data needed by feature extraction modules.
    
    Call this once at the start, then pass the result to feature functions.
    
    Parameters
    ----------
    epochs : mne.Epochs
        Input epochs
    bands : List[str]
        Frequency bands to precompute
    config : Any
        Configuration object
    logger : Any
        Logger instance
    compute_bands : bool
        Whether to precompute band-filtered data
    compute_psd_data : bool
        Whether to precompute PSD
    windows_spec : Any, optional
        Time window specification
    frequency_bands_override : Any, optional
        Override frequency band definitions
        
    Returns
    -------
    PrecomputedData
        Container with all precomputed data
    """
    picks = mne.pick_types(
        epochs.info,
        eeg=True,
        meg=False,
        eog=False,
        stim=False,
        exclude="bads"
    )
    
    if len(picks) == 0:
        if logger:
            logger.warning("No EEG channels available")
        return _create_empty_precomputed_data(epochs, config, logger)
    
    transform_type = _get_spatial_transform_type(config)
    epochs_picked = epochs.copy().pick(picks)
    epochs_transformed = _apply_spatial_transform(
        epochs_picked,
        transform_type,
        config,
        logger,
    )
    
    data = epochs_transformed.get_data()
    times = epochs_transformed.times
    sfreq = float(epochs_transformed.info["sfreq"])
    ch_names = list(epochs_transformed.ch_names)
    
    precomputed = PrecomputedData(
        data=data,
        times=times,
        sfreq=sfreq,
        ch_names=ch_names,
        picks=picks,
        config=config,
        logger=logger,
    )
    
    _populate_basic_qc(precomputed, data, sfreq)
    
    windows, windows_qc = _compute_time_windows(
        times,
        sfreq,
        config,
        logger,
        windows_spec,
    )
    precomputed.windows = windows
    precomputed.qc.time_windows = windows_qc
    
    if windows is None and logger:
        error_msg = windows_qc.get("errors", ["Unknown error"])[0]
        precomputed.qc.errors.append(f"time_windows: {error_msg}")
        logger.error(
            "Time window computation failed; downstream features will be skipped: %s",
            error_msg,
        )
    
    gfp, gfp_qc = _compute_gfp_with_qc(data)
    precomputed.gfp = gfp
    precomputed.qc.gfp = gfp_qc
    
    frequency_bands = _determine_frequency_bands(
        config,
        frequency_bands_override,
        data,
        sfreq,
        ch_names,
        precomputed.windows,
        precomputed.qc,
        logger,
    )
    precomputed.frequency_bands = frequency_bands
    
    if compute_bands and bands:
        _compute_and_store_bands(
            data,
            sfreq,
            bands,
            frequency_bands,
            config,
            precomputed,
            logger,
        )
    
    if compute_psd_data:
        baseline_mask = (
            precomputed.windows.baseline_mask
            if precomputed.windows is not None
            else None
        )
        psd_data, psd_qc = _compute_psd_with_qc(
            data,
            sfreq,
            baseline_mask,
            config,
            logger,
        )
        precomputed.psd_data = psd_data
        precomputed.qc.psd = psd_qc
        
        if psd_data is not None and logger:
            logger.info(
                "Precomputed PSD: %d freq bins",
                len(psd_data.freqs),
            )
    
    return precomputed


def _determine_frequency_bands(
    config: Any,
    frequency_bands_override: Any,
    data: np.ndarray,
    sfreq: float,
    ch_names: List[str],
    windows: Optional[TimeWindows],
    qc: PrecomputedQC,
    logger: Any,
) -> Optional[dict]:
    """
    Determine frequency band definitions, optionally using IAF estimation.
    
    Returns frequency bands dict or None.
    """
    base_bands = frequency_bands_override or get_frequency_bands(config)
    frequency_bands = dict(base_bands) if isinstance(base_bands, dict) else {}
    
    use_iaf = (
        hasattr(config, "get") and
        bool(config.get("feature_engineering.bands.use_iaf", False))
    )
    
    if not use_iaf or windows is None:
        return frequency_bands if frequency_bands else None
    
    try:
        baseline_mask = getattr(windows, "baseline_mask", None)
        iaf = _estimate_individual_alpha_frequency(
            data,
            sfreq,
            ch_names,
            baseline_mask,
            config,
        )
        
        if iaf is not None:
            qc.time_windows["iaf_hz"] = float(iaf)
            frequency_bands = _adjust_frequency_bands_for_iaf(
                frequency_bands,
                iaf,
                config,
            )
    except Exception as exc:
        if logger:
            logger.warning(
                "IAF estimation failed; using config bands (%s).",
                exc,
            )
    
    return frequency_bands if frequency_bands else None


def _compute_and_store_bands(
    data: np.ndarray,
    sfreq: float,
    bands: List[str],
    frequency_bands: Optional[dict],
    config: Any,
    precomputed: PrecomputedData,
    logger: Any,
) -> None:
    """Compute and store band data for specified bands."""
    if not frequency_bands:
        frequency_bands = get_frequency_bands(config)
    
    band_definitions = [
        (band, frequency_bands[band])
        for band in bands
        if band in frequency_bands
    ]
    
    if not band_definitions:
        if logger:
            logger.warning(
                "No valid bands found in config; skipping band precomputation."
            )
        return
    
    results = _compute_bands_parallel_or_sequential(
        data,
        sfreq,
        band_definitions,
        config,
        logger,
    )
    
    _process_band_results(results, precomputed, logger)
    
    if logger and precomputed.band_data:
        logger.info(
            "Precomputed band data for: %s",
            list(precomputed.band_data.keys()),
        )
