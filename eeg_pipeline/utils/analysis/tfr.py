from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import mne
import numpy as np
import pandas as pd

from ..config.loader import get_constants, get_config_value, ensure_config, get_frequency_bands
from eeg_pipeline.domain.features.naming import NamingSchema
from eeg_pipeline.utils.analysis.windowing import time_mask
from eeg_pipeline.utils.analysis.stats import (
    validate_baseline_window_pre_stimulus,
)


###################################################################
# Constants Loading
###################################################################

# Numerical constants
_PERCENT_TO_RATIO_DIVISOR = 100.0

def _get_tfr_constants(config=None):
    if config is None:
        return get_constants("time_frequency_analysis")
    return get_constants("time_frequency_analysis", config)


def _get_min_baseline_samples(config) -> int:
    """Get minimum baseline samples from config.
    
    Supports both min_baseline_samples (standard) and min_samples_for_baseline_validation (legacy).
    """
    val = get_config_value(config, "time_frequency_analysis.constants.min_baseline_samples", None)
    if val is None:
        val = get_config_value(config, "time_frequency_analysis.constants.min_samples_for_baseline_validation", 10)
    return int(val)


###################################################################
# Configuration Helpers
###################################################################

def get_tfr_config(config) -> Tuple[float, float, int, float, int, Union[str, list]]:
    """
    Parses TFR configuration from settings with fallback defaults.
    
    Returns:
        tuple: (freq_min, freq_max, n_freqs, n_cycles_factor, decim, picks)
    """
    tfr_config = config.get("time_frequency_analysis.tfr", {})

    freq_min = float(tfr_config.get("freq_min", 1.0))
    freq_max = float(tfr_config.get("freq_max", 100.0))
    n_freqs = int(tfr_config.get("n_freqs", 40))
    n_cycles_factor = float(tfr_config.get("n_cycles_factor", 2.0))
    decim = int(tfr_config.get("decim", 4))
    picks = tfr_config.get("picks", "eeg")
    
    return freq_min, freq_max, n_freqs, n_cycles_factor, decim, picks


def get_tfr_decim(config, mode: str = "power") -> int:
    """
    Get decimation factor for TFR based on mode.
    
    Parameters
    ----------
    config : Any
        Configuration object
    mode : str
        "power" for power TFR (can be aggressive) or "phase" for complex TFR (preserve time structure)
        
    Returns
    -------
    int
        Decimation factor
    """
    tfr_config = config.get("time_frequency_analysis.tfr", {})
    
    if mode == "phase":
        return int(tfr_config.get("decim_phase", 1))
    else:
        return int(tfr_config.get("decim_power", 4))


###################################################################
# TFR Computation Helpers
###################################################################


def filter_freqs_for_signal_length(
    freqs: np.ndarray,
    n_cycles: np.ndarray,
    sfreq: float,
    n_samples: int,
    logger: Optional[logging.Logger] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Filter frequencies whose wavelets would be longer than the signal.
    
    Morlet wavelet length is approximately: n_cycles / freq * sfreq * 2
    (the factor of 2 accounts for the full wavelet extent).
    
    Parameters
    ----------
    freqs : np.ndarray
        Frequency array
    n_cycles : np.ndarray
        Number of cycles per frequency
    sfreq : float
        Sampling frequency
    n_samples : int
        Number of samples in the signal
    logger : Optional[logging.Logger]
        Logger for warnings
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Filtered (freqs, n_cycles) arrays
    """
    wavelet_lengths = (n_cycles / freqs) * sfreq * 2
    valid_mask = wavelet_lengths < n_samples
    
    if not np.all(valid_mask):
        n_excluded = np.sum(~valid_mask)
        excluded_freqs = freqs[~valid_mask]
        if logger:
            logger.warning(
                f"Excluding {n_excluded} low frequencies (< {excluded_freqs.max():.2f} Hz) "
                f"whose wavelets exceed signal length ({n_samples} samples). "
                f"Consider using longer time windows for low-frequency analysis."
            )
        
        if not np.any(valid_mask):
            raise ValueError(
                f"All frequencies excluded: signal too short ({n_samples} samples) "
                f"for wavelet analysis. Minimum frequency {freqs.min():.2f} Hz requires "
                f"~{int(wavelet_lengths.min())} samples."
            )
    
    return freqs[valid_mask], n_cycles[valid_mask]


def compute_tfr_morlet(
    epochs: mne.Epochs,
    config,
    logger: Optional[logging.Logger] = None,
    freqs: Optional[np.ndarray] = None,
    picks: Optional[Union[str, List[int]]] = None,
    decim: Optional[int] = None,
) -> mne.time_frequency.EpochsTFR:
    """
    Compute TFR using Morlet wavelets with consistent pipeline parameters.
    
    Parameters
    ----------
    epochs : mne.Epochs
        Epochs to compute TFR for
    config : Any
        Configuration object
    logger : Optional[logging.Logger]
        Logger instance (optional)
    freqs : Optional[np.ndarray]
        Frequency array. If None, uses config defaults with logspace.
    picks : Optional[Union[str, List[int]]]
        Channel picks. If None, uses config defaults.
    decim : Optional[int]
        Decimation factor. If None, uses config decim_power for power TFR.
        
    Returns
    -------
    mne.time_frequency.EpochsTFR
        Computed TFR object
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    freq_min, freq_max, n_freqs, n_cycles_factor, _, tfr_picks = get_tfr_config(config)
    
    if freqs is None:
        freqs = np.logspace(np.log10(freq_min), np.log10(freq_max), n_freqs)
    if picks is None:
        picks = tfr_picks
    if decim is None:
        decim = get_tfr_decim(config, mode="power")
    
    n_cycles = compute_adaptive_n_cycles(freqs, cycles_factor=n_cycles_factor, config=config)
    
    n_samples = len(epochs.times)
    sfreq = epochs.info["sfreq"]
    freqs, n_cycles = filter_freqs_for_signal_length(
        freqs, n_cycles, sfreq, n_samples, logger
    )
    
    workers = resolve_tfr_workers(workers_default=int(config.get("time_frequency_analysis.tfr.workers", -1)))

    resolved_picks = _resolve_picks(epochs, picks) if isinstance(picks, str) else picks

    compute_kwargs = dict(
        method="morlet",
        freqs=freqs,
        n_cycles=n_cycles,
        decim=decim,
        picks=resolved_picks,
        use_fft=True,
        return_itc=False,
    )

    try:
        power = epochs.compute_tfr(**compute_kwargs, n_jobs=workers)
    except PermissionError as exc:
        if workers not in (None, 1):
            logger.warning(
                "TFR computation failed with PermissionError using n_jobs=%s; retrying with n_jobs=1. Error=%s",
                str(workers),
                str(exc),
            )
            power = epochs.compute_tfr(**compute_kwargs, n_jobs=1)
        else:
            raise
    
    return power


def compute_tfr_for_visualization(
    epochs: mne.Epochs,
    config,
    logger: logging.Logger,
) -> mne.time_frequency.EpochsTFR:
    logger.info("Computing TFR for visualization...")
    return compute_tfr_morlet(epochs, config, logger=logger)


def _resolve_picks(epochs: mne.Epochs, config_picks: str) -> str:
    """
    Resolve channel picks based on what's available in epochs.
    
    After CSD transform, channels become type 'csd' instead of 'eeg'.
    This function detects the actual channel types and returns appropriate picks.
    
    Parameters
    ----------
    epochs : mne.Epochs
        Epochs object to check
    config_picks : str
        Configured picks (e.g., 'eeg')
        
    Returns
    -------
    str
        Resolved picks that will work with the actual channel types
    """
    if config_picks == "data":
        return "data"
    
    ch_types = set(epochs.get_channel_types())
    
    if config_picks == "eeg":
        if "eeg" in ch_types:
            return "eeg"
        if "csd" in ch_types:
            return "csd"
        return "data"
    
    if config_picks in ch_types:
        return config_picks
    
    return "data"


def compute_complex_tfr(
    epochs: mne.Epochs,
    config,
    logger: Optional[logging.Logger] = None,
    freqs: Optional[np.ndarray] = None,
) -> mne.time_frequency.EpochsTFR:
    """
    Compute complex-valued TFR for phase-based metrics (ITPC, PAC).
    
    Uses decim_phase (default=1) to preserve time structure for phase metrics.
    
    Parameters
    ----------
    epochs : mne.Epochs
        Epochs to compute TFR for
    config : Any
        Configuration object
    logger : Optional[logging.Logger]
        Logger instance
    freqs : Optional[np.ndarray]
        Frequency array. If None, uses config defaults.
        
    Returns
    -------
    mne.time_frequency.EpochsTFR
        Complex-valued TFR object (output="complex")
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    freq_min, freq_max, n_freqs, n_cycles_factor, _, tfr_picks = get_tfr_config(config)
    
    if freqs is None:
        freqs = np.logspace(np.log10(freq_min), np.log10(freq_max), n_freqs)
    
    decim_phase = get_tfr_decim(config, mode="phase")
    n_cycles = compute_adaptive_n_cycles(freqs, cycles_factor=n_cycles_factor, config=config)
    
    n_samples = len(epochs.times)
    sfreq = epochs.info["sfreq"]
    freqs, n_cycles = filter_freqs_for_signal_length(
        freqs, n_cycles, sfreq, n_samples, logger
    )
    
    workers = resolve_tfr_workers(workers_default=int(config.get("time_frequency_analysis.tfr.workers", -1)))
    
    resolved_picks = _resolve_picks(epochs, tfr_picks)
    
    logger.info("Computing complex TFR for phase-based metrics (decim=%d, %d freqs)...", decim_phase, len(freqs))
    compute_kwargs = dict(
        method="morlet",
        freqs=freqs,
        n_cycles=n_cycles,
        decim=decim_phase,
        picks=resolved_picks,
        use_fft=True,
        return_itc=False,
        average=False,
        output="complex",
    )

    try:
        return epochs.compute_tfr(**compute_kwargs, n_jobs=workers)
    except PermissionError as exc:
        if workers not in (None, 1):
            logger.warning(
                "Complex TFR computation failed with PermissionError using n_jobs=%s; retrying with n_jobs=1. Error=%s",
                str(workers),
                str(exc),
            )
            return epochs.compute_tfr(**compute_kwargs, n_jobs=1)
        raise


def _extract_baseline_power_features(
    tfr: mne.time_frequency.EpochsTFR,
    bands: Dict[str, Tuple[float, float]],
    baseline_indices: Tuple[int, int],
    logger: logging.Logger,
) -> Tuple[pd.DataFrame, List[str]]:
    """Extract baseline power features from TFR data."""
    b_start, b_end = baseline_indices
    data_baseline = tfr.data[..., int(b_start):int(b_end)]
    data_mean_time = np.nanmean(data_baseline, axis=-1)
    
    results = {}
    ch_names = tfr.info["ch_names"]
    len(tfr)
    
    freqs = tfr.freqs
    
    for band, (fmin, fmax) in bands.items():
        if fmax is None:
            fmax = freqs[-1]
        freq_mask = (freqs >= fmin) & (freqs <= fmax)
        if not np.any(freq_mask):
            continue
        
        band_freqs = np.asarray(freqs[freq_mask], dtype=float)
        if band_freqs.size >= 2 and np.all(np.isfinite(band_freqs)):
            w = np.gradient(band_freqs).astype(float)
            w = np.where(np.isfinite(w) & (w > 0), w, np.nan)
        else:
            w = np.ones((band_freqs.size,), dtype=float)

        p = data_mean_time[..., freq_mask]
        w3 = w[None, None, :]
        finite = np.isfinite(p) & np.isfinite(w3)
        num = np.nansum(np.where(finite, p * w3, 0.0), axis=-1)
        den = np.nansum(np.where(finite, w3, 0.0), axis=-1)
        band_power = np.where(den > 0, num / den, np.nan)
        
        for i, ch in enumerate(ch_names):
            col = NamingSchema.build("power", "baseline", band, "ch", "mean", channel=ch)
            results[col] = band_power[:, i]
            
    df = pd.DataFrame(results)
    return df, list(df.columns)


def extract_roi_tfrs(
    power: mne.time_frequency.EpochsTFR,
    config,
    logger: logging.Logger,
) -> dict:
    logger.info("Extracting ROI averages from TFR...")
    roi_map = build_rois_from_info(power.info, config=config)

    if len(roi_map) == 0:
        logger.warning("No ROI channels found in montage; skipping ROI analysis")
        return {}

    roi_tfrs = {}
    for roi, chs in roi_map.items():
        if not chs:
            continue

        roi_chs = [ch for ch in chs if ch in power.ch_names]
        if not roi_chs:
            logger.warning(f"No channels found for ROI {roi}")
            continue

        picks = mne.pick_channels(power.ch_names, include=roi_chs, ordered=True)
        if len(picks) == 0:
            continue

        roi_data = np.nanmean(power.data[:, picks, :, :], axis=1, keepdims=True)
        roi_info = mne.create_info([f"ROI:{roi}"], sfreq=power.info['sfreq'], ch_types='eeg')

        roi_tfr = power.copy()
        roi_tfr.data = roi_data
        roi_tfr.info = roi_info
        roi_tfrs[roi] = roi_tfr

    logger.info(f"Extracted TFRs for {len(roi_tfrs)} ROIs")
    return roi_tfrs


def compute_tfr_for_subject(
    epochs: mne.Epochs,
    aligned_events: pd.DataFrame,
    subject: str,
    task: str,
    config,
    deriv_root: Path,
    logger: logging.Logger,
    tfr_computed: Optional[mne.time_frequency.EpochsTFR] = None,
    baseline_window: Optional[Tuple[float, float]] = None,
    power_bands: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Tuple[mne.time_frequency.EpochsTFR, pd.DataFrame, List[str], float, float]:
    freq_min, freq_max, n_freqs, n_cycles_factor, tfr_decim, tfr_picks = get_tfr_config(config)

    freqs = np.logspace(np.log10(freq_min), np.log10(freq_max), n_freqs)
    n_cycles = compute_adaptive_n_cycles(freqs, cycles_factor=n_cycles_factor, config=config)

    if tfr_computed is not None:
        logger.info("Using pre-computed TFR...")
        tfr = tfr_computed
    else:
        logger.info("Computing TFR...")
        tfr = compute_tfr_morlet(epochs, config, logger=logger)

    if len(aligned_events) != len(tfr):
        raise ValueError(
            f"Alignment mismatch: aligned_events has {len(aligned_events)} rows "
            f"but TFR has {len(tfr)} epochs"
        )

    tfr.metadata = aligned_events.copy()

    times = np.asarray(tfr.times)
    tfr_analysis = config.get("time_frequency_analysis", {})
    if baseline_window is not None:
        tfr_baseline_raw = tuple(baseline_window)
    else:
        tfr_baseline_raw = tuple(tfr_analysis.get("baseline_window", [-2.0, 0.0]))
    strict_baseline_validation = bool(
        tfr_analysis.get("strict_baseline_validation", True)
    )
    tfr_baseline = validate_baseline_window_pre_stimulus(
        tfr_baseline_raw,
        logger=logger,
        strict=strict_baseline_validation,
    )
    min_baseline_samples = _get_min_baseline_samples(config)
    
    b_start, b_end = tfr_baseline
    b_start = float(times.min()) if b_start is None else float(b_start)
    b_end = 0.0 if b_end is None else float(b_end)
    
    baseline_mask = (times >= b_start) & (times < b_end)
    b_idxs = np.where(baseline_mask)[0]

    # Scientific validity warning: if baseline starts too close to the epoch boundary,
    # low-frequency Morlet wavelets will be edge-affected and bias baseline-normalized power.
    if freqs.size and n_cycles.size and b_idxs.size:
        try:
            min_idx = int(np.nanargmin(freqs))
            f_low = float(freqs[min_idx])
            ncy_low = float(n_cycles[min_idx])
            if np.isfinite(f_low) and f_low > 0 and np.isfinite(ncy_low) and ncy_low > 0:
                half_wavelet_sec = (ncy_low / f_low) / 2.0
                baseline_first_time = float(times[b_idxs[0]])
                epoch_start = float(times.min())
                margin_sec = baseline_first_time - epoch_start
                if np.isfinite(margin_sec) and margin_sec < half_wavelet_sec:
                    logger.warning(
                        "Baseline starts %.3fs after epoch start, but the lowest-frequency Morlet wavelet "
                        "has ~%.3fs half-length (fmin=%.2fHz, n_cycles=%.1f). "
                        "Baseline power will be edge-affected. Consider setting epochs.tmin earlier "
                        "or moving the baseline window later.",
                        margin_sec,
                        half_wavelet_sec,
                        f_low,
                        ncy_low,
                    )
        except (ValueError, TypeError, IndexError):
            pass
    
    if len(b_idxs) < min_baseline_samples:
        logger.info(
            f"Baseline window [{b_start:.3f}, {b_end:.3f}] outside current time range "
            f"[{times.min():.3f}, {times.max():.3f}]; skipping baseline power extraction."
        )
        return tfr, pd.DataFrame(), [], b_start, b_end

    if power_bands is None:
        power_bands = get_frequency_bands(config)

    tfr_comment = getattr(tfr, "comment", None)
    if isinstance(tfr_comment, str) and "BASELINED:" in tfr_comment:
        raise ValueError(f"TFR already baseline-corrected (comment: '{tfr_comment}')")

    logger.info("Extracting baseline power features (raw power)...")
    logger.info("Cropping TFR to range [%.3f, %.3f]", times.min(), times.max())
    
    baseline_df, baseline_cols = _extract_baseline_power_features(
        tfr, power_bands, (b_idxs[0], b_idxs[-1] + 1), logger
    )

    return tfr, baseline_df, baseline_cols, b_start, b_end




###################################################################
# Channel Extraction & Finding
###################################################################

def extract_eeg_channels(epochs: mne.Epochs) -> List[str]:
    return [
        ch for ch in epochs.info["ch_names"]
        if epochs.get_channel_types(picks=[ch])[0] == "eeg"
    ]


def find_common_channels_train_test(
    train_subjects: List[str],
    test_subject: str,
    subj_to_epochs: Dict[str, mne.Epochs]
) -> List[str]:
    train_channel_sets = [
        set(extract_eeg_channels(subj_to_epochs[s]))
        for s in train_subjects
    ]
    
    if len(train_channel_sets) == 1:
        common_train = sorted(list(train_channel_sets[0]))
    else:
        common_train = sorted(list(set.intersection(*train_channel_sets)))
    
    test_channels = set(extract_eeg_channels(subj_to_epochs[test_subject]))
    return sorted([ch for ch in common_train if ch in test_channels])



###################################################################
# ROI Channel Operations
###################################################################

def canonicalize_ch_name(ch: str) -> str:
    cleaned = ch.strip()
    try:
        cleaned = re.sub(r"^(EEG[ \-_]*)", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.split(r"[-/]", cleaned)[0]
        cleaned = re.sub(r"\s+", "", cleaned)
        cleaned = re.sub(r"(Ref|LE|RE|M1|M2|A1|A2|AVG|AVE)$", "", cleaned, flags=re.IGNORECASE)
        return cleaned
    except (re.error, TypeError, AttributeError) as e:
        logger = logging.getLogger(__name__)
        logger.warning(f"Error canonicalizing channel name '{ch}': {e}; returning original")
        return ch


def get_rois(config) -> Dict[str, List[str]]:
    if config is None:
        raise ValueError("config is required for get_rois")
    rois = config.get("time_frequency_analysis.rois")
    if rois is None:
        raise ValueError("time_frequency_analysis.rois not found in config")
    return dict(rois)


def find_roi_channels(info: mne.Info, patterns: List[str]) -> List[str]:
    channel_names = info["ch_names"]
    canon_map = {ch: canonicalize_ch_name(ch) for ch in channel_names}
    matched_channels = set()
    
    for pattern in patterns:
        regex = re.compile(pattern, flags=re.IGNORECASE)
        for ch_name in channel_names:
            canon_name = canon_map.get(ch_name, ch_name)
            if regex.match(ch_name) or regex.match(canon_name):
                matched_channels.add(ch_name)
    
    ordered_channels = []
    seen = set()
    for ch_name in channel_names:
        if ch_name in matched_channels and ch_name not in seen:
            ordered_channels.append(ch_name)
            seen.add(ch_name)
    
    return ordered_channels


def build_rois_from_info(info: mne.Info, config=None) -> Dict[str, List[str]]:
    rois = {}
    roi_defs = get_rois(config)
    for roi, pats in roi_defs.items():
        chans = find_roi_channels(info, pats)
        if chans:
            rois[roi] = chans
    return rois




###################################################################
# TFR Parameter Computation
###################################################################

def compute_adaptive_n_cycles(
    freqs: Union[np.ndarray, list],
    cycles_factor: Optional[float] = None,
    min_cycles: Optional[float] = None,
    max_cycles: Optional[float] = None,
    config: Optional[Any] = None
) -> np.ndarray:
    """
    Compute adaptive n_cycles for Morlet wavelets.
    
    Formula: n_cycles = freq / cycles_factor, clamped to [min_cycles, max_cycles].
    
    With n_cycles_factor=2.0 (default), this gives:
    - 4 Hz -> 2 cycles (clamped to min_cycles=3)
    - 10 Hz -> 5 cycles
    - 40 Hz -> 20 cycles (clamped to max_cycles=15 if set)
    - 80 Hz -> 40 cycles (clamped to max_cycles=15 if set)
    
    The max_cycles cap prevents extreme temporal smoothing at high frequencies
    (gamma band), which can degrade PAC/ITPC time structure.
    """
    if cycles_factor is None:
        cycles_factor = _get_config_float(config, "time_frequency_analysis.tfr.n_cycles_factor", 2.0)
    if min_cycles is None:
        min_cycles = _get_config_float(config, "time_frequency_analysis.tfr.min_cycles", 3.0)
    if max_cycles is None:
        max_cycles = _get_config_float(config, "time_frequency_analysis.tfr.max_cycles", None)
    
    freqs = np.asarray(freqs, dtype=float)
    base_cycles = freqs / cycles_factor
    n_cycles = np.maximum(base_cycles, min_cycles)
    
    if max_cycles is not None and np.isfinite(max_cycles) and max_cycles > 0:
        n_cycles = np.minimum(n_cycles, max_cycles)
    
    return n_cycles


def _get_config_float(config: Optional[Any], key: str, default: float) -> float:
    """Get float value from config with fallback to default."""
    if config is None:
        return default
    return float(config.get(key, default))




def _get_logger(logger: Optional[logging.Logger]) -> logging.Logger:
    if logger is None:
        return logging.getLogger(__name__)
    return logger


def resolve_tfr_workers(workers_default: int = -1) -> int:
    """Resolve TFR worker count from environment variable or use default."""
    raw = os.getenv("EEG_TFR_WORKERS")
    if not raw or raw.strip().lower() in {"auto", ""}:
        return workers_default

    try:
        workers = int(raw)
        return max(1, workers)
    except ValueError:
        logger = _get_logger(None)
        logger.warning(f"EEG_TFR_WORKERS={raw} invalid; using {workers_default}")
        return workers_default


def get_bands_for_tfr(
    tfr=None,
    max_freq_available: Optional[float] = None,
    band_bounds: Optional[Dict[str, Tuple[float, Optional[float]]]] = None,
    config=None,
) -> Dict[str, Tuple[float, float]]:
    if band_bounds is None:
        config = ensure_config(config)
        from ..config.loader import get_frequency_bands
        
        config_bands = get_frequency_bands(config)
        if not config_bands:
            raise ValueError("No frequency bands found in config. Check time_frequency_analysis.bands")
        
        band_bounds = {
            k: (v[0], v[1] if v[1] is not None else None)
            for k, v in dict(config_bands).items()
        }

    max_freq = max_freq_available
    if max_freq is None:
        if tfr is not None:
            max_freq = float(np.max(tfr.freqs))
        else:
            config = ensure_config(config)
            max_freq = float(config.get("time_frequency_analysis.tfr.freq_max"))
    
    standard_bands = ["delta", "theta", "alpha", "beta"]
    bands = {k: v for k, v in band_bounds.items() if k in standard_bands}

    gamma_lower, gamma_upper = band_bounds.get("gamma", (None, None))
    if gamma_lower is None or gamma_upper is None:
        config = ensure_config(config)
        from ..config.loader import get_frequency_bands
        config_bands = get_frequency_bands(config)
        if "gamma" not in config_bands:
            raise ValueError("Gamma band not found in config frequency bands")
        default_gamma = config_bands["gamma"]
        gamma_lower = gamma_lower if gamma_lower is not None else default_gamma[0]
        gamma_upper = gamma_upper if gamma_upper is not None else default_gamma[1]
    
    bands["gamma"] = (gamma_lower, min(gamma_upper or max_freq, max_freq))
    return bands


###################################################################
# TFR I/O with Unit Standardization
###################################################################

def save_tfr_with_sidecar(
    tfr: Union["mne.time_frequency.EpochsTFR", "mne.time_frequency.AverageTFR"],
    out_path: Union[str, "os.PathLike[str]"],
    baseline_window: Tuple[float, float],
    mode: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
    config=None,
) -> None:
    logger = _get_logger(logger)
    
    if mode is None:
        config = ensure_config(config)
        mode = str(config.get("time_frequency_analysis.baseline_mode"))

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tfr.save(str(path), overwrite=True)
    
    sidecar = {
        "baseline_applied": True,
        "baseline_mode": str(mode),
        "units": ("log10ratio" if str(mode).lower() == "logratio" else str(mode)),
        "baseline_window": [float(baseline_window[0]), float(baseline_window[1])],
        "created_by": "analysis.tfr.save_tfr_with_sidecar",
        "comment": getattr(tfr, "comment", None),
    }
    with open(path.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(sidecar, f, indent=2)
    
    logger.info(f"Saved TFR and sidecar: {path} (+ .json)")


###################################################################
# TFR Baseline Operations
###################################################################

def validate_baseline_window(
    times: np.ndarray,
    baseline: Tuple[float, float],
    min_samples: Optional[int] = None,
    config=None,
) -> Tuple[float, float, np.ndarray]:
    if min_samples is None:
        config = ensure_config(config)
        min_samples = _get_min_baseline_samples(config)
    
    b_start, b_end = baseline
    b_start = float(times.min()) if b_start is None else float(b_start)
    b_end = 0.0 if b_end is None else float(b_end)
    
    if b_end > 0:
        raise ValueError(f"Baseline window must end at or before 0 s, got [{b_start}, {b_end}]")
    
    if b_start >= b_end:
        raise ValueError(
            f"Baseline window start ({b_start}) must be < end ({b_end}). "
            f"Invalid baseline window configuration."
        )
    
    mask = (times >= b_start) & (times < b_end)
    n_samples = int(mask.sum())
    
    if n_samples < min_samples:
        msg = (
            f"Baseline window [{b_start:.3f}, {b_end:.3f}] s has {n_samples} samples; "
            f"at least {min_samples} required"
        )
        logger = _get_logger(None)
        logger.error(msg)
        raise ValueError(msg)
    
    return b_start, b_end, mask


def restrict_epochs_to_roi(
    epochs: mne.Epochs,
    roi_selection: Optional[str],
    config,
    logger,
) -> mne.Epochs:
    """
    Restrict epochs to channels within a specified ROI.
    
    Parameters
    ----------
    epochs : mne.Epochs
        Epochs object to restrict
    roi_selection : Optional[str]
        ROI name to restrict to. If None, returns epochs unchanged.
    config : Any
        Configuration object
    logger : logging.Logger
        Logger instance
        
    Returns
    -------
    mne.Epochs
        Epochs restricted to ROI channels, or original epochs if roi_selection is None
    """
    if roi_selection is None:
        return epochs
    
    roi_map = build_rois_from_info(epochs.info, config=config)
    if roi_selection not in roi_map:
        logger.warning(f"ROI '{roi_selection}' not found; using all channels")
        return epochs
    
    channels = roi_map[roi_selection]
    epochs_restricted = epochs.pick_channels(channels)
    logger.info(f"Restricted TF computation to ROI '{roi_selection}' ({len(channels)} channels)")
    return epochs_restricted


def apply_baseline_to_tfr(
    tfr, config, logger
) -> Tuple[bool, Optional[Tuple[float, float]]]:
    """
    Apply baseline correction to a TFR object using logratio mode.
    
    Parameters
    ----------
    tfr : mne.time_frequency.EpochsTFR or mne.time_frequency.AverageTFR
        TFR object to baseline correct
    config : Any
        Configuration object
    logger : logging.Logger
        Logger instance
        
    Returns
    -------
    Tuple[bool, Optional[Tuple[float, float]]]
        (baseline_applied, baseline_window_used) where baseline_window_used is (start, end) in seconds
    """
    baseline_applied = False
    baseline_window_used = None
    baseline_window = config.get(
        "time_frequency_analysis.baseline_window", [-5.0, -0.01]
    )
    min_samples_roi = config.get("behavior_analysis.statistics.min_samples_roi", 20)
    min_baseline_samples = _get_min_baseline_samples(config)
    if min_baseline_samples < min_samples_roi:
        min_baseline_samples = min_samples_roi
    
    try:
        b_start, b_end, _ = validate_baseline_window(
            tfr.times,
            tuple(baseline_window),
            min_samples=min_baseline_samples,
        )
        baseline_applied = apply_baseline_safe(
            tfr,
            baseline=(b_start, b_end),
            mode="logratio",
            logger=logger,
            min_samples=min_baseline_samples,
            config=config,
        )
        baseline_window_used = _extract_baseline_from_comment(tfr, (b_start, b_end))
    except (ValueError, RuntimeError) as err:
        logger.error(
            f"Baseline validation failed ({err}); raising error"
        )
        raise
    
    return baseline_applied, baseline_window_used


def validate_baseline_indices(
    times: np.ndarray,
    baseline: Tuple[Optional[float], Optional[float]],
    min_samples: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
    config: Optional[Any] = None,
) -> Tuple[float, float, np.ndarray]:
    if min_samples is None:
        config = ensure_config(config)
        min_samples = _get_min_baseline_samples(config)
    b_start, b_end = baseline
    b_start = float(times.min()) if b_start is None else float(b_start)
    b_end = 0.0 if b_end is None else float(b_end)

    if b_end > 0:
        raise ValueError("Baseline window must end at or before 0 s (stimulus onset)")
    
    if b_start >= b_end:
        raise ValueError(
            f"Baseline window start ({b_start}) must be < end ({b_end}). "
            f"Invalid baseline window configuration."
        )

    baseline_mask = (times >= b_start) & (times < b_end)
    idx = np.where(baseline_mask)[0]

    if len(idx) < min_samples:
        raise ValueError(
            f"Baseline window contains only {len(idx)} samples "
            f"(minimum {min_samples} required)"
        )

    if logger is not None:
        timespan = (float(times[idx[0]]), float(times[idx[-1]]))
        logger.info(
            f"Baseline indices: window [{b_start:.3f}, {b_end:.3f}] s maps to "
            f"indices [{idx[0]}, {idx[-1]}] with actual timespan [{timespan[0]:.3f}, {timespan[1]:.3f}] s "
            f"(n_samples={len(idx)})"
        )

    return b_start, b_end, idx


def _check_baseline_already_applied(
    tfr_obj: Any,
    force: bool,
    logger: logging.Logger,
    config: Optional[Any] = None,
) -> bool:
    if force:
        return False
    
    constants = _get_tfr_constants(config)
    comment = getattr(tfr_obj, "comment", None)
    baseline_sentinel = constants["baseline_sentinel"]
    if not isinstance(comment, str) or baseline_sentinel not in comment:
        return False
    
    logger.warning(
        f"Detected baseline-corrected TFR by sentinel '{baseline_sentinel}' in comment; "
        f"skipping re-application to prevent double-baselining. "
        f"Use force=True to override."
    )
    return True


def _clip_baseline_window(
    baseline_start: float,
    baseline_end: float,
    times: np.ndarray,
    logger: logging.Logger,
) -> Tuple[float, float]:
    time_min = float(times[0])
    time_max = float(times[-1])
    
    baseline_start_clipped = max(baseline_start, time_min)
    baseline_end_clipped = min(baseline_end, time_max)
    
    if baseline_end_clipped > 0:
        baseline_end_clipped = min(0.0, time_max)
        logger.warning(f"Clipping baseline end to 0.0 (was {baseline_end})")
    
    was_clipped = (
        baseline_start_clipped != baseline_start or 
        baseline_end_clipped != baseline_end
    )
    if was_clipped:
        logger.info(
            f"Clipped baseline window from [{baseline_start}, {baseline_end}] to "
            f"[{baseline_start_clipped}, {baseline_end_clipped}] to fit data range."
        )
    
    return baseline_start_clipped, baseline_end_clipped


def apply_baseline_safe(
    tfr_obj: Any,
    baseline: Tuple[Optional[float], Optional[float]],
    mode: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
    force: bool = False,
    min_samples: Optional[int] = None,
    config: Optional[Any] = None,
) -> bool:
    logger = _get_logger(logger)
    config = ensure_config(config)

    if mode is None:
        mode = str(config.get("time_frequency_analysis.baseline_mode"))

    if _check_baseline_already_applied(tfr_obj, force, logger):
        return True

    times = np.asarray(tfr_obj.times)
    
    if min_samples is None:
        min_samples = _get_min_baseline_samples(config)
    
    baseline_start = float(times.min()) if baseline[0] is None else float(baseline[0])
    baseline_end = 0.0 if baseline[1] is None else float(baseline[1])
    
    if baseline_end > 0:
        error_msg = (
            f"Baseline window must end at or before 0 s (pre-stimulus), "
            f"got [{baseline_start}, {baseline_end}]. "
            f"Post-stimulus baseline windows are invalid and indicate a configuration error."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    baseline_start_clipped, baseline_end_clipped = _clip_baseline_window(
        baseline_start, baseline_end, times, logger
    )
    
    _validate_baseline_samples(
        baseline_start_clipped, baseline_end_clipped, times, min_samples, logger
    )
    
    tfr_obj.apply_baseline(baseline=(baseline_start_clipped, baseline_end_clipped), mode=mode)
    _add_baseline_comment(tfr_obj, mode, baseline_start_clipped, baseline_end_clipped, config=config)
    logger.info(f"Applied baseline {(baseline_start_clipped, baseline_end_clipped)} with mode='{mode}'.")
    return True


def _validate_baseline_samples(
    baseline_start_clipped: float,
    baseline_end_clipped: float,
    times: np.ndarray,
    min_samples: int,
    logger: logging.Logger
) -> None:
    times_array = np.asarray(times)
    baseline_mask = (times_array >= baseline_start_clipped) & (times_array < baseline_end_clipped)
    n_samples = int(baseline_mask.sum())
    min_required_samples = max(1, min_samples)
    
    is_invalid_window = baseline_start_clipped >= baseline_end_clipped
    has_insufficient_samples = n_samples < min_required_samples
    
    if is_invalid_window or has_insufficient_samples:
        time_min_available = float(times[0])
        time_max_available = float(times[-1])
        error_msg = (
            f"Baseline window [{baseline_start_clipped}, {baseline_end_clipped}] "
            f"invalid/insufficient for available times "
            f"[{time_min_available}, {time_max_available}] (samples={n_samples}); "
            f"at least {min_required_samples} required"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)


def _add_baseline_comment(
    tfr_obj: Any,
    mode: str,
    baseline_start_clipped: float,
    baseline_end_clipped: float,
    config: Optional[Any] = None
) -> None:
    constants = _get_tfr_constants(config)
    previous_comment = getattr(tfr_obj, "comment", "")
    baseline_sentinel = constants['baseline_sentinel']
    baseline_tag = (
        f"{baseline_sentinel}mode={mode};"
        f"win=({baseline_start_clipped:.3f},{baseline_end_clipped:.3f})"
    )
    if previous_comment:
        tfr_obj.comment = f"{previous_comment} | {baseline_tag}"
    else:
        tfr_obj.comment = baseline_tag


def apply_baseline_and_crop(
    tfr_obj,
    baseline: Tuple[Optional[float], Optional[float]],
    crop_window: Optional[Tuple[Optional[float], Optional[float]]] = None,
    mode: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
    force_baseline: bool = False,
    min_samples: Optional[int] = None,
    config=None,
) -> Tuple[float, float]:
    logger = _get_logger(logger)

    baseline_applied = apply_baseline_safe(
        tfr_obj,
        baseline=baseline,
        mode=mode,
        logger=logger,
        force=force_baseline,
        min_samples=min_samples,
        config=config,
    )

    baseline_used = _extract_baseline_from_comment(tfr_obj, baseline) if baseline_applied else baseline

    if crop_window is not None:
        _apply_crop_window(tfr_obj, crop_window, logger)
    
    return baseline_used


def _extract_baseline_from_comment(
    tfr_obj: Any,
    default_baseline: Tuple[Optional[float], Optional[float]]
) -> Tuple[float, float]:
    if not hasattr(tfr_obj, "comment"):
        return default_baseline
    
    comment = str(tfr_obj.comment)
    match = re.search(r"BASELINED:.*?win=\(([^,]+),([^)]+)\)", comment)
    if match:
        return (float(match.group(1)), float(match.group(2)))
    
    return default_baseline


def _apply_crop_window(
    tfr_obj: Any,
    crop_window: Tuple[Optional[float], Optional[float]],
    logger: logging.Logger
) -> None:
    times = np.asarray(tfr_obj.times)
    tmin_req, tmax_req = crop_window
    tmin_avail, tmax_avail = float(times[0]), float(times[-1])
    
    tmin_req = float(times.min()) if tmin_req is None else float(tmin_req)
    tmax_req = float(times.max()) if tmax_req is None else float(tmax_req)
    
    tmin_clip = max(tmin_req, tmin_avail)
    tmax_clip = min(tmax_req, tmax_avail)
    
    is_invalid_window = tmin_clip > tmax_clip
    if is_invalid_window:
        logger.warning(
            f"Requested crop window [{tmin_req}, {tmax_req}] invalid for available times "
            f"[{tmin_avail}, {tmax_avail}]; using full range."
        )
        tmin_clip, tmax_clip = tmin_avail, tmax_avail
    else:
        was_clipped = tmin_clip != tmin_req or tmax_clip != tmax_req
        if was_clipped:
            logger.info(
                f"Clipped crop window from [{tmin_req}, {tmax_req}] to "
                f"[{tmin_clip}, {tmax_clip}] to fit data range."
            )
    
    tfr_obj.crop(tmin=tmin_clip, tmax=tmax_clip)


###################################################################
# TFR Data Extraction and Masking
###################################################################

def average_tfr_band(tfr_avg, fmin: float, fmax: float, tmin: float, tmax: float):
    freqs = np.asarray(tfr_avg.freqs)
    times = np.asarray(tfr_avg.times)
    f_mask = (freqs >= fmin) & (freqs <= fmax)
    t_mask = (times >= tmin) & (times < tmax)
    if f_mask.sum() == 0 or t_mask.sum() == 0:
        return None
    sel = tfr_avg.data[:, f_mask, :][:, :, t_mask]
    return sel.mean(axis=(1, 2))






###################################################################
# Time Window Utilities
###################################################################


def clip_time_range(times: np.ndarray, tmin_req: float, tmax_req: float) -> Optional[Tuple[float, float]]:
    tmin_clip = float(max(times.min(), tmin_req))
    tmax_clip = float(min(times.max(), tmax_req))
    
    is_finite = np.isfinite(tmin_clip) and np.isfinite(tmax_clip)
    is_valid_range = tmax_clip > tmin_clip
    if not is_finite or not is_valid_range:
        return None
    
    return tmin_clip, tmax_clip






###################################################################
# TFR Object Extraction Utilities
###################################################################

def extract_trial_band_power(tfr_epochs, fmin: float, fmax: float, tmin: float, tmax: float) -> Optional[np.ndarray]:
    if not isinstance(tfr_epochs, mne.time_frequency.EpochsTFR):
        return None
    
    freqs = np.asarray(tfr_epochs.freqs)
    times = np.asarray(tfr_epochs.times)
    f_mask = (freqs >= float(fmin)) & (freqs <= float(fmax))
    t_mask = (times >= float(tmin)) & (times < float(tmax))
    
    if f_mask.sum() == 0 or t_mask.sum() == 0:
        return None
    
    sel = np.asarray(tfr_epochs.data)[:, :, f_mask, :][:, :, :, t_mask]  # (trials, ch, f, t)
    if sel.size == 0:
        return None

    # Average over time first, then frequency-weighted average over frequencies.
    # This avoids bias when freqs are non-uniform/log-spaced (the pipeline default).
    # weights ~ df, computed via gradient on the selected frequency vector.
    sel_mean_time = np.nanmean(sel, axis=-1)  # (trials, ch, f)
    band_freqs = np.asarray(freqs[f_mask], dtype=float)
    if band_freqs.size >= 2 and np.all(np.isfinite(band_freqs)):
        w = np.gradient(band_freqs).astype(float)
        w = np.where(np.isfinite(w) & (w > 0), w, np.nan)
    else:
        w = np.ones((band_freqs.size,), dtype=float)

    w3 = w[None, None, :]
    finite = np.isfinite(sel_mean_time) & np.isfinite(w3)
    num = np.nansum(np.where(finite, sel_mean_time * w3, 0.0), axis=-1)
    den = np.nansum(np.where(finite, w3, 0.0), axis=-1)
    out = np.where(den > 0, num / den, np.nan)
    return out


def build_roi_channel_mask(ch_names: List[str], roi_channels: List[str]) -> np.ndarray:
    return np.array([ch in roi_channels for ch in ch_names], dtype=bool)


def extract_significant_roi_channels(ch_names: List[str], mask_vec: np.ndarray, sig_mask: np.ndarray) -> Tuple[List[int], List[str]]:
    roi_sig_indices = [i for i in range(len(ch_names)) if mask_vec[i] and sig_mask[i]]
    roi_sig_chs = [ch_names[i] for i in roi_sig_indices]
    return roi_sig_indices, roi_sig_chs


def extract_roi_from_tfr(avg_tfr, roi: str, roi_map: Optional[Dict[str, List[str]]], config) -> Optional[Any]:
    if roi_map is not None:
        chs_all = roi_map.get(roi)
        if chs_all is not None:
            subj_chs = avg_tfr.info['ch_names']
            canon_subj = {canonicalize_ch_name(ch).upper(): ch for ch in subj_chs}
            want = {canonicalize_ch_name(ch).upper() for ch in chs_all}
            chs = [canon_subj[canonicalize_ch_name(ch).upper()] for ch in subj_chs if canonicalize_ch_name(ch).upper() in want]
            if len(chs) > 0:
                picks = mne.pick_channels(subj_chs, include=chs, exclude=[])
                roi_tfr = avg_tfr.copy()
                roi_tfr.data = np.nanmean(np.asarray(avg_tfr.data)[picks, :, :], axis=0, keepdims=True)
                roi_tfr.info = mne.create_info([f"ROI:{roi}"], sfreq=avg_tfr.info['sfreq'], ch_types='eeg')
                return roi_tfr
    
    roi_defs = get_rois(config)
    pats = roi_defs.get(roi, [])
    chs = find_roi_channels(avg_tfr.info, pats)
    if len(chs) > 0:
        picks = mne.pick_channels(avg_tfr.info['ch_names'], include=chs, exclude=[])
        roi_tfr = avg_tfr.copy()
        roi_tfr.data = np.nanmean(np.asarray(avg_tfr.data)[picks, :, :], axis=0, keepdims=True)
        roi_tfr.info = mne.create_info([f"ROI:{roi}"], sfreq=avg_tfr.info['sfreq'], ch_types='eeg')
        return roi_tfr
    return None


def extract_tfr_object(tfr: Any):
    if tfr is None or (isinstance(tfr, list) and len(tfr) == 0):
        return None
    return tfr[0] if isinstance(tfr, list) else tfr






def create_tfr_subset(tfr, n: int):
    return tfr.copy()[:n]


def apply_baseline_and_average(
    tfr,
    baseline: Tuple[Optional[float], Optional[float]],
    logger: Optional[logging.Logger] = None
):
    tfr_copy = tfr.copy()
    if isinstance(tfr_copy, mne.time_frequency.EpochsTFR):
        # Apply baseline at the epoch level before averaging.
        # For nonlinear baseline modes (e.g., logratio), baseline(average(x)) != average(baseline(x)).
        baseline_used = apply_baseline_and_crop(
            tfr_copy, baseline=baseline, mode="logratio", logger=logger
        )
        tfr_avg = tfr_copy.average()
    else:
        tfr_avg = tfr_copy
        baseline_used = apply_baseline_and_crop(
            tfr_avg, baseline=baseline, mode="logratio", logger=logger
        )

    return tfr_avg, baseline_used


__all__ = [
    "canonicalize_ch_name",
    "find_roi_channels",
    "build_rois_from_info",
    "get_rois",
    "get_tfr_config",
    "compute_adaptive_n_cycles",
    "resolve_tfr_workers",
    "get_bands_for_tfr",
    "save_tfr_with_sidecar",
    "validate_baseline_window",
    "validate_baseline_indices",
    "apply_baseline_safe",
    "apply_baseline_and_crop",
    "average_tfr_band",
    "time_mask",
    "clip_time_range",
    # TFR data extraction utilities
    "extract_trial_band_power",
    # ROI processing utilities
    "build_roi_channel_mask",
    "extract_significant_roi_channels",
    "extract_roi_from_tfr",
    # TFR object extraction utilities
    "extract_tfr_object",
    # TFR manipulation utilities
    "create_tfr_subset",
    "apply_baseline_and_average",
    "restrict_epochs_to_roi",
    "apply_baseline_to_tfr",
    "compute_tfr_morlet",
]
