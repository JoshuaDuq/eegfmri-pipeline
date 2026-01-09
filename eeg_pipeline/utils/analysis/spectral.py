"""
Spectral Analysis Utilities
===========================

Functions for computing spectral features (PSD, Band Power) using methods
other than Morlet wavelets (which are handled in tfr.py).
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

import numpy as np
import mne
from mne.time_frequency import psd_array_welch
from scipy.signal import hilbert

from eeg_pipeline.types import BandData, PSDData


# Filter design constants
FILTER_LENGTH_MULTIPLIER = 6.6
MIN_FILTER_LENGTH = 3
DEFAULT_LOW_FREQ_HZ = 0.1

# Padding defaults
DEFAULT_PAD_SECONDS = 0.5
DEFAULT_PAD_CYCLES = 3.0

# PSD defaults
DEFAULT_PSD_FMIN_HZ = 1.0
DEFAULT_PSD_FMAX_OFFSET_HZ = 0.5
DEFAULT_PSD_FMAX_HZ = 80.0
DEFAULT_PSD_FFT_MULTIPLIER = 2.0
MIN_SAMPLES_FOR_PSD = 64


def _safe_filter_length(n_times: int, sfreq: float, l_freq: float) -> str:
    """Compute filter length that fits within signal, or 'auto' if safe."""
    if l_freq <= 0:
        l_freq = DEFAULT_LOW_FREQ_HZ
    
    default_length = int(FILTER_LENGTH_MULTIPLIER * sfreq / l_freq)
    
    if default_length >= n_times:
        safe_length = n_times - 1
        if safe_length % 2 == 0:
            safe_length -= 1
        return str(max(safe_length, MIN_FILTER_LENGTH))
    
    return 'auto'


def _parse_padding_config(
    config: Any,
    pad_sec: Optional[float],
    pad_cycles: Optional[float],
) -> Tuple[float, float]:
    """Extract padding parameters from config or function arguments."""
    pad_seconds = DEFAULT_PAD_SECONDS
    pad_cycles = DEFAULT_PAD_CYCLES
    
    if config is not None and hasattr(config, "get"):
        config_pad_sec = config.get("feature_engineering.band_envelope.pad_sec")
        if config_pad_sec is not None:
            try:
                pad_seconds = float(config_pad_sec)
            except (ValueError, TypeError):
                pass
        
        config_pad_cycles = config.get("feature_engineering.band_envelope.pad_cycles")
        if config_pad_cycles is not None:
            try:
                pad_cycles = float(config_pad_cycles)
            except (ValueError, TypeError):
                pass
    
    if pad_sec is not None:
        try:
            pad_seconds = float(pad_sec)
        except (ValueError, TypeError):
            pass
    
    if pad_cycles is not None:
        try:
            pad_cycles = float(pad_cycles)
        except (ValueError, TypeError):
            pass
    
    return pad_seconds, pad_cycles


def _compute_padding_samples(
    pad_seconds: float,
    pad_cycles: float,
    fmin: float,
    sfreq: float,
    n_times: int,
) -> int:
    """Compute number of padding samples needed, clamped to valid range."""
    if not (np.isfinite(fmin) and fmin > 0 and pad_cycles > 0):
        cycle_pad_seconds = 0.0
    else:
        cycle_pad_seconds = pad_cycles / fmin
    
    effective_pad_seconds = max(pad_seconds, cycle_pad_seconds)
    
    if not (np.isfinite(effective_pad_seconds) and effective_pad_seconds > 0):
        return 0
    
    pad_samples = int(round(effective_pad_seconds * sfreq))
    
    if n_times <= 1:
        return 0
    
    return max(0, min(pad_samples, n_times - 1))


def _apply_padding(data: np.ndarray, pad_samples: int) -> Tuple[np.ndarray, int]:
    """Apply symmetric padding to data and return padded data and new length."""
    if pad_samples <= 0:
        return data, data.shape[-1]
    
    padded = np.pad(
        data,
        pad_width=((0, 0), (pad_samples, pad_samples)),
        mode="reflect",
    )
    return padded, padded.shape[-1]


def _remove_padding(
    filtered: np.ndarray,
    analytic: np.ndarray,
    pad_samples: int,
    n_times_padded: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Remove padding from filtered and analytic signals."""
    if pad_samples <= 0 or n_times_padded <= (2 * pad_samples):
        return filtered, analytic
    
    filtered_unpadded = filtered[:, pad_samples:-pad_samples]
    analytic_unpadded = analytic[:, pad_samples:-pad_samples]
    return filtered_unpadded, analytic_unpadded


def compute_band_data(
    data: np.ndarray,
    sfreq: float,
    band: str,
    fmin: float,
    fmax: float,
    logger: Optional[logging.Logger] = None,
    n_jobs: int = 1,
    *,
    pad_sec: Optional[float] = None,
    pad_cycles: Optional[float] = None,
    config: Any = None,
) -> Optional[BandData]:
    """
    Compute all band-related quantities once.
    
    Parameters
    ----------
    data : np.ndarray
        Input data with shape (n_epochs, n_channels, n_times)
    sfreq : float
        Sampling frequency in Hz
    band : str
        Frequency band name
    fmin : float
        Lower frequency bound in Hz
    fmax : float
        Upper frequency bound in Hz
    logger : Optional[logging.Logger]
        Logger instance for error reporting
    n_jobs : int
        Number of parallel jobs for filtering
    pad_sec : Optional[float]
        Padding duration in seconds (overrides config)
    pad_cycles : Optional[float]
        Padding in cycles (overrides config)
    config : Any
        Configuration object with padding parameters
    
    Returns
    -------
    Optional[BandData]
        Band data container or None on failure
    """
    if data.ndim != 3:
        if logger:
            logger.error(f"Expected 3D data, got {data.ndim}D")
        return None
    
    if sfreq <= 0:
        if logger:
            logger.error(f"Invalid sampling frequency: {sfreq}")
        return None
    
    if fmin >= fmax or fmin < 0:
        if logger:
            logger.error(f"Invalid frequency range: [{fmin}, {fmax}]")
        return None
    
    if fmax > sfreq / 2:
        if logger:
            logger.error(f"fmax {fmax} exceeds Nyquist frequency {sfreq / 2}")
        return None
    
    n_epochs, n_channels, n_times = data.shape
    
    if n_times < 1:
        if logger:
            logger.error("Data has no time samples")
        return None
    
    try:
        flat_data = data.reshape(-1, n_times)
        
        pad_seconds, pad_cycles_value = _parse_padding_config(
            config, pad_sec, pad_cycles
        )
        
        pad_samples = _compute_padding_samples(
            pad_seconds, pad_cycles_value, fmin, sfreq, n_times
        )
        
        flat_data_padded, n_times_padded = _apply_padding(flat_data, pad_samples)
        
        filter_length = _safe_filter_length(n_times_padded, sfreq, fmin)
        
        filtered = mne.filter.filter_data(
            flat_data_padded,
            sfreq,
            l_freq=fmin,
            h_freq=fmax,
            filter_length=filter_length,
            n_jobs=n_jobs,
            verbose=False,
        )
        
        analytic = hilbert(filtered, axis=-1)
        
        filtered, analytic = _remove_padding(
            filtered, analytic, pad_samples, n_times_padded
        )
        
        filtered = filtered.reshape(n_epochs, n_channels, n_times)
        analytic = analytic.reshape(n_epochs, n_channels, n_times)
        
        envelope = np.abs(analytic)
        phase = np.angle(analytic)
        power = envelope ** 2
        
        return BandData(
            band=band,
            fmin=fmin,
            fmax=fmax,
            filtered=filtered,
            analytic=analytic,
            envelope=envelope,
            phase=phase,
            power=power,
        )
        
    except (ValueError, IndexError, RuntimeError) as exc:
        if logger:
            logger.error(f"Failed to compute band data for {band}: {exc}")
        return None


def _parse_psd_config(config: Any, n_times: int, sfreq: float) -> dict[str, Any]:
    """Extract PSD computation parameters from config with defaults."""
    psd_cfg = {}
    if config is not None:
        feature_eng = config.get("feature_engineering", {})
        psd_cfg = feature_eng.get("psd", {})
    
    nyquist_freq = sfreq / 2.0
    default_fmax = min(DEFAULT_PSD_FMAX_HZ, nyquist_freq - DEFAULT_PSD_FMAX_OFFSET_HZ)
    default_n_fft = min(n_times, int(DEFAULT_PSD_FFT_MULTIPLIER * sfreq))
    
    return {
        "fmin": float(psd_cfg.get("fmin", DEFAULT_PSD_FMIN_HZ)),
        "fmax": float(psd_cfg.get("fmax", default_fmax)),
        "n_fft": int(psd_cfg.get("n_fft", default_n_fft)),
        "n_overlap": int(psd_cfg.get("n_overlap", 0)),
        "window": psd_cfg.get("window", "hann"),
        "n_jobs": int(psd_cfg.get("n_jobs", 1)),
    }


def compute_psd(
    data: np.ndarray,
    sfreq: float,
    *,
    config: Any = None,
    logger: Optional[logging.Logger] = None,
    min_samples: int = MIN_SAMPLES_FOR_PSD,
) -> Optional[PSDData]:
    """
    Compute power spectral density using Welch's method.
    
    Parameters
    ----------
    data : np.ndarray
        Input data with shape (n_epochs, n_channels, n_times)
    sfreq : float
        Sampling frequency in Hz
    config : Any
        Configuration object with PSD parameters
    logger : Optional[logging.Logger]
        Logger instance for warnings and errors
    min_samples : int
        Minimum number of time samples required
    
    Returns
    -------
    Optional[PSDData]
        PSD data container or None on failure
    """
    if data.ndim != 3:
        if logger:
            logger.error(f"Expected 3D data, got {data.ndim}D")
        return None
    
    if sfreq <= 0:
        if logger:
            logger.error(f"Invalid sampling frequency: {sfreq}")
        return None
    
    n_epochs, n_channels, n_times = data.shape
    
    if n_times < min_samples:
        if logger:
            logger.warning(
                "PSD skipped: only %d samples (< MIN_SAMPLES=%d).",
                n_times,
                min_samples,
            )
        return None
    
    psd_params = _parse_psd_config(config, n_times, sfreq)
    
    if psd_params["fmin"] >= psd_params["fmax"]:
        if logger:
            logger.error(
                f"Invalid frequency range: [{psd_params['fmin']}, {psd_params['fmax']}]"
            )
        return None
    
    if psd_params["fmax"] > sfreq / 2.0:
        if logger:
            logger.error(
                f"fmax {psd_params['fmax']} exceeds Nyquist frequency {sfreq / 2.0}"
            )
        return None
    
    try:
        psd_all, freqs = psd_array_welch(
            data,
            sfreq=sfreq,
            fmin=psd_params["fmin"],
            fmax=psd_params["fmax"],
            n_fft=psd_params["n_fft"],
            n_overlap=psd_params["n_overlap"],
            window=psd_params["window"],
            n_jobs=psd_params["n_jobs"],
            verbose=False,
        )
    except (ValueError, IndexError, RuntimeError) as exc:
        if logger:
            logger.error("PSD computation failed: %s", exc)
        return None
    
    return PSDData(freqs=freqs, psd=psd_all)


def bandpass_filter_epochs(
    data: np.ndarray,
    sfreq: float,
    fmin: float,
    fmax: float,
    n_jobs: int = 1,
) -> Optional[np.ndarray]:
    """
    Bandpass filter data (2D or 3D).
    
    Parameters
    ----------
    data : np.ndarray
        Input data with shape (n_channels, n_times) or (n_epochs, n_channels, n_times)
    sfreq : float
        Sampling frequency in Hz
    fmin : float
        Lower frequency bound in Hz
    fmax : float
        Upper frequency bound in Hz
    n_jobs : int
        Number of parallel jobs for filtering
    
    Returns
    -------
    Optional[np.ndarray]
        Filtered data with same shape as input, or None on error
    """
    if data.ndim not in (2, 3):
        return None
    
    if sfreq <= 0:
        return None
    
    if fmin >= fmax or fmin < 0:
        return None
    
    if fmax > sfreq / 2:
        return None
    
    try:
        original_shape = data.shape
        
        if data.ndim == 2:
            flat_data = data
        else:
            n_epochs, n_channels, n_times = data.shape
            flat_data = data.reshape(-1, n_times)
        
        n_times = flat_data.shape[-1]
        filter_length = _safe_filter_length(n_times, sfreq, fmin)
        
        filtered = mne.filter.filter_data(
            flat_data,
            sfreq,
            l_freq=fmin,
            h_freq=fmax,
            filter_length=filter_length,
            n_jobs=n_jobs,
            verbose=False,
        )
        
        return filtered.reshape(original_shape)
        
    except (ValueError, IndexError, RuntimeError):
        return None
