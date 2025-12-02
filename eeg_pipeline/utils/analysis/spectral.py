"""
Spectral Analysis Utilities
===========================

Functions for computing spectral features (PSD, Band Power) using methods
other than Morlet wavelets (which are handled in tfr.py).
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Tuple, List

import numpy as np
import mne
from mne.time_frequency import psd_array_welch
from scipy.signal import hilbert

from eeg_pipeline.utils.config.loader import get_frequency_bands
from eeg_pipeline.types import BandData, PSDData


def _safe_filter_length(n_times: int, sfreq: float, l_freq: float) -> str:
    """Compute filter length that fits within signal, or 'auto' if safe."""
    default_length = int(6.6 * sfreq / l_freq) if l_freq else int(6.6 * sfreq / 0.1)
    if default_length >= n_times:
        safe_length = n_times - 1
        if safe_length % 2 == 0:
            safe_length -= 1
        return str(max(safe_length, 3))
    return 'auto'


def compute_band_data(
    data: np.ndarray,
    sfreq: float,
    band: str,
    fmin: float,
    fmax: float,
    logger: Optional[logging.Logger] = None,
    n_jobs: int = 1,
) -> Optional[BandData]:
    """
    Compute all band-related quantities once.
    """
    n_epochs, n_channels, n_times = data.shape
    
    try:
        # Reshape for filtering: (epochs * channels, times)
        flat_data = data.reshape(-1, n_times)
        
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
        
        analytic = hilbert(filtered, axis=-1)
        
        # Reshape back
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
        
    except Exception as exc:
        if logger:
            logger.error(f"Failed to compute band data for {band}: {exc}")
        return None


def compute_psd(
    data: np.ndarray,
    sfreq: float,
    *,
    config: Any = None,
    logger: Optional[logging.Logger] = None,
    min_samples: int = 64,
) -> Optional[PSDData]:
    # ... (same) ...
    n_epochs, n_channels, n_times = data.shape
    if n_times < min_samples:
        if logger:
            logger.warning(
                "PSD skipped: only %d samples (< MIN_SAMPLES=%d).",
                n_times,
                min_samples,
            )
        return None

    psd_cfg = (config or {}).get("feature_engineering", {}).get("psd", {})
    fmin = float(psd_cfg.get("fmin", 1.0))
    fmax = float(psd_cfg.get("fmax", min(80.0, sfreq / 2.0 - 0.5)))
    n_fft = int(psd_cfg.get("n_fft", min(n_times, int(2 * sfreq))))
    n_overlap = int(psd_cfg.get("n_overlap", 0))
    window = psd_cfg.get("window", "hann")
    n_jobs = int(psd_cfg.get("n_jobs", 1))

    try:
        psd_all, freqs = psd_array_welch(
            data,
            sfreq=sfreq,
            fmin=fmin,
            fmax=fmax,
            n_fft=n_fft,
            n_overlap=n_overlap,
            window=window,
            n_jobs=n_jobs,
            verbose=False,
        )
    except (ValueError, IndexError, RuntimeError) as exc:
        if logger:
            logger.error("PSD computation failed: %s", exc)
        return None

    return PSDData(freqs=freqs, psd=psd_all)


def bandpass_filter_epochs(
    data: np.ndarray, sfreq: float, fmin: float, fmax: float, n_jobs: int = 1
) -> Optional[np.ndarray]:
    """Bandpass filter data (2D or 3D). Returns None on error."""
    try:
        original_shape = data.shape
        
        if data.ndim == 2:
            flat_data = data
        else:  # 3D
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
