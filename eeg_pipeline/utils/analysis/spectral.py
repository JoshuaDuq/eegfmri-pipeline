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
from mne.time_frequency import psd_array_multitaper, psd_array_welch
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


def compute_frequency_weights(frequencies: np.ndarray) -> np.ndarray:
    """Compute trapezoidal integration weights for a frequency axis."""
    freqs = np.asarray(frequencies, dtype=float)
    n_freqs = freqs.size
    if n_freqs <= 1:
        return np.ones(n_freqs, dtype=float)

    weights = np.zeros(n_freqs, dtype=float)
    weights[0] = (freqs[1] - freqs[0]) / 2.0
    weights[1:-1] = (freqs[2:] - freqs[:-2]) / 2.0
    weights[-1] = (freqs[-1] - freqs[-2]) / 2.0

    if np.all(np.isfinite(weights)) and np.all(weights > 0):
        return weights

    fallback = np.gradient(freqs).astype(float)
    fallback = np.where(np.isfinite(fallback) & (fallback > 0), fallback, np.nan)
    if np.isfinite(fallback).any():
        return fallback
    return np.ones(n_freqs, dtype=float)


def _safe_filter_length(n_times: int, sfreq: float, l_freq: float) -> str:
    """Compute filter length that fits within signal, or 'auto' if safe."""
    effective_l_freq = l_freq if l_freq > 0 else DEFAULT_LOW_FREQ_HZ
    default_length = int(FILTER_LENGTH_MULTIPLIER * sfreq / effective_l_freq)
    
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
    pad_cycles_value = DEFAULT_PAD_CYCLES
    
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
                pad_cycles_value = float(config_pad_cycles)
            except (ValueError, TypeError):
                pass
    
    if pad_sec is not None:
        try:
            pad_seconds = float(pad_sec)
        except (ValueError, TypeError):
            pass
    
    if pad_cycles is not None:
        try:
            pad_cycles_value = float(pad_cycles)
        except (ValueError, TypeError):
            pass
    
    return pad_seconds, pad_cycles_value


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
        
        pad_seconds, pad_cycles = _parse_padding_config(
            config, pad_sec, pad_cycles
        )
        
        pad_samples = _compute_padding_samples(
            pad_seconds, pad_cycles, fmin, sfreq, n_times
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
    psd_cfg: dict[str, Any] = {}
    if config is not None and hasattr(config, "get"):
        psd_cfg = config.get("feature_engineering.psd", {}) or {}
        if not psd_cfg:
            psd_cfg = config.get("feature_engineering.spectral", {}) or {}
    
    nyquist_freq = sfreq / 2.0
    default_fmax = min(DEFAULT_PSD_FMAX_HZ, nyquist_freq - DEFAULT_PSD_FMAX_OFFSET_HZ)
    default_n_fft = min(n_times, int(DEFAULT_PSD_FFT_MULTIPLIER * sfreq))
    default_n_overlap = max(0, default_n_fft // 2)
    
    n_fft = int(psd_cfg.get("n_fft", default_n_fft))
    n_fft = max(2, min(n_fft, n_times))
    
    n_overlap_raw = psd_cfg.get("n_overlap", None)
    if n_overlap_raw is None:
        n_overlap = default_n_overlap if n_fft == default_n_fft else max(0, n_fft // 2)
    else:
        n_overlap = int(n_overlap_raw)
    n_overlap = max(0, min(n_overlap, n_fft - 1))

    return {
        "fmin": float(psd_cfg.get("fmin", DEFAULT_PSD_FMIN_HZ)),
        "fmax": float(psd_cfg.get("fmax", default_fmax)),
        "n_fft": n_fft,
        "n_overlap": n_overlap,
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


def compute_psd_bandpower(
    data: np.ndarray,
    sfreq: float,
    band_ranges: dict[str, tuple[float, float]],
    *,
    psd_method: str = "multitaper",
    fmin: float = 1.0,
    fmax: float = 80.0,
    bandwidth: float = 2.0,
    adaptive: bool = False,
    normalize_by_bandwidth: bool = True,
    exclude_line_noise: bool = False,
    line_freqs: Optional[list[float]] = None,
    line_width: Optional[float] = None,
    n_harmonics: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
) -> Optional[dict[str, np.ndarray]]:
    """
    Compute PSD-integrated band power (scientifically valid for ratios/asymmetry).
    
    This is the correct approach for band power ratios and asymmetry metrics.
    Unlike Hilbert envelope², PSD integration:
    - Properly accounts for 1/f spectral slope
    - Is bandwidth-normalized (power per Hz)
    - Has well-defined statistical properties
    - Is comparable across bands of different widths
    
    Parameters
    ----------
    data : np.ndarray
        Input data with shape (n_epochs, n_channels, n_times)
    sfreq : float
        Sampling frequency in Hz
    band_ranges : dict
        Dictionary mapping band names to (fmin, fmax) tuples
    psd_method : str
        PSD method: 'multitaper' (recommended) or 'welch'
    fmin : float
        Minimum frequency for PSD computation
    fmax : float
        Maximum frequency for PSD computation
    bandwidth : float
        Frequency smoothing bandwidth for multitaper (Hz)
    normalize_by_bandwidth : bool
        If True, return power per Hz (recommended for ratios).
        If False, return total integrated power in band.
    exclude_line_noise : bool
        If True, exclude line noise frequencies and harmonics from band integration.
    line_freqs : Optional[list[float]]
        Line noise frequencies (e.g., [50.0] or [60.0]). Required if exclude_line_noise=True.
    line_width : Optional[float]
        Half-width (Hz) around each line frequency to exclude (i.e., excludes
        [f0 - line_width, f0 + line_width]). Default 1.0 Hz.
    n_harmonics : Optional[int]
        Number of harmonics to exclude. Default 3.
    logger : Optional[logging.Logger]
        Logger for warnings
    
    Returns
    -------
    Optional[dict[str, np.ndarray]]
        Dictionary mapping band names to power arrays (n_epochs, n_channels).
        Returns None on failure.
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
    
    if n_times < 64:
        if logger:
            logger.warning(
                "PSD bandpower skipped: only %d samples (< 64 minimum).",
                n_times,
            )
        return None
    
    nyquist = sfreq / 2.0
    fmax = min(fmax, nyquist - 0.5)
    
    try:
        if psd_method == "multitaper":
            psds, freqs = psd_array_multitaper(
                data,
                sfreq=sfreq,
                fmin=fmin,
                fmax=fmax,
                bandwidth=bandwidth,
                adaptive=adaptive,
                normalization="full",
                verbose=False,
            )
        else:
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
    except Exception as exc:
        if logger:
            logger.error("PSD computation failed: %s", exc)
        return None
    
    freqs = np.asarray(freqs, dtype=float)
    psds = np.asarray(psds, dtype=float)
    
    # Compute frequency bin widths for integration
    if len(freqs) > 1:
        df = np.gradient(freqs)
    else:
        df = np.ones_like(freqs)
    
    # Build line-noise exclusion mask if requested
    line_noise_mask = np.zeros(len(freqs), dtype=bool)
    if exclude_line_noise and line_freqs:
        width = line_width if line_width is not None else 1.0
        n_harm = n_harmonics if n_harmonics is not None else 3
        for base_freq in line_freqs:
            for harmonic in range(1, n_harm + 1):
                center = base_freq * harmonic
                line_noise_mask |= (freqs >= center - width) & (freqs <= center + width)
        if logger and np.any(line_noise_mask):
            n_excluded = np.sum(line_noise_mask)
            logger.debug("Excluding %d frequency bins for line noise", n_excluded)
    
    band_power: dict[str, np.ndarray] = {}
    
    for band_name, (band_fmin, band_fmax) in band_ranges.items():
        band_mask = (freqs >= band_fmin) & (freqs <= band_fmax)
        
        # Exclude line noise bins from band integration
        if exclude_line_noise:
            band_mask = band_mask & ~line_noise_mask
        
        if not np.any(band_mask):
            if logger:
                logger.warning(
                    "Band '%s' [%.1f-%.1f Hz] outside PSD range [%.1f-%.1f Hz]; skipping.",
                    band_name, band_fmin, band_fmax, freqs.min(), freqs.max()
                )
            band_power[band_name] = np.full((n_epochs, n_channels), np.nan)
            continue
        
        # Integrate PSD over band: sum(PSD * df)
        psd_band = psds[..., band_mask]
        df_band = df[band_mask]
        
        # Weighted integration (handles non-uniform frequency spacing)
        integrated_power = np.sum(psd_band * df_band, axis=-1)
        
        if normalize_by_bandwidth:
            # Power per Hz (comparable across bands of different widths)
            # Use actual integrated bandwidth (excluding line noise gaps)
            actual_bandwidth = np.sum(df_band)
            if actual_bandwidth > 0:
                integrated_power = integrated_power / actual_bandwidth
        
        band_power[band_name] = integrated_power
    
    return band_power


def subtract_evoked(
    data: np.ndarray,
    condition_labels: Optional[np.ndarray] = None,
    *,
    train_mask: Optional[np.ndarray] = None,
    min_trials_per_condition: int = 2,
) -> np.ndarray:
    """
    Subtract evoked (phase-locked) response to isolate induced activity.
    
    For spectral analysis of event-related data (e.g., pain paradigms), the
    evoked response (ERP) can contaminate power/spectral estimates, especially
    at low frequencies. Subtracting the evoked response isolates "induced"
    oscillatory activity that is time-locked but not phase-locked to the stimulus.
    
    Parameters
    ----------
    data : np.ndarray
        Input data with shape (n_epochs, n_channels, n_times)
    condition_labels : Optional[np.ndarray]
        If provided, subtract condition-specific evoked responses. Array of
        length n_epochs with condition labels. If None, subtract grand average
        across all epochs.
    
    Returns
    -------
    np.ndarray
        Induced data with same shape as input (data - evoked)
    
    Notes
    -----
    This is the standard approach for computing "induced" power in EEG/MEG:
    - Total power = Evoked power + Induced power
    - Induced = Total - Evoked
    
    For pain paradigms, induced power better reflects ongoing oscillatory
    changes related to pain processing, while evoked power reflects the
    transient sensory response.
    
    References
    ----------
    Tallon-Baudry & Bertrand (1999). Oscillatory gamma activity in humans
    and its role in object representation. Trends in Cognitive Sciences.
    """
    if data.ndim != 3:
        raise ValueError(f"Expected 3D data (epochs, channels, times), got {data.ndim}D")
    
    n_epochs, n_channels, n_times = data.shape
    induced = data.copy()
    
    if train_mask is not None:
        train_mask = np.asarray(train_mask, dtype=bool).ravel()
        if train_mask.size != n_epochs:
            raise ValueError(
                f"train_mask length ({train_mask.size}) must match n_epochs ({n_epochs})"
            )
        if int(np.sum(train_mask)) == 0:
            train_mask = None

    ref_mask = train_mask if train_mask is not None else np.ones(n_epochs, dtype=bool)
    ref_data = data[ref_mask]
    grand_evoked = np.nanmean(ref_data, axis=0, keepdims=True)

    if condition_labels is None:
        induced = data - grand_evoked
    else:
        condition_labels = np.asarray(condition_labels)
        if len(condition_labels) != n_epochs:
            raise ValueError(
                f"condition_labels length ({len(condition_labels)}) must match "
                f"n_epochs ({n_epochs})"
            )

        if min_trials_per_condition < 1:
            min_trials_per_condition = 1

        if condition_labels.dtype.kind in {"f", "i", "u"}:
            valid_labels = np.isfinite(condition_labels.astype(float))
        else:
            valid_labels = np.array(
                [
                    lbl is not None
                    and not (isinstance(lbl, (float, np.floating)) and np.isnan(lbl))
                    for lbl in condition_labels
                ],
                dtype=bool,
            )

        unique_conditions = np.unique(condition_labels[valid_labels])

        for condition in unique_conditions:
            cond_mask_all = condition_labels == condition
            cond_mask_ref = cond_mask_all & ref_mask
            n_ref = int(np.sum(cond_mask_ref))

            if n_ref >= min_trials_per_condition:
                condition_evoked = np.nanmean(data[cond_mask_ref], axis=0, keepdims=True)
            else:
                condition_evoked = grand_evoked

            induced[cond_mask_all] = data[cond_mask_all] - condition_evoked

        # For missing/invalid condition labels, fall back to grand evoked subtraction
        missing_mask = ~valid_labels
        if np.any(missing_mask):
            induced[missing_mask] = data[missing_mask] - grand_evoked
    
    return induced


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
