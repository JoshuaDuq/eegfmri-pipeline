"""
Feature Extraction Core Module
==============================

Single Source of Truth for:
1. Data structures (PrecomputedData, BandData, FeatureExtractionContext)
2. Shared computations (filtering, GFP, PSD, envelopes)
3. Feature extractor protocol for consistent interfaces
4. Utility functions used across multiple extractors
5. Centralized thresholds and constants

Other modules should import from here rather than reimplementing.
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Protocol, runtime_checkable

import numpy as np
import pandas as pd
import mne
from scipy.signal import hilbert, welch, find_peaks

from eeg_pipeline.utils.config.loader import get_frequency_bands
from eeg_pipeline.utils.analysis.tfr import time_mask


###################################################################
# Centralized Thresholds and Constants
###################################################################

# Numerical stability
EPSILON_STD = 1e-10
EPSILON_PSD = 1e-12
EPSILON_AMP = 1e-10

# Feature extraction thresholds
MIN_EPOCHS_FOR_FEATURES = 10
MIN_CHANNELS_FOR_CONNECTIVITY = 3
MIN_SAMPLES_FOR_PSD = 64
MIN_VALID_FRACTION = 0.5  # Minimum fraction of valid data

# Microstate thresholds
MIN_EPOCHS_FOR_MICROSTATES = 20
MAX_GFP_PEAKS_PER_EPOCH = 100

# Connectivity thresholds
MIN_EPOCHS_FOR_PLV = 10
MIN_EDGE_SAMPLES = 30

# Complexity thresholds
MIN_SAMPLES_FOR_ENTROPY = 100
DEFAULT_PE_ORDER = 3
DEFAULT_PE_DELAY = 1


###################################################################
# Core Module - Single Source of Truth for Shared Computations
###################################################################


@dataclass
class BandData:
    """Pre-computed band-filtered data and derived quantities."""
    band: str
    fmin: float
    fmax: float
    filtered: np.ndarray  # (epochs, channels, times)
    analytic: np.ndarray  # Complex analytic signal
    envelope: np.ndarray  # Amplitude envelope
    phase: np.ndarray     # Instantaneous phase
    power: np.ndarray     # Envelope squared


@dataclass
class PSDData:
    """Pre-computed power spectral density."""
    freqs: np.ndarray
    psd: np.ndarray  # (epochs, channels, freqs)
    

@dataclass 
class TimeWindows:
    """Pre-computed time window masks."""
    baseline_mask: np.ndarray
    active_mask: np.ndarray
    plateau_masks: List[np.ndarray] = field(default_factory=list)
    window_labels: List[str] = field(default_factory=list)


@dataclass
class PrecomputedData:
    """Container for all pre-computed intermediate data."""
    
    # Raw data
    data: np.ndarray  # (epochs, channels, times)
    times: np.ndarray
    sfreq: float
    ch_names: List[str]
    picks: np.ndarray
    
    # Time windows
    windows: Optional[TimeWindows] = None
    
    # Band-filtered data (computed on demand)
    band_data: Dict[str, BandData] = field(default_factory=dict)
    
    # PSD (computed on demand)
    psd_data: Optional[PSDData] = None
    
    # GFP (computed on demand)
    gfp: Optional[np.ndarray] = None  # (epochs, times)
    gfp_band: Dict[str, np.ndarray] = field(default_factory=dict)
    
    # Configuration
    config: Any = None
    logger: Any = None


###################################################################
# Feature Extraction Context
###################################################################


@dataclass
class FeatureExtractionContext:
    """
    Shared context for feature extraction - compute expensive things ONCE.
    
    Similar to BehaviorContext, this holds all state needed during
    feature extraction, enabling sharing of expensive computations
    across multiple feature extractors.
    
    Usage
    -----
    ```python
    ctx = FeatureExtractionContext.from_epochs(epochs, config, logger)
    ctx.ensure_loaded()  # Preload epochs if needed
    
    # Extract features using shared computations
    power_df = extract_power_features(ctx)
    connectivity_df = extract_connectivity_features(ctx)
    ```
    
    Attributes
    ----------
    subject : str
        Subject identifier
    task : str
        Task name
    epochs : mne.Epochs
        The epochs object
    config : Any
        Configuration object
    logger : logging.Logger
        Logger instance
    precomputed : PrecomputedData
        Pre-computed data (filtered signals, GFP, PSD, etc.)
    """
    
    # Identity
    subject: str
    task: str
    
    # Data
    epochs: Optional[mne.Epochs] = None
    aligned_events: Optional[pd.DataFrame] = None
    
    # Configuration
    config: Any = None
    logger: Optional[logging.Logger] = None
    deriv_root: Path = None
    features_dir: Path = None
    
    # Pre-computed data (computed lazily)
    precomputed: Optional[PrecomputedData] = None
    tfr_complex: Optional[mne.time_frequency.EpochsTFR] = None
    
    # Feature results
    power_df: Optional[pd.DataFrame] = None
    connectivity_df: Optional[pd.DataFrame] = None
    microstates_df: Optional[pd.DataFrame] = None
    precomputed_df: Optional[pd.DataFrame] = None
    
    # State tracking
    _data_loaded: bool = False
    _precomputed_ready: bool = False
    
    @classmethod
    def from_epochs(
        cls,
        epochs: mne.Epochs,
        aligned_events: pd.DataFrame,
        subject: str,
        task: str,
        config: Any,
        logger: logging.Logger,
        deriv_root: Path = None,
    ) -> "FeatureExtractionContext":
        """Create context from epochs."""
        from eeg_pipeline.utils.io.general import deriv_features_path
        
        features_dir = deriv_features_path(deriv_root, subject) if deriv_root else None
        
        return cls(
            subject=subject,
            task=task,
            epochs=epochs,
            aligned_events=aligned_events,
            config=config,
            logger=logger,
            deriv_root=deriv_root,
            features_dir=features_dir,
        )
    
    def ensure_loaded(self) -> bool:
        """Ensure epochs data is loaded."""
        if self.epochs is None:
            return False
        
        if not self.epochs.preload:
            self.logger.info("Preloading epochs data...")
            self.epochs.load_data()
        
        self._data_loaded = True
        return True
    
    def ensure_precomputed(self) -> bool:
        """Ensure precomputed data is ready."""
        if self._precomputed_ready and self.precomputed is not None:
            return True
        
        if not self.ensure_loaded():
            return False
        
        self.logger.info("Computing shared intermediate data...")
        # Get bands from config for precomputation
        bands = self.config.get("power.bands_to_use", ["delta", "theta", "alpha", "beta", "gamma"])
        self.precomputed = precompute_data(
            self.epochs, bands, self.config, self.logger
        )
        self._precomputed_ready = True
        return True
    
    @property
    def n_epochs(self) -> int:
        """Number of epochs."""
        return len(self.epochs) if self.epochs is not None else 0
    
    @property
    def n_channels(self) -> int:
        """Number of channels."""
        return len(self.epochs.ch_names) if self.epochs is not None else 0
    
    @property
    def sfreq(self) -> float:
        """Sampling frequency."""
        return self.epochs.info["sfreq"] if self.epochs is not None else 0.0
    
    @property
    def ch_names(self) -> List[str]:
        """Channel names."""
        return list(self.epochs.ch_names) if self.epochs is not None else []
    
    @property
    def times(self) -> np.ndarray:
        """Time vector."""
        return self.epochs.times if self.epochs is not None else np.array([])
    
    def get_data(self) -> np.ndarray:
        """Get epochs data array."""
        self.ensure_loaded()
        return self.epochs.get_data()
    
    def get_frequency_bands(self) -> Dict[str, Tuple[float, float]]:
        """Get configured frequency bands."""
        return get_frequency_bands(self.config)


def compute_time_windows(
    times: np.ndarray,
    config: Any,
    n_plateau_windows: int = 5,
) -> TimeWindows:
    """Compute all time window masks once."""
    tf_cfg = config.get("time_frequency_analysis", {})
    baseline_window = tf_cfg.get("baseline_window", [-5.0, -0.01])
    plateau_window = tf_cfg.get("plateau_window", [3.0, 10.5])
    
    baseline_mask = time_mask(times, baseline_window[0], baseline_window[1])
    active_mask = time_mask(times, plateau_window[0], plateau_window[1])
    
    # Compute plateau sub-windows
    plateau_start, plateau_end = plateau_window
    window_duration = (plateau_end - plateau_start) / n_plateau_windows
    
    plateau_masks = []
    window_labels = []
    for i in range(n_plateau_windows):
        win_start = plateau_start + i * window_duration
        win_end = win_start + window_duration
        mask = time_mask(times, win_start, win_end)
        plateau_masks.append(mask)
        window_labels.append(f"w{i}")
    
    return TimeWindows(
        baseline_mask=baseline_mask,
        active_mask=active_mask,
        plateau_masks=plateau_masks,
        window_labels=window_labels,
    )


def compute_band_data(
    data: np.ndarray,
    sfreq: float,
    band: str,
    fmin: float,
    fmax: float,
    logger: Any,
) -> Optional[BandData]:
    """Compute all band-related quantities once."""
    n_epochs, n_channels, n_times = data.shape
    
    try:
        # Reshape for filtering: (epochs * channels, times)
        flat_data = data.reshape(-1, n_times)
        
        filtered = mne.filter.filter_data(
            flat_data,
            sfreq,
            l_freq=fmin,
            h_freq=fmax,
            n_jobs=1,
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
    fmin: float = 1.0,
    fmax: float = 80.0,
) -> Optional[PSDData]:
    """Compute PSD for all epochs and channels once."""
    n_epochs, n_channels, n_times = data.shape
    nperseg = min(n_times, int(2 * sfreq))
    
    try:
        # Compute for first channel to get freq array
        freqs, _ = welch(data[0, 0], fs=sfreq, nperseg=nperseg)
        freq_mask = (freqs >= fmin) & (freqs <= fmax)
        freqs = freqs[freq_mask]
        
        # Compute for all
        psd_all = np.zeros((n_epochs, n_channels, len(freqs)))
        
        for ep in range(n_epochs):
            for ch in range(n_channels):
                _, psd = welch(data[ep, ch], fs=sfreq, nperseg=nperseg)
                psd_all[ep, ch] = psd[freq_mask]
        
        return PSDData(freqs=freqs, psd=psd_all)
        
    except (ValueError, IndexError, RuntimeError):
        return None


def compute_gfp(data: np.ndarray) -> np.ndarray:
    """Compute Global Field Power for all epochs."""
    return np.std(data, axis=1)  # (epochs, times)


def precompute_data(
    epochs: mne.Epochs,
    bands: List[str],
    config: Any,
    logger: Any,
    *,
    compute_bands: bool = True,
    compute_psd_data: bool = True,
    n_plateau_windows: int = 5,
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
    compute_psd : bool
        Whether to precompute PSD
    n_plateau_windows : int
        Number of temporal windows in plateau period
        
    Returns
    -------
    PrecomputedData
        Container with all precomputed data
    """
    picks = mne.pick_types(epochs.info, eeg=True, meg=False, eog=False, stim=False, exclude="bads")
    
    if len(picks) == 0:
        logger.warning("No EEG channels available")
        return PrecomputedData(
            data=np.array([]),
            times=epochs.times,
            sfreq=float(epochs.info["sfreq"]),
            ch_names=[],
            picks=picks,
            config=config,
            logger=logger,
        )
    
    data = epochs.get_data(picks=picks)
    times = epochs.times
    sfreq = float(epochs.info["sfreq"])
    ch_names = [epochs.info["ch_names"][p] for p in picks]
    
    # Create container
    precomputed = PrecomputedData(
        data=data,
        times=times,
        sfreq=sfreq,
        ch_names=ch_names,
        picks=picks,
        config=config,
        logger=logger,
    )
    
    # Compute time windows
    precomputed.windows = compute_time_windows(times, config, n_plateau_windows)
    
    # Compute GFP
    precomputed.gfp = compute_gfp(data)
    
    # Compute band data
    if compute_bands and bands:
        freq_bands = get_frequency_bands(config)
        
        for band in bands:
            if band not in freq_bands:
                logger.warning(f"Band '{band}' not in config, skipping")
                continue
            
            fmin, fmax = freq_bands[band]
            band_data = compute_band_data(data, sfreq, band, fmin, fmax, logger)
            
            if band_data is not None:
                precomputed.band_data[band] = band_data
                # Also compute band-specific GFP
                precomputed.gfp_band[band] = compute_gfp(band_data.filtered)
        
        logger.info(f"Precomputed band data for: {list(precomputed.band_data.keys())}")
    
    # Compute PSD
    if compute_psd_data:
        precomputed.psd_data = compute_psd(data, sfreq)
        if precomputed.psd_data is not None:
            logger.info(f"Precomputed PSD: {len(precomputed.psd_data.freqs)} freq bins")
    
    return precomputed


def get_band_power_in_window(
    precomputed: PrecomputedData,
    band: str,
    window_mask: np.ndarray,
) -> Optional[np.ndarray]:
    """
    Get mean band power in a time window.
    
    Returns
    -------
    np.ndarray
        Shape (epochs, channels)
    """
    if band not in precomputed.band_data:
        return None
    
    power = precomputed.band_data[band].power  # (epochs, channels, times)
    return np.mean(power[:, :, window_mask], axis=2)


def get_psd_band_power(
    precomputed: PrecomputedData,
    fmin: float,
    fmax: float,
) -> Optional[np.ndarray]:
    """
    Get mean PSD power in a frequency band.
    
    Returns
    -------
    np.ndarray
        Shape (epochs, channels)
    """
    if precomputed.psd_data is None:
        return None
    
    freqs = precomputed.psd_data.freqs
    psd = precomputed.psd_data.psd
    
    freq_mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(freq_mask):
        return None
    
    return np.mean(psd[:, :, freq_mask], axis=2)


###################################################################
# Shared Utility Functions - Import from here, don't reimplement!
###################################################################


def pick_eeg_channels(epochs: mne.Epochs) -> Tuple[np.ndarray, List[str]]:
    """
    Pick EEG channels from epochs.
    
    This is the SINGLE implementation - other modules should import this.
    
    Parameters
    ----------
    epochs : mne.Epochs
        Input epochs
        
    Returns
    -------
    Tuple[np.ndarray, List[str]]
        (picks array, channel names list)
    """
    picks = mne.pick_types(epochs.info, eeg=True, meg=False, eog=False, stim=False, exclude="bads")
    ch_names = [epochs.info["ch_names"][p] for p in picks]
    return picks, ch_names


def bandpass_filter_epochs(
    data: np.ndarray,
    sfreq: float,
    fmin: float,
    fmax: float,
) -> Optional[np.ndarray]:
    """
    Bandpass filter epoch data.
    
    This is the SINGLE implementation - other modules should import this.
    
    Parameters
    ----------
    data : np.ndarray
        Input data, shape (epochs, channels, times) or (channels, times)
    sfreq : float
        Sampling frequency
    fmin, fmax : float
        Band limits
        
    Returns
    -------
    np.ndarray or None
        Filtered data with same shape as input
    """
    try:
        original_shape = data.shape
        
        if data.ndim == 2:
            flat_data = data
        else:  # 3D
            n_epochs, n_channels, n_times = data.shape
            flat_data = data.reshape(-1, n_times)
        
        filtered = mne.filter.filter_data(
            flat_data,
            sfreq,
            l_freq=fmin,
            h_freq=fmax,
            n_jobs=1,
            verbose=False,
        )
        
        return filtered.reshape(original_shape)
        
    except (ValueError, IndexError, RuntimeError):
        return None


def compute_gfp_with_peaks(
    data: np.ndarray,
    min_peak_distance: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute GFP and find peaks.
    
    Parameters
    ----------
    data : np.ndarray
        Shape (channels, times) or (epochs, channels, times)
    min_peak_distance : int
        Minimum distance between peaks in samples
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (gfp, peak_indices)
    """
    if data.ndim == 2:
        gfp = np.std(data, axis=0)
    else:
        gfp = np.std(data, axis=1)
    
    if gfp.ndim == 1:
        peaks, _ = find_peaks(gfp, distance=min_peak_distance)
    else:
        # For 2D (epochs, times), return peaks for first epoch
        peaks, _ = find_peaks(gfp[0], distance=min_peak_distance)
    
    return gfp, peaks


def compute_band_envelope_fast(
    data: np.ndarray,
    sfreq: float,
    fmin: float,
    fmax: float,
) -> Optional[np.ndarray]:
    """
    Compute band-limited envelope (amplitude) using Hilbert transform.
    
    This is the SINGLE implementation - other modules should import this.
    
    Parameters
    ----------
    data : np.ndarray
        Input data. Shape can be (times,), (channels, times), or (epochs, channels, times)
    sfreq : float
        Sampling frequency
    fmin, fmax : float
        Band limits
        
    Returns
    -------
    np.ndarray or None
        Envelope with same shape as input
    """
    try:
        original_shape = data.shape
        
        # Handle different input dimensions
        if data.ndim == 1:
            flat_data = data.reshape(1, -1)
        elif data.ndim == 2:
            flat_data = data
        else:  # 3D
            n_epochs, n_channels, n_times = data.shape
            flat_data = data.reshape(-1, n_times)
        
        # Filter
        filtered = mne.filter.filter_data(
            flat_data,
            sfreq,
            l_freq=fmin,
            h_freq=fmax,
            n_jobs=1,
            verbose=False,
        )
        
        # Hilbert transform
        analytic = hilbert(filtered, axis=-1)
        envelope = np.abs(analytic)
        
        return envelope.reshape(original_shape)
        
    except (ValueError, IndexError, RuntimeError):
        return None


def compute_band_phase_fast(
    data: np.ndarray,
    sfreq: float,
    fmin: float,
    fmax: float,
) -> Optional[np.ndarray]:
    """
    Compute band-limited instantaneous phase using Hilbert transform.
    
    Parameters
    ----------
    data : np.ndarray
        Input data
    sfreq : float
        Sampling frequency
    fmin, fmax : float
        Band limits
        
    Returns
    -------
    np.ndarray or None
        Phase with same shape as input
    """
    try:
        original_shape = data.shape
        
        if data.ndim == 1:
            flat_data = data.reshape(1, -1)
        elif data.ndim == 2:
            flat_data = data
        else:
            n_epochs, n_channels, n_times = data.shape
            flat_data = data.reshape(-1, n_times)
        
        filtered = mne.filter.filter_data(
            flat_data,
            sfreq,
            l_freq=fmin,
            h_freq=fmax,
            n_jobs=1,
            verbose=False,
        )
        
        analytic = hilbert(filtered, axis=-1)
        phase = np.angle(analytic)
        
        return phase.reshape(original_shape)
        
    except (ValueError, IndexError, RuntimeError):
        return None


###################################################################
# Feature Result Types
###################################################################


@dataclass
class FeatureResult:
    """Standard result type for feature extraction functions."""
    df: pd.DataFrame
    columns: List[str]
    name: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def empty(self) -> bool:
        return self.df.empty
    
    def __len__(self) -> int:
        return len(self.df)


@runtime_checkable
class FeatureExtractor(Protocol):
    """
    Protocol defining the interface for feature extractors.
    
    All feature extraction functions should follow this pattern.
    This enables consistent usage across the pipeline.
    """
    
    def __call__(
        self,
        epochs: mne.Epochs,
        config: Any,
        logger: Any,
        **kwargs,
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Extract features from epochs.
        
        Parameters
        ----------
        epochs : mne.Epochs
            Input epochs
        config : Any
            Configuration object
        logger : Any
            Logger instance
        **kwargs
            Additional extractor-specific parameters
            
        Returns
        -------
        Tuple[pd.DataFrame, List[str]]
            (features_dataframe, column_names)
        """
        ...


###################################################################
# ROI Utilities (shared across roi_features.py and others)
###################################################################


def match_channels_to_pattern(
    ch_names: List[str],
    patterns: List[str],
) -> List[int]:
    """Match channel names to regex patterns, return indices."""
    matched = []
    for idx, ch in enumerate(ch_names):
        for pattern in patterns:
            if re.match(pattern, ch):
                matched.append(idx)
                break
    return matched


def build_roi_map(
    ch_names: List[str],
    roi_definitions: Dict[str, List[str]],
) -> Dict[str, List[int]]:
    """Build mapping from ROI names to channel indices."""
    roi_map: Dict[str, List[int]] = {}
    for roi_name, patterns in roi_definitions.items():
        indices = match_channels_to_pattern(ch_names, patterns)
        if indices:
            roi_map[roi_name] = indices
    return roi_map

