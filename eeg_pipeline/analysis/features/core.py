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
from typing import Optional, List, Dict, Any, Tuple, Protocol, runtime_checkable, Callable, Union

import numpy as np
import pandas as pd
import mne
from mne.time_frequency import psd_array_welch
from scipy.signal import hilbert, find_peaks

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
# Type Protocols
###################################################################


@runtime_checkable
class ConfigLike(Protocol):
    """Protocol for configuration objects with dict-like access."""
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by dot-separated key."""
        ...


# Progress callback type: receives (stage_name, fraction_complete 0-1)
ProgressCallback = Callable[[str, float], None]


def null_progress(stage: str, fraction: float) -> None:
    """No-op progress callback for when progress reporting is disabled."""
    pass


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
    """Pre-computed time window masks for feature extraction."""
    baseline_mask: np.ndarray
    active_mask: np.ndarray
    baseline_range: Tuple[float, float] = (np.nan, np.nan)
    active_range: Tuple[float, float] = (np.nan, np.nan)
    clamped: bool = False
    valid: bool = True
    errors: List[str] = field(default_factory=list)
    # Coarse temporal bins (early, mid, late)
    coarse_masks: List[np.ndarray] = field(default_factory=list)
    coarse_labels: List[str] = field(default_factory=list)
    # Fine temporal bins (t1-t7 for HRF modeling)
    fine_masks: List[np.ndarray] = field(default_factory=list)
    fine_labels: List[str] = field(default_factory=list)
    # Legacy compatibility
    plateau_masks: List[np.ndarray] = field(default_factory=list)
    window_labels: List[str] = field(default_factory=list)


@dataclass
class PrecomputedQC:
    """Lightweight QC summary for precomputed intermediates."""

    data_finite_fraction: float = np.nan
    n_epochs: int = 0
    n_channels: int = 0
    n_times: int = 0
    sfreq: float = np.nan
    time_windows: Dict[str, Any] = field(default_factory=dict)
    psd: Dict[str, Any] = field(default_factory=dict)
    bands: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    gfp: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        """Convert QC to a JSON-serializable dictionary."""
        return {
            "data_finite_fraction": self.data_finite_fraction,
            "n_epochs": self.n_epochs,
            "n_channels": self.n_channels,
            "n_times": self.n_times,
            "sfreq": self.sfreq,
            "time_windows": self.time_windows,
            "psd": self.psd,
            "bands": self.bands,
            "gfp": self.gfp,
            "errors": list(self.errors),
        }


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
    qc: PrecomputedQC = field(default_factory=PrecomputedQC)


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
        from eeg_pipeline.utils.config.loader import get_frequency_band_names
        bands = get_frequency_band_names(self.config)
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
    *,
    logger: Any = None,
    strict: bool = True,
) -> TimeWindows:
    """
    Compute all time window masks once.
    
    Creates both coarse (early/mid/late) and fine (t1-t7) temporal bins
    based on config settings.
    """
    errors: List[str] = []
    tf_cfg = config.get("time_frequency_analysis", {})
    baseline_window = tf_cfg.get("baseline_window", [-5.0, -0.01])
    fe_cfg = config.get("feature_engineering.features", {})
    plateau_window = tf_cfg.get("plateau_window", [3.0, 10.5])
    available_range = (
        (float(times[0]), float(times[-1])) if times.size > 0 else (np.nan, np.nan)
    )

    def _build_mask(start: float, end: float, label: str) -> Tuple[np.ndarray, Tuple[float, float], bool]:
        mask = time_mask(times, start, end)
        if np.any(mask):
            return mask, (start, end), False

        # Clamp to available time range to avoid empty masks
        if times.size == 0:
            if logger:
                logger.warning("Time vector is empty; mask '%s' will be empty.", label)
            return np.zeros_like(times, dtype=bool), (start, end), True

        clamped_start = max(start, float(times[0]))
        clamped_end = min(end, float(times[-1]))
        clamped = time_mask(times, clamped_start, clamped_end)
        if np.any(clamped):
            if logger:
                logger.warning(
                    "Clamped %s window from [%.3f, %.3f] to available range [%.3f, %.3f].",
                    label,
                    start,
                    end,
                    clamped_start,
                    clamped_end,
                )
            return clamped, (clamped_start, clamped_end), True

        if logger:
            logger.warning(
                "No samples found for %s window [%.3f, %.3f]; features using this window will be NaN.",
                label,
                start,
                end,
            )
        return clamped, (clamped_start, clamped_end), True

    baseline_mask, (baseline_start, baseline_end), baseline_clamped = _build_mask(
        baseline_window[0], baseline_window[1], "baseline"
    )
    active_mask, (plateau_start, plateau_end), active_clamped = _build_mask(
        plateau_window[0], plateau_window[1], "plateau"
    )
    clamped_any = baseline_clamped or active_clamped
    if not np.any(baseline_mask):
        errors.append(
            f"Baseline window [{baseline_start:.3f}, {baseline_end:.3f}] is empty; "
            f"available time range: [{available_range[0]:.3f}, {available_range[1]:.3f}]"
        )
    if not np.any(active_mask):
        errors.append(
            f"Active/plateau window [{plateau_start:.3f}, {plateau_end:.3f}] is empty; "
            f"available time range: [{available_range[0]:.3f}, {available_range[1]:.3f}]"
        )
    
    # Coarse temporal bins from config
    coarse_bins = fe_cfg.get("temporal_bins", [
        {"start": 3.0, "end": 5.0, "label": "early"},
        {"start": 5.0, "end": 7.5, "label": "mid"},
        {"start": 7.5, "end": 10.5, "label": "late"},
    ])
    coarse_masks = []
    coarse_labels = []
    for bin_def in coarse_bins:
        mask, _, _ = _build_mask(bin_def["start"], bin_def["end"], f"coarse-{bin_def['label']}")
        coarse_masks.append(mask)
        coarse_labels.append(bin_def["label"])
    
    # Fine temporal bins from config (for HRF modeling)
    fine_masks = []
    fine_labels = []
    use_fine = fe_cfg.get("use_fine_temporal_bins", True)
    if use_fine:
        fine_bins = fe_cfg.get("temporal_bins_fine", [])
        if not fine_bins:
            # Generate default fine bins (7 bins of ~1s each)
            n_fine = 7
            duration = (plateau_end - plateau_start) / n_fine
            fine_bins = [
                {"start": plateau_start + i * duration, 
                 "end": plateau_start + (i + 1) * duration, 
                 "label": f"t{i+1}"}
                for i in range(n_fine)
            ]
        for bin_def in fine_bins:
            label = bin_def["label"]
            mask, _, _ = _build_mask(bin_def["start"], bin_def["end"], f"fine-{label}")
            fine_masks.append(mask)
            fine_labels.append(label)
    
    # Legacy plateau windows (for backward compatibility)
    plateau_masks = []
    window_labels = []
    if n_plateau_windows > 0:
        window_duration = (plateau_end - plateau_start) / n_plateau_windows
        for i in range(n_plateau_windows):
            win_start = plateau_start + i * window_duration
            win_end = win_start + window_duration
            mask, _, _ = _build_mask(win_start, win_end, f"plateau-w{i}")
            plateau_masks.append(mask)
            window_labels.append(f"w{i}")
    
    # Explicitly warn when masks are empty after clamping so downstream code can bail early
    if errors:
        if logger:
            logger_method = logger.error if strict else logger.warning
            logger_method("Time window validation failed: %s", "; ".join(errors))
        if strict:
            raise ValueError("; ".join(errors))

    return TimeWindows(
        baseline_mask=baseline_mask,
        active_mask=active_mask,
        baseline_range=(baseline_start, baseline_end),
        active_range=(plateau_start, plateau_end),
        clamped=clamped_any,
        valid=not errors,
        errors=errors,
        coarse_masks=coarse_masks,
        coarse_labels=coarse_labels,
        fine_masks=fine_masks,
        fine_labels=fine_labels,
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
    *,
    config: Any = None,
    logger: Any = None,
) -> Optional[PSDData]:
    """Compute PSD for all epochs and channels once with config-aware parameters."""
    n_epochs, n_channels, n_times = data.shape

    if n_times < MIN_SAMPLES_FOR_PSD:
        if logger:
            logger.warning(
                "PSD skipped: only %d samples (< MIN_SAMPLES_FOR_PSD=%d).",
                n_times,
                MIN_SAMPLES_FOR_PSD,
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

    # psd_array_welch returns (epochs, channels, freqs)
    return PSDData(freqs=freqs, psd=psd_all)


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

    # Lightweight QC for downstream inspection
    if data.size > 0:
        precomputed.qc.data_finite_fraction = float(np.isfinite(data).sum() / data.size)
        precomputed.qc.n_epochs = data.shape[0]
        precomputed.qc.n_channels = data.shape[1]
        precomputed.qc.n_times = data.shape[2]
        precomputed.qc.sfreq = sfreq
    
    # Compute time windows
    try:
        precomputed.windows = compute_time_windows(
            times, config, n_plateau_windows, logger=logger, strict=True
        )
        precomputed.qc.time_windows = {
            "baseline_samples": int(np.sum(precomputed.windows.baseline_mask)),
            "active_samples": int(np.sum(precomputed.windows.active_mask)),
            "baseline_range": getattr(precomputed.windows, "baseline_range", (np.nan, np.nan)),
            "active_range": getattr(precomputed.windows, "active_range", (np.nan, np.nan)),
            "clamped": getattr(precomputed.windows, "clamped", False),
            "errors": list(getattr(precomputed.windows, "errors", [])),
        }
        if logger and precomputed.windows:
            logger.info(
                "Using time windows baseline=%s, active=%s (clamped=%s)",
                getattr(precomputed.windows, "baseline_range", None),
                getattr(precomputed.windows, "active_range", None),
                getattr(precomputed.windows, "clamped", False),
            )
    except ValueError as exc:
        precomputed.windows = None
        precomputed.qc.time_windows = {
            "baseline_samples": 0,
            "active_samples": 0,
            "baseline_range": (np.nan, np.nan),
            "active_range": (np.nan, np.nan),
            "clamped": False,
            "errors": [str(exc)],
        }
        precomputed.qc.errors.append(f"time_windows: {exc}")
        if logger:
            logger.error("Time window computation failed; downstream features will be skipped: %s", exc)
    
    # Compute GFP
    precomputed.gfp = compute_gfp(data)
    if precomputed.gfp.size > 0:
        finite_fraction = float(np.isfinite(precomputed.gfp).sum() / precomputed.gfp.size)
        precomputed.qc.gfp = {
            "finite_fraction": finite_fraction,
            "median": float(np.nanmedian(precomputed.gfp)),
        }
    
    # Compute band data (optionally in parallel)
    if compute_bands and bands:
        freq_bands = get_frequency_bands(config)
        band_defs = [(band, freq_bands[band]) for band in bands if band in freq_bands]
        if not band_defs and logger:
            logger.warning("No valid bands found in config; skipping band precomputation.")
        else:
            n_jobs_bands = int(config.get("feature_engineering.parallel.n_jobs_bands", 1))

            def _compute_single_band(band_name: str, fmin: float, fmax: float):
                bd = compute_band_data(data, sfreq, band_name, fmin, fmax, logger)
                if bd is None:
                    return None
                gfp_band = compute_gfp(bd.filtered)
                band_power = bd.power
                qc_entry = {}
                if band_power.size > 0:
                    qc_entry = {
                        "finite_fraction": float(np.isfinite(band_power).sum() / band_power.size),
                        "median_power": float(np.nanmedian(band_power)),
                        "fmin": fmin,
                        "fmax": fmax,
                    }
                return band_name, bd, gfp_band, qc_entry

            results = []
            if n_jobs_bands > 1:
                try:
                    from joblib import Parallel, delayed

                    results = Parallel(n_jobs=n_jobs_bands, prefer="processes")(
                        delayed(_compute_single_band)(band, fmin, fmax) for band, (fmin, fmax) in band_defs
                    )
                except Exception as exc:  # pragma: no cover - defensive fallback
                    if logger:
                        logger.warning(
                            "Parallel band computation failed (%s); falling back to sequential.", exc
                        )
                    n_jobs_bands = 1

            if n_jobs_bands == 1:
                results = [_compute_single_band(band, fmin, fmax) for band, (fmin, fmax) in band_defs]

            for res in results:
                if res is None:
                    continue
                band_name, bd, gfp_band, qc_entry = res
                precomputed.band_data[band_name] = bd
                precomputed.gfp_band[band_name] = gfp_band
                if qc_entry:
                    precomputed.qc.bands[band_name] = qc_entry

            logger.info(f"Precomputed band data for: {list(precomputed.band_data.keys())}")
    
    # Compute PSD
    if compute_psd_data:
        precomputed.psd_data = compute_psd(data, sfreq, config=config, logger=logger)
        if precomputed.psd_data is not None:
            logger.info(f"Precomputed PSD: {len(precomputed.psd_data.freqs)} freq bins")
            psd_arr = precomputed.psd_data.psd
            precomputed.qc.psd = {
                "n_freq_bins": int(len(precomputed.psd_data.freqs)),
                "finite_fraction": float(np.isfinite(psd_arr).sum() / psd_arr.size),
                "freq_range": (
                    float(precomputed.psd_data.freqs[0]),
                    float(precomputed.psd_data.freqs[-1]),
                )
                if len(precomputed.psd_data.freqs) > 1
                else (np.nan, np.nan),
            }
    
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
    
    freq_subset = freqs[freq_mask]
    psd_subset = psd[:, :, freq_mask]
    if freq_subset.size < 2:
        return None
    
    return np.trapz(psd_subset, freq_subset, axis=2)


###################################################################
# Shared Utility Functions - Import from here, don't reimplement!
###################################################################


def pick_eeg_channels(epochs: mne.Epochs) -> Tuple[np.ndarray, List[str]]:
    """Pick EEG channels from epochs. Returns (picks array, channel names list)."""
    picks = mne.pick_types(epochs.info, eeg=True, meg=False, eog=False, stim=False, exclude="bads")
    ch_names = [epochs.info["ch_names"][p] for p in picks]
    return picks, ch_names


def get_eeg_data(epochs: mne.Epochs, logger: Any = None, context: str = "") -> Optional[Tuple[np.ndarray, List[str], np.ndarray]]:
    """Get EEG data with channel picking and validation. Returns None if no channels."""
    picks, ch_names = pick_eeg_channels(epochs)
    if len(picks) == 0:
        if logger:
            logger.warning(f"No EEG channels available{' for ' + context if context else ''}")
        return None
    return epochs.get_data(picks=picks), ch_names, picks


def bandpass_filter_epochs(
    data: np.ndarray, sfreq: float, fmin: float, fmax: float
) -> Optional[np.ndarray]:
    """Bandpass filter data (2D or 3D). Returns None on error."""
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


def compute_gfp_with_peaks(data: np.ndarray, min_peak_distance: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """Compute GFP and find peaks. Returns (gfp, peak_indices)."""
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


def compute_band_envelope_fast(data: np.ndarray, sfreq: float, fmin: float, fmax: float) -> Optional[np.ndarray]:
    """Compute band-limited envelope via Hilbert transform. Returns None on error."""
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


def compute_band_phase_fast(data: np.ndarray, sfreq: float, fmin: float, fmax: float) -> Optional[np.ndarray]:
    """Compute band-limited instantaneous phase via Hilbert. Returns None on error."""
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

# Import from types to avoid duplication
from eeg_pipeline.types import FeatureResult


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
