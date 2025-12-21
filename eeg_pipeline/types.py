"""
Type definitions for the EEG pipeline.

This module provides type hints, protocols, and dataclasses used throughout
the pipeline for type safety and IDE support.
"""

from __future__ import annotations

import os
# Ensure environment variable is set before numpy import
os.environ.setdefault("NUMPY_SKIP_MACOS_CHECK", "1")

from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Protocol,
    Tuple,
    TypedDict,
    Union,
    runtime_checkable,
)

import numpy as np
import pandas as pd

# Type aliases
PathLike = Union[str, Path]
ArrayLike = Union[np.ndarray, List[float], pd.Series]
FrequencyBand = Tuple[float, float]
CorrelationMethod = Literal["spearman", "pearson"]
AlignMode = Literal["strict", "warn", "none"]


###################################################################
# Configuration Protocol
###################################################################


@runtime_checkable
class EEGConfig(Protocol):
    """Protocol for configuration objects."""

    def get(self, key: str, default: Any = None) -> Any:
        """Get nested config value using dot notation."""
        ...

    @property
    def deriv_root(self) -> Path:
        """Path to derivatives directory."""
        ...

    @property
    def bids_root(self) -> Path:
        """Path to BIDS root directory."""
        ...


@runtime_checkable
class ConfigLike(Protocol):
    """Protocol for configuration objects with dict-like access."""
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value by dot-separated key."""
        ...


###################################################################
# Feature Extraction Types
###################################################################


class FrequencyBands(TypedDict, total=False):
    """Frequency band definitions."""

    delta: Tuple[float, float]
    theta: Tuple[float, float]
    alpha: Tuple[float, float]
    beta: Tuple[float, float]
    gamma: Tuple[float, float]
    low_gamma: Tuple[float, float]
    high_gamma: Tuple[float, float]


# Re-export ValidationResult from utils.validation to avoid duplication
from eeg_pipeline.utils.validation import ValidationResult


###################################################################
# Correlation Types
###################################################################


@dataclass
class CorrelationResult:
    """Result of a correlation analysis."""

    r: float
    p: float
    n: int
    method: CorrelationMethod = "spearman"
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None
    r_partial: Optional[float] = None
    p_partial: Optional[float] = None
    p_perm: Optional[float] = None
    q: Optional[float] = None  # FDR-corrected p-value

    def to_dict(self) -> Dict[str, Any]:
        return {
            "r": self.r,
            "p": self.p,
            "n": self.n,
            "method": self.method,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "r_partial": self.r_partial,
            "p_partial": self.p_partial,
            "p_perm": self.p_perm,
            "q": self.q,
        }


###################################################################
# Decoding Types
###################################################################


@dataclass
class DecodingResult:
    """Result of a decoding analysis."""

    y_true: np.ndarray
    y_pred: np.ndarray
    r: float
    r2: float
    mse: float
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None
    p_perm: Optional[float] = None
    feature_importance: Optional[Dict[str, float]] = None


@dataclass
class CVFoldResult:
    """Result from a single cross-validation fold."""

    fold: int
    train_idx: np.ndarray
    test_idx: np.ndarray
    y_true: np.ndarray
    y_pred: np.ndarray
    best_params: Optional[Dict[str, Any]] = None
    subject: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fold": self.fold,
            "y_true": self.y_true.tolist(),
            "y_pred": self.y_pred.tolist(),
            "test_idx": self.test_idx.tolist(),
            "best_params": self.best_params,
            "subject": self.subject,
        }


###################################################################
# Pipeline Types
###################################################################


@dataclass
class SubjectResult:
    """Result of processing a single subject."""

    subject: str
    success: bool
    error: Optional[str] = None
    duration_seconds: Optional[float] = None
    outputs: Dict[str, Path] = field(default_factory=dict)


@dataclass
class BatchResult:
    """Result of batch processing multiple subjects."""

    subjects: List[str]
    results: List[SubjectResult]
    total_duration: float

    @property
    def n_success(self) -> int:
        return sum(1 for r in self.results if r.success)

    @property
    def n_failed(self) -> int:
        return sum(1 for r in self.results if not r.success)

    @property
    def failed_subjects(self) -> List[str]:
        return [r.subject for r in self.results if not r.success]


###################################################################
# Callback Types
###################################################################

# Progress callback: receives (stage_name, fraction_complete 0-1)
ProgressCallback = Callable[[str, float], None]
LoggerType = Any  # logging.Logger, but avoid import


def null_progress(stage: str, fraction: float) -> None:
    """No-op progress callback for when progress reporting is disabled."""
    pass


###################################################################
# Constants
###################################################################

DEFAULT_BANDS: FrequencyBands = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 80.0),
}

EPSILON = 1e-12


###################################################################
# Precomputed Data Structures
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
    
    def crop(self, tmin_idx: int, tmax_idx: int) -> BandData:
        """Crop band data to specific time indices."""
        return BandData(
            band=self.band,
            fmin=self.fmin,
            fmax=self.fmax,
            filtered=self.filtered[..., tmin_idx:tmax_idx],
            analytic=self.analytic[..., tmin_idx:tmax_idx],
            envelope=self.envelope[..., tmin_idx:tmax_idx],
            phase=self.phase[..., tmin_idx:tmax_idx],
            power=self.power[..., tmin_idx:tmax_idx],
        )


@dataclass
class PSDData:
    """Pre-computed power spectral density."""
    freqs: np.ndarray
    psd: np.ndarray  # (epochs, channels, freqs)


@dataclass
class TimeWindows:
    """Pre-computed time window masks for feature extraction."""

    # Required fields (no defaults)
    baseline_mask: np.ndarray
    active_mask: np.ndarray

    # Fields with defaults
    baseline_range: Tuple[float, float] = (np.nan, np.nan)
    active_range: Tuple[float, float] = (np.nan, np.nan)

    # Generic containers for ANY number of arbitrary windows
    # keys are window names (e.g. 'stimulus', 'response', 'segment_a')
    masks: Dict[str, np.ndarray] = field(default_factory=dict)
    ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    clamped: bool = False
    valid: bool = True
    errors: List[str] = field(default_factory=list)

    # Time vector for derived mask computation
    times: Optional[np.ndarray] = None
    # Current range name if in a targeted iteration
    name: Optional[str] = None

    def _empty_mask(self) -> np.ndarray:
        """Return an empty mask matching the stored mask length."""
        if self.baseline_mask is not None:
            return np.zeros_like(self.baseline_mask, dtype=bool)
        if self.active_mask is not None:
            return np.zeros_like(self.active_mask, dtype=bool)
        if self.times is not None:
            return np.zeros_like(self.times, dtype=bool)
        return np.array([], dtype=bool)

    def get_mask(self, name: str) -> np.ndarray:
        """Retrieve a boolean mask by semantic name."""
        if name is None:
            return self._empty_mask()
        key = str(name).lower()

        # 1. Check specialized fields
        if key in {"baseline", "pre", "prestim"}:
            return self.baseline_mask
        if key in {"plateau", "active", "stim", "task"}:
            return self.active_mask

        # 2. Check generic dictionary
        if key in self.masks:
            return self.masks[key]

        # 3. Dynamic "ramp" computation (compatibility)
        if key == "ramp":
            if (
                self.times is not None
                and np.isfinite(self.baseline_range[1])
                and np.isfinite(self.active_range[0])
            ):
                ramp_mask = (self.times >= self.baseline_range[1]) & (
                    self.times < self.active_range[0]
                )
                if np.any(ramp_mask):
                    return ramp_mask

        return self._empty_mask()


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

    def crop(self, tmin: float, tmax: float) -> PrecomputedData:
        """Create a new PrecomputedData object cropped to the time range."""
        from eeg_pipeline.utils.analysis.tfr import time_mask
        
        mask = time_mask(self.times, tmin, tmax)
        if not np.any(mask):
            return self
            
        tmin_idx = int(np.where(mask)[0][0])
        tmax_idx = int(np.where(mask)[0][-1]) + 1
        
        new_times = self.times[mask]
        new_data = self.data[..., mask]
        
        cropped = PrecomputedData(
            data=new_data,
            times=new_times,
            sfreq=self.sfreq,
            ch_names=self.ch_names,
            picks=self.picks,
            config=self.config,
            logger=self.logger,
        )
        
        # Crop band data
        for band, bd in self.band_data.items():
            cropped.band_data[band] = bd.crop(tmin_idx, tmax_idx)
            
        # Crop GFP
        if self.gfp is not None:
            cropped.gfp = self.gfp[..., mask]
        for band, gfp_arr in self.gfp_band.items():
            cropped.gfp_band[band] = gfp_arr[..., mask]
            
        # PSD needs to be recomputed if it was computed on a different window,
        # but for now we just clear it or leave it as is if it's already range-specific.
        # Most extractors will recompute PSD if missing.
        cropped.psd_data = None 
            
        # Recompute windows for the new time range
        from eeg_pipeline.utils.analysis.windowing import TimeWindowSpec, time_windows_from_spec
        try:
            spec = TimeWindowSpec(times=new_times, config=self.config, sampling_rate=self.sfreq, logger=self.logger)
            cropped.windows = time_windows_from_spec(spec, logger=self.logger, strict=False)
        except Exception:
            cropped.windows = None
            
        return cropped
