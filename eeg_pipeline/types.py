"""
Type definitions for the EEG pipeline.

This module provides type hints, protocols, and dataclasses used throughout
the pipeline for type safety and IDE support.
"""

from __future__ import annotations

import os

# Suppress macOS NumPy compatibility warning
# Must be set before numpy import
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


###################################################################
# Configuration Protocol
###################################################################


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
# Machine Learning Types
###################################################################


@dataclass
class MLResult:
    """Result of a machine learning analysis."""

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

ProgressCallback = Callable[[str, float], None]


def null_progress(stage: str, fraction: float) -> None:
    """No-op progress callback when progress reporting is disabled."""


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

    # Fields with defaults
    baseline_mask: Optional[np.ndarray] = None
    active_mask: Optional[np.ndarray] = None
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
        """Return an empty boolean mask matching stored mask dimensions."""
        if self.baseline_mask is not None:
            reference = self.baseline_mask
        elif self.active_mask is not None:
            reference = self.active_mask
        elif self.times is not None:
            reference = self.times
        else:
            return np.array([], dtype=bool)

        return np.zeros_like(reference, dtype=bool)

    def get_mask(self, name: str) -> np.ndarray:
        """Retrieve a boolean mask by name."""
        raw_key = str(name)
        key = raw_key.lower()

        # 1. Exact match in masks dict (user-defined names)
        if raw_key in self.masks:
            return self.masks[raw_key]
        if key in self.masks:
            return self.masks[key]

        # 2. Match against primary baseline/active fields for internal compatibility
        if key == "baseline" and self.baseline_mask is not None:
            return self.baseline_mask
        if key == "active" and self.active_mask is not None:
            return self.active_mask

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

    # Trial metadata (aligned to epochs axis)
    metadata: Optional[pd.DataFrame] = None
    condition_labels: Optional[np.ndarray] = None
    train_mask: Optional[np.ndarray] = None

    # Provenance flags (important for scientific interpretation)
    evoked_subtracted: bool = False
    evoked_subtracted_conditionwise: bool = False
    
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
    spatial_modes: Optional[List[str]] = None
    frequency_bands: Optional[Dict[str, List[float]]] = None
    feature_family: Optional[str] = None
    spatial_transform: Optional[str] = None

    def crop(self, tmin: float, tmax: float) -> PrecomputedData:
        """Create a new PrecomputedData object cropped to the time range."""
        from eeg_pipeline.utils.analysis.tfr import time_mask

        mask = time_mask(self.times, tmin, tmax)
        if not np.any(mask):
            return self

        tmin_idx, tmax_idx = self._get_crop_indices(mask)
        new_times = self.times[mask]
        new_data = self.data[..., mask]

        cropped = self._create_cropped_base(new_times, new_data)
        self._crop_band_data(cropped, tmin_idx, tmax_idx)
        self._crop_gfp(cropped, mask)
        cropped.psd_data = None
        cropped.windows = self._recompute_windows(new_times)

        return cropped

    def _get_crop_indices(self, mask: np.ndarray) -> Tuple[int, int]:
        """Extract start and end indices from a boolean mask."""
        indices = np.where(mask)[0]
        return int(indices[0]), int(indices[-1]) + 1

    def _create_cropped_base(
        self, new_times: np.ndarray, new_data: np.ndarray
    ) -> "PrecomputedData":
        """Create base PrecomputedData with cropped time and data."""
        return PrecomputedData(
            data=new_data,
            times=new_times,
            sfreq=self.sfreq,
            ch_names=self.ch_names,
            picks=self.picks,
            windows=None,
            metadata=self.metadata,
            condition_labels=self.condition_labels,
            train_mask=self.train_mask,
            evoked_subtracted=self.evoked_subtracted,
            evoked_subtracted_conditionwise=self.evoked_subtracted_conditionwise,
            config=self.config,
            logger=self.logger,
            spatial_modes=self.spatial_modes,
            frequency_bands=self.frequency_bands,
            feature_family=self.feature_family,
            spatial_transform=self.spatial_transform,
        )

    def _crop_band_data(
        self, cropped: "PrecomputedData", tmin_idx: int, tmax_idx: int
    ) -> None:
        """Crop all band data to the specified indices."""
        for band, band_data in self.band_data.items():
            cropped.band_data[band] = band_data.crop(tmin_idx, tmax_idx)

    def _crop_gfp(self, cropped: "PrecomputedData", mask: np.ndarray) -> None:
        """Crop GFP arrays using the time mask."""
        if self.gfp is not None:
            cropped.gfp = self.gfp[..., mask]
        for band, gfp_arr in self.gfp_band.items():
            cropped.gfp_band[band] = gfp_arr[..., mask]

    def _recompute_windows(self, new_times: np.ndarray) -> Optional[TimeWindows]:
        """Recompute time windows for the cropped time range."""
        from eeg_pipeline.utils.analysis.windowing import (
            TimeWindowSpec,
            time_windows_from_spec,
        )

        explicit_windows = self._extract_explicit_windows()
        window_name = self.windows.name if self.windows is not None else None

        try:
            spec = TimeWindowSpec(
                times=new_times,
                config=self.config,
                sampling_rate=self.sfreq,
                logger=self.logger,
                name=window_name,
                explicit_windows=explicit_windows,
            )
            return time_windows_from_spec(spec, logger=self.logger, strict=False)
        except (ValueError, TypeError, KeyError):
            return None

    def _extract_explicit_windows(self) -> Optional[List[Dict[str, Any]]]:
        """Extract explicit window ranges from existing windows."""
        if self.windows is None or not hasattr(self.windows, "ranges"):
            return None

        ranges = self.windows.ranges
        if not ranges:
            return None

        explicit_windows = []
        for name, rng in ranges.items():
            if isinstance(rng, (list, tuple)) and len(rng) >= 2:
                explicit_windows.append({"name": name, "tmin": rng[0], "tmax": rng[1]})

        return explicit_windows if explicit_windows else None

    def with_windows(self, windows: Optional[TimeWindows]) -> PrecomputedData:
        """Return a shallow copy with updated time windows."""
        if windows is None:
            return self

        return PrecomputedData(
            data=self.data,
            times=self.times,
            sfreq=self.sfreq,
            ch_names=self.ch_names,
            picks=self.picks,
            windows=windows,
            band_data=self.band_data,
            psd_data=self.psd_data,
            gfp=self.gfp,
            gfp_band=self.gfp_band,
            config=self.config,
            logger=self.logger,
            qc=self.qc,
            spatial_modes=self.spatial_modes,
            frequency_bands=self.frequency_bands,
            feature_family=self.feature_family,
            spatial_transform=self.spatial_transform,
            evoked_subtracted=self.evoked_subtracted,
            evoked_subtracted_conditionwise=self.evoked_subtracted_conditionwise,
        )
