"""
Type definitions for the EEG pipeline.

This module provides type hints, protocols, and dataclasses used throughout
the pipeline for type safety and IDE support.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Protocol, Tuple, runtime_checkable

import numpy as np
import pandas as pd

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
    phase: np.ndarray  # Instantaneous phase
    power: np.ndarray  # Envelope squared

    # Epochs whose spectral availability permits this contiguous band
    eligible_epochs: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if self.eligible_epochs is None:
            return

        mask = np.asarray(self.eligible_epochs, dtype=bool)
        n_epochs = self.filtered.shape[0]
        if mask.ndim != 1 or len(mask) != n_epochs:
            raise ValueError(
                f"BandData eligible_epochs length ({mask.size}) does not match "
                f"band epochs ({n_epochs})."
            )
        self.eligible_epochs = mask

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
            eligible_epochs=self.eligible_epochs,
        )


@dataclass
class PSDData:
    """Pre-computed power spectral density."""

    freqs: np.ndarray
    psd: np.ndarray  # (epochs, channels, freqs)

    # Per-epoch estimator validity, set only when spectral availability is active
    valid_frequency_mask: Optional[np.ndarray] = None
    half_support_hz: Optional[float] = None

    def __post_init__(self) -> None:
        if self.valid_frequency_mask is None:
            return

        mask = np.asarray(self.valid_frequency_mask, dtype=bool)
        expected = (self.psd.shape[0], len(self.freqs))
        if mask.shape != expected:
            raise ValueError(
                f"PSDData valid_frequency_mask shape {mask.shape} does not match "
                f"the (epochs, freqs) axes {expected}."
            )
        self.valid_frequency_mask = mask


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

    # Epoch-aligned unavailable frequency intervals (None when not configured)
    spectral_availability: Optional[Any] = None

    def __post_init__(self) -> None:
        """Validate axis contracts shared by all feature extractors."""
        self.data = np.asarray(self.data)
        self.times = np.asarray(self.times, dtype=float)
        self.picks = np.asarray(self.picks)

        if self.data.ndim != 3:
            raise ValueError(
                "PrecomputedData.data must have shape (epochs, channels, times); "
                f"got {self.data.shape}."
            )

        n_epochs, n_channels, n_times = self.data.shape
        if self.times.ndim != 1 or len(self.times) != n_times:
            raise ValueError(
                f"PrecomputedData times length ({len(self.times)}) does not match "
                f"data time axis ({n_times})."
            )
        if not np.all(np.isfinite(self.times)):
            raise ValueError("PrecomputedData.times must contain only finite values.")
        if len(self.times) > 1 and np.any(np.diff(self.times) <= 0):
            raise ValueError("PrecomputedData.times must be strictly increasing.")
        if not np.isfinite(self.sfreq) or float(self.sfreq) <= 0:
            raise ValueError("PrecomputedData.sfreq must be a positive finite number.")
        if len(self.ch_names) != n_channels:
            raise ValueError(
                f"PrecomputedData channel names ({len(self.ch_names)}) do not match "
                f"data channels ({n_channels})."
            )
        if self.picks.ndim != 1 or len(self.picks) != n_channels:
            raise ValueError(
                f"PrecomputedData picks ({len(self.picks)}) do not match "
                f"data channels ({n_channels})."
            )

        self._validate_trial_metadata(n_epochs)
        self._validate_window_masks(n_times)
        self._validate_spectral_availability(n_epochs)

    def _validate_trial_metadata(self, n_epochs: int) -> None:
        """Require optional trial metadata to align with the epoch axis."""
        if self.metadata is not None and len(self.metadata) != n_epochs:
            raise ValueError(
                f"PrecomputedData metadata rows ({len(self.metadata)}) do not match "
                f"data epochs ({n_epochs})."
            )
        if self.condition_labels is not None and len(self.condition_labels) != n_epochs:
            raise ValueError(
                "PrecomputedData condition_labels length "
                f"({len(self.condition_labels)}) does not match data epochs ({n_epochs})."
            )
        if self.train_mask is not None and len(self.train_mask) != n_epochs:
            raise ValueError(
                f"PrecomputedData train_mask length ({len(self.train_mask)}) does not "
                f"match data epochs ({n_epochs})."
            )

    def _validate_spectral_availability(self, n_epochs: int) -> None:
        """Require optional spectral availability to align with the epoch axis."""
        if self.spectral_availability is None:
            return

        n_keys = len(self.spectral_availability.recording_keys)
        if n_keys != n_epochs:
            raise ValueError(
                f"PrecomputedData spectral_availability length ({n_keys}) does not "
                f"match data epochs ({n_epochs})."
            )

    def _validate_window_masks(self, n_times: int) -> None:
        """Require stored time-window masks to align with the time axis."""
        if self.windows is None:
            return

        masks = dict(self.windows.masks)
        if self.windows.baseline_mask is not None:
            masks.setdefault("baseline", self.windows.baseline_mask)
        if self.windows.active_mask is not None:
            masks.setdefault("active", self.windows.active_mask)

        for name, mask in masks.items():
            mask_array = np.asarray(mask)
            if mask_array.ndim != 1 or len(mask_array) != n_times:
                raise ValueError(
                    f"PrecomputedData window mask '{name}' length ({len(mask_array)}) "
                    f"does not match data time axis ({n_times})."
                )

    def crop(self, tmin: float, tmax: float) -> PrecomputedData:
        """Create a new PrecomputedData object cropped to the time range."""
        from eeg_pipeline.utils.analysis.tfr import time_mask

        if not np.isfinite(tmin) or not np.isfinite(tmax):
            raise ValueError("PrecomputedData crop bounds must be finite.")
        if tmax <= tmin:
            raise ValueError("PrecomputedData crop tmax must be greater than tmin.")

        mask = time_mask(self.times, tmin, tmax)
        if not np.any(mask):
            raise ValueError(
                f"Requested crop [{tmin}, {tmax}] does not overlap precomputed times "
                f"[{self.times[0]}, {self.times[-1]}]."
            )

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
            spectral_availability=self.spectral_availability,
        )

    def _crop_band_data(self, cropped: "PrecomputedData", tmin_idx: int, tmax_idx: int) -> None:
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

        spec = TimeWindowSpec(
            times=new_times,
            config=self.config,
            sampling_rate=self.sfreq,
            logger=self.logger,
            name=window_name,
            explicit_windows=explicit_windows,
        )
        return time_windows_from_spec(spec, logger=self.logger, strict=True)

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
            metadata=self.metadata,
            condition_labels=self.condition_labels,
            train_mask=self.train_mask,
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
            spectral_availability=self.spectral_availability,
        )
