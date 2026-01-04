"""Feature extraction context."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import mne
import numpy as np
import pandas as pd

from eeg_pipeline.types import PrecomputedData
from eeg_pipeline.utils.config.loader import get_frequency_band_names
from eeg_pipeline.domain.features.constants import FEATURE_CATEGORIES, SPATIAL_MODES


@dataclass
class FeatureContext:
    """Context for feature extraction pipeline."""
    subject: str
    task: str
    config: Any
    deriv_root: Path
    logger: logging.Logger
    epochs: mne.Epochs
    aligned_events: pd.DataFrame
    fixed_templates: Optional[np.ndarray] = None
    fixed_template_ch_names: Optional[List[str]] = None
    feature_categories: List[str] = field(default_factory=lambda: list(FEATURE_CATEGORIES))
    bands: Optional[List[str]] = None  # Runtime override for frequency bands
    spatial_modes: List[str] = field(default_factory=lambda: ["roi", "global"])  # Spatial aggregation
    tmin: Optional[float] = None  # Custom time window start (seconds)
    tmax: Optional[float] = None  # Custom time window end (seconds)
    name: Optional[str] = None  # Name of the current time range
    aggregation_method: str = "mean"  # Aggregation method: 'mean' or 'median'
    explicit_windows: Optional[List[Dict[str, Any]]] = None  # User-defined time windows (TUI/CLI)
    
    # Pre-computed data (computed lazily)
    precomputed: Optional[PrecomputedData] = None
    _precomputed_ready: bool = False
    
    # Results
    tfr: Optional[Any] = None
    tfr_complex: Optional[Any] = None
    baseline_df: Optional[pd.DataFrame] = None
    baseline_cols: List[str] = field(default_factory=list)
    results: Dict[str, Any] = field(default_factory=dict)

    # Windows
    _windows: Optional[Any] = None # TimeWindowSpec placeholder

    def __post_init__(self):
        """Initialize windows and spatial modes if not provided."""
        # Resolve default spatial modes from config if not explicitly passed
        # We compare against the hardcoded default in the field definition
        if self.spatial_modes == ["roi", "global"]:
            config_modes = self.config.get("feature_engineering.spatial_modes")
            if config_modes and isinstance(config_modes, (list, tuple)):
                self.spatial_modes = list(config_modes)
            else:
                # Default to all three if nothing specified in config either
                self.spatial_modes = ["roi", "channels", "global"]
        
        self._ensure_windows()

    def _ensure_windows(self) -> None:
        if self._windows is None and self.epochs is not None:
            # Lazy import
            from eeg_pipeline.utils.analysis.windowing import TimeWindowSpec, time_windows_from_spec
            spec = TimeWindowSpec(
                times=self.epochs.times,
                config=self.config,
                sampling_rate=self.epochs.info["sfreq"],
                logger=self.logger,
                name=self.name,
                explicit_windows=self.explicit_windows,
            )
            # Convert spec to TimeWindows dataclass
            self._windows = time_windows_from_spec(spec, logger=self.logger, strict=False)

        fail_on_missing = bool(self.config.get("feature_engineering.validation.fail_on_missing_windows", False))
        if fail_on_missing and self._windows is not None:
            baseline_ok = self._windows.baseline_mask is not None and np.any(self._windows.baseline_mask)
            active_ok = self._windows.active_mask is not None and np.any(self._windows.active_mask)
            if not baseline_ok or not active_ok:
                raise ValueError(
                    "Missing required time windows for feature extraction: "
                    f"baseline_ok={baseline_ok}, active_ok={active_ok}. "
                    "Check time_frequency_analysis.baseline_window / active_window and epoch time range."
                )

        fail_on_missing_named = bool(
            self.config.get("feature_engineering.validation.fail_on_missing_named_window", True)
        )
        if fail_on_missing_named and self.name and self._windows is not None:
            named_mask = self._windows.get_mask(self.name)
            if named_mask is None or not np.any(named_mask):
                raise ValueError(
                    f"Named time window '{self.name}' is missing or empty. "
                    "Define it in feature_engineering.windows/custom windows or pass explicit ranges."
                )

    @property
    def windows(self):
        """Access centralized time windows."""
        self._ensure_windows()
        return self._windows

    def add_result(self, key: str, value: Any, cols: Optional[List[str]] = None) -> None:
        """Add a result to the context."""
        self.results[key] = value
        if cols is not None:
            self.results[f"{key}_cols"] = cols

    def set_precomputed(self, precomputed: Optional[PrecomputedData]) -> None:
        self.precomputed = precomputed
        self._precomputed_ready = precomputed is not None
        if precomputed is not None:
            precomputed.spatial_modes = list(self.spatial_modes) if self.spatial_modes else None
    
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
