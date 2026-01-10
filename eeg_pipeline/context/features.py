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
from eeg_pipeline.domain.features.constants import FEATURE_CATEGORIES


_DEFAULT_SPATIAL_MODES = ["roi", "global"]


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
    bands: Optional[List[str]] = None
    frequency_bands: Optional[Dict[str, List[float]]] = None
    spatial_modes: List[str] = field(default_factory=lambda: list(_DEFAULT_SPATIAL_MODES))
    tmin: Optional[float] = None
    tmax: Optional[float] = None
    name: Optional[str] = None
    aggregation_method: str = "mean"
    explicit_windows: Optional[List[Dict[str, Any]]] = None
    train_mask: Optional[np.ndarray] = None
    precomputed: Optional[PrecomputedData] = None
    _precomputed_ready: bool = False
    tfr: Optional[Any] = None
    tfr_complex: Optional[Any] = None
    baseline_df: Optional[pd.DataFrame] = None
    baseline_cols: List[str] = field(default_factory=list)
    results: Dict[str, Any] = field(default_factory=dict)
    _windows: Optional[Any] = None
    _original_epochs: Optional[mne.Epochs] = None

    def __post_init__(self) -> None:
        """Initialize spatial modes and windows."""
        self._resolve_spatial_modes()
        self._initialize_windows()
        self._validate_windows()

    def _resolve_spatial_modes(self) -> None:
        """Resolve spatial modes from config if using default."""
        if self.spatial_modes == _DEFAULT_SPATIAL_MODES:
            config_modes = self.config.get("feature_engineering.spatial_modes")
            if config_modes and isinstance(config_modes, (list, tuple)):
                self.spatial_modes = list(config_modes)
            else:
                self.spatial_modes = ["roi", "channels", "global"]

    def _initialize_windows(self) -> None:
        """Lazily initialize time windows from epochs."""
        if self._windows is None and self.epochs is not None:
            from eeg_pipeline.utils.analysis.windowing import (
                TimeWindowSpec,
                time_windows_from_spec,
            )
            spec = TimeWindowSpec(
                times=self.epochs.times,
                config=self.config,
                sampling_rate=self.epochs.info["sfreq"],
                logger=self.logger,
                name=self.name,
                explicit_windows=self.explicit_windows,
            )
            self._windows = time_windows_from_spec(
                spec, logger=self.logger, strict=False
            )

    def _validate_windows(self) -> None:
        """Validate time windows according to config settings."""
        if self._windows is None:
            return

        self._validate_required_windows()
        self._validate_named_window()

    def _validate_required_windows(self) -> None:
        """Validate that at least one valid time window exists from Step 5 input."""
        fail_on_missing = bool(
            self.config.get("feature_engineering.validation.fail_on_missing_windows", False)
        )
        if not fail_on_missing:
            return

        # Check for at least one non-empty mask
        has_valid_window = any(
            np.any(mask) for name, mask in self._windows.masks.items()
        )

        if not has_valid_window:
            raise ValueError(
                "No valid time windows found for feature extraction. "
                "Please define at least one time range in Step 5 of the TUI "
                "(e.g., 'baseline', 'active', or a custom segment)."
            )

    def _validate_named_window(self) -> None:
        """Validate that named window exists if specified."""
        fail_on_missing_named = bool(
            self.config.get(
                "feature_engineering.validation.fail_on_missing_named_window", True
            )
        )
        if not fail_on_missing_named or not self.name:
            return

        named_mask = self._windows.get_mask(self.name)
        named_exists = named_mask is not None and np.any(named_mask)

        if not named_exists:
            raise ValueError(
                f"Named time window '{self.name}' is missing or empty. "
                "Define it in feature_engineering.windows/custom windows "
                "or pass explicit ranges."
            )

    @property
    def windows(self) -> Any:
        """Access centralized time windows."""
        self._initialize_windows()
        return self._windows

    def add_result(self, key: str, value: Any, cols: Optional[List[str]] = None) -> None:
        """Add a result to the context."""
        self.results[key] = value
        if cols is not None:
            self.results[f"{key}_cols"] = cols

    def set_precomputed(self, precomputed: Optional[PrecomputedData]) -> None:
        """Set precomputed data and sync spatial modes and frequency bands."""
        self.precomputed = precomputed
        self._precomputed_ready = precomputed is not None

        if precomputed is not None:
            precomputed.spatial_modes = (
                list(self.spatial_modes) if self.spatial_modes else None
            )
            if precomputed.frequency_bands is not None:
                self.frequency_bands = dict(precomputed.frequency_bands)

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
