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
from eeg_pipeline.domain.features.constants import FEATURE_CATEGORIES, PRECOMPUTED_GROUP_CHOICES


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
        """Initialize windows if not provided."""
        self._ensure_windows()

    def _ensure_windows(self) -> None:
        if self._windows is None and self.epochs is not None:
            # Lazy import
            from eeg_pipeline.utils.analysis.windowing import TimeWindowSpec
            self._windows = TimeWindowSpec(
                times=self.epochs.times,
                config=self.config,
                sampling_rate=self.epochs.info["sfreq"],
                logger=self.logger,
            )

        fail_on_missing = bool(self.config.get("feature_engineering.validation.fail_on_missing_windows", False))
        if fail_on_missing and self._windows is not None:
            baseline_meta = self._windows.metadata.get("baseline")
            plateau_meta = self._windows.metadata.get("plateau")
            baseline_ok = bool(baseline_meta is not None and getattr(baseline_meta, "valid", False))
            plateau_ok = bool(plateau_meta is not None and getattr(plateau_meta, "valid", False))
            if not baseline_ok or not plateau_ok:
                raise ValueError(
                    "Missing required time windows for feature extraction: "
                    f"baseline_ok={baseline_ok}, plateau_ok={plateau_ok}. "
                    "Check time_frequency_analysis.baseline_window / plateau_window and epoch time range."
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
