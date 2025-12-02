"""
Visualization Context
=====================

Context object for holding state during feature visualization.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import matplotlib.pyplot as plt
import mne

from eeg_pipeline.utils.io.general import ensure_dir, read_tsv as _read_tsv, save_fig as _save_fig
from eeg_pipeline.plotting.config import get_plot_config

@dataclass
class FeaturePlotContext:
    """Context for feature visualization."""
    subject: str
    plots_dir: Path
    features_dir: Path
    config: Any = None
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger("eeg_pipeline.plotting"))
    n_trials: int = 0
    epochs_info: Optional[mne.Info] = None
    aligned_events: Optional[pd.DataFrame] = None  # For pain comparison
    
    # Loaded dataframes
    power_df: Optional[pd.DataFrame] = None
    microstate_df: Optional[pd.DataFrame] = None
    connectivity_df: Optional[pd.DataFrame] = None
    pac_df: Optional[pd.DataFrame] = None
    
    # Additional
    complexity_df: Optional[pd.DataFrame] = None
    dynamics_df: Optional[pd.DataFrame] = None
    aperiodic_df: Optional[pd.DataFrame] = None
    pac_trials_df: Optional[pd.DataFrame] = None
    pac_time_df: Optional[pd.DataFrame] = None
    itpc_df: Optional[pd.DataFrame] = None
    all_features: Optional[pd.DataFrame] = None
    
    # MNE Objects (for topographic/time-frequency plots)
    epochs: Optional[mne.Epochs] = None
    tfr: Optional[mne.time_frequency.EpochsTFR] = None
    _tfr_cached: bool = False  # Track if TFR has been computed
    
    def subdir(self, name: str) -> Path:
        path = self.plots_dir / name
        ensure_dir(path)
        return path
    
    @property
    def plot_cfg(self):
        return get_plot_config(self.config)
    
    def save(self, fig, path: Path) -> None:
        _save_fig(fig, path, formats=self.plot_cfg.formats, dpi=self.plot_cfg.dpi)
        plt.close(fig)
    
    def _read_tsv(self, filename: str) -> Optional[pd.DataFrame]:
        """Helper to read a TSV file from the features directory."""
        path = self.features_dir / filename
        if path.exists():
            self.logger.debug(f"Loading {filename}...")
            return _read_tsv(path)
        self.logger.debug(f"{filename} not found, skipping.")
        return None

    def load_data(self) -> None:
        """Load all feature data into memory."""
        self.logger.info("Loading feature data frames...")
        
        # Load core features
        self.power_df = self._read_tsv("features_eeg_direct.tsv")
        self.connectivity_df = self._read_tsv("features_connectivity.tsv")
        
        # Load complexity/dynamics features
        self.complexity_df = self._read_tsv("features_complexity.tsv")
        self.dynamics_df = self._read_tsv("features_dynamics.tsv")
        self.aperiodic_df = self._read_tsv("features_aperiodic.tsv")
        
        # Load microstates features
        self.microstate_df = self._read_tsv("features_microstates.tsv")
        
        # Load PAC features
        self.pac_df = self._read_tsv("features_pac.tsv") 
        self.pac_trials_df = self._read_tsv("features_pac_trials.tsv")
        self.pac_time_df = self._read_tsv("features_pac_time.tsv")
        
        # Load ITPC features
        self.itpc_df = self._read_tsv("features_itpc.tsv")
        
        # Load combined features for convenience
        self.all_features = self._read_tsv("features_all.tsv")
        
        if self.all_features is not None:
            self.logger.info(f"Loaded features_all.tsv with {self.all_features.shape[1]} columns")
        else:
            self.logger.warning("features_all.tsv could not be loaded")
            
        if self.power_df is not None:
            self.n_trials = len(self.power_df)
            self.logger.info(f"Loaded {self.n_trials} trials")
    
    def get_or_compute_tfr(self, freqs=None, n_cycles=None):
        """Get cached TFR or compute if not available.
        
        Caches TFR after first computation to avoid redundant processing.
        
        Args:
            freqs: Frequency array (uses config default if None)
            n_cycles: Number of cycles (uses config default if None)
        
        Returns:
            EpochsTFR object or None if epochs not available
        """
        if self.tfr is not None and self._tfr_cached:
            return self.tfr
        
        if self.epochs is None:
            self.logger.warning("Cannot compute TFR: no epochs available")
            return None
        
        try:
            from eeg_pipeline.utils.analysis.tfr import compute_tfr_for_visualization
            
            self.logger.info("Computing TFR (will be cached for subsequent plots)...")
            self.tfr = compute_tfr_for_visualization(
                self.epochs, 
                config=self.config,
                freqs=freqs,
                n_cycles=n_cycles,
                logger=self.logger
            )
            self._tfr_cached = True
            self.logger.info("TFR computed and cached")
            return self.tfr
        except Exception as e:
            self.logger.warning(f"Failed to compute TFR: {e}")
            return None
