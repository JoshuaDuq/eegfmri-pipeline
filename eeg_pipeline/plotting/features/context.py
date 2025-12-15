"""
Feature Visualization Context and Registry
===========================================

Unified module for feature visualization state management and plot registration.
This consolidates context/visualization.FeaturePlotContext and plotting/manager.py.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import mne
import pandas as pd

from eeg_pipeline.io.paths import ensure_dir
from eeg_pipeline.io.tsv import read_tsv as _read_tsv
from eeg_pipeline.plotting.io.figures import save_fig as _save_fig
from eeg_pipeline.plotting.config import get_plot_config
from eeg_pipeline.plotting.core.registry import (
    CategorizedPlotManager,
    CategorizedPlotRegistry,
    PlotterFunc,
)


class VisualizationRegistry(CategorizedPlotRegistry["FeaturePlotContext"]):
    """Registry for feature plotting functions."""


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
    aligned_events: Optional[pd.DataFrame] = None
    
    power_df: Optional[pd.DataFrame] = None
    microstate_df: Optional[pd.DataFrame] = None
    connectivity_df: Optional[pd.DataFrame] = None
    pac_df: Optional[pd.DataFrame] = None
    complexity_df: Optional[pd.DataFrame] = None
    dynamics_df: Optional[pd.DataFrame] = None
    aperiodic_df: Optional[pd.DataFrame] = None
    pac_trials_df: Optional[pd.DataFrame] = None
    pac_time_df: Optional[pd.DataFrame] = None
    itpc_df: Optional[pd.DataFrame] = None
    all_features: Optional[pd.DataFrame] = None
    
    epochs: Optional[mne.Epochs] = None
    tfr: Optional[mne.time_frequency.EpochsTFR] = None
    _tfr_cached: bool = False
    
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
    
    def load_data(self) -> None:
        """Load all feature data into memory using the canonical loader."""
        from eeg_pipeline.utils.data.loading import load_feature_bundle
        
        self.logger.info("Loading feature data frames...")
        
        deriv_root = self.features_dir.parent.parent.parent
        bundle = load_feature_bundle(self.subject, deriv_root, self.logger)
        
        self.power_df = bundle.power_df
        self.connectivity_df = bundle.connectivity_df
        self.complexity_df = bundle.complexity_df
        self.dynamics_df = bundle.dynamics_df
        self.aperiodic_df = bundle.aperiodic_df
        self.microstate_df = bundle.microstate_df
        self.pac_df = bundle.pac_df
        self.pac_trials_df = bundle.pac_trials_df
        self.pac_time_df = bundle.pac_time_df
        self.itpc_df = bundle.itpc_df
        self.all_features = bundle.all_features_df
        
        if self.all_features is not None:
            self.logger.info(f"Loaded features_all.tsv with {self.all_features.shape[1]} columns")
        else:
            self.logger.warning("features_all.tsv could not be loaded")
            
        self.n_trials = bundle.n_trials
        if self.n_trials > 0:
            self.logger.info(f"Loaded {self.n_trials} trials")
    
    def get_or_compute_tfr(self, freqs=None, n_cycles=None):
        """Get cached TFR or compute if not available."""
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
                logger=self.logger
            )
            self._tfr_cached = True
            self.logger.info("TFR computed and cached")
            return self.tfr
        except Exception as e:
            self.logger.warning(f"Failed to compute TFR: {e}")
            return None


class VisualizationManager(CategorizedPlotManager["FeaturePlotContext"]):
    """Orchestrates the execution of registered plotters."""

    def __init__(self, ctx: FeaturePlotContext):
        super().__init__(ctx, logger=ctx.logger)

    def run_category(self, category: str) -> None:
        plotters = VisualizationRegistry.get_plotters(category)
        from eeg_pipeline.plotting.style import use_style

        with use_style():
            super().run_category(category, plotters=plotters)

    def run_all(self) -> Dict[str, Path]:
        categories = VisualizationRegistry.get_categories()
        preferred_order = [
            "power",
            "connectivity",
            "microstates",
            "pac",
            "complexity",
            "burst",
            "erds",
            "aperiodic",
            "itpc",
        ]

        ordered = [c for c in preferred_order if c in categories]
        ordered += [c for c in categories if c not in preferred_order]

        for cat in ordered:
            self.run_category(cat)

        return self.saved_plots


__all__ = [
    "FeaturePlotContext",
    "VisualizationRegistry",
    "VisualizationManager",
    "PlotterFunc",
]
