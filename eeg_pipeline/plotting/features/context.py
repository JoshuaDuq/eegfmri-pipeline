"""
Feature Visualization Context and Registry
===========================================

Unified module for feature visualization state management and plot registration.
This consolidates context/visualization.FeaturePlotContext and plotting/manager.py.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    import matplotlib.figure

import matplotlib.pyplot as plt
import mne
import pandas as pd

from eeg_pipeline.infra.tsv import read_table as _read_table
from eeg_pipeline.plotting.io.figures import save_fig as _save_fig
from eeg_pipeline.plotting.config import PlotConfig, get_plot_config
from eeg_pipeline.plotting.core.registry import (
    CategorizedPlotManager,
    CategorizedPlotRegistry,
    PlotterFunc,
)


class VisualizationRegistry(CategorizedPlotRegistry["FeaturePlotContext"]):
    """Registry for feature plotting functions."""


# Feature table specifications: (attribute_name, file_stem, extensions, mode)
_FEATURE_TABLE_SPECS: List[Tuple[str, str, List[str], str]] = [
    ("power_df", "features_power", [".tsv"], "wide"),
    ("connectivity_df", "features_connectivity", [".tsv"], "wide"),
    ("aperiodic_df", "features_aperiodic", [".tsv"], "wide"),
    ("erds_df", "features_erds", [".tsv"], "wide"),
    ("bursts_df", "features_bursts", [".tsv"], "wide"),
    ("quality_df", "features_quality", [".tsv"], "wide"),
    ("spectral_df", "features_spectral", [".tsv"], "wide"),
    ("ratios_df", "features_ratios", [".tsv"], "wide"),
    ("asymmetry_df", "features_asymmetry", [".tsv"], "wide"),
    ("complexity_df", "features_complexity", [".tsv"], "wide"),
    ("pac_df", "features_pac", [".tsv"], "wide"),
    ("pac_trials_df", "features_pac_trials", [".tsv"], "wide"),
    ("pac_time_df", "features_pac_time", [".tsv"], "long"),
    ("itpc_df", "features_itpc", [".tsv"], "wide"),
    ("temporal_df", "features_temporal", [".tsv"], "wide"),
    ("erp_df", "features_erp", [".tsv"], "wide"),
]


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
    connectivity_df: Optional[pd.DataFrame] = None
    aperiodic_df: Optional[pd.DataFrame] = None
    erp_df: Optional[pd.DataFrame] = None
    erds_df: Optional[pd.DataFrame] = None
    bursts_df: Optional[pd.DataFrame] = None
    quality_df: Optional[pd.DataFrame] = None
    spectral_df: Optional[pd.DataFrame] = None
    ratios_df: Optional[pd.DataFrame] = None
    asymmetry_df: Optional[pd.DataFrame] = None
    pac_df: Optional[pd.DataFrame] = None
    complexity_df: Optional[pd.DataFrame] = None
    pac_trials_df: Optional[pd.DataFrame] = None
    pac_time_df: Optional[pd.DataFrame] = None
    itpc_df: Optional[pd.DataFrame] = None
    temporal_df: Optional[pd.DataFrame] = None

    window_ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    time_range_suffixes: List[str] = field(default_factory=list)
    
    epochs: Optional[mne.Epochs] = None
    tfr: Optional[mne.time_frequency.EpochsTFR] = None
    _tfr_cached: bool = False
    
    def subdir(self, name: str) -> Path:
        """Get subdirectory path within plots directory."""
        return self.plots_dir / name
    
    @property
    def plot_cfg(self) -> PlotConfig:
        """Get plotting configuration."""
        return get_plot_config(self.config)
    
    def save(self, fig: "matplotlib.figure.Figure", path: Path) -> None:
        _save_fig(fig, path, formats=self.plot_cfg.formats, dpi=self.plot_cfg.dpi)
        plt.close(fig)
    
    def load_data(self) -> None:
        """Load all feature data into memory with time-range awareness."""
        self.logger.info("Loading feature data frames...")

        self._load_extraction_configs()
        self._apply_window_overrides()
        self._load_feature_tables()

        self.n_trials = self._infer_trial_count()
        if self.n_trials > 0:
            self.logger.info("Loaded %d trials", self.n_trials)

    def _load_extraction_configs(self) -> None:
        """Load extraction configuration files and extract window ranges.
        
        Searches for configs in per-category metadata folders: 
        features/{category}/metadata/extraction_config*.json
        """
        self.window_ranges = {}
        self.time_range_suffixes = []
        
        ExtractionConfig = Tuple[str, Optional[str], Optional[float], Optional[float]]
        configs: List[ExtractionConfig] = []
        
        # Search per-category metadata folders (new location)
        for category_dir in sorted(self.features_dir.iterdir()):
            if not category_dir.is_dir() or category_dir.name == "metadata":
                continue
            metadata_dir = category_dir / "metadata"
            if metadata_dir.is_dir():
                for path in sorted(metadata_dir.glob("extraction_config*.json")):
                    config_entry = self._parse_extraction_config(path)
                    if config_entry:
                        configs.append(config_entry)

        name_to_suffix: Dict[str, Optional[str]] = {}
        for suffix, name, tmin, tmax in configs:
            if name:
                name_key = str(name).strip().lower()
                name_to_suffix[name_key] = suffix
                if tmin is not None and tmax is not None:
                    try:
                        self.window_ranges[name_key] = (float(tmin), float(tmax))
                    except (TypeError, ValueError):
                        self.logger.warning(
                            "Invalid time range for %s: tmin=%s tmax=%s",
                            name,
                            str(tmin),
                            str(tmax),
                        )

        preferred = ["baseline", "active"]
        ordered_suffixes: List[str] = []
        for name in preferred:
            suffix = name_to_suffix.get(name)
            if suffix:
                ordered_suffixes.append(suffix)

        for suffix, name, _, _ in configs:
            if suffix and suffix not in ordered_suffixes:
                ordered_suffixes.append(suffix)

        self.time_range_suffixes = ordered_suffixes

        if self.window_ranges:
            formatted = ", ".join(
                f"{name}=[{rng[0]:.2f}, {rng[1]:.2f}]"
                for name, rng in self.window_ranges.items()
            )
            self.logger.info("Detected extracted windows: %s", formatted)
    
    def _parse_extraction_config(self, path: Path) -> Optional[Tuple[Optional[str], Optional[str], Optional[float], Optional[float]]]:
        """Parse an extraction config JSON file.
        
        Returns:
            (suffix, name, tmin, tmax) tuple or None if parsing fails
        """
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            self.logger.warning("Failed to read extraction config %s: %s", path, exc)
            return None

        stem = path.stem
        suffix = None
        if stem != "extraction_config":
            suffix = stem.replace("extraction_config_", "", 1)
        name = payload.get("name") or suffix
        tmin = payload.get("tmin")
        tmax = payload.get("tmax")
        return (suffix, name, tmin, tmax)

    def _apply_window_overrides(self) -> None:
        """Apply window range overrides to config if available."""
        if not self.window_ranges or self.config is None:
            return

        try:
            self.config = copy.deepcopy(self.config)
        except (TypeError, AttributeError, ValueError):
            # Config may not be deepcopy-able, continue with original
            pass

        baseline = self.window_ranges.get("baseline")
        active = self.window_ranges.get("active")
        if baseline:
            self._set_config_window("baseline", baseline)
        if active:
            self._set_config_window("active", active)

    def _set_config_window(self, label: str, window: Tuple[float, float]) -> None:
        if self.config is None:
            return
        win = [float(window[0]), float(window[1])]
        self._set_config_value(f"time_frequency_analysis.{label}_window", win)
        self._set_config_value(f"feature_engineering.features.{label}_window", win)

    def _set_config_value(self, key: str, value: Any) -> None:
        """Set nested config value, creating intermediate dicts as needed."""
        if self.config is None:
            return
        
        # Try direct assignment first (for dict-like configs)
        try:
            self.config[key] = value
            return
        except (TypeError, KeyError, AttributeError):
            # Config doesn't support direct assignment, use nested path
            pass
        
        keys = key.split(".")
        current = self.config
        for part in keys[:-1]:
            if part not in current or not isinstance(current[part], dict):
                current[part] = {}
            current = current[part]
        current[keys[-1]] = value

    def _load_feature_tables(self) -> None:
        """Load all feature data tables from TSV files."""
        for attr_name, stem, exts, mode in _FEATURE_TABLE_SPECS:
            paths = self._collect_feature_paths(stem, exts)
            df = self._load_feature_set(paths, mode=mode, stem=stem)
            if df is not None and not df.empty:
                setattr(self, attr_name, df)

    def _collect_feature_paths(self, stem: str, exts: Sequence[str]) -> List[Path]:
        """Collect feature file paths, prioritizing base files and known suffixes."""
        paths: List[Path] = []
        suffixes = self.time_range_suffixes

        # First, try base files and known suffix variants
        for ext in exts:
            base_path = self.features_dir / f"{stem}{ext}"
            if base_path.exists():
                paths.append(base_path)

            for suffix in suffixes:
                candidate_path = self.features_dir / f"{stem}_{suffix}{ext}"
                if candidate_path.exists():
                    paths.append(candidate_path)

        if paths:
            return self._dedupe_paths(paths)

        # Fallback: glob for any matching pattern
        for ext in exts:
            pattern = f"{stem}_*{ext}"
            for path in sorted(self.features_dir.glob(pattern)):
                if not self._is_feature_payload(path):
                    continue
                paths.append(path)

        return self._dedupe_paths(paths)

    def _load_feature_set(
        self,
        paths: Sequence[Path],
        *,
        mode: str,
        stem: str,
    ) -> Optional[pd.DataFrame]:
        """Load and combine feature data from multiple files.
        
        Args:
            paths: File paths to load
            mode: "wide" (column-wise concatenation) or "long" (row-wise)
            stem: Base filename stem for suffix extraction
            
        Returns:
            Combined DataFrame or None if no valid data found
        """
        if not paths:
            return None

        data_frames: List[pd.DataFrame] = []
        base_len: Optional[int] = None

        for path in paths:
            df = self._safe_read_table(path)
            if df is None or df.empty:
                continue

            df = df.reset_index(drop=True)
            if base_len is None:
                base_len = len(df)
            elif len(df) != base_len and mode == "wide":
                self.logger.warning(
                    "Skipping %s (rows=%d) due to length mismatch (expected %d)",
                    path.name,
                    len(df),
                    base_len,
                )
                continue

            if mode == "long":
                suffix = self._suffix_from_path(path, stem)
                if suffix and "segment" not in df.columns:
                    df = df.copy()
                    df["segment"] = suffix

            data_frames.append(df)

        if not data_frames:
            return None

        if mode == "long":
            combined = pd.concat(data_frames, axis=0, ignore_index=True)
        else:
            combined = pd.concat(data_frames, axis=1)

        if combined.columns.duplicated().any():
            dupes = combined.columns[combined.columns.duplicated()].unique().tolist()
            dupe_preview = ", ".join(map(str, dupes[:6]))
            self.logger.warning(
                "Dropping %d duplicate columns (%s)",
                len(dupes),
                dupe_preview,
            )
            combined = combined.loc[:, ~combined.columns.duplicated()]

        return combined

    def _safe_read_table(self, path: Path) -> Optional[pd.DataFrame]:
        """Safely read table file, returning None on failure."""
        if not path.exists():
            return None
        try:
            return _read_table(path)
        except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            self.logger.warning("Failed to read %s: %s", path, exc)
            return None

    def _is_metadata_file(self, path: Path) -> bool:
        """Check if path points to a metadata file (not feature data)."""
        stem_lower = path.stem.lower()
        excluded_keywords = ["columns", "config", "qc", "manifest"]
        return any(keyword in stem_lower for keyword in excluded_keywords)

    def _suffix_from_path(self, path: Path, stem: str) -> Optional[str]:
        """Extract time range suffix from feature file path."""
        if self._is_metadata_file(path):
            return None
        base = path.stem
        if base == stem:
            return None
        prefix = f"{stem}_"
        if base.startswith(prefix):
            return base[len(prefix):]
        return None

    def _is_feature_payload(self, path: Path) -> bool:
        """Check if path points to a feature data file (not metadata)."""
        return not self._is_metadata_file(path)

    def _dedupe_paths(self, paths: Sequence[Path]) -> List[Path]:
        """Remove duplicate paths while preserving order."""
        seen = set()
        unique_paths = []
        for path in paths:
            if path not in seen:
                seen.add(path)
                unique_paths.append(path)
        return unique_paths

    def _infer_trial_count(self) -> int:
        """Infer trial count from first available non-empty dataframe."""
        candidate_dataframes = [
            self.power_df,
            self.connectivity_df,
            self.aperiodic_df,
            self.erp_df,
            self.erds_df,
            self.spectral_df,
            self.ratios_df,
            self.asymmetry_df,
            self.complexity_df,
            self.bursts_df,
            self.itpc_df,
            self.pac_trials_df,
            self.temporal_df,
            self.quality_df,
        ]
        for df in candidate_dataframes:
            if df is not None and not df.empty:
                return len(df)
        return 0
    
    def get_or_compute_tfr(self) -> Optional[mne.time_frequency.EpochsTFR]:
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
                logger=self.logger,
            )
            self._tfr_cached = True
            self.logger.info("TFR computed and cached")
            return self.tfr
        except Exception as exc:
            self.logger.warning("Failed to compute TFR: %s", exc)
            return None


class VisualizationManager(CategorizedPlotManager["FeaturePlotContext"]):
    """Orchestrates the execution of registered plotters."""

    def __init__(self, ctx: FeaturePlotContext):
        super().__init__(ctx, logger=ctx.logger)

    def run_category(
        self,
        category: str,
        *,
        plotters: Optional[List[Tuple[str, PlotterFunc["FeaturePlotContext"]]]] = None,
    ) -> None:
        """Run plotters for a specific category."""
        if plotters is None:
            plotters = VisualizationRegistry.get_plotters(category)
        from eeg_pipeline.plotting.style import use_style

        with use_style():
            super().run_category(category, plotters=plotters)

    def run_all(self) -> Dict[str, Path]:
        """Run all registered plotters in preferred order."""
        categories = VisualizationRegistry.get_categories()
        preferred_order = [
            "power",
            "connectivity",
            "complexity",
            "aperiodic",
            "erds",
            "spectral",
            "ratios",
            "asymmetry",
            "bursts",
            "temporal",
            "itpc",
            "pac",
            "erp",
            "quality",
        ]

        ordered_categories = [c for c in preferred_order if c in categories]
        remaining_categories = [c for c in categories if c not in preferred_order]
        ordered_categories.extend(remaining_categories)

        for category in ordered_categories:
            self.run_category(category)

        return self.saved_plots


__all__ = [
    "FeaturePlotContext",
    "VisualizationRegistry",
    "VisualizationManager",
    "PlotterFunc",
]
