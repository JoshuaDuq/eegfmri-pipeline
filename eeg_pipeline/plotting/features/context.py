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
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import mne
import pandas as pd

from eeg_pipeline.infra.paths import ensure_dir
from eeg_pipeline.infra.tsv import read_table as _read_table
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
    all_features: Optional[pd.DataFrame] = None

    window_ranges: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    time_range_suffixes: List[str] = field(default_factory=list)
    
    epochs: Optional[mne.Epochs] = None
    tfr: Optional[mne.time_frequency.EpochsTFR] = None
    _tfr_cached: bool = False
    
    def subdir(self, name: str) -> Path:
        path = self.plots_dir / name
        return path
    
    @property
    def plot_cfg(self):
        return get_plot_config(self.config)
    
    def save(self, fig, path: Path) -> None:
        _save_fig(fig, path, formats=self.plot_cfg.formats, dpi=self.plot_cfg.dpi)
        plt.close(fig)
    
    def load_data(self) -> None:
        """Load all feature data into memory with time-range awareness."""
        self.logger.info("Loading feature data frames...")

        self._load_extraction_configs()
        self._apply_window_overrides()
        self._load_feature_tables()

        if self.all_features is not None:
            self.logger.info("Loaded features_all.tsv with %d columns", self.all_features.shape[1])
        else:
            self.logger.warning("features_all.tsv could not be loaded")

        self.n_trials = self._infer_trial_count()
        if self.n_trials > 0:
            self.logger.info("Loaded %d trials", self.n_trials)

    def _load_extraction_configs(self) -> None:
        self.window_ranges = {}
        self.time_range_suffixes = []
        configs: List[Tuple[str, Optional[str], Optional[float], Optional[float]]] = []

        for path in sorted(self.features_dir.glob("extraction_config*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                self.logger.warning("Failed to read extraction config %s: %s", path, exc)
                continue

            stem = path.stem
            suffix = None
            if stem != "extraction_config":
                suffix = stem.replace("extraction_config_", "", 1)
            name = payload.get("name") or suffix
            tmin = payload.get("tmin")
            tmax = payload.get("tmax")
            configs.append((suffix, name, tmin, tmax))

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

    def _apply_window_overrides(self) -> None:
        if not self.window_ranges or self.config is None:
            return

        try:
            self.config = copy.deepcopy(self.config)
        except Exception:
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
        self._set_config_value(f"feature_engineering.windows.{label}_window", win)
        self._set_config_value(f"feature_engineering.features.{label}_window", win)

    def _set_config_value(self, key: str, value: Any) -> None:
        if self.config is None:
            return
        try:
            self.config[key] = value
            return
        except Exception:
            pass
        keys = key.split(".")
        current = self.config
        for part in keys[:-1]:
            if part not in current or not isinstance(current[part], dict):
                current[part] = {}
            current = current[part]
        current[keys[-1]] = value

    def _load_feature_tables(self) -> None:
        table_specs = [
            ("power_df", "features_power", [".tsv"], "wide"),
            ("connectivity_df", "features_connectivity", [".parquet", ".tsv"], "wide"),
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
            ("all_features", "features_all", [".tsv"], "wide"),
        ]

        for attr_name, stem, exts, mode in table_specs:
            paths = self._collect_feature_paths(stem, exts)
            df = self._load_feature_set(paths, mode=mode, stem=stem)
            if df is not None and not df.empty:
                setattr(self, attr_name, df)

    def _collect_feature_paths(self, stem: str, exts: Sequence[str]) -> List[Path]:
        paths: List[Path] = []
        suffixes = self.time_range_suffixes

        for ext in exts:
            base = self.features_dir / f"{stem}{ext}"
            if base.exists():
                paths.append(base)

            for suffix in suffixes:
                candidate = self.features_dir / f"{stem}_{suffix}{ext}"
                if candidate.exists():
                    paths.append(candidate)

        if paths:
            return self._dedupe_paths(paths)

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
            self.logger.warning("Dropping %d duplicate columns (%s)", len(dupes), ", ".join(map(str, dupes[:6])))
            combined = combined.loc[:, ~combined.columns.duplicated()]

        return combined

    def _safe_read_table(self, path: Path) -> Optional[pd.DataFrame]:
        if not path.exists():
            return None
        try:
            return _read_table(path)
        except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            self.logger.warning("Failed to read %s: %s", path, exc)
            return None

    def _suffix_from_path(self, path: Path, stem: str) -> Optional[str]:
        base = path.stem
        if "_columns" in base or "_qc" in base or "_config" in base:
            return None
        if base == stem:
            return None
        prefix = f"{stem}_"
        if base.startswith(prefix):
            return base[len(prefix):]
        return None

    def _is_feature_payload(self, path: Path) -> bool:
        stem = path.stem.lower()
        if "columns" in stem:
            return False
        if "config" in stem:
            return False
        if stem.endswith("_manifest"):
            return False
        return True

    def _dedupe_paths(self, paths: Sequence[Path]) -> List[Path]:
        seen = set()
        out = []
        for path in paths:
            if path in seen:
                continue
            seen.add(path)
            out.append(path)
        return out

    def _infer_trial_count(self) -> int:
        candidates = [
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
            self.all_features,
        ]
        for df in candidates:
            if df is not None and not df.empty:
                return len(df)
        return 0
    
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

    def run_category(self, category: str, *, plotters: Optional[List[Tuple[str, PlotterFunc["FeaturePlotContext"]]]] = None) -> None:
        if plotters is None:
            plotters = VisualizationRegistry.get_plotters(category)
        from eeg_pipeline.plotting.style import use_style

        with use_style():
            super().run_category(category, plotters=plotters)

    def run_all(self) -> Dict[str, Path]:
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
