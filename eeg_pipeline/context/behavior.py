"""
Behavior Analysis Context
=========================

Context objects for behavior analysis data loading and state management.

Note: For new code, consider using BehaviorPipeline from pipelines/behavior.py
which provides a higher-level interface with the same functionality.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from eeg_pipeline.utils.analysis.stats.correlation import CorrelationRecord


class ComputationStatus(Enum):
    PENDING = auto()
    RUNNING = auto()
    SUCCESS = auto()
    FAILED = auto()
    SKIPPED = auto()


@dataclass
class ComputationResult:
    """Result of a computation."""
    name: str
    status: ComputationStatus
    records: List[CorrelationRecord] = field(default_factory=list)
    dataframe: Optional[pd.DataFrame] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def n_significant(self) -> int:
        return sum(1 for r in self.records if r.is_significant)

    def to_dataframe(self) -> pd.DataFrame:
        if self.dataframe is not None:
            return self.dataframe
        return pd.DataFrame([r.to_dict() for r in self.records]) if self.records else pd.DataFrame()


@dataclass
class BehaviorContext:
    """Shared context for behavior analysis."""
    subject: str
    task: str
    config: Any
    logger: Any
    deriv_root: Path
    stats_dir: Path
    use_spearman: bool = True
    bootstrap: int = 0
    n_perm: int = 100
    rng: Optional[np.random.Generator] = None
    partial_covars: Optional[List[str]] = None
    control_temperature: bool = True
    control_trial_order: bool = True
    compute_change_scores: bool = True
    compute_reliability: bool = True
    
    epochs: Any = None
    epochs_info: Any = None
    aligned_events: Optional[pd.DataFrame] = None
    power_df: Optional[pd.DataFrame] = None
    connectivity_df: Optional[pd.DataFrame] = None
    microstates_df: Optional[pd.DataFrame] = None
    aperiodic_df: Optional[pd.DataFrame] = None
    precomputed_df: Optional[pd.DataFrame] = None
    itpc_df: Optional[pd.DataFrame] = None
    pac_df: Optional[pd.DataFrame] = None
    targets: Optional[pd.Series] = None
    temperature: Optional[pd.Series] = None
    temperature_column: Optional[str] = None
    covariates_df: Optional[pd.DataFrame] = None
    covariates_without_temp_df: Optional[pd.DataFrame] = None
    group_ids: Optional[np.ndarray] = None
    results: Dict[str, ComputationResult] = field(default_factory=dict)
    _data_loaded: bool = False
    _change_scores_added: bool = False

    @property
    def method(self) -> str:
        return "spearman" if self.use_spearman else "pearson"

    @property
    def n_trials(self) -> int:
        if self.targets is not None: return len(self.targets)
        if self.power_df is not None: return len(self.power_df)
        return 0

    @property
    def power_bands(self) -> List[str]:
        from eeg_pipeline.utils.config.loader import get_frequency_band_names
        return get_frequency_band_names(self.config)

    def get_min_samples(self, sample_type: str = "default") -> int:
        """Get minimum samples threshold from config."""
        # Avoid circular imports by using local constants or passed config
        defaults = {"channel": 10, "roi": 20, "default": 5, "edge": 30, "temporal": 15}
        if self.config is None:
            return defaults.get(sample_type, 5)
        return int(self.config.get(f"behavior_analysis.min_samples.{sample_type}",
                             defaults.get(sample_type, 5)))

    @property
    def min_samples_channel(self) -> int:
        """Minimum samples for channel-level statistics."""
        return self.get_min_samples("channel")

    @property
    def min_samples_roi(self) -> int:
        """Minimum samples for ROI-level statistics."""
        return self.get_min_samples("roi")

    @property
    def has_temperature(self) -> bool:
        return self.temperature is not None and len(self.temperature) > 0

    @property
    def has_covariates(self) -> bool:
        return self.covariates_df is not None and not self.covariates_df.empty

    def load_data(self) -> bool:
        """Load all features and targets. Returns True if successful."""
        if self._data_loaded:
            return self.targets is not None

        from eeg_pipeline.utils.data.loading import (
            load_epochs_for_analysis, load_feature_bundle,
            extract_temperature_data, build_covariate_matrix, build_covariates_without_temp,
        )

        self.logger.info("Loading data...")
        try:
            self.epochs, self.aligned_events = load_epochs_for_analysis(
                self.subject, self.task, align="strict", preload=True,
                deriv_root=self.deriv_root, logger=self.logger, config=self.config
            )
            
            if self.epochs is None or self.aligned_events is None:
                self.logger.error("Failed to load epochs or events")
                return False

            self.epochs_info = self.epochs.info

            bundle = load_feature_bundle(
                self.subject, self.deriv_root, self.logger, include_targets=True
            )
            
            self.power_df = bundle.power_df
            self.connectivity_df = bundle.connectivity_df
            self.pac_df = bundle.pac_trials_df
            self.microstates_df = bundle.microstate_df
            self.aperiodic_df = bundle.aperiodic_df
            self.itpc_df = bundle.itpc_df
            # Consolidate any precomputed/derived tables
            precomputed_sources = [
                bundle.all_features_df,
                bundle.complexity_df,
                bundle.dynamics_df,
            ]
            self.precomputed_df = None
            for df in precomputed_sources:
                if df is None or df.empty:
                    continue
                self.precomputed_df = df if self.precomputed_df is None else pd.concat(
                    [self.precomputed_df, df], axis=1
                )
            feature_tables = {
                "power": self.power_df,
                "connectivity": self.connectivity_df,
                "pac": self.pac_df,
                "microstates": self.microstates_df,
                "aperiodic": self.aperiodic_df,
                "itpc": self.itpc_df,
                "precomputed": self.precomputed_df,
            }
            feature_counts = {
                name: 0 if table is None or table.empty else table.shape[1]
                for name, table in feature_tables.items()
            }
            if all(count == 0 for count in feature_counts.values()):
                self.logger.error(
                    "Feature bundle is empty for sub-%s; aborting analysis", self.subject
                )
                return False
            else:
                self.logger.info(
                    "Feature coverage loaded: %s",
                    ", ".join(f"{k}={v}" for k, v in feature_counts.items() if v > 0),
                )
            self.targets = bundle.targets

            if self.targets is None or len(self.targets) == 0:
                self.logger.warning("No targets found")
                return False

            self.temperature, self.temperature_column = extract_temperature_data(
                self.aligned_events, self.config
            )

            self.covariates_df = build_covariate_matrix(
                self.aligned_events, self.partial_covars, self.config
            )
            self.covariates_without_temp_df = build_covariates_without_temp(
                self.covariates_df, self.temperature_column
            )

            # Align potential grouping variables for LOSO/clustered evaluation
            for candidate in ["subject", "session", "run", "block"]:
                if self.aligned_events is not None and candidate in self.aligned_events.columns:
                    try:
                        self.group_ids = np.asarray(self.aligned_events[candidate])
                        break
                    except Exception:
                        self.group_ids = None

            self._data_loaded = True
            return True

        except Exception as e:
            self.logger.error(f"Data loading failed: {e}")
            return False

    def add_result(self, result: ComputationResult) -> None:
        """Add a computation result."""
        self.results[result.name] = result


###################################################################
# AnalysisConfig
###################################################################


@dataclass
class AnalysisConfig:
    """Lightweight configuration extracted from BehaviorContext."""
    subject: str
    config: Any
    logger: Any
    rng: np.random.Generator
    stats_dir: Optional[Path] = None
    bootstrap: int = 0
    n_perm: int = 0
    use_spearman: bool = True
    method: str = "spearman"
    min_samples_channel: int = 10
    min_samples_roi: int = 20
    groups: Optional[np.ndarray] = None

    @classmethod
    def from_context(cls, ctx: "BehaviorContext") -> "AnalysisConfig":
        return cls(
            subject=ctx.subject,
            config=ctx.config,
            logger=ctx.logger,
            rng=ctx.rng or np.random.default_rng(42),
            stats_dir=ctx.stats_dir,
            bootstrap=ctx.bootstrap,
            n_perm=ctx.n_perm,
            use_spearman=ctx.use_spearman,
            method=ctx.method,
            min_samples_channel=ctx.min_samples_channel,
            min_samples_roi=ctx.min_samples_roi,
        )
