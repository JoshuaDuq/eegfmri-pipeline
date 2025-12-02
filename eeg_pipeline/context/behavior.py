"""Behavior analysis context."""

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
    results: Dict[str, ComputationResult] = field(default_factory=dict)
    _data_loaded: bool = False

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
            _load_features_and_targets, load_epochs_for_analysis,
            extract_temperature_data, build_covariate_matrix, build_covariates_without_temp,
        )
        from eeg_pipeline.utils.io.general import deriv_features_path, read_tsv

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

            # Load features
            features_dir = deriv_features_path(self.deriv_root, self.subject)
            
            def _safe_read(filename: str) -> Optional[pd.DataFrame]:
                path = features_dir / filename
                if not path.exists():
                    return None
                try:
                    return read_tsv(path)
                except Exception as e:
                    self.logger.warning(f"Failed to read {filename}: {e}")
                    return None

            # Load main features file (contains power, microstates, aperiodic, itpc)
            self.power_df = _safe_read("features_eeg_direct.tsv")
            
            # Only load separate files if they contain unique features not in direct
            # Connectivity and PAC are separate pipelines, always load them
            self.connectivity_df = _safe_read("features_connectivity.tsv")
            self.pac_df = _safe_read("features_pac_trials.tsv")
            
            # These are included in features_eeg_direct.tsv, set to None to avoid duplicates
            self.microstates_df = None
            self.aperiodic_df = None
            self.itpc_df = None
            self.precomputed_df = None
            
            # Load targets
            targets_df = _safe_read("target_vas_ratings.tsv")
            if targets_df is not None:
                if targets_df.shape[1] == 1:
                    self.targets = pd.to_numeric(targets_df.iloc[:, 0], errors="coerce")
                else:
                    # Try to find a numeric column
                    numeric_cols = targets_df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        self.targets = pd.to_numeric(targets_df[numeric_cols[0]], errors="coerce")
                    else:
                        self.logger.warning("No numeric columns found in targets file")
                        self.targets = None
            else:
                self.targets = None

            if self.targets is None or len(self.targets) == 0:
                self.logger.warning("No targets found")
                return False

            # Load temperature
            self.temperature, self.temperature_column = extract_temperature_data(
                self.aligned_events, self.config
            )

            # Build covariates
            self.covariates_df = build_covariate_matrix(
                self.aligned_events, self.partial_covars, self.config
            )
            self.covariates_without_temp_df = build_covariates_without_temp(
                self.covariates_df, self.temperature_column
            )

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
