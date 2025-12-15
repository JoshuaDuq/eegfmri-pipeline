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
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from eeg_pipeline.utils.analysis.stats.correlation import CorrelationRecord, compute_correlation
from eeg_pipeline.io.columns import pick_target_column
from eeg_pipeline.io.paths import deriv_features_path


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
    group_ids: Optional[Union[np.ndarray, pd.Series]] = None
    group_column: Optional[str] = None
    results: Dict[str, ComputationResult] = field(default_factory=dict)
    data_qc: Dict[str, Any] = field(default_factory=dict)
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
                self.subject,
                self.deriv_root,
                self.logger,
                include_targets=True,
                config=self.config,
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

            if self.aligned_events is None or len(self.aligned_events) != len(self.targets):
                self.logger.error(
                    "Trial alignment mismatch for sub-%s: aligned_events=%s, targets=%s",
                    self.subject,
                    0 if self.aligned_events is None else len(self.aligned_events),
                    len(self.targets),
                )
                return False

            features_dir = deriv_features_path(self.deriv_root, self.subject)
            manifest_path = features_dir / "trial_alignment.tsv"
            if not manifest_path.exists():
                raise FileNotFoundError(
                    f"Trial alignment manifest missing: {manifest_path}. "
                    f"Re-run feature extraction to generate aligned trial manifests."
                )

            manifest = pd.read_csv(manifest_path, sep="\t")
            if len(manifest) != len(self.targets) or len(manifest) != len(self.aligned_events):
                raise ValueError(
                    "Trial alignment manifest length mismatch for sub-%s: manifest=%d, targets=%d, aligned_events=%d"
                    % (self.subject, len(manifest), len(self.targets), len(self.aligned_events))
                )

            self.data_qc["trial_alignment_manifest"] = {
                "path": str(manifest_path),
                "n_trials": int(len(manifest)),
                "has_target_value": bool("target_value" in manifest.columns),
            }

            if "target_value" in manifest.columns:
                m = pd.to_numeric(manifest["target_value"], errors="coerce")
                t = pd.to_numeric(self.targets, errors="coerce")
                valid = m.notna() & t.notna()
                if int(valid.sum()) > 0:
                    diff = (m[valid].to_numpy(dtype=float) - t[valid].to_numpy(dtype=float))
                    self.data_qc["trial_alignment_manifest_target_value"] = {
                        "n_valid": int(valid.sum()),
                        "mean_abs_error": float(np.mean(np.abs(diff))),
                        "max_abs_error": float(np.max(np.abs(diff))),
                    }

            rating_col = None
            try:
                target_columns = list(self.config.get("event_columns.rating", []) or []) if self.config is not None else []
                rating_col = pick_target_column(self.aligned_events, target_columns=target_columns)
            except Exception:
                rating_col = None
            if rating_col is not None and rating_col in self.aligned_events.columns:
                events_rating = pd.to_numeric(self.aligned_events[rating_col], errors="coerce")
                targets_rating = pd.to_numeric(self.targets, errors="coerce")
                valid = events_rating.notna() & targets_rating.notna()
                if int(valid.sum()) >= 3:
                    r, p = compute_correlation(
                        events_rating[valid].values,
                        targets_rating[valid].values,
                        method="spearman",
                    )
                    diff = (events_rating[valid].to_numpy(dtype=float) - targets_rating[valid].to_numpy(dtype=float))
                    self.data_qc["events_targets_rating_consistency"] = {
                        "events_rating_column": str(rating_col),
                        "n_valid": int(valid.sum()),
                        "spearman_r": float(r) if np.isfinite(r) else np.nan,
                        "spearman_p": float(p) if np.isfinite(p) else np.nan,
                        "mean_abs_error": float(np.mean(np.abs(diff))) if len(diff) else np.nan,
                        "max_abs_error": float(np.max(np.abs(diff))) if len(diff) else np.nan,
                    }

            base_index = self.targets.index

            def _align_feature_df(name: str, df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
                if df is None or df.empty:
                    return df
                if len(df) != len(base_index):
                    self.logger.error(
                        "Feature table length mismatch for %s: %s rows vs %s targets for sub-%s",
                        name,
                        len(df),
                        len(base_index),
                        self.subject,
                    )
                    return None
                if not df.index.equals(base_index):
                    df = df.copy()
                    df.index = base_index
                return df

            self.power_df = _align_feature_df("power", self.power_df)
            self.connectivity_df = _align_feature_df("connectivity", self.connectivity_df)
            self.pac_df = _align_feature_df("pac", self.pac_df)
            self.microstates_df = _align_feature_df("microstates", self.microstates_df)
            self.aperiodic_df = _align_feature_df("aperiodic", self.aperiodic_df)
            self.itpc_df = _align_feature_df("itpc", self.itpc_df)
            self.precomputed_df = _align_feature_df("precomputed", self.precomputed_df)

            if all(
                df is None or df.empty
                for df in [
                    self.power_df,
                    self.connectivity_df,
                    self.pac_df,
                    self.microstates_df,
                    self.aperiodic_df,
                    self.itpc_df,
                    self.precomputed_df,
                ]
            ):
                self.logger.error("All feature tables failed alignment for sub-%s", self.subject)
                return False

            self.temperature, self.temperature_column = extract_temperature_data(
                self.aligned_events, self.config
            )

            self.covariates_df = build_covariate_matrix(
                self.aligned_events, self.partial_covars, self.config
            )

            def _sanitize_covariates(cov: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
                if cov is None or cov.empty:
                    return None
                cov = cov.copy()
                if self.aligned_events is not None and not cov.index.equals(self.aligned_events.index):
                    cov.index = self.aligned_events.index
                for col in list(cov.columns):
                    cov[col] = pd.to_numeric(cov[col], errors="coerce")
                cov = cov.replace([np.inf, -np.inf], np.nan)
                cov = cov.dropna(axis=1, how="all")
                if cov.empty:
                    return None
                constant_cols = [
                    c
                    for c in cov.columns
                    if int(pd.to_numeric(cov[c], errors="coerce").nunique(dropna=True)) <= 1
                ]
                if constant_cols:
                    cov = cov.drop(columns=constant_cols, errors="ignore")
                return None if cov.empty else cov

            self.covariates_df = _sanitize_covariates(self.covariates_df)

            self.data_qc["trial_order"] = {"enabled": bool(self.control_trial_order)}
            if self.control_trial_order and self.aligned_events is not None:
                trial_candidates = [
                    "trial_index",
                    "trial_in_run",
                    "trial",
                    "trial_number",
                ]
                trial_col = None
                for cand in trial_candidates:
                    if cand in self.aligned_events.columns:
                        trial_col = cand
                        break

                if trial_col is None:
                    self.control_trial_order = False
                    self.data_qc["trial_order"].update({"enabled": False, "reason": "missing_trial_column"})
                else:
                    trial_index_series = pd.to_numeric(self.aligned_events[trial_col], errors="coerce")
                    max_missing = float(
                        self.config.get("behavior_analysis.trial_order.max_missing_fraction", 0.1)
                        if self.config is not None
                        else 0.1
                    )
                    missing_fraction = float(trial_index_series.isna().mean()) if len(trial_index_series) else 1.0
                    n_unique = int(trial_index_series.dropna().nunique())
                    is_monotonic = bool(trial_index_series.dropna().is_monotonic_increasing)

                    self.data_qc["trial_order"].update(
                        {
                            "source_column": str(trial_col),
                            "missing_fraction": missing_fraction,
                            "n_unique_non_nan": n_unique,
                            "is_monotonic_increasing_non_nan": is_monotonic,
                            "max_missing_fraction_threshold": max_missing,
                        }
                    )

                    if trial_index_series.isna().all() or missing_fraction > max_missing or not is_monotonic:
                        self.control_trial_order = False
                        self.data_qc["trial_order"].update(
                            {
                                "enabled": False,
                                "reason": "unreliable_trial_order_column",
                            }
                        )
                    else:
                        if self.covariates_df is None or self.covariates_df.empty:
                            self.covariates_df = pd.DataFrame(index=self.aligned_events.index)
                        self.covariates_df["trial_index"] = trial_index_series
                        self.covariates_df = _sanitize_covariates(self.covariates_df)

            if (
                not self.control_temperature
                and self.temperature_column
                and self.covariates_df is not None
                and not self.covariates_df.empty
                and self.temperature_column in self.covariates_df.columns
            ):
                self.covariates_df = self.covariates_df.drop(
                    columns=[self.temperature_column],
                    errors="ignore",
                )
                self.covariates_df = _sanitize_covariates(self.covariates_df)

            self.covariates_without_temp_df = build_covariates_without_temp(
                self.covariates_df, self.temperature_column
            )

            if self.covariates_without_temp_df is not None and self.aligned_events is not None:
                if not self.covariates_without_temp_df.index.equals(self.aligned_events.index):
                    self.covariates_without_temp_df = self.covariates_without_temp_df.copy()
                    self.covariates_without_temp_df.index = self.aligned_events.index
            self.covariates_without_temp_df = _sanitize_covariates(self.covariates_without_temp_df)

            self.group_ids = None
            self.group_column = None
            if self.aligned_events is not None:
                for candidate in ["run", "block", "session", "subject"]:
                    if candidate not in self.aligned_events.columns:
                        continue
                    try:
                        values = self.aligned_events[candidate]
                        if hasattr(values, "nunique") and int(values.nunique(dropna=False)) <= 1:
                            continue
                        self.group_ids = pd.Series(values, index=self.aligned_events.index, name=candidate)
                        self.group_column = candidate
                        break
                    except Exception:
                        self.group_ids = None
                        self.group_column = None

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
