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
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import mne

from eeg_pipeline.infra.tsv import read_tsv

from eeg_pipeline.utils.analysis.stats.correlation import CorrelationRecord, compute_correlation
from eeg_pipeline.utils.data.columns import pick_target_column
from eeg_pipeline.infra.paths import deriv_features_path


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
    stats_config: Optional[Any] = None
    feature_categories: Optional[List[str]] = None
    selected_feature_files: Optional[List[str]] = None  # Specific files to load (e.g., ["power", "aperiodic"])
    
    epochs: Any = None
    epochs_info: Any = None
    aligned_events: Optional[pd.DataFrame] = None
    power_df: Optional[pd.DataFrame] = None
    connectivity_df: Optional[pd.DataFrame] = None
    aperiodic_df: Optional[pd.DataFrame] = None
    erp_df: Optional[pd.DataFrame] = None
    complexity_df: Optional[pd.DataFrame] = None
    itpc_df: Optional[pd.DataFrame] = None
    pac_df: Optional[pd.DataFrame] = None
    bursts_df: Optional[pd.DataFrame] = None
    quality_df: Optional[pd.DataFrame] = None
    erds_df: Optional[pd.DataFrame] = None
    spectral_df: Optional[pd.DataFrame] = None
    ratios_df: Optional[pd.DataFrame] = None
    asymmetry_df: Optional[pd.DataFrame] = None
    temporal_df: Optional[pd.DataFrame] = None
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
    _combined_features_df: Optional[pd.DataFrame] = None
    _combined_features_signature: Optional[str] = None

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
        """Get minimum samples threshold from config.
        
        Delegates to centralized get_min_samples in utils.config.loader.
        """
        from eeg_pipeline.utils.config.loader import get_min_samples
        return get_min_samples(self.config, sample_type)

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

        self.logger.info("Loading data...")
        try:
            if not self._load_epochs():
                return False
            if not self._load_features():
                return False
            if not self._validate_alignment():
                return False
            self._build_covariates()
            self._extract_group_ids()
            self._data_loaded = True
            return True
        except Exception as e:
            self.logger.error(f"Data loading failed: {e}")
            return False

    def _load_epochs(self) -> bool:
        """Load epochs and aligned events."""
        from eeg_pipeline.utils.data.epochs import load_epochs_for_analysis

        self.epochs, self.aligned_events = load_epochs_for_analysis(
            self.subject, self.task, align="strict", preload=True,
            deriv_root=self.deriv_root, logger=self.logger, config=self.config
        )
        if self.epochs is None or self.aligned_events is None:
            self.logger.error("Failed to load epochs or events")
            return False
        self.epochs_info = self.epochs.info
        return True

    def _load_features(self) -> bool:
        """Load feature bundle and apply category/file filtering."""
        from eeg_pipeline.utils.data.feature_io import load_feature_bundle
        from eeg_pipeline.utils.data.feature_discovery import STANDARD_FEATURE_FILES
        from eeg_pipeline.infra.paths import deriv_features_path
        from eeg_pipeline.infra.tsv import read_tsv
        
        # If specific files are selected, load them selectively
        if self.selected_feature_files:
            self.logger.info(f"Loading selected feature files: {', '.join(self.selected_feature_files)}")
            features_dir = deriv_features_path(self.deriv_root, self.subject)
            
            # Map of feature file key to context attribute
            file_attr_map = {
                "power": "power_df",
                "connectivity": "connectivity_df",
                "aperiodic": "aperiodic_df",
                "erp": "erp_df",
                "itpc": "itpc_df",
                "pac": "pac_df",
                "complexity": "complexity_df",
                "bursts": "bursts_df",
                "quality": "quality_df",
                "erds": "erds_df",
                "spectral": "spectral_df",
                "ratios": "ratios_df",
                "asymmetry": "asymmetry_df",
                "temporal": "temporal_df",
            }
            
            for key in self.selected_feature_files:
                if key not in STANDARD_FEATURE_FILES:
                    self.logger.warning(f"Unknown feature file key: {key}")
                    continue

                if key == "all":
                    self.logger.warning("Feature file key 'all' is not supported in behavior analysis; skipping.")
                    continue
                
                filename = STANDARD_FEATURE_FILES[key]
                path = features_dir / filename
                
                if not path.exists():
                    self.logger.warning(f"Feature file not found: {path}")
                    continue
                
                attr_name = file_attr_map.get(key)
                if not attr_name:
                    self.logger.warning(f"Feature file key '{key}' has no mapped context attribute; skipping.")
                    continue

                try:
                    if path.suffix == ".parquet":
                        df = pd.read_parquet(path)
                    else:
                        df = read_tsv(path)
                    
                    current = getattr(self, attr_name)
                    if current is None:
                        setattr(self, attr_name, df)
                    else:
                        # Merge for precomputed-style attributes
                        new_cols = [c for c in df.columns if c not in current.columns]
                        if new_cols:
                            setattr(self, attr_name, pd.concat([current, df[new_cols]], axis=1))
                    
                    self.logger.info(f"Loaded {key}: {df.shape[1]} columns, {df.shape[0]} rows")
                except Exception as e:
                    self.logger.warning(f"Failed to load {key}: {e}")
            
            # Load targets separately
            targets_path = features_dir / "target_vas_ratings.tsv"
            if targets_path.exists():
                targets_df = read_tsv(targets_path)
                if targets_df.shape[1] == 1:
                    self.targets = pd.to_numeric(targets_df.iloc[:, 0], errors="coerce")
                else:
                    from eeg_pipeline.utils.data.columns import pick_target_column
                    target_col = pick_target_column(targets_df, target_columns=self.config.get("event_columns.rating", []) if self.config else [])
                    if target_col:
                        self.targets = pd.to_numeric(targets_df[target_col], errors="coerce")
                    else:
                        numeric_cols = targets_df.select_dtypes(include=[np.number]).columns
                        if len(numeric_cols) > 0:
                            self.targets = pd.to_numeric(targets_df[numeric_cols[0]], errors="coerce")
            elif self.aligned_events is not None:
                target_columns = list(self.config.get("event_columns.rating", []) or []) if self.config is not None else []
                target_col = pick_target_column(self.aligned_events, target_columns=target_columns)
                if target_col:
                    self.targets = pd.to_numeric(self.aligned_events[target_col], errors="coerce")
                    self.logger.info(
                        "Targets loaded from events column '%s' (target_vas_ratings.tsv missing)",
                        target_col,
                    )
                else:
                    numeric_cols = self.aligned_events.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        fallback_col = numeric_cols[0]
                        self.targets = pd.to_numeric(self.aligned_events[fallback_col], errors="coerce")
                        self.logger.info(
                            "Targets loaded from events numeric column '%s' (target_vas_ratings.tsv missing)",
                            fallback_col,
                        )
        else:
            # Default: load all via bundle (existing behavior)
            bundle = load_feature_bundle(
                self.subject, self.deriv_root, self.logger,
                include_targets=True, config=self.config,
            )

            self.power_df = bundle.power_df
            self.connectivity_df = bundle.connectivity_df
            self.pac_df = bundle.pac_trials_df
            self.aperiodic_df = bundle.aperiodic_df
            self.erp_df = bundle.erp_df
            self.itpc_df = bundle.itpc_df
            self.complexity_df = bundle.complexity_df
            self.bursts_df = bundle.bursts_df
            self.quality_df = bundle.quality_df
            self.erds_df = bundle.erds_df
            self.spectral_df = bundle.spectral_df
            self.ratios_df = bundle.ratios_df
            self.asymmetry_df = bundle.asymmetry_df
            self.temporal_df = bundle.temporal_df
            self.targets = bundle.targets

            if (self.targets is None or len(self.targets) == 0) and self.aligned_events is not None:
                target_columns = list(self.config.get("event_columns.rating", []) or []) if self.config is not None else []
                target_col = pick_target_column(self.aligned_events, target_columns=target_columns)
                if target_col:
                    self.targets = pd.to_numeric(self.aligned_events[target_col], errors="coerce")
                    self.logger.info(
                        "Targets loaded from events column '%s' (bundle targets missing)",
                        target_col,
                    )
                else:
                    numeric_cols = self.aligned_events.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        fallback_col = numeric_cols[0]
                        self.targets = pd.to_numeric(self.aligned_events[fallback_col], errors="coerce")
                        self.logger.info(
                            "Targets loaded from events numeric column '%s' (bundle targets missing)",
                            fallback_col,
                        )

        feature_counts = self._get_feature_counts()
        if all(count == 0 for count in feature_counts.values()):
            self.logger.error("Feature bundle is empty for sub-%s; aborting analysis", self.subject)
            return False
        self.logger.info(
            "Feature coverage loaded: %s",
            ", ".join(f"{k}={v}" for k, v in feature_counts.items() if v > 0),
        )

        self._apply_category_filter(feature_counts)

        if self.targets is None or len(self.targets) == 0:
            self.logger.warning("No targets found")
            return False
        return True

    def iter_feature_tables(self) -> List[Tuple[str, Optional[pd.DataFrame]]]:
        """Iterate over all feature tables as (name, dataframe) pairs.
        
        Centralizes the canonical ordering of feature types to avoid duplication.
        Uses FEATURE_TYPES from domain.features for consistency.
        """
        from eeg_pipeline.domain.features import FEATURE_TYPES
        
        attr_map = {
            "power": self.power_df,
            "connectivity": self.connectivity_df,
            "aperiodic": self.aperiodic_df,
            "erp": self.erp_df,
            "itpc": self.itpc_df,
            "pac": self.pac_df,
            "complexity": self.complexity_df,
            "bursts": self.bursts_df,
            "quality": self.quality_df,
            "erds": self.erds_df,
            "spectral": self.spectral_df,
            "ratios": self.ratios_df,
            "asymmetry": self.asymmetry_df,
            "temporal": self.temporal_df,
        }
        return [(ft, attr_map.get(ft)) for ft in FEATURE_TYPES]

    def _get_feature_counts(self) -> Dict[str, int]:
        """Get feature counts for each feature type."""
        return {
            name: 0 if table is None or table.empty else table.shape[1]
            for name, table in self.iter_feature_tables()
        }

    def _apply_category_filter(self, feature_counts: Dict[str, int]) -> None:
        """Apply feature category filtering if specified."""
        if not self.feature_categories:
            return

        category_map = {
            "power": "power_df",
            "connectivity": "connectivity_df",
            "spectral": "spectral_df",
            "aperiodic": "aperiodic_df",
            "erp": "erp_df",
            "erds": "erds_df",
            "ratios": "ratios_df",
            "asymmetry": "asymmetry_df",
            "itpc": "itpc_df",
            "pac": "pac_df",
            "complexity": "complexity_df",
            "bursts": "bursts_df",
            "quality": "quality_df",
            "temporal": "temporal_df",
            "psychometrics": None,
            "dose_response": None,
        }
        keep_attrs = {category_map.get(cat) for cat in self.feature_categories if category_map.get(cat)}

        if keep_attrs:
            all_attrs = [
                "power_df", "connectivity_df", "spectral_df", "aperiodic_df",
                "erp_df", "erds_df", "ratios_df", "asymmetry_df",
                "itpc_df", "pac_df", "complexity_df", "bursts_df",
                "quality_df", "temporal_df",
            ]
            for attr_name in all_attrs:
                if attr_name not in keep_attrs:
                    setattr(self, attr_name, None)
            self.logger.info("Filtered to categories: %s", ", ".join(self.feature_categories))

    def _validate_alignment(self) -> bool:
        """Validate trial alignment between features, events, and targets."""
        import json

        if self.aligned_events is None or len(self.aligned_events) != len(self.targets):
            self.logger.error(
                "Trial alignment mismatch for sub-%s: aligned_events=%s, targets=%s",
                self.subject,
                0 if self.aligned_events is None else len(self.aligned_events),
                len(self.targets),
            )
            return False

        features_dir = deriv_features_path(self.deriv_root, self.subject)
        manifest_path = features_dir / "trial_alignment.json"
        
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Trial alignment manifest missing: {manifest_path}. "
                f"Re-run feature extraction to generate aligned trial manifests."
            )
        actual_path = manifest_path

        with open(actual_path, "r") as f:
            manifest = json.load(f)

        manifest_n_epochs = manifest.get("n_epochs", 0)
        if manifest_n_epochs != len(self.targets) or manifest_n_epochs != len(self.aligned_events):
            raise ValueError(
                "Trial alignment manifest length mismatch for sub-%s: manifest=%d, targets=%d, aligned_events=%d"
                % (self.subject, manifest_n_epochs, len(self.targets), len(self.aligned_events))
            )

        self.data_qc["trial_alignment_manifest"] = {
            "path": str(actual_path),
            "n_trials": manifest_n_epochs,
            "has_target_value": "events_trial_type" in manifest,
        }

        self._validate_rating_consistency()
        if not self._align_feature_tables():
            return False
        return True

    def _validate_rating_consistency(self) -> None:
        """Validate consistency between events rating and targets."""
        rating_col = None
        try:
            target_columns = list(self.config.get("event_columns.rating", []) or []) if self.config is not None else []
            rating_col = pick_target_column(self.aligned_events, target_columns=target_columns)
        except Exception:
            rating_col = None

        if rating_col is None or rating_col not in self.aligned_events.columns:
            return

        events_rating = pd.to_numeric(self.aligned_events[rating_col], errors="coerce")
        targets_rating = pd.to_numeric(self.targets, errors="coerce")
        valid = events_rating.notna() & targets_rating.notna()

        if int(valid.sum()) >= 3:
            r, p = compute_correlation(
                events_rating[valid].values,
                targets_rating[valid].values,
                method="spearman",
            )
            diff = events_rating[valid].to_numpy(dtype=float) - targets_rating[valid].to_numpy(dtype=float)
            self.data_qc["events_targets_rating_consistency"] = {
                "events_rating_column": str(rating_col),
                "n_valid": int(valid.sum()),
                "spearman_r": float(r) if np.isfinite(r) else np.nan,
                "spearman_p": float(p) if np.isfinite(p) else np.nan,
                "mean_abs_error": float(np.mean(np.abs(diff))) if len(diff) else np.nan,
                "max_abs_error": float(np.max(np.abs(diff))) if len(diff) else np.nan,
            }

    def _align_feature_tables(self) -> bool:
        """Align all feature tables to target index using iter_feature_tables."""
        base_index = self.targets.index
        if not base_index.is_unique:
            self.logger.error("Target index contains duplicates for sub-%s", self.subject)
            return False

        alignment_report: Dict[str, Any] = {}

        def align_df(name: str, df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
            if df is None or df.empty:
                return df
            if not df.index.is_unique:
                self.logger.error("Feature table index contains duplicates for %s (sub-%s)", name, self.subject)
                alignment_report[name] = {"status": "failed", "reason": "duplicate_index"}
                return None
            if len(df) != len(base_index):
                self.logger.error(
                    "Feature table length mismatch for %s: %s rows vs %s targets for sub-%s",
                    name, len(df), len(base_index), self.subject,
                )
                alignment_report[name] = {
                    "status": "failed",
                    "reason": "length_mismatch",
                    "n_rows": int(len(df)),
                    "n_targets": int(len(base_index)),
                }
                return None
            if not df.index.equals(base_index):
                try:
                    if df.index.sort_values().equals(base_index.sort_values()):
                        df = df.loc[base_index]
                        alignment_report[name] = {"status": "reindexed_by_label"}
                    else:
                        self.logger.error(
                            "Feature table index mismatch for %s (sub-%s): cannot align indices",
                            name, self.subject,
                        )
                        alignment_report[name] = {"status": "failed", "reason": "index_mismatch"}
                        return None
                except Exception as exc:
                    self.logger.error(
                        "Feature table index compare failed for %s (sub-%s): %s",
                        name, self.subject, exc,
                    )
                    alignment_report[name] = {"status": "failed", "reason": "index_compare_failed"}
                    return None
            else:
                alignment_report[name] = {"status": "aligned"}
            return df

        attr_map = {
            "power": "power_df",
            "connectivity": "connectivity_df",
            "aperiodic": "aperiodic_df",
            "erp": "erp_df",
            "itpc": "itpc_df",
            "pac": "pac_df",
            "complexity": "complexity_df",
            "bursts": "bursts_df",
            "quality": "quality_df",
            "erds": "erds_df",
            "spectral": "spectral_df",
            "ratios": "ratios_df",
            "asymmetry": "asymmetry_df",
            "temporal": "temporal_df",
        }
        for name, attr in attr_map.items():
            setattr(self, attr, align_df(name, getattr(self, attr)))

        if all(df is None or df.empty for _, df in self.iter_feature_tables()):
            self.logger.error("All feature tables failed alignment for sub-%s", self.subject)
            return False
        self.data_qc["alignment_checks"] = alignment_report
        return True

    def _build_covariates(self) -> None:
        """Build covariate matrices for partial correlations."""
        from eeg_pipeline.utils.data.covariates import (
            extract_temperature_data,
            build_covariate_matrix,
            build_covariates_without_temp,
        )

        self.temperature, self.temperature_column = extract_temperature_data(
            self.aligned_events, self.config
        )
        cov_raw = build_covariate_matrix(
            self.aligned_events, self.partial_covars, self.config
        )
        cov_report = self._summarize_covariates(cov_raw)
        self.covariates_df = self._sanitize_covariates(cov_raw)
        drop_reasons = {}
        if cov_raw is not None and not cov_raw.empty:
            for col in cov_raw.columns:
                series = pd.to_numeric(cov_raw[col], errors="coerce")
                if series.isna().all():
                    drop_reasons[str(col)] = "all_nan"
                elif int(series.nunique(dropna=True)) <= 1:
                    drop_reasons[str(col)] = "constant"
        if drop_reasons:
            cov_report["drop_reasons"] = drop_reasons

        self._setup_trial_order_covariate()
        cov_report["trial_order_added"] = bool(
            self.covariates_df is not None and "trial_index" in self.covariates_df.columns
        )

        if (
            not self.control_temperature
            and self.temperature_column
            and self.covariates_df is not None
            and not self.covariates_df.empty
            and self.temperature_column in self.covariates_df.columns
        ):
            self.covariates_df = self.covariates_df.drop(columns=[self.temperature_column], errors="ignore")
            self.covariates_df = self._sanitize_covariates(self.covariates_df)
            cov_report.setdefault("dropped_by_rule", []).append(self.temperature_column)

        self.covariates_without_temp_df = build_covariates_without_temp(
            self.covariates_df, self.temperature_column
        )
        if self.covariates_without_temp_df is not None and self.aligned_events is not None:
            if not self.covariates_without_temp_df.index.equals(self.aligned_events.index):
                self.covariates_without_temp_df = self.covariates_without_temp_df.copy()
                self.covariates_without_temp_df.index = self.aligned_events.index
        self.covariates_without_temp_df = self._sanitize_covariates(self.covariates_without_temp_df)

        cov_report["final_columns"] = [] if self.covariates_df is None else [str(c) for c in self.covariates_df.columns]
        cov_report["dropped_columns"] = [
            c for c in cov_report.get("raw_columns", []) if c not in cov_report.get("final_columns", [])
        ]
        self.data_qc["covariates_qc"] = cov_report

    def _sanitize_covariates(self, cov: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        """Clean covariate DataFrame: convert to numeric, remove constants and NaN columns."""
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
            c for c in cov.columns
            if int(pd.to_numeric(cov[c], errors="coerce").nunique(dropna=True)) <= 1
        ]
        if constant_cols:
            cov = cov.drop(columns=constant_cols, errors="ignore")
        return None if cov.empty else cov

    def _summarize_covariates(self, cov: Optional[pd.DataFrame]) -> Dict[str, Any]:
        if cov is None or cov.empty:
            return {"raw_columns": [], "column_stats": {}}
        summary: Dict[str, Any] = {}
        for col in cov.columns:
            series = pd.to_numeric(cov[col], errors="coerce")
            summary[str(col)] = {
                "missing_fraction": float(series.isna().mean()),
                "n_unique": int(series.nunique(dropna=True)),
            }
        return {
            "raw_columns": [str(c) for c in cov.columns],
            "column_stats": summary,
        }

    def _setup_trial_order_covariate(self) -> None:
        """Setup trial order as a covariate if enabled and valid."""
        self.data_qc["trial_order"] = {"enabled": bool(self.control_trial_order)}
        if not self.control_trial_order or self.aligned_events is None:
            return

        trial_candidates = ["trial_index", "trial_in_run", "trial", "trial_number"]
        trial_col = None
        for cand in trial_candidates:
            if cand in self.aligned_events.columns:
                trial_col = cand
                break

        if trial_col is None:
            self.control_trial_order = False
            self.data_qc["trial_order"].update({"enabled": False, "reason": "missing_trial_column"})
            return

        trial_index_series = pd.to_numeric(self.aligned_events[trial_col], errors="coerce")
        max_missing = float(
            self.config.get("behavior_analysis.trial_order.max_missing_fraction", 0.1)
            if self.config is not None else 0.1
        )
        missing_fraction = float(trial_index_series.isna().mean()) if len(trial_index_series) else 1.0
        n_unique = int(trial_index_series.dropna().nunique())
        is_monotonic = bool(trial_index_series.dropna().is_monotonic_increasing)

        self.data_qc["trial_order"].update({
            "source_column": str(trial_col),
            "missing_fraction": missing_fraction,
            "n_unique_non_nan": n_unique,
            "is_monotonic_increasing_non_nan": is_monotonic,
            "max_missing_fraction_threshold": max_missing,
        })

        if trial_index_series.isna().all() or missing_fraction > max_missing or not is_monotonic:
            self.control_trial_order = False
            self.data_qc["trial_order"].update({"enabled": False, "reason": "unreliable_trial_order_column"})
        else:
            if self.covariates_df is None or self.covariates_df.empty:
                self.covariates_df = pd.DataFrame(index=self.aligned_events.index)
            self.covariates_df["trial_index"] = trial_index_series
            self.covariates_df = self._sanitize_covariates(self.covariates_df)

    def _extract_group_ids(self) -> None:
        """Extract group IDs for permutation testing."""
        self.group_ids = None
        self.group_column = None
        if self.aligned_events is None:
            return

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
