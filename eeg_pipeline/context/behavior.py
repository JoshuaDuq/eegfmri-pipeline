"""
Behavior Analysis Context
=========================

Context objects for behavior analysis data loading and state management.

Note: For new code, consider using BehaviorPipeline from pipelines/behavior.py
which provides a higher-level interface with the same functionality.
"""

from __future__ import annotations

import json

from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from eeg_pipeline.infra.paths import deriv_features_path
from eeg_pipeline.utils.analysis.stats.correlation import (
    CorrelationRecord,
    compute_correlation,
)
from eeg_pipeline.utils.data.columns import pick_target_column


###################################################################
# Constants
###################################################################


_FEATURE_FILE_TO_ATTR = {
    "power": "power_df",
    "connectivity": "connectivity_df",
    "directedconnectivity": "directed_connectivity_df",
    "sourcelocalization": "source_localization_df",
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

_TARGETS_FILENAME = "target_vas_ratings.parquet"
_TRIAL_ALIGNMENT_MANIFEST = "trial_alignment.json"
_MIN_VALID_SAMPLES_FOR_CORRELATION = 3
_DEFAULT_TRIAL_ORDER_MAX_MISSING_FRACTION = 0.1


###################################################################
# Enums and Data Classes
###################################################################


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
        if not self.records:
            return pd.DataFrame()
        return pd.DataFrame([record.to_dict() for record in self.records])


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
    selected_bands: Optional[List[str]] = None  # Specific bands to include (e.g., ["alpha", "beta"])
    computation_features: Optional[Dict[str, List[str]]] = None  # Per-computation feature category filters
    also_save_csv: bool = False  # Also save output tables as CSV files
    
    epochs: Any = None
    epochs_info: Any = None
    aligned_events: Optional[pd.DataFrame] = None
    power_df: Optional[pd.DataFrame] = None
    connectivity_df: Optional[pd.DataFrame] = None
    directed_connectivity_df: Optional[pd.DataFrame] = None
    source_localization_df: Optional[pd.DataFrame] = None
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
    _trial_table_df: Optional[pd.DataFrame] = None
    feature_manifests: Dict[str, Any] = field(default_factory=dict)
    feature_paths: Dict[str, Path] = field(default_factory=dict)

    @property
    def method(self) -> str:
        return "spearman" if self.use_spearman else "pearson"

    @property
    def n_trials(self) -> int:
        if self.targets is not None:
            return len(self.targets)
        if self.power_df is not None:
            return len(self.power_df)
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
        except (FileNotFoundError, ValueError) as e:
            self.logger.error(f"Data loading failed: {e}")
            return False
        except Exception as e:
            self.logger.error(f"Unexpected error during data loading: {e}")
            raise

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
        if self.selected_feature_files:
            self._load_selected_feature_files()
        else:
            self._load_all_features_from_bundle()

        feature_counts = self._get_feature_counts()
        if all(count == 0 for count in feature_counts.values()):
            self.logger.error(
                "Feature bundle is empty for sub-%s; aborting analysis",
                self.subject,
            )
            return False

        coverage_info = ", ".join(
            f"{k}={v}" for k, v in feature_counts.items() if v > 0
        )
        self.logger.info("Feature coverage loaded: %s", coverage_info)

        self._apply_category_filter(feature_counts)

        if self.targets is None or len(self.targets) == 0:
            self.logger.warning("No targets found")
            return False
        return True

    def _load_selected_feature_files(self) -> None:
        """Load selected feature files individually."""
        from eeg_pipeline.utils.data.feature_discovery import STANDARD_FEATURE_FILES

        self.logger.info(
            "Loading selected feature files: %s",
            ", ".join(self.selected_feature_files),
        )
        features_dir = deriv_features_path(self.deriv_root, self.subject)

        for key in self.selected_feature_files:
            if key == "all":
                self.logger.warning(
                    "Feature file key 'all' is not supported in behavior analysis; "
                    "skipping."
                )
                continue
            if key not in STANDARD_FEATURE_FILES:
                self.logger.warning("Unknown feature file key: %s", key)
                continue

            self._load_single_feature_file(key, features_dir, STANDARD_FEATURE_FILES)

        self._load_targets_from_file(features_dir)
        if self.targets is None and self.aligned_events is not None:
            self._load_targets_from_events()

    def _load_single_feature_file(
        self, key: str, features_dir: Path, standard_files: Dict[str, str]
    ) -> None:
        """Load a single feature file and assign to context attribute."""
        from eeg_pipeline.infra.tsv import read_table
        from eeg_pipeline.utils.data.feature_discovery import _find_feature_file_path

        filename = standard_files[key]
        path = _find_feature_file_path(features_dir, key, filename)

        if not path.exists():
            self.logger.warning("Feature file not found: %s", path)
            return

        attr_name = _FEATURE_FILE_TO_ATTR.get(key)
        if not attr_name:
            self.logger.warning(
                "Feature file key '%s' has no mapped context attribute; skipping.",
                key,
            )
            return

        try:
            df = read_table(path)
            self.feature_paths[key] = path
            try:
                meta_path = path.parent / "metadata" / f"{path.stem}.json"
                if meta_path.exists():
                    self.feature_manifests[key] = json.loads(
                        meta_path.read_text(encoding="utf-8")
                    )
            except (OSError, json.JSONDecodeError) as exc:
                self.logger.warning("Failed to load feature metadata for %s: %s", path, exc)
            current_df = getattr(self, attr_name)
            if current_df is None:
                setattr(self, attr_name, df)
            else:
                new_columns = [
                    col for col in df.columns if col not in current_df.columns
                ]
                if new_columns:
                    merged_df = pd.concat([current_df, df[new_columns]], axis=1)
                    setattr(self, attr_name, merged_df)

            self.logger.info(
                "Loaded %s: %d columns, %d rows", key, df.shape[1], df.shape[0]
            )
        except (OSError, pd.errors.EmptyDataError, ValueError) as e:
            self.logger.warning("Failed to load %s: %s", key, e)

    def _load_targets_from_file(self, features_dir: Path) -> None:
        """Load targets from behavior/target_vas_ratings.parquet file."""
        from eeg_pipeline.infra.tsv import read_table
        
        targets_path = features_dir / "behavior" / _TARGETS_FILENAME
        if not targets_path.exists():
            return

        targets_df = read_table(targets_path)
        if targets_df.shape[1] == 1:
            self.targets = pd.to_numeric(
                targets_df.iloc[:, 0], errors="coerce"
            )
            return

        rating_columns = (
            self.config.get("event_columns.rating", []) if self.config else []
        )
        target_col = pick_target_column(targets_df, target_columns=rating_columns)
        if target_col:
            self.targets = pd.to_numeric(targets_df[target_col], errors="coerce")
            return

        numeric_columns = targets_df.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            self.targets = pd.to_numeric(
                targets_df[numeric_columns[0]], errors="coerce"
            )

    def _load_targets_from_events(self) -> None:
        """Load targets from aligned_events DataFrame as fallback."""
        if self.aligned_events is None:
            return

        rating_columns = (
            list(self.config.get("event_columns.rating", []) or [])
            if self.config is not None
            else []
        )
        target_col = pick_target_column(
            self.aligned_events, target_columns=rating_columns
        )

        if target_col:
            self.targets = pd.to_numeric(
                self.aligned_events[target_col], errors="coerce"
            )
            self.logger.info(
                "Targets loaded from events column '%s' (%s missing)",
                target_col,
                _TARGETS_FILENAME,
            )
            return

        numeric_columns = self.aligned_events.select_dtypes(
            include=[np.number]
        ).columns
        if len(numeric_columns) > 0:
            fallback_col = numeric_columns[0]
            self.targets = pd.to_numeric(
                self.aligned_events[fallback_col], errors="coerce"
            )
            self.logger.info(
                "Targets loaded from events numeric column '%s' (%s missing)",
                fallback_col,
                _TARGETS_FILENAME,
            )

    def _load_all_features_from_bundle(self) -> None:
        """Load all features using feature bundle."""
        from eeg_pipeline.utils.data.feature_io import load_feature_bundle

        bundle = load_feature_bundle(
            self.subject,
            self.deriv_root,
            self.logger,
            include_targets=True,
            config=self.config,
        )

        self.feature_manifests = dict(getattr(bundle, "manifests", {}) or {})
        self.feature_paths = dict(getattr(bundle, "paths", {}) or {})

        self.power_df = bundle.power_df
        self.connectivity_df = bundle.connectivity_df
        self.directed_connectivity_df = bundle.directed_connectivity_df
        self.source_localization_df = bundle.source_localization_df
        # Prefer trial-level PAC if available, otherwise fall back to any PAC table.
        # Keep the canonical key name "pac" for downstream prefixing.
        if bundle.pac_trials_df is not None and not bundle.pac_trials_df.empty:
            self.pac_df = bundle.pac_trials_df
            if "pac_trials" in self.feature_manifests and "pac" not in self.feature_manifests:
                self.feature_manifests["pac"] = self.feature_manifests["pac_trials"]
            if "pac_trials" in self.feature_paths and "pac" not in self.feature_paths:
                self.feature_paths["pac"] = self.feature_paths["pac_trials"]
        else:
            self.pac_df = bundle.pac_df
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

        if (self.targets is None or len(self.targets) == 0) and (
            self.aligned_events is not None
        ):
            self._load_targets_from_events()

    def iter_feature_tables(self) -> List[Tuple[str, Optional[pd.DataFrame]]]:
        """Iterate over all feature tables as (name, dataframe) pairs.
        
        Centralizes the canonical ordering of feature types to avoid duplication.
        Uses FEATURE_CATEGORIES from domain.features.constants for consistency.
        """
        from eeg_pipeline.domain.features.constants import FEATURE_CATEGORIES

        feature_dataframes = {
            "power": self.power_df,
            "connectivity": self.connectivity_df,
            "directedconnectivity": self.directed_connectivity_df,
            "sourcelocalization": self.source_localization_df,
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
        feature_types = list(FEATURE_CATEGORIES) + ["temporal"]
        return [
            (feature_type, feature_dataframes.get(feature_type))
            for feature_type in feature_types
        ]

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

        category_to_attr = {
            "power": "power_df",
            "connectivity": "connectivity_df",
            "directedconnectivity": "directed_connectivity_df",
            "sourcelocalization": "source_localization_df",
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
        keep_attributes = {
            category_to_attr.get(category)
            for category in self.feature_categories
            if category_to_attr.get(category) is not None
        }

        if not keep_attributes:
            return

        all_feature_attributes = list(_FEATURE_FILE_TO_ATTR.values())
        for attr_name in all_feature_attributes:
            if attr_name not in keep_attributes:
                setattr(self, attr_name, None)

        categories_str = ", ".join(self.feature_categories)
        self.logger.info("Filtered to categories: %s", categories_str)

    def _validate_alignment(self) -> bool:
        """Validate trial alignment between features, events, and targets."""
        if not self._check_events_targets_length_match():
            return False

        manifest = self._load_alignment_manifest()
        if manifest is None:
            return False

        if not self._validate_manifest_length(manifest):
            return False

        self._record_alignment_manifest(manifest)
        self._validate_rating_consistency()

        return self._align_feature_tables()

    def _check_events_targets_length_match(self) -> bool:
        """Check if aligned_events and targets have matching lengths."""
        n_events = 0 if self.aligned_events is None else len(self.aligned_events)
        n_targets = len(self.targets) if self.targets is not None else 0

        if self.aligned_events is None or n_events != n_targets:
            self.logger.error(
                "Trial alignment mismatch for sub-%s: aligned_events=%d, "
                "targets=%d",
                self.subject,
                n_events,
                n_targets,
            )
            return False
        return True

    def _load_alignment_manifest(self) -> Optional[Dict[str, Any]]:
        """Load trial alignment manifest from disk."""
        features_dir = deriv_features_path(self.deriv_root, self.subject)
        manifest_path = features_dir / "metadata" / _TRIAL_ALIGNMENT_MANIFEST

        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Trial alignment manifest missing: {manifest_path}. "
                f"Re-run feature extraction to generate aligned trial manifests."
            )

        with open(manifest_path, "r") as f:
            return json.load(f)

    def _validate_manifest_length(self, manifest: Dict[str, Any]) -> bool:
        """Validate manifest length matches targets and events."""
        manifest_n_epochs = manifest.get("n_epochs", 0)
        n_targets = len(self.targets) if self.targets is not None else 0
        n_events = len(self.aligned_events) if self.aligned_events is not None else 0

        if manifest_n_epochs != n_targets or manifest_n_epochs != n_events:
            raise ValueError(
                f"Trial alignment manifest length mismatch for sub-{self.subject}: "
                f"manifest={manifest_n_epochs}, targets={n_targets}, "
                f"aligned_events={n_events}"
            )
        return True

    def _record_alignment_manifest(self, manifest: Dict[str, Any]) -> None:
        """Record alignment manifest information in data_qc."""
        features_dir = deriv_features_path(self.deriv_root, self.subject)
        manifest_path = features_dir / _TRIAL_ALIGNMENT_MANIFEST

        self.data_qc["trial_alignment_manifest"] = {
            "path": str(manifest_path),
            "n_trials": manifest.get("n_epochs", 0),
            "has_target_value": "events_trial_type" in manifest,
        }

    def _validate_rating_consistency(self) -> None:
        """Validate consistency between events rating and targets."""
        rating_col = self._find_rating_column()
        if rating_col is None:
            return

        events_rating = pd.to_numeric(
            self.aligned_events[rating_col], errors="coerce"
        )
        targets_rating = pd.to_numeric(self.targets, errors="coerce")
        valid_mask = events_rating.notna() & targets_rating.notna()
        n_valid = int(valid_mask.sum())

        if n_valid < _MIN_VALID_SAMPLES_FOR_CORRELATION:
            return

        correlation_r, correlation_p = compute_correlation(
            events_rating[valid_mask].values,
            targets_rating[valid_mask].values,
            method="spearman",
        )
        differences = (
            events_rating[valid_mask].to_numpy(dtype=float)
            - targets_rating[valid_mask].to_numpy(dtype=float)
        )

        self.data_qc["events_targets_rating_consistency"] = {
            "events_rating_column": str(rating_col),
            "n_valid": n_valid,
            "spearman_r": float(correlation_r) if np.isfinite(correlation_r) else np.nan,
            "spearman_p": float(correlation_p) if np.isfinite(correlation_p) else np.nan,
            "mean_abs_error": float(np.mean(np.abs(differences))),
            "max_abs_error": float(np.max(np.abs(differences))),
        }

    def _find_rating_column(self) -> Optional[str]:
        """Find rating column in aligned_events."""
        if self.aligned_events is None:
            return None

        rating_columns = (
            list(self.config.get("event_columns.rating", []) or [])
            if self.config is not None
            else []
        )
        try:
            return pick_target_column(
                self.aligned_events, target_columns=rating_columns
            )
        except (KeyError, AttributeError):
            return None

    def _align_feature_tables(self) -> bool:
        """Align all feature tables to target index using iter_feature_tables."""
        base_index = self.targets.index
        if not base_index.is_unique:
            self.logger.error(
                "Target index contains duplicates for sub-%s", self.subject
            )
            return False

        alignment_report: Dict[str, Any] = {}

        for feature_type, dataframe in self.iter_feature_tables():
            attr_name = _FEATURE_FILE_TO_ATTR.get(feature_type)
            if attr_name is None:
                continue

            aligned_df = self._align_single_feature_table(
                feature_type, dataframe, base_index, alignment_report
            )
            setattr(self, attr_name, aligned_df)

        if all(
            df is None or df.empty for _, df in self.iter_feature_tables()
        ):
            self.logger.error(
                "All feature tables failed alignment for sub-%s", self.subject
            )
            return False

        self.data_qc["alignment_checks"] = alignment_report
        return True

    def _align_single_feature_table(
        self,
        name: str,
        df: Optional[pd.DataFrame],
        base_index: pd.Index,
        alignment_report: Dict[str, Any],
    ) -> Optional[pd.DataFrame]:
        """Align a single feature table to base_index."""
        if df is None or df.empty:
            return df

        if not df.index.is_unique:
            self.logger.error(
                "Feature table index contains duplicates for %s (sub-%s)",
                name,
                self.subject,
            )
            alignment_report[name] = {
                "status": "failed",
                "reason": "duplicate_index",
            }
            return None

        if len(df) != len(base_index):
            self.logger.error(
                "Feature table length mismatch for %s: %d rows vs %d targets "
                "for sub-%s",
                name,
                len(df),
                len(base_index),
                self.subject,
            )
            alignment_report[name] = {
                "status": "failed",
                "reason": "length_mismatch",
                "n_rows": int(len(df)),
                "n_targets": int(len(base_index)),
            }
            return None

        if df.index.equals(base_index):
            alignment_report[name] = {"status": "aligned"}
            return df

        return self._reindex_feature_table(name, df, base_index, alignment_report)

    def _reindex_feature_table(
        self,
        name: str,
        df: pd.DataFrame,
        base_index: pd.Index,
        alignment_report: Dict[str, Any],
    ) -> Optional[pd.DataFrame]:
        """Reindex feature table to match base_index if possible."""
        try:
            sorted_df_index = df.index.sort_values()
            sorted_base_index = base_index.sort_values()

            if sorted_df_index.equals(sorted_base_index):
                reindexed_df = df.loc[base_index]
                alignment_report[name] = {"status": "reindexed_by_label"}
                return reindexed_df

            self.logger.error(
                "Feature table index mismatch for %s (sub-%s): cannot align "
                "indices",
                name,
                self.subject,
            )
            alignment_report[name] = {
                "status": "failed",
                "reason": "index_mismatch",
            }
            return None
        except (KeyError, ValueError) as exc:
            self.logger.error(
                "Feature table index compare failed for %s (sub-%s): %s",
                name,
                self.subject,
                exc,
            )
            alignment_report[name] = {
                "status": "failed",
                "reason": "index_compare_failed",
            }
            return None

    def _build_covariates(self) -> None:
        """Build covariate matrices for partial correlations."""
        from eeg_pipeline.utils.data.covariates import (
            build_covariate_matrix,
            build_covariates_without_temp,
            extract_temperature_data,
        )

        self._extract_temperature()
        raw_covariates = self._build_raw_covariate_matrix()
        cov_report = self._summarize_covariates(raw_covariates)
        self.covariates_df = self._sanitize_covariates(raw_covariates)

        drop_reasons = self._identify_dropped_columns(raw_covariates)
        if drop_reasons:
            cov_report["drop_reasons"] = drop_reasons

        self._setup_trial_order_covariate()
        cov_report["trial_order_added"] = self._has_trial_index_column()

        self._remove_temperature_if_disabled(cov_report)
        self._build_covariates_without_temperature()

        self._finalize_covariate_report(cov_report)

    def _extract_temperature(self) -> None:
        """Extract temperature data from aligned events."""
        from eeg_pipeline.utils.data.covariates import extract_temperature_data

        self.temperature, self.temperature_column = extract_temperature_data(
            self.aligned_events, self.config
        )

    def _build_raw_covariate_matrix(self) -> Optional[pd.DataFrame]:
        """Build raw covariate matrix from aligned events."""
        from eeg_pipeline.utils.data.covariates import build_covariate_matrix

        return build_covariate_matrix(
            self.aligned_events, self.partial_covars, self.config
        )

    def _identify_dropped_columns(
        self, cov_raw: Optional[pd.DataFrame]
    ) -> Dict[str, str]:
        """Identify columns that were dropped and why."""
        drop_reasons = {}
        if cov_raw is None or cov_raw.empty:
            return drop_reasons

        for col in cov_raw.columns:
            series = pd.to_numeric(cov_raw[col], errors="coerce")
            if series.isna().all():
                drop_reasons[str(col)] = "all_nan"
            elif int(series.nunique(dropna=True)) <= 1:
                drop_reasons[str(col)] = "constant"
        return drop_reasons

    def _has_trial_index_column(self) -> bool:
        """Check if covariates_df contains trial_index column."""
        return (
            self.covariates_df is not None
            and not self.covariates_df.empty
            and "trial_index" in self.covariates_df.columns
        )

    def _remove_temperature_if_disabled(self, cov_report: Dict[str, Any]) -> None:
        """Remove temperature column if control_temperature is disabled."""
        should_remove = (
            not self.control_temperature
            and self.temperature_column
            and self.covariates_df is not None
            and not self.covariates_df.empty
            and self.temperature_column in self.covariates_df.columns
        )

        if not should_remove:
            return

        self.covariates_df = self.covariates_df.drop(
            columns=[self.temperature_column], errors="ignore"
        )
        self.covariates_df = self._sanitize_covariates(self.covariates_df)
        cov_report.setdefault("dropped_by_rule", []).append(
            self.temperature_column
        )

    def _build_covariates_without_temperature(self) -> None:
        """Build covariates DataFrame without temperature column."""
        from eeg_pipeline.utils.data.covariates import build_covariates_without_temp

        self.covariates_without_temp_df = build_covariates_without_temp(
            self.covariates_df, self.temperature_column
        )

        if (
            self.covariates_without_temp_df is not None
            and self.aligned_events is not None
        ):
            if not self.covariates_without_temp_df.index.equals(
                self.aligned_events.index
            ):
                self.covariates_without_temp_df = (
                    self.covariates_without_temp_df.copy()
                )
                self.covariates_without_temp_df.index = self.aligned_events.index

        self.covariates_without_temp_df = self._sanitize_covariates(
            self.covariates_without_temp_df
        )

    def _finalize_covariate_report(self, cov_report: Dict[str, Any]) -> None:
        """Finalize covariate QC report with final and dropped columns."""
        final_columns = (
            []
            if self.covariates_df is None
            else [str(c) for c in self.covariates_df.columns]
        )
        raw_columns = cov_report.get("raw_columns", [])
        dropped_columns = [
            c for c in raw_columns if c not in final_columns
        ]

        cov_report["final_columns"] = final_columns
        cov_report["dropped_columns"] = dropped_columns
        self.data_qc["covariates_qc"] = cov_report

    def _sanitize_covariates(
        self, cov: Optional[pd.DataFrame]
    ) -> Optional[pd.DataFrame]:
        """Clean covariate DataFrame: convert to numeric, remove constants and NaN columns."""
        if cov is None or cov.empty:
            return None

        sanitized = cov.copy()
        self._align_covariate_index(sanitized)
        self._convert_to_numeric(sanitized)
        self._remove_infinite_values(sanitized)
        self._remove_all_nan_columns(sanitized)

        if sanitized.empty:
            return None

        self._remove_constant_columns(sanitized)
        return None if sanitized.empty else sanitized

    def _align_covariate_index(self, cov: pd.DataFrame) -> None:
        """Align covariate DataFrame index with aligned_events."""
        if (
            self.aligned_events is not None
            and not cov.index.equals(self.aligned_events.index)
        ):
            cov.index = self.aligned_events.index

    def _convert_to_numeric(self, cov: pd.DataFrame) -> None:
        """Convert all columns to numeric."""
        for col in list(cov.columns):
            cov[col] = pd.to_numeric(cov[col], errors="coerce")

    def _remove_infinite_values(self, cov: pd.DataFrame) -> None:
        """Replace infinite values with NaN."""
        cov.replace([np.inf, -np.inf], np.nan, inplace=True)

    def _remove_all_nan_columns(self, cov: pd.DataFrame) -> None:
        """Remove columns that are entirely NaN."""
        cov.dropna(axis=1, how="all", inplace=True)

    def _remove_constant_columns(self, cov: pd.DataFrame) -> None:
        """Remove columns with constant values."""
        constant_columns = [
            col
            for col in cov.columns
            if int(
                pd.to_numeric(cov[col], errors="coerce").nunique(dropna=True)
            )
            <= 1
        ]
        if constant_columns:
            cov.drop(columns=constant_columns, errors="ignore", inplace=True)

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

        trial_column = self._find_trial_column()
        if trial_column is None:
            self.control_trial_order = False
            self.data_qc["trial_order"].update(
                {"enabled": False, "reason": "missing_trial_column"}
            )
            return

        trial_series = pd.to_numeric(
            self.aligned_events[trial_column], errors="coerce"
        )
        max_missing_fraction = self._get_max_missing_fraction_threshold()
        validation_result = self._validate_trial_order_column(
            trial_series, max_missing_fraction
        )

        self._record_trial_order_metadata(
            trial_column, trial_series, max_missing_fraction, validation_result
        )

        if not validation_result["is_valid"]:
            self.control_trial_order = False
            self.data_qc["trial_order"].update(
                {"enabled": False, "reason": "unreliable_trial_order_column"}
            )
            return

        self._add_trial_index_to_covariates(trial_series)

    def _find_trial_column(self) -> Optional[str]:
        """Find trial order column in aligned_events."""
        trial_candidates = [
            "trial_index",
            "trial_in_run",
            "trial",
            "trial_number",
        ]
        for candidate in trial_candidates:
            if candidate in self.aligned_events.columns:
                return candidate
        return None

    def _get_max_missing_fraction_threshold(self) -> float:
        """Get maximum missing fraction threshold from config."""
        if self.config is None:
            return _DEFAULT_TRIAL_ORDER_MAX_MISSING_FRACTION

        return float(
            self.config.get(
                "behavior_analysis.trial_order.max_missing_fraction",
                _DEFAULT_TRIAL_ORDER_MAX_MISSING_FRACTION,
            )
        )

    def _validate_trial_order_column(
        self, trial_series: pd.Series, max_missing: float
    ) -> Dict[str, Any]:
        """Validate trial order column quality."""
        n_samples = len(trial_series)
        missing_fraction = (
            float(trial_series.isna().mean()) if n_samples > 0 else 1.0
        )
        n_unique = int(trial_series.dropna().nunique())
        is_monotonic = bool(trial_series.dropna().is_monotonic_increasing)

        is_all_missing = trial_series.isna().all()
        exceeds_missing_threshold = missing_fraction > max_missing
        is_not_monotonic = not is_monotonic

        is_valid = not (
            is_all_missing or exceeds_missing_threshold or is_not_monotonic
        )

        return {
            "is_valid": is_valid,
            "missing_fraction": missing_fraction,
            "n_unique": n_unique,
            "is_monotonic": is_monotonic,
        }

    def _record_trial_order_metadata(
        self,
        trial_column: str,
        trial_series: pd.Series,
        max_missing: float,
        validation: Dict[str, Any],
    ) -> None:
        """Record trial order metadata in data_qc."""
        self.data_qc["trial_order"].update(
            {
                "source_column": str(trial_column),
                "missing_fraction": validation["missing_fraction"],
                "n_unique_non_nan": validation["n_unique"],
                "is_monotonic_increasing_non_nan": validation["is_monotonic"],
                "max_missing_fraction_threshold": max_missing,
            }
        )

    def _add_trial_index_to_covariates(self, trial_series: pd.Series) -> None:
        """Add trial_index column to covariates DataFrame."""
        if self.covariates_df is None or self.covariates_df.empty:
            self.covariates_df = pd.DataFrame(index=self.aligned_events.index)
        self.covariates_df["trial_index"] = trial_series
        self.covariates_df = self._sanitize_covariates(self.covariates_df)

    def _extract_group_ids(self) -> None:
        """Extract group IDs for permutation testing.
        
        Uses configurable preference order for group columns.
        Default order is block→run→session (more conservative for within-subject tests),
        preferring smaller units for permutation to be more conservative about
        temporal autocorrelation.
        """
        from eeg_pipeline.utils.config.loader import get_config_value

        self.group_ids = None
        self.group_column = None
        if self.aligned_events is None:
            return

        run_column = self._get_configured_run_column()
        preference_order = self._build_group_column_preference_order(run_column)

        for candidate in preference_order:
            resolved_column = self._resolve_column_name(candidate, run_column)
            if resolved_column is None:
                continue

            group_series = self._try_extract_group_series(resolved_column)
            if group_series is not None:
                self.group_ids = group_series
                self.group_column = resolved_column
                break

    def _get_configured_run_column(self) -> str:
        """Get configured run column name."""
        from eeg_pipeline.utils.config.loader import get_config_value

        run_col = str(
            get_config_value(
                self.config,
                "behavior_analysis.run_adjustment.column",
                "run_id",
            )
            or "run_id"
        ).strip()
        return run_col if run_col else "run_id"

    def _build_group_column_preference_order(
        self, run_column: str
    ) -> List[str]:
        """Build preference order for group columns."""
        from eeg_pipeline.utils.config.loader import get_config_value

        default_preference = ["block", run_column, "run", "session", "subject"]
        pref_order = get_config_value(
            self.config,
            "behavior_analysis.permutation.group_column_preference",
            default_preference,
        )

        if not isinstance(pref_order, (list, tuple)):
            pref_order = default_preference

        normalized_order = [str(col).strip().lower() for col in pref_order]

        if run_column.lower() not in normalized_order:
            normalized_order.append(run_column.lower())

        return self._deduplicate_preserving_order(normalized_order)

    def _deduplicate_preserving_order(self, items: List[str]) -> List[str]:
        """Remove duplicates while preserving order."""
        seen = set()
        return [item for item in items if item not in seen and not seen.add(item)]

    def _resolve_column_name(
        self, candidate: str, run_column: str
    ) -> Optional[str]:
        """Resolve column name, handling aliases."""
        if candidate in self.aligned_events.columns:
            return candidate

        if candidate == "run" and run_column in self.aligned_events.columns:
            return run_column

        return None

    def _try_extract_group_series(
        self, column_name: str
    ) -> Optional[pd.Series]:
        """Try to extract group series from column, validating it."""
        try:
            values = self.aligned_events[column_name]
            n_unique = int(values.nunique(dropna=False))
            if n_unique <= 1:
                return None

            return pd.Series(
                values, index=self.aligned_events.index, name=column_name
            )
        except (KeyError, AttributeError, ValueError):
            return None


    def add_result(self, result: ComputationResult) -> None:
        """Add a computation result."""
        self.results[result.name] = result
