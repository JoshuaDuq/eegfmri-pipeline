"""Behavior pipeline orchestration.

This module contains the implementation of the behavior pipeline stages.
The pipeline layer (`eeg_pipeline.pipelines.behavior`) should remain a thin wrapper.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
import hashlib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from eeg_pipeline.context.behavior import BehaviorContext
from eeg_pipeline.analysis.behavior.config_resolver import resolve_correlation_targets
from eeg_pipeline.utils.analysis.stats.correlation import compute_correlation, format_correlation_method_label
from eeg_pipeline.utils.config.loader import get_config_value, get_config_float, get_config_int, get_config_bool
from eeg_pipeline.infra.paths import ensure_dir


def _also_save_csv_from_config(config: Any) -> bool:
    return bool(get_config_value(config, "behavior_analysis.output.also_save_csv", False))


def _write_parquet_with_optional_csv(
    df: pd.DataFrame,
    path: Path,
    *,
    also_save_csv: bool,
) -> None:
    from eeg_pipeline.infra.tsv import write_parquet, write_csv

    write_parquet(df, path)
    if also_save_csv:
        write_csv(df, path.with_suffix(".csv"), index=False)


###################################################################
# Result Caching Layer
###################################################################


class _ResultCache:
    """In-memory cache for expensive computations to avoid repeated disk I/O and processing."""

    def __init__(self):
        self._trial_table_df: Optional[pd.DataFrame] = None
        self._trial_table_path: Optional[Path] = None
        self._feature_cols: Dict[str, List[str]] = {}
        self._filtered_feature_cols: Dict[Tuple[str, ...], List[str]] = {}
        self._discovered_files: Dict[str, List[Path]] = {}
        self._fdr_results: Optional[Dict[str, Any]] = None
        self._feature_types: Dict[str, str] = {}
        self._feature_bands: Dict[str, str] = {}
        self._manifest: Optional[Dict[str, Any]] = None
        self._manifest_loaded: bool = False
        self._stats_subfolders: Dict[Tuple[str, str, bool], Path] = {}

    def get_stats_subfolder(
        self,
        stats_dir: Path,
        kind: str,
        overwrite: bool,
        *,
        ensure: bool,
    ) -> Path:
        """Return stable stats subfolder for this run (per kind)."""
        key = (str(stats_dir.resolve()), str(kind), bool(overwrite))
        if overwrite:
            path = stats_dir / kind
        else:
            if key in self._stats_subfolders:
                path = self._stats_subfolders[key]
            else:
                from datetime import datetime

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = stats_dir / f"{kind}_{timestamp}"
                self._stats_subfolders[key] = path
        if ensure:
            ensure_dir(path)
        return path

    def get_trial_table(self, ctx: BehaviorContext) -> Optional[pd.DataFrame]:
        """Get cached trial table or load from disk."""
        if self._trial_table_df is not None:
            return self._trial_table_df

        suffix = _feature_suffix_from_context(ctx)
        fname = f"trials{suffix}"
        expected_dir = get_behavior_output_dir(ctx, "trial_table", ensure=False)
        fmt = str(get_config_value(ctx.config, "behavior_analysis.trial_table.format", "tsv")).strip().lower()
        preferred_ext = ".parquet" if fmt == "parquet" else ".tsv"
        trial_table_path = expected_dir / f"{fname}{preferred_ext}"
        if not trial_table_path.exists():
            # Allow reading from the alternative format if present.
            alt_ext = ".tsv" if preferred_ext == ".parquet" else ".parquet"
            alt_path = expected_dir / f"{fname}{alt_ext}"
            if alt_path.exists():
                trial_table_path = alt_path

        if not trial_table_path.exists():
            feature_files = ctx.selected_feature_files or ctx.feature_categories or None
            if feature_files:
                found = _find_trial_table_path(ctx.stats_dir, feature_files=feature_files)
                if found is None:
                    return None
                trial_table_path = found
            else:
                return None

        from eeg_pipeline.infra.tsv import read_table
        self._trial_table_df = read_table(trial_table_path)
        _validate_trial_table_contract_metadata(ctx, trial_table_path, self._trial_table_df)
        self._trial_table_path = trial_table_path
        return self._trial_table_df

    def get_feature_cols(self, df: pd.DataFrame, ctx: BehaviorContext) -> List[str]:
        """Get cached feature columns or compute from DataFrame."""
        cache_key = id(df)
        if cache_key not in self._feature_cols:
            self._feature_cols[cache_key] = [c for c in df.columns if str(c).startswith(FEATURE_COLUMN_PREFIXES)]
        return self._feature_cols[cache_key]

    def get_filtered_feature_cols(
        self,
        feature_cols: List[str],
        ctx: BehaviorContext,
        computation_name: Optional[str] = None,
    ) -> List[str]:
        """Get cached filtered feature columns."""
        bands_key = tuple(sorted(ctx.selected_bands)) if ctx.selected_bands else None
        cache_key = (id(feature_cols), bands_key, computation_name)

        if cache_key not in self._filtered_feature_cols:
            filtered = feature_cols.copy()
            if ctx.selected_bands:
                filtered = _filter_feature_cols_by_band(filtered, ctx)
            if computation_name:
                filtered = _filter_feature_cols_for_computation(filtered, computation_name, ctx)
            filtered = _filter_feature_cols_by_provenance(filtered, ctx, computation_name)
            self._filtered_feature_cols[cache_key] = filtered

        return self._filtered_feature_cols[cache_key]

    def get_discovered_files(self, ctx: BehaviorContext, patterns: List[str]) -> List[Path]:
        """Get cached discovered files matching patterns."""
        cache_key = "_".join(sorted(patterns))
        if cache_key not in self._discovered_files:
            files: List[Path] = []
            for pat in patterns:
                files.extend(sorted(ctx.stats_dir.rglob(pat)))
            self._discovered_files[cache_key] = sorted({p.resolve() for p in files if p.exists()})
        return self._discovered_files[cache_key]

    def set_fdr_results(self, results: Dict[str, Any]) -> None:
        """Store FDR correction results for downstream stages."""
        self._fdr_results = results

    def get_fdr_results(self) -> Optional[Dict[str, Any]]:
        """Get cached FDR results."""
        return self._fdr_results

    def load_manifest(self, ctx: BehaviorContext) -> Optional[Dict[str, Any]]:
        """Load feature manifest from disk if available."""
        if self._manifest_loaded:
            return self._manifest

        self._manifest_loaded = True
        from eeg_pipeline.infra.paths import deriv_features_path

        features_dir = deriv_features_path(ctx.deriv_root, ctx.subject)
        manifest_patterns = ["*_manifest.json", "*_features_manifest.json"]

        for pattern in manifest_patterns:
            for manifest_path in features_dir.glob(pattern):
                if manifest_path.exists():
                    text = manifest_path.read_text(encoding="utf-8")
                    try:
                        self._manifest = json.loads(text)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Invalid feature manifest JSON: {manifest_path}") from exc

                    self._populate_from_manifest()
                    ctx.logger.info("Loaded feature manifest: %s", manifest_path.name)
                    return self._manifest

        return None

    def _populate_from_manifest(self) -> None:
        """Populate type/band caches from loaded manifest."""
        if self._manifest is None:
            return

        features = self._manifest.get("features", [])
        for entry in features:
            name = entry.get("name")
            if not name:
                continue
            group = entry.get("group", "unknown")
            band = entry.get("band", "broadband")
            self._feature_types[name] = group
            self._feature_bands[name] = band if band and band != "unknown" else "broadband"

    def get_feature_type(self, feature: str, config: Any) -> str:
        """Get cached feature type from manifest or compute and cache."""
        if feature in self._feature_types:
            return self._feature_types[feature]
        self._feature_types[feature] = _infer_feature_type(feature, config)
        return self._feature_types[feature]

    def get_feature_band(self, feature: str, config: Any) -> str:
        """Get cached feature band from manifest or compute and cache."""
        if feature in self._feature_bands:
            return self._feature_bands[feature]
        self._feature_bands[feature] = _infer_feature_band(feature, config)
        return self._feature_bands[feature]

    def clear_feature_types(self) -> None:
        """Clear feature type cache."""
        self._feature_types.clear()
        self._feature_bands.clear()

    def clear(self) -> None:
        """Clear all cached data."""
        self._trial_table_df = None
        self._trial_table_path = None
        self._feature_cols.clear()
        self._filtered_feature_cols.clear()
        self._discovered_files.clear()
        self._fdr_results = None
        self._feature_types.clear()
        self._feature_bands.clear()
        self._manifest = None
        self._manifest_loaded = False
        self._stats_subfolders.clear()


# Global cache instance (reset per pipeline run)
_cache = _ResultCache()


###################################################################
# Stage Registry - Dependency Graph & Metadata
###################################################################


@dataclass(frozen=True)
class StageSpec:
    """Specification for a pipeline stage."""
    name: str
    description: str
    requires: Tuple[str, ...]  # Required resources/prior stages
    produces: Tuple[str, ...]  # What this stage outputs
    config_key: Optional[str] = None  # Config key to enable/disable
    group: str = "core"  # Stage group for UI organization


class StageRegistry:
    """Registry of pipeline stages with dependency resolution."""
    
    _stages: Dict[str, StageSpec] = {}
    
    # Resource types
    RESOURCE_TRIAL_TABLE = "trial_table"
    RESOURCE_EPOCHS = "epochs"
    RESOURCE_TFR = "tfr"
    RESOURCE_POWER_DF = "power_df"
    RESOURCE_TEMPERATURE = "temperature"
    RESOURCE_RATING = "rating"
    RESOURCE_FEATURES = "features"
    RESOURCE_DESIGN = "correlate_design"
    RESOURCE_EFFECT_SIZES = "effect_sizes"
    RESOURCE_PVALUES = "pvalues"
    RESOURCE_CORRELATIONS = "correlations"
    RESOURCE_CONDITION_EFFECTS = "condition_effects"
    
    # Stage groups
    GROUP_DATA_PREP = "data_prep"
    GROUP_CORRELATIONS = "correlations"
    GROUP_CONDITION = "condition"
    GROUP_TEMPORAL = "temporal"
    GROUP_ADVANCED = "advanced"
    GROUP_VALIDATION = "validation"
    GROUP_EXPORT = "export"
    
    @classmethod
    def register(cls, spec: StageSpec) -> None:
        cls._stages[spec.name] = spec
    
    @classmethod
    def get(cls, name: str) -> Optional[StageSpec]:
        return cls._stages.get(name)
    
    @classmethod
    def all_stages(cls) -> Dict[str, StageSpec]:
        return cls._stages.copy()
    
    @classmethod
    def stages_in_group(cls, group: str) -> List[StageSpec]:
        return [s for s in cls._stages.values() if s.group == group]
    
    @classmethod
    def get_prerequisites(cls, stage_name: str) -> List[str]:
        """Get all prerequisite stages for a given stage."""
        spec = cls._stages.get(stage_name)
        if not spec:
            return []
        
        prerequisites: List[str] = []
        for req in spec.requires:
            # Find stages that produce this requirement
            for other_name, other_spec in cls._stages.items():
                if req in other_spec.produces and other_name != stage_name:
                    if other_name not in prerequisites:
                        prerequisites.append(other_name)
                    # Recursively get prerequisites
                    for pre in cls.get_prerequisites(other_name):
                        if pre not in prerequisites:
                            prerequisites.append(pre)
        return prerequisites
    
    @classmethod
    def validate_stage_combo(cls, stages: List[str]) -> Tuple[bool, List[str]]:
        """Validate a combination of stages, return (valid, missing_prerequisites)."""
        missing: List[str] = []
        for stage in stages:
            prereqs = cls.get_prerequisites(stage)
            for prereq in prereqs:
                if prereq not in stages and prereq not in missing:
                    missing.append(prereq)
        return (len(missing) == 0, missing)
    
    @classmethod
    def auto_resolve_stages(cls, stages: List[str]) -> List[str]:
        """Given selected stages, return full list with prerequisites in execution order."""
        resolved: List[str] = []
        
        def add_with_deps(stage: str) -> None:
            if stage in resolved:
                return
            prereqs = cls.get_prerequisites(stage)
            for prereq in prereqs:
                add_with_deps(prereq)
            if stage not in resolved:
                resolved.append(stage)
        
        for stage in stages:
            add_with_deps(stage)
        return resolved
    
    @classmethod
    def compute_progress_steps(cls, stages: List[str]) -> List[Dict[str, Any]]:
        """Compute progress steps for UI display."""
        resolved = cls.auto_resolve_stages(stages)
        steps: List[Dict[str, Any]] = []
        for i, stage in enumerate(resolved):
            spec = cls._stages.get(stage)
            if spec:
                steps.append({
                    "index": i,
                    "name": stage,
                    "description": spec.description,
                    "group": spec.group,
                    "total": len(resolved),
                })
        return steps
    
    @classmethod
    def get_available_stages_for_context(cls, ctx: "BehaviorContext") -> List[str]:
        """Get stages that can run given the context's available data."""
        available_resources = set()
        
        # Check what resources the context has
        if ctx.aligned_events is not None:
            available_resources.add(cls.RESOURCE_TRIAL_TABLE)
        if ctx.power_df is not None and not ctx.power_df.empty:
            available_resources.add(cls.RESOURCE_POWER_DF)
            available_resources.add(cls.RESOURCE_FEATURES)
        if ctx.temperature is not None:
            available_resources.add(cls.RESOURCE_TEMPERATURE)
        # Check if rating column is available in aligned_events
        if ctx.aligned_events is not None:
            rating_col = ctx._find_rating_column() if hasattr(ctx, "_find_rating_column") else None
            if rating_col is not None:
                available_resources.add(cls.RESOURCE_RATING)
        if ctx.epochs_info is not None:
            available_resources.add(cls.RESOURCE_EPOCHS)
        
        available: List[str] = []
        for name, spec in cls._stages.items():
            # Check if all non-stage requirements are met
            can_run = True
            for req in spec.requires:
                # Skip stage-produced requirements (those are handled by prerequisites)
                is_stage_output = any(req in s.produces for s in cls._stages.values())
                if not is_stage_output and req not in available_resources:
                    can_run = False
                    break
            if can_run:
                available.append(name)
        return available
    
    @classmethod
    def list_stages(cls) -> List[Dict[str, Any]]:
        """List all registered stages with metadata for CLI discoverability."""
        return [
            {
                "name": spec.name,
                "description": spec.description,
                "group": spec.group,
                "requires": list(spec.requires),
                "produces": list(spec.produces),
                "config_key": spec.config_key,
            }
            for spec in cls._stages.values()
        ]
    
    @classmethod
    def dry_run(cls, stages: List[str]) -> Dict[str, Any]:
        """Return DAG resolution and expected outputs without executing."""
        resolved = cls.auto_resolve_stages(stages)
        expected_outputs = set()
        for stage in resolved:
            spec = cls.get(stage)
            if spec:
                expected_outputs.update(spec.produces)
        return {
            "requested": stages,
            "resolved": resolved,
            "execution_order": resolved,
            "expected_outputs": list(expected_outputs),
            "n_stages": len(resolved),
        }


###################################################################
# Stage Output Contracts (Dataclasses)
###################################################################


@dataclass
class CorrelationDesign:
    """Output contract for correlate_design stage."""
    targets: List[str]
    feature_cols: List[str]
    partial_covars: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class CorrelationResults:
    """Output contract for correlation stages."""
    records: List[Dict[str, Any]]
    df: Optional[pd.DataFrame] = None
    n_tests: int = 0
    n_significant: int = 0
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ConditionResults:
    """Output contract for condition comparison stages."""
    df: pd.DataFrame
    comparison_type: str = "column"
    n_pain: int = 0
    n_nonpain: int = 0
    n_significant: int = 0
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class RegressionResults:
    """Output contract for regression stage."""
    df: pd.DataFrame
    primary_unit: str = "trial"
    n_features: int = 0
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class TemporalResults:
    """Output contract for temporal stages."""
    power: Optional[Dict[str, Any]] = None
    itpc: Optional[Dict[str, Any]] = None
    erds: Optional[Dict[str, Any]] = None
    correction_method: str = "fdr"
    n_tests: int = 0
    n_significant: int = 0


@dataclass
class ClusterResults:
    """Output contract for cluster permutation stage."""
    n_clusters: int = 0
    n_significant: int = 0
    clusters: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass  
class FDRResults:
    """Output contract for FDR correction stages."""
    n_tests: int = 0
    n_significant: int = 0
    alpha: float = 0.05
    method: str = "fdr_bh"
    family_structure: Optional[Dict[str, Any]] = None
    family_df: Optional[pd.DataFrame] = None


@dataclass
class MixedEffectsResult:
    """Output contract for mixed-effects models (group-level)."""
    df: pd.DataFrame
    n_subjects: int = 0
    n_features: int = 0
    n_significant: int = 0
    random_effects: str = "intercept"
    family_structure: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class GroupLevelResult:
    """Output contract for group-level analysis."""
    mixed_effects: Optional[MixedEffectsResult] = None
    multilevel_correlations: Optional[pd.DataFrame] = None
    n_subjects: int = 0
    subjects: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


# Register all stages
_STAGE_SPECS = [
    # Data Preparation
    StageSpec(
        name="load",
        description="Load behavioral data",
        requires=(),
        produces=(StageRegistry.RESOURCE_RATING, StageRegistry.RESOURCE_TEMPERATURE, StageRegistry.RESOURCE_FEATURES),
        group=StageRegistry.GROUP_DATA_PREP,
    ),
    StageSpec(
        name="trial_table",
        description="Build canonical trial table",
        requires=(StageRegistry.RESOURCE_FEATURES,),
        produces=(StageRegistry.RESOURCE_TRIAL_TABLE,),
        config_key="build_trial_table",
        group=StageRegistry.GROUP_DATA_PREP,
    ),
    StageSpec(
        name="lag_features",
        description="Add lag/delta features",
        requires=(StageRegistry.RESOURCE_TRIAL_TABLE,),
        produces=("lag_features",),
        group=StageRegistry.GROUP_DATA_PREP,
    ),
    StageSpec(
        name="pain_residual",
        description="Compute pain residual",
        requires=(StageRegistry.RESOURCE_TRIAL_TABLE, StageRegistry.RESOURCE_TEMPERATURE, StageRegistry.RESOURCE_RATING),
        produces=("pain_residual",),
        group=StageRegistry.GROUP_DATA_PREP,
    ),
    StageSpec(
        name="temperature_models",
        description="Compare temperature-rating models",
        requires=(StageRegistry.RESOURCE_TRIAL_TABLE, StageRegistry.RESOURCE_TEMPERATURE, StageRegistry.RESOURCE_RATING),
        produces=("temperature_models",),
        group=StageRegistry.GROUP_DATA_PREP,
    ),
    StageSpec(
        name="feature_qc",
        description="Screen feature quality",
        requires=(StageRegistry.RESOURCE_TRIAL_TABLE,),
        produces=("feature_qc",),
        config_key="behavior_analysis.feature_qc.enabled",
        group=StageRegistry.GROUP_DATA_PREP,
    ),
    
    # Correlations Pipeline
    StageSpec(
        name="correlate_design",
        description="Assemble correlation design matrix",
        requires=(StageRegistry.RESOURCE_TRIAL_TABLE,),
        produces=(StageRegistry.RESOURCE_DESIGN,),
        group=StageRegistry.GROUP_CORRELATIONS,
    ),
    StageSpec(
        name="correlate_effect_sizes",
        description="Compute correlation effect sizes",
        requires=(StageRegistry.RESOURCE_DESIGN,),
        produces=(StageRegistry.RESOURCE_EFFECT_SIZES,),
        group=StageRegistry.GROUP_CORRELATIONS,
    ),
    StageSpec(
        name="correlate_pvalues",
        description="Compute permutation p-values",
        requires=(StageRegistry.RESOURCE_DESIGN, StageRegistry.RESOURCE_EFFECT_SIZES),
        produces=(StageRegistry.RESOURCE_PVALUES,),
        config_key="behavior_analysis.correlations.permutation.enabled",
        group=StageRegistry.GROUP_CORRELATIONS,
    ),
    StageSpec(
        name="correlate_primary_selection",
        description="Select primary p-value and effect size for FDR",
        requires=(StageRegistry.RESOURCE_DESIGN, StageRegistry.RESOURCE_EFFECT_SIZES),
        produces=("primary_selection",),
        group=StageRegistry.GROUP_CORRELATIONS,
    ),
    StageSpec(
        name="correlate_fdr",
        description="Apply FDR correction",
        requires=("primary_selection",),
        produces=(StageRegistry.RESOURCE_CORRELATIONS,),
        group=StageRegistry.GROUP_CORRELATIONS,
    ),
    StageSpec(
        name="pain_sensitivity",
        description="Pain sensitivity correlations",
        requires=(StageRegistry.RESOURCE_TRIAL_TABLE, StageRegistry.RESOURCE_TEMPERATURE, StageRegistry.RESOURCE_RATING),
        produces=("pain_sensitivity",),
        config_key="compute_pain_sensitivity",
        group=StageRegistry.GROUP_CORRELATIONS,
    ),
    StageSpec(
        name="regression",
        description="Trialwise regression",
        requires=(StageRegistry.RESOURCE_TRIAL_TABLE,),
        produces=("regression",),
        config_key="run_regression",
        group=StageRegistry.GROUP_CORRELATIONS,
    ),
    StageSpec(
        name="models",
        description="Fit model families",
        requires=(StageRegistry.RESOURCE_TRIAL_TABLE,),
        produces=("models",),
        config_key="run_models",
        group=StageRegistry.GROUP_CORRELATIONS,
    ),
    StageSpec(
        name="stability",
        description="Groupwise stability",
        requires=(StageRegistry.RESOURCE_TRIAL_TABLE,),
        produces=("stability",),
        group=StageRegistry.GROUP_CORRELATIONS,
    ),
    StageSpec(
        name="consistency",
        description="Effect direction consistency",
        requires=(StageRegistry.RESOURCE_CORRELATIONS,),
        produces=("consistency",),
        group=StageRegistry.GROUP_VALIDATION,
    ),
    StageSpec(
        name="influence",
        description="Influence diagnostics",
        requires=(StageRegistry.RESOURCE_TRIAL_TABLE, StageRegistry.RESOURCE_CORRELATIONS),
        produces=("influence",),
        group=StageRegistry.GROUP_VALIDATION,
    ),
    
    # Condition Comparisons
    StageSpec(
        name="condition_column",
        description="Column-based condition contrast",
        requires=(StageRegistry.RESOURCE_TRIAL_TABLE,),
        produces=(StageRegistry.RESOURCE_CONDITION_EFFECTS,),
        config_key="run_condition_comparison",
        group=StageRegistry.GROUP_CONDITION,
    ),
    StageSpec(
        name="condition_window",
        description="Window-based condition contrast",
        requires=(StageRegistry.RESOURCE_TRIAL_TABLE,),
        produces=("window_effects",),
        config_key="behavior_analysis.condition.compare_windows",
        group=StageRegistry.GROUP_CONDITION,
    ),
    
    # Temporal Analysis
    StageSpec(
        name="temporal_tfr",
        description="Time-frequency correlations",
        requires=(StageRegistry.RESOURCE_TFR, StageRegistry.RESOURCE_RATING),
        produces=("tfr_correlations",),
        config_key="run_temporal_correlations",
        group=StageRegistry.GROUP_TEMPORAL,
    ),
    StageSpec(
        name="temporal_stats",
        description="Temporal statistics (power/ITPC/ERDS)",
        requires=(StageRegistry.RESOURCE_POWER_DF, StageRegistry.RESOURCE_RATING),
        produces=("temporal_stats",),
        config_key="run_temporal_correlations",
        group=StageRegistry.GROUP_TEMPORAL,
    ),
    StageSpec(
        name="cluster",
        description="Cluster permutation tests",
        requires=(StageRegistry.RESOURCE_EPOCHS,),
        produces=("cluster_results",),
        config_key="run_cluster_tests",
        group=StageRegistry.GROUP_TEMPORAL,
    ),
    
    # Advanced Analysis
    StageSpec(
        name="mediation",
        description="Mediation analysis",
        requires=(StageRegistry.RESOURCE_TRIAL_TABLE, StageRegistry.RESOURCE_TEMPERATURE, StageRegistry.RESOURCE_RATING),
        produces=("mediation",),
        config_key="run_mediation",
        group=StageRegistry.GROUP_ADVANCED,
    ),
    StageSpec(
        name="moderation",
        description="Moderation analysis",
        requires=(StageRegistry.RESOURCE_TRIAL_TABLE, StageRegistry.RESOURCE_TEMPERATURE, StageRegistry.RESOURCE_RATING),
        produces=("moderation",),
        config_key="run_moderation",
        group=StageRegistry.GROUP_ADVANCED,
    ),
    StageSpec(
        name="mixed_effects",
        description="Mixed-effects models (group-level)",
        requires=(StageRegistry.RESOURCE_TRIAL_TABLE,),
        produces=("mixed_effects",),
        config_key="run_mixed_effects",
        group=StageRegistry.GROUP_ADVANCED,
    ),
    StageSpec(
        name="hierarchical_fdr_summary",
        description="Summarize hierarchical FDR across analyses",
        requires=(),
        produces=("hierarchical_fdr_summary",),
        group=StageRegistry.GROUP_VALIDATION,
    ),
    
    # Validation & Export
    StageSpec(
        name="report",
        description="Generate subject report",
        requires=(),
        produces=("report",),
        group=StageRegistry.GROUP_EXPORT,
    ),
    StageSpec(
        name="export",
        description="Export results",
        requires=(),
        produces=("exports",),
        group=StageRegistry.GROUP_EXPORT,
    ),
]


for _spec in _STAGE_SPECS:
    StageRegistry.register(_spec)


###################################################################
# Stage Runners Registry (Data-Driven Dispatch)
###################################################################


class _ResultsFromOutputs:
    """Lightweight results object built from DAG stage outputs for export stage."""
    
    def __init__(self, outputs: Dict[str, Any]):
        # Map stage output keys to result attributes
        self.correlations = outputs.get("correlate_fdr")
        self.pain_sensitivity = outputs.get("pain_sensitivity")
        self.condition_effects = outputs.get("condition_column")
        self.condition_effects_window = outputs.get("condition_window")
        self.mediation = outputs.get("mediation")
        self.moderation = outputs.get("moderation")
        self.mixed_effects = outputs.get("mixed_effects")
        self.regression = outputs.get("regression")
        self.models = outputs.get("models")
        self.stability = outputs.get("stability")
        self.consistency = outputs.get("consistency")
        self.influence = outputs.get("influence")
        self.trial_table_path = outputs.get("trial_table")
        self.report_path = outputs.get("report")
        self.subject = None
        self.summary = {}
    
    def to_summary(self) -> Dict[str, Any]:
        """Return summary dict compatible with BehaviorPipelineResults.to_summary()."""
        summary = {}
        if self.trial_table_path:
            summary["trial_table_path"] = str(self.trial_table_path)
        if self.report_path:
            summary["report_path"] = str(self.report_path)
        
        n_total = 0
        n_sig_raw = 0
        n_sig_fdr = 0
        
        if self.correlations is not None and hasattr(self.correlations, 'empty') and not self.correlations.empty:
            n_total = len(self.correlations)
            if "p_raw" in self.correlations.columns:
                n_sig_raw = int((self.correlations["p_raw"] < 0.05).sum())
            if "p_fdr" in self.correlations.columns:
                n_sig_fdr = int((self.correlations["p_fdr"] < 0.05).sum())
        
        summary["n_features"] = n_total
        summary["n_significant_raw"] = n_sig_raw
        summary["n_significant_fdr"] = n_sig_fdr
        
        return summary


def _build_results_from_outputs(outputs: Dict[str, Any]) -> Any:
    """Build a results-like object from DAG stage outputs.
    
    The export stage expects a results object with attributes like correlations,
    pain_sensitivity, etc. When running via the DAG, we need to construct this
    from the outputs dictionary.
    """
    if not outputs:
        return None
    return _ResultsFromOutputs(outputs)


def _get_stage_runners() -> Dict[str, callable]:
    """Return data-driven stage runner mapping.
    
    Replaces the long if/elif chain in run_selected_stages with a dict lookup.
    Adding/removing stages only requires updating this dict.
    """
    return {
        "load": lambda ctx, config, outputs: stage_load(ctx),
        "trial_table": lambda ctx, config, outputs: stage_trial_table(ctx, config),
        "lag_features": lambda ctx, config, outputs: stage_lag_features(ctx, config),
        "pain_residual": lambda ctx, config, outputs: stage_pain_residual(ctx, config),
        "temperature_models": lambda ctx, config, outputs: stage_temperature_models(ctx, config),
        "feature_qc": lambda ctx, config, outputs: stage_feature_qc_screen(ctx, config),
        "correlate_design": lambda ctx, config, outputs: stage_correlate_design(ctx, config),
        "correlate_effect_sizes": lambda ctx, config, outputs: stage_correlate_effect_sizes(
            ctx, config, outputs.get("correlate_design")
        ),
        "correlate_pvalues": lambda ctx, config, outputs: stage_correlate_pvalues(
            ctx, config, outputs.get("correlate_design"), outputs.get("correlate_effect_sizes", [])
        ),
        "correlate_primary_selection": lambda ctx, config, outputs: stage_correlate_primary_selection(
            ctx,
            config,
            outputs.get("correlate_design"),
            outputs.get("correlate_pvalues") or outputs.get("correlate_effect_sizes", []),
        ),
        "correlate_fdr": lambda ctx, config, outputs: stage_correlate_fdr(
            ctx,
            config,
            outputs.get("correlate_primary_selection")
            or outputs.get("correlate_pvalues")
            or outputs.get("correlate_effect_sizes", []),
        ),
        "pain_sensitivity": lambda ctx, config, outputs: stage_pain_sensitivity(ctx, config),
        "regression": lambda ctx, config, outputs: stage_regression(ctx, config),
        "models": lambda ctx, config, outputs: stage_models(ctx, config),
        "stability": lambda ctx, config, outputs: stage_stability(ctx, config),
        "consistency": lambda ctx, config, outputs: stage_consistency(
            ctx, config, _build_results_from_outputs(outputs)
        ),
        "influence": lambda ctx, config, outputs: stage_influence(
            ctx, config, _build_results_from_outputs(outputs)
        ),
        "condition_column": lambda ctx, config, outputs: stage_condition_column(ctx, config),
        "condition_window": lambda ctx, config, outputs: stage_condition_window(ctx, config),
        "temporal_tfr": lambda ctx, config, outputs: stage_temporal_tfr(ctx),
        "temporal_stats": lambda ctx, config, outputs: stage_temporal_stats(ctx),
        "cluster": lambda ctx, config, outputs: stage_cluster(ctx, config),
        "mediation": lambda ctx, config, outputs: stage_mediation(ctx, config),
        "moderation": lambda ctx, config, outputs: stage_moderation(ctx, config),
        "mixed_effects": lambda ctx, config, outputs: stage_mixed_effects(ctx, config),
        "hierarchical_fdr_summary": lambda ctx, config, outputs: stage_hierarchical_fdr_summary(ctx, config),
        "report": lambda ctx, config, outputs: stage_report(ctx, config),
        "export": lambda ctx, config, outputs: stage_export(ctx, config, _build_results_from_outputs(outputs)),
    }


STAGE_RUNNERS: Dict[str, callable] = _get_stage_runners()


def _is_stage_enabled_by_config(stage_name: str, config: Any) -> bool:
    """Check if stage is enabled via its config_key.
    
    Centralizes config-based stage skipping logic.
    """
    spec = StageRegistry.get(stage_name)
    if spec is None or spec.config_key is None:
        return True
    
    config_value = get_config_value(config, spec.config_key, None)
    if config_value is None:
        return True
    
    return bool(config_value)


###################################################################
# Feature QC Screen Stage
###################################################################


@dataclass
class FeatureQCResult:
    """Result from feature QC screening."""
    passed_features: List[str]
    failed_features: Dict[str, List[str]]  # reason -> list of features
    qc_df: pd.DataFrame
    metadata: Dict[str, Any]


def _check_within_run_variance(
    df_trials: pd.DataFrame,
    feature_col: str,
    run_col: str,
    min_variance: float,
) -> bool:
    """Check if feature has sufficient variance within runs."""
    run_variances = df_trials.groupby(run_col)[feature_col].apply(
        lambda x: pd.to_numeric(x, errors="coerce").var()
    )
    all_runs_constant = (run_variances.fillna(0) < min_variance).all()
    return not all_runs_constant


def _evaluate_feature_quality(
    feature_col: str,
    values: pd.Series,
    df_trials: pd.DataFrame,
    max_missing_pct: float,
    min_variance: float,
    check_within_run: bool,
    run_col: str,
) -> Dict[str, Any]:
    """Evaluate quality metrics for a single feature."""
    total_count = len(values)
    missing_count = values.isna().sum()
    missing_pct = missing_count / total_count if total_count > 0 else 1.0
    valid_count = values.notna().sum()
    variance = values.var() if valid_count > 1 else 0.0
    
    within_run_ok = True
    if check_within_run and run_col in df_trials.columns:
        within_run_ok = _check_within_run_variance(df_trials, feature_col, run_col, min_variance)
    
    return {
        "feature": feature_col,
        "n_total": total_count,
        "n_missing": missing_count,
        "missing_pct": missing_pct,
        "variance": variance,
        "within_run_variance_ok": within_run_ok,
        "passed": True,
    }


def _classify_feature_failure(
    feature_col: str,
    qc_metrics: Dict[str, Any],
    max_missing_pct: float,
    min_variance: float,
) -> Optional[str]:
    """Classify why a feature failed QC, returning failure reason or None if passed."""
    if qc_metrics["missing_pct"] > max_missing_pct:
        return "high_missingness"
    if qc_metrics["variance"] < min_variance:
        return "near_zero_variance"
    if not qc_metrics["within_run_variance_ok"]:
        return "constant_within_run"
    return None


def stage_feature_qc_screen(
    ctx: BehaviorContext,
    config: Any,
) -> FeatureQCResult:
    """Filter features by data quality before inference.
    
    Single responsibility: QC-based feature filtering (no circular selection).
    
    Filters by:
    - Missingness: > threshold missing values
    - Near-zero variance: variance < threshold
    - Constant within-run: no variation within runs
    - Reliability: ICC or split-half < threshold (optional)
    """
    df_trials = _load_trial_table_df(ctx)
    if not _is_dataframe_valid(df_trials):
        ctx.logger.warning("Feature QC: trial table missing; skipping.")
        return FeatureQCResult([], {}, pd.DataFrame(), {"status": "skipped"})

    feature_cols = [col for col in df_trials.columns if str(col).startswith(FEATURE_COLUMN_PREFIXES)]
    if not feature_cols:
        return FeatureQCResult([], {}, pd.DataFrame(), {"status": "no_features"})

    max_missing_pct = get_config_float(
        ctx.config, "behavior_analysis.feature_qc.max_missing_pct", MAX_MISSING_PCT_DEFAULT
    )
    min_variance = get_config_float(
        ctx.config, "behavior_analysis.feature_qc.min_variance", MIN_VARIANCE_THRESHOLD
    )
    check_within_run = get_config_bool(
        ctx.config, "behavior_analysis.feature_qc.check_within_run_variance", True
    )
    run_col = str(get_config_value(
        ctx.config, "behavior_analysis.run_adjustment.column", "run_id"
    ) or "run_id").strip()

    passed_features = []
    failed_features: Dict[str, List[str]] = {
        "high_missingness": [],
        "near_zero_variance": [],
        "constant_within_run": [],
    }
    qc_records: List[Dict[str, Any]] = []

    for feature_col in feature_cols:
        values = pd.to_numeric(df_trials[feature_col], errors="coerce")
        qc_metrics = _evaluate_feature_quality(
            feature_col, values, df_trials, max_missing_pct, min_variance, check_within_run, run_col
        )
        
        failure_reason = _classify_feature_failure(feature_col, qc_metrics, max_missing_pct, min_variance)
        if failure_reason:
            failed_features[failure_reason].append(feature_col)
            qc_metrics["passed"] = False
        else:
            passed_features.append(feature_col)
        
        qc_records.append(qc_metrics)

    qc_df = pd.DataFrame(qc_records)
    n_failed = sum(len(feature_list) for feature_list in failed_features.values())
    
    n_total = len(feature_cols)
    n_passed = len(passed_features)
    pass_rate = 100 * n_passed / n_total if n_total > 0 else 0.0
    ctx.logger.info(
        "Feature QC: %d/%d passed (%.1f%%), %d failed",
        n_passed, n_total, pass_rate, n_failed
    )
    
    for reason, feature_list in failed_features.items():
        if feature_list:
            ctx.logger.info("  %s: %d features", reason, len(feature_list))

    suffix = _feature_suffix_from_context(ctx)
    out_dir = _get_stats_subfolder(ctx, "feature_qc")
    out_path = out_dir / f"feature_qc_screen{suffix}.parquet"
    _write_parquet_with_optional_csv(qc_df, out_path, also_save_csv=ctx.also_save_csv)

    metadata = {
        "status": "ok",
        "n_total": n_total,
        "n_passed": n_passed,
        "n_failed": n_failed,
        "thresholds": {
            "max_missing_pct": max_missing_pct,
            "min_variance": min_variance,
            "check_within_run": check_within_run,
        },
    }
    ctx.data_qc["feature_qc_screen"] = metadata

    return FeatureQCResult(passed_features, failed_features, qc_df, metadata)


###################################################################
# Stage Executor - Run Selected Stages from Registry
###################################################################


def config_to_stage_names(pipeline_config: Any) -> List[str]:
    """Map pipeline config flags to stage names for DAG-based execution.
    
    This replaces hard-coded if/else chains in the pipeline layer.
    """
    stages = ["load"]  # Always run load
    
    if getattr(pipeline_config, "run_trial_table", True):
        stages.append("trial_table")
    
    if getattr(pipeline_config, "run_lag_features", False):
        stages.append("lag_features")
    
    if getattr(pipeline_config, "run_pain_residual", False):
        stages.append("pain_residual")
    
    if getattr(pipeline_config, "run_temperature_models", False):
        stages.append("temperature_models")
    
    if getattr(pipeline_config, "run_feature_qc", False):
        stages.append("feature_qc")
    
    if getattr(pipeline_config, "run_regression", False):
        stages.append("regression")
    
    if getattr(pipeline_config, "run_models", False):
        stages.append("models")
    
    if getattr(pipeline_config, "run_stability", False):
        stages.append("stability")
    
    if getattr(pipeline_config, "run_correlations", True):
        stages.extend(["correlate_design", "correlate_effect_sizes", "correlate_pvalues", "correlate_primary_selection", "correlate_fdr"])
    
    if getattr(pipeline_config, "compute_pain_sensitivity", False):
        stages.append("pain_sensitivity")
    
    if getattr(pipeline_config, "run_consistency", False):
        stages.append("consistency")
    
    if getattr(pipeline_config, "run_influence", False):
        stages.append("influence")
    
    if getattr(pipeline_config, "run_condition_comparison", False):
        stages.extend(["condition_column", "condition_window"])
    
    if getattr(pipeline_config, "run_temporal_correlations", False):
        stages.extend(["temporal_tfr", "temporal_stats"])
    
    if getattr(pipeline_config, "run_cluster_tests", False):
        stages.append("cluster")
    
    if getattr(pipeline_config, "run_mediation", False):
        stages.append("mediation")
    
    if getattr(pipeline_config, "run_moderation", False):
        stages.append("moderation")
    
    if getattr(pipeline_config, "run_mixed_effects", False):
        stages.append("mixed_effects")
    
    if getattr(pipeline_config, "run_validation", True):
        stages.append("hierarchical_fdr_summary")
    
    if getattr(pipeline_config, "run_report", False):
        stages.append("report")
    
    stages.append("export")  # Always run export
    
    return stages


def run_selected_stages(
    ctx: BehaviorContext,
    config: Any,
    selected_stages: List[str],
    results: Optional[Any] = None,
    progress: Optional[Any] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Execute selected stages in dependency order using the stage registry.
    
    Uses data-driven STAGE_RUNNERS dict instead of if/elif chains.
    
    Args:
        ctx: Behavior context
        config: Pipeline configuration
        selected_stages: List of stage names to run
        results: Optional results object to populate
        progress: Optional progress reporter
        dry_run: If True, return execution plan without running stages
    
    Returns:
        Dict mapping stage names to their outputs
    """
    resolved = StageRegistry.auto_resolve_stages(selected_stages)
    
    # Filter by config_key (auto-skip disabled stages)
    enabled_stages = [
        s for s in resolved
        if _is_stage_enabled_by_config(s, ctx.config)
    ]
    
    skipped = set(resolved) - set(enabled_stages)
    if skipped:
        ctx.logger.info("Auto-skipped stages (disabled by config): %s", ", ".join(skipped))
    
    ctx.logger.info("Running %d stages: %s", len(enabled_stages), ", ".join(enabled_stages))
    
    if dry_run:
        return StageRegistry.dry_run(enabled_stages)

    outputs: Dict[str, Any] = {}
    progress_steps = StageRegistry.compute_progress_steps(enabled_stages)

    def _run_stage(name: str) -> Any:
        """Run a single stage using STAGE_RUNNERS dict lookup."""
        runner = STAGE_RUNNERS.get(name)
        if runner is None:
            raise KeyError(f"Stage '{name}' has no implementation in STAGE_RUNNERS")
        return runner(ctx, config, outputs)

    import time as _time

    for step in progress_steps:
        stage_name = step["name"]
        
        if progress is not None:
            progress.step(stage_name, current=step["index"] + 1, total=step["total"])
        
        t0 = _time.perf_counter()
        try:
            output = _run_stage(stage_name)
            stage_elapsed = _time.perf_counter() - t0
            outputs[stage_name] = output

            if results is not None:
                _update_results_from_stage(results, stage_name, output)

            _log_stage_outcome(ctx.logger, stage_name, output, stage_elapsed,
                               step["index"] + 1, step["total"])
        except Exception as exc:
            ctx.logger.error("Stage '%s' failed after %.1fs: %s",
                             stage_name, _time.perf_counter() - t0, exc)
            raise

    return outputs


def _log_stage_outcome(
    logger: Any,
    stage_name: str,
    output: Any,
    elapsed: float,
    step_num: int,
    total_steps: int,
) -> None:
    """Log concise outcome for a completed behavior stage."""
    detail = ""
    if isinstance(output, pd.DataFrame) and not output.empty:
        detail = f" ({len(output)} rows, {output.shape[1]} cols)"
    elif isinstance(output, dict):
        n_keys = len(output)
        if n_keys > 0:
            detail = f" ({n_keys} outputs)"
    elif isinstance(output, (str, Path)) and output:
        detail = f" -> {Path(str(output)).name}"
    logger.info(
        "[%d/%d] \u2713 %s%s (%.1fs)",
        step_num, total_steps, stage_name, detail, elapsed,
    )


# Stage-to-attribute mapping for results object
_STAGE_TO_ATTR_MAP = {
    "trial_table": "trial_table_path",
    "correlate_fdr": "correlations",
    "pain_sensitivity": "pain_sensitivity",
    "regression": "regression",
    "models": "models",
    "stability": "stability",
    "consistency": "consistency",
    "influence": "influence",
    "condition_column": "condition_effects",
    "condition_window": "condition_effects_window",
    "temporal_tfr": "tf",
    "temporal_stats": "temporal",
    "cluster": "cluster",
    "mediation": "mediation",
    "moderation": "moderation",
    "mixed_effects": "mixed_effects",
    "report": "report_path",
}


def _update_results_from_stage(results: Any, stage_name: str, output: Any) -> None:
    """Update BehaviorPipelineResults from stage output."""
    attr = _STAGE_TO_ATTR_MAP.get(stage_name)
    if attr and hasattr(results, attr):
        setattr(results, attr, output)


def run_behavior_stages(
    ctx: BehaviorContext,
    pipeline_config: Any,
    results: Optional[Any] = None,
    progress: Optional[Any] = None,
) -> Dict[str, Any]:
    """Run behavior pipeline stages based on pipeline config.
    
    This is the main entry point for DAG-based stage execution.
    Replaces hard-coded if/else chains in BehaviorPipeline.process_subject.
    
    Args:
        ctx: Behavior context
        pipeline_config: Pipeline configuration with run_* flags
        results: Optional BehaviorPipelineResults to populate
        progress: Optional progress reporter
    
    Returns:
        Dict mapping stage names to their outputs
    """
    stages = config_to_stage_names(pipeline_config)
    return run_selected_stages(ctx, pipeline_config, stages, results, progress)


# Centralized feature column prefixes - single source of truth
FEATURE_COLUMN_PREFIXES = (
    "power_",
    "connectivity_",
    "directedconnectivity_",
    "sourcelocalization_",
    "aperiodic_",
    "erp_",
    "itpc_",
    "pac_",
    "complexity_",
    "bursts_",
    "quality_",
    "erds_",
    "spectral_",
    "ratios_",
    "asymmetry_",
    "microstates_",
    "temporal_",
)

CATEGORY_PREFIX_MAP = {prefix.rstrip("_"): prefix for prefix in FEATURE_COLUMN_PREFIXES}

# Constants for validation thresholds
MIN_SAMPLES_DEFAULT = 10
MIN_SAMPLES_RUN_LEVEL = 3
MIN_VARIANCE_THRESHOLD = 1e-10
CONSTANT_VARIANCE_THRESHOLD = 1e-12
MAX_MISSING_PCT_DEFAULT = 0.2
FDR_ALPHA_DEFAULT = 0.05
MIN_FEATURES_FOR_ANALYSIS = 1
MIN_TRIALS_FOR_ANALYSIS = 1


def _check_early_exit_conditions(
    df: Optional[pd.DataFrame],
    feature_cols: Optional[List[str]] = None,
    min_features: int = MIN_FEATURES_FOR_ANALYSIS,
    min_trials: int = MIN_TRIALS_FOR_ANALYSIS,
) -> Tuple[bool, Optional[str]]:
    """Check if analysis should skip due to insufficient data.

    Returns:
        (should_skip, skip_reason)
    """
    if not _is_dataframe_valid(df):
        return True, "no_dataframe"

    if len(df) < min_trials:
        return True, f"insufficient_trials ({len(df)} < {min_trials})"

    if feature_cols is not None:
        if len(feature_cols) < min_features:
            return True, f"insufficient_features ({len(feature_cols)} < {min_features})"

        # Check if any features have valid data
        has_valid_features = False
        for col in feature_cols:
            if col in df.columns:
                valid_count = pd.to_numeric(df[col], errors="coerce").notna().sum()
                if valid_count >= min_trials:
                    has_valid_features = True
                    break

        if not has_valid_features:
            return True, "no_features_with_valid_data"

    return False, None


def _is_dataframe_valid(df: Optional[pd.DataFrame]) -> bool:
    """Check if DataFrame is not None and not empty.

    Encapsulates boundary condition for DataFrame validation.
    """
    return df is not None and not df.empty


def _require_trial_table(stage_name: str):
    """Decorator to skip stage if trial table missing.

    Injects loaded trial table as first argument after ctx and config.
    Returns empty DataFrame if trial table unavailable.
    """
    def decorator(func):
        def wrapper(ctx: BehaviorContext, config: Any, *args, **kwargs):
            df_trials = _load_trial_table_df(ctx)
            if not _is_dataframe_valid(df_trials):
                ctx.logger.warning(f"{stage_name}: trial table missing; skipping.")
                return pd.DataFrame()
            return func(ctx, config, df_trials, *args, **kwargs)
        return wrapper
    return decorator


def _get_stats_subfolder(ctx: BehaviorContext, kind: str) -> Path:
    """Helper to get a subfolder within stats_dir and ensure it exists.
    
    If ctx.overwrite is False, appends a timestamp to the folder name
    (e.g., 'trial_table_20260120_143022') to preserve previous outputs.
    """
    return get_behavior_output_dir(ctx, kind, ensure=True)


def _get_stats_subfolder_with_overwrite(
    stats_dir: Path,
    kind: str,
    overwrite: bool,
    *,
    ensure: bool = True,
) -> Path:
    """Helper to get a subfolder within stats_dir with overwrite control.
    
    If overwrite is False, appends a timestamp to the folder name
    (e.g., 'trial_table_20260120_143022') to preserve previous outputs.
    """
    return _cache.get_stats_subfolder(stats_dir, kind, overwrite, ensure=ensure)


def _sanitize_path_component(value: str) -> str:
    """Sanitize a string for use as a single path component."""
    value = str(value).strip()
    if not value:
        return "all"
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")
    out = []
    for ch in value:
        out.append(ch if ch in allowed else "_")
    cleaned = "".join(out).strip("._-")
    return cleaned if cleaned else "all"


def _feature_folder_from_context(ctx: BehaviorContext) -> str:
    """Resolve the feature folder name for this run."""
    raw = ctx.selected_feature_files or ctx.feature_categories or []
    items = [str(x).strip() for x in raw if str(x).strip()]
    items = [x for x in items if x.lower() != "all"]
    if not items:
        return "all"
    return _sanitize_path_component("_".join(sorted(items)))


def _feature_folder_from_list(feature_files: List[str]) -> str:
    items = [str(x).strip() for x in feature_files if str(x).strip()]
    items = [x for x in items if x.lower() != "all"]
    if not items:
        return "all"
    return _sanitize_path_component("_".join(sorted(items)))


def get_behavior_output_dir(ctx: BehaviorContext, kind: str, *, ensure: bool = True) -> Path:
    """Return `stats_dir/<kind>/<feature_folder>` (and optionally create it)."""
    kind_dir = _get_stats_subfolder_with_overwrite(ctx.stats_dir, kind, ctx.overwrite, ensure=ensure)
    feature_dir = kind_dir / _feature_folder_from_context(ctx)
    if ensure:
        ensure_dir(feature_dir)
    return feature_dir


def _write_stats_table(
    ctx: BehaviorContext,
    df: pd.DataFrame,
    path: Path,
    force_tsv: bool = False,
) -> Path:
    """Write a stats table, using parquet for large DataFrames.
    
    Automatically uses parquet format for DataFrames with >100 rows
    unless force_tsv is True.
    
    Args:
        ctx: BehaviorContext
        df: DataFrame to write
        path: Output path (extension adjusted automatically)
        force_tsv: If True, always use TSV regardless of size
        
    Returns:
        Actual path written
    """
    from eeg_pipeline.infra.tsv import write_stats_table
    actual_path = write_stats_table(df, path, force_tsv=force_tsv)
    
    if ctx.also_save_csv:
        from eeg_pipeline.infra.tsv import write_csv
        csv_path = actual_path.with_suffix(".csv")
        write_csv(df, csv_path, index=False)
        ctx.logger.info("Also saved stats table as CSV: %s", csv_path.name)
    
    return actual_path


def _get_feature_columns(
    df: pd.DataFrame,
    ctx: BehaviorContext,
    computation_name: Optional[str] = None,
) -> List[str]:
    """Extract and filter feature columns from DataFrame.
    
    Centralizes the pattern of extracting feature columns and applying
    band and computation-specific filters.
    
    Args:
        df: DataFrame containing feature columns
        ctx: BehaviorContext with filtering preferences
        computation_name: Optional computation name for feature filtering
        
    Returns:
        List of filtered feature column names
    """
    feature_cols = [c for c in df.columns if str(c).startswith(FEATURE_COLUMN_PREFIXES)]
    return _cache.get_filtered_feature_cols(feature_cols, ctx, computation_name)


def _attach_temperature_metadata(
    df: pd.DataFrame,
    metadata_dict: Dict[str, Any],
    target_col: Optional[str] = None,
) -> pd.DataFrame:
    """Attach temperature control metadata to result DataFrame.
    
    Centralizes the repeated pattern of adding temperature control columns.
    
    Args:
        df: Result DataFrame to modify
        metadata_dict: Metadata dictionary containing temperature control info
        target_col: Optional target column name for per-target metadata mapping
        
    Returns:
        Modified DataFrame with temperature metadata columns
    """
    if df is None or df.empty:
        return df
    
    if "temperature_control" not in df.columns:
        df = df.copy()
        df["temperature_control"] = metadata_dict.get("temperature_control", None)
        df["temperature_control_used"] = metadata_dict.get("temperature_control_used", None)
        
        spline_meta = metadata_dict.get("temperature_spline", None)
        if isinstance(spline_meta, dict):
            df["temperature_spline_status"] = spline_meta.get("status", None)
            df["temperature_spline_n_knots"] = spline_meta.get("n_knots", None)
            df["temperature_spline_quantile_low"] = spline_meta.get("quantile_low", None)
            df["temperature_spline_quantile_high"] = spline_meta.get("quantile_high", None)
    
    if target_col and target_col in df.columns:
        ctrl_by_out = metadata_dict.get("temperature_control_by_outcome", None)
        if isinstance(ctrl_by_out, dict):
            used_map = {str(k): (v or {}).get("temperature_control_used", None) for k, v in ctrl_by_out.items()}
            status_map = {}
            nknots_map = {}
            for k, v in ctrl_by_out.items():
                s = (v or {}).get("temperature_spline", None)
                if isinstance(s, dict):
                    status_map[str(k)] = s.get("status", None)
                    nknots_map[str(k)] = s.get("n_knots", None)
            df["temperature_control_used"] = df[target_col].astype(str).map(used_map)
            df["temperature_spline_status"] = df[target_col].astype(str).map(status_map)
            df["temperature_spline_n_knots"] = df[target_col].astype(str).map(nknots_map)
    
    return df


def _find_stats_path(ctx: BehaviorContext, filename: str) -> Optional[Path]:
    """Helper to find a file in stats_dir or its subfolders."""
    kind = _infer_output_kind(filename)
    if kind != "unknown":
        expected_dir = get_behavior_output_dir(ctx, kind, ensure=False)
        expected_path = expected_dir / filename
        if expected_path.exists():
            return expected_path

        candidate_path = ctx.stats_dir / kind / filename
        if candidate_path.exists():
            return candidate_path
    
    root_path = ctx.stats_dir / filename
    if root_path.exists():
        return root_path
    
    for candidate_path in ctx.stats_dir.rglob(filename):
        if candidate_path.is_file():
            return candidate_path
            
    return None


def _compute_feature_signature(ctx: BehaviorContext) -> str:
    """Compute hash signature of feature tables for caching."""
    parts = []
    for name, df in ctx.iter_feature_tables():
        if not _is_dataframe_valid(df):
            continue
        column_names = ",".join(str(c) for c in df.columns)
        parts.append(f"{name}:{df.shape[0]}:{df.shape[1]}:{column_names}")
    payload = "|".join(parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _compute_trial_table_input_hash(ctx: BehaviorContext) -> str:
    """Compute a deterministic input hash for trial-table stage caching."""
    parts: List[str] = [
        f"subject={ctx.subject}",
        f"task={ctx.task}",
        f"feature_signature={_compute_feature_signature(ctx)}",
    ]

    events = getattr(ctx, "aligned_events", None)
    if isinstance(events, pd.DataFrame) and not events.empty:
        try:
            events_hash_raw = pd.util.hash_pandas_object(
                events.reset_index(drop=True),
                index=True,
            ).to_numpy(dtype=np.uint64)
            events_hash = hashlib.sha256(events_hash_raw.tobytes()).hexdigest()
        except Exception:
            cols = ",".join(str(c) for c in events.columns)
            events_hash = f"fallback:{len(events)}:{events.shape[1]}:{cols}"
        parts.append(f"events_hash={events_hash}")
    else:
        parts.append("events_hash=none")

    feature_paths = getattr(ctx, "feature_paths", {}) or {}
    for name, path in sorted(feature_paths.items()):
        try:
            st = Path(path).stat()
        except Exception:
            continue
        parts.append(
            f"feature_path:{name}:{Path(path).resolve()}:{st.st_size}:{st.st_mtime_ns}"
        )

    payload = "|".join(parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _trial_table_metadata_path(table_path: Path) -> Path:
    return table_path.parent / f"{table_path.stem}.metadata.json"


def _validate_trial_table_contract_metadata(
    ctx: BehaviorContext,
    table_path: Path,
    df: pd.DataFrame,
) -> None:
    """Validate trial-table metadata contract (if present)."""
    from eeg_pipeline.utils.data.trial_table import validate_trial_table_contract

    meta_path = _trial_table_metadata_path(table_path)
    if not meta_path.exists():
        ctx.logger.warning("Trial table metadata missing: %s", meta_path)
        return
    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid trial table metadata JSON: {meta_path}") from exc

    if not isinstance(metadata, dict):
        raise ValueError(f"Invalid trial table metadata payload (not a dict): {meta_path}")

    errors = validate_trial_table_contract(df, metadata)
    if errors:
        raise ValueError(
            f"Trial table contract validation failed for {table_path.name}: "
            + "; ".join(errors)
        )


def _validate_feature_index_alignment(df: pd.DataFrame, name: str, base_index: pd.Index, logger: Any) -> None:
    """Validate that feature DataFrame index matches base index."""
    if not df.index.equals(base_index):
        expected_rows = len(base_index)
        message = (
            f"Feature index mismatch for {name}: expected alignment of "
            f"{expected_rows} rows."
        )
        logger.error(message)
        raise ValueError(message)


def _check_duplicate_columns(df: pd.DataFrame, context: str, logger: Any) -> None:
    """Check for duplicate columns and raise if found."""
    if df.columns.duplicated().any():
        duplicate_names = [str(c) for c in df.columns[df.columns.duplicated()].unique()]
        message = f"Duplicate feature columns {context}: {duplicate_names}"
        logger.error(message)
        raise ValueError(message)


def _rename_feature_columns_with_prefix(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Rename columns to include prefix if not already present."""
    rename_map = {
        col: col if str(col).startswith(prefix) else f"{prefix}{col}"
        for col in df.columns
    }
    if all(k == v for k, v in rename_map.items()):
        return df
    return df.rename(columns=rename_map)


def _check_cross_table_column_overlap(new_df: pd.DataFrame, existing_dfs: List[pd.DataFrame], logger: Any) -> None:
    """Check for column name overlap between new DataFrame and existing ones."""
    existing_columns = pd.Index([])
    for prev_df in existing_dfs:
        existing_columns = existing_columns.append(prev_df.columns)
    
    overlap = existing_columns.intersection(new_df.columns)
    if not overlap.empty:
        overlap_list = [str(c) for c in overlap.tolist()]
        message = f"Duplicate feature columns across tables: {overlap_list}"
        logger.error(message)
        raise ValueError(message)


def combine_features(ctx: BehaviorContext) -> pd.DataFrame:
    """Combine all feature tables into a single DataFrame with caching."""
    signature = _compute_feature_signature(ctx)
    if ctx._combined_features_df is not None and ctx._combined_features_signature == signature:
        return ctx._combined_features_df

    feature_dataframes = []
    base_index = None
    
    for name, df in ctx.iter_feature_tables():
        if not _is_dataframe_valid(df):
            continue
        
        if base_index is None:
            base_index = df.index
        else:
            _validate_feature_index_alignment(df, name, base_index, ctx.logger)
        
        prefix = f"{name}_"
        df_renamed = _rename_feature_columns_with_prefix(df, prefix)
        _check_duplicate_columns(df_renamed, f"within {name}", ctx.logger)
        
        if feature_dataframes:
            _check_cross_table_column_overlap(df_renamed, feature_dataframes, ctx.logger)
        
        feature_dataframes.append(df_renamed)

    combined = pd.concat(feature_dataframes, axis=1) if feature_dataframes else pd.DataFrame()
    if not combined.empty:
        _check_duplicate_columns(combined, "after combining", ctx.logger)
    
    ctx._combined_features_df = combined
    ctx._combined_features_signature = signature
    return combined


def _has_precomputed_change_scores(df: Optional[pd.DataFrame]) -> bool:
    """Check if DataFrame already has change score columns from feature pipeline."""
    if df is None or df.empty:
        return False
    return any("_change_" in str(c) for c in df.columns)


def _augment_dataframe_with_change_scores(df: Optional[pd.DataFrame], config: Any) -> Optional[pd.DataFrame]:
    """Add change score columns to a feature DataFrame if not already present.
    
    Skips computation if change scores were pre-computed in the feature pipeline.
    """
    if not _is_dataframe_valid(df):
        return df

    if _has_precomputed_change_scores(df):
        return df
    
    from eeg_pipeline.utils.analysis.stats.transforms import compute_change_features
    
    change_df = compute_change_features(df, config=config)
    if not _is_dataframe_valid(change_df):
        return df
    
    new_columns = [col for col in change_df.columns if col not in df.columns]
    if not new_columns:
        return df
    
    return pd.concat([df, change_df[new_columns]], axis=1)


def add_change_scores(ctx: BehaviorContext) -> None:
    """Compute and append change scores (active-baseline) once per context.
    
    Skips if change scores were pre-computed in the feature pipeline.
    """
    if ctx._change_scores_added or not ctx.compute_change_scores:
        return

    n_precomputed = sum(
        1 for df in [ctx.power_df, ctx.connectivity_df, ctx.aperiodic_df]
        if _has_precomputed_change_scores(df)
    )
    if n_precomputed > 0:
        ctx.logger.info("Using %d pre-computed change score tables from feature pipeline", n_precomputed)

    ctx.power_df = _augment_dataframe_with_change_scores(ctx.power_df, ctx.config)
    ctx.connectivity_df = _augment_dataframe_with_change_scores(ctx.connectivity_df, ctx.config)
    ctx.directed_connectivity_df = _augment_dataframe_with_change_scores(ctx.directed_connectivity_df, ctx.config)
    ctx.source_localization_df = _augment_dataframe_with_change_scores(ctx.source_localization_df, ctx.config)
    ctx.aperiodic_df = _augment_dataframe_with_change_scores(ctx.aperiodic_df, ctx.config)
    ctx.itpc_df = _augment_dataframe_with_change_scores(ctx.itpc_df, ctx.config)
    ctx.pac_df = _augment_dataframe_with_change_scores(ctx.pac_df, ctx.config)
    ctx.complexity_df = _augment_dataframe_with_change_scores(ctx.complexity_df, ctx.config)
    ctx._change_scores_added = True


def stage_load(ctx: BehaviorContext) -> bool:
    if not ctx.load_data():
        ctx.logger.warning("Failed to load data")
        return False

    _cache.load_manifest(ctx)

    ctx.logger.info(f"Loaded {ctx.n_trials} trials")
    return True


###################################################################
# Correlate Stage - Single Responsibility Components
###################################################################


@dataclass
class CorrelateDesign:
    """Design matrix components for correlation analysis."""
    df_trials: pd.DataFrame
    feature_cols: List[str]
    targets: List[str]
    cov_df: Optional[pd.DataFrame]
    temperature_series: Optional[pd.Series]
    run_col: str
    run_adjust_in_correlations: bool
    groups_for_perm: Optional[pd.Series]


def stage_correlate_design(ctx: BehaviorContext, config: Any) -> Optional[CorrelateDesign]:
    """Assemble design matrix: targets, covariates, feature columns.
    
    Single responsibility: Prepare all inputs needed for correlation computation.
    """
    df_trials = _load_trial_table_df(ctx)
    if _is_dataframe_valid(df_trials):
        ctx.logger.info(
            "Correlations design: loaded trial table (%d rows, %d cols)",
            df_trials.shape[0], df_trials.shape[1]
        )
    else:
        ctx.logger.warning("Correlations design: trial table missing; skipping.")
        return None

    primary_unit = str(
        get_config_value(ctx.config, "behavior_analysis.correlations.primary_unit", "trial") or "trial"
    ).strip().lower()
    allow_iid_trials = get_config_bool(ctx.config, "behavior_analysis.statistics.allow_iid_trials", False)
    perm_enabled = get_config_bool(ctx.config, "behavior_analysis.correlations.permutation.enabled", False)
    if primary_unit in {"trial", "trialwise"} and not perm_enabled and not allow_iid_trials:
        raise ValueError(
            "Trial-level correlations require a valid non-i.i.d inference method. "
            "Enable permutation testing (behavior_analysis.correlations.permutation.enabled=true) "
            "or use run-level aggregation (behavior_analysis.correlations.primary_unit=run_mean). "
            "Set behavior_analysis.statistics.allow_iid_trials=true to override (not recommended)."
        )

    feature_cols = _get_feature_columns(df_trials, ctx, "correlations")
    
    if not feature_cols:
        ctx.logger.warning("Correlations design: no feature columns after filtering.")
        return None

    prefer_pain_residual = get_config_bool(ctx.config, "behavior_analysis.correlations.prefer_pain_residual", True)
    default_targets = ["pain_residual", "rating", "temperature"] if prefer_pain_residual else ["rating", "temperature", "pain_residual"]
    targets = resolve_correlation_targets(
        ctx.config,
        logger=ctx.logger,
        default_targets=default_targets,
    )
    use_cv_resid = get_config_bool(ctx.config, "behavior_analysis.correlations.use_crossfit_pain_residual", False)
    if use_cv_resid and "pain_residual_cv" in df_trials.columns:
        has_explicit_targets = bool(
            get_config_value(ctx.config, "behavior_analysis.correlations.targets", None)
            or get_config_value(ctx.config, "behavior_analysis.targets", None)
        )
        if not has_explicit_targets:
            targets = ["pain_residual_cv", *[t for t in targets if t != "pain_residual_cv"]]
    targets = [t for t in targets if t in df_trials.columns]

    if not targets:
        ctx.logger.warning("Correlations design: no valid target columns found.")
        return None

    run_adjust_enabled = get_config_bool(ctx.config, "behavior_analysis.run_adjustment.enabled", False)
    run_col = str(get_config_value(ctx.config, "behavior_analysis.run_adjustment.column", "run_id") or "run_id").strip()
    if not run_col:
        run_col = "run_id"
    if primary_unit in {"run", "run_mean", "runmean", "run_level"} and run_col not in df_trials.columns:
        raise ValueError(
            f"Run-level correlations requested (primary_unit={primary_unit!r}) "
            f"but run column '{run_col}' is missing from the trial table."
        )
    run_adjust_in_correlations = bool(
        get_config_value(ctx.config, "behavior_analysis.run_adjustment.include_in_correlations", run_adjust_enabled)
    )
    max_run_dummies = get_config_int(ctx.config, "behavior_analysis.run_adjustment.max_dummies", 20)

    cov_parts = []
    if bool(getattr(config, "control_trial_order", True)):
        for c in ["trial_index_within_group", "trial_index"]:
            if c in df_trials.columns:
                cov_parts.append(pd.DataFrame({c: pd.to_numeric(df_trials[c], errors="coerce")}, index=df_trials.index))
                break
    if run_adjust_in_correlations and run_col in df_trials.columns:
        run_s = df_trials[run_col]
        n_levels = int(pd.Series(run_s).nunique(dropna=True))
        if n_levels > 1 and n_levels <= max(1, max_run_dummies + 1):
            run_dum = pd.get_dummies(run_s.astype("category"), prefix=run_col, drop_first=True)
            cov_parts.append(run_dum)
        elif n_levels > max_run_dummies + 1:
            ctx.logger.warning(
                "Correlations design: run adjustment requested but %s has %d levels (> max %d dummies); skipping.",
                run_col, n_levels, max_run_dummies,
            )
    cov_df = pd.concat(cov_parts, axis=1) if cov_parts else None

    temperature_series = None
    if bool(getattr(config, "control_temperature", True)) and "temperature" in df_trials.columns:
        temperature_series = pd.to_numeric(df_trials["temperature"], errors="coerce")

    groups_for_perm = None
    if getattr(ctx, "group_ids", None) is not None:
        groups_candidate = np.asarray(ctx.group_ids)
        if len(groups_candidate) == len(df_trials):
            groups_for_perm = groups_candidate
        else:
            ctx.logger.warning(
                "Correlations design: ignoring ctx.group_ids length=%d because trial table has %d rows.",
                len(groups_candidate),
                len(df_trials),
            )
    if groups_for_perm is None and run_col in df_trials.columns:
        groups_for_perm = df_trials[run_col].to_numpy()
    groups_for_perm = _sanitize_permutation_groups(
        groups_for_perm,
        ctx.logger,
        "Correlations",
    )
    if primary_unit in {"trial", "trialwise"} and not allow_iid_trials and perm_enabled and groups_for_perm is None:
        raise ValueError(
            "Trial-level correlations require grouped permutation labels for non-i.i.d trials. "
            "Provide behavior_analysis.run_adjustment.column in the trial table (or ctx.group_ids), "
            "or set behavior_analysis.statistics.allow_iid_trials=true to override (not recommended)."
        )
    groups_for_perm_series = (
        pd.Series(groups_for_perm, index=df_trials.index, name=run_col)
        if groups_for_perm is not None
        else None
    )

    ctx.logger.info(
        "Correlations design: %d features x %d targets, covariates=%s, temp_control=%s",
        len(feature_cols), len(targets),
        cov_df.shape[1] if cov_df is not None else 0,
        temperature_series is not None,
    )

    return CorrelateDesign(
        df_trials=df_trials,
        feature_cols=feature_cols,
        targets=targets,
        cov_df=cov_df,
        temperature_series=temperature_series,
        run_col=run_col,
        run_adjust_in_correlations=run_adjust_in_correlations,
        groups_for_perm=groups_for_perm_series,
    )


def _compute_single_effect_size(
    feat: str,
    target: str,
    df_trials: pd.DataFrame,
    cov_df: Optional[pd.DataFrame],
    temperature_series: Optional[pd.Series],
    run_col: str,
    run_adjust_in_correlations: bool,
    method: str,
    robust_method: Optional[str],
    method_label: str,
    min_samples: int,
    want_raw: bool,
    want_partial_cov: bool,
    want_partial_temp: bool,
    want_partial_cov_temp: bool,
    want_run_mean: bool,
    config: Any,
) -> Dict[str, Any]:
    """Compute effect size for a single feature-target pair (worker function for parallelization)."""
    from eeg_pipeline.utils.analysis.stats.correlation import safe_correlation
    from eeg_pipeline.utils.analysis.stats import compute_partial_correlations_with_cov_temp

    x = pd.to_numeric(df_trials[feat], errors="coerce")
    y = pd.to_numeric(df_trials[target], errors="coerce")
    x_arr = x.to_numpy(dtype=float)
    y_arr = y.to_numpy(dtype=float)
    finite_xy = np.isfinite(x_arr) & np.isfinite(y_arr)

    skip_reason = None
    if int(finite_xy.sum()) >= min_samples:
        feature_std = float(np.nanstd(x_arr[finite_xy], ddof=1))
        target_std = float(np.nanstd(y_arr[finite_xy], ddof=1))
        if feature_std <= CONSTANT_VARIANCE_THRESHOLD:
            skip_reason = "feature_constant"
        elif target_std <= CONSTANT_VARIANCE_THRESHOLD:
            skip_reason = "target_constant"

    r_raw, p_raw, n = safe_correlation(x_arr, y_arr, method, min_samples, robust_method=robust_method)

    rec: Dict[str, Any] = {
        "feature": str(feat),
        "feature_type": _cache.get_feature_type(str(feat), config),
        "band": _cache.get_feature_band(str(feat), config),
        "target": str(target),
        "method": method,
        "robust_method": robust_method,
        "method_label": method_label,
        "n": int(n),
        "r_raw": float(r_raw) if np.isfinite(r_raw) else np.nan if want_raw else np.nan,
        "p_raw": float(p_raw) if np.isfinite(p_raw) else np.nan if want_raw else np.nan,
        "r": float(r_raw) if np.isfinite(r_raw) else np.nan if want_raw else np.nan,
        "p": float(p_raw) if np.isfinite(p_raw) else np.nan if want_raw else np.nan,
        "p_value": float(p_raw) if np.isfinite(p_raw) else np.nan if want_raw else np.nan,
        "skip_reason": skip_reason,
        "run_adjustment_enabled": bool(run_adjust_in_correlations and run_col in df_trials.columns),
        "run_column": run_col if run_col in df_trials.columns else None,
    }

    temp_for_partial = temperature_series if (temperature_series is not None and target != "temperature") else None

    if want_partial_cov or want_partial_temp or want_partial_cov_temp:
        r_pc, p_pc, n_pc, r_pt, p_pt, n_pt, r_pct, p_pct, n_pct = compute_partial_correlations_with_cov_temp(
            roi_values=x,
            target_values=y,
            covariates_df=cov_df,
            temperature_series=temp_for_partial,
            method=method,
            context="trial_table",
            logger=None,
            min_samples=min_samples,
            config=config,
        )

        if want_partial_cov or want_partial_cov_temp:
            rec.update({
                "r_partial_cov": r_pc,
                "p_partial_cov": p_pc,
                "n_partial_cov": n_pc,
            })

        if want_partial_temp or want_partial_cov_temp:
            rec.update({
                "r_partial_temp": r_pt,
                "p_partial_temp": p_pt,
                "n_partial_temp": n_pt,
            })

        if want_partial_cov_temp:
            rec.update({
                "r_partial_cov_temp": r_pct,
                "p_partial_cov_temp": p_pct,
                "n_partial_cov_temp": n_pct,
            })

    if want_run_mean and run_col in df_trials.columns:
        df_run = pd.DataFrame({run_col: df_trials[run_col], "x": x, "y": y})
        run_means = df_run.groupby(run_col, dropna=True)[["x", "y"]].mean(numeric_only=True)
        r_run, p_run, n_run = safe_correlation(
            run_means["x"].to_numpy(dtype=float),
            run_means["y"].to_numpy(dtype=float),
            method,
            min_samples=3,
            robust_method=None,
        )
        rec.update(
            {
                "n_runs": int(n_run),
                "r_run_mean": float(r_run) if np.isfinite(r_run) else np.nan,
                "p_run_mean": float(p_run) if np.isfinite(p_run) else np.nan,
            }
        )

    return rec


def stage_correlate_effect_sizes(
    ctx: BehaviorContext,
    config: Any,
    design: CorrelateDesign,
) -> List[Dict[str, Any]]:
    """Compute raw and partial correlation effect sizes.
    
    Single responsibility: Compute effect sizes (r values) without p-value inference.
    Only computes requested correlation types to avoid unnecessary computation.
    Uses joblib parallelization for large feature sets.
    """
    if design is None:
        ctx.logger.warning("Correlations effect sizes: design missing; skipping.")
        return []

    from joblib import Parallel, delayed
    from eeg_pipeline.utils.parallel import get_n_jobs, _normalize_n_jobs

    method = getattr(config, "method", "spearman")
    robust_method = getattr(config, "robust_method", None)
    method_label = getattr(config, "method_label", "")
    min_samples = int(getattr(config, "min_samples", 10))

    correlation_types = get_config_value(
        ctx.config,
        "behavior_analysis.correlations.types",
        ["partial_cov_temp"]
    )
    if not isinstance(correlation_types, (list, tuple)):
        correlation_types = [correlation_types]

    want_raw = "raw" in correlation_types
    want_partial_cov = "partial_cov" in correlation_types
    want_partial_temp = "partial_temp" in correlation_types
    want_partial_cov_temp = "partial_cov_temp" in correlation_types
    primary_unit = str(
        get_config_value(ctx.config, "behavior_analysis.correlations.primary_unit", "trial") or "trial"
    ).strip().lower()
    want_run_mean = ("run_mean" in correlation_types) or (
        primary_unit in {"run", "run_mean", "runmean", "run_level"}
    )

    if robust_method not in (None, "", False):
        if want_partial_cov or want_partial_temp or want_partial_cov_temp:
            ctx.logger.info(
                "Correlations: robust_method=%s disables partial correlations; using raw only.",
                robust_method,
            )
        want_raw = True
        want_partial_cov = False
        want_partial_temp = False
        want_partial_cov_temp = False

    tasks = [
        (feat, target)
        for target in design.targets
        for feat in design.feature_cols
    ]
    n_tasks = len(tasks)
    n_jobs = get_n_jobs(ctx.config, default=-1, config_path="behavior_analysis.n_jobs")
    n_jobs_actual = _normalize_n_jobs(n_jobs)

    ctx.logger.info(
        "Correlations effect sizes: %d feature-target pairs, n_jobs=%d",
        n_tasks, n_jobs_actual,
    )

    if n_tasks == 0:
        return []

    if n_jobs_actual > 1 and n_tasks >= 100:
        records = Parallel(n_jobs=n_jobs_actual, backend="loky")(
            delayed(_compute_single_effect_size)(
                feat=feat,
                target=target,
                df_trials=design.df_trials,
                cov_df=design.cov_df,
                temperature_series=design.temperature_series,
                run_col=design.run_col,
                run_adjust_in_correlations=design.run_adjust_in_correlations,
                method=method,
                robust_method=robust_method,
                method_label=method_label,
                min_samples=min_samples,
                want_raw=want_raw,
                want_partial_cov=want_partial_cov,
                want_partial_temp=want_partial_temp,
                want_partial_cov_temp=want_partial_cov_temp,
                want_run_mean=want_run_mean,
                config=ctx.config,
            )
            for feat, target in tasks
        )
    else:
        records = [
            _compute_single_effect_size(
                feat=feat,
                target=target,
                df_trials=design.df_trials,
                cov_df=design.cov_df,
                temperature_series=design.temperature_series,
                run_col=design.run_col,
                run_adjust_in_correlations=design.run_adjust_in_correlations,
                method=method,
                robust_method=robust_method,
                method_label=method_label,
                min_samples=min_samples,
                want_raw=want_raw,
                want_partial_cov=want_partial_cov,
                want_partial_temp=want_partial_temp,
                want_partial_cov_temp=want_partial_cov_temp,
                want_run_mean=want_run_mean,
                config=ctx.config,
            )
            for feat, target in tasks
        ]

    ctx.logger.info("Correlations effect sizes: computed %d feature-target pairs", len(records))
    return records


def _compute_single_pvalue(
    rec: Dict[str, Any],
    df_trials: pd.DataFrame,
    df_index: pd.Index,
    cov_df: Optional[pd.DataFrame],
    temperature_series: Optional[pd.Series],
    groups_for_perm: Optional[pd.Series],
    method: str,
    robust_method: Optional[str],
    n_perm: int,
    perm_scheme: str,
    rng_seed: int,
    config: Any,
    perm_ok_robust: bool,
) -> Dict[str, Any]:
    """Compute permutation p-values for a single record (worker function for parallelization)."""
    from eeg_pipeline.utils.analysis.stats.permutation import compute_permutation_pvalues_with_cov_temp

    feat = rec["feature"]
    target = rec["target"]
    r_raw = rec.get("r_raw", np.nan)
    n = rec.get("n", 0)

    result = rec.copy()

    if not (np.isfinite(r_raw) and int(n) > 0):
        result.update({
            "n_permutations": int(n_perm),
            "p_perm_raw": np.nan,
            "p_perm_partial_cov": np.nan,
            "p_perm_partial_temp": np.nan,
            "p_perm_partial_cov_temp": np.nan,
        })
        return result

    rng = np.random.default_rng(rng_seed)
    x = pd.to_numeric(df_trials[feat], errors="coerce")
    y = pd.to_numeric(df_trials[target], errors="coerce")

    if perm_ok_robust:
        from eeg_pipeline.utils.analysis.stats.correlation import compute_robust_correlation
        from eeg_pipeline.utils.analysis.stats.permutation import permute_within_groups

        x_vec = x.to_numpy(dtype=float)
        y_vec = y.to_numpy(dtype=float)
        valid = np.isfinite(x_vec) & np.isfinite(y_vec)

        if valid.sum() < 4:
            p_perm_raw = np.nan
        else:
            x_v = x_vec[valid]
            y_v = y_vec[valid]
            groups_v = np.asarray(groups_for_perm)[valid] if groups_for_perm is not None else None

            r_obs, _ = compute_robust_correlation(x_v, y_v, method=str(robust_method).strip().lower())
            if not np.isfinite(r_obs):
                p_perm_raw = np.nan
            else:
                extreme = 0
                for _ in range(int(n_perm)):
                    perm_idx = permute_within_groups(len(y_v), rng, groups_v, scheme=perm_scheme)
                    y_perm = y_v[perm_idx]
                    r_perm, _ = compute_robust_correlation(x_v, y_perm, method=str(robust_method).strip().lower())
                    if np.isfinite(r_perm) and abs(r_perm) >= abs(r_obs):
                        extreme += 1
                p_perm_raw = float((extreme + 1) / (int(n_perm) + 1))

        result.update({
            "n_permutations": int(n_perm),
            "p_perm_raw": float(p_perm_raw) if np.isfinite(p_perm_raw) else np.nan,
            "p_perm_partial_cov": np.nan,
            "p_perm_partial_temp": np.nan,
            "p_perm_partial_cov_temp": np.nan,
        })
    else:
        temp_for_partial = temperature_series if (temperature_series is not None and target != "temperature") else None
        p_perm, p_perm_cov, p_perm_temp, p_perm_cov_temp = compute_permutation_pvalues_with_cov_temp(
            x_aligned=pd.Series(x.to_numpy(dtype=float), index=df_index),
            y_aligned=pd.Series(y.to_numpy(dtype=float), index=df_index),
            covariates_df=cov_df,
            temp_series=temp_for_partial,
            method=method.strip().lower(),
            n_perm=n_perm,
            n_eff=int(n),
            rng=rng,
            config=config,
            groups=groups_for_perm,
        )
        result.update({
            "n_permutations": int(n_perm),
            "p_perm_raw": float(p_perm) if np.isfinite(p_perm) else np.nan,
            "p_perm_partial_cov": float(p_perm_cov) if np.isfinite(p_perm_cov) else np.nan,
            "p_perm_partial_temp": float(p_perm_temp) if np.isfinite(p_perm_temp) else np.nan,
            "p_perm_partial_cov_temp": float(p_perm_cov_temp) if np.isfinite(p_perm_cov_temp) else np.nan,
        })

    return result


def stage_correlate_pvalues(
    ctx: BehaviorContext,
    config: Any,
    design: CorrelateDesign,
    records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Compute permutation p-values for correlations.
    
    Single responsibility: Add permutation-based p-values to existing effect size records.
    Uses joblib parallelization for large record sets.
    """
    if not records or not isinstance(records, list):
        ctx.logger.warning("Correlations pvalues: no valid records; skipping.")
        return []

    from joblib import Parallel, delayed
    from eeg_pipeline.utils.parallel import get_n_jobs, _normalize_n_jobs

    method = getattr(config, "method", "spearman")
    robust_method = getattr(config, "robust_method", None)
    perm_enabled = get_config_bool(ctx.config, "behavior_analysis.correlations.permutation.enabled", False)
    n_perm = get_config_int(ctx.config, "behavior_analysis.correlations.permutation.n_permutations", ctx.n_perm or 0)
    perm_scheme = str(get_config_value(ctx.config, "behavior_analysis.permutation.scheme", "shuffle") or "shuffle").strip().lower()

    perm_ok_standard = (
        perm_enabled
        and n_perm > 0
        and (robust_method in (None, "", False))
        and isinstance(method, str)
        and method.strip().lower() in {"spearman", "pearson"}
    )
    perm_ok_robust = (
        perm_enabled
        and n_perm > 0
        and (robust_method not in (None, "", False))
        and isinstance(robust_method, str)
        and robust_method.strip().lower() in {"percentage_bend", "winsorized", "shepherd"}
    )

    if not (perm_ok_standard or perm_ok_robust):
        if perm_enabled and robust_method not in (None, "", False):
            ctx.logger.debug("Correlations pvalues: permutation disabled for robust_method=%s", robust_method)
        for rec in records:
            rec.update({
                "n_permutations": int(n_perm) if perm_enabled else 0,
                "p_perm_raw": np.nan,
                "p_perm_partial_cov": np.nan,
                "p_perm_partial_temp": np.nan,
                "p_perm_partial_cov_temp": np.nan,
            })
        return records

    base_seed = 42 if ctx.rng is None else int(ctx.rng.integers(0, 2**31))
    n_records = len(records)
    n_jobs = get_n_jobs(ctx.config, default=-1, config_path="behavior_analysis.n_jobs")
    n_jobs_actual = _normalize_n_jobs(n_jobs)

    ctx.logger.info(
        "Correlations pvalues: %d records, n_perm=%d, n_jobs=%d",
        n_records, n_perm, n_jobs_actual,
    )

    if n_jobs_actual > 1 and n_records >= 100:
        updated_records = Parallel(n_jobs=n_jobs_actual, backend="loky")(
            delayed(_compute_single_pvalue)(
                rec=rec,
                df_trials=design.df_trials,
                df_index=design.df_trials.index,
                cov_df=design.cov_df,
                temperature_series=design.temperature_series,
                groups_for_perm=design.groups_for_perm,
                method=method,
                robust_method=robust_method,
                n_perm=n_perm,
                perm_scheme=perm_scheme,
                rng_seed=base_seed + i,
                config=ctx.config,
                perm_ok_robust=perm_ok_robust,
            )
            for i, rec in enumerate(records)
        )
    else:
        updated_records = [
            _compute_single_pvalue(
                rec=rec,
                df_trials=design.df_trials,
                df_index=design.df_trials.index,
                cov_df=design.cov_df,
                temperature_series=design.temperature_series,
                groups_for_perm=design.groups_for_perm,
                method=method,
                robust_method=robust_method,
                n_perm=n_perm,
                perm_scheme=perm_scheme,
                rng_seed=base_seed + i,
                config=ctx.config,
                perm_ok_robust=perm_ok_robust,
            )
            for i, rec in enumerate(records)
        ]

    n_computed = sum(1 for r in updated_records if np.isfinite(r.get("p_perm_raw", np.nan)))
    ctx.logger.info("Correlations pvalues: computed %d permutation tests (n_perm=%d)", n_computed, n_perm)
    return updated_records


def stage_correlate_primary_selection(
    ctx: BehaviorContext,
    config: Any,
    design: CorrelateDesign,
    records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Select primary p-value and effect size for each test.
    
    Single responsibility: Determine which p-value column to use for FDR.
    """
    if not records or not isinstance(records, list):
        ctx.logger.warning("Correlations primary selection: no valid records; skipping.")
        return []

    p_primary_mode = str(get_config_value(ctx.config, "behavior_analysis.correlations.p_primary_mode", "perm_if_available")).strip().lower()
    primary_unit = str(get_config_value(ctx.config, "behavior_analysis.correlations.primary_unit", "trial")).strip().lower()
    use_run_unit = primary_unit in {"run", "run_mean", "runmean", "run_level"}

    for rec in records:
        target = rec.get("target", "")
        p_kind = "p_raw"
        p_primary = rec["p_raw"]
        r_primary = rec["r_raw"]
        src = "raw"

        robust_method = rec.get("robust_method", None)
        if robust_method not in (None, "", False):
            p_kind = "p_raw"
            p_primary = rec.get("p_raw", np.nan)
            r_primary = rec.get("r_raw", np.nan)
            src = "raw_robust"
        elif use_run_unit:
            # Never downgrade run-level inference to trial-level p-values.
            p_kind = "p_run_mean"
            p_primary = rec.get("p_run_mean", np.nan)
            r_primary = rec.get("r_run_mean", np.nan)
            src = "run_mean" if pd.notna(p_primary) else "run_mean_unavailable"
        else:
            want_partial_cov = design.cov_df is not None and not design.cov_df.empty
            want_partial_temp = bool(getattr(config, "control_temperature", True)) and target != "temperature" and design.temperature_series is not None

            if want_partial_temp and want_partial_cov:
                p_kind = "p_partial_cov_temp"
                p_primary = rec.get("p_partial_cov_temp", np.nan)
                r_primary = rec.get("r_partial_cov_temp", np.nan)
                src = "partial_cov_temp"
            elif want_partial_temp:
                p_kind = "p_partial_temp"
                p_primary = rec.get("p_partial_temp", np.nan)
                r_primary = rec.get("r_partial_temp", np.nan)
                src = "partial_temp"
            elif want_partial_cov:
                p_kind = "p_partial_cov"
                p_primary = rec.get("p_partial_cov", np.nan)
                r_primary = rec.get("r_partial_cov", np.nan)
                src = "partial_cov"

            if p_primary_mode in {"perm", "permutation", "perm_if_available", "permutation_if_available"}:
                perm_map = {
                    "p_raw": "p_perm_raw",
                    "p_partial_cov": "p_perm_partial_cov",
                    "p_partial_temp": "p_perm_partial_temp",
                    "p_partial_cov_temp": "p_perm_partial_cov_temp",
                }
                perm_key = perm_map.get(p_kind)
                if perm_key and pd.notna(rec.get(perm_key, np.nan)):
                    p_kind = perm_key
                    p_primary = rec.get(perm_key, np.nan)
                    src = f"{src}_perm"

            # If the selected primary statistic is unavailable, keep it missing.
            # Do not silently downgrade controlled/permutation inference to raw.
            if not (pd.notna(p_primary) and np.isfinite(float(p_primary))):
                p_primary = np.nan
                r_primary = np.nan
                src = f"{src}_missing"

        rec["p_kind_primary"] = p_kind
        rec["p_primary"] = p_primary
        rec["r_primary"] = r_primary
        rec["p_primary_source"] = src

    return records


def _compute_unified_fdr(
    ctx: BehaviorContext,
    config: Any,
    df: pd.DataFrame,
    p_col: str = "p_primary",
    family_cols: Optional[List[str]] = None,
    analysis_type: str = "correlations",
) -> pd.DataFrame:
    """Compute unified FDR corrections (within-family, hierarchical, global) in one pass.

    Returns DataFrame with added columns:
    - p_fdr: Within-family FDR-corrected p-values
    - q_global: Global FDR-corrected p-values across all tests
    - fdr_family: Family identifier for hierarchical correction

    Results are cached for downstream stages.
    """
    from eeg_pipeline.utils.analysis.stats.fdr import fdr_bh

    if df.empty or p_col not in df.columns:
        return df

    df = df.copy()
    fdr_alpha = float(getattr(config, "fdr_alpha", 0.05))
    hierarchical_fdr = get_config_bool(ctx.config, "behavior_analysis.statistics.hierarchical_fdr", True)

    # Determine family columns (filter to only those that exist in df)
    if family_cols is None:
        family_cols = ["feature_type", "band", "target", "analysis_kind"]
    family_cols = [col for col in family_cols if col in df.columns]

    # Convert p-values to numeric
    p_vals = pd.to_numeric(df[p_col], errors="coerce").to_numpy()

    # Global FDR (across all tests)
    df["q_global"] = fdr_bh(p_vals, alpha=fdr_alpha, config=ctx.config)

    # Hierarchical/within-family FDR
    if hierarchical_fdr and family_cols:
        df["fdr_family"] = df[family_cols].astype(str).agg("_".join, axis=1)
        df["p_fdr"] = np.nan

        family_stats = []
        for family_id, family_group in df.groupby("fdr_family"):
            family_p_vals = pd.to_numeric(family_group[p_col], errors="coerce").to_numpy()
            family_p_fdr = fdr_bh(family_p_vals, alpha=fdr_alpha, config=ctx.config)
            df.loc[family_group.index, "p_fdr"] = family_p_fdr

            family_stats.append({
                "family": family_id,
                "n_tests": len(family_p_vals),
                "n_significant": int((family_p_fdr < fdr_alpha).sum()),
            })

        # Store in context and cache
        fdr_metadata = {
            "family_columns": family_cols,
            "n_families": len(family_stats),
            "family_stats": family_stats,
            "hierarchical": True,
            "analysis_type": analysis_type,
            "n_total_tests": len(df),
            "n_sig_global": int((df["q_global"] < fdr_alpha).sum()),
            "n_sig_within": int((df["p_fdr"] < fdr_alpha).sum()),
        }
        ctx.data_qc["fdr_family_structure"] = fdr_metadata

        # Update cache
        cached_fdr = _cache.get_fdr_results() or {}
        cached_fdr[analysis_type] = {
            "df": df,
            "metadata": fdr_metadata,
        }
        _cache.set_fdr_results(cached_fdr)

        n_sig_total = int((df["p_fdr"] < fdr_alpha).sum())
        ctx.logger.info(
            "Unified FDR [%s]: %d/%d significant within families, %d/%d globally (alpha=%.2f)",
            analysis_type, n_sig_total, len(df), fdr_metadata["n_sig_global"], len(df), fdr_alpha
        )
    else:
        # Flat FDR (no family structure)
        df["p_fdr"] = df["q_global"]
        df["fdr_family"] = "all"

        fdr_metadata = {
            "family_columns": [],
            "n_families": 1,
            "hierarchical": False,
            "analysis_type": analysis_type,
            "n_total_tests": len(df),
            "n_sig_global": int((df["q_global"] < fdr_alpha).sum()),
            "n_sig_within": int((df["p_fdr"] < fdr_alpha).sum()),
        }
        ctx.data_qc["fdr_family_structure"] = fdr_metadata

        # Update cache
        cached_fdr = _cache.get_fdr_results() or {}
        cached_fdr[analysis_type] = {
            "df": df,
            "metadata": fdr_metadata,
        }
        _cache.set_fdr_results(cached_fdr)

        n_sig = int((df["p_fdr"] < fdr_alpha).sum())
        ctx.logger.info(
            "Unified FDR [%s]: %d/%d significant (flat, alpha=%.2f)",
            analysis_type, n_sig, len(df), fdr_alpha
        )

    return df


def stage_correlate_fdr(
    ctx: BehaviorContext,
    config: Any,
    records: List[Dict[str, Any]],
) -> pd.DataFrame:
    """Apply unified FDR correction with explicit family structure.

    Family structure: feature_type × band × target × analysis_kind
    This is critical for EEG feature banks to control FWER properly.

    Single responsibility: FDR correction on p_primary column with family awareness.
    Now uses unified FDR computation for consistency across all stages.
    """
    if records is None:
        raise ValueError("Correlations FDR: records is None")
    if not isinstance(records, list):
        raise TypeError(f"Correlations FDR: records must be a list, got {type(records)!r}")
    if not records:
        ctx.logger.info("Correlations FDR: no records; skipping.")
        return pd.DataFrame()

    try:
        corr_df = pd.DataFrame(records)
    except Exception as exc:
        raise ValueError("Correlations FDR: failed to build DataFrame from records") from exc

    if corr_df.empty:
        return corr_df

    if "p_primary" not in corr_df.columns:
        ctx.logger.error("Missing 'p_primary' column. Ensure 'correlate_primary_selection' stage runs before 'correlate_fdr'.")
        raise KeyError("p_primary")

    # Add analysis_kind if missing
    if "analysis_kind" not in corr_df.columns:
        corr_df["analysis_kind"] = "correlation"

    return _compute_unified_fdr(
        ctx,
        config,
        corr_df,
        p_col="p_primary",
        family_cols=["feature_type", "band", "target", "analysis_kind"],
        analysis_type="correlations",
    )


def stage_correlate(ctx: BehaviorContext, config: Any) -> pd.DataFrame:
    """Backward-compatible correlations stage (composed).
    
    BehaviorPipeline expects a single `stage_correlate(ctx, config)` entrypoint.
    Internally, we keep single-responsibility sub-stages:
    - stage_correlate_design
    - stage_correlate_effect_sizes
    - stage_correlate_pvalues
    - stage_correlate_primary_selection
    - stage_correlate_fdr
    """
    design = stage_correlate_design(ctx, config)
    if design is None:
        return pd.DataFrame()

    records = stage_correlate_effect_sizes(ctx, config, design)
    records = stage_correlate_pvalues(ctx, config, design, records)
    records = stage_correlate_primary_selection(ctx, config, design, records)
    return stage_correlate_fdr(ctx, config, records)


def _sanitize_permutation_groups(
    groups: Optional[np.ndarray],
    logger: Any,
    context: str,
    *,
    min_group_size: int = 2,
) -> Optional[np.ndarray]:
    """Return groups if valid for grouped permutation, otherwise None."""
    if groups is None:
        return None
    groups_array = np.asarray(groups)
    if groups_array.size == 0:
        return None
    unique_groups, counts = np.unique(groups_array, return_counts=True)
    small_group_count = int((counts < int(min_group_size)).sum())
    if small_group_count > 0:
        logger.warning(
            "%s: disabling grouped permutation because %d/%d groups had fewer than %d samples.",
            context,
            small_group_count,
            len(unique_groups),
            int(min_group_size),
        )
        return None
    return groups_array


def stage_pain_sensitivity(ctx: BehaviorContext, config: Any) -> pd.DataFrame:
    """Compute pain sensitivity correlations (independent stage)."""
    from eeg_pipeline.analysis.behavior.api import run_pain_sensitivity_correlations

    df_trials = _load_trial_table_df(ctx)
    if not _is_dataframe_valid(df_trials):
        ctx.logger.warning("Pain sensitivity: trial table missing; skipping.")
        return pd.DataFrame()

    required_columns = {"temperature", "rating"}
    missing_columns = required_columns - set(df_trials.columns)
    if missing_columns:
        ctx.logger.warning(
            "Pain sensitivity: requires %s columns; missing: %s. Skipping.",
            required_columns, missing_columns
        )
        return pd.DataFrame()

    method = getattr(config, "method", "spearman")
    primary_unit = str(
        get_config_value(ctx.config, "behavior_analysis.pain_sensitivity.primary_unit", "trial") or "trial"
    ).strip().lower()
    use_run_unit = primary_unit in {"run", "run_mean", "runmean", "run_level"}
    allow_iid_trials = get_config_bool(ctx.config, "behavior_analysis.statistics.allow_iid_trials", False)
    n_perm = get_config_int(
        ctx.config,
        "behavior_analysis.pain_sensitivity.n_permutations",
        get_config_int(ctx.config, "behavior_analysis.statistics.n_permutations", 0),
    )
    p_primary_mode = str(
        get_config_value(ctx.config, "behavior_analysis.pain_sensitivity.p_primary_mode", "perm_if_available")
        or "perm_if_available"
    ).strip().lower()
    run_col = str(get_config_value(ctx.config, "behavior_analysis.run_adjustment.column", "run_id") or "run_id").strip()
    perm_scheme = str(get_config_value(ctx.config, "behavior_analysis.permutation.scheme", "shuffle") or "shuffle").strip().lower()
    if use_run_unit and run_col not in df_trials.columns:
        raise ValueError(
            f"Run-level pain sensitivity requested but run column '{run_col}' is missing from trial table."
        )
    if primary_unit in {"trial", "trialwise"} and not allow_iid_trials and n_perm <= 0:
        raise ValueError(
            "Trial-level pain sensitivity requires a valid non-i.i.d inference method. "
            "Set behavior_analysis.pain_sensitivity.n_permutations > 0, "
            "use run-level aggregation (behavior_analysis.pain_sensitivity.primary_unit=run_mean), "
            "or set behavior_analysis.statistics.allow_iid_trials=true to override (not recommended)."
        )

    if use_run_unit:
        ctx.logger.info("Pain sensitivity: aggregating to run-level (primary_unit=%s)", primary_unit)
        psi_feature_cols_run = _get_feature_columns(df_trials, ctx, "pain_sensitivity")
        agg_cols = [c for c in (psi_feature_cols_run + ["rating", "temperature"]) if c in df_trials.columns]
        if not agg_cols:
            ctx.logger.warning("Pain sensitivity: no aggregatable columns found; skipping.")
            return pd.DataFrame()
        df_trials = df_trials.groupby(run_col)[agg_cols].mean(numeric_only=True).reset_index()

    robust_method_cfg = get_config_value(ctx.config, "behavior_analysis.robust_correlation", None)
    if robust_method_cfg is not None:
        robust_method_cfg = str(robust_method_cfg).strip().lower() or None

    psi_feature_cols = _get_feature_columns(df_trials, ctx, "pain_sensitivity")

    if not psi_feature_cols:
        ctx.logger.warning("Pain sensitivity: no feature columns found; skipping.")
        return pd.DataFrame()

    ctx.logger.info("Pain sensitivity: analyzing %d features", len(psi_feature_cols))
    psi_features = df_trials[psi_feature_cols].copy()
    groups_for_perm = None
    if getattr(ctx, "group_ids", None) is not None:
        groups_candidate = np.asarray(ctx.group_ids)
        if len(groups_candidate) == len(df_trials):
            groups_for_perm = groups_candidate
        else:
            ctx.logger.warning(
                "Pain sensitivity: ignoring ctx.group_ids length=%d because trial table has %d rows.",
                len(groups_candidate),
                len(df_trials),
            )
    if groups_for_perm is None and run_col in df_trials.columns:
        groups_for_perm = df_trials[run_col].to_numpy()
    groups_for_perm = _sanitize_permutation_groups(
        groups_for_perm,
        ctx.logger,
        "Pain sensitivity",
    )
    if primary_unit in {"trial", "trialwise"} and not allow_iid_trials and groups_for_perm is None:
        raise ValueError(
            "Trial-level pain sensitivity requires grouped permutation labels for non-i.i.d trials. "
            "Provide behavior_analysis.run_adjustment.column in the trial table (or ctx.group_ids), "
            "or set behavior_analysis.statistics.allow_iid_trials=true to override (not recommended)."
        )
    psi_df = run_pain_sensitivity_correlations(
        features_df=psi_features,
        ratings=pd.to_numeric(df_trials["rating"], errors="coerce"),
        temperatures=pd.to_numeric(df_trials["temperature"], errors="coerce"),
        method=method,
        robust_method=robust_method_cfg,
        min_samples=int(getattr(config, "min_samples", 10)),
        logger=ctx.logger,
        config=ctx.config,
        n_perm=n_perm,
        groups=groups_for_perm,
        permutation_scheme=perm_scheme,
        p_primary_mode=p_primary_mode,
        rng=getattr(ctx, "rng", None),
    )

    if _is_dataframe_valid(psi_df):
        # Add analysis_kind for FDR
        if "analysis_kind" not in psi_df.columns:
            psi_df["analysis_kind"] = "pain_sensitivity"

        # Preserve p_primary from run_pain_sensitivity_correlations when available
        # (e.g., permutation-aware primary selection).
        if "p_primary" in psi_df.columns:
            psi_df["p_primary"] = pd.to_numeric(psi_df["p_primary"], errors="coerce")
        else:
            p_column = next((col for col in ["p_psi", "p_value", "p"] if col in psi_df.columns), None)
            if p_column:
                psi_df["p_primary"] = pd.to_numeric(psi_df[p_column], errors="coerce")

        if "p_raw" not in psi_df.columns:
            p_raw_col = next((col for col in ["p_psi", "p_value", "p"] if col in psi_df.columns), None)
            if p_raw_col:
                psi_df["p_raw"] = pd.to_numeric(psi_df[p_raw_col], errors="coerce")

        if "p_primary" in psi_df.columns:
            # Use unified FDR
            psi_df = _compute_unified_fdr(
                ctx,
                config,
                psi_df,
                p_col="p_primary",
                family_cols=["feature_type", "band", "analysis_kind"],
                analysis_type="pain_sensitivity",
            )

    return psi_df if _is_dataframe_valid(psi_df) else pd.DataFrame()



def _feature_suffix_from_context(ctx: BehaviorContext) -> str:
    feature_files = ctx.selected_feature_files or ctx.feature_categories or []
    return "_" + "_".join(sorted(str(x) for x in feature_files)) if feature_files else ""


def _filter_feature_cols_by_band(
    feature_cols: List[str],
    ctx: BehaviorContext,
) -> List[str]:
    """Filter feature columns to only include user-selected bands.
    
    If ctx.selected_bands is None or empty, returns all feature_cols unchanged.
    Otherwise, filters to keep only columns whose parsed 'band' matches one of
    the selected bands. Columns that don't have a band (e.g., ERP, complexity)
    are kept by default.
    """
    if not ctx.selected_bands:
        return feature_cols
    
    from eeg_pipeline.domain.features.naming import NamingSchema
    prefixes = sorted(FEATURE_COLUMN_PREFIXES, key=len, reverse=True)
    
    selected = set(b.lower() for b in ctx.selected_bands)
    filtered: List[str] = []
    
    for col in feature_cols:
        col_str = str(col)
        parsed = NamingSchema.parse(col_str)
        if not parsed.get("valid"):
            # Try stripping the feature-table prefix used in the trial table
            matched_prefix = next((p for p in prefixes if col_str.startswith(p)), None)
            if not matched_prefix:
                raise ValueError(
                    f"Band filter: cannot parse feature column {col_str!r} "
                    f"(selected_bands={sorted(selected)})"
                )

            candidate = col_str[len(matched_prefix):]
            parsed2 = NamingSchema.parse(candidate)
            if not parsed2.get("valid"):
                raise ValueError(
                    f"Band filter: cannot parse feature column {col_str!r} "
                    f"after stripping prefix {matched_prefix!r} -> {candidate!r} "
                    f"(selected_bands={sorted(selected)})"
                )
            parsed = parsed2

        band = parsed.get("band")
        if not band:
            # Features without bands (ERP, complexity, etc.) - keep them
            filtered.append(col)
        elif str(band).lower() in selected:
            # Band matches user selection
            filtered.append(col)
        # else: band doesn't match, exclude
    
    if len(filtered) < len(feature_cols):
        ctx.logger.info(
            "Band filter: kept %d/%d features for bands: %s",
            len(filtered),
            len(feature_cols),
            ", ".join(sorted(selected)),
        )
    
    return filtered


def _filter_feature_cols_for_computation(
    feature_cols: List[str],
    computation_name: str,
    ctx: BehaviorContext,
) -> List[str]:
    """Filter feature columns based on per-computation feature selection."""
    if not ctx.computation_features or computation_name not in ctx.computation_features:
        return feature_cols
    
    selected_features = ctx.computation_features[computation_name]
    if not selected_features:
        return feature_cols

    allowed_prefixes = tuple(
        CATEGORY_PREFIX_MAP[cat] for cat in selected_features if cat in CATEGORY_PREFIX_MAP
    )
    if not allowed_prefixes:
        ctx.logger.warning(
            f"Computation '{computation_name}' has feature filter {selected_features} but no matching prefixes found. Using all features."
        )
        return feature_cols

    filtered = [c for c in feature_cols if str(c).startswith(allowed_prefixes)]
    
    if len(filtered) < len(feature_cols):
        ctx.logger.info(
            "%s: filtered features to %s (%d/%d kept)",
            computation_name, selected_features, len(filtered), len(feature_cols)
        )
        
    return filtered


def _primary_unit_for_computation(ctx: BehaviorContext, computation_name: Optional[str]) -> str:
    mapping = {
        "correlations": "behavior_analysis.correlations.primary_unit",
        "regression": "behavior_analysis.regression.primary_unit",
        "condition": "behavior_analysis.condition.primary_unit",
        "pain_sensitivity": "behavior_analysis.pain_sensitivity.primary_unit",
        "condition_window_comparison": "behavior_analysis.condition.window_comparison.primary_unit",
    }
    key = mapping.get(str(computation_name or "").strip().lower(), None)
    if key is None:
        key = "behavior_analysis.primary_unit"
    primary_unit = str(get_config_value(ctx.config, key, "trial") or "trial").strip().lower()
    return primary_unit


def _filter_feature_cols_by_provenance(
    feature_cols: List[str],
    ctx: BehaviorContext,
    computation_name: Optional[str] = None,
) -> List[str]:
    """Exclude non-i.i.d./broadcast features when performing trial-wise analyses."""
    if not feature_cols:
        return feature_cols

    primary_unit = _primary_unit_for_computation(ctx, computation_name)
    is_trial_unit = primary_unit in {"trial", "trial_level", "trialwise"}
    if not is_trial_unit:
        return feature_cols

    enabled = bool(
        get_config_value(
            ctx.config,
            "behavior_analysis.features.exclude_non_trialwise_features",
            True,
        )
    )
    if not enabled:
        return feature_cols

    from eeg_pipeline.domain.features.naming import infer_feature_provenance

    manifests = getattr(ctx, "feature_manifests", {}) or {}
    dropped: List[str] = []
    kept: List[str] = []

    prefixes = sorted(FEATURE_COLUMN_PREFIXES, key=len, reverse=True)

    for col in feature_cols:
        col_str = str(col)
        matched_prefix = next((p for p in prefixes if col_str.startswith(p)), None)
        if matched_prefix is None:
            kept.append(col_str)
            continue

        table_key = matched_prefix.rstrip("_")
        raw_name = col_str[len(matched_prefix) :]

        manifest = manifests.get(table_key) or {}
        prov_cols = (manifest.get("provenance") or {}).get("columns") or {}

        # Some feature tables already use the same prefix as the table key
        # (e.g., itpc_*). In those cases, the manifest uses the full column name.
        props = prov_cols.get(raw_name)
        if props is None:
            props = prov_cols.get(col_str)

        if props is None:
            inferred = infer_feature_provenance(
                feature_columns=[raw_name],
                config=ctx.config,
                df_attrs={},
            )
            props = (inferred.get("columns") or {}).get(raw_name, {})
            if not props:
                inferred = infer_feature_provenance(
                    feature_columns=[col_str],
                    config=ctx.config,
                    df_attrs={},
                )
                props = (inferred.get("columns") or {}).get(col_str, {})

        trialwise_valid = bool(props.get("trialwise_valid", True))
        broadcasted = bool(props.get("broadcasted", False))
        if (not trialwise_valid) or broadcasted:
            dropped.append(col_str)
        else:
            kept.append(col_str)

    if dropped and ctx.logger is not None:
        ctx.logger.warning(
            "%s: excluded %d/%d non-trialwise (broadcast/cross-trial) features because primary_unit='trial'. Examples=%s",
            computation_name or "analysis",
            len(dropped),
            len(feature_cols),
            ",".join(dropped[:5]),
        )

    return kept


@dataclass
class TrialTableResult:
    """Result from compute_trial_table."""
    df: pd.DataFrame
    metadata: Dict[str, Any]


def compute_trial_table(ctx: BehaviorContext, config: Any) -> Optional[TrialTableResult]:
    """Build the canonical per-trial analysis table (compute only, no I/O)."""
    from eeg_pipeline.utils.data.trial_table import build_subject_trial_table

    result = build_subject_trial_table(ctx)
    return TrialTableResult(df=result.df, metadata=result.metadata)


def write_trial_table(ctx: BehaviorContext, result: TrialTableResult) -> Path:
    """Write trial table and metadata to disk.
    
    Single responsibility: Persist trial table artifacts.
    """
    from eeg_pipeline.utils.data.trial_table import save_trial_table

    fmt = str(get_config_value(ctx.config, "behavior_analysis.trial_table.format", "tsv")).strip().lower()
    suffix = _feature_suffix_from_context(ctx)
    fname = f"trials{suffix}"
    out_dir = _get_stats_subfolder(ctx, "trial_table")
    out_ext = ".parquet" if fmt == "parquet" else ".tsv"
    out_path = out_dir / f"{fname}{out_ext}"

    # Create a simple object with df and metadata for save_trial_table
    class _TableWrapper:
        def __init__(self, df, metadata):
            self.df = df
            self.metadata = metadata

    save_trial_table(_TableWrapper(result.df, result.metadata), out_path, format=fmt)

    if ctx.also_save_csv:
        from eeg_pipeline.infra.tsv import write_csv
        csv_path = out_dir / f"{fname}.csv"
        write_csv(result.df, csv_path, index=False)
        ctx.logger.info("Also saved trial table as CSV: %s/%s", out_dir.name, csv_path.name)

    meta_path = out_dir / f"{fname}.metadata.json"
    _write_metadata_file(meta_path, result.metadata)
    ctx.logger.info("Saved trial table: %s/%s (%d rows, %d cols)", out_dir.name, out_path.name, len(result.df), result.df.shape[1])

    _cache._trial_table_df = result.df
    _cache._trial_table_path = out_path

    return out_path


def _try_reuse_cached_trial_table(
    ctx: BehaviorContext,
    *,
    input_hash: str,
) -> Optional[Path]:
    """Reuse existing trial-table artifact when input hash is unchanged."""
    fmt = str(get_config_value(ctx.config, "behavior_analysis.trial_table.format", "tsv")).strip().lower()
    suffix = _feature_suffix_from_context(ctx)
    fname = f"trials{suffix}"
    out_dir = _get_stats_subfolder(ctx, "trial_table")
    out_ext = ".parquet" if fmt == "parquet" else ".tsv"
    out_path = out_dir / f"{fname}{out_ext}"

    if not out_path.exists():
        return None

    meta_path = _trial_table_metadata_path(out_path)
    if not meta_path.exists():
        return None

    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(metadata, dict):
        return None

    contract = metadata.get("contract", {}) or {}
    if str(contract.get("input_hash", "")) != str(input_hash):
        return None

    from eeg_pipeline.infra.tsv import read_table

    df_cached = read_table(out_path)
    _validate_trial_table_contract_metadata(ctx, out_path, df_cached)

    _cache._trial_table_df = df_cached
    _cache._trial_table_path = out_path
    ctx.logger.info(
        "Trial table unchanged (input hash match); reusing existing artifact: %s/%s",
        out_dir.name,
        out_path.name,
    )
    return out_path


def stage_trial_table(ctx: BehaviorContext, config: Any) -> Optional[Path]:
    """Build and save trial table (composed stage).
    
    Composes: compute_trial_table + write_trial_table
    """
    input_hash = _compute_trial_table_input_hash(ctx)
    cached = _try_reuse_cached_trial_table(ctx, input_hash=input_hash)
    if cached is not None:
        return cached

    result = compute_trial_table(ctx, config)
    if result is None or not _is_dataframe_valid(result.df):
        ctx.logger.warning("Trial table: no data to write")
        return None
    contract = result.metadata.setdefault("contract", {})
    contract["input_hash"] = str(input_hash)
    return write_trial_table(ctx, result)


def stage_lag_features(ctx: BehaviorContext, config: Any) -> Optional[Path]:
    """Add lagged and delta variables to the trial table.
    
    Computes:
    - prev_temperature, delta_temperature
    - prev_rating, delta_rating
    - trial_index_within_group
    
    These variables are useful for habituation/dynamics analyses.
    """
    from eeg_pipeline.utils.data.trial_table import add_lag_and_delta_features

    df = _load_trial_table_df(ctx)
    if not _is_dataframe_valid(df):
        ctx.logger.warning("Lag features: trial table missing; skipping.")
        return None

    run_col = str(get_config_value(ctx.config, "behavior_analysis.run_adjustment.column", "run_id") or "run_id").strip()
    group_cols = [c for c in [run_col, "run_id", "run", "block"] if c]
    seen = set()
    group_cols = [c for c in group_cols if not (c in seen or seen.add(c))]

    df_augmented, lag_meta = add_lag_and_delta_features(df, group_columns=group_cols)

    suffix = _feature_suffix_from_context(ctx)
    out_dir = _get_stats_subfolder(ctx, "lag_features")
    out_path = out_dir / f"trials_with_lags{suffix}.parquet"
    _write_parquet_with_optional_csv(df_augmented, out_path, also_save_csv=ctx.also_save_csv)

    meta_path = out_dir / f"lag_features{suffix}.metadata.json"
    _write_metadata_file(meta_path, lag_meta)
    ctx.data_qc["lag_features"] = lag_meta
    ctx.logger.info("Lag features saved: %s/%s", out_dir.name, out_path.name)

    return out_path


def stage_pain_residual(ctx: BehaviorContext, config: Any) -> Optional[Path]:
    """Compute pain residual = rating - f(temperature).
    
    Fits a flexible temperature→rating curve and computes residuals
    representing 'pain beyond stimulus intensity'. Optionally performs
    cross-validated (out-of-run) prediction to avoid overfitting.
    """
    from eeg_pipeline.utils.data.trial_table import add_pain_residual

    df = _load_trial_table_df(ctx)
    if not _is_dataframe_valid(df):
        ctx.logger.warning("Pain residual: trial table missing; skipping.")
        return None

    required_columns = {"temperature", "rating"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        ctx.logger.warning(
            "Pain residual: requires %s columns; missing: %s. Skipping.",
            required_columns, missing_columns
        )
        return None

    df_augmented, resid_meta = add_pain_residual(df, ctx.config)

    suffix = _feature_suffix_from_context(ctx)
    out_dir = _get_stats_subfolder(ctx, "pain_residual")
    out_path = out_dir / f"trials_with_residual{suffix}.parquet"
    _write_parquet_with_optional_csv(df_augmented, out_path, also_save_csv=ctx.also_save_csv)

    meta_path = out_dir / f"pain_residual{suffix}.metadata.json"
    _write_metadata_file(meta_path, resid_meta)
    ctx.data_qc["pain_residual"] = resid_meta
    ctx.logger.info("Pain residual saved: %s/%s", out_dir.name, out_path.name)

    return out_path


@dataclass
class TempModelComparisonResult:
    """Result from compute_temp_model_comparison."""
    df: Optional[pd.DataFrame]
    metadata: Dict[str, Any]


@dataclass
class TempBreakpointResult:
    """Result from compute_temp_breakpoints."""
    df: Optional[pd.DataFrame]
    metadata: Dict[str, Any]


def compute_temp_model_comparison(
    temperature: pd.Series,
    rating: pd.Series,
    config: Any,
) -> TempModelComparisonResult:
    """Compare temperature→rating model fits (linear vs polynomial vs spline).
    
    Single responsibility: Model comparison computation.
    """
    from eeg_pipeline.utils.analysis.stats.temperature_models import compare_temperature_rating_models

    df_cmp, meta_cmp = compare_temperature_rating_models(temperature, rating, config=config)
    return TempModelComparisonResult(df=df_cmp, metadata=meta_cmp)


def compute_temp_breakpoints(
    temperature: pd.Series,
    rating: pd.Series,
    config: Any,
) -> TempBreakpointResult:
    """Detect threshold temperatures where sensitivity changes.
    
    Single responsibility: Breakpoint detection computation.
    """
    from eeg_pipeline.utils.analysis.stats.temperature_models import fit_temperature_breakpoint_test

    df_bp, meta_bp = fit_temperature_breakpoint_test(temperature, rating, config=config)
    return TempBreakpointResult(df=df_bp, metadata=meta_bp)


def write_temperature_models(
    ctx: BehaviorContext,
    model_comparison: Optional[TempModelComparisonResult],
    breakpoint: Optional[TempBreakpointResult],
) -> Path:
    """Write temperature model results to disk.
    
    Single responsibility: Persist temperature model artifacts.
    """
    suffix = _feature_suffix_from_context(ctx)
    out_dir = _get_stats_subfolder(ctx, "temperature_models")

    if model_comparison is not None:
        if _is_dataframe_valid(model_comparison.df):
            comparison_path = out_dir / f"model_comparison{suffix}.parquet"
            _write_parquet_with_optional_csv(
                model_comparison.df, comparison_path, also_save_csv=ctx.also_save_csv
            )
        
        metadata_path = out_dir / f"model_comparison{suffix}.metadata.json"
        metadata_path.write_text(
            json.dumps(model_comparison.metadata, indent=2, default=str)
        )
        ctx.data_qc["temperature_model_comparison"] = model_comparison.metadata
        ctx.logger.info("Temperature model comparison saved: %s", out_dir.name)

    if breakpoint is not None:
        if _is_dataframe_valid(breakpoint.df):
            breakpoint_path = out_dir / f"breakpoint_candidates{suffix}.parquet"
            _write_parquet_with_optional_csv(
                breakpoint.df, breakpoint_path, also_save_csv=ctx.also_save_csv
            )
        
        metadata_path = out_dir / f"breakpoint_test{suffix}.metadata.json"
        metadata_path.write_text(
            json.dumps(breakpoint.metadata, indent=2, default=str)
        )
        ctx.data_qc["temperature_breakpoint_test"] = breakpoint.metadata
        ctx.logger.info("Temperature breakpoint test saved: %s", out_dir.name)

    return out_dir


def stage_temperature_models(ctx: BehaviorContext, config: Any) -> Dict[str, Any]:
    """Compare temperature→rating model fits and test for breakpoints (composed).
    
    Composes: compute_temp_model_comparison + compute_temp_breakpoints + write_temperature_models
    """
    df = _load_trial_table_df(ctx)
    if not _is_dataframe_valid(df):
        ctx.logger.warning("Temperature models: trial table missing; skipping.")
        return {"status": "skipped_missing_data"}

    required_columns = {"temperature", "rating"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        ctx.logger.warning(
            "Temperature models: requires %s columns; missing: %s. Skipping.",
            required_columns, missing_columns
        )
        return {"status": "skipped_missing_columns"}

    meta: Dict[str, Any] = {"status": "init"}
    model_comparison = None
    breakpoint = None

    mc_enabled = get_config_bool(ctx.config, "behavior_analysis.temperature_models.model_comparison.enabled", True)
    if mc_enabled:
        model_comparison = compute_temp_model_comparison(df["temperature"], df["rating"], ctx.config)
        meta["model_comparison"] = model_comparison.metadata

    bp_enabled = get_config_bool(ctx.config, "behavior_analysis.temperature_models.breakpoint_test.enabled", True)
    if bp_enabled:
        breakpoint = compute_temp_breakpoints(df["temperature"], df["rating"], ctx.config)
        meta["breakpoint_test"] = breakpoint.metadata

    write_temperature_models(ctx, model_comparison, breakpoint)
    meta["status"] = "ok"
    return meta


def _load_trial_table_df(ctx: BehaviorContext) -> Optional[pd.DataFrame]:
    """Load trial table with caching."""
    return _cache.get_trial_table(ctx)


def stage_regression(ctx: BehaviorContext, config: Any) -> pd.DataFrame:
    """Trialwise regression: rating (or pain_residual) ~ temperature + trial order + feature (+ interaction).

    Supports primary_unit=trial|run to control unit of analysis.
    Run-level aggregates features/outcomes per run before fitting, avoiding pseudo-replication.
    """
    from eeg_pipeline.utils.analysis.stats.trialwise_regression import run_trialwise_feature_regressions

    suffix = _feature_suffix_from_context(ctx)
    method_label = config.method_label
    method_suffix = f"_{method_label}" if method_label else ""

    df_trials = _load_trial_table_df(ctx)
    if not _is_dataframe_valid(df_trials):
        ctx.logger.warning("Regression: trial table missing; skipping.")
        return pd.DataFrame()

    feature_cols = _get_feature_columns(df_trials, ctx)

    should_skip, skip_reason = _check_early_exit_conditions(
        df_trials,
        feature_cols,
        min_features=1,
        min_trials=10,
    )
    if should_skip:
        ctx.logger.info(f"Regression: skipping due to {skip_reason}")
        return pd.DataFrame()

    # Unit of analysis control
    primary_unit = str(get_config_value(ctx.config, "behavior_analysis.regression.primary_unit", "trial")).strip().lower()
    use_run_unit = primary_unit in {"run", "run_mean", "runmean", "run_level"}
    run_col = str(get_config_value(ctx.config, "behavior_analysis.run_adjustment.column", "run_id") or "run_id").strip()
    allow_iid_trials = get_config_bool(ctx.config, "behavior_analysis.statistics.allow_iid_trials", False)
    n_perm = get_config_int(ctx.config, "behavior_analysis.regression.n_permutations", 0)

    if use_run_unit and run_col not in df_trials.columns:
        raise ValueError(
            f"Run-level regression requested (primary_unit={primary_unit!r}) "
            f"but run column '{run_col}' is missing from trial table."
        )
    if primary_unit in {"trial", "trialwise"} and not allow_iid_trials and n_perm <= 0:
        raise ValueError(
            "Trial-level regression requires a valid non-i.i.d inference method. "
            "Set behavior_analysis.regression.n_permutations > 0, "
            "use run-level aggregation (behavior_analysis.regression.primary_unit=run_mean), "
            "or set behavior_analysis.statistics.allow_iid_trials=true to override (not recommended)."
        )

    # Aggregate to run-level if requested (avoids pseudo-replication)
    if use_run_unit and run_col in df_trials.columns:
        ctx.logger.info("Regression: aggregating to run-level (primary_unit=%s)", primary_unit)
        agg_cols = feature_cols + ["rating", "temperature"]
        agg_cols = [c for c in agg_cols if c in df_trials.columns]
        df_trials = df_trials.groupby(run_col)[agg_cols].mean().reset_index()
        ctx.logger.info("  Run-level: %d observations", len(df_trials))

    groups = None
    if getattr(ctx, "group_ids", None) is not None:
        groups_candidate = np.asarray(ctx.group_ids)
        if len(groups_candidate) == len(df_trials):
            groups = groups_candidate
        else:
            ctx.logger.warning(
                "Regression: ignoring ctx.group_ids length=%d because current data has %d rows.",
                len(groups_candidate),
                len(df_trials),
            )
    if groups is None:
        if run_col in df_trials.columns:
            groups = df_trials[run_col].to_numpy()
        elif "block" in df_trials.columns:
            groups = df_trials["block"].to_numpy()
        elif "run" in df_trials.columns:
            groups = df_trials["run"].to_numpy()
    groups = _sanitize_permutation_groups(
        groups,
        ctx.logger,
        "Regression",
    )
    if primary_unit in {"trial", "trialwise"} and not allow_iid_trials and groups is None:
        raise ValueError(
            "Trial-level regression with permutation inference requires grouped labels. "
            "Provide behavior_analysis.run_adjustment.column in the trial table (or ctx.group_ids), "
            "or set behavior_analysis.statistics.allow_iid_trials=true to override (not recommended)."
        )

    reg_df, reg_meta = run_trialwise_feature_regressions(
        df_trials,
        feature_cols=feature_cols,
        config=ctx.config,
        groups_for_permutation=groups,
    )
    reg_meta["primary_unit"] = primary_unit
    ctx.data_qc["trialwise_regression"] = reg_meta
    reg_df = _attach_temperature_metadata(reg_df, reg_meta)

    out_dir = _get_stats_subfolder(ctx, "trialwise_regression")
    out_path = out_dir / f"regression_feature_effects{suffix}{method_suffix}.parquet"
    if not reg_df.empty:
        actual_path = _write_stats_table(ctx, reg_df, out_path)
        ctx.logger.info("Regression results saved: %s (%d features)", actual_path.name, len(reg_df))
    return reg_df


def stage_models(ctx: BehaviorContext, config: Any) -> pd.DataFrame:
    """Fit multiple model families per feature (OLS-HC3 / robust / quantile / logistic)."""
    from eeg_pipeline.utils.analysis.stats.feature_models import run_feature_model_families

    suffix = _feature_suffix_from_context(ctx)
    method_label = getattr(config, "method_label", "")
    method_suffix = f"_{method_label}" if method_label else ""

    df_trials = _load_trial_table_df(ctx)
    if not _is_dataframe_valid(df_trials):
        ctx.logger.warning("Models: trial table missing; skipping.")
        return pd.DataFrame()

    feature_cols = _get_feature_columns(df_trials, ctx)

    should_skip, skip_reason = _check_early_exit_conditions(
        df_trials,
        feature_cols,
        min_features=1,
        min_trials=10,
    )
    if should_skip:
        ctx.logger.info(f"Models: skipping due to {skip_reason}")
        return pd.DataFrame()

    primary_unit = str(get_config_value(ctx.config, "behavior_analysis.models.primary_unit", "trial") or "trial").strip().lower()
    use_run_unit = primary_unit in {"run", "run_mean", "runmean", "run_level"}
    allow_iid_trials = get_config_bool(ctx.config, "behavior_analysis.statistics.allow_iid_trials", False)
    run_col = str(get_config_value(ctx.config, "behavior_analysis.run_adjustment.column", "run_id") or "run_id").strip()

    if primary_unit in {"trial", "trialwise"} and not allow_iid_trials:
        raise ValueError(
            "Trial-level feature-model inference assumes i.i.d trials and is not recommended. "
            "Use run-level aggregation (behavior_analysis.models.primary_unit=run_mean) "
            "or set behavior_analysis.statistics.allow_iid_trials=true to override."
        )

    if use_run_unit:
        if run_col not in df_trials.columns:
            raise ValueError(
                f"Run-level models requested (primary_unit={primary_unit!r}) "
                f"but run column '{run_col}' is missing from trial table."
            )
        ctx.logger.info("Models: aggregating to run-level (primary_unit=%s)", primary_unit)
        outcomes_cfg = get_config_value(ctx.config, "behavior_analysis.models.outcomes", ["rating", "pain_residual"])
        if isinstance(outcomes_cfg, str):
            outcomes_cfg = [outcomes_cfg]
        elif not isinstance(outcomes_cfg, (list, tuple)):
            outcomes_cfg = ["rating", "pain_residual"]
        binary_outcome = str(
            get_config_value(ctx.config, "behavior_analysis.models.binary_outcome", "pain_binary") or "pain_binary"
        ).strip()
        extra_cols = [
            "temperature",
            "rating",
            "pain_residual",
            "pain_binary",
            "trial_index",
            "trial_index_within_group",
            "prev_temperature",
            "prev_rating",
            "delta_temperature",
            "delta_rating",
        ]
        agg_cols = [
            c
            for c in set(feature_cols + list(outcomes_cfg) + [binary_outcome] + extra_cols)
            if c in df_trials.columns
        ]
        agg_numeric = {c: "mean" for c in agg_cols if c != binary_outcome}
        grouped = df_trials.groupby(run_col).agg(agg_numeric)
        if binary_outcome in agg_cols:
            grouped[binary_outcome] = df_trials.groupby(run_col)[binary_outcome].apply(
                lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0]
            )
        df_trials = grouped.reset_index()
        ctx.logger.info("  Run-level: %d observations", len(df_trials))

    model_df, model_meta = run_feature_model_families(
        df_trials,
        feature_cols=feature_cols,
        config=ctx.config,
    )
    ctx.data_qc["feature_models"] = model_meta
    model_df = _attach_temperature_metadata(model_df, model_meta, target_col="target")

    out_dir = _get_stats_subfolder(ctx, "feature_models")
    out_path = out_dir / f"models_feature_effects{suffix}{method_suffix}.parquet"
    if model_df is not None and not model_df.empty:
        actual_path = _write_stats_table(ctx, model_df, out_path)
        ctx.logger.info("Model families results saved: %s (%d rows)", actual_path.name, len(model_df))
    return model_df if model_df is not None else pd.DataFrame()


def stage_stability(ctx: BehaviorContext, config: Any) -> pd.DataFrame:
    """Assess within-subject run/block stability of feature→outcome associations (non-gating)."""
    from eeg_pipeline.utils.analysis.stats.stability import compute_groupwise_stability

    filename = _build_output_filename(ctx, config, "stability_groupwise")

    df_trials = _load_trial_table_df(ctx)
    if not _is_dataframe_valid(df_trials):
        ctx.logger.warning("Stability: trial table missing; skipping.")
        return pd.DataFrame()

    run_col = str(get_config_value(ctx.config, "behavior_analysis.run_adjustment.column", "run_id") or "run_id").strip()
    group_col = str(get_config_value(ctx.config, "behavior_analysis.stability.group_column", "")).strip()
    if group_col and group_col not in df_trials.columns:
        # Support common alias: "run" means configured run column (often run_id).
        if group_col == "run" and run_col in df_trials.columns:
            group_col = run_col
        else:
            ctx.logger.warning("Stability: configured group_column '%s' not found; falling back to auto.", group_col)
            group_col = ""
    if not group_col:
        if run_col in df_trials.columns:
            group_col = run_col
        elif "run_id" in df_trials.columns:
            group_col = "run_id"
        else:
            group_col = "run" if "run" in df_trials.columns else ("block" if "block" in df_trials.columns else "")
    if not group_col:
        ctx.logger.info("Stability: no run/block column available; skipping.")
        return pd.DataFrame()

    outcome = str(get_config_value(ctx.config, "behavior_analysis.stability.outcome", "")).strip().lower()
    if not outcome:
        outcome = "pain_residual" if "pain_residual" in df_trials.columns else "rating"

    feature_cols = _get_feature_columns(df_trials, ctx)

    should_skip, skip_reason = _check_early_exit_conditions(
        df_trials,
        feature_cols,
        min_features=1,
        min_trials=10,
    )
    if should_skip:
        ctx.logger.info(f"Stability: skipping due to {skip_reason}")
        return pd.DataFrame()

    stab_df, stab_meta = compute_groupwise_stability(
        df_trials,
        feature_cols=feature_cols,
        outcome=outcome,
        group_col=group_col,
        config=ctx.config,
    )
    ctx.data_qc["stability_groupwise"] = stab_meta

    out_dir = _get_stats_subfolder(ctx, "stability_groupwise")
    out_path = out_dir / f"{filename}.parquet"
    if stab_df is not None and not stab_df.empty:
        actual_path = _write_stats_table(ctx, stab_df, out_path)
        ctx.logger.info("Stability results saved: %s (%d features)", actual_path.name, len(stab_df))
    _write_metadata_file(out_dir / f"{filename}.metadata.json", stab_meta)
    return stab_df if stab_df is not None else pd.DataFrame()


def stage_consistency(ctx: BehaviorContext, config: Any, results: Any) -> pd.DataFrame:
    """Merge correlations/regression/models and flag effect-direction contradictions (non-gating)."""
    from eeg_pipeline.utils.analysis.stats.consistency import build_effect_direction_consistency_summary

    filename = _build_output_filename(ctx, config, "consistency_summary")

    corr_df = getattr(results, "correlations", None)
    reg_df = getattr(results, "regression", None)
    models_df = getattr(results, "models", None)
    out_df, meta = build_effect_direction_consistency_summary(
        corr_df=corr_df,
        regression_df=reg_df,
        models_df=models_df,
    )
    ctx.data_qc["effect_direction_consistency"] = meta
    if out_df is None or out_df.empty:
        return pd.DataFrame()

    out_dir = _get_stats_subfolder(ctx, "consistency_summary")
    out_path = out_dir / f"{filename}.parquet"
    actual_path = _write_stats_table(ctx, out_df, out_path)
    _write_metadata_file(out_dir / f"{filename}.metadata.json", meta)
    ctx.logger.info("Consistency summary saved: %s (%d features)", actual_path.name, len(out_df))
    return out_df


def stage_influence(ctx: BehaviorContext, config: Any, results: Any) -> pd.DataFrame:
    """Compute leverage/Cook's summaries for top effects (non-gating)."""
    from eeg_pipeline.utils.analysis.stats.influence import compute_influence_diagnostics

    df_trials = _load_trial_table_df(ctx)
    if not _is_dataframe_valid(df_trials):
        ctx.logger.info("Influence: trial table missing; skipping.")
        return pd.DataFrame()

    feature_cols = _get_feature_columns(df_trials, ctx)

    should_skip, skip_reason = _check_early_exit_conditions(
        df_trials,
        feature_cols,
        min_features=1,
        min_trials=10,
    )
    if should_skip:
        ctx.logger.info(f"Influence: skipping due to {skip_reason}")
        return pd.DataFrame()

    out_df, meta = compute_influence_diagnostics(
        df_trials,
        corr_df=getattr(results, "correlations", None),
        regression_df=getattr(results, "regression", None),
        models_df=getattr(results, "models", None),
        config=ctx.config,
    )
    ctx.data_qc["influence_diagnostics"] = meta
    if not _is_dataframe_valid(out_df):
        return pd.DataFrame()
    
    influence_meta = meta if isinstance(meta, dict) else {}
    influence_meta["temperature_control"] = get_config_value(ctx.config, "behavior_analysis.influence.temperature_control", None)
    out_df = _attach_temperature_metadata(out_df, influence_meta, target_col="outcome")

    out_dir = _get_stats_subfolder(ctx, "influence_diagnostics")
    filename = _build_output_filename(ctx, config, "influence_diagnostics")
    out_path = out_dir / f"{filename}.parquet"
    actual_path = _write_stats_table(ctx, out_df, out_path)
    _write_metadata_file(out_dir / f"{filename}.metadata.json", meta)
    ctx.logger.info("Influence diagnostics saved: %s (%d rows)", actual_path.name, len(out_df))
    return out_df


def _compute_series_statistics(series: pd.Series) -> Dict[str, Any]:
    """Compute basic statistics for a numeric series."""
    valid_values = series.notna()
    n_valid = int(valid_values.sum())
    
    if n_valid == 0:
        return {
            "n_non_nan": 0,
            "min": np.nan,
            "max": np.nan,
            "mean": np.nan,
            "std": np.nan,
        }
    
    numeric_series = pd.to_numeric(series, errors="coerce")
    has_values = numeric_series.notna().any()
    has_multiple = n_valid > 1
    
    return {
        "n_non_nan": n_valid,
        "min": float(numeric_series.min()) if has_values else np.nan,
        "max": float(numeric_series.max()) if has_values else np.nan,
        "mean": float(numeric_series.mean()) if has_values else np.nan,
        "std": float(numeric_series.std(ddof=1)) if has_multiple else np.nan,
    }


def build_behavior_qc(ctx: BehaviorContext) -> Dict[str, Any]:
    """Build behavior quality control summary."""
    qc: Dict[str, Any] = {
        "subject": ctx.subject,
        "task": ctx.task,
        "n_trials": ctx.n_trials,
        "has_temperature": ctx.has_temperature,
        "temperature_column": ctx.temperature_column,
        "group_column": ctx.group_column,
    }

    if ctx.data_qc:
        qc["data_qc"] = ctx.data_qc

    # Get rating from aligned_events
    rating_series = None
    rating_col = ctx._find_rating_column() if hasattr(ctx, "_find_rating_column") else None
    if rating_col is not None and ctx.aligned_events is not None:
        rating_series = pd.to_numeric(ctx.aligned_events[rating_col], errors="coerce")

    if rating_series is not None:
        qc["rating"] = _compute_series_statistics(rating_series)

    if ctx.temperature is not None:
        qc["temperature"] = _compute_series_statistics(ctx.temperature)

    if rating_series is not None and ctx.temperature is not None:
        s = pd.to_numeric(rating_series, errors="coerce")
        t = pd.to_numeric(ctx.temperature, errors="coerce")
        valid = s.notna() & t.notna()
        if int(valid.sum()) >= 3:
            r, p = compute_correlation(s[valid].values, t[valid].values, method="spearman")
            qc["rating_temperature_sanity"] = {
                "method": "spearman",
                "n": int(valid.sum()),
                "r": float(r) if np.isfinite(r) else np.nan,
                "p": float(p) if np.isfinite(p) else np.nan,
            }

    if ctx.aligned_events is not None and rating_series is not None:
        from eeg_pipeline.analysis.behavior.api import split_by_condition

        pain_mask, nonpain_mask, n_pain, n_nonpain = split_by_condition(
            ctx.aligned_events, ctx.config, ctx.logger
        )
        if int(n_pain) > 0 or int(n_nonpain) > 0:
            s = pd.to_numeric(rating_series, errors="coerce")
            pain_ratings = s[pain_mask] if len(pain_mask) == len(s) else pd.Series(dtype=float)
            nonpain_ratings = s[nonpain_mask] if len(nonpain_mask) == len(s) else pd.Series(dtype=float)
            qc["pain_vs_nonpain"] = {
                "status": "ok",
                "n_pain": int(n_pain),
                "n_nonpain": int(n_nonpain),
                "mean_rating_pain": float(pain_ratings.mean()) if pain_ratings.notna().any() else np.nan,
                "mean_rating_nonpain": float(nonpain_ratings.mean()) if nonpain_ratings.notna().any() else np.nan,
                "mean_rating_difference_pain_minus_nonpain": (
                    float(pain_ratings.mean() - nonpain_ratings.mean())
                    if pain_ratings.notna().any() and nonpain_ratings.notna().any()
                    else np.nan
                ),
            }

    if _is_dataframe_valid(ctx.covariates_df):
        cov = ctx.covariates_df
        qc["covariates"] = {
            "n_covariates": int(cov.shape[1]),
            "columns": [str(col) for col in cov.columns],
            "missing_fraction_by_column": {
                str(col): float(pd.to_numeric(cov[col], errors="coerce").isna().mean())
                for col in cov.columns
            },
        }

    feature_counts: Dict[str, int] = {}
    for name, df in ctx.iter_feature_tables():
        if df is None or df.empty:
            continue
        feature_counts[name] = int(df.shape[1])
    qc["feature_counts"] = feature_counts

    return qc


def _infer_feature_type(feature: str, config: Any) -> str:
    try:
        from eeg_pipeline.domain.features.registry import classify_feature, get_feature_registry

        registry = get_feature_registry(config)
        ftype, _, _ = classify_feature(feature, include_subtype=False, registry=registry)
        return ftype
    except Exception:
        name = str(feature or "").strip().lower()
        for prefix in FEATURE_COLUMN_PREFIXES:
            if name.startswith(prefix):
                return prefix.rstrip("_")
        return "unknown"


def _infer_feature_band(feature: str, config: Any) -> str:
    """Extract frequency band from feature name using naming schema."""
    try:
        from eeg_pipeline.domain.features.registry import classify_feature, get_feature_registry

        registry = get_feature_registry(config)
        _, _, meta = classify_feature(feature, include_subtype=True, registry=registry)
        band = meta.get("band", "N/A")
        return str(band) if band and band != "N/A" else "broadband"
    except Exception:
        name = str(feature or "").strip().lower()
        for band in ("delta", "theta", "alpha", "beta", "gamma"):
            token = f"_{band}"
            if token in name:
                return band
        return "broadband"


def _summarize_covariates_qc(ctx: BehaviorContext) -> Dict[str, Any]:
    cov = ctx.covariates_df
    if cov is None or cov.empty:
        return {"status": "empty"}
    return {
        "status": "ok",
        "columns": [str(c) for c in cov.columns],
        "missing_fraction_by_column": {
            str(c): float(pd.to_numeric(cov[c], errors="coerce").isna().mean()) for c in cov.columns
        },
    }


def write_analysis_metadata(
    ctx: BehaviorContext,
    pipeline_config: Any,
    results: Any,
    stage_metrics: Optional[Dict[str, Any]] = None,
    outputs_manifest: Optional[Path] = None,
) -> Path:
    robust_method = pipeline_config.robust_method
    method_label = pipeline_config.method_label
    payload: Dict[str, Any] = {
        "subject": ctx.subject,
        "task": ctx.task,
        "method": pipeline_config.method,
        "method_label": method_label,
        "robust_method": robust_method,
        "min_samples": pipeline_config.min_samples,
        "control_temperature": pipeline_config.control_temperature,
        "control_trial_order": pipeline_config.control_trial_order,
        "compute_change_scores": pipeline_config.compute_change_scores,
        "compute_pain_sensitivity": pipeline_config.compute_pain_sensitivity,
        "compute_reliability": pipeline_config.compute_reliability,
        "n_permutations": pipeline_config.n_permutations,
        "fdr_alpha": pipeline_config.fdr_alpha,
        "n_trials": ctx.n_trials,
        "statistics_config": {
            "method": pipeline_config.method,
            "robust_method": robust_method,
            "method_label": method_label,
            "min_samples": pipeline_config.min_samples,
            "bootstrap": pipeline_config.bootstrap,
            "n_permutations": pipeline_config.n_permutations,
            "fdr_alpha": pipeline_config.fdr_alpha,
            "control_temperature": pipeline_config.control_temperature,
            "control_trial_order": pipeline_config.control_trial_order,
            "compute_change_scores": pipeline_config.compute_change_scores,
            "compute_reliability": pipeline_config.compute_reliability,
            "compute_bayes_factors": getattr(pipeline_config, "compute_bayes_factors", False),
            "compute_loso_stability": getattr(pipeline_config, "compute_loso_stability", False),
        },
        "outputs": {
            "has_trial_table": bool(getattr(results, "trial_table_path", None)),
            "has_regression": bool(getattr(results, "regression", None) is not None and not results.regression.empty),
            "has_stability": bool(getattr(results, "stability", None) is not None and not getattr(results, "stability").empty) if getattr(results, "stability", None) is not None else False,
            "has_models": bool(getattr(results, "models", None) is not None and not getattr(results, "models").empty) if getattr(results, "models", None) is not None else False,
            "has_correlations": bool(getattr(results, "correlations", None) is not None and not results.correlations.empty),
            "has_pain_sensitivity": bool(getattr(results, "pain_sensitivity", None) is not None and not results.pain_sensitivity.empty),
            "has_condition_effects": bool(getattr(results, "condition_effects", None) is not None and not results.condition_effects.empty),
            "has_mediation": bool(getattr(results, "mediation", None) is not None and not getattr(results, "mediation").empty) if getattr(results, "mediation", None) is not None else False,
            "has_moderation": bool(getattr(results, "moderation", None) is not None and not getattr(results, "moderation").empty) if getattr(results, "moderation", None) is not None else False,
        },
        "qc": build_behavior_qc(ctx),
    }

    payload["temperature_status"] = {
        "available": bool(ctx.temperature is not None and ctx.temperature.notna().any()) if ctx.temperature is not None else False,
        "control_enabled": bool(ctx.control_temperature),
    }
    if not payload["temperature_status"]["available"]:
        payload["temperature_status"]["reason"] = "missing_temperature"

    if not pipeline_config.compute_pain_sensitivity:
        payload["pain_sensitivity_status"] = "disabled"
    elif payload["temperature_status"]["available"]:
        payload["pain_sensitivity_status"] = "computed" if payload["outputs"]["has_pain_sensitivity"] else "skipped"
    else:
        payload["pain_sensitivity_status"] = "skipped_no_temperature"

    payload["covariates_qc"] = _summarize_covariates_qc(ctx)

    if stage_metrics:
        payload["stage_metrics"] = stage_metrics

    if outputs_manifest is not None:
        payload["outputs_manifest"] = str(outputs_manifest)

    if ctx.data_qc:
        payload["data_qc"] = ctx.data_qc

    corr_df = getattr(results, "correlations", None)
    if corr_df is not None and not corr_df.empty:
        df = corr_df
        partial_cols = [
            ("p_partial_cov", "partial_cov"),
            ("p_partial_temp", "partial_temp"),
            ("p_partial_cov_temp", "partial_cov_temp"),
        ]
        partial_ok: Dict[str, Any] = {}
        for col, label in partial_cols:
            if col not in df.columns:
                continue
            pvals = pd.to_numeric(df[col], errors="coerce")
            partial_ok[label] = {
                "n_non_nan": int(pvals.notna().sum()),
                "fraction_non_nan": float(pvals.notna().mean()) if len(pvals) else np.nan,
            }
        if partial_ok:
            payload["partial_correlation_feasibility"] = partial_ok

        if "p_primary_source" in df.columns and df["p_primary_source"].notna().any():
            payload["primary_test_source_counts"] = df["p_primary_source"].fillna("unknown").value_counts().to_dict()

        if "within_family_p_kind" in df.columns and df["within_family_p_kind"].notna().any():
            payload["within_family_p_kind_counts"] = df["within_family_p_kind"].fillna("unknown").value_counts().to_dict()

    out_dir = _get_stats_subfolder(ctx, "analysis_metadata")
    path = out_dir / "analysis_metadata.json"
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


###################################################################
# Condition Stage - Single Responsibility Components
###################################################################


def _resolve_condition_compare_column(df_trials: pd.DataFrame, config: Any) -> str:
    """Resolve configured condition column, falling back to configured pain column."""
    from eeg_pipeline.utils.data.columns import get_pain_column_from_config

    compare_col = str(
        get_config_value(config, "behavior_analysis.condition.compare_column", "") or ""
    ).strip()
    if compare_col and compare_col in df_trials.columns:
        return compare_col

    fallback_col = get_pain_column_from_config(config, df_trials)
    if fallback_col and fallback_col in df_trials.columns:
        return str(fallback_col)

    # Keep configured name for diagnostics if present; otherwise use legacy default.
    return compare_col or "pain"


def stage_condition_column(
    ctx: BehaviorContext,
    config: Any,
    df_trials: Optional[pd.DataFrame] = None,
    feature_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Run column-based condition comparison (e.g., pain vs non-pain).
    
    Single responsibility: Column contrast comparison.
    Supports primary_unit=trial|run to control unit of analysis.
    
    When overwrite=false, includes compare_column name in output filename to allow
    multiple comparisons without overwriting previous results.
    
    If compare_values has 3+ values, delegates to multigroup comparison instead.
    """
    from eeg_pipeline.analysis.behavior.api import split_by_condition, compute_condition_effects
    from eeg_pipeline.infra.tsv import write_parquet

    # Check if multigroup comparison is needed (3+ values)
    compare_values = get_config_value(ctx.config, "behavior_analysis.condition.compare_values", [])
    use_multigroup = isinstance(compare_values, (list, tuple)) and len(compare_values) > 2
    
    if use_multigroup:
        # Delegate to multigroup comparison
        ctx.logger.info(
            f"Condition column: {len(compare_values)} values specified, "
            "delegating to multigroup comparison"
        )
        return stage_condition_multigroup(ctx, config, df_trials=df_trials, feature_cols=feature_cols)

    fail_fast = get_config_value(ctx.config, "behavior_analysis.condition.fail_fast", True)
    primary_unit = str(get_config_value(
        ctx.config, "behavior_analysis.condition.primary_unit", "trial"
    )).strip().lower()
    allow_iid_trials = get_config_bool(ctx.config, "behavior_analysis.statistics.allow_iid_trials", False)
    perm_enabled = get_config_bool(ctx.config, "behavior_analysis.condition.permutation.enabled", False)
    if primary_unit in {"trial", "trialwise"} and not perm_enabled and not allow_iid_trials:
        raise ValueError(
            "Trial-level condition comparisons require a valid non-i.i.d inference method. "
            "Enable permutation testing (behavior_analysis.condition.permutation.enabled=true) "
            "or use run-level aggregation (behavior_analysis.condition.primary_unit=run_mean). "
            "Set behavior_analysis.statistics.allow_iid_trials=true to override (not recommended)."
        )
    run_unit_values = {"run", "run_mean", "runmean", "run_level"}
    use_run_unit = primary_unit in run_unit_values
    run_col = str(get_config_value(
        ctx.config, "behavior_analysis.run_adjustment.column", "run_id"
    ) or "run_id").strip()

    if df_trials is None:
        df_trials = _load_trial_table_df(ctx)
    if not _is_dataframe_valid(df_trials):
        ctx.logger.warning("Condition column: trial table missing; skipping.")
        return pd.DataFrame()

    if feature_cols is None:
        feature_cols = _get_feature_columns(df_trials, ctx, "condition")
    
    # Aggregate to run-level if requested (avoids pseudo-replication)
    compare_col = _resolve_condition_compare_column(df_trials, ctx.config)
    overwrite = get_config_bool(ctx.config, "behavior_analysis.condition.overwrite", True)
    if use_run_unit and run_col in df_trials.columns and compare_col in df_trials.columns:
        ctx.logger.info("Condition: aggregating to run×condition level (primary_unit=%s)", primary_unit)
        group_keys = [run_col, compare_col]
        df_agg = (
            df_trials.groupby(group_keys, dropna=True)[feature_cols]
            .mean(numeric_only=True)
            .reset_index()
        )
        cell_counts = (
            df_trials.groupby(group_keys, dropna=True)
            .size()
            .rename("n_trials_cell")
            .reset_index()
        )
        df_trials = df_agg.merge(cell_counts, on=group_keys, how="left")
        ctx.logger.info("  Run×condition level: %d observations", len(df_trials))
    
    if not feature_cols:
        ctx.logger.info("Condition column: no feature columns found; skipping.")
        return pd.DataFrame()

    min_trials_required = 2 if use_run_unit else 10
    should_skip, skip_reason = _check_early_exit_conditions(
        df_trials,
        feature_cols,
        min_features=1,
        min_trials=min_trials_required,
    )
    if should_skip:
        ctx.logger.info(f"Condition column: skipping due to {skip_reason}")
        return pd.DataFrame()

    suffix = _feature_suffix_from_context(ctx)
    out_dir = _get_stats_subfolder(ctx, "condition_effects")

    try:
        pain_mask, nonpain_mask, n_pain, n_nonpain = split_by_condition(df_trials, ctx.config, ctx.logger)

        if n_pain == 0 and n_nonpain == 0:
            msg = (
                "Condition split produced zero trials; check "
                "behavior_analysis.condition.compare_column / behavior_analysis.condition.compare_values "
                "and/or config event_columns.pain_binary"
            )
            if fail_fast:
                raise ValueError(msg)
            ctx.logger.warning(msg)
            return pd.DataFrame()

        # Extract condition values from config or data
        compare_values = get_config_value(ctx.config, "behavior_analysis.condition.compare_values", None)
        if compare_values and len(compare_values) >= 2:
            condition_value1, condition_value2 = str(compare_values[0]), str(compare_values[1])
        else:
            # Fallback: extract from actual data values
            if compare_col in df_trials.columns:
                condition_series = df_trials[compare_col]
                unique_vals = condition_series.dropna().unique()
                if len(unique_vals) >= 2:
                    condition_value1, condition_value2 = str(unique_vals[0]), str(unique_vals[1])
                else:
                    condition_value1, condition_value2 = "1", "0"
            else:
                condition_value1, condition_value2 = "1", "0"

        features = df_trials[feature_cols].copy()
        groups = None
        if getattr(ctx, "group_ids", None) is not None:
            groups_candidate = np.asarray(ctx.group_ids)
            if len(groups_candidate) == len(df_trials):
                groups = groups_candidate
            else:
                ctx.logger.warning(
                    "Condition column: ignoring ctx.group_ids length=%d because current data has %d rows.",
                    len(groups_candidate),
                    len(df_trials),
                )
        if groups is None:
            run_col = str(get_config_value(ctx.config, "behavior_analysis.run_adjustment.column", "run_id") or "run_id").strip()
            if run_col and run_col in df_trials.columns:
                groups = df_trials[run_col].to_numpy()
        groups = _sanitize_permutation_groups(
            groups,
            ctx.logger,
            "Condition column",
        )
        if primary_unit in {"trial", "trialwise"} and not allow_iid_trials and perm_enabled and groups is None:
            raise ValueError(
                "Trial-level condition comparison requires grouped permutation labels for non-i.i.d trials. "
                "Provide behavior_analysis.run_adjustment.column in the trial table (or ctx.group_ids), "
                "or set behavior_analysis.statistics.allow_iid_trials=true to override (not recommended)."
            )

        column_df = compute_condition_effects(
            features,
            pain_mask,
            nonpain_mask,
            min_samples=max(int(getattr(config, "min_samples", 10)), 2),
            fdr_alpha=config.fdr_alpha,
            logger=ctx.logger,
            n_jobs=config.n_jobs,
            config=ctx.config,
            groups=groups,
        )

        if column_df is not None and not column_df.empty:
            column_df = column_df.copy()
            column_df["comparison_type"] = "column"
            column_df["analysis_kind"] = "condition_column"
            column_df["condition_column"] = compare_col
            column_df["condition_value1"] = condition_value1
            column_df["condition_value2"] = condition_value2

            # Standardize p-value columns
            if "p_value" in column_df.columns and "p_raw" not in column_df.columns:
                column_df["p_raw"] = pd.to_numeric(column_df["p_value"], errors="coerce")
            if "p_value" in column_df.columns and "p_primary" not in column_df.columns:
                column_df["p_primary"] = pd.to_numeric(column_df["p_value"], errors="coerce")
            if "q_value" in column_df.columns and "p_fdr" not in column_df.columns:
                column_df["p_fdr"] = pd.to_numeric(column_df["q_value"], errors="coerce")

            # Use unified FDR
            column_df = _compute_unified_fdr(
                ctx,
                config,
                column_df,
                p_col="p_primary",
                family_cols=["feature_type", "analysis_kind"],
                analysis_type="condition_column",
            )

            # Always include condition column name in filename
            col_path = out_dir / f"condition_effects_column{suffix}_{compare_col}.parquet"
            _write_parquet_with_optional_csv(column_df, col_path, also_save_csv=ctx.also_save_csv)
            ctx.logger.info(f"Condition column comparison: {len(column_df)} features saved to {col_path}")
            return column_df

    except Exception as exc:
        ctx.logger.error("Condition column comparison failed: %s", exc)
        raise

    return pd.DataFrame()



def _compute_pairwise_effect_sizes(
    v1: np.ndarray,
    v2: np.ndarray,
) -> Tuple[float, float, float, float, float]:
    """Compute paired (within-subject) effect sizes (Cohen's dz and Hedge's gz).

    Returns:
        (mean_diff, std_diff, cohens_d, hedges_g, hedges_correction)
    """
    diff = v2 - v1
    mean_diff = float(np.nanmean(diff))
    std_diff = float(np.nanstd(diff, ddof=1))
    cohens_d = mean_diff / std_diff if np.isfinite(std_diff) and std_diff > 0 else np.nan

    # Hedge's g correction
    n = int(np.sum(np.isfinite(v1) & np.isfinite(v2)))
    hedges_correction = 1 - (3 / (4 * n - 1)) if n > 1 else 1.0
    hedges_g = cohens_d * hedges_correction if np.isfinite(cohens_d) else np.nan

    return mean_diff, std_diff, cohens_d, hedges_g, hedges_correction


def _compute_batch_pairwise_effect_sizes(
    v1_matrix: np.ndarray,
    v2_matrix: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute paired effect sizes for multiple feature pairs vectorized.
    
    Args:
        v1_matrix: shape (n_samples, n_features) - window 1 values
        v2_matrix: shape (n_samples, n_features) - window 2 values
    
    Returns:
        (mean_diff, std_diff, cohens_d, hedges_g) - all shape (n_features,)
    """
    diff = v2_matrix - v1_matrix
    mean_diff = np.nanmean(diff, axis=0)
    std_diff = np.nanstd(diff, axis=0, ddof=1)

    # Cohen's dz (paired)
    cohens_d = np.where(std_diff > 1e-12, mean_diff / std_diff, np.nan)

    # Hedge's g correction
    n = np.sum(np.isfinite(v1_matrix) & np.isfinite(v2_matrix), axis=0)
    hedges_correction = np.where(n > 1, 1 - (3 / (4 * n - 1)), 1.0)
    hedges_g = cohens_d * hedges_correction
    
    return mean_diff, std_diff, cohens_d, hedges_g


def stage_condition_window(
    ctx: BehaviorContext,
    config: Any,
    df_trials: Optional[pd.DataFrame] = None,
    feature_cols: Optional[List[str]] = None,
    compare_windows: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Run window-based condition comparison (e.g., baseline vs active).
    
    Single responsibility: Window contrast comparison.
    """
    from eeg_pipeline.infra.tsv import write_parquet

    if compare_windows is None:
        compare_windows = get_config_value(
            ctx.config, "behavior_analysis.condition.compare_windows", []
        )
    
    min_windows_required = 2
    if not compare_windows or len(compare_windows) < min_windows_required:
        if compare_windows:
            ctx.logger.warning(
                "Window comparison requires at least %d windows, got: %s",
                min_windows_required, compare_windows
            )
        return pd.DataFrame()

    if df_trials is None:
        df_trials = _load_trial_table_df(ctx)
    if not _is_dataframe_valid(df_trials):
        ctx.logger.warning("Condition window: trial table missing; skipping.")
        return pd.DataFrame()

    if feature_cols is None:
        feature_cols = _get_feature_columns(df_trials, ctx, "condition")
    
    if not feature_cols:
        ctx.logger.info("Condition window: no feature columns found; skipping.")
        return pd.DataFrame()

    should_skip, skip_reason = _check_early_exit_conditions(df_trials, feature_cols, min_features=1, min_trials=10)
    if should_skip:
        ctx.logger.info(f"Condition window: skipping due to {skip_reason}")
        return pd.DataFrame()

    suffix = _feature_suffix_from_context(ctx)
    compare_col = _resolve_condition_compare_column(df_trials, ctx.config)
    out_dir = _get_stats_subfolder(ctx, "condition_effects")

    ctx.logger.info(f"Running window comparison: {compare_windows}")
    window_df = _run_window_comparison(
        ctx, df_trials, feature_cols, compare_windows, 0, config.fdr_alpha, suffix
    )
    
    if not window_df.empty:
        # Store windows information in dataframe (windows are already stored as window1/window2)
        # Store condition column name for reference but don't include in filename
        window_df["condition_column"] = compare_col
        
        # Window files don't include condition column name in filename (windows are the comparison dimension)
        win_path = out_dir / f"condition_effects_window{suffix}.parquet"
        _write_parquet_with_optional_csv(window_df, win_path, also_save_csv=ctx.also_save_csv)
        ctx.logger.info(f"Condition window comparison: {len(window_df)} features saved to {win_path}")

    return window_df


def stage_condition(ctx: BehaviorContext, config: Any) -> pd.DataFrame:
    """Backward-compatible condition stage (column + optional window + optional multigroup).
    
    The pipeline wrapper historically called a single stage and expected a DataFrame.
    Internally, we keep single-responsibility sub-stages:
    - stage_condition_column (2-group comparison)
    - stage_condition_window (paired window comparison)
    - stage_condition_multigroup (3+ group comparison)
    """
    df_trials = _load_trial_table_df(ctx)
    if not _is_dataframe_valid(df_trials):
        ctx.logger.warning("Condition: trial table missing; skipping.")
        return pd.DataFrame()

    feature_cols = [c for c in df_trials.columns if str(c).startswith(FEATURE_COLUMN_PREFIXES)]
    feature_cols = _cache.get_filtered_feature_cols(feature_cols, ctx, "condition")
    if not feature_cols:
        ctx.logger.info("Condition: no feature columns found; skipping.")
        return pd.DataFrame()

    result_dfs = []
    
    compare_values = get_config_value(ctx.config, "behavior_analysis.condition.compare_values", [])
    use_multigroup = isinstance(compare_values, (list, tuple)) and len(compare_values) > 2
    
    if use_multigroup:
        multigroup_df = stage_condition_multigroup(ctx, config, df_trials=df_trials, feature_cols=feature_cols)
        if multigroup_df is not None and not multigroup_df.empty:
            result_dfs.append(multigroup_df)
    else:
        col_df = stage_condition_column(ctx, config, df_trials=df_trials, feature_cols=feature_cols)
        if col_df is not None and not col_df.empty:
            result_dfs.append(col_df)

    compare_windows = get_config_value(ctx.config, "behavior_analysis.condition.compare_windows", [])
    win_df = stage_condition_window(
        ctx,
        config,
        df_trials=df_trials,
        feature_cols=feature_cols,
        compare_windows=compare_windows if isinstance(compare_windows, list) else None,
    )
    if win_df is not None and not win_df.empty:
        result_dfs.append(win_df)

    if result_dfs:
        return pd.concat(result_dfs, ignore_index=True)
    return pd.DataFrame()


def stage_condition_multigroup(
    ctx: BehaviorContext,
    config: Any,
    df_trials: Optional[pd.DataFrame] = None,
    feature_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Run multi-group condition comparison (3+ groups).
    
    Computes all pairwise Mann-Whitney U tests between groups with FDR correction.
    Results are saved to condition_effects_multigroup*.tsv.
    
    When overwrite=false, includes compare_column name in output filename to allow
    multiple comparisons without overwriting previous results.
    """
    from eeg_pipeline.utils.analysis.stats.effect_size import compute_multigroup_condition_effects
    
    if df_trials is None:
        df_trials = _load_trial_table_df(ctx)
    if not _is_dataframe_valid(df_trials):
        ctx.logger.warning("Condition multigroup: trial table missing; skipping.")
        return pd.DataFrame()
    
    if feature_cols is None:
        feature_cols = _get_feature_columns(df_trials, ctx, "condition")
    if not feature_cols:
        ctx.logger.info("Condition multigroup: no feature columns found; skipping.")
        return pd.DataFrame()

    primary_unit = str(
        get_config_value(ctx.config, "behavior_analysis.condition.primary_unit", "trial") or "trial"
    ).strip().lower()
    allow_iid_trials = get_config_bool(ctx.config, "behavior_analysis.statistics.allow_iid_trials", False)
    if primary_unit in {"trial", "trialwise"} and not allow_iid_trials:
        raise ValueError(
            "Trial-level multigroup condition comparisons assume i.i.d trials. "
            "Use run-level aggregation (behavior_analysis.condition.primary_unit=run_mean) "
            "or set behavior_analysis.statistics.allow_iid_trials=true to override (not recommended)."
        )
    
    compare_column = _resolve_condition_compare_column(df_trials, ctx.config)
    compare_values = get_config_value(ctx.config, "behavior_analysis.condition.compare_values", [])
    overwrite = get_config_bool(ctx.config, "behavior_analysis.condition.overwrite", True)
    
    compare_labels = get_config_value(ctx.config, "behavior_analysis.condition.compare_labels", None)
    
    if not isinstance(compare_values, (list, tuple)) or len(compare_values) < 3:
        ctx.logger.info("Condition multigroup: requires 3+ compare_values; skipping.")
        return pd.DataFrame()
    
    if compare_column not in df_trials.columns:
        ctx.logger.warning(f"Condition multigroup: column '{compare_column}' not found; skipping.")
        return pd.DataFrame()

    run_col = str(get_config_value(ctx.config, "behavior_analysis.run_adjustment.column", "run_id") or "run_id").strip()
    use_run_unit = primary_unit in {"run", "run_mean", "runmean", "run_level"}
    if use_run_unit:
        if run_col not in df_trials.columns:
            raise ValueError(
                f"Run-level multigroup condition comparisons requested but run column '{run_col}' is missing."
            )
        ctx.logger.info(
            "Condition multigroup: aggregating to run×condition level (primary_unit=%s)",
            primary_unit,
        )
        group_keys = [run_col, compare_column]
        df_agg = (
            df_trials.groupby(group_keys, dropna=True)[feature_cols]
            .mean(numeric_only=True)
            .reset_index()
        )
        cell_counts = (
            df_trials.groupby(group_keys, dropna=True)
            .size()
            .rename("n_trials_cell")
            .reset_index()
        )
        df_trials = df_agg.merge(cell_counts, on=group_keys, how="left")
    
    if isinstance(compare_labels, (list, tuple)) and len(compare_labels) >= len(compare_values):
        group_labels = [str(l).strip() for l in compare_labels[:len(compare_values)]]
    else:
        group_labels = [str(v) for v in compare_values]
    
    column_values = df_trials[compare_column]
    group_masks = {}
    
    for val, label in zip(compare_values, group_labels):
        try:
            numeric_val = float(val)
            mask = (pd.to_numeric(column_values, errors="coerce") == numeric_val).values
        except (ValueError, TypeError):
            val_str = str(val).strip().lower()
            mask = (column_values.astype(str).str.strip().str.lower() == val_str).values
        
        if np.any(mask):
            group_masks[label] = mask
            ctx.logger.debug(f"  Group '{label}': {np.sum(mask)} trials")
    
    if len(group_masks) < 2:
        ctx.logger.warning("Condition multigroup: fewer than 2 groups have data; skipping.")
        return pd.DataFrame()
    
    ctx.logger.info(
        f"Condition multigroup ({compare_column}): {len(group_masks)} groups, "
        f"{len(feature_cols)} features"
    )
    
    features_df = df_trials[feature_cols].copy()
    
    multigroup_df = compute_multigroup_condition_effects(
        features_df=features_df,
        group_masks=group_masks,
        group_labels=group_labels,
        fdr_alpha=config.fdr_alpha,
        logger=ctx.logger,
        config=ctx.config,
    )
    
    if multigroup_df is not None and not multigroup_df.empty:
        multigroup_df["compare_column"] = compare_column
        suffix = _feature_suffix_from_context(ctx)
        # Always include condition column name in filename
        out_dir = _get_stats_subfolder(ctx, "condition_effects")
        filename = f"condition_effects_multigroup{suffix}_{compare_column}.tsv"
        path = out_dir / filename
        _write_stats_table(ctx, multigroup_df, path)
        ctx.logger.info(f"Saved multi-group condition effects to {path}")
    
    return multigroup_df


def _run_window_comparison(
    ctx: BehaviorContext,
    df_trials: pd.DataFrame,
    feature_cols: List[str],
    windows: List[str],
    min_samples: int,
    fdr_alpha: float,
    suffix: str,
) -> pd.DataFrame:
    """Run paired window comparison on feature columns.
    
    Compares features between two time windows (e.g., baseline vs active).
    Uses vectorized computation for effect sizes and batch processing.
    """
    from eeg_pipeline.utils.analysis.stats.fdr import fdr_bh
    from scipy import stats as sp_stats

    if len(windows) < 2:
        ctx.logger.warning("Window comparison requires at least 2 windows")
        return pd.DataFrame()

    window1, window2 = windows[0], windows[1]

    run_col = str(get_config_value(ctx.config, "behavior_analysis.run_adjustment.column", "run_id") or "run_id").strip()
    wc_primary_unit = str(
        get_config_value(ctx.config, "behavior_analysis.condition.window_comparison.primary_unit", "trial") or "trial"
    ).strip().lower()
    allow_iid_trials = get_config_bool(ctx.config, "behavior_analysis.statistics.allow_iid_trials", False)
    if wc_primary_unit in {"trial", "trialwise"} and not allow_iid_trials:
        raise ValueError(
            "Trial-level window comparisons assume i.i.d trials. "
            "Use run-level aggregation (behavior_analysis.condition.window_comparison.primary_unit=run_mean) "
            "or set behavior_analysis.statistics.allow_iid_trials=true to override (not recommended)."
        )
    use_run_unit = wc_primary_unit in {"run", "run_mean", "runmean", "run_level"}
    if use_run_unit and run_col not in df_trials.columns:
        raise ValueError(
            f"Run-level window comparison requested but run column '{run_col}' is missing from trial table."
        )

    from eeg_pipeline.domain.features.naming import NamingSchema

    # Parse NamingSchema to find window-specific features.
    # This avoids substring collisions when window names appear elsewhere in feature identifiers.
    window1_features: Dict[Tuple[str, str, str, str, str], str] = {}
    window2_features: Dict[Tuple[str, str, str, str, str], str] = {}

    prefixes = sorted(FEATURE_COLUMN_PREFIXES, key=len, reverse=True)
    w1 = str(window1).strip().lower()
    w2 = str(window2).strip().lower()

    n_unparseable = 0
    for col in feature_cols:
        col_str = str(col)
        matched_prefix = next((p for p in prefixes if col_str.startswith(p)), None)
        raw_name = col_str[len(matched_prefix) :] if matched_prefix else col_str

        parsed = NamingSchema.parse(raw_name)
        if not parsed.get("valid"):
            n_unparseable += 1
            continue

        # When prefix is stripped, parse() shifts fields: segment becomes group, band becomes segment.
        # Example: "power_baseline_alpha_ch_Fp1_mean" -> strip "power_" -> "baseline_alpha_ch_Fp1_mean"
        # Parse interprets: group="baseline", segment="alpha", band=""
        # Correct interpretation: group="power", segment="baseline", band="alpha"
        parsed_group = str(parsed.get("group") or "").strip().lower()
        parsed_segment = str(parsed.get("segment") or "").strip().lower()
        parsed_band = str(parsed.get("band") or "").strip()
        
        if matched_prefix and parsed_group in {w1, w2}:
            # Prefix was stripped: parsed.group is actually the segment
            seg = parsed_group
            band = parsed_segment if parsed_segment else parsed_band
        else:
            # No prefix or group doesn't match window: use parsed segment
            seg = parsed_segment if parsed_segment else parsed_group
            band = parsed_band if parsed_band else parsed_segment
        
        seg = seg.strip().lower()
        if seg not in {w1, w2}:
            continue

        group_name = matched_prefix.rstrip("_") if matched_prefix else str(parsed.get("group") or "")
        key = (
            group_name,
            band,
            str(parsed.get("scope") or ""),
            str(parsed.get("identifier") or ""),
            str(parsed.get("stat") or ""),
        )
        if seg == w1:
            window1_features[key] = col_str
        else:
            window2_features[key] = col_str

    if n_unparseable and ctx.logger is not None:
        ctx.logger.debug(
            "Window comparison: skipped %d unparseable feature columns (NamingSchema invalid).",
            int(n_unparseable),
        )

    # Find matching pairs
    common_bases = sorted(set(window1_features.keys()) & set(window2_features.keys()))
    
    if not common_bases:
        # Diagnose why no matches: check if it's a stat mismatch
        w1_stats = {k[4] for k in window1_features.keys()}
        w2_stats = {k[4] for k in window2_features.keys()}
        common_stats = w1_stats & w2_stats
        
        if not common_stats:
            ctx.logger.warning(
                f"No matching feature pairs found for windows {window1} and {window2}. "
                f"Reason: different stat types ({window1} has {sorted(w1_stats)}, {window2} has {sorted(w2_stats)}). "
                f"Window comparisons require features with matching (group, band, scope, identifier, stat)."
            )
        else:
            ctx.logger.warning(
                f"No matching feature pairs found for windows {window1} and {window2}. "
                f"Found {len(window1_features)} {window1} features and {len(window2_features)} {window2} features, "
                f"but none share the same (group, band, scope, identifier, stat) combination."
            )
        return pd.DataFrame()

    n_pairs = len(common_bases)
    ctx.logger.info(f"Window comparison: {n_pairs} feature pairs for {window1} vs {window2} (vectorized)")

    # Build aligned column lists for batch processing
    cols1 = [window1_features[base] for base in common_bases]
    cols2 = [window2_features[base] for base in common_bases]
    
    # Extract data matrices for vectorized operations
    v1_matrix = df_trials[cols1].to_numpy(dtype=np.float64, na_value=np.nan)
    v2_matrix = df_trials[cols2].to_numpy(dtype=np.float64, na_value=np.nan)

    # Compute statistics vectorized
    valid_mask = np.isfinite(v1_matrix) & np.isfinite(v2_matrix)
    n_valid_per_pair = valid_mask.sum(axis=0)

    # Compute means and stds vectorized
    mean_v1 = np.nanmean(v1_matrix, axis=0)
    mean_v2 = np.nanmean(v2_matrix, axis=0)
    std_v1 = np.nanstd(v1_matrix, axis=0, ddof=1)
    std_v2 = np.nanstd(v2_matrix, axis=0, ddof=1)

    # Compute effect sizes vectorized
    mean_diff, std_diff, cohens_d, hedges_g = _compute_batch_pairwise_effect_sizes(v1_matrix, v2_matrix)

    # Wilcoxon tests must be done per-pair (no vectorized scipy version)
    records: List[Dict[str, Any]] = []

    for i, base_name in enumerate(common_bases):
        col1 = cols1[i]
        col2 = cols2[i]
        n_valid = int(n_valid_per_pair[i])

        if n_valid < max(min_samples, 2):
            continue

        v1_valid = v1_matrix[valid_mask[:, i], i]
        v2_valid = v2_matrix[valid_mask[:, i], i]

        try:
            stat, p_val = sp_stats.wilcoxon(v1_valid, v2_valid)
        except (ValueError, TypeError, sp_stats.Error) as exc:
            raise RuntimeError(
                f"Wilcoxon failed for window comparison '{window1}' vs '{window2}' "
                f"(col1={col1}, col2={col2})"
            ) from exc

        stat_run = np.nan
        p_val_run = np.nan
        n_runs = np.nan
        if use_run_unit and run_col in df_trials.columns:
            df_run = (
                pd.DataFrame(
                    {
                        "run": df_trials[run_col],
                        "v1": v1_matrix[:, i],
                        "v2": v2_matrix[:, i],
                    }
                )
                .dropna()
            )
            run_means = df_run.groupby("run", dropna=True)[["v1", "v2"]].mean(numeric_only=True)
            n_runs = int(len(run_means))
            if n_runs >= 2:
                try:
                    stat_run, p_val_run = sp_stats.wilcoxon(
                        run_means["v1"].to_numpy(),
                        run_means["v2"].to_numpy(),
                    )
                except (ValueError, TypeError, sp_stats.Error, KeyError) as exc:
                    raise RuntimeError(
                        f"Run-level Wilcoxon failed for window comparison '{window1}' vs '{window2}' "
                        f"(col1={col1}, col2={col2}, run_col={run_col})"
                    ) from exc

        col1_prefix = next((p for p in prefixes if str(col1).startswith(p)), "")
        col1_raw = str(col1)[len(col1_prefix) :] if col1_prefix else str(col1)

        records.append(
            {
                "feature": "::".join(str(x) for x in base_name),
                "feature_col_window1": col1,
                "feature_col_window2": col2,
                "feature_type": _cache.get_feature_type(col1_raw, ctx.config),
                "analysis_kind": "condition_window",
                "comparison_type": "window",
                "window1": window1,
                "window2": window2,
                "n_pairs": n_valid,
                "n_runs": n_runs,
                "mean_window1": float(mean_v1[i]),
                "mean_window2": float(mean_v2[i]),
                "std_window1": float(std_v1[i]),
                "std_window2": float(std_v2[i]),
                "mean_diff": float(mean_diff[i]),
                "std_diff": float(std_diff[i]) if np.isfinite(std_diff[i]) else np.nan,
                "statistic": float(stat),
                "p_raw": float(p_val),
                "statistic_run": float(stat_run) if np.isfinite(stat_run) else np.nan,
                "p_value_run": float(p_val_run) if np.isfinite(p_val_run) else np.nan,
                "cohens_d": float(cohens_d[i]),
                "hedges_g": float(hedges_g[i]),
            }
        )

    if not records:
        ctx.logger.info("Window comparison: no valid feature pairs with sufficient samples")
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # Standardize p-value columns
    if use_run_unit and "p_value_run" in df.columns:
        df["p_primary"] = pd.to_numeric(df["p_value_run"], errors="coerce")
        df["p_primary_source"] = "run_mean"
    else:
        df["p_primary"] = pd.to_numeric(df["p_raw"], errors="coerce")
        df["p_primary_source"] = "trial"

    # Use unified FDR
    df = _compute_unified_fdr(
        ctx,
        ctx.config,
        df,
        p_col="p_primary",
        family_cols=["feature_type", "analysis_kind"],
        analysis_type="condition_window",
    )

    return df


###################################################################
# Temporal Stage - Single Responsibility Components
###################################################################


def stage_temporal_tfr(ctx: BehaviorContext) -> Optional[Dict[str, Any]]:
    """Compute time-frequency representation correlations.
    
    Single responsibility: TFR computation only.
    """
    from eeg_pipeline.analysis.behavior.api import compute_time_frequency_from_context

    ctx.logger.info("Computing time-frequency correlations...")
    return compute_time_frequency_from_context(ctx)


def stage_temporal_stats(
    ctx: BehaviorContext,
    selected_features: Optional[List[str]] = None,
) -> Dict[str, Optional[Dict[str, Any]]]:
    """Compute temporal statistics (power, ITPC, ERDS correlations).
    
    Single responsibility: Temporal statistics computation.
    
    Family-wise correction modes (behavior_analysis.temporal.correction_method):
    - "fdr" (default): FDR correction across time×freq cells
    - "cluster": Cluster-based permutation test (preferred for dense grids)
    - "bonferroni": Conservative Bonferroni correction
    - "none": No correction (use with caution)
    """
    from eeg_pipeline.analysis.behavior.api import compute_temporal_from_context
    from eeg_pipeline.utils.analysis.stats.temporal import compute_itpc_temporal_from_context
    from eeg_pipeline.infra.paths import ensure_dir
    from statsmodels.stats.multitest import multipletests

    if selected_features is None:
        selected_features = ctx.selected_feature_files or ctx.feature_categories
    if not selected_features:
        selected_features = ["power", "itpc", "erds"]
        ctx.logger.info(f"No feature files specified, defaulting to: {selected_features}")
    
    # Family-wise correction settings
    correction_method = str(get_config_value(
        ctx.config, "behavior_analysis.temporal.correction_method", "fdr"
    )).strip().lower()
    fdr_alpha = get_config_float(ctx.config, "behavior_analysis.statistics.fdr_alpha", 0.05)
    ctx.logger.info(f"Temporal: using {correction_method} correction (alpha={fdr_alpha})")

    results: Dict[str, Optional[Dict[str, Any]]] = {
        "power": None,
        "itpc": None,
        "erds": None,
    }

    ctx.logger.info("Computing temporal correlations by condition...")
    results["power"] = compute_temporal_from_context(ctx)

    if "itpc" in selected_features:
        ctx.logger.info("Computing ITPC temporal correlations...")
        itpc_results = compute_itpc_temporal_from_context(ctx)
        results["itpc"] = itpc_results
        if itpc_results:
            ctx.logger.info(
                f"ITPC temporal: {itpc_results.get('n_tests', 0)} tests, "
                f"{itpc_results.get('n_sig_raw', 0)} significant"
            )

    if "erds" in selected_features:
        from eeg_pipeline.utils.analysis.stats.temporal import compute_erds_temporal_from_context
        ctx.logger.info("Computing ERDS temporal correlations...")
        erds_results = compute_erds_temporal_from_context(ctx)
        results["erds"] = erds_results
        if erds_results:
            ctx.logger.info(
                f"ERDS temporal: {erds_results.get('n_tests', 0)} tests, "
                f"{erds_results.get('n_sig_raw', 0)} significant"
            )

    all_temporal_records = []
    for res in results.values():
        if res and "records" in res:
            all_temporal_records.extend(res["records"])
    
    if all_temporal_records:
        out_dir = _get_stats_subfolder(ctx, "temporal_correlations")
        method_suffix = "_spearman" if ctx.use_spearman else "_pearson"
        
        # Apply family-wise correction across all temporal tests
        df_temporal = pd.DataFrame(all_temporal_records)
        
        # Ensure p column exists for correction
        if "p" in df_temporal.columns and "p_raw" not in df_temporal.columns:
            df_temporal["p_raw"] = df_temporal["p"]
        
        if "p_raw" in df_temporal.columns:
            p_vals = pd.to_numeric(df_temporal["p_raw"], errors="coerce").fillna(1.0).to_numpy()
            
            if correction_method == "fdr":
                reject, p_corrected, _, _ = multipletests(p_vals, alpha=fdr_alpha, method="fdr_bh")
                df_temporal["p_fdr"] = p_corrected
                df_temporal["sig_fdr"] = reject
                df_temporal["p_primary"] = df_temporal["p_fdr"]
                n_sig = int(reject.sum())
                ctx.logger.info(f"Temporal FDR: {n_sig}/{len(p_vals)} significant at alpha={fdr_alpha}")
                
            elif correction_method == "bonferroni":
                reject, p_corrected, _, _ = multipletests(p_vals, alpha=fdr_alpha, method="bonferroni")
                df_temporal["p_bonferroni"] = p_corrected
                df_temporal["sig_bonferroni"] = reject
                df_temporal["p_primary"] = df_temporal["p_bonferroni"]
                n_sig = int(reject.sum())
                ctx.logger.info(f"Temporal Bonferroni: {n_sig}/{len(p_vals)} significant at alpha={fdr_alpha}")
                
            elif correction_method == "cluster":
                if "p_cluster" in df_temporal.columns:
                    p_cluster_vals = pd.to_numeric(df_temporal["p_cluster"], errors="coerce").fillna(1.0).to_numpy()
                    df_temporal["p_primary"] = p_cluster_vals
                    if "cluster_significant" in df_temporal.columns:
                        df_temporal["sig_cluster"] = df_temporal["cluster_significant"].fillna(False).astype(bool)
                    else:
                        df_temporal["sig_cluster"] = p_cluster_vals < fdr_alpha
                    n_sig = int(df_temporal["sig_cluster"].sum())
                    ctx.logger.info(f"Temporal cluster: {n_sig}/{len(p_cluster_vals)} significant at alpha={fdr_alpha}")
                else:
                    ctx.logger.warning(
                        "Temporal: cluster correction requested but no p_cluster column present; falling back to FDR."
                    )
                    reject, p_corrected, _, _ = multipletests(p_vals, alpha=fdr_alpha, method="fdr_bh")
                    df_temporal["p_fdr"] = p_corrected
                    df_temporal["sig_fdr"] = reject
                    df_temporal["p_primary"] = df_temporal["p_fdr"]
                    df_temporal["correction_note"] = "fdr_fallback_from_cluster_missing_p_cluster"
                
            elif correction_method == "none":
                ctx.logger.warning("Temporal: no multiple comparison correction applied (use with caution)")
                df_temporal["sig_raw"] = p_vals < fdr_alpha
                df_temporal["p_primary"] = df_temporal["p_raw"]
            
            df_temporal["correction_method"] = correction_method
        
        # Save combined temporal correlations using consistent naming (like regular correlations)
        combined_path = out_dir / f"temporal_correlations{method_suffix}.parquet"
        _write_stats_table(ctx, df_temporal, combined_path)
        ctx.logger.info(
            f"Saved combined temporal correlations: {len(all_temporal_records)} tests -> {combined_path.name}"
        )
        
        # Save normalized results for temporal correlations (consistent with other correlation outputs)
        normalized_records = []
        method = "spearman" if ctx.use_spearman else "pearson"
        method_label = format_correlation_method_label(method, None)
        target_label = str(get_config_value(
            ctx.config, "behavior_analysis.temporal.target_column", ""
        ) or "").strip() or "rating"
        for idx, row in df_temporal.iterrows():
            normalized_records.append({
                "analysis_type": "temporal_correlations",
                "feature_id": row.get("channel", ""),
                "feature_type": row.get("feature", "temporal"),
                "target": target_label,
                "method": method,
                "robust_method": None,
                "method_label": method_label,
                "n": row.get("n", np.nan),
                "r": row.get("r", np.nan),
                "p_raw": row.get("p_raw", row.get("p", np.nan)),
                "p_primary": row.get("p_primary", row.get("p_raw", row.get("p", np.nan))),
                "p_fdr": row.get("p_fdr", np.nan),
                "notes": f"band={row.get('band', '')}, time={row.get('time_start', '')}–{row.get('time_end', '')}s, condition={row.get('condition', '')}",
            })
        
        if normalized_records:
            df_normalized = pd.DataFrame(normalized_records)
            normalized_path = out_dir / f"normalized_results{method_suffix}.parquet"
            _write_stats_table(ctx, df_normalized, normalized_path)
            ctx.logger.debug(
                f"Temporal normalized results: {len(normalized_records)} records -> {normalized_path.name}"
            )

    return results




def stage_cluster(ctx: BehaviorContext, config: Any) -> Dict[str, Any]:
    from eeg_pipeline.analysis.behavior.api import run_cluster_test_from_context

    ctx.logger.info("Running cluster permutation tests...")
    ctx.n_perm = config.n_permutations
    results = run_cluster_test_from_context(ctx)
    return results if results else {"status": "completed"}


###################################################################
# Advanced Stage - Single Responsibility Components
###################################################################


def stage_mediation(ctx: BehaviorContext, config: Any) -> pd.DataFrame:
    """Run mediation analysis: test if neural features mediate the temperature→rating relationship.

    Single responsibility: Mediation analysis (indirect effects).
    """
    from eeg_pipeline.analysis.behavior.api import run_mediation_analysis

    df_trials = _load_trial_table_df(ctx)
    if not _is_dataframe_valid(df_trials):
        ctx.logger.info("Mediation: trial table missing; skipping.")
        return pd.DataFrame()

    required_columns = {"temperature", "rating"}
    missing_columns = required_columns - set(df_trials.columns)
    if missing_columns:
        ctx.logger.warning(
            "Mediation: requires %s columns; missing: %s. Skipping.",
            required_columns, missing_columns
        )
        return pd.DataFrame()

    feature_cols = _get_feature_columns(df_trials, ctx, "mediation")

    # Early exit: check if we have enough valid features
    should_skip, skip_reason = _check_early_exit_conditions(df_trials, feature_cols, min_features=1, min_trials=10)
    if should_skip:
        ctx.logger.info(f"Mediation: skipping due to {skip_reason}")
        return pd.DataFrame()

    if not feature_cols:
        ctx.logger.info("Mediation: no feature columns found; skipping.")
        return pd.DataFrame()

    ctx.logger.info("Running mediation analysis...")
    n_bootstrap = get_config_int(ctx.config, "behavior_analysis.mediation.n_bootstrap", 1000)
    n_permutations = get_config_int(ctx.config, "behavior_analysis.mediation.n_permutations", 0)
    allow_iid_trials = get_config_bool(ctx.config, "behavior_analysis.statistics.allow_iid_trials", False)
    p_primary_mode = str(
        get_config_value(ctx.config, "behavior_analysis.mediation.p_primary_mode", "perm_if_available") or "perm_if_available"
    ).strip().lower()
    min_effect_size = get_config_float(ctx.config, "behavior_analysis.mediation.min_effect_size", 0.05)
    max_mediators = get_config_value(ctx.config, "behavior_analysis.mediation.max_mediators", None)
    run_col = str(get_config_value(ctx.config, "behavior_analysis.run_adjustment.column", "run_id") or "run_id").strip()
    perm_scheme = str(get_config_value(ctx.config, "behavior_analysis.permutation.scheme", "shuffle") or "shuffle").strip().lower()

    if max_mediators is not None:
        max_mediators = int(max_mediators)
        variances = df_trials[feature_cols].var()
        mediators = variances.nlargest(max(1, max_mediators)).index.tolist()
        ctx.logger.info("Limiting to top %d mediators by variance", max_mediators)
    else:
        mediators = feature_cols
        ctx.logger.info("Testing all %d features as mediators (no limit)", len(mediators))

    groups_for_resampling = None
    if getattr(ctx, "group_ids", None) is not None:
        groups_candidate = np.asarray(ctx.group_ids)
        if len(groups_candidate) == len(df_trials):
            groups_for_resampling = groups_candidate
        else:
            ctx.logger.warning(
                "Mediation: ignoring ctx.group_ids length=%d because trial table has %d rows.",
                len(groups_candidate),
                len(df_trials),
            )
    if groups_for_resampling is None and run_col in df_trials.columns:
        groups_for_resampling = df_trials[run_col].to_numpy()
    groups_for_resampling = _sanitize_permutation_groups(
        groups_for_resampling,
        ctx.logger,
        "Mediation",
    )
    if not allow_iid_trials:
        if n_permutations <= 0:
            raise ValueError(
                "Mediation requires non-i.i.d inference under repeated measures. "
                "Set behavior_analysis.mediation.n_permutations > 0 or set "
                "behavior_analysis.statistics.allow_iid_trials=true to override (not recommended)."
            )
        if groups_for_resampling is None:
            raise ValueError(
                "Mediation requires grouped resampling labels for non-i.i.d trials. "
                "Provide behavior_analysis.run_adjustment.column in the trial table (or ctx.group_ids), "
                "or set behavior_analysis.statistics.allow_iid_trials=true to override (not recommended)."
            )

    result = run_mediation_analysis(
        df_trials,
        "temperature",
        mediators,
        "rating",
        n_bootstrap=n_bootstrap,
        n_permutations=n_permutations,
        groups=groups_for_resampling,
        permutation_scheme=perm_scheme,
        min_effect_size=min_effect_size,
    )
    if result is None or result.empty:
        return pd.DataFrame()

    med_df = result.copy()
    if "analysis_kind" not in med_df.columns:
        med_df["analysis_kind"] = "mediation"

    if "p_raw" not in med_df.columns:
        med_df["p_raw"] = pd.to_numeric(
            med_df.get("sobel_p", med_df.get("p_value", np.nan)),
            errors="coerce",
        )
    if "p_ab_perm" not in med_df.columns:
        med_df["p_ab_perm"] = np.nan

    use_perm = p_primary_mode in {"perm", "permutation", "perm_if_available", "permutation_if_available"}
    if use_perm:
        perm_p = pd.to_numeric(med_df["p_ab_perm"], errors="coerce")
        raw_p = pd.to_numeric(med_df["p_raw"], errors="coerce")
        if not allow_iid_trials:
            med_df["p_primary"] = perm_p.where(perm_p.notna(), np.nan)
            med_df["p_primary_source"] = np.where(perm_p.notna(), "perm", "perm_missing_required")
        else:
            med_df["p_primary"] = perm_p.where(perm_p.notna(), raw_p)
            med_df["p_primary_source"] = np.where(perm_p.notna(), "perm", "sobel")
    else:
        med_df["p_primary"] = pd.to_numeric(med_df["p_raw"], errors="coerce")
        med_df["p_primary_source"] = "sobel"

    mediator_col = "mediator" if "mediator" in med_df.columns else ("feature" if "feature" in med_df.columns else None)
    if mediator_col is not None:
        try:
            med_df["feature_type"] = [
                _cache.get_feature_type(str(m), ctx.config) for m in med_df[mediator_col].astype(str).tolist()
            ]
        except Exception:
            med_df["feature_type"] = "unknown"
    else:
        med_df["feature_type"] = "unknown"

    med_df = _compute_unified_fdr(
        ctx,
        config,
        med_df,
        p_col="p_primary",
        family_cols=["feature_type", "analysis_kind"],
        analysis_type="mediation",
    )
    med_df["significant_mediation"] = pd.to_numeric(med_df.get("p_fdr", np.nan), errors="coerce") < float(
        getattr(config, "fdr_alpha", 0.05)
    )
    return med_df


def stage_mixed_effects(ctx: BehaviorContext, config: Any) -> pd.DataFrame:
    """Run mixed-effects analysis.
    
    Single responsibility: Mixed-effects models (requires multiple subjects).
    
    Note: Subject-level behavior analysis should skip this stage.
    Call run_group_level_mixed_effects() for proper multi-subject analysis.
    """
    ctx.logger.warning(
        "Skipping mixed-effects analysis in subject-level mode; "
        "run this at group level with multiple subjects via run_group_level()."
    )
    return pd.DataFrame()


###################################################################
# Group-Level Analysis (Multi-Subject)
###################################################################


def _find_trial_table_path(stats_dir: Path, feature_files: Optional[List[str]] = None) -> Optional[Path]:
    """Find trial table file with feature suffix.
    
    If feature_files is provided, builds exact filename. Otherwise searches for:
    - trials_*.{parquet,tsv}
    - trials.{parquet,tsv}
    """
    trial_table_roots = sorted(p for p in stats_dir.glob("trial_table*") if p.is_dir())
    if not trial_table_roots:
        return None
    
    if feature_files:
        suffix = "_" + "_".join(sorted(str(x) for x in feature_files))
        fname = f"trials{suffix}"
        feature_dir = _feature_folder_from_list(feature_files)
        candidates: List[Path] = []
        for root in trial_table_roots:
            candidates.append(root / feature_dir / f"{fname}.parquet")
            candidates.append(root / feature_dir / f"{fname}.tsv")
        existing = [p for p in candidates if p.exists()]
        if len(existing) == 0:
            return None
        if len(existing) > 1:
            raise ValueError(
                f"Multiple trial table files found in {stats_dir} for {feature_dir}: {existing}. "
                "Specify a unique stats_dir or clean old outputs to disambiguate."
            )
        return existing[0]
    
    pattern_paths: List[Path] = []
    for root in trial_table_roots:
        pattern_paths.extend(list(root.glob("*/trials_*.parquet")))
        pattern_paths.extend(list(root.glob("*/trials_*.tsv")))
        pattern_paths.extend(list(root.glob("*/trials.parquet")))
        pattern_paths.extend(list(root.glob("*/trials.tsv")))
    if len(pattern_paths) == 0:
        return None
    if len(pattern_paths) > 1:
        raise ValueError(
            f"Multiple trial table files found in {stats_dir}: {pattern_paths}. "
            "Specify feature files to disambiguate."
        )
    return pattern_paths[0]


def run_group_level_mixed_effects(
    subjects: List[str],
    deriv_root: Path,
    config: Any,
    logger: Any,
    random_effects: str = "intercept",
    max_features: int = 50,
    fdr_alpha: float = 0.05,
) -> MixedEffectsResult:
    """Run proper mixed-effects models across all subjects.
    
    Fits statsmodels MixedLM with subject as random effect for each feature.
    Uses hierarchical FDR correction by feature family.
    
    Parameters
    ----------
    subjects : List[str]
        List of subject IDs
    deriv_root : Path
        Derivatives root directory
    config : Any
        Configuration object
    logger : Any
        Logger instance
    random_effects : str
        Random effects structure: 'intercept' or 'slope'
    max_features : int
        Maximum features to test
    fdr_alpha : float
        FDR alpha level
    
    Returns
    -------
    MixedEffectsResult
        Results with coefficients, p-values, and family structure
    """
    from eeg_pipeline.infra.paths import deriv_stats_path
    from eeg_pipeline.infra.tsv import read_table

    import statsmodels.formula.api as smf
    feature_files = get_config_value(config, "behavior_analysis.feature_files", None)
    if isinstance(feature_files, str):
        feature_files = [feature_files]
    elif feature_files is None:
        feature_files = get_config_value(config, "behavior_analysis.feature_categories", None)
        if isinstance(feature_files, str):
            feature_files = [feature_files]
    
    all_trials: List[pd.DataFrame] = []
    
    for sub in subjects:
        stats_dir = deriv_stats_path(deriv_root, sub)
        trial_path = _find_trial_table_path(stats_dir, feature_files=feature_files)
        
        if trial_path is None:
            logger.warning(f"No trial table for sub-{sub}; skipping.")
            continue
        
        df = read_table(trial_path)
        if df.empty:
            continue
        
        df["subject_id"] = sub
        all_trials.append(df)
    
    if len(all_trials) < 2:
        logger.warning("Mixed-effects requires >=2 subjects; only %d found.", len(all_trials))
        return MixedEffectsResult(df=pd.DataFrame(), metadata={"status": "insufficient_subjects"})
    
    combined = pd.concat(all_trials, ignore_index=True)
    logger.info("Mixed-effects: %d subjects, %d total trials", len(all_trials), len(combined))
    
    if "rating" not in combined.columns:
        raise KeyError("Mixed-effects requires a 'rating' column in the combined trial table.")
    
    feature_cols = [c for c in combined.columns if str(c).startswith(FEATURE_COLUMN_PREFIXES)]
    
    if len(feature_cols) > max_features:
        variances = combined[feature_cols].var()
        feature_cols = variances.nlargest(max_features).index.tolist()
        logger.info("Mixed-effects: limited to top %d features by variance", max_features)
    
    n_subjects = combined["subject_id"].nunique()
    if n_subjects < 3:
        logger.warning(
            "Mixed-effects: only %d subjects. Convergence warnings are expected with small sample sizes. "
            "Results may be unreliable.",
            n_subjects
        )
    
    run_col_cfg = str(get_config_value(config, "behavior_analysis.run_adjustment.column", "run_id") or "run_id").strip()
    run_col_candidates = [run_col_cfg, "run_id", "run", "block"]
    run_col = next((c for c in run_col_candidates if c and c in combined.columns), None)
    trial_order_col = next((c for c in ("trial_index_within_group", "trial_index") if c in combined.columns), None)
    include_temperature = bool(get_config_value(config, "behavior_analysis.mixed_effects.include_temperature", True))
    max_run_dummies = int(get_config_value(config, "behavior_analysis.run_adjustment.max_dummies", 20))

    records: List[Dict[str, Any]] = []
    family_records: List[Dict[str, Any]] = []
    
    for feat in feature_cols:
        feat_type = _cache.get_feature_type(str(feat), config)
        family_id = f"mixed_{feat_type}"
        
        model_cols = ["subject_id", "rating", feat]
        if include_temperature and "temperature" in combined.columns:
            model_cols.append("temperature")
        if trial_order_col is not None:
            model_cols.append(trial_order_col)
        if run_col is not None:
            model_cols.append(run_col)
        model_cols = list(dict.fromkeys(model_cols))

        df_valid = combined[model_cols].dropna()
        if len(df_valid) < 10 or df_valid["subject_id"].nunique() < 2:
            continue
        
        df_valid = df_valid.rename(columns={feat: "feature_value"})
        formula_terms = ["feature_value"]
        covariate_terms: List[str] = []
        if include_temperature and "temperature" in df_valid.columns and df_valid["temperature"].nunique(dropna=True) > 1:
            formula_terms.append("temperature")
            covariate_terms.append("temperature")
        if trial_order_col is not None and trial_order_col in df_valid.columns and df_valid[trial_order_col].nunique(dropna=True) > 1:
            formula_terms.append(trial_order_col)
            covariate_terms.append(trial_order_col)
        if run_col is not None and run_col in df_valid.columns:
            n_run_levels = int(df_valid[run_col].nunique(dropna=True))
            if n_run_levels > 1 and n_run_levels <= max(1, max_run_dummies + 1):
                formula_terms.append(f"C({run_col})")
                covariate_terms.append(f"C({run_col})")

        formula = "rating ~ " + " + ".join(formula_terms)
        
        try:
            if random_effects == "slope":
                model = smf.mixedlm(
                    formula,
                    df_valid,
                    groups=df_valid["subject_id"],
                    re_formula="~feature_value"
                )
            else:
                model = smf.mixedlm(
                    formula,
                    df_valid,
                    groups=df_valid["subject_id"]
                )
            
            result = model.fit(reml=True)
            
            fixed_coef = result.fe_params.get("feature_value", np.nan)
            fixed_se = result.bse.get("feature_value", np.nan)
            fixed_z = result.tvalues.get("feature_value", np.nan)
            fixed_p = result.pvalues.get("feature_value", np.nan)
            
            records.append({
                "feature": str(feat),
                "feature_type": feat_type,
                "family_id": family_id,
                "family_kind": "feature_type",
                "n_subjects": df_valid["subject_id"].nunique(),
                "n_trials": len(df_valid),
                "fixed_coef": fixed_coef,
                "fixed_se": fixed_se,
                "fixed_z": fixed_z,
                "fixed_p": fixed_p,
                "random_effects": random_effects,
                "formula": formula,
                "covariates": "|".join(covariate_terms) if covariate_terms else "",
                "aic": result.aic,
                "bic": result.bic,
                "converged": result.converged,
            })
            
            family_records.append({
                "feature": str(feat),
                "family_id": family_id,
                "family_kind": "feature_type",
            })
            
        except Exception as exc:
            logger.warning("Mixed-effects failed for feature '%s': %s", feat, exc)
            continue
    
    if not records:
        logger.warning("Mixed-effects: no valid results.")
        return MixedEffectsResult(df=pd.DataFrame(), metadata={"status": "no_valid_results"})
    
    results_df = pd.DataFrame(records)
    family_df = pd.DataFrame(family_records)
    
    from eeg_pipeline.utils.analysis.stats.fdr import hierarchical_fdr
    
    results_df = hierarchical_fdr(
        results_df,
        p_col="fixed_p",
        family_col="family_id",
        alpha=fdr_alpha,
        config=config,
    )
    
    n_sig = int((results_df["q_within_family"] < fdr_alpha).sum()) if "q_within_family" in results_df.columns else 0
    
    logger.info("Mixed-effects: %d features tested, %d significant (hierarchical FDR)", len(results_df), n_sig)
    
    family_structure = {
        "method": "hierarchical_fdr",
        "families": results_df["family_id"].unique().tolist() if "family_id" in results_df.columns else [],
        "n_families": results_df["family_id"].nunique() if "family_id" in results_df.columns else 0,
    }
    
    return MixedEffectsResult(
        df=results_df,
        n_subjects=len(all_trials),
        n_features=len(results_df),
        n_significant=n_sig,
        random_effects=random_effects,
        family_structure=family_structure,
        metadata={"status": "ok"},
    )


def run_group_level_correlations(
    subjects: List[str],
    deriv_root: Path,
    config: Any,
    logger: Any,
    use_block_permutation: bool = True,
    n_perm: int = 1000,
    fdr_alpha: float = 0.05,
) -> pd.DataFrame:
    """Run multilevel correlations across subjects with block-aware permutations.
    
    Uses block/run-restricted permutations for valid p-values under dependence.
    
    Parameters
    ----------
    subjects : List[str]
        List of subject IDs
    use_block_permutation : bool
        If True, permute within blocks/runs to preserve temporal structure
    n_perm : int
        Number of permutations
    """
    from eeg_pipeline.infra.paths import deriv_stats_path
    from eeg_pipeline.infra.tsv import read_table
    from eeg_pipeline.utils.analysis.stats.fdr import hierarchical_fdr
    from eeg_pipeline.utils.analysis.stats.permutation import permute_within_groups
    
    feature_files = get_config_value(config, "behavior_analysis.feature_files", None)
    if isinstance(feature_files, str):
        feature_files = [feature_files]
    elif feature_files is None:
        feature_files = get_config_value(config, "behavior_analysis.feature_categories", None)
        if isinstance(feature_files, str):
            feature_files = [feature_files]
    
    all_trials: List[pd.DataFrame] = []
    
    for sub in subjects:
        stats_dir = deriv_stats_path(deriv_root, sub)
        trial_path = _find_trial_table_path(stats_dir, feature_files=feature_files)
        
        if trial_path is None:
            continue
        
        df = read_table(trial_path)
        if df is None or df.empty:
            continue
        
        df["subject_id"] = sub
        all_trials.append(df)
    
    if len(all_trials) < 2:
        logger.warning("Multilevel correlations require >=2 subjects.")
        return pd.DataFrame()
    
    combined = pd.concat(all_trials, ignore_index=True)
    
    if "rating" not in combined.columns:
        logger.warning("Multilevel correlations: 'rating' column not found.")
        return pd.DataFrame()
    
    block_col = None
    for cand in ("block", "run_id", "run", "session"):
        if cand in combined.columns:
            block_col = cand
            break
    
    feature_cols = [c for c in combined.columns if str(c).startswith(FEATURE_COLUMN_PREFIXES)]
    rating = pd.to_numeric(combined["rating"], errors="coerce").to_numpy()
    subject_all = combined["subject_id"].astype(str).to_numpy()
    
    records: List[Dict[str, Any]] = []
    rng = np.random.default_rng(42)
    
    for feat in feature_cols:
        feat_type = _cache.get_feature_type(str(feat), config)
        family_id = f"corr_{feat_type}"
        
        feature_vals = pd.to_numeric(combined[feat], errors="coerce").to_numpy()
        valid_mask = np.isfinite(rating) & np.isfinite(feature_vals)
        
        if valid_mask.sum() < 10:
            continue

        feature_valid = feature_vals[valid_mask]
        rating_valid = rating[valid_mask]
        subjects_valid = subject_all[valid_mask]
        block_valid = combined[block_col].to_numpy()[valid_mask] if block_col is not None else None

        subject_counts = pd.Series(subjects_valid).value_counts()
        eligible = pd.Series(subjects_valid).map(subject_counts).to_numpy() >= 2
        if int(eligible.sum()) < 10:
            continue
        feature_valid = feature_valid[eligible]
        rating_valid = rating_valid[eligible]
        subjects_valid = subjects_valid[eligible]
        if block_valid is not None:
            block_valid = block_valid[eligible]

        feature_ws = feature_valid - pd.Series(feature_valid).groupby(pd.Series(subjects_valid)).transform("mean").to_numpy()
        rating_ws = rating_valid - pd.Series(rating_valid).groupby(pd.Series(subjects_valid)).transform("mean").to_numpy()
        finite_ws = np.isfinite(feature_ws) & np.isfinite(rating_ws)
        if int(finite_ws.sum()) < 10:
            continue
        feature_ws = feature_ws[finite_ws]
        rating_ws = rating_ws[finite_ws]
        subjects_valid = subjects_valid[finite_ws]
        if block_valid is not None:
            block_valid = block_valid[finite_ws]

        r_obs, _ = compute_correlation(
            feature_ws,
            rating_ws,
            method="spearman",
        )
        
        if not np.isfinite(r_obs):
            continue

        if use_block_permutation and block_col is not None and block_valid is not None:
            block_groups = np.array([f"{subj}::{blk}" for subj, blk in zip(subjects_valid, block_valid)], dtype=object)
            _, block_counts = np.unique(block_groups, return_counts=True)
            if np.all(block_counts >= 2):
                perm_groups = block_groups
                perm_method = "subject_block_restricted"
            else:
                perm_groups = subjects_valid
                perm_method = "subject_restricted_fallback"
        else:
            perm_groups = subjects_valid
            perm_method = "subject_restricted"

        null_rs = []
        if int(n_perm) > 0:
            for _ in range(int(n_perm)):
                perm_idx = permute_within_groups(
                    len(rating_ws),
                    rng,
                    perm_groups,
                    scheme="shuffle",
                )
                rating_perm = rating_ws[perm_idx]
                r_perm, _ = compute_correlation(feature_ws, rating_perm, method="spearman")
                if np.isfinite(r_perm):
                    null_rs.append(r_perm)
        p_perm = (np.sum(np.abs(null_rs) >= np.abs(r_obs)) + 1) / (len(null_rs) + 1) if null_rs else np.nan
        
        records.append({
            "feature": str(feat),
            "feature_type": feat_type,
            "family_id": family_id,
            "family_kind": "feature_type",
            "r": r_obs,
            "n": int(len(feature_ws)),
            "n_subjects": int(pd.Series(subjects_valid).nunique()),
            "estimator": "within_subject_centered_spearman",
            "p_perm": p_perm,
            "permutation_method": perm_method,
            "n_perm": n_perm,
        })
    
    if not records:
        return pd.DataFrame()
    
    results_df = pd.DataFrame(records)
    
    results_df = hierarchical_fdr(
        results_df,
        p_col="p_perm",
        family_col="family_id",
        alpha=fdr_alpha,
        config=config,
    )
    
    return results_df


def run_group_level_analysis(
    subjects: List[str],
    deriv_root: Path,
    config: Any,
    logger: Any,
    run_mixed_effects: bool = False,
    run_multilevel_correlations: bool = False,
    output_dir: Optional[Path] = None,
) -> GroupLevelResult:
    """Run all group-level analyses.
    
    Entry point for multi-subject behavior analysis including:
    - Mixed-effects models with hierarchical FDR
    - Multilevel correlations with block-restricted permutations
    
    Parameters
    ----------
    subjects : List[str]
        List of subject IDs
    deriv_root : Path
        Derivatives root directory
    config : Any
        Configuration object
    logger : Any
        Logger instance
    run_mixed_effects : bool, default False
        Run mixed-effects models (only runs if explicitly requested)
    run_multilevel_correlations : bool, default False
        Run multilevel correlations (opt-in, only runs if explicitly requested)
    output_dir : Path, optional
        Output directory for results
    
    Returns
    -------
    GroupLevelResult
        Aggregated group-level results
    """
    from eeg_pipeline.infra.paths import ensure_dir
    
    logger.info("="*60)
    logger.info("Group-Level Behavior Analysis")
    logger.info("="*60)
    logger.info("Subjects: %s", ", ".join(subjects))
    
    mixed_result = None
    multilevel_df = None
    
    if run_mixed_effects:
        logger.info("Running mixed-effects models...")
        mixed_result = run_group_level_mixed_effects(
            subjects=subjects,
            deriv_root=deriv_root,
            config=config,
            logger=logger,
            random_effects=get_config_value(config, "behavior_analysis.mixed_effects.random_effects", "intercept"),
            max_features=get_config_int(config, "behavior_analysis.mixed_effects.max_features", 50),
            fdr_alpha=get_config_float(config, "behavior_analysis.statistics.fdr_alpha", 0.05),
        )
        
        if output_dir and mixed_result.df is not None and not mixed_result.df.empty:
            ensure_dir(output_dir)
            out_path = output_dir / "group_mixed_effects.parquet"
            _write_parquet_with_optional_csv(
                mixed_result.df, out_path, also_save_csv=_also_save_csv_from_config(config)
            )
            logger.info("Saved mixed-effects results: %s", out_path)
    
    if run_multilevel_correlations:
        logger.info("Running multilevel correlations with block-restricted permutations...")
        multilevel_df = run_group_level_correlations(
            subjects=subjects,
            deriv_root=deriv_root,
            config=config,
            logger=logger,
            use_block_permutation=get_config_bool(config, "behavior_analysis.group_level.block_permutation", True),
            n_perm=get_config_int(config, "behavior_analysis.statistics.n_permutations", 1000),
            fdr_alpha=get_config_float(config, "behavior_analysis.statistics.fdr_alpha", 0.05),
        )
        
        if output_dir and multilevel_df is not None and not multilevel_df.empty:
            ensure_dir(output_dir)
            out_path = output_dir / "group_multilevel_correlations.parquet"
            _write_parquet_with_optional_csv(
                multilevel_df, out_path, also_save_csv=_also_save_csv_from_config(config)
            )
            logger.info("Saved multilevel correlations: %s", out_path)
    
    return GroupLevelResult(
        mixed_effects=mixed_result,
        multilevel_correlations=multilevel_df,
        n_subjects=len(subjects),
        subjects=subjects,
        metadata={"status": "ok"},
    )


def stage_moderation(ctx: BehaviorContext, config: Any) -> pd.DataFrame:
    """Run moderation analysis: test if neural features moderate the temperature→rating relationship.

    Model: rating = b0 + b1*temperature + b2*feature + b3*(temperature*feature) + error

    If b3 is significant, the feature moderates how temperature affects pain rating.
    """
    from eeg_pipeline.utils.analysis.stats.moderation import run_moderation_analysis

    suffix = _feature_suffix_from_context(ctx)
    method_label = getattr(config, "method_label", "")
    method_suffix = f"_{method_label}" if method_label else ""

    df_trials = _load_trial_table_df(ctx)
    if not _is_dataframe_valid(df_trials):
        ctx.logger.warning("Moderation: trial table missing; skipping.")
        return pd.DataFrame()

    required_columns = {"temperature", "rating"}
    missing_columns = required_columns - set(df_trials.columns)
    if missing_columns:
        ctx.logger.warning(
            "Moderation: requires %s columns; missing: %s. Skipping.",
            required_columns, missing_columns
        )
        return pd.DataFrame()

    feature_cols = _get_feature_columns(df_trials, ctx, "moderation")

    # Early exit: check if we have enough valid features
    should_skip, skip_reason = _check_early_exit_conditions(df_trials, feature_cols, min_features=1, min_trials=10)
    if should_skip:
        ctx.logger.info(f"Moderation: skipping due to {skip_reason}")
        return pd.DataFrame()

    if not feature_cols:
        ctx.logger.info("Moderation: no feature columns found; skipping.")
        return pd.DataFrame()

    max_features = getattr(config, "moderation_max_features", None)  # None = unlimited
    fdr_alpha = float(getattr(config, "fdr_alpha", 0.05))
    n_permutations = get_config_int(ctx.config, "behavior_analysis.moderation.n_permutations", 0)
    min_samples = max(int(getattr(config, "moderation_min_samples", 15)), 2)
    allow_iid_trials = get_config_bool(ctx.config, "behavior_analysis.statistics.allow_iid_trials", False)
    p_primary_mode = str(
        get_config_value(ctx.config, "behavior_analysis.moderation.p_primary_mode", "perm_if_available") or "perm_if_available"
    ).strip().lower()
    run_col = str(get_config_value(ctx.config, "behavior_analysis.run_adjustment.column", "run_id") or "run_id").strip()
    perm_scheme = str(get_config_value(ctx.config, "behavior_analysis.permutation.scheme", "shuffle") or "shuffle").strip().lower()

    groups_for_resampling = None
    if getattr(ctx, "group_ids", None) is not None:
        groups_candidate = np.asarray(ctx.group_ids)
        if len(groups_candidate) == len(df_trials):
            groups_for_resampling = groups_candidate
        else:
            ctx.logger.warning(
                "Moderation: ignoring ctx.group_ids length=%d because trial table has %d rows.",
                len(groups_candidate),
                len(df_trials),
            )
    if groups_for_resampling is None and run_col in df_trials.columns:
        groups_for_resampling = df_trials[run_col].to_numpy()
    groups_for_resampling = _sanitize_permutation_groups(
        groups_for_resampling,
        ctx.logger,
        "Moderation",
    )
    if not allow_iid_trials:
        if n_permutations <= 0:
            raise ValueError(
                "Moderation requires non-i.i.d inference under repeated measures. "
                "Set behavior_analysis.moderation.n_permutations > 0 or set "
                "behavior_analysis.statistics.allow_iid_trials=true to override (not recommended)."
            )
        if groups_for_resampling is None:
            raise ValueError(
                "Moderation requires grouped resampling labels for non-i.i.d trials. "
                "Provide behavior_analysis.run_adjustment.column in the trial table (or ctx.group_ids), "
                "or set behavior_analysis.statistics.allow_iid_trials=true to override (not recommended)."
            )

    if max_features is not None and len(feature_cols) > max_features:
        variances = df_trials[feature_cols].var()
        feature_cols = variances.nlargest(max_features).index.tolist()
        ctx.logger.info("Moderation: limited to top %d features by variance", max_features)
    else:
        ctx.logger.info("Moderation: testing all %d features (no limit)", len(feature_cols))

    temperature = pd.to_numeric(df_trials["temperature"], errors="coerce").to_numpy()
    rating = pd.to_numeric(df_trials["rating"], errors="coerce").to_numpy()

    records: List[Dict[str, Any]] = []
    for feat in feature_cols:
        feature_values = pd.to_numeric(df_trials[feat], errors="coerce").to_numpy()

        valid_mask = np.isfinite(temperature) & np.isfinite(rating) & np.isfinite(feature_values)
        n_valid = int(valid_mask.sum())
        if n_valid < min_samples:
            continue
        groups_valid = None
        if groups_for_resampling is not None and len(groups_for_resampling) == len(valid_mask):
            groups_valid = np.asarray(groups_for_resampling)[valid_mask]
        groups_valid = _sanitize_permutation_groups(
            groups_valid,
            ctx.logger,
            f"Moderation[{feat}]",
        )
        if not allow_iid_trials and groups_valid is None:
            continue

        result = run_moderation_analysis(
            X=temperature[valid_mask],
            W=feature_values[valid_mask],
            Y=rating[valid_mask],
            n_perm=n_permutations,
            x_label="temperature",
            w_label=str(feat),
            y_label="rating",
            center_predictors=True,
            rng=getattr(ctx, "rng", None),
            groups=groups_valid,
            permutation_scheme=perm_scheme,
        )

        rec = {
            "feature": str(feat),
            "feature_type": _cache.get_feature_type(str(feat), ctx.config),
            "n": result.n,
            "b1_temperature": result.b1,
            "b2_feature": result.b2,
            "b3_interaction": result.b3,
            "se_b3": result.se_b3,
            "p_interaction": result.p_b3,
            "p_interaction_perm": result.p_b3_perm,
            "n_permutations": int(getattr(result, "n_permutations", n_permutations) or 0),
            "slope_low_w": result.slope_low_w,
            "slope_mean_w": result.slope_mean_w,
            "slope_high_w": result.slope_high_w,
            "p_slope_low": result.p_slope_low,
            "p_slope_mean": result.p_slope_mean,
            "p_slope_high": result.p_slope_high,
            "r_squared": result.r_squared,
            "r_squared_change": result.r_squared_change,
            "f_interaction": result.f_interaction,
            "p_f_interaction": result.p_f_interaction,
            "jn_low": result.jn_low,
            "jn_high": result.jn_high,
            "jn_type": result.jn_type,
            "significant_moderation_raw": result.is_significant_moderation(fdr_alpha),
        }
        records.append(rec)

    mod_df = pd.DataFrame(records) if records else pd.DataFrame()

    if not mod_df.empty:
        # Add analysis_kind for FDR
        if "analysis_kind" not in mod_df.columns:
            mod_df["analysis_kind"] = "moderation"

        # Ensure primary p-value selection honors permutation setting when available.
        mod_df["p_raw"] = pd.to_numeric(mod_df["p_interaction"], errors="coerce")
        use_perm = p_primary_mode in {"perm", "permutation", "perm_if_available", "permutation_if_available"}
        if use_perm and "p_interaction_perm" in mod_df.columns:
            p_perm = pd.to_numeric(mod_df["p_interaction_perm"], errors="coerce")
            if not allow_iid_trials:
                mod_df["p_primary"] = p_perm.where(p_perm.notna(), np.nan)
                mod_df["p_primary_source"] = np.where(p_perm.notna(), "perm", "perm_missing_required")
            else:
                mod_df["p_primary"] = p_perm.where(p_perm.notna(), mod_df["p_raw"])
                mod_df["p_primary_source"] = np.where(p_perm.notna(), "perm", "asymptotic")
        else:
            mod_df["p_primary"] = mod_df["p_raw"]
            mod_df["p_primary_source"] = "asymptotic"

        # Use unified FDR
        mod_df = _compute_unified_fdr(
            ctx,
            config,
            mod_df,
            p_col="p_primary",
            family_cols=["feature_type", "analysis_kind"],
            analysis_type="moderation",
        )
        mod_df["significant_moderation"] = pd.to_numeric(mod_df.get("p_fdr", np.nan), errors="coerce") < fdr_alpha

    out_dir = _get_stats_subfolder(ctx, "moderation")
    out_path = out_dir / f"moderation_results{suffix}{method_suffix}.parquet"
    if not mod_df.empty:
        _write_parquet_with_optional_csv(mod_df, out_path, also_save_csv=ctx.also_save_csv)
        n_sig = int((mod_df["p_fdr"] < fdr_alpha).sum())
        ctx.logger.info(
            "Moderation: %d features tested, %d significant (FDR < %.2f)",
            len(mod_df), n_sig, fdr_alpha
        )
    else:
        ctx.logger.info("Moderation: no valid results.")

    return mod_df


def stage_hierarchical_fdr_summary(ctx: BehaviorContext, config: Any) -> pd.DataFrame:
    """Compute hierarchical FDR summary across analysis types from cached FDR results.

    Uses cached FDR results instead of re-reading from disk.
    Single responsibility: Hierarchical FDR summary computation and output.
    """
    fdr_alpha = config.fdr_alpha

    # Get cached FDR results
    cached_fdr = _cache.get_fdr_results()
    if not cached_fdr:
        ctx.logger.info("No cached FDR results found; skipping hierarchical FDR summary.")
        return pd.DataFrame()

    ctx.logger.info("Computing hierarchical FDR summary from cached results...")

    # Build summary from cached metadata
    summary_records = []
    for analysis_type, fdr_data in cached_fdr.items():
        metadata = fdr_data.get("metadata", {})
        if metadata:
            summary_records.append({
                "analysis_type": analysis_type,
                "n_tests": metadata.get("n_total_tests", 0),
                "n_reject_within": metadata.get("n_sig_within", 0),
                "n_reject_global": metadata.get("n_sig_global", 0),
                "pct_reject_within": (
                    100 * metadata.get("n_sig_within", 0) / metadata.get("n_total_tests", 1)
                    if metadata.get("n_total_tests", 0) > 0 else 0
                ),
                "pct_reject_global": (
                    100 * metadata.get("n_sig_global", 0) / metadata.get("n_total_tests", 1)
                    if metadata.get("n_total_tests", 0) > 0 else 0
                ),
                "n_families": metadata.get("n_families", 0),
                "hierarchical": metadata.get("hierarchical", False),
            })

    if not summary_records:
        ctx.logger.warning("No FDR metadata found in cache.")
        return pd.DataFrame()

    hier_summary = pd.DataFrame(summary_records)

    # Save summary
    if not hier_summary.empty:
        hier_dir = _get_stats_subfolder(ctx, "fdr")
        hier_path = hier_dir / "hierarchical_fdr_summary.parquet"
        _write_parquet_with_optional_csv(
            hier_summary, hier_path, also_save_csv=ctx.also_save_csv
        )
        ctx.logger.info(f"Hierarchical FDR summary saved to {hier_path}")

        for _, row in hier_summary.iterrows():
            ctx.logger.info(
                f"  {row['analysis_type']}: {row['n_reject_within']}/{row['n_tests']} "
                f"({row['pct_reject_within']:.1f}%) reject within-family"
            )

    return hier_summary




def stage_report(ctx: BehaviorContext, pipeline_config: Any) -> Path:
    """Write a single-subject, self-diagnosing Markdown report (fail-fast)."""
    suffix = _feature_suffix_from_context(ctx)
    method_label = getattr(pipeline_config, "method_label", "")
    method_suffix = f"_{method_label}" if method_label else ""

    top_n = get_config_int(ctx.config, "behavior_analysis.report.top_n", 15)
    alpha = float(getattr(pipeline_config, "fdr_alpha", 0.05))

    # Trial-table summary
    df_trials = _load_trial_table_df(ctx)
    n_trials = int(len(df_trials)) if df_trials is not None else 0
    n_features = 0
    if df_trials is not None and not df_trials.empty:
        n_features = int(sum(1 for c in df_trials.columns if str(c).startswith(FEATURE_COLUMN_PREFIXES)))

    def _read_tsv(path: Path) -> pd.DataFrame:
        return pd.read_csv(path, sep="\t")

    def _sig_counts(df: pd.DataFrame) -> Dict[str, Any]:
        out: Dict[str, Any] = {"n": int(len(df))}
        for col in ["q_global", "p_fdr", "p_primary"]:
            if col in df.columns:
                vals = pd.to_numeric(df[col], errors="coerce")
                out[f"n_sig_{col}"] = int((vals < alpha).sum()) if vals.notna().any() else 0
        return out

    def _top_rows(df: pd.DataFrame) -> pd.DataFrame:
        pcols = [c for c in ["q_global", "p_fdr", "p_primary"] if c in df.columns]
        if not pcols:
            return df.head(0)
        pcol = pcols[0]
        out = df.copy()
        out[pcol] = pd.to_numeric(out[pcol], errors="coerce")
        out = out.sort_values(pcol, ascending=True)
        keep = [c for c in ["feature", "target", "r_primary", "beta_feature", "hedges_g", "p_primary", "p_fdr", "q_global"] if c in out.columns]
        return out[keep].head(max(1, int(top_n))) if keep else out.head(0)

    def _to_md_table(df: pd.DataFrame) -> str:
        if df is None or df.empty:
            return ""
        df2 = df.copy()
        for c in df2.columns:
            df2[c] = df2[c].apply(lambda x: "" if pd.isna(x) else str(x))
        cols = [str(c) for c in df2.columns]
        header = "| " + " | ".join(cols) + " |"
        sep = "| " + " | ".join(["---"] * len(cols)) + " |"
        rows = ["| " + " | ".join(row) + " |" for row in df2.to_numpy(dtype=str).tolist()]
        return "\n".join([header, sep, *rows])

    patterns = [
        "correlations*.parquet",
        "pain_sensitivity*.parquet",
        "regression_feature_effects*.parquet",
        "models_feature_effects*.parquet",
        "condition_effects*.parquet",
        "consistency_summary*.parquet",
        "influence_diagnostics*.parquet",
        "temperature_model_comparison*.parquet",
        "temperature_breakpoint_candidates*.parquet",
        "hierarchical_fdr_summary.parquet",
        "normalized_results*.parquet",
        "summary.json",
        "analysis_metadata.json",
        "outputs_manifest.json",
    ]

    files: List[Path] = []
    for pat in patterns:
        found = sorted(ctx.stats_dir.rglob(pat))
        files.extend(found)
    # Include any other TSV outputs not covered above.
    extra = sorted(p for p in ctx.stats_dir.rglob("*.tsv") if p not in files)
    files.extend(extra)
    files = sorted({p.resolve() for p in files if p.exists()})

    lines: List[str] = []
    lines.append(f"# Subject Report: sub-{ctx.subject}")
    lines.append("")
    lines.append(f"- Task: `{ctx.task}`")
    lines.append(f"- Trials: `{n_trials}`")
    lines.append(f"- Features in trial table: `{n_features}`")
    lines.append(f"- Method: `{getattr(pipeline_config, 'method', '')}` (`{method_label}`)")
    lines.append(f"- Controls: temperature=`{bool(getattr(pipeline_config, 'control_temperature', True))}`, trial_order=`{bool(getattr(pipeline_config, 'control_trial_order', True))}`")
    lines.append(f"- Global FDR alpha: `{alpha}`")
    
    # Summaries per output file (TSVs)
    tsvs = [p for p in files if p.suffix == ".tsv"]
    if tsvs:
        lines.append("")
        lines.append("## Outputs")
        for p in tsvs:
            df = _read_tsv(p)
            if df.empty:
                lines.append(f"- `{p.name}`: (empty)")
                continue
            counts = _sig_counts(df)
            sig_bits = []
            for k in ["n_sig_q_global", "n_sig_p_fdr", "n_sig_p_primary"]:
                if k in counts:
                    sig_bits.append(f"{k}={counts[k]}")
            sig_str = ", ".join(sig_bits) if sig_bits else "no p-columns"
            lines.append(f"- `{p.name}`: n={counts['n']}, {sig_str}")

            top = _top_rows(df)
            if not top.empty and ("feature" in top.columns or "target" in top.columns):
                lines.append("")
                lines.append(f"### Top ({min(len(top), top_n)}) — `{p.name}`")
                lines.append("")
                lines.append(_to_md_table(top))
                lines.append("")

    report_dir = _get_stats_subfolder(ctx, "subject_report")
    out_path = report_dir / f"subject_report{suffix}{method_suffix}.md"
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    ctx.logger.info("Subject report saved: %s/%s", report_dir.name, out_path.name)
    return out_path


def _build_output_filename(
    ctx: BehaviorContext,
    pipeline_config: Any,
    base_name: str,
) -> str:
    """Build standardized output filename with feature and method suffixes.
    
    Centralizes the pattern of constructing output filenames across stages.
    
    Args:
        ctx: BehaviorContext with selected feature files/categories
        pipeline_config: Pipeline configuration with method_label
        base_name: Base filename (e.g., "correlations", "influence_diagnostics")
        
    Returns:
        Filename with feature and method suffixes (no extension)
    """
    feature_suffix = _feature_suffix_from_context(ctx)
    method_label = getattr(pipeline_config, "method_label", "")
    method_suffix = f"_{method_label}" if method_label else ""
    return f"{base_name}{feature_suffix}{method_suffix}"


def _write_metadata_file(path: Path, metadata: Dict[str, Any]) -> None:
    """Write metadata JSON file (fail-fast)."""
    path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")


def _is_valid_df(obj: Any) -> bool:
    """Check if obj is a valid non-empty DataFrame (not a failed stage result dict)."""
    if obj is None:
        return False
    if isinstance(obj, dict):
        return False
    if not hasattr(obj, 'empty'):
        return False
    return not obj.empty


def stage_export(ctx: BehaviorContext, pipeline_config: Any, results: Any) -> List[Path]:
    """Export all analysis results to disk with normalization.

    Consolidates normalization logic into a single stage to avoid duplicate file writes.
    """
    from eeg_pipeline.infra.paths import ensure_dir
    from eeg_pipeline.infra.tsv import write_tsv

    ensure_dir(ctx.stats_dir)
    saved: List[Path] = []

    # Write results using _is_valid_df for safe checks
    if _is_valid_df(getattr(results, "correlations", None)):
        out_dir = _get_stats_subfolder(ctx, "correlations")
        filename = _build_output_filename(ctx, pipeline_config, "correlations")
        path = out_dir / f"{filename}.tsv"
        saved.append(_write_stats_table(ctx, results.correlations, path))

    if _is_valid_df(getattr(results, "pain_sensitivity", None)):
        out_dir = _get_stats_subfolder(ctx, "pain_sensitivity")
        filename = _build_output_filename(ctx, pipeline_config, "pain_sensitivity")
        path = out_dir / f"{filename}.tsv"
        saved.append(_write_stats_table(ctx, results.pain_sensitivity, path))

    if _is_valid_df(getattr(results, "condition_effects", None)):
        out_dir = _get_stats_subfolder(ctx, "condition_effects")
        filename = _build_output_filename(ctx, pipeline_config, "condition_effects")
        path = out_dir / f"{filename}.tsv"
        saved.append(_write_stats_table(ctx, results.condition_effects, path))

    if _is_valid_df(getattr(results, "mediation", None)):
        out_dir = _get_stats_subfolder(ctx, "mediation")
        filename = _build_output_filename(ctx, pipeline_config, "mediation")
        path = out_dir / f"{filename}.tsv"
        saved.append(_write_stats_table(ctx, results.mediation, path))

    if _is_valid_df(getattr(results, "mixed_effects", None)):
        out_dir = _get_stats_subfolder(ctx, "mixed_effects")
        filename = _build_output_filename(ctx, pipeline_config, "mixed_effects")
        path = out_dir / f"{filename}.tsv"
        saved.append(_write_stats_table(ctx, results.mixed_effects, path))

    # NOTE: regression and models are already written by their respective stages.
    # The files already exist in trialwise_regression/ and feature_models/ folders.

    if _is_valid_df(getattr(results, "regression", None)):
        out_dir = _get_stats_subfolder(ctx, "trialwise_regression")
        filename = _build_output_filename(ctx, pipeline_config, "regression_feature_effects")
        path = out_dir / f"{filename}.tsv"
        if path.exists():
            saved.append(path)

    if _is_valid_df(getattr(results, "models", None)):
        out_dir = _get_stats_subfolder(ctx, "feature_models")
        filename = _build_output_filename(ctx, pipeline_config, "models_feature_effects")
        path = out_dir / f"{filename}.tsv"
        if path.exists():
            saved.append(path)

    return saved


# Output kind inference - single source of truth for file-to-kind mapping
_OUTPUT_KIND_PATTERNS = [
    ("corr_stats_", "correlations"),
    ("correlations", "correlations"),
    ("pain_sensitivity", "pain_sensitivity"),
    ("condition_effects", "condition_effects"),
    ("mediation", "mediation"),
    ("moderation", "moderation"),
    ("mixed_effects", "mixed_effects"),
    ("regression_feature_effects", "trialwise_regression"),
    ("models_feature_effects", "feature_models"),
    ("trials_with_lags", "lag_features"),
    ("trials_with_residual", "pain_residual"),
    ("lag_features", "lag_features"),
    ("pain_residual", "pain_residual"),
    ("model_comparison", "temperature_models"),
    ("breakpoint_candidates", "temperature_models"),
    ("breakpoint_test", "temperature_models"),
    ("trials", "trial_table"),
    ("temperature_model_comparison", "temperature_model_comparison"),
    ("temperature_breakpoint", "temperature_breakpoint_test"),
    ("stability_groupwise", "stability_groupwise"),
    ("consistency_summary", "consistency_summary"),
    ("influence_diagnostics", "influence_diagnostics"),
    ("normalized_results", "normalized"),
    ("feature_screening", "feature_screening"),
    ("paired_comparisons", "paired_comparisons"),
    ("summary", "summary"),
    ("analysis_metadata", "analysis_metadata"),
    ("subject_report", "subject_report"),
    ("tf_grid", "time_frequency"),
    ("temporal_correlations", "temporal_correlations"),
    ("hierarchical_fdr_summary", "fdr"),
]


def _infer_output_kind(name: str) -> str:
    """Infer output kind from filename using pattern matching."""
    for prefix, kind in _OUTPUT_KIND_PATTERNS:
        if name.startswith(prefix):
            return kind
    return "unknown"


def _count_rows(path: Path) -> Optional[int]:
    if path.suffix not in {".tsv", ".csv"}:
        return None
    with path.open("r", encoding="utf-8") as f:
        header = f.readline()
        if not header:
            return 0
        return sum(1 for _ in f)


def write_outputs_manifest(
    ctx: BehaviorContext,
    pipeline_config: Any,
    results: Any,
    stage_metrics: Optional[Dict[str, Any]] = None,
) -> Path:
    from datetime import datetime

    feature_folder = _feature_folder_from_context(ctx)
    out_dir = _get_stats_subfolder(ctx, "summary")
    manifest_path = out_dir / "outputs_manifest.json"

    outputs = []
    for path in sorted(p for p in ctx.stats_dir.rglob("*") if p.is_file()):
        # Skip hidden files or logs
        if path.name.startswith(".") or path.suffix == ".log":
            continue
        if path.name == "outputs_manifest.json":
            continue
        rel = path.relative_to(ctx.stats_dir)
        parts = rel.parts
        if len(parts) < 2 or parts[1] != feature_folder:
            continue
        outputs.append({
            "name": path.name,
            "path": str(path),
            "kind": _infer_output_kind(path.name),
            "subfolder": str(path.parent.relative_to(ctx.stats_dir)),
            "rows": _count_rows(path),
            "size_bytes": int(path.stat().st_size),
            "method_label": pipeline_config.method_label,
        })

    feature_types = [name for name, df in ctx.iter_feature_tables() if df is not None and not df.empty]

    payload = {
        "subject": ctx.subject,
        "task": ctx.task,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "method": pipeline_config.method,
        "robust_method": pipeline_config.robust_method,
        "method_label": pipeline_config.method_label,
        "n_trials": ctx.n_trials,
        "feature_types": feature_types,
        "feature_categories": ctx.feature_categories or [],
        "feature_files": ctx.selected_feature_files or [],
        "targets": {
            "rating": bool(ctx._find_rating_column() is not None) if hasattr(ctx, "_find_rating_column") else False,
            "temperature": bool(ctx.temperature is not None and ctx.temperature.notna().any()) if ctx.temperature is not None else False,
        },
        "covariates_qc": ctx.data_qc.get("covariates_qc", {}),
        "outputs": outputs,
    }

    if stage_metrics:
        payload["stage_metrics"] = stage_metrics

    manifest_path.write_text(json.dumps(payload, indent=2, default=str))
    return manifest_path
