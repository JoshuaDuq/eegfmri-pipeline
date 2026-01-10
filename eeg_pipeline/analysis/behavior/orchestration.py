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
from eeg_pipeline.utils.analysis.stats.correlation import compute_correlation
from eeg_pipeline.utils.config.loader import get_config_value
from eeg_pipeline.infra.paths import ensure_dir


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
        if ctx.targets is not None:
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
class TrialTableResult:
    """Output contract for trial_table stage."""
    df: pd.DataFrame
    path: Optional[Path] = None
    n_trials: int = 0
    n_features: int = 0
    metadata: Optional[Dict[str, Any]] = None


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
class ConfoundsResult:
    """Output contract for confounds stage."""
    df: pd.DataFrame
    selection_mode: str = "descriptive"
    covariates_applied: List[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.covariates_applied is None:
            self.covariates_applied = []


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
        name="trial_table_validate",
        description="Validate trial table",
        requires=(StageRegistry.RESOURCE_TRIAL_TABLE,),
        produces=(),
        config_key="behavior_analysis.trial_table.validate.enabled",
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
        requires=(StageRegistry.RESOURCE_DESIGN, StageRegistry.RESOURCE_PVALUES),
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
        name="confounds",
        description="Audit QC confounds",
        requires=(StageRegistry.RESOURCE_TRIAL_TABLE,),
        produces=("confounds_audit",),
        config_key="run_confounds",
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
        name="temporal_topomaps",
        description="Topomap correlations",
        requires=(StageRegistry.RESOURCE_POWER_DF, StageRegistry.RESOURCE_EPOCHS),
        produces=("topomap_correlations",),
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
    
    # Validation & Export
    StageSpec(
        name="validate",
        description="Apply global FDR correction",
        requires=(StageRegistry.RESOURCE_CORRELATIONS,),
        produces=("global_fdr",),
        group=StageRegistry.GROUP_VALIDATION,
    ),
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
        self.confounds = outputs.get("confounds")
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
        "trial_table_validate": lambda ctx, config, outputs: stage_trial_table_validate(ctx, config),
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
            ctx, config, outputs.get("correlate_design"), outputs.get("correlate_pvalues", [])
        ),
        "correlate_fdr": lambda ctx, config, outputs: stage_correlate_fdr(
            ctx, config, outputs.get("correlate_primary_selection") or outputs.get("correlate_pvalues", [])
        ),
        "pain_sensitivity": lambda ctx, config, outputs: stage_pain_sensitivity(ctx, config),
        "confounds": lambda ctx, config, outputs: stage_confounds(ctx, config),
        "regression": lambda ctx, config, outputs: stage_regression(ctx, config),
        "models": lambda ctx, config, outputs: stage_models(ctx, config),
        "stability": lambda ctx, config, outputs: stage_stability(ctx, config),
        "consistency": lambda ctx, config, outputs: stage_consistency(ctx, config, None),
        "influence": lambda ctx, config, outputs: stage_influence(ctx, config, None),
        "condition_column": lambda ctx, config, outputs: stage_condition_column(ctx, config),
        "condition_window": lambda ctx, config, outputs: stage_condition_window(ctx, config),
        "temporal_tfr": lambda ctx, config, outputs: stage_temporal_tfr(ctx),
        "temporal_stats": lambda ctx, config, outputs: stage_temporal_stats(ctx),
        "temporal_topomaps": lambda ctx, config, outputs: stage_temporal_topomaps(ctx),
        "cluster": lambda ctx, config, outputs: stage_cluster(ctx, config),
        "mediation": lambda ctx, config, outputs: stage_mediation(ctx, config),
        "moderation": lambda ctx, config, outputs: stage_moderation(ctx, config),
        "mixed_effects": lambda ctx, config, outputs: stage_mixed_effects(ctx, config),
        "global_fdr": lambda ctx, config, outputs: stage_global_fdr(ctx, config),
        "hierarchical_fdr_summary": lambda ctx, config, outputs: stage_hierarchical_fdr_summary(ctx, config),
        "validate": lambda ctx, config, outputs: stage_validate(ctx, config, None),
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
# Standardized Result Schema
###################################################################

STANDARD_RESULT_COLUMNS = (
    "analysis_type",   # e.g., "correlation", "condition", "regression", "mediation"
    "feature_id",      # Feature identifier
    "target",          # Target variable (rating, temperature, pain_residual)
    "effect_size",     # Primary effect size (r, d, beta, etc.)
    "effect_size_ci_lo",  # 95% CI lower bound
    "effect_size_ci_hi",  # 95% CI upper bound
    "p_raw",           # Uncorrected p-value
    "p_primary",       # Primary p-value (parametric or permutation)
    "p_within_fdr",    # Within-family FDR-corrected p-value
    "q_global",        # Global FDR q-value across all tests
)


def standardize_result_schema(
    df: pd.DataFrame,
    analysis_type: str,
    feature_col: str = "feature",
    target_col: str = "target",
) -> pd.DataFrame:
    """Standardize result DataFrame to canonical schema for global correction.
    
    Ensures all inferential outputs have consistent columns for FDR.
    """
    if df.empty:
        return df
    
    df = df.copy()
    
    # Add analysis_type if missing
    if "analysis_type" not in df.columns:
        df["analysis_type"] = analysis_type
    
    # Standardize feature_id
    if "feature_id" not in df.columns:
        if feature_col in df.columns:
            df["feature_id"] = df[feature_col].astype(str)
        elif "feature" in df.columns:
            df["feature_id"] = df["feature"].astype(str)
        elif "mediator" in df.columns:
            df["feature_id"] = df["mediator"].astype(str)
        else:
            df["feature_id"] = ""
    
    # Standardize target
    if "target" not in df.columns:
        if target_col in df.columns:
            df["target"] = df[target_col].astype(str)
        else:
            df["target"] = "rating"
    
    # Ensure p-value columns exist
    for col in ["p_raw", "p_primary", "p_within_fdr", "q_global"]:
        if col not in df.columns:
            df[col] = np.nan
    
    # Map existing columns to standard names
    col_mappings = {
        "p_value": "p_raw",
        "p": "p_raw",
        "pvalue": "p_raw",
        "p_fdr": "p_within_fdr",
        "q_value": "p_within_fdr",
    }
    for old_col, new_col in col_mappings.items():
        if old_col in df.columns and df[new_col].isna().all():
            df[new_col] = pd.to_numeric(df[old_col], errors="coerce")
    
    return df


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

    max_missing_pct = float(get_config_value(
        ctx.config, "behavior_analysis.feature_qc.max_missing_pct", MAX_MISSING_PCT_DEFAULT
    ))
    min_variance = float(get_config_value(
        ctx.config, "behavior_analysis.feature_qc.min_variance", MIN_VARIANCE_THRESHOLD
    ))
    check_within_run = bool(get_config_value(
        ctx.config, "behavior_analysis.feature_qc.check_within_run_variance", True
    ))
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

    from eeg_pipeline.infra.tsv import write_tsv
    suffix = _feature_suffix_from_context(ctx)
    out_dir = _get_stats_subfolder(ctx, "feature_qc")
    write_tsv(qc_df, out_dir / f"feature_qc_screen{suffix}.tsv")

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
        stages.append("trial_table_validate")
    
    if getattr(pipeline_config, "run_lag_features", False):
        stages.append("lag_features")
    
    if getattr(pipeline_config, "run_pain_residual", False):
        stages.append("pain_residual")
    
    if getattr(pipeline_config, "run_temperature_models", False):
        stages.append("temperature_models")
    
    if getattr(pipeline_config, "run_feature_qc", False):
        stages.append("feature_qc")
    
    if getattr(pipeline_config, "run_confounds", False):
        stages.append("confounds")
    
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
        stages.extend(["temporal_tfr", "temporal_stats", "temporal_topomaps"])
    
    if getattr(pipeline_config, "run_cluster_tests", False):
        stages.append("cluster")
    
    if getattr(pipeline_config, "run_mediation", False):
        stages.append("mediation")
    
    if getattr(pipeline_config, "run_moderation", False):
        stages.append("moderation")
    
    if getattr(pipeline_config, "run_mixed_effects", False):
        stages.append("mixed_effects")
    
    if getattr(pipeline_config, "run_validation", True):
        stages.extend(["global_fdr", "hierarchical_fdr_summary"])
    
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
            ctx.logger.warning("Stage '%s' has no implementation in STAGE_RUNNERS", name)
            return None
        return runner(ctx, config, outputs)

    for step in progress_steps:
        stage_name = step["name"]
        
        if progress is not None:
            progress.step(stage_name, current=step["index"] + 1, total=step["total"])
        
        ctx.logger.info("[%d/%d] Running stage: %s", step["index"] + 1, step["total"], stage_name)
        
        try:
            output = _run_stage(stage_name)
            outputs[stage_name] = output
            
            if results is not None:
                _update_results_from_stage(results, stage_name, output)
                    
        except Exception as exc:
            ctx.logger.error("Stage '%s' failed: %s", stage_name, exc)
            outputs[stage_name] = {"status": "failed", "error": str(exc)}

    return outputs


def _update_results_from_stage(results: Any, stage_name: str, output: Any) -> None:
    """Update BehaviorPipelineResults from stage output."""
    stage_to_attr = {
        "trial_table": "trial_table_path",
        "trial_table_validate": "trial_table_validation",
        "correlate_fdr": "correlations",
        "pain_sensitivity": "pain_sensitivity",
        "confounds": "confounds",
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
    
    attr = stage_to_attr.get(stage_name)
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


# Centralized feature column prefixes - matches FEATURE_TYPES from domain.features
FEATURE_COLUMN_PREFIXES = (
    "power_",
    "connectivity_",
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
    "temporal_",
)

CATEGORY_PREFIX_MAP = {
    "power": "power_",
    "connectivity": "connectivity_",
    "aperiodic": "aperiodic_",
    "erp": "erp_",
    "itpc": "itpc_",
    "pac": "pac_",
    "complexity": "complexity_",
    "bursts": "bursts_",
    "quality": "quality_",
    "erds": "erds_",
    "spectral": "spectral_",
    "ratios": "ratios_",
    "asymmetry": "asymmetry_",
    "temporal": "temporal_",
}

# Constants for validation thresholds
MIN_SAMPLES_DEFAULT = 10
MIN_SAMPLES_RUN_LEVEL = 3
MIN_VARIANCE_THRESHOLD = 1e-10
CONSTANT_VARIANCE_THRESHOLD = 1e-12
MAX_MISSING_PCT_DEFAULT = 0.2
FDR_ALPHA_DEFAULT = 0.05
MIN_FEATURES_FOR_ANALYSIS = 1
MIN_TRIALS_FOR_ANALYSIS = 1


def _is_dataframe_valid(df: Optional[pd.DataFrame]) -> bool:
    """Check if DataFrame is not None and not empty.
    
    Encapsulates boundary condition for DataFrame validation.
    """
    return df is not None and not df.empty


def _get_stats_subfolder(ctx: BehaviorContext, kind: str) -> Path:
    """Helper to get a subfolder within stats_dir and ensure it exists."""
    path = ctx.stats_dir / kind
    ensure_dir(path)
    return path


def _write_stats_table(
    ctx: BehaviorContext,
    df: pd.DataFrame,
    path: Path,
) -> List[Path]:
    """Write a stats table respecting the also_save_csv context setting.
    
    Args:
        ctx: BehaviorContext with also_save_csv flag
        df: DataFrame to write
        path: Primary output path (TSV)
        
    Returns:
        List of paths written (TSV and optionally CSV)
    """
    from eeg_pipeline.infra.tsv import write_table_with_formats
    return write_table_with_formats(df, path, also_save_csv=ctx.also_save_csv)


def _find_stats_path(ctx: BehaviorContext, filename: str) -> Optional[Path]:
    """Helper to find a file in stats_dir or its subfolders."""
    kind = _infer_output_kind(filename)
    if kind != "unknown":
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


def _augment_dataframe_with_change_scores(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """Add change score columns to a feature DataFrame."""
    if not _is_dataframe_valid(df):
        return df
    
    from eeg_pipeline.utils.analysis.stats.transforms import compute_change_features
    
    change_df = compute_change_features(df)
    if not _is_dataframe_valid(change_df):
        return df
    
    new_columns = [col for col in change_df.columns if col not in df.columns]
    if not new_columns:
        return df
    
    return pd.concat([df, change_df[new_columns]], axis=1)


def add_change_scores(ctx: BehaviorContext) -> None:
    """Compute and append change scores (active-baseline) once per context."""
    if ctx._change_scores_added or not ctx.compute_change_scores:
        return

    ctx.power_df = _augment_dataframe_with_change_scores(ctx.power_df)
    ctx.connectivity_df = _augment_dataframe_with_change_scores(ctx.connectivity_df)
    ctx.aperiodic_df = _augment_dataframe_with_change_scores(ctx.aperiodic_df)
    ctx.itpc_df = _augment_dataframe_with_change_scores(ctx.itpc_df)
    ctx.pac_df = _augment_dataframe_with_change_scores(ctx.pac_df)
    ctx.complexity_df = _augment_dataframe_with_change_scores(ctx.complexity_df)
    ctx._change_scores_added = True


def stage_load(ctx: BehaviorContext) -> bool:
    if not ctx.load_data():
        ctx.logger.warning("Failed to load data")
        return False

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

    feature_cols = [c for c in df_trials.columns if str(c).startswith(FEATURE_COLUMN_PREFIXES)]
    feature_cols = _filter_feature_cols_by_band(feature_cols, ctx)
    feature_cols = _filter_feature_cols_for_computation(feature_cols, "correlations", ctx)
    
    if not feature_cols:
        ctx.logger.warning("Correlations design: no feature columns after filtering.")
        return None

    prefer_pain_residual = bool(get_config_value(ctx.config, "behavior_analysis.correlations.prefer_pain_residual", True))
    default_targets = ["pain_residual", "rating", "temperature"] if prefer_pain_residual else ["rating", "temperature", "pain_residual"]
    targets_cfg = get_config_value(ctx.config, "behavior_analysis.correlations.targets", None)
    if isinstance(targets_cfg, (list, tuple)) and targets_cfg:
        targets = [str(t).strip().lower() for t in targets_cfg]
    else:
        targets = default_targets
    use_cv_resid = bool(get_config_value(ctx.config, "behavior_analysis.correlations.use_crossfit_pain_residual", False))
    if use_cv_resid and "pain_residual_cv" in df_trials.columns:
        if not (isinstance(targets_cfg, (list, tuple)) and targets_cfg):
            targets = ["pain_residual_cv", *[t for t in targets if t != "pain_residual_cv"]]
    targets = [t for t in targets if t in df_trials.columns]

    if not targets:
        ctx.logger.warning("Correlations design: no valid target columns found.")
        return None

    run_adjust_enabled = bool(get_config_value(ctx.config, "behavior_analysis.run_adjustment.enabled", False))
    run_col = str(get_config_value(ctx.config, "behavior_analysis.run_adjustment.column", "run_id") or "run_id").strip()
    if not run_col:
        run_col = "run_id"
    run_adjust_in_correlations = bool(
        get_config_value(ctx.config, "behavior_analysis.run_adjustment.include_in_correlations", run_adjust_enabled)
    )
    max_run_dummies = int(get_config_value(ctx.config, "behavior_analysis.run_adjustment.max_dummies", 20))

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

    groups_for_perm = ctx.group_ids if getattr(ctx, "group_ids", None) is not None else None
    if groups_for_perm is None and run_col in df_trials.columns:
        groups_for_perm = pd.Series(df_trials[run_col], index=df_trials.index, name=run_col)

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
        groups_for_perm=groups_for_perm,
    )


def stage_correlate_effect_sizes(
    ctx: BehaviorContext,
    config: Any,
    design: CorrelateDesign,
) -> List[Dict[str, Any]]:
    """Compute raw and partial correlation effect sizes.
    
    Single responsibility: Compute effect sizes (r values) without p-value inference.
    """
    from eeg_pipeline.utils.analysis.stats.correlation import safe_correlation
    from eeg_pipeline.utils.analysis.stats import compute_partial_correlations_with_cov_temp

    method = getattr(config, "method", "spearman")
    robust_method = getattr(config, "robust_method", None)
    method_label = getattr(config, "method_label", "")
    min_samples = int(getattr(config, "min_samples", 10))

    records: List[Dict[str, Any]] = []
    for target in design.targets:
        y = pd.to_numeric(design.df_trials[target], errors="coerce")
        temp_for_partial = design.temperature_series if (design.temperature_series is not None and target != "temperature") else None

        for feat in design.feature_cols:
            x = pd.to_numeric(design.df_trials[feat], errors="coerce")
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
                "feature_type": _infer_feature_type(str(feat), ctx.config),
                "target": str(target),
                "method": method,
                "robust_method": robust_method,
                "method_label": method_label,
                "n": int(n),
                "r_raw": float(r_raw) if np.isfinite(r_raw) else np.nan,
                "p_raw": float(p_raw) if np.isfinite(p_raw) else np.nan,
                "r": float(r_raw) if np.isfinite(r_raw) else np.nan,
                "p": float(p_raw) if np.isfinite(p_raw) else np.nan,
                "p_value": float(p_raw) if np.isfinite(p_raw) else np.nan,
                "skip_reason": skip_reason,
                "run_adjustment_enabled": bool(design.run_adjust_in_correlations and design.run_col in design.df_trials.columns),
                "run_column": design.run_col if design.run_col in design.df_trials.columns else None,
            }

            r_pc, p_pc, n_pc, r_pt, p_pt, n_pt, r_pct, p_pct, n_pct = compute_partial_correlations_with_cov_temp(
                roi_values=x,
                target_values=y,
                covariates_df=design.cov_df,
                temperature_series=temp_for_partial,
                method=method,
                context="trial_table",
                logger=ctx.logger,
                min_samples=min_samples,
                config=ctx.config,
            )
            rec.update({
                "r_partial_cov": r_pc,
                "p_partial_cov": p_pc,
                "n_partial_cov": n_pc,
                "r_partial_temp": r_pt,
                "p_partial_temp": p_pt,
                "n_partial_temp": n_pt,
                "r_partial_cov_temp": r_pct,
                "p_partial_cov_temp": p_pct,
                "n_partial_cov_temp": n_pct,
            })

            if design.run_col in design.df_trials.columns:
                try:
                    df_run = pd.DataFrame({design.run_col: design.df_trials[design.run_col], "x": x, "y": y})
                    run_means = df_run.groupby(design.run_col, dropna=True)[["x", "y"]].mean(numeric_only=True)
                    r_run, p_run, n_run = safe_correlation(
                        run_means["x"].to_numpy(dtype=float),
                        run_means["y"].to_numpy(dtype=float),
                        method,
                        min_samples=3,
                        robust_method=None,
                    )
                    rec.update({
                        "n_runs": int(n_run),
                        "r_run_mean": float(r_run) if np.isfinite(r_run) else np.nan,
                        "p_run_mean": float(p_run) if np.isfinite(p_run) else np.nan,
                    })
                except Exception:
                    rec.update({"n_runs": np.nan, "r_run_mean": np.nan, "p_run_mean": np.nan})

            records.append(rec)

    ctx.logger.info("Correlations effect sizes: computed %d feature-target pairs", len(records))
    return records


def stage_correlate_pvalues(
    ctx: BehaviorContext,
    config: Any,
    design: CorrelateDesign,
    records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Compute permutation p-values for correlations.
    
    Single responsibility: Add permutation-based p-values to existing effect size records.
    """
    from eeg_pipeline.utils.analysis.stats.permutation import compute_permutation_pvalues_with_cov_temp

    method = getattr(config, "method", "spearman")
    robust_method = getattr(config, "robust_method", None)
    perm_enabled = bool(get_config_value(ctx.config, "behavior_analysis.correlations.permutation.enabled", False))
    n_perm = int(get_config_value(ctx.config, "behavior_analysis.correlations.permutation.n_permutations", ctx.n_perm or 0))

    perm_ok = (
        perm_enabled
        and n_perm > 0
        and robust_method in (None, "", False)
        and isinstance(method, str)
        and method.strip().lower() in {"spearman", "pearson"}
    )

    if not perm_ok:
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

    rng = ctx.rng or np.random.default_rng(42)
    n_computed = 0

    for rec in records:
        feat = rec["feature"]
        target = rec["target"]
        r_raw = rec.get("r_raw", np.nan)
        n = rec.get("n", 0)

        if not (np.isfinite(r_raw) and int(n) > 0):
            rec.update({
                "n_permutations": int(n_perm),
                "p_perm_raw": np.nan,
                "p_perm_partial_cov": np.nan,
                "p_perm_partial_temp": np.nan,
                "p_perm_partial_cov_temp": np.nan,
            })
            continue

        x = pd.to_numeric(design.df_trials[feat], errors="coerce")
        y = pd.to_numeric(design.df_trials[target], errors="coerce")
        temp_for_partial = design.temperature_series if (design.temperature_series is not None and target != "temperature") else None

        p_perm, p_perm_cov, p_perm_temp, p_perm_cov_temp = compute_permutation_pvalues_with_cov_temp(
            x_aligned=pd.Series(x.to_numpy(dtype=float), index=design.df_trials.index),
            y_aligned=pd.Series(y.to_numpy(dtype=float), index=design.df_trials.index),
            covariates_df=design.cov_df,
            temp_series=temp_for_partial,
            method=method.strip().lower(),
            n_perm=n_perm,
            n_eff=int(n),
            rng=rng,
            config=ctx.config,
            groups=design.groups_for_perm,
        )
        rec.update({
            "n_permutations": int(n_perm),
            "p_perm_raw": float(p_perm) if np.isfinite(p_perm) else np.nan,
            "p_perm_partial_cov": float(p_perm_cov) if np.isfinite(p_perm_cov) else np.nan,
            "p_perm_partial_temp": float(p_perm_temp) if np.isfinite(p_perm_temp) else np.nan,
            "p_perm_partial_cov_temp": float(p_perm_cov_temp) if np.isfinite(p_perm_cov_temp) else np.nan,
        })
        n_computed += 1

    ctx.logger.info("Correlations pvalues: computed %d permutation tests (n_perm=%d)", n_computed, n_perm)
    return records


def stage_correlate_primary_selection(
    ctx: BehaviorContext,
    config: Any,
    design: CorrelateDesign,
    records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Select primary p-value and effect size for each test.
    
    Single responsibility: Determine which p-value column to use for FDR.
    """
    p_primary_mode = str(get_config_value(ctx.config, "behavior_analysis.correlations.p_primary_mode", "perm_if_available")).strip().lower()
    primary_unit = str(get_config_value(ctx.config, "behavior_analysis.correlations.primary_unit", "trial")).strip().lower()
    use_run_unit = primary_unit in {"run", "run_mean", "runmean", "run_level"}

    for rec in records:
        target = rec.get("target", "")
        p_kind = "p_raw"
        p_primary = rec["p_raw"]
        r_primary = rec["r_raw"]
        src = "raw"

        if use_run_unit and pd.notna(rec.get("p_run_mean", np.nan)):
            p_kind = "p_run_mean"
            p_primary = rec.get("p_run_mean", np.nan)
            r_primary = rec.get("r_run_mean", np.nan)
            src = "run_mean"
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

        rec["p_kind_primary"] = p_kind
        rec["p_primary"] = p_primary
        rec["r_primary"] = r_primary
        rec["p_primary_source"] = src

    return records


def stage_correlate_fdr(
    ctx: BehaviorContext,
    config: Any,
    records: List[Dict[str, Any]],
) -> pd.DataFrame:
    """Apply hierarchical FDR correction with explicit family structure.
    
    Family structure: feature_type × band × target × analysis_kind
    This is critical for EEG feature banks to control FWER properly.
    
    Single responsibility: FDR correction on p_primary column with family awareness.
    """
    from eeg_pipeline.utils.analysis.stats.fdr import fdr_bh

    corr_df = pd.DataFrame(records) if records else pd.DataFrame()
    if corr_df.empty:
        return corr_df

    if "p_primary" not in corr_df.columns:
        ctx.logger.error("Missing 'p_primary' column. Ensure 'correlate_primary_selection' stage runs before 'correlate_fdr'.")
        raise KeyError("p_primary")

    fdr_alpha = float(getattr(config, "fdr_alpha", 0.05))
    hierarchical_fdr = bool(get_config_value(ctx.config, "behavior_analysis.statistics.hierarchical_fdr", True))
    
    # Define family structure columns
    family_cols = []
    for col in ["feature_type", "band", "target", "analysis_kind"]:
        if col in corr_df.columns:
            family_cols.append(col)
    
    if hierarchical_fdr and family_cols:
        # Apply FDR within each family
        corr_df["fdr_family"] = corr_df[family_cols].astype(str).agg("_".join, axis=1)
        corr_df["p_fdr"] = np.nan
        
        family_stats = []
        for family_id, family_group in corr_df.groupby("fdr_family"):
            p_vals = pd.to_numeric(family_group["p_primary"], errors="coerce").to_numpy()
            p_fdr = fdr_bh(p_vals, alpha=fdr_alpha, config=ctx.config)
            corr_df.loc[family_group.index, "p_fdr"] = p_fdr
            
            n_tests = len(p_vals)
            n_sig = int((p_fdr < fdr_alpha).sum())
            family_stats.append({
                "family": family_id,
                "n_tests": n_tests,
                "n_significant": n_sig,
            })
        
        # Store family structure in context for downstream use
        ctx.data_qc["fdr_family_structure"] = {
            "family_columns": family_cols,
            "n_families": len(family_stats),
            "family_stats": family_stats,
            "hierarchical": True,
        }
        
        n_sig_total = int((corr_df["p_fdr"] < fdr_alpha).sum())
        n_families = corr_df["fdr_family"].nunique()
        ctx.logger.info("Hierarchical FDR: %d/%d significant across %d families (alpha=%.2f)", 
                       n_sig_total, len(corr_df), n_families, fdr_alpha)
    else:
        # Flat FDR (original behavior)
        p_for_fdr = pd.to_numeric(corr_df["p_primary"], errors="coerce").to_numpy()
        corr_df["p_fdr"] = fdr_bh(p_for_fdr, alpha=fdr_alpha, config=ctx.config)
        corr_df["fdr_family"] = "all"
        
        ctx.data_qc["fdr_family_structure"] = {
            "family_columns": [],
            "n_families": 1,
            "hierarchical": False,
        }
        
        n_sig = int((corr_df["p_fdr"] < fdr_alpha).sum())
        ctx.logger.info("Flat FDR: %d/%d significant at alpha=%.2f", n_sig, len(corr_df), fdr_alpha)

    return corr_df


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


def stage_pain_sensitivity(ctx: BehaviorContext, config: Any) -> pd.DataFrame:
    """Compute pain sensitivity correlations (independent stage)."""
    from eeg_pipeline.utils.analysis.stats.fdr import fdr_bh
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
    robust_method_cfg = get_config_value(ctx.config, "behavior_analysis.robust_correlation", None)
    if robust_method_cfg is not None:
        robust_method_cfg = str(robust_method_cfg).strip().lower() or None
    
    psi_feature_cols = [c for c in df_trials.columns if str(c).startswith(FEATURE_COLUMN_PREFIXES)]
    psi_feature_cols = _filter_feature_cols_by_band(psi_feature_cols, ctx)
    psi_feature_cols = _filter_feature_cols_for_computation(psi_feature_cols, "pain_sensitivity", ctx)
    
    if not psi_feature_cols:
        ctx.logger.warning("Pain sensitivity: no feature columns found; skipping.")
        return pd.DataFrame()
    
    ctx.logger.info("Pain sensitivity: analyzing %d features", len(psi_feature_cols))
    psi_features = df_trials[psi_feature_cols].copy()
    psi_df = run_pain_sensitivity_correlations(
        features_df=psi_features,
        ratings=pd.to_numeric(df_trials["rating"], errors="coerce"),
        temperatures=pd.to_numeric(df_trials["temperature"], errors="coerce"),
        method=method,
        robust_method=robust_method_cfg,
        min_samples=int(getattr(config, "min_samples", 10)),
        logger=ctx.logger,
        config=ctx.config,
    )
    
    if _is_dataframe_valid(psi_df):
        p_value_columns = ["p_psi", "p_value"]
        p_column = next((col for col in p_value_columns if col in psi_df.columns), None)
        if p_column:
            p_values = pd.to_numeric(psi_df[p_column], errors="coerce").to_numpy()
            fdr_alpha = float(getattr(config, "fdr_alpha", FDR_ALPHA_DEFAULT))
            psi_df["p_fdr"] = fdr_bh(p_values, alpha=fdr_alpha, config=ctx.config)

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
    
    selected = set(b.lower() for b in ctx.selected_bands)
    filtered: List[str] = []
    
    for col in feature_cols:
        try:
            parsed = NamingSchema.parse(str(col))
            if not parsed.get("valid"):
                # Keep unparseable columns (might be summary or derived features)
                filtered.append(col)
                continue
            
            band = parsed.get("band")
            if not band:
                # Features without bands (ERP, complexity, etc.) - keep them
                filtered.append(col)
            elif str(band).lower() in selected:
                # Band matches user selection
                filtered.append(col)
            # else: band doesn't match, exclude
        except Exception:
            # On parse error, keep the column
            filtered.append(col)
    
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


@dataclass
class TrialTableResult:
    """Result from compute_trial_table."""
    df: pd.DataFrame
    metadata: Dict[str, Any]


def compute_trial_table(ctx: BehaviorContext, config: Any) -> Optional[TrialTableResult]:
    """Build the canonical per-trial analysis table (compute only, no I/O).
    
    Single responsibility: Compute trial table DataFrame and metadata.
    """
    from eeg_pipeline.utils.data.trial_table import build_subject_trial_table

    include_features = bool(get_config_value(ctx.config, "behavior_analysis.trial_table.include_features", True))
    include_covariates = bool(get_config_value(ctx.config, "behavior_analysis.trial_table.include_covariates", True))
    include_events = bool(get_config_value(ctx.config, "behavior_analysis.trial_table.include_events", True))

    extra_cols = get_config_value(ctx.config, "behavior_analysis.trial_table.extra_event_columns", None)
    extra_cols_list = [str(c) for c in extra_cols] if isinstance(extra_cols, (list, tuple)) else None

    result = build_subject_trial_table(
        ctx,
        include_features=include_features,
        include_covariates=include_covariates,
        include_events=include_events,
        extra_event_columns=extra_cols_list,
    )

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
    out_path = out_dir / f"{fname}.tsv"

    # Create a simple object with df and metadata for save_trial_table
    class _TableWrapper:
        def __init__(self, df, metadata):
            self.df = df
            self.metadata = metadata

    save_trial_table(_TableWrapper(result.df, result.metadata), out_path, format=fmt)

    meta_path = out_dir / f"{fname}.metadata.json"
    meta_path.write_text(json.dumps(result.metadata, indent=2, default=str))
    ctx.logger.info("Saved trial table: %s/%s (%d rows, %d cols)", out_dir.name, out_path.name, len(result.df), result.df.shape[1])

    return out_path


def stage_trial_table(ctx: BehaviorContext, config: Any) -> Optional[Path]:
    """Build and save trial table (composed stage).
    
    Composes: compute_trial_table + write_trial_table
    """
    result = compute_trial_table(ctx, config)
    if result is None or not _is_dataframe_valid(result.df):
        ctx.logger.warning("Trial table: no data to write")
        return None
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
    from eeg_pipeline.infra.tsv import write_tsv

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
    out_path = out_dir / f"trials_with_lags{suffix}.tsv"
    write_tsv(df_augmented, out_path)

    meta_path = out_dir / f"lag_features{suffix}.metadata.json"
    meta_path.write_text(json.dumps(lag_meta, indent=2, default=str))
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
    from eeg_pipeline.infra.tsv import write_tsv

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
    out_path = out_dir / f"trials_with_residual{suffix}.tsv"
    write_tsv(df_augmented, out_path)

    meta_path = out_dir / f"pain_residual{suffix}.metadata.json"
    meta_path.write_text(json.dumps(resid_meta, indent=2, default=str))
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
    from eeg_pipeline.infra.tsv import write_tsv

    suffix = _feature_suffix_from_context(ctx)
    out_dir = _get_stats_subfolder(ctx, "temperature_models")

    if model_comparison is not None:
        if _is_dataframe_valid(model_comparison.df):
            comparison_path = out_dir / f"model_comparison{suffix}.tsv"
            write_tsv(model_comparison.df, comparison_path)
        
        metadata_path = out_dir / f"model_comparison{suffix}.metadata.json"
        metadata_path.write_text(
            json.dumps(model_comparison.metadata, indent=2, default=str)
        )
        ctx.data_qc["temperature_model_comparison"] = model_comparison.metadata
        ctx.logger.info("Temperature model comparison saved: %s", out_dir.name)

    if breakpoint is not None:
        if _is_dataframe_valid(breakpoint.df):
            breakpoint_path = out_dir / f"breakpoint_candidates{suffix}.tsv"
            write_tsv(breakpoint.df, breakpoint_path)
        
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

    mc_enabled = bool(get_config_value(ctx.config, "behavior_analysis.temperature_models.model_comparison.enabled", True))
    if mc_enabled:
        try:
            model_comparison = compute_temp_model_comparison(df["temperature"], df["rating"], ctx.config)
            meta["model_comparison"] = model_comparison.metadata
        except Exception as exc:
            ctx.logger.warning("Temperature model comparison failed: %s", exc)
            meta["model_comparison"] = {"status": "failed", "error": str(exc)}

    bp_enabled = bool(get_config_value(ctx.config, "behavior_analysis.temperature_models.breakpoint_test.enabled", True))
    if bp_enabled:
        try:
            breakpoint = compute_temp_breakpoints(df["temperature"], df["rating"], ctx.config)
            meta["breakpoint_test"] = breakpoint.metadata
        except Exception as exc:
            ctx.logger.warning("Temperature breakpoint test failed: %s", exc)
            meta["breakpoint_test"] = {"status": "failed", "error": str(exc)}

    write_temperature_models(ctx, model_comparison, breakpoint)
    meta["status"] = "ok"
    return meta


def _load_trial_table_df(ctx: BehaviorContext) -> Optional[pd.DataFrame]:
    """Load trial table DataFrame from disk with fallback search."""
    from eeg_pipeline.infra.tsv import read_table
    
    suffix = _feature_suffix_from_context(ctx)
    preferred_filenames = [f"trials{suffix}.tsv", f"trials{suffix}.parquet"]
    
    for filename in preferred_filenames:
        path = _find_stats_path(ctx, filename)
        if path is not None:
            return read_table(path)

    tsv_candidates = sorted(ctx.stats_dir.rglob("trials*.tsv"))
    parquet_candidates = sorted(ctx.stats_dir.rglob("trials*.parquet"))
    all_candidates = tsv_candidates or parquet_candidates
    
    if all_candidates:
        return read_table(all_candidates[0])
    
    return None


def stage_trial_table_validate(ctx: BehaviorContext, config: Any) -> Dict[str, Any]:
    """Validate the canonical trial table (non-gating) and write a contract report."""
    from eeg_pipeline.utils.data.trial_table_validation import validate_trial_table
    from eeg_pipeline.infra.tsv import write_tsv

    enabled = bool(get_config_value(ctx.config, "behavior_analysis.trial_table.validate.enabled", True))
    if not enabled:
        return {"enabled": False, "status": "disabled"}

    df = _load_trial_table_df(ctx)
    if not _is_dataframe_valid(df):
        return {"enabled": True, "status": "missing_trial_table"}

    suffix = _feature_suffix_from_context(ctx)
    result = validate_trial_table(df, config=ctx.config)

    out_dir = _get_stats_subfolder(ctx, "trial_table_validation")
    summary_path = out_dir / f"trial_table_validation_summary{suffix}.tsv"
    report_path = out_dir / f"trial_table_validation{suffix}.json"
    try:
        if _is_dataframe_valid(result.summary_df):
            write_tsv(result.summary_df, summary_path)
        report_path.write_text(json.dumps(result.report, indent=2, default=str))
    except Exception as exc:
        ctx.logger.warning("Trial table validation write failed: %s", exc)

    ctx.data_qc["trial_table_validation"] = {
        "enabled": True,
        "status": result.report.get("status", "unknown"),
        "warnings": result.report.get("warnings", []),
        "summary_tsv": summary_path.name,
        "report_json": report_path.name,
    }
    if result.report.get("warnings"):
        ctx.logger.info("Trial table validation warnings: %s", "; ".join(result.report["warnings"][:10]))
    return result.report


@dataclass
class ConfoundsAuditResult:
    """Result from compute_confounds_audit."""
    df: pd.DataFrame
    metadata: Dict[str, Any]


def compute_confounds_audit(
    df_trials: pd.DataFrame,
    config: Any,
    method: str,
    robust_method: Optional[str],
    min_samples: int,
) -> ConfoundsAuditResult:
    """Audit QC confounds against targets.
    
    Single responsibility: Compute confounds audit (no I/O, no state mutation).
    """
    from eeg_pipeline.utils.analysis.stats.confounds import audit_qc_confounds

    audit_df, audit_meta = audit_qc_confounds(
        df_trials,
        config=config,
        targets=["rating", "temperature"],
        method=method,
        robust_method=robust_method,
        min_samples=min_samples,
    )
    return ConfoundsAuditResult(df=audit_df, metadata=audit_meta)


def _crossfit_confound_selection(
    df_trials: pd.DataFrame,
    audit_df: pd.DataFrame,
    run_col: str,
    runs: np.ndarray,
    alpha: float,
    max_covariates: int,
    logger: Any,
) -> Dict[str, Any]:
    """Cross-fitting for confound selection: select on held-in, evaluate on held-out.
    
    This avoids post-selection inference inflation by using a split-sample approach:
    - For each run, select confounds using all OTHER runs
    - Track which confounds are consistently selected across folds
    - Only apply confounds that pass a consistency threshold
    
    Returns:
        Dict with crossfit results including consistent_covariates list
    """
    from collections import Counter
    
    if "qc_col" not in audit_df.columns:
        logger.warning("Crossfit: audit_df missing 'qc_col'; cannot perform crossfit")
        return {"consistent_covariates": [], "fold_selections": {}}
    
    qc_cols = audit_df["qc_col"].unique().tolist()
    if not qc_cols:
        return {"consistent_covariates": [], "fold_selections": {}}
    
    fold_selections: Dict[str, List[str]] = {}
    all_selected: List[str] = []
    
    for held_out_run in runs:
        # Train fold: all runs except held_out_run
        train_mask = df_trials[run_col] != held_out_run
        df_train = df_trials[train_mask]
        
        min_samples_for_crossfit = MIN_SAMPLES_DEFAULT
        if len(df_train) < min_samples_for_crossfit:
            logger.debug(
                "Crossfit fold %s: too few train samples (%d)",
                held_out_run, len(df_train)
            )
            continue
        
        # Select confounds on training data
        fold_selected = []
        for qc_col in qc_cols:
            if qc_col not in df_train.columns:
                continue
            
            if "rating" not in df_train.columns:
                continue
            
            valid_mask = df_train[qc_col].notna() & df_train["rating"].notna()
            if valid_mask.sum() < MIN_SAMPLES_DEFAULT:
                continue
            
            from scipy.stats import spearmanr
            correlation_coef, p_value = spearmanr(
                df_train.loc[valid_mask, qc_col],
                df_train.loc[valid_mask, "rating"]
            )
            if p_value < alpha:
                fold_selected.append(qc_col)
        
        # Limit to max_covariates
        fold_selected = fold_selected[:max_covariates]
        fold_selections[str(held_out_run)] = fold_selected
        all_selected.extend(fold_selected)
    
    # Find covariates selected in majority of folds (>=50%)
    n_folds = len(fold_selections)
    if n_folds == 0:
        return {"consistent_covariates": [], "fold_selections": fold_selections}
    
    selection_counts = Counter(all_selected)
    min_fold_agreement = 0.5
    consistency_threshold = max(1, int(n_folds * min_fold_agreement))
    
    consistent_covariates = [
        cov for cov, count in selection_counts.items()
        if count >= consistency_threshold
    ][:max_covariates]
    
    logger.info(
        "Crossfit: %d folds, %d consistent covariates (threshold=%d)",
        n_folds, len(consistent_covariates), consistency_threshold
    )
    
    return {
        "consistent_covariates": consistent_covariates,
        "fold_selections": fold_selections,
        "selection_counts": dict(selection_counts),
        "consistency_threshold": consistency_threshold,
        "n_folds": n_folds,
    }


def apply_selected_qc_covariates(
    ctx: BehaviorContext,
    df_trials: pd.DataFrame,
    audit_df: pd.DataFrame,
    alpha: float,
    max_covariates: int,
) -> List[str]:
    """Select and apply significant QC covariates to context.
    
    Single responsibility: Mutate ctx.covariates_df with selected QC columns.
    """
    from eeg_pipeline.utils.analysis.stats.confounds import select_significant_qc_covariates
    from eeg_pipeline.utils.formatting import sanitize_label

    qc_cols = select_significant_qc_covariates(
        audit_df,
        config=ctx.config,
        alpha=alpha,
        max_covariates=max_covariates,
        prefer_target="rating",
    )
    if not qc_cols:
        return []

    cov_add = pd.DataFrame(index=np.arange(len(df_trials)))
    for col in qc_cols:
        safe_name = sanitize_label(f"qc_{col}")
        if safe_name in cov_add.columns:
            continue
        cov_add[safe_name] = pd.to_numeric(df_trials[col], errors="coerce")

    if cov_add.empty:
        return []

    if not _is_dataframe_valid(ctx.covariates_df):
        index = ctx.aligned_events.index if ctx.aligned_events is not None else None
        ctx.covariates_df = pd.DataFrame(index=index)

    cov_add.index = ctx.covariates_df.index
    for c in cov_add.columns:
        if c not in ctx.covariates_df.columns:
            ctx.covariates_df[c] = cov_add[c].to_numpy()

    from eeg_pipeline.utils.data.covariates import build_covariates_without_temp

    ctx.covariates_without_temp_df = build_covariates_without_temp(ctx.covariates_df, ctx.temperature_column)
    ctx.data_qc["confounds_qc_covariates_added"] = {
        "columns": list(cov_add.columns),
        "source_metrics": qc_cols,
        "alpha": alpha,
    }
    ctx.logger.info("Added QC covariates: %s", ", ".join(cov_add.columns))

    return list(cov_add.columns)


def stage_confounds(ctx: BehaviorContext, config: Any) -> pd.DataFrame:
    """Audit QC confounds and optionally apply covariates (composed).
    
    Composes: compute_confounds_audit + apply_selected_qc_covariates
    
    Statistical validity modes (behavior_analysis.confounds.selection_mode):
    - "descriptive" (default): Report QC-outcome associations without adjusting downstream.
      This avoids post-selection inflation from "select significant then adjust".
    - "prespecified": Use only pre-specified covariates from config (no data-driven selection).
    - "crossfit": Cross-fitting approach - select covariates on held-in runs, evaluate on held-out.
      This controls post-selection bias at the cost of reduced power.
    """
    from eeg_pipeline.infra.tsv import write_tsv

    df_trials = _load_trial_table_df(ctx)
    if not _is_dataframe_valid(df_trials):
        ctx.logger.warning("Confounds: trial table missing; skipping.")
        return pd.DataFrame()

    method = config.method
    robust_method = config.robust_method
    method_label = config.method_label
    min_samples = int(get_config_value(ctx.config, "behavior_analysis.min_samples.default", 10))

    result = compute_confounds_audit(df_trials, ctx.config, method, robust_method, min_samples)

    suffix = _feature_suffix_from_context(ctx)
    method_suffix = f"_{method_label}" if method_label else ""
    out_dir = _get_stats_subfolder(ctx, "confounds_audit")
    out_path = out_dir / f"confounds_audit{suffix}{method_suffix}.tsv"
    if not result.df.empty:
        write_tsv(result.df, out_path)
        ctx.logger.info("Confounds audit saved: %s/%s (%d rows)", out_dir.name, out_path.name, len(result.df))

    ctx.data_qc["confounds_audit"] = result.metadata

    # Statistical validity: control post-selection inflation
    selection_mode = str(get_config_value(
        ctx.config, "behavior_analysis.confounds.selection_mode", "descriptive"
    )).strip().lower()
    
    if selection_mode == "descriptive":
        ctx.logger.info("Confounds: descriptive mode (no downstream adjustment to avoid post-selection bias)")
        result.metadata["selection_mode"] = "descriptive"
        result.metadata["covariates_applied"] = []
        
    elif selection_mode == "prespecified":
        prespec = get_config_value(ctx.config, "behavior_analysis.confounds.prespecified_covariates", [])
        if prespec:
            ctx.logger.info("Confounds: prespecified mode (%d covariates)", len(prespec))
            ctx.partial_covars = list(prespec)
            result.metadata["selection_mode"] = "prespecified"
            result.metadata["covariates_applied"] = list(prespec)
        else:
            ctx.logger.info("Confounds: prespecified mode but no covariates configured")
            result.metadata["selection_mode"] = "prespecified"
            result.metadata["covariates_applied"] = []
            
    elif selection_mode == "crossfit":
        # Cross-fitting: select covariates on held-in runs, evaluate on held-out
        run_col = str(get_config_value(ctx.config, "behavior_analysis.run_adjustment.column", "run_id") or "run_id").strip()
        alpha = float(get_config_value(ctx.config, "behavior_analysis.statistics.fdr_alpha", 0.05))
        max_cov = int(get_config_value(ctx.config, "behavior_analysis.confounds.max_qc_covariates", 3))
        
        if run_col not in df_trials.columns:
            ctx.logger.warning("Confounds: crossfit requires run column '%s'; falling back to descriptive", run_col)
            result.metadata["selection_mode"] = "descriptive_fallback"
            result.metadata["covariates_applied"] = []
        else:
            runs = df_trials[run_col].unique()
            if len(runs) < 2:
                ctx.logger.warning("Confounds: crossfit requires >=2 runs; falling back to descriptive")
                result.metadata["selection_mode"] = "descriptive_fallback"
                result.metadata["covariates_applied"] = []
            else:
                # Leave-one-run-out cross-fitting
                crossfit_results = _crossfit_confound_selection(
                    df_trials, result.df, run_col, runs, alpha, max_cov, ctx.logger
                )
                
                # Apply covariates that were selected consistently across folds
                consistent_covars = crossfit_results.get("consistent_covariates", [])
                if consistent_covars:
                    ctx.partial_covars = list(consistent_covars)
                    ctx.logger.info("Confounds: crossfit selected %d covariates: %s",
                                   len(consistent_covars), ", ".join(consistent_covars))
                else:
                    ctx.logger.info("Confounds: crossfit found no consistent covariates")

                result.metadata["selection_mode"] = "crossfit"
                result.metadata["covariates_applied"] = consistent_covars
                result.metadata["crossfit_details"] = crossfit_results

    return result.df


def stage_regression(ctx: BehaviorContext, config: Any) -> pd.DataFrame:
    """Trialwise regression: rating (or pain_residual) ~ temperature + trial order + feature (+ interaction).
    
    Supports primary_unit=trial|run to control unit of analysis.
    Run-level aggregates features/outcomes per run before fitting, avoiding pseudo-replication.
    """
    from eeg_pipeline.utils.analysis.stats.trialwise_regression import run_trialwise_feature_regressions
    from eeg_pipeline.infra.tsv import write_tsv

    suffix = _feature_suffix_from_context(ctx)
    method_label = config.method_label
    method_suffix = f"_{method_label}" if method_label else ""

    df_trials = _load_trial_table_df(ctx)
    if not _is_dataframe_valid(df_trials):
        ctx.logger.warning("Regression: trial table missing; skipping.")
        return pd.DataFrame()

    # Unit of analysis control
    primary_unit = str(get_config_value(ctx.config, "behavior_analysis.regression.primary_unit", "trial")).strip().lower()
    use_run_unit = primary_unit in {"run", "run_mean", "runmean", "run_level"}
    run_col = str(get_config_value(ctx.config, "behavior_analysis.run_adjustment.column", "run_id") or "run_id").strip()

    # Feature columns: use naming schema prefixes for safety.
    feature_cols = [c for c in df_trials.columns if str(c).startswith(FEATURE_COLUMN_PREFIXES)]
    feature_cols = _filter_feature_cols_by_band(feature_cols, ctx)

    # Aggregate to run-level if requested (avoids pseudo-replication)
    if use_run_unit and run_col in df_trials.columns:
        ctx.logger.info("Regression: aggregating to run-level (primary_unit=%s)", primary_unit)
        agg_cols = feature_cols + ["rating", "temperature"]
        agg_cols = [c for c in agg_cols if c in df_trials.columns]
        df_trials = df_trials.groupby(run_col)[agg_cols].mean().reset_index()
        ctx.logger.info("  Run-level: %d observations", len(df_trials))

    groups = None
    if getattr(ctx, "group_ids", None) is not None:
        try:
            groups = np.asarray(ctx.group_ids)
        except Exception:
            groups = None

    reg_df, reg_meta = run_trialwise_feature_regressions(
        df_trials,
        feature_cols=feature_cols,
        config=ctx.config,
        groups_for_permutation=groups,
    )
    reg_meta["primary_unit"] = primary_unit
    ctx.data_qc["trialwise_regression"] = reg_meta
    if reg_df is not None and not reg_df.empty and "temperature_control" not in reg_df.columns:
        reg_df = reg_df.copy()
        reg_df["temperature_control"] = reg_meta.get("temperature_control", None)
        reg_df["temperature_control_used"] = reg_meta.get("temperature_control_used", None)
        spline_meta = reg_meta.get("temperature_spline", None)
        if isinstance(spline_meta, dict):
            reg_df["temperature_spline_status"] = spline_meta.get("status", None)
            reg_df["temperature_spline_n_knots"] = spline_meta.get("n_knots", None)
            reg_df["temperature_spline_quantile_low"] = spline_meta.get("quantile_low", None)
            reg_df["temperature_spline_quantile_high"] = spline_meta.get("quantile_high", None)

    out_dir = _get_stats_subfolder(ctx, "trialwise_regression")
    out_path = out_dir / f"regression_feature_effects{suffix}{method_suffix}.tsv"
    if not reg_df.empty:
        write_tsv(reg_df, out_path)
        ctx.logger.info("Regression results saved: %s/%s (%d features)", out_dir.name, out_path.name, len(reg_df))
    return reg_df


def stage_models(ctx: BehaviorContext, config: Any) -> pd.DataFrame:
    """Fit multiple model families per feature (OLS-HC3 / robust / quantile / logistic)."""
    from eeg_pipeline.utils.analysis.stats.feature_models import run_feature_model_families
    from eeg_pipeline.infra.tsv import write_tsv

    suffix = _feature_suffix_from_context(ctx)
    method_label = getattr(config, "method_label", "")
    method_suffix = f"_{method_label}" if method_label else ""

    df_trials = _load_trial_table_df(ctx)
    if not _is_dataframe_valid(df_trials):
        ctx.logger.warning("Models: trial table missing; skipping.")
        return pd.DataFrame()

    feature_cols = [c for c in df_trials.columns if str(c).startswith(FEATURE_COLUMN_PREFIXES)]
    feature_cols = _filter_feature_cols_by_band(feature_cols, ctx)

    model_df, model_meta = run_feature_model_families(
        df_trials,
        feature_cols=feature_cols,
        config=ctx.config,
    )
    ctx.data_qc["feature_models"] = model_meta
    if model_df is not None and not model_df.empty and "temperature_control" not in model_df.columns:
        model_df = model_df.copy()
        model_df["temperature_control"] = model_meta.get("temperature_control", None)
        # Optional per-target (outcome) temperature control diagnostics.
        ctrl_by_out = model_meta.get("temperature_control_by_outcome", None)
        if isinstance(ctrl_by_out, dict) and "target" in model_df.columns:
            used_map = {str(k): (v or {}).get("temperature_control_used", None) for k, v in ctrl_by_out.items()}
            status_map = {}
            nknots_map = {}
            for k, v in ctrl_by_out.items():
                s = (v or {}).get("temperature_spline", None)
                if isinstance(s, dict):
                    status_map[str(k)] = s.get("status", None)
                    nknots_map[str(k)] = s.get("n_knots", None)
            model_df["temperature_control_used"] = model_df["target"].astype(str).map(used_map)
            model_df["temperature_spline_status"] = model_df["target"].astype(str).map(status_map)
            model_df["temperature_spline_n_knots"] = model_df["target"].astype(str).map(nknots_map)

    out_dir = _get_stats_subfolder(ctx, "feature_models")
    out_path = out_dir / f"models_feature_effects{suffix}{method_suffix}.tsv"
    if model_df is not None and not model_df.empty:
        write_tsv(model_df, out_path)
        ctx.logger.info("Model families results saved: %s/%s (%d rows)", out_dir.name, out_path.name, len(model_df))
    return model_df if model_df is not None else pd.DataFrame()


def stage_stability(ctx: BehaviorContext, config: Any) -> pd.DataFrame:
    """Assess within-subject run/block stability of feature→outcome associations (non-gating)."""
    from eeg_pipeline.utils.analysis.stats.stability import compute_groupwise_stability
    from eeg_pipeline.infra.tsv import write_tsv

    suffix = _feature_suffix_from_context(ctx)
    method_label = getattr(config, "method_label", "")
    method_suffix = f"_{method_label}" if method_label else ""

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

    feature_cols = [c for c in df_trials.columns if str(c).startswith(FEATURE_COLUMN_PREFIXES)]
    feature_cols = _filter_feature_cols_by_band(feature_cols, ctx)

    stab_df, stab_meta = compute_groupwise_stability(
        df_trials,
        feature_cols=feature_cols,
        outcome=outcome,
        group_col=group_col,
        config=ctx.config,
    )
    ctx.data_qc["stability_groupwise"] = stab_meta

    out_dir = _get_stats_subfolder(ctx, "stability_groupwise")
    out_path = out_dir / f"stability_groupwise{suffix}{method_suffix}.tsv"
    if stab_df is not None and not stab_df.empty:
        write_tsv(stab_df, out_path)
        ctx.logger.info("Stability results saved: %s/%s (%d features)", out_dir.name, out_path.name, len(stab_df))
    try:
        (out_dir / f"stability_groupwise{suffix}{method_suffix}.metadata.json").write_text(
            json.dumps(stab_meta, indent=2, default=str)
        )
    except Exception:
        pass
    return stab_df if stab_df is not None else pd.DataFrame()


def stage_consistency(ctx: BehaviorContext, config: Any, results: Any) -> pd.DataFrame:
    """Merge correlations/regression/models and flag effect-direction contradictions (non-gating)."""
    from eeg_pipeline.utils.analysis.stats.consistency import build_effect_direction_consistency_summary
    from eeg_pipeline.infra.tsv import write_tsv

    suffix = _feature_suffix_from_context(ctx)
    method_label = getattr(config, "method_label", "")
    method_suffix = f"_{method_label}" if method_label else ""

    corr_df = getattr(results, "correlations", None)
    reg_df = getattr(results, "regression", None)
    models_df = getattr(results, "models", None)
    out_df, meta = build_effect_direction_consistency_summary(
        corr_df=corr_df,
        regression_df=reg_df,
        models_df=models_df,
        config=ctx.config,
    )
    ctx.data_qc["effect_direction_consistency"] = meta
    if out_df is None or out_df.empty:
        return pd.DataFrame()

    out_dir = _get_stats_subfolder(ctx, "consistency_summary")
    out_path = out_dir / f"consistency_summary{suffix}{method_suffix}.tsv"
    write_tsv(out_df, out_path)
    try:
        (out_dir / f"consistency_summary{suffix}{method_suffix}.metadata.json").write_text(
            json.dumps(meta, indent=2, default=str)
        )
    except Exception:
        pass
    ctx.logger.info("Consistency summary saved: %s (%d features)", out_path.name, len(out_df))
    return out_df


def stage_influence(ctx: BehaviorContext, config: Any, results: Any) -> pd.DataFrame:
    """Compute leverage/Cook's summaries for top effects (non-gating)."""
    from eeg_pipeline.utils.analysis.stats.influence import compute_influence_diagnostics
    from eeg_pipeline.infra.tsv import write_tsv

    suffix = _feature_suffix_from_context(ctx)
    method_label = getattr(config, "method_label", "")
    method_suffix = f"_{method_label}" if method_label else ""

    df_trials = _load_trial_table_df(ctx)
    if not _is_dataframe_valid(df_trials):
        ctx.logger.info("Influence: trial table missing; skipping.")
        return pd.DataFrame()

    feature_cols = [c for c in df_trials.columns if str(c).startswith(FEATURE_COLUMN_PREFIXES)]
    feature_cols = _filter_feature_cols_by_band(feature_cols, ctx)

    out_df, meta = compute_influence_diagnostics(
        df_trials,
        feature_cols=feature_cols,
        corr_df=getattr(results, "correlations", None),
        regression_df=getattr(results, "regression", None),
        models_df=getattr(results, "models", None),
        config=ctx.config,
    )
    ctx.data_qc["influence_diagnostics"] = meta
    if not _is_dataframe_valid(out_df):
        return pd.DataFrame()
    # Attach high-level temperature-control info (useful when spline/rating_hat are enabled).
    try:
        if "temperature_control" not in out_df.columns:
            out_df = out_df.copy()
            out_df["temperature_control"] = get_config_value(ctx.config, "behavior_analysis.influence.temperature_control", None)
        ctrl_by_out = meta.get("temperature_control_by_outcome", None) if isinstance(meta, dict) else None
        if isinstance(ctrl_by_out, dict) and "outcome" in out_df.columns:
            used_map = {str(k): (v or {}).get("temperature_control_used", None) for k, v in ctrl_by_out.items()}
            out_df["temperature_control_used"] = out_df["outcome"].astype(str).map(used_map)
    except Exception:
        pass

    out_dir = _get_stats_subfolder(ctx, "influence_diagnostics")
    out_path = out_dir / f"influence_diagnostics{suffix}{method_suffix}.tsv"
    write_tsv(out_df, out_path)
    try:
        (out_dir / f"influence_diagnostics{suffix}{method_suffix}.metadata.json").write_text(
            json.dumps(meta, indent=2, default=str)
        )
    except Exception:
        pass
    ctx.logger.info("Influence diagnostics saved: %s/%s (%d rows)", out_dir.name, out_path.name, len(out_df))
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

    if ctx.targets is not None:
        qc["rating"] = _compute_series_statistics(ctx.targets)

    if ctx.temperature is not None:
        qc["temperature"] = _compute_series_statistics(ctx.temperature)

    if ctx.targets is not None and ctx.temperature is not None:
        s = pd.to_numeric(ctx.targets, errors="coerce")
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

    if ctx.aligned_events is not None and ctx.targets is not None:
        from eeg_pipeline.analysis.behavior.api import split_by_condition

        try:
            pain_mask, nonpain_mask, n_pain, n_nonpain = split_by_condition(
                ctx.aligned_events, ctx.config, ctx.logger
            )
        except Exception as exc:
            qc["pain_vs_nonpain"] = {
                "status": "failed",
                "error": str(exc),
            }
        else:
            if int(n_pain) > 0 or int(n_nonpain) > 0:
                s = pd.to_numeric(ctx.targets, errors="coerce")
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


def _ensure_method_columns(
    df: pd.DataFrame,
    method: str,
    robust_method: Optional[str],
    method_label: str,
) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df = df.copy()
    if "method" not in df.columns:
        df["method"] = method
    if "robust_method" not in df.columns:
        df["robust_method"] = robust_method
    if "method_label" not in df.columns:
        df["method_label"] = method_label
    return df


_STANDARD_COLUMNS = (
    "analysis_type",
    "feature_id",
    "feature_type",
    "target",
    "method",
    "robust_method",
    "method_label",
    "n",
    "r",
    "p_raw",
    "p_primary",
    "p_fdr",
    "notes",
)


def _infer_feature_type(feature: str, config: Any) -> str:
    from eeg_pipeline.domain.features.registry import classify_feature, get_feature_registry
    try:
        registry = get_feature_registry(config)
        ftype, _, _ = classify_feature(feature, include_subtype=False, registry=registry)
        return ftype
    except Exception:
        return "unknown"


def _build_normalized_records(
    ctx: BehaviorContext,
    pipeline_config: Any,
    results: Any,
) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    method = pipeline_config.method
    robust_method = pipeline_config.robust_method
    method_label = pipeline_config.method_label

    corr_df = getattr(results, "correlations", None)
    if corr_df is not None and not corr_df.empty:
        for _, row in corr_df.iterrows():
            feature = row.get("feature")
            feature_type = row.get("feature_type") or _infer_feature_type(str(feature), ctx.config)
            target = row.get("target", "rating")
            p_raw = row.get("p_raw", row.get("p_value", row.get("p")))
            p_primary = row.get("p_primary", p_raw)
            p_fdr = row.get("p_fdr", row.get("q_value"))
            r_val = row.get("r_primary", row.get("r"))
            records.append({
                "analysis_type": "correlations",
                "feature_id": feature,
                "feature_type": feature_type,
                "target": target,
                "method": row.get("method", method),
                "robust_method": row.get("robust_method", robust_method),
                "method_label": row.get("method_label", method_label),
                "n": row.get("n"),
                "r": r_val,
                "p_raw": p_raw,
                "p_primary": p_primary,
                "p_fdr": p_fdr,
                "notes": None,
            })

    psi_df = getattr(results, "pain_sensitivity", None)
    if psi_df is not None and not psi_df.empty:
        for _, row in psi_df.iterrows():
            feature = row.get("feature")
            feature_type = row.get("feature_type") or _infer_feature_type(str(feature), ctx.config)
            p_raw = row.get("p_psi", row.get("p_value"))
            r_val = row.get("r_psi", row.get("r"))
            records.append({
                "analysis_type": "pain_sensitivity",
                "feature_id": feature,
                "feature_type": feature_type,
                "target": row.get("target", "pain_sensitivity"),
                "method": row.get("method", method),
                "robust_method": row.get("robust_method", robust_method),
                "method_label": row.get("method_label", method_label),
                "n": row.get("n"),
                "r": r_val,
                "p_raw": p_raw,
                "p_primary": p_raw,
                "p_fdr": row.get("p_fdr", row.get("q_value")),
                "notes": None,
            })

    cond_df = getattr(results, "condition_effects", None)
    if cond_df is not None and not cond_df.empty:
        for _, row in cond_df.iterrows():
            feature = row.get("feature")
            feature_type = row.get("feature_type") or _infer_feature_type(str(feature), ctx.config)
            n_pain = row.get("n_pain")
            n_nonpain = row.get("n_nonpain")
            n_total = None
            if pd.notna(n_pain) and pd.notna(n_nonpain):
                n_total = int(n_pain) + int(n_nonpain)
            records.append({
                "analysis_type": "condition_effects",
                "feature_id": feature,
                "feature_type": feature_type,
                "target": "pain_vs_nonpain",
                "method": row.get("method", method),
                "robust_method": row.get("robust_method", robust_method),
                "method_label": row.get("method_label", method_label),
                "n": n_total,
                "r": row.get("hedges_g"),
                "p_raw": row.get("p_value"),
                "p_primary": row.get("p_value"),
                "p_fdr": row.get("q_value"),
                "notes": "hedges_g",
            })

    med_df = getattr(results, "mediation", None)
    if med_df is not None and not med_df.empty:
        for _, row in med_df.iterrows():
            feature = row.get("mediator")
            records.append({
                "analysis_type": "mediation",
                "feature_id": feature,
                "feature_type": "mediator",
                "target": "rating",
                "method": method,
                "robust_method": robust_method,
                "method_label": method_label,
                "n": None,
                "r": row.get("indirect_effect"),
                "p_raw": row.get("sobel_p"),
                "p_primary": row.get("sobel_p"),
                "p_fdr": None,
                "notes": "indirect_effect",
            })

    mod_df = getattr(results, "moderation", None)
    if mod_df is not None and not mod_df.empty:
        for _, row in mod_df.iterrows():
            feature = row.get("feature")
            feature_type = row.get("feature_type") or _infer_feature_type(str(feature), ctx.config)
            records.append({
                "analysis_type": "moderation",
                "feature_id": feature,
                "feature_type": feature_type,
                "target": "rating",
                "method": method,
                "robust_method": robust_method,
                "method_label": method_label,
                "n": row.get("n"),
                "r": row.get("b3_interaction"),
                "p_raw": row.get("p_interaction"),
                "p_primary": row.get("p_interaction"),
                "p_fdr": row.get("p_fdr"),
                "notes": "interaction_effect",
            })

    mixed_df = getattr(results, "mixed_effects", None)
    if mixed_df is not None and not mixed_df.empty:
        for _, row in mixed_df.iterrows():
            feature = row.get("feature")
            behavior = row.get("behavior", "rating")
            records.append({
                "analysis_type": "mixed_effects",
                "feature_id": feature,
                "feature_type": row.get("feature_type", "mixed_effects"),
                "target": behavior,
                "method": method,
                "robust_method": robust_method,
                "method_label": method_label,
                "n": row.get("n_observations"),
                "r": row.get("fixed_effect"),
                "p_raw": row.get("fixed_p"),
                "p_primary": row.get("fixed_p"),
                "p_fdr": row.get("fixed_p_fdr"),
                "notes": "fixed_effect",
            })

    return records


def _write_normalized_results(
    ctx: BehaviorContext,
    pipeline_config: Any,
    results: Any,
) -> Optional[Path]:
    records = _build_normalized_records(ctx, pipeline_config, results)
    if not records:
        return None
    df = pd.DataFrame(records)
    df = df.reindex(columns=_STANDARD_COLUMNS)
    
    # Build feature suffix from selected feature files or categories
    feature_files = ctx.selected_feature_files or ctx.feature_categories or []
    feature_suffix = "_" + "_".join(sorted(feature_files)) if feature_files else ""
    
    out_dir = _get_stats_subfolder(ctx, "normalized")
    path = out_dir / f"normalized_results{feature_suffix}.tsv"
    paths = _write_stats_table(ctx, df, path)
    return paths[0] if paths else path


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
) -> None:
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
            "has_trial_table_validation": bool(getattr(results, "trial_table_validation", None)),
            "has_confounds_audit": bool(getattr(results, "confounds", None) is not None and not results.confounds.empty),
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

    (ctx.stats_dir / "analysis_metadata.json").write_text(json.dumps(payload, indent=2, default=str))


###################################################################
# Condition Stage - Single Responsibility Components
###################################################################


def stage_condition_column(
    ctx: BehaviorContext,
    config: Any,
    df_trials: Optional[pd.DataFrame] = None,
    feature_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Run column-based condition comparison (e.g., pain vs non-pain).
    
    Single responsibility: Column contrast comparison.
    Supports primary_unit=trial|run to control unit of analysis.
    """
    from eeg_pipeline.analysis.behavior.api import split_by_condition, compute_condition_effects
    from eeg_pipeline.infra.tsv import write_tsv

    fail_fast = get_config_value(ctx.config, "behavior_analysis.condition.fail_fast", True)
    primary_unit = str(get_config_value(
        ctx.config, "behavior_analysis.condition.primary_unit", "trial"
    )).strip().lower()
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
        feature_cols = [c for c in df_trials.columns if str(c).startswith(FEATURE_COLUMN_PREFIXES)]
        feature_cols = _filter_feature_cols_by_band(feature_cols, ctx)
        feature_cols = _filter_feature_cols_for_computation(feature_cols, "condition", ctx)
    
    # Aggregate to run-level if requested (avoids pseudo-replication)
    compare_col = str(get_config_value(ctx.config, "behavior_analysis.condition.compare_column", "pain") or "pain").strip()
    if use_run_unit and run_col in df_trials.columns and compare_col in df_trials.columns:
        ctx.logger.info("Condition: aggregating to run-level (primary_unit=%s)", primary_unit)
        agg_cols = feature_cols + [compare_col]
        agg_cols = [c for c in agg_cols if c in df_trials.columns]
        # For condition column, take mode of condition and mean of features
        df_agg = df_trials.groupby(run_col)[feature_cols].mean()
        df_agg[compare_col] = df_trials.groupby(run_col)[compare_col].apply(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0])
        df_trials = df_agg.reset_index()
        ctx.logger.info("  Run-level: %d observations", len(df_trials))
    
    if not feature_cols:
        ctx.logger.info("Condition column: no feature columns found; skipping.")
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

        features = df_trials[feature_cols].copy()
        groups = None
        if getattr(ctx, "group_ids", None) is not None:
            try:
                groups = np.asarray(ctx.group_ids)
            except Exception:
                groups = None
        if groups is None:
            run_col = str(get_config_value(ctx.config, "behavior_analysis.run_adjustment.column", "run_id") or "run_id").strip()
            if run_col and run_col in df_trials.columns:
                groups = pd.to_numeric(df_trials[run_col], errors="ignore").to_numpy()

        # Suppress numpy RuntimeWarnings during condition effects computation
        # These occur when computing stats on features with insufficient variance
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            
            column_df = compute_condition_effects(
                features,
                pain_mask,
                nonpain_mask,
                min_samples=0,
                fdr_alpha=config.fdr_alpha,
                logger=ctx.logger,
                n_jobs=config.n_jobs,
                config=ctx.config,
                groups=groups,
            )

        if column_df is not None and not column_df.empty:
            column_df = column_df.copy()
            column_df["comparison_type"] = "column"
            if "p_value" in column_df.columns and "p_raw" not in column_df.columns:
                column_df["p_raw"] = pd.to_numeric(column_df["p_value"], errors="coerce")
            if "p_value" in column_df.columns and "p_primary" not in column_df.columns:
                column_df["p_primary"] = pd.to_numeric(column_df["p_value"], errors="coerce")
            if "q_value" in column_df.columns and "p_fdr" not in column_df.columns:
                column_df["p_fdr"] = pd.to_numeric(column_df["q_value"], errors="coerce")
            
            col_path = out_dir / f"condition_effects_column{suffix}.tsv"
            write_tsv(column_df, col_path)
            ctx.logger.info(f"Condition column comparison: {len(column_df)} features saved to {col_path}")
            return column_df

    except Exception as exc:
        if fail_fast:
            raise
        ctx.logger.warning(f"Condition column comparison failed: {exc}")

    return pd.DataFrame()


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
    from eeg_pipeline.infra.tsv import write_tsv

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
        feature_cols = [c for c in df_trials.columns if str(c).startswith(FEATURE_COLUMN_PREFIXES)]
        feature_cols = _filter_feature_cols_by_band(feature_cols, ctx)
        feature_cols = _filter_feature_cols_for_computation(feature_cols, "condition", ctx)
    
    if not feature_cols:
        ctx.logger.info("Condition window: no feature columns found; skipping.")
        return pd.DataFrame()

    suffix = _feature_suffix_from_context(ctx)
    out_dir = _get_stats_subfolder(ctx, "condition_effects")

    ctx.logger.info(f"Running window comparison: {compare_windows}")
    window_df = _run_window_comparison(
        ctx, df_trials, feature_cols, compare_windows, 0, config.fdr_alpha, suffix
    )
    
    if not window_df.empty:
        win_path = out_dir / f"condition_effects_window{suffix}.tsv"
        write_tsv(window_df, win_path)
        ctx.logger.info(f"Condition window comparison: {len(window_df)} features saved to {win_path}")

    return window_df


def stage_condition(ctx: BehaviorContext, config: Any) -> pd.DataFrame:
    """Backward-compatible condition stage (column + optional window).
    
    The pipeline wrapper historically called a single stage and expected a DataFrame.
    Internally, we keep single-responsibility sub-stages:
    - stage_condition_column
    - stage_condition_window
    """
    df_trials = _load_trial_table_df(ctx)
    if not _is_dataframe_valid(df_trials):
        ctx.logger.warning("Condition: trial table missing; skipping.")
        return pd.DataFrame()

    feature_cols = [c for c in df_trials.columns if str(c).startswith(FEATURE_COLUMN_PREFIXES)]
    feature_cols = _filter_feature_cols_by_band(feature_cols, ctx)
    feature_cols = _filter_feature_cols_for_computation(feature_cols, "condition", ctx)
    if not feature_cols:
        ctx.logger.info("Condition: no feature columns found; skipping.")
        return pd.DataFrame()

    col_df = stage_condition_column(ctx, config, df_trials=df_trials, feature_cols=feature_cols)

    compare_windows = get_config_value(ctx.config, "behavior_analysis.condition.compare_windows", [])
    win_df = stage_condition_window(
        ctx,
        config,
        df_trials=df_trials,
        feature_cols=feature_cols,
        compare_windows=compare_windows if isinstance(compare_windows, list) else None,
    )

    if col_df is not None and not col_df.empty and win_df is not None and not win_df.empty:
        return pd.concat([col_df, win_df], ignore_index=True)
    if col_df is not None and not col_df.empty:
        return col_df
    if win_df is not None and not win_df.empty:
        return win_df
    return pd.DataFrame()


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
    """
    from eeg_pipeline.utils.analysis.stats.fdr import fdr_bh
    from scipy import stats as sp_stats

    if len(windows) < 2:
        ctx.logger.warning("Window comparison requires at least 2 windows")
        return pd.DataFrame()

    window1, window2 = windows[0], windows[1]
    records: List[Dict[str, Any]] = []

    run_col = str(get_config_value(ctx.config, "behavior_analysis.run_adjustment.column", "run_id") or "run_id").strip()
    run_adjust_enabled = bool(get_config_value(ctx.config, "behavior_analysis.run_adjustment.enabled", False))
    use_run_unit = str(
        get_config_value(ctx.config, "behavior_analysis.condition.window_comparison.primary_unit", "trial")
    ).strip().lower() in {"run", "run_mean", "runmean", "run_level"}

    # Parse feature columns to find window-specific features
    # Feature naming pattern: {category}_{window}_{band}_{roi} or similar
    window1_features = {}
    window2_features = {}
    
    for col in feature_cols:
        col_lower = str(col).lower()
        if f"_{window1.lower()}_" in col_lower or col_lower.endswith(f"_{window1.lower()}"):
            base_name = col_lower.replace(f"_{window1.lower()}", "").replace(f"{window1.lower()}_", "")
            window1_features[base_name] = col
        elif f"_{window2.lower()}_" in col_lower or col_lower.endswith(f"_{window2.lower()}"):
            base_name = col_lower.replace(f"_{window2.lower()}", "").replace(f"{window2.lower()}_", "")
            window2_features[base_name] = col

    # Find matching pairs
    common_bases = set(window1_features.keys()) & set(window2_features.keys())
    
    if not common_bases:
        ctx.logger.warning(f"No matching feature pairs found for windows {window1} and {window2}")
        ctx.logger.info(f"Looking for patterns like '*_{window1}_*' and '*_{window2}_*' in feature columns")
        return pd.DataFrame()

    ctx.logger.info(f"Window comparison: found {len(common_bases)} feature pairs for {window1} vs {window2}")

    # Suppress numpy RuntimeWarnings during window comparisons
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        
        for base_name in common_bases:
            col1 = window1_features[base_name]
            col2 = window2_features[base_name]
            
            v1 = pd.to_numeric(df_trials[col1], errors="coerce").values
            v2 = pd.to_numeric(df_trials[col2], errors="coerce").values
            
            valid_mask = np.isfinite(v1) & np.isfinite(v2)
            n_valid = int(valid_mask.sum())
            
            v1_valid = v1[valid_mask]
            v2_valid = v2[valid_mask]
            
            # Paired Wilcoxon signed-rank test
            try:
                stat, p_val = sp_stats.wilcoxon(v1_valid, v2_valid)
            except Exception:
                continue

            # Optional run-level Wilcoxon on per-run means (more conservative unit-of-analysis).
            stat_run = np.nan
            p_val_run = np.nan
            n_runs = np.nan
            if use_run_unit and run_adjust_enabled and run_col in df_trials.columns:
                try:
                    df_run = pd.DataFrame(
                        {
                            "run": df_trials[run_col],
                            "v1": pd.to_numeric(df_trials[col1], errors="coerce"),
                            "v2": pd.to_numeric(df_trials[col2], errors="coerce"),
                        }
                    ).dropna()
                    run_means = df_run.groupby("run", dropna=True)[["v1", "v2"]].mean(numeric_only=True)
                    n_runs = int(len(run_means))
                    if n_runs >= 2:
                        stat_run, p_val_run = sp_stats.wilcoxon(run_means["v1"].to_numpy(), run_means["v2"].to_numpy())
                except Exception:
                    stat_run = np.nan
                    p_val_run = np.nan
                    n_runs = np.nan
            
            # Effect sizes
            diff = v2_valid - v1_valid
            mean_diff = float(np.nanmean(diff))
            pooled_std = float(np.sqrt((np.nanvar(v1_valid) + np.nanvar(v2_valid)) / 2))
            cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0.0
            
            # Hedge's g correction
            n = len(v1_valid)
            hedges_correction = 1 - (3 / (4 * n - 1)) if n > 1 else 1.0
            hedges_g = cohens_d * hedges_correction
            
            records.append({
                "feature": base_name,
                "feature_type": _infer_feature_type(col1, ctx.config),
                "comparison_type": "window",
                "window1": window1,
                "window2": window2,
                "n_pairs": n_valid,
                "n_runs": n_runs,
                "mean_window1": float(np.nanmean(v1_valid)),
                "mean_window2": float(np.nanmean(v2_valid)),
                "std_window1": float(np.nanstd(v1_valid, ddof=1)),
                "std_window2": float(np.nanstd(v2_valid, ddof=1)),
                "mean_diff": mean_diff,
                "statistic": float(stat),
                "p_value": float(p_val),
                "statistic_run": float(stat_run) if np.isfinite(stat_run) else np.nan,
                "p_value_run": float(p_val_run) if np.isfinite(p_val_run) else np.nan,
                "cohens_d": cohens_d,
                "hedges_g": hedges_g,
            })

    if not records:
        ctx.logger.info("Window comparison: no valid feature pairs with sufficient samples")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    
    # Apply FDR correction
    df["p_raw"] = pd.to_numeric(df["p_value"], errors="coerce")
    if use_run_unit and "p_value_run" in df.columns:
        p_primary = pd.to_numeric(df["p_value_run"], errors="coerce")
        df["p_primary"] = p_primary.where(p_primary.notna(), df["p_raw"])
    else:
        df["p_primary"] = df["p_raw"]
    p_vals = pd.to_numeric(df["p_primary"], errors="coerce").to_numpy()
    df["p_fdr"] = fdr_bh(p_vals, alpha=fdr_alpha, config=ctx.config)
    df["q_value"] = df["p_fdr"]

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
    try:
        tf_results = compute_time_frequency_from_context(ctx)
        return tf_results
    except Exception as exc:
        ctx.logger.error(f"Time-frequency correlations failed: {exc}")
        return None


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
    from eeg_pipeline.infra.tsv import write_tsv
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
    fdr_alpha = float(get_config_value(ctx.config, "behavior_analysis.statistics.fdr_alpha", 0.05))
    ctx.logger.info(f"Temporal: using {correction_method} correction (alpha={fdr_alpha})")

    results: Dict[str, Optional[Dict[str, Any]]] = {
        "power": None,
        "itpc": None,
        "erds": None,
    }

    ctx.logger.info("Computing temporal correlations by condition...")
    try:
        results["power"] = compute_temporal_from_context(ctx)
    except Exception as exc:
        ctx.logger.error(f"Temporal correlations failed: {exc}")

    if "itpc" in selected_features:
        ctx.logger.info("Computing ITPC temporal correlations...")
        try:
            itpc_results = compute_itpc_temporal_from_context(ctx)
            results["itpc"] = itpc_results
            if itpc_results:
                ctx.logger.info(
                    f"ITPC temporal: {itpc_results.get('n_tests', 0)} tests, "
                    f"{itpc_results.get('n_sig_raw', 0)} significant"
                )
        except Exception as exc:
            ctx.logger.error(f"ITPC temporal correlations failed: {exc}")

    if "erds" in selected_features:
        from eeg_pipeline.utils.analysis.stats.temporal import compute_erds_temporal_from_context
        ctx.logger.info("Computing ERDS temporal correlations...")
        try:
            erds_results = compute_erds_temporal_from_context(ctx)
            results["erds"] = erds_results
            if erds_results:
                ctx.logger.info(
                    f"ERDS temporal: {erds_results.get('n_tests', 0)} tests, "
                    f"{erds_results.get('n_sig_raw', 0)} significant"
                )
        except Exception as exc:
            ctx.logger.error(f"ERDS temporal correlations failed: {exc}")

    all_temporal_records = []
    for res in results.values():
        if res and "records" in res:
            all_temporal_records.extend(res["records"])
    
    if all_temporal_records:
        out_dir = ctx.stats_dir / "temporal_correlations"
        ensure_dir(out_dir)
        sfx = "_spearman" if ctx.use_spearman else "_pearson"
        
        # Apply family-wise correction across all temporal tests
        df_temporal = pd.DataFrame(all_temporal_records)
        if "p_value" in df_temporal.columns or "p_raw" in df_temporal.columns:
            p_col = "p_raw" if "p_raw" in df_temporal.columns else "p_value"
            p_vals = pd.to_numeric(df_temporal[p_col], errors="coerce").fillna(1.0).to_numpy()
            
            if correction_method == "fdr":
                reject, p_corrected, _, _ = multipletests(p_vals, alpha=fdr_alpha, method="fdr_bh")
                df_temporal["p_fdr"] = p_corrected
                df_temporal["sig_fdr"] = reject
                n_sig = int(reject.sum())
                ctx.logger.info(f"Temporal FDR: {n_sig}/{len(p_vals)} significant at alpha={fdr_alpha}")
                
            elif correction_method == "bonferroni":
                reject, p_corrected, _, _ = multipletests(p_vals, alpha=fdr_alpha, method="bonferroni")
                df_temporal["p_bonferroni"] = p_corrected
                df_temporal["sig_bonferroni"] = reject
                n_sig = int(reject.sum())
                ctx.logger.info(f"Temporal Bonferroni: {n_sig}/{len(p_vals)} significant at alpha={fdr_alpha}")
                
            elif correction_method == "cluster":
                ctx.logger.warning(
                    "Temporal: cluster correction requested but requires spatial/temporal adjacency info. "
                    "Falling back to FDR correction."
                )
                reject, p_corrected, _, _ = multipletests(p_vals, alpha=fdr_alpha, method="fdr_bh")
                df_temporal["p_fdr"] = p_corrected
                df_temporal["sig_fdr"] = reject
                df_temporal["correction_note"] = "fdr_fallback_from_cluster"
                
            elif correction_method == "none":
                ctx.logger.warning("Temporal: no multiple comparison correction applied (use with caution)")
                df_temporal["sig_raw"] = p_vals < fdr_alpha
            
            df_temporal["correction_method"] = correction_method
        
        combined_tsv_path = out_dir / f"corr_stats_temporal_combined{sfx}.tsv"
        write_tsv(df_temporal, combined_tsv_path)
        ctx.logger.info(
            f"Saved combined temporal correlations: {len(all_temporal_records)} tests -> {combined_tsv_path.name}"
        )

    return results


def stage_temporal_topomaps(ctx: BehaviorContext) -> bool:
    """Run power topomap correlations.
    
    Single responsibility: Topomap correlation computation.
    """
    from eeg_pipeline.analysis.behavior.api import run_power_topomap_correlations

    ctx.logger.info("Computing topomap correlations...")
    try:
        run_power_topomap_correlations(
            subject=ctx.subject,
            task=ctx.task,
            power_df=ctx.power_df,
            temperature=ctx.temperature,
            epochs_info=ctx.epochs_info,
            stats_dir=ctx.stats_dir,
            config=ctx.config,
            logger=ctx.logger,
            use_spearman=ctx.use_spearman,
            rng=ctx.rng,
            bootstrap=ctx.bootstrap,
            n_perm=ctx.n_perm,
        )
        return True
    except Exception as exc:
        ctx.logger.error(f"Topomap correlations failed: {exc}")
        return False


def stage_cluster(ctx: BehaviorContext, config: Any) -> Dict[str, Any]:
    from eeg_pipeline.analysis.behavior.api import run_cluster_test_from_context

    ctx.logger.info("Running cluster permutation tests...")
    ctx.n_perm = config.n_permutations

    try:
        results = run_cluster_test_from_context(ctx)
        return results if results else {"status": "completed"}
    except Exception as exc:
        ctx.logger.warning(f"Cluster tests failed: {exc}")
        return {"status": "failed", "error": str(exc)}


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

    feature_cols = [c for c in df_trials.columns if str(c).startswith(FEATURE_COLUMN_PREFIXES)]
    feature_cols = _filter_feature_cols_by_band(feature_cols, ctx)
    feature_cols = _filter_feature_cols_for_computation(feature_cols, "mediation", ctx)
    
    if not feature_cols:
        ctx.logger.info("Mediation: no feature columns found; skipping.")
        return pd.DataFrame()

    ctx.logger.info("Running mediation analysis...")
    n_bootstrap = int(get_config_value(ctx.config, "behavior_analysis.mediation.n_bootstrap", 1000))
    min_effect_size = float(get_config_value(ctx.config, "behavior_analysis.mediation.min_effect_size", 0.05))
    max_mediators = get_config_value(ctx.config, "behavior_analysis.mediation.max_mediators", None)

    if max_mediators is not None:
        max_mediators = int(max_mediators)
        variances = df_trials[feature_cols].var()
        mediators = variances.nlargest(max(1, max_mediators)).index.tolist()
        ctx.logger.info("Limiting to top %d mediators by variance", max_mediators)
    else:
        mediators = feature_cols
        ctx.logger.info("Testing all %d features as mediators (no limit)", len(mediators))

    result = run_mediation_analysis(
        df_trials,
        "temperature",
        mediators,
        "rating",
        n_bootstrap=n_bootstrap,
        min_effect_size=min_effect_size,
    )
    return result if result is not None else pd.DataFrame()


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
    from eeg_pipeline.infra.tsv import read_tsv
    
    try:
        import statsmodels.formula.api as smf
        from statsmodels.regression.mixed_linear_model import MixedLM
    except ImportError:
        logger.warning("statsmodels not available; skipping mixed-effects.")
        return MixedEffectsResult(df=pd.DataFrame(), metadata={"status": "statsmodels_unavailable"})
    
    all_trials: List[pd.DataFrame] = []
    
    for sub in subjects:
        stats_dir = deriv_stats_path(deriv_root, sub)
        trial_path = stats_dir / "trial_table.tsv"
        if not trial_path.exists():
            trial_path = stats_dir / "trials_with_features.tsv"
        
        if not trial_path.exists():
            logger.warning(f"No trial table for sub-{sub}; skipping.")
            continue
        
        df = read_tsv(trial_path)
        if df is None or df.empty:
            continue
        
        df["subject_id"] = sub
        all_trials.append(df)
    
    if len(all_trials) < 2:
        logger.warning("Mixed-effects requires >=2 subjects; only %d found.", len(all_trials))
        return MixedEffectsResult(df=pd.DataFrame(), metadata={"status": "insufficient_subjects"})
    
    combined = pd.concat(all_trials, ignore_index=True)
    logger.info("Mixed-effects: %d subjects, %d total trials", len(all_trials), len(combined))
    
    if "rating" not in combined.columns:
        logger.warning("Mixed-effects: 'rating' column not found.")
        return MixedEffectsResult(df=pd.DataFrame(), metadata={"status": "no_rating_column"})
    
    feature_cols = [c for c in combined.columns if str(c).startswith(FEATURE_COLUMN_PREFIXES)]
    
    if len(feature_cols) > max_features:
        variances = combined[feature_cols].var()
        feature_cols = variances.nlargest(max_features).index.tolist()
        logger.info("Mixed-effects: limited to top %d features by variance", max_features)
    
    records: List[Dict[str, Any]] = []
    family_records: List[Dict[str, Any]] = []
    
    for feat in feature_cols:
        feat_type = _infer_feature_type(str(feat), config)
        family_id = f"mixed_{feat_type}"
        
        df_valid = combined[["subject_id", "rating", feat]].dropna()
        if len(df_valid) < 10 or df_valid["subject_id"].nunique() < 2:
            continue
        
        df_valid = df_valid.rename(columns={feat: "feature_value"})
        
        try:
            if random_effects == "slope":
                model = smf.mixedlm(
                    "rating ~ feature_value",
                    df_valid,
                    groups=df_valid["subject_id"],
                    re_formula="~feature_value"
                )
            else:
                model = smf.mixedlm(
                    "rating ~ feature_value",
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
                "aic": result.aic,
                "bic": result.bic,
                "converged": result.converged,
            })
            
            family_records.append({
                "feature": str(feat),
                "family_id": family_id,
                "family_kind": "feature_type",
            })
            
        except Exception as e:
            logger.warning(f"Mixed-effects failed for {feat}: {e}")
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
    from eeg_pipeline.infra.tsv import read_tsv
    from eeg_pipeline.utils.analysis.stats.fdr import hierarchical_fdr
    
    all_trials: List[pd.DataFrame] = []
    
    for sub in subjects:
        stats_dir = deriv_stats_path(deriv_root, sub)
        trial_path = stats_dir / "trial_table.tsv"
        if not trial_path.exists():
            trial_path = stats_dir / "trials_with_features.tsv"
        
        if not trial_path.exists():
            continue
        
        df = read_tsv(trial_path)
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
    
    records: List[Dict[str, Any]] = []
    rng = np.random.default_rng(42)
    
    for feat in feature_cols:
        feat_type = _infer_feature_type(str(feat), config)
        family_id = f"corr_{feat_type}"
        
        feature_vals = pd.to_numeric(combined[feat], errors="coerce").to_numpy()
        valid_mask = np.isfinite(rating) & np.isfinite(feature_vals)
        
        if valid_mask.sum() < 10:
            continue
        
        r_obs, _ = compute_correlation(
            feature_vals[valid_mask],
            rating[valid_mask],
            method="spearman",
        )
        
        if not np.isfinite(r_obs):
            continue
        
        # Block/run-restricted permutation for valid p-values
        if use_block_permutation and block_col is not None:
            blocks = combined[block_col].to_numpy()[valid_mask]
            subjects_arr = combined["subject_id"].to_numpy()[valid_mask]
            
            null_rs = []
            for _ in range(n_perm):
                rating_perm = rating[valid_mask].copy()
                
                # Permute within each subject-block combination
                for subj in np.unique(subjects_arr):
                    subj_mask = subjects_arr == subj
                    for blk in np.unique(blocks[subj_mask]):
                        block_mask = subj_mask & (blocks == blk)
                        rating_perm[block_mask] = rng.permutation(rating_perm[block_mask])
                
                r_perm, _ = compute_correlation(feature_vals[valid_mask], rating_perm, method="spearman")
                if np.isfinite(r_perm):
                    null_rs.append(r_perm)
            
            if null_rs:
                p_perm = (np.sum(np.abs(null_rs) >= np.abs(r_obs)) + 1) / (len(null_rs) + 1)
            else:
                p_perm = np.nan
        else:
            # Standard permutation (ignores block structure)
            null_rs = []
            for _ in range(n_perm):
                rating_perm = rng.permutation(rating[valid_mask])
                r_perm, _ = compute_correlation(feature_vals[valid_mask], rating_perm, method="spearman")
                if np.isfinite(r_perm):
                    null_rs.append(r_perm)
            
            p_perm = (np.sum(np.abs(null_rs) >= np.abs(r_obs)) + 1) / (len(null_rs) + 1) if null_rs else np.nan
        
        records.append({
            "feature": str(feat),
            "feature_type": feat_type,
            "family_id": family_id,
            "family_kind": "feature_type",
            "r": r_obs,
            "n": int(valid_mask.sum()),
            "n_subjects": combined.loc[valid_mask, "subject_id"].nunique(),
            "p_perm": p_perm,
            "permutation_method": "block_restricted" if use_block_permutation and block_col else "standard",
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
    run_mixed_effects: bool = True,
    run_multilevel_correlations: bool = True,
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
    run_mixed_effects : bool
        Run mixed-effects models
    run_multilevel_correlations : bool
        Run multilevel correlations
    output_dir : Path, optional
        Output directory for results
    
    Returns
    -------
    GroupLevelResult
        Aggregated group-level results
    """
    from eeg_pipeline.infra.tsv import write_tsv
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
            max_features=int(get_config_value(config, "behavior_analysis.mixed_effects.max_features", 50)),
            fdr_alpha=float(get_config_value(config, "behavior_analysis.statistics.fdr_alpha", 0.05)),
        )
        
        if output_dir and mixed_result.df is not None and not mixed_result.df.empty:
            ensure_dir(output_dir)
            write_tsv(mixed_result.df, output_dir / "group_mixed_effects.tsv")
            logger.info("Saved mixed-effects results: %s", output_dir / "group_mixed_effects.tsv")
    
    if run_multilevel_correlations:
        logger.info("Running multilevel correlations with block-restricted permutations...")
        multilevel_df = run_group_level_correlations(
            subjects=subjects,
            deriv_root=deriv_root,
            config=config,
            logger=logger,
            use_block_permutation=bool(get_config_value(config, "behavior_analysis.group_level.block_permutation", True)),
            n_perm=int(get_config_value(config, "behavior_analysis.statistics.n_permutations", 1000)),
            fdr_alpha=float(get_config_value(config, "behavior_analysis.statistics.fdr_alpha", 0.05)),
        )
        
        if output_dir and multilevel_df is not None and not multilevel_df.empty:
            ensure_dir(output_dir)
            write_tsv(multilevel_df, output_dir / "group_multilevel_correlations.tsv")
            logger.info("Saved multilevel correlations: %s", output_dir / "group_multilevel_correlations.tsv")
    
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
    from eeg_pipeline.utils.analysis.stats.fdr import fdr_bh
    from eeg_pipeline.infra.tsv import write_tsv

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

    feature_cols = [c for c in df_trials.columns if str(c).startswith(FEATURE_COLUMN_PREFIXES)]
    feature_cols = _filter_feature_cols_by_band(feature_cols, ctx)
    feature_cols = _filter_feature_cols_for_computation(feature_cols, "moderation", ctx)

    if not feature_cols:
        ctx.logger.info("Moderation: no feature columns found; skipping.")
        return pd.DataFrame()

    max_features = getattr(config, "moderation_max_features", None)  # None = unlimited
    fdr_alpha = float(getattr(config, "fdr_alpha", 0.05))

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

        result = run_moderation_analysis(
            X=temperature[valid_mask],
            W=feature_values[valid_mask],
            Y=rating[valid_mask],
            x_label="temperature",
            w_label=str(feat),
            y_label="rating",
            center_predictors=True,
        )

        rec = {
            "feature": str(feat),
            "feature_type": _infer_feature_type(str(feat), ctx.config),
            "n": result.n,
            "b1_temperature": result.b1,
            "b2_feature": result.b2,
            "b3_interaction": result.b3,
            "se_b3": result.se_b3,
            "p_interaction": result.p_b3,
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
            "significant_moderation": result.is_significant_moderation(fdr_alpha),
        }
        records.append(rec)

    mod_df = pd.DataFrame(records) if records else pd.DataFrame()

    if not mod_df.empty:
        p_vals = pd.to_numeric(mod_df["p_interaction"], errors="coerce").to_numpy()
        mod_df["p_fdr"] = fdr_bh(p_vals, alpha=fdr_alpha, config=ctx.config)
        mod_df["p_raw"] = mod_df["p_interaction"]
        mod_df["p_primary"] = mod_df["p_interaction"]

    out_dir = _get_stats_subfolder(ctx, "moderation")
    out_path = out_dir / f"moderation_results{suffix}{method_suffix}.tsv"
    if not mod_df.empty:
        write_tsv(mod_df, out_path)
        n_sig = int((mod_df["p_fdr"] < fdr_alpha).sum())
        ctx.logger.info(
            "Moderation: %d features tested, %d significant (FDR < %.2f)",
            len(mod_df), n_sig, fdr_alpha
        )
    else:
        ctx.logger.info("Moderation: no valid results.")

    return mod_df


def _get_fdr_patterns(config: Any) -> List[str]:
    """Get file patterns for FDR correction based on config flags."""
    patterns = []
    
    if getattr(config, "run_condition_comparison", False):
        patterns.append("condition_effects*.tsv")
    if getattr(config, "run_confounds", False):
        patterns.append("confounds_audit*.tsv")
    if getattr(config, "run_regression", False):
        patterns.append("regression_feature_effects*.tsv")
    if getattr(config, "run_models", False):
        patterns.append("models_feature_effects*.tsv")
    if getattr(config, "run_mediation", False):
        patterns.append("mediation*.tsv")
    if getattr(config, "run_moderation", False):
        patterns.append("moderation_results*.tsv")
    if getattr(config, "run_mixed_effects", False):
        patterns.append("mixed_effects*.tsv")
    if getattr(config, "run_correlations", True):
        patterns.append("correlations*.tsv")
    if getattr(config, "compute_pain_sensitivity", False):
        patterns.append("pain_sensitivity*.tsv")
    if getattr(config, "run_temporal_correlations", False):
        patterns.extend([
            "corr_stats_tf_*.tsv",
            "corr_stats_temporal_*.tsv",
            "*_topomap_*_correlations_*.tsv"
        ])
    if getattr(config, "run_cluster_tests", False):
        patterns.append("cluster_results_*.tsv")

    patterns = list(set(patterns))
    if not patterns:
        patterns = ["correlations*.tsv", "condition_effects*.tsv", "pain_sensitivity*.tsv"]
    return patterns


def stage_global_fdr(ctx: BehaviorContext, config: Any) -> Dict[str, Any]:
    """Apply global FDR correction across all analysis outputs.
    
    Single responsibility: Global FDR correction.
    """
    from eeg_pipeline.utils.analysis.stats.fdr import apply_global_fdr

    patterns = _get_fdr_patterns(config)
    fdr_alpha = config.fdr_alpha

    ctx.logger.info("Applying global FDR correction...")
    summary = apply_global_fdr(
        ctx.stats_dir,
        alpha=fdr_alpha,
        logger=ctx.logger,
        include_glob=patterns,
    )
    return summary


def stage_hierarchical_fdr_summary(ctx: BehaviorContext, config: Any) -> pd.DataFrame:
    """Compute hierarchical FDR summary across analysis types.
    
    Single responsibility: Hierarchical FDR summary computation and output.
    """
    from eeg_pipeline.utils.analysis.stats.reliability import compute_hierarchical_fdr_summary

    patterns = _get_fdr_patterns(config)
    fdr_alpha = config.fdr_alpha

    ctx.logger.info("Computing hierarchical FDR summary...")
    try:
        hier_summary = compute_hierarchical_fdr_summary(
            ctx.stats_dir,
            alpha=fdr_alpha,
            config=ctx.config,
            include_glob=patterns,
        )
        if not hier_summary.empty:
            hier_dir = _get_stats_subfolder(ctx, "fdr")
            hier_path = hier_dir / "hierarchical_fdr_summary.tsv"
            hier_summary.to_csv(hier_path, sep="\t", index=False)
            ctx.logger.info(f"Hierarchical FDR summary saved to {hier_path}")
            
            for _, row in hier_summary.iterrows():
                ctx.logger.info(
                    f"  {row['analysis_type']}: {row['n_reject_within']}/{row['n_tests']} "
                    f"({row['pct_reject_within']:.1f}%) reject within-family"
                )
        return hier_summary
    except Exception as exc:
        ctx.logger.warning(f"Hierarchical FDR failed: {exc}")
        return pd.DataFrame()


def _apply_global_fdr_to_results(
    ctx: BehaviorContext,
    config: Any,
    results: Any,
) -> None:
    """Update results object with global FDR values from disk.
    
    Single responsibility: Merge q_global into in-memory results.
    """
    from eeg_pipeline.infra.tsv import read_tsv

    fdr_alpha = config.fdr_alpha
    mapping = {
        "confounds": "confounds_audit*.tsv",
        "regression": "regression_feature_effects*.tsv",
        "models": "models_feature_effects*.tsv",
        "correlations": "correlations*.tsv",
        "pain_sensitivity": "pain_sensitivity*.tsv",
        "condition_effects": "condition_effects*.tsv",
        "tf": "corr_stats_tf_*.tsv",
        "temporal": "corr_stats_temporal_*.tsv",
        "cluster": "cluster_results_*.tsv",
        "mediation": "mediation*.tsv",
        "moderation": "moderation_results*.tsv",
        "mixed_effects": "mixed_effects*.tsv",
    }
    
    for attr, pattern in mapping.items():
        current_value = getattr(results, attr, None)
        if current_value is None:
            continue
            
        match_files = list(ctx.stats_dir.rglob(pattern))
        if not match_files:
            continue
        
        for fpath in match_files:
            df_disk = read_tsv(fpath)
            if df_disk is None or "q_global" not in df_disk.columns:
                continue

            if isinstance(current_value, pd.DataFrame):
                if current_value.empty:
                    continue
                merge_cols = [c for c in ["feature", "feature_id", "target", "mediator"] if c in df_disk.columns]
                if merge_cols:
                    if "q_global" in current_value.columns:
                        current_value = current_value.drop(columns=["q_global"])
                    current_value = pd.merge(current_value, df_disk[merge_cols + ["q_global"]], on=merge_cols, how="left")
                    setattr(results, attr, current_value)
                elif len(current_value) == len(df_disk):
                    current_value["q_global"] = df_disk["q_global"].values
                    setattr(results, attr, current_value)
            elif isinstance(current_value, dict):
                if attr == "cluster":
                    band_name = fpath.stem.replace("cluster_results_", "")
                    if band_name in current_value and "cluster_records" in current_value[band_name]:
                        recs = current_value[band_name]["cluster_records"]
                        if len(recs) == len(df_disk):
                            for r, q in zip(recs, df_disk["q_global"]):
                                r["q_global"] = q
                else:
                    current_value["n_sig_fdr"] = int((df_disk["q_global"] < fdr_alpha).sum())
                    setattr(results, attr, current_value)


def stage_validate(ctx: BehaviorContext, config: Any, results: Optional[Any] = None) -> None:
    """Apply global FDR and compute hierarchical summary (composed).
    
    Composes: stage_global_fdr + stage_hierarchical_fdr_summary + _apply_global_fdr_to_results
    """
    stage_global_fdr(ctx, config)
    
    if results is not None:
        _apply_global_fdr_to_results(ctx, config, results)

    stage_hierarchical_fdr_summary(ctx, config)


def stage_report(ctx: BehaviorContext, pipeline_config: Any) -> Optional[Path]:
    """Write a single-subject, self-diagnosing Markdown report (non-gating)."""
    suffix = _feature_suffix_from_context(ctx)
    method_label = getattr(pipeline_config, "method_label", "")
    method_suffix = f"_{method_label}" if method_label else ""

    top_n = int(get_config_value(ctx.config, "behavior_analysis.report.top_n", 15))
    alpha = float(getattr(pipeline_config, "fdr_alpha", 0.05))

    # Trial-table summary
    df_trials = _load_trial_table_df(ctx)
    n_trials = int(len(df_trials)) if df_trials is not None else 0
    n_features = 0
    if df_trials is not None and not df_trials.empty:
        n_features = int(sum(1 for c in df_trials.columns if str(c).startswith(FEATURE_COLUMN_PREFIXES)))

    # Trial-table validation (if present)
    val_status = None
    val_warnings: List[str] = []
    try:
        vpath = _find_stats_path(ctx, f"trial_table_validation{suffix}.json")
        if vpath and vpath.exists():
            payload = json.loads(vpath.read_text())
            val_status = payload.get("status")
            val_warnings = [str(x) for x in (payload.get("warnings") or [])]
    except Exception:
        pass

    def _read_tsv(path: Path) -> Optional[pd.DataFrame]:
        try:
            return pd.read_csv(path, sep="\t")
        except Exception:
            return None

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
        "correlations*.tsv",
        "pain_sensitivity*.tsv",
        "regression_feature_effects*.tsv",
        "models_feature_effects*.tsv",
        "condition_effects*.tsv",
        "consistency_summary*.tsv",
        "influence_diagnostics*.tsv",
        "trial_table_validation_summary*.tsv",
        "temperature_model_comparison*.tsv",
        "temperature_breakpoint_candidates*.tsv",
        "hierarchical_fdr_summary.tsv",
        "normalized_results*.tsv",
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
    if val_status is not None:
        lines.append(f"- Trial table validation: `{val_status}`")
    if val_warnings:
        lines.append("")
        lines.append("## Validation Warnings")
        for w in val_warnings[: min(20, len(val_warnings))]:
            lines.append(f"- {w}")
        if len(val_warnings) > 20:
            lines.append(f"- ... {len(val_warnings) - 20} more")

    # Summaries per output file (TSVs)
    tsvs = [p for p in files if p.suffix == ".tsv"]
    if tsvs:
        lines.append("")
        lines.append("## Outputs")
        for p in tsvs:
            df = _read_tsv(p)
            if df is None or df.empty:
                lines.append(f"- `{p.name}`: (empty or unreadable)")
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
    try:
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        ctx.logger.info("Subject report saved: %s/%s", report_dir.name, out_path.name)
        return out_path
    except Exception as exc:
        ctx.logger.warning("Failed to write subject report: %s", exc)
        return None


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
    from eeg_pipeline.infra.paths import ensure_dir
    from eeg_pipeline.infra.tsv import write_tsv

    ensure_dir(ctx.stats_dir)
    saved: List[Path] = []
    robust_method = pipeline_config.robust_method
    method_label = pipeline_config.method_label
    method_suffix = f"_{method_label}" if method_label else ""
    
    # Build feature suffix from selected feature files or categories
    feature_files = ctx.selected_feature_files or ctx.feature_categories or []
    feature_suffix = "_" + "_".join(sorted(feature_files)) if feature_files else ""

    # Use _is_valid_df to safely check results before accessing .empty
    # (Failed stages return dicts instead of DataFrames)
    corr = getattr(results, "correlations", None)
    if _is_valid_df(corr):
        results.correlations = _ensure_method_columns(
            results.correlations, pipeline_config.method, robust_method, method_label
        )
    
    pain_sens = getattr(results, "pain_sensitivity", None)
    if _is_valid_df(pain_sens):
        results.pain_sensitivity = _ensure_method_columns(
            results.pain_sensitivity, pipeline_config.method, robust_method, method_label
        )
    
    cond_eff = getattr(results, "condition_effects", None)
    if _is_valid_df(cond_eff):
        results.condition_effects = _ensure_method_columns(
            results.condition_effects, pipeline_config.method, robust_method, method_label
        )
    
    med = getattr(results, "mediation", None)
    if _is_valid_df(med):
        results.mediation = _ensure_method_columns(
            results.mediation, pipeline_config.method, robust_method, method_label
        )
    
    mixed = getattr(results, "mixed_effects", None)
    if _is_valid_df(mixed):
        results.mixed_effects = _ensure_method_columns(
            results.mixed_effects, pipeline_config.method, robust_method, method_label
        )
    
    conf = getattr(results, "confounds", None)
    if _is_valid_df(conf):
        results.confounds = _ensure_method_columns(
            results.confounds, pipeline_config.method, robust_method, method_label
        )
    
    reg = getattr(results, "regression", None)
    if _is_valid_df(reg):
        results.regression = _ensure_method_columns(
            results.regression, pipeline_config.method, robust_method, method_label
        )
    
    mod = getattr(results, "models", None)
    if _is_valid_df(mod):
        results.models = _ensure_method_columns(
            results.models, pipeline_config.method, robust_method, method_label
        )

    # Write results using _is_valid_df for safe checks
    if _is_valid_df(getattr(results, "correlations", None)):
        out_dir = _get_stats_subfolder(ctx, "correlations")
        path = out_dir / f"correlations{feature_suffix}{method_suffix}.tsv"
        saved.extend(_write_stats_table(ctx, results.correlations, path))

    if _is_valid_df(getattr(results, "pain_sensitivity", None)):
        out_dir = _get_stats_subfolder(ctx, "pain_sensitivity")
        path = out_dir / f"pain_sensitivity{feature_suffix}{method_suffix}.tsv"
        saved.extend(_write_stats_table(ctx, results.pain_sensitivity, path))

    if _is_valid_df(getattr(results, "condition_effects", None)):
        out_dir = _get_stats_subfolder(ctx, "condition_effects")
        path = out_dir / f"condition_effects{feature_suffix}.tsv"
        saved.extend(_write_stats_table(ctx, results.condition_effects, path))

    if _is_valid_df(getattr(results, "mediation", None)):
        out_dir = _get_stats_subfolder(ctx, "mediation")
        path = out_dir / f"mediation{feature_suffix}.tsv"
        saved.extend(_write_stats_table(ctx, results.mediation, path))

    if _is_valid_df(getattr(results, "mixed_effects", None)):
        out_dir = _get_stats_subfolder(ctx, "mixed_effects")
        path = out_dir / f"mixed_effects{feature_suffix}.tsv"
        saved.extend(_write_stats_table(ctx, results.mixed_effects, path))

    # NOTE: confounds, regression, and models are already written by their respective stages.
    # We only ensure method columns are updated in-memory (done above) for consistency.
    # The files already exist in confounds_audit/, trialwise_regression/, and feature_models/ folders.
    if _is_valid_df(getattr(results, "confounds", None)):
        out_dir = _get_stats_subfolder(ctx, "confounds_audit")
        path = out_dir / f"confounds_audit{feature_suffix}{method_suffix}.tsv"
        if path.exists():
            saved.append(path)

    if _is_valid_df(getattr(results, "regression", None)):
        out_dir = _get_stats_subfolder(ctx, "trialwise_regression")
        path = out_dir / f"regression_feature_effects{feature_suffix}{method_suffix}.tsv"
        if path.exists():
            saved.append(path)

    if _is_valid_df(getattr(results, "models", None)):
        out_dir = _get_stats_subfolder(ctx, "feature_models")
        path = out_dir / f"models_feature_effects{feature_suffix}{method_suffix}.tsv"
        if path.exists():
            saved.append(path)

    normalized_path = _write_normalized_results(ctx, pipeline_config, results)
    if normalized_path is not None:
        saved.append(normalized_path)

    if results is None:
        ctx.logger.warning("Export stage: results is None, skipping summary generation")
    else:
        summary = results.to_summary()
        (ctx.stats_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
        saved.append(ctx.stats_dir / "summary.json")

    ctx.logger.info(f"Saved {len(saved)} output files")
    return saved


def _infer_output_kind(name: str) -> str:
    if name.startswith("corr_stats_"):
        return "correlations"
    if name.startswith("correlations"):
        return "correlations"
    if name.startswith("pain_sensitivity"):
        return "pain_sensitivity"
    if name.startswith("condition_effects"):
        return "condition_effects"
    if name.startswith("mediation"):
        return "mediation"
    if name.startswith("moderation"):
        return "moderation"
    if name.startswith("mixed_effects"):
        return "mixed_effects"
    if name.startswith("confounds_audit"):
        return "confounds_audit"
    if name.startswith("regression_feature_effects"):
        return "trialwise_regression"
    if name.startswith("models_feature_effects"):
        return "feature_models"
    if name.startswith("trials_with_lags"):
        return "lag_features"
    if name.startswith("trials_with_residual"):
        return "pain_residual"
    if name.startswith("lag_features"):
        return "lag_features"
    if name.startswith("pain_residual"):
        return "pain_residual"
    if name.startswith("model_comparison"):
        return "temperature_models"
    if name.startswith("breakpoint_candidates") or name.startswith("breakpoint_test"):
        return "temperature_models"
    if name.startswith("trials"):
        return "trial_table"
    if name.startswith("trial_table_validation"):
        return "trial_table_validation"
    if name.startswith("temperature_model_comparison"):
        return "temperature_model_comparison"
    if name.startswith("temperature_breakpoint"):
        return "temperature_breakpoint_test"
    if name.startswith("stability_groupwise"):
        return "stability_groupwise"
    if name.startswith("consistency_summary"):
        return "consistency_summary"
    if name.startswith("influence_diagnostics"):
        return "influence_diagnostics"
    if name.startswith("normalized_results"):
        return "normalized"
    if name.startswith("feature_screening"):
        return "feature_screening"
    if name.startswith("paired_comparisons"):
        return "paired_comparisons"
    if name.startswith("summary"):
        return "summary"
    if name.startswith("analysis_metadata"):
        return "analysis_metadata"
    if name.startswith("subject_report"):
        return "subject_report"
    if name.startswith("time_frequency_correlation_data"):
        return "time_frequency"
    if name.startswith("temporal_correlations"):
        return "temporal_correlations"
    if name.startswith("hierarchical_fdr_summary"):
        return "fdr"
    return "unknown"


def _count_rows(path: Path) -> Optional[int]:
    if path.suffix not in {".tsv", ".csv"}:
        return None
    try:
        with path.open("r") as f:
            header = f.readline()
            if not header:
                return 0
            return sum(1 for _ in f)
    except Exception:
        return None


def write_outputs_manifest(
    ctx: BehaviorContext,
    pipeline_config: Any,
    results: Any,
    stage_metrics: Optional[Dict[str, Any]] = None,
) -> Path:
    from datetime import datetime

    outputs = []
    for path in sorted(p for p in ctx.stats_dir.rglob("*") if p.is_file() and p.name != "outputs_manifest.json"):
        # Skip hidden files or logs
        if path.name.startswith(".") or path.suffix == ".log":
            continue
        outputs.append({
            "name": path.name,
            "path": str(path),
            "kind": _infer_output_kind(path.name),
            "subfolder": path.parent.name if path.parent != ctx.stats_dir else None,
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
            "rating": bool(ctx.targets is not None and ctx.targets.notna().any()) if ctx.targets is not None else False,
            "temperature": bool(ctx.temperature is not None and ctx.temperature.notna().any()) if ctx.temperature is not None else False,
        },
        "covariates_qc": ctx.data_qc.get("covariates_qc", {}),
        "outputs": outputs,
    }

    if stage_metrics:
        payload["stage_metrics"] = stage_metrics

    path = ctx.stats_dir / "outputs_manifest.json"
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path
