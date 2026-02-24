from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Mapping, Tuple


@dataclass(frozen=True)
class PipelineStageRule:
    """Mapping from a pipeline-config flag to one or more stage names."""

    flag_attr: str
    stage_names: Tuple[str, ...]
    default: bool = False


@dataclass(frozen=True)
class StageSpecDefinition:
    """Serializable stage specification source for StageRegistry bootstrapping."""

    name: str
    description: str
    requires: Tuple[str, ...]
    produces: Tuple[str, ...]
    config_key: str | None = None
    group: str = "core"


# Ordered rules to preserve stage execution ordering from config flags.
PIPELINE_STAGE_RULES: Tuple[PipelineStageRule, ...] = (
    PipelineStageRule("run_trial_table", ("trial_table",), default=True),
    PipelineStageRule("run_lag_features", ("lag_features",)),
    PipelineStageRule("run_predictor_residual", ("predictor_residual",)),
    PipelineStageRule("run_predictor_models", ("predictor_models",)),
    PipelineStageRule("run_feature_qc", ("feature_qc",)),
    PipelineStageRule("run_regression", ("regression",)),
    PipelineStageRule("run_models", ("models",)),
    PipelineStageRule("run_stability", ("stability",)),
    PipelineStageRule("run_icc", ("icc",)),
    PipelineStageRule(
        "run_correlations",
        (
            "correlate_design",
            "correlate_effect_sizes",
            "correlate_pvalues",
            "correlate_primary_selection",
            "correlate_fdr",
        ),
        default=True,
    ),
    PipelineStageRule("compute_predictor_sensitivity", ("predictor_sensitivity",)),
    PipelineStageRule("run_consistency", ("consistency",)),
    PipelineStageRule("run_influence", ("influence",)),
    PipelineStageRule("run_condition_comparison", ("condition_column", "condition_window")),
    PipelineStageRule("run_temporal_correlations", ("temporal_tfr", "temporal_stats")),
    PipelineStageRule("run_cluster_tests", ("cluster",)),
    PipelineStageRule("run_mediation", ("mediation",)),
    PipelineStageRule("run_moderation", ("moderation",)),
    PipelineStageRule("run_mixed_effects", ("mixed_effects",)),
    PipelineStageRule("run_validation", ("hierarchical_fdr_summary",), default=True),
    PipelineStageRule("run_report", ("report",)),
)


# Keep this mapping as the single source for --computations overrides.
COMPUTATION_TO_PIPELINE_ATTR = {
    "trial_table": "run_trial_table",
    "lag_features": "run_lag_features",
    "predictor_residual": "run_predictor_residual",
    "predictor_models": "run_predictor_models",
    "regression": "run_regression",
    "models": "run_models",
    "stability": "run_stability",
    "icc": "run_icc",
    "consistency": "run_consistency",
    "influence": "run_influence",
    "report": "run_report",
    "correlations": "run_correlations",
    "predictor_sensitivity": "compute_predictor_sensitivity",
    "condition": "run_condition_comparison",
    "temporal": "run_temporal_correlations",
    "cluster": "run_cluster_tests",
    "mediation": "run_mediation",
    "moderation": "run_moderation",
    "mixed_effects": "run_mixed_effects",
    "multilevel_correlations": "run_multilevel_correlations",
}

# Stage metadata lives with stage-selection rules to avoid split ownership.
STAGE_SPEC_DEFINITIONS: Tuple[StageSpecDefinition, ...] = (
    StageSpecDefinition(
        name="load",
        description="Load behavioral data",
        requires=(),
        produces=("rating", "predictor", "features"),
        group="data_prep",
    ),
    StageSpecDefinition(
        name="trial_table",
        description="Build canonical trial table",
        requires=("features",),
        produces=("trial_table",),
        config_key="behavior_analysis.trial_table.enabled",
        group="data_prep",
    ),
    StageSpecDefinition(
        name="lag_features",
        description="Add lag/delta features",
        requires=("trial_table",),
        produces=("lag_features",),
        config_key="behavior_analysis.lag_features.enabled",
        group="data_prep",
    ),
    StageSpecDefinition(
        name="predictor_residual",
        description="Compute pain residual",
        requires=("trial_table", "predictor", "rating"),
        produces=("predictor_residual",),
        config_key="behavior_analysis.predictor_residual.enabled",
        group="data_prep",
    ),
    StageSpecDefinition(
        name="predictor_models",
        description="Compare predictor-outcome models",
        requires=("trial_table", "predictor", "rating"),
        produces=("predictor_models",),
        config_key="behavior_analysis.predictor_models.enabled",
        group="data_prep",
    ),
    StageSpecDefinition(
        name="feature_qc",
        description="Screen feature quality",
        requires=("trial_table",),
        produces=("feature_qc",),
        config_key="behavior_analysis.feature_qc.enabled",
        group="data_prep",
    ),
    StageSpecDefinition(
        name="correlate_design",
        description="Assemble correlation design matrix",
        requires=("trial_table",),
        produces=("correlate_design",),
        config_key="behavior_analysis.correlations.enabled",
        group="correlations",
    ),
    StageSpecDefinition(
        name="correlate_effect_sizes",
        description="Compute correlation effect sizes",
        requires=("correlate_design",),
        produces=("effect_sizes",),
        config_key="behavior_analysis.correlations.enabled",
        group="correlations",
    ),
    StageSpecDefinition(
        name="correlate_pvalues",
        description="Compute permutation p-values",
        requires=("correlate_design", "effect_sizes"),
        produces=("pvalues",),
        config_key="behavior_analysis.correlations.permutation.enabled",
        group="correlations",
    ),
    StageSpecDefinition(
        name="correlate_primary_selection",
        description="Select primary p-value and effect size for FDR",
        requires=("correlate_design", "effect_sizes"),
        produces=("primary_selection",),
        config_key="behavior_analysis.correlations.enabled",
        group="correlations",
    ),
    StageSpecDefinition(
        name="correlate_fdr",
        description="Apply FDR correction",
        requires=("primary_selection",),
        produces=("correlations",),
        config_key="behavior_analysis.correlations.enabled",
        group="correlations",
    ),
    StageSpecDefinition(
        name="predictor_sensitivity",
        description="Pain sensitivity correlations",
        requires=("trial_table", "predictor", "rating"),
        produces=("predictor_sensitivity",),
        config_key="behavior_analysis.predictor_sensitivity.enabled",
        group="correlations",
    ),
    StageSpecDefinition(
        name="regression",
        description="Trialwise regression",
        requires=("trial_table",),
        produces=("regression",),
        config_key="behavior_analysis.regression.enabled",
        group="correlations",
    ),
    StageSpecDefinition(
        name="models",
        description="Fit model families",
        requires=("trial_table",),
        produces=("models",),
        config_key="behavior_analysis.models.enabled",
        group="correlations",
    ),
    StageSpecDefinition(
        name="stability",
        description="Groupwise stability",
        requires=("trial_table",),
        produces=("stability",),
        config_key="behavior_analysis.stability.enabled",
        group="correlations",
    ),
    StageSpecDefinition(
        name="icc",
        description="Run-to-run feature reliability",
        requires=("trial_table",),
        produces=("icc",),
        config_key="behavior_analysis.icc.enabled",
        group="correlations",
    ),
    StageSpecDefinition(
        name="consistency",
        description="Effect direction consistency",
        requires=("correlations",),
        produces=("consistency",),
        config_key="behavior_analysis.consistency.enabled",
        group="validation",
    ),
    StageSpecDefinition(
        name="influence",
        description="Influence diagnostics",
        requires=("trial_table", "correlations"),
        produces=("influence",),
        config_key="behavior_analysis.influence.enabled",
        group="validation",
    ),
    StageSpecDefinition(
        name="condition_column",
        description="Column-based condition contrast",
        requires=("trial_table",),
        produces=("condition_effects",),
        config_key="behavior_analysis.condition.enabled",
        group="condition",
    ),
    StageSpecDefinition(
        name="condition_window",
        description="Window-based condition contrast",
        requires=("trial_table",),
        produces=("window_effects",),
        config_key="behavior_analysis.condition.compare_windows",
        group="condition",
    ),
    StageSpecDefinition(
        name="temporal_tfr",
        description="Time-frequency correlations",
        requires=("tfr", "rating"),
        produces=("tfr_correlations",),
        config_key="behavior_analysis.temporal.enabled",
        group="temporal",
    ),
    StageSpecDefinition(
        name="temporal_stats",
        description="Temporal statistics (power/ITPC/ERDS)",
        requires=("power_df", "rating"),
        produces=("temporal_stats",),
        config_key="behavior_analysis.temporal.enabled",
        group="temporal",
    ),
    StageSpecDefinition(
        name="cluster",
        description="Cluster permutation tests",
        requires=("epochs",),
        produces=("cluster_results",),
        config_key="behavior_analysis.cluster.enabled",
        group="temporal",
    ),
    StageSpecDefinition(
        name="mediation",
        description="Mediation analysis",
        requires=("trial_table", "predictor", "rating"),
        produces=("mediation",),
        config_key="behavior_analysis.mediation.enabled",
        group="advanced",
    ),
    StageSpecDefinition(
        name="moderation",
        description="Moderation analysis",
        requires=("trial_table", "predictor", "rating"),
        produces=("moderation",),
        config_key="behavior_analysis.moderation.enabled",
        group="advanced",
    ),
    StageSpecDefinition(
        name="mixed_effects",
        description="Mixed-effects models (group-level)",
        requires=("trial_table",),
        produces=("mixed_effects",),
        config_key="behavior_analysis.mixed_effects.enabled",
        group="advanced",
    ),
    StageSpecDefinition(
        name="hierarchical_fdr_summary",
        description="Summarize hierarchical FDR across analyses",
        requires=(),
        produces=("hierarchical_fdr_summary",),
        config_key="behavior_analysis.validation.enabled",
        group="validation",
    ),
    StageSpecDefinition(
        name="report",
        description="Generate subject report",
        requires=(),
        produces=("report",),
        config_key="behavior_analysis.report.enabled",
        group="export",
    ),
    StageSpecDefinition(
        name="export",
        description="Export results",
        requires=(),
        produces=("exports",),
        group="export",
    ),
)


def config_to_stage_names_impl(pipeline_config: Any) -> List[str]:
    """Map pipeline config flags to stage names for DAG-based execution."""
    stages = ["load"]
    for rule in PIPELINE_STAGE_RULES:
        if bool(getattr(pipeline_config, rule.flag_attr, rule.default)):
            stages.extend(rule.stage_names)
    stages.append("export")
    return stages


def apply_computation_flags_impl(
    pipeline_config: Any,
    computation_flags: Mapping[str, bool],
) -> None:
    """Apply normalized --computations flags onto BehaviorPipelineConfig."""
    unknown = sorted(set(computation_flags) - set(COMPUTATION_TO_PIPELINE_ATTR))
    if unknown:
        raise KeyError(
            f"Unknown computation flags without pipeline mapping: {', '.join(unknown)}"
        )

    for computation_name, attr_name in COMPUTATION_TO_PIPELINE_ATTR.items():
        if computation_name in computation_flags:
            setattr(pipeline_config, attr_name, bool(computation_flags[computation_name]))

    # Preserve established behavior where requesting stability also enables ICC.
    if bool(computation_flags.get("stability", False)):
        setattr(pipeline_config, "run_icc", True)
