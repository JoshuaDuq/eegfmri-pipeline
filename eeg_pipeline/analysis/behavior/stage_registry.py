from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from eeg_pipeline.utils.config.loader import get_config_value

if TYPE_CHECKING:
    from eeg_pipeline.context.behavior import BehaviorContext


@dataclass(frozen=True)
class StageSpec:
    """Specification for a pipeline stage."""

    name: str
    description: str
    requires: Tuple[str, ...]
    produces: Tuple[str, ...]
    config_key: Optional[str] = None
    group: str = "core"


class StageRegistry:
    """Registry of pipeline stages with dependency resolution."""

    _stages: Dict[str, StageSpec] = {}

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
        spec = cls._stages.get(stage_name)
        if not spec:
            return []

        prerequisites: List[str] = []
        for req in spec.requires:
            for other_name, other_spec in cls._stages.items():
                if req in other_spec.produces and other_name != stage_name:
                    if other_name not in prerequisites:
                        prerequisites.append(other_name)
                    for pre in cls.get_prerequisites(other_name):
                        if pre not in prerequisites:
                            prerequisites.append(pre)
        return prerequisites

    @classmethod
    def validate_stage_combo(cls, stages: List[str]) -> Tuple[bool, List[str]]:
        missing: List[str] = []
        for stage in stages:
            prereqs = cls.get_prerequisites(stage)
            for prereq in prereqs:
                if prereq not in stages and prereq not in missing:
                    missing.append(prereq)
        return (len(missing) == 0, missing)

    @classmethod
    def auto_resolve_stages(cls, stages: List[str]) -> List[str]:
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
        resolved = cls.auto_resolve_stages(stages)
        steps: List[Dict[str, Any]] = []
        for i, stage in enumerate(resolved):
            spec = cls._stages.get(stage)
            if spec:
                steps.append(
                    {
                        "index": i,
                        "name": stage,
                        "description": spec.description,
                        "group": spec.group,
                        "total": len(resolved),
                    }
                )
        return steps

    @classmethod
    def get_available_stages_for_context(cls, ctx: "BehaviorContext") -> List[str]:
        available_resources = set()

        if ctx.aligned_events is not None:
            available_resources.add(cls.RESOURCE_TRIAL_TABLE)
        if ctx.power_df is not None and not ctx.power_df.empty:
            available_resources.add(cls.RESOURCE_POWER_DF)
            available_resources.add(cls.RESOURCE_FEATURES)
        if ctx.temperature is not None:
            available_resources.add(cls.RESOURCE_TEMPERATURE)
        if ctx.aligned_events is not None:
            rating_col = ctx._find_rating_column() if hasattr(ctx, "_find_rating_column") else None
            if rating_col is not None:
                available_resources.add(cls.RESOURCE_RATING)
        if ctx.epochs_info is not None:
            available_resources.add(cls.RESOURCE_EPOCHS)

        available: List[str] = []
        for name, spec in cls._stages.items():
            can_run = True
            for req in spec.requires:
                is_stage_output = any(req in s.produces for s in cls._stages.values())
                if not is_stage_output and req not in available_resources:
                    can_run = False
                    break
            if can_run:
                available.append(name)
        return available

    @classmethod
    def list_stages(cls) -> List[Dict[str, Any]]:
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


class _ResultsFromOutputs:
    """Lightweight results object built from DAG stage outputs for export stage."""

    def __init__(self, outputs: Dict[str, Any]):
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
        self.icc = outputs.get("icc")
        self.consistency = outputs.get("consistency")
        self.influence = outputs.get("influence")
        self.trial_table_path = outputs.get("trial_table")
        self.report_path = outputs.get("report")
        self.subject = None
        self.summary = {}

    def to_summary(self) -> Dict[str, Any]:
        summary = {}
        if self.trial_table_path:
            summary["trial_table_path"] = str(self.trial_table_path)
        if self.report_path:
            summary["report_path"] = str(self.report_path)

        n_total = 0
        n_sig_raw = 0
        n_sig_fdr = 0

        if self.correlations is not None and hasattr(self.correlations, "empty") and not self.correlations.empty:
            n_total = len(self.correlations)
            if "p_raw" in self.correlations.columns:
                n_sig_raw = int((self.correlations["p_raw"] < 0.05).sum())
            if "p_fdr" in self.correlations.columns:
                n_sig_fdr = int((self.correlations["p_fdr"] < 0.05).sum())

        summary["n_features"] = n_total
        summary["n_significant_raw"] = n_sig_raw
        summary["n_significant_fdr"] = n_sig_fdr

        return summary


def build_results_from_outputs(outputs: Dict[str, Any]) -> Any:
    """Build a results-like object from DAG stage outputs."""
    if not outputs:
        return None
    return _ResultsFromOutputs(outputs)


def is_stage_enabled_by_config(stage_name: str, config: Any) -> bool:
    """Check if stage is enabled via its config_key."""
    spec = StageRegistry.get(stage_name)
    if spec is None or spec.config_key is None:
        return True

    config_value = get_config_value(config, spec.config_key, None)
    if config_value is None:
        return True

    return bool(config_value)


def config_to_stage_names(pipeline_config: Any) -> List[str]:
    """Map pipeline config flags to stage names for DAG-based execution."""
    stages = ["load"]

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
    if getattr(pipeline_config, "run_icc", False):
        stages.append("icc")
    if getattr(pipeline_config, "run_correlations", True):
        stages.extend(
            [
                "correlate_design",
                "correlate_effect_sizes",
                "correlate_pvalues",
                "correlate_primary_selection",
                "correlate_fdr",
            ]
        )
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

    stages.append("export")
    return stages


_STAGE_SPECS = [
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
        name="icc",
        description="Run-to-run feature reliability",
        requires=(StageRegistry.RESOURCE_TRIAL_TABLE,),
        produces=("icc",),
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
