from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from eeg_pipeline.analysis.behavior.stage_catalog import (
    STAGE_SPEC_DEFINITIONS,
    config_to_stage_names_impl as _config_to_stage_names_impl,
)
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
    RESOURCE_PREDICTOR = "predictor"
    RESOURCE_OUTCOME = "outcome"
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
        if ctx.predictor_series is not None:
            available_resources.add(cls.RESOURCE_PREDICTOR)
        if ctx.aligned_events is not None:
            outcome_col = ctx._find_outcome_column() if hasattr(ctx, "_find_outcome_column") else None
            if outcome_col is not None:
                available_resources.add(cls.RESOURCE_OUTCOME)
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
        self.condition_effects = outputs.get("condition_column")
        self.regression = outputs.get("regression")
        self.icc = outputs.get("icc")
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
    return _config_to_stage_names_impl(pipeline_config)


for _spec in STAGE_SPEC_DEFINITIONS:
    StageRegistry.register(
        StageSpec(
            name=_spec.name,
            description=_spec.description,
            requires=_spec.requires,
            produces=_spec.produces,
            config_key=_spec.config_key,
            group=_spec.group,
        )
    )
