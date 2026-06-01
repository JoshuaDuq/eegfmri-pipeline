"""Study 2: source interpretation of NPS-predictive EEG activity."""

from studies.pain_study.study2.association import (
    SourcePowerAssociationMap,
    compute_source_power_association_map,
)
from studies.pain_study.study2.contributions import (
    compute_band_contribution_scores,
    standardize_contribution_scores,
)
from studies.pain_study.study2.gates import (
    Study1ConfirmatoryGateQC,
    evaluate_study1_confirmatory_gates,
)
from studies.pain_study.study2.implementation_status import (
    STUDY2_IMPLEMENTATION_COMPONENTS,
    Study2ImplementationComponent,
    assert_confirmatory_pipeline_ready,
    unimplemented_confirmatory_components,
)
from studies.pain_study.study2.source_model_qc import (
    SourceModelQC,
    evaluate_source_model_qc,
)
from studies.pain_study.study2.source_stage import (
    SourceStageCohortQC,
    SourceStageSubjectQC,
    evaluate_band_unique_source_stage_cohort,
    evaluate_band_unique_source_stage_subject,
    evaluate_source_stage_cohort,
    evaluate_source_stage_subject,
)

__all__ = [
    "SourcePowerAssociationMap",
    "SourceModelQC",
    "Study1ConfirmatoryGateQC",
    "Study2ImplementationComponent",
    "SourceStageCohortQC",
    "SourceStageSubjectQC",
    "STUDY2_IMPLEMENTATION_COMPONENTS",
    "assert_confirmatory_pipeline_ready",
    "compute_source_power_association_map",
    "evaluate_study1_confirmatory_gates",
    "evaluate_source_model_qc",
    "evaluate_band_unique_source_stage_cohort",
    "evaluate_band_unique_source_stage_subject",
    "evaluate_source_stage_cohort",
    "evaluate_source_stage_subject",
    "compute_band_contribution_scores",
    "standardize_contribution_scores",
    "unimplemented_confirmatory_components",
]
