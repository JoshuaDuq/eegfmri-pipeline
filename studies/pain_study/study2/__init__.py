"""Study 2: source-space analyses of NPS-predictive EEG activity."""

from studies.pain_study.study2.association import (
    SourcePowerAssociationMap,
    compute_source_power_association_map,
)
from studies.pain_study.study2.artifact_controls import (
    ArtifactControlQC,
    RobustnessQC,
    evaluate_artifact_controls,
    evaluate_robustness_summary,
)
from studies.pain_study.study2.behavioral_convergence import (
    BehavioralConvergenceResult,
    compute_behavioral_convergence,
)
from studies.pain_study.study2.contributions import (
    compute_band_contribution_scores,
    standardize_contribution_scores,
)
from studies.pain_study.study2.directional_consistency import (
    DirectionalConsistencyQC,
    evaluate_directional_consistency,
)
from studies.pain_study.study2.gates import (
    Study1ConfirmatoryCriteriaQC,
    evaluate_study1_confirmatory_criteria,
)
from studies.pain_study.study2.haufe import (
    HaufePattern,
    compute_haufe_pattern,
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
from studies.pain_study.study2.point_spread import (
    PointSpreadFWHMReport,
    compute_point_spread_fwhm,
)
from studies.pain_study.study2.reporting import (
    BootstrapMeanInterval,
    bootstrap_mean_interval,
)
from studies.pain_study.study2.source_power import (
    SourcePowerExtraction,
    compute_sloreta_hilbert_logratio_power,
    compute_sloreta_source_estimates,
)
from studies.pain_study.study2.source_maps import (
    CohortSourceAssociationResult,
    SubjectSourceAssociationResult,
    compute_band_unique_cohort_source_association_maps,
    compute_band_unique_subject_source_association_map,
    compute_cohort_source_association_maps,
    compute_subject_source_association_map,
)
from studies.pain_study.study2.source_inference import (
    GroupSourceInferenceResult,
    SourceCluster,
    compute_group_source_inference,
)
from studies.pain_study.study2.source_family import (
    SourceFamilyBandResult,
    SourceFamilyInferenceResult,
    compute_source_family_inference,
)
from studies.pain_study.study2.source_stage import (
    SourceStageAssociationInputs,
    SourceStageCohortQC,
    SourceStageSubjectQC,
    evaluate_band_unique_source_stage_cohort,
    evaluate_band_unique_source_stage_subject,
    evaluate_source_stage_cohort,
    evaluate_source_stage_subject,
    prepare_band_unique_source_stage_association_inputs,
    prepare_source_stage_association_inputs,
)
from studies.pain_study.study2.spatial_comparison import (
    SpatialCorrespondenceResult,
    compute_spatial_correspondence,
)
from studies.pain_study.study2.target_permutations import (
    InvalidPermutationDraw,
    TargetRetrainedSourcePermutationResult,
    run_target_retrained_source_permutations,
)
from studies.pain_study.study2.study1_context import load_study1_model_context
from studies.pain_study.study2.target_retrained_null import (
    Study1ModelContext,
    build_target_retrained_null_maps,
)

__all__ = [
    "ArtifactControlQC",
    "BehavioralConvergenceResult",
    "BootstrapMeanInterval",
    "CohortSourceAssociationResult",
    "DirectionalConsistencyQC",
    "GroupSourceInferenceResult",
    "HaufePattern",
    "InvalidPermutationDraw",
    "PointSpreadFWHMReport",
    "RobustnessQC",
    "SourceFamilyBandResult",
    "SourceFamilyInferenceResult",
    "SourcePowerExtraction",
    "SourcePowerAssociationMap",
    "SourceCluster",
    "SourceModelQC",
    "SpatialCorrespondenceResult",
    "Study1ConfirmatoryCriteriaQC",
    "Study2ImplementationComponent",
    "SubjectSourceAssociationResult",
    "SourceStageAssociationInputs",
    "SourceStageCohortQC",
    "SourceStageSubjectQC",
    "STUDY2_IMPLEMENTATION_COMPONENTS",
    "assert_confirmatory_pipeline_ready",
    "bootstrap_mean_interval",
    "compute_point_spread_fwhm",
    "compute_source_power_association_map",
    "compute_band_unique_cohort_source_association_maps",
    "compute_band_unique_subject_source_association_map",
    "compute_behavioral_convergence",
    "compute_cohort_source_association_maps",
    "compute_source_family_inference",
    "compute_group_source_inference",
    "compute_haufe_pattern",
    "compute_sloreta_hilbert_logratio_power",
    "compute_sloreta_source_estimates",
    "compute_spatial_correspondence",
    "compute_subject_source_association_map",
    "evaluate_artifact_controls",
    "evaluate_robustness_summary",
    "evaluate_directional_consistency",
    "evaluate_study1_confirmatory_criteria",
    "evaluate_source_model_qc",
    "evaluate_band_unique_source_stage_cohort",
    "evaluate_band_unique_source_stage_subject",
    "evaluate_source_stage_cohort",
    "evaluate_source_stage_subject",
    "compute_band_contribution_scores",
    "prepare_band_unique_source_stage_association_inputs",
    "prepare_source_stage_association_inputs",
    "run_target_retrained_source_permutations",
    "build_target_retrained_null_maps",
    "load_study1_model_context",
    "Study1ModelContext",
    "standardize_contribution_scores",
    "TargetRetrainedSourcePermutationResult",
    "unimplemented_confirmatory_components",
]
