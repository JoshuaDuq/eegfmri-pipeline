from __future__ import annotations


def test_assert_confirmatory_pipeline_ready_accepts_completed_protocol_components() -> None:
    from studies.pain_study.study2.implementation_status import (
        assert_confirmatory_pipeline_ready,
    )

    assert_confirmatory_pipeline_ready()


def test_study2_public_api_exports_analysis_core_components() -> None:
    import studies.pain_study.study2 as study2

    assert callable(study2.compute_group_source_inference)
    assert callable(study2.evaluate_directional_consistency)
    assert callable(study2.evaluate_artifact_controls)
    assert callable(study2.evaluate_robustness_summary)
    assert callable(study2.compute_spatial_correspondence)
    assert callable(study2.compute_behavioral_convergence)
    assert callable(study2.bootstrap_mean_interval)
    assert callable(study2.compute_haufe_pattern)
    assert callable(study2.compute_source_family_inference)
    assert callable(study2.compute_sloreta_source_estimates)
    assert callable(study2.compute_sloreta_hilbert_logratio_power)
    assert callable(study2.compute_point_spread_fwhm)
    assert callable(study2.run_target_retrained_source_permutations)
    assert callable(study2.prepare_source_stage_association_inputs)
    assert callable(study2.compute_subject_source_association_map)
    assert callable(study2.compute_band_unique_subject_source_association_map)
    assert callable(study2.compute_cohort_source_association_maps)
    assert callable(study2.compute_band_unique_cohort_source_association_maps)
