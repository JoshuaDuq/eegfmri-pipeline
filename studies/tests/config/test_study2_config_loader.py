from __future__ import annotations


def test_study2_default_confirmatory_cell_matches_readme() -> None:
    from studies.pain_study.study2.config import load_study2_config

    config = load_study2_config()
    cell = config["study2"]["confirmatory"]["study1_cell"]

    assert cell["target"] == "NPS"
    assert cell["model"] == "elasticnet"
    assert cell["feature_family"] == "power"
    assert cell["frequency_preset"] == "alpha_beta"
    assert cell["feature_resolution"] == "individual_channel"
    assert cell["residualization_level"] == 2
    assert cell["contribution_bands"] == ["alpha", "beta"]

    gates = config["study2"]["confirmatory"]["study1_gates"]
    assert gates["max_holm_p"] == 0.05
    assert gates["min_delta_r2"] == 0.02
    assert gates["min_delta_r2_lower_ci"] == 0.005
    assert gates["min_level2_delta_r2"] == 0.005
    assert gates["require_positive_within_subject_delta_r2"] is True
    assert gates["require_temporal_negative_controls"] is True
    assert gates["require_artifact_censoring_robustness"] is True

    source_stage = config["study2"]["source_stage"]
    assert source_stage["min_source_valid_subjects"] == 30
    assert source_stage["min_valid_blocks_per_subject"] == 3
    assert source_stage["min_retained_trials_per_subject"] == 25
    assert source_stage["min_residual_degrees_of_freedom"] == 15
    assert source_stage["max_condition_number"] == 100
    assert source_stage["max_opposite_band_vif"] == 5
    assert source_stage["max_collinearity_failure_fraction"] == 0.20

    source_modeling = config["study2"]["source_modeling"]
    assert source_modeling["primary_inverse_method"] == "sLORETA"
    assert source_modeling["rank_excluded_channels"] == ["Fp1", "Fp2"]
    assert source_modeling["noise_covariance_baseline_s"] == [-5.0, -0.01]
    assert source_modeling["immediate_baseline_s"] == [-0.2, -0.01]
    assert source_modeling["early_precue_baseline_s"] == [-7.0, -5.5]
    assert source_modeling["active_plateau_window_s"] == [3.0, 10.5]
    assert source_modeling["regularization"]["snr"] == 3.0
    assert source_modeling["regularization"]["loose_orientation"] == 0.2
    assert source_modeling["regularization"]["depth_weighting"] == 0.8
    assert source_modeling["sensitivity_snr"] == [1.0, 5.0]

    source_qc = config["study2"]["source_model_qc"]
    assert source_qc["min_valid_eeg_channel_location_fraction"] == 0.90
    assert source_qc["max_mean_coregistration_error_mm"] == 5.0
    assert source_qc["max_coregistration_error_mm"] == 10.0

    artifact_controls = config["study2"]["artifact_controls"]
    assert artifact_controls["sensor_template_abs_r_threshold"] == 0.80
    assert artifact_controls["source_artifact_map_abs_r_threshold"] == 0.50
    assert artifact_controls["holm_alpha"] == 0.05

    directional = config["study2"]["directional_consistency"]
    assert directional["min_true_target_spatial_r"] == 0.20
    assert directional["min_cluster_same_sign_fraction"] == 0.60

    source_inference = config["study2"]["source_inference"]
    assert source_inference["calibration_null_fwer_interval"] == [0.025, 0.075]
    assert source_inference["max_r015_ci_half_width"] == 0.10
    assert source_inference["min_r015_cluster_recovery"] == 0.80
    assert source_inference["primary_cluster_forming_p"] == 0.01
    assert source_inference["sensitivity_cluster_forming_p"] == [0.001, 0.05]

    permutations = config["study2"]["permutations"]
    assert permutations["target_retrained_valid_draws"] == 1000
    assert permutations["extended_valid_draws"] == 5000
    assert permutations["full_selection_sensitivity_draws"] == 250
    assert permutations["max_invalid_draw_fraction"] == 0.20
    assert permutations["compute_budget_hours"] == 72
    assert permutations["extension_cluster_p_threshold"] == 0.10
    assert permutations["extension_null_threshold_margin"] == 0.10

    spatial = config["study2"]["spatial_comparison"]
    assert spatial["min_meaningful_eeg_fmri_abs_r"] == 0.15
    assert spatial["brainsmash_surrogates"] == 5000
    assert spatial["spin_rotations"] == 10000

    criterion = config["study2"]["criterion_overlap"]
    assert criterion["min_rated_trials"] == 25
    assert criterion["min_permutation_valid_blocks"] == 3
    assert criterion["permutations"] == 5000
    assert criterion["holm_alpha"] == 0.05
    assert criterion["min_adjusted_pain_class_variance_fraction"] == 0.10
    assert criterion["min_adjusted_pain_class_subjects"] == 30

    reporting = config["study2"]["reporting"]
    assert reporting["bca_bootstrap_resamples"] == 10000

    robustness = config["study2"]["robustness"]
    assert robustness["framewise_displacement_mm"] == 0.5
    assert robustness["dvars_robust_z"] == 3.0
    assert robustness["fp1_fp2_artifact_robust_z"] == 3.0
    assert robustness["cardiac_phase_locking_percentile"] == 95
    assert robustness["scanner_residual_robust_z"] == 3.0
    assert robustness["min_unthresholded_spatial_r"] == 0.50
    assert robustness["min_cluster_dice"] == 0.40
    assert robustness["max_centroid_displacement_mm"] == 15.0
