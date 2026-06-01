from __future__ import annotations


def test_study2_default_confirmatory_cell_matches_readme() -> None:
    from studies.pain_study.study2.config import load_study2_config

    config = load_study2_config()
    cell = config["study2"]["confirmatory"]["study1_cell"]

    assert cell["target"] == "NPS"
    assert cell["model"] == "elasticnet"
    assert cell["feature_family"] == "power"
    assert cell["frequency_preset"] == "alpha_beta_gamma"
    assert cell["feature_resolution"] == "individual_channel"
    assert cell["residualization_level"] == 2
    assert cell["contribution_bands"] == ["alpha", "beta", "gamma"]

    contributions = config["study2"]["contributions"]
    assert contributions["combined_column"] == "eta_combined"
    assert contributions["combined_standardized_column"] == "eta_combined_z"
    assert contributions["alpha_column"] == "eta_alpha"
    assert contributions["beta_column"] == "eta_beta"
    assert contributions["gamma_column"] == "eta_gamma"
    assert contributions["alpha_standardized_column"] == "eta_alpha_z"
    assert contributions["beta_standardized_column"] == "eta_beta_z"
    assert contributions["gamma_standardized_column"] == "eta_gamma_z"

    gates = config["study2"]["confirmatory"]["study1_gates"]
    assert gates["max_holm_p"] == 0.05
    assert gates["min_delta_r2"] == 0.02
    assert gates["min_delta_r2_lower_ci"] == 0.005
    assert gates["min_level2_delta_r2"] == 0.005
    assert gates["min_target_split_half_reliability"] == 0.4
    assert gates["min_target_reliability_n_trials"] == 30
    assert gates["require_positive_within_subject_delta_r2"] is True
    assert gates["require_temporal_negative_controls"] is True
    assert gates["require_artifact_censoring_robustness"] is True

    source_stage = config["study2"]["source_stage"]
    assert source_stage["min_source_valid_subjects"] == 30
    assert source_stage["min_feasibility_subjects"] == 20
    assert source_stage["min_valid_blocks_per_subject"] == 3
    assert source_stage["min_retained_trials_per_subject"] == 25
    assert source_stage["min_residual_degrees_of_freedom"] == 15
    assert source_stage["max_condition_number"] == 100
    assert source_stage["max_opposite_band_vif"] == 5
    assert source_stage["max_collinearity_failure_fraction"] == 0.20
    assert source_stage["continuous_columns"] == [
        "onset",
        "trial_index_within_block",
        "hrf_weighted_framewise_displacement",
        "hrf_weighted_std_dvars",
        "hrf_weighted_fp1_fp2_high_frequency_power",
        "residual_ecg_coupling",
    ]
    assert source_stage["categorical_columns"] == [
        "block",
        "stimulus_temp",
        "selected_surface",
    ]
    assert source_stage["fixed_categorical_levels"] == {
        "block": [1, 2, 3, 4, 5, 6],
        "stimulus_temp": [44.3, 45.3, 46.3, 47.3, 48.3, 49.3],
        "selected_surface": [1, 2, 3, 4, 5],
    }

    source_modeling = config["study2"]["source_modeling"]
    assert source_modeling["primary_power_estimand"] == "total_power"
    assert source_modeling["erp_subtracted_power"] == "sensitivity_only"
    assert source_modeling["primary_inverse_method"] == "sLORETA"
    assert source_modeling["rank_excluded_channels"] == ["Fp1", "Fp2"]
    assert source_modeling["noise_covariance_baseline_s"] == [-5.0, -0.01]
    assert source_modeling["immediate_baseline_s"] == [-0.2, -0.01]
    assert "early_precue_baseline_s" not in source_modeling
    assert source_modeling["active_plateau_window_s"] == [3.0, 10.5]
    assert source_modeling["regularization"]["snr"] == 3.0
    assert source_modeling["regularization"]["loose_orientation"] == 0.2
    assert source_modeling["regularization"]["depth_weighting"] == 0.8
    assert source_modeling["sensitivity_snr"] == [1.0, 5.0]
    assert source_modeling["report_point_spread_fwhm"] is True

    source_qc = config["study2"]["source_model_qc"]
    assert source_qc["min_valid_eeg_channel_location_fraction"] == 0.90
    assert source_qc["max_mean_coregistration_error_mm"] == 5.0
    assert source_qc["max_coregistration_error_mm"] == 10.0

    artifact_controls = config["study2"]["artifact_controls"]
    assert artifact_controls["sensor_template_abs_r_threshold"] == 0.80
    assert artifact_controls["source_artifact_map_abs_r_threshold"] == 0.50
    assert artifact_controls["holm_alpha"] == 0.05
    assert artifact_controls["gamma_requires_artifact_survival"] is True

    directional = config["study2"]["directional_consistency"]
    assert directional["min_true_target_spatial_r"] == 0.20
    assert directional["min_cluster_same_sign_fraction"] == 0.60

    source_inference = config["study2"]["source_inference"]
    assert "calibration_null_fwer_interval" not in source_inference
    assert source_inference["primary_cluster_forming_p"] == 0.01
    assert source_inference["sensitivity_cluster_forming_p"] == [0.001]

    permutations = config["study2"]["permutations"]
    assert permutations["target_retrained_valid_draws"] == 1000
    assert permutations["max_invalid_draw_fraction"] == 0.20
    assert permutations["compute_budget_hours"] == 72
    assert "extended_valid_draws" not in permutations
    assert "full_selection_sensitivity_draws" not in permutations

    spatial = config["study2"]["spatial_comparison"]
    assert spatial["min_meaningful_eeg_fmri_abs_r"] == 0.15
    assert spatial["brainsmash_surrogates"] == 5000
    assert spatial["fmri_smoothing_kernel_source"] == "median_point_spread_fwhm"
    assert "spin_rotations" not in spatial

    behavioral = config["study2"]["behavioral_convergence"]
    assert behavioral["min_rated_trials"] == 25
    assert behavioral["min_permutation_valid_blocks"] == 3
    assert behavioral["permutations"] == 5000

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
