"""
Statistics Utilities
====================

Statistical utilities for EEG analysis.

Modules:
    - base: Constants and config helpers
    - fdr: FDR correction
    - cluster: Cluster permutation tests (1D)
    - cluster_2d: Cluster permutation tests (2D time-frequency)
    - correlation: Correlation and partial correlation
    - bootstrap: Bootstrap and permutation tests
    - effect_size: Effect size metrics
    - validation: Data validation
    - formatting: Statistical formatting
    - aggregation: Group statistics
    - eeg_stats: EEG-specific statistics
    - transform: Data transformation
    - aperiodic: Aperiodic fitting
    - regression: Linear regression
    - visualization: Visualization statistics
    - roi: ROI statistics
    - band_stats: Band statistics
    - coupling: Inter-band coupling
    - permutation: Permutation tests
    - partial: Partial correlation
    - group: Group correlation statistics
"""

from .base import (
    CorrelationStats,
    get_statistics_constants,
    get_fdr_alpha,
    get_ci_level,
    get_n_permutations,
    get_n_bootstrap,
    get_config_value,
    ensure_config,
)

from .fdr import (
    fdr_bh,
    fdr_bh_reject,
    fdr_bh_mask,
    fdr_bh_values,
    bh_adjust,
    select_p_values_for_fdr,
    filter_significant_predictors,
    apply_fdr_correction_and_save,
    get_pvalue_series,
    extract_pvalue_from_dataframe,
    should_apply_fisher_transform,
    get_cluster_correction_config,
    compute_fdr_rejections_for_heatmap,
    build_correlation_matrices_for_prefix,
)

from .cluster import (
    build_distance_adjacency,
    get_eeg_adjacency,
    build_full_mask_from_eeg,
    extract_cluster_indices,
    compute_cluster_masses,
    cluster_mask_from_clusters,
    resolve_cluster_n_jobs,
    cluster_test_two_sample,
    cluster_test_epochs,
)

from .cluster_2d import (
    compute_cluster_masses_2d,
    compute_permutation_max_masses,
    compute_cluster_pvalues,
    compute_cluster_correction_2d,
    compute_cluster_masses_1d,
    compute_topomap_permutation_masses,
    compute_cluster_pvalues_1d,
)

from .correlation import (
    get_correlation_method,
    compute_correlation,
    fisher_z,
    inverse_fisher_z,
    fisher_ci,
    fisher_aggregate,
    weighted_fisher_aggregate,
    joint_valid_mask,
    partial_corr_xy_given_Z,
    compute_partial_corr,
    normalize_series,
    compute_correlation_pvalue,
    # New robust/Bayesian statistics
    compute_bayes_factor_correlation,
    compute_robust_correlation,
    compute_loso_correlation_stability,
    compute_correlation_reliability,
)

from .bootstrap import (
    permute_within_groups,
    perm_pval_simple,
    bootstrap_corr_ci,
    bootstrap_ci_bca,
)

from .effect_size import (
    cohens_d,
    hedges_g,
    glass_delta,
    fisher_z_test,
    cohens_q,
    correlation_difference_effect,
    r_to_d,
    d_to_r,
    compute_effect_sizes,
)

from .validation import (
    validate_pain_binary_values,
    validate_temperature_values,
    validate_baseline_window_pre_stimulus,
    check_pyriemann,
    extract_finite_mask,
    extract_pain_masks,
    extract_duration_data,
)

from .formatting import (
    format_p_value,
    format_correlation_text,
    format_cluster_ann,
    format_fdr_ann,
    format_correlation_stats_text,
    _safe_float,
    _compute_fisher_z_ci,
    _compute_bf10_correlation,
    _interpret_bayes_factor,
)

from .aggregation import (
    compute_group_channel_statistics,
    compute_channel_confidence_interval,
    pool_data_by_strategy,
    compute_band_summary_statistics,
    compute_band_summaries,
    compute_fisher_transformed_mean,
    compute_group_band_statistics,
    compute_error_bars_from_ci_dicts,
    compute_error_bars_from_arrays,
    count_trials_by_condition,
    compute_duration_p_value,
)

from .eeg_stats import (
    align_epochs_to_pivot_chs,
    compute_correlation_for_metric_state,
    compute_residuals,
    compute_robust_correlation,
    get_fdr_alpha_from_config,
    compute_bootstrap_ci,
)

from .transform import (
    center_series,
    zscore_series,
    apply_pooling_strategy,
    prepare_data_for_plotting,
    prepare_data_without_validation,
    prepare_group_data,
)

from .aperiodic import (
    fit_aperiodic,
    fit_aperiodic_to_all_epochs,
)

from .regression import (
    compute_linear_residuals,
    fit_linear_regression,
    compute_binned_statistics,
)

from .visualization import (
    compute_kde_scale,
    compute_correlation_vmax,
)

from .roi import (
    extract_roi_statistics,
    extract_overall_statistics,
    update_stats_from_dataframe,
    compute_roi_percentage_change,
    compute_roi_pvalue,
    compute_statistics_for_mask,
    compute_coverage_statistics,
)

from .band_stats import (
    compute_band_spatial_correlation,
    compute_band_pair_correlation,
    compute_subject_band_correlation_matrix,
    compute_group_band_correlation_matrix,
    fisher_z_transform_mean,
    compute_band_statistics_array,
    compute_inter_band_correlation_statistics,
    compute_correlation_ci_fisher,
    compute_band_correlations,
    compute_connectivity_correlations,
    compute_correlation_stats,
    compute_partial_residuals_stats,
)

from .coupling import (
    compute_consensus_labels,
    compute_inter_band_coupling_matrix,
    compute_group_channel_power_statistics,
)

from .permutation import (
    perm_pval_partial_freedman_lane,
    compute_perm_and_partial_perm,
    compute_permutation_pvalue_partial,
    compute_permutation_pvalues,
    compute_temp_permutation_pvalues,
    compute_permutation_pvalues_for_roi_pair,
)

from .partial import (
    partial_residuals_xy_given_Z,
    compute_partial_residuals,
    compute_partial_correlation_with_covariates,
    compute_partial_correlations,
    compute_partial_correlation_for_roi_pair,
    prepare_aligned_data,
)


from .meta_analysis import (
    MetaAnalysisResult,
    fisher_z as meta_fisher_z,
    inverse_fisher_z as meta_inverse_fisher_z,
    correlation_se,
    compute_heterogeneity,
    fixed_effects_meta,
    random_effects_meta,
    leave_one_out_meta,
    permutation_null_distribution,
    bootstrap_correlation_ci,
    ensure_bootstrap_ci,
    bayes_factor_correlation,
    equivalence_test_correlation,
)

from .reliability import (
    compute_icc,
    compute_split_half_reliability,
    compute_feature_reliability,
    hierarchical_fdr,
    compute_hierarchical_fdr_summary,
    cross_validated_prediction,
    compute_calibration_curve,
    compute_required_n_for_correlation,
    assess_statistical_power,
    is_underpowered,
)

from .validation import (
    AssumptionCheckResult,
    ValidationReport,
    check_normality_shapiro,
    check_normality_dagostino,
    compute_qq_data,
    check_variance_levene,
    check_variance_bartlett,
    validate_permutation_distribution,
    check_randomization_balance,
    compute_fwer_bonferroni,
    compute_fwer_holm,
    compute_fwer_sidak,
    validate_fwer_control,
    validate_behavioral_contrast,
)

from .visualization import (
    compute_permutation_distribution_data,
    compute_cluster_mass_histogram_data,
    compute_pp_plot_data,
    compute_qq_plot_data,
    compute_effect_size_distribution_data,
    compute_bootstrap_distribution_data,
    compute_raincloud_data,
    compute_spaghetti_plot_data,
    compute_correction_comparison_data,
    create_provenance_block,
    format_provenance_text,
    save_stats_for_plot,
)

# Aliases
cluster_test_two_sample_arrays = cluster_test_two_sample

__all__ = [
    # Base
    "CorrelationStats",
    "get_statistics_constants",
    "get_fdr_alpha",
    "get_ci_level",
    "get_n_permutations",
    "get_n_bootstrap",
    "get_config_value",
    "ensure_config",
    # FDR
    "fdr_bh",
    "fdr_bh_reject",
    "fdr_bh_mask",
    "fdr_bh_values",
    "bh_adjust",
    "select_p_values_for_fdr",
    "filter_significant_predictors",
    "apply_fdr_correction_and_save",
    "get_pvalue_series",
    "extract_pvalue_from_dataframe",
    "should_apply_fisher_transform",
    "get_cluster_correction_config",
    "compute_fdr_rejections_for_heatmap",
    "build_correlation_matrices_for_prefix",
    # Cluster
    "build_distance_adjacency",
    "get_eeg_adjacency",
    "build_full_mask_from_eeg",
    "extract_cluster_indices",
    "compute_cluster_masses",
    "cluster_mask_from_clusters",
    "resolve_cluster_n_jobs",
    "cluster_test_two_sample",
    "cluster_test_two_sample_arrays",
    "cluster_test_epochs",
    # Cluster 2D
    "compute_cluster_masses_2d",
    "compute_permutation_max_masses",
    "compute_cluster_pvalues",
    "compute_cluster_correction_2d",
    "compute_cluster_masses_1d",
    "compute_topomap_permutation_masses",
    "compute_cluster_pvalues_1d",
    # Correlation
    "get_correlation_method",
    "compute_correlation",
    "fisher_z",
    "inverse_fisher_z",
    "fisher_ci",
    "fisher_aggregate",
    "weighted_fisher_aggregate",
    "joint_valid_mask",
    "partial_corr_xy_given_Z",
    "compute_partial_corr",
    "normalize_series",
    "compute_correlation_pvalue",
    # Robust/Bayesian statistics
    "compute_bayes_factor_correlation",
    "compute_robust_correlation",
    "compute_loso_correlation_stability",
    "compute_correlation_reliability",
    # Bootstrap
    "permute_within_groups",
    "perm_pval_simple",
    "bootstrap_corr_ci",
    "bootstrap_ci_bca",
    # Effect Size
    "cohens_d",
    "hedges_g",
    "glass_delta",
    "fisher_z_test",
    "cohens_q",
    "correlation_difference_effect",
    "r_to_d",
    "d_to_r",
    "compute_effect_sizes",
    # Validation
    "validate_pain_binary_values",
    "validate_temperature_values",
    "validate_baseline_window_pre_stimulus",
    "check_pyriemann",
    "extract_finite_mask",
    "extract_pain_masks",
    "extract_duration_data",
    # Formatting
    "format_p_value",
    "format_correlation_text",
    "format_cluster_ann",
    "format_fdr_ann",
    "format_correlation_stats_text",
    "_safe_float",
    # Aggregation
    "compute_group_channel_statistics",
    "compute_channel_confidence_interval",
    "pool_data_by_strategy",
    "compute_band_summary_statistics",
    "compute_band_summaries",
    "compute_fisher_transformed_mean",
    "compute_group_band_statistics",
    "compute_error_bars_from_ci_dicts",
    "compute_error_bars_from_arrays",
    "count_trials_by_condition",
    "compute_duration_p_value",
    # EEG Stats
    "align_epochs_to_pivot_chs",
    "compute_correlation_for_metric_state",
    "prepare_aligned_data",
    "compute_residuals",
    "compute_robust_correlation",
    "get_fdr_alpha_from_config",
    "compute_bootstrap_ci",
    # Transform
    "center_series",
    "zscore_series",
    "apply_pooling_strategy",
    "prepare_data_for_plotting",
    "prepare_data_without_validation",
    "prepare_group_data",
    # Aperiodic
    "fit_aperiodic",
    "fit_aperiodic_to_all_epochs",
    # Regression
    "compute_linear_residuals",
    "fit_linear_regression",
    "compute_binned_statistics",
    # Visualization
    "compute_kde_scale",
    "compute_correlation_vmax",
    # ROI
    "extract_roi_statistics",
    "extract_overall_statistics",
    "update_stats_from_dataframe",
    "compute_roi_percentage_change",
    "compute_roi_pvalue",
    "compute_statistics_for_mask",
    "compute_coverage_statistics",
    # Band Stats
    "compute_band_spatial_correlation",
    "compute_band_pair_correlation",
    "compute_subject_band_correlation_matrix",
    "compute_group_band_correlation_matrix",
    "fisher_z_transform_mean",
    "compute_band_statistics_array",
    "compute_inter_band_correlation_statistics",
    "compute_correlation_ci_fisher",
    "compute_band_correlations",
    "compute_connectivity_correlations",
    "compute_correlation_stats",
    "compute_partial_residuals_stats",
    # Coupling
    "compute_consensus_labels",
    "compute_inter_band_coupling_matrix",
    "compute_group_channel_power_statistics",
    # Permutation
    "perm_pval_partial_freedman_lane",
    "compute_perm_and_partial_perm",
    "compute_permutation_pvalue_partial",
    "compute_permutation_pvalues",
    "compute_temp_permutation_pvalues",
    "compute_permutation_pvalues_for_roi_pair",
    # Partial
    "partial_residuals_xy_given_Z",
    "compute_partial_residuals",
    "compute_partial_correlation_with_covariates",
    "compute_partial_correlations",
    "compute_partial_correlation_for_roi_pair",
    # Meta-analysis
    "MetaAnalysisResult",
    "meta_fisher_z",
    "meta_inverse_fisher_z",
    "correlation_se",
    "compute_heterogeneity",
    "fixed_effects_meta",
    "random_effects_meta",
    "leave_one_out_meta",
    "permutation_null_distribution",
    "bootstrap_correlation_ci",
    "ensure_bootstrap_ci",
    "bayes_factor_correlation",
    "equivalence_test_correlation",
    # Validation
    "AssumptionCheckResult",
    "ValidationReport",
    "check_normality_shapiro",
    "check_normality_dagostino",
    "compute_qq_data",
    "check_variance_levene",
    "check_variance_bartlett",
    "validate_permutation_distribution",
    "check_randomization_balance",
    "compute_fwer_bonferroni",
    "compute_fwer_holm",
    "compute_fwer_sidak",
    "validate_fwer_control",
    "validate_behavioral_contrast",
    # Visualization diagnostics
    "compute_permutation_distribution_data",
    "compute_cluster_mass_histogram_data",
    "compute_pp_plot_data",
    "compute_qq_plot_data",
    "compute_effect_size_distribution_data",
    "compute_bootstrap_distribution_data",
    "compute_raincloud_data",
    "compute_spaghetti_plot_data",
    "compute_correction_comparison_data",
    "create_provenance_block",
    "format_provenance_text",
    "save_stats_for_plot",
    # Reliability and validity
    "compute_icc",
    "compute_split_half_reliability",
    "compute_feature_reliability",
    "hierarchical_fdr",
    "compute_hierarchical_fdr_summary",
    "cross_validated_prediction",
    "compute_calibration_curve",
    "compute_required_n_for_correlation",
    "assess_statistical_power",
    "is_underpowered",
]
