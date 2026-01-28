"""
Statistics Utilities
====================

Statistical utilities for EEG analysis.

Modules:
    - base: Constants, config helpers, and core utilities
    - fdr: FDR correction
    - cluster: Cluster permutation tests (1D and 2D)
    - correlation: Correlation and partial correlation
    - bootstrap: Bootstrap and permutation tests
    - effect_size: Effect size metrics
    - validation: Data validation
    - formatting: Statistical formatting
    - transforms: Data transformation and aperiodic fitting
    - visualization: Visualization statistics
    - roi: ROI statistics
    - permutation: Permutation tests
    - partial: Partial correlation
    - mediation: Mediation analysis
    - moderation: Moderation analysis
    - reliability: Reliability and ICC
    - paired_comparisons: Paired comparison tests
    - feature_models: Feature-level regression models
    - trialwise_regression: Trial-wise regression
    - temporal: Temporal correlation analysis
    - stability: Stability analysis
    - consistency: Consistency checks
    - influence: Influence diagnostics
    - splines: Spline fitting
    - temperature_models: Temperature-rating models
    - pain_residual: Pain residual computation
    - mixed_effects: Mixed effects models
"""

from .base import (
    get_statistics_constants,
    get_fdr_alpha,
    get_ci_level,
    get_z_critical_value,
    get_n_permutations,
    get_n_bootstrap,
    get_config_value,
    ensure_config,
    _safe_float,
)

from .fdr import (
    fdr_bh,
    fdr_bh_values,
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
    compute_cluster_masses_2d,
    compute_permutation_max_masses,
    compute_cluster_pvalues_2d as compute_cluster_pvalues,
    compute_cluster_correction_2d,
)

from .correlation import (
    compute_correlation,
    fisher_z,
    inverse_fisher_z,
    fisher_ci,
    joint_valid_mask,
    # New robust/Bayesian statistics
    compute_bayes_factor_correlation,
    compute_robust_correlation,
    compute_loso_correlation_stability,
    # Correlation statistics helpers
    compute_correlation_stats,
)

from .bootstrap import (
    bootstrap_corr_ci,
    compute_bootstrap_ci,
)

from .effect_size import (
    cohens_d,
    hedges_g,
    compute_cohens_d_with_bootstrap_ci,
)

from eeg_pipeline.utils.analysis.arrays import extract_finite_mask

from .formatting import (
    format_p_value,
    format_cluster_ann,
    format_correlation_stats_text,
)

from .transforms import (
    prepare_data_for_plotting,
    fit_linear_regression,
    compute_binned_statistics,
    compute_residuals,
)

from .visualization import compute_kde_scale

from .roi import (
    compute_roi_percentage_change,
    compute_roi_pvalue,
)

from .permutation import (
    permute_within_groups,
    perm_pval_simple,
    compute_permutation_pvalues,
    compute_permutation_pvalues_with_cov_temp,
)

from .partial import (
    partial_corr_xy_given_Z,
    partial_residuals_xy_given_Z,
    compute_partial_residuals,
    compute_partial_corr,
    compute_partial_correlation_with_covariates,
    compute_partial_correlations_with_cov_temp,
)



from .paired_comparisons import compute_paired_cohens_d

from .mediation import (
    MediationResult,
    compute_mediation_paths,
    bootstrap_indirect_effect,
    run_full_mediation_analysis,
    analyze_mediation_for_features,
)

from .moderation import (
    ModerationResult,
    compute_moderation_effect,
    run_moderation_analysis,
)

from .reliability import (
    compute_icc,
)

from .validation import validate_baseline_window_pre_stimulus


__all__ = [
    # Base
    "get_statistics_constants",
    "get_fdr_alpha",
    "get_ci_level",
    "get_z_critical_value",
    "get_n_permutations",
    "get_n_bootstrap",
    "get_config_value",
    "ensure_config",
    # FDR
    "fdr_bh",
    "fdr_bh_values",
    # Cluster
    "build_distance_adjacency",
    "get_eeg_adjacency",
    "build_full_mask_from_eeg",
    "extract_cluster_indices",
    "compute_cluster_masses",
    "cluster_mask_from_clusters",
    "resolve_cluster_n_jobs",
    "cluster_test_two_sample",
    "cluster_test_epochs",
    # Cluster 2D
    "compute_cluster_masses_2d",
    "compute_permutation_max_masses",
    "compute_cluster_pvalues",
    "compute_cluster_correction_2d",
    # Correlation
    "compute_correlation",
    "fisher_z",
    "inverse_fisher_z",
    "fisher_ci",
    "joint_valid_mask",
    "compute_partial_corr",
    # Robust/Bayesian statistics
    "compute_bayes_factor_correlation",
    "compute_robust_correlation",
    "compute_loso_correlation_stability",
    # Correlation statistics helpers
    "compute_correlation_stats",
    # Bootstrap
    "bootstrap_corr_ci",
    # Effect Size
    "cohens_d",
    "hedges_g",
    "compute_cohens_d_with_bootstrap_ci",
    # Validation
    "validate_baseline_window_pre_stimulus",
    "extract_finite_mask",
    # Formatting
    "format_p_value",
    "format_cluster_ann",
    "format_correlation_stats_text",
    "_safe_float",
    # EEG Stats
    "compute_residuals",
    "compute_bootstrap_ci",
    # Transform
    "prepare_data_for_plotting",
    # Regression
    "fit_linear_regression",
    "compute_binned_statistics",
    # Visualization
    "compute_kde_scale",
    # ROI
    "compute_roi_percentage_change",
    "compute_roi_pvalue",
    # Permutation
    "permute_within_groups",
    "perm_pval_simple",
    "compute_permutation_pvalues",
    "compute_permutation_pvalues_with_cov_temp",
    # Partial
    "partial_corr_xy_given_Z",
    "partial_residuals_xy_given_Z",
    "compute_partial_residuals",
    "compute_partial_correlation_with_covariates",
    "compute_partial_correlations_with_cov_temp",
    # Reliability and validity
    "compute_icc",
    # Paired Comparisons
    "compute_paired_cohens_d",
    # Mediation
    "MediationResult",
    "compute_mediation_paths",
    "bootstrap_indirect_effect",
    "run_full_mediation_analysis",
    "analyze_mediation_for_features",
    # Moderation
    "ModerationResult",
    "compute_moderation_effect",
    "run_moderation_analysis",
]
