"""Parser construction for behavior analysis CLI command."""

from __future__ import annotations

import argparse

from eeg_pipeline.cli.common import (
    add_common_subject_args,
    add_task_arg,
    add_output_format_args,
    add_path_args,
)
from eeg_pipeline.cli.commands.base import (
    BEHAVIOR_COMPUTATIONS,
    BEHAVIOR_VISUALIZE_CATEGORIES,
)
from eeg_pipeline.utils.data.feature_discovery import STANDARD_FEATURE_FILES

FEATURE_FILE_CHOICES = list(STANDARD_FEATURE_FILES.keys())

def setup_behavior(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """Configure the behavior command parser."""
    parser = subparsers.add_parser(
        "behavior",
        help="Behavior analysis: compute correlations or visualize",
        description="Behavior pipeline: compute correlations or visualize",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("mode", choices=["compute", "visualize"], help="Pipeline mode")
    
    # Discoverability options
    parser.add_argument(
        "--list-stages",
        action="store_true",
        help="List all available pipeline stages with descriptions",
    )
    
    add_common_subject_args(parser)
    add_task_arg(parser)
    add_output_format_args(parser)
    parser.add_argument(
        "--categories",
        nargs="+",
        choices=BEHAVIOR_VISUALIZE_CATEGORIES,
        default=None,
        metavar="CATEGORY",
        help="Feature categories to process (e.g., power, connectivity, itpc)",
    )
    
    compute_group = parser.add_argument_group("Compute mode options")
    compute_group.add_argument(
        "--predictor-type",
        choices=["continuous", "binary", "categorical"],
        default=None,
        help=(
            "Nature of the predictor variable. "
            "'continuous' enables curve-fitting analyses (predictor_residual, "
            "predictor_models, psychometrics, spline/outcome_hat control). "
            "'binary' or 'categorical' disables these analyses."
        ),
    )
    compute_group.add_argument("--correlation-method", choices=["spearman", "pearson"], default=None)
    compute_group.add_argument("--robust-correlation", choices=["none", "percentage_bend", "winsorized", "shepherd"], default=None)
    compute_group.add_argument("--bootstrap", type=int, default=None)
    compute_group.add_argument("--n-perm", type=int, default=None)
    compute_group.add_argument("--global-n-bootstrap", type=int, default=None)
    compute_group.add_argument("--perm-scheme", choices=["shuffle", "circular_shift"], default=None)
    compute_group.add_argument("--rng-seed", type=int, default=None)
    compute_group.add_argument("--n-jobs", type=int, default=None)
    compute_group.add_argument("--min-samples", type=int, default=None)
    compute_group.add_argument(
        "--predictor-column",
        type=str,
        default=None,
        help="Canonical predictor column for behavior analyses (e.g., dose, intensity, predictor)",
    )
    compute_group.add_argument(
        "--outcome-column",
        type=str,
        default=None,
        help="Canonical outcome column for behavior analyses (e.g., vas_rating, arousal, confidence)",
    )
    compute_group.add_argument("--predictor-control", action="store_true", default=None, dest="predictor_control")
    compute_group.add_argument("--no-predictor-control", action="store_false", dest="predictor_control")
    compute_group.add_argument("--control-trial-order", action="store_true", default=None)
    compute_group.add_argument("--no-control-trial-order", action="store_false", dest="control_trial_order")
    # Run adjustment (subject-level; optional)
    compute_group.add_argument("--run-adjustment", action="store_true", default=None, dest="run_adjustment")
    compute_group.add_argument("--no-run-adjustment", action="store_false", dest="run_adjustment")
    compute_group.add_argument("--run-adjustment-column", type=str, default=None, help="Run identifier column name (e.g., run_id)")
    compute_group.add_argument(
        "--run-adjustment-include-in-correlations",
        action="store_true",
        default=None,
        dest="run_adjustment_include_in_correlations",
        help="Include run dummies in trial-table correlations partial covariates",
    )
    compute_group.add_argument(
        "--no-run-adjustment-include-in-correlations",
        action="store_false",
        dest="run_adjustment_include_in_correlations",
    )
    compute_group.add_argument("--run-adjustment-max-dummies", type=int, default=None)
    compute_group.add_argument("--fdr-alpha", type=float, default=None)
    compute_group.add_argument(
        "--statistics-alpha",
        type=float,
        default=None,
        help="Global significance alpha fallback used by shared stats helpers.",
    )
    compute_group.add_argument("--stats-predictor-control", choices=["linear", "spline"], default=None, dest="stats_predictor_control")
    compute_group.add_argument("--stats-allow-iid-trials", action="store_true", default=None)
    compute_group.add_argument("--no-stats-allow-iid-trials", action="store_false", dest="stats_allow_iid_trials")
    compute_group.add_argument("--compute-change-scores", action="store_true", default=None)
    compute_group.add_argument("--no-compute-change-scores", action="store_false", dest="compute_change_scores")
    compute_group.add_argument("--loso-stability", action="store_true", default=None)
    compute_group.add_argument("--no-loso-stability", action="store_false", dest="loso_stability")
    compute_group.add_argument("--compute-bayes-factors", action="store_true", default=None)
    compute_group.add_argument("--no-compute-bayes-factors", action="store_false", dest="compute_bayes_factors")
    compute_group.add_argument("--stats-hierarchical-fdr", action="store_true", default=None, dest="stats_hierarchical_fdr")
    compute_group.add_argument("--no-stats-hierarchical-fdr", action="store_false", dest="stats_hierarchical_fdr")
    compute_group.add_argument("--stats-compute-reliability", action="store_true", default=None, dest="stats_compute_reliability")
    compute_group.add_argument("--no-stats-compute-reliability", action="store_false", dest="stats_compute_reliability")
    compute_group.add_argument("--perm-group-column-preference", nargs="+", default=None, metavar="COL", dest="perm_group_column_preference", help="Preferred columns for permutation grouping (e.g. run_id block)")
    compute_group.add_argument("--exclude-non-trialwise-features", action="store_true", default=None, dest="exclude_non_trialwise_features")
    compute_group.add_argument("--no-exclude-non-trialwise-features", action="store_false", dest="exclude_non_trialwise_features")
    compute_group.add_argument(
        "--feature-registry-files-json",
        type=str,
        default=None,
        help="JSON object for behavior_analysis.feature_registry.files",
    )
    compute_group.add_argument(
        "--feature-registry-source-to-feature-type-json",
        type=str,
        default=None,
        dest="feature_registry_source_to_feature_type_json",
        help="JSON object for behavior_analysis.feature_registry.source_to_feature_type",
    )
    compute_group.add_argument(
        "--feature-registry-type-hierarchy-json",
        type=str,
        default=None,
        help="JSON object for behavior_analysis.feature_registry.feature_type_hierarchy",
    )
    compute_group.add_argument(
        "--feature-registry-patterns-json",
        type=str,
        default=None,
        help="JSON object for behavior_analysis.feature_registry.feature_patterns",
    )
    compute_group.add_argument(
        "--feature-registry-classifiers-json",
        type=str,
        default=None,
        help="JSON array for behavior_analysis.feature_registry.feature_classifiers",
    )
    compute_group.add_argument("--predictor-range", nargs=2, type=float, default=None, metavar=("MIN", "MAX"), dest="predictor_range", help="Valid predictor range (e.g. 0.0 1.0)")
    compute_group.add_argument("--max-missing-channels-fraction", type=float, default=None, help="Max fraction of missing channels allowed")
    compute_group.add_argument("--computations", nargs="+", choices=BEHAVIOR_COMPUTATIONS, default=None)
    compute_group.add_argument(
        "--validate-only",
        action="store_true",
        help="Load and validate data only (no statistics). Still writes metadata and outputs manifest.",
    )
    compute_group.add_argument("--consistency", action="store_true", default=None, dest="consistency_enabled")
    compute_group.add_argument("--no-consistency", action="store_false", dest="consistency_enabled")
    compute_group.add_argument("--cluster-correction-enabled", action="store_true", default=None, dest="cluster_correction_enabled")
    compute_group.add_argument("--no-cluster-correction-enabled", action="store_false", dest="cluster_correction_enabled")
    compute_group.add_argument("--cluster-correction-alpha", type=float, default=None)
    compute_group.add_argument("--cluster-correction-min-cluster-size", type=int, default=None)
    compute_group.add_argument("--cluster-correction-tail", type=int, choices=[-1, 0, 1], default=None)
    compute_group.add_argument("--validation-min-epochs", type=int, default=None)
    compute_group.add_argument("--validation-min-channels", type=int, default=None)
    compute_group.add_argument("--validation-max-amplitude-uv", type=float, default=None)
    
    feature_choices = [
        "power", "connectivity", "directedconnectivity", "sourcelocalization",
        "aperiodic", "erp", "bursts", "itpc", "pac",
        "complexity", "quality", "erds", "spectral", "ratios", "asymmetry",
    ]
    compute_group.add_argument(
        "--correlations-features", nargs="+", choices=feature_choices, default=None,
        help="Feature categories for correlations analysis"
    )
    compute_group.add_argument(
        "--predictor-sensitivity-features", nargs="+", choices=feature_choices, default=None,
        help="Feature categories for predictor sensitivity analysis"
    )
    compute_group.add_argument(
        "--condition-features", nargs="+", choices=feature_choices, default=None,
        help="Feature categories for condition comparison"
    )
    compute_group.add_argument(
        "--temporal-features", nargs="+", choices=feature_choices, default=None,
        help="Feature categories for temporal analysis"
    )
    compute_group.add_argument(
        "--cluster-features", nargs="+", choices=feature_choices, default=None,
        help="Feature categories for cluster permutation tests"
    )
    compute_group.add_argument(
        "--mediation-features", nargs="+", choices=feature_choices, default=None,
        help="Feature categories for mediation analysis"
    )
    compute_group.add_argument(
        "--moderation-features", nargs="+", choices=feature_choices, default=None,
        help="Feature categories for moderation analysis"
    )
    compute_group.add_argument(
        "--feature-files",
        nargs="+",
        choices=FEATURE_FILE_CHOICES,
        default=None,
        metavar="FILE",
        help="Specific feature files to load (e.g., power, aperiodic, connectivity). Default: all available",
    )
    compute_group.add_argument(
        "--bands",
        nargs="+",
        choices=["delta", "theta", "alpha", "beta", "gamma"],
        default=None,
        help="Frequency bands to use for analysis (default: all)",
    )

    trial_table_group = parser.add_argument_group("Trial table options")
    trial_table_group.add_argument("--trial-table-format", choices=["parquet", "tsv"], default=None)
    trial_table_group.add_argument("--trial-table-add-lag-features", action="store_true", default=None)
    trial_table_group.add_argument("--no-trial-table-add-lag-features", action="store_false", dest="trial_table_add_lag_features")
    trial_table_group.add_argument(
        "--trial-table-disallow-positional-alignment",
        action="store_true",
        default=None,
        dest="trial_table_disallow_positional_alignment",
        help="Fail when trial-table alignment would rely on positional fallback.",
    )
    trial_table_group.add_argument(
        "--no-trial-table-disallow-positional-alignment",
        action="store_false",
        dest="trial_table_disallow_positional_alignment",
    )
    trial_table_group.add_argument("--feature-summaries", action="store_true", default=None, dest="feature_summaries_enabled")
    trial_table_group.add_argument("--no-feature-summaries", action="store_false", dest="feature_summaries_enabled")
    trial_table_group.add_argument(
        "--trial-order-max-missing-fraction",
        type=float,
        default=None,
        help="Max fraction of missing trial-order values before disabling trial-order control (default: 0.1)",
    )

    feature_qc_group = parser.add_argument_group("Feature QC options")
    feature_qc_group.add_argument("--feature-qc-enabled", action="store_true", default=None, dest="feature_qc_enabled")
    feature_qc_group.add_argument("--no-feature-qc-enabled", action="store_false", dest="feature_qc_enabled")
    feature_qc_group.add_argument("--feature-qc-max-missing-pct", type=float, default=None)
    feature_qc_group.add_argument("--feature-qc-min-variance", type=float, default=None)
    feature_qc_group.add_argument(
        "--feature-qc-check-within-run-variance",
        action="store_true",
        default=None,
        dest="feature_qc_check_within_run_variance",
    )
    feature_qc_group.add_argument(
        "--no-feature-qc-check-within-run-variance",
        action="store_false",
        dest="feature_qc_check_within_run_variance",
    )

    residual_group = parser.add_argument_group("Predictor residual / predictor-model diagnostics")
    residual_group.add_argument("--predictor-residual", action="store_true", default=None, dest="predictor_residual_enabled")
    residual_group.add_argument("--no-predictor-residual", action="store_false", dest="predictor_residual_enabled")
    residual_group.add_argument("--predictor-residual-method", choices=["spline", "poly"], default=None, dest="predictor_residual_method")
    residual_group.add_argument("--predictor-residual-min-samples", type=int, default=None, dest="predictor_residual_min_samples")
    residual_group.add_argument(
        "--predictor-residual-spline-df-candidates",
        nargs="+",
        type=int,
        default=None,
        dest="predictor_residual_spline_df_candidates",
        help="Candidate spline degrees of freedom for predictor→outcome residual model (e.g., 3 4 5)",
    )
    residual_group.add_argument("--predictor-residual-poly-degree", type=int, default=None, dest="predictor_residual_poly_degree")
    residual_group.add_argument("--predictor-residual-model-compare", action="store_true", default=None, dest="predictor_residual_model_compare_enabled")
    residual_group.add_argument("--no-predictor-residual-model-compare", action="store_false", dest="predictor_residual_model_compare_enabled")
    residual_group.add_argument("--predictor-residual-model-compare-min-samples", type=int, default=None, dest="predictor_residual_model_compare_min_samples")
    residual_group.add_argument(
        "--predictor-residual-model-compare-poly-degrees",
        nargs="+",
        type=int,
        default=None,
        dest="predictor_residual_model_compare_poly_degrees",
        help="Polynomial degrees to compare in predictor-residual model comparison (e.g., 2 3)",
    )
    residual_group.add_argument("--predictor-residual-breakpoint-test", action="store_true", default=None, dest="predictor_residual_breakpoint_enabled")
    residual_group.add_argument("--no-predictor-residual-breakpoint-test", action="store_false", dest="predictor_residual_breakpoint_enabled")
    residual_group.add_argument("--predictor-residual-breakpoint-min-samples", type=int, default=None, dest="predictor_residual_breakpoint_min_samples")
    residual_group.add_argument("--predictor-residual-breakpoint-candidates", type=int, default=None, dest="predictor_residual_breakpoint_candidates")
    residual_group.add_argument("--predictor-residual-breakpoint-quantile-low", type=float, default=None, dest="predictor_residual_breakpoint_quantile_low")
    residual_group.add_argument("--predictor-residual-breakpoint-quantile-high", type=float, default=None, dest="predictor_residual_breakpoint_quantile_high")
    # Optional cross-fit residualization (out-of-run prediction)
    residual_group.add_argument("--predictor-residual-crossfit", action="store_true", default=None, dest="predictor_residual_crossfit_enabled")
    residual_group.add_argument("--no-predictor-residual-crossfit", action="store_false", dest="predictor_residual_crossfit_enabled")
    residual_group.add_argument("--predictor-residual-crossfit-group-column", type=str, default=None, dest="predictor_residual_crossfit_group_column")
    residual_group.add_argument("--predictor-residual-crossfit-n-splits", type=int, default=None, dest="predictor_residual_crossfit_n_splits")
    residual_group.add_argument("--predictor-residual-crossfit-method", choices=["spline", "poly"], default=None, dest="predictor_residual_crossfit_method")
    residual_group.add_argument("--predictor-residual-crossfit-spline-n-knots", type=int, default=None, dest="predictor_residual_crossfit_spline_n_knots")

    regression_group = parser.add_argument_group("Trialwise regression options")
    regression_group.add_argument("--regression-outcome", choices=["outcome", "predictor_residual", "predictor"], default=None)
    regression_group.add_argument("--regression-include-predictor", action="store_true", default=None, dest="regression_include_predictor")
    regression_group.add_argument("--no-regression-include-predictor", action="store_false", dest="regression_include_predictor")
    regression_group.add_argument("--regression-predictor-control", choices=["linear", "outcome_hat", "spline"], default=None, dest="regression_predictor_control")
    regression_group.add_argument("--regression-predictor-spline-knots", type=int, default=None, dest="regression_predictor_spline_knots")
    regression_group.add_argument("--regression-predictor-spline-quantile-low", type=float, default=None, dest="regression_predictor_spline_quantile_low")
    regression_group.add_argument("--regression-predictor-spline-quantile-high", type=float, default=None, dest="regression_predictor_spline_quantile_high")
    regression_group.add_argument("--regression-predictor-spline-min-samples", type=int, default=None, dest="regression_predictor_spline_min_samples")
    regression_group.add_argument("--regression-include-trial-order", action="store_true", default=None)
    regression_group.add_argument("--no-regression-include-trial-order", action="store_false", dest="regression_include_trial_order")
    regression_group.add_argument("--regression-include-prev-terms", action="store_true", default=None)
    regression_group.add_argument("--no-regression-include-prev-terms", action="store_false", dest="regression_include_prev_terms")
    regression_group.add_argument("--regression-include-run-block", action="store_true", default=None)
    regression_group.add_argument("--no-regression-include-run-block", action="store_false", dest="regression_include_run_block")
    regression_group.add_argument("--regression-include-interaction", action="store_true", default=None)
    regression_group.add_argument("--no-regression-include-interaction", action="store_false", dest="regression_include_interaction")
    regression_group.add_argument("--regression-standardize", action="store_true", default=None)
    regression_group.add_argument("--no-regression-standardize", action="store_false", dest="regression_standardize")
    regression_group.add_argument("--regression-min-samples", type=int, default=None)
    regression_group.add_argument("--regression-primary-unit", choices=["trial", "run_mean"], default=None)
    regression_group.add_argument("--regression-permutations", type=int, default=None)
    regression_group.add_argument("--regression-max-features", type=int, default=None)

    models_group = parser.add_argument_group("Model families options")
    models_group.add_argument("--models-outcomes", nargs="+", choices=["outcome", "predictor_residual", "predictor", "binary_outcome"], default=None)
    models_group.add_argument("--models-families", nargs="+", choices=["ols_hc3", "robust_rlm", "quantile_50", "logit"], default=None)
    models_group.add_argument("--models-include-predictor", action="store_true", default=None, dest="models_include_predictor")
    models_group.add_argument("--no-models-include-predictor", action="store_false", dest="models_include_predictor")
    models_group.add_argument("--models-predictor-control", choices=["linear", "outcome_hat", "spline"], default=None, dest="models_predictor_control")
    models_group.add_argument("--models-predictor-spline-knots", type=int, default=None, dest="models_predictor_spline_knots")
    models_group.add_argument("--models-predictor-spline-quantile-low", type=float, default=None, dest="models_predictor_spline_quantile_low")
    models_group.add_argument("--models-predictor-spline-quantile-high", type=float, default=None, dest="models_predictor_spline_quantile_high")
    models_group.add_argument("--models-predictor-spline-min-samples", type=int, default=None, dest="models_predictor_spline_min_samples")
    models_group.add_argument("--models-include-trial-order", action="store_true", default=None)
    models_group.add_argument("--no-models-include-trial-order", action="store_false", dest="models_include_trial_order")
    models_group.add_argument("--models-include-prev-terms", action="store_true", default=None)
    models_group.add_argument("--no-models-include-prev-terms", action="store_false", dest="models_include_prev_terms")
    models_group.add_argument("--models-include-run-block", action="store_true", default=None)
    models_group.add_argument("--no-models-include-run-block", action="store_false", dest="models_include_run_block")
    models_group.add_argument("--models-include-interaction", action="store_true", default=None)
    models_group.add_argument("--no-models-include-interaction", action="store_false", dest="models_include_interaction")
    models_group.add_argument("--models-standardize", action="store_true", default=None)
    models_group.add_argument("--no-models-standardize", action="store_false", dest="models_standardize")
    models_group.add_argument("--models-min-samples", type=int, default=None)
    models_group.add_argument("--models-max-features", type=int, default=None)
    models_group.add_argument("--models-binary-outcome", choices=["binary_outcome", "outcome_median"], default=None)
    models_group.add_argument("--models-primary-unit", choices=["trial", "run_mean"], default=None)
    models_group.add_argument(
        "--models-force-trial-iid-asymptotic",
        action="store_true",
        default=None,
        dest="models_force_trial_iid_asymptotic",
        help="Explicitly allow trial-level feature models with i.i.d asymptotic inference",
    )
    models_group.add_argument(
        "--no-models-force-trial-iid-asymptotic",
        action="store_false",
        dest="models_force_trial_iid_asymptotic",
    )

    stability_group = parser.add_argument_group("Stability options")
    stability_group.add_argument("--stability-method", choices=["spearman", "pearson"], default=None)
    stability_group.add_argument("--stability-outcome", choices=["auto", "outcome", "predictor_residual"], default=None)
    stability_group.add_argument("--stability-group-column", choices=["auto", "run", "block"], default=None)
    stability_group.add_argument("--stability-partial-predictor", action="store_true", default=None, dest="stability_partial_predictor")
    stability_group.add_argument("--no-stability-partial-predictor", action="store_false", dest="stability_partial_predictor")
    stability_group.add_argument("--stability-min-group-trials", type=int, default=None)
    stability_group.add_argument("--stability-max-features", type=int, default=None)
    stability_group.add_argument("--stability-alpha", type=float, default=None)

    influence_group = parser.add_argument_group("Influence diagnostics options")
    influence_group.add_argument("--influence-outcomes", nargs="+", choices=["outcome", "predictor_residual", "predictor"], default=None)
    influence_group.add_argument("--influence-max-features", type=int, default=None)
    influence_group.add_argument("--influence-include-predictor", action="store_true", default=None, dest="influence_include_predictor")
    influence_group.add_argument("--no-influence-include-predictor", action="store_false", dest="influence_include_predictor")
    influence_group.add_argument("--influence-predictor-control", choices=["linear", "outcome_hat", "spline"], default=None, dest="influence_predictor_control")
    influence_group.add_argument("--influence-predictor-spline-knots", type=int, default=None, dest="influence_predictor_spline_knots")
    influence_group.add_argument("--influence-predictor-spline-quantile-low", type=float, default=None, dest="influence_predictor_spline_quantile_low")
    influence_group.add_argument("--influence-predictor-spline-quantile-high", type=float, default=None, dest="influence_predictor_spline_quantile_high")
    influence_group.add_argument("--influence-predictor-spline-min-samples", type=int, default=None, dest="influence_predictor_spline_min_samples")
    influence_group.add_argument("--influence-include-trial-order", action="store_true", default=None)
    influence_group.add_argument("--no-influence-include-trial-order", action="store_false", dest="influence_include_trial_order")
    influence_group.add_argument("--influence-include-run-block", action="store_true", default=None)
    influence_group.add_argument("--no-influence-include-run-block", action="store_false", dest="influence_include_run_block")
    influence_group.add_argument("--influence-include-interaction", action="store_true", default=None)
    influence_group.add_argument("--no-influence-include-interaction", action="store_false", dest="influence_include_interaction")
    influence_group.add_argument("--influence-standardize", action="store_true", default=None)
    influence_group.add_argument("--no-influence-standardize", action="store_false", dest="influence_standardize")
    influence_group.add_argument("--influence-cooks-threshold", type=float, default=None)
    influence_group.add_argument("--influence-leverage-threshold", type=float, default=None)

    predictor_sensitivity_group = parser.add_argument_group("Predictor sensitivity options")
    predictor_sensitivity_group.add_argument("--predictor-sensitivity-min-trials", type=int, default=None)
    predictor_sensitivity_group.add_argument("--predictor-sensitivity-primary-unit", choices=["trial", "run_mean"], default=None)
    predictor_sensitivity_group.add_argument("--predictor-sensitivity-permutations", type=int, default=None)
    predictor_sensitivity_group.add_argument(
        "--predictor-sensitivity-permutation-primary",
        action="store_true",
        default=None,
        dest="predictor_sensitivity_permutation_primary",
        help="Use permutation-based p_primary when available",
    )
    predictor_sensitivity_group.add_argument(
        "--no-predictor-sensitivity-permutation-primary",
        action="store_false",
        dest="predictor_sensitivity_permutation_primary",
    )

    correlations_group = parser.add_argument_group("Correlations (trial-table) options")
    correlations_group.add_argument(
        "--correlations-types",
        nargs="+",
        choices=["raw", "partial_cov", "partial_predictor", "partial_cov_predictor", "run_mean"],
        default=None,
        help="Correlation types to compute (default from config)",
    )
    correlations_group.add_argument("--correlations-primary-unit", choices=["trial", "run_mean"], default=None)
    correlations_group.add_argument(
        "--correlations-min-runs",
        type=int,
        default=None,
        help="Minimum runs required for run-mean correlation estimates",
    )
    correlations_group.add_argument(
        "--correlations-prefer-predictor-residual",
        action="store_true",
        default=None,
        dest="correlations_prefer_predictor_residual",
        help="Prefer predictor_residual (or predictor_residual_cv) when selecting correlation targets",
    )
    correlations_group.add_argument(
        "--no-correlations-prefer-predictor-residual",
        action="store_false",
        dest="correlations_prefer_predictor_residual",
    )
    correlations_group.add_argument(
        "--correlations-permutations",
        type=int,
        default=None,
        dest="correlations_permutations",
        help="Override permutation iterations for correlations only (unset=use global --n-perm)",
    )
    correlations_group.add_argument("--correlations-use-crossfit-predictor-residual", action="store_true", default=None, dest="correlations_use_crossfit_predictor_residual")
    correlations_group.add_argument("--no-correlations-use-crossfit-predictor-residual", action="store_false", dest="correlations_use_crossfit_predictor_residual")
    correlations_group.add_argument(
        "--correlations-permutation-primary",
        action="store_true",
        default=None,
        dest="correlations_permutation_primary",
        help="Use within-run/block permutation p-values for p_primary when available",
    )
    correlations_group.add_argument("--no-correlations-permutation-primary", action="store_false", dest="correlations_permutation_primary")
    correlations_group.add_argument(
        "--correlations-target-column",
        type=str,
        default=None,
        dest="correlations_target_column",
        help="Custom target column name from events (e.g., 'vas_rating', 'pain_intensity')",
    )
    correlations_group.add_argument(
        "--correlations-power-segment",
        type=str,
        default=None,
        dest="correlations_power_segment",
        help=(
            "Optional segment name used for ROI power correlations (NamingSchema segment, "
            "e.g. 'active', 'stimulation'). Empty uses all available segments."
        ),
    )
    correlations_group.add_argument(
        "--group-level-block-permutation",
        action="store_true",
        default=None,
        dest="group_level_block_permutation",
        help="Use block-restricted permutations when a block/run column is available (default: True)",
    )
    correlations_group.add_argument(
        "--no-group-level-block-permutation",
        action="store_false",
        dest="group_level_block_permutation",
    )
    correlations_group.add_argument(
        "--group-level-target",
        type=str,
        default=None,
        help="Target column for multilevel group correlations",
    )
    correlations_group.add_argument(
        "--group-level-control-predictor",
        action="store_true",
        default=None,
        dest="group_level_control_predictor",
        help="Control predictor in multilevel group correlations",
    )
    correlations_group.add_argument(
        "--no-group-level-control-predictor",
        action="store_false",
        dest="group_level_control_predictor",
    )
    correlations_group.add_argument(
        "--group-level-control-trial-order",
        action="store_true",
        default=None,
        dest="group_level_control_trial_order",
        help="Control trial order in multilevel group correlations",
    )
    correlations_group.add_argument(
        "--no-group-level-control-trial-order",
        action="store_false",
        dest="group_level_control_trial_order",
    )
    correlations_group.add_argument(
        "--group-level-control-run-effects",
        action="store_true",
        default=None,
        dest="group_level_control_run_effects",
        help="Control run effects in multilevel group correlations",
    )
    correlations_group.add_argument(
        "--no-group-level-control-run-effects",
        action="store_false",
        dest="group_level_control_run_effects",
    )
    correlations_group.add_argument(
        "--group-level-max-run-dummies",
        type=int,
        default=None,
        dest="group_level_max_run_dummies",
        help="Maximum run dummy columns for multilevel group-level controls",
    )
    correlations_group.add_argument(
        "--group-level-allow-parametric-fallback",
        action="store_true",
        default=None,
        dest="group_level_allow_parametric_fallback",
        help="Allow parametric fallback when permutation testing is unavailable in multilevel correlations",
    )
    correlations_group.add_argument(
        "--no-group-level-allow-parametric-fallback",
        action="store_false",
        dest="group_level_allow_parametric_fallback",
    )

    report_group = parser.add_argument_group("Report options")
    report_group.add_argument("--report-top-n", type=int, default=None, help="Top N rows per analysis table in subject_report*.md")

    temporal_group = parser.add_argument_group("Temporal options")
    temporal_group.add_argument(
        "--temporal-target-column",
        type=str,
        default=None,
        help="events.tsv column to correlate against for temporal analyses (default: outcome from event_columns.outcome)",
    )
    temporal_group.add_argument("--temporal-correction-method", choices=["fdr", "cluster"], default=None)
    temporal_group.add_argument("--temporal-time-resolution-ms", type=int, default=None)
    temporal_group.add_argument("--temporal-freqs-hz", nargs="+", type=float, default=None, help="Frequency bins for temporal TF analyses (Hz)")
    temporal_group.add_argument("--temporal-time-min-ms", type=int, default=None)
    temporal_group.add_argument("--temporal-time-max-ms", type=int, default=None)
    temporal_group.add_argument("--temporal-smooth-window-ms", type=int, default=None)
    temporal_group.add_argument("--temporal-split-by-condition", action="store_true", default=None, dest="temporal_split_by_condition")
    temporal_group.add_argument("--no-temporal-split-by-condition", action="store_false", dest="temporal_split_by_condition")
    temporal_group.add_argument("--temporal-condition-column", type=str, default=None, help="events.tsv column to split/filter by (default: event_columns.binary_outcome)")
    temporal_group.add_argument("--temporal-condition-values", nargs="+", default=None, metavar="VALUE", help="Subset of values to compute (empty = all unique values)")
    temporal_group.add_argument("--temporal-include-roi-averages", action="store_true", default=None, dest="temporal_include_roi_averages", help="Include ROI-averaged rows in output")
    temporal_group.add_argument("--no-temporal-include-roi-averages", action="store_false", dest="temporal_include_roi_averages", help="Exclude ROI-averaged rows from output")
    temporal_group.add_argument("--temporal-include-tf-grid", action="store_true", default=None, dest="temporal_include_tf_grid", help="Include individual frequency (TF grid) rows in output")
    temporal_group.add_argument("--no-temporal-include-tf-grid", action="store_false", dest="temporal_include_tf_grid", help="Exclude TF grid rows from output")
    # Temporal feature selection
    temporal_group.add_argument("--temporal-feature-power", action="store_true", default=None, help="Enable power temporal correlations")
    temporal_group.add_argument("--no-temporal-feature-power", action="store_false", dest="temporal_feature_power", help="Disable power temporal correlations")
    temporal_group.add_argument("--temporal-feature-itpc", action="store_true", default=None, help="Enable ITPC temporal correlations")
    temporal_group.add_argument("--no-temporal-feature-itpc", action="store_false", dest="temporal_feature_itpc", help="Disable ITPC temporal correlations")
    temporal_group.add_argument("--temporal-feature-erds", action="store_true", default=None, help="Enable ERDS temporal correlations")
    temporal_group.add_argument("--no-temporal-feature-erds", action="store_false", dest="temporal_feature_erds", help="Disable ERDS temporal correlations")
    # ITPC-specific temporal options
    temporal_group.add_argument("--temporal-itpc-baseline-min", type=float, default=None, help="ITPC baseline window start (seconds)")
    temporal_group.add_argument("--temporal-itpc-baseline-max", type=float, default=None, help="ITPC baseline window end (seconds)")
    temporal_group.add_argument("--temporal-itpc-baseline-correction", action="store_true", default=None, help="Enable ITPC baseline correction")
    temporal_group.add_argument("--no-temporal-itpc-baseline-correction", action="store_false", dest="temporal_itpc_baseline_correction", help="Disable ITPC baseline correction")
    # ERDS-specific temporal options
    temporal_group.add_argument("--temporal-erds-baseline-min", type=float, default=None, help="ERDS baseline window start (seconds)")
    temporal_group.add_argument("--temporal-erds-baseline-max", type=float, default=None, help="ERDS baseline window end (seconds)")
    temporal_group.add_argument("--temporal-erds-method", choices=["percent", "zscore"], default=None, help="ERDS computation method")
    
    # Cluster-specific options
    cluster_group = parser.add_argument_group("Cluster permutation options")
    cluster_group.add_argument("--cluster-threshold", type=float, default=None, help="Cluster forming threshold")
    cluster_group.add_argument("--cluster-min-size", type=int, default=None, help="Minimum cluster size")
    cluster_group.add_argument("--cluster-tail", type=int, choices=[-1, 0, 1], default=None, help="Test tail: 0=two-tailed, 1=upper, -1=lower")
    cluster_group.add_argument("--cluster-condition-column", type=str, default=None, help="events.tsv column to split by (default: event_columns.binary_outcome)")
    cluster_group.add_argument("--cluster-condition-values", nargs="+", default=None, metavar="VALUE", help="Exactly 2 values to compare (e.g., 0 1 or condition_a condition_b)")
    
    # Mediation-specific options
    mediation_group = parser.add_argument_group("Mediation analysis options")
    mediation_group.add_argument("--mediation-bootstrap", type=int, default=None, help="Bootstrap iterations for mediation")
    mediation_group.add_argument("--mediation-permutations", type=int, default=None, help="Permutation iterations for mediation (0=disabled)")
    mediation_group.add_argument(
        "--mediation-permutation-primary",
        action="store_true",
        default=None,
        dest="mediation_permutation_primary",
        help="Use permutation-based p_primary when available",
    )
    mediation_group.add_argument(
        "--no-mediation-permutation-primary",
        action="store_false",
        dest="mediation_permutation_primary",
    )
    mediation_group.add_argument("--mediation-min-effect-size", type=float, default=None, help="Minimum mediation effect size")
    mediation_group.add_argument("--mediation-max-mediators", type=int, default=None, help="Maximum mediators to test")
    
    # Moderation-specific options
    moderation_group = parser.add_argument_group("Moderation analysis options")
    moderation_group.add_argument("--moderation-max-features", type=int, default=None, help="Maximum features for moderation")
    moderation_group.add_argument("--moderation-min-samples", type=int, default=None, help="Minimum samples for moderation")
    moderation_group.add_argument("--moderation-permutations", type=int, default=None, help="Permutation iterations for moderation (0=disabled)")
    moderation_group.add_argument(
        "--moderation-permutation-primary",
        action="store_true",
        default=None,
        dest="moderation_permutation_primary",
        help="Use permutation-based p_primary when available",
    )
    moderation_group.add_argument(
        "--no-moderation-permutation-primary",
        action="store_false",
        dest="moderation_permutation_primary",
    )
    
    # Mixed effects-specific options
    mixed_group = parser.add_argument_group("Mixed effects options")
    mixed_group.add_argument("--mixed-random-effects", choices=["intercept", "intercept_slope"], default=None, help="Random effects specification")
    mixed_group.add_argument("--mixed-include-predictor", action="store_true", default=None, dest="mixed_include_predictor")
    mixed_group.add_argument("--no-mixed-include-predictor", action="store_false", dest="mixed_include_predictor")
    mixed_group.add_argument("--mixed-max-features", type=int, default=None, help="Maximum features for mixed effects")
    
    # Condition-specific options
    condition_group = parser.add_argument_group("Condition comparison options")
    condition_group.add_argument("--condition-fail-fast", action="store_true", default=None, dest="condition_fail_fast")
    condition_group.add_argument("--no-condition-fail-fast", action="store_false", dest="condition_fail_fast")
    condition_group.add_argument("--condition-effect-threshold", type=float, default=None, help="Minimum effect size (Cohen's d) to report")
    condition_group.add_argument("--condition-min-trials", type=int, default=None, help="Minimum trials per condition")
    condition_group.add_argument("--condition-compare-column", type=str, default=None, help="events.tsv column to use for condition split (default: event_columns.binary_outcome)")
    condition_group.add_argument("--condition-compare-values", nargs="+", default=None, metavar="VALUE", help="Values in the column to compare (e.g., 0 1 or condition_a condition_b)")
    condition_group.add_argument("--condition-compare-labels", nargs="+", default=None, metavar="LABEL", help="Optional labels aligned to --condition-compare-values")
    condition_group.add_argument("--condition-overwrite", action="store_true", default=None, dest="condition_overwrite", help="Overwrite existing condition effects files (default)")
    condition_group.add_argument("--no-condition-overwrite", action="store_false", dest="condition_overwrite", help="Include compare_column in filename to avoid overwriting")
    condition_group.add_argument("--condition-primary-unit", choices=["trial", "run_mean"], default=None)
    condition_group.add_argument("--condition-compare-windows", nargs="+", default=None, metavar="WINDOW", help="Time windows to compare (e.g., baseline active)")
    condition_group.add_argument("--condition-window-primary-unit", choices=["trial", "run_mean"], default=None)
    condition_group.add_argument("--condition-window-min-samples", type=int, default=None)
    condition_group.add_argument("--condition-permutation-primary", action="store_true", default=None, dest="condition_permutation_primary")
    condition_group.add_argument("--no-condition-permutation-primary", action="store_false", dest="condition_permutation_primary")

    output_group = parser.add_argument_group("Output options")
    output_group.add_argument(
        "--also-save-csv",
        action="store_true",
        default=None,
        dest="also_save_csv",
        help="Also save output tables as CSV files (in addition to TSV)",
    )
    output_group.add_argument(
        "--no-also-save-csv",
        action="store_false",
        dest="also_save_csv",
    )
    output_group.add_argument(
        "--overwrite",
        action="store_true",
        default=None,
        dest="overwrite",
        help="Overwrite existing output folders (default: True)",
    )
    output_group.add_argument(
        "--no-overwrite",
        action="store_false",
        dest="overwrite",
        help="Append timestamp to output folders instead of overwriting",
    )

    visualize_group = parser.add_argument_group("Visualize mode options")
    plot_group = visualize_group.add_mutually_exclusive_group()
    plot_group.add_argument("--plots", nargs="+", metavar="PLOT")
    plot_group.add_argument("--all-plots", action="store_true")
    visualize_group.add_argument("--skip-scatter", action="store_true")

    add_path_args(parser)

    return parser
