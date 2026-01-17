"""Behavior analysis CLI command."""

from __future__ import annotations

import argparse
from typing import Any, List

from eeg_pipeline.cli.common import (
    add_common_subject_args,
    add_task_arg,
    add_output_format_args,
    add_path_args,
    resolve_task,
    create_progress_reporter,
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
    compute_group.add_argument("--correlation-method", choices=["spearman", "pearson"], default=None)
    compute_group.add_argument("--robust-correlation", choices=["none", "percentage_bend", "winsorized", "shepherd"], default=None)
    compute_group.add_argument("--bootstrap", type=int, default=None)
    compute_group.add_argument("--n-perm", type=int, default=None)
    compute_group.add_argument("--rng-seed", type=int, default=None)
    compute_group.add_argument("--n-jobs", type=int, default=None)
    compute_group.add_argument("--min-samples", type=int, default=None)
    compute_group.add_argument("--control-temperature", action="store_true", default=None)
    compute_group.add_argument("--no-control-temperature", action="store_false", dest="control_temperature")
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
    compute_group.add_argument("--compute-change-scores", action="store_true", default=None)
    compute_group.add_argument("--no-compute-change-scores", action="store_false", dest="compute_change_scores")
    compute_group.add_argument("--loso-stability", action="store_true", default=None)
    compute_group.add_argument("--no-loso-stability", action="store_false", dest="loso_stability")
    compute_group.add_argument("--compute-bayes-factors", action="store_true", default=None)
    compute_group.add_argument("--no-compute-bayes-factors", action="store_false", dest="compute_bayes_factors")
    compute_group.add_argument("--computations", nargs="+", choices=BEHAVIOR_COMPUTATIONS, default=None)
    compute_group.add_argument(
        "--validate-only",
        action="store_true",
        help="Load and validate data only (no statistics). Still writes metadata and outputs manifest.",
    )
    compute_group.add_argument("--consistency", action="store_true", default=None, dest="consistency_enabled")
    compute_group.add_argument("--no-consistency", action="store_false", dest="consistency_enabled")
    
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
        "--pain-sensitivity-features", nargs="+", choices=feature_choices, default=None,
        help="Feature categories for pain sensitivity analysis"
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
    trial_table_group.add_argument("--trial-table-include-features", action="store_true", default=None)
    trial_table_group.add_argument("--no-trial-table-include-features", action="store_false", dest="trial_table_include_features")
    trial_table_group.add_argument("--trial-table-include-covariates", action="store_true", default=None)
    trial_table_group.add_argument("--no-trial-table-include-covariates", action="store_false", dest="trial_table_include_covariates")
    trial_table_group.add_argument("--trial-table-include-events", action="store_true", default=None)
    trial_table_group.add_argument("--no-trial-table-include-events", action="store_false", dest="trial_table_include_events")
    trial_table_group.add_argument("--trial-table-add-lag-features", action="store_true", default=None)
    trial_table_group.add_argument("--no-trial-table-add-lag-features", action="store_false", dest="trial_table_add_lag_features")
    trial_table_group.add_argument("--trial-table-extra-event-columns", nargs="+", default=None, metavar="COL")
    trial_table_group.add_argument("--feature-summaries", action="store_true", default=None, dest="feature_summaries_enabled")
    trial_table_group.add_argument("--no-feature-summaries", action="store_false", dest="feature_summaries_enabled")

    residual_group = parser.add_argument_group("Pain residual / temperature-model diagnostics")
    residual_group.add_argument("--pain-residual", action="store_true", default=None, dest="pain_residual_enabled")
    residual_group.add_argument("--no-pain-residual", action="store_false", dest="pain_residual_enabled")
    residual_group.add_argument("--pain-residual-method", choices=["spline", "poly"], default=None)
    residual_group.add_argument("--pain-residual-min-samples", type=int, default=None)
    residual_group.add_argument("--pain-residual-poly-degree", type=int, default=None)
    residual_group.add_argument("--pain-residual-model-compare", action="store_true", default=None, dest="pain_residual_model_compare_enabled")
    residual_group.add_argument("--no-pain-residual-model-compare", action="store_false", dest="pain_residual_model_compare_enabled")
    residual_group.add_argument("--pain-residual-model-compare-min-samples", type=int, default=None)
    residual_group.add_argument("--pain-residual-breakpoint-test", action="store_true", default=None, dest="pain_residual_breakpoint_enabled")
    residual_group.add_argument("--no-pain-residual-breakpoint-test", action="store_false", dest="pain_residual_breakpoint_enabled")
    residual_group.add_argument("--pain-residual-breakpoint-min-samples", type=int, default=None)
    residual_group.add_argument("--pain-residual-breakpoint-candidates", type=int, default=None)
    residual_group.add_argument("--pain-residual-breakpoint-quantile-low", type=float, default=None)
    residual_group.add_argument("--pain-residual-breakpoint-quantile-high", type=float, default=None)
    # Optional cross-fit residualization (out-of-run prediction)
    residual_group.add_argument("--pain-residual-crossfit", action="store_true", default=None, dest="pain_residual_crossfit_enabled")
    residual_group.add_argument("--no-pain-residual-crossfit", action="store_false", dest="pain_residual_crossfit_enabled")
    residual_group.add_argument("--pain-residual-crossfit-group-column", type=str, default=None)
    residual_group.add_argument("--pain-residual-crossfit-n-splits", type=int, default=None)
    residual_group.add_argument("--pain-residual-crossfit-method", choices=["spline", "poly"], default=None)
    residual_group.add_argument("--pain-residual-crossfit-spline-n-knots", type=int, default=None)

    regression_group = parser.add_argument_group("Trialwise regression options")
    regression_group.add_argument("--regression-outcome", choices=["rating", "pain_residual", "temperature"], default=None)
    regression_group.add_argument("--regression-include-temperature", action="store_true", default=None)
    regression_group.add_argument("--no-regression-include-temperature", action="store_false", dest="regression_include_temperature")
    regression_group.add_argument("--regression-temperature-control", choices=["linear", "rating_hat", "spline"], default=None)
    regression_group.add_argument("--regression-temperature-spline-knots", type=int, default=None)
    regression_group.add_argument("--regression-temperature-spline-quantile-low", type=float, default=None)
    regression_group.add_argument("--regression-temperature-spline-quantile-high", type=float, default=None)
    regression_group.add_argument("--regression-temperature-spline-min-samples", type=int, default=None)
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
    regression_group.add_argument("--regression-permutations", type=int, default=None)
    regression_group.add_argument("--regression-max-features", type=int, default=None)

    models_group = parser.add_argument_group("Model families options")
    models_group.add_argument("--models-outcomes", nargs="+", choices=["rating", "pain_residual", "temperature", "pain_binary"], default=None)
    models_group.add_argument("--models-families", nargs="+", choices=["ols_hc3", "robust_rlm", "quantile_50", "logit"], default=None)
    models_group.add_argument("--models-include-temperature", action="store_true", default=None)
    models_group.add_argument("--no-models-include-temperature", action="store_false", dest="models_include_temperature")
    models_group.add_argument("--models-temperature-control", choices=["linear", "rating_hat", "spline"], default=None)
    models_group.add_argument("--models-temperature-spline-knots", type=int, default=None)
    models_group.add_argument("--models-temperature-spline-quantile-low", type=float, default=None)
    models_group.add_argument("--models-temperature-spline-quantile-high", type=float, default=None)
    models_group.add_argument("--models-temperature-spline-min-samples", type=int, default=None)
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
    models_group.add_argument("--models-binary-outcome", choices=["pain_binary", "rating_median"], default=None)

    stability_group = parser.add_argument_group("Stability options")
    stability_group.add_argument("--stability-method", choices=["spearman", "pearson"], default=None)
    stability_group.add_argument("--stability-outcome", choices=["auto", "rating", "pain_residual"], default=None)
    stability_group.add_argument("--stability-group-column", choices=["auto", "run", "block"], default=None)
    stability_group.add_argument("--stability-partial-temperature", action="store_true", default=None)
    stability_group.add_argument("--no-stability-partial-temperature", action="store_false", dest="stability_partial_temperature")
    stability_group.add_argument("--stability-min-group-trials", type=int, default=None)
    stability_group.add_argument("--stability-max-features", type=int, default=None)
    stability_group.add_argument("--stability-alpha", type=float, default=None)

    influence_group = parser.add_argument_group("Influence diagnostics options")
    influence_group.add_argument("--influence-outcomes", nargs="+", choices=["rating", "pain_residual", "temperature"], default=None)
    influence_group.add_argument("--influence-max-features", type=int, default=None)
    influence_group.add_argument("--influence-include-temperature", action="store_true", default=None)
    influence_group.add_argument("--no-influence-include-temperature", action="store_false", dest="influence_include_temperature")
    influence_group.add_argument("--influence-temperature-control", choices=["linear", "rating_hat", "spline"], default=None)
    influence_group.add_argument("--influence-temperature-spline-knots", type=int, default=None)
    influence_group.add_argument("--influence-temperature-spline-quantile-low", type=float, default=None)
    influence_group.add_argument("--influence-temperature-spline-quantile-high", type=float, default=None)
    influence_group.add_argument("--influence-temperature-spline-min-samples", type=int, default=None)
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

    pain_sensitivity_group = parser.add_argument_group("Pain sensitivity options")
    pain_sensitivity_group.add_argument("--pain-sensitivity-min-trials", type=int, default=None)

    correlations_group = parser.add_argument_group("Correlations (trial-table) options")
    correlations_group.add_argument("--correlations-targets", nargs="+", choices=["rating", "temperature", "pain_residual"], default=None)
    correlations_group.add_argument("--correlations-primary-unit", choices=["trial", "run_mean"], default=None)
    correlations_group.add_argument("--correlations-prefer-pain-residual", action="store_true", default=None, dest="correlations_prefer_pain_residual")
    correlations_group.add_argument("--no-correlations-prefer-pain-residual", action="store_false", dest="correlations_prefer_pain_residual")
    correlations_group.add_argument("--correlations-use-crossfit-pain-residual", action="store_true", default=None, dest="correlations_use_crossfit_pain_residual")
    correlations_group.add_argument("--no-correlations-use-crossfit-pain-residual", action="store_false", dest="correlations_use_crossfit_pain_residual")
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

    report_group = parser.add_argument_group("Report options")
    report_group.add_argument("--report-top-n", type=int, default=None, help="Top N rows per analysis table in subject_report*.md")

    temporal_group = parser.add_argument_group("Temporal options")
    temporal_group.add_argument("--temporal-time-resolution-ms", type=int, default=None)
    temporal_group.add_argument("--temporal-time-min-ms", type=int, default=None)
    temporal_group.add_argument("--temporal-time-max-ms", type=int, default=None)
    temporal_group.add_argument("--temporal-smooth-window-ms", type=int, default=None)
    temporal_group.add_argument("--temporal-split-by-condition", action="store_true", default=None, dest="temporal_split_by_condition")
    temporal_group.add_argument("--no-temporal-split-by-condition", action="store_false", dest="temporal_split_by_condition")
    temporal_group.add_argument("--temporal-condition-column", type=str, default=None, help="events.tsv column to split/filter by (default: event_columns.pain_binary)")
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
    
    # Time-frequency heatmap options
    tfheatmap_group = parser.add_argument_group("Time-frequency heatmap options")
    tfheatmap_group.add_argument("--tf-heatmap-enabled", action="store_true", default=None, help="Enable TF heatmap computation")
    tfheatmap_group.add_argument("--no-tf-heatmap-enabled", action="store_false", dest="tf_heatmap_enabled", help="Disable TF heatmap computation")
    tfheatmap_group.add_argument("--tf-heatmap-freqs", nargs="+", type=float, default=None, help="Frequencies for TF heatmap")
    tfheatmap_group.add_argument("--tf-heatmap-time-resolution-ms", type=int, default=None, help="Time resolution for TF heatmap (ms)")
    
    
    # Cluster-specific options
    cluster_group = parser.add_argument_group("Cluster permutation options")
    cluster_group.add_argument("--cluster-threshold", type=float, default=None, help="Cluster forming threshold")
    cluster_group.add_argument("--cluster-min-size", type=int, default=None, help="Minimum cluster size")
    cluster_group.add_argument("--cluster-tail", type=int, choices=[-1, 0, 1], default=None, help="Test tail: 0=two-tailed, 1=upper, -1=lower")
    cluster_group.add_argument("--cluster-condition-column", type=str, default=None, help="events.tsv column to split by (default: event_columns.pain_binary)")
    cluster_group.add_argument("--cluster-condition-values", nargs="+", default=None, metavar="VALUE", help="Exactly 2 values to compare (e.g., 0 1 or pain nonpain)")
    
    # Mediation-specific options
    mediation_group = parser.add_argument_group("Mediation analysis options")
    mediation_group.add_argument("--mediation-bootstrap", type=int, default=None, help="Bootstrap iterations for mediation")
    mediation_group.add_argument("--mediation-permutations", type=int, default=None, help="Permutation iterations for mediation (0=disabled)")
    mediation_group.add_argument("--mediation-min-effect-size", type=float, default=None, help="Minimum mediation effect size")
    mediation_group.add_argument("--mediation-max-mediators", type=int, default=None, help="Maximum mediators to test")
    
    # Moderation-specific options
    moderation_group = parser.add_argument_group("Moderation analysis options")
    moderation_group.add_argument("--moderation-max-features", type=int, default=None, help="Maximum features for moderation")
    moderation_group.add_argument("--moderation-min-samples", type=int, default=None, help="Minimum samples for moderation")
    moderation_group.add_argument("--moderation-permutations", type=int, default=None, help="Permutation iterations for moderation (0=disabled)")
    
    # Mixed effects-specific options
    mixed_group = parser.add_argument_group("Mixed effects options")
    mixed_group.add_argument("--mixed-random-effects", choices=["intercept", "intercept_slope"], default=None, help="Random effects specification")
    mixed_group.add_argument("--mixed-max-features", type=int, default=None, help="Maximum features for mixed effects")
    
    # Condition-specific options
    condition_group = parser.add_argument_group("Condition comparison options")
    condition_group.add_argument("--condition-fail-fast", action="store_true", default=None, dest="condition_fail_fast")
    condition_group.add_argument("--no-condition-fail-fast", action="store_false", dest="condition_fail_fast")
    condition_group.add_argument("--condition-effect-threshold", type=float, default=None, help="Minimum effect size (Cohen's d) to report")
    condition_group.add_argument("--condition-min-trials", type=int, default=None, help="Minimum trials per condition")
    condition_group.add_argument("--condition-compare-column", type=str, default=None, help="events.tsv column to use for condition split (default: event_columns.pain_binary)")
    condition_group.add_argument("--condition-compare-values", nargs="+", default=None, metavar="VALUE", help="Values in the column to compare (e.g., 0 1 or pain nonpain)")
    condition_group.add_argument("--condition-compare-windows", nargs="+", default=None, metavar="WINDOW", help="Time windows to compare (e.g., baseline active)")
    condition_group.add_argument("--condition-window-primary-unit", choices=["trial", "run_mean"], default=None)
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

    visualize_group = parser.add_argument_group("Visualize mode options")
    plot_group = visualize_group.add_mutually_exclusive_group()
    plot_group.add_argument("--plots", nargs="+", metavar="PLOT")
    plot_group.add_argument("--all-plots", action="store_true")
    visualize_group.add_argument("--skip-scatter", action="store_true")

    add_path_args(parser)

    return parser


def _configure_behavior_compute_mode(args: argparse.Namespace, config: Any) -> None:
    """
    Populate behavior-analysis configuration from CLI arguments.

    This mutates ``config`` in place but does not have other side effects.
    """
    ba = config.setdefault("behavior_analysis", {})
    stats_cfg = ba.setdefault("statistics", {})
    corr_cfg = ba.setdefault("correlations", {})
    run_adj_cfg = ba.setdefault("run_adjustment", {})

    rng_seed = args.rng_seed if args.rng_seed is not None else config.get("project.random_state")
    if rng_seed is not None:
        config.setdefault("project", {})["random_state"] = rng_seed

    if args.correlation_method:
        stats_cfg["correlation_method"] = args.correlation_method

    if getattr(args, "robust_correlation", None) is not None:
        rc = str(args.robust_correlation).strip().lower()
        ba["robust_correlation"] = None if rc in ("", "none") else rc

    if args.bootstrap is not None:
        ba["bootstrap"] = int(args.bootstrap)

    if args.n_perm is not None:
        stats_cfg["n_permutations"] = int(args.n_perm)
        ba.setdefault("cluster", {})["n_permutations"] = int(args.n_perm)

    if getattr(args, "n_jobs", None) is not None:
        ba["n_jobs"] = int(args.n_jobs)

    if getattr(args, "min_samples", None) is not None:
        ba.setdefault("min_samples", {})["default"] = int(args.min_samples)

    if getattr(args, "control_temperature", None) is not None:
        ba["control_temperature"] = bool(args.control_temperature)

    if getattr(args, "control_trial_order", None) is not None:
        ba["control_trial_order"] = bool(args.control_trial_order)

    # Run adjustment (optional; paradigms may have run_id or none)
    if getattr(args, "run_adjustment", None) is not None:
        run_adj_cfg["enabled"] = bool(args.run_adjustment)
    if getattr(args, "run_adjustment_column", None) is not None:
        run_adj_cfg["column"] = str(args.run_adjustment_column).strip()
    if getattr(args, "run_adjustment_include_in_correlations", None) is not None:
        run_adj_cfg["include_in_correlations"] = bool(args.run_adjustment_include_in_correlations)
    if getattr(args, "run_adjustment_max_dummies", None) is not None:
        run_adj_cfg["max_dummies"] = int(args.run_adjustment_max_dummies)

    if getattr(args, "fdr_alpha", None) is not None:
        stats_cfg["fdr_alpha"] = float(args.fdr_alpha)

    if getattr(args, "compute_change_scores", None) is not None:
        corr_cfg["compute_change_scores"] = bool(args.compute_change_scores)
    if getattr(args, "loso_stability", None) is not None:
        corr_cfg["loso_stability"] = bool(args.loso_stability)
    if getattr(args, "compute_bayes_factors", None) is not None:
        corr_cfg["compute_bayes_factors"] = bool(args.compute_bayes_factors)

    if getattr(args, "consistency_enabled", None) is not None:
        ba.setdefault("consistency", {})["enabled"] = bool(args.consistency_enabled)

    # Trial table
    tt = ba.setdefault("trial_table", {})
    if getattr(args, "trial_table_format", None) is not None:
        tt["format"] = str(args.trial_table_format).strip().lower()
    if getattr(args, "trial_table_include_features", None) is not None:
        tt["include_features"] = bool(args.trial_table_include_features)
    if getattr(args, "trial_table_include_covariates", None) is not None:
        tt["include_covariates"] = bool(args.trial_table_include_covariates)
    if getattr(args, "trial_table_include_events", None) is not None:
        tt["include_events"] = bool(args.trial_table_include_events)
    if getattr(args, "trial_table_add_lag_features", None) is not None:
        tt["add_lag_features"] = bool(args.trial_table_add_lag_features)
    if getattr(args, "trial_table_extra_event_columns", None) is not None:
        cols = [str(c) for c in (args.trial_table_extra_event_columns or [])]
        if len(cols) == 1 and cols[0].strip().lower() == "none":
            tt["extra_event_columns"] = []
        else:
            tt["extra_event_columns"] = cols

    if getattr(args, "feature_summaries_enabled", None) is not None:
        ba.setdefault("feature_summaries", {})["enabled"] = bool(args.feature_summaries_enabled)

    # Pain residual / temperature-model diagnostics
    pr = ba.setdefault("pain_residual", {})
    if getattr(args, "pain_residual_enabled", None) is not None:
        pr["enabled"] = bool(args.pain_residual_enabled)
    if getattr(args, "pain_residual_method", None) is not None:
        pr["method"] = str(args.pain_residual_method).strip().lower()
    if getattr(args, "pain_residual_min_samples", None) is not None:
        pr["min_samples"] = int(args.pain_residual_min_samples)
    if getattr(args, "pain_residual_poly_degree", None) is not None:
        pr["poly_degree"] = int(args.pain_residual_poly_degree)

    mc = pr.setdefault("model_comparison", {})
    if getattr(args, "pain_residual_model_compare_enabled", None) is not None:
        mc["enabled"] = bool(args.pain_residual_model_compare_enabled)
    if getattr(args, "pain_residual_model_compare_min_samples", None) is not None:
        mc["min_samples"] = int(args.pain_residual_model_compare_min_samples)

    bp = pr.setdefault("breakpoint_test", {})
    if getattr(args, "pain_residual_breakpoint_enabled", None) is not None:
        bp["enabled"] = bool(args.pain_residual_breakpoint_enabled)
    if getattr(args, "pain_residual_breakpoint_min_samples", None) is not None:
        bp["min_samples"] = int(args.pain_residual_breakpoint_min_samples)
    if getattr(args, "pain_residual_breakpoint_candidates", None) is not None:
        bp["n_candidates"] = int(args.pain_residual_breakpoint_candidates)
    if getattr(args, "pain_residual_breakpoint_quantile_low", None) is not None:
        bp["quantile_low"] = float(args.pain_residual_breakpoint_quantile_low)
    if getattr(args, "pain_residual_breakpoint_quantile_high", None) is not None:
        bp["quantile_high"] = float(args.pain_residual_breakpoint_quantile_high)

    crossfit = pr.setdefault("crossfit", {})
    if getattr(args, "pain_residual_crossfit_enabled", None) is not None:
        crossfit["enabled"] = bool(args.pain_residual_crossfit_enabled)
    if getattr(args, "pain_residual_crossfit_group_column", None) is not None:
        crossfit["group_column"] = str(args.pain_residual_crossfit_group_column).strip()
    if getattr(args, "pain_residual_crossfit_n_splits", None) is not None:
        crossfit["n_splits"] = int(args.pain_residual_crossfit_n_splits)
    if getattr(args, "pain_residual_crossfit_method", None) is not None:
        crossfit["method"] = str(args.pain_residual_crossfit_method).strip().lower()
    if getattr(args, "pain_residual_crossfit_spline_n_knots", None) is not None:
        crossfit["spline_n_knots"] = int(args.pain_residual_crossfit_spline_n_knots)

    # Regression
    reg = ba.setdefault("regression", {})
    if getattr(args, "regression_outcome", None) is not None:
        reg["outcome"] = str(args.regression_outcome).strip().lower()
    if getattr(args, "regression_include_temperature", None) is not None:
        reg["include_temperature"] = bool(args.regression_include_temperature)
    if getattr(args, "regression_temperature_control", None) is not None:
        reg["temperature_control"] = str(args.regression_temperature_control).strip().lower()
    if (
        getattr(args, "regression_temperature_spline_knots", None) is not None
        or getattr(args, "regression_temperature_spline_quantile_low", None) is not None
        or getattr(args, "regression_temperature_spline_quantile_high", None) is not None
        or getattr(args, "regression_temperature_spline_min_samples", None) is not None
    ):
        ts = reg.setdefault("temperature_spline", {})
        if getattr(args, "regression_temperature_spline_knots", None) is not None:
            ts["n_knots"] = int(args.regression_temperature_spline_knots)
        if getattr(args, "regression_temperature_spline_quantile_low", None) is not None:
            ts["quantile_low"] = float(args.regression_temperature_spline_quantile_low)
        if getattr(args, "regression_temperature_spline_quantile_high", None) is not None:
            ts["quantile_high"] = float(args.regression_temperature_spline_quantile_high)
        if getattr(args, "regression_temperature_spline_min_samples", None) is not None:
            ts["min_samples"] = int(args.regression_temperature_spline_min_samples)
    if getattr(args, "regression_include_trial_order", None) is not None:
        reg["include_trial_order"] = bool(args.regression_include_trial_order)
    if getattr(args, "regression_include_prev_terms", None) is not None:
        reg["include_prev_terms"] = bool(args.regression_include_prev_terms)
    if getattr(args, "regression_include_run_block", None) is not None:
        reg["include_run_block"] = bool(args.regression_include_run_block)
    if getattr(args, "regression_include_interaction", None) is not None:
        reg["include_interaction"] = bool(args.regression_include_interaction)
    if getattr(args, "regression_standardize", None) is not None:
        reg["standardize"] = bool(args.regression_standardize)
    if getattr(args, "regression_min_samples", None) is not None:
        reg["min_samples"] = int(args.regression_min_samples)
    if getattr(args, "regression_permutations", None) is not None:
        reg["n_permutations"] = int(args.regression_permutations)
    if getattr(args, "regression_max_features", None) is not None:
        max_f = int(args.regression_max_features)
        reg["max_features"] = None if max_f <= 0 else max_f

    # Models
    mdl = ba.setdefault("models", {})
    if getattr(args, "models_outcomes", None) is not None:
        mdl["outcomes"] = [str(o).strip().lower() for o in (args.models_outcomes or [])]
    if getattr(args, "models_families", None) is not None:
        mdl["families"] = [str(f).strip().lower() for f in (args.models_families or [])]
    if getattr(args, "models_include_temperature", None) is not None:
        mdl["include_temperature"] = bool(args.models_include_temperature)
    if getattr(args, "models_temperature_control", None) is not None:
        mdl["temperature_control"] = str(args.models_temperature_control).strip().lower()
    if (
        getattr(args, "models_temperature_spline_knots", None) is not None
        or getattr(args, "models_temperature_spline_quantile_low", None) is not None
        or getattr(args, "models_temperature_spline_quantile_high", None) is not None
        or getattr(args, "models_temperature_spline_min_samples", None) is not None
    ):
        ts = mdl.setdefault("temperature_spline", {})
        if getattr(args, "models_temperature_spline_knots", None) is not None:
            ts["n_knots"] = int(args.models_temperature_spline_knots)
        if getattr(args, "models_temperature_spline_quantile_low", None) is not None:
            ts["quantile_low"] = float(args.models_temperature_spline_quantile_low)
        if getattr(args, "models_temperature_spline_quantile_high", None) is not None:
            ts["quantile_high"] = float(args.models_temperature_spline_quantile_high)
        if getattr(args, "models_temperature_spline_min_samples", None) is not None:
            ts["min_samples"] = int(args.models_temperature_spline_min_samples)
    if getattr(args, "models_include_trial_order", None) is not None:
        mdl["include_trial_order"] = bool(args.models_include_trial_order)
    if getattr(args, "models_include_prev_terms", None) is not None:
        mdl["include_prev_terms"] = bool(args.models_include_prev_terms)
    if getattr(args, "models_include_run_block", None) is not None:
        mdl["include_run_block"] = bool(args.models_include_run_block)
    if getattr(args, "models_include_interaction", None) is not None:
        mdl["include_interaction"] = bool(args.models_include_interaction)
    if getattr(args, "models_standardize", None) is not None:
        mdl["standardize"] = bool(args.models_standardize)
    if getattr(args, "models_min_samples", None) is not None:
        mdl["min_samples"] = int(args.models_min_samples)
    if getattr(args, "models_max_features", None) is not None:
        max_f = int(args.models_max_features)
        mdl["max_features"] = None if max_f <= 0 else max_f
    if getattr(args, "models_binary_outcome", None) is not None:
        mdl["binary_outcome"] = str(args.models_binary_outcome).strip().lower()

    # Stability
    stab = ba.setdefault("stability", {})
    if getattr(args, "stability_method", None) is not None:
        stab["method"] = str(args.stability_method).strip().lower()
    if getattr(args, "stability_outcome", None) is not None:
        val = str(args.stability_outcome).strip().lower()
        stab["outcome"] = "" if val == "auto" else val
    if getattr(args, "stability_group_column", None) is not None:
        val = str(args.stability_group_column).strip().lower()
        stab["group_column"] = "" if val == "auto" else val
    if getattr(args, "stability_partial_temperature", None) is not None:
        stab["partial_temperature"] = bool(args.stability_partial_temperature)
    if getattr(args, "stability_min_group_trials", None) is not None:
        stab["min_group_trials"] = int(args.stability_min_group_trials)
    if getattr(args, "stability_max_features", None) is not None:
        max_f = int(args.stability_max_features)
        stab["max_features"] = None if max_f <= 0 else max_f
    if getattr(args, "stability_alpha", None) is not None:
        stab["alpha"] = float(args.stability_alpha)

    # Influence
    infl = ba.setdefault("influence", {})
    if getattr(args, "influence_outcomes", None) is not None:
        infl["outcomes"] = [str(o).strip().lower() for o in (args.influence_outcomes or [])]
    if getattr(args, "influence_max_features", None) is not None:
        infl["max_features"] = int(args.influence_max_features)
    if getattr(args, "influence_include_temperature", None) is not None:
        infl["include_temperature"] = bool(args.influence_include_temperature)
    if getattr(args, "influence_temperature_control", None) is not None:
        infl["temperature_control"] = str(args.influence_temperature_control).strip().lower()
    if (
        getattr(args, "influence_temperature_spline_knots", None) is not None
        or getattr(args, "influence_temperature_spline_quantile_low", None) is not None
        or getattr(args, "influence_temperature_spline_quantile_high", None) is not None
        or getattr(args, "influence_temperature_spline_min_samples", None) is not None
    ):
        ts = infl.setdefault("temperature_spline", {})
        if getattr(args, "influence_temperature_spline_knots", None) is not None:
            ts["n_knots"] = int(args.influence_temperature_spline_knots)
        if getattr(args, "influence_temperature_spline_quantile_low", None) is not None:
            ts["quantile_low"] = float(args.influence_temperature_spline_quantile_low)
        if getattr(args, "influence_temperature_spline_quantile_high", None) is not None:
            ts["quantile_high"] = float(args.influence_temperature_spline_quantile_high)
        if getattr(args, "influence_temperature_spline_min_samples", None) is not None:
            ts["min_samples"] = int(args.influence_temperature_spline_min_samples)
    if getattr(args, "influence_include_trial_order", None) is not None:
        infl["include_trial_order"] = bool(args.influence_include_trial_order)
    if getattr(args, "influence_include_run_block", None) is not None:
        infl["include_run_block"] = bool(args.influence_include_run_block)
    if getattr(args, "influence_include_interaction", None) is not None:
        infl["include_interaction"] = bool(args.influence_include_interaction)
    if getattr(args, "influence_standardize", None) is not None:
        infl["standardize"] = bool(args.influence_standardize)
    if getattr(args, "influence_cooks_threshold", None) is not None:
        infl["cooks_threshold"] = (
            None if float(args.influence_cooks_threshold) <= 0 else float(args.influence_cooks_threshold)
        )
    if getattr(args, "influence_leverage_threshold", None) is not None:
        infl["leverage_threshold"] = (
            None if float(args.influence_leverage_threshold) <= 0 else float(args.influence_leverage_threshold)
        )

    # Pain sensitivity
    if getattr(args, "pain_sensitivity_min_trials", None) is not None:
        ba.setdefault("pain_sensitivity", {})["min_trials"] = int(args.pain_sensitivity_min_trials)

    # Correlations (trial-table)
    if getattr(args, "correlations_targets", None) is not None:
        corr_cfg["targets"] = [
            str(t).strip().lower() for t in (args.correlations_targets or [])
        ]
    if getattr(args, "correlations_primary_unit", None) is not None:
        corr_cfg["primary_unit"] = str(args.correlations_primary_unit).strip().lower()
    if getattr(args, "correlations_prefer_pain_residual", None) is not None:
        corr_cfg["prefer_pain_residual"] = bool(args.correlations_prefer_pain_residual)
    if getattr(args, "correlations_use_crossfit_pain_residual", None) is not None:
        corr_cfg["use_crossfit_pain_residual"] = bool(args.correlations_use_crossfit_pain_residual)
    if getattr(args, "correlations_permutation_primary", None) is not None:
        enabled = bool(args.correlations_permutation_primary)
        corr_cfg["p_primary_mode"] = "perm_if_available" if enabled else "asymptotic"
        corr_cfg.setdefault("permutation", {})["enabled"] = enabled
    if getattr(args, "correlations_target_column", None) is not None:
        corr_cfg["target_column"] = str(args.correlations_target_column).strip()

    # Report
    if getattr(args, "report_top_n", None) is not None:
        ba.setdefault("report", {})["top_n"] = int(args.report_top_n)

    # Condition
    if getattr(args, "condition_fail_fast", None) is not None:
        ba.setdefault("condition", {})["fail_fast"] = bool(args.condition_fail_fast)
    if getattr(args, "condition_effect_threshold", None) is not None:
        ba.setdefault("condition", {})["effect_size_threshold"] = float(args.condition_effect_threshold)
    if getattr(args, "condition_min_trials", None) is not None:
        ba.setdefault("condition", {})["min_trials_per_condition"] = int(args.condition_min_trials)
    if getattr(args, "condition_compare_column", None) is not None:
        ba.setdefault("condition", {})["compare_column"] = str(args.condition_compare_column).strip()
    if getattr(args, "condition_compare_values", None) is not None:
        values = [str(v).strip() for v in (args.condition_compare_values or [])]
        ba.setdefault("condition", {})["compare_values"] = values
    if getattr(args, "condition_compare_windows", None) is not None:
        windows = [str(w).strip() for w in (args.condition_compare_windows or [])]
        ba.setdefault("condition", {})["compare_windows"] = windows
    if getattr(args, "condition_window_primary_unit", None) is not None:
        ba.setdefault("condition", {}).setdefault("window_comparison", {})["primary_unit"] = (
            str(args.condition_window_primary_unit).strip().lower()
        )
    if getattr(args, "condition_permutation_primary", None) is not None:
        enabled = bool(args.condition_permutation_primary)
        cond = ba.setdefault("condition", {})
        cond["p_primary_mode"] = "perm_if_available" if enabled else "asymptotic"
        cond.setdefault("permutation", {})["enabled"] = enabled

    # Temporal
    temporal_cfg = ba.setdefault("temporal", {})
    if getattr(args, "temporal_split_by_condition", None) is not None:
        temporal_cfg["split_by_condition"] = bool(args.temporal_split_by_condition)
    if getattr(args, "temporal_condition_column", None) is not None:
        temporal_cfg["condition_column"] = str(args.temporal_condition_column).strip()
    if getattr(args, "temporal_condition_values", None) is not None:
        temporal_cfg["condition_values"] = [str(v).strip() for v in (args.temporal_condition_values or [])]
    if getattr(args, "temporal_include_roi_averages", None) is not None:
        temporal_cfg["include_roi_averages"] = bool(args.temporal_include_roi_averages)
    if getattr(args, "temporal_include_tf_grid", None) is not None:
        temporal_cfg["include_tf_grid"] = bool(args.temporal_include_tf_grid)
    if getattr(args, "temporal_time_resolution_ms", None) is not None:
        temporal_cfg["time_resolution_ms"] = int(args.temporal_time_resolution_ms)
    if getattr(args, "temporal_smooth_window_ms", None) is not None:
        temporal_cfg["smooth_window_ms"] = int(args.temporal_smooth_window_ms)
    tmin = getattr(args, "temporal_time_min_ms", None)
    tmax = getattr(args, "temporal_time_max_ms", None)
    if tmin is not None or tmax is not None:
        cur = temporal_cfg.get("time_range_ms", [-200, 1000])
        lo = int(tmin) if tmin is not None else int(cur[0])
        hi = int(tmax) if tmax is not None else int(cur[1])
        temporal_cfg["time_range_ms"] = [lo, hi]

    # ITPC-specific temporal options
    itpc_cfg = temporal_cfg.setdefault("itpc", {})
    if getattr(args, "temporal_itpc_baseline_min", None) is not None or getattr(
        args,
        "temporal_itpc_baseline_max",
        None,
    ) is not None:
        baseline_window = list(itpc_cfg.get("baseline_window", [-0.5, -0.01]))
        if getattr(args, "temporal_itpc_baseline_min", None) is not None:
            baseline_window[0] = float(args.temporal_itpc_baseline_min)
        if getattr(args, "temporal_itpc_baseline_max", None) is not None:
            baseline_window[1] = float(args.temporal_itpc_baseline_max)
        itpc_cfg["baseline_window"] = baseline_window
    if getattr(args, "temporal_itpc_baseline_correction", None) is not None:
        itpc_cfg["baseline_correction"] = bool(args.temporal_itpc_baseline_correction)

    # ERDS-specific temporal options
    erds_cfg = temporal_cfg.setdefault("erds", {})
    if getattr(args, "temporal_erds_baseline_min", None) is not None or getattr(
        args,
        "temporal_erds_baseline_max",
        None,
    ) is not None:
        baseline_window = list(erds_cfg.get("baseline_window", [-0.5, -0.1]))
        if getattr(args, "temporal_erds_baseline_min", None) is not None:
            baseline_window[0] = float(args.temporal_erds_baseline_min)
        if getattr(args, "temporal_erds_baseline_max", None) is not None:
            baseline_window[1] = float(args.temporal_erds_baseline_max)
        erds_cfg["baseline_window"] = baseline_window
    if getattr(args, "temporal_erds_method", None) is not None:
        erds_cfg["method"] = str(args.temporal_erds_method).lower()

    # Temporal feature selection
    features_cfg = temporal_cfg.setdefault("features", {})
    if getattr(args, "temporal_feature_power", None) is not None:
        features_cfg["power"] = bool(args.temporal_feature_power)
    if getattr(args, "temporal_feature_itpc", None) is not None:
        features_cfg["itpc"] = bool(args.temporal_feature_itpc)
    if getattr(args, "temporal_feature_erds", None) is not None:
        features_cfg["erds"] = bool(args.temporal_feature_erds)

    # Time-frequency heatmap
    tfheatmap_cfg = ba.setdefault("time_frequency_heatmap", {})
    if getattr(args, "tf_heatmap_enabled", None) is not None:
        tfheatmap_cfg["enabled"] = bool(args.tf_heatmap_enabled)
    if getattr(args, "tf_heatmap_freqs", None) is not None:
        tfheatmap_cfg["freqs"] = list(args.tf_heatmap_freqs)
    if getattr(args, "tf_heatmap_time_resolution_ms", None) is not None:
        tfheatmap_cfg["time_resolution_ms"] = int(args.tf_heatmap_time_resolution_ms)

    # Cluster
    if getattr(args, "cluster_threshold", None) is not None:
        ba.setdefault("cluster", {})["forming_threshold"] = float(args.cluster_threshold)
    if getattr(args, "cluster_min_size", None) is not None:
        ba.setdefault("cluster", {})["min_cluster_size"] = int(args.cluster_min_size)
    if getattr(args, "cluster_tail", None) is not None:
        ba.setdefault("cluster", {})["tail"] = int(args.cluster_tail)
    if getattr(args, "cluster_condition_column", None) is not None:
        ba.setdefault("cluster", {})["condition_column"] = str(args.cluster_condition_column).strip()
    if getattr(args, "cluster_condition_values", None) is not None:
        ba.setdefault("cluster", {})["condition_values"] = [
            str(v).strip() for v in (args.cluster_condition_values or [])
        ]

    # Mediation / mixed effects
    if getattr(args, "mediation_bootstrap", None) is not None:
        ba.setdefault("mediation", {})["n_bootstrap"] = int(args.mediation_bootstrap)
    if getattr(args, "mediation_permutations", None) is not None:
        ba.setdefault("mediation", {})["n_permutations"] = int(args.mediation_permutations)
    if getattr(args, "mediation_min_effect_size", None) is not None:
        ba.setdefault("mediation", {})["min_effect_size"] = float(args.mediation_min_effect_size)
    if getattr(args, "mediation_max_mediators", None) is not None:
        ba.setdefault("mediation", {})["max_mediators"] = int(args.mediation_max_mediators)

    # Moderation
    if getattr(args, "moderation_max_features", None) is not None:
        ba.setdefault("moderation", {})["max_features"] = int(args.moderation_max_features)
    if getattr(args, "moderation_min_samples", None) is not None:
        ba.setdefault("moderation", {})["min_samples"] = int(args.moderation_min_samples)
    if getattr(args, "moderation_permutations", None) is not None:
        ba.setdefault("moderation", {})["n_permutations"] = int(args.moderation_permutations)

    if getattr(args, "mixed_random_effects", None) is not None:
        ba.setdefault("mixed_effects", {})["random_effects"] = str(args.mixed_random_effects).strip().lower()
    if getattr(args, "mixed_max_features", None) is not None:
        ba.setdefault("mixed_effects", {})["max_features"] = int(args.mixed_max_features)

    # Output options
    out = ba.setdefault("output", {})
    if getattr(args, "also_save_csv", None) is not None:
        out["also_save_csv"] = bool(args.also_save_csv)


def _build_computation_features(args: argparse.Namespace) -> dict[str, list[str]] | None:
    """
    Collect feature-category selections per computation type from CLI args.

    Returns ``None`` when no computation-specific feature overrides are given.
    """
    computation_features: dict[str, list[str]] = {}

    if getattr(args, "correlations_features", None):
        computation_features["correlations"] = args.correlations_features
    if getattr(args, "pain_sensitivity_features", None):
        computation_features["pain_sensitivity"] = args.pain_sensitivity_features
    if getattr(args, "condition_features", None):
        computation_features["condition"] = args.condition_features
    if getattr(args, "temporal_features", None):
        computation_features["temporal"] = args.temporal_features
    if getattr(args, "cluster_features", None):
        computation_features["cluster"] = args.cluster_features
    if getattr(args, "mediation_features", None):
        computation_features["mediation"] = args.mediation_features
    if getattr(args, "moderation_features", None):
        computation_features["moderation"] = args.moderation_features

    return computation_features or None


def run_behavior(args: argparse.Namespace, subjects: List[str], config: Any) -> None:
    """Execute the behavior command."""
    from eeg_pipeline.pipelines.behavior import BehaviorPipeline
    from eeg_pipeline.plotting.orchestration.behavior import visualize_behavior_for_subjects
    from eeg_pipeline.analysis.behavior.orchestration import StageRegistry, config_to_stage_names
    
    # Handle discoverability options first
    if getattr(args, "list_stages", False):
        stages = StageRegistry.list_stages()
        print("\n=== Available Behavior Pipeline Stages ===\n")
        for stage in stages:
            print(f"  {stage['name']}")
            print(f"    Description: {stage['description']}")
            print(f"    Group: {stage['group']}")
            if stage['requires']:
                print(f"    Requires: {', '.join(stage['requires'])}")
            if stage['produces']:
                print(f"    Produces: {', '.join(stage['produces'])}")
            print()
        return
    
    if getattr(args, "dry_run", False):
        # Build a mock pipeline config to resolve stages
        from eeg_pipeline.pipelines.behavior import BehaviorPipelineConfig
        pipeline_config = BehaviorPipelineConfig()
        # Apply CLI args to config
        if hasattr(args, "computations") and args.computations:
            for comp in args.computations:
                setattr(pipeline_config, f"run_{comp}", True)
        
        stages = config_to_stage_names(pipeline_config)
        dry_run_result = StageRegistry.dry_run(stages)
        
        print("\n=== Dry Run: Behavior Pipeline ===\n")
        print(f"Requested stages: {', '.join(dry_run_result['requested'])}")
        print(f"Resolved stages ({dry_run_result['n_stages']} total):")
        for i, stage in enumerate(dry_run_result['execution_order'], 1):
            print(f"  {i}. {stage}")
        print(f"\nExpected outputs: {', '.join(dry_run_result['expected_outputs'])}")
        return
    
    categories = getattr(args, "categories", None)
    progress = create_progress_reporter(args)
    task = resolve_task(args.task, config)
    
    if getattr(args, "bids_root", None):
        config.setdefault("paths", {})["bids_root"] = args.bids_root
    if getattr(args, "deriv_root", None):
        config.setdefault("paths", {})["deriv_root"] = args.deriv_root
    
    if args.mode == "compute":
        _configure_behavior_compute_mode(args, config)

        computation_features = _build_computation_features(args)
        
        pipeline = BehaviorPipeline(
            config=config,
            computations=args.computations,
            feature_categories=categories,
            feature_files=getattr(args, "feature_files", None),
            computation_features=computation_features,
        )
        
        pipeline.run_batch(
            subjects=subjects,
            task=task,
            bands=getattr(args, "bands", None),
            validate_only=bool(getattr(args, "validate_only", False)),
            progress=progress,
        )
    elif args.mode == "visualize":
        selected_plots = getattr(args, "plots", None)
        run_all_plots = bool(getattr(args, "all_plots", False))
        skip_scatter = bool(getattr(args, "skip_scatter", False))

        if selected_plots is not None:
            visualize_categories = None
            plots = selected_plots
        elif skip_scatter:
            visualize_categories = None
            plots = [
                "psychometrics",
                "temporal_topomaps",
                "pain_clusters",
                "dose_response",
            ]
        elif run_all_plots:
            visualize_categories = None
            plots = []
        else:
            visualize_categories = categories
            plots = None

        visualize_behavior_for_subjects(
            subjects=subjects,
            task=task,
            config=config,
            visualize_categories=visualize_categories,
            plots=plots,
        )
