"""Configuration helpers for behavior analysis CLI command."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Callable, Sequence


CastFn = Callable[[Any], Any]


@dataclass(frozen=True)
class ConfigOverrideRule:
    """Declarative mapping from CLI arg attribute to nested config path."""

    arg_attr: str
    config_path: str
    cast: CastFn


def _to_bool(value: Any) -> bool:
    return bool(value)


def _to_int(value: Any) -> int:
    return int(value)


def _to_float(value: Any) -> float:
    return float(value)


def _to_stripped(value: Any) -> str:
    return str(value).strip()


def _to_lower_stripped(value: Any) -> str:
    return str(value).strip().lower()


def _to_list(value: Any) -> list[Any]:
    return list(value)


def _to_stripped_list(value: Any) -> list[str]:
    return [str(v).strip() for v in (value or [])]


def _to_lower_stripped_list(value: Any) -> list[str]:
    return [str(v).strip().lower() for v in (value or [])]


def _to_optional_int_max(value: Any) -> int | None:
    parsed = int(value)
    return None if parsed <= 0 else parsed


def _to_optional_float_threshold(value: Any) -> float | None:
    parsed = float(value)
    return None if parsed <= 0 else parsed


def _to_auto_blank_or_lower(value: Any) -> str:
    parsed = _to_lower_stripped(value)
    return "" if parsed == "auto" else parsed


def _to_optional_robust_method(value: Any) -> str | None:
    parsed = _to_lower_stripped(value)
    return None if parsed in ("", "none") else parsed


def _set_nested_config_value(config: Any, config_path: str, value: Any) -> None:
    parts = config_path.split(".")
    current = config
    for part in parts[:-1]:
        if part not in current or not isinstance(current[part], dict):
            current[part] = {}
        current = current[part]
    current[parts[-1]] = value


def _apply_override_rules(
    args: argparse.Namespace,
    config: Any,
    rules: Sequence[ConfigOverrideRule],
) -> None:
    for rule in rules:
        value = getattr(args, rule.arg_attr, None)
        if value is None:
            continue
        _set_nested_config_value(config, rule.config_path, rule.cast(value))


def _has_any_arg(args: argparse.Namespace, arg_names: Sequence[str]) -> bool:
    return any(getattr(args, name, None) is not None for name in arg_names)


_GENERAL_OVERRIDE_RULES = (
    ConfigOverrideRule("correlation_method", "behavior_analysis.statistics.correlation_method", str),
    ConfigOverrideRule("bootstrap", "behavior_analysis.bootstrap", _to_int),
    ConfigOverrideRule("global_n_bootstrap", "behavior_analysis.statistics.default_n_bootstrap", _to_int),
    ConfigOverrideRule("perm_scheme", "behavior_analysis.permutation.scheme", _to_lower_stripped),
    ConfigOverrideRule("n_jobs", "behavior_analysis.n_jobs", _to_int),
    ConfigOverrideRule("min_samples", "behavior_analysis.min_samples.default", _to_int),
    ConfigOverrideRule("predictor_column", "behavior_analysis.predictor_column", _to_stripped),
    ConfigOverrideRule("outcome_column", "behavior_analysis.outcome_column", _to_stripped),
    ConfigOverrideRule("predictor_type", "behavior_analysis.predictor_type", _to_lower_stripped),
    ConfigOverrideRule("predictor_control", "behavior_analysis.predictor_control_enabled", _to_bool),
    ConfigOverrideRule("control_trial_order", "behavior_analysis.control_trial_order", _to_bool),
    ConfigOverrideRule("run_adjustment", "behavior_analysis.run_adjustment.enabled", _to_bool),
    ConfigOverrideRule("run_adjustment_column", "behavior_analysis.run_adjustment.column", _to_stripped),
    ConfigOverrideRule(
        "run_adjustment_include_in_correlations",
        "behavior_analysis.run_adjustment.include_in_correlations",
        _to_bool,
    ),
    ConfigOverrideRule("run_adjustment_max_dummies", "behavior_analysis.run_adjustment.max_dummies", _to_int),
    ConfigOverrideRule("fdr_alpha", "behavior_analysis.statistics.fdr_alpha", _to_float),
    ConfigOverrideRule("stats_predictor_control", "behavior_analysis.statistics.predictor_control", _to_lower_stripped),
    ConfigOverrideRule("stats_allow_iid_trials", "behavior_analysis.statistics.allow_iid_trials", _to_bool),
    ConfigOverrideRule("stats_hierarchical_fdr", "behavior_analysis.statistics.hierarchical_fdr", _to_bool),
    ConfigOverrideRule("stats_compute_reliability", "behavior_analysis.statistics.compute_reliability", _to_bool),
    ConfigOverrideRule(
        "exclude_non_trialwise_features",
        "behavior_analysis.features.exclude_non_trialwise_features",
        _to_bool,
    ),
    ConfigOverrideRule("compute_change_scores", "behavior_analysis.correlations.compute_change_scores", _to_bool),
    ConfigOverrideRule("loso_stability", "behavior_analysis.correlations.loso_stability", _to_bool),
    ConfigOverrideRule("compute_bayes_factors", "behavior_analysis.correlations.compute_bayes_factors", _to_bool),
    ConfigOverrideRule("consistency_enabled", "behavior_analysis.consistency.enabled", _to_bool),
    ConfigOverrideRule("cluster_correction_enabled", "behavior_analysis.cluster_correction.enabled", _to_bool),
    ConfigOverrideRule("cluster_correction_alpha", "behavior_analysis.cluster_correction.alpha", _to_float),
    ConfigOverrideRule(
        "cluster_correction_min_cluster_size",
        "behavior_analysis.cluster_correction.min_cluster_size",
        _to_int,
    ),
    ConfigOverrideRule("cluster_correction_tail", "behavior_analysis.cluster_correction.tail", _to_int),
    ConfigOverrideRule("validation_min_epochs", "validation.min_epochs", _to_int),
    ConfigOverrideRule("validation_min_channels", "validation.min_channels", _to_int),
    ConfigOverrideRule("validation_max_amplitude_uv", "validation.max_amplitude_uv", _to_float),
)

_TRIAL_TABLE_OVERRIDE_RULES = (
    ConfigOverrideRule("trial_table_format", "behavior_analysis.trial_table.format", _to_lower_stripped),
    ConfigOverrideRule("trial_table_add_lag_features", "behavior_analysis.trial_table.add_lag_features", _to_bool),
    ConfigOverrideRule(
        "trial_order_max_missing_fraction",
        "behavior_analysis.trial_order.max_missing_fraction",
        _to_float,
    ),
    ConfigOverrideRule("feature_summaries_enabled", "behavior_analysis.feature_summaries.enabled", _to_bool),
)

_FEATURE_QC_OVERRIDE_RULES = (
    ConfigOverrideRule("feature_qc_enabled", "behavior_analysis.feature_qc.enabled", _to_bool),
    ConfigOverrideRule("feature_qc_max_missing_pct", "behavior_analysis.feature_qc.max_missing_pct", _to_float),
    ConfigOverrideRule("feature_qc_min_variance", "behavior_analysis.feature_qc.min_variance", _to_float),
    ConfigOverrideRule(
        "feature_qc_check_within_run_variance",
        "behavior_analysis.feature_qc.check_within_run_variance",
        _to_bool,
    ),
)

_PAIN_RESIDUAL_OVERRIDE_RULES = (
    ConfigOverrideRule("predictor_residual_enabled", "behavior_analysis.predictor_residual.enabled", _to_bool),
    ConfigOverrideRule("predictor_residual_method", "behavior_analysis.predictor_residual.method", _to_lower_stripped),
    ConfigOverrideRule("predictor_residual_min_samples", "behavior_analysis.predictor_residual.min_samples", _to_int),
    ConfigOverrideRule(
        "predictor_residual_spline_df_candidates",
        "behavior_analysis.predictor_residual.spline_df_candidates",
        _to_list,
    ),
    ConfigOverrideRule("predictor_residual_poly_degree", "behavior_analysis.predictor_residual.poly_degree", _to_int),
)

_PAIN_RESIDUAL_CROSSFIT_OVERRIDE_RULES = (
    ConfigOverrideRule("predictor_residual_crossfit_enabled", "behavior_analysis.predictor_residual.crossfit.enabled", _to_bool),
    ConfigOverrideRule(
        "predictor_residual_crossfit_group_column",
        "behavior_analysis.predictor_residual.crossfit.group_column",
        _to_stripped,
    ),
    ConfigOverrideRule(
        "predictor_residual_crossfit_n_splits",
        "behavior_analysis.predictor_residual.crossfit.n_splits",
        _to_int,
    ),
    ConfigOverrideRule(
        "predictor_residual_crossfit_method",
        "behavior_analysis.predictor_residual.crossfit.method",
        _to_lower_stripped,
    ),
    ConfigOverrideRule(
        "predictor_residual_crossfit_spline_n_knots",
        "behavior_analysis.predictor_residual.crossfit.spline_n_knots",
        _to_int,
    ),
)

_REGRESSION_OVERRIDE_RULES = (
    ConfigOverrideRule("regression_outcome", "behavior_analysis.regression.outcome", _to_lower_stripped),
    ConfigOverrideRule("regression_include_predictor", "behavior_analysis.regression.include_predictor", _to_bool),
    ConfigOverrideRule(
        "regression_predictor_control",
        "behavior_analysis.regression.predictor_control",
        _to_lower_stripped,
    ),
    ConfigOverrideRule("regression_include_trial_order", "behavior_analysis.regression.include_trial_order", _to_bool),
    ConfigOverrideRule("regression_include_prev_terms", "behavior_analysis.regression.include_prev_terms", _to_bool),
    ConfigOverrideRule("regression_include_run_block", "behavior_analysis.regression.include_run_block", _to_bool),
    ConfigOverrideRule("regression_include_interaction", "behavior_analysis.regression.include_interaction", _to_bool),
    ConfigOverrideRule("regression_standardize", "behavior_analysis.regression.standardize", _to_bool),
    ConfigOverrideRule("regression_min_samples", "behavior_analysis.regression.min_samples", _to_int),
    ConfigOverrideRule("regression_primary_unit", "behavior_analysis.regression.primary_unit", _to_lower_stripped),
    ConfigOverrideRule("regression_permutations", "behavior_analysis.regression.n_permutations", _to_int),
    ConfigOverrideRule("regression_max_features", "behavior_analysis.regression.max_features", _to_optional_int_max),
)

_REGRESSION_PREDICTOR_SPLINE_OVERRIDE_RULES = (
    ConfigOverrideRule(
        "regression_predictor_spline_knots",
        "behavior_analysis.regression.predictor_spline.n_knots",
        _to_int,
    ),
    ConfigOverrideRule(
        "regression_predictor_spline_quantile_low",
        "behavior_analysis.regression.predictor_spline.quantile_low",
        _to_float,
    ),
    ConfigOverrideRule(
        "regression_predictor_spline_quantile_high",
        "behavior_analysis.regression.predictor_spline.quantile_high",
        _to_float,
    ),
    ConfigOverrideRule(
        "regression_predictor_spline_min_samples",
        "behavior_analysis.regression.predictor_spline.min_samples",
        _to_int,
    ),
)

_MODELS_OVERRIDE_RULES = (
    ConfigOverrideRule("models_outcomes", "behavior_analysis.models.outcomes", _to_lower_stripped_list),
    ConfigOverrideRule("models_families", "behavior_analysis.models.families", _to_lower_stripped_list),
    ConfigOverrideRule("models_include_predictor", "behavior_analysis.models.include_predictor", _to_bool),
    ConfigOverrideRule("models_predictor_control", "behavior_analysis.models.predictor_control", _to_lower_stripped),
    ConfigOverrideRule("models_include_trial_order", "behavior_analysis.models.include_trial_order", _to_bool),
    ConfigOverrideRule("models_include_prev_terms", "behavior_analysis.models.include_prev_terms", _to_bool),
    ConfigOverrideRule("models_include_run_block", "behavior_analysis.models.include_run_block", _to_bool),
    ConfigOverrideRule("models_include_interaction", "behavior_analysis.models.include_interaction", _to_bool),
    ConfigOverrideRule("models_standardize", "behavior_analysis.models.standardize", _to_bool),
    ConfigOverrideRule("models_min_samples", "behavior_analysis.models.min_samples", _to_int),
    ConfigOverrideRule("models_max_features", "behavior_analysis.models.max_features", _to_optional_int_max),
    ConfigOverrideRule("models_binary_outcome", "behavior_analysis.models.binary_outcome", _to_lower_stripped),
    ConfigOverrideRule("models_primary_unit", "behavior_analysis.models.primary_unit", _to_lower_stripped),
    ConfigOverrideRule(
        "models_force_trial_iid_asymptotic",
        "behavior_analysis.models.force_trial_iid_asymptotic",
        _to_bool,
    ),
)

_MODELS_PREDICTOR_SPLINE_OVERRIDE_RULES = (
    ConfigOverrideRule("models_predictor_spline_knots", "behavior_analysis.models.predictor_spline.n_knots", _to_int),
    ConfigOverrideRule(
        "models_predictor_spline_quantile_low",
        "behavior_analysis.models.predictor_spline.quantile_low",
        _to_float,
    ),
    ConfigOverrideRule(
        "models_predictor_spline_quantile_high",
        "behavior_analysis.models.predictor_spline.quantile_high",
        _to_float,
    ),
    ConfigOverrideRule(
        "models_predictor_spline_min_samples",
        "behavior_analysis.models.predictor_spline.min_samples",
        _to_int,
    ),
)

_STABILITY_OVERRIDE_RULES = (
    ConfigOverrideRule("stability_method", "behavior_analysis.stability.method", _to_lower_stripped),
    ConfigOverrideRule("stability_outcome", "behavior_analysis.stability.outcome", _to_auto_blank_or_lower),
    ConfigOverrideRule("stability_group_column", "behavior_analysis.stability.group_column", _to_auto_blank_or_lower),
    ConfigOverrideRule("stability_partial_predictor", "behavior_analysis.stability.partial_predictor", _to_bool),
    ConfigOverrideRule("stability_min_group_trials", "behavior_analysis.stability.min_group_trials", _to_int),
    ConfigOverrideRule("stability_max_features", "behavior_analysis.stability.max_features", _to_optional_int_max),
    ConfigOverrideRule("stability_alpha", "behavior_analysis.stability.alpha", _to_float),
)

_INFLUENCE_OVERRIDE_RULES = (
    ConfigOverrideRule("influence_outcomes", "behavior_analysis.influence.outcomes", _to_lower_stripped_list),
    ConfigOverrideRule("influence_max_features", "behavior_analysis.influence.max_features", _to_int),
    ConfigOverrideRule("influence_include_predictor", "behavior_analysis.influence.include_predictor", _to_bool),
    ConfigOverrideRule(
        "influence_predictor_control",
        "behavior_analysis.influence.predictor_control",
        _to_lower_stripped,
    ),
    ConfigOverrideRule("influence_include_trial_order", "behavior_analysis.influence.include_trial_order", _to_bool),
    ConfigOverrideRule("influence_include_run_block", "behavior_analysis.influence.include_run_block", _to_bool),
    ConfigOverrideRule("influence_include_interaction", "behavior_analysis.influence.include_interaction", _to_bool),
    ConfigOverrideRule("influence_standardize", "behavior_analysis.influence.standardize", _to_bool),
    ConfigOverrideRule(
        "influence_cooks_threshold",
        "behavior_analysis.influence.cooks_threshold",
        _to_optional_float_threshold,
    ),
    ConfigOverrideRule(
        "influence_leverage_threshold",
        "behavior_analysis.influence.leverage_threshold",
        _to_optional_float_threshold,
    ),
)

_INFLUENCE_PREDICTOR_SPLINE_OVERRIDE_RULES = (
    ConfigOverrideRule(
        "influence_predictor_spline_knots",
        "behavior_analysis.influence.predictor_spline.n_knots",
        _to_int,
    ),
    ConfigOverrideRule(
        "influence_predictor_spline_quantile_low",
        "behavior_analysis.influence.predictor_spline.quantile_low",
        _to_float,
    ),
    ConfigOverrideRule(
        "influence_predictor_spline_quantile_high",
        "behavior_analysis.influence.predictor_spline.quantile_high",
        _to_float,
    ),
    ConfigOverrideRule(
        "influence_predictor_spline_min_samples",
        "behavior_analysis.influence.predictor_spline.min_samples",
        _to_int,
    ),
)

_PAIN_SENSITIVITY_OVERRIDE_RULES = (
    ConfigOverrideRule("predictor_sensitivity_min_trials", "behavior_analysis.predictor_sensitivity.min_trials", _to_int),
    ConfigOverrideRule(
        "predictor_sensitivity_primary_unit",
        "behavior_analysis.predictor_sensitivity.primary_unit",
        _to_lower_stripped,
    ),
    ConfigOverrideRule(
        "predictor_sensitivity_permutations",
        "behavior_analysis.predictor_sensitivity.n_permutations",
        _to_int,
    ),
)

_CORRELATIONS_OVERRIDE_RULES = (
    ConfigOverrideRule("correlations_types", "behavior_analysis.correlations.types", _to_list),
    ConfigOverrideRule("correlations_primary_unit", "behavior_analysis.correlations.primary_unit", _to_lower_stripped),
    ConfigOverrideRule("correlations_min_runs", "behavior_analysis.correlations.min_runs", _to_int),
    ConfigOverrideRule(
        "correlations_prefer_predictor_residual",
        "behavior_analysis.correlations.prefer_predictor_residual",
        _to_bool,
    ),
    ConfigOverrideRule(
        "correlations_permutations",
        "behavior_analysis.correlations.permutation.n_permutations",
        _to_int,
    ),
    ConfigOverrideRule(
        "correlations_use_crossfit_predictor_residual",
        "behavior_analysis.correlations.use_crossfit_predictor_residual",
        _to_bool,
    ),
    ConfigOverrideRule("correlations_target_column", "behavior_analysis.correlations.target_column", _to_stripped),
)

_GROUP_LEVEL_OVERRIDE_RULES = (
    ConfigOverrideRule(
        "group_level_target",
        "behavior_analysis.group_level.multilevel_correlations.target",
        _to_lower_stripped,
    ),
    ConfigOverrideRule(
        "group_level_control_predictor",
        "behavior_analysis.group_level.multilevel_correlations.control_predictor",
        _to_bool,
    ),
    ConfigOverrideRule(
        "group_level_control_trial_order",
        "behavior_analysis.group_level.multilevel_correlations.control_trial_order",
        _to_bool,
    ),
    ConfigOverrideRule(
        "group_level_control_run_effects",
        "behavior_analysis.group_level.multilevel_correlations.control_run_effects",
        _to_bool,
    ),
    ConfigOverrideRule(
        "group_level_max_run_dummies",
        "behavior_analysis.group_level.multilevel_correlations.max_run_dummies",
        _to_int,
    ),
    ConfigOverrideRule(
        "group_level_allow_parametric_fallback",
        "behavior_analysis.group_level.multilevel_correlations.allow_parametric_fallback",
        _to_bool,
    ),
)

_REPORT_OVERRIDE_RULES = (
    ConfigOverrideRule("report_top_n", "behavior_analysis.report.top_n", _to_int),
)

_CONDITION_OVERRIDE_RULES = (
    ConfigOverrideRule("condition_fail_fast", "behavior_analysis.condition.fail_fast", _to_bool),
    ConfigOverrideRule("condition_effect_threshold", "behavior_analysis.condition.effect_size_threshold", _to_float),
    ConfigOverrideRule("condition_min_trials", "behavior_analysis.condition.min_trials_per_condition", _to_int),
    ConfigOverrideRule("condition_compare_column", "behavior_analysis.condition.compare_column", _to_stripped),
    ConfigOverrideRule("condition_compare_values", "behavior_analysis.condition.compare_values", _to_stripped_list),
    ConfigOverrideRule("condition_compare_labels", "behavior_analysis.condition.compare_labels", _to_stripped_list),
    ConfigOverrideRule("condition_overwrite", "behavior_analysis.condition.overwrite", _to_bool),
    ConfigOverrideRule("condition_primary_unit", "behavior_analysis.condition.primary_unit", _to_lower_stripped),
    ConfigOverrideRule("condition_compare_windows", "behavior_analysis.condition.compare_windows", _to_stripped_list),
)

_CONDITION_WINDOW_OVERRIDE_RULES = (
    ConfigOverrideRule(
        "condition_window_primary_unit",
        "behavior_analysis.condition.window_comparison.primary_unit",
        _to_lower_stripped,
    ),
    ConfigOverrideRule(
        "condition_window_min_samples",
        "behavior_analysis.condition.window_comparison.min_samples",
        _to_int,
    ),
)

_TEMPORAL_OVERRIDE_RULES = (
    ConfigOverrideRule("temporal_target_column", "behavior_analysis.temporal.target_column", _to_stripped),
    ConfigOverrideRule(
        "temporal_correction_method",
        "behavior_analysis.temporal.correction_method",
        _to_lower_stripped,
    ),
    ConfigOverrideRule("temporal_split_by_condition", "behavior_analysis.temporal.split_by_condition", _to_bool),
    ConfigOverrideRule("temporal_condition_column", "behavior_analysis.temporal.condition_column", _to_stripped),
    ConfigOverrideRule("temporal_condition_values", "behavior_analysis.temporal.condition_values", _to_stripped_list),
    ConfigOverrideRule(
        "temporal_include_roi_averages",
        "behavior_analysis.temporal.include_roi_averages",
        _to_bool,
    ),
    ConfigOverrideRule("temporal_include_tf_grid", "behavior_analysis.temporal.include_tf_grid", _to_bool),
    ConfigOverrideRule("temporal_time_resolution_ms", "behavior_analysis.temporal.time_resolution_ms", _to_int),
    ConfigOverrideRule("temporal_freqs_hz", "behavior_analysis.temporal.freqs_hz", _to_list),
    ConfigOverrideRule("temporal_smooth_window_ms", "behavior_analysis.temporal.smooth_window_ms", _to_int),
)

_TEMPORAL_ITPC_OVERRIDE_RULES = (
    ConfigOverrideRule(
        "temporal_itpc_baseline_correction",
        "behavior_analysis.temporal.itpc.baseline_correction",
        _to_bool,
    ),
)

_TEMPORAL_ERDS_OVERRIDE_RULES = (
    ConfigOverrideRule("temporal_erds_method", "behavior_analysis.temporal.erds.method", _to_lower_stripped),
)

_TEMPORAL_FEATURES_OVERRIDE_RULES = (
    ConfigOverrideRule("temporal_feature_power", "behavior_analysis.temporal.features.power", _to_bool),
    ConfigOverrideRule("temporal_feature_itpc", "behavior_analysis.temporal.features.itpc", _to_bool),
    ConfigOverrideRule("temporal_feature_erds", "behavior_analysis.temporal.features.erds", _to_bool),
)

_CLUSTER_OVERRIDE_RULES = (
    ConfigOverrideRule("cluster_threshold", "behavior_analysis.cluster.forming_threshold", _to_float),
    ConfigOverrideRule("cluster_min_size", "behavior_analysis.cluster.min_cluster_size", _to_int),
    ConfigOverrideRule("cluster_tail", "behavior_analysis.cluster.tail", _to_int),
    ConfigOverrideRule("cluster_condition_column", "behavior_analysis.cluster.condition_column", _to_stripped),
    ConfigOverrideRule("cluster_condition_values", "behavior_analysis.cluster.condition_values", _to_stripped_list),
)

_MEDIATION_OVERRIDE_RULES = (
    ConfigOverrideRule("mediation_bootstrap", "behavior_analysis.mediation.n_bootstrap", _to_int),
    ConfigOverrideRule("mediation_permutations", "behavior_analysis.mediation.n_permutations", _to_int),
    ConfigOverrideRule("mediation_min_effect_size", "behavior_analysis.mediation.min_effect_size", _to_float),
    ConfigOverrideRule("mediation_max_mediators", "behavior_analysis.mediation.max_mediators", _to_int),
)

_MODERATION_OVERRIDE_RULES = (
    ConfigOverrideRule("moderation_max_features", "behavior_analysis.moderation.max_features", _to_int),
    ConfigOverrideRule("moderation_min_samples", "behavior_analysis.moderation.min_samples", _to_int),
    ConfigOverrideRule("moderation_permutations", "behavior_analysis.moderation.n_permutations", _to_int),
)

_MIXED_EFFECTS_OVERRIDE_RULES = (
    ConfigOverrideRule("mixed_random_effects", "behavior_analysis.mixed_effects.random_effects", _to_lower_stripped),
    ConfigOverrideRule(
        "mixed_include_predictor",
        "behavior_analysis.mixed_effects.include_predictor",
        _to_bool,
    ),
    ConfigOverrideRule("mixed_max_features", "behavior_analysis.mixed_effects.max_features", _to_int),
)

_OUTPUT_OVERRIDE_RULES = (
    ConfigOverrideRule("also_save_csv", "behavior_analysis.output.also_save_csv", _to_bool),
    ConfigOverrideRule("overwrite", "behavior_analysis.output.overwrite", _to_bool),
)


def _configure_behavior_compute_mode(args: argparse.Namespace, config: Any) -> None:
    """
    Populate behavior-analysis configuration from CLI arguments.

    This mutates ``config`` in place but does not have other side effects.
    """
    ba = config.setdefault("behavior_analysis", {})
    stats_cfg = ba.setdefault("statistics", {})
    ba.setdefault("correlations", {})
    ba.setdefault("run_adjustment", {})

    rng_seed = args.rng_seed
    if rng_seed is None:
        rng_seed = config.get("project.random_state")
    if rng_seed is None:
        rng_seed = config.get("behavior_analysis.statistics.base_seed", 42)
    if rng_seed is not None:
        config.setdefault("project", {})["random_state"] = rng_seed
        # Behavior stages derive deterministic subject seeds from statistics.base_seed.
        stats_cfg["base_seed"] = int(rng_seed)

    _apply_override_rules(args, config, _GENERAL_OVERRIDE_RULES)

    robust_correlation = getattr(args, "robust_correlation", None)
    if robust_correlation is not None:
        _set_nested_config_value(
            config,
            "behavior_analysis.robust_correlation",
            _to_optional_robust_method(robust_correlation),
        )

    if args.n_perm is not None:
        n_perm = int(args.n_perm)
        _set_nested_config_value(config, "behavior_analysis.statistics.n_permutations", n_perm)
        _set_nested_config_value(config, "behavior_analysis.cluster.n_permutations", n_perm)

    perm_group_column_preference = getattr(args, "perm_group_column_preference", None)
    if perm_group_column_preference:
        parts: list[str] = []
        for token in perm_group_column_preference:
            parts.extend(str(token).replace(",", " ").split())
        if parts:
            _set_nested_config_value(
                config,
                "behavior_analysis.permutation.group_column_preference",
                [p.strip() for p in parts if p.strip()],
            )

    if getattr(args, "predictor_range", None) is not None:
        _set_nested_config_value(
            config,
            "io.constants.predictor_range",
            [float(args.predictor_range[0]), float(args.predictor_range[1])],
        )
    if getattr(args, "max_missing_channels_fraction", None) is not None:
        _set_nested_config_value(
            config,
            "io.constants.max_missing_channels_fraction",
            float(args.max_missing_channels_fraction),
        )

    _apply_override_rules(args, config, _TRIAL_TABLE_OVERRIDE_RULES)
    _apply_override_rules(args, config, _FEATURE_QC_OVERRIDE_RULES)
    _apply_override_rules(args, config, _PAIN_RESIDUAL_OVERRIDE_RULES)

    # Predictor-model diagnostics are consumed by behavior_analysis.predictor_models.*
    if getattr(args, "predictor_residual_model_compare_enabled", None) is not None:
        enabled = bool(args.predictor_residual_model_compare_enabled)
        _set_nested_config_value(config, "behavior_analysis.predictor_models.model_comparison.enabled", enabled)
    if getattr(args, "predictor_residual_model_compare_min_samples", None) is not None:
        min_samples = int(args.predictor_residual_model_compare_min_samples)
        _set_nested_config_value(
            config,
            "behavior_analysis.predictor_models.model_comparison.min_samples",
            min_samples,
        )
    if getattr(args, "predictor_residual_model_compare_poly_degrees", None) is not None:
        poly_degrees = list(args.predictor_residual_model_compare_poly_degrees)
        _set_nested_config_value(
            config,
            "behavior_analysis.predictor_models.model_comparison.poly_degrees",
            poly_degrees,
        )

    if getattr(args, "predictor_residual_breakpoint_enabled", None) is not None:
        enabled = bool(args.predictor_residual_breakpoint_enabled)
        _set_nested_config_value(config, "behavior_analysis.predictor_models.breakpoint_test.enabled", enabled)
    if getattr(args, "predictor_residual_breakpoint_min_samples", None) is not None:
        min_samples = int(args.predictor_residual_breakpoint_min_samples)
        _set_nested_config_value(
            config,
            "behavior_analysis.predictor_models.breakpoint_test.min_samples",
            min_samples,
        )
    if getattr(args, "predictor_residual_breakpoint_candidates", None) is not None:
        candidates = int(args.predictor_residual_breakpoint_candidates)
        _set_nested_config_value(
            config,
            "behavior_analysis.predictor_models.breakpoint_test.n_candidates",
            candidates,
        )
    if getattr(args, "predictor_residual_breakpoint_quantile_low", None) is not None:
        q_low = float(args.predictor_residual_breakpoint_quantile_low)
        _set_nested_config_value(
            config,
            "behavior_analysis.predictor_models.breakpoint_test.quantile_low",
            q_low,
        )
    if getattr(args, "predictor_residual_breakpoint_quantile_high", None) is not None:
        q_high = float(args.predictor_residual_breakpoint_quantile_high)
        _set_nested_config_value(
            config,
            "behavior_analysis.predictor_models.breakpoint_test.quantile_high",
            q_high,
        )

    _apply_override_rules(args, config, _PAIN_RESIDUAL_CROSSFIT_OVERRIDE_RULES)

    _apply_override_rules(args, config, _REGRESSION_OVERRIDE_RULES)
    if _has_any_arg(
        args,
        (
            "regression_predictor_spline_knots",
            "regression_predictor_spline_quantile_low",
            "regression_predictor_spline_quantile_high",
            "regression_predictor_spline_min_samples",
        ),
    ):
        _apply_override_rules(args, config, _REGRESSION_PREDICTOR_SPLINE_OVERRIDE_RULES)

    _apply_override_rules(args, config, _MODELS_OVERRIDE_RULES)
    if _has_any_arg(
        args,
        (
            "models_predictor_spline_knots",
            "models_predictor_spline_quantile_low",
            "models_predictor_spline_quantile_high",
            "models_predictor_spline_min_samples",
        ),
    ):
        _apply_override_rules(args, config, _MODELS_PREDICTOR_SPLINE_OVERRIDE_RULES)

    _apply_override_rules(args, config, _STABILITY_OVERRIDE_RULES)

    _apply_override_rules(args, config, _INFLUENCE_OVERRIDE_RULES)
    if _has_any_arg(
        args,
        (
            "influence_predictor_spline_knots",
            "influence_predictor_spline_quantile_low",
            "influence_predictor_spline_quantile_high",
            "influence_predictor_spline_min_samples",
        ),
    ):
        _apply_override_rules(args, config, _INFLUENCE_PREDICTOR_SPLINE_OVERRIDE_RULES)

    _apply_override_rules(args, config, _PAIN_SENSITIVITY_OVERRIDE_RULES)
    if getattr(args, "predictor_sensitivity_permutation_primary", None) is not None:
        _set_nested_config_value(
            config,
            "behavior_analysis.predictor_sensitivity.p_primary_mode",
            "perm_if_available" if bool(args.predictor_sensitivity_permutation_primary) else "asymptotic",
        )

    _apply_override_rules(args, config, _CORRELATIONS_OVERRIDE_RULES)
    if getattr(args, "correlations_permutation_primary", None) is not None:
        enabled = bool(args.correlations_permutation_primary)
        _set_nested_config_value(
            config,
            "behavior_analysis.correlations.p_primary_mode",
            "perm_if_available" if enabled else "asymptotic",
        )
        _set_nested_config_value(config, "behavior_analysis.correlations.permutation.enabled", enabled)

    _apply_override_rules(args, config, _GROUP_LEVEL_OVERRIDE_RULES)
    if getattr(args, "group_level_block_permutation", None) is not None:
        enabled = bool(args.group_level_block_permutation)
        _set_nested_config_value(config, "behavior_analysis.group_level.block_permutation", enabled)
        _set_nested_config_value(
            config,
            "behavior_analysis.group_level.multilevel_correlations.block_permutation",
            enabled,
        )

    _apply_override_rules(args, config, _REPORT_OVERRIDE_RULES)

    _apply_override_rules(args, config, _CONDITION_OVERRIDE_RULES)
    _apply_override_rules(args, config, _CONDITION_WINDOW_OVERRIDE_RULES)
    if getattr(args, "condition_permutation_primary", None) is not None:
        enabled = bool(args.condition_permutation_primary)
        _set_nested_config_value(
            config,
            "behavior_analysis.condition.p_primary_mode",
            "perm_if_available" if enabled else "asymptotic",
        )
        _set_nested_config_value(config, "behavior_analysis.condition.permutation.enabled", enabled)

    _apply_override_rules(args, config, _TEMPORAL_OVERRIDE_RULES)

    tmin = getattr(args, "temporal_time_min_ms", None)
    tmax = getattr(args, "temporal_time_max_ms", None)
    if tmin is not None or tmax is not None:
        current_range = config.get("behavior_analysis.temporal.time_range_ms", [-200, 1000])
        lo = int(tmin) if tmin is not None else int(current_range[0])
        hi = int(tmax) if tmax is not None else int(current_range[1])
        _set_nested_config_value(config, "behavior_analysis.temporal.time_range_ms", [lo, hi])

    if _has_any_arg(args, ("temporal_itpc_baseline_min", "temporal_itpc_baseline_max")):
        baseline_window = list(config.get("behavior_analysis.temporal.itpc.baseline_window", [-0.5, -0.01]))
        if getattr(args, "temporal_itpc_baseline_min", None) is not None:
            baseline_window[0] = float(args.temporal_itpc_baseline_min)
        if getattr(args, "temporal_itpc_baseline_max", None) is not None:
            baseline_window[1] = float(args.temporal_itpc_baseline_max)
        _set_nested_config_value(config, "behavior_analysis.temporal.itpc.baseline_window", baseline_window)
    _apply_override_rules(args, config, _TEMPORAL_ITPC_OVERRIDE_RULES)

    if _has_any_arg(args, ("temporal_erds_baseline_min", "temporal_erds_baseline_max")):
        baseline_window = list(config.get("behavior_analysis.temporal.erds.baseline_window", [-0.5, -0.1]))
        if getattr(args, "temporal_erds_baseline_min", None) is not None:
            baseline_window[0] = float(args.temporal_erds_baseline_min)
        if getattr(args, "temporal_erds_baseline_max", None) is not None:
            baseline_window[1] = float(args.temporal_erds_baseline_max)
        _set_nested_config_value(config, "behavior_analysis.temporal.erds.baseline_window", baseline_window)
    _apply_override_rules(args, config, _TEMPORAL_ERDS_OVERRIDE_RULES)

    _apply_override_rules(args, config, _TEMPORAL_FEATURES_OVERRIDE_RULES)

    _apply_override_rules(args, config, _CLUSTER_OVERRIDE_RULES)

    _apply_override_rules(args, config, _MEDIATION_OVERRIDE_RULES)
    if getattr(args, "mediation_permutation_primary", None) is not None:
        _set_nested_config_value(
            config,
            "behavior_analysis.mediation.p_primary_mode",
            "perm_if_available" if bool(args.mediation_permutation_primary) else "asymptotic",
        )

    _apply_override_rules(args, config, _MODERATION_OVERRIDE_RULES)
    if getattr(args, "moderation_permutation_primary", None) is not None:
        _set_nested_config_value(
            config,
            "behavior_analysis.moderation.p_primary_mode",
            "perm_if_available" if bool(args.moderation_permutation_primary) else "asymptotic",
        )

    _apply_override_rules(args, config, _MIXED_EFFECTS_OVERRIDE_RULES)

    _apply_override_rules(args, config, _OUTPUT_OVERRIDE_RULES)


def _build_computation_features(args: argparse.Namespace) -> dict[str, list[str]] | None:
    """
    Collect feature-category selections per computation type from CLI args.

    Returns ``None`` when no computation-specific feature overrides are given.
    """
    computation_features: dict[str, list[str]] = {}

    if getattr(args, "correlations_features", None):
        computation_features["correlations"] = args.correlations_features
    if getattr(args, "predictor_sensitivity_features", None):
        computation_features["predictor_sensitivity"] = args.predictor_sensitivity_features
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
