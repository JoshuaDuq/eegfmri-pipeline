"""Configuration helpers for behavior analysis CLI command."""

from __future__ import annotations

import argparse
import warnings
from typing import Any

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
    if getattr(args, "global_n_bootstrap", None) is not None:
        stats_cfg["default_n_bootstrap"] = int(args.global_n_bootstrap)

    if args.n_perm is not None:
        stats_cfg["n_permutations"] = int(args.n_perm)
        ba.setdefault("cluster", {})["n_permutations"] = int(args.n_perm)
    if getattr(args, "perm_scheme", None) is not None:
        ba.setdefault("permutation", {})["scheme"] = str(args.perm_scheme).strip().lower()

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
    if getattr(args, "stats_temp_control", None) is not None:
        stats_cfg["temperature_control"] = str(args.stats_temp_control).strip().lower()
    if getattr(args, "stats_allow_iid_trials", None) is not None:
        stats_cfg["allow_iid_trials"] = bool(args.stats_allow_iid_trials)
    if getattr(args, "stats_hierarchical_fdr", None) is not None:
        stats_cfg["hierarchical_fdr"] = bool(args.stats_hierarchical_fdr)
    if getattr(args, "stats_compute_reliability", None) is not None:
        stats_cfg["compute_reliability"] = bool(args.stats_compute_reliability)
    if getattr(args, "perm_group_column_preference", None):
        parts = []
        for token in args.perm_group_column_preference:
            parts.extend(str(token).replace(",", " ").split())
        if parts:
            ba.setdefault("permutation", {})["group_column_preference"] = [p.strip() for p in parts if p.strip()]
    if getattr(args, "exclude_non_trialwise_features", None) is not None:
        ba.setdefault("features", {})["exclude_non_trialwise_features"] = bool(args.exclude_non_trialwise_features)
    if getattr(args, "temperature_range", None) is not None:
        config.setdefault("io", {}).setdefault("constants", {})["temperature_range"] = [float(args.temperature_range[0]), float(args.temperature_range[1])]
    if getattr(args, "max_missing_channels_fraction", None) is not None:
        config.setdefault("io", {}).setdefault("constants", {})["max_missing_channels_fraction"] = float(args.max_missing_channels_fraction)

    if getattr(args, "compute_change_scores", None) is not None:
        corr_cfg["compute_change_scores"] = bool(args.compute_change_scores)
    if getattr(args, "loso_stability", None) is not None:
        corr_cfg["loso_stability"] = bool(args.loso_stability)
    if getattr(args, "compute_bayes_factors", None) is not None:
        corr_cfg["compute_bayes_factors"] = bool(args.compute_bayes_factors)

    if getattr(args, "consistency_enabled", None) is not None:
        ba.setdefault("consistency", {})["enabled"] = bool(args.consistency_enabled)
    ccfg = ba.setdefault("cluster_correction", {})
    if getattr(args, "cluster_correction_enabled", None) is not None:
        ccfg["enabled"] = bool(args.cluster_correction_enabled)
    if getattr(args, "cluster_correction_alpha", None) is not None:
        ccfg["alpha"] = float(args.cluster_correction_alpha)
    if getattr(args, "cluster_correction_min_cluster_size", None) is not None:
        ccfg["min_cluster_size"] = int(args.cluster_correction_min_cluster_size)
    if getattr(args, "cluster_correction_tail", None) is not None:
        ccfg["tail"] = int(args.cluster_correction_tail)
    if getattr(args, "validation_min_epochs", None) is not None:
        config.setdefault("validation", {})["min_epochs"] = int(args.validation_min_epochs)
    if getattr(args, "validation_min_channels", None) is not None:
        config.setdefault("validation", {})["min_channels"] = int(args.validation_min_channels)
    if getattr(args, "validation_max_amplitude_uv", None) is not None:
        config.setdefault("validation", {})["max_amplitude_uv"] = float(args.validation_max_amplitude_uv)

    # Trial table
    tt = ba.setdefault("trial_table", {})
    if getattr(args, "trial_table_format", None) is not None:
        tt["format"] = str(args.trial_table_format).strip().lower()
    if getattr(args, "trial_table_add_lag_features", None) is not None:
        tt["add_lag_features"] = bool(args.trial_table_add_lag_features)
    if getattr(args, "trial_order_max_missing_fraction", None) is not None:
        ba.setdefault("trial_order", {})["max_missing_fraction"] = float(
            args.trial_order_max_missing_fraction
        )

    if getattr(args, "feature_summaries_enabled", None) is not None:
        ba.setdefault("feature_summaries", {})["enabled"] = bool(args.feature_summaries_enabled)

    # Feature QC
    fqc = ba.setdefault("feature_qc", {})
    if getattr(args, "feature_qc_enabled", None) is not None:
        fqc["enabled"] = bool(args.feature_qc_enabled)
    if getattr(args, "feature_qc_max_missing_pct", None) is not None:
        fqc["max_missing_pct"] = float(args.feature_qc_max_missing_pct)
    if getattr(args, "feature_qc_min_variance", None) is not None:
        fqc["min_variance"] = float(args.feature_qc_min_variance)
    if getattr(args, "feature_qc_check_within_run_variance", None) is not None:
        fqc["check_within_run_variance"] = bool(args.feature_qc_check_within_run_variance)

    # Pain residual / temperature-model diagnostics
    pr = ba.setdefault("pain_residual", {})
    if getattr(args, "pain_residual_enabled", None) is not None:
        pr["enabled"] = bool(args.pain_residual_enabled)
    if getattr(args, "pain_residual_method", None) is not None:
        pr["method"] = str(args.pain_residual_method).strip().lower()
    if getattr(args, "pain_residual_min_samples", None) is not None:
        pr["min_samples"] = int(args.pain_residual_min_samples)
    if getattr(args, "pain_residual_spline_df_candidates", None) is not None:
        pr["spline_df_candidates"] = list(args.pain_residual_spline_df_candidates)
    if getattr(args, "pain_residual_poly_degree", None) is not None:
        pr["poly_degree"] = int(args.pain_residual_poly_degree)

    mc = pr.setdefault("model_comparison", {})
    if getattr(args, "pain_residual_model_compare_enabled", None) is not None:
        mc["enabled"] = bool(args.pain_residual_model_compare_enabled)
    if getattr(args, "pain_residual_model_compare_min_samples", None) is not None:
        mc["min_samples"] = int(args.pain_residual_model_compare_min_samples)
    if getattr(args, "pain_residual_model_compare_poly_degrees", None) is not None:
        mc["poly_degrees"] = list(args.pain_residual_model_compare_poly_degrees)

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
    if getattr(args, "models_primary_unit", None) is not None:
        mdl["primary_unit"] = str(args.models_primary_unit).strip().lower()
    if getattr(args, "models_force_trial_iid_asymptotic", None) is not None:
        mdl["force_trial_iid_asymptotic"] = bool(args.models_force_trial_iid_asymptotic)

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
    if getattr(args, "correlations_types", None) is not None:
        corr_cfg["types"] = list(args.correlations_types)
    if getattr(args, "correlations_primary_unit", None) is not None:
        corr_cfg["primary_unit"] = str(args.correlations_primary_unit).strip().lower()
    if getattr(args, "correlations_use_crossfit_pain_residual", None) is not None:
        corr_cfg["use_crossfit_pain_residual"] = bool(args.correlations_use_crossfit_pain_residual)
    if getattr(args, "correlations_permutation_primary", None) is not None:
        enabled = bool(args.correlations_permutation_primary)
        corr_cfg["p_primary_mode"] = "perm_if_available" if enabled else "asymptotic"
        corr_cfg.setdefault("permutation", {})["enabled"] = enabled
    if getattr(args, "correlations_target_column", None) is not None:
        target_col = str(args.correlations_target_column).strip()
        corr_cfg["target_column"] = target_col
        # Explicit target-column selection should drive correlation targets.
        corr_cfg["targets"] = [target_col] if target_col else []
    gl_corr_cfg = ba.setdefault("group_level", {}).setdefault("multilevel_correlations", {})
    if getattr(args, "group_level_block_permutation", None) is not None:
        enabled = bool(args.group_level_block_permutation)
        ba.setdefault("group_level", {})["block_permutation"] = enabled
        gl_corr_cfg["block_permutation"] = enabled
    if getattr(args, "group_level_target", None) is not None:
        gl_corr_cfg["target"] = str(args.group_level_target).strip().lower()
    if getattr(args, "group_level_control_temperature", None) is not None:
        gl_corr_cfg["control_temperature"] = bool(args.group_level_control_temperature)
    if getattr(args, "group_level_control_trial_order", None) is not None:
        gl_corr_cfg["control_trial_order"] = bool(args.group_level_control_trial_order)
    if getattr(args, "group_level_control_run_effects", None) is not None:
        gl_corr_cfg["control_run_effects"] = bool(args.group_level_control_run_effects)
    if getattr(args, "group_level_max_run_dummies", None) is not None:
        gl_corr_cfg["max_run_dummies"] = int(args.group_level_max_run_dummies)

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
    if getattr(args, "condition_overwrite", None) is not None:
        ba.setdefault("condition", {})["overwrite"] = bool(args.condition_overwrite)
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
    if getattr(args, "temporal_target_column", None) is not None:
        temporal_cfg["target_column"] = str(args.temporal_target_column).strip()
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

    # Deprecated TF-heatmap aliases now map to canonical temporal settings.
    tf_flag_used = any(
        getattr(args, name, None) is not None
        for name in (
            "tf_heatmap_enabled",
            "tf_heatmap_freqs",
            "tf_heatmap_time_resolution_ms",
        )
    )
    if tf_flag_used:
        warnings.warn(
            "TF-heatmap flags are deprecated. Use temporal options "
            "(--temporal-include-tf-grid, --temporal-time-resolution-ms) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if getattr(args, "tf_heatmap_enabled", None) is not None:
            temporal_cfg["include_tf_grid"] = bool(args.tf_heatmap_enabled)
        if getattr(args, "tf_heatmap_time_resolution_ms", None) is not None:
            temporal_cfg["time_resolution_ms"] = int(args.tf_heatmap_time_resolution_ms)
        if getattr(args, "tf_heatmap_freqs", None) is not None:
            temporal_cfg["freqs_hz"] = [float(v) for v in (args.tf_heatmap_freqs or [])]

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
    if getattr(args, "overwrite", None) is not None:
        out["overwrite"] = bool(args.overwrite)


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

