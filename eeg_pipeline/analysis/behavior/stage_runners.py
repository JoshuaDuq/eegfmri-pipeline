from __future__ import annotations

from typing import Any, Callable, Dict


def build_stage_runners_impl(
    *,
    stage_load_fn: Callable[[Any], Any],
    stage_trial_table_fn: Callable[[Any, Any], Any],
    stage_predictor_residual_fn: Callable[[Any, Any], Any],
    stage_correlate_design_fn: Callable[[Any, Any], Any],
    stage_correlate_effect_sizes_fn: Callable[[Any, Any, Any], Any],
    stage_correlate_pvalues_fn: Callable[[Any, Any, Any, Any], Any],
    stage_correlate_primary_selection_fn: Callable[[Any, Any, Any, Any], Any],
    stage_correlate_fdr_fn: Callable[[Any, Any, Any], Any],
    stage_regression_fn: Callable[[Any, Any], Any],
    stage_condition_column_fn: Callable[[Any, Any], Any],
    stage_temporal_tfr_fn: Callable[[Any], Any],
    stage_temporal_stats_fn: Callable[[Any], Any],
    stage_cluster_fn: Callable[[Any, Any], Any],
    stage_hierarchical_fdr_summary_fn: Callable[[Any, Any], Any],
    stage_report_fn: Callable[[Any, Any], Any],
    stage_export_fn: Callable[[Any, Any, Any], Any],
    build_results_from_outputs_fn: Callable[[Dict[str, Any]], Any],
    ) -> Dict[str, Callable[[Any, Any, Dict[str, Any]], Any]]:
    """Return data-driven stage runner mapping."""
    return {
        "load": lambda ctx, config, outputs: stage_load_fn(ctx),
        "trial_table": lambda ctx, config, outputs: stage_trial_table_fn(ctx, config),
        "predictor_residual": lambda ctx, config, outputs: stage_predictor_residual_fn(ctx, config),
        "correlate_design": lambda ctx, config, outputs: stage_correlate_design_fn(ctx, config),
        "correlate_effect_sizes": lambda ctx, config, outputs: stage_correlate_effect_sizes_fn(
            ctx, config, outputs.get("correlate_design")
        ),
        "correlate_pvalues": lambda ctx, config, outputs: stage_correlate_pvalues_fn(
            ctx, config, outputs.get("correlate_design"), outputs.get("correlate_effect_sizes", [])
        ),
        "correlate_primary_selection": lambda ctx, config, outputs: stage_correlate_primary_selection_fn(
            ctx,
            config,
            outputs.get("correlate_design"),
            outputs.get("correlate_pvalues") or outputs.get("correlate_effect_sizes", []),
        ),
        "correlate_fdr": lambda ctx, config, outputs: stage_correlate_fdr_fn(
            ctx,
            config,
            outputs.get("correlate_primary_selection")
            or outputs.get("correlate_pvalues")
            or outputs.get("correlate_effect_sizes", []),
        ),
        "regression": lambda ctx, config, outputs: stage_regression_fn(ctx, config),
        "condition_column": lambda ctx, config, outputs: stage_condition_column_fn(ctx, config),
        "temporal_tfr": lambda ctx, config, outputs: stage_temporal_tfr_fn(ctx),
        "temporal_stats": lambda ctx, config, outputs: stage_temporal_stats_fn(ctx),
        "cluster": lambda ctx, config, outputs: stage_cluster_fn(ctx, config),
        "hierarchical_fdr_summary": lambda ctx, config, outputs: stage_hierarchical_fdr_summary_fn(ctx, config),
        "report": lambda ctx, config, outputs: stage_report_fn(ctx, config),
        "export": lambda ctx, config, outputs: stage_export_fn(ctx, config, build_results_from_outputs_fn(outputs)),
    }


def build_stage_runners_from_namespace_impl(
    ns: Dict[str, Any],
    *,
    build_results_from_outputs_fn: Callable[[Dict[str, Any]], Any],
) -> Dict[str, Callable[[Any, Any, Dict[str, Any]], Any]]:
    """Build stage runners by resolving expected stage callables from a namespace dict."""
    return build_stage_runners_impl(
        stage_load_fn=ns["stage_load"],
        stage_trial_table_fn=ns["stage_trial_table"],
        stage_predictor_residual_fn=ns["stage_predictor_residual"],
        stage_correlate_design_fn=ns["stage_correlate_design"],
        stage_correlate_effect_sizes_fn=ns["stage_correlate_effect_sizes"],
        stage_correlate_pvalues_fn=ns["stage_correlate_pvalues"],
        stage_correlate_primary_selection_fn=ns["stage_correlate_primary_selection"],
        stage_correlate_fdr_fn=ns["stage_correlate_fdr"],
        stage_regression_fn=ns["stage_regression"],
        stage_condition_column_fn=ns["stage_condition_column"],
        stage_temporal_tfr_fn=ns["stage_temporal_tfr"],
        stage_temporal_stats_fn=ns["stage_temporal_stats"],
        stage_cluster_fn=ns["stage_cluster"],
        stage_hierarchical_fdr_summary_fn=ns["stage_hierarchical_fdr_summary"],
        stage_report_fn=ns["stage_report"],
        stage_export_fn=ns["stage_export"],
        build_results_from_outputs_fn=build_results_from_outputs_fn,
    )
