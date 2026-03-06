from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import pandas as pd

from eeg_pipeline.utils.config.loader import get_config_value
from eeg_pipeline.utils.data.columns import get_binary_outcome_column_from_config


def compute_series_statistics(series: pd.Series) -> Dict[str, Any]:
    """Compute basic statistics for a numeric series."""
    valid_values = series.notna()
    n_valid = int(valid_values.sum())

    if n_valid == 0:
        return {
            "n_non_nan": 0,
            "min": np.nan,
            "max": np.nan,
            "mean": np.nan,
            "std": np.nan,
        }

    numeric_series = pd.to_numeric(series, errors="coerce")
    has_values = numeric_series.notna().any()
    has_multiple = n_valid > 1

    return {
        "n_non_nan": n_valid,
        "min": float(numeric_series.min()) if has_values else np.nan,
        "max": float(numeric_series.max()) if has_values else np.nan,
        "mean": float(numeric_series.mean()) if has_values else np.nan,
        "std": float(numeric_series.std(ddof=1)) if has_multiple else np.nan,
    }


def summarize_covariates_qc_impl(ctx: Any) -> Dict[str, Any]:
    cov = ctx.covariates_df
    if cov is None or cov.empty:
        return {"status": "empty"}
    return {
        "status": "ok",
        "columns": [str(c) for c in cov.columns],
        "missing_fraction_by_column": {
            str(c): float(pd.to_numeric(cov[c], errors="coerce").isna().mean()) for c in cov.columns
        },
    }


def build_behavior_qc_impl(
    ctx: Any,
    *,
    compute_series_statistics_fn: Callable[[pd.Series], Dict[str, Any]],
    compute_correlation_fn: Callable[..., Any],
) -> Dict[str, Any]:
    """Build behavior quality control summary."""
    qc: Dict[str, Any] = {
        "subject": ctx.subject,
        "task": ctx.task,
        "n_trials": ctx.n_trials,
        "has_predictor": ctx.has_predictor,
        "predictor_column": ctx.predictor_column,
        "group_column": ctx.group_column,
    }

    if ctx.data_qc:
        qc["data_qc"] = ctx.data_qc

    outcome_series = None
    outcome_col = ctx._find_outcome_column() if hasattr(ctx, "_find_outcome_column") else None
    qc["outcome_column"] = outcome_col
    if outcome_col is not None and ctx.aligned_events is not None:
        outcome_series = pd.to_numeric(ctx.aligned_events[outcome_col], errors="coerce")

    if outcome_series is not None:
        qc["outcome"] = compute_series_statistics_fn(outcome_series)

    if ctx.predictor_series is not None:
        qc["predictor"] = compute_series_statistics_fn(ctx.predictor_series)

    if outcome_series is not None and ctx.predictor_series is not None:
        s = pd.to_numeric(outcome_series, errors="coerce")
        t = pd.to_numeric(ctx.predictor_series, errors="coerce")
        valid = s.notna() & t.notna()
        if int(valid.sum()) >= 3:
            r, p = compute_correlation_fn(s[valid].values, t[valid].values, method="spearman")
            qc["outcome_predictor_sanity"] = {
                "method": "spearman",
                "n": int(valid.sum()),
                "r": float(r) if np.isfinite(r) else np.nan,
                "p": float(p) if np.isfinite(p) else np.nan,
            }

    if ctx.aligned_events is not None and outcome_series is not None:
        from eeg_pipeline.analysis.behavior.api import split_by_condition

        compare_col = str(
            get_config_value(ctx.config, "behavior_analysis.condition.compare_column", "") or ""
        ).strip()
        condition_enabled = bool(
            get_config_value(ctx.config, "behavior_analysis.condition.enabled", True)
        )
        resolved_condition_col = (
            compare_col if compare_col and compare_col in ctx.aligned_events.columns else None
        )
        if resolved_condition_col is None:
            resolved_condition_col = get_binary_outcome_column_from_config(ctx.config, ctx.aligned_events)

        if condition_enabled and resolved_condition_col is not None:
            try:
                cond_a_mask, cond_b_mask, n_condition_a, n_condition_b = split_by_condition(
                    ctx.aligned_events,
                    ctx.config,
                    ctx.logger,
                )
            except ValueError as exc:
                qc["contrast"] = {
                    "status": "skipped",
                    "reason": str(exc),
                }
                cond_a_mask, cond_b_mask, n_condition_a, n_condition_b = np.array([]), np.array([]), 0, 0
        else:
            cond_a_mask, cond_b_mask, n_condition_a, n_condition_b = np.array([]), np.array([]), 0, 0

        if int(n_condition_a) > 0 or int(n_condition_b) > 0:
            s = pd.to_numeric(outcome_series, errors="coerce")
            cond_a_outcomes = s[cond_a_mask] if len(cond_a_mask) == len(s) else pd.Series(dtype=float)
            cond_b_outcomes = s[cond_b_mask] if len(cond_b_mask) == len(s) else pd.Series(dtype=float)
            qc["contrast"] = {
                "status": "ok",
                "n_condition_a": int(n_condition_a),
                "n_condition_b": int(n_condition_b),
                "mean_outcome_condition_a": float(cond_a_outcomes.mean()) if cond_a_outcomes.notna().any() else np.nan,
                "mean_outcome_condition_b": float(cond_b_outcomes.mean()) if cond_b_outcomes.notna().any() else np.nan,
                "mean_outcome_difference_a_minus_b": (
                    float(cond_a_outcomes.mean() - cond_b_outcomes.mean())
                    if cond_a_outcomes.notna().any() and cond_b_outcomes.notna().any()
                    else np.nan
                ),
            }

    if ctx.covariates_df is not None and not ctx.covariates_df.empty:
        cov = ctx.covariates_df
        qc["covariates"] = {
            "n_covariates": int(cov.shape[1]),
            "columns": [str(col) for col in cov.columns],
            "missing_fraction_by_column": {
                str(col): float(pd.to_numeric(cov[col], errors="coerce").isna().mean())
                for col in cov.columns
            },
        }

    feature_counts: Dict[str, int] = {}
    for name, df in ctx.iter_feature_tables():
        if df is None or df.empty:
            continue
        feature_counts[name] = int(df.shape[1])
    qc["feature_counts"] = feature_counts

    return qc


def write_analysis_metadata_impl(
    ctx: Any,
    pipeline_config: Any,
    results: Any,
    stage_metrics: Optional[Dict[str, Any]] = None,
    outputs_manifest: Optional[Path] = None,
    *,
    build_behavior_qc_fn: Callable[[Any], Dict[str, Any]],
    summarize_covariates_qc_fn: Callable[[Any], Dict[str, Any]],
    get_stats_subfolder_fn: Callable[[Any, str], Path],
) -> Path:
    robust_method = pipeline_config.robust_method
    method_label = pipeline_config.method_label
    payload: Dict[str, Any] = {
        "subject": ctx.subject,
        "task": ctx.task,
        "method": pipeline_config.method,
        "method_label": method_label,
        "robust_method": robust_method,
        "min_samples": pipeline_config.min_samples,
        "control_predictor": pipeline_config.control_predictor,
        "control_trial_order": pipeline_config.control_trial_order,
        "compute_change_scores": pipeline_config.compute_change_scores,
        "compute_reliability": pipeline_config.compute_reliability,
        "n_permutations": pipeline_config.n_permutations,
        "fdr_alpha": pipeline_config.fdr_alpha,
        "n_trials": ctx.n_trials,
        "statistics_config": {
            "method": pipeline_config.method,
            "robust_method": robust_method,
            "method_label": method_label,
            "min_samples": pipeline_config.min_samples,
            "bootstrap": pipeline_config.bootstrap,
            "n_permutations": pipeline_config.n_permutations,
            "fdr_alpha": pipeline_config.fdr_alpha,
            "control_predictor": pipeline_config.control_predictor,
            "control_trial_order": pipeline_config.control_trial_order,
            "compute_change_scores": pipeline_config.compute_change_scores,
            "compute_reliability": pipeline_config.compute_reliability,
            "compute_bayes_factors": getattr(pipeline_config, "compute_bayes_factors", False),
        },
        "outputs": {
            "has_trial_table": bool(getattr(results, "trial_table_path", None)),
            "has_regression": bool(getattr(results, "regression", None) is not None and not results.regression.empty),
            "has_correlations": bool(getattr(results, "correlations", None) is not None and not results.correlations.empty),
            "has_condition_effects": bool(
                getattr(results, "condition_effects", None) is not None and not results.condition_effects.empty
            ),
        },
        "qc": build_behavior_qc_fn(ctx),
    }

    payload["predictor_status"] = {
        "available": bool(ctx.predictor_series is not None and ctx.predictor_series.notna().any())
        if ctx.predictor_series is not None
        else False,
        "control_enabled": bool(ctx.control_predictor),
    }
    if not payload["predictor_status"]["available"]:
        payload["predictor_status"]["reason"] = "missing_predictor"

    payload["covariates_qc"] = summarize_covariates_qc_fn(ctx)

    if stage_metrics:
        payload["stage_metrics"] = stage_metrics

    if outputs_manifest is not None:
        payload["outputs_manifest"] = str(outputs_manifest)

    if ctx.data_qc:
        payload["data_qc"] = ctx.data_qc

    corr_df = getattr(results, "correlations", None)
    if corr_df is not None and not corr_df.empty:
        df = corr_df
        partial_cols = [
            ("p_partial_cov", "partial_cov"),
            ("p_partial_predictor", "partial_predictor"),
            ("p_partial_cov_predictor", "partial_cov_predictor"),
        ]
        partial_ok: Dict[str, Any] = {}
        for col, label in partial_cols:
            if col not in df.columns:
                continue
            pvals = pd.to_numeric(df[col], errors="coerce")
            partial_ok[label] = {
                "n_non_nan": int(pvals.notna().sum()),
                "fraction_non_nan": float(pvals.notna().mean()) if len(pvals) else np.nan,
            }
        if partial_ok:
            payload["partial_correlation_feasibility"] = partial_ok

        if "p_primary_source" in df.columns and df["p_primary_source"].notna().any():
            payload["primary_test_source_counts"] = df["p_primary_source"].fillna("unknown").value_counts().to_dict()

        if "within_family_p_kind" in df.columns and df["within_family_p_kind"].notna().any():
            payload["within_family_p_kind_counts"] = df["within_family_p_kind"].fillna("unknown").value_counts().to_dict()

    out_dir = get_stats_subfolder_fn(ctx, "analysis_metadata")
    path = out_dir / "analysis_metadata.json"
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path
