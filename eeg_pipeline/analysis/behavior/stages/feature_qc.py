from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import pandas as pd

from eeg_pipeline.analysis.behavior.result_types import FeatureQCResult
from eeg_pipeline.utils.config.loader import get_config_bool, get_config_float, get_config_value


def _check_within_run_variance(
    df_trials: pd.DataFrame,
    feature_col: str,
    run_col: str,
    min_variance: float,
) -> bool:
    run_variances = df_trials.groupby(run_col)[feature_col].apply(lambda x: pd.to_numeric(x, errors="coerce").var())
    all_runs_constant = (run_variances.fillna(0) < min_variance).all()
    return not all_runs_constant


def _evaluate_feature_quality(
    feature_col: str,
    values: pd.Series,
    df_trials: pd.DataFrame,
    max_missing_pct: float,
    min_variance: float,
    check_within_run: bool,
    run_col: str,
) -> Dict[str, Any]:
    total_count = len(values)
    missing_count = values.isna().sum()
    missing_pct = missing_count / total_count if total_count > 0 else 1.0
    valid_count = values.notna().sum()
    variance = values.var() if valid_count > 1 else 0.0

    within_run_ok = True
    if check_within_run and run_col in df_trials.columns:
        within_run_ok = _check_within_run_variance(df_trials, feature_col, run_col, min_variance)

    return {
        "feature": feature_col,
        "n_total": total_count,
        "n_missing": missing_count,
        "missing_pct": missing_pct,
        "variance": variance,
        "within_run_variance_ok": within_run_ok,
        "passed": True,
    }


def _classify_feature_failure(
    qc_metrics: Dict[str, Any],
    max_missing_pct: float,
    min_variance: float,
) -> Optional[str]:
    if qc_metrics["missing_pct"] > max_missing_pct:
        return "high_missingness"
    if qc_metrics["variance"] < min_variance:
        return "near_zero_variance"
    if not qc_metrics["within_run_variance_ok"]:
        return "constant_within_run"
    return None


def stage_feature_qc_screen_impl(
    ctx: Any,
    config: Any,
    *,
    load_trial_table_df_fn: Callable[[Any], Optional[pd.DataFrame]],
    is_dataframe_valid_fn: Callable[[Optional[pd.DataFrame]], bool],
    feature_column_prefixes: Sequence[str],
    feature_suffix_from_context_fn: Callable[[Any], str],
    get_stats_subfolder_fn: Callable[[Any, str], Path],
    write_parquet_with_optional_csv_fn: Callable[[pd.DataFrame, Path, bool], None],
    max_missing_pct_default: float,
    min_variance_threshold: float,
) -> FeatureQCResult:
    """Filter features by data quality before inference."""
    _ = config

    df_trials = load_trial_table_df_fn(ctx)
    if not is_dataframe_valid_fn(df_trials):
        ctx.logger.warning("Feature QC: trial table missing; skipping.")
        return FeatureQCResult([], {}, pd.DataFrame(), {"status": "skipped"})

    feature_cols = [col for col in df_trials.columns if str(col).startswith(tuple(feature_column_prefixes))]
    if not feature_cols:
        return FeatureQCResult([], {}, pd.DataFrame(), {"status": "no_features"})

    max_missing_pct = get_config_float(ctx.config, "behavior_analysis.feature_qc.max_missing_pct", max_missing_pct_default)
    min_variance = get_config_float(ctx.config, "behavior_analysis.feature_qc.min_variance", min_variance_threshold)
    check_within_run = get_config_bool(ctx.config, "behavior_analysis.feature_qc.check_within_run_variance", True)
    run_col = str(get_config_value(ctx.config, "behavior_analysis.run_adjustment.column", "run_id") or "run_id").strip()

    passed_features = []
    failed_features: Dict[str, List[str]] = {
        "high_missingness": [],
        "near_zero_variance": [],
        "constant_within_run": [],
    }
    qc_records: List[Dict[str, Any]] = []

    for feature_col in feature_cols:
        values = pd.to_numeric(df_trials[feature_col], errors="coerce")
        qc_metrics = _evaluate_feature_quality(
            feature_col,
            values,
            df_trials,
            max_missing_pct,
            min_variance,
            check_within_run,
            run_col,
        )

        failure_reason = _classify_feature_failure(qc_metrics, max_missing_pct, min_variance)
        if failure_reason:
            failed_features[failure_reason].append(feature_col)
            qc_metrics["passed"] = False
        else:
            passed_features.append(feature_col)

        qc_records.append(qc_metrics)

    qc_df = pd.DataFrame(qc_records)
    n_failed = sum(len(feature_list) for feature_list in failed_features.values())

    n_total = len(feature_cols)
    n_passed = len(passed_features)
    pass_rate = 100 * n_passed / n_total if n_total > 0 else 0.0
    ctx.logger.info("Feature QC: %d/%d passed (%.1f%%), %d failed", n_passed, n_total, pass_rate, n_failed)

    for reason, feature_list in failed_features.items():
        if feature_list:
            ctx.logger.info("  %s: %d features", reason, len(feature_list))

    suffix = feature_suffix_from_context_fn(ctx)
    out_dir = get_stats_subfolder_fn(ctx, "feature_qc")
    out_path = out_dir / f"feature_qc_screen{suffix}.parquet"
    write_parquet_with_optional_csv_fn(qc_df, out_path, also_save_csv=ctx.also_save_csv)

    metadata = {
        "status": "ok",
        "n_total": n_total,
        "n_passed": n_passed,
        "n_failed": n_failed,
        "thresholds": {
            "max_missing_pct": max_missing_pct,
            "min_variance": min_variance,
            "check_within_run": check_within_run,
        },
    }
    ctx.data_qc["feature_qc_screen"] = metadata

    return FeatureQCResult(passed_features, failed_features, qc_df, metadata)
