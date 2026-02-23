from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from eeg_pipeline.analysis.behavior.result_types import TempBreakpointResult, TempModelComparisonResult
from eeg_pipeline.utils.config.loader import get_config_bool, get_config_int, get_config_value


def compute_temp_model_comparison_impl(
    temperature: pd.Series,
    rating: pd.Series,
    config: Any,
) -> TempModelComparisonResult:
    """Compare temperature→rating model fits (linear vs polynomial vs spline)."""
    from eeg_pipeline.utils.analysis.stats.temperature_models import compare_temperature_rating_models

    df_cmp, meta_cmp = compare_temperature_rating_models(temperature, rating, config=config)
    return TempModelComparisonResult(df=df_cmp, metadata=meta_cmp)


def compute_temp_breakpoints_impl(
    temperature: pd.Series,
    rating: pd.Series,
    config: Any,
) -> TempBreakpointResult:
    """Detect threshold temperatures where sensitivity changes."""
    from eeg_pipeline.utils.analysis.stats.temperature_models import fit_temperature_breakpoint_test

    df_bp, meta_bp = fit_temperature_breakpoint_test(temperature, rating, config=config)
    return TempBreakpointResult(df=df_bp, metadata=meta_bp)


def write_temperature_models_impl(
    ctx: Any,
    model_comparison: Optional[TempModelComparisonResult],
    breakpoint: Optional[TempBreakpointResult],
    *,
    feature_suffix_from_context_fn: Callable[[Any], str],
    get_stats_subfolder_fn: Callable[[Any, str], Path],
    is_dataframe_valid_fn: Callable[[Optional[pd.DataFrame]], bool],
    write_parquet_with_optional_csv_fn: Callable[[pd.DataFrame, Path, bool], None],
) -> Path:
    """Write temperature model results to disk."""
    suffix = feature_suffix_from_context_fn(ctx)
    out_dir = get_stats_subfolder_fn(ctx, "temperature_models")

    if model_comparison is not None:
        if is_dataframe_valid_fn(model_comparison.df):
            comparison_path = out_dir / f"model_comparison{suffix}.parquet"
            write_parquet_with_optional_csv_fn(model_comparison.df, comparison_path, also_save_csv=ctx.also_save_csv)

        metadata_path = out_dir / f"model_comparison{suffix}.metadata.json"
        metadata_path.write_text(json.dumps(model_comparison.metadata, indent=2, default=str))
        ctx.data_qc["temperature_model_comparison"] = model_comparison.metadata
        ctx.logger.info("Temperature model comparison saved: %s", out_dir.name)

    if breakpoint is not None:
        if is_dataframe_valid_fn(breakpoint.df):
            breakpoint_path = out_dir / f"breakpoint_candidates{suffix}.parquet"
            write_parquet_with_optional_csv_fn(breakpoint.df, breakpoint_path, also_save_csv=ctx.also_save_csv)

        metadata_path = out_dir / f"breakpoint_test{suffix}.metadata.json"
        metadata_path.write_text(json.dumps(breakpoint.metadata, indent=2, default=str))
        ctx.data_qc["temperature_breakpoint_test"] = breakpoint.metadata
        ctx.logger.info("Temperature breakpoint test saved: %s", out_dir.name)

    return out_dir


def stage_temperature_models_impl(
    ctx: Any,
    config: Any,
    *,
    load_trial_table_df_fn: Callable[[Any], Optional[pd.DataFrame]],
    is_dataframe_valid_fn: Callable[[Optional[pd.DataFrame]], bool],
    compute_temp_model_comparison_fn: Callable[[pd.Series, pd.Series, Any], TempModelComparisonResult],
    compute_temp_breakpoints_fn: Callable[[pd.Series, pd.Series, Any], TempBreakpointResult],
    write_temperature_models_fn: Callable[[Any, Optional[TempModelComparisonResult], Optional[TempBreakpointResult]], Path],
) -> Dict[str, Any]:
    """Compare temperature→rating model fits and test for breakpoints."""
    from eeg_pipeline.utils.data.columns import (
        resolve_outcome_column,
        resolve_predictor_column,
    )

    df = load_trial_table_df_fn(ctx)
    if not is_dataframe_valid_fn(df):
        ctx.logger.warning("Temperature models: trial table missing; skipping.")
        return {"status": "skipped_missing_data"}

    predictor_column = resolve_predictor_column(df, ctx.config) or "temperature"
    outcome_column = resolve_outcome_column(df, ctx.config) or "rating"
    required_columns = {predictor_column, outcome_column}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        ctx.logger.warning(
            "Predictor-model stage requires predictor/outcome columns %s; missing: %s. Skipping.",
            required_columns,
            missing_columns,
        )
        return {"status": "skipped_missing_columns"}

    meta: Dict[str, Any] = {
        "status": "init",
        "predictor_column": predictor_column,
        "outcome_column": outcome_column,
    }
    model_comparison = None
    breakpoint = None

    mc_enabled = get_config_bool(ctx.config, "behavior_analysis.temperature_models.model_comparison.enabled", True)
    if mc_enabled:
        model_comparison = compute_temp_model_comparison_fn(
            df[predictor_column],
            df[outcome_column],
            ctx.config,
        )
        meta["model_comparison"] = model_comparison.metadata

    bp_enabled = get_config_bool(ctx.config, "behavior_analysis.temperature_models.breakpoint_test.enabled", True)
    if bp_enabled:
        breakpoint = compute_temp_breakpoints_fn(
            df[predictor_column],
            df[outcome_column],
            ctx.config,
        )
        meta["breakpoint_test"] = breakpoint.metadata

    write_temperature_models_fn(ctx, model_comparison, breakpoint)
    meta["status"] = "ok"
    return meta


def stage_regression_impl(
    ctx: Any,
    config: Any,
    *,
    feature_suffix_from_context_fn: Callable[[Any], str],
    load_trial_table_df_fn: Callable[[Any], Optional[pd.DataFrame]],
    is_dataframe_valid_fn: Callable[[Optional[pd.DataFrame]], bool],
    get_feature_columns_fn: Callable[[pd.DataFrame, Any], List[str]],
    check_early_exit_conditions_fn: Callable[..., tuple[bool, Optional[str]]],
    sanitize_permutation_groups_fn: Callable[[Any, Any, str], Any],
    attach_temperature_metadata_fn: Callable[[pd.DataFrame, Dict[str, Any], str], pd.DataFrame],
    get_stats_subfolder_fn: Callable[[Any, str], Path],
    write_stats_table_fn: Callable[[Any, pd.DataFrame, Path], Path],
) -> pd.DataFrame:
    """Trialwise regression stage with optional run-level aggregation."""
    from eeg_pipeline.utils.analysis.stats.trialwise_regression import run_trialwise_feature_regressions
    from eeg_pipeline.utils.data.columns import (
        resolve_outcome_column,
        resolve_predictor_column,
    )

    suffix = feature_suffix_from_context_fn(ctx)
    method_label = getattr(config, "method_label", "")
    method_suffix = f"_{method_label}" if method_label else ""

    df_trials = load_trial_table_df_fn(ctx)
    if not is_dataframe_valid_fn(df_trials):
        ctx.logger.warning("Regression: trial table missing; skipping.")
        return pd.DataFrame()

    feature_cols = get_feature_columns_fn(df_trials, ctx)
    should_skip, skip_reason = check_early_exit_conditions_fn(
        df_trials,
        feature_cols,
        min_features=1,
        min_trials=10,
    )
    if should_skip:
        ctx.logger.info("Regression: skipping due to %s", skip_reason)
        return pd.DataFrame()

    primary_unit = str(get_config_value(ctx.config, "behavior_analysis.regression.primary_unit", "trial")).strip().lower()
    use_run_unit = primary_unit in {"run", "run_mean", "runmean", "run_level"}
    run_col = str(get_config_value(ctx.config, "behavior_analysis.run_adjustment.column", "run_id") or "run_id").strip()
    allow_iid_trials = get_config_bool(ctx.config, "behavior_analysis.statistics.allow_iid_trials", False)
    n_perm = get_config_int(ctx.config, "behavior_analysis.regression.n_permutations", 0)

    if use_run_unit and run_col not in df_trials.columns:
        raise ValueError(
            f"Run-level regression requested (primary_unit={primary_unit!r}) "
            f"but run column '{run_col}' is missing from trial table."
        )
    if primary_unit in {"trial", "trialwise"} and not allow_iid_trials and n_perm <= 0:
        raise ValueError(
            "Trial-level regression requires a valid non-i.i.d inference method. "
            "Set behavior_analysis.regression.n_permutations > 0, "
            "use run-level aggregation (behavior_analysis.regression.primary_unit=run_mean), "
            "or set behavior_analysis.statistics.allow_iid_trials=true to override (not recommended)."
        )

    if use_run_unit and run_col in df_trials.columns:
        ctx.logger.info("Regression: aggregating to run-level (primary_unit=%s)", primary_unit)
        outcome_column = resolve_outcome_column(df_trials, ctx.config) or "rating"
        predictor_column = resolve_predictor_column(df_trials, ctx.config) or "temperature"
        agg_cols = [
            c
            for c in (
                feature_cols
                + [outcome_column, predictor_column]
            )
            if c in df_trials.columns
        ]
        df_trials = df_trials.groupby(run_col)[agg_cols].mean().reset_index()
        ctx.logger.info("  Run-level: %d observations", len(df_trials))

    groups = None
    if getattr(ctx, "group_ids", None) is not None:
        groups_candidate = np.asarray(ctx.group_ids)
        if len(groups_candidate) == len(df_trials):
            groups = groups_candidate
        else:
            ctx.logger.warning(
                "Regression: ignoring ctx.group_ids length=%d because current data has %d rows.",
                len(groups_candidate),
                len(df_trials),
            )
    if groups is None:
        if run_col in df_trials.columns:
            groups = df_trials[run_col].to_numpy()
        elif "block" in df_trials.columns:
            groups = df_trials["block"].to_numpy()
        elif "run" in df_trials.columns:
            groups = df_trials["run"].to_numpy()
    groups = sanitize_permutation_groups_fn(groups, ctx.logger, "Regression")
    if primary_unit in {"trial", "trialwise"} and not allow_iid_trials and groups is None:
        raise ValueError(
            "Trial-level regression with permutation inference requires grouped labels. "
            "Provide behavior_analysis.run_adjustment.column in the trial table (or ctx.group_ids), "
            "or set behavior_analysis.statistics.allow_iid_trials=true to override (not recommended)."
        )
    strict_permutation_primary = bool(
        primary_unit in {"trial", "trialwise"} and not allow_iid_trials and n_perm > 0
    )

    reg_df, reg_meta = run_trialwise_feature_regressions(
        df_trials,
        feature_cols=feature_cols,
        config=ctx.config,
        groups_for_permutation=groups,
        strict_permutation_primary=strict_permutation_primary,
    )
    reg_meta["primary_unit"] = primary_unit
    ctx.data_qc["trialwise_regression"] = reg_meta
    reg_df = attach_temperature_metadata_fn(reg_df, reg_meta)

    out_dir = get_stats_subfolder_fn(ctx, "trialwise_regression")
    out_path = out_dir / f"regression_feature_effects{suffix}{method_suffix}.parquet"
    if not reg_df.empty:
        actual_path = write_stats_table_fn(ctx, reg_df, out_path)
        ctx.logger.info("Regression results saved: %s (%d features)", actual_path.name, len(reg_df))
    return reg_df


def stage_models_impl(
    ctx: Any,
    config: Any,
    *,
    feature_suffix_from_context_fn: Callable[[Any], str],
    load_trial_table_df_fn: Callable[[Any], Optional[pd.DataFrame]],
    is_dataframe_valid_fn: Callable[[Optional[pd.DataFrame]], bool],
    get_feature_columns_fn: Callable[[pd.DataFrame, Any], List[str]],
    check_early_exit_conditions_fn: Callable[..., tuple[bool, Optional[str]]],
    attach_temperature_metadata_fn: Callable[[pd.DataFrame, Dict[str, Any], str], pd.DataFrame],
    get_stats_subfolder_fn: Callable[[Any, str], Path],
    write_stats_table_fn: Callable[[Any, pd.DataFrame, Path], Path],
) -> pd.DataFrame:
    """Fit multiple model families per feature (OLS-HC3 / robust / quantile / logistic)."""
    from eeg_pipeline.utils.analysis.stats.feature_models import run_feature_model_families
    from eeg_pipeline.utils.data.columns import (
        resolve_outcome_column,
        resolve_predictor_column,
    )

    suffix = feature_suffix_from_context_fn(ctx)
    method_label = getattr(config, "method_label", "")
    method_suffix = f"_{method_label}" if method_label else ""

    df_trials = load_trial_table_df_fn(ctx)
    if not is_dataframe_valid_fn(df_trials):
        ctx.logger.warning("Models: trial table missing; skipping.")
        return pd.DataFrame()

    feature_cols = get_feature_columns_fn(df_trials, ctx)
    should_skip, skip_reason = check_early_exit_conditions_fn(
        df_trials,
        feature_cols,
        min_features=1,
        min_trials=10,
    )
    if should_skip:
        ctx.logger.info("Models: skipping due to %s", skip_reason)
        return pd.DataFrame()

    primary_unit = str(get_config_value(ctx.config, "behavior_analysis.models.primary_unit", "trial") or "trial").strip().lower()
    use_run_unit = primary_unit in {"run", "run_mean", "runmean", "run_level"}
    allow_iid_trials = get_config_bool(ctx.config, "behavior_analysis.statistics.allow_iid_trials", False)
    force_trial_iid_asymptotic = get_config_bool(
        ctx.config,
        "behavior_analysis.models.force_trial_iid_asymptotic",
        False,
    )
    run_col = str(get_config_value(ctx.config, "behavior_analysis.run_adjustment.column", "run_id") or "run_id").strip()

    if primary_unit in {"trial", "trialwise"} and not allow_iid_trials:
        raise ValueError(
            "Trial-level feature-model inference assumes i.i.d trials and is not recommended. "
            "Use run-level aggregation (behavior_analysis.models.primary_unit=run_mean) "
            "or set behavior_analysis.statistics.allow_iid_trials=true to override."
        )
    if primary_unit in {"trial", "trialwise"} and allow_iid_trials and not force_trial_iid_asymptotic:
        raise ValueError(
            "Trial-level feature models require explicit override to use asymptotic i.i.d inference. "
            "Set behavior_analysis.models.force_trial_iid_asymptotic=true to proceed, "
            "or use run-level aggregation (behavior_analysis.models.primary_unit=run_mean)."
        )

    if use_run_unit:
        if run_col not in df_trials.columns:
            raise ValueError(
                f"Run-level models requested (primary_unit={primary_unit!r}) "
                f"but run column '{run_col}' is missing from trial table."
            )
        ctx.logger.info("Models: aggregating to run-level (primary_unit=%s)", primary_unit)
        outcomes_cfg = get_config_value(ctx.config, "behavior_analysis.models.outcomes", ["rating", "predictor_residual"])
        if isinstance(outcomes_cfg, str):
            outcomes_cfg = [outcomes_cfg]
        elif not isinstance(outcomes_cfg, (list, tuple)):
            outcomes_cfg = ["rating", "predictor_residual"]
        binary_outcome = str(
            get_config_value(ctx.config, "behavior_analysis.models.binary_outcome", "binary_outcome") or "binary_outcome"
        ).strip()
        outcome_column = resolve_outcome_column(df_trials, ctx.config) or "rating"
        predictor_column = resolve_predictor_column(df_trials, ctx.config) or "temperature"
        extra_cols = [
            predictor_column,
            outcome_column,
            "predictor_residual",
            "binary_outcome",
            "trial_index",
            "trial_index_within_group",
            "prev_temperature",
            "prev_rating",
            "delta_temperature",
            "delta_rating",
        ]
        agg_cols = [c for c in set(feature_cols + list(outcomes_cfg) + [binary_outcome] + extra_cols) if c in df_trials.columns]
        agg_numeric = {c: "mean" for c in agg_cols if c != binary_outcome}
        grouped = df_trials.groupby(run_col).agg(agg_numeric)
        if binary_outcome in agg_cols:
            grouped[binary_outcome] = df_trials.groupby(run_col)[binary_outcome].apply(
                lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0]
            )
        df_trials = grouped.reset_index()
        ctx.logger.info("  Run-level: %d observations", len(df_trials))

    model_df, model_meta = run_feature_model_families(
        df_trials,
        feature_cols=feature_cols,
        config=ctx.config,
    )
    ctx.data_qc["feature_models"] = model_meta
    model_df = attach_temperature_metadata_fn(model_df, model_meta, target_col="target")

    out_dir = get_stats_subfolder_fn(ctx, "feature_models")
    out_path = out_dir / f"models_feature_effects{suffix}{method_suffix}.parquet"
    if model_df is not None and not model_df.empty:
        actual_path = write_stats_table_fn(ctx, model_df, out_path)
        ctx.logger.info("Model families results saved: %s (%d rows)", actual_path.name, len(model_df))
    return model_df if model_df is not None else pd.DataFrame()
