from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from eeg_pipeline.utils.config.loader import get_config_value


def stage_stability_impl(
    ctx: Any,
    config: Any,
    *,
    build_output_filename_fn: Callable[[Any, Any, str], str],
    load_trial_table_df_fn: Callable[[Any], Optional[pd.DataFrame]],
    is_dataframe_valid_fn: Callable[[Optional[pd.DataFrame]], bool],
    get_feature_columns_fn: Callable[[pd.DataFrame, Any], List[str]],
    check_early_exit_conditions_fn: Callable[..., Tuple[bool, Optional[str]]],
    get_stats_subfolder_fn: Callable[[Any, str], Path],
    write_stats_table_fn: Callable[[Any, pd.DataFrame, Path], Path],
    write_metadata_file_fn: Callable[[Path, Dict[str, Any]], None],
) -> pd.DataFrame:
    """Assess within-subject run/block stability of feature→outcome associations."""
    from eeg_pipeline.utils.analysis.stats.stability import compute_groupwise_stability

    filename = build_output_filename_fn(ctx, config, "stability_groupwise")

    df_trials = load_trial_table_df_fn(ctx)
    if not is_dataframe_valid_fn(df_trials):
        ctx.logger.warning("Stability: trial table missing; skipping.")
        return pd.DataFrame()

    run_col = str(get_config_value(ctx.config, "behavior_analysis.run_adjustment.column", "run_id") or "run_id").strip()
    group_col = str(get_config_value(ctx.config, "behavior_analysis.stability.group_column", "")).strip()
    if group_col and group_col not in df_trials.columns:
        if group_col == "run" and run_col in df_trials.columns:
            group_col = run_col
        else:
            ctx.logger.warning("Stability: configured group_column '%s' not found; falling back to auto.", group_col)
            group_col = ""
    if not group_col:
        if run_col in df_trials.columns:
            group_col = run_col
        elif "run_id" in df_trials.columns:
            group_col = "run_id"
        else:
            group_col = "run" if "run" in df_trials.columns else ("block" if "block" in df_trials.columns else "")
    if not group_col:
        ctx.logger.info("Stability: no run/block column available; skipping.")
        return pd.DataFrame()

    outcome = str(get_config_value(ctx.config, "behavior_analysis.stability.outcome", "")).strip().lower()
    if not outcome:
        outcome = "pain_residual" if "pain_residual" in df_trials.columns else "rating"

    feature_cols = get_feature_columns_fn(df_trials, ctx)

    should_skip, skip_reason = check_early_exit_conditions_fn(
        df_trials,
        feature_cols,
        min_features=1,
        min_trials=10,
    )
    if should_skip:
        ctx.logger.info(f"Stability: skipping due to {skip_reason}")
        return pd.DataFrame()

    stab_df, stab_meta = compute_groupwise_stability(
        df_trials,
        feature_cols=feature_cols,
        outcome=outcome,
        group_col=group_col,
        config=ctx.config,
    )
    ctx.data_qc["stability_groupwise"] = stab_meta

    out_dir = get_stats_subfolder_fn(ctx, "stability_groupwise")
    out_path = out_dir / f"{filename}.parquet"
    if stab_df is not None and not stab_df.empty:
        actual_path = write_stats_table_fn(ctx, stab_df, out_path)
        ctx.logger.info("Stability results saved: %s (%d features)", actual_path.name, len(stab_df))
    write_metadata_file_fn(out_dir / f"{filename}.metadata.json", stab_meta)
    return stab_df if stab_df is not None else pd.DataFrame()


def stage_consistency_impl(
    ctx: Any,
    config: Any,
    results: Any,
    *,
    build_output_filename_fn: Callable[[Any, Any, str], str],
    get_stats_subfolder_fn: Callable[[Any, str], Path],
    write_stats_table_fn: Callable[[Any, pd.DataFrame, Path], Path],
    write_metadata_file_fn: Callable[[Path, Dict[str, Any]], None],
) -> pd.DataFrame:
    """Merge correlations/regression/models and flag effect-direction contradictions."""
    from eeg_pipeline.utils.analysis.stats.consistency import build_effect_direction_consistency_summary

    filename = build_output_filename_fn(ctx, config, "consistency_summary")

    corr_df = getattr(results, "correlations", None)
    reg_df = getattr(results, "regression", None)
    models_df = getattr(results, "models", None)
    out_df, meta = build_effect_direction_consistency_summary(
        corr_df=corr_df,
        regression_df=reg_df,
        models_df=models_df,
    )
    ctx.data_qc["effect_direction_consistency"] = meta
    if out_df is None or out_df.empty:
        return pd.DataFrame()

    out_dir = get_stats_subfolder_fn(ctx, "consistency_summary")
    out_path = out_dir / f"{filename}.parquet"
    actual_path = write_stats_table_fn(ctx, out_df, out_path)
    write_metadata_file_fn(out_dir / f"{filename}.metadata.json", meta)
    ctx.logger.info("Consistency summary saved: %s (%d features)", actual_path.name, len(out_df))
    return out_df


def stage_influence_impl(
    ctx: Any,
    config: Any,
    results: Any,
    *,
    load_trial_table_df_fn: Callable[[Any], Optional[pd.DataFrame]],
    is_dataframe_valid_fn: Callable[[Optional[pd.DataFrame]], bool],
    get_feature_columns_fn: Callable[[pd.DataFrame, Any], List[str]],
    check_early_exit_conditions_fn: Callable[..., Tuple[bool, Optional[str]]],
    attach_temperature_metadata_fn: Callable[[pd.DataFrame, Dict[str, Any], Optional[str]], pd.DataFrame],
    get_stats_subfolder_fn: Callable[[Any, str], Path],
    build_output_filename_fn: Callable[[Any, Any, str], str],
    write_stats_table_fn: Callable[[Any, pd.DataFrame, Path], Path],
    write_metadata_file_fn: Callable[[Path, Dict[str, Any]], None],
) -> pd.DataFrame:
    """Compute leverage/Cook's summaries for top effects."""
    from eeg_pipeline.utils.analysis.stats.influence import compute_influence_diagnostics

    df_trials = load_trial_table_df_fn(ctx)
    if not is_dataframe_valid_fn(df_trials):
        ctx.logger.info("Influence: trial table missing; skipping.")
        return pd.DataFrame()

    feature_cols = get_feature_columns_fn(df_trials, ctx)

    should_skip, skip_reason = check_early_exit_conditions_fn(
        df_trials,
        feature_cols,
        min_features=1,
        min_trials=10,
    )
    if should_skip:
        ctx.logger.info(f"Influence: skipping due to {skip_reason}")
        return pd.DataFrame()

    out_df, meta = compute_influence_diagnostics(
        df_trials,
        corr_df=getattr(results, "correlations", None),
        regression_df=getattr(results, "regression", None),
        models_df=getattr(results, "models", None),
        config=ctx.config,
    )
    ctx.data_qc["influence_diagnostics"] = meta
    if not is_dataframe_valid_fn(out_df):
        return pd.DataFrame()

    influence_meta = meta if isinstance(meta, dict) else {}
    influence_meta["temperature_control"] = get_config_value(ctx.config, "behavior_analysis.influence.temperature_control", None)
    out_df = attach_temperature_metadata_fn(out_df, influence_meta, target_col="outcome")

    out_dir = get_stats_subfolder_fn(ctx, "influence_diagnostics")
    filename = build_output_filename_fn(ctx, config, "influence_diagnostics")
    out_path = out_dir / f"{filename}.parquet"
    actual_path = write_stats_table_fn(ctx, out_df, out_path)
    write_metadata_file_fn(out_dir / f"{filename}.metadata.json", meta)
    ctx.logger.info("Influence diagnostics saved: %s (%d rows)", actual_path.name, len(out_df))
    return out_df
