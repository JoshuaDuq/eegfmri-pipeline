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

    from eeg_pipeline.utils.data.columns import resolve_outcome_column

    outcome = str(get_config_value(ctx.config, "behavior_analysis.stability.outcome", "")).strip().lower()
    if not outcome:
        if "predictor_residual" in df_trials.columns:
            outcome = "predictor_residual"
        else:
            outcome = resolve_outcome_column(df_trials, ctx.config) or "outcome"

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
    attach_predictor_metadata_fn: Callable[[pd.DataFrame, Dict[str, Any], Optional[str]], pd.DataFrame],
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
    influence_meta["predictor_control"] = get_config_value(ctx.config, "behavior_analysis.influence.predictor_control", None)
    out_df = attach_predictor_metadata_fn(out_df, influence_meta, target_col="outcome")

    out_dir = get_stats_subfolder_fn(ctx, "influence_diagnostics")
    filename = build_output_filename_fn(ctx, config, "influence_diagnostics")
    out_path = out_dir / f"{filename}.parquet"
    actual_path = write_stats_table_fn(ctx, out_df, out_path)
    write_metadata_file_fn(out_dir / f"{filename}.metadata.json", meta)
    ctx.logger.info("Influence diagnostics saved: %s (%d rows)", actual_path.name, len(out_df))
    return out_df

def stage_icc_impl(
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
    """Assess within-subject run-to-run reliability (ICC) of EEG features."""
    from eeg_pipeline.utils.analysis.stats.reliability import compute_icc

    filename = build_output_filename_fn(ctx, config, "icc_reliability")

    df_trials = load_trial_table_df_fn(ctx)
    if not is_dataframe_valid_fn(df_trials):
        ctx.logger.warning("ICC: trial table missing; skipping.")
        return pd.DataFrame()

    run_col = str(get_config_value(ctx.config, "behavior_analysis.run_adjustment.column", "run_id") or "run_id").strip()
    if run_col not in df_trials.columns:
        if "run" in df_trials.columns:
            run_col = "run"
        elif "block" in df_trials.columns:
            run_col = "block"
        else:
            ctx.logger.info("ICC: no run/block column available; skipping.")
            return pd.DataFrame()

    feature_cols = get_feature_columns_fn(df_trials, ctx)

    should_skip, skip_reason = check_early_exit_conditions_fn(
        df_trials,
        feature_cols,
        min_features=1,
        min_trials=10,
    )
    if should_skip:
        ctx.logger.info(f"ICC: skipping due to {skip_reason}")
        return pd.DataFrame()

    trial_col = None
    for cand in ("trial_index_within_group", "trial_in_run", "trial", "trial_number", "trial_index"):
        if cand in df_trials.columns:
            trial_col = cand
            break

    if not trial_col:
        ctx.logger.warning("ICC: no intra-run trial column available to align trials across runs; skipping.")
        return pd.DataFrame()
        
    records = []
    
    unique_runs = df_trials[run_col].dropna().unique()
    if len(unique_runs) < 2:
        ctx.logger.info("ICC: only one run available; skipping.")
        return pd.DataFrame()
        
    for feat in feature_cols:
        if feat not in df_trials.columns:
            continue
            
        feat_df = df_trials[[trial_col, run_col, feat]].dropna()
        if feat_df.empty:
            continue
            
        pivoted = feat_df.pivot_table(index=trial_col, columns=run_col, values=feat, aggfunc="mean")
        valid_pivoted = pivoted.dropna()
        
        if len(valid_pivoted) < 2:
            continue
            
        data = valid_pivoted.to_numpy()
        icc_val, ci_low, ci_high = compute_icc(data, icc_type="ICC(3,1)")
        
        records.append({
            "feature": feat,
            "icc": float(icc_val),
            "ci_lower_95": float(ci_low) if not pd.isna(ci_low) else float('nan'),
            "ci_upper_95": float(ci_high) if not pd.isna(ci_high) else float('nan'),
            "n_trials_aligned": int(len(valid_pivoted)),
            "n_runs": int(data.shape[1])
        })

    out_df = pd.DataFrame(records)
    meta = {
        "status": "ok" if not out_df.empty else "empty",
        "icc_type": "ICC(3,1)",
        "run_col": run_col,
        "trial_col": trial_col
    }
    
    ctx.data_qc["icc_reliability"] = meta

    out_dir = get_stats_subfolder_fn(ctx, "icc_reliability")
    out_path = out_dir / f"{filename}.parquet"
    if not out_df.empty:
        actual_path = write_stats_table_fn(ctx, out_df, out_path)
        ctx.logger.info("ICC reliability saved: %s (%d features)", actual_path.name, len(out_df))
    write_metadata_file_fn(out_dir / f"{filename}.metadata.json", meta)
    return out_df
