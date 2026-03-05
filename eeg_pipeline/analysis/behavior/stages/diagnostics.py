from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from eeg_pipeline.utils.config.loader import get_config_value

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
