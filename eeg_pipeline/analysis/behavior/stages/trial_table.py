from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import pandas as pd

from eeg_pipeline.analysis.behavior.result_types import TrialTableResult
from eeg_pipeline.utils.config.loader import get_config_value


def compute_trial_table_impl(ctx: Any, config: Any) -> Optional[TrialTableResult]:
    """Build the canonical per-trial analysis table (compute only, no I/O)."""
    _ = config
    from eeg_pipeline.utils.data.trial_table import build_subject_trial_table

    result = build_subject_trial_table(ctx)
    return TrialTableResult(df=result.df, metadata=result.metadata)


def write_trial_table_impl(
    ctx: Any,
    result: TrialTableResult,
    *,
    trial_table_suffix_from_context_fn: Callable[[Any], str],
    trial_table_output_dir_fn: Callable[[Any], Path],
    write_metadata_file_fn: Callable[[Path, Dict[str, Any]], None],
) -> Path:
    """Write trial table and metadata to disk."""
    from eeg_pipeline.utils.data.trial_table import save_trial_table

    fmt = str(get_config_value(ctx.config, "behavior_analysis.trial_table.format", "tsv")).strip().lower()
    suffix = trial_table_suffix_from_context_fn(ctx)
    fname = f"trials{suffix}"
    out_dir = trial_table_output_dir_fn(ctx)
    out_ext = ".parquet" if fmt == "parquet" else ".tsv"
    out_path = out_dir / f"{fname}{out_ext}"

    class _TableWrapper:
        def __init__(self, df: pd.DataFrame, metadata: Dict[str, Any]):
            self.df = df
            self.metadata = metadata

    save_trial_table(_TableWrapper(result.df, result.metadata), out_path, format=fmt)

    if ctx.also_save_csv:
        from eeg_pipeline.infra.tsv import write_csv

        csv_path = out_dir / f"{fname}.csv"
        write_csv(result.df, csv_path, index=False)
        ctx.logger.info("Also saved trial table as CSV: %s/%s", out_dir.name, csv_path.name)

    meta_path = out_dir / f"{fname}.metadata.json"
    write_metadata_file_fn(meta_path, result.metadata)
    ctx.logger.info(
        "Saved trial table: %s/%s (%d rows, %d cols)",
        out_dir.name,
        out_path.name,
        len(result.df),
        result.df.shape[1],
    )

    return out_path


def try_reuse_cached_trial_table_impl(
    ctx: Any,
    *,
    input_hash: str,
    trial_table_suffix_from_context_fn: Callable[[Any], str],
    trial_table_output_dir_fn: Callable[[Any], Path],
    trial_table_metadata_path_fn: Callable[[Path], Path],
    validate_trial_table_contract_metadata_fn: Callable[[Any, Path, pd.DataFrame], None],
) -> Optional[tuple[Path, pd.DataFrame]]:
    """Reuse existing trial-table artifact when input hash is unchanged."""
    fmt = str(get_config_value(ctx.config, "behavior_analysis.trial_table.format", "tsv")).strip().lower()
    suffix = trial_table_suffix_from_context_fn(ctx)
    fname = f"trials{suffix}"
    out_dir = trial_table_output_dir_fn(ctx)
    out_ext = ".parquet" if fmt == "parquet" else ".tsv"
    out_path = out_dir / f"{fname}{out_ext}"

    if not out_path.exists():
        return None

    meta_path = trial_table_metadata_path_fn(out_path)
    if not meta_path.exists():
        return None

    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(metadata, dict):
        return None

    contract = metadata.get("contract", {}) or {}
    if str(contract.get("input_hash", "")) != str(input_hash):
        return None

    from eeg_pipeline.infra.tsv import read_table

    df_cached = read_table(out_path)
    validate_trial_table_contract_metadata_fn(ctx, out_path, df_cached)

    ctx.logger.info(
        "Trial table unchanged (input hash match); reusing existing artifact: %s/%s",
        out_dir.name,
        out_path.name,
    )
    return out_path, df_cached


def stage_trial_table_impl(
    ctx: Any,
    config: Any,
    *,
    compute_trial_table_input_hash_fn: Callable[[Any], str],
    try_reuse_cached_trial_table_fn: Callable[[Any, str], Optional[Path]],
    compute_trial_table_fn: Callable[[Any, Any], Optional[TrialTableResult]],
    is_dataframe_valid_fn: Callable[[Optional[pd.DataFrame]], bool],
    write_trial_table_fn: Callable[[Any, TrialTableResult], Path],
) -> Optional[Path]:
    """Build and save trial table (composed stage)."""
    input_hash = compute_trial_table_input_hash_fn(ctx)
    cached = try_reuse_cached_trial_table_fn(ctx, input_hash)
    if cached is not None:
        return cached

    result = compute_trial_table_fn(ctx, config)
    if result is None or not is_dataframe_valid_fn(result.df):
        ctx.logger.warning("Trial table: no data to write")
        return None
    contract = result.metadata.setdefault("contract", {})
    contract["input_hash"] = str(input_hash)
    return write_trial_table_fn(ctx, result)


def stage_lag_features_impl(
    ctx: Any,
    config: Any,
    *,
    load_trial_table_df_fn: Callable[[Any], Optional[pd.DataFrame]],
    is_dataframe_valid_fn: Callable[[Optional[pd.DataFrame]], bool],
    feature_suffix_from_context_fn: Callable[[Any], str],
    get_stats_subfolder_fn: Callable[[Any, str], Path],
    write_parquet_with_optional_csv_fn: Callable[[pd.DataFrame, Path, bool], None],
    write_metadata_file_fn: Callable[[Path, Dict[str, Any]], None],
    set_trial_table_cache_fn: Optional[Callable[[pd.DataFrame], None]] = None,
) -> Optional[Path]:
    """Add lagged and delta variables to the trial table."""
    _ = config
    from eeg_pipeline.utils.data.trial_table import add_lag_and_delta_features

    df = load_trial_table_df_fn(ctx)
    if not is_dataframe_valid_fn(df):
        ctx.logger.warning("Lag features: trial table missing; skipping.")
        return None

    run_col = str(get_config_value(ctx.config, "behavior_analysis.run_adjustment.column", "run_id") or "run_id").strip()
    group_cols = [c for c in [run_col, "run_id", "run", "block"] if c]
    seen = set()
    group_cols = [c for c in group_cols if not (c in seen or seen.add(c))]

    df_augmented, lag_meta = add_lag_and_delta_features(df, group_columns=group_cols)

    suffix = feature_suffix_from_context_fn(ctx)
    out_dir = get_stats_subfolder_fn(ctx, "lag_features")
    out_path = out_dir / f"trials_with_lags{suffix}.parquet"
    write_parquet_with_optional_csv_fn(df_augmented, out_path, also_save_csv=ctx.also_save_csv)

    meta_path = out_dir / f"lag_features{suffix}.metadata.json"
    write_metadata_file_fn(meta_path, lag_meta)
    ctx.data_qc["lag_features"] = lag_meta
    if set_trial_table_cache_fn is not None:
        set_trial_table_cache_fn(df_augmented)
    ctx.logger.info("Lag features saved: %s/%s", out_dir.name, out_path.name)

    return out_path


def stage_predictor_residual_impl(
    ctx: Any,
    config: Any,
    *,
    load_trial_table_df_fn: Callable[[Any], Optional[pd.DataFrame]],
    is_dataframe_valid_fn: Callable[[Optional[pd.DataFrame]], bool],
    feature_suffix_from_context_fn: Callable[[Any], str],
    get_stats_subfolder_fn: Callable[[Any, str], Path],
    write_parquet_with_optional_csv_fn: Callable[[pd.DataFrame, Path, bool], None],
    write_metadata_file_fn: Callable[[Path, Dict[str, Any]], None],
    set_trial_table_cache_fn: Optional[Callable[[pd.DataFrame], None]] = None,
) -> Optional[Path]:
    """Compute predictor residual = rating - f(predictor)."""
    _ = config
    from eeg_pipeline.utils.data.trial_table import add_predictor_residual
    from eeg_pipeline.utils.data.columns import (
        resolve_outcome_column,
        resolve_predictor_column,
    )

    df = load_trial_table_df_fn(ctx)
    if not is_dataframe_valid_fn(df):
        ctx.logger.warning("Pain residual: trial table missing; skipping.")
        return None

    predictor_column = resolve_predictor_column(df, ctx.config) or "predictor"
    outcome_column = resolve_outcome_column(df, ctx.config) or "outcome"

    required_columns = {predictor_column, outcome_column}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        ctx.logger.warning(
            "Residual stage requires predictor/outcome columns %s; missing: %s. Skipping.",
            required_columns,
            missing_columns,
        )
        return None

    df_augmented, resid_meta = add_predictor_residual(
        df,
        ctx.config,
        predictor_col=predictor_column,
        outcome_col=outcome_column,
    )

    suffix = feature_suffix_from_context_fn(ctx)
    out_dir = get_stats_subfolder_fn(ctx, "predictor_residual")
    out_path = out_dir / f"trials_with_residual{suffix}.parquet"
    write_parquet_with_optional_csv_fn(df_augmented, out_path, also_save_csv=ctx.also_save_csv)

    meta_path = out_dir / f"predictor_residual{suffix}.metadata.json"
    write_metadata_file_fn(meta_path, resid_meta)
    ctx.data_qc["predictor_residual"] = resid_meta
    if set_trial_table_cache_fn is not None:
        set_trial_table_cache_fn(df_augmented)
    ctx.logger.info("Pain residual saved: %s/%s", out_dir.name, out_path.name)

    return out_path
