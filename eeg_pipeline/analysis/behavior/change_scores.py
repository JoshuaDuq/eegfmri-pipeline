from __future__ import annotations

from typing import Any, Optional

import pandas as pd


def has_precomputed_change_scores_impl(df: Optional[pd.DataFrame]) -> bool:
    """Check if DataFrame already has change score columns from feature pipeline."""
    if df is None or df.empty:
        return False
    return any("_change_" in str(c) for c in df.columns)


def augment_dataframe_with_change_scores_impl(
    df: Optional[pd.DataFrame],
    config: Any,
    *,
    is_dataframe_valid_fn,
) -> Optional[pd.DataFrame]:
    """Add change score columns to a feature DataFrame if not already present."""
    if not is_dataframe_valid_fn(df):
        return df

    if has_precomputed_change_scores_impl(df):
        return df

    from eeg_pipeline.utils.analysis.stats.transforms import compute_change_features

    change_df = compute_change_features(df, config=config)
    if not is_dataframe_valid_fn(change_df):
        return df

    new_columns = [col for col in change_df.columns if col not in df.columns]
    if not new_columns:
        return df

    return pd.concat([df, change_df[new_columns]], axis=1)


def add_change_scores_impl(
    ctx: Any,
    *,
    augment_dataframe_with_change_scores_fn,
    has_precomputed_change_scores_fn,
) -> None:
    """Compute and append change scores (active-baseline) once per context."""
    if ctx._change_scores_added or not ctx.compute_change_scores:
        return

    n_precomputed = sum(
        1
        for df in [ctx.power_df, ctx.connectivity_df, ctx.aperiodic_df]
        if has_precomputed_change_scores_fn(df)
    )
    if n_precomputed > 0:
        ctx.logger.info("Using %d pre-computed change score tables from feature pipeline", n_precomputed)

    ctx.power_df = augment_dataframe_with_change_scores_fn(ctx.power_df, ctx.config)
    ctx.connectivity_df = augment_dataframe_with_change_scores_fn(ctx.connectivity_df, ctx.config)
    ctx.directed_connectivity_df = augment_dataframe_with_change_scores_fn(ctx.directed_connectivity_df, ctx.config)
    ctx.source_localization_df = augment_dataframe_with_change_scores_fn(ctx.source_localization_df, ctx.config)
    ctx.aperiodic_df = augment_dataframe_with_change_scores_fn(ctx.aperiodic_df, ctx.config)
    ctx.itpc_df = augment_dataframe_with_change_scores_fn(ctx.itpc_df, ctx.config)
    ctx.pac_df = augment_dataframe_with_change_scores_fn(ctx.pac_df, ctx.config)
    ctx.complexity_df = augment_dataframe_with_change_scores_fn(ctx.complexity_df, ctx.config)
    ctx._change_scores_added = True
