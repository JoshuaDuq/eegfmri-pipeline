from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging

import numpy as np
import pandas as pd

from .feature_columns import infer_power_band
from .loading import (
    register_feature_block,
    validate_feature_block_lengths,
    validate_trial_alignment_manifest,
)


def align_feature_dataframes(
    pow_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    conn_df: Optional[pd.DataFrame],
    ms_df: Optional[pd.DataFrame],
    aper_df: Optional[pd.DataFrame],
    y: pd.Series,
    aligned_events: pd.DataFrame,
    features_dir: Path,
    logger: logging.Logger,
    config,
    initial_trial_count: Optional[int] = None,
    critical_features: Optional[List[str]] = None,
    extra_blocks: Optional[Dict[str, pd.DataFrame]] = None,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    Optional[pd.DataFrame],
    Optional[pd.DataFrame],
    Optional[pd.DataFrame],
    pd.Series,
    Dict[str, Any],
]:
    """Align feature blocks and target vector while preserving valid trials."""

    def _block_length(block):
        if block is None or getattr(block, "empty", False):
            return None
        return len(block)

    n_trials = next(
        (
            l
            for l in [
                _block_length(pow_df),
                _block_length(baseline_df),
                _block_length(conn_df),
                _block_length(ms_df),
                _block_length(aper_df),
                len(y) if y is not None else None,
            ]
            if l is not None
        ),
        None,
    )

    drop_mask: Optional[np.ndarray] = None
    if n_trials is not None:
        combined_mask = np.ones(n_trials, dtype=bool)

        if pow_df is not None and not getattr(pow_df, "empty", False):
            min_valid_band_fraction = float(
                config.get("feature_engineering.features.min_valid_band_fraction", 0.0)
            )

            band_names = sorted({b for b in (infer_power_band(c) for c in pow_df.columns) if b})
            band_drop_counts = {}
            for band in band_names:
                band_cols = [c for c in pow_df.columns if infer_power_band(c) == band]
                if not band_cols:
                    continue
                values = pow_df[band_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
                valid_fraction = np.isfinite(values).mean(axis=1)
                band_invalid = valid_fraction <= min_valid_band_fraction
                band_drop_counts[band] = int(band_invalid.sum())
                if np.any(band_invalid):
                    pow_df.loc[band_invalid, band_cols] = np.nan
                    qc_col = f"qc_power_{band}"
                    if qc_col not in pow_df.columns:
                        pow_df[qc_col] = True
                    pow_df.loc[band_invalid, qc_col] = False
            for band, n_bad in band_drop_counts.items():
                if n_bad > 0:
                    logger.warning(
                        "Nulling %d trials with insufficient valid power samples for band %s (threshold=%.2f); trials retained for other bands.",
                        n_bad,
                        band,
                        min_valid_band_fraction,
                    )

        def _finite_mask(block):
            if block is None or getattr(block, "empty", False):
                return np.ones(n_trials, dtype=bool)
            if isinstance(block, pd.DataFrame):
                try:
                    data = block.to_numpy(dtype=float, copy=False)
                except (TypeError, ValueError):
                    data = block.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            else:
                data = pd.to_numeric(block, errors="coerce").to_numpy(dtype=float)
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            return np.isfinite(data).any(axis=1)

        combined_mask &= _finite_mask(pow_df)
        combined_mask &= _finite_mask(baseline_df)
        combined_mask &= _finite_mask(ms_df)
        combined_mask &= _finite_mask(aper_df)
        combined_mask &= _finite_mask(y)

        if not np.all(combined_mask):
            drop_mask = combined_mask

    if drop_mask is not None:
        n_mask = len(drop_mask)

        def _safe_mask_apply(df, name: str):
            if df is None or getattr(df, "empty", False):
                return df
            if len(df) != n_mask:
                logger.warning(
                    "Length mismatch for %s: DataFrame has %d rows but mask has %d. Skipping mask application to avoid misalignment.",
                    name,
                    len(df),
                    n_mask,
                )
                return df
            return df.loc[drop_mask].reset_index(drop=True)

        pow_df = _safe_mask_apply(pow_df, "power")
        baseline_df = _safe_mask_apply(baseline_df, "baseline")
        conn_df = _safe_mask_apply(conn_df, "connectivity")
        ms_df = _safe_mask_apply(ms_df, "microstates")
        aper_df = _safe_mask_apply(aper_df, "aperiodic")
        if y is not None and len(y) == n_mask:
            y = y.loc[drop_mask].reset_index(drop=True)
        elif y is not None:
            logger.warning(
                "Length mismatch for target: Series has %d rows but mask has %d. Skipping mask application to avoid misalignment.",
                len(y),
                n_mask,
            )

    block_registry: Dict[str, pd.DataFrame] = {}
    before_lengths: Dict[str, int] = {}

    def _is_valid_df(df):
        return df if df is not None and (not hasattr(df, "empty") or not df.empty) else None

    register_feature_block("power", _is_valid_df(pow_df), block_registry, before_lengths)
    register_feature_block("baseline", _is_valid_df(baseline_df), block_registry, before_lengths)
    register_feature_block("connectivity", _is_valid_df(conn_df), block_registry, before_lengths)
    register_feature_block("microstates", _is_valid_df(ms_df), block_registry, before_lengths)
    register_feature_block("aperiodic", _is_valid_df(aper_df), block_registry, before_lengths)
    register_feature_block("target", y, block_registry, before_lengths)

    if extra_blocks:
        for block_name, block_df in extra_blocks.items():
            register_feature_block(block_name, _is_valid_df(block_df), block_registry, before_lengths)

    if not block_registry:
        logger.warning("No features extracted; skipping save")
        return None, None, None, None, None, None, None

    validate_feature_block_lengths(before_lengths, logger, critical_features=critical_features)

    if aligned_events is not None and len(aligned_events) > 0:
        validate_trial_alignment_manifest(aligned_events, features_dir, logger)

    logger.info(
        "Validated feature block lengths (%s)",
        ", ".join(f"{k}={v}" for k, v in before_lengths.items()),
    )

    pow_df_aligned = block_registry.get("power")
    if pow_df_aligned is not None:
        pow_df_aligned = pow_df_aligned.reset_index(drop=True)

    baseline_df_aligned = block_registry.get("baseline")
    if baseline_df_aligned is not None:
        baseline_df_aligned = baseline_df_aligned.reset_index(drop=True)
    else:
        baseline_df_aligned = baseline_df if baseline_df is not None else pd.DataFrame()

    conn_df_aligned = block_registry.get("connectivity")
    if conn_df_aligned is not None:
        conn_df_aligned = conn_df_aligned.reset_index(drop=True)

    ms_df_aligned = block_registry.get("microstates")
    if ms_df_aligned is not None:
        ms_df_aligned = ms_df_aligned.reset_index(drop=True)

    aper_df_aligned = block_registry.get("aperiodic")
    if aper_df_aligned is not None:
        aper_df_aligned = aper_df_aligned.reset_index(drop=True)

    target_block = block_registry.get("target")
    if target_block is not None:
        y_aligned = target_block.iloc[:, 0] if target_block.shape[1] == 1 else target_block
    else:
        y_aligned = y

    def _get_length(df):
        if df is None:
            return 0
        if hasattr(df, "empty") and df.empty:
            return 0
        return len(df) if hasattr(df, "__len__") else 0

    after_lengths = {
        "power": _get_length(pow_df_aligned),
        "baseline": _get_length(baseline_df_aligned),
        "connectivity": _get_length(conn_df_aligned),
        "microstates": _get_length(ms_df_aligned),
        "aperiodic": _get_length(aper_df_aligned),
        "target": _get_length(y_aligned),
    }

    extra_aligned: Dict[str, Optional[pd.DataFrame]] = {}
    if extra_blocks:
        for block_name in extra_blocks.keys():
            block = block_registry.get(block_name)
            if block is not None:
                block = block.reset_index(drop=True)
            extra_aligned[block_name] = block
            after_lengths[block_name] = _get_length(block)

    retention_stats = {
        "initial_trial_count": initial_trial_count,
        "before_alignment": before_lengths,
        "after_alignment": after_lengths,
        "mask": drop_mask,
        "extra_aligned": extra_aligned,
    }

    return (
        pow_df_aligned,
        baseline_df_aligned,
        conn_df_aligned,
        ms_df_aligned,
        aper_df_aligned,
        y_aligned,
        retention_stats,
    )


__all__ = ["align_feature_dataframes"]
