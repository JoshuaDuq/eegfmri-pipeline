"""
Subject-Level Trial Table (Canonical)
====================================

Builds a single per-trial table that merges:
- aligned events metadata
- targets (rating)
- temperature and pain condition labels
- optional covariates and derived columns
- optional feature columns (prefixed by feature type)

This table is intended to be the single source of truth for subject-level
behavior/statistics and plotting, to avoid silent misalignment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class TrialTableBuildResult:
    df: pd.DataFrame
    metadata: Dict[str, Any]


def _safe_numeric(series: Optional[pd.Series]) -> pd.Series:
    if series is None:
        return pd.Series(dtype=float)
    return pd.to_numeric(series, errors="coerce")


def _pick_event_columns(events: pd.DataFrame, candidates: List[str]) -> Dict[str, pd.Series]:
    out: Dict[str, pd.Series] = {}
    for c in candidates:
        if c in events.columns and c not in out:
            out[c] = events[c]
    return out


def build_subject_trial_table(
    ctx: Any,  # BehaviorContext
    *,
    include_features: bool = True,
    include_covariates: bool = True,
    include_events: bool = True,
    extra_event_columns: Optional[List[str]] = None,
) -> TrialTableBuildResult:
    """
    Build the canonical subject-level trial table.

    Notes
    -----
    - Assumes ctx.load_data() already ran and alignment is validated.
    - Uses ctx.epochs.selection as the original event index when available.
    """
    if ctx.aligned_events is None:
        raise ValueError("ctx.aligned_events is required")
    if ctx.targets is None:
        raise ValueError("ctx.targets is required")

    n = int(len(ctx.targets))
    events = ctx.aligned_events.reset_index(drop=True)
    if len(events) != n:
        raise ValueError(f"Trial table requires aligned events length {len(events)} == targets length {n}")

    meta: Dict[str, Any] = {
        "subject": getattr(ctx, "subject", None),
        "task": getattr(ctx, "task", None),
        "n_trials": n,
        "include_features": bool(include_features),
        "include_covariates": bool(include_covariates),
        "include_events": bool(include_events),
    }

    df = pd.DataFrame(index=np.arange(n))
    df["subject"] = str(getattr(ctx, "subject", ""))
    df["task"] = str(getattr(ctx, "task", ""))
    df["epoch"] = np.arange(n, dtype=int)

    selection = getattr(getattr(ctx, "epochs", None), "selection", None)
    if selection is not None:
        try:
            selection_arr = np.asarray(selection, dtype=int)
            if selection_arr.shape[0] == n:
                df["original_event_index"] = selection_arr
                meta["has_original_event_index"] = True
        except Exception:
            meta["has_original_event_index"] = False

    df["rating"] = _safe_numeric(ctx.targets).reset_index(drop=True)
    if getattr(ctx, "temperature", None) is not None:
        df["temperature"] = _safe_numeric(ctx.temperature).reset_index(drop=True)

    # Pain condition coding (if present)
    try:
        from eeg_pipeline.utils.data.columns import get_pain_column_from_config

        pain_col = get_pain_column_from_config(ctx.config, events)
    except Exception:
        pain_col = None
    if pain_col is not None and pain_col in events.columns:
        df["pain_binary"] = pd.to_numeric(events[pain_col], errors="coerce")
        meta["pain_column"] = str(pain_col)

    if include_events:
        keep = [
            "trial",
            "trial_type",
            "condition",
            "run",
            "block",
            "response_time",
        ]
        if extra_event_columns:
            keep.extend([str(c) for c in extra_event_columns])
        cols = _pick_event_columns(events, keep)
        for k, v in cols.items():
            if k in df.columns:
                continue
            df[k] = v.reset_index(drop=True)

    if include_covariates and getattr(ctx, "covariates_df", None) is not None and not ctx.covariates_df.empty:
        cov = ctx.covariates_df.copy()
        cov = cov.reset_index(drop=True)
        for c in cov.columns:
            name = str(c)
            if name in df.columns:
                name = f"cov_{name}"
            df[name] = pd.to_numeric(cov[c], errors="coerce")

    if include_features:
        from eeg_pipeline.analysis.behavior.orchestration import combine_features

        feats = combine_features(ctx)
        if feats is not None and not feats.empty:
            feats = feats.reset_index(drop=True)
            # Avoid accidental overwrites of metadata columns.
            collisions = set(df.columns).intersection(set(str(c) for c in feats.columns))
            if collisions:
                feats = feats.rename(columns={c: f"feat_{c}" for c in feats.columns if str(c) in collisions})
                meta["feature_column_collisions"] = sorted(str(c) for c in collisions)
            df = pd.concat([df, feats], axis=1)

    meta["n_columns"] = int(df.shape[1])
    return TrialTableBuildResult(df=df, metadata=meta)


def add_lag_and_delta_features(
    df: pd.DataFrame,
    *,
    temperature_col: str = "temperature",
    rating_col: str = "rating",
    group_columns: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Add lagged and delta variables (prev_*, delta_*) within runs/blocks if available."""
    df = df.copy()

    group_columns = [c for c in (group_columns or ["run", "block"]) if c in df.columns]
    meta: Dict[str, Any] = {"group_columns": group_columns}

    if "epoch" in df.columns:
        df = df.sort_values("epoch", kind="stable").reset_index(drop=True)

    has_temp = temperature_col in df.columns
    has_rating = rating_col in df.columns
    if not has_temp and not has_rating:
        meta["status"] = "skipped_no_columns"
        return df, meta

    def _apply(group: pd.DataFrame) -> pd.DataFrame:
        g = group.copy()
        if has_temp:
            t = pd.to_numeric(g[temperature_col], errors="coerce")
            g["prev_temperature"] = t.shift(1)
            g["delta_temperature"] = t - t.shift(1)
        if has_rating:
            y = pd.to_numeric(g[rating_col], errors="coerce")
            g["prev_rating"] = y.shift(1)
            g["delta_rating"] = y - y.shift(1)
        g["trial_index_within_group"] = np.arange(len(g), dtype=int)
        return g

    if group_columns:
        df = df.groupby(group_columns, dropna=False, sort=False, group_keys=False).apply(_apply)
    else:
        df = _apply(df)

    meta["status"] = "ok"
    return df.reset_index(drop=True), meta


def add_pain_residual(
    df: pd.DataFrame,
    config: Any,
    *,
    temperature_col: str = "temperature",
    rating_col: str = "rating",
    out_pred_col: str = "rating_hat_from_temp",
    out_resid_col: str = "pain_residual",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Add a flexible temperature→rating fit and define pain_residual = rating - f(temp)."""
    enabled = bool(
        getattr(config, "get", lambda *_args, **_kwargs: False)(
            "behavior_analysis.pain_residual.enabled", True
        )
    )
    meta: Dict[str, Any] = {"enabled": enabled}
    if not enabled:
        return df, meta
    if temperature_col not in df.columns or rating_col not in df.columns:
        meta["status"] = "skipped_missing_columns"
        return df, meta

    from eeg_pipeline.utils.analysis.stats.pain_residual import fit_temperature_rating_curve

    temp = pd.to_numeric(df[temperature_col], errors="coerce")
    rating = pd.to_numeric(df[rating_col], errors="coerce")
    pred, resid, model_meta = fit_temperature_rating_curve(temp, rating, config=config)
    meta.update(model_meta)

    out = df.copy()
    out[out_pred_col] = pred
    out[out_resid_col] = resid
    return out, meta




def save_trial_table(
    result: TrialTableBuildResult,
    out_path: Path,
    *,
    format: str = "tsv",
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = str(format).strip().lower()
    if fmt in {"tsv", "txt"}:
        from eeg_pipeline.infra.tsv import write_tsv
        write_tsv(result.df, out_path, index=False)
    elif fmt == "parquet":
        # Discouraged but kept for legacy fallback if explicitly requested
        result.df.to_parquet(out_path, index=False)
    else:
        raise ValueError(f"Unsupported trial table format: {format}")
    return out_path


__all__ = [
    "TrialTableBuildResult",
    "build_subject_trial_table",
    "add_lag_and_delta_features",
    "add_pain_residual",
    "save_trial_table",
]
