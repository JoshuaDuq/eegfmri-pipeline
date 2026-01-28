"""
Feature metadata + aggregation helpers for ML interpretability.

This module turns feature column names into structured metadata using the
canonical :class:`~eeg_pipeline.domain.features.naming.NamingSchema` parser,
and provides simple aggregation utilities (e.g., band/ROI-level importance).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from eeg_pipeline.domain.features.naming import NamingSchema
from eeg_pipeline.utils.analysis.spatial import get_roi_definitions
from eeg_pipeline.utils.analysis.channels import build_roi_map


def build_feature_metadata(
    feature_names: Sequence[str],
    *,
    config: Optional[Any] = None,
) -> pd.DataFrame:
    """Build a metadata table for feature names.

    Columns include:
      - feature, valid, group, segment, band, scope, identifier, stat
      - channel (if scope == "ch"), roi (if scope == "roi" or channel maps to an ROI)
    """
    rows: List[Dict[str, Any]] = []
    for name in feature_names:
        parsed = NamingSchema.parse(str(name))
        row: Dict[str, Any] = {
            "feature": str(name),
            "valid": bool(parsed.get("valid", False)),
            "group": parsed.get("group"),
            "segment": parsed.get("segment"),
            "band": parsed.get("band"),
            "scope": parsed.get("scope"),
            "identifier": parsed.get("identifier"),
            "stat": parsed.get("stat"),
        }
        if parsed.get("scope") == "ch":
            row["channel"] = parsed.get("identifier")
        elif parsed.get("scope") == "roi":
            row["roi"] = parsed.get("identifier")
        rows.append(row)

    meta = pd.DataFrame(rows)

    # Optional channel→ROI mapping for channel-scoped features (helps interpretability).
    if config is not None and "channel" in meta.columns:
        channels = sorted({c for c in meta["channel"].dropna().astype(str).tolist() if c})
        if channels:
            roi_defs = get_roi_definitions(config) or {}
            if isinstance(roi_defs, dict) and roi_defs:
                roi_map = build_roi_map(channels, roi_defs)
                channel_to_roi: Dict[str, str] = {}
                for roi_name, idxs in roi_map.items():
                    for idx in idxs:
                        ch = channels[int(idx)]
                        # Prefer first ROI match if overlaps occur.
                        channel_to_roi.setdefault(str(ch), str(roi_name))
                if "roi" not in meta.columns:
                    meta["roi"] = np.nan
                meta["roi"] = meta["roi"].fillna(meta["channel"].map(channel_to_roi))

    return meta


def aggregate_importance(
    importance_df: pd.DataFrame,
    *,
    feature_col: str = "feature",
    value_col: str,
    group_cols: List[str],
    normalize: bool = True,
) -> pd.DataFrame:
    """Aggregate per-feature importance into grouped summaries.

    Returns a DataFrame with columns:
      - group_cols + [n_features, importance_sum, importance_mean, importance_share?]
    """
    if importance_df is None or importance_df.empty:
        return pd.DataFrame()
    if feature_col not in importance_df.columns or value_col not in importance_df.columns:
        raise ValueError(f"Missing required columns: {feature_col}, {value_col}")

    df = importance_df.copy()
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df[np.isfinite(df[value_col].to_numpy(dtype=float))]
    if df.empty:
        return pd.DataFrame()

    # Fill missing group keys with "unknown" so aggregation is explicit.
    for col in group_cols:
        if col not in df.columns:
            df[col] = "unknown"
        df[col] = df[col].astype(object).where(df[col].notna(), "unknown")

    grouped = (
        df.groupby(group_cols, dropna=False, as_index=False)
        .agg(
            n_features=(feature_col, "count"),
            importance_sum=(value_col, "sum"),
            importance_mean=(value_col, "mean"),
        )
        .sort_values("importance_sum", ascending=False)
        .reset_index(drop=True)
    )

    if normalize:
        total = float(grouped["importance_sum"].sum())
        grouped["importance_share"] = grouped["importance_sum"] / total if total > 0 else np.nan

    return grouped


__all__ = ["build_feature_metadata", "aggregate_importance"]
