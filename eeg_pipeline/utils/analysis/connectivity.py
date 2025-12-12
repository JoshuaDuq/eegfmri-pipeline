"""Connectivity analysis helpers (non-plot utilities).

These helpers support parsing connectivity feature columns and constructing matrices.
They are intentionally kept out of plotting modules.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from eeg_pipeline.utils.analysis.events import extract_pain_mask
from eeg_pipeline.utils.analysis.stats import fdr_bh
from eeg_pipeline.utils.config.loader import get_config_value


def parse_connectivity_columns(
    columns: List[str],
    measure: str,
    band: str,
) -> Tuple[List[str], List[Tuple[str, str]], List[int]]:
    """Parse connectivity feature column names to recover edges.

    Supports:
    - New schema: conn_{segment}_{band}_chpair_{ch1}-{ch2}_{stat}
    - Older schema: {measure}_{band}_{ch1}-{ch2} (and variants)
    """
    relevant_cols: List[str] = []
    edges: List[Tuple[str, str]] = []
    indices: List[int] = []

    for idx, col in enumerate(columns):
        # 1) New schema
        if col.startswith("conn_") and f"_{band}_" in col and "_chpair_" in col:
            if measure in col:
                try:
                    parts = col.split("_chpair_")[1].split("_")
                    pair_str = parts[0]
                    if "-" in pair_str:
                        ch1, ch2 = pair_str.split("-")
                        relevant_cols.append(col)
                        edges.append((ch1, ch2))
                        indices.append(idx)
                        continue
                except (IndexError, ValueError):
                    pass

        # 2) Old schema
        prefix = f"{measure}_{band}_"
        if col.startswith(prefix):
            remainder = col[len(prefix) :]

            if "__" in remainder:
                parts = remainder.split("__")
                if len(parts) == 2:
                    relevant_cols.append(col)
                    edges.append((parts[0], parts[1]))
                    indices.append(idx)
                    continue

            if "-" in remainder:
                parts = remainder.split("-")
                if len(parts) == 2:
                    relevant_cols.append(col)
                    edges.append((parts[0], parts[1]))
                    indices.append(idx)
                    continue

            if "_" in remainder:
                parts = remainder.split("_")
                if len(parts) == 2:
                    relevant_cols.append(col)
                    edges.append((parts[0], parts[1]))
                    indices.append(idx)
                    continue

    return relevant_cols, edges, indices


def build_matrix_from_edges(
    edge_values: Dict[Tuple[str, str], float],
    node_order: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Create a symmetric adjacency matrix from an edge-value dict."""
    nodes = node_order or sorted({n for pair in edge_values.keys() for n in pair})
    idx = {n: i for i, n in enumerate(nodes)}
    mat = np.zeros((len(nodes), len(nodes)), dtype=float)
    for (u, v), val in edge_values.items():
        if u not in idx or v not in idx:
            continue
        i, j = idx[u], idx[v]
        mat[i, j] = val
        mat[j, i] = val
    return mat, nodes


def build_adjacency_from_edges(
    features_df: pd.DataFrame,
    edge_cols: List[str],
    channel_order: List[str],
) -> np.ndarray:
    """Build an adjacency matrix by averaging edge columns across trials."""
    n_ch = len(channel_order)
    adj = np.zeros((n_ch, n_ch), dtype=float)
    idx = {ch: i for i, ch in enumerate(channel_order)}

    for col in edge_cols:
        try:
            nodes_str = col.split("_")[-1]
            ch1, ch2 = nodes_str.split("__")
        except ValueError:
            continue

        if ch1 not in idx or ch2 not in idx:
            continue

        i = idx[ch1]
        j = idx[ch2]
        vals = pd.to_numeric(features_df[col], errors="coerce")
        adj[i, j] = float(np.nanmean(vals))
        adj[j, i] = adj[i, j]

    return adj


def compute_significant_edges(
    features_df: pd.DataFrame,
    edge_cols: List[str],
    events_df: Optional[pd.DataFrame],
    config: Any,
) -> Optional[Set[str]]:
    """Return set of edge column names significant (FDR) between pain/non-pain."""
    if events_df is None or events_df.empty:
        return None

    pain_mask = extract_pain_mask(events_df, config)
    if pain_mask is None:
        return None

    n_pain = int(pain_mask.sum())
    n_nonpain = int((~pain_mask).sum())
    if n_pain < 3 or n_nonpain < 3:
        return None

    p_values: List[float] = []
    edge_map: List[str] = []

    df_pain = features_df[pain_mask]
    df_nonpain = features_df[~pain_mask]

    for col in edge_cols:
        vals_pain = pd.to_numeric(df_pain[col], errors="coerce").values
        vals_nonpain = pd.to_numeric(df_nonpain[col], errors="coerce").values

        vals_pain = vals_pain[np.isfinite(vals_pain)]
        vals_nonpain = vals_nonpain[np.isfinite(vals_nonpain)]

        if len(vals_pain) < 3 or len(vals_nonpain) < 3:
            p_values.append(np.nan)
            edge_map.append(col)
            continue

        try:
            _, p = mannwhitneyu(vals_pain, vals_nonpain, alternative="two-sided")
            p_values.append(float(p))
        except ValueError:
            p_values.append(np.nan)

        edge_map.append(col)

    p_arr = np.asarray(p_values, dtype=float)
    finite_mask = np.isfinite(p_arr)
    if not np.any(finite_mask):
        return None

    q_vals = np.full_like(p_arr, np.nan, dtype=float)
    q_vals[finite_mask] = fdr_bh(p_arr[finite_mask], config=config)

    alpha = float(get_config_value(config, "statistics.fdr_alpha", 0.05))

    sig_edges = {edge_map[i] for i, q in enumerate(q_vals) if np.isfinite(q) and q < alpha}
    return sig_edges or None


# Backwards-compatible alias (older name)
compute_significance_mask = compute_significant_edges


__all__ = [
    "parse_connectivity_columns",
    "build_matrix_from_edges",
    "build_adjacency_from_edges",
    "compute_significant_edges",
    "compute_significance_mask",
]


