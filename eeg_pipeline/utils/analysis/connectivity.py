"""Connectivity analysis helpers (non-plot utilities).

These helpers support parsing connectivity feature columns and constructing matrices.
They are intentionally kept out of plotting modules.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import re
from scipy.stats import mannwhitneyu

from eeg_pipeline.utils.analysis.events import extract_comparison_mask
from eeg_pipeline.utils.analysis.stats import fdr_bh
from eeg_pipeline.utils.config.loader import get_config_value


def _compile_token_pattern(token: str) -> re.Pattern:
    """Compile regex pattern to match token at word boundaries."""
    return re.compile(rf"(?:^|[_-]){re.escape(token)}(?:[_-]|$)")


def _extract_chpair_edge(column: str) -> Optional[Tuple[str, str]]:
    """Extract channel pair from column name using chpair schema."""
    column_lower = column.lower()
    if "_chpair_" not in column_lower and "-chpair-" not in column_lower:
        return None

    separator = "_chpair_" if "_chpair_" in column_lower else "-chpair-"
    try:
        remainder = column.split(separator)[1]
        match = re.search(r"([^-]+)-([^_.-]+)", remainder)
        if match:
            return (match.group(1), match.group(2))
    except (IndexError, ValueError):
        pass
    return None


def _extract_global_or_roi_edge(column: str) -> Optional[Tuple[str, str]]:
    """Extract edge label for global or ROI columns."""
    column_lower = column.lower()
    global_tokens = ["_global_", "-global-"]
    roi_tokens = ["_roi_", "-roi-"]

    if any(token in column_lower for token in global_tokens):
        return ("Global", "Global")
    if any(token in column_lower for token in roi_tokens):
        return ("ROI", "ROI")
    return None


def _extract_old_schema_edge(column: str, prefix: str) -> Optional[Tuple[str, str]]:
    """Extract edge from old schema: {measure}_{band}_{ch1}-{ch2} variants."""
    if not column.lower().startswith(prefix.lower()):
        return None

    remainder = column[len(prefix):]
    separators = ["__", "-", "_"]

    for sep in separators:
        if sep in remainder:
            parts = remainder.split(sep)
            if len(parts) == 2:
                return (parts[0], parts[1])
    return None


def parse_connectivity_columns(
    columns: List[str],
    measure: str,
    band: str,
    segment: Optional[str] = None,
) -> Tuple[List[str], List[Tuple[str, str]], List[int]]:
    """Parse connectivity feature column names to recover edges.

    Supports:
    - New schema: conn_{segment}_{band}_chpair_{ch1}-{ch2}_{stat}
    - Older schema: {measure}_{band}_{ch1}-{ch2} (and variants)
    """
    relevant_cols: List[str] = []
    edges: List[Tuple[str, str]] = []
    indices: List[int] = []

    measure_lower = measure.lower()
    band_lower = band.lower()
    segment_lower = segment.lower() if segment else None

    measure_pattern = _compile_token_pattern(measure_lower)
    band_pattern = _compile_token_pattern(band_lower)
    segment_pattern = _compile_token_pattern(segment_lower) if segment_lower else None
    old_schema_prefix = f"{measure_lower}_{band_lower}_"

    for index, column in enumerate(columns):
        column_lower = column.lower()

        if not (measure_pattern.search(column_lower) and band_pattern.search(column_lower)):
            continue

        if segment_pattern and not segment_pattern.search(column_lower):
            continue

        edge = _extract_chpair_edge(column)
        if edge:
            relevant_cols.append(column)
            edges.append(edge)
            indices.append(index)
            continue

        edge = _extract_global_or_roi_edge(column)
        if edge:
            relevant_cols.append(column)
            edges.append(edge)
            indices.append(index)
            continue

        edge = _extract_old_schema_edge(column, old_schema_prefix)
        if edge:
            relevant_cols.append(column)
            edges.append(edge)
            indices.append(index)

    return relevant_cols, edges, indices


def build_matrix_from_edges(
    edge_values: Dict[Tuple[str, str], float],
    node_order: Optional[List[str]] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Create a symmetric adjacency matrix from an edge-value dict."""
    if not edge_values:
        nodes = node_order or []
        matrix = np.zeros((len(nodes), len(nodes)), dtype=float)
        return matrix, nodes

    nodes = node_order or sorted({node for pair in edge_values.keys() for node in pair})
    node_to_index = {node: index for index, node in enumerate(nodes)}
    matrix = np.zeros((len(nodes), len(nodes)), dtype=float)

    for (node1, node2), value in edge_values.items():
        if node1 not in node_to_index or node2 not in node_to_index:
            continue
        index1 = node_to_index[node1]
        index2 = node_to_index[node2]
        matrix[index1, index2] = value
        matrix[index2, index1] = value

    return matrix, nodes


def _parse_edge_from_column(column: str) -> Optional[Tuple[str, str]]:
    """Parse edge from column name as fallback when edges list not provided."""
    edge = _extract_chpair_edge(column)
    if edge:
        return edge

    if "__" in column:
        try:
            nodes_str = column.split("_")[-1]
            if "__" in nodes_str:
                channel1, channel2 = nodes_str.split("__")
                return (channel1, channel2)
        except ValueError:
            pass
    return None


def build_adjacency_from_edges(
    features_df: pd.DataFrame,
    edge_cols: List[str],
    channel_order: List[str],
    edges: Optional[List[Tuple[str, str]]] = None,
) -> np.ndarray:
    """Build an adjacency matrix by averaging edge columns across trials.
    
    If 'edges' is provided (List of (ch1, ch2) tuples matching edge_cols),
    it uses those directly. Otherwise, it falls back to parsing column names.
    """
    num_channels = len(channel_order)
    adjacency_matrix = np.zeros((num_channels, num_channels), dtype=float)
    channel_to_index = {channel: index for index, channel in enumerate(channel_order)}

    for column_index, column in enumerate(edge_cols):
        if edges is not None and column_index < len(edges):
            channel1, channel2 = edges[column_index]
        else:
            parsed_edge = _parse_edge_from_column(column)
            if parsed_edge is None:
                continue
            channel1, channel2 = parsed_edge

        if channel1 not in channel_to_index or channel2 not in channel_to_index:
            continue

        index1 = channel_to_index[channel1]
        index2 = channel_to_index[channel2]
        values = pd.to_numeric(features_df[column], errors="coerce")
        mean_value = float(np.nanmean(values))
        adjacency_matrix[index1, index2] = mean_value
        adjacency_matrix[index2, index1] = mean_value

    return adjacency_matrix


def _compute_p_value_for_edge(
    values1: np.ndarray,
    values2: np.ndarray,
) -> float:
    """Compute Mann-Whitney U p-value for two value arrays."""
    if len(values1) < 3 or len(values2) < 3:
        return np.nan

    try:
        _, p_value = mannwhitneyu(values1, values2, alternative="two-sided")
        return float(p_value)
    except ValueError:
        return np.nan


def compute_significant_edges(
    features_df: pd.DataFrame,
    edge_cols: List[str],
    events_df: Optional[pd.DataFrame],
    config: Any,
) -> Optional[Set[str]]:
    """Return set of edge column names significant (FDR) between two configured conditions."""
    if events_df is None or events_df.empty:
        return None

    comparison = extract_comparison_mask(events_df, config, require_enabled=False)
    if comparison is None:
        return None
    mask1, mask2, _label1, _label2 = comparison

    num_samples1 = int(mask1.sum())
    num_samples2 = int(mask2.sum())
    min_samples_required = 3
    if num_samples1 < min_samples_required or num_samples2 < min_samples_required:
        return None

    p_values: List[float] = []
    edge_column_map: List[str] = []

    condition1_df = features_df[mask1]
    condition2_df = features_df[mask2]

    for column in edge_cols:
        values1_raw = pd.to_numeric(condition1_df[column], errors="coerce").values
        values2_raw = pd.to_numeric(condition2_df[column], errors="coerce").values

        values1_clean = values1_raw[np.isfinite(values1_raw)]
        values2_clean = values2_raw[np.isfinite(values2_raw)]

        p_value = _compute_p_value_for_edge(values1_clean, values2_clean)
        p_values.append(p_value)
        edge_column_map.append(column)

    p_array = np.asarray(p_values, dtype=float)
    finite_mask = np.isfinite(p_array)
    if not np.any(finite_mask):
        return None

    q_values = np.full_like(p_array, np.nan, dtype=float)
    q_values[finite_mask] = fdr_bh(p_array[finite_mask], config=config)

    alpha = float(get_config_value(config, "statistics.fdr_alpha", 0.05))

    significant_edges = {
        edge_column_map[index]
        for index, q_value in enumerate(q_values)
        if np.isfinite(q_value) and q_value < alpha
    }
    return significant_edges if significant_edges else None


__all__ = [
    "parse_connectivity_columns",
    "build_matrix_from_edges",
    "build_adjacency_from_edges",
    "compute_significant_edges",
]
