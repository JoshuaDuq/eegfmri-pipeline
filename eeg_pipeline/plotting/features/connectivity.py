import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import re
import networkx as nx
from scipy import stats
from mne_connectivity.viz import plot_connectivity_circle

from eeg_pipeline.infra.paths import ensure_dir
from eeg_pipeline.infra.logging import get_logger
from eeg_pipeline.plotting.io.figures import log_if_present, save_fig, get_band_color
from eeg_pipeline.utils.data.columns import find_column_in_events, find_pain_column_in_events
from eeg_pipeline.utils.config.loader import get_config_value, get_frequency_band_names
from ..config import get_plot_config
from eeg_pipeline.plotting.features.utils import get_fdr_alpha
from eeg_pipeline.utils.analysis.events import extract_comparison_mask
from eeg_pipeline.utils.analysis.connectivity import (
    build_adjacency_from_edges,
    build_matrix_from_edges,
    compute_significant_edges,
    parse_connectivity_columns,
)


@lru_cache(maxsize=256)
def _parse_connectivity_columns_cached(
    columns: Tuple[str, ...],
    measure: str,
    band: str,
    segment: Optional[str] = None,
) -> Tuple[Tuple[str, ...], Tuple[Tuple[str, str], ...]]:
    """Parse and cache connectivity columns for a measure and band."""
    cols, edges, _ = parse_connectivity_columns(list(columns), measure, band, segment=segment)
    return tuple(cols), tuple(edges)


def _filter_non_self_edges(
    columns: List[str],
    edges: List[Tuple[str, str]],
) -> Tuple[List[str], List[Tuple[str, str]]]:
    """Filter out self-connections (ch1 == ch2) from connectivity data."""
    filtered_columns = [col for col, edge in zip(columns, edges) if edge[0] != edge[1]]
    filtered_edges = [edge for edge in edges if edge[0] != edge[1]]
    return filtered_columns, filtered_edges


def _extract_unique_nodes(edges: List[Tuple[str, str]]) -> List[str]:
    """Extract sorted unique channel names from edge list."""
    unique_nodes = set()
    for ch1, ch2 in edges:
        unique_nodes.update([ch1, ch2])
    return sorted(list(unique_nodes))


def _get_channel_order(
    edges: List[Tuple[str, str]],
    info: Optional[mne.Info],
) -> Optional[List[str]]:
    """Get ordered channel list from edges, using info if available."""
    unique_nodes = _extract_unique_nodes(edges)
    if not unique_nodes:
        return None
    
    if info is not None:
        channel_order = [ch for ch in info.ch_names if ch in unique_nodes]
        return channel_order if channel_order else None
    
    return unique_nodes


def _build_connectivity_matrix(
    mean_connectivity: np.ndarray,
    edges: List[Tuple[str, str]],
    node_names: List[str],
    threshold: float,
) -> Tuple[np.ndarray, int]:
    """Build symmetric connectivity matrix from edge values above threshold."""
    n_nodes = len(node_names)
    node_indices = {name: idx for idx, name in enumerate(node_names)}
    matrix = np.zeros((n_nodes, n_nodes))
    n_significant = 0
    
    for value, (ch1, ch2) in zip(mean_connectivity, edges):
        if abs(value) >= threshold and ch1 in node_indices and ch2 in node_indices:
            idx1 = node_indices[ch1]
            idx2 = node_indices[ch2]
            matrix[idx1, idx2] = value
            matrix[idx2, idx1] = value
            n_significant += 1
    
    return matrix, n_significant


def _get_connectivity_colormap_and_range(measure: str) -> Tuple[str, Optional[float], Optional[float]]:
    """Get appropriate colormap and value range for connectivity measure."""
    measure_lower = measure.lower()
    if "wpli" in measure_lower or "pli" in measure_lower or "coherence" in measure_lower:
        return "viridis", 0.0, 1.0
    return "RdBu", None, None


def _filter_connectivity_columns_by_roi(
    columns: List[str],
    roi_name: str,
    roi_definitions: Dict[str, Any],
    all_features_columns: List[str],
) -> List[str]:
    """Filter connectivity columns to within-ROI edges."""
    if roi_name == "all" or roi_name not in roi_definitions:
        return columns
    
    from eeg_pipeline.plotting.features.roi import get_roi_channels
    
    channel_pattern = re.compile(r'_chpair_([^_]+)_([^_]+)_')
    all_channel_names = set()
    for col in all_features_columns:
        match = channel_pattern.search(str(col))
        if match:
            all_channel_names.add(match.group(1))
            all_channel_names.add(match.group(2))
    
    roi_channels = set(get_roi_channels(roi_definitions[roi_name], list(all_channel_names)))
    
    filtered_columns = []
    for col in columns:
        match = channel_pattern.search(str(col))
        if match:
            ch1, ch2 = match.group(1), match.group(2)
            if ch1 in roi_channels and ch2 in roi_channels:
                filtered_columns.append(col)
    
    return filtered_columns if filtered_columns else columns


def _get_roi_names_for_comparison(
    config: Any,
    roi_definitions: Dict[str, Any],
) -> List[str]:
    """Get list of ROI names to use for comparison plots."""
    comparison_rois = get_config_value(config, "plotting.comparisons.comparison_rois", [])
    if comparison_rois:
        roi_names = []
        for roi in comparison_rois:
            if roi.lower() == "all":
                if "all" not in roi_names:
                    roi_names.append("all")
            elif roi in roi_definitions:
                roi_names.append(roi)
        return roi_names
    
    roi_names = ["all"]
    if roi_definitions:
        roi_names.extend(list(roi_definitions.keys()))
    return roi_names


def _detect_segments_from_data(
    features_df: pd.DataFrame,
    config: Any,
    logger: Optional[logging.Logger],
) -> List[str]:
    """Auto-detect segments from data if not in config."""
    segments = get_config_value(config, "plotting.comparisons.comparison_windows", [])
    if segments and len(segments) >= 2:
        return segments
    
    default_measures = ["aec", "wpli", "pli", "plv", "coherence"]
    from eeg_pipeline.plotting.features.utils import get_named_segments
    
    for measure in default_measures:
        detected = get_named_segments(features_df, group=measure)
        if len(detected) >= 2:
            if logger:
                logger.info(f"Auto-detected segments for connectivity comparison: {detected}")
            return detected[:2]
    
    return []


def _plot_window_comparison_connectivity(
    features_df: pd.DataFrame,
    segments: List[str],
    measures: List[str],
    bands: List[str],
    roi_names: List[str],
    roi_definitions: Dict[str, Any],
    subject: str,
    save_dir: Path,
    config: Any,
    logger: logging.Logger,
    stats_dir: Optional[Path],
) -> None:
    """Plot paired window comparison for connectivity measures."""
    from eeg_pipeline.plotting.features.utils import plot_paired_comparison
    
    segment1, segment2 = segments[0], segments[1]
    
    for roi_name in roi_names:
        for measure in measures:
            data_by_band = {}
            for band in bands:
                cols1, _, _ = parse_connectivity_columns(
                    list(features_df.columns), measure, band, segment=segment1
                )
                cols2, _, _ = parse_connectivity_columns(
                    list(features_df.columns), measure, band, segment=segment2
                )
                
                cols1 = _filter_connectivity_columns_by_roi(
                    cols1, roi_name, roi_definitions, list(features_df.columns)
                )
                cols2 = _filter_connectivity_columns_by_roi(
                    cols2, roi_name, roi_definitions, list(features_df.columns)
                )
                
                if not cols1 or not cols2:
                    continue
                
                series1 = features_df[cols1].apply(pd.to_numeric, errors="coerce").mean(axis=1)
                series2 = features_df[cols2].apply(pd.to_numeric, errors="coerce").mean(axis=1)
                
                valid_mask = series1.notna() & series2.notna()
                values1 = series1[valid_mask].values
                values2 = series2[valid_mask].values
                
                if len(values1) > 0:
                    data_by_band[band] = (values1, values2)
            
            if data_by_band:
                measure_safe = measure.lower()
                roi_safe = roi_name.replace(" ", "_").lower() if roi_name != "all" else ""
                suffix = f"_roi-{roi_safe}" if roi_safe else ""
                save_path = save_dir / (
                    f"sub-{subject}_connectivity_{measure_safe}_by_condition{suffix}_window"
                )
                
                plot_paired_comparison(
                    data_by_band=data_by_band,
                    subject=subject,
                    save_path=save_path,
                    feature_label=f"Connectivity ({measure.upper()})",
                    config=config,
                    logger=logger,
                    label1=segment1.capitalize(),
                    label2=segment2.capitalize(),
                    roi_name=roi_name,
                    stats_dir=stats_dir,
                )
    
    log_if_present(logger, "info", 
                  f"Saved connectivity paired comparison plots for "
                  f"{len(measures)} measures × {len(roi_names)} ROIs")


def plot_connectivity_circle_for_band(
    features_df: pd.DataFrame,
    info: mne.Info,
    subject: str,
    save_dir: Path,
    logger: logging.Logger,
    config: Any,
    measure: str = "wpli",
    band: str = "alpha",
    n_lines: Optional[int] = None,
    significance_threshold: Optional[float] = None,
) -> None:
    """Plot connectivity circle diagram for a single band."""
    if features_df is None or features_df.empty:
        log_if_present(logger, "warning", "No feature data for connectivity plot")
        return

    plot_cfg = get_plot_config(config)
    columns_tuple, edges_tuple = _parse_connectivity_columns_cached(
        tuple(features_df.columns), measure, band
    )
    columns, edges = _filter_non_self_edges(list(columns_tuple), list(edges_tuple))
    
    if not columns:
        log_if_present(logger, "debug", 
                      f"No channel-pair connectivity columns found for {measure} {band}")
        return
        
    n_trials = len(features_df)
    mean_connectivity = features_df[columns].mean(axis=0).values
    
    default_top_fraction = float(get_config_value(
        config, "plotting.plots.features.connectivity.circle_top_fraction", 0.1
    ))
    top_fraction = significance_threshold if significance_threshold is not None else default_top_fraction
    
    absolute_connectivity = np.abs(mean_connectivity)
    threshold = np.percentile(absolute_connectivity, (1 - top_fraction) * 100)
    
    node_names = _extract_unique_nodes(edges)
    n_nodes = len(node_names)
    n_edges = len(edges)
    
    connectivity_matrix, n_significant = _build_connectivity_matrix(
        mean_connectivity, edges, node_names, threshold
    )

    figure_size = plot_cfg.get_figure_size("square", plot_type="connectivity")
    fig, ax = plt.subplots(figsize=figure_size, subplot_kw=dict(polar=True))
    
    colormap, vmin, vmax = _get_connectivity_colormap_and_range(measure)
    
    min_lines_config = int(get_config_value(
        config, "plotting.plots.features.connectivity.circle_min_lines", 20
    ))
    n_lines_to_show = n_lines if n_lines is not None else max(min_lines_config, n_significant)
    
    try:
        plot_connectivity_circle(
            connectivity_matrix,
            node_names,
            n_lines=n_lines_to_show,
            node_angles=None,
            node_colors=None,
            title="",
            ax=ax,
            show=False,
            vmin=vmin,
            vmax=vmax,
            colorbar=True,
            colormap=colormap
        )
    except Exception as e:
        log_if_present(logger, "error", f"Failed to plot connectivity circle: {e}")
        plt.close(fig)
        return

    title_text = (
        f"{measure.upper()} Connectivity: {band.capitalize()} Band\n"
        f"Subject: {subject} | Top {int(top_fraction*100)}% connections "
        f"(threshold ≥ {threshold:.3f})"
    )
    fig.suptitle(title_text, fontsize=plot_cfg.font.figure_title, 
                 fontweight="bold", y=0.98)
    
    percentage_significant = (n_significant / n_edges * 100) if n_edges > 0 else 0.0
    footer_text = (
        f"n = {n_trials} trials | {n_nodes} nodes | "
        f"{n_significant}/{n_edges} significant edges ({percentage_significant:.1f}%)"
    )
    fig.text(
        0.5, 0.02, footer_text,
        ha='center', va='bottom',
        fontsize=plot_cfg.font.large, color='gray', alpha=0.8
    )

    output_name = f"sub-{subject}_connectivity_{measure}_{band}_circle"
    save_fig(
        fig,
        save_dir / output_name,
        formats=plot_cfg.formats,
        dpi=plot_cfg.dpi,
        bbox_inches=plot_cfg.bbox_inches,
        pad_inches=plot_cfg.pad_inches
    )
    plt.close(fig)
    log_if_present(logger, "info", 
                  f"Saved {measure} {band} connectivity circle ({n_significant} significant edges)")


def plot_connectivity_circle_by_condition(
    features_df: pd.DataFrame,
    events_df: pd.DataFrame,
    info: mne.Info,
    subject: str,
    save_dir: Path,
    logger: logging.Logger,
    config: Any,
    measure: str = "wpli",
    band: str = "alpha",
    n_lines: Optional[int] = None,
    significance_threshold: Optional[float] = None,
) -> None:
    """Plot connectivity circle diagrams comparing two conditions."""
    if features_df is None or features_df.empty or events_df is None:
        log_if_present(logger, "warning", "No feature data for connectivity plot")
        return
    
    if plot_connectivity_circle is None:
        log_if_present(logger, "warning", "mne-connectivity not installed")
        return

    comparison_info = extract_comparison_mask(events_df, config, require_enabled=False)
    if comparison_info is None:
        return
    mask1, mask2, label1, label2 = comparison_info

    n_samples = min(len(features_df), len(mask1))
    if n_samples <= 0:
        return
    if n_samples != len(features_df):
        features_df = features_df.iloc[:n_samples].copy()
    mask1 = np.asarray(mask1[:n_samples], dtype=bool)
    mask2 = np.asarray(mask2[:n_samples], dtype=bool)

    if int(mask1.sum()) == 0 or int(mask2.sum()) == 0:
        return
    
    plot_cfg = get_plot_config(config)
    condition_colors = {
        "c1": plot_cfg.get_color("condition_1"),
        "c2": plot_cfg.get_color("condition_2"),
    }

    columns_tuple, edges_tuple = _parse_connectivity_columns_cached(
        tuple(features_df.columns), measure, band
    )
    columns, edges = _filter_non_self_edges(list(columns_tuple), list(edges_tuple))
    
    if not columns:
        log_if_present(logger, "debug", 
                      f"No channel-pair connectivity columns found for {measure} {band}")
        return
    
    node_names = _extract_unique_nodes(edges)
    n_nodes = len(node_names)
    n_edges = len(edges)
    
    default_top_fraction = float(get_config_value(
        config, "plotting.plots.features.connectivity.circle_top_fraction", 0.1
    ))
    top_fraction = (significance_threshold if significance_threshold is not None 
                   else default_top_fraction)
    
    pooled_connectivity = features_df[columns].mean(axis=0).values
    absolute_connectivity = np.abs(pooled_connectivity)
    threshold = np.percentile(absolute_connectivity, (1 - top_fraction) * 100)
    
    def build_matrix_for_condition(condition_mask: np.ndarray) -> Tuple[np.ndarray, int]:
        """Build connectivity matrix for a specific condition."""
        mean_connectivity = features_df.loc[condition_mask, columns].mean(axis=0).values
        return _build_connectivity_matrix(mean_connectivity, edges, node_names, threshold)
    
    matrix_condition1, n_sig_condition1 = build_matrix_for_condition(mask1)
    matrix_condition2, n_sig_condition2 = build_matrix_for_condition(mask2)
    
    n_trials_condition1 = int(mask1.sum())
    n_trials_condition2 = int(mask2.sum())
    
    colormap, vmin, vmax = _get_connectivity_colormap_and_range(measure)
    if vmin is None:
        vmin, vmax = 0.0, 1.0
        colormap = "viridis"
    
    min_lines_config = int(get_config_value(
        config, "plotting.plots.features.connectivity.circle_min_lines", 20
    ))
    n_lines_to_show = (n_lines if n_lines is not None 
                       else max(min_lines_config, max(n_sig_condition1, n_sig_condition2)))
    
    width_per_circle = float(plot_cfg.plot_type_configs.get("connectivity", {})
                             .get("width_per_circle", 9.0))
    fig, axes = plt.subplots(
        1, 2, figsize=(width_per_circle * 2, width_per_circle), 
        subplot_kw=dict(polar=True)
    )
    
    try:
        plot_connectivity_circle(
            matrix_condition1, node_names, n_lines=n_lines_to_show, ax=axes[0],
            title="", show=False,
            vmin=vmin, vmax=vmax, colorbar=False, colormap=colormap
        )
        axes[0].set_title(
            f"{label1}\n(n={n_trials_condition1} trials, {n_sig_condition1} edges)",
            fontsize=plot_cfg.font.suptitle,
            fontweight="bold",
            color=condition_colors["c1"],
        )
        
        plot_connectivity_circle(
            matrix_condition2, node_names, n_lines=n_lines_to_show, ax=axes[1],
            title="", show=False,
            vmin=vmin, vmax=vmax, colorbar=True, colormap=colormap
        )
        axes[1].set_title(
            f"{label2}\n(n={n_trials_condition2} trials, {n_sig_condition2} edges)",
            fontsize=plot_cfg.font.suptitle,
            fontweight="bold",
            color=condition_colors["c2"],
        )
    except Exception as e:
        log_if_present(logger, "error", f"Failed to plot: {e}")
        plt.close(fig)
        return
    
    title_text = (
        f"{measure.upper()} Connectivity: {band.capitalize()} Band\n"
        f"Subject: {subject} | Top {int(top_fraction*100)}% connections "
        f"(threshold ≥ {threshold:.3f})"
    )
    fig.suptitle(title_text, fontsize=plot_cfg.font.figure_title, 
                 fontweight="bold", y=0.98)
    
    footer_text = (
        f"{n_nodes} nodes | {n_edges} total edges | "
        f"Showing connections ≥ {threshold:.3f}"
    )
    fig.text(0.5, 0.02, footer_text, ha='center', va='bottom', 
             fontsize=plot_cfg.font.large, color='gray')
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.93])
    output_name = f"sub-{subject}_connectivity_{measure}_{band}_circle_by_condition"
    save_fig(
        fig, save_dir / output_name,
        formats=plot_cfg.formats, dpi=plot_cfg.dpi,
        bbox_inches=plot_cfg.bbox_inches, pad_inches=plot_cfg.pad_inches
    )
    plt.close(fig)
    log_if_present(logger, "info", 
                  f"Saved {measure} {band} connectivity circle by condition")


def plot_sliding_connectivity_trajectories(
    conn_df: pd.DataFrame,
    window_indices: List[int],
    window_centers: np.ndarray,
    aligned_events: Optional[pd.DataFrame],
    subject: str,
    plots_dir: Path,
    logger: logging.Logger,
    config: Any,
) -> None:
    """Plot sliding window connectivity trajectories over time."""
    if conn_df is None or conn_df.empty or not window_indices:
        return
    
    plot_cfg = get_plot_config(config)
    mean_traces = []
    labels = []

    for window_idx in window_indices:
        prefix = f"sw{window_idx}corr_all_"
        window_columns = [
            col for col in conn_df.columns 
            if str(col).startswith(prefix) and "__" in str(col)
        ]
        if not window_columns:
            continue
        mean_trace = conn_df[window_columns].apply(
            pd.to_numeric, errors="coerce"
        ).mean(axis=1)
        mean_traces.append(mean_trace)
        labels.append(window_idx)

    if not mean_traces:
        log_if_present(logger, "warning", 
                      "No sliding connectivity columns found for trajectories.")
        return

    trajectory_matrix = np.vstack([np.asarray(trace) for trace in mean_traces])
    fig, ax = plt.subplots(
        figsize=plot_cfg.get_figure_size("sliding", plot_type="connectivity")
    )
    
    mean_all = np.nanmean(trajectory_matrix, axis=1)
    n_finite = np.maximum(1, np.sum(np.isfinite(trajectory_matrix), axis=1))
    sem_all = np.nanstd(trajectory_matrix, axis=1) / np.sqrt(n_finite)
    
    blue_color = plot_cfg.get_color("blue")
    ax.plot(window_centers[:len(mean_all)], mean_all, 
           color=blue_color, label="All trials")
    ax.fill_between(
        window_centers[:len(mean_all)],
        mean_all - sem_all,
        mean_all + sem_all,
        color=blue_color,
        alpha=0.2,
    )

    n_trials = trajectory_matrix.shape[1]
    n_windows = len(window_indices)
    
    if aligned_events is not None:
        comparison_info = extract_comparison_mask(
            aligned_events, config, require_enabled=False
        )
        if comparison_info is not None:
            mask1, mask2, label1, label2 = comparison_info
            n_samples = min(trajectory_matrix.shape[1], len(mask1))
            if n_samples > 0:
                matrix_aligned = trajectory_matrix[:, :n_samples]
                mask1_array = np.asarray(mask1[:n_samples], dtype=bool)
                mask2_array = np.asarray(mask2[:n_samples], dtype=bool)
                
                for condition_mask, condition_label, condition_color in [
                    (mask1_array, label1, plot_cfg.get_color("blue")),
                    (mask2_array, label2, plot_cfg.get_color("red")),
                ]:
                    if int(condition_mask.sum()) == 0:
                        continue
                    
                    condition_data = matrix_aligned[:, condition_mask]
                    n_condition = condition_data.shape[1]
                    if condition_data.size == 0:
                        continue
                    
                    mean_condition = np.nanmean(condition_data, axis=1)
                    n_finite_cond = np.maximum(
                        1, np.sum(np.isfinite(condition_data), axis=1)
                    )
                    sem_condition = np.nanstd(condition_data, axis=1) / np.sqrt(n_finite_cond)
                    
                    ax.plot(window_centers[:len(mean_condition)], mean_condition, 
                           label=f"{condition_label} (n={n_condition})", 
                           color=condition_color)
                    ax.fill_between(
                        window_centers[:len(mean_condition)],
                        mean_condition - sem_condition,
                        mean_condition + sem_condition,
                        color=condition_color,
                        alpha=0.2,
                    )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mean sliding connectivity")
    ax.set_title(f"Sliding connectivity trajectories (sub-{subject})")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False)
    
    footer_text = f"n={n_trials} trials | {n_windows} time windows"
    fig.text(
        0.99, 0.01, footer_text,
        ha='right', va='bottom',
        fontsize=plot_cfg.font.small,
        color='gray', alpha=0.8
    )
    
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    save_fig(
        fig,
        plots_dir / f"sub-{subject}_sliding_connectivity_trajectories",
        formats=plot_cfg.formats,
        dpi=plot_cfg.dpi,
        bbox_inches=plot_cfg.bbox_inches,
        pad_inches=plot_cfg.pad_inches,
    )
    plt.close(fig)
    log_if_present(logger, "info", "Saved sliding connectivity trajectories")


def plot_sliding_degree_heatmap(
    conn_df: pd.DataFrame,
    window_indices: List[int],
    window_centers: np.ndarray,
    subject: str,
    plots_dir: Path,
    logger: logging.Logger,
    config: Any,
) -> None:
    deg_cols = [c for c in conn_df.columns if "corr_all_deg_" in str(c)]
    if not deg_cols:
        return
    channels = sorted({c.split("deg_")[-1] for c in deg_cols if "deg_" in c})
    if not channels:
        return
    data = np.full((len(channels), len(window_indices)), np.nan, dtype=float)
    for win_pos, win in enumerate(window_indices):
        for ch_idx, ch in enumerate(channels):
            col = f"sw{win}corr_all_deg_{ch}"
            if col in conn_df.columns:
                vals = pd.to_numeric(conn_df[col], errors="coerce")
                data[ch_idx, win_pos] = np.nanmean(vals)

    if not np.isfinite(data).any():
        log_if_present(logger, "warning", "Sliding degree heatmap has no finite values.")
        return

    plot_cfg = get_plot_config(config)
    fig, ax = plt.subplots(figsize=plot_cfg.get_figure_size("wide", plot_type="connectivity"))
    vmax = np.nanmax(np.abs(data))
    im = ax.imshow(
        data,
        aspect="auto",
        origin="lower",
        cmap="magma",
        extent=[window_centers[0], window_centers[len(window_indices)-1], -0.5, len(channels)-0.5],
        vmin=0,
        vmax=vmax if np.isfinite(vmax) and vmax > 0 else None,
    )
    ax.set_yticks(range(len(channels)))
    ax.set_yticklabels(channels)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Channel")
    ax.set_title(f"Sliding degree (mean across trials) - sub-{subject}")
    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Mean degree")
    plt.tight_layout()
    save_fig(
        fig,
        plots_dir / f"sub-{subject}_sliding_degree_heatmap",
        formats=plot_cfg.formats,
        dpi=plot_cfg.dpi,
        bbox_inches=plot_cfg.bbox_inches,
        pad_inches=plot_cfg.pad_inches,
    )
    plt.close(fig)
    log_if_present(logger, "info", "Saved sliding degree heatmap")


def plot_edge_significance_circle_from_stats(
    stats_df: pd.DataFrame,
    prefix: str,
    save_dir: Path,
    logger: logging.Logger,
    config: Any,
    sig_edges: Optional[set] = None,
) -> None:
    if stats_df is None or stats_df.empty or "edge" not in stats_df.columns:
        return

    edge_vals = {}
    for _, row in stats_df.iterrows():
        edge_str = str(row.get("edge"))
        if "__" not in edge_str:
            continue
        ch1, ch2 = edge_str.split("__", 1)
        val = float(row.get("effect", row.get("r", 0.0)))
        edge_vals[(ch1, ch2)] = val

    if not edge_vals:
        return

    mat, nodes = build_matrix_from_edges(edge_vals)
    plot_cfg = get_plot_config(config)
    if plot_connectivity_circle is None:
        log_if_present(logger, "warning", "mne-connectivity not installed; skipping significance circle")
        return

    fig_size = plot_cfg.get_figure_size("standard", plot_type="connectivity")
    fig, ax = plt.subplots(figsize=fig_size, subplot_kw=dict(polar=True))
    vmax = np.nanmax(np.abs(mat)) if np.isfinite(mat).any() else 1.0
    plot_connectivity_circle(
        mat,
        nodes,
        n_lines=None,
        title=f"{prefix} significant edges",
        ax=ax,
        show=False,
        vmin=-vmax,
        vmax=vmax,
        colormap="RdBu_r",
        node_angles=None,
        node_colors=None,
        colorbar=True,
    )

    if sig_edges:
        highlight = []
        for e in sig_edges:
            if "__" in str(e):
                ch1, ch2 = str(e).split("__", 1)
                if ch1 in nodes and ch2 in nodes:
                    highlight.append((ch1, ch2))
        if highlight:
            plot_connectivity_circle(
                mat,
                nodes,
                n_lines=None,
                title=None,
                ax=ax,
                show=False,
                vmin=-vmax,
                vmax=vmax,
                colormap="RdBu_r",
                node_angles=None,
                node_colors=None,
                colorbar=False,
                linewidth=3.0,
                edge_threshold=None,
                facecolor="none",
                edge_colors="lime",
                indices=highlight,
            )

    ensure_dir(save_dir)
    save_fig(fig, save_dir / f"{prefix}_edge_significance")
    plt.close(fig)
    log_if_present(logger, "info", f"Saved edge significance circle for {prefix}")


def plot_graph_metric_distributions(
    connectivity_df: pd.DataFrame,
    events_df: Optional[pd.DataFrame],
    save_dir: Path,
    logger: logging.Logger,
    config: Any,
) -> None:
    if connectivity_df is None or connectivity_df.empty:
        return
    ensure_dir(save_dir)
    plot_cfg = get_plot_config(config)

    metric_cols = [c for c in connectivity_df.columns if any(k in c for k in ["geff", "clust", "pc", "smallworld", "modularity"])]
    if not metric_cols:
        return

    n_cols = min(3, len(metric_cols))
    n_rows = (len(metric_cols) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows), squeeze=False)
    axes = axes.flatten()
    
    for idx, col in enumerate(metric_cols):
        ax = axes[idx]
        vals = pd.to_numeric(connectivity_df[col], errors="coerce").dropna().values
        
        if len(vals) > 0:
            parts = ax.violinplot([vals], positions=[0], showmedians=True, widths=0.6)
            parts["bodies"][0].set_facecolor("#3b528b")
            parts["bodies"][0].set_alpha(0.6)
            
            jitter = np.random.uniform(-0.1, 0.1, len(vals))
            ax.scatter(jitter, vals, c="#3b528b", alpha=0.3, s=10)
            
            ax.axhline(np.mean(vals), color="black", linestyle="--", linewidth=1, alpha=0.7)
        
        ax.set_xticks([0])
        ax.set_xticklabels([col.split("_")[-1]])
        ax.set_ylabel(col.split("_")[-1])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    
    for idx in range(len(metric_cols), len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle("Graph Metric Distributions", fontsize=plot_cfg.font.figure_title, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, save_dir / "connectivity_graph_metrics_violin")
    plt.close(fig)


def plot_graph_metrics_bar(
    features_df: pd.DataFrame,
    save_dir: Path,
    measure: str = "wpli",
    band: str = "alpha",
    config: Any = None,
) -> None:
    if features_df is None or features_df.empty:
        return

    metric_keys = ["geff", "clust", "pc", "smallworld"]
    columns = [f"{measure}_{band}_{k}" for k in metric_keys]
    available = [c for c in columns if c in features_df.columns]
    if len(available) == 0:
        return

    plot_cfg = get_plot_config(config)
    means = [np.nanmean(features_df[c]) for c in available]
    sems = [stats.sem(pd.to_numeric(features_df[c], errors="coerce"), nan_policy="omit") for c in available]

    fig, ax = plt.subplots(figsize=plot_cfg.get_figure_size("wide", plot_type="connectivity"))
    x = np.arange(len(available))
    ax.bar(x, means, yerr=sems, color=plot_cfg.style.colors.gray, alpha=plot_cfg.style.bar.alpha,
           width=plot_cfg.style.bar.width, capsize=plot_cfg.style.errorbar_capsize)
    ax.set_xticks(x)
    labels = [col.split("_")[-1] for col in available]
    ax.set_xticklabels(labels, rotation=0)
    ax.set_ylabel("Value")
    ax.set_title(f"{measure.upper()} {band}: Graph metrics")
    ensure_dir(save_dir)
    out = save_dir / f"connectivity_{measure}_{band}_graph_metrics"
    save_fig(fig, out, formats=plot_cfg.formats)
    plt.close(fig)


def plot_rsn_radar(
    features_df: pd.DataFrame,
    save_dir: Path,
    measure: str = "wpli",
    band: str = "alpha",
    config: Any = None,
) -> None:
    if features_df is None or features_df.empty:
        return

    prefix = f"{measure}_{band}_rsn_"
    strength_cols = [c for c in features_df.columns if c.startswith(prefix) and c.endswith("_strength")]
    if not strength_cols:
        return

    plot_cfg = get_plot_config(config)
    rsn_names = [c[len(prefix):-len("_strength")] for c in strength_cols]
    values = [np.nanmean(features_df[c]) for c in strength_cols]
    if len(rsn_names) == 0:
        return

    angles = np.linspace(0, 2 * np.pi, len(rsn_names), endpoint=False)
    values_cycle = values + [values[0]]
    angles_cycle = list(angles) + [angles[0]]

    fig, ax = plt.subplots(subplot_kw=dict(polar=True), figsize=plot_cfg.get_figure_size("square", plot_type="connectivity"))
    ax.plot(angles_cycle, values_cycle, color=plot_cfg.style.colors.gray, linewidth=plot_cfg.style.line.width_thick)
    ax.fill(angles_cycle, values_cycle, color=plot_cfg.style.colors.gray, alpha=0.3)
    ax.set_xticks(angles)
    ax.set_xticklabels(rsn_names)
    ax.set_title(f"{measure.upper()} {band}: RSN strength")

    ensure_dir(save_dir)
    out = save_dir / f"connectivity_{measure}_{band}_rsn_radar"
    save_fig(fig, out, formats=plot_cfg.formats)
    plt.close(fig)
    log_if_present(get_logger(__name__), "info", f"Saved RSN radar for {measure} {band}")


def _plot_column_comparison_connectivity(
    features_df: pd.DataFrame,
    events_df: pd.DataFrame,
    measures: List[str],
    bands: List[str],
    roi_names: List[str],
    roi_definitions: Dict[str, Any],
    subject: str,
    save_dir: Path,
    config: Any,
    logger: logging.Logger,
    stats_dir: Optional[Path],
) -> None:
    """Plot unpaired column comparison for connectivity measures."""
    from eeg_pipeline.utils.analysis.events import extract_comparison_mask
    from eeg_pipeline.plotting.features.utils import compute_or_load_column_stats, get_band_color
    
    comparison_info = extract_comparison_mask(events_df, config)
    if not comparison_info:
        if logger:
            logger.debug("Column comparison requested but config incomplete")
        return
    
    mask1, mask2, label1, label2 = comparison_info
    segment = get_config_value(config, "plotting.comparisons.comparison_segment", "active")
    
    plot_cfg = get_plot_config(config)
    segment_colors = {"v1": "#5a7d9a", "v2": "#c44e52"}
    band_colors = {band: get_band_color(band, config) for band in bands}
    n_bands = len(bands)
    n_trials = len(features_df)
    
    for roi_name in roi_names:
        for measure in measures:
            cell_data = {}
            for band_idx, band in enumerate(bands):
                columns, _, _ = parse_connectivity_columns(
                    list(features_df.columns), measure, band, segment=segment
                )
                columns = _filter_connectivity_columns_by_roi(
                    columns, roi_name, roi_definitions, list(features_df.columns)
                )
                
                if not columns:
                    cell_data[band_idx] = None
                    continue
                
                value_series = features_df[columns].apply(
                    pd.to_numeric, errors="coerce"
                ).mean(axis=1)
                values1 = value_series[mask1].dropna().values
                values2 = value_series[mask2].dropna().values
                
                cell_data[band_idx] = {"v1": values1, "v2": values2}
            
            qvalues, n_significant, use_precomputed = compute_or_load_column_stats(
                stats_dir=stats_dir,
                feature_type="connectivity",
                feature_keys=bands,
                cell_data=cell_data,
                config=config,
                logger=logger,
            )
            
            fig, axes = plt.subplots(1, n_bands, figsize=(3 * n_bands, 5), squeeze=False)
            
            for band_idx, band in enumerate(bands):
                ax = axes.flatten()[band_idx]
                data = cell_data.get(band_idx)
                
                if data is None or len(data.get("v1", [])) == 0 or len(data.get("v2", [])) == 0:
                    ax.text(0.5, 0.5, "No data", ha="center", va="center",
                           transform=ax.transAxes, fontsize=plot_cfg.font.title, color="gray")
                    ax.set_xticks([])
                    continue
                
                values1, values2 = data["v1"], data["v2"]
                
                boxplot = ax.boxplot([values1, values2], positions=[0, 1], 
                                    widths=0.4, patch_artist=True)
                boxplot["boxes"][0].set_facecolor(segment_colors["v1"])
                boxplot["boxes"][0].set_alpha(0.6)
                boxplot["boxes"][1].set_facecolor(segment_colors["v2"])
                boxplot["boxes"][1].set_alpha(0.6)
                
                jitter_range = 0.08
                ax.scatter(np.random.uniform(-jitter_range, jitter_range, len(values1)), 
                          values1, c=segment_colors["v1"], alpha=0.3, s=6)
                ax.scatter(1 + np.random.uniform(-jitter_range, jitter_range, len(values2)), 
                          values2, c=segment_colors["v2"], alpha=0.3, s=6)
                
                all_values = np.concatenate([values1, values2])
                ymin, ymax = np.nanmin(all_values), np.nanmax(all_values)
                yrange = ymax - ymin if ymax > ymin else 0.1
                ax.set_ylim(ymin - 0.1 * yrange, ymax + 0.3 * yrange)
                
                if band_idx in qvalues:
                    _, qvalue, effect_size, is_significant = qvalues[band_idx]
                    sig_marker = "†" if is_significant else ""
                    sig_color = "#d62728" if is_significant else "#333333"
                    annotation_text = f"q={qvalue:.3f}{sig_marker}\nd={effect_size:.2f}"
                    ax.annotate(annotation_text, xy=(0.5, ymax + 0.05 * yrange),
                               ha="center", fontsize=plot_cfg.font.medium, color=sig_color,
                               fontweight="bold" if is_significant else "normal")
                
                ax.set_xticks([0, 1])
                ax.set_xticklabels([label1, label2], fontsize=9)
                ax.set_title(band.capitalize(), fontweight="bold", 
                           color=band_colors.get(band, "gray"))
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
            
            n_tests = len(qvalues)
            roi_display = roi_name.replace("_", " ").title() if roi_name != "all" else "All Edges"
            stats_source = "pre-computed" if use_precomputed else "Mann-Whitney U"
            title_text = (
                f"Connectivity ({measure.upper()}): {label1} vs {label2} (Column Comparison)\n"
                f"Subject: {subject} | ROI: {roi_display} | N: {n_trials} trials | "
                f"{stats_source} | FDR: {n_significant}/{n_tests} significant (†=q<0.05)"
            )
            fig.suptitle(title_text, fontsize=plot_cfg.font.suptitle, 
                        fontweight="bold", y=1.02)
            
            plt.tight_layout()
            
            measure_safe = measure.lower()
            roi_safe = roi_name.replace(" ", "_").lower() if roi_name != "all" else ""
            suffix = f"_roi-{roi_safe}" if roi_safe else ""
            filename = f"sub-{subject}_connectivity_{measure_safe}_by_condition{suffix}_column"
            
            save_fig(fig, save_dir / filename, formats=plot_cfg.formats, dpi=plot_cfg.dpi,
                     bbox_inches=plot_cfg.bbox_inches, pad_inches=plot_cfg.pad_inches)
            plt.close(fig)
    
    log_if_present(logger, "info", 
                  f"Saved connectivity column comparison plots for "
                  f"{len(measures)} measures × {len(roi_names)} ROIs")


def plot_connectivity_by_condition(
    features_df: pd.DataFrame,
    events_df: pd.DataFrame,
    subject: str,
    save_dir: Path,
    logger: logging.Logger,
    config: Any,
    stats_dir: Optional[Path] = None,
) -> None:
    """Connectivity comparison by Measure × Band.
    
    For window comparisons (paired): Uses the unified plot_paired_comparison helper.
    For column comparisons (unpaired): Uses Mann-Whitney U test with consistent styling.
    Creates one figure per connectivity measure per ROI.
    
    If stats_dir is provided, uses pre-computed statistics from the behavior pipeline.
    """
    from eeg_pipeline.plotting.features.roi import get_roi_definitions
    
    if features_df is None or features_df.empty:
        return

    compare_windows = get_config_value(config, "plotting.comparisons.compare_windows", True)
    compare_columns = get_config_value(config, "plotting.comparisons.compare_columns", False)
    
    segments = _detect_segments_from_data(features_df, config, logger)
    measures = get_config_value(
        config, "plotting.plots.features.connectivity.measures", 
        ["aec", "wpli", "pli", "plv", "coherence"]
    )
    bands = list(get_frequency_band_names(config) or ['theta', 'alpha', 'beta', 'gamma'])
    
    roi_definitions = get_roi_definitions(config)
    roi_names = _get_roi_names_for_comparison(config, roi_definitions)
    
    if logger:
        logger.info(
            f"Connectivity comparison: segments={segments}, ROIs={roi_names}, "
            f"measures={measures}, compare_windows={compare_windows}, "
            f"compare_columns={compare_columns}"
        )
    
    if compare_windows and segments and len(segments) >= 2:
        _plot_window_comparison_connectivity(
            features_df, segments, measures, bands, roi_names, roi_definitions,
            subject, save_dir, config, logger, stats_dir
        )

    if compare_columns:
        _plot_column_comparison_connectivity(
            features_df, events_df, measures, bands, roi_names, roi_definitions,
            subject, save_dir, config, logger, stats_dir
        )


def plot_connectivity_band_segment_condition(
    features_df: pd.DataFrame,
    events_df: pd.DataFrame,
    subject: str,
    save_dir: Path,
    logger: logging.Logger,
    config: Any,
) -> None:
    """Connectivity comparison by Measure × Band.

    
    Delegates to plot_connectivity_by_condition for unified implementation.
    """
    plot_connectivity_by_condition(
        features_df=features_df,
        events_df=events_df,
        subject=subject,
        save_dir=save_dir,
        logger=logger,
        config=config,
    )






def plot_connectivity_heatmap(
    features_df: pd.DataFrame,
    info: mne.Info,
    subject: str,
    save_dir: Path,
    logger: logging.Logger,
    config: Any,
    measure: str = "wpli",
    band: str = "alpha",
    events_df: Optional[pd.DataFrame] = None,
) -> None:
    """Plot connectivity heatmap for a given measure and band."""
    columns_all, edges_all, _ = parse_connectivity_columns(
        list(features_df.columns), measure, band
    )
    columns, edges = _filter_non_self_edges(columns_all, edges_all)
    
    if not columns:
        log_if_present(logger, "debug", 
                      f"No channel-pair connectivity columns for {measure} {band} heatmap")
        return

    channel_order = _get_channel_order(edges, info)
    if not channel_order:
        log_if_present(logger, "debug", 
                      f"No channel names found for {measure} {band}")
        return
        
    adjacency_matrix = build_adjacency_from_edges(
        features_df, columns, channel_order, edges=edges
    )
    if not np.any(np.isfinite(adjacency_matrix)):
        return

    significant_edges = compute_significant_edges(features_df, columns, events_df, config)
    plot_cfg = get_plot_config(config)
    vmax = (float(np.nanmax(np.abs(adjacency_matrix))) 
            if np.any(np.isfinite(adjacency_matrix)) else 1.0)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(adjacency_matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(channel_order)))
    ax.set_yticks(range(len(channel_order)))
    ax.set_xticklabels(channel_order, rotation=90, fontsize=plot_cfg.font.annotation)
    ax.set_yticklabels(channel_order, fontsize=plot_cfg.font.annotation)
    ax.set_title(f"{measure.upper()} {band.capitalize()} mean connectivity (sub-{subject})")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Connectivity")

    if significant_edges:
        significant_mask = np.zeros_like(adjacency_matrix, dtype=bool)
        edge_to_indices = {
            (ch1, ch2): (channel_order.index(ch1), channel_order.index(ch2))
            for ch1, ch2 in edges 
            if ch1 in channel_order and ch2 in channel_order
        }
        for col in significant_edges:
            if col in columns:
                col_index = columns.index(col)
                edge = edges[col_index]
                if edge in edge_to_indices:
                    i, j = edge_to_indices[edge]
                    significant_mask[i, j] = significant_mask[j, i] = True
        if np.any(significant_mask):
            ax.contour(significant_mask, colors="k", levels=[0.5], linewidths=0.5)

    ensure_dir(save_dir)
    output_name = save_dir / f"sub-{subject}_connectivity_heatmap_{measure}_{band}"
    save_fig(
        fig,
        output_name,
        formats=plot_cfg.formats,
        dpi=plot_cfg.dpi,
        bbox_inches=plot_cfg.bbox_inches,
        pad_inches=plot_cfg.pad_inches,
    )
    plt.close(fig)
    log_if_present(logger, "info", f"Saved connectivity heatmap for {measure} {band}")


def plot_connectivity_network(
    features_df: pd.DataFrame,
    info: mne.Info,
    subject: str,
    save_dir: Path,
    logger: logging.Logger,
    config: Any,
    measure: str = "wpli",
    band: str = "alpha",
    events_df: Optional[pd.DataFrame] = None,
) -> None:
    """Plot connectivity network graph for a given measure and band."""
    columns_all, edges_all, _ = parse_connectivity_columns(
        list(features_df.columns), measure, band
    )
    columns, edges = _filter_non_self_edges(columns_all, edges_all)
    
    if not columns:
        log_if_present(logger, "debug", 
                      f"No channel-pair connectivity columns for {measure} {band} network")
        return

    channel_order = _get_channel_order(edges, info)
    if not channel_order:
        log_if_present(logger, "debug", 
                      f"No channel names found for {measure} {band}")
        return
        
    adjacency_matrix = build_adjacency_from_edges(
        features_df, columns, channel_order, edges=edges
    )
    if not np.any(np.isfinite(adjacency_matrix)):
        return

    significant_edges = compute_significant_edges(features_df, columns, events_df, config)
    significant_set = significant_edges if isinstance(significant_edges, set) else set()

    plot_cfg = get_plot_config(config)
    graph = nx.Graph()
    for i, channel_i in enumerate(channel_order):
        graph.add_node(channel_i)
        for j in range(i + 1, len(channel_order)):
            weight = adjacency_matrix[i, j]
            if np.isfinite(weight) and np.abs(weight) > 0:
                graph.add_edge(channel_i, channel_order[j], weight=float(weight))

    if graph.number_of_edges() == 0:
        return

    node_positions = nx.spring_layout(graph, seed=42)
    edge_weights = np.array(
        [data["weight"] for _, _, data in graph.edges(data=True)], dtype=float
    )
    vmax = float(np.nanmax(np.abs(edge_weights))) if edge_weights.size else 1.0

    fig, ax = plt.subplots(figsize=(6, 5))
    graph_edges = list(graph.edges())
    edge_colors = [graph[u][v]["weight"] for u, v in graph_edges]
    node_color = plot_cfg.get_color("network_node")
    
    nx.draw_networkx_nodes(graph, node_positions, node_size=120, 
                          node_color=node_color, alpha=0.8, ax=ax)
    nx.draw_networkx_edges(
        graph,
        node_positions,
        edgelist=graph_edges,
        edge_color=edge_colors,
        edge_cmap=plt.cm.RdBu_r,
        edge_vmin=-vmax,
        edge_vmax=vmax,
        width=2.0,
        alpha=0.7,
        ax=ax,
    )
    nx.draw_networkx_labels(graph, node_positions, font_size=7, 
                           font_weight="bold", ax=ax)

    if significant_set:
        column_to_edge = {col: edges[i] for i, col in enumerate(columns)}
        highlight_edges = []
        for col in significant_set:
            if col in column_to_edge:
                ch1, ch2 = column_to_edge[col]
                if graph.has_edge(ch1, ch2):
                    highlight_edges.append((ch1, ch2))
                elif graph.has_edge(ch2, ch1):
                    highlight_edges.append((ch2, ch1))
        if highlight_edges:
            nx.draw_networkx_edges(
                graph, node_positions, edgelist=highlight_edges, 
                edge_color="lime", width=3.0, alpha=0.9, ax=ax
            )

    scalar_mappable = plt.cm.ScalarMappable(
        cmap="RdBu_r", norm=plt.Normalize(vmin=-vmax, vmax=vmax)
    )
    scalar_mappable.set_array([])
    cbar = plt.colorbar(scalar_mappable, ax=ax)
    cbar.set_label("Connectivity")
    ax.set_title(
        f"Connectivity network ({measure.upper()} {band.capitalize()}, sub-{subject})"
    )
    ax.axis("off")

    ensure_dir(save_dir)
    output_name = save_dir / f"sub-{subject}_connectivity_network_{measure}_{band}"
    save_fig(
        fig,
        output_name,
        formats=plot_cfg.formats,
        dpi=plot_cfg.dpi,
        bbox_inches=plot_cfg.bbox_inches,
        pad_inches=plot_cfg.pad_inches,
    )
    plt.close(fig)
    log_if_present(logger, "info", f"Saved connectivity network for {measure} {band}")


def plot_sliding_state_centroids(
    centroids: np.ndarray,
    edge_pairs: List[Tuple[str, str]],
    ch_names: Optional[List[str]],
    save_dir: Path,
    logger: logging.Logger,
    config: Any,
) -> None:
    if centroids is None or centroids.size == 0 or not edge_pairs:
        return
    ensure_dir(save_dir)
    plot_cfg = get_plot_config(config)
    n_states = centroids.shape[0]
    nodes = sorted({n for pair in edge_pairs for n in pair})
    node_idx = {n: i for i, n in enumerate(nodes)}

    if plot_connectivity_circle is None:
        log_if_present(logger, "warning", "mne-connectivity not installed; skipping centroid circles")
        return

    for s_idx in range(n_states):
        adj = np.zeros((len(nodes), len(nodes)), dtype=float)
        for val, (u, v) in zip(centroids[s_idx], edge_pairs):
            if u in node_idx and v in node_idx:
                i, j = node_idx[u], node_idx[v]
                adj[i, j] = val
                adj[j, i] = val
        fig_size = plot_cfg.get_figure_size("standard", plot_type="connectivity")
        fig, ax = plt.subplots(figsize=fig_size, subplot_kw=dict(polar=True))
        vmax = np.nanmax(np.abs(adj)) if np.isfinite(adj).any() else 1.0
        plot_connectivity_circle(
            adj,
            nodes,
            n_lines=None,
            node_angles=None,
            node_colors=None,
            title=f"Sliding state {s_idx}",
            ax=ax,
            show=False,
            vmin=-vmax,
            vmax=vmax,
            colorbar=True,
            colormap="RdBu_r",
        )
        out_path = save_dir / f"sliding_state_centroid_{s_idx}"
        save_fig(
            fig,
            out_path,
            formats=plot_cfg.formats,
            dpi=plot_cfg.dpi,
            bbox_inches=plot_cfg.bbox_inches,
            pad_inches=plot_cfg.pad_inches,
        )
        plt.close(fig)
        log_if_present(logger, "info", f"Saved sliding centroid for state {s_idx}")


def plot_sliding_state_sequences(
    state_matrix: np.ndarray,
    window_indices: List[int],
    save_dir: Path,
    logger: logging.Logger,
    config: Any,
) -> None:
    if state_matrix is None or state_matrix.size == 0:
        return
    ensure_dir(save_dir)
    plot_cfg = get_plot_config(config)
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(state_matrix, aspect="auto", interpolation="nearest", cmap="tab20")
    ax.set_xlabel("Sliding window")
    ax.set_ylabel("Trial")
    ax.set_xticks(range(len(window_indices)))
    ax.set_xticklabels(window_indices)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("State")
    fig.tight_layout()
    save_fig(fig, save_dir / "sliding_state_sequences", formats=plot_cfg.formats, dpi=plot_cfg.dpi)
    plt.close(fig)
    log_if_present(logger, "info", "Saved sliding state sequence plot")


def plot_sliding_state_occupancy_boxplot(
    occupancy: np.ndarray,
    events_df: Optional[pd.DataFrame],
    save_dir: Path,
    logger: logging.Logger,
    config: Any,
) -> None:
    if occupancy is None or occupancy.size == 0:
        return
    ensure_dir(save_dir)
    plot_cfg = get_plot_config(config)
    n_states = occupancy.shape[1]

    pain_col = None
    if events_df is not None:
        pain_col = find_column_in_events(events_df, "pain") or find_pain_column_in_events(events_df)

    fig, ax = plt.subplots(figsize=(6, 4))
    data = [occupancy[:, s] for s in range(n_states)]
    ax.boxplot(data, labels=[f"S{s}" for s in range(n_states)], patch_artist=True)
    ax.set_ylabel("Occupancy fraction")
    ax.set_title("Sliding-state occupancy (all trials)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    save_fig(fig, save_dir / "sliding_state_occupancy", formats=plot_cfg.formats, dpi=plot_cfg.dpi)
    plt.close(fig)

    if pain_col and pain_col in events_df.columns:
        fig, ax = plt.subplots(figsize=(7, 4))
        pain_vals = pd.to_numeric(events_df[pain_col], errors="coerce")
        groups = ["nonpain", "pain"]
        positions = []
        vals = []
        for s in range(n_states):
            for g_idx, g_val in enumerate([0, 1]):
                mask = (pain_vals == g_val) & np.isfinite(occupancy[:, s])
                if mask.sum() == 0:
                    continue
                vals.append(occupancy[:, s][mask])
                positions.append(s + (0.15 if g_idx == 1 else -0.15))
        if vals:
            ax.boxplot(vals, positions=positions, widths=0.25, patch_artist=True)
            ax.set_xticks(range(n_states))
            ax.set_xticklabels([f"S{s}" for s in range(n_states)])
            ax.set_ylabel("Occupancy fraction")
            ax.set_title("Occupancy by pain group")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            save_fig(fig, save_dir / "sliding_state_occupancy_by_pain", formats=plot_cfg.formats, dpi=plot_cfg.dpi)
        plt.close(fig)


def plot_sliding_state_occupancy_ribbons(
    occupancy_mean: np.ndarray,
    occupancy_sem: np.ndarray,
    window_centers: np.ndarray,
    save_dir: Path,
    logger: logging.Logger,
    config: Any,
    state_labels: Optional[List[str]] = None,
) -> None:
    if occupancy_mean is None or occupancy_mean.size == 0 or window_centers is None or window_centers.size == 0:
        log_if_present(logger, "warning", "No occupancy data for ribbons; skipping plot")
        return
    ensure_dir(save_dir)
    plot_cfg = get_plot_config(config)
    n_states, _ = occupancy_mean.shape
    state_labels = state_labels or [f"S{idx}" for idx in range(n_states)]

    fig, ax = plt.subplots(figsize=plot_cfg.get_figure_size("sliding", plot_type="connectivity"))
    colors = plt.cm.get_cmap("tab10", n_states)
    for s_idx in range(n_states):
        mean_vals = occupancy_mean[s_idx, :]
        sem_vals = occupancy_sem[s_idx, :] if occupancy_sem is not None else None
        ax.plot(window_centers, mean_vals, label=state_labels[s_idx], color=colors(s_idx))
        if sem_vals is not None and sem_vals.size == mean_vals.size:
            ax.fill_between(
                window_centers,
                mean_vals - sem_vals,
                mean_vals + sem_vals,
                color=colors(s_idx),
                alpha=0.2,
            )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Occupancy (fraction)")
    ax.set_title("Sliding-state occupancy trajectories")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", frameon=False)

    save_fig(
        fig,
        save_dir / "sliding_state_occupancy_ribbons",
        formats=plot_cfg.formats,
        dpi=plot_cfg.dpi,
        bbox_inches=plot_cfg.bbox_inches,
        pad_inches=plot_cfg.pad_inches,
    )
    plt.close(fig)
    log_if_present(logger, "info", "Saved sliding state occupancy ribbons")


def plot_sliding_state_lagged_correlation_surfaces(
    window_centers: np.ndarray,
    corr_r: np.ndarray,
    corr_p: np.ndarray,
    save_dir: Path,
    logger: logging.Logger,
    config: Any,
    target_label: str = "VAS",
    state_labels: Optional[List[str]] = None,
) -> None:
    if corr_r is None or corr_r.size == 0 or window_centers is None or window_centers.size == 0:
        log_if_present(logger, "warning", "No lagged correlation data; skipping plot")
        return

    ensure_dir(save_dir)
    plot_cfg = get_plot_config(config)
    n_states, _ = corr_r.shape
    state_labels = state_labels or [f"S{idx}" for idx in range(n_states)]
    vmax = float(np.nanmax(np.abs(corr_r))) if np.isfinite(corr_r).any() else 1.0
    vmax = vmax if vmax > 0 else 1.0
    alpha = get_fdr_alpha(config)

    fig, ax = plt.subplots(figsize=plot_cfg.get_figure_size("sliding", plot_type="connectivity"))
    im = ax.imshow(
        corr_r,
        aspect="auto",
        origin="lower",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        extent=[window_centers[0], window_centers[-1], -0.5, n_states - 0.5],
    )
    ax.set_yticks(range(n_states))
    ax.set_yticklabels(state_labels)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("State")
    ax.set_title(f"Sliding-state correlation vs {target_label}")

    if corr_p is not None and corr_p.shape == corr_r.shape:
        sig_mask = (corr_p < alpha) & np.isfinite(corr_p)
        if np.any(sig_mask):
            y_idx, x_idx = np.where(sig_mask)
            ax.scatter(window_centers[x_idx], y_idx, marker="o", color="k", s=12, linewidths=0.5)

    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Spearman r")

    save_fig(
        fig,
        save_dir / f"sliding_state_corr_surface_{target_label.lower()}",
        formats=plot_cfg.formats,
        dpi=plot_cfg.dpi,
        bbox_inches=plot_cfg.bbox_inches,
        pad_inches=plot_cfg.pad_inches,
    )
    plt.close(fig)
    log_if_present(logger, "info", f"Saved sliding state correlation surface for {target_label}")
