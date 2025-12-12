import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import re
import networkx as nx
from scipy import stats
from scipy.stats import spearmanr, mannwhitneyu
try:
    from mne_connectivity.viz import plot_connectivity_circle
except ImportError:
    plot_connectivity_circle = None

from eeg_pipeline.utils.io.paths import ensure_dir
from eeg_pipeline.utils.io.logging import get_logger
from eeg_pipeline.utils.io.plotting import log_if_present, save_fig, get_band_color
from eeg_pipeline.utils.io.columns import find_column_in_events, get_column_from_config, find_pain_column_in_events
from eeg_pipeline.utils.config.loader import get_config_value, get_frequency_band_names
from eeg_pipeline.utils.analysis.events import extract_pain_mask
from ..config import get_plot_config
from eeg_pipeline.utils.analysis.connectivity import (
    build_adjacency_from_edges,
    build_matrix_from_edges,
    compute_significance_mask,
    parse_connectivity_columns,
)


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
    if features_df is None or features_df.empty:
        log_if_present(logger, "warning", "No feature data for connectivity plot")
        return

    plot_cfg = get_plot_config(config)
    condition_colors = {
        "nonpain": plot_cfg.get_color("nonpain"),
        "pain": plot_cfg.get_color("pain"),
    }
    
    cols, edges, _ = parse_connectivity_columns(features_df.columns, measure, band)
    
    if not cols:
        log_if_present(logger, "warning", f"No connectivity columns found for {measure} {band}")
        return
        
    n_trials = len(features_df)
    mean_conn = features_df[cols].mean(axis=0).values
    
    cfg_thresh = float(get_config_value(config, "plotting.plots.features.connectivity.circle_top_fraction", 0.1))
    top_fraction = significance_threshold if significance_threshold is not None else cfg_thresh
    abs_conn = np.abs(mean_conn)
    threshold = np.percentile(abs_conn, (1 - top_fraction) * 100)
    n_significant = np.sum(abs_conn >= threshold)
    
    node_names = sorted(list(set([ch for edge in edges for ch in edge])))
    n_nodes = len(node_names)
    n_edges = len(edges)
    node_indices = {name: i for i, name in enumerate(node_names)}
    
    con_matrix = np.zeros((n_nodes, n_nodes))
    
    # Only include connections above threshold
    for val, (ch1, ch2) in zip(mean_conn, edges):
        if abs(val) >= threshold and ch1 in node_indices and ch2 in node_indices:
            idx1 = node_indices[ch1]
            idx2 = node_indices[ch2]
            con_matrix[idx1, idx2] = val
            con_matrix[idx2, idx1] = val
            
    if plot_connectivity_circle is None:
        log_if_present(logger, "warning", "mne-connectivity not installed; cannot plot connectivity circle")
        return

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    vmin, vmax = None, None
    colormap = "RdBu"
    measure_lower = measure.lower()
    if "wpli" in measure_lower or "pli" in measure_lower or "coherence" in measure_lower:
        vmin = 0.0
        vmax = 1.0
        colormap = "viridis"
    
    # Calculate number of lines to show
    min_lines = int(get_config_value(config, "plotting.plots.features.connectivity.circle_min_lines", 20))
    if n_lines is None:
        n_lines = max(min_lines, n_significant)
    
    try:
        plot_connectivity_circle(
            con_matrix,
            node_names,
            n_lines=n_lines,
            node_angles=None,
            node_colors=None,
            title="",  # We'll add custom title
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

    # Detailed title
    title = (f"{measure.upper()} Connectivity: {band.capitalize()} Band\n"
             f"Subject: {subject} | Top {int(top_fraction*100)}% connections (threshold ≥ {threshold:.3f})")
    fig.suptitle(title, fontsize=plot_cfg.font.figure_title, fontweight="bold", y=0.98)
    
    # Footer annotation
    footer_text = (f"n = {n_trials} trials | {n_nodes} nodes | "
                   f"{n_significant}/{n_edges} significant edges ({n_significant/n_edges*100:.1f}%)")
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
    log_if_present(logger, "info", f"Saved {measure} {band} connectivity circle ({n_significant} significant edges)")


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
    if features_df is None or features_df.empty or events_df is None:
        log_if_present(logger, "warning", "No feature data for connectivity plot")
        return
    
    if plot_connectivity_circle is None:
        log_if_present(logger, "warning", "mne-connectivity not installed")
        return

    pain_mask = extract_pain_mask(events_df, config)
    if pain_mask is None:
        return
    
    plot_cfg = get_plot_config(config)
    condition_colors = {
        "nonpain": plot_cfg.get_color("nonpain"),
        "pain": plot_cfg.get_color("pain"),
    }
    
    cols, edges, _ = parse_connectivity_columns(features_df.columns, measure, band)
    
    if not cols:
        log_if_present(logger, "warning", f"No connectivity columns found for {measure} {band}")
        return
    
    node_names = sorted(list(set([ch for edge in edges for ch in edge])))
    n_nodes = len(node_names)
    n_edges = len(edges)
    node_indices = {name: i for i, name in enumerate(node_names)}
    
    cfg_thresh = float(get_config_value(config, "plotting.plots.features.connectivity.circle_top_fraction", 0.1))
    top_fraction = significance_threshold if significance_threshold is not None else cfg_thresh
    # Calculate global threshold from pooled data
    all_conn = features_df[cols].mean(axis=0).values
    abs_conn = np.abs(all_conn)
    threshold = np.percentile(abs_conn, (1 - top_fraction) * 100)
    
    def build_matrix_thresholded(mask):
        mean_conn = features_df.loc[mask, cols].mean(axis=0).values
        mat = np.zeros((n_nodes, n_nodes))
        n_sig = 0
        for val, (ch1, ch2) in zip(mean_conn, edges):
            if abs(val) >= threshold and ch1 in node_indices and ch2 in node_indices:
                mat[node_indices[ch1], node_indices[ch2]] = val
                mat[node_indices[ch2], node_indices[ch1]] = val
                n_sig += 1
        return mat, n_sig
    
    matrix_nonpain, n_sig_nonpain = build_matrix_thresholded(~pain_mask)
    matrix_pain, n_sig_pain = build_matrix_thresholded(pain_mask)
    
    n_nonpain = int((~pain_mask).sum())
    n_pain = int(pain_mask.sum())
    
    vmin, vmax = 0.0, 1.0
    colormap = "viridis"
    
    # Auto-calculate n_lines if not specified
    min_lines = int(get_config_value(config, "plotting.plots.features.connectivity.circle_min_lines", 20))
    if n_lines is None:
        n_lines = max(min_lines, max(n_sig_nonpain, n_sig_pain))
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 9), subplot_kw=dict(polar=True))
    
    try:
        plot_connectivity_circle(
            matrix_nonpain, node_names, n_lines=n_lines, ax=axes[0],
            title="", show=False,
            vmin=vmin, vmax=vmax, colorbar=False, colormap=colormap
        )
        axes[0].set_title(f"Non-Pain\n(n={n_nonpain} trials, {n_sig_nonpain} edges)", 
                         fontsize=plot_cfg.font.suptitle, fontweight="bold", color=condition_colors["nonpain"])
        
        plot_connectivity_circle(
            matrix_pain, node_names, n_lines=n_lines, ax=axes[1],
            title="", show=False,
            vmin=vmin, vmax=vmax, colorbar=True, colormap=colormap
        )
        axes[1].set_title(f"Pain\n(n={n_pain} trials, {n_sig_pain} edges)", 
                         fontsize=plot_cfg.font.suptitle, fontweight="bold", color=condition_colors["pain"])
    except Exception as e:
        log_if_present(logger, "error", f"Failed to plot: {e}")
        plt.close(fig)
        return
    
    # Main title
    title = (f"{measure.upper()} Connectivity: {band.capitalize()} Band\n"
             f"Subject: {subject} | Top {int(top_fraction*100)}% connections (threshold ≥ {threshold:.3f})")
    fig.suptitle(title, fontsize=plot_cfg.font.figure_title, fontweight="bold", y=0.98)
    
    # Footer annotation
    footer_text = (f"{n_nodes} nodes | {n_edges} total edges | "
                   f"Showing connections ≥ {threshold:.3f}")
    fig.text(0.5, 0.02, footer_text, ha='center', va='bottom', fontsize=plot_cfg.font.large, color='gray')
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.93])
    save_fig(fig, save_dir / f"sub-{subject}_connectivity_{measure}_{band}_circle_by_condition",
             formats=plot_cfg.formats, dpi=plot_cfg.dpi,
             bbox_inches=plot_cfg.bbox_inches, pad_inches=plot_cfg.pad_inches)
    plt.close(fig)
    log_if_present(logger, "info", f"Saved {measure} {band} connectivity circle by condition")


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
    if conn_df is None or conn_df.empty or not window_indices:
        return
    plot_cfg = get_plot_config(config)
    mean_traces = []
    labels = []

    for win in window_indices:
        prefix = f"sw{win}corr_all_"
        win_cols = [c for c in conn_df.columns if str(c).startswith(prefix) and "__" in str(c)]
        if not win_cols:
            continue
        mean_traces.append(conn_df[win_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1))
        labels.append(win)

    if not mean_traces:
        log_if_present(logger, "warning", "No sliding connectivity columns found for trajectories.")
        return

    mat = np.vstack([np.asarray(t) for t in mean_traces])
    fig, ax = plt.subplots(figsize=plot_cfg.get_figure_size("sliding", plot_type="connectivity"))
    mean_all = np.nanmean(mat, axis=1)
    sem_all = np.nanstd(mat, axis=1) / np.sqrt(np.maximum(1, np.sum(np.isfinite(mat), axis=1)))
    ax.plot(window_centers[:len(mean_all)], mean_all, color=plot_cfg.get_color("blue"), label="All trials")
    ax.fill_between(
        window_centers[:len(mean_all)],
        mean_all - sem_all,
        mean_all + sem_all,
        color=plot_cfg.get_color("blue"),
        alpha=0.2,
    )

    n_trials = mat.shape[1]
    n_windows = len(window_indices)
    
    pain_mask = extract_pain_mask(aligned_events, config)
    if pain_mask is not None and len(pain_mask) == mat.shape[1]:
        for mask_val, label, color in [(False, "Non-pain", plot_cfg.get_color("nonpain")), (True, "Pain", plot_cfg.get_color("pain"))]:
            m = mat[:, pain_mask.to_numpy() == mask_val]
            n_cond = m.shape[1]
            if m.size == 0:
                continue
            mean_cond = np.nanmean(m, axis=1)
            sem_cond = np.nanstd(m, axis=1) / np.sqrt(np.maximum(1, np.sum(np.isfinite(m), axis=1)))
            ax.plot(window_centers[:len(mean_cond)], mean_cond, label=f"{label} (n={n_cond})", color=color)
            ax.fill_between(
                window_centers[:len(mean_cond)],
                mean_cond - sem_cond,
                mean_cond + sem_cond,
                color=color,
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

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
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


def plot_connectivity_by_condition(
    features_df: pd.DataFrame,
    events_df: pd.DataFrame,
    subject: str,
    save_dir: Path,
    logger: logging.Logger,
    config: Any,
) -> None:
    if features_df is None or features_df.empty or events_df is None:
        return

    pain_mask = extract_pain_mask(events_df, config)
    if pain_mask is None:
        return

    from eeg_pipeline.plotting.features.utils import (
        compute_condition_stats,
        apply_fdr_correction,
        format_stats_annotation,
        format_footer_annotation,
    )

    measures = get_config_value(config, "plotting.plots.features.connectivity.measures", ["aec", "wpli"])
    bands = list(get_frequency_band_names(config) or ['theta', 'alpha', 'beta', 'gamma'])
    
    plot_cfg = get_plot_config(config)
    condition_colors = {"pain": plot_cfg.get_color("pain"), "nonpain": plot_cfg.get_color("nonpain")}
    band_colors = {band: get_band_color(band, config) for band in bands}
    
    all_stats = []
    all_pvals = []
    cell_data = {}
    
    for m_idx, measure in enumerate(measures):
        for b_idx, band in enumerate(bands):
            edge_cols = [
                c for c in features_df.columns 
                if measure in c and f'_{band}_' in c and ('__' in c or '_chpair_' in c)
            ]
            
            if not edge_cols:
                cell_data[(m_idx, b_idx)] = None
                continue
            
            trial_means = features_df[edge_cols].mean(axis=1)
            vals_pain = trial_means[pain_mask].dropna().values
            vals_nonpain = trial_means[~pain_mask].dropna().values
            
            if len(vals_nonpain) >= 3 and len(vals_pain) >= 3:
                stats_result = compute_condition_stats(vals_nonpain, vals_pain, n_boot=1000, config=config)
                all_stats.append(stats_result)
                all_pvals.append(stats_result["p_raw"])
                cell_data[(m_idx, b_idx)] = {
                    "vals_nonpain": vals_nonpain,
                    "vals_pain": vals_pain,
                    "stats": stats_result,
                    "stats_idx": len(all_stats) - 1,
                }
            else:
                cell_data[(m_idx, b_idx)] = {
                    "vals_nonpain": vals_nonpain,
                    "vals_pain": vals_pain,
                    "stats": None,
                    "stats_idx": None,
                }
    
    if all_pvals:
        valid_pvals = [p for p in all_pvals if np.isfinite(p)]
        if valid_pvals:
            rejected, qvals, _ = apply_fdr_correction(valid_pvals, config=config)
            q_idx = 0
            for i, p in enumerate(all_pvals):
                if np.isfinite(p):
                    all_stats[i]["q_fdr"] = qvals[q_idx]
                    all_stats[i]["fdr_significant"] = rejected[q_idx]
                    q_idx += 1
                else:
                    all_stats[i]["q_fdr"] = np.nan
                    all_stats[i]["fdr_significant"] = False
            n_significant = int(np.sum(rejected))
        else:
            n_significant = 0
    else:
        n_significant = 0
    
    fig, axes = plt.subplots(len(measures), len(bands), figsize=(14, 8), sharey="row")
    
    for m_idx, measure in enumerate(measures):
        for b_idx, band in enumerate(bands):
            ax = axes[m_idx, b_idx]
            
            data = cell_data.get((m_idx, b_idx))
            
            if data is None:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                continue
            
            vals_nonpain = data["vals_nonpain"]
            vals_pain = data["vals_pain"]
            
            if len(vals_nonpain) > 0 and len(vals_pain) > 0:
                bp = ax.boxplot([vals_nonpain, vals_pain], 
                               positions=[0, 1], widths=0.4, patch_artist=True)
                bp["boxes"][0].set_facecolor(condition_colors["nonpain"])
                bp["boxes"][0].set_alpha(0.6)
                bp["boxes"][1].set_facecolor(condition_colors["pain"])
                bp["boxes"][1].set_alpha(0.6)
                
                ax.scatter(np.random.uniform(-0.1, 0.1, len(vals_nonpain)), 
                          vals_nonpain, c=condition_colors["nonpain"], alpha=0.4, s=10)
                ax.scatter(1 + np.random.uniform(-0.1, 0.1, len(vals_pain)), 
                          vals_pain, c=condition_colors["pain"], alpha=0.4, s=10)
                
                if data["stats"] is not None:
                    s = data["stats"]
                    annotation = format_stats_annotation(
                        p_raw=s["p_raw"],
                        q_fdr=s.get("q_fdr"),
                        cohens_d=s["cohens_d"],
                        ci_low=s["ci_low"],
                        ci_high=s["ci_high"],
                        compact=True,
                    )
                    text_color = "#d62728" if s.get("fdr_significant", False) else "#333333"
                    ax.text(0.5, 0.98, annotation, transform=ax.transAxes, 
                           ha="center", fontsize=6, va="top", color=text_color,
                           bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
            
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["NP", "P"], fontsize=plot_cfg.font.medium)
            ax.set_title(f"{band.capitalize()}", fontweight="bold", color=band_colors[band])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            
            if b_idx == 0:
                ax.set_ylabel(f"{measure.upper()}")
    
    n_pain = int(pain_mask.sum())
    n_nonpain = int((~pain_mask).sum())
    fig.suptitle(f"Connectivity by Condition (sub-{subject})\nN: {n_nonpain} non-pain, {n_pain} pain", 
                fontsize=plot_cfg.font.figure_title, fontweight="bold", y=1.02)
    
    n_tests = len([p for p in all_pvals if np.isfinite(p)])
    footer = format_footer_annotation(
        n_tests=n_tests,
        correction_method="FDR-BH",
        alpha=0.05,
        n_significant=n_significant,
        additional_info="Mann-Whitney U | Bootstrap 95% CI | †=FDR significant"
    )
    fig.text(0.5, 0.01, footer, ha="center", va="bottom", fontsize=8, color="gray")
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    save_fig(
        fig,
        save_dir / f"sub-{subject}_connectivity_by_condition",
        formats=plot_cfg.formats,
        dpi=plot_cfg.dpi,
        bbox_inches=plot_cfg.bbox_inches,
        pad_inches=plot_cfg.pad_inches
    )
    plt.close(fig)
    log_if_present(logger, "info", f"Saved connectivity by condition ({n_significant}/{n_tests} FDR significant)")


def plot_connectivity_heatmap(
    features_df: pd.DataFrame,
    info: mne.Info,
    subject: str,
    save_dir: Path,
    logger: logging.Logger,
    config: Any,
    prefix: str,
    events_df: Optional[pd.DataFrame] = None,
) -> None:
    edge_cols = [c for c in features_df.columns if c.startswith(prefix + "_") and "__" in c]
    if not edge_cols:
        return

    edge_nodes = set()
    for col in edge_cols:
        try:
            nodes_str = col.split("_")[-1]
            ch1, ch2 = nodes_str.split("__")
            edge_nodes.update([ch1, ch2])
        except ValueError:
            continue
    channel_order = [ch for ch in info.ch_names if ch in edge_nodes]
    adj = build_adjacency_from_edges(features_df, edge_cols, channel_order)
    if not np.any(np.isfinite(adj)):
        return

    sig_edges = compute_significance_mask(features_df, edge_cols, events_df, config)
    plot_cfg = get_plot_config(config)
    vmax = float(np.nanmax(np.abs(adj))) if np.any(np.isfinite(adj)) else 1.0

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(adj, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(channel_order)))
    ax.set_yticks(range(len(channel_order)))
    ax.set_xticklabels(channel_order, rotation=90, fontsize=plot_cfg.font.annotation)
    ax.set_yticklabels(channel_order, fontsize=plot_cfg.font.annotation)
    ax.set_title(f"{prefix} mean connectivity (sub-{subject})")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Connectivity")

    if sig_edges:
        sig_mask = np.zeros_like(adj, dtype=bool)
        for col in sig_edges:
            try:
                nodes_str = col.split("_")[-1]
                ch1, ch2 = nodes_str.split("__")
            except ValueError:
                continue
            if ch1 in channel_order and ch2 in channel_order:
                i = channel_order.index(ch1)
                j = channel_order.index(ch2)
                sig_mask[i, j] = sig_mask[j, i] = True
        ax.contour(sig_mask, colors="k", levels=[0.5], linewidths=0.5)

    ensure_dir(save_dir)
    output_name = save_dir / f"sub-{subject}_connectivity_heatmap_{prefix}"
    save_fig(
        fig,
        output_name,
        formats=plot_cfg.formats,
        dpi=plot_cfg.dpi,
        bbox_inches=plot_cfg.bbox_inches,
        pad_inches=plot_cfg.pad_inches,
    )
    plt.close(fig)
    log_if_present(logger, "info", f"Saved connectivity heatmap for {prefix}")


def plot_connectivity_network(
    features_df: pd.DataFrame,
    info: mne.Info,
    subject: str,
    save_dir: Path,
    logger: logging.Logger,
    config: Any,
    prefix: str,
    events_df: Optional[pd.DataFrame] = None,
) -> None:
    edge_cols = [c for c in features_df.columns if c.startswith(prefix + "_") and "__" in c]
    if not edge_cols:
        return

    edge_nodes = set()
    for col in edge_cols:
        try:
            nodes_str = col.split("_")[-1]
            ch1, ch2 = nodes_str.split("__")
            edge_nodes.update([ch1, ch2])
        except ValueError:
            continue
    channel_order = [ch for ch in info.ch_names if ch in edge_nodes]
    adj = build_adjacency_from_edges(features_df, edge_cols, channel_order)
    if not np.any(np.isfinite(adj)):
        return

    sig_edges = compute_significance_mask(features_df, edge_cols, events_df, config)
    sig_set = sig_edges if isinstance(sig_edges, set) else set()

    plot_cfg = get_plot_config(config)
    G = nx.Graph()
    for i, ch_i in enumerate(channel_order):
        G.add_node(ch_i)
        for j in range(i + 1, len(channel_order)):
            w = adj[i, j]
            if np.isfinite(w) and np.abs(w) > 0:
                G.add_edge(ch_i, channel_order[j], weight=float(w))

    if G.number_of_edges() == 0:
        return

    pos = nx.spring_layout(G, seed=42)
    weights = np.array([d["weight"] for _, _, d in G.edges(data=True)], dtype=float)
    vmax = float(np.nanmax(np.abs(weights))) if weights.size else 1.0

    fig, ax = plt.subplots(figsize=(6, 5))
    edges = G.edges()
    colors = [G[u][v]["weight"] for u, v in edges]
    node_color = plot_cfg.get_color("network_node")
    nx.draw_networkx_nodes(G, pos, node_size=120, node_color=node_color, alpha=0.8, ax=ax)
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=edges,
        edge_color=colors,
        edge_cmap=plt.cm.RdBu_r,
        edge_vmin=-vmax,
        edge_vmax=vmax,
        width=2.0,
        alpha=0.7,
        ax=ax,
    )
    nx.draw_networkx_labels(G, pos, font_size=7, font_weight="bold", ax=ax)

    if sig_set:
        highlight = []
        for col in sig_set:
            try:
                nodes_str = col.split("_")[-1]
                ch1, ch2 = nodes_str.split("__")
            except ValueError:
                continue
            if G.has_edge(ch1, ch2):
                highlight.append((ch1, ch2))
        nx.draw_networkx_edges(G, pos, edgelist=highlight, edge_color="lime", width=3.0, alpha=0.9, ax=ax)

    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(vmin=-vmax, vmax=vmax))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Connectivity")
    ax.set_title(f"Connectivity network ({prefix}, sub-{subject})")
    ax.axis("off")

    ensure_dir(save_dir)
    output_name = save_dir / f"sub-{subject}_connectivity_network_{prefix}"
    save_fig(
        fig,
        output_name,
        formats=plot_cfg.formats,
        dpi=plot_cfg.dpi,
        bbox_inches=plot_cfg.bbox_inches,
        pad_inches=plot_cfg.pad_inches,
    )
    plt.close(fig)
    log_if_present(logger, "info", f"Saved connectivity network for {prefix}")


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
        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
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
    alpha = float(get_config_value(config, "behavior_analysis.statistics.fdr_alpha", get_config_value(config, "statistics.fdr_alpha", 0.05)))

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
