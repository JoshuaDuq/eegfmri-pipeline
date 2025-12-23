"""
ERDS Visualization
==================

Clean, publication-quality visualizations for ERD/ERS features.
Uses violin/strip plots for distributions and summary comparisons.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from eeg_pipeline.domain.features.naming import NamingSchema
from eeg_pipeline.plotting.io.figures import save_fig
from eeg_pipeline.plotting.config import get_plot_config
from eeg_pipeline.plotting.features.utils import get_band_names, get_band_colors


###################################################################
# ERDS Distribution Plots
###################################################################

def _get_erds_segments(features_df: pd.DataFrame) -> List[str]:
    segments = set()
    for col in features_df.columns:
        parsed = NamingSchema.parse(str(col))
        if not parsed.get("valid"):
            continue
        if parsed.get("group") != "erds":
            continue
        segment = str(parsed.get("segment") or "")
        if segment:
            segments.add(segment)
    return sorted(segments)


def _select_erds_segment(features_df: pd.DataFrame, preferred: str = "active") -> Optional[str]:
    segments = _get_erds_segments(features_df)
    if not segments:
        return None
    if preferred in segments:
        return preferred
    return segments[0]


def _collect_erds_values(
    features_df: pd.DataFrame,
    *,
    band: str,
    segment: str,
    stat: str,
    scope: str = "global",
) -> np.ndarray:
    cols: List[str] = []
    for col in features_df.columns:
        parsed = NamingSchema.parse(str(col))
        if not parsed.get("valid"):
            continue
        if parsed.get("group") != "erds":
            continue
        if str(parsed.get("segment") or "") != str(segment):
            continue
        if str(parsed.get("band") or "") != str(band):
            continue
        if scope and str(parsed.get("scope") or "") != str(scope):
            continue
        if str(parsed.get("stat") or "") != str(stat):
            continue
        cols.append(str(col))

    if cols:
        if len(cols) == 1:
            series = pd.to_numeric(features_df[cols[0]], errors="coerce")
        else:
            series = features_df[cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
        vals = series.dropna().values
        return vals[np.isfinite(vals)]

    if scope == "global":
        base_stat = stat
        if base_stat.endswith("_mean"):
            base_stat = base_stat[:-5]
        elif base_stat.endswith("_std"):
            base_stat = base_stat[:-4]
        if base_stat != stat:
            return _collect_erds_values(
                features_df, band=band, segment=segment, stat=base_stat, scope="ch"
            )

    return np.array([])


def plot_erds_temporal_evolution(
    features_df: pd.DataFrame,
    save_path: Path,
    *,
    config: Any = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> plt.Figure:
    """ERDS percent/dB distributions by band for the active segment."""
    bands = get_band_names(config)
    band_colors = get_band_colors(config)
    plot_cfg = get_plot_config(config)
    if figsize is None:
        figsize = plot_cfg.get_figure_size("wide", plot_type="features")

    segment = _select_erds_segment(features_df, preferred="active")
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    ax = axes[0]
    data_list = []
    positions = []
    colors = []

    if segment is not None:
        for i, band in enumerate(bands):
            vals = _collect_erds_values(
                features_df,
                band=band,
                segment=segment,
                stat="percent_mean",
                scope="global",
            )
            if vals.size > 0:
                data_list.append(vals)
                positions.append(i)
                colors.append(band_colors[band])

    if data_list:
        parts = ax.violinplot(data_list, positions=positions, showmedians=True, widths=0.7)
        for i, pc in enumerate(parts.get("bodies", [])):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.6)
        ax.set_xticks(range(len(bands)))
        ax.set_xticklabels([b.capitalize() for b in bands])
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])

    ax.set_xlabel("Band")
    ax.set_ylabel("ERDS (%)")
    ax.set_title("ERDS Percent Change")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1]
    data_list = []
    positions = []
    colors = []

    if segment is not None:
        for i, band in enumerate(bands):
            vals = _collect_erds_values(
                features_df,
                band=band,
                segment=segment,
                stat="db_mean",
                scope="global",
            )
            if vals.size > 0:
                data_list.append(vals)
                positions.append(i)
                colors.append(band_colors[band])

    if data_list:
        parts = ax.violinplot(data_list, positions=positions, showmedians=True, widths=0.7)
        for i, pc in enumerate(parts.get("bodies", [])):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.6)
        ax.axhline(0, color="black", linestyle="-", linewidth=1)
        ax.set_xticks(range(len(bands)))
        ax.set_xticklabels([b.capitalize() for b in bands])
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])

    ax.set_xlabel("Band")
    ax.set_ylabel("ERDS (dB)")
    ax.set_title("Log-Ratio Change")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    seg_label = segment if segment is not None else "unknown"
    fig.suptitle(
        f"ERDS by Band ({seg_label})",
        fontsize=plot_cfg.font.figure_title,
        fontweight="bold",
        y=1.02,
    )

    plt.tight_layout()
    save_fig(fig, save_path)
    plt.close(fig)

    return fig


def plot_erds_latency_distribution(
    features_df: pd.DataFrame,
    save_path: Path,
    *,
    config: Any = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> plt.Figure:
    """ERDS peak/onset latency distributions by band."""
    plot_cfg = get_plot_config(config)
    if figsize is None:
        figsize = plot_cfg.get_figure_size("wide", plot_type="features")
    bands = get_band_names(config)
    band_colors = get_band_colors(config)
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    ax = axes[0]
    data_list = []
    positions = []
    colors = []

    segment = _select_erds_segment(features_df, preferred="active")
    if segment is not None:
        for i, band in enumerate(bands):
            vals = _collect_erds_values(
                features_df,
                band=band,
                segment=segment,
                stat="peak_latency",
                scope="ch",
            )
            if vals.size > 0:
                data_list.append(vals * 1000.0)
                positions.append(i)
                colors.append(band_colors[band])

    if data_list:
        parts = ax.violinplot(data_list, positions=positions, showmedians=True, widths=0.7)
        for i, pc in enumerate(parts.get("bodies", [])):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.6)
        for i, (pos, vals) in enumerate(zip(positions, data_list)):
            jitter = np.random.uniform(-0.1, 0.1, len(vals))
            ax.scatter(pos + jitter, vals, c=colors[i], alpha=0.3, s=8)
        ax.set_xticks(range(len(bands)))
        ax.set_xticklabels([b.capitalize() for b in bands])
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
    
    ax.set_xlabel("Band")
    ax.set_ylabel("Peak Latency (ms)")
    ax.set_title("Peak Latency")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    ax = axes[1]
    data_list = []
    positions = []
    colors = []
    
    if segment is not None:
        for i, band in enumerate(bands):
            vals = _collect_erds_values(
                features_df,
                band=band,
                segment=segment,
                stat="onset_latency",
                scope="ch",
            )
            if vals.size > 0:
                data_list.append(vals * 1000.0)
                positions.append(i)
                colors.append(band_colors[band])

    if data_list:
        parts = ax.violinplot(data_list, positions=positions, showmedians=True, widths=0.7)
        for i, pc in enumerate(parts.get("bodies", [])):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.6)
        for i, (pos, vals) in enumerate(zip(positions, data_list)):
            jitter = np.random.uniform(-0.1, 0.1, len(vals))
            ax.scatter(pos + jitter, vals, c=colors[i], alpha=0.3, s=8)
        ax.set_xticks(range(len(bands)))
        ax.set_xticklabels([b.capitalize() for b in bands])
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
    
    ax.set_xlabel("Band")
    ax.set_ylabel("Onset Latency (ms)")
    ax.set_title("Onset Latency")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    seg_label = segment if segment is not None else "unknown"
    fig.suptitle(
        f"ERDS Latencies ({seg_label})",
        fontsize=plot_cfg.font.figure_title,
        fontweight="bold",
        y=1.02,
    )
    
    plt.tight_layout()
    save_fig(fig, save_path)
    plt.close(fig)
    
    return fig


def plot_erds_erd_ers_separation(
    features_df: pd.DataFrame,
    save_path: Path,
    *,
    config: Any = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> plt.Figure:
    """ERD vs ERS magnitude distributions by band."""
    plot_cfg = get_plot_config(config)
    if figsize is None:
        figsize = plot_cfg.get_figure_size("wide", plot_type="features")
    bands = get_band_names(config)
    band_colors = get_band_colors(config)
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    ax = axes[0]
    data_list = []
    positions = []
    colors = []

    segment = _select_erds_segment(features_df, preferred="active")
    if segment is not None:
        for i, band in enumerate(bands):
            vals = _collect_erds_values(
                features_df,
                band=band,
                segment=segment,
                stat="erd_magnitude",
                scope="ch",
            )
            if vals.size > 0:
                data_list.append(vals)
                positions.append(i)
                colors.append(band_colors[band])

    if data_list:
        parts = ax.violinplot(data_list, positions=positions, showmedians=True, widths=0.7)
        for i, pc in enumerate(parts.get("bodies", [])):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.6)
        ax.set_xticks(range(len(bands)))
        ax.set_xticklabels([b.capitalize() for b in bands])
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])

    ax.set_xlabel("Band")
    ax.set_ylabel("ERD Magnitude (%)")
    ax.set_title("ERD Magnitude")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    ax = axes[1]
    data_list = []
    positions = []
    colors = []
    
    if segment is not None:
        for i, band in enumerate(bands):
            vals = _collect_erds_values(
                features_df,
                band=band,
                segment=segment,
                stat="ers_magnitude",
                scope="ch",
            )
            if vals.size > 0:
                data_list.append(vals)
                positions.append(i)
                colors.append(band_colors[band])

    if data_list:
        parts = ax.violinplot(data_list, positions=positions, showmedians=True, widths=0.7)
        for i, pc in enumerate(parts.get("bodies", [])):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.6)
        ax.set_xticks(range(len(bands)))
        ax.set_xticklabels([b.capitalize() for b in bands])
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])

    ax.set_xlabel("Band")
    ax.set_ylabel("ERS Magnitude (%)")
    ax.set_title("ERS Magnitude")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    seg_label = segment if segment is not None else "unknown"
    fig.suptitle(
        f"ERD/ERS Magnitudes ({seg_label})",
        fontsize=plot_cfg.font.figure_title,
        fontweight="bold",
        y=1.02,
    )
    
    plt.tight_layout()
    save_fig(fig, save_path)
    plt.close(fig)
    
    return fig


def plot_erds_global_summary(
    features_df: pd.DataFrame,
    save_path: Path,
    *,
    config: Any = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> plt.Figure:
    """Global ERDS summary by band."""
    plot_cfg = get_plot_config(config)
    if figsize is None:
        figsize = plot_cfg.get_figure_size("wide", plot_type="features")
    bands = get_band_names(config)
    band_colors = get_band_colors(config)
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    ax = axes[0]
    data_list = []
    positions = []
    colors = []

    segment = _select_erds_segment(features_df, preferred="active")
    if segment is not None:
        for i, band in enumerate(bands):
            vals = _collect_erds_values(
                features_df,
                band=band,
                segment=segment,
                stat="percent_mean",
                scope="global",
            )
            if vals.size > 0:
                data_list.append(vals)
                positions.append(i)
                colors.append(band_colors[band])

    if data_list:
        parts = ax.violinplot(data_list, positions=positions, showmedians=True, widths=0.7)
        for i, pc in enumerate(parts.get("bodies", [])):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.6)
        ax.axhline(0, color="black", linestyle="-", linewidth=1)
        ax.set_xticks(range(len(bands)))
        ax.set_xticklabels([b.capitalize() for b in bands])
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])

    ax.set_xlabel("Band")
    ax.set_ylabel("ERDS (%)")
    ax.set_title("Mean ERDS")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    ax = axes[1]
    data_list = []
    positions = []
    colors = []
    
    if segment is not None:
        for i, band in enumerate(bands):
            vals = _collect_erds_values(
                features_df,
                band=band,
                segment=segment,
                stat="percent_std",
                scope="global",
            )
            if vals.size > 0:
                data_list.append(vals)
                positions.append(i)
                colors.append(band_colors[band])

    if data_list:
        parts = ax.violinplot(data_list, positions=positions, showmedians=True, widths=0.7)
        for i, pc in enumerate(parts.get("bodies", [])):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.6)
        ax.set_xticks(range(len(bands)))
        ax.set_xticklabels([b.capitalize() for b in bands])
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])

    ax.set_xlabel("Band")
    ax.set_ylabel("ERDS Std (%)")
    ax.set_title("Across-Channel Variability")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    seg_label = segment if segment is not None else "unknown"
    fig.suptitle(
        f"ERDS Summary ({seg_label})",
        fontsize=plot_cfg.font.figure_title,
        fontweight="bold",
        y=1.02,
    )
    
    plt.tight_layout()
    save_fig(fig, save_path)
    plt.close(fig)
    
    return fig


__all__ = [
    "plot_erds_temporal_evolution",
    "plot_erds_latency_distribution",
    "plot_erds_erd_ers_separation",
    "plot_erds_global_summary",
]
