"""
Spectral Feature Visualization
==============================

Plots for spectral peak metrics and edge frequencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from eeg_pipeline.domain.features.naming import NamingSchema
from eeg_pipeline.plotting.config import get_plot_config
from eeg_pipeline.plotting.io.figures import save_fig
from eeg_pipeline.plotting.features.utils import get_band_names
from eeg_pipeline.utils.config.loader import get_config_value


@dataclass
class SpectralColumn:
    name: str
    segment: str
    band: str
    scope: str
    stat: str


def _parse_spectral_column(col: str) -> Optional[SpectralColumn]:
    parsed = NamingSchema.parse(str(col))
    if parsed.get("valid") and parsed.get("group") == "spectral":
        return SpectralColumn(
            name=str(col),
            segment=str(parsed.get("segment") or ""),
            band=str(parsed.get("band") or ""),
            scope=str(parsed.get("scope") or ""),
            stat=str(parsed.get("stat") or ""),
        )

    name = str(col)
    if not name.startswith("spectral_"):
        return None

    parts = name.split("_")
    if len(parts) < 5:
        return None

    segment = parts[1]
    band = parts[2]
    scope = parts[3]
    if scope not in {"ch", "roi", "global"}:
        return None

    if scope == "global":
        stat = "_".join(parts[4:])
    else:
        if len(parts) < 6:
            return None
        stat = "_".join(parts[5:])
    return SpectralColumn(
        name=name,
        segment=segment,
        band=band,
        scope=scope,
        stat=stat,
    )


def _collect_spectral_columns(features_df: pd.DataFrame) -> List[SpectralColumn]:
    cols: List[SpectralColumn] = []
    for col in features_df.columns:
        entry = _parse_spectral_column(col)
        if entry is not None:
            cols.append(entry)
    return cols


def _select_segment(segments: List[str], preferred: str = "active") -> Optional[str]:
    if not segments:
        return None
    if preferred in segments:
        return preferred
    return segments[0]


def _select_columns(
    entries: List[SpectralColumn],
    *,
    segment: str,
    band: str,
    stat_preference: List[str],
    scope_preference: List[str],
) -> Tuple[List[str], Optional[str], Optional[str]]:
    for scope in scope_preference:
        for stat in stat_preference:
            cols = [
                e.name
                for e in entries
                if e.segment == segment and e.band == band and e.scope == scope and e.stat == stat
            ]
            if cols:
                return cols, scope, stat
    return [], None, None


def plot_spectral_summary(
    features_df: pd.DataFrame,
    save_path: Path,
    *,
    config: Any = None,
    metrics: Optional[List[str]] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> plt.Figure:
    """Plot spectral feature distributions by band."""
    plot_cfg = get_plot_config(config)
    entries = _collect_spectral_columns(features_df)
    if not entries:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No spectral data", ha="center", va="center")
        return fig

    segments = sorted({e.segment for e in entries if e.segment})
    segment = _select_segment(segments, preferred="active")
    if segment is None:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No spectral data", ha="center", va="center")
        return fig

    bands = sorted({e.band for e in entries if e.segment == segment and e.band})
    if not bands:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No spectral data", ha="center", va="center")
        return fig

    if metrics is None:
        metrics = get_config_value(
            config,
            "plotting.plots.features.spectral.metrics",
            [
                "peak_freq",
                "center_freq",
                "bandwidth",
                "entropy",
                "peak_power",
                "logratio_mean",
                "logratio_std",
                "slope",
            ],
        )
    metrics = list(metrics or [])
    metrics = [m for m in metrics if any(e.stat == m for e in entries)]
    if not metrics:
        metrics = sorted({e.stat for e in entries if e.stat})[:4]

    band_order = get_band_names(config)
    bands = [b for b in band_order if b in bands] + [b for b in bands if b not in band_order]

    n_rows = len(metrics)
    if figsize is None:
        figsize = (max(8.0, len(bands) * 1.2), max(4.5, n_rows * 3.0))

    fig, axes = plt.subplots(n_rows, 1, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    for ax, metric in zip(axes, metrics):
        data_list = []
        positions = []
        for i, band in enumerate(bands):
            cols, _, _ = _select_columns(
                entries,
                segment=segment,
                band=band,
                stat_preference=[metric],
                scope_preference=["global", "roi", "ch"],
            )
            if not cols:
                continue
            series = (
                pd.to_numeric(features_df[cols[0]], errors="coerce")
                if len(cols) == 1
                else features_df[cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
            )
            vals = series.dropna().values
            if vals.size == 0:
                continue
            data_list.append(vals)
            positions.append(i)

        if data_list:
            parts = ax.violinplot(data_list, positions=positions, showmedians=True, widths=0.7)
            for pc in parts.get("bodies", []):
                pc.set_facecolor("#7C3AED")
                pc.set_alpha(0.6)
            ax.set_xticks(range(len(bands)))
            ax.set_xticklabels([b.capitalize() for b in bands])
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])

        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(metric.replace("_", " ").title(), fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        f"Spectral Features by Band ({segment})",
        fontsize=plot_cfg.font.figure_title,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    save_fig(
        fig,
        save_path,
        formats=plot_cfg.formats,
        dpi=plot_cfg.dpi,
        bbox_inches=plot_cfg.bbox_inches,
        pad_inches=plot_cfg.pad_inches,
    )
    plt.close(fig)
    return fig


def plot_spectral_edge_frequency(
    features_df: pd.DataFrame,
    save_path: Path,
    *,
    config: Any = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> plt.Figure:
    """Plot broadband spectral edge frequency distribution."""
    plot_cfg = get_plot_config(config)
    entries = _collect_spectral_columns(features_df)
    if not entries:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No spectral data", ha="center", va="center")
        return fig

    edge_cols = [e.name for e in entries if e.band == "broadband" and "edge" in e.stat]
    if not edge_cols:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No edge frequency data", ha="center", va="center")
        return fig

    series = (
        pd.to_numeric(features_df[edge_cols[0]], errors="coerce")
        if len(edge_cols) == 1
        else features_df[edge_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    )
    vals = series.dropna().values
    if vals.size == 0:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No edge frequency data", ha="center", va="center")
        return fig

    if figsize is None:
        figsize = (8.0, 4.5)

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(vals, bins=25, color="#7C3AED", alpha=0.7, edgecolor="white")
    ax.set_xlabel("Edge Frequency (Hz)")
    ax.set_ylabel("Count")
    ax.set_title("Spectral Edge Frequency", fontweight="bold")
    ax.axvline(np.nanmean(vals), color="black", linestyle="--", linewidth=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Broadband Spectral Edge Frequency",
        fontsize=plot_cfg.font.figure_title,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    save_fig(
        fig,
        save_path,
        formats=plot_cfg.formats,
        dpi=plot_cfg.dpi,
        bbox_inches=plot_cfg.bbox_inches,
        pad_inches=plot_cfg.pad_inches,
    )
    plt.close(fig)
    return fig


__all__ = [
    "plot_spectral_summary",
    "plot_spectral_edge_frequency",
]
