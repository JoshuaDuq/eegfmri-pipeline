"""
Burst Feature Visualization
===========================

Plots for oscillatory burst metrics across bands and conditions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from eeg_pipeline.plotting.config import get_plot_config
from eeg_pipeline.plotting.io.figures import save_fig
from eeg_pipeline.utils.analysis.events import extract_pain_mask
from eeg_pipeline.plotting.features.utils import (
    get_named_segments,
    get_named_bands,
    get_band_names,
    collect_named_series,
    compute_condition_stats,
    apply_fdr_correction,
    format_stats_annotation,
    format_footer_annotation,
    get_condition_colors,
)
from eeg_pipeline.utils.config.loader import get_config_value


def _select_segment(segments: List[str], preferred: str = "active") -> Optional[str]:
    if not segments:
        return None
    if preferred in segments:
        return preferred
    return segments[0]


def plot_bursts_by_band(
    features_df: pd.DataFrame,
    save_path: Path,
    *,
    config: Any = None,
    metrics: Optional[List[str]] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> plt.Figure:
    """Plot burst metric distributions across bands."""
    plot_cfg = get_plot_config(config)
    segments = get_named_segments(features_df, group="bursts")
    segment = _select_segment(segments, preferred="active")
    if segment is None:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No burst data", ha="center", va="center")
        return fig
    bands = get_named_bands(features_df, group="bursts", segment=segment)
    if not bands:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No burst data", ha="center", va="center")
        return fig

    if metrics is None:
        metrics = get_config_value(
            config,
            "plotting.plots.features.bursts.metrics",
            ["rate", "count", "duration_mean", "amp_mean", "fraction"],
        )
    metrics = list(metrics or [])
    metrics = [m for m in metrics if isinstance(m, str)]
    if not metrics:
        metrics = ["rate"]

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
            series, _, _ = collect_named_series(
                features_df,
                group="bursts",
                segment=segment,
                band=band,
                stat_preference=[metric],
                scope_preference=["global", "roi", "ch"],
            )
            vals = series.dropna().values
            if vals.size == 0:
                continue
            data_list.append(vals)
            positions.append(i)

        if data_list:
            parts = ax.violinplot(data_list, positions=positions, showmedians=True, widths=0.7)
            for pc in parts.get("bodies", []):
                pc.set_facecolor("#D97706")
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

    seg_label = segment if segment is not None else "unknown"
    fig.suptitle(
        f"Burst Metrics by Band ({seg_label})",
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


def plot_bursts_by_condition(
    features_df: pd.DataFrame,
    events_df: pd.DataFrame,
    subject: str,
    save_path: Path,
    *,
    config: Any = None,
    metric: str = "rate",
    figsize: Optional[Tuple[float, float]] = None,
) -> plt.Figure:
    """Compare burst metrics between pain vs non-pain conditions."""
    if features_df is None or features_df.empty or events_df is None:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No burst data", ha="center", va="center")
        return fig
    if len(features_df) != len(events_df):
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Length mismatch", ha="center", va="center")
        return fig

    pain_mask = extract_pain_mask(events_df, config)
    if pain_mask is None:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No pain labels", ha="center", va="center")
        return fig

    plot_cfg = get_plot_config(config)
    condition_colors = get_condition_colors(config)

    segments = get_named_segments(features_df, group="bursts")
    segment = _select_segment(segments, preferred="active")
    if segment is None:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No burst data", ha="center", va="center")
        return fig
    bands = get_named_bands(features_df, group="bursts", segment=segment)
    if not bands:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No burst data", ha="center", va="center")
        return fig

    band_order = get_band_names(config)
    bands = [b for b in band_order if b in bands] + [b for b in bands if b not in band_order]

    if figsize is None:
        figsize = (max(6.0, len(bands) * 2.5), 5.0)

    fig, axes = plt.subplots(1, len(bands), figsize=figsize, squeeze=False)
    axes = axes.flatten()

    all_stats = []
    all_pvals = []
    band_data = {}

    for band in bands:
        series, _, _ = collect_named_series(
            features_df,
            group="bursts",
            segment=segment,
            band=band,
            stat_preference=[metric],
            scope_preference=["global", "roi", "ch"],
        )
        vals_pain = series[pain_mask].dropna().values
        vals_nonpain = series[~pain_mask].dropna().values
        if len(vals_pain) >= 3 and len(vals_nonpain) >= 3:
            stats_result = compute_condition_stats(vals_nonpain, vals_pain, n_boot=1000, config=config)
            all_stats.append(stats_result)
            all_pvals.append(stats_result["p_raw"])
            band_data[band] = (vals_nonpain, vals_pain, stats_result, len(all_stats) - 1)
        else:
            band_data[band] = (vals_nonpain, vals_pain, None, None)

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
            n_significant = int(np.sum(rejected))
        else:
            n_significant = 0
    else:
        n_significant = 0

    for idx, band in enumerate(bands):
        ax = axes[idx]
        vals_nonpain, vals_pain, stats_result, _ = band_data[band]
        if len(vals_nonpain) < 3 or len(vals_pain) < 3:
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center", transform=ax.transAxes)
            continue

        bp = ax.boxplot([vals_nonpain, vals_pain], positions=[0, 1], widths=0.4, patch_artist=True)
        bp["boxes"][0].set_facecolor(condition_colors["nonpain"])
        bp["boxes"][0].set_alpha(0.6)
        bp["boxes"][1].set_facecolor(condition_colors["pain"])
        bp["boxes"][1].set_alpha(0.6)

        ax.scatter(np.random.uniform(-0.1, 0.1, len(vals_nonpain)), vals_nonpain, c=condition_colors["nonpain"], alpha=0.4, s=12)
        ax.scatter(1 + np.random.uniform(-0.1, 0.1, len(vals_pain)), vals_pain, c=condition_colors["pain"], alpha=0.4, s=12)

        if stats_result is not None:
            annotation = format_stats_annotation(
                p_raw=stats_result["p_raw"],
                q_fdr=stats_result.get("q_fdr"),
                cohens_d=stats_result["cohens_d"],
                ci_low=stats_result["ci_low"],
                ci_high=stats_result["ci_high"],
                compact=True,
            )
            text_color = plot_cfg.style.colors.significant if stats_result.get("fdr_significant", False) else plot_cfg.style.colors.gray
            ax.text(0.5, 0.98, annotation, ha="center", va="top", transform=ax.transAxes,
                    fontsize=plot_cfg.font.annotation, color=text_color,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["NP", "P"], fontsize=plot_cfg.font.large)
        ax.set_title(band.capitalize(), fontsize=plot_cfg.font.title, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    n_pain = int(pain_mask.sum())
    n_nonpain = int((~pain_mask).sum())
    seg_label = segment if segment is not None else "unknown"
    fig.suptitle(
        f"Bursts ({metric}, {seg_label}, sub-{subject})\nN: {n_nonpain} NP, {n_pain} P",
        fontsize=plot_cfg.font.figure_title,
        fontweight="bold",
        y=1.02,
    )

    n_tests = len([p for p in all_pvals if np.isfinite(p)])
    footer = format_footer_annotation(
        n_tests=n_tests,
        correction_method="FDR-BH",
        alpha=0.05,
        n_significant=n_significant,
        additional_info="Mann-Whitney U | Bootstrap 95% CI | †=FDR significant",
    )
    fig.text(0.5, 0.01, footer, ha="center", va="bottom", fontsize=8, color="gray")

    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
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
    "plot_bursts_by_band",
    "plot_bursts_by_condition",
]
