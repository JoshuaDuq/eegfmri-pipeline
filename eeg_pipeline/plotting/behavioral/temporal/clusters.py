from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, List, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from eeg_pipeline.plotting.config import get_plot_config
from eeg_pipeline.utils.config.loader import get_frequency_band_names
from eeg_pipeline.infra.logging import get_default_logger as _get_default_logger
from eeg_pipeline.infra.paths import ensure_dir
from eeg_pipeline.plotting.io.figures import (
    get_default_config as _get_default_config,
    log_if_present as _log_if_present,
    save_fig,
)
from eeg_pipeline.infra.tsv import read_tsv


def plot_pain_nonpain_clusters(
    subject: str,
    stats_dir: Path,
    plots_dir: Path,
    config: Any,
    logger: Optional[logging.Logger] = None,
    alpha: Optional[float] = None,
) -> None:
    logger = logger or _get_default_logger()
    plot_cfg = get_plot_config(config)
    ensure_dir(plots_dir)

    if alpha is not None:
        alpha_used = float(alpha)
    else:
        try:
            alpha_used = float(
                config.get(
                    "behavior_analysis.statistics.fdr_alpha",
                    config.get("statistics.sig_alpha", 0.05),
                )
            )
        except (ValueError, TypeError):
            alpha_used = 0.05

    if not np.isfinite(alpha_used) or alpha_used <= 0:
        alpha_used = 0.05

    files = sorted(stats_dir.glob("pain_nonpain_time_clusters_*.tsv"))
    if not files:
        _log_if_present(logger, "warning", f"No pain/non-pain cluster stats found in {stats_dir}")
        return

    frames: List[pd.DataFrame] = []
    for fpath in files:
        df = read_tsv(fpath)
        if df is None or df.empty:
            continue
        band = (
            df["band"].iloc[0]
            if "band" in df.columns
            else fpath.stem.replace("pain_nonpain_time_clusters_", "")
        )
        df["band"] = df.get("band", band)
        df["p_value"] = pd.to_numeric(df.get("p_value", np.nan), errors="coerce")
        if "fdr_reject_global" in df.columns:
            df["significant"] = df["fdr_reject_global"].fillna(False)
        else:
            df["significant"] = df["p_value"] < alpha_used
        df["t_start"] = pd.to_numeric(df.get("t_start", np.nan), errors="coerce")
        df["t_end"] = pd.to_numeric(df.get("t_end", np.nan), errors="coerce")
        df["cluster_index"] = pd.to_numeric(df.get("cluster_index", np.nan), errors="coerce")
        frames.append(df)

    if not frames:
        _log_if_present(logger, "warning", "Pain/non-pain cluster TSVs were found but empty.")
        return

    all_clusters = pd.concat(frames, ignore_index=True)
    sig_clusters = all_clusters[all_clusters["significant"]]
    if sig_clusters.empty:
        _log_if_present(logger, "warning", "No significant pain vs. non-pain clusters to plot.")
        return

    topomap_dir = plots_dir / "topomaps"
    ensure_dir(topomap_dir)
    sig_out = topomap_dir / "pain_nonpain_clusters_significant.tsv"
    sig_clusters.to_csv(sig_out, sep="\t", index=False)
    _log_if_present(logger, "info", f"Saved significant cluster table to {sig_out}")

    band_order = get_frequency_band_names(config)
    if not band_order:
        band_order = sorted(all_clusters["band"].unique())
    band_order = [b for b in band_order if not all_clusters[all_clusters["band"] == b].empty]
    if not band_order:
        _log_if_present(logger, "warning", "No clusters available after filtering bands.")
        return

    fig, ax = plt.subplots(figsize=plot_cfg.get_figure_size("wide", plot_type="behavioral"))
    for idx, band in enumerate(band_order):
        band_df = all_clusters[all_clusters["band"] == band]
        if band_df.empty:
            continue
        t_start_min = band_df["t_start"].dropna().min() if not band_df["t_start"].dropna().empty else 0
        t_end_max = band_df["t_end"].dropna().max() if not band_df["t_end"].dropna().empty else 0
        if np.isfinite(t_start_min) and np.isfinite(t_end_max):
            ax.hlines(
                idx,
                t_start_min,
                t_end_max,
                colors=plot_cfg.get_color("light_gray"),
                linestyles="--",
                linewidth=0.5,
                alpha=0.5,
            )

        for _, row in band_df.iterrows():
            t_start_val = row.get("t_start")
            t_end_val = row.get("t_end")

            if pd.isna(t_start_val) or pd.isna(t_end_val):
                continue

            try:
                t_start = float(t_start_val)
                t_end = float(t_end_val)
            except (ValueError, TypeError):
                continue

            if not np.isfinite(t_start) or not np.isfinite(t_end):
                continue

            width = t_end - t_start
            if width <= 0:
                continue

            color = (
                plot_cfg.get_color("significant")
                if row.get("significant", False)
                else plot_cfg.get_color("nonsignificant")
            )
            ax.broken_barh(
                [(t_start, width)],
                (idx - 0.35, 0.7),
                facecolors=color,
                edgecolors="none",
                alpha=0.8,
            )

    ax.set_yticks(range(len(band_order)))
    ax.set_yticklabels(band_order)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Band")
    ax.set_title(f"Pain vs. non-pain cluster summary (sub-{subject})")
    legend_patches = [
        mpatches.Patch(color=plot_cfg.get_color("significant"), label="Significant"),
        mpatches.Patch(color=plot_cfg.get_color("nonsignificant"), label="Non-significant"),
    ]
    ax.legend(handles=legend_patches, loc="upper right", frameon=False)
    plt.tight_layout()

    topomap_dir = plots_dir / "topomaps"
    ensure_dir(topomap_dir)
    save_fig(
        fig,
        topomap_dir / f"sub-{subject}_pain_nonpain_cluster_ribbons",
        formats=plot_cfg.formats,
        dpi=plot_cfg.dpi,
        bbox_inches=plot_cfg.bbox_inches,
        pad_inches=plot_cfg.pad_inches,
    )
    plt.close(fig)

    for band in band_order:
        band_df = sig_clusters[sig_clusters["band"] == band]
        if band_df.empty:
            continue

        t_start_vals = band_df["t_start"].dropna()
        t_end_vals = band_df["t_end"].dropna()

        if t_start_vals.empty or t_end_vals.empty:
            continue

        try:
            t_min = float(t_start_vals.min())
            t_max = float(t_end_vals.max())
        except (ValueError, TypeError):
            continue

        if not np.isfinite(t_min) or not np.isfinite(t_max) or t_max <= t_min:
            continue

        time_grid = np.linspace(t_min, t_max, 200)
        channels: List[str] = []
        for ch_list in band_df.get("channels", []):
            if pd.isna(ch_list):
                continue
            channels.extend([ch.strip() for ch in str(ch_list).split(",") if ch.strip()])
        channels = sorted(set(channels))
        if not channels:
            continue

        mask = np.zeros((len(channels), len(time_grid) - 1), dtype=int)
        for _, row in band_df.iterrows():
            ch_list = row.get("channels", "")
            if pd.isna(ch_list):
                continue
            row_channels = [ch.strip() for ch in str(ch_list).split(",") if ch.strip()]

            t_start_val = row.get("t_start")
            t_end_val = row.get("t_end")
            if pd.isna(t_start_val) or pd.isna(t_end_val):
                continue

            try:
                t_start = float(t_start_val)
                t_end = float(t_end_val)
            except (ValueError, TypeError):
                continue

            if not np.isfinite(t_start) or not np.isfinite(t_end):
                continue

            overlaps = (time_grid[:-1] < t_end) & (time_grid[1:] > t_start)
            if not np.any(overlaps):
                continue

            for ch in row_channels:
                try:
                    ch_idx = channels.index(ch)
                except ValueError:
                    continue
                mask[ch_idx, overlaps] = 1

        fig, ax = plt.subplots(figsize=plot_cfg.get_figure_size("standard", plot_type="behavioral"))
        mesh = ax.pcolormesh(
            time_grid,
            np.arange(len(channels) + 1),
            mask,
            cmap="Reds",
            shading="auto",
            vmin=0,
            vmax=1,
        )
        ax.set_title(f"Pain vs. non-pain clusters (band: {band}, sub-{subject})")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Channel")
        ax.set_yticks(np.arange(len(channels)) + 0.5)
        ax.set_yticklabels(channels)
        cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
        cbar.set_label("Significant cluster coverage")
        plt.tight_layout()

        topomap_dir = plots_dir / "topomaps"
        ensure_dir(topomap_dir)
        save_fig(
            fig,
            topomap_dir / f"sub-{subject}_pain_nonpain_cluster_mask_{band}",
            formats=plot_cfg.formats,
            dpi=plot_cfg.dpi,
            bbox_inches=plot_cfg.bbox_inches,
            pad_inches=plot_cfg.pad_inches,
        )
        plt.close(fig)
