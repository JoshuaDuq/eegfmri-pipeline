from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from eeg_pipeline.plotting.config import get_plot_config
from eeg_pipeline.utils.config.loader import get_frequency_band_names
from eeg_pipeline.infra.logging import get_default_logger as _get_default_logger
from eeg_pipeline.infra.paths import ensure_dir
from eeg_pipeline.plotting.io.figures import (
    log_if_present as _log_if_present,
    save_fig,
)
from eeg_pipeline.infra.tsv import read_tsv

DEFAULT_ALPHA = 0.05
TIME_GRID_POINTS = 200
CLUSTER_PREFIX = "pain_nonpain_time_clusters_"


def _get_alpha_value(config: Any, alpha: Optional[float]) -> float:
    """Extract and validate alpha value from config or parameter."""
    if alpha is not None:
        alpha_value = float(alpha)
    else:
        try:
            alpha_value = float(
                config.get(
                    "behavior_analysis.statistics.fdr_alpha",
                    config.get("statistics.sig_alpha", DEFAULT_ALPHA),
                )
            )
        except (ValueError, TypeError):
            alpha_value = DEFAULT_ALPHA

    if not np.isfinite(alpha_value) or alpha_value <= 0:
        alpha_value = DEFAULT_ALPHA

    return alpha_value


def _extract_band_name(df: pd.DataFrame, filepath: Path) -> str:
    """Extract frequency band name from dataframe or filename."""
    if "band" in df.columns:
        return df["band"].iloc[0]
    return filepath.stem.replace(CLUSTER_PREFIX, "")


def _normalize_cluster_dataframe(df: pd.DataFrame, band: str, alpha: float) -> pd.DataFrame:
    """Normalize cluster dataframe with required columns and significance flags."""
    df = df.copy()
    df["band"] = band
    df["p_value"] = pd.to_numeric(df.get("p_value", np.nan), errors="coerce")

    if "fdr_reject_global" in df.columns:
        df["significant"] = df["fdr_reject_global"].fillna(False)
    else:
        df["significant"] = df["p_value"] < alpha

    df["t_start"] = pd.to_numeric(df.get("t_start", np.nan), errors="coerce")
    df["t_end"] = pd.to_numeric(df.get("t_end", np.nan), errors="coerce")
    df["cluster_index"] = pd.to_numeric(df.get("cluster_index", np.nan), errors="coerce")

    return df


def _load_cluster_dataframes(
    stats_dir: Path,
    alpha: float,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Load and combine all cluster statistics files."""
    files = sorted(stats_dir.glob(f"{CLUSTER_PREFIX}*.tsv"))
    if not files:
        _log_if_present(logger, "warning", f"No pain/non-pain cluster stats found in {stats_dir}")
        return pd.DataFrame()

    frames = []
    for filepath in files:
        df = read_tsv(filepath)
        if df is None or df.empty:
            continue

        band = _extract_band_name(df, filepath)
        normalized_df = _normalize_cluster_dataframe(df, band, alpha)
        frames.append(normalized_df)

    if not frames:
        _log_if_present(logger, "warning", "Condition contrast cluster TSVs were found but empty.")
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def _get_band_order(config: Any, all_clusters: pd.DataFrame) -> list[str]:
    """Determine frequency band order from config or data."""
    band_order = get_frequency_band_names(config)
    if not band_order:
        band_order = sorted(all_clusters["band"].unique())

    available_bands = [
        band
        for band in band_order
        if not all_clusters[all_clusters["band"] == band].empty
    ]

    return available_bands


def _extract_time_range(band_df: pd.DataFrame) -> Optional[tuple[float, float]]:
    """Extract valid time range from band dataframe."""
    t_start_vals = band_df["t_start"].dropna()
    t_end_vals = band_df["t_end"].dropna()

    if t_start_vals.empty or t_end_vals.empty:
        return None

    try:
        t_min = float(t_start_vals.min())
        t_max = float(t_end_vals.max())
    except (ValueError, TypeError):
        return None

    if not np.isfinite(t_min) or not np.isfinite(t_max) or t_max <= t_min:
        return None

    return (t_min, t_max)


def _extract_valid_time_point(row: pd.Series, key: str) -> Optional[float]:
    """Extract and validate a time point from a dataframe row."""
    value = row.get(key)
    if pd.isna(value):
        return None

    try:
        time_value = float(value)
    except (ValueError, TypeError):
        return None

    if not np.isfinite(time_value):
        return None

    return time_value


def _extract_channels_from_dataframe(band_df: pd.DataFrame) -> list[str]:
    """Extract and deduplicate channel names from cluster dataframe."""
    channels = []
    for channel_list in band_df.get("channels", []):
        if pd.isna(channel_list):
            continue
        channel_names = [ch.strip() for ch in str(channel_list).split(",") if ch.strip()]
        channels.extend(channel_names)

    return sorted(set(channels))


def _create_cluster_mask(
    band_df: pd.DataFrame,
    channels: list[str],
    time_grid: np.ndarray,
) -> np.ndarray:
    """Create binary mask indicating cluster coverage across channels and time."""
    mask = np.zeros((len(channels), len(time_grid) - 1), dtype=int)

    for _, row in band_df.iterrows():
        channel_list = row.get("channels", "")
        if pd.isna(channel_list):
            continue

        row_channels = [ch.strip() for ch in str(channel_list).split(",") if ch.strip()]

        t_start = _extract_valid_time_point(row, "t_start")
        t_end = _extract_valid_time_point(row, "t_end")
        if t_start is None or t_end is None:
            continue

        if t_end <= t_start:
            continue

        overlaps = (time_grid[:-1] < t_end) & (time_grid[1:] > t_start)
        if not np.any(overlaps):
            continue

        for channel in row_channels:
            try:
                channel_idx = channels.index(channel)
            except ValueError:
                continue
            mask[channel_idx, overlaps] = 1

    return mask


def _plot_cluster_ribbons(
    all_clusters: pd.DataFrame,
    band_order: list[str],
    subject: str,
    plot_cfg: Any,
) -> None:
    """Create ribbon plot showing cluster time ranges across frequency bands."""
    fig, ax = plt.subplots(figsize=plot_cfg.get_figure_size("wide", plot_type="behavioral"))

    for band_idx, band in enumerate(band_order):
        band_df = all_clusters[all_clusters["band"] == band]
        if band_df.empty:
            continue

        time_range = _extract_time_range(band_df)
        if time_range is not None:
            t_start_min, t_end_max = time_range
            ax.hlines(
                band_idx,
                t_start_min,
                t_end_max,
                colors=plot_cfg.get_color("light_gray"),
                linestyles="--",
                linewidth=0.5,
                alpha=0.5,
            )

        for _, row in band_df.iterrows():
            t_start = _extract_valid_time_point(row, "t_start")
            t_end = _extract_valid_time_point(row, "t_end")

            if t_start is None or t_end is None:
                continue

            width = t_end - t_start
            if width <= 0:
                continue

            is_significant = row.get("significant", False)
            color = (
                plot_cfg.get_color("significant")
                if is_significant
                else plot_cfg.get_color("nonsignificant")
            )

            ax.broken_barh(
                [(t_start, width)],
                (band_idx - 0.35, 0.7),
                facecolors=color,
                edgecolors="none",
                alpha=0.8,
            )

    ax.set_yticks(range(len(band_order)))
    ax.set_yticklabels(band_order)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Band")
    ax.set_title(f"Condition contrast cluster summary (sub-{subject})")

    legend_patches = [
        mpatches.Patch(color=plot_cfg.get_color("significant"), label="Significant"),
        mpatches.Patch(color=plot_cfg.get_color("nonsignificant"), label="Non-significant"),
    ]
    ax.legend(handles=legend_patches, loc="upper right", frameon=False)
    plt.tight_layout()

    return fig


def _plot_cluster_mask(
    band_df: pd.DataFrame,
    band: str,
    subject: str,
    plot_cfg: Any,
) -> None:
    """Create heatmap showing cluster coverage across channels and time."""
    time_range = _extract_time_range(band_df)
    if time_range is None:
        return

    t_min, t_max = time_range
    time_grid = np.linspace(t_min, t_max, TIME_GRID_POINTS)

    channels = _extract_channels_from_dataframe(band_df)
    if not channels:
        return

    mask = _create_cluster_mask(band_df, channels, time_grid)

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

    ax.set_title(f"Condition contrast clusters (band: {band}, sub-{subject})")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Channel")
    ax.set_yticks(np.arange(len(channels)) + 0.5)
    ax.set_yticklabels(channels)

    colorbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    colorbar.set_label("Significant cluster coverage")
    plt.tight_layout()

    return fig


def _save_figure(fig: plt.Figure, filepath: Path, plot_cfg: Any) -> None:
    """Save figure with configured settings."""
    save_fig(
        fig,
        filepath,
        formats=plot_cfg.formats,
        dpi=plot_cfg.dpi,
        bbox_inches=plot_cfg.bbox_inches,
        pad_inches=plot_cfg.pad_inches,
    )
    plt.close(fig)


def plot_pain_nonpain_clusters(
    subject: str,
    stats_dir: Path,
    plots_dir: Path,
    config: Any,
    logger: Optional[logging.Logger] = None,
    alpha: Optional[float] = None,
) -> None:
    """Plot pain vs non-pain cluster statistics across time and frequency bands."""
    logger = logger or _get_default_logger()
    plot_cfg = get_plot_config(config)
    ensure_dir(plots_dir)

    alpha_value = _get_alpha_value(config, alpha)

    all_clusters = _load_cluster_dataframes(stats_dir, alpha_value, logger)
    if all_clusters.empty:
        return

    significant_clusters = all_clusters[all_clusters["significant"]]
    if significant_clusters.empty:
        _log_if_present(logger, "warning", "No significant pain vs. non-pain clusters to plot.")
        return

    topomap_dir = plots_dir / "topomaps"
    ensure_dir(topomap_dir)

    significant_output_path = topomap_dir / "condition_contrast_clusters_significant.tsv"
    significant_clusters.to_csv(significant_output_path, sep="\t", index=False)
    _log_if_present(logger, "info", f"Saved significant cluster table to {significant_output_path}")

    band_order = _get_band_order(config, all_clusters)
    if not band_order:
        _log_if_present(logger, "warning", "No clusters available after filtering bands.")
        return

    ribbon_fig = _plot_cluster_ribbons(all_clusters, band_order, subject, plot_cfg)
    ribbon_path = topomap_dir / f"sub-{subject}_condition_cluster_ribbons"
    _save_figure(ribbon_fig, ribbon_path, plot_cfg)

    for band in band_order:
        band_df = significant_clusters[significant_clusters["band"] == band]
        if band_df.empty:
            continue

        mask_fig = _plot_cluster_mask(band_df, band, subject, plot_cfg)
        if mask_fig is None:
            continue

        mask_path = topomap_dir / f"sub-{subject}_condition_cluster_mask_{band}"
        _save_figure(mask_fig, mask_path, plot_cfg)
