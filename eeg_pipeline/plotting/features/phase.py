"""
Phase analysis visualization plotting functions.

Functions for creating phase-related plots including ITPC (Inter-Trial Phase Coherence)
heatmaps, topomaps, behavior scatter plots, and PAC (Phase-Amplitude Coupling)
comodulograms and time ribbons.
"""

import logging
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

from ...utils.io.general import ensure_dir, save_fig, log_if_present, get_column_from_config
from ..config import get_plot_config
from ...utils.analysis.stats import fdr_bh


def plot_itpc_heatmap(
    itpc_map: np.ndarray,
    freqs: np.ndarray,
    times: np.ndarray,
    subject: str,
    save_dir: Path,
    logger: logging.Logger,
    config: Any,
) -> None:
    """Plot ITPC heatmap averaged across channels.
    
    Args:
        itpc_map: ITPC map array (n_channels, n_freqs, n_times)
        freqs: Frequency array
        times: Time array
        subject: Subject identifier
        save_dir: Directory to save plot
        logger: Logger instance
        config: Configuration object
    """
    if itpc_map is None or freqs is None or times is None:
        log_if_present(logger, "warning", "ITPC map missing; skipping heatmap")
        return

    if itpc_map.ndim != 3:
        log_if_present(logger, "warning", f"Unexpected ITPC map shape {itpc_map.shape}; skipping heatmap")
        return

    plot_cfg = get_plot_config(config)
    itpc_avg = np.nanmean(itpc_map, axis=0)

    fig, ax = plt.subplots(figsize=plot_cfg.get_figure_size("standard", plot_type="tfr"))
    im = ax.imshow(
        itpc_avg,
        origin="lower",
        aspect="auto",
        extent=[float(times[0]), float(times[-1]), float(freqs[0]), float(freqs[-1])],
        cmap="magma",
        vmin=0.0,
        vmax=1.0,
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(f"ITPC (channel-mean, sub-{subject})")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("ITPC")

    ensure_dir(save_dir)
    output_name = save_dir / f"sub-{subject}_itpc_heatmap"
    save_fig(
        fig,
        output_name,
        formats=plot_cfg.formats,
        dpi=plot_cfg.dpi,
        bbox_inches=plot_cfg.bbox_inches,
        pad_inches=plot_cfg.pad_inches,
    )
    plt.close(fig)
    log_if_present(logger, "info", "Saved ITPC heatmap")


def plot_itpc_topomaps(
    itpc_df,
    info: mne.Info,
    subject: str,
    save_dir: Path,
    logger: logging.Logger,
    config: Any,
) -> None:
    """Plot ITPC topomaps in a grid organized by band and time bin.
    
    Args:
        itpc_df: DataFrame with columns: channel, band, time_bin, itpc
        info: MNE Info object
        subject: Subject identifier
        save_dir: Directory to save plot
        logger: Logger instance
        config: Configuration object
    """
    if itpc_df is None or len(itpc_df) == 0:
        log_if_present(logger, "warning", "ITPC dataframe empty; skipping topomaps")
        return

    required_cols = {"channel", "band", "time_bin", "itpc"}
    if not required_cols.issubset(set(itpc_df.columns)):
        log_if_present(logger, "warning", f"ITPC dataframe missing required columns {required_cols}; skipping topomaps")
        return

    plot_cfg = get_plot_config(config)
    ensure_dir(save_dir)

    bands = sorted(itpc_df["band"].unique())
    time_bins = sorted(itpc_df["time_bin"].unique())
    
    if not bands or not time_bins:
        log_if_present(logger, "warning", "No bands or time bins found in ITPC data")
        return

    n_rows = len(bands)
    n_cols = len(time_bins)
    
    fig_width = 4.5 * n_cols
    fig_height = 4.0 * n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), squeeze=False)
    
    topomap_data = {}
    all_values = []
    for (band, time_bin), df_sub in itpc_df.groupby(["band", "time_bin"]):
        ch_names = df_sub["channel"].astype(str).tolist()
        values = df_sub["itpc"].to_numpy(dtype=float)

        picks = mne.pick_channels(info.ch_names, include=ch_names, ordered=True)
        if len(picks) == 0:
            continue

        info_subset = mne.pick_info(info, picks)
        values_ordered = []
        for ch in info_subset.ch_names:
            try:
                idx = ch_names.index(ch)
                values_ordered.append(values[idx])
            except ValueError:
                values_ordered.append(np.nan)

        if not np.any(np.isfinite(values_ordered)):
            continue
            
        topomap_data[(band, time_bin)] = (values_ordered, info_subset)
        all_values.extend([v for v in values_ordered if np.isfinite(v)])
    
    shared_colorbar = bool(config.get("plotting.plots.itpc.shared_colorbar", True))
    if all_values and shared_colorbar:
        vmin = np.percentile(all_values, 5)
        vmax = np.percentile(all_values, 95)
        if np.isclose(vmin, vmax):
            vmin = np.min(all_values)
            vmax = np.max(all_values)
        if vmax - vmin < 0.01:
            center = (vmin + vmax) / 2
            vmin = center - 0.005
            vmax = center + 0.005
    else:
        vmin, vmax = 0.0, 1.0
    
    im = None
    for row_idx, band in enumerate(bands):
        for col_idx, time_bin in enumerate(time_bins):
            ax = axes[row_idx, col_idx]
            
            if (band, time_bin) in topomap_data:
                values_ordered, info_subset = topomap_data[(band, time_bin)]
                
                vmin_use, vmax_use = vmin, vmax
                if not shared_colorbar and np.any(np.isfinite(values_ordered)):
                    local = np.asarray(values_ordered)[np.isfinite(values_ordered)]
                    vmin_use = np.percentile(local, 5)
                    vmax_use = np.percentile(local, 95)
                    if np.isclose(vmin_use, vmax_use):
                        vmin_use = float(np.min(local))
                        vmax_use = float(np.max(local))
                    if vmax_use - vmin_use < 0.01:
                        center = (vmin_use + vmax_use) / 2
                        vmin_use = center - 0.005
                        vmax_use = center + 0.005

                im, _ = mne.viz.plot_topomap(
                    values_ordered,
                    info_subset,
                    axes=ax,
                    show=False,
                    cmap="viridis",
                    vlim=(vmin_use, vmax_use),
                    contours=6,
                )
                ax.set_title(f"{band} | {time_bin}")
            else:
                ax.axis("off")
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
    
    for row_idx, band in enumerate(bands):
        axes[row_idx, 0].set_ylabel(band.capitalize(), fontsize=12, fontweight="bold")
    
    for col_idx, time_bin in enumerate(time_bins):
        axes[0, col_idx].set_title(f"{time_bin}\n{axes[0, col_idx].get_title()}", fontweight="bold")
    
    if shared_colorbar and im is not None:
        fig.subplots_adjust(right=0.92)
        cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cbar_ax, label=f"ITPC\\n[{vmin:.3f}, {vmax:.3f}]")
    
    fig.suptitle(f"ITPC Topomaps (sub-{subject})", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 0.92, 0.96])
    
    output_name = save_dir / f"sub-{subject}_itpc_topomaps_grid"
    save_fig(
        fig,
        output_name,
        formats=plot_cfg.formats,
        dpi=plot_cfg.dpi,
        bbox_inches=plot_cfg.bbox_inches,
        pad_inches=plot_cfg.pad_inches,
    )
    plt.close(fig)
    log_if_present(logger, "info", f"Saved ITPC topomap grid ({n_rows} bands × {n_cols} time bins)")


def _parse_itpc_columns(columns):
    """Parse ITPC column names to extract band, time_bin, and channel.
    
    Args:
        columns: List of column names
    
    Returns:
        List of tuples (col_name, band, time_bin, channel)
    """
    parsed = []
    for col in columns:
        if not isinstance(col, str) or not col.startswith("itpc_"):
            continue
        parts = col.split("_")
        if len(parts) < 4:
            continue
        band = parts[1]
        time_bin = parts[-1]
        channel = "_".join(parts[2:-1])
        parsed.append((col, band, time_bin, channel))
    return parsed


def plot_itpc_behavior_scatter(
    itpc_df: pd.DataFrame,
    events_df: pd.DataFrame,
    subject: str,
    save_dir: Path,
    logger: logging.Logger,
    config: Any,
) -> None:
    """Plot ITPC vs behavior rating scatter plots organized by band and time bin.
    
    Args:
        itpc_df: DataFrame with ITPC columns
        events_df: Events DataFrame with rating column
        subject: Subject identifier
        save_dir: Directory to save plot
        logger: Logger instance
        config: Configuration object
    """
    if itpc_df is None or itpc_df.empty or events_df is None or events_df.empty:
        log_if_present(logger, "warning", "ITPC scatter: missing ITPC data or events; skipping")
        return

    rating_col = get_column_from_config(config, "event_columns.rating", events_df)
    if rating_col is None or rating_col not in events_df.columns:
        log_if_present(logger, "warning", "ITPC scatter: rating column not found; skipping")
        return

    parsed_cols = _parse_itpc_columns(itpc_df.columns)
    if not parsed_cols:
        log_if_present(logger, "warning", "ITPC scatter: no itpc_* columns found")
        return

    rating = pd.to_numeric(events_df[rating_col], errors="coerce")
    valid_mask = np.isfinite(rating)
    if valid_mask.sum() < 3:
        log_if_present(logger, "warning", "ITPC scatter: insufficient valid ratings")
        return

    plot_cfg = get_plot_config(config)
    bands = sorted({p[1] for p in parsed_cols})
    time_bins = sorted({p[2] for p in parsed_cols})
    n_rows, n_cols = len(bands), len(time_bins)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.0 * n_cols, 3.0 * n_rows), squeeze=False)
    marker_size = getattr(getattr(plot_cfg, "style", None), "scatter", getattr(plot_cfg, "style", None))
    marker_size_val = getattr(marker_size, "marker_size", 20) if marker_size is not None else 20
    marker_alpha_val = getattr(marker_size, "alpha", 0.6) if marker_size is not None else 0.6

    for r, band in enumerate(bands):
        for c, tbin in enumerate(time_bins):
            ax = axes[r, c]
            cols = [col for col, b, tb, _ in parsed_cols if b == band and tb == tbin]
            if not cols:
                ax.axis("off")
                continue
            values = pd.to_numeric(itpc_df[cols].mean(axis=1), errors="coerce")
            vals = values[valid_mask]
            rat = rating[valid_mask]
            if vals.size < 3:
                ax.axis("off")
                continue
            ax.scatter(
                rat,
                vals,
                s=marker_size_val,
                alpha=marker_alpha_val,
                color=plot_cfg.style.colors.gray if hasattr(plot_cfg.style, "colors") else "gray",
            )
            try:
                r_val = rat.corr(vals, method="spearman")
            except Exception:
                r_val = np.nan
            ax.set_title(f"{band} {tbin}\nρ={r_val:.2f}" if np.isfinite(r_val) else f"{band} {tbin}")
            ax.set_xlabel("Rating")
            ax.set_ylabel("ITPC (chan-mean)")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    fig.tight_layout()
    ensure_dir(save_dir)
    output = save_dir / f"sub-{subject}_itpc_rating_scatter"
    save_fig(
        fig,
        output,
        formats=plot_cfg.formats,
        dpi=plot_cfg.dpi,
        bbox_inches=plot_cfg.bbox_inches,
        pad_inches=plot_cfg.pad_inches,
    )
    plt.close(fig)
    log_if_present(logger, "info", "Saved ITPC rating scatter grid")


def _get_pac_plot_cfg(config):
    """Get PAC-specific plot configuration.
    
    Args:
        config: Configuration object
    
    Returns:
        Tuple of (plot_cfg, pac_cfg)
    """
    plot_cfg = get_plot_config(config)
    pac_cfg = plot_cfg.plot_type_configs.get("pac", {})
    return plot_cfg, pac_cfg


def plot_pac_comodulograms(
    pac_df: pd.DataFrame,
    subject: str,
    save_dir: Path,
    logger: logging.Logger,
    config: Any,
) -> None:
    """Plot PAC comodulograms (phase-amplitude coupling matrices) for each ROI.
    
    Args:
        pac_df: DataFrame with columns: roi, phase_freq, amp_freq, pac, (optional) p_perm, q_perm
        subject: Subject identifier
        save_dir: Directory to save plots
        logger: Logger instance
        config: Configuration object
    """
    if pac_df is None or pac_df.empty:
        log_if_present(logger, "warning", "PAC dataframe empty; skipping comodulograms")
        return

    required_cols = {"roi", "phase_freq", "amp_freq", "pac"}
    if not required_cols.issubset(set(pac_df.columns)):
        log_if_present(logger, "warning", f"PAC dataframe missing required columns {required_cols}; skipping")
        return

    plot_cfg, pac_plot_cfg = _get_pac_plot_cfg(config)
    cmap = pac_plot_cfg.get("cmap", "magma")
    alpha_sig = pac_plot_cfg.get("alpha_sig", config.get("statistics.sig_alpha", 0.05) if config else 0.05)
    
    all_pac_values = pac_df["pac"].dropna().values
    if len(all_pac_values) > 0:
        vmin = np.percentile(all_pac_values, 5)
        vmax = np.percentile(all_pac_values, 95)
        if np.isclose(vmin, vmax):
            vmin = np.min(all_pac_values)
            vmax = np.max(all_pac_values)
        if vmax - vmin < 0.01:
            center = (vmin + vmax) / 2
            vmin = max(0, center - 0.005)
            vmax = center + 0.005
    else:
        vmin, vmax = 0.0, 0.5

    ensure_dir(save_dir)
    for roi, df_roi in pac_df.groupby("roi"):
        if "q_perm" not in df_roi.columns and "p_perm" in df_roi.columns:
            pvals = pd.to_numeric(df_roi["p_perm"], errors="coerce")
            if pvals.notna().any():
                qvals = fdr_bh(pvals.to_numpy(dtype=float), config=config)
                df_roi = df_roi.copy()
                df_roi["q_perm"] = qvals
            else:
                df_roi = df_roi.copy()

        phase_freqs = np.sort(df_roi["phase_freq"].unique())
        amp_freqs = np.sort(df_roi["amp_freq"].unique())
        grid = df_roi.pivot(index="amp_freq", columns="phase_freq", values="pac").reindex(index=amp_freqs, columns=phase_freqs)

        fig, ax = plt.subplots(figsize=plot_cfg.get_figure_size("standard", plot_type="pac"))
        c = ax.pcolormesh(
            phase_freqs,
            amp_freqs,
            grid.to_numpy(),
            cmap=cmap,
            shading="auto",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_xlabel("Phase frequency (Hz)")
        ax.set_ylabel("Amplitude frequency (Hz)")
        ax.set_title(f"PAC Comodulogram (ROI: {roi}, sub-{subject})")
        cbar = plt.colorbar(c, ax=ax)
        cbar.set_label(f"PAC [{vmin:.3f}, {vmax:.3f}]")

        if "q_perm" in df_roi.columns:
            q_grid = df_roi.pivot(index="amp_freq", columns="phase_freq", values="q_perm").reindex(index=amp_freqs, columns=phase_freqs)
            sig_mask = (q_grid.to_numpy() <= alpha_sig)
            if np.any(sig_mask):
                ax.contour(phase_freqs, amp_freqs, sig_mask, colors="white", linewidths=1.0, linestyles="--")

        output_name = save_dir / f"sub-{subject}_pac_comod_{roi}"
        save_fig(
            fig,
            output_name,
            formats=plot_cfg.formats,
            dpi=plot_cfg.dpi,
            bbox_inches=plot_cfg.bbox_inches,
            pad_inches=plot_cfg.pad_inches,
        )
        plt.close(fig)
        log_if_present(logger, "info", f"Saved PAC comodulogram for ROI {roi}")


def plot_pac_time_ribbons(
    pac_time_df: pd.DataFrame,
    subject: str,
    save_dir: Path,
    logger: logging.Logger,
    config: Any,
) -> None:
    """Plot PAC over time (time x amp_freq) for each ROI and phase frequency.
    
    Args:
        pac_time_df: DataFrame with columns: roi, phase_freq, amp_freq, time, pac
        subject: Subject identifier
        save_dir: Directory to save plots
        logger: Logger instance
        config: Configuration object
    """
    if pac_time_df is None or pac_time_df.empty:
        return

    required_cols = {"roi", "phase_freq", "amp_freq", "time", "pac"}
    if not required_cols.issubset(set(pac_time_df.columns)):
        return

    plot_cfg, pac_plot_cfg = _get_pac_plot_cfg(config)
    cmap = pac_plot_cfg.get("cmap", "magma")
    ensure_dir(save_dir)

    for (roi, phase_f), df_sub in pac_time_df.groupby(["roi", "phase_freq"]):
        times = np.sort(df_sub["time"].unique())
        amp_freqs = np.sort(df_sub["amp_freq"].unique())
        grid = df_sub.pivot(index="amp_freq", columns="time", values="pac").reindex(index=amp_freqs, columns=times)

        fig, ax = plt.subplots(figsize=plot_cfg.get_figure_size("wide", plot_type="pac"))
        vmin = np.nanpercentile(grid.to_numpy().flatten(), 5) if np.any(np.isfinite(grid)) else 0.0
        vmax = np.nanpercentile(grid.to_numpy().flatten(), 95) if np.any(np.isfinite(grid)) else 1.0
        if np.isclose(vmin, vmax):
            vmax = vmin + 1e-3
        c = ax.pcolormesh(times, amp_freqs, grid.to_numpy(), cmap=cmap, shading="auto", vmin=vmin, vmax=vmax)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude frequency (Hz)")
        ax.set_title(f"PAC over time (ROI: {roi}, phase {phase_f:.1f} Hz)")
        plt.colorbar(c, ax=ax, label="PAC")

        out = save_dir / f"sub-{subject}_pac_time_roi-{roi}_phase-{phase_f:.1f}"
        save_fig(
            fig,
            out,
            formats=plot_cfg.formats,
            dpi=plot_cfg.dpi,
            bbox_inches=plot_cfg.bbox_inches,
            pad_inches=plot_cfg.pad_inches,
        )
        plt.close(fig)
        log_if_present(logger, "info", f"Saved PAC time ribbon for ROI {roi}, phase {phase_f:.1f}")


def plot_pac_behavior_scatter(
    pac_trials_df: pd.DataFrame,
    events_df: pd.DataFrame,
    subject: str,
    save_dir: Path,
    logger: logging.Logger,
    config: Any,
) -> None:
    """Plot PAC vs behavior (rating/temperature) scatter plots.
    
    Args:
        pac_trials_df: DataFrame with trial-level PAC data
        events_df: Events DataFrame with rating and optionally temperature
        subject: Subject identifier
        save_dir: Directory to save plot
        logger: Logger instance
        config: Configuration object
    """
    if pac_trials_df is None or pac_trials_df.empty or events_df is None or events_df.empty:
        return

    rating_col = get_column_from_config(config, "event_columns.rating", events_df)
    if rating_col is None or rating_col not in events_df.columns:
        log_if_present(logger, "warning", "No rating column for PAC-behavior scatter; skipping")
        return

    ratings = pd.to_numeric(events_df[rating_col], errors="coerce")
    temp_col = get_column_from_config(config, "event_columns.temperature", events_df)
    temps = pd.to_numeric(events_df[temp_col], errors="coerce") if temp_col and temp_col in events_df.columns else None

    pac_plot_cfg = config.get("feature_engineering.pac", {})
    target_phase = pac_plot_cfg.get("plot_target_phase_hz")
    target_amp = pac_plot_cfg.get("plot_target_amp_hz")

    unique_pairs = pac_trials_df[["phase_freq", "amp_freq"]].drop_duplicates()
    if unique_pairs.empty:
        return

    if target_phase is None or target_amp is None:
        # Auto-select most common pair to still produce a plot
        pair_counts = pac_trials_df.groupby(["phase_freq", "amp_freq"]).size().reset_index(name="count")
        top_pair = pair_counts.sort_values("count", ascending=False).iloc[0]
        target_phase, target_amp = float(top_pair["phase_freq"]), float(top_pair["amp_freq"])
        log_if_present(
            logger,
            "info",
            f"PAC-behavior scatter: using most frequent pair ({target_phase}->{target_amp} Hz) "
            "because plot_target_phase_hz/plot_target_amp_hz not set.",
        )

    target_pair = pd.Series({"phase_freq": float(target_phase), "amp_freq": float(target_amp)})
    mask_pair = (
        (pac_trials_df["phase_freq"] == target_pair["phase_freq"])
        & (pac_trials_df["amp_freq"] == target_pair["amp_freq"])
    )
    df_pair = pac_trials_df.loc[mask_pair].copy()
    if df_pair.empty:
        log_if_present(logger, "warning", f"PAC-behavior scatter: target pair {target_phase}->{target_amp} Hz not found; skipping.")
        return

    merged = df_pair.merge(
        ratings.rename("rating"),
        left_on="trial",
        right_index=True,
        how="left",
    )
    if merged["rating"].isna().any() or merged["trial"].max() >= len(ratings):
        log_if_present(logger, "warning", "PAC-behavior scatter: trial indices do not align with ratings; skipping.")
        return
    if temps is not None:
        merged = merged.merge(temps.rename("temperature"), left_on="trial", right_index=True, how="left")

    plot_cfg, _ = _get_pac_plot_cfg(config)
    fig, axes = plt.subplots(1, 2 if temps is not None else 1, figsize=plot_cfg.get_figure_size("wide", plot_type="behavioral"))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    roi_groups = merged.groupby("roi")
    panels: Dict[str, str] = {"rating": "Rating"}
    if temps is not None:
        panels["temperature"] = "Temperature"

    for ax_idx, (col_key, label) in enumerate(panels.items()):
        ax = axes[ax_idx]
        for roi, df_roi in roi_groups:
            ax.scatter(df_roi[col_key], df_roi["pac"], alpha=0.6, label=roi, s=30)
        valid = merged[[col_key, "pac"]].dropna()
        if len(valid) >= 3:
            r = valid[col_key].corr(valid["pac"], method="spearman")
            ax.text(0.05, 0.95, f"Spearman r={r:.2f}", transform=ax.transAxes, va="top", fontsize=plot_cfg.font.small)
        ax.set_xlabel(label)
        ax.set_ylabel(f"PAC ({float(target_pair['phase_freq']):.1f}→{float(target_pair['amp_freq']):.1f} Hz)")
        ax.set_title(f"PAC vs {label} (sub-{subject})")
        ax.grid(alpha=0.3)

    if len(axes) > 1:
        axes[0].legend(title="ROI", fontsize=plot_cfg.font.small)

    ensure_dir(save_dir)
    output_name = save_dir / f"sub-{subject}_pac_behavior_scatter"
    save_fig(
        fig,
        output_name,
        formats=plot_cfg.formats,
        dpi=plot_cfg.dpi,
        bbox_inches=plot_cfg.bbox_inches,
        pad_inches=plot_cfg.pad_inches,
    )
    plt.close(fig)
    log_if_present(logger, "info", "Saved PAC-behavior scatter")
