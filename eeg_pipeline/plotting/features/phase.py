"""
Phase analysis visualization plotting functions.

Functions for creating phase-related plots including ITPC (Inter-Trial Phase Coherence)
heatmaps, topomaps, behavior scatter plots, and PAC (Phase-Amplitude Coupling)
comodulograms and time ribbons.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

from eeg_pipeline.infra.paths import ensure_dir
from eeg_pipeline.plotting.io.figures import save_fig, log_if_present
from ..config import get_plot_config
from ...utils.analysis.stats import fdr_bh
from eeg_pipeline.utils.analysis.events import extract_pain_mask
from scipy import stats


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
        itpc_df: DataFrame with either:
            - Long format columns: channel, band, time_bin, itpc
            - Wide format columns: itpc_{segment}_{band}_ch_{channel}_val
        info: MNE Info object
        subject: Subject identifier
        save_dir: Directory to save plot
        logger: Logger instance
        config: Configuration object
    """
    if itpc_df is None or len(itpc_df) == 0:
        log_if_present(logger, "warning", "ITPC dataframe empty; skipping topomaps")
        return

    # Check if already in long format
    required_cols = {"channel", "band", "time_bin", "itpc"}
    if required_cols.issubset(set(itpc_df.columns)):
        df_long = itpc_df
    else:
        # Convert from wide format: itpc_{segment}_{band}_ch_{channel}_val
        itpc_cols = [c for c in itpc_df.columns if c.startswith("itpc_") and "_ch_" in c]
        if not itpc_cols:
            log_if_present(logger, "warning", "No ITPC columns found in wide format; skipping topomaps")
            return
        
        rows = []
        for col in itpc_cols:
            # Parse column name: itpc_{segment}_{band}_ch_{channel}_val
            parts = col.replace("itpc_", "").replace("_val", "").split("_ch_")
            if len(parts) != 2:
                continue
            segment_band = parts[0]
            channel = parts[1]
            # Split segment and band
            sb_parts = segment_band.split("_")
            if len(sb_parts) >= 2:
                segment = sb_parts[0]
                band = "_".join(sb_parts[1:])  # Handle multi-word bands
                # Use mean across trials as the ITPC value for topomap
                itpc_val = itpc_df[col].mean()
                rows.append({
                    "channel": channel,
                    "band": band,
                    "time_bin": segment,  # Use segment as time_bin
                    "itpc": itpc_val
                })
        
        if not rows:
            log_if_present(logger, "warning", "Could not parse ITPC columns; skipping topomaps")
            return
        
        df_long = pd.DataFrame(rows)

    plot_cfg = get_plot_config(config)
    ensure_dir(save_dir)

    bands = sorted(df_long["band"].unique())
    time_bins = sorted(df_long["time_bin"].unique())
    
    if not bands or not time_bins:
        log_if_present(logger, "warning", "No bands or time bins found in ITPC data")
        return

    n_rows = len(bands)
    n_cols = len(time_bins)
    
    width_per_state = float(plot_cfg.plot_type_configs.get("itpc", {}).get("width_per_bin", 4.5))
    height_per_state = float(plot_cfg.plot_type_configs.get("itpc", {}).get("height_per_band", 4.0))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(width_per_state * n_cols, height_per_state * n_rows), squeeze=False)
    
    topomap_data = {}
    all_values = []
    for (band, time_bin), df_sub in df_long.groupby(["band", "time_bin"]):
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
        axes[row_idx, 0].set_ylabel(band.capitalize(), fontsize=plot_cfg.font.figure_title, fontweight="bold")
    
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
    from eeg_pipeline.domain.features.naming import NamingSchema

    parsed = []
    for col in columns:
        parsed_name = NamingSchema.parse(str(col))
        if not parsed_name.get("valid"):
            continue
        if parsed_name.get("group") != "itpc":
            continue
        band = parsed_name.get("band")
        time_bin = parsed_name.get("segment")
        channel = parsed_name.get("identifier")
        parsed.append((col, band, time_bin, channel))
    return parsed


def plot_itpc_by_condition(
    itpc_df: pd.DataFrame,
    events_df: pd.DataFrame,
    subject: str,
    save_dir: Path,
    logger: logging.Logger,
    config: Any,
) -> None:
    """Compare ITPC between Pain and Non-pain conditions using box+strip.
    
    Statistical improvements:
    - Shows both raw p-value and FDR-corrected q-value
    - Includes bootstrap 95% CI for mean difference
    - Reports Cohen's d effect size
    - Footer shows total tests and correction method
    """
    if itpc_df is None or itpc_df.empty or events_df is None:
        return
    
    pain_mask = extract_pain_mask(events_df, config)
    if pain_mask is None:
        return
    
    from eeg_pipeline.domain.features.naming import NamingSchema
    from eeg_pipeline.plotting.features.utils import (
        compute_condition_stats,
        apply_fdr_correction,
        format_stats_annotation,
        format_footer_annotation,
        get_band_colors,
        get_band_names,
        get_condition_colors,
    )

    condition_colors = get_condition_colors(config)
    band_colors = get_band_colors(config)

    itpc_entries = []
    band_set = set()
    segment_set = set()
    for col in itpc_df.columns:
        parsed = NamingSchema.parse(str(col))
        if not parsed.get("valid"):
            continue
        if parsed.get("group") != "itpc":
            continue
        segment = parsed.get("segment")
        band = parsed.get("band")
        if not segment or not band:
            continue
        itpc_entries.append(
            (
                str(col),
                str(band),
                str(segment),
                str(parsed.get("scope") or ""),
            )
        )
        band_set.add(str(band))
        segment_set.add(str(segment))

    if not itpc_entries or not band_set:
        return

    segment = "active" if "active" in segment_set else sorted(segment_set)[0]
    itpc_entries = [e for e in itpc_entries if e[2] == segment]
    band_set = {e[1] for e in itpc_entries}

    band_order = get_band_names(config)
    bands = [b for b in band_order if b in band_set]
    bands += [b for b in sorted(band_set) if b not in bands]
    if not bands:
        return

    plot_cfg = get_plot_config(config)
    n_bands = len(bands)

    # Calculate figure size dynamically
    width_per_band = float(plot_cfg.plot_type_configs.get("itpc", {}).get("width_per_band_box", 4.0))
    fig_height = float(plot_cfg.plot_type_configs.get("itpc", {}).get("height_box", 5.0))
    figsize = (width_per_band * n_bands, fig_height)
    
    all_stats = []
    all_pvals = []
    band_data = {}
    
    for band in bands:
        band_cols = [c for c, b, _, scope in itpc_entries if b == band and scope == "global"]
        if not band_cols:
            band_cols = [c for c, b, _, scope in itpc_entries if b == band and scope == "roi"]
        if not band_cols:
            band_cols = [c for c, b, _, scope in itpc_entries if b == band and scope == "ch"]
        
        if not band_cols:
            band_data[band] = None
            continue
        
        mean_itpc = itpc_df[band_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
        vals_pain = mean_itpc[pain_mask].dropna().values
        vals_nonpain = mean_itpc[~pain_mask].dropna().values
        
        if len(vals_pain) >= 3 and len(vals_nonpain) >= 3:
            stats_result = compute_condition_stats(vals_nonpain, vals_pain, n_boot=1000, config=config)
            all_stats.append(stats_result)
            all_pvals.append(stats_result["p_raw"])
            band_data[band] = {
                "vals_nonpain": vals_nonpain,
                "vals_pain": vals_pain,
                "stats": stats_result,
                "stats_idx": len(all_stats) - 1,
            }
        else:
            band_data[band] = {
                "vals_nonpain": vals_nonpain,
                "vals_pain": vals_pain,
                "stats": None,
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
    
    fig, axes = plt.subplots(1, n_bands, figsize=figsize, squeeze=False)
    axes = axes.flatten()
    
    for idx, band in enumerate(bands):
        ax = axes[idx]
        data = band_data.get(band)
        
        if data is None:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue
        
        vals_nonpain = data["vals_nonpain"]
        vals_pain = data["vals_pain"]
        
        if len(vals_pain) < 3 or len(vals_nonpain) < 3:
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center", transform=ax.transAxes)
            continue
        
        bp = ax.boxplot([vals_nonpain, vals_pain], positions=[0, 1], widths=0.4, patch_artist=True)
        bp["boxes"][0].set_facecolor(condition_colors["nonpain"])
        bp["boxes"][0].set_alpha(0.6)
        bp["boxes"][1].set_facecolor(condition_colors["pain"])
        bp["boxes"][1].set_alpha(0.6)
        
        ax.scatter(np.random.uniform(-0.1, 0.1, len(vals_nonpain)), 
                  vals_nonpain, c=condition_colors["nonpain"], alpha=0.4, s=12)
        ax.scatter(1 + np.random.uniform(-0.1, 0.1, len(vals_pain)), 
                  vals_pain, c=condition_colors["pain"], alpha=0.4, s=12)
        
        if data.get("stats") is not None:
            s = data["stats"]
            annotation = format_stats_annotation(
                p_raw=s["p_raw"],
                q_fdr=s.get("q_fdr"),
                cohens_d=s["cohens_d"],
                ci_low=s["ci_low"],
                ci_high=s["ci_high"],
                compact=True,
            )
            text_color = plot_cfg.style.colors.significant if s.get("fdr_significant", False) else plot_cfg.style.colors.gray
            ax.text(0.5, 0.98, annotation, ha="center", va="top", 
                   transform=ax.transAxes, fontsize=plot_cfg.font.annotation, color=text_color,
                   bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
        
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["NP", "P"], fontsize=plot_cfg.font.large)
        ax.set_ylabel("Mean ITPC")
        ax.set_title(f"{band.capitalize()}", fontweight="bold", 
                    color=band_colors.get(band, "#333333"))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    
    n_pain = int(pain_mask.sum())
    n_nonpain = int((~pain_mask).sum())
    fig.suptitle(
        f"ITPC by Condition ({segment}, sub-{subject})\nN: {n_nonpain} NP, {n_pain} P",
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
        additional_info="Mann-Whitney U | Bootstrap 95% CI | †=FDR significant"
    )
    fig.text(0.5, 0.01, footer, ha="center", va="bottom", fontsize=8, color="gray")
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    ensure_dir(save_dir)
    save_fig(
        fig,
        save_dir / f"sub-{subject}_itpc_by_condition",
        formats=plot_cfg.formats,
        dpi=plot_cfg.dpi,
        bbox_inches=plot_cfg.bbox_inches,
        pad_inches=plot_cfg.pad_inches
    )
    plt.close(fig)
    log_if_present(logger, "info", f"Saved ITPC by condition ({n_significant}/{n_tests} FDR significant)")



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
    """Plot PAC comodulograms (phase-amplitude coupling matrices) as a single combined figure.
    
    Creates a grid of comodulograms for all ROIs in a single figure to reduce file count
    and provide a comprehensive overview.
    
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

    from ...utils.config.loader import get_config_value, ensure_config
    config = ensure_config(config)
    plot_cfg, pac_plot_cfg = _get_pac_plot_cfg(config)
    cmap = pac_plot_cfg.get("cmap", "magma")
    alpha_sig = pac_plot_cfg.get("alpha_sig", get_config_value(config, "statistics.sig_alpha", 0.05))
    
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
    
    # Get unique ROIs
    rois = sorted(pac_df["roi"].unique())
    n_rois = len(rois)
    
    if n_rois == 0:
        log_if_present(logger, "warning", "No ROIs found in PAC data")
        return
    
    # Create grid layout
    n_cols = min(3, n_rois)
    n_rows = (n_rois + n_cols - 1) // n_cols
    
    width_per_col = float(pac_plot_cfg.get("width_per_col", 4.0))
    height_per_row = float(pac_plot_cfg.get("height_per_row", 3.5))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(width_per_col * n_cols, height_per_row * n_rows), squeeze=False)
    
    for idx, roi in enumerate(rois):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]
        
        df_roi = pac_df[pac_df["roi"] == roi].copy()
        
        if "q_perm" not in df_roi.columns and "p_perm" in df_roi.columns:
            pvals = pd.to_numeric(df_roi["p_perm"], errors="coerce")
            if pvals.notna().any():
                qvals = fdr_bh(pvals.to_numpy(dtype=float), config=config)
                df_roi["q_perm"] = qvals

        phase_freqs = np.sort(df_roi["phase_freq"].unique())
        amp_freqs = np.sort(df_roi["amp_freq"].unique())
        
        if len(phase_freqs) == 0 or len(amp_freqs) == 0:
            ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(roi[:20], fontsize=plot_cfg.font.title)
            continue
            
        grid = df_roi.pivot(index="amp_freq", columns="phase_freq", values="pac").reindex(index=amp_freqs, columns=phase_freqs)

        c = ax.pcolormesh(
            phase_freqs,
            amp_freqs,
            grid.to_numpy(),
            cmap=cmap,
            shading="auto",
            vmin=vmin,
            vmax=vmax,
        )
        
        # Truncate long ROI names
        roi_short = roi[:20] + "..." if len(roi) > 20 else roi
        ax.set_title(roi_short, fontsize=plot_cfg.font.title, fontweight='bold')
        
        if row == n_rows - 1:
            ax.set_xlabel("Phase (Hz)", fontsize=plot_cfg.font.large)
        if col == 0:
            ax.set_ylabel("Amp (Hz)", fontsize=plot_cfg.font.large)
        
        ax.tick_params(labelsize=8)

        if "q_perm" in df_roi.columns:
            q_grid = df_roi.pivot(index="amp_freq", columns="phase_freq", values="q_perm").reindex(index=amp_freqs, columns=phase_freqs)
            sig_mask = (q_grid.to_numpy() <= alpha_sig)
            if np.any(sig_mask):
                ax.contour(phase_freqs, amp_freqs, sig_mask, colors="white", linewidths=0.8, linestyles="--")
    
    # Hide empty subplots
    for idx in range(n_rois, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].axis('off')
    
    # Add colorbar
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(c, cax=cbar_ax)
    cbar.set_label("PAC (MI)", fontsize=plot_cfg.font.title)
    
    fig.suptitle(f"Phase-Amplitude Coupling Comodulograms\nsub-{subject} | {n_rois} ROIs",
                fontsize=plot_cfg.font.figure_title, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 0.88, 0.95])
    
    output_name = save_dir / f"sub-{subject}_pac_comodulograms_grid"
    save_fig(
        fig,
        output_name,
        formats=plot_cfg.formats,
        dpi=plot_cfg.dpi,
        bbox_inches=plot_cfg.bbox_inches,
        pad_inches=plot_cfg.pad_inches,
    )
    plt.close(fig)
    log_if_present(logger, "info", f"Saved PAC comodulograms grid ({n_rois} ROIs)")


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




def plot_pac_by_condition(
    pac_trials_df: pd.DataFrame,
    events_df: pd.DataFrame,
    subject: str,
    save_dir: Path,
    logger: logging.Logger,
    config: Any,
) -> None:
    """Compare PAC between Pain and Non-pain using box+strip plots."""
    if pac_trials_df is None or pac_trials_df.empty or events_df is None:
        return

    pain_mask = extract_pain_mask(events_df, config)
    if pain_mask is None:
        return
    
    # Check if DataFrame has required 'roi' column (long format)
    if 'roi' not in pac_trials_df.columns:
        log_if_present(logger, "debug", "PAC DataFrame missing 'roi' column - skipping condition plot")
        return
        
    if len(pac_trials_df) % len(events_df) != 0:
        return
    
    rois = sorted(pac_trials_df['roi'].unique())
    n_rois = len(rois)
    if n_rois == 0:
        return
    
    condition_colors = get_condition_colors(config)
    plot_cfg = get_plot_config(config)
    
    # Calculate figure size dynamically
    width_per_roi = float(plot_cfg.plot_type_configs.get("pac", {}).get("width_per_roi", 4.0))
    fig_height = float(plot_cfg.plot_type_configs.get("pac", {}).get("height_box", 5.0))
    fig, axes = plt.subplots(1, n_rois, figsize=(width_per_roi * n_rois, fig_height), squeeze=False)
    axes = axes.flatten()
        
    for i, roi in enumerate(rois):
        ax = axes[i]
        
        df_roi = pac_trials_df[pac_trials_df['roi'] == roi]
        
        if 'trial_idx' not in df_roi.columns:
            ax.text(0.5, 0.5, "No trial data", ha="center", va="center", transform=ax.transAxes)
            continue
            
        trial_pac = df_roi.groupby('trial_idx')['pac'].mean().reindex(range(len(events_df)), fill_value=np.nan)
        
        vals_pain = trial_pac[pain_mask].dropna().values
        vals_nonpain = trial_pac[~pain_mask].dropna().values
        
        if len(vals_pain) < 3 or len(vals_nonpain) < 3:
            ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center", transform=ax.transAxes)
            continue
        
        bp = ax.boxplot([vals_nonpain, vals_pain], positions=[0, 1], widths=0.4, patch_artist=True)
        bp["boxes"][0].set_facecolor(condition_colors["nonpain"])
        bp["boxes"][0].set_alpha(0.6)
        bp["boxes"][1].set_facecolor(condition_colors["pain"])
        bp["boxes"][1].set_alpha(0.6)
        
        ax.scatter(np.random.uniform(-0.1, 0.1, len(vals_nonpain)), 
                  vals_nonpain, c=condition_colors["nonpain"], alpha=0.4, s=12)
        ax.scatter(1 + np.random.uniform(-0.1, 0.1, len(vals_pain)), 
                  vals_pain, c=condition_colors["pain"], alpha=0.4, s=12)
        
        try:
            _, p = stats.mannwhitneyu(vals_nonpain, vals_pain)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            if sig:
                ax.text(0.5, 0.95, f"p={p:.3f}{sig}", ha="center", va="top", 
                       transform=ax.transAxes, fontsize=plot_cfg.font.large)
        except ValueError:
            pass
        
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Non-pain", "Pain"], fontsize=plot_cfg.font.large)
        ax.set_ylabel("Mean PAC")
        ax.set_title(f"ROI: {roi[:15]}" if len(roi) > 15 else f"ROI: {roi}", fontsize=plot_cfg.font.title)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    
    fig.suptitle("PAC by Condition", fontsize=plot_cfg.font.figure_title, fontweight="bold", y=1.02)
        
    plt.tight_layout()
    save_fig(fig, save_dir / f"sub-{subject}_pac_by_condition", 
             formats=plot_cfg.formats, dpi=plot_cfg.dpi)
    plt.close(fig)
    log_if_present(logger, "info", "Saved PAC by condition comparison")


def convert_pac_wide_to_long(pac_df: pd.DataFrame, logger: Optional[logging.Logger] = None) -> Optional[pd.DataFrame]:
    """Convert PAC data from wide format to long format for comodulograms.
    
    This function parses standard PAC column names (pac_segment_phase_amp_ch_stat)
    and produces a simplified DataFrame with [roi, phase_freq, amp_freq, pac].
    Use this if you have flat feature files and need to reconstruct the matrix structure.
    """
    records = []
    
    # Heuristic mapping if real freqs not in name
    # If the band naming scheme changes, this needs update or config lookup
    # This assumes standard "delta", "theta", etc. names.
    # It attempts to map them to center frequencies.
    phase_freq_map = {'delta': 2.5, 'theta': 6, 'alpha': 10, 'beta': 20, 'gamma': 40}
    amp_freq_map = {'delta': 2.5, 'theta': 6, 'alpha': 10, 'beta': 20, 'gamma': 40}
    
    # Iterate columns
    for col in pac_df.columns:
        # Expected: pac_active_theta_beta_ch_Fp1_pac
        # or pac_active_theta_beta_ch_Fp1_mi
        if not str(col).startswith('pac_'):
            continue
            
        parts = str(col).split('_')
        # parts: [pac, <segment>, <phase>, <amp>, ..., val]
        # We need identifying band names.
        
        # Heuristic: look for band names in parts
        phase_band = None
        amp_band = None
        
        # Try to find bands from known maps
        # Note: this is fragile if band names are non-standard or duplicates
        # We assume order: phase then amp
        parts_bands = [p for p in parts if p in phase_freq_map]
        if len(parts_bands) >= 2:
            phase_band = parts_bands[0]
            amp_band = parts_bands[1]
        
        if not phase_band or not amp_band:
            continue
            
        # ROI?
        # Look for "ch" or "ch_"
        roi = "Global"
        if "ch" in parts:
            idx = parts.index("ch")
            if idx + 1 < len(parts):
                roi = parts[idx+1]
        elif any(p.startswith("chpair") for p in parts):
             # PAC usually done per channel but could be cross-channel
             # If cross-channel, maybe skip or handle
             continue
             
        val = pac_df[col].mean() # Average over trials if multiple rows, or single row
        
        if pd.notna(val):
            records.append({
                'roi': roi,
                'phase_freq': phase_freq_map[phase_band],
                'amp_freq': amp_freq_map[amp_band],
                'pac': val
            })
            
    if not records:
        if logger:
             logger.warning("No PAC columns matched heuristic parsing for wide-to-long conversion")
        return None
        
    return pd.DataFrame(records)


def plot_pac_summary(
    pac_df: pd.DataFrame,
    subject: str,
    save_dir: Path,
    logger: logging.Logger,
    config: Any,
) -> None:
    """PAC summary showing aggregate coupling across frequency pairs.
    
    Answers: "What is the overall PAC profile?"
    """
    if pac_df is None or pac_df.empty:
        return
    
    plot_cfg = get_plot_config(config)
    
    pac_cols = [c for c in pac_df.columns if c.startswith("pac_") or "pac" in c.lower()]
    
    if not pac_cols:
        return
    
    phase_bands = ["delta", "theta", "alpha"]
    amp_bands = ["alpha", "beta", "gamma", "highgamma"]
    
    pac_matrix = np.zeros((len(phase_bands), len(amp_bands)))
    pac_counts = np.zeros((len(phase_bands), len(amp_bands)))
    
    for col in pac_cols:
        col_lower = col.lower()
        for pi, phase in enumerate(phase_bands):
            for ai, amp in enumerate(amp_bands):
                if phase in col_lower and amp in col_lower:
                    vals = pac_df[col].dropna().values
                    if len(vals) > 0:
                        pac_matrix[pi, ai] += np.mean(vals)
                        pac_counts[pi, ai] += 1
    
    pac_counts[pac_counts == 0] = 1
    pac_matrix = pac_matrix / pac_counts
    
    if pac_matrix.sum() == 0:
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    ax1 = axes[0]
    im = ax1.imshow(pac_matrix, cmap="hot", aspect="auto")
    ax1.set_xticks(range(len(amp_bands)))
    ax1.set_xticklabels([b.capitalize() for b in amp_bands])
    ax1.set_yticks(range(len(phase_bands)))
    ax1.set_yticklabels([b.capitalize() for b in phase_bands])
    ax1.set_xlabel("Amplitude Band")
    ax1.set_ylabel("Phase Band")
    ax1.set_title("Mean PAC Matrix", fontweight="bold")
    plt.colorbar(im, ax=ax1, shrink=0.8, label="PAC")
    
    for i in range(len(phase_bands)):
        for j in range(len(amp_bands)):
            if pac_matrix[i, j] > 0:
                ax1.text(j, i, f"{pac_matrix[i, j]:.3f}", ha="center", va="center", 
                        fontsize=8, color="white" if pac_matrix[i, j] > pac_matrix.max()/2 else "black")
    
    ax2 = axes[1]
    all_pac_vals = []
    for col in pac_cols:
        vals = pac_df[col].dropna().values
        all_pac_vals.extend(vals)
    
    if all_pac_vals:
        ax2.hist(all_pac_vals, bins=30, color="#14B8A6", alpha=0.7, edgecolor="white")
        ax2.axvline(np.mean(all_pac_vals), color="red", linestyle="--", linewidth=2,
                   label=f"Mean: {np.mean(all_pac_vals):.3f}")
        ax2.axvline(np.median(all_pac_vals), color="orange", linestyle=":", linewidth=2,
                   label=f"Median: {np.median(all_pac_vals):.3f}")
        ax2.set_xlabel("PAC Value")
        ax2.set_ylabel("Count")
        ax2.set_title("PAC Distribution", fontweight="bold")
        ax2.legend()
    
    ax3 = axes[2]
    phase_means = pac_matrix.mean(axis=1)
    amp_means = pac_matrix.mean(axis=0)
    
    x = np.arange(max(len(phase_bands), len(amp_bands)))
    width = 0.35
    
    ax3.bar(x[:len(phase_bands)] - width/2, phase_means, width, label="Phase", color="#3B82F6", alpha=0.8)
    ax3.bar(x[:len(amp_bands)] + width/2, amp_means, width, label="Amplitude", color="#22C55E", alpha=0.8)
    
    all_labels = phase_bands + amp_bands[len(phase_bands):]
    ax3.set_xticks(x[:len(all_labels)])
    ax3.set_xticklabels([b[:3].capitalize() for b in all_labels])
    ax3.set_ylabel("Mean PAC")
    ax3.set_title("PAC by Band Role", fontweight="bold")
    ax3.legend()
    
    fig.suptitle(f"Phase-Amplitude Coupling Summary (sub-{subject})", 
                fontsize=plot_cfg.font.figure_title, fontweight="bold", y=1.02)
    
    max_pac = pac_matrix.max()
    max_idx = np.unravel_index(pac_matrix.argmax(), pac_matrix.shape)
    max_pair = f"{phase_bands[max_idx[0]]}-{amp_bands[max_idx[1]]}"
    footer = f"n={len(pac_df)} trials | Strongest coupling: {max_pair} (PAC={max_pac:.3f})"
    fig.text(0.5, 0.01, footer, ha="center", va="bottom", fontsize=plot_cfg.font.small, color="gray")
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    ensure_dir(save_dir)
    save_fig(fig, save_dir / f"sub-{subject}_pac_summary",
             formats=plot_cfg.formats, dpi=plot_cfg.dpi,
             bbox_inches=plot_cfg.bbox_inches, pad_inches=plot_cfg.pad_inches)
    plt.close(fig)
    
    if logger:
        logger.info(f"Saved PAC summary (max: {max_pair}={max_pac:.3f})")
