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


###################################################################
# ERDS Condition Comparison
###################################################################

def plot_erds_by_condition(
    features_df: pd.DataFrame,
    events_df: pd.DataFrame,
    subject: str,
    save_dir: Path,
    logger: Any,
    config: Any,
    stats_dir: Optional[Path] = None,
) -> None:
    """Compare ERDS between conditions per band.
    
    For window comparisons (paired): Uses the unified plot_paired_comparison helper.
    For column comparisons (unpaired): Uses Mann-Whitney U test with consistent styling.
    Creates one figure per ROI.
    
    If stats_dir is provided, uses pre-computed statistics from the behavior pipeline.
    """
    import logging
    from scipy import stats
    from eeg_pipeline.infra.paths import ensure_dir
    from eeg_pipeline.utils.config.loader import get_config_value
    from eeg_pipeline.utils.analysis.events import extract_comparison_mask
    from eeg_pipeline.plotting.features.utils import (
        plot_paired_comparison,
        apply_fdr_correction,
        get_named_segments,
        get_band_color,
    )
    from eeg_pipeline.plotting.features.roi import get_roi_definitions, get_roi_channels
    from eeg_pipeline.plotting.io.figures import log_if_present
    
    if features_df is None or features_df.empty or events_df is None:
        return

    compare_wins = get_config_value(config, "plotting.comparisons.compare_windows", True)
    compare_cols = get_config_value(config, "plotting.comparisons.compare_columns", False)
    
    # Get segments from config or auto-detect from data
    segments = get_config_value(config, "plotting.comparisons.comparison_windows", [])
    if not segments or len(segments) < 2:
        detected = _get_erds_segments(features_df)
        if len(detected) >= 2:
            segments = detected[:2]
            if logger:
                log_if_present(logger, "info", f"Auto-detected segments for ERDS comparison: {segments}")
    
    bands = get_band_names(config)
    if not bands:
        return
    
    # Get ROI definitions
    rois = get_roi_definitions(config)
    all_channels = set()
    for col in features_df.columns:
        parsed = NamingSchema.parse(str(col))
        if parsed.get("valid") and parsed.get("group") == "erds" and parsed.get("scope") == "ch":
            ch = parsed.get("identifier")
            if ch:
                all_channels.add(str(ch))
    all_channels = list(all_channels)
    
    comp_rois = get_config_value(config, "plotting.comparisons.comparison_rois", [])
    if comp_rois:
        roi_names = []
        for r in comp_rois:
            if r.lower() == "all":
                if "all" not in roi_names:
                    roi_names.append("all")
            elif r in rois:
                roi_names.append(r)
    else:
        roi_names = ["all"]
        if rois:
            roi_names.extend(list(rois.keys()))
    
    if logger:
        log_if_present(logger, "info", f"ERDS comparison: segments={segments}, ROIs={roi_names}, bands={bands}, compare_windows={compare_wins}, compare_columns={compare_cols}")
    
    plot_cfg = get_plot_config(config)
    ensure_dir(save_dir)
    
    # Helper to get ERDS columns for a segment/band/ROI
    def get_erds_columns(segment, band, roi_name):
        """Get ERDS columns filtered by segment, band, and ROI."""
        cols = []
        roi_channels = all_channels if roi_name == "all" else get_roi_channels(rois.get(roi_name, []), all_channels)
        roi_set = set(roi_channels) if roi_channels else set(all_channels)
        
        for col in features_df.columns:
            parsed = NamingSchema.parse(str(col))
            if not parsed.get("valid"):
                continue
            if parsed.get("group") != "erds":
                continue
            if str(parsed.get("segment") or "") != segment:
                continue
            if str(parsed.get("band") or "") != band:
                continue
            # Accept global scope or ch scope in ROI
            scope = parsed.get("scope") or ""
            stat = str(parsed.get("stat") or "")
            # Prefer percent_mean or db_mean
            if stat not in ("percent_mean", "db_mean", "percent", "db"):
                continue
            if scope in ("global", "roi"):
                cols.append(col)
            elif scope == "ch":
                ch_id = str(parsed.get("identifier") or "")
                if ch_id in roi_set:
                    cols.append(col)
        return cols
    
    # Window comparison (paired) - use unified helper
    if compare_wins and len(segments) >= 2:
        seg1, seg2 = segments[0], segments[1]
        
        for roi_name in roi_names:
            data_by_band = {}
            for band in bands:
                cols1 = get_erds_columns(seg1, band, roi_name)
                cols2 = get_erds_columns(seg2, band, roi_name)
                
                if not cols1 or not cols2:
                    continue
                
                s1 = features_df[cols1].apply(pd.to_numeric, errors="coerce").mean(axis=1)
                s2 = features_df[cols2].apply(pd.to_numeric, errors="coerce").mean(axis=1)
                
                valid_mask = s1.notna() & s2.notna()
                v1, v2 = s1[valid_mask].values, s2[valid_mask].values
                
                if len(v1) > 0:
                    data_by_band[band] = (v1, v2)
            
            if data_by_band:
                roi_safe = roi_name.replace(" ", "_").lower() if roi_name != "all" else ""
                suffix = f"_roi-{roi_safe}" if roi_safe else ""
                save_path = save_dir / f"sub-{subject}_erds_by_condition{suffix}_window"
                
                plot_paired_comparison(
                    data_by_band=data_by_band,
                    subject=subject,
                    save_path=save_path,
                    feature_label="ERDS",
                    config=config,
                    logger=logger,
                    label1=seg1.capitalize(),
                    label2=seg2.capitalize(),
                    roi_name=roi_name,
                    stats_dir=stats_dir,
                )
        
        log_if_present(logger, "info", f"Saved ERDS paired comparison plots for {len(roi_names)} ROIs")

    # Column comparison (unpaired)
    if compare_cols:
        comp_mask_info = extract_comparison_mask(events_df, config)
        if not comp_mask_info:
            if logger:
                log_if_present(logger, "debug", "Column comparison requested but config incomplete")
        else:
            m1, m2, label1, label2 = comp_mask_info
            seg_name = get_config_value(config, "plotting.comparisons.comparison_segment", "active")
            
            segment_colors = {"v1": "#5a7d9a", "v2": "#c44e52"}
            band_colors = {band: get_band_color(band, config) for band in bands}
            n_bands = len(bands)
            n_trials = len(features_df)
            
            for roi_name in roi_names:
                all_pvals, pvalue_keys, cell_data = [], [], {}
                
                for col_idx, band in enumerate(bands):
                    cols = get_erds_columns(seg_name, band, roi_name)
                    
                    if not cols:
                        cell_data[col_idx] = None
                        continue
                    
                    val_series = features_df[cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
                    v1 = val_series[m1].dropna().values
                    v2 = val_series[m2].dropna().values
                    
                    cell_data[col_idx] = {"v1": v1, "v2": v2}
                    
                    if len(v1) >= 3 and len(v2) >= 3:
                        try:
                            _, p = stats.mannwhitneyu(v1, v2, alternative="two-sided")
                            diff = np.mean(v2) - np.mean(v1)
                            pooled_std = np.sqrt(((len(v1)-1)*np.var(v1, ddof=1) + (len(v2)-1)*np.var(v2, ddof=1)) / (len(v1)+len(v2)-2))
                            d = diff / pooled_std if pooled_std > 0 else 0
                            all_pvals.append(p)
                            pvalue_keys.append((col_idx, p, d))
                        except Exception:
                            pass
                
                qvalues = {}
                n_significant = 0
                if all_pvals:
                    rejected, qvals, _ = apply_fdr_correction(all_pvals, config=config)
                    for i, (key, p, d) in enumerate(pvalue_keys):
                        qvalues[key] = (p, qvals[i], d, rejected[i])
                    n_significant = int(np.sum(rejected))
                
                fig, axes = plt.subplots(1, n_bands, figsize=(3 * n_bands, 5), squeeze=False)
                
                for col_idx, band in enumerate(bands):
                    ax = axes.flatten()[col_idx]
                    data = cell_data.get(col_idx)
                    
                    if data is None or len(data.get("v1", [])) == 0 or len(data.get("v2", [])) == 0:
                        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                               transform=ax.transAxes, fontsize=plot_cfg.font.title, color="gray")
                        ax.set_xticks([])
                        continue
                    
                    v1, v2 = data["v1"], data["v2"]
                    
                    bp = ax.boxplot([v1, v2], positions=[0, 1], widths=0.4, patch_artist=True)
                    bp["boxes"][0].set_facecolor(segment_colors["v1"])
                    bp["boxes"][0].set_alpha(0.6)
                    bp["boxes"][1].set_facecolor(segment_colors["v2"])
                    bp["boxes"][1].set_alpha(0.6)
                    
                    ax.scatter(np.random.uniform(-0.08, 0.08, len(v1)), v1, c=segment_colors["v1"], alpha=0.3, s=6)
                    ax.scatter(1 + np.random.uniform(-0.08, 0.08, len(v2)), v2, c=segment_colors["v2"], alpha=0.3, s=6)
                    
                    all_vals = np.concatenate([v1, v2])
                    ymin, ymax = np.nanmin(all_vals), np.nanmax(all_vals)
                    yrange = ymax - ymin if ymax > ymin else 0.1
                    ax.set_ylim(ymin - 0.1 * yrange, ymax + 0.3 * yrange)
                    
                    if col_idx in qvalues:
                        _, q, d, sig = qvalues[col_idx]
                        sig_marker = "†" if sig else ""
                        sig_color = "#d62728" if sig else "#333333"
                        ax.annotate(f"q={q:.3f}{sig_marker}\nd={d:.2f}", xy=(0.5, ymax + 0.05 * yrange),
                                   ha="center", fontsize=plot_cfg.font.medium, color=sig_color,
                                   fontweight="bold" if sig else "normal")
                    
                    ax.set_xticks([0, 1])
                    ax.set_xticklabels([label1, label2], fontsize=9)
                    ax.set_title(band.capitalize(), fontweight="bold", color=band_colors.get(band, "gray"))
                    ax.spines["top"].set_visible(False)
                    ax.spines["right"].set_visible(False)
                
                n_tests = len(all_pvals)
                roi_display = roi_name.replace("_", " ").title() if roi_name != "all" else "All Channels"
                
                title = (f"ERDS: {label1} vs {label2} (Column Comparison)\n"
                         f"Subject: {subject} | ROI: {roi_display} | N: {n_trials} trials | Mann-Whitney U | "
                         f"FDR: {n_significant}/{n_tests} significant (†=q<0.05)")
                fig.suptitle(title, fontsize=plot_cfg.font.suptitle, fontweight="bold", y=1.02)
                
                plt.tight_layout()
                
                roi_safe = roi_name.replace(" ", "_").lower() if roi_name != "all" else ""
                suffix = f"_roi-{roi_safe}" if roi_safe else ""
                filename = f"sub-{subject}_erds_by_condition{suffix}_column"
                
                save_fig(fig, save_dir / filename, formats=plot_cfg.formats, dpi=plot_cfg.dpi,
                         bbox_inches=plot_cfg.bbox_inches, pad_inches=plot_cfg.pad_inches)
                plt.close(fig)
            
            log_if_present(logger, "info", f"Saved ERDS column comparison plots for {len(roi_names)} ROIs")


__all__ = [
    "plot_erds_temporal_evolution",
    "plot_erds_latency_distribution",
    "plot_erds_erd_ers_separation",
    "plot_erds_global_summary",
    "plot_erds_by_condition",
]

