"""
Complexity Feature Visualization
=================================

Clean, publication-quality visualizations for nonlinear dynamics features.
Uses violin/strip plots for distributions, shows individual data points.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from eeg_pipeline.plotting.io.figures import save_fig
from eeg_pipeline.plotting.config import get_plot_config
from eeg_pipeline.plotting.features.utils import (
    compute_cohens_d,
    get_band_names,
    get_band_colors,
    get_condition_colors,
)


###################################################################
# Permutation Entropy Plots
###################################################################


def plot_permutation_entropy_distribution(
    features_df: pd.DataFrame,
    save_path: Path,
    *,
    config: Any = None,
    figsize: Tuple[float, float] = (10, 5),
) -> plt.Figure:
    """Plot removed."""
    return plt.figure(figsize=figsize)


def plot_hjorth_parameters_distribution(
    features_df: pd.DataFrame,
    save_path: Path,
    *,
    config: Any = None,
    figsize: Tuple[float, float] = (12, 5),
) -> plt.Figure:
    """Plot removed."""
    return plt.figure(figsize=figsize)


def plot_hjorth_by_band(
    features_df: pd.DataFrame,
    save_path: Path,
    *,
    config: Any = None,
    figsize: Tuple[float, float] = (10, 5),
) -> plt.Figure:
    """Hjorth mobility across frequency bands."""
    plot_cfg = get_plot_config(config)
    fig, ax = plt.subplots(figsize=figsize)
    
    data_list = []
    positions = []
    colors = []
    
    bands = get_band_names(config)
    band_colors = get_band_colors(config)
    for i, band in enumerate(bands):
        cols = [c for c in features_df.columns if f"_{band}_" in c and "hjorth_mobility" in c]
        if cols:
            vals = features_df[cols].values.flatten()
            vals = vals[np.isfinite(vals)]
            if len(vals) > 0:
                data_list.append(vals)
                positions.append(i)
                colors.append(band_colors[band])
    
    if data_list:
        parts = ax.violinplot(data_list, positions=positions, showmedians=True, widths=0.7)
        
        for i, pc in enumerate(parts.get("bodies", [])):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.6)
        
        for i, (pos, vals) in enumerate(zip(positions, data_list)):
            jitter = np.random.uniform(-0.1, 0.1, len(vals))
            ax.scatter(pos + jitter, vals, c=colors[i], alpha=0.2, s=5)
    
    ax.set_xticks(range(len(bands)))
    ax.set_xticklabels([b.capitalize() for b in bands])
    ax.set_xlabel("Frequency Band")
    ax.set_ylabel("Hjorth Mobility")
    ax.set_title("Hjorth Mobility by Frequency Band")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    plt.tight_layout()
    save_fig(fig, save_path)
    plt.close(fig)
    
    return fig


def plot_lempel_ziv_complexity_distribution(
    features_df: pd.DataFrame,
    save_path: Path,
    *,
    config: Any = None,
    figsize: Tuple[float, float] = (10, 5),
) -> plt.Figure:
    """Plot removed."""
    return plt.figure(figsize=figsize)


def plot_complexity_comparison(
    features_df: pd.DataFrame,
    save_path: Path,
    *,
    config: Any = None,
    figsize: Tuple[float, float] = (10, 8),
) -> plt.Figure:
    """Plot removed."""
    return plt.figure(figsize=figsize)


def plot_complexity_by_condition(
    features_df: pd.DataFrame,
    events_df: pd.DataFrame,
    subject: str,
    save_dir: Path,
    logger: Any = None,
    config: Any = None,
    stats_dir: Optional[Path] = None,
) -> None:
    """Compare complexity metrics between conditions per band.
    
    Complexity uses frequency bands (delta, theta, alpha, beta, gamma) with
    metrics like lzc (Lempel-Ziv Complexity) and pe (Permutation Entropy).
    
    For window comparisons (paired): Uses the unified plot_paired_comparison helper.
    For column comparisons (unpaired): Uses Mann-Whitney U test with consistent styling.
    Creates one figure per ROI per metric.
    
    If stats_dir is provided, uses pre-computed statistics from the behavior pipeline.
    """
    from eeg_pipeline.domain.features.naming import NamingSchema
    from eeg_pipeline.infra.paths import ensure_dir
    from eeg_pipeline.utils.config.loader import get_config_value
    from eeg_pipeline.utils.analysis.events import extract_comparison_mask
    from eeg_pipeline.plotting.features.utils import (
        plot_paired_comparison,
        apply_fdr_correction,
        get_band_color,
    )
    from eeg_pipeline.plotting.features.roi import get_roi_definitions
    from eeg_pipeline.plotting.io.figures import log_if_present
    
    if features_df is None or features_df.empty or events_df is None:
        return

    compare_wins = get_config_value(config, "plotting.comparisons.compare_windows", True)
    compare_cols = get_config_value(config, "plotting.comparisons.compare_columns", False)
    
    # Get segments from data (the group is "comp", not "complexity")
    segment_set = set()
    for col in features_df.columns:
        parsed = NamingSchema.parse(str(col))
        if parsed.get("valid") and parsed.get("group") == "comp":
            seg = parsed.get("segment")
            if seg:
                segment_set.add(str(seg))
    
    # Get segments from config or auto-detect from data
    segments = get_config_value(config, "plotting.comparisons.comparison_windows", [])
    if not segments or len(segments) < 2:
        if len(segment_set) >= 2:
            segments = sorted(segment_set)[:2]
            if logger:
                log_if_present(logger, "info", f"Auto-detected segments for complexity comparison: {segments}")
    
    # Get frequency bands from config
    bands = get_band_names(config)
    if not bands:
        # Try to detect bands from data
        bands_found = set()
        for col in features_df.columns:
            parsed = NamingSchema.parse(str(col))
            if parsed.get("valid") and parsed.get("group") == "comp":
                band = parsed.get("band")
                if band:
                    bands_found.add(str(band))
        bands = sorted(bands_found)
    
    if not bands:
        return
    
    # Get metrics (lzc, pe, etc.)
    metrics_found = set()
    for col in features_df.columns:
        parsed = NamingSchema.parse(str(col))
        if parsed.get("valid") and parsed.get("group") == "comp":
            stat = parsed.get("stat")
            if stat:
                metrics_found.add(str(stat))
    
    metrics = sorted(metrics_found) if metrics_found else ["lzc", "pe"]
    metric_labels = {"lzc": "LZC", "pe": "PE"}
    
    # Get ROI definitions
    rois = get_roi_definitions(config)
    
    # Get ROIs from data
    data_rois = set()
    for col in features_df.columns:
        parsed = NamingSchema.parse(str(col))
        if parsed.get("valid") and parsed.get("group") == "comp" and parsed.get("scope") == "roi":
            roi_id = parsed.get("identifier")
            if roi_id:
                data_rois.add(str(roi_id))
    
    # Use config-defined ROIs (keys from 'rois' config)
    # The feature computation uses these same names in column names
    config_roi_names = list(rois.keys()) if rois else []
    
    comp_rois = get_config_value(config, "plotting.comparisons.comparison_rois", [])
    if comp_rois:
        roi_names = []
        for r in comp_rois:
            if r.lower() == "all":
                if "all" not in roi_names:
                    roi_names.append("all")
            else:
                # Check if config ROI matches by name
                for config_roi in config_roi_names:
                    if r.lower().replace("_", "").replace("-", "") == config_roi.lower().replace("_", "").replace("-", ""):
                        roi_names.append(config_roi)
                        break
    else:
        # Default: all + each config-defined ROI
        roi_names = ["all"]
        roi_names.extend(config_roi_names)

    
    if logger:
        log_if_present(logger, "info", f"Complexity comparison: segments={segments}, ROIs={roi_names}, bands={bands}, metrics={metrics}, compare_windows={compare_wins}, compare_columns={compare_cols}")
    
    plot_cfg = get_plot_config(config)
    ensure_dir(save_dir)
    
    # Helper to get complexity columns for a segment/band/metric/ROI
    def get_complexity_columns(segment, band, metric, roi_name):
        """Get complexity columns filtered by segment, band, metric, and ROI."""
        cols = []
        for col in features_df.columns:
            parsed = NamingSchema.parse(str(col))
            if not parsed.get("valid"):
                continue
            if parsed.get("group") != "comp":
                continue
            if str(parsed.get("segment") or "") != segment:
                continue
            if str(parsed.get("band") or "") != band:
                continue
            if str(parsed.get("stat") or "") != metric:
                continue
            
            scope = parsed.get("scope") or ""
            if roi_name == "all":
                # For "all", prefer global scope
                if scope == "global":
                    cols.append(col)
            else:
                # For specific ROI, match the roi identifier
                if scope == "roi":
                    roi_id = str(parsed.get("identifier") or "")
                    # Flexible matching (case-insensitive, ignore underscores)
                    if roi_id.lower().replace("_", "") == roi_name.lower().replace("_", ""):
                        cols.append(col)
        
        # If no global columns for "all", fall back to averaging all roi columns
        if roi_name == "all" and not cols:
            for col in features_df.columns:
                parsed = NamingSchema.parse(str(col))
                if not parsed.get("valid"):
                    continue
                if parsed.get("group") != "comp":
                    continue
                if str(parsed.get("segment") or "") != segment:
                    continue
                if str(parsed.get("band") or "") != band:
                    continue
                if str(parsed.get("stat") or "") != metric:
                    continue
                if parsed.get("scope") == "roi":
                    cols.append(col)
        
        return cols
    
    # Create plots per metric
    for metric in metrics:
        metric_label = metric_labels.get(metric, metric.upper())
        
        # Window comparison (paired) - use unified helper
        if compare_wins and len(segments) >= 2:
            seg1, seg2 = segments[0], segments[1]
            
            for roi_name in roi_names:
                data_by_band = {}
                for band in bands:
                    cols1 = get_complexity_columns(seg1, band, metric, roi_name)
                    cols2 = get_complexity_columns(seg2, band, metric, roi_name)
                    
                    if not cols1 or not cols2:
                        continue
                    
                    s1 = features_df[cols1].apply(pd.to_numeric, errors="coerce").mean(axis=1)
                    s2 = features_df[cols2].apply(pd.to_numeric, errors="coerce").mean(axis=1)
                    
                    valid_mask = s1.notna() & s2.notna()
                    v1, v2 = s1[valid_mask].values, s2[valid_mask].values
                    
                    if len(v1) > 0:
                        data_by_band[band] = (v1, v2)
                
                if data_by_band:
                    roi_safe = roi_name.replace(" ", "_").lower() if roi_name.lower() != "all" else ""
                    suffix = f"_roi-{roi_safe}" if roi_safe else ""
                    save_path = save_dir / f"sub-{subject}_complexity_{metric}_by_condition{suffix}_window"
                    
                    plot_paired_comparison(
                        data_by_band=data_by_band,
                        subject=subject,
                        save_path=save_path,
                        feature_label=f"Complexity ({metric_label})",
                        config=config,
                        logger=logger,
                        label1=seg1.capitalize(),
                        label2=seg2.capitalize(),
                        roi_name=roi_name,
                        stats_dir=stats_dir,
                    )
            
            if logger:
                log_if_present(logger, "info", f"Saved complexity {metric_label} paired comparison plots for {len(roi_names)} ROIs")

        # Column comparison (unpaired)
        if compare_cols:
            comp_mask_info = extract_comparison_mask(events_df, config)
            if not comp_mask_info:
                if logger:
                    log_if_present(logger, "debug", "Column comparison requested but config incomplete")
            else:
                m1, m2, label1, label2 = comp_mask_info
                seg_name = get_config_value(config, "plotting.comparisons.comparison_segment", "active")
                
                from eeg_pipeline.plotting.features.utils import compute_or_load_column_stats
                
                segment_colors = {"v1": "#5a7d9a", "v2": "#c44e52"}
                band_colors = {band: get_band_color(band, config) for band in bands}
                n_bands = len(bands)
                n_trials = len(features_df)
                
                for roi_name in roi_names:
                    # Collect cell data first
                    cell_data = {}
                    for col_idx, band in enumerate(bands):
                        cols = get_complexity_columns(seg_name, band, metric, roi_name)
                        
                        if not cols:
                            cell_data[col_idx] = None
                            continue
                        
                        val_series = features_df[cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
                        v1 = val_series[m1].dropna().values
                        v2 = val_series[m2].dropna().values
                        
                        cell_data[col_idx] = {"v1": v1, "v2": v2}
                    
                    # Compute or load column comparison stats
                    qvalues, n_significant, use_precomputed = compute_or_load_column_stats(
                        stats_dir=stats_dir,
                        feature_type="complexity",
                        feature_keys=bands,
                        cell_data=cell_data,
                        config=config,
                        logger=logger,
                    )
                    
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
                    
                    n_tests = len(qvalues)
                    roi_display = roi_name.replace("_", " ").title() if roi_name.lower() != "all" else "All Channels"
                    
                    stats_source = "pre-computed" if use_precomputed else "Mann-Whitney U"
                    title = (f"Complexity ({metric_label}): {label1} vs {label2} (Column Comparison)\n"
                             f"Subject: {subject} | ROI: {roi_display} | N: {n_trials} trials | {stats_source} | "
                             f"FDR: {n_significant}/{n_tests} significant (†=q<0.05)")
                    fig.suptitle(title, fontsize=plot_cfg.font.suptitle, fontweight="bold", y=1.02)
                    
                    plt.tight_layout()
                    
                    roi_safe = roi_name.replace(" ", "_").lower() if roi_name.lower() != "all" else ""
                    suffix = f"_roi-{roi_safe}" if roi_safe else ""
                    filename = f"sub-{subject}_complexity_{metric}_by_condition{suffix}_column"
                    
                    save_fig(fig, save_dir / filename, formats=plot_cfg.formats, dpi=plot_cfg.dpi,
                             bbox_inches=plot_cfg.bbox_inches, pad_inches=plot_cfg.pad_inches)
                    plt.close(fig)
                
                if logger:
                    log_if_present(logger, "info", f"Saved complexity {metric_label} column comparison plots for {len(roi_names)} ROIs")







def plot_complexity_summary(
    features_df: pd.DataFrame,
    save_path: Path,
    *,
    config: Any = None,
    figsize: Tuple[float, float] = (10, 8),
) -> plt.Figure:
    """Plot removed."""
    return plt.figure(figsize=figsize)



# Alias for backward compatibility
plot_complexity_by_band = plot_hjorth_by_band

__all__ = [
    "plot_hjorth_by_band",
    "plot_complexity_by_band",
    "plot_complexity_by_condition",
]
