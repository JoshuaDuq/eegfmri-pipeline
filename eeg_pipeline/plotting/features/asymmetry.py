"""
Asymmetry Visualization
=======================

Plots for hemispheric asymmetry indices across bands.
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


def plot_asymmetry_by_band(
    features_df: pd.DataFrame,
    save_path: Path,
    *,
    config: Any = None,
    stat: Optional[str] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> plt.Figure:
    """Plot hemispheric asymmetry distributions by band."""
    plot_cfg = get_plot_config(config)
    segments = get_named_segments(features_df, group="asymmetry")
    segment = _select_segment(segments, preferred="active")
    if segment is None:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No asymmetry data", ha="center", va="center")
        return fig
    bands = get_named_bands(features_df, group="asymmetry", segment=segment)
    if not bands:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No asymmetry data", ha="center", va="center")
        return fig

    if stat is None:
        stat = str(get_config_value(config, "plotting.plots.features.asymmetry.stat", "index"))
    stat = stat if stat else "index"

    band_order = get_band_names(config)
    bands = [b for b in band_order if b in bands] + [b for b in bands if b not in band_order]

    if figsize is None:
        figsize = (max(8.0, len(bands) * 1.2), 5.0)

    fig, ax = plt.subplots(figsize=figsize)
    data_list = []
    positions = []

    for i, band in enumerate(bands):
        series, _, _ = collect_named_series(
            features_df,
            group="asymmetry",
            segment=segment,
            band=band,
            stat_preference=[stat, "index", "logdiff"],
            scope_preference=["chpair"],
        )
        vals = series.dropna().values
        if vals.size == 0:
            continue
        data_list.append(vals)
        positions.append(i)

    if data_list:
        parts = ax.violinplot(data_list, positions=positions, showmedians=True, widths=0.7)
        for pc in parts.get("bodies", []):
            pc.set_facecolor("#0EA5E9")
            pc.set_alpha(0.6)
        ax.set_xticks(range(len(bands)))
        ax.set_xticklabels([b.capitalize() for b in bands])
    else:
        ax.text(0.5, 0.5, "No asymmetry data", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])

    ax.set_xlabel("Band")
    ax.set_ylabel(stat.replace("_", " ").title())
    seg_label = segment if segment is not None else "unknown"
    ax.set_title(f"Asymmetry ({stat}) by Band ({seg_label})", fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

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


def plot_asymmetry_by_condition(
    features_df: pd.DataFrame,
    events_df: pd.DataFrame,
    subject: str,
    save_dir: Path,
    logger: Any = None,
    config: Any = None,
    stats_dir: Optional[Path] = None,
) -> None:
    """Compare asymmetry indices between conditions.
    
    For window comparisons (paired): Uses the unified plot_paired_comparison helper.
    For column comparisons (unpaired): Uses Mann-Whitney U test with consistent styling.
    Creates one figure per ROI.
    
    If stats_dir is provided, uses pre-computed statistics from the behavior pipeline.
    """
    from eeg_pipeline.domain.features.naming import NamingSchema
    from eeg_pipeline.infra.paths import ensure_dir
    from eeg_pipeline.utils.config.loader import get_config_value
    from eeg_pipeline.utils.analysis.events import extract_comparison_mask
    from eeg_pipeline.plotting.features.utils import (
        plot_paired_comparison,
        apply_fdr_correction,
        get_named_segments,
    )
    from eeg_pipeline.plotting.features.roi import get_roi_definitions, get_roi_channels
    from eeg_pipeline.plotting.io.figures import log_if_present
    from scipy import stats
    
    if features_df is None or features_df.empty or events_df is None:
        return

    compare_wins = get_config_value(config, "plotting.comparisons.compare_windows", True)
    compare_cols = get_config_value(config, "plotting.comparisons.compare_columns", False)
    
    # Get segments from config or auto-detect from data
    segments = get_config_value(config, "plotting.comparisons.comparison_windows", [])
    if not segments or len(segments) < 2:
        detected = get_named_segments(features_df, group="asymmetry")
        if len(detected) >= 2:
            segments = detected[:2]
            if logger:
                log_if_present(logger, "info", f"Auto-detected segments for asymmetry comparison: {segments}")
    
    # Get bands from config or data
    bands = get_band_names(config)
    if not bands:
        detected_bands = set()
        for col in features_df.columns:
            parsed = NamingSchema.parse(str(col))
            if parsed.get("valid") and parsed.get("group") == "asymmetry":
                b = parsed.get("band")
                if b:
                    detected_bands.add(str(b))
        bands = sorted(list(detected_bands))
    
    if not bands:
        return
    
    # Get metrics
    metrics = get_config_value(config, "plotting.plots.features.asymmetry.comparison_metrics", ["index", "logdiff"])
    
    # Get ROI definitions
    rois = get_roi_definitions(config)
    config_roi_names = list(rois.keys()) if rois else []
    
    comp_rois = get_config_value(config, "plotting.comparisons.comparison_rois", [])
    if comp_rois:
        roi_names = []
        for r in comp_rois:
            if r.lower() == "all":
                if "all" not in roi_names:
                    roi_names.append("all")
            else:
                for config_roi in config_roi_names:
                    if r.lower().replace("_", "").replace("-", "") == config_roi.lower().replace("_", "").replace("-", ""):
                        roi_names.append(config_roi)
                        break
    else:
        roi_names = ["all"]
        roi_names.extend(config_roi_names)
    
    if logger:
        log_if_present(logger, "info", f"Asymmetry comparison: segments={segments}, ROIs={roi_names}, compare_windows={compare_wins}, compare_columns={compare_cols}")
    
    plot_cfg = get_plot_config(config)
    ensure_dir(save_dir)
    
    # Helper to get asymmetry columns for a segment/band/metric/ROI
    def get_asymmetry_columns(segment, band, metric, roi_name):
        """Get asymmetry columns filtered by segment, band, metric, and ROI."""
        cols = []
        for col in features_df.columns:
            parsed = NamingSchema.parse(str(col))
            if not parsed.get("valid") or parsed.get("group") != "asymmetry":
                continue
            if str(parsed.get("segment") or "") != segment:
                continue
            if str(parsed.get("band") or "") != band:
                continue
            if str(parsed.get("stat") or "") != metric:
                continue
            
            scope = parsed.get("scope") or ""
            if roi_name == "all":
                # For asymmetry, we usually have chpair. "all" means all chpairs.
                cols.append(col)
            else:
                # If we have an ROI, we check if the channels in the pair belong to the ROI
                ident = str(parsed.get("identifier") or "")
                if scope == "roi":
                    if ident.lower().replace("_", "") == roi_name.lower().replace("_", ""):
                        cols.append(col)
                elif scope == "chpair":
                    # ident is "CH1-CH2"
                    if "-" in ident:
                        ch1, ch2 = ident.split("-", 1)
                        roi_chans = get_roi_channels(rois.get(roi_name, []), [ch1, ch2])
                        if ch1 in roi_chans or ch2 in roi_chans:
                            cols.append(col)
        return cols

    for metric in metrics:
        metric_label = metric.replace("_", " ").title()
        
        # Window comparison (paired)
        if compare_wins and len(segments) >= 2:
            seg1, seg2 = segments[0], segments[1]
            
            for roi_name in roi_names:
                data_by_band = {}
                for band in bands:
                    cols1 = get_asymmetry_columns(seg1, band, metric, roi_name)
                    cols2 = get_asymmetry_columns(seg2, band, metric, roi_name)
                    
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
                    save_path = save_dir / f"sub-{subject}_asymmetry_{metric}_by_condition{suffix}_window"
                    
                    plot_paired_comparison(
                        data_by_band=data_by_band,
                        subject=subject,
                        save_path=save_path,
                        feature_label=f"Asymmetry ({metric_label})",
                        config=config,
                        logger=logger,
                        label1=seg1.capitalize(),
                        label2=seg2.capitalize(),
                        roi_name=roi_name,
                        stats_dir=stats_dir,
                    )

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
                n_bands = len(bands)
                n_trials = len(features_df)
                
                for roi_name in roi_names:
                    all_pvals, pvalue_keys, cell_data = [], [], {}
                    
                    for col_idx, band in enumerate(bands):
                        cols = get_asymmetry_columns(seg_name, band, metric, roi_name)
                        
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
                        ax.set_title(band.capitalize(), fontweight="bold")
                        ax.spines["top"].set_visible(False)
                        ax.spines["right"].set_visible(False)
                    
                    n_tests = len(all_pvals)
                    roi_display = roi_name.replace("_", " ").title() if roi_name.lower() != "all" else "All Pairs"
                    
                    title = (f"Asymmetry ({metric_label}): {label1} vs {label2} (Column Comparison)\n"
                             f"Subject: {subject} | ROI: {roi_display} | N: {n_trials} trials | Mann-Whitney U | "
                             f"FDR: {n_significant}/{n_tests} significant (†=q<0.05)")
                    fig.suptitle(title, fontsize=plot_cfg.font.suptitle, fontweight="bold", y=1.02)
                    
                    plt.tight_layout()
                    
                    roi_safe = roi_name.replace(" ", "_").lower() if roi_name.lower() != "all" else ""
                    suffix = f"_roi-{roi_safe}" if roi_safe else ""
                    filename = f"sub-{subject}_asymmetry_{metric}_by_condition{suffix}_column"
                    
                    save_fig(fig, save_dir / filename, formats=plot_cfg.formats, dpi=plot_cfg.dpi,
                             bbox_inches=plot_cfg.bbox_inches, pad_inches=plot_cfg.pad_inches)
                    plt.close(fig)



__all__ = [
    "plot_asymmetry_by_band",
    "plot_asymmetry_by_condition",
]
