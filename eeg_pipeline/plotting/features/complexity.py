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
from eeg_pipeline.utils.analysis.events import extract_pain_mask
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


def plot_hjorth_by_band(
    features_df: pd.DataFrame,
    save_path: Path,
    *,
    config: Any = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> plt.Figure:
    """Hjorth mobility across frequency bands."""
    plot_cfg = get_plot_config(config)
    if figsize is None:
        figsize = plot_cfg.get_figure_size("wide", plot_type="features")
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


def plot_complexity_by_condition(
    features_df: pd.DataFrame,
    events_df: pd.DataFrame,
    subject: str,
    save_path: Path,
    config: Any = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> plt.Figure:
    """Complexity metrics by condition and timing (baseline vs plateau).
    
    Grid: rows = timing (baseline/plateau), cols = complexity metrics.
    
    Statistical improvements:
    - Shows both raw p-value and FDR-corrected q-value
    - Includes bootstrap 95% CI for mean difference
    - Reports Cohen's d effect size
    - Footer shows total tests and correction method
    """
    pain_mask = extract_pain_mask(events_df, config)
    if pain_mask is None or len(features_df) != len(pain_mask):
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "Cannot identify conditions", ha="center", va="center")
        return fig
    
    plot_cfg = get_plot_config(config)
    if figsize is None:
        # 3 columns, 2 rows
        width_per_col = float(plot_cfg.plot_type_configs.get("complexity", {}).get("width_per_measure", 4.5))
        height_per_row = float(plot_cfg.plot_type_configs.get("complexity", {}).get("height_per_segment", 4.0))
        figsize = (width_per_col * 3, height_per_row * 2)
    
    from eeg_pipeline.utils.config.loader import get_config_value
    from .utils import (
        compute_condition_stats,
        apply_fdr_correction,
        format_stats_annotation,
        format_footer_annotation,
    )
    
    baseline_window = get_config_value(config, "time_frequency_analysis.baseline_window", [-3.0, -0.5])
    plateau_window = get_config_value(config, "plateau_window", [3.0, 10.5])
    
    segment_labels = {
        "baseline": ("BASELINE", f"{baseline_window[0]:.1f} to {baseline_window[1]:.1f}s"),
        "plateau": ("PLATEAU", f"{plateau_window[0]:.1f} to {plateau_window[1]:.1f}s")
    }
    segments = list(segment_labels.keys())

    measures = [
        ("_lzc", "LZC"),
        ("hjorth_mobility", "Mobility"),
        ("hjorth_complexity", "Complexity")
    ]
    
    all_stats = []
    all_pvals = []
    cell_data = {}
    
    for row_idx, segment in enumerate(segments):
        for col_idx, (pattern, label) in enumerate(measures):
            cols = [c for c in features_df.columns 
                   if pattern in c and f"_{segment}_" in c]
            
            if not cols:
                cell_data[(row_idx, col_idx)] = None
                continue
            
            vals = features_df[cols].mean(axis=1)
            nonpain_vals = vals[~pain_mask].dropna().values
            pain_vals = vals[pain_mask].dropna().values
            
            if len(nonpain_vals) >= 3 and len(pain_vals) >= 3:
                stats_result = compute_condition_stats(nonpain_vals, pain_vals, n_boot=1000, config=config)
                all_stats.append(stats_result)
                all_pvals.append(stats_result["p_raw"])
                cell_data[(row_idx, col_idx)] = {
                    "nonpain_vals": nonpain_vals,
                    "pain_vals": pain_vals,
                    "stats": stats_result,
                    "stats_idx": len(all_stats) - 1,
                }
            else:
                cell_data[(row_idx, col_idx)] = {
                    "nonpain_vals": nonpain_vals,
                    "pain_vals": pain_vals,
                    "stats": None,
                    "stats_idx": None,
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
    
    fig, axes = plt.subplots(len(segments), len(measures), figsize=figsize, sharey="row")
    condition_colors = get_condition_colors(config)
    
    for row_idx, segment in enumerate(segments):
        seg_name, seg_time = segment_labels[segment]
        for col_idx, (pattern, label) in enumerate(measures):
            ax = axes[row_idx, col_idx]
            
            data = cell_data.get((row_idx, col_idx))
            
            if data is None:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", 
                       transform=ax.transAxes, fontsize=plot_cfg.font.large, color="gray")
                ax.set_xticks([])
                continue
            
            nonpain_vals = data["nonpain_vals"]
            pain_vals = data["pain_vals"]
            
            if len(nonpain_vals) > 0 and len(pain_vals) > 0:
                bp = ax.boxplot([nonpain_vals, pain_vals], 
                               positions=[0, 1], widths=0.4, patch_artist=True)
                bp["boxes"][0].set_facecolor(condition_colors["nonpain"])
                bp["boxes"][0].set_alpha(0.6)
                bp["boxes"][1].set_facecolor(condition_colors["pain"])
                bp["boxes"][1].set_alpha(0.6)
                
                ax.scatter(np.random.uniform(-0.1, 0.1, len(nonpain_vals)), 
                          nonpain_vals, c=condition_colors["nonpain"], alpha=0.3, s=8)
                ax.scatter(1 + np.random.uniform(-0.1, 0.1, len(pain_vals)), 
                          pain_vals, c=condition_colors["pain"], alpha=0.3, s=8)
                
                if data["stats"] is not None:
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
                    ax.text(0.5, 0.98, annotation, transform=ax.transAxes, 
                           ha="center", fontsize=plot_cfg.font.annotation, va="top", color=text_color,
                           bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
            
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["NP", "P"], fontsize=plot_cfg.font.small)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            
            if row_idx == 0:
                ax.set_title(label, fontweight="bold", fontsize=plot_cfg.font.title)
            if col_idx == 0:
                ax.set_ylabel(f"{seg_name}\n({seg_time})", fontsize=plot_cfg.font.medium)
    
    n_pain = int(pain_mask.sum())
    n_nonpain = int((~pain_mask).sum())
    fig.suptitle(f"Complexity by Condition: Baseline vs Plateau (sub-{subject})\nN: {n_nonpain} non-pain, {n_pain} pain", 
                fontsize=plot_cfg.font.figure_title, fontweight="bold", y=1.02)
    
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
    save_fig(fig, save_path)
    plt.close(fig)
    
    return fig



__all__ = [
    "plot_hjorth_by_band",
    "plot_complexity_by_condition",
]
