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

from eeg_pipeline.domain.features.naming import NamingSchema
from eeg_pipeline.plotting.io.figures import save_fig
from eeg_pipeline.utils.analysis.events import extract_pain_mask
from eeg_pipeline.plotting.config import get_plot_config
from eeg_pipeline.plotting.features.utils import (
    get_band_names,
    get_band_colors,
    get_condition_colors,
)


###################################################################
# Permutation Entropy Plots
###################################################################

def _get_complexity_segments(features_df: pd.DataFrame) -> List[str]:
    segments = set()
    for col in features_df.columns:
        parsed = NamingSchema.parse(str(col))
        if not parsed.get("valid"):
            continue
        if parsed.get("group") != "comp":
            continue
        segment = str(parsed.get("segment") or "")
        if segment:
            segments.add(segment)
    return sorted(segments)


def _select_complexity_segment(features_df: pd.DataFrame, preferred: str = "active") -> Optional[str]:
    segments = _get_complexity_segments(features_df)
    if not segments:
        return None
    if preferred in segments:
        return preferred
    return segments[0]


def _collect_complexity_values(
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
        if parsed.get("group") != "comp":
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
        return _collect_complexity_values(
            features_df,
            band=band,
            segment=segment,
            stat=stat,
            scope="ch",
        )

    return np.array([])


def plot_complexity_by_band(
    features_df: pd.DataFrame,
    save_path: Path,
    *,
    config: Any = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> plt.Figure:
    """LZC and permutation entropy distributions across frequency bands."""
    plot_cfg = get_plot_config(config)
    if figsize is None:
        figsize = plot_cfg.get_figure_size("wide", plot_type="features")

    bands = get_band_names(config)
    band_colors = get_band_colors(config)
    segment = _select_complexity_segment(features_df, preferred="active")

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    metrics = [("lzc", "LZC"), ("pe", "PE")]
    for ax, (stat, label) in zip(axes, metrics):
        data_list = []
        positions = []
        colors = []

        if segment is not None:
            for i, band in enumerate(bands):
                vals = _collect_complexity_values(
                    features_df,
                    band=band,
                    segment=segment,
                    stat=stat,
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
            for i, (pos, vals) in enumerate(zip(positions, data_list)):
                jitter = np.random.uniform(-0.1, 0.1, len(vals))
                ax.scatter(pos + jitter, vals, c=colors[i], alpha=0.2, s=5)
            ax.set_xticks(range(len(bands)))
            ax.set_xticklabels([b.capitalize() for b in bands])
        else:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([])

        ax.set_xlabel("Frequency Band")
        ax.set_ylabel(label)
        ax.set_title(f"{label} by Band")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    seg_label = segment if segment is not None else "unknown"
    fig.suptitle(
        f"Complexity by Band ({seg_label})",
        fontsize=plot_cfg.font.figure_title,
        fontweight="bold",
        y=1.02,
    )

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
    """Complexity metrics by condition and timing (baseline vs active).
    
    Grid: rows = timing (baseline/active), cols = complexity metrics.
    
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
    measures = [
        ("lzc", "LZC"),
        ("pe", "PE"),
    ]
    if figsize is None:
        width_per_col = float(plot_cfg.plot_type_configs.get("complexity", {}).get("width_per_measure", 4.5))
        height_per_row = float(plot_cfg.plot_type_configs.get("complexity", {}).get("height_per_segment", 4.0))
        figsize = (width_per_col * len(measures), height_per_row * 2)
    
    from eeg_pipeline.utils.config.loader import get_config_value
    from .utils import (
        compute_condition_stats,
        apply_fdr_correction,
        format_stats_annotation,
        format_footer_annotation,
    )
    
    baseline_window = get_config_value(config, "time_frequency_analysis.baseline_window", [-3.0, -0.5])
    active_window = get_config_value(config, "time_frequency_analysis.active_window", [3.0, 10.5])
    
    segment_labels = {
        "baseline": ("BASELINE", f"{baseline_window[0]:.1f} to {baseline_window[1]:.1f}s"),
        "active": ("ACTIVE", f"{active_window[0]:.1f} to {active_window[1]:.1f}s")
    }
    segments = list(segment_labels.keys())

    all_stats = []
    all_pvals = []
    cell_data = {}
    
    for row_idx, segment in enumerate(segments):
        for col_idx, (stat, label) in enumerate(measures):
            cols = []
            for c in features_df.columns:
                parsed = NamingSchema.parse(str(c))
                if not parsed.get("valid"):
                    continue
                if parsed.get("group") != "comp":
                    continue
                if str(parsed.get("segment") or "") != str(segment):
                    continue
                if str(parsed.get("stat") or "") != str(stat):
                    continue
                cols.append(str(c))
            
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
        for col_idx, (stat, label) in enumerate(measures):
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
    fig.suptitle(f"Complexity by Condition: Baseline vs Active (sub-{subject})\nN: {n_nonpain} non-pain, {n_pain} pain", 
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
    "plot_complexity_by_band",
    "plot_complexity_by_condition",
]
