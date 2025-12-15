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
    get_band_names,
    get_band_colors,
    get_condition_colors,
)


###################################################################
# Burst Detection Plots
###################################################################


def plot_burst_duration_distribution(
    features_df: pd.DataFrame,
    save_path: Path,
    *,
    config: Any = None,
    figsize: Tuple[float, float] = (10, 5),
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    
    bands = get_band_names(config)
    band_colors = get_band_colors(config)

    data_list = []
    positions = []
    colors = []
    
    for i, band in enumerate(bands):
        cols = [c for c in features_df.columns if f"dynamics_{band}_burst_mean_duration" in c]
        if cols:
            vals = features_df[cols[0]].dropna().values
            vals = vals[np.isfinite(vals)]
            if len(vals) > 0:
                vals_ms = vals * 1000
                data_list.append(vals_ms)
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
    ax.set_xlabel("Frequency Band")
    ax.set_ylabel("Burst Duration (ms)")
    ax.set_title("Burst Duration Distribution by Band")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    plt.tight_layout()
    save_fig(fig, save_path)
    plt.close(fig)
    
    return fig


def plot_burst_amplitude_distribution(
    features_df: pd.DataFrame,
    save_path: Path,
    *,
    config: Any = None,
    figsize: Tuple[float, float] = (10, 5),
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    
    bands = get_band_names(config)
    band_colors = get_band_colors(config)

    data_list = []
    positions = []
    colors = []
    
    for i, band in enumerate(bands):
        cols = [c for c in features_df.columns if f"dynamics_{band}_burst_mean_amplitude" in c]
        if cols:
            vals = features_df[cols[0]].dropna().values
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
            ax.scatter(pos + jitter, vals, c=colors[i], alpha=0.3, s=8)
    
    ax.set_xticks(range(len(bands)))
    ax.set_xticklabels([b.capitalize() for b in bands])
    ax.set_xlabel("Frequency Band")
    ax.set_ylabel("Burst Amplitude (a.u.)")
    ax.set_title("Burst Amplitude Distribution by Band")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    plt.tight_layout()
    save_fig(fig, save_path)
    plt.close(fig)
    
    return fig


def plot_burst_summary_by_band(
    features_df: pd.DataFrame,
    save_path: Path,
    *,
    config: Any = None,
    figsize: Tuple[float, float] = (12, 8),
) -> plt.Figure:
    plot_cfg = get_plot_config(config)
    bands = get_band_names(config)
    band_colors = get_band_colors(config)
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    ax = axes[0]
    
    for i, band in enumerate(bands):
        rate_cols = [c for c in features_df.columns if f"dynamics_{band}_burst_rate" in c]
        dur_cols = [c for c in features_df.columns if f"dynamics_{band}_burst_mean_duration" in c]
        
        if rate_cols and dur_cols:
            rates = features_df[rate_cols[0]].values
            durations = features_df[dur_cols[0]].values * 1000
            
            valid = np.isfinite(rates) & np.isfinite(durations)
            if np.sum(valid) > 0:
                ax.scatter(rates[valid], durations[valid], 
                          c=band_colors[band], alpha=0.5, s=20, 
                          label=band.capitalize())
    
    ax.set_xlabel("Burst Rate (count)")
    ax.set_ylabel("Burst Duration (ms)")
    ax.set_title("A. Rate vs Duration")
    ax.legend(fontsize=plot_cfg.font.medium)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    ax = axes[1]
    
    for i, band in enumerate(bands):
        rate_cols = [c for c in features_df.columns if f"dynamics_{band}_burst_rate" in c]
        amp_cols = [c for c in features_df.columns if f"dynamics_{band}_burst_mean_amplitude" in c]
        
        if rate_cols and amp_cols:
            rates = features_df[rate_cols[0]].values
            amps = features_df[amp_cols[0]].values
            
            valid = np.isfinite(rates) & np.isfinite(amps)
            if np.sum(valid) > 0:
                ax.scatter(rates[valid], amps[valid], 
                          c=band_colors[band], alpha=0.5, s=20,
                          label=band.capitalize())
    
    ax.set_xlabel("Burst Rate (count)")
    ax.set_ylabel("Burst Amplitude (a.u.)")
    ax.set_title("B. Rate vs Amplitude")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    fig.suptitle("Burst Characteristics", fontsize=plot_cfg.font.figure_title, fontweight="bold", y=1.02)
    
    plt.tight_layout()
    save_fig(fig, save_path)
    plt.close(fig)
    
    return fig


###################################################################
# GFP Dynamics Plots
###################################################################


def plot_gfp_by_band(
    features_df: pd.DataFrame,
    save_path: Path,
    *,
    config: Any = None,
    figsize: Tuple[float, float] = (10, 5),
) -> plt.Figure:
    plot_cfg = get_plot_config(config)
    bands = get_band_names(config)
    band_colors = get_band_colors(config)
    fig, ax = plt.subplots(figsize=figsize)
    
    data_list = []
    positions = []
    colors = []
    
    for i, band in enumerate(bands):
        cols = [c for c in features_df.columns if f"gfp_{band}_mean_active" in c]
        if cols:
            vals = features_df[cols[0]].dropna().values
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
            ax.scatter(pos + jitter, vals, c=colors[i], alpha=0.3, s=8)
    
    ax.set_xticks(range(len(bands)))
    ax.set_xticklabels([b.capitalize() for b in bands])
    ax.set_xlabel("Frequency Band")
    ax.set_ylabel("GFP (mean active)")
    ax.set_title("Band-Specific Global Field Power")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    plt.tight_layout()
    save_fig(fig, save_path)
    plt.close(fig)
    
    return fig


###################################################################
# Power Dynamics Plots
###################################################################


def plot_power_fano_factor(
    features_df: pd.DataFrame,
    save_path: Path,
    *,
    config: Any = None,
    figsize: Tuple[float, float] = (10, 5),
) -> plt.Figure:
    plot_cfg = get_plot_config(config)
    bands = get_band_names(config)
    band_colors = get_band_colors(config)
    fig, ax = plt.subplots(figsize=figsize)
    
    data_list = []
    positions = []
    colors = []
    
    for i, band in enumerate(bands):
        cols = [c for c in features_df.columns if f"dynamics_{band}_power_fano" in c]
        if cols:
            vals = features_df[cols[0]].dropna().values
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
            ax.scatter(pos + jitter, vals, c=colors[i], alpha=0.3, s=8)
    
    ax.axhline(1, color="gray", linestyle="--", linewidth=1.5, label="Poisson (F=1)")
    
    ax.set_xticks(range(len(bands)))
    ax.set_xticklabels([b.capitalize() for b in bands])
    ax.set_xlabel("Frequency Band")
    ax.set_ylabel("Fano Factor")
    ax.set_title("Power Variability by Band")
    ax.legend(fontsize=plot_cfg.font.medium, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    plt.tight_layout()
    save_fig(fig, save_path)
    plt.close(fig)
    
    return fig


def plot_power_logratio(
    features_df: pd.DataFrame,
    save_path: Path,
    *,
    config: Any = None,
    figsize: Tuple[float, float] = (10, 5),
) -> plt.Figure:
    plot_cfg = get_plot_config(config)
    bands = get_band_names(config)
    band_colors = get_band_colors(config)
    fig, ax = plt.subplots(figsize=figsize)
    
    data_list = []
    positions = []
    colors = []
    
    for i, band in enumerate(bands):
        cols = [c for c in features_df.columns if f"dynamics_{band}_logratio" in c]
        if cols:
            vals = features_df[cols[0]].dropna().values
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
            ax.scatter(pos + jitter, vals, c=colors[i], alpha=0.3, s=8)
    
    ax.axhline(0, color="black", linestyle="-", linewidth=1)
    
    ax.set_xticks(range(len(bands)))
    ax.set_xticklabels([b.capitalize() for b in bands])
    ax.set_xlabel("Frequency Band")
    ax.set_ylabel("Log Ratio (Active / Baseline)")
    ax.set_title("Power Change Relative to Baseline")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    plt.tight_layout()
    save_fig(fig, save_path)
    plt.close(fig)
    
    return fig


def plot_gamma_ramp_bursts(
    features_df: pd.DataFrame,
    save_path: Path,
    *,
    figsize: Tuple[float, float] = (8, 5),
) -> plt.Figure:
    """Gamma burst characteristics during ramp period."""
    fig, ax = plt.subplots(figsize=figsize)
    
    rate_cols = [c for c in features_df.columns if "dynamics_gamma_burst_rate" in c]
    dur_cols = [c for c in features_df.columns if "dynamics_gamma_burst_mean_duration" in c]
    
    if rate_cols and dur_cols:
        rates = features_df[rate_cols[0]].values
        durations = features_df[dur_cols[0]].values * 1000
        
        valid = np.isfinite(rates) & np.isfinite(durations)
        if np.sum(valid) > 0:
            ax.scatter(rates[valid], durations[valid], 
                      c=BAND_COLORS["gamma"], alpha=0.6, s=30, edgecolors="white")
            
            r, p = stats.spearmanr(rates[valid], durations[valid])
            ax.text(0.05, 0.95, f"ρ = {r:.2f}, p = {p:.3f}", 
                   transform=ax.transAxes, fontsize=plot_cfg.font.large, va="top")
    
    ax.set_xlabel("Gamma Burst Rate (count)")
    ax.set_ylabel("Gamma Burst Duration (ms)")
    ax.set_title("Gamma Burst Characteristics")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    plt.tight_layout()
    save_fig(fig, save_path)
    plt.close(fig)
    
    return fig


def plot_dynamics_by_condition(
    features_df: pd.DataFrame,
    events_df: pd.DataFrame,
    subject: str,
    save_path: Path,
    config: Any = None,
    figsize: Tuple[float, float] = (16, 5),
) -> plt.Figure:
    """Dynamics by condition per frequency band during plateau.
    
    Shows burst dynamics during plateau for each band, pain vs non-pain.
    Timing windows are read from config.
    
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
    
    from eeg_pipeline.utils.config.loader import get_config_value
    from eeg_pipeline.plotting.features.utils import (
        compute_condition_stats,
        apply_fdr_correction,
        format_stats_annotation,
        format_footer_annotation,
    )
    
    plateau_window = get_config_value(config, "plateau_window", [3.0, 10.5])
    plateau_label = f"{plateau_window[0]:.1f} to {plateau_window[1]:.1f}s"
    
    plot_cfg = get_plot_config(config)
    bands = get_band_names(config)
    band_colors = get_band_colors(config)
    condition_colors = get_condition_colors(config)
    
    all_stats = []
    all_pvals = []
    band_data = {}
    
    for band in bands:
        cols = [c for c in features_df.columns 
               if f"dynamics_{band}_" in c or f"dynamics_plateau_{band}_" in c]
        if not cols:
            cols = [c for c in features_df.columns if f"_{band}_burst" in c]
        
        if not cols:
            band_data[band] = None
            continue
        
        vals = features_df[cols].mean(axis=1)
        nonpain_vals = vals[~pain_mask].dropna().values
        pain_vals = vals[pain_mask].dropna().values
        
        if len(nonpain_vals) >= 3 and len(pain_vals) >= 3:
            stats_result = compute_condition_stats(nonpain_vals, pain_vals, n_boot=1000, config=config)
            all_stats.append(stats_result)
            all_pvals.append(stats_result["p_raw"])
            band_data[band] = {
                "nonpain": nonpain_vals,
                "pain": pain_vals,
                "stats": stats_result,
                "stats_idx": len(all_stats) - 1,
            }
        else:
            band_data[band] = {
                "nonpain": nonpain_vals,
                "pain": pain_vals,
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
    
    fig, axes = plt.subplots(1, len(bands), figsize=figsize, sharey=True)
    
    for idx, band in enumerate(bands):
        ax = axes[idx]
        data = band_data.get(band)
        
        if data is None:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", 
                   transform=ax.transAxes, fontsize=plot_cfg.font.large, color="gray")
            ax.set_xticks([])
            continue
        
        nonpain_vals = data["nonpain"]
        pain_vals = data["pain"]
        
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
                text_color = "#d62728" if s.get("fdr_significant", False) else "#333333"
                ax.text(0.5, 0.98, annotation, transform=ax.transAxes, 
                       ha="center", fontsize=6, va="top", color=text_color,
                       bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
        
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["NP", "P"], fontsize=plot_cfg.font.medium)
        ax.set_title(band.capitalize(), fontweight="bold", color=band_colors.get(band, "#333"))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        
        if idx == 0:
            ax.set_ylabel("Burst Dynamics")
    
    n_pain = int(pain_mask.sum())
    n_nonpain = int((~pain_mask).sum())
    fig.suptitle(f"Dynamics by Condition: Plateau ({plateau_label}) (sub-{subject})\nN: {n_nonpain} NP, {n_pain} P", 
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
    "plot_burst_duration_distribution",
    "plot_burst_amplitude_distribution",
    "plot_burst_summary_by_band",
    "plot_gfp_by_band",
    "plot_power_fano_factor",
    "plot_power_logratio",
    "plot_gamma_ramp_bursts",
    "plot_dynamics_by_condition",
]
