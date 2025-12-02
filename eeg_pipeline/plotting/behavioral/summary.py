"""
Summary Dashboard Visualizations.

Creates comprehensive summary figures combining multiple analysis results
into publication-ready dashboard layouts.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns

from eeg_pipeline.plotting.config import get_plot_config
from eeg_pipeline.utils.io.general import save_fig, ensure_dir, get_band_color


def plot_analysis_dashboard(
    subject: str,
    stats_df: Optional[pd.DataFrame],
    power_df: Optional[pd.DataFrame],
    targets: Optional[pd.Series],
    save_path: Path,
    *,
    temperature: Optional[pd.Series] = None,
    title: str = "Brain-Behavior Analysis Dashboard",
    figsize: Tuple[float, float] = (16, 12),
    config: Any = None,
    logger: Optional[logging.Logger] = None,
) -> plt.Figure:
    """Create comprehensive analysis dashboard.
    
    Layout:
    ┌─────────────────┬─────────────────┐
    │  Top Predictors │  Effect Size    │
    │  (Forest Plot)  │  Distribution   │
    ├─────────────────┼─────────────────┤
    │  Rating vs Temp │  Feature        │
    │  Scatter        │  Heatmap        │
    └─────────────────┴─────────────────┘
    
    Parameters
    ----------
    subject : str
        Subject ID
    stats_df : pd.DataFrame
        Correlation statistics (must have 'r', 'p' columns)
    power_df : pd.DataFrame
        Power features
    targets : pd.Series
        Behavioral target (e.g., pain rating)
    temperature : pd.Series, optional
        Temperature data
    """
    plot_cfg = get_plot_config(config)
    # Defensive: ensure stats_df has expected columns to avoid RGBA errors
    if stats_df is not None and not stats_df.empty:
        stats_df = stats_df.copy()
        if "r" not in stats_df.columns and "correlation" in stats_df.columns:
            stats_df["r"] = stats_df["correlation"]
        if "p" not in stats_df.columns and "p_value" in stats_df.columns:
            stats_df["p"] = stats_df["p_value"]
        if "q" not in stats_df.columns and "p_fdr" in stats_df.columns:
            stats_df["q"] = stats_df["p_fdr"]
    
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25)
    
    # ══════════════════════════════════════════════════════════════
    # Panel A: Top Predictors (Forest Plot)
    # ══════════════════════════════════════════════════════════════
    ax_forest = fig.add_subplot(gs[0, 0])
    
    if stats_df is not None and not stats_df.empty and "r" in stats_df.columns:
        df = stats_df.copy()
        df["abs_r"] = df["r"].abs()
        # Show top 30 features in dashboard
        df_top = df.nlargest(30, "abs_r")
        
        if not df_top.empty:
            y_pos = np.arange(len(df_top))[::-1]
            r_vals = df_top["r"].values
            
            # Color by significance
            sig_color = "#C42847"
            nonsig_color = "#666666"
            if "q" in df_top.columns:
                colors = [sig_color if q < 0.05 else nonsig_color for q in df_top["q"]]
            elif "p" in df_top.columns:
                colors = [sig_color if p < 0.05 else nonsig_color for p in df_top["p"]]
            else:
                colors = [nonsig_color] * len(df_top)
            
            # Error bars if CI available
            if "ci_low" in df_top.columns and "ci_high" in df_top.columns:
                ci_low = df_top["ci_low"].values
                ci_high = df_top["ci_high"].values
                # Plot error bars individually to avoid RGBA issues with color lists
                for i in range(len(r_vals)):
                    ax_forest.errorbar(
                        r_vals[i], y_pos[i], 
                        xerr=[[r_vals[i] - ci_low[i]], [ci_high[i] - r_vals[i]]], 
                        fmt="none",
                        ecolor=colors[i], elinewidth=1.5, capsize=3, zorder=1
                    )
            
            ax_forest.scatter(r_vals, y_pos, c=colors, s=50, zorder=3, edgecolors="white")
            ax_forest.axvline(0, color="#808080", linestyle="--", linewidth=1, alpha=0.7)
            
            # Labels
            if "identifier" in df_top.columns:
                labels = df_top["identifier"].astype(str).tolist()
            elif "channel" in df_top.columns and "band" in df_top.columns:
                labels = [f"{c} ({b})" for c, b in zip(df_top["channel"], df_top["band"])]
            elif "feature" in df_top.columns:
                labels = df_top["feature"].astype(str).tolist()
            else:
                labels = [f"Feature {i}" for i in range(len(df_top))]
            
            ax_forest.set_yticks(y_pos)
            ax_forest.set_yticklabels(labels, fontsize=8)
            ax_forest.set_xlabel("Correlation (r)")
            ax_forest.grid(axis="x", alpha=0.3, linestyle=":")
    else:
        ax_forest.text(0.5, 0.5, "No correlation data", ha="center", va="center")
    
    ax_forest.set_title("A. Top Brain-Behavior Correlations", fontsize=11, fontweight="bold", loc="left")
    
    # ══════════════════════════════════════════════════════════════
    # Panel B: Effect Size Distribution
    # ══════════════════════════════════════════════════════════════
    ax_dist = fig.add_subplot(gs[0, 1])
    
    if stats_df is not None and not stats_df.empty and "r" in stats_df.columns:
        r_all = stats_df["r"].dropna()
        
        ax_dist.hist(r_all, bins=30, color="#3B82F6", alpha=0.7, edgecolor="white", density=True)
        
        # Add KDE
        if len(r_all) > 10:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(r_all)
            x_range = np.linspace(r_all.min(), r_all.max(), 100)
            ax_dist.plot(x_range, kde(x_range), "r-", linewidth=2, label="KDE")
        
        ax_dist.axvline(0, color="#808080", linestyle="--", linewidth=1, alpha=0.7)
        ax_dist.axvline(r_all.mean(), color="red", linestyle="-", linewidth=2, alpha=0.8,
                       label=f"Mean: {r_all.mean():.3f}")
        
        # Significance threshold markers
        if "q" in stats_df.columns:
            n_sig = (stats_df["q"] < 0.05).sum()
            n_total = len(stats_df)
            ax_dist.text(0.98, 0.98, f"Significant: {n_sig}/{n_total}\n({100*n_sig/n_total:.1f}%)",
                        transform=ax_dist.transAxes, ha="right", va="top", fontsize=9,
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        
        ax_dist.set_xlabel("Correlation (r)")
        ax_dist.set_ylabel("Density")
        ax_dist.legend(loc="upper left", fontsize=8)
        ax_dist.grid(alpha=0.3)
    else:
        ax_dist.text(0.5, 0.5, "No data", ha="center", va="center")
    
    ax_dist.set_title("B. Effect Size Distribution", fontsize=11, fontweight="bold", loc="left")
    
    # ══════════════════════════════════════════════════════════════
    # Panel C: Rating vs Temperature
    # ══════════════════════════════════════════════════════════════
    ax_scatter = fig.add_subplot(gs[1, 0])
    
    if targets is not None and temperature is not None:
        # Align indices to ensure same length
        common_idx = targets.index.intersection(temperature.index)
        if len(common_idx) > 0:
            targets_aligned = targets.loc[common_idx]
            temp_aligned = temperature.loc[common_idx]
            valid = targets_aligned.notna() & temp_aligned.notna()
            
            if valid.sum() > 5:
                temp_valid = temp_aligned[valid]
                targets_valid = targets_aligned[valid]
                
                ax_scatter.scatter(temp_valid, targets_valid, 
                                  alpha=0.5, s=20, c="#22C55E", edgecolors="white", linewidths=0.3)
                
                # Regression line
                from scipy import stats as scipy_stats
                slope, intercept, r, p, _ = scipy_stats.linregress(temp_valid, targets_valid)
                x_line = np.array([temp_valid.min(), temp_valid.max()])
                y_line = slope * x_line + intercept
                ax_scatter.plot(x_line, y_line, "r-", linewidth=2, alpha=0.8)
            
                ax_scatter.text(0.05, 0.95, f"r = {r:.3f}\np = {p:.3f}",
                               transform=ax_scatter.transAxes, va="top", fontsize=10,
                               bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
                
                ax_scatter.set_xlabel("Temperature (°C)")
                ax_scatter.set_ylabel("Pain Rating")
                ax_scatter.grid(alpha=0.3)
            else:
                ax_scatter.text(0.5, 0.5, "Insufficient data", ha="center", va="center")
        else:
            ax_scatter.text(0.5, 0.5, "No overlapping data", ha="center", va="center")
    elif targets is not None:
        targets_valid = targets.dropna()
        if len(targets_valid) > 0:
            ax_scatter.hist(targets_valid, bins=min(25, len(targets_valid)//2), color="#22C55E", alpha=0.7, edgecolor="white")
            ax_scatter.set_xlabel("Pain Rating")
            ax_scatter.set_ylabel("Count")
            mean_val = targets_valid.mean()
            ax_scatter.axvline(mean_val, color="red", linestyle="--", linewidth=2,
                              label=f"Mean: {mean_val:.2f}")
            ax_scatter.legend(fontsize=8)
            ax_scatter.grid(alpha=0.3)
        else:
            ax_scatter.text(0.5, 0.5, "No valid behavioral data", ha="center", va="center", fontsize=10)
    else:
        ax_scatter.text(0.5, 0.5, "No behavioral data", ha="center", va="center", fontsize=10)
    
    ax_scatter.set_title("C. Behavioral Response", fontsize=11, fontweight="bold", loc="left")
    
    # ══════════════════════════════════════════════════════════════
    # Panel D: Band-wise Summary Heatmap
    # ══════════════════════════════════════════════════════════════
    ax_heat = fig.add_subplot(gs[1, 1])
    
    if stats_df is not None and "band" in stats_df.columns and "r" in stats_df.columns:
        # Pivot: bands x ROI (or some grouping)
        if "roi" in stats_df.columns:
            pivot = stats_df.pivot_table(index="band", columns="roi", values="r", aggfunc="mean")
        elif "channel" in stats_df.columns:
            # Summarize by band
            band_summary = stats_df.groupby("band").agg({
                "r": ["mean", "std", "min", "max"],
            })
            band_summary.columns = ["Mean r", "SD", "Min", "Max"]
            
            bands = band_summary.index.tolist()
            y_pos = np.arange(len(bands))
            mean_r = band_summary["Mean r"].values
            sd_vals = band_summary["SD"].fillna(0).values
            
            # Get colors with safe fallback
            band_colors = []
            for b in bands:
                try:
                    color_val = get_band_color(b, config)
                    band_colors.append(color_val)
                except Exception:
                    band_colors.append("#3B82F6")  # Blue fallback
            
            ax_heat.barh(y_pos, mean_r, xerr=sd_vals,
                        capsize=3, color=band_colors,
                        edgecolor="white", linewidth=0.5, height=0.6)
            ax_heat.set_yticks(y_pos)
            ax_heat.set_yticklabels([b.title() for b in bands])
            ax_heat.axvline(0, color="#808080", linestyle="--", linewidth=1, alpha=0.7)
            ax_heat.set_xlabel("Mean Correlation ± SD")
            ax_heat.grid(axis="x", alpha=0.3)
            pivot = None
        else:
            pivot = None
        
        if pivot is not None and not pivot.empty:
            vmax = max(abs(pivot.values.min()), abs(pivot.values.max()))
            sns.heatmap(pivot, cmap="RdBu_r", center=0, vmin=-vmax, vmax=vmax,
                       ax=ax_heat, cbar_kws={"label": "r", "shrink": 0.8},
                       annot=True, fmt=".2f", annot_kws={"fontsize": 7},
                       linewidths=0.5)
    else:
        ax_heat.text(0.5, 0.5, "No band data", ha="center", va="center")
    
    ax_heat.set_title("D. Frequency Band Summary", fontsize=11, fontweight="bold", loc="left")
    
    # ══════════════════════════════════════════════════════════════
    # Main title and save
    # ══════════════════════════════════════════════════════════════
    fig.suptitle(f"{title} — sub-{subject}", fontsize=14, fontweight="bold", y=0.98)
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    save_fig(fig, save_path)
    plt.close(fig)
    
    if logger:
        logger.info(f"Saved analysis dashboard to {save_path}")
    
    return fig


def plot_group_summary_dashboard(
    subjects: List[str],
    group_stats_df: Optional[pd.DataFrame],
    subject_stats: Dict[str, pd.DataFrame],
    save_path: Path,
    *,
    title: str = "Group Analysis Summary",
    figsize: Tuple[float, float] = (18, 12),
    config: Any = None,
    logger: Optional[logging.Logger] = None,
) -> plt.Figure:
    """Create group-level summary dashboard.
    
    Shows:
    - Group-level effect sizes
    - Subject variability
    - Reliability metrics
    - Significant predictors across subjects
    """
    plot_cfg = get_plot_config(config)
    
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # Panel A: Group effect sizes
    ax_group = fig.add_subplot(gs[0, 0])
    if group_stats_df is not None and not group_stats_df.empty and "r_pooled" in group_stats_df.columns:
        df = group_stats_df.copy()
        df["abs_r"] = df["r_pooled"].abs()
        df_top = df.nlargest(15, "abs_r")
        
        y_pos = np.arange(len(df_top))[::-1]
        colors = [plot_cfg.get_color("significant") if r else plot_cfg.get_color("nonsignificant")
                  for r in df_top.get("fdr_reject", [True] * len(df_top))]
        
        ax_group.barh(y_pos, df_top["r_pooled"], color=colors, edgecolor="white", height=0.7)
        ax_group.axvline(0, color="gray", linestyle="--")
        
        if "roi" in df_top.columns and "band" in df_top.columns:
            labels = [f"{r} ({b})" for r, b in zip(df_top["roi"], df_top["band"])]
        else:
            labels = [f"Feature {i}" for i in range(len(df_top))]
        ax_group.set_yticks(y_pos)
        ax_group.set_yticklabels(labels, fontsize=8)
        ax_group.set_xlabel("Pooled Correlation")
        ax_group.grid(axis="x", alpha=0.3)
    else:
        ax_group.text(0.5, 0.5, "No group stats", ha="center", va="center")
    ax_group.set_title("A. Group-Level Effects", fontsize=11, fontweight="bold", loc="left")
    
    # Panel B: Subject variability
    ax_subj = fig.add_subplot(gs[0, 1])
    if subject_stats:
        subj_means = []
        subj_ids = []
        for subj, df in subject_stats.items():
            if df is not None and "r" in df.columns:
                subj_means.append(df["r"].mean())
                subj_ids.append(subj)
        
        if subj_means:
            y_pos = np.arange(len(subj_ids))
            colors = plt.cm.Set2(np.linspace(0, 1, len(subj_ids)))
            ax_subj.barh(y_pos, subj_means, color=colors, edgecolor="white", height=0.7)
            ax_subj.axvline(np.mean(subj_means), color="red", linestyle="--", linewidth=2,
                           label=f"Grand mean: {np.mean(subj_means):.3f}")
            ax_subj.set_yticks(y_pos)
            ax_subj.set_yticklabels([f"sub-{s}" for s in subj_ids], fontsize=8)
            ax_subj.set_xlabel("Mean Correlation")
            ax_subj.legend(fontsize=8)
            ax_subj.grid(axis="x", alpha=0.3)
        else:
            ax_subj.text(0.5, 0.5, "No subject data", ha="center", va="center")
    else:
        ax_subj.text(0.5, 0.5, "No subject stats", ha="center", va="center")
    ax_subj.set_title("B. Subject Variability", fontsize=11, fontweight="bold", loc="left")
    
    # Panel C: Consistency across subjects
    ax_consist = fig.add_subplot(gs[0, 2])
    if subject_stats and len(subject_stats) > 1:
        # Count how many subjects show significant effects per feature
        feature_counts = {}
        for subj, df in subject_stats.items():
            if df is not None and "p" in df.columns:
                sig_features = df[df["p"] < 0.05]
                for feat in sig_features.get("channel", sig_features.get("feature", [])):
                    feature_counts[feat] = feature_counts.get(feat, 0) + 1
        
        if feature_counts:
            sorted_feats = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)[:15]
            feats, counts = zip(*sorted_feats)
            y_pos = np.arange(len(feats))[::-1]
            
            ax_consist.barh(y_pos, counts, color="#3B82F6", edgecolor="white", height=0.7)
            ax_consist.set_yticks(y_pos)
            ax_consist.set_yticklabels(feats, fontsize=8)
            ax_consist.set_xlabel(f"# Subjects (of {len(subjects)})")
            ax_consist.axvline(len(subjects) * 0.5, color="red", linestyle="--", 
                              label="50% threshold")
            ax_consist.legend(fontsize=8)
            ax_consist.grid(axis="x", alpha=0.3)
        else:
            ax_consist.text(0.5, 0.5, "No significant features", ha="center", va="center")
    else:
        ax_consist.text(0.5, 0.5, "Need multiple subjects", ha="center", va="center")
    ax_consist.set_title("C. Cross-Subject Consistency", fontsize=11, fontweight="bold", loc="left")
    
    # Panel D: Effect size by band
    ax_band = fig.add_subplot(gs[1, 0])
    if group_stats_df is not None and "band" in group_stats_df.columns:
        band_summary = group_stats_df.groupby("band")["r_pooled"].agg(["mean", "std"]).reset_index()
        bands = band_summary["band"].values
        y_pos = np.arange(len(bands))
        
        ax_band.barh(y_pos, band_summary["mean"], xerr=band_summary["std"], capsize=3,
                    color=[get_band_color(b, config) or "#3B82F6" for b in bands], edgecolor="white", height=0.6)
        ax_band.axvline(0, color="gray", linestyle="--")
        ax_band.set_yticks(y_pos)
        ax_band.set_yticklabels([b.title() for b in bands])
        ax_band.set_xlabel("Mean Pooled Correlation ± SD")
        ax_band.grid(axis="x", alpha=0.3)
    else:
        ax_band.text(0.5, 0.5, "No band data", ha="center", va="center")
    ax_band.set_title("D. Band-wise Group Effects", fontsize=11, fontweight="bold", loc="left")
    
    # Panel E: ROI summary
    ax_roi = fig.add_subplot(gs[1, 1])
    if group_stats_df is not None and "roi" in group_stats_df.columns:
        roi_summary = group_stats_df.groupby("roi")["r_pooled"].agg(["mean", "std"]).reset_index()
        roi_summary = roi_summary.nlargest(10, "mean")
        
        y_pos = np.arange(len(roi_summary))[::-1]
        ax_roi.barh(y_pos, roi_summary["mean"], xerr=roi_summary["std"], capsize=3,
                   color="#22C55E", edgecolor="white", height=0.6)
        ax_roi.axvline(0, color="gray", linestyle="--")
        ax_roi.set_yticks(y_pos)
        ax_roi.set_yticklabels(roi_summary["roi"].values)
        ax_roi.set_xlabel("Mean Pooled Correlation")
        ax_roi.grid(axis="x", alpha=0.3)
    else:
        ax_roi.text(0.5, 0.5, "No ROI data", ha="center", va="center")
    ax_roi.set_title("E. ROI-wise Effects", fontsize=11, fontweight="bold", loc="left")
    
    # Panel F: Summary statistics text
    ax_summary = fig.add_subplot(gs[1, 2])
    ax_summary.axis("off")
    
    summary_text = f"Analysis Summary\n{'=' * 30}\n\n"
    summary_text += f"Subjects: {len(subjects)}\n"
    
    if group_stats_df is not None and not group_stats_df.empty:
        n_total = len(group_stats_df)
        n_sig = group_stats_df.get("fdr_reject", pd.Series()).sum()
        summary_text += f"Total tests: {n_total}\n"
        summary_text += f"Significant (FDR): {n_sig} ({100*n_sig/n_total:.1f}%)\n"
        
        if "r_pooled" in group_stats_df.columns:
            r_mean = group_stats_df["r_pooled"].mean()
            r_max = group_stats_df["r_pooled"].abs().max()
            summary_text += f"\nEffect sizes:\n"
            summary_text += f"  Mean r: {r_mean:.3f}\n"
            summary_text += f"  Max |r|: {r_max:.3f}\n"
    
    ax_summary.text(0.1, 0.9, summary_text, transform=ax_summary.transAxes,
                   fontsize=10, family="monospace", va="top",
                   bbox=dict(boxstyle="round", facecolor="#f0f0f0", alpha=0.8))
    ax_summary.set_title("F. Summary", fontsize=11, fontweight="bold", loc="left")
    
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    save_fig(fig, save_path)
    plt.close(fig)
    
    if logger:
        logger.info(f"Saved group summary dashboard to {save_path}")
    
    return fig


def plot_quality_overview(
    quality_report: Dict[str, Any],
    save_path: Path,
    *,
    subject: str = "",
    title: str = "Feature Quality Overview",
    figsize: Tuple[float, float] = (14, 8),
    config: Any = None,
) -> plt.Figure:
    """Visualize feature quality metrics.
    
    Shows:
    - Missing data patterns
    - Variance/ICC summary
    - Outlier counts
    - Feature quality scores
    """
    plot_cfg = get_plot_config(config)
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Panel A: Missing data
    ax = axes[0, 0]
    if "missing_per_feature" in quality_report and quality_report["missing_per_feature"]:
        missing = quality_report["missing_per_feature"]
        # Ensure it's a dict and has values
        if isinstance(missing, dict) and len(missing) > 0:
            features = list(missing.keys())[:20]
            values = [float(missing[f]) if pd.notna(missing[f]) else 0.0 for f in features]
            if any(v > 0 for v in values):  # Only plot if there's actual missing data
                y_pos = np.arange(len(features))[::-1]
                colors = ["#EF4444" if v > 0.1 else "#22C55E" for v in values]
                ax.barh(y_pos, values, color=colors, edgecolor="white", height=0.7)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(features, fontsize=7)
                ax.set_xlabel("Proportion Missing")
                ax.axvline(0.1, color="red", linestyle="--", label="10% threshold")
                ax.legend(fontsize=8)
            else:
                ax.text(0.5, 0.5, "No missing data", ha="center", va="center", fontsize=10)
        else:
            ax.text(0.5, 0.5, "No missing data info", ha="center", va="center", fontsize=10)
    else:
        ax.text(0.5, 0.5, "No missing data info", ha="center", va="center", fontsize=10)
    ax.set_title("A. Missing Data", fontsize=10, fontweight="bold", loc="left")
    ax.grid(axis="x", alpha=0.3)
    
    # Panel B: Variance summary
    ax = axes[0, 1]
    if "variance_summary" in quality_report and quality_report["variance_summary"]:
        var_summary = quality_report["variance_summary"]
        if isinstance(var_summary, dict) and len(var_summary) > 0:
            var_values = [float(v) for v in var_summary.values() if pd.notna(v) and v > 0]
            if len(var_values) > 0:
                ax.hist(var_values, bins=min(20, len(var_values)), color="#3B82F6", 
                       edgecolor="white", alpha=0.7)
                median_val = np.median(var_values)
                ax.axvline(median_val, color="red", 
                          linestyle="--", label=f"Median: {median_val:.2e}")
                ax.set_xlabel("Feature Variance")
                ax.set_ylabel("Count")
                ax.legend(fontsize=8)
                ax.grid(axis="y", alpha=0.3)
            else:
                ax.text(0.5, 0.5, "No valid variance data", ha="center", va="center", fontsize=10)
        else:
            ax.text(0.5, 0.5, "No variance info", ha="center", va="center", fontsize=10)
    else:
        ax.text(0.5, 0.5, "No variance info", ha="center", va="center", fontsize=10)
    ax.set_title("B. Variance Distribution", fontsize=10, fontweight="bold", loc="left")
    ax.grid(alpha=0.3)
    
    # Panel C: Quality scores
    ax = axes[1, 0]
    if "quality_scores" in quality_report and quality_report["quality_scores"]:
        scores = quality_report["quality_scores"]
        if isinstance(scores, dict) and len(scores) > 0:
            categories = list(scores.keys())[:10]
            values = []
            valid_categories = []
            for c in categories:
                v = scores.get(c)
                if v is not None and pd.notna(v):
                    try:
                        v_float = float(v)
                        if 0 <= v_float <= 1:
                            values.append(v_float)
                            valid_categories.append(c)
                    except (ValueError, TypeError):
                        continue
            
            if len(values) > 0:
                y_pos = np.arange(len(valid_categories))
                colors = ["#22C55E" if v >= 0.8 else "#F59E0B" if v >= 0.5 else "#EF4444" for v in values]
                ax.barh(y_pos, values, color=colors, edgecolor="white", height=0.6)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(valid_categories)
                ax.set_xlabel("Quality Score (0-1)")
                ax.set_xlim(0, 1)
                ax.axvline(0.8, color="gray", linestyle="--", alpha=0.7)
                ax.grid(axis="x", alpha=0.3)
            else:
                ax.text(0.5, 0.5, "No valid quality scores", ha="center", va="center", fontsize=10)
        else:
            ax.text(0.5, 0.5, "No quality scores", ha="center", va="center", fontsize=10)
    else:
        ax.text(0.5, 0.5, "No quality scores", ha="center", va="center", fontsize=10)
    ax.set_title("C. Quality Scores", fontsize=10, fontweight="bold", loc="left")
    ax.grid(axis="x", alpha=0.3)
    
    # Panel D: Summary text
    ax = axes[1, 1]
    ax.axis("off")
    
    summary = f"Quality Report Summary\n{'=' * 25}\n\n"
    if "n_features" in quality_report:
        summary += f"Total features: {quality_report['n_features']}\n"
    if "n_samples" in quality_report:
        summary += f"Total samples: {quality_report['n_samples']}\n"
    if "overall_quality" in quality_report:
        summary += f"Overall quality: {quality_report['overall_quality']:.2f}\n"
    if "n_low_quality" in quality_report:
        summary += f"Low quality features: {quality_report['n_low_quality']}\n"
    if "recommendations" in quality_report:
        summary += f"\nRecommendations:\n"
        for rec in quality_report["recommendations"][:3]:
            summary += f"  • {rec}\n"
    
    ax.text(0.1, 0.9, summary, transform=ax.transAxes,
           fontsize=10, family="monospace", va="top",
           bbox=dict(boxstyle="round", facecolor="#f0f0f0", alpha=0.8))
    ax.set_title("D. Summary", fontsize=10, fontweight="bold", loc="left")
    
    suptitle = title
    if subject:
        suptitle += f" — sub-{subject}"
    fig.suptitle(suptitle, fontsize=12, fontweight="bold")
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    save_fig(fig, save_path)
    plt.close(fig)
    
    return fig
