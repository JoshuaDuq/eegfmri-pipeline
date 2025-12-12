"""
ROI-based feature visualization with Band × Condition grids.

Provides structured visualization of EEG features organized by:
- ROI (9 regions from config)
- Frequency band (delta, theta, alpha, beta, gamma)
- Condition (pain vs non-pain)
"""

from __future__ import annotations

import re
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from eeg_pipeline.utils.analysis.events import extract_pain_mask
from eeg_pipeline.plotting.config import get_plot_config
from eeg_pipeline.utils.io.plotting import save_fig
from eeg_pipeline.plotting.features.utils import (
    apply_fdr_correction,
    get_band_colors,
    get_band_names,
    get_condition_colors,
    compute_cohens_d,
    compute_paired_cohens_d,
    format_significance_annotation,
    get_significance_color,
    safe_mannwhitneyu,
    safe_wilcoxon,
)


def get_roi_definitions(config: Any) -> Dict[str, List[str]]:
    """Get ROI definitions from config.
    
    Returns dict mapping ROI name to list of regex patterns.
    """
    from eeg_pipeline.utils.config.loader import get_config_value
    
    rois = get_config_value(config, "rois", {})
    if not rois:
        rois = get_config_value(config, "time_frequency_analysis.rois", {})
    return rois


def get_roi_channels(roi_patterns: List[str], all_channels: List[str]) -> List[str]:
    """Match channels to ROI regex patterns."""
    matched = []
    for ch in all_channels:
        for pattern in roi_patterns:
            if re.match(pattern, ch):
                matched.append(ch)
                break
    return matched


def extract_channels_from_columns(columns: List[str]) -> List[str]:
    """Extract unique channel names from feature column names."""
    channels = set()
    for col in columns:
        match = re.search(r'_ch_([A-Za-z0-9]+)_', col)
        if match:
            channels.add(match.group(1))
        elif col.endswith(('_mean', '_logratio', '_lzc', '_pe')):
            match = re.search(r'_ch_([A-Za-z0-9]+)', col)
            if match:
                channels.add(match.group(1))
    return sorted(list(channels))


def aggregate_by_roi(
    features_df: pd.DataFrame,
    col_pattern: str,
    roi_channels: List[str],
) -> pd.Series:
    """Aggregate feature columns by ROI (mean across channels in ROI)."""
    cols = []
    for col in features_df.columns:
        if col_pattern in col:
            for ch in roi_channels:
                if f"_ch_{ch}_" in col or col.endswith(f"_ch_{ch}"):
                    cols.append(col)
                    break
    
    if not cols:
        return pd.Series([np.nan] * len(features_df), index=features_df.index)
    
    return features_df[cols].mean(axis=1)


def _get_bands_and_palettes(config: Any) -> Tuple[List[str], Dict[str, str], Dict[str, str]]:
    """Convenience accessor for bands and color palettes."""
    bands = get_band_names(config)
    band_colors = get_band_colors(config)
    condition_colors = get_condition_colors(config)
    return bands, band_colors, condition_colors


def plot_power_by_roi_band_condition(
    features_df: pd.DataFrame,
    events_df: pd.DataFrame,
    subject: str,
    save_dir: Path,
    logger: logging.Logger,
    config: Any,
) -> None:
    """Power by ROI, Band, and Condition (plateau timing).
    
    Creates a grid: rows = ROIs (9), cols = frequency bands (5).
    Each cell shows pain vs non-pain box+strip comparison.
    FDR-corrected significance markers applied across all tests.
    """
    if features_df is None or features_df.empty or events_df is None:
        return

    pain_mask = extract_pain_mask(events_df, config)
    if pain_mask is None:
        return
    
    from eeg_pipeline.utils.config.loader import get_config_value
    from statsmodels.stats.multitest import multipletests
    
    plateau_window = get_config_value(config, "plateau_window", [3.0, 10.5])
    plateau_label = f"{plateau_window[0]:.1f}-{plateau_window[1]:.1f}s"
    
    rois = get_roi_definitions(config)
    if not rois:
        if logger:
            logger.warning("No ROI definitions found in config")
        return
    
    all_channels = extract_channels_from_columns(list(features_df.columns))
    
    bands, band_colors, condition_colors = _get_bands_and_palettes(config)
    roi_names = list(rois.keys())
    n_rois = len(roi_names)
    n_bands = len(bands)
    
    # First pass: collect all data and compute statistics
    plot_data = {}
    all_pvalues = []
    pvalue_keys = []
    
    for row_idx, roi_name in enumerate(roi_names):
        roi_patterns = rois[roi_name]
        roi_channels = get_roi_channels(roi_patterns, all_channels)
        
        for col_idx, band in enumerate(bands):
            key = (row_idx, col_idx)
            roi_vals = aggregate_by_roi(
                features_df, 
                f"power_plateau_{band}_",
                roi_channels
            )
            
            vals_nonpain = roi_vals[~pain_mask].dropna().values
            vals_pain = roi_vals[pain_mask].dropna().values
            
            plot_data[key] = (vals_nonpain, vals_pain)
            
            if len(vals_nonpain) > 3 and len(vals_pain) > 3:
                _, p = stats.mannwhitneyu(vals_nonpain, vals_pain)
                pooled_std = np.sqrt(((len(vals_nonpain) - 1) * np.var(vals_nonpain, ddof=1) + 
                                     (len(vals_pain) - 1) * np.var(vals_pain, ddof=1)) / 
                                    (len(vals_nonpain) + len(vals_pain) - 2))
                d = (np.mean(vals_pain) - np.mean(vals_nonpain)) / pooled_std if pooled_std > 0 else 0
                all_pvalues.append(p)
                pvalue_keys.append((key, p, d))
    
    # FDR correction
    qvalues = {}
    if all_pvalues:
        rejected, qvals, _, _ = multipletests(all_pvalues, method='fdr_bh')
        for i, (key, p, d) in enumerate(pvalue_keys):
            qvalues[key] = (p, qvals[i], d, rejected[i])
    
    # Second pass: plot with corrected significance
    plot_cfg = get_plot_config(config)
    fig, axes = plt.subplots(n_rois, n_bands, figsize=(16, 2.5 * n_rois))
    
    for row_idx, roi_name in enumerate(roi_names):
        for col_idx, band in enumerate(bands):
            ax = axes[row_idx, col_idx]
            key = (row_idx, col_idx)
            vals_nonpain, vals_pain = plot_data[key]
            
            if len(vals_nonpain) == 0 and len(vals_pain) == 0:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", 
                       transform=ax.transAxes, fontsize=plot_cfg.font.medium, color="gray")
                ax.set_xticks([])
                continue
            
            if len(vals_nonpain) > 0 and len(vals_pain) > 0:
                bp = ax.boxplot([vals_nonpain, vals_pain], 
                               positions=[0, 1], widths=0.4, patch_artist=True)
                bp["boxes"][0].set_facecolor(condition_colors["nonpain"])
                bp["boxes"][0].set_alpha(0.6)
                bp["boxes"][1].set_facecolor(condition_colors["pain"])
                bp["boxes"][1].set_alpha(0.6)
                
                ax.scatter(np.random.uniform(-0.08, 0.08, len(vals_nonpain)), 
                          vals_nonpain, c=condition_colors["nonpain"], alpha=0.3, s=6)
                ax.scatter(1 + np.random.uniform(-0.08, 0.08, len(vals_pain)), 
                          vals_pain, c=condition_colors["pain"], alpha=0.3, s=6)
                
                if key in qvalues:
                    p, q, d, sig = qvalues[key]
                    sig_marker = "†" if sig else ""  # † = FDR significant
                    ax.text(0.5, 0.95, f"q={q:.3f}{sig_marker}\nd={d:.2f}", 
                           transform=ax.transAxes, ha="center", fontsize=plot_cfg.font.annotation, va="top")
            
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["NP", "P"], fontsize=plot_cfg.font.small)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            
            if row_idx == 0:
                ax.set_title(band.capitalize(), fontweight="bold", 
                           color=band_colors.get(band, None), fontsize=plot_cfg.font.title)
            if col_idx == 0:
                short_name = roi_name.replace("_", "\n").replace("Contra", "C").replace("Ipsi", "I")
                ax.set_ylabel(short_name, fontsize=plot_cfg.font.medium)
    
    n_pain = int(pain_mask.sum())
    n_nonpain = int((~pain_mask).sum())
    n_tests = len(all_pvalues)
    n_sig = sum(1 for k in qvalues if qvalues[k][3])
    
    title = (f"Band Power by ROI: Pain vs Non-Pain Condition Comparison\n"
             f"Plateau Phase ({plateau_label}), Baseline-Normalized dB\n"
             f"Subject: {subject} | N: {n_nonpain} non-pain, {n_pain} pain | "
             f"FDR: {n_sig}/{n_tests} significant (†=q<0.05)")
    fig.suptitle(title, fontsize=plot_cfg.font.suptitle, fontweight="bold", y=1.02)
    
    plt.tight_layout()
    save_fig(fig, save_dir / f"sub-{subject}_power_roi_band_condition", 
             formats=plot_cfg.formats, dpi=plot_cfg.dpi,
             bbox_inches=plot_cfg.bbox_inches, pad_inches=plot_cfg.pad_inches)
    plt.close(fig)
    
    if logger:
        logger.info(f"Saved power ROI × band × condition plot ({n_sig}/{n_tests} FDR significant)")


def plot_dynamics_by_roi_band_condition(
    features_df: pd.DataFrame,
    events_df: pd.DataFrame,
    subject: str,
    save_dir: Path,
    logger: logging.Logger,
    config: Any,
    metric: str = "lzc",
) -> None:
    """Complexity (LZC/PE/Hjorth) by ROI, Band, and Condition.
    
    Creates a grid: rows = ROIs (9), cols = frequency bands (5).
    FDR-corrected significance markers applied across all tests.
    """
    if features_df is None or features_df.empty or events_df is None:
        return

    pain_mask = extract_pain_mask(events_df, config)
    if pain_mask is None:
        return
    
    from eeg_pipeline.utils.config.loader import get_config_value
    from statsmodels.stats.multitest import multipletests
    
    plateau_window = get_config_value(config, "plateau_window", [3.0, 10.5])
    plateau_label = f"{plateau_window[0]:.1f}-{plateau_window[1]:.1f}s"
    
    rois = get_roi_definitions(config)
    if not rois:
        return
    
    _, _, condition_colors = _get_bands_and_palettes(config)
    
    all_channels = extract_channels_from_columns(list(features_df.columns))
    
    bands, band_colors, condition_colors = _get_bands_and_palettes(config)
    roi_names = list(rois.keys())
    n_rois = len(roi_names)
    n_bands = len(bands)
    
    metric_labels = {"lzc": "Lempel-Ziv Complexity", "pe": "Permutation Entropy", 
                     "hjorth_mobility": "Hjorth Mobility", "hjorth_complexity": "Hjorth Complexity"}
    metric_label = metric_labels.get(metric, metric.upper())
    
    # First pass: collect data and statistics
    plot_data = {}
    all_pvalues = []
    pvalue_keys = []
    
    for row_idx, roi_name in enumerate(roi_names):
        roi_patterns = rois[roi_name]
        roi_channels = get_roi_channels(roi_patterns, all_channels)
        
        for col_idx, band in enumerate(bands):
            key = (row_idx, col_idx)
            
            metric_cols = [c for c in features_df.columns 
                          if f"dynamics_plateau_{band}_" in c and f"_{metric}" in c]
            if metric_cols:
                roi_metric_cols = [c for c in metric_cols 
                                  if any(f"_ch_{ch}_" in c or c.endswith(f"_ch_{ch}") 
                                        for ch in roi_channels)]
                if roi_metric_cols:
                    roi_vals = features_df[roi_metric_cols].mean(axis=1)
                else:
                    roi_vals = pd.Series([np.nan] * len(features_df))
            else:
                roi_vals = pd.Series([np.nan] * len(features_df))
            
            vals_nonpain = roi_vals[~pain_mask].dropna().values
            vals_pain = roi_vals[pain_mask].dropna().values
            plot_data[key] = (vals_nonpain, vals_pain)
            
            if len(vals_nonpain) > 3 and len(vals_pain) > 3:
                _, p = stats.mannwhitneyu(vals_nonpain, vals_pain)
                pooled_std = np.sqrt(((len(vals_nonpain) - 1) * np.var(vals_nonpain, ddof=1) + 
                                     (len(vals_pain) - 1) * np.var(vals_pain, ddof=1)) / 
                                    (len(vals_nonpain) + len(vals_pain) - 2))
                d = (np.mean(vals_pain) - np.mean(vals_nonpain)) / pooled_std if pooled_std > 0 else 0
                all_pvalues.append(p)
                pvalue_keys.append((key, p, d))
    
    # FDR correction
    qvalues = {}
    if all_pvalues:
        rejected, qvals, _, _ = multipletests(all_pvalues, method='fdr_bh')
        for i, (key, p, d) in enumerate(pvalue_keys):
            qvalues[key] = (p, qvals[i], d, rejected[i])
    
    # Second pass: plot
    plot_cfg = get_plot_config(config)
    fig, axes = plt.subplots(n_rois, n_bands, figsize=(16, 2.5 * n_rois))
    
    for row_idx, roi_name in enumerate(roi_names):
        for col_idx, band in enumerate(bands):
            ax = axes[row_idx, col_idx]
            key = (row_idx, col_idx)
            vals_nonpain, vals_pain = plot_data[key]
            
            if len(vals_nonpain) == 0 and len(vals_pain) == 0:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", 
                       transform=ax.transAxes, fontsize=plot_cfg.font.medium, color="gray")
                ax.set_xticks([])
                continue
            
            if len(vals_nonpain) > 0 and len(vals_pain) > 0:
                bp = ax.boxplot([vals_nonpain, vals_pain], 
                               positions=[0, 1], widths=0.4, patch_artist=True)
                bp["boxes"][0].set_facecolor(condition_colors["nonpain"])
                bp["boxes"][0].set_alpha(0.6)
                bp["boxes"][1].set_facecolor(condition_colors["pain"])
                bp["boxes"][1].set_alpha(0.6)
                
                ax.scatter(np.random.uniform(-0.08, 0.08, len(vals_nonpain)), 
                          vals_nonpain, c=condition_colors["nonpain"], alpha=0.3, s=6)
                ax.scatter(1 + np.random.uniform(-0.08, 0.08, len(vals_pain)), 
                          vals_pain, c=condition_colors["pain"], alpha=0.3, s=6)
                
                if key in qvalues:
                    p, q, d, sig = qvalues[key]
                    sig_marker = "†" if sig else ""
                    ax.text(0.5, 0.95, f"q={q:.3f}{sig_marker}\nd={d:.2f}", 
                           transform=ax.transAxes, ha="center", fontsize=plot_cfg.font.annotation, va="top")
            
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["NP", "P"], fontsize=plot_cfg.font.small)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            
            if row_idx == 0:
                ax.set_title(band.capitalize(), fontweight="bold", 
                           color=band_colors.get(band, None), fontsize=plot_cfg.font.title)
            if col_idx == 0:
                short_name = roi_name.replace("_", "\n").replace("Contra", "C").replace("Ipsi", "I")
                ax.set_ylabel(short_name, fontsize=plot_cfg.font.medium)
    
    n_pain = int(pain_mask.sum())
    n_nonpain = int((~pain_mask).sum())
    n_tests = len(all_pvalues)
    n_sig = sum(1 for k in qvalues if qvalues[k][3])
    
    title = (f"{metric_label} by ROI: Pain vs Non-Pain Condition Comparison\n"
             f"Plateau Phase ({plateau_label})\n"
             f"Subject: {subject} | N: {n_nonpain} non-pain, {n_pain} pain | "
             f"FDR: {n_sig}/{n_tests} significant (†=q<0.05)")
    fig.suptitle(title, fontsize=plot_cfg.font.suptitle, fontweight="bold", y=1.02)
    
    plt.tight_layout()
    save_fig(fig, save_dir / f"sub-{subject}_{metric}_roi_band_condition", 
             formats=plot_cfg.formats, dpi=plot_cfg.dpi,
             bbox_inches=plot_cfg.bbox_inches, pad_inches=plot_cfg.pad_inches)
    plt.close(fig)
    
    if logger:
        logger.info(f"Saved {metric} ROI × band × condition plot ({n_sig}/{n_tests} FDR significant)")


__all__ = [
    "get_roi_definitions",
    "get_roi_channels",
    "plot_power_by_roi_band_condition",
    "plot_dynamics_by_roi_band_condition",
    "plot_connectivity_by_roi_band_condition",
    "plot_aperiodic_by_roi_condition",
    "plot_itpc_by_roi_band_condition",
    "plot_itpc_plateau_vs_baseline",
    "plot_pac_by_roi_condition",
    "plot_band_segment_condition",
    "plot_power_plateau_vs_baseline",
    "plot_temporal_evolution",
    "plot_feature_correlation_heatmap",
]


def plot_aperiodic_by_roi_condition(
    features_df: pd.DataFrame,
    events_df: pd.DataFrame,
    subject: str,
    save_dir: Path,
    logger: logging.Logger,
    config: Any,
) -> None:
    """Aperiodic slope and offset by ROI and Condition.
    
    Creates a grid: rows = ROIs (9), cols = metrics (slope, offset).
    FDR-corrected significance markers applied across all tests.
    """
    if features_df is None or features_df.empty or events_df is None:
        return

    pain_mask = extract_pain_mask(events_df, config)
    if pain_mask is None:
        return
    
    from eeg_pipeline.utils.config.loader import get_config_value
    from statsmodels.stats.multitest import multipletests
    
    plateau_window = get_config_value(config, "plateau_window", [3.0, 10.5])
    plateau_label = f"{plateau_window[0]:.1f}-{plateau_window[1]:.1f}s"
    
    rois = get_roi_definitions(config)
    if not rois:
        return
    
    all_channels = extract_channels_from_columns(list(features_df.columns))
    
    roi_names = list(rois.keys())
    n_rois = len(roi_names)
    metrics = ["slope", "offset"]
    n_metrics = len(metrics)
    
    condition_colors = get_condition_colors(config)
    
    # First pass: collect data and statistics
    plot_data = {}
    all_pvalues = []
    pvalue_keys = []
    
    for row_idx, roi_name in enumerate(roi_names):
        roi_patterns = rois[roi_name]
        roi_channels = get_roi_channels(roi_patterns, all_channels)
        
        for col_idx, metric in enumerate(metrics):
            key = (row_idx, col_idx)
            
            cols = []
            for col in features_df.columns:
                if f"aperiodic_plateau_broadband_ch_" in col and f"_{metric}" in col:
                    for ch in roi_channels:
                        if f"_ch_{ch}_" in col:
                            cols.append(col)
                            break
            
            if cols:
                roi_vals = features_df[cols].mean(axis=1)
                vals_nonpain = roi_vals[~pain_mask].dropna().values
                vals_pain = roi_vals[pain_mask].dropna().values
            else:
                vals_nonpain = np.array([])
                vals_pain = np.array([])
            
            plot_data[key] = (vals_nonpain, vals_pain)
            
            if len(vals_nonpain) > 3 and len(vals_pain) > 3:
                _, p = stats.mannwhitneyu(vals_nonpain, vals_pain)
                pooled_std = np.sqrt(((len(vals_nonpain) - 1) * np.var(vals_nonpain, ddof=1) + 
                                     (len(vals_pain) - 1) * np.var(vals_pain, ddof=1)) / 
                                    (len(vals_nonpain) + len(vals_pain) - 2))
                d = (np.mean(vals_pain) - np.mean(vals_nonpain)) / pooled_std if pooled_std > 0 else 0
                all_pvalues.append(p)
                pvalue_keys.append((key, p, d))
    
    # FDR correction
    qvalues = {}
    if all_pvalues:
        rejected, qvals, _, _ = multipletests(all_pvalues, method='fdr_bh')
        for i, (key, p, d) in enumerate(pvalue_keys):
            qvalues[key] = (p, qvals[i], d, rejected[i])
    
    # Second pass: plot
    plot_cfg = get_plot_config(config)
    fig, axes = plt.subplots(n_rois, n_metrics, figsize=(8, 2.5 * n_rois))
    
    for row_idx, roi_name in enumerate(roi_names):
        for col_idx, metric in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            key = (row_idx, col_idx)
            vals_nonpain, vals_pain = plot_data[key]
            
            if len(vals_nonpain) == 0 and len(vals_pain) == 0:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", 
                       transform=ax.transAxes, fontsize=plot_cfg.font.medium, color="gray")
                ax.set_xticks([])
                continue
            
            if len(vals_nonpain) > 0 and len(vals_pain) > 0:
                bp = ax.boxplot([vals_nonpain, vals_pain], 
                               positions=[0, 1], widths=0.4, patch_artist=True)
                bp["boxes"][0].set_facecolor(condition_colors["nonpain"])
                bp["boxes"][0].set_alpha(0.6)
                bp["boxes"][1].set_facecolor(condition_colors["pain"])
                bp["boxes"][1].set_alpha(0.6)
                
                ax.scatter(np.random.uniform(-0.08, 0.08, len(vals_nonpain)), 
                          vals_nonpain, c=condition_colors["nonpain"], alpha=0.3, s=6)
                ax.scatter(1 + np.random.uniform(-0.08, 0.08, len(vals_pain)), 
                          vals_pain, c=condition_colors["pain"], alpha=0.3, s=6)
                
                if key in qvalues:
                    p, q, d, sig = qvalues[key]
                    sig_marker = "†" if sig else ""
                    ax.text(0.5, 0.95, f"q={q:.3f}{sig_marker}\nd={d:.2f}", 
                           transform=ax.transAxes, ha="center", fontsize=plot_cfg.font.annotation, va="top")
            
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["NP", "P"], fontsize=plot_cfg.font.small)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            
            if row_idx == 0:
                ax.set_title(metric.capitalize(), fontweight="bold", fontsize=plot_cfg.font.title)
            if col_idx == 0:
                short_name = roi_name.replace("_", "\n").replace("Contra", "C").replace("Ipsi", "I")
                ax.set_ylabel(short_name, fontsize=plot_cfg.font.medium)
    
    n_pain = int(pain_mask.sum())
    n_nonpain = int((~pain_mask).sum())
    n_tests = len(all_pvalues)
    n_sig = sum(1 for k in qvalues if qvalues[k][3])
    
    title = (f"Aperiodic 1/f Parameters by ROI: Pain vs Non-Pain Comparison\n"
             f"Plateau Phase ({plateau_label})\n"
             f"Subject: {subject} | N: {n_nonpain} non-pain, {n_pain} pain | "
             f"FDR: {n_sig}/{n_tests} significant (†=q<0.05)")
    fig.suptitle(title, fontsize=plot_cfg.font.suptitle, fontweight="bold", y=1.02)
    
    plt.tight_layout()
    save_fig(fig, save_dir / f"sub-{subject}_aperiodic_roi_condition", 
             formats=plot_cfg.formats, dpi=plot_cfg.dpi,
             bbox_inches=plot_cfg.bbox_inches, pad_inches=plot_cfg.pad_inches)
    plt.close(fig)
    
    if logger:
        logger.info(f"Saved aperiodic ROI × condition plot ({n_sig}/{n_tests} FDR significant)")


def extract_channel_pairs_from_columns(columns: List[str]) -> List[Tuple[str, str]]:
    """Extract unique channel pairs from connectivity column names."""
    pairs = set()
    for col in columns:
        match = re.search(r'_chpair_([A-Za-z0-9]+)-([A-Za-z0-9]+)_', col)
        if match:
            pairs.add((match.group(1), match.group(2)))
    return list(pairs)


def aggregate_connectivity_by_roi(
    features_df: pd.DataFrame,
    col_pattern: str,
    roi_channels: List[str],
) -> pd.Series:
    """Aggregate connectivity within ROI (mean of edges where both channels in ROI)."""
    cols = []
    for col in features_df.columns:
        if col_pattern in col:
            match = re.search(r'_chpair_([A-Za-z0-9]+)-([A-Za-z0-9]+)_', col)
            if match:
                ch1, ch2 = match.group(1), match.group(2)
                if ch1 in roi_channels and ch2 in roi_channels:
                    cols.append(col)
    
    if not cols:
        return pd.Series([np.nan] * len(features_df), index=features_df.index)
    
    return features_df[cols].mean(axis=1)


def plot_connectivity_by_roi_band_condition(
    features_df: pd.DataFrame,
    events_df: pd.DataFrame,
    subject: str,
    save_dir: Path,
    logger: logging.Logger,
    config: Any,
    measure: str = "wpli",
) -> None:
    """Within-ROI connectivity by Band and Condition.
    
    Creates a grid: rows = ROIs (9), cols = frequency bands (5).
    Each cell shows pain vs non-pain within-ROI mean connectivity.
    """
    if features_df is None or features_df.empty or events_df is None:
        return

    pain_mask = extract_pain_mask(events_df, config)
    if pain_mask is None:
        return
    
    from eeg_pipeline.utils.config.loader import get_config_value
    
    plateau_window = get_config_value(config, "plateau_window", [3.0, 10.5])
    plateau_label = f"{plateau_window[0]:.1f}-{plateau_window[1]:.1f}s"
    
    rois = get_roi_definitions(config)
    if not rois:
        return
    
    all_pairs = extract_channel_pairs_from_columns(list(features_df.columns))
    all_channels = list(set([p[0] for p in all_pairs] + [p[1] for p in all_pairs]))
    
    bands, band_colors, condition_colors = _get_bands_and_palettes(config)
    roi_names = list(rois.keys())
    n_rois = len(roi_names)
    n_bands = len(bands)
    
    measure_labels = {"wpli": "wPLI", "aec": "AEC"}
    measure_label = measure_labels.get(measure, measure.upper())
    
    plot_cfg = get_plot_config(config)
    fig, axes = plt.subplots(n_rois, n_bands, figsize=(16, 2.5 * n_rois))
    
    for row_idx, roi_name in enumerate(roi_names):
        roi_patterns = rois[roi_name]
        roi_channels = get_roi_channels(roi_patterns, all_channels)
        
        for col_idx, band in enumerate(bands):
            ax = axes[row_idx, col_idx]
            
            roi_vals = aggregate_connectivity_by_roi(
                features_df, 
                f"conn_plateau_{band}_",
                roi_channels
            )
            
            if measure != "wpli":
                measure_cols = [c for c in features_df.columns 
                              if f"conn_plateau_{band}_" in c and f"_{measure}" in c]
                if measure_cols:
                    roi_measure_cols = []
                    for c in measure_cols:
                        match = re.search(r'_chpair_([A-Za-z0-9]+)-([A-Za-z0-9]+)_', c)
                        if match and match.group(1) in roi_channels and match.group(2) in roi_channels:
                            roi_measure_cols.append(c)
                    if roi_measure_cols:
                        roi_vals = features_df[roi_measure_cols].mean(axis=1)
            
            if roi_vals.isna().all():
                ax.text(0.5, 0.5, "No data", ha="center", va="center", 
                       transform=ax.transAxes, fontsize=plot_cfg.font.medium, color="gray")
                ax.set_xticks([])
                continue
            
            vals_nonpain = roi_vals[~pain_mask].dropna().values
            vals_pain = roi_vals[pain_mask].dropna().values
            
            if len(vals_nonpain) > 0 and len(vals_pain) > 0:
                bp = ax.boxplot([vals_nonpain, vals_pain], 
                               positions=[0, 1], widths=0.4, patch_artist=True)
                bp["boxes"][0].set_facecolor(condition_colors["nonpain"])
                bp["boxes"][0].set_alpha(0.6)
                bp["boxes"][1].set_facecolor(condition_colors["pain"])
                bp["boxes"][1].set_alpha(0.6)
                
                ax.scatter(np.random.uniform(-0.08, 0.08, len(vals_nonpain)), 
                          vals_nonpain, c=condition_colors["nonpain"], alpha=0.3, s=6)
                ax.scatter(1 + np.random.uniform(-0.08, 0.08, len(vals_pain)), 
                          vals_pain, c=condition_colors["pain"], alpha=0.3, s=6)
                
                if len(vals_nonpain) > 3 and len(vals_pain) > 3:
                    _, p = stats.mannwhitneyu(vals_nonpain, vals_pain)
                    pooled_std = np.sqrt(((len(vals_nonpain) - 1) * np.var(vals_nonpain, ddof=1) + 
                                         (len(vals_pain) - 1) * np.var(vals_pain, ddof=1)) / 
                                        (len(vals_nonpain) + len(vals_pain) - 2))
                    d = (np.mean(vals_pain) - np.mean(vals_nonpain)) / pooled_std if pooled_std > 0 else 0
                    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                    ax.text(0.5, 0.95, f"p={p:.3f}{sig}\nd={d:.2f}", transform=ax.transAxes, 
                           ha="center", fontsize=plot_cfg.font.annotation, va="top")
            
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["NP", "P"], fontsize=plot_cfg.font.small)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            
            if row_idx == 0:
                ax.set_title(band.capitalize(), fontweight="bold", 
                           color=band_colors.get(band, None), fontsize=plot_cfg.font.title)
            if col_idx == 0:
                short_name = roi_name.replace("_", "\n").replace("Contra", "C").replace("Ipsi", "I")
                ax.set_ylabel(short_name, fontsize=plot_cfg.font.medium)
    n_pain = int(pain_mask.sum())
    n_nonpain = int((~pain_mask).sum())
    fig.suptitle(f"Within-ROI {measure_label} by Band: Plateau ({plateau_label}) (sub-{subject})\nN: {n_nonpain} NP, {n_pain} P", 
                fontsize=plot_cfg.font.figure_title, fontweight="bold", y=1.01)
    
    plt.tight_layout()
    save_fig(fig, save_dir / f"sub-{subject}_conn_{measure}_roi_band_condition", 
             formats=plot_cfg.formats, dpi=plot_cfg.dpi,
             bbox_inches=plot_cfg.bbox_inches, pad_inches=plot_cfg.pad_inches)
    plt.close(fig)
    
    if logger:
        logger.info(f"Saved {measure} ROI × band × condition plot")


def plot_itpc_by_roi_band_condition(
    features_df: pd.DataFrame,
    events_df: pd.DataFrame,
    subject: str,
    save_dir: Path,
    logger: logging.Logger,
    config: Any,
    segment: str = "plateau",
) -> None:
    """ITPC by ROI, Band, and Condition.
    
    Creates a grid: rows = ROIs (9), cols = frequency bands (5).
    Each cell shows pain vs non-pain box+strip comparison.
    FDR-corrected significance markers applied across all tests.
    """
    if features_df is None or features_df.empty or events_df is None:
        return

    pain_mask = extract_pain_mask(events_df, config)
    if pain_mask is None:
        return
    
    from eeg_pipeline.utils.config.loader import get_config_value
    from statsmodels.stats.multitest import multipletests
    
    plateau_window = get_config_value(config, "plateau_window", [3.0, 10.5])
    plateau_label = f"{plateau_window[0]:.1f}-{plateau_window[1]:.1f}s"
    
    rois = get_roi_definitions(config)
    if not rois:
        if logger:
            logger.warning("No ROI definitions found in config")
        return
    
    all_channels = extract_channels_from_columns(list(features_df.columns))
    
    bands, band_colors, condition_colors = _get_bands_and_palettes(config)
    roi_names = list(rois.keys())
    n_rois = len(roi_names)
    n_bands = len(bands)
    
    # First pass: collect all data and compute statistics
    plot_data = {}
    all_pvalues = []
    pvalue_keys = []
    
    for row_idx, roi_name in enumerate(roi_names):
        roi_patterns = rois[roi_name]
        roi_channels = get_roi_channels(roi_patterns, all_channels)
        
        for col_idx, band in enumerate(bands):
            key = (row_idx, col_idx)
            
            # Find ITPC columns for this band, segment, and ROI channels
            # Column pattern: itpc_{segment}_{band}_ch_{channel}_val
            itpc_cols = [c for c in features_df.columns 
                        if f"itpc_{segment}_{band}_ch_" in c]
            if itpc_cols:
                roi_itpc_cols = [c for c in itpc_cols 
                                if any(f"_ch_{ch}_" in c or c.endswith(f"_ch_{ch}") 
                                      for ch in roi_channels)]
                if roi_itpc_cols:
                    roi_vals = features_df[roi_itpc_cols].mean(axis=1)
                else:
                    roi_vals = pd.Series([np.nan] * len(features_df))
            else:
                roi_vals = pd.Series([np.nan] * len(features_df))
            
            vals_nonpain = roi_vals[~pain_mask].dropna().values
            vals_pain = roi_vals[pain_mask].dropna().values
            
            plot_data[key] = (vals_nonpain, vals_pain)
            
            if len(vals_nonpain) > 3 and len(vals_pain) > 3:
                _, p = stats.mannwhitneyu(vals_nonpain, vals_pain)
                pooled_std = np.sqrt(((len(vals_nonpain) - 1) * np.var(vals_nonpain, ddof=1) + 
                                     (len(vals_pain) - 1) * np.var(vals_pain, ddof=1)) / 
                                    (len(vals_nonpain) + len(vals_pain) - 2))
                d = (np.mean(vals_pain) - np.mean(vals_nonpain)) / pooled_std if pooled_std > 0 else 0
                all_pvalues.append(p)
                pvalue_keys.append((key, p, d))
    
    # FDR correction
    qvalues = {}
    if all_pvalues:
        rejected, qvals, _, _ = multipletests(all_pvalues, method='fdr_bh')
        for i, (key, p, d) in enumerate(pvalue_keys):
            qvalues[key] = (p, qvals[i], d, rejected[i])
    
    # Second pass: plot with corrected significance
    plot_cfg = get_plot_config(config)
    fig, axes = plt.subplots(n_rois, n_bands, figsize=(16, 2.5 * n_rois))
    
    for row_idx, roi_name in enumerate(roi_names):
        for col_idx, band in enumerate(bands):
            ax = axes[row_idx, col_idx]
            key = (row_idx, col_idx)
            vals_nonpain, vals_pain = plot_data[key]
            
            if len(vals_nonpain) == 0 and len(vals_pain) == 0:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", 
                       transform=ax.transAxes, fontsize=plot_cfg.font.medium, color="gray")
                ax.set_xticks([])
                continue
            
            if len(vals_nonpain) > 0 and len(vals_pain) > 0:
                bp = ax.boxplot([vals_nonpain, vals_pain], 
                               positions=[0, 1], widths=0.4, patch_artist=True)
                bp["boxes"][0].set_facecolor(condition_colors["nonpain"])
                bp["boxes"][0].set_alpha(0.6)
                bp["boxes"][1].set_facecolor(condition_colors["pain"])
                bp["boxes"][1].set_alpha(0.6)
                
                ax.scatter(np.random.uniform(-0.08, 0.08, len(vals_nonpain)), 
                          vals_nonpain, c=condition_colors["nonpain"], alpha=0.3, s=6)
                ax.scatter(1 + np.random.uniform(-0.08, 0.08, len(vals_pain)), 
                          vals_pain, c=condition_colors["pain"], alpha=0.3, s=6)
                
                # Auto-scale y-axis with padding
                all_vals = np.concatenate([vals_nonpain, vals_pain])
                ymin, ymax = np.nanmin(all_vals), np.nanmax(all_vals)
                yrange = ymax - ymin if ymax > ymin else 0.1
                ax.set_ylim(ymin - 0.1 * yrange, ymax + 0.25 * yrange)
                
                if key in qvalues:
                    p, q, d, sig = qvalues[key]
                    sig_marker = "†" if sig else ""
                    sig_color = "#d62728" if sig else "#333333"
                    ax.annotate(f"q={q:.3f}{sig_marker}\nd={d:.2f}", 
                               xy=(0.5, ymax + 0.05 * yrange), 
                               ha="center", fontsize=plot_cfg.font.annotation, color=sig_color,
                               fontweight="bold" if sig else "normal")
            
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["NP", "P"], fontsize=plot_cfg.font.small)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            
            if row_idx == 0:
                ax.set_title(band.capitalize(), fontweight="bold", 
                           color=band_colors.get(band, None), fontsize=plot_cfg.font.title)
            if col_idx == 0:
                short_name = roi_name.replace("_", "\n").replace("Contra", "C").replace("Ipsi", "I")
                ax.set_ylabel(short_name, fontsize=plot_cfg.font.medium)
    
    n_pain = int(pain_mask.sum())
    n_nonpain = int((~pain_mask).sum())
    n_tests = len(all_pvalues)
    n_sig = sum(1 for k in qvalues if qvalues[k][3])
    
    title = (f"Inter-Trial Phase Coherence by ROI: Pain vs Non-Pain Comparison\n"
             f"Plateau Phase ({plateau_label}), LOO-ITPC\n"
             f"Subject: {subject} | N: {n_nonpain} non-pain, {n_pain} pain | "
             f"FDR: {n_sig}/{n_tests} significant (†=q<0.05)")
    fig.suptitle(title, fontsize=plot_cfg.font.suptitle, fontweight="bold", y=1.02)
    
    plt.tight_layout()
    save_fig(fig, save_dir / f"sub-{subject}_itpc_roi_band_condition", 
             formats=plot_cfg.formats, dpi=plot_cfg.dpi,
             bbox_inches=plot_cfg.bbox_inches, pad_inches=plot_cfg.pad_inches)
    plt.close(fig)
    
    if logger:
        logger.info(f"Saved ITPC ROI × band × condition plot ({n_sig}/{n_tests} FDR significant)")


def plot_itpc_plateau_vs_baseline(
    features_df: pd.DataFrame,
    subject: str,
    save_dir: Path,
    logger: logging.Logger,
    config: Any,
) -> None:
    """ITPC Plateau vs Baseline Paired Comparison.
    
    Shows whether thermal stimulation evokes phase-locked responses.
    Paired comparison: each y-point is same trial (baseline vs plateau).
    """
    if features_df is None or features_df.empty:
        return
    
    from eeg_pipeline.utils.config.loader import get_config_value
    from statsmodels.stats.multitest import multipletests
    from scipy.stats import wilcoxon
    
    plateau_window = get_config_value(config, "plateau_window", [3.0, 10.5])
    baseline_window = get_config_value(config, "baseline_window", [-3.0, -0.5])
    
    rois = get_roi_definitions(config)
    if not rois:
        if logger:
            logger.warning("No ROI definitions found")
        return
    
    all_channels = extract_channels_from_columns(list(features_df.columns))
    
    bands, band_colors, _ = _get_bands_and_palettes(config)
    roi_names = list(rois.keys())
    n_rois = len(roi_names)
    n_bands = len(bands)
    
    # First pass: collect data and statistics
    plot_data = {}
    all_pvalues = []
    pvalue_keys = []
    
    for row_idx, roi_name in enumerate(roi_names):
        roi_patterns = rois[roi_name]
        roi_channels = get_roi_channels(roi_patterns, all_channels)
        
        for col_idx, band in enumerate(bands):
            key = (row_idx, col_idx)
            
            # Find baseline and plateau ITPC columns for this band and ROI
            baseline_cols = [c for c in features_df.columns 
                           if f"itpc_baseline_{band}_ch_" in c]
            plateau_cols = [c for c in features_df.columns 
                          if f"itpc_plateau_{band}_ch_" in c]
            
            vals_baseline = np.array([])
            vals_plateau = np.array([])
            
            if baseline_cols and plateau_cols:
                roi_baseline_cols = [c for c in baseline_cols 
                                    if any(f"_ch_{ch}_" in c for ch in roi_channels)]
                roi_plateau_cols = [c for c in plateau_cols 
                                   if any(f"_ch_{ch}_" in c for ch in roi_channels)]
                
                if roi_baseline_cols and roi_plateau_cols:
                    vals_baseline = features_df[roi_baseline_cols].mean(axis=1).dropna().values
                    vals_plateau = features_df[roi_plateau_cols].mean(axis=1).dropna().values
            
            plot_data[key] = (vals_baseline, vals_plateau)
            
            if len(vals_baseline) > 5 and len(vals_plateau) > 5 and len(vals_baseline) == len(vals_plateau):
                try:
                    _, p = wilcoxon(vals_plateau, vals_baseline)
                    diff = vals_plateau - vals_baseline
                    pooled_std = np.std(diff, ddof=1)
                    d = np.mean(diff) / pooled_std if pooled_std > 0 else 0
                    all_pvalues.append(p)
                    pvalue_keys.append((key, p, d))
                except Exception:
                    pass
    
    # FDR correction
    qvalues = {}
    if all_pvalues:
        rejected, qvals, _, _ = multipletests(all_pvalues, method='fdr_bh')
        for i, (key, p, d) in enumerate(pvalue_keys):
            qvalues[key] = (p, qvals[i], d, rejected[i])
    
    # Second pass: plot
    plot_cfg = get_plot_config(config)
    fig, axes = plt.subplots(n_rois, n_bands, figsize=(16, 2.5 * n_rois))
    
    for row_idx, roi_name in enumerate(roi_names):
        for col_idx, band in enumerate(bands):
            ax = axes[row_idx, col_idx]
            key = (row_idx, col_idx)
            vals_baseline, vals_plateau = plot_data[key]
            
            if len(vals_baseline) == 0 or len(vals_plateau) == 0:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", 
                       transform=ax.transAxes, fontsize=plot_cfg.font.medium, color="gray")
                ax.set_xticks([])
                continue
            
            bp = ax.boxplot([vals_baseline, vals_plateau], 
                           positions=[0, 1], widths=0.4, patch_artist=True)
            bp["boxes"][0].set_facecolor("#1f77b4")  # Blue: baseline
            bp["boxes"][0].set_alpha(0.6)
            bp["boxes"][1].set_facecolor("#d62728")  # Red: plateau
            bp["boxes"][1].set_alpha(0.6)
            
            ax.scatter(np.random.uniform(-0.08, 0.08, len(vals_baseline)), 
                      vals_baseline, c="#1f77b4", alpha=0.3, s=6)
            ax.scatter(1 + np.random.uniform(-0.08, 0.08, len(vals_plateau)), 
                      vals_plateau, c="#d62728", alpha=0.3, s=6)
            
            # Draw paired lines (sample to avoid clutter)
            if len(vals_baseline) == len(vals_plateau) and len(vals_baseline) <= 100:
                for i in range(len(vals_baseline)):
                    ax.plot([0, 1], [vals_baseline[i], vals_plateau[i]], 
                           c="gray", alpha=0.15, lw=0.5)
            
            # Auto-scale y-axis with padding
            all_vals = np.concatenate([vals_baseline, vals_plateau])
            ymin, ymax = np.nanmin(all_vals), np.nanmax(all_vals)
            yrange = ymax - ymin if ymax > ymin else 0.1
            ax.set_ylim(ymin - 0.1 * yrange, ymax + 0.25 * yrange)
            
            if key in qvalues:
                p, q, d, sig = qvalues[key]
                sig_marker = "†" if sig else ""
                sig_color = "#d62728" if sig else "#333333"
                ax.annotate(f"q={q:.3f}{sig_marker}\nd={d:.2f}", 
                           xy=(0.5, ymax + 0.05 * yrange), 
                           ha="center", fontsize=plot_cfg.font.annotation, color=sig_color,
                           fontweight="bold" if sig else "normal")
            
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["BL", "PL"], fontsize=plot_cfg.font.small)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            
            if row_idx == 0:
                ax.set_title(band.capitalize(), fontweight="bold", 
                           color=band_colors.get(band, None), fontsize=plot_cfg.font.title)
            if col_idx == 0:
                short_name = roi_name.replace("_", "\n").replace("Contra", "C").replace("Ipsi", "I")
                ax.set_ylabel(short_name, fontsize=plot_cfg.font.medium)
    
    n_tests = len(all_pvalues)
    n_sig = sum(1 for k in qvalues if qvalues[k][3])
    
    title = (f"ITPC Plateau vs Baseline by ROI: Paired Comparison\n"
             f"Baseline ({baseline_window[0]:.1f}-{baseline_window[1]:.1f}s) vs "
             f"Plateau ({plateau_window[0]:.1f}-{plateau_window[1]:.1f}s)\n"
             f"Subject: {subject} | Wilcoxon signed-rank | "
             f"FDR: {n_sig}/{n_tests} significant (†=q<0.05)")
    fig.suptitle(title, fontsize=plot_cfg.font.suptitle, fontweight="bold", y=1.02)
    
    plt.tight_layout()
    save_fig(fig, save_dir / f"sub-{subject}_itpc_plateau_vs_baseline", 
             formats=plot_cfg.formats, dpi=plot_cfg.dpi,
             bbox_inches=plot_cfg.bbox_inches, pad_inches=plot_cfg.pad_inches)
    plt.close(fig)
    
    if logger:
        logger.info(f"Saved ITPC plateau vs baseline plot ({n_sig}/{n_tests} FDR significant)")


def plot_pac_by_roi_condition(
    features_df: pd.DataFrame,
    events_df: pd.DataFrame,
    subject: str,
    save_dir: Path,
    logger: logging.Logger,
    config: Any,
) -> None:
    """PAC by ROI × Frequency Pair × Condition.
    
    Shows phase-amplitude coupling for different band pairs.
    PAC columns: pac_plateau_{phase}_{amp}_ch_{channel}_val
    """
    if features_df is None or features_df.empty or events_df is None:
        return

    pain_mask = extract_pain_mask(events_df, config)
    if pain_mask is None:
        return
    
    from statsmodels.stats.multitest import multipletests
    
    rois = get_roi_definitions(config)
    if not rois:
        return
    
    all_channels = extract_channels_from_columns(list(features_df.columns))
    
    # Define PAC frequency pairs of interest
    from eeg_pipeline.utils.config.loader import get_config_value
    pac_pairs = get_config_value(config, "plotting.plots.features.pac_pairs", ["theta_beta", "theta_gamma", "alpha_beta", "alpha_gamma"])
    
    _, band_colors, condition_colors = _get_bands_and_palettes(config)
    roi_names = list(rois.keys())
    n_rois = len(roi_names)
    n_pairs = len(pac_pairs)
    
    # First pass: collect data and statistics
    plot_data = {}
    all_pvalues = []
    pvalue_keys = []
    
    for row_idx, roi_name in enumerate(roi_names):
        roi_patterns = rois[roi_name]
        roi_channels = get_roi_channels(roi_patterns, all_channels)
        
        for col_idx, pair in enumerate(pac_pairs):
            key = (row_idx, col_idx)
            
            # Find PAC columns for this frequency pair and ROI
            pac_cols = [c for c in features_df.columns 
                       if f"pac_plateau_{pair}_ch_" in c]
            
            vals_nonpain = np.array([])
            vals_pain = np.array([])
            
            if pac_cols:
                roi_pac_cols = [c for c in pac_cols 
                               if any(f"_ch_{ch}_" in c for ch in roi_channels)]
                if roi_pac_cols:
                    roi_vals = features_df[roi_pac_cols].mean(axis=1)
                    vals_nonpain = roi_vals[~pain_mask].dropna().values
                    vals_pain = roi_vals[pain_mask].dropna().values
            
            plot_data[key] = (vals_nonpain, vals_pain)
            
            if len(vals_nonpain) > 3 and len(vals_pain) > 3:
                _, p = stats.mannwhitneyu(vals_nonpain, vals_pain)
                pooled_std = np.sqrt(((len(vals_nonpain) - 1) * np.var(vals_nonpain, ddof=1) + 
                                     (len(vals_pain) - 1) * np.var(vals_pain, ddof=1)) / 
                                    (len(vals_nonpain) + len(vals_pain) - 2))
                d = (np.mean(vals_pain) - np.mean(vals_nonpain)) / pooled_std if pooled_std > 0 else 0
                all_pvalues.append(p)
                pvalue_keys.append((key, p, d))
    
    # FDR correction
    qvalues = {}
    if all_pvalues:
        rejected, qvals, _, _ = multipletests(all_pvalues, method='fdr_bh')
        for i, (key, p, d) in enumerate(pvalue_keys):
            qvalues[key] = (p, qvals[i], d, rejected[i])
    
    # Second pass: plot
    plot_cfg = get_plot_config(config)
    fig, axes = plt.subplots(n_rois, n_pairs, figsize=(12, 2.5 * n_rois))
    
    base_palette = list(band_colors.values()) if band_colors else ["#440154", "#3b528b", "#21918c", "#fde725"]
    pair_colors = [base_palette[i % len(base_palette)] for i in range(n_pairs)]
    
    for row_idx, roi_name in enumerate(roi_names):
        for col_idx, pair in enumerate(pac_pairs):
            ax = axes[row_idx, col_idx]
            key = (row_idx, col_idx)
            vals_nonpain, vals_pain = plot_data[key]
            
            if len(vals_nonpain) == 0 and len(vals_pain) == 0:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", 
                       transform=ax.transAxes, fontsize=plot_cfg.font.medium, color="gray")
                ax.set_xticks([])
                continue
            
            if len(vals_nonpain) > 0 and len(vals_pain) > 0:
                bp = ax.boxplot([vals_nonpain, vals_pain], 
                               positions=[0, 1], widths=0.4, patch_artist=True)
                bp["boxes"][0].set_facecolor(condition_colors["nonpain"])
                bp["boxes"][0].set_alpha(0.6)
                bp["boxes"][1].set_facecolor(condition_colors["pain"])
                bp["boxes"][1].set_alpha(0.6)
                
                ax.scatter(np.random.uniform(-0.08, 0.08, len(vals_nonpain)), 
                          vals_nonpain, c=condition_colors["nonpain"], alpha=0.3, s=6)
                ax.scatter(1 + np.random.uniform(-0.08, 0.08, len(vals_pain)), 
                          vals_pain, c=condition_colors["pain"], alpha=0.3, s=6)
                
                if key in qvalues:
                    p, q, d, sig = qvalues[key]
                    sig_marker = "†" if sig else ""
                    ax.text(0.5, 0.95, f"q={q:.3f}{sig_marker}\nd={d:.2f}", 
                           transform=ax.transAxes, ha="center", fontsize=plot_cfg.font.annotation, va="top")
            
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["NP", "P"], fontsize=plot_cfg.font.small)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            
            if row_idx == 0:
                pair_label = pair.replace("_", "→")
                ax.set_title(pair_label.capitalize(), fontweight="bold", 
                           color=pair_colors[col_idx], fontsize=plot_cfg.font.title)
            if col_idx == 0:
                short_name = roi_name.replace("_", "\n").replace("Contra", "C").replace("Ipsi", "I")
                ax.set_ylabel(short_name, fontsize=plot_cfg.font.medium)
    
    n_pain = int(pain_mask.sum())
    n_nonpain = int((~pain_mask).sum())
    n_tests = len(all_pvalues)
    n_sig = sum(1 for k in qvalues if qvalues[k][3])
    
    title = (f"Phase-Amplitude Coupling by ROI: Pain vs Non-Pain Comparison\n"
             f"Plateau Phase, MVL Method\n"
             f"Subject: {subject} | N: {n_nonpain} non-pain, {n_pain} pain | "
             f"FDR: {n_sig}/{n_tests} significant (†=q<0.05)")
    fig.suptitle(title, fontsize=plot_cfg.font.suptitle, fontweight="bold", y=1.02)
    
    plt.tight_layout()
    save_fig(fig, save_dir / f"sub-{subject}_pac_roi_condition", 
             formats=plot_cfg.formats, dpi=plot_cfg.dpi,
             bbox_inches=plot_cfg.bbox_inches, pad_inches=plot_cfg.pad_inches)
    plt.close(fig)
    
    if logger:
        logger.info(f"Saved PAC ROI × condition plot ({n_sig}/{n_tests} FDR significant)")


def plot_band_segment_condition(
    features_df: pd.DataFrame,
    events_df: pd.DataFrame,
    subject: str,
    save_dir: Path,
    logger: logging.Logger,
    config: Any,
    feature_prefix: str,
    feature_label: str,
    segments: List[str] = None,
) -> None:
    """Unified Band × Segment × Condition plot.
    
    Creates grid: rows = frequency bands, columns = segments (baseline, plateau).
    Each cell shows mean across ALL channels, comparing pain vs non-pain.
    
    Args:
        feature_prefix: Column pattern prefix (e.g., 'itpc', 'pow')
        feature_label: Human-readable label (e.g., 'ITPC', 'Band Power')
        segments: List of segments to plot (default: ['baseline', 'plateau'])
    """
    if features_df is None or features_df.empty or events_df is None:
        return

    pain_mask = extract_pain_mask(events_df, config)
    if pain_mask is None:
        return
    
    from statsmodels.stats.multitest import multipletests
    
    if segments is None:
        segments = ["baseline", "plateau"]
    
    bands, band_colors, condition_colors = _get_bands_and_palettes(config)
    n_bands = len(bands)
    n_segments = len(segments)
    
    # First pass: collect data and statistics
    plot_data = {}
    all_pvalues = []
    pvalue_keys = []
    
    for row_idx, band in enumerate(bands):
        for col_idx, segment in enumerate(segments):
            key = (row_idx, col_idx)
            
            # Find columns matching pattern: {prefix}_{segment}_{band}_ch_*
            pattern = f"{feature_prefix}_{segment}_{band}_ch_"
            cols = [c for c in features_df.columns if pattern in c]
            
            vals_nonpain = np.array([])
            vals_pain = np.array([])
            
            if cols:
                # Mean across ALL channels
                mean_vals = features_df[cols].mean(axis=1)
                vals_nonpain = mean_vals[~pain_mask].dropna().values
                vals_pain = mean_vals[pain_mask].dropna().values
            
            plot_data[key] = (vals_nonpain, vals_pain)
            
            if len(vals_nonpain) > 3 and len(vals_pain) > 3:
                _, p = stats.mannwhitneyu(vals_nonpain, vals_pain)
                pooled_std = np.sqrt(((len(vals_nonpain) - 1) * np.var(vals_nonpain, ddof=1) + 
                                     (len(vals_pain) - 1) * np.var(vals_pain, ddof=1)) / 
                                    (len(vals_nonpain) + len(vals_pain) - 2))
                d = (np.mean(vals_pain) - np.mean(vals_nonpain)) / pooled_std if pooled_std > 0 else 0
                all_pvalues.append(p)
                pvalue_keys.append((key, p, d))
    
    # Check if we have any data
    has_data = any(len(pdata[0]) > 0 or len(pdata[1]) > 0 for pdata in plot_data.values())
    if not has_data:
        if logger:
            logger.warning(f"No {feature_label} data found for band × segment × condition plot")
        return
    
    # FDR correction
    qvalues = {}
    if all_pvalues:
        rejected, qvals, _, _ = multipletests(all_pvalues, method='fdr_bh')
        for i, (key, p, d) in enumerate(pvalue_keys):
            qvalues[key] = (p, qvals[i], d, rejected[i])
    
    # Second pass: plot
    plot_cfg = get_plot_config(config)
    fig, axes = plt.subplots(n_bands, n_segments, figsize=(5 * n_segments, 3 * n_bands))
    
    for row_idx, band in enumerate(bands):
        for col_idx, segment in enumerate(segments):
            ax = axes[row_idx, col_idx]
            key = (row_idx, col_idx)
            vals_nonpain, vals_pain = plot_data[key]
            
            if len(vals_nonpain) == 0 and len(vals_pain) == 0:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", 
                       transform=ax.transAxes, fontsize=plot_cfg.font.title, color="gray")
                ax.set_xticks([])
                continue
            
            if len(vals_nonpain) > 0 and len(vals_pain) > 0:
                bp = ax.boxplot([vals_nonpain, vals_pain], 
                               positions=[0, 1], widths=0.5, patch_artist=True)
                bp["boxes"][0].set_facecolor(condition_colors["nonpain"])
                bp["boxes"][0].set_alpha(0.6)
                bp["boxes"][1].set_facecolor(condition_colors["pain"])
                bp["boxes"][1].set_alpha(0.6)
                
                ax.scatter(np.random.uniform(-0.1, 0.1, len(vals_nonpain)), 
                          vals_nonpain, c=condition_colors["nonpain"], alpha=0.4, s=10)
                ax.scatter(1 + np.random.uniform(-0.1, 0.1, len(vals_pain)), 
                          vals_pain, c=condition_colors["pain"], alpha=0.4, s=10)
                
                # Auto-scale y-axis with padding
                all_vals = np.concatenate([vals_nonpain, vals_pain])
                ymin, ymax = np.nanmin(all_vals), np.nanmax(all_vals)
                yrange = ymax - ymin if ymax > ymin else 1.0
                ax.set_ylim(ymin - 0.1 * yrange, ymax + 0.25 * yrange)  # Extra top space for annotation
                
                if key in qvalues:
                    p, q, d, sig = qvalues[key]
                    sig_marker = "†" if sig else ""
                    sig_color = get_significance_color(sig, config)
                    # Position annotation above the data
                    ax.annotate(f"q={q:.3f}{sig_marker}\nd={d:.2f}", 
                               xy=(0.5, ymax + 0.05 * yrange), 
                               ha="center", fontsize=plot_cfg.font.medium, color=sig_color,
                               fontweight="bold" if sig else "normal")
            
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Non-Pain", "Pain"], fontsize=plot_cfg.font.title)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            
            # Row labels (band)
            if col_idx == 0:
                ax.set_ylabel(band.capitalize(), fontsize=plot_cfg.font.figure_title, fontweight="bold",
                            color=band_colors.get(band, None))
            
            # Column labels (segment)
            if row_idx == 0:
                ax.set_title(segment.capitalize(), fontsize=plot_cfg.font.figure_title, fontweight="bold")
    
    n_pain = int(pain_mask.sum())
    n_nonpain = int((~pain_mask).sum())
    n_tests = len(all_pvalues)
    n_sig = sum(1 for k in qvalues if qvalues[k][3])
    
    title = (f"{feature_label}: Mean Across Channels by Band × Segment\n"
             f"Subject: {subject} | N: {n_nonpain} non-pain, {n_pain} pain | "
             f"Mann-Whitney U, FDR-corrected | {n_sig}/{n_tests} significant (†=q<0.05)")
    fig.suptitle(title, fontsize=plot_cfg.font.figure_title, fontweight="bold", y=1.02)
    
    plt.tight_layout()
    safe_name = feature_prefix.lower().replace(" ", "_")
    save_fig(fig, save_dir / f"sub-{subject}_{safe_name}_band_segment_condition", 
             formats=plot_cfg.formats, dpi=plot_cfg.dpi,
             bbox_inches=plot_cfg.bbox_inches, pad_inches=plot_cfg.pad_inches)
    plt.close(fig)
    
    if logger:
        logger.info(f"Saved {feature_label} band × segment × condition plot ({n_sig}/{n_tests} FDR significant)")


###################################################################
# POWER PLATEAU VS BASELINE (PAIRED WILCOXON)
###################################################################

def plot_power_plateau_vs_baseline(
    features_df: pd.DataFrame,
    subject: str,
    save_dir: Path,
    logger: logging.Logger,
    config: Any,
) -> None:
    """Power Plateau vs Baseline Paired Comparison.
    
    Uses paired Wilcoxon signed-rank test for within-trial comparison.
    Shows whether power changes from baseline to plateau period.
    """
    if features_df is None or features_df.empty:
        return
    
    from eeg_pipeline.utils.config.loader import get_config_value
    from statsmodels.stats.multitest import multipletests
    from scipy.stats import wilcoxon
    
    plateau_window = get_config_value(config, "plateau_window", [3.0, 10.5])
    baseline_window = get_config_value(config, "baseline_window", [-3.0, -0.5])
    
    bands, band_colors, condition_colors = _get_bands_and_palettes(config)
    n_bands = len(bands)
    
    plot_data = {}
    all_pvalues = []
    pvalue_keys = []
    
    for band_idx, band in enumerate(bands):
        baseline_cols = [c for c in features_df.columns 
                        if f"power_baseline_{band}_ch_" in c]
        plateau_cols = [c for c in features_df.columns 
                       if f"power_plateau_{band}_ch_" in c]
        
        vals_baseline = np.array([])
        vals_plateau = np.array([])
        
        if baseline_cols and plateau_cols:
            vals_baseline = features_df[baseline_cols].mean(axis=1).dropna().values
            vals_plateau = features_df[plateau_cols].mean(axis=1).dropna().values
        
        plot_data[band_idx] = (vals_baseline, vals_plateau)
        
        if len(vals_baseline) > 5 and len(vals_plateau) > 5 and len(vals_baseline) == len(vals_plateau):
            try:
                _, p = wilcoxon(vals_plateau, vals_baseline)
                diff = vals_plateau - vals_baseline
                pooled_std = np.std(diff, ddof=1)
                d = np.mean(diff) / pooled_std if pooled_std > 0 else 0
                all_pvalues.append(p)
                pvalue_keys.append((band_idx, p, d))
            except Exception:
                pass
    
    qvalues = {}
    if all_pvalues:
        rejected, qvals, _, _ = multipletests(all_pvalues, method='fdr_bh')
        for i, (key, p, d) in enumerate(pvalue_keys):
            qvalues[key] = (p, qvals[i], d, rejected[i])
    
    plot_cfg = get_plot_config(config)
    fig, axes = plt.subplots(1, n_bands, figsize=(3 * n_bands, 5))
    
    for band_idx, band in enumerate(bands):
        ax = axes[band_idx]
        vals_baseline, vals_plateau = plot_data[band_idx]
        
        if len(vals_baseline) == 0 or len(vals_plateau) == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", 
                   transform=ax.transAxes, fontsize=plot_cfg.font.title, color="gray")
            ax.set_xticks([])
            continue
        
        bp = ax.boxplot([vals_baseline, vals_plateau], 
                       positions=[0, 1], widths=0.4, patch_artist=True)
        bp["boxes"][0].set_facecolor(condition_colors["nonpain"])
        bp["boxes"][0].set_alpha(0.6)
        bp["boxes"][1].set_facecolor(condition_colors["pain"])
        bp["boxes"][1].set_alpha(0.6)
        
        ax.scatter(np.random.uniform(-0.08, 0.08, len(vals_baseline)), 
                  vals_baseline, c=condition_colors["nonpain"], alpha=0.3, s=6)
        ax.scatter(1 + np.random.uniform(-0.08, 0.08, len(vals_plateau)), 
                  vals_plateau, c=condition_colors["pain"], alpha=0.3, s=6)
        
        if len(vals_baseline) == len(vals_plateau) and len(vals_baseline) <= 100:
            for i in range(len(vals_baseline)):
                ax.plot([0, 1], [vals_baseline[i], vals_plateau[i]], 
                       c="gray", alpha=0.15, lw=0.5)
        
        all_vals = np.concatenate([vals_baseline, vals_plateau])
        ymin, ymax = np.nanmin(all_vals), np.nanmax(all_vals)
        yrange = ymax - ymin if ymax > ymin else 0.1
        ax.set_ylim(ymin - 0.1 * yrange, ymax + 0.3 * yrange)
        
        if band_idx in qvalues:
            p, q, d, sig = qvalues[band_idx]
            sig_marker = "†" if sig else ""
            sig_color = get_significance_color(sig, config)
            ax.annotate(f"q={q:.3f}{sig_marker}\nd={d:.2f}", 
                       xy=(0.5, ymax + 0.05 * yrange), 
                       ha="center", fontsize=plot_cfg.font.medium, color=sig_color,
                       fontweight="bold" if sig else "normal")
        
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Baseline", "Plateau"], fontsize=9)
        ax.set_title(band.capitalize(), fontweight="bold", color=band_colors.get(band, None))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    
    n_tests = len(all_pvalues)
    n_sig = sum(1 for k in qvalues if qvalues[k][3])
    
    title = (f"Band Power: Baseline vs Plateau (Paired Comparison)\n"
             f"Subject: {subject} | Wilcoxon signed-rank | "
             f"FDR: {n_sig}/{n_tests} significant (†=q<0.05)")
    fig.suptitle(title, fontsize=plot_cfg.font.suptitle, fontweight="bold", y=1.02)
    
    plt.tight_layout()
    save_fig(fig, save_dir / f"sub-{subject}_power_plateau_vs_baseline", 
             formats=plot_cfg.formats, dpi=plot_cfg.dpi,
             bbox_inches=plot_cfg.bbox_inches, pad_inches=plot_cfg.pad_inches)
    plt.close(fig)
    
    if logger:
        logger.info(f"Saved Power plateau vs baseline plot ({n_sig}/{n_tests} FDR significant)")


###################################################################
# TEMPORAL EVOLUTION PLOT
###################################################################

def plot_temporal_evolution(
    features_df: pd.DataFrame,
    events_df: pd.DataFrame,
    subject: str,
    save_dir: Path,
    logger: logging.Logger,
    config: Any,
    feature_prefix: str = "power",
    feature_label: str = "Band Power",
) -> None:
    """Temporal Evolution: Early vs Mid vs Late Plateau.
    
    Shows how features evolve across the trial using time bins.
    Compares pain vs non-pain at each temporal phase.
    """
    if features_df is None or features_df.empty or events_df is None:
        return

    pain_mask = extract_pain_mask(events_df, config)
    if pain_mask is None:
        return
    
    from eeg_pipeline.utils.config.loader import get_config_value
    
    temporal_cfg = get_config_value(config, "plotting.plots.features.temporal", {})
    time_bins = temporal_cfg.get("time_bins", ["coarse_early", "coarse_mid", "coarse_late"])
    time_labels = temporal_cfg.get("time_labels", ["Early", "Mid", "Late"])
    
    bands, band_colors, condition_colors = _get_bands_and_palettes(config)
    
    plot_cfg = get_plot_config(config)
    fig, axes = plt.subplots(len(bands), 1, figsize=(8, 3 * len(bands)), sharex=True)
    
    for band_idx, band in enumerate(bands):
        ax = axes[band_idx]
        
        pain_means, nonpain_means = [], []
        pain_sems, nonpain_sems = [], []
        
        for time_bin in time_bins:
            cols = [c for c in features_df.columns 
                   if f"{feature_prefix}_{time_bin}_{band}_ch_" in c]
            
            if cols:
                mean_vals = features_df[cols].mean(axis=1)
                pain_vals = mean_vals[pain_mask].dropna().values
                nonpain_vals = mean_vals[~pain_mask].dropna().values
                
                pain_means.append(np.mean(pain_vals) if len(pain_vals) > 0 else np.nan)
                nonpain_means.append(np.mean(nonpain_vals) if len(nonpain_vals) > 0 else np.nan)
                pain_sems.append(stats.sem(pain_vals) if len(pain_vals) > 1 else 0)
                nonpain_sems.append(stats.sem(nonpain_vals) if len(nonpain_vals) > 1 else 0)
            else:
                pain_means.append(np.nan)
                nonpain_means.append(np.nan)
                pain_sems.append(0)
                nonpain_sems.append(0)
        
        x = np.arange(len(time_bins))
        
        ax.errorbar(x - 0.1, nonpain_means, yerr=nonpain_sems, 
                   fmt='o-', color=condition_colors["nonpain"], 
                   label="Non-Pain", capsize=3, markersize=8)
        ax.errorbar(x + 0.1, pain_means, yerr=pain_sems, 
                   fmt='o-', color=condition_colors["pain"], 
                   label="Pain", capsize=3, markersize=8)
        
        ax.set_xticks(x)
        ax.set_xticklabels(time_labels)
        ax.set_ylabel(band.capitalize(), fontweight="bold", color=band_colors.get(band, None))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc="upper right", fontsize=plot_cfg.font.medium)
    
    axes[0].set_title(f"{feature_label} Temporal Evolution", fontsize=plot_cfg.font.figure_title, fontweight="bold")
    axes[-1].set_xlabel("Trial Phase", fontsize=plot_cfg.font.suptitle)
    
    title = f"{feature_label} Temporal Evolution: Pain vs Non-Pain\nSubject: {subject}"
    fig.suptitle(title, fontsize=plot_cfg.font.figure_title, fontweight="bold", y=1.01)
    
    plt.tight_layout()
    save_fig(fig, save_dir / f"sub-{subject}_{feature_prefix}_temporal_evolution", 
             formats=plot_cfg.formats, dpi=plot_cfg.dpi,
             bbox_inches=plot_cfg.bbox_inches, pad_inches=plot_cfg.pad_inches)
    plt.close(fig)
    
    if logger:
        logger.info(f"Saved {feature_label} temporal evolution plot")


###################################################################
# FEATURE CORRELATION HEATMAP
###################################################################

def plot_feature_correlation_heatmap(
    features_df: pd.DataFrame,
    subject: str,
    save_dir: Path,
    logger: logging.Logger,
    config: Any,
    max_features: int = 50,
) -> None:
    """Feature Correlation Heatmap.
    
    Shows inter-feature correlations to identify redundancy.
    Clusters features by correlation for better visualization.
    """
    if features_df is None or features_df.empty:
        return
    
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import squareform
    from eeg_pipeline.utils.config.loader import get_config_value
    
    skip_cols = ["condition", "trial", "epoch", "subject", "vas", "rating", "temp"]
    numeric_cols = [c for c in features_df.select_dtypes(include=[np.number]).columns
                   if not any(skip in c.lower() for skip in skip_cols)]
    
    if len(numeric_cols) < 3:
        if logger:
            logger.info("Too few numeric features for correlation heatmap")
        return
    
    max_features = int(get_config_value(config, "plotting.plots.features.correlation.max_features", max_features))
    if len(numeric_cols) > max_features:
        variance = features_df[numeric_cols].var()
        top_cols = variance.nlargest(max_features).index.tolist()
        numeric_cols = top_cols
    
    df_subset = features_df[numeric_cols].dropna(axis=1, how='all')
    
    if df_subset.shape[1] < 3:
        return
    
    corr_matrix = df_subset.corr()
    
    dissimilarity = 1 - np.abs(corr_matrix.values)
    np.fill_diagonal(dissimilarity, 0)
    dissimilarity = np.clip(dissimilarity, 0, None)
    dissimilarity = (dissimilarity + dissimilarity.T) / 2
    
    try:
        condensed = squareform(dissimilarity)
        linkage_matrix = linkage(condensed, method='average')
        order = leaves_list(linkage_matrix)
    except Exception:
        order = np.arange(len(numeric_cols))
    
    corr_ordered = corr_matrix.iloc[order, order]
    
    plot_cfg = get_plot_config(config)
    fig_size = max(10, min(20, len(numeric_cols) * 0.3))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    
    im = ax.imshow(corr_ordered.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    
    ax.set_xticks(np.arange(len(corr_ordered.columns)))
    ax.set_yticks(np.arange(len(corr_ordered.index)))
    
    short_labels = [c.split('_')[-2] + "_" + c.split('_')[-1] 
                   if len(c.split('_')) > 2 else c[-15:] 
                   for c in corr_ordered.columns]
    ax.set_xticklabels(short_labels, rotation=90, fontsize=6)
    ax.set_yticklabels(short_labels, fontsize=6)
    
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Pearson Correlation", fontsize=plot_cfg.font.title)
    
    title = f"Feature Correlation Heatmap\nSubject: {subject} | {len(numeric_cols)} features (clustered)"
    ax.set_title(title, fontsize=plot_cfg.font.figure_title, fontweight="bold", pad=10)
    
    plt.tight_layout()
    save_fig(fig, save_dir / f"sub-{subject}_feature_correlation_heatmap", 
             formats=plot_cfg.formats, dpi=plot_cfg.dpi,
             bbox_inches=plot_cfg.bbox_inches, pad_inches=plot_cfg.pad_inches)
    plt.close(fig)
    
    if logger:
        logger.info(f"Saved feature correlation heatmap ({len(numeric_cols)} features)")
