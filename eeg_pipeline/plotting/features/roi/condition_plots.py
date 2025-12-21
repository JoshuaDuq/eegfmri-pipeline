from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from eeg_pipeline.domain.features.naming import NamingSchema
from eeg_pipeline.utils.analysis.events import extract_pain_mask
from eeg_pipeline.plotting.config import get_plot_config
from eeg_pipeline.plotting.io.figures import save_fig
from eeg_pipeline.plotting.features.utils import (
    apply_fdr_correction,
    get_band_colors,
    get_band_names,
    get_condition_colors,
    get_significance_color,
)

from .core import (
    _get_bands_and_palettes,
    aggregate_by_roi,
    aggregate_connectivity_by_roi,
    extract_channel_pairs_from_columns,
    extract_channels_from_columns,
    get_roi_channels,
    get_roi_definitions,
)


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

    plot_data: Dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    all_pvalues = []
    pvalue_keys = []

    for row_idx, roi_name in enumerate(roi_names):
        roi_patterns = rois[roi_name]
        roi_channels = get_roi_channels(roi_patterns, all_channels)

        for col_idx, band in enumerate(bands):
            key = (row_idx, col_idx)

            cols = []
            roi_set = set(roi_channels)
            for c in features_df.columns:
                parsed = NamingSchema.parse(str(c))
                if not parsed.get("valid"):
                    continue
                if parsed.get("group") != "power":
                    continue
                if parsed.get("segment") != "plateau":
                    continue
                if parsed.get("band") != band:
                    continue
                if parsed.get("scope") != "ch":
                    continue
                if parsed.get("identifier") not in roi_set:
                    continue
                cols.append(str(c))
            roi_vals = (
                features_df[cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
                if cols
                else pd.Series([np.nan] * len(features_df), index=features_df.index)
            )

            vals_nonpain = roi_vals[~pain_mask].dropna().values
            vals_pain = roi_vals[pain_mask].dropna().values

            plot_data[key] = (vals_nonpain, vals_pain)

            if len(vals_nonpain) > 3 and len(vals_pain) > 3:
                _, p = stats.mannwhitneyu(vals_nonpain, vals_pain)
                pooled_std = np.sqrt(
                    (
                        (len(vals_nonpain) - 1) * np.var(vals_nonpain, ddof=1)
                        + (len(vals_pain) - 1) * np.var(vals_pain, ddof=1)
                    )
                    / (len(vals_nonpain) + len(vals_pain) - 2)
                )
                d = (
                    (np.mean(vals_pain) - np.mean(vals_nonpain)) / pooled_std
                    if pooled_std > 0
                    else 0
                )
                all_pvalues.append(p)
                pvalue_keys.append((key, p, d))

    qvalues: Dict[tuple[int, int], tuple[float, float, float, bool]] = {}
    if all_pvalues:
        rejected, qvals, _ = apply_fdr_correction(all_pvalues, config=config)
        for i, (key, p, d) in enumerate(pvalue_keys):
            qvalues[key] = (p, qvals[i], d, rejected[i])

    width_per_col = float(plot_cfg.plot_type_configs.get("roi", {}).get("width_per_band", 3.2))
    height_per_row = float(plot_cfg.plot_type_configs.get("roi", {}).get("height_per_roi", 2.5))
    fig, axes = plt.subplots(n_rois, n_bands, figsize=(width_per_col * n_bands, height_per_row * n_rois))

    for row_idx, roi_name in enumerate(roi_names):
        for col_idx, band in enumerate(bands):
            ax = axes[row_idx, col_idx]
            key = (row_idx, col_idx)
            vals_nonpain, vals_pain = plot_data[key]

            if len(vals_nonpain) == 0 and len(vals_pain) == 0:
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=plot_cfg.font.medium,
                    color="gray",
                )
                ax.set_xticks([])
                continue

            if len(vals_nonpain) > 0 and len(vals_pain) > 0:
                bp = ax.boxplot(
                    [vals_nonpain, vals_pain],
                    positions=[0, 1],
                    widths=0.4,
                    patch_artist=True,
                )
                bp["boxes"][0].set_facecolor(condition_colors["nonpain"])
                bp["boxes"][0].set_alpha(0.6)
                bp["boxes"][1].set_facecolor(condition_colors["pain"])
                bp["boxes"][1].set_alpha(0.6)

                ax.scatter(
                    np.random.uniform(-0.08, 0.08, len(vals_nonpain)),
                    vals_nonpain,
                    c=condition_colors["nonpain"],
                    alpha=0.3,
                    s=6,
                )
                ax.scatter(
                    1 + np.random.uniform(-0.08, 0.08, len(vals_pain)),
                    vals_pain,
                    c=condition_colors["pain"],
                    alpha=0.3,
                    s=6,
                )

                if key in qvalues:
                    _, q, d, sig = qvalues[key]
                    sig_marker = "†" if sig else ""
                    ax.text(
                        0.5,
                        0.95,
                        f"q={q:.3f}{sig_marker}\nd={d:.2f}",
                        transform=ax.transAxes,
                        ha="center",
                        fontsize=plot_cfg.font.annotation,
                        va="top",
                    )

            ax.set_xticks([0, 1])
            ax.set_xticklabels(["NP", "P"], fontsize=plot_cfg.font.small)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            if row_idx == 0:
                ax.set_title(
                    band.capitalize(),
                    fontweight="bold",
                    color=band_colors.get(band, None),
                    fontsize=plot_cfg.font.title,
                )
            if col_idx == 0:
                short_name = roi_name.replace("_", "\n").replace("Contra", "C").replace(
                    "Ipsi", "I"
                )
                ax.set_ylabel(short_name, fontsize=plot_cfg.font.medium)

    n_pain = int(pain_mask.sum())
    n_nonpain = int((~pain_mask).sum())
    n_tests = len(all_pvalues)
    n_sig = sum(1 for k in qvalues if qvalues[k][3])

    title = (
        "Band Power by ROI: Pain vs Non-Pain Condition Comparison\n"
        f"Plateau Phase ({plateau_label}), Baseline-Normalized dB\n"
        f"Subject: {subject} | N: {n_nonpain} non-pain, {n_pain} pain | "
        f"FDR: {n_sig}/{n_tests} significant (†=q<0.05)"
    )
    fig.suptitle(title, fontsize=plot_cfg.font.suptitle, fontweight="bold", y=1.02)

    plt.tight_layout()
    save_fig(
        fig,
        save_dir / f"sub-{subject}_power_roi_band_condition",
        formats=plot_cfg.formats,
        dpi=plot_cfg.dpi,
        bbox_inches=plot_cfg.bbox_inches,
        pad_inches=plot_cfg.pad_inches,
    )
    plt.close(fig)

    if logger:
        logger.info(
            f"Saved power ROI × band × condition plot ({n_sig}/{n_tests} FDR significant)"
        )


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

    plateau_window = get_config_value(config, "plateau_window", [3.0, 10.5])
    plateau_label = f"{plateau_window[0]:.1f}-{plateau_window[1]:.1f}s"

    rois = get_roi_definitions(config)
    if not rois:
        return

    all_channels = extract_channels_from_columns(list(features_df.columns))

    bands, band_colors, condition_colors = _get_bands_and_palettes(config)
    roi_names = list(rois.keys())
    n_rois = len(roi_names)
    n_bands = len(bands)

    metric_labels = {
        "lzc": "Lempel-Ziv Complexity",
        "pe": "Permutation Entropy",
        "hjorth_mobility": "Hjorth Mobility",
        "hjorth_complexity": "Hjorth Complexity",
    }
    metric_label = metric_labels.get(metric, metric.upper())

    plot_data: Dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    all_pvalues = []
    pvalue_keys = []

    for row_idx, roi_name in enumerate(roi_names):
        roi_patterns = rois[roi_name]
        roi_channels = get_roi_channels(roi_patterns, all_channels)

        for col_idx, band in enumerate(bands):
            key = (row_idx, col_idx)

            metric_cols = [
                c
                for c in features_df.columns
                if f"dynamics_plateau_{band}_" in c and f"_{metric}" in c
            ]
            if metric_cols:
                roi_metric_cols = [
                    c
                    for c in metric_cols
                    if any(
                        f"_ch_{ch}_" in c or c.endswith(f"_ch_{ch}") for ch in roi_channels
                    )
                ]
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
                pooled_std = np.sqrt(
                    (
                        (len(vals_nonpain) - 1) * np.var(vals_nonpain, ddof=1)
                        + (len(vals_pain) - 1) * np.var(vals_pain, ddof=1)
                    )
                    / (len(vals_nonpain) + len(vals_pain) - 2)
                )
                d = (
                    (np.mean(vals_pain) - np.mean(vals_nonpain)) / pooled_std
                    if pooled_std > 0
                    else 0
                )
                all_pvalues.append(p)
                pvalue_keys.append((key, p, d))

    qvalues: Dict[tuple[int, int], tuple[float, float, float, bool]] = {}
    if all_pvalues:
        rejected, qvals, _ = apply_fdr_correction(all_pvalues, config=config)
        for i, (key, p, d) in enumerate(pvalue_keys):
            qvalues[key] = (p, qvals[i], d, rejected[i])

    plot_cfg = get_plot_config(config)
    width_per_col = float(plot_cfg.plot_type_configs.get("roi", {}).get("width_per_band", 3.2))
    height_per_row = float(plot_cfg.plot_type_configs.get("roi", {}).get("height_per_roi", 2.5))
    fig, axes = plt.subplots(n_rois, n_bands, figsize=(width_per_col * n_bands, height_per_row * n_rois))

    for row_idx, roi_name in enumerate(roi_names):
        for col_idx, band in enumerate(bands):
            ax = axes[row_idx, col_idx]
            key = (row_idx, col_idx)
            vals_nonpain, vals_pain = plot_data[key]

            if len(vals_nonpain) == 0 and len(vals_pain) == 0:
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=plot_cfg.font.medium,
                    color="gray",
                )
                ax.set_xticks([])
                continue

            if len(vals_nonpain) > 0 and len(vals_pain) > 0:
                bp = ax.boxplot(
                    [vals_nonpain, vals_pain],
                    positions=[0, 1],
                    widths=0.4,
                    patch_artist=True,
                )
                bp["boxes"][0].set_facecolor(condition_colors["nonpain"])
                bp["boxes"][0].set_alpha(0.6)
                bp["boxes"][1].set_facecolor(condition_colors["pain"])
                bp["boxes"][1].set_alpha(0.6)

                ax.scatter(
                    np.random.uniform(-0.08, 0.08, len(vals_nonpain)),
                    vals_nonpain,
                    c=condition_colors["nonpain"],
                    alpha=0.3,
                    s=6,
                )
                ax.scatter(
                    1 + np.random.uniform(-0.08, 0.08, len(vals_pain)),
                    vals_pain,
                    c=condition_colors["pain"],
                    alpha=0.3,
                    s=6,
                )

                if key in qvalues:
                    _, q, d, sig = qvalues[key]
                    sig_marker = "†" if sig else ""
                    ax.text(
                        0.5,
                        0.95,
                        f"q={q:.3f}{sig_marker}\nd={d:.2f}",
                        transform=ax.transAxes,
                        ha="center",
                        fontsize=plot_cfg.font.annotation,
                        va="top",
                    )

            ax.set_xticks([0, 1])
            ax.set_xticklabels(["NP", "P"], fontsize=plot_cfg.font.small)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            if row_idx == 0:
                ax.set_title(
                    band.capitalize(),
                    fontweight="bold",
                    color=band_colors.get(band, None),
                    fontsize=plot_cfg.font.title,
                )
            if col_idx == 0:
                short_name = roi_name.replace("_", "\n").replace("Contra", "C").replace(
                    "Ipsi", "I"
                )
                ax.set_ylabel(short_name, fontsize=plot_cfg.font.medium)

    n_pain = int(pain_mask.sum())
    n_nonpain = int((~pain_mask).sum())
    n_tests = len(all_pvalues)
    n_sig = sum(1 for k in qvalues if qvalues[k][3])

    title = (
        f"{metric_label} by ROI: Pain vs Non-Pain Condition Comparison\n"
        f"Plateau Phase ({plateau_label})\n"
        f"Subject: {subject} | N: {n_nonpain} non-pain, {n_pain} pain | "
        f"FDR: {n_sig}/{n_tests} significant (†=q<0.05)"
    )
    fig.suptitle(title, fontsize=plot_cfg.font.suptitle, fontweight="bold", y=1.02)

    plt.tight_layout()
    save_fig(
        fig,
        save_dir / f"sub-{subject}_{metric}_roi_band_condition",
        formats=plot_cfg.formats,
        dpi=plot_cfg.dpi,
        bbox_inches=plot_cfg.bbox_inches,
        pad_inches=plot_cfg.pad_inches,
    )
    plt.close(fig)

    if logger:
        logger.info(
            f"Saved {metric} ROI × band × condition plot ({n_sig}/{n_tests} FDR significant)"
        )


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

    plot_data: Dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
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
                pooled_std = np.sqrt(
                    (
                        (len(vals_nonpain) - 1) * np.var(vals_nonpain, ddof=1)
                        + (len(vals_pain) - 1) * np.var(vals_pain, ddof=1)
                    )
                    / (len(vals_nonpain) + len(vals_pain) - 2)
                )
                d = (
                    (np.mean(vals_pain) - np.mean(vals_nonpain)) / pooled_std
                    if pooled_std > 0
                    else 0
                )
                all_pvalues.append(p)
                pvalue_keys.append((key, p, d))

    qvalues: Dict[tuple[int, int], tuple[float, float, float, bool]] = {}
    if all_pvalues:
        rejected, qvals, _ = apply_fdr_correction(all_pvalues, config=config)
        for i, (key, p, d) in enumerate(pvalue_keys):
            qvalues[key] = (p, qvals[i], d, rejected[i])

    plot_cfg = get_plot_config(config)
    width_per_col = float(plot_cfg.plot_type_configs.get("roi", {}).get("width_per_metric", 4.0))
    height_per_row = float(plot_cfg.plot_type_configs.get("roi", {}).get("height_per_roi", 2.5))
    fig, axes = plt.subplots(n_rois, n_metrics, figsize=(width_per_col * n_metrics, height_per_row * n_rois))

    for row_idx, roi_name in enumerate(roi_names):
        for col_idx, metric in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            key = (row_idx, col_idx)
            vals_nonpain, vals_pain = plot_data[key]

            if len(vals_nonpain) == 0 and len(vals_pain) == 0:
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=plot_cfg.font.medium,
                    color="gray",
                )
                ax.set_xticks([])
                continue

            if len(vals_nonpain) > 0 and len(vals_pain) > 0:
                bp = ax.boxplot(
                    [vals_nonpain, vals_pain],
                    positions=[0, 1],
                    widths=0.4,
                    patch_artist=True,
                )
                bp["boxes"][0].set_facecolor(condition_colors["nonpain"])
                bp["boxes"][0].set_alpha(0.6)
                bp["boxes"][1].set_facecolor(condition_colors["pain"])
                bp["boxes"][1].set_alpha(0.6)

                ax.scatter(
                    np.random.uniform(-0.08, 0.08, len(vals_nonpain)),
                    vals_nonpain,
                    c=condition_colors["nonpain"],
                    alpha=0.3,
                    s=6,
                )
                ax.scatter(
                    1 + np.random.uniform(-0.08, 0.08, len(vals_pain)),
                    vals_pain,
                    c=condition_colors["pain"],
                    alpha=0.3,
                    s=6,
                )

                if key in qvalues:
                    _, q, d, sig = qvalues[key]
                    sig_marker = "†" if sig else ""
                    ax.text(
                        0.5,
                        0.95,
                        f"q={q:.3f}{sig_marker}\nd={d:.2f}",
                        transform=ax.transAxes,
                        ha="center",
                        fontsize=plot_cfg.font.annotation,
                        va="top",
                    )

            ax.set_xticks([0, 1])
            ax.set_xticklabels(["NP", "P"], fontsize=plot_cfg.font.small)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            if row_idx == 0:
                ax.set_title(metric.capitalize(), fontweight="bold", fontsize=plot_cfg.font.title)
            if col_idx == 0:
                short_name = roi_name.replace("_", "\n").replace("Contra", "C").replace(
                    "Ipsi", "I"
                )
                ax.set_ylabel(short_name, fontsize=plot_cfg.font.medium)

    n_pain = int(pain_mask.sum())
    n_nonpain = int((~pain_mask).sum())
    n_tests = len(all_pvalues)
    n_sig = sum(1 for k in qvalues if qvalues[k][3])

    title = (
        "Aperiodic 1/f Parameters by ROI: Pain vs Non-Pain Comparison\n"
        f"Plateau Phase ({plateau_label})\n"
        f"Subject: {subject} | N: {n_nonpain} non-pain, {n_pain} pain | "
        f"FDR: {n_sig}/{n_tests} significant (†=q<0.05)"
    )
    fig.suptitle(title, fontsize=plot_cfg.font.suptitle, fontweight="bold", y=1.02)

    plt.tight_layout()
    save_fig(
        fig,
        save_dir / f"sub-{subject}_aperiodic_roi_condition",
        formats=plot_cfg.formats,
        dpi=plot_cfg.dpi,
        bbox_inches=plot_cfg.bbox_inches,
        pad_inches=plot_cfg.pad_inches,
    )
    plt.close(fig)

    if logger:
        logger.info(
            f"Saved aperiodic ROI × condition plot ({n_sig}/{n_tests} FDR significant)"
        )


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
    width_per_col = float(plot_cfg.plot_type_configs.get("roi", {}).get("width_per_band", 3.2))
    height_per_row = float(plot_cfg.plot_type_configs.get("roi", {}).get("height_per_roi", 2.5))
    fig, axes = plt.subplots(n_rois, n_bands, figsize=(width_per_col * n_bands, height_per_row * n_rois))

    for row_idx, roi_name in enumerate(roi_names):
        roi_patterns = rois[roi_name]
        roi_channels = get_roi_channels(roi_patterns, all_channels)

        for col_idx, band in enumerate(bands):
            ax = axes[row_idx, col_idx]

            roi_vals = aggregate_connectivity_by_roi(features_df, f"conn_plateau_{band}_", roi_channels)

            if measure != "wpli":
                measure_cols = [
                    c
                    for c in features_df.columns
                    if f"conn_plateau_{band}_" in c and f"_{measure}" in c
                ]
                if measure_cols:
                    roi_measure_cols = []
                    import re

                    for c in measure_cols:
                        match = re.search(r"_chpair_([A-Za-z0-9]+)-([A-Za-z0-9]+)_", c)
                        if match and match.group(1) in roi_channels and match.group(2) in roi_channels:
                            roi_measure_cols.append(c)
                    if roi_measure_cols:
                        roi_vals = features_df[roi_measure_cols].mean(axis=1)

            if roi_vals.isna().all():
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=plot_cfg.font.medium,
                    color="gray",
                )
                ax.set_xticks([])
                continue

            vals_nonpain = roi_vals[~pain_mask].dropna().values
            vals_pain = roi_vals[pain_mask].dropna().values

            if len(vals_nonpain) > 0 and len(vals_pain) > 0:
                bp = ax.boxplot(
                    [vals_nonpain, vals_pain],
                    positions=[0, 1],
                    widths=0.4,
                    patch_artist=True,
                )
                bp["boxes"][0].set_facecolor(condition_colors["nonpain"])
                bp["boxes"][0].set_alpha(0.6)
                bp["boxes"][1].set_facecolor(condition_colors["pain"])
                bp["boxes"][1].set_alpha(0.6)

                ax.scatter(
                    np.random.uniform(-0.08, 0.08, len(vals_nonpain)),
                    vals_nonpain,
                    c=condition_colors["nonpain"],
                    alpha=0.3,
                    s=6,
                )
                ax.scatter(
                    1 + np.random.uniform(-0.08, 0.08, len(vals_pain)),
                    vals_pain,
                    c=condition_colors["pain"],
                    alpha=0.3,
                    s=6,
                )

                if len(vals_nonpain) > 3 and len(vals_pain) > 3:
                    _, p = stats.mannwhitneyu(vals_nonpain, vals_pain)
                    pooled_std = np.sqrt(
                        (
                            (len(vals_nonpain) - 1) * np.var(vals_nonpain, ddof=1)
                            + (len(vals_pain) - 1) * np.var(vals_pain, ddof=1)
                        )
                        / (len(vals_nonpain) + len(vals_pain) - 2)
                    )
                    d = (
                        (np.mean(vals_pain) - np.mean(vals_nonpain)) / pooled_std
                        if pooled_std > 0
                        else 0
                    )
                    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                    ax.text(
                        0.5,
                        0.95,
                        f"p={p:.3f}{sig}\nd={d:.2f}",
                        transform=ax.transAxes,
                        ha="center",
                        fontsize=plot_cfg.font.annotation,
                        va="top",
                    )

            ax.set_xticks([0, 1])
            ax.set_xticklabels(["NP", "P"], fontsize=plot_cfg.font.small)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            if row_idx == 0:
                ax.set_title(
                    band.capitalize(),
                    fontweight="bold",
                    color=band_colors.get(band, None),
                    fontsize=plot_cfg.font.title,
                )
            if col_idx == 0:
                short_name = roi_name.replace("_", "\n").replace("Contra", "C").replace(
                    "Ipsi", "I"
                )
                ax.set_ylabel(short_name, fontsize=plot_cfg.font.medium)

    n_pain = int(pain_mask.sum())
    n_nonpain = int((~pain_mask).sum())
    fig.suptitle(
        f"Within-ROI {measure_label} by Band: Plateau ({plateau_label}) (sub-{subject})\nN: {n_nonpain} NP, {n_pain} P",
        fontsize=plot_cfg.font.figure_title,
        fontweight="bold",
        y=1.01,
    )

    plt.tight_layout()
    save_fig(
        fig,
        save_dir / f"sub-{subject}_conn_{measure}_roi_band_condition",
        formats=plot_cfg.formats,
        dpi=plot_cfg.dpi,
        bbox_inches=plot_cfg.bbox_inches,
        pad_inches=plot_cfg.pad_inches,
    )
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

    plot_data: Dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    all_pvalues = []
    pvalue_keys = []

    for row_idx, roi_name in enumerate(roi_names):
        roi_patterns = rois[roi_name]
        roi_channels = get_roi_channels(roi_patterns, all_channels)

        for col_idx, band in enumerate(bands):
            key = (row_idx, col_idx)

            itpc_cols = [c for c in features_df.columns if f"itpc_{segment}_{band}_ch_" in c]
            if itpc_cols:
                roi_itpc_cols = [
                    c
                    for c in itpc_cols
                    if any(f"_ch_{ch}_" in c or c.endswith(f"_ch_{ch}") for ch in roi_channels)
                ]
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
                pooled_std = np.sqrt(
                    (
                        (len(vals_nonpain) - 1) * np.var(vals_nonpain, ddof=1)
                        + (len(vals_pain) - 1) * np.var(vals_pain, ddof=1)
                    )
                    / (len(vals_nonpain) + len(vals_pain) - 2)
                )
                d = (
                    (np.mean(vals_pain) - np.mean(vals_nonpain)) / pooled_std
                    if pooled_std > 0
                    else 0
                )
                all_pvalues.append(p)
                pvalue_keys.append((key, p, d))

    qvalues: Dict[tuple[int, int], tuple[float, float, float, bool]] = {}
    if all_pvalues:
        rejected, qvals, _ = apply_fdr_correction(all_pvalues, config=config)
        for i, (key, p, d) in enumerate(pvalue_keys):
            qvalues[key] = (p, qvals[i], d, rejected[i])

    plot_cfg = get_plot_config(config)
    width_per_col = float(plot_cfg.plot_type_configs.get("roi", {}).get("width_per_band", 3.2))
    height_per_row = float(plot_cfg.plot_type_configs.get("roi", {}).get("height_per_roi", 2.5))
    fig, axes = plt.subplots(n_rois, n_bands, figsize=(width_per_col * n_bands, height_per_row * n_rois))

    for row_idx, roi_name in enumerate(roi_names):
        for col_idx, band in enumerate(bands):
            ax = axes[row_idx, col_idx]
            key = (row_idx, col_idx)
            vals_nonpain, vals_pain = plot_data[key]

            if len(vals_nonpain) == 0 and len(vals_pain) == 0:
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=plot_cfg.font.medium,
                    color="gray",
                )
                ax.set_xticks([])
                continue

            if len(vals_nonpain) > 0 and len(vals_pain) > 0:
                bp = ax.boxplot(
                    [vals_nonpain, vals_pain],
                    positions=[0, 1],
                    widths=0.4,
                    patch_artist=True,
                )
                bp["boxes"][0].set_facecolor(condition_colors["nonpain"])
                bp["boxes"][0].set_alpha(0.6)
                bp["boxes"][1].set_facecolor(condition_colors["pain"])
                bp["boxes"][1].set_alpha(0.6)

                ax.scatter(
                    np.random.uniform(-0.08, 0.08, len(vals_nonpain)),
                    vals_nonpain,
                    c=condition_colors["nonpain"],
                    alpha=0.3,
                    s=6,
                )
                ax.scatter(
                    1 + np.random.uniform(-0.08, 0.08, len(vals_pain)),
                    vals_pain,
                    c=condition_colors["pain"],
                    alpha=0.3,
                    s=6,
                )

                all_vals = np.concatenate([vals_nonpain, vals_pain])
                ymin, ymax = np.nanmin(all_vals), np.nanmax(all_vals)
                yrange = ymax - ymin if ymax > ymin else 0.1
                ax.set_ylim(ymin - 0.1 * yrange, ymax + 0.25 * yrange)

                if key in qvalues:
                    _, q, d, sig = qvalues[key]
                    sig_marker = "†" if sig else ""
                    sig_color = "#d62728" if sig else "#333333"
                    ax.annotate(
                        f"q={q:.3f}{sig_marker}\nd={d:.2f}",
                        xy=(0.5, ymax + 0.05 * yrange),
                        ha="center",
                        fontsize=plot_cfg.font.annotation,
                        color=sig_color,
                        fontweight="bold" if sig else "normal",
                    )

            ax.set_xticks([0, 1])
            ax.set_xticklabels(["NP", "P"], fontsize=plot_cfg.font.small)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            if row_idx == 0:
                ax.set_title(
                    band.capitalize(),
                    fontweight="bold",
                    color=band_colors.get(band, None),
                    fontsize=plot_cfg.font.title,
                )
            if col_idx == 0:
                short_name = roi_name.replace("_", "\n").replace("Contra", "C").replace(
                    "Ipsi", "I"
                )
                ax.set_ylabel(short_name, fontsize=plot_cfg.font.medium)

    n_pain = int(pain_mask.sum())
    n_nonpain = int((~pain_mask).sum())
    n_tests = len(all_pvalues)
    n_sig = sum(1 for k in qvalues if qvalues[k][3])

    title = (
        "Inter-Trial Phase Coherence by ROI: Pain vs Non-Pain Comparison\n"
        f"Plateau Phase ({plateau_label}), LOO-ITPC\n"
        f"Subject: {subject} | N: {n_nonpain} non-pain, {n_pain} pain | "
        f"FDR: {n_sig}/{n_tests} significant (†=q<0.05)"
    )
    fig.suptitle(title, fontsize=plot_cfg.font.suptitle, fontweight="bold", y=1.02)

    plt.tight_layout()
    save_fig(
        fig,
        save_dir / f"sub-{subject}_itpc_roi_band_condition",
        formats=plot_cfg.formats,
        dpi=plot_cfg.dpi,
        bbox_inches=plot_cfg.bbox_inches,
        pad_inches=plot_cfg.pad_inches,
    )
    plt.close(fig)

    if logger:
        logger.info(
            f"Saved ITPC ROI × band × condition plot ({n_sig}/{n_tests} FDR significant)"
        )


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

    plot_data: Dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    all_pvalues = []
    pvalue_keys = []

    for row_idx, roi_name in enumerate(roi_names):
        roi_patterns = rois[roi_name]
        roi_channels = get_roi_channels(roi_patterns, all_channels)

        for col_idx, band in enumerate(bands):
            key = (row_idx, col_idx)

            baseline_cols = [
                c for c in features_df.columns if f"itpc_baseline_{band}_ch_" in c
            ]
            plateau_cols = [
                c for c in features_df.columns if f"itpc_plateau_{band}_ch_" in c
            ]

            vals_baseline = np.array([])
            vals_plateau = np.array([])

            if baseline_cols and plateau_cols:
                roi_baseline_cols = [
                    c for c in baseline_cols if any(f"_ch_{ch}_" in c for ch in roi_channels)
                ]
                roi_plateau_cols = [
                    c for c in plateau_cols if any(f"_ch_{ch}_" in c for ch in roi_channels)
                ]

                if roi_baseline_cols and roi_plateau_cols:
                    vals_baseline = (
                        features_df[roi_baseline_cols].mean(axis=1).dropna().values
                    )
                    vals_plateau = (
                        features_df[roi_plateau_cols].mean(axis=1).dropna().values
                    )

            plot_data[key] = (vals_baseline, vals_plateau)

            if (
                len(vals_baseline) > 5
                and len(vals_plateau) > 5
                and len(vals_baseline) == len(vals_plateau)
            ):
                try:
                    _, p = wilcoxon(vals_plateau, vals_baseline)
                    diff = vals_plateau - vals_baseline
                    pooled_std = np.std(diff, ddof=1)
                    d = np.mean(diff) / pooled_std if pooled_std > 0 else 0
                    all_pvalues.append(p)
                    pvalue_keys.append((key, p, d))
                except Exception:
                    pass

    qvalues: Dict[tuple[int, int], tuple[float, float, float, bool]] = {}
    if all_pvalues:
        rejected, qvals, _ = apply_fdr_correction(all_pvalues, config=config)
        for i, (key, p, d) in enumerate(pvalue_keys):
            qvalues[key] = (p, qvals[i], d, rejected[i])

    plot_cfg = get_plot_config(config)
    width_per_col = float(plot_cfg.plot_type_configs.get("roi", {}).get("width_per_band", 3.2))
    height_per_row = float(plot_cfg.plot_type_configs.get("roi", {}).get("height_per_roi", 2.5))
    fig, axes = plt.subplots(n_rois, n_bands, figsize=(width_per_col * n_bands, height_per_row * n_rois))

    for row_idx, roi_name in enumerate(roi_names):
        for col_idx, band in enumerate(bands):
            ax = axes[row_idx, col_idx]
            key = (row_idx, col_idx)
            vals_baseline, vals_plateau = plot_data[key]

            if len(vals_baseline) == 0 or len(vals_plateau) == 0:
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=plot_cfg.font.medium,
                    color="gray",
                )
                ax.set_xticks([])
                continue

            bp = ax.boxplot(
                [vals_baseline, vals_plateau],
                positions=[0, 1],
                widths=0.4,
                patch_artist=True,
            )
            bp["boxes"][0].set_facecolor("#1f77b4")
            bp["boxes"][0].set_alpha(0.6)
            bp["boxes"][1].set_facecolor("#d62728")
            bp["boxes"][1].set_alpha(0.6)

            ax.scatter(
                np.random.uniform(-0.08, 0.08, len(vals_baseline)),
                vals_baseline,
                c="#1f77b4",
                alpha=0.3,
                s=6,
            )
            ax.scatter(
                1 + np.random.uniform(-0.08, 0.08, len(vals_plateau)),
                vals_plateau,
                c="#d62728",
                alpha=0.3,
                s=6,
            )

            if len(vals_baseline) == len(vals_plateau) and len(vals_baseline) <= 100:
                for i in range(len(vals_baseline)):
                    ax.plot(
                        [0, 1],
                        [vals_baseline[i], vals_plateau[i]],
                        c="gray",
                        alpha=0.15,
                        lw=0.5,
                    )

            all_vals = np.concatenate([vals_baseline, vals_plateau])
            ymin, ymax = np.nanmin(all_vals), np.nanmax(all_vals)
            yrange = ymax - ymin if ymax > ymin else 0.1
            ax.set_ylim(ymin - 0.1 * yrange, ymax + 0.25 * yrange)

            if key in qvalues:
                _, q, d, sig = qvalues[key]
                sig_marker = "†" if sig else ""
                sig_color = "#d62728" if sig else "#333333"
                ax.annotate(
                    f"q={q:.3f}{sig_marker}\nd={d:.2f}",
                    xy=(0.5, ymax + 0.05 * yrange),
                    ha="center",
                    fontsize=plot_cfg.font.annotation,
                    color=sig_color,
                    fontweight="bold" if sig else "normal",
                )

            ax.set_xticks([0, 1])
            ax.set_xticklabels(["BL", "PL"], fontsize=plot_cfg.font.small)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            if row_idx == 0:
                ax.set_title(
                    band.capitalize(),
                    fontweight="bold",
                    color=band_colors.get(band, None),
                    fontsize=plot_cfg.font.title,
                )
            if col_idx == 0:
                short_name = roi_name.replace("_", "\n").replace("Contra", "C").replace(
                    "Ipsi", "I"
                )
                ax.set_ylabel(short_name, fontsize=plot_cfg.font.medium)

    n_tests = len(all_pvalues)
    n_sig = sum(1 for k in qvalues if qvalues[k][3])

    title = (
        "ITPC Plateau vs Baseline by ROI: Paired Comparison\n"
        f"Baseline ({baseline_window[0]:.1f}-{baseline_window[1]:.1f}s) vs "
        f"Plateau ({plateau_window[0]:.1f}-{plateau_window[1]:.1f}s)\n"
        f"Subject: {subject} | Wilcoxon signed-rank | "
        f"FDR: {n_sig}/{n_tests} significant (†=q<0.05)"
    )
    fig.suptitle(title, fontsize=plot_cfg.font.suptitle, fontweight="bold", y=1.02)

    plt.tight_layout()
    save_fig(
        fig,
        save_dir / f"sub-{subject}_itpc_plateau_vs_baseline",
        formats=plot_cfg.formats,
        dpi=plot_cfg.dpi,
        bbox_inches=plot_cfg.bbox_inches,
        pad_inches=plot_cfg.pad_inches,
    )
    plt.close(fig)

    if logger:
        logger.info(
            f"Saved ITPC plateau vs baseline plot ({n_sig}/{n_tests} FDR significant)"
        )


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

    rois = get_roi_definitions(config)
    if not rois:
        return

    all_channels = extract_channels_from_columns(list(features_df.columns))

    from eeg_pipeline.utils.config.loader import get_config_value

    pac_pairs = get_config_value(
        config,
        "plotting.plots.features.pac_pairs",
        ["theta_beta", "theta_gamma", "alpha_beta", "alpha_gamma"],
    )

    _, band_colors, condition_colors = _get_bands_and_palettes(config)
    roi_names = list(rois.keys())
    n_rois = len(roi_names)
    n_pairs = len(pac_pairs)

    plot_data: Dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    all_pvalues = []
    pvalue_keys = []

    for row_idx, roi_name in enumerate(roi_names):
        roi_patterns = rois[roi_name]
        roi_channels = get_roi_channels(roi_patterns, all_channels)

        for col_idx, pair in enumerate(pac_pairs):
            key = (row_idx, col_idx)

            pac_cols = [c for c in features_df.columns if f"pac_plateau_{pair}_ch_" in c]

            vals_nonpain = np.array([])
            vals_pain = np.array([])

            if pac_cols:
                roi_pac_cols = [
                    c for c in pac_cols if any(f"_ch_{ch}_" in c for ch in roi_channels)
                ]
                if roi_pac_cols:
                    roi_vals = features_df[roi_pac_cols].mean(axis=1)
                    vals_nonpain = roi_vals[~pain_mask].dropna().values
                    vals_pain = roi_vals[pain_mask].dropna().values

            plot_data[key] = (vals_nonpain, vals_pain)

            if len(vals_nonpain) > 3 and len(vals_pain) > 3:
                _, p = stats.mannwhitneyu(vals_nonpain, vals_pain)
                pooled_std = np.sqrt(
                    (
                        (len(vals_nonpain) - 1) * np.var(vals_nonpain, ddof=1)
                        + (len(vals_pain) - 1) * np.var(vals_pain, ddof=1)
                    )
                    / (len(vals_nonpain) + len(vals_pain) - 2)
                )
                d = (
                    (np.mean(vals_pain) - np.mean(vals_nonpain)) / pooled_std
                    if pooled_std > 0
                    else 0
                )
                all_pvalues.append(p)
                pvalue_keys.append((key, p, d))

    qvalues: Dict[tuple[int, int], tuple[float, float, float, bool]] = {}
    if all_pvalues:
        rejected, qvals, _ = apply_fdr_correction(all_pvalues, config=config)
        for i, (key, p, d) in enumerate(pvalue_keys):
            qvalues[key] = (p, qvals[i], d, rejected[i])

    plot_cfg = get_plot_config(config)
    fig, axes = plt.subplots(n_rois, n_pairs, figsize=(12, 2.5 * n_rois))

    base_palette = (
        list(band_colors.values()) if band_colors else ["#440154", "#3b528b", "#21918c", "#fde725"]
    )
    pair_colors = [base_palette[i % len(base_palette)] for i in range(n_pairs)]

    for row_idx, roi_name in enumerate(roi_names):
        for col_idx, pair in enumerate(pac_pairs):
            ax = axes[row_idx, col_idx]
            key = (row_idx, col_idx)
            vals_nonpain, vals_pain = plot_data[key]

            if len(vals_nonpain) == 0 and len(vals_pain) == 0:
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=plot_cfg.font.medium,
                    color="gray",
                )
                ax.set_xticks([])
                continue

            if len(vals_nonpain) > 0 and len(vals_pain) > 0:
                bp = ax.boxplot(
                    [vals_nonpain, vals_pain],
                    positions=[0, 1],
                    widths=0.4,
                    patch_artist=True,
                )
                bp["boxes"][0].set_facecolor(condition_colors["nonpain"])
                bp["boxes"][0].set_alpha(0.6)
                bp["boxes"][1].set_facecolor(condition_colors["pain"])
                bp["boxes"][1].set_alpha(0.6)

                ax.scatter(
                    np.random.uniform(-0.08, 0.08, len(vals_nonpain)),
                    vals_nonpain,
                    c=condition_colors["nonpain"],
                    alpha=0.3,
                    s=6,
                )
                ax.scatter(
                    1 + np.random.uniform(-0.08, 0.08, len(vals_pain)),
                    vals_pain,
                    c=condition_colors["pain"],
                    alpha=0.3,
                    s=6,
                )

                if key in qvalues:
                    _, q, d, sig = qvalues[key]
                    sig_marker = "†" if sig else ""
                    ax.text(
                        0.5,
                        0.95,
                        f"q={q:.3f}{sig_marker}\nd={d:.2f}",
                        transform=ax.transAxes,
                        ha="center",
                        fontsize=plot_cfg.font.annotation,
                        va="top",
                    )

            ax.set_xticks([0, 1])
            ax.set_xticklabels(["NP", "P"], fontsize=plot_cfg.font.small)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            if row_idx == 0:
                pair_label = pair.replace("_", "→")
                ax.set_title(
                    pair_label.capitalize(),
                    fontweight="bold",
                    color=pair_colors[col_idx],
                    fontsize=plot_cfg.font.title,
                )
            if col_idx == 0:
                short_name = roi_name.replace("_", "\n").replace("Contra", "C").replace(
                    "Ipsi", "I"
                )
                ax.set_ylabel(short_name, fontsize=plot_cfg.font.medium)

    n_pain = int(pain_mask.sum())
    n_nonpain = int((~pain_mask).sum())
    n_tests = len(all_pvalues)
    n_sig = sum(1 for k in qvalues if qvalues[k][3])

    title = (
        "Phase-Amplitude Coupling by ROI: Pain vs Non-Pain Comparison\n"
        "Plateau Phase, MVL Method\n"
        f"Subject: {subject} | N: {n_nonpain} non-pain, {n_pain} pain | "
        f"FDR: {n_sig}/{n_tests} significant (†=q<0.05)"
    )
    fig.suptitle(title, fontsize=plot_cfg.font.suptitle, fontweight="bold", y=1.02)

    plt.tight_layout()
    save_fig(
        fig,
        save_dir / f"sub-{subject}_pac_roi_condition",
        formats=plot_cfg.formats,
        dpi=plot_cfg.dpi,
        bbox_inches=plot_cfg.bbox_inches,
        pad_inches=plot_cfg.pad_inches,
    )
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
    """

    if features_df is None or features_df.empty or events_df is None:
        return

    pain_mask = extract_pain_mask(events_df, config)
    if pain_mask is None:
        return

    if segments is None:
        segments = ["baseline", "plateau"]

    bands, band_colors, condition_colors = _get_bands_and_palettes(config)
    n_bands = len(bands)
    n_segments = len(segments)

    plot_data = {}
    all_pvalues = []
    pvalue_keys = []

    for row_idx, band in enumerate(bands):
        for col_idx, segment in enumerate(segments):
            key = (row_idx, col_idx)

            pattern = f"{feature_prefix}_{segment}_{band}_ch_"
            cols = [c for c in features_df.columns if pattern in c]

            vals_nonpain = np.array([])
            vals_pain = np.array([])

            if cols:
                mean_vals = features_df[cols].mean(axis=1)
                vals_nonpain = mean_vals[~pain_mask].dropna().values
                vals_pain = mean_vals[pain_mask].dropna().values

            plot_data[key] = (vals_nonpain, vals_pain)

            if len(vals_nonpain) > 3 and len(vals_pain) > 3:
                _, p = stats.mannwhitneyu(vals_nonpain, vals_pain)
                pooled_std = np.sqrt(
                    (
                        (len(vals_nonpain) - 1) * np.var(vals_nonpain, ddof=1)
                        + (len(vals_pain) - 1) * np.var(vals_pain, ddof=1)
                    )
                    / (len(vals_nonpain) + len(vals_pain) - 2)
                )
                d = (
                    (np.mean(vals_pain) - np.mean(vals_nonpain)) / pooled_std
                    if pooled_std > 0
                    else 0
                )
                all_pvalues.append(p)
                pvalue_keys.append((key, p, d))

    has_data = any(len(pdata[0]) > 0 or len(pdata[1]) > 0 for pdata in plot_data.values())
    if not has_data:
        if logger:
            logger.warning(
                f"No {feature_label} data found for band × segment × condition plot"
            )
        return

    qvalues = {}
    if all_pvalues:
        rejected, qvals, _ = apply_fdr_correction(all_pvalues, config=config)
        for i, (key, p, d) in enumerate(pvalue_keys):
            qvalues[key] = (p, qvals[i], d, rejected[i])

    plot_cfg = get_plot_config(config)
    fig, axes = plt.subplots(n_bands, n_segments, figsize=(5 * n_segments, 3 * n_bands))

    for row_idx, band in enumerate(bands):
        for col_idx, segment in enumerate(segments):
            ax = axes[row_idx, col_idx]
            key = (row_idx, col_idx)
            vals_nonpain, vals_pain = plot_data[key]

            if len(vals_nonpain) == 0 and len(vals_pain) == 0:
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=plot_cfg.font.title,
                    color="gray",
                )
                ax.set_xticks([])
                continue

            if len(vals_nonpain) > 0 and len(vals_pain) > 0:
                bp = ax.boxplot(
                    [vals_nonpain, vals_pain],
                    positions=[0, 1],
                    widths=0.5,
                    patch_artist=True,
                )
                bp["boxes"][0].set_facecolor(condition_colors["nonpain"])
                bp["boxes"][0].set_alpha(0.6)
                bp["boxes"][1].set_facecolor(condition_colors["pain"])
                bp["boxes"][1].set_alpha(0.6)

                ax.scatter(
                    np.random.uniform(-0.1, 0.1, len(vals_nonpain)),
                    vals_nonpain,
                    c=condition_colors["nonpain"],
                    alpha=0.4,
                    s=10,
                )
                ax.scatter(
                    1 + np.random.uniform(-0.1, 0.1, len(vals_pain)),
                    vals_pain,
                    c=condition_colors["pain"],
                    alpha=0.4,
                    s=10,
                )

                all_vals = np.concatenate([vals_nonpain, vals_pain])
                ymin, ymax = np.nanmin(all_vals), np.nanmax(all_vals)
                yrange = ymax - ymin if ymax > ymin else 1.0
                ax.set_ylim(ymin - 0.1 * yrange, ymax + 0.25 * yrange)

                if key in qvalues:
                    _, q, d, sig = qvalues[key]
                    sig_marker = "†" if sig else ""
                    sig_color = get_significance_color(sig, config)
                    ax.annotate(
                        f"q={q:.3f}{sig_marker}\nd={d:.2f}",
                        xy=(0.5, ymax + 0.05 * yrange),
                        ha="center",
                        fontsize=plot_cfg.font.medium,
                        color=sig_color,
                        fontweight="bold" if sig else "normal",
                    )

            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Non-Pain", "Pain"], fontsize=plot_cfg.font.title)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            if col_idx == 0:
                ax.set_ylabel(
                    band.capitalize(),
                    fontsize=plot_cfg.font.figure_title,
                    fontweight="bold",
                    color=band_colors.get(band, None),
                )

            if row_idx == 0:
                ax.set_title(
                    segment.capitalize(),
                    fontsize=plot_cfg.font.figure_title,
                    fontweight="bold",
                )

    n_pain = int(pain_mask.sum())
    n_nonpain = int((~pain_mask).sum())
    n_tests = len(all_pvalues)
    n_sig = sum(1 for k in qvalues if qvalues[k][3])

    title = (
        f"{feature_label}: Mean Across Channels by Band × Segment\n"
        f"Subject: {subject} | N: {n_nonpain} non-pain, {n_pain} pain | "
        f"Mann-Whitney U, FDR-corrected | {n_sig}/{n_tests} significant (†=q<0.05)"
    )
    fig.suptitle(title, fontsize=plot_cfg.font.figure_title, fontweight="bold", y=1.02)

    plt.tight_layout()
    safe_name = feature_prefix.lower().replace(" ", "_")
    save_fig(
        fig,
        save_dir / f"sub-{subject}_{safe_name}_band_segment_condition",
        formats=plot_cfg.formats,
        dpi=plot_cfg.dpi,
        bbox_inches=plot_cfg.bbox_inches,
        pad_inches=plot_cfg.pad_inches,
    )
    plt.close(fig)

    if logger:
        logger.info(
            f"Saved {feature_label} band × segment × condition plot ({n_sig}/{n_tests} FDR significant)"
        )


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
    from scipy.stats import wilcoxon

    plateau_window = get_config_value(config, "plateau_window", [3.0, 10.5])
    baseline_window = get_config_value(config, "baseline_window", [-3.0, -0.5])

    bands, band_colors, condition_colors = _get_bands_and_palettes(config)
    n_bands = len(bands)

    plot_data = {}
    all_pvalues = []
    pvalue_keys = []

    for band_idx, band in enumerate(bands):
        baseline_cols = [c for c in features_df.columns if f"power_baseline_{band}_ch_" in c]
        plateau_cols = [c for c in features_df.columns if f"power_plateau_{band}_ch_" in c]

        vals_baseline = np.array([])
        vals_plateau = np.array([])

        if baseline_cols and plateau_cols:
            vals_baseline = features_df[baseline_cols].mean(axis=1).dropna().values
            vals_plateau = features_df[plateau_cols].mean(axis=1).dropna().values

        plot_data[band_idx] = (vals_baseline, vals_plateau)

        if (
            len(vals_baseline) > 5
            and len(vals_plateau) > 5
            and len(vals_baseline) == len(vals_plateau)
        ):
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
        rejected, qvals, _ = apply_fdr_correction(all_pvalues, config=config)
        for i, (key, p, d) in enumerate(pvalue_keys):
            qvalues[key] = (p, qvals[i], d, rejected[i])

    plot_cfg = get_plot_config(config)
    fig, axes = plt.subplots(1, n_bands, figsize=(3 * n_bands, 5))

    for band_idx, band in enumerate(bands):
        ax = axes[band_idx]
        vals_baseline, vals_plateau = plot_data[band_idx]

        if len(vals_baseline) == 0 or len(vals_plateau) == 0:
            ax.text(
                0.5,
                0.5,
                "No data",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=plot_cfg.font.title,
                color="gray",
            )
            ax.set_xticks([])
            continue

        bp = ax.boxplot([vals_baseline, vals_plateau], positions=[0, 1], widths=0.4, patch_artist=True)
        bp["boxes"][0].set_facecolor(condition_colors["nonpain"])
        bp["boxes"][0].set_alpha(0.6)
        bp["boxes"][1].set_facecolor(condition_colors["pain"])
        bp["boxes"][1].set_alpha(0.6)

        ax.scatter(
            np.random.uniform(-0.08, 0.08, len(vals_baseline)),
            vals_baseline,
            c=condition_colors["nonpain"],
            alpha=0.3,
            s=6,
        )
        ax.scatter(
            1 + np.random.uniform(-0.08, 0.08, len(vals_plateau)),
            vals_plateau,
            c=condition_colors["pain"],
            alpha=0.3,
            s=6,
        )

        if len(vals_baseline) == len(vals_plateau) and len(vals_baseline) <= 100:
            for i in range(len(vals_baseline)):
                ax.plot([0, 1], [vals_baseline[i], vals_plateau[i]], c="gray", alpha=0.15, lw=0.5)

        all_vals = np.concatenate([vals_baseline, vals_plateau])
        ymin, ymax = np.nanmin(all_vals), np.nanmax(all_vals)
        yrange = ymax - ymin if ymax > ymin else 0.1
        ax.set_ylim(ymin - 0.1 * yrange, ymax + 0.3 * yrange)

        if band_idx in qvalues:
            _, q, d, sig = qvalues[band_idx]
            sig_marker = "†" if sig else ""
            sig_color = get_significance_color(sig, config)
            ax.annotate(
                f"q={q:.3f}{sig_marker}\nd={d:.2f}",
                xy=(0.5, ymax + 0.05 * yrange),
                ha="center",
                fontsize=plot_cfg.font.medium,
                color=sig_color,
                fontweight="bold" if sig else "normal",
            )

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Baseline", "Plateau"], fontsize=9)
        ax.set_title(band.capitalize(), fontweight="bold", color=band_colors.get(band, None))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    n_tests = len(all_pvalues)
    n_sig = sum(1 for k in qvalues if qvalues[k][3])

    title = (
        "Band Power: Baseline vs Plateau (Paired Comparison)\n"
        f"Subject: {subject} | Wilcoxon signed-rank | "
        f"FDR: {n_sig}/{n_tests} significant (†=q<0.05)"
    )
    fig.suptitle(title, fontsize=plot_cfg.font.suptitle, fontweight="bold", y=1.02)

    plt.tight_layout()
    save_fig(
        fig,
        save_dir / f"sub-{subject}_power_plateau_vs_baseline",
        formats=plot_cfg.formats,
        dpi=plot_cfg.dpi,
        bbox_inches=plot_cfg.bbox_inches,
        pad_inches=plot_cfg.pad_inches,
    )
    plt.close(fig)

    if logger:
        logger.info(f"Saved Power plateau vs baseline plot ({n_sig}/{n_tests} FDR significant)")


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
    """Temporal Evolution: Early vs Mid vs Late Plateau."""

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
            cols = [c for c in features_df.columns if f"{feature_prefix}_{time_bin}_{band}_ch_" in c]

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

        ax.errorbar(
            x - 0.1,
            nonpain_means,
            yerr=nonpain_sems,
            fmt="o-",
            color=condition_colors["nonpain"],
            label="Non-Pain",
            capsize=3,
            markersize=8,
        )
        ax.errorbar(
            x + 0.1,
            pain_means,
            yerr=pain_sems,
            fmt="o-",
            color=condition_colors["pain"],
            label="Pain",
            capsize=3,
            markersize=8,
        )

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
    save_fig(
        fig,
        save_dir / f"sub-{subject}_{feature_prefix}_temporal_evolution",
        formats=plot_cfg.formats,
        dpi=plot_cfg.dpi,
        bbox_inches=plot_cfg.bbox_inches,
        pad_inches=plot_cfg.pad_inches,
    )
    plt.close(fig)

    if logger:
        logger.info(f"Saved {feature_label} temporal evolution plot")


__all__ = [
    "plot_power_by_roi_band_condition",
    "plot_dynamics_by_roi_band_condition",
    "plot_aperiodic_by_roi_condition",
    "plot_connectivity_by_roi_band_condition",
    "plot_itpc_by_roi_band_condition",
    "plot_itpc_plateau_vs_baseline",
    "plot_pac_by_roi_condition",
    "plot_band_segment_condition",
    "plot_power_plateau_vs_baseline",
    "plot_temporal_evolution",
]
