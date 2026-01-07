from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from eeg_pipeline.domain.features.naming import NamingSchema
from eeg_pipeline.utils.analysis.events import extract_comparison_mask
from eeg_pipeline.plotting.config import get_plot_config
from eeg_pipeline.plotting.io.figures import save_fig
from eeg_pipeline.plotting.features.utils import (
    apply_fdr_correction,
    get_band_colors,
    get_band_names,
    get_condition_colors,
    get_significance_color,
    collect_named_series,
    get_named_segments,
    get_named_bands,
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


def _get_comparison_masks(events_df: pd.DataFrame, config: Any) -> Optional[tuple[np.ndarray, np.ndarray, str, str]]:
    if events_df is None or events_df.empty:
        return None
    return extract_comparison_mask(events_df, config, require_enabled=False)


def _get_named_identifiers(
    features_df: pd.DataFrame,
    *,
    group: str,
    segment: str,
    scope: str,
    band: Optional[str] = None,
) -> List[str]:
    identifiers = set()
    for col in features_df.columns:
        parsed = NamingSchema.parse(str(col))
        if not parsed.get("valid"):
            continue
        if parsed.get("group") != group:
            continue
        if str(parsed.get("segment") or "") != str(segment):
            continue
        if str(parsed.get("scope") or "") != str(scope):
            continue
        if band and str(parsed.get("band") or "") != str(band):
            continue
        identifier = parsed.get("identifier")
        if identifier:
            identifiers.add(str(identifier))
    return sorted(identifiers)


def _collect_roi_series(
    features_df: pd.DataFrame,
    *,
    group: str,
    segment: str,
    band: str,
    roi_name: str,
    roi_channels: Optional[List[str]] = None,
    stat_preference: Optional[List[str]] = None,
) -> pd.Series:
    stat_preference = list(stat_preference or [])
    if not stat_preference:
        stat_preference = ["val"]

    roi_series, _, _ = collect_named_series(
        features_df,
        group=group,
        segment=segment,
        band=band,
        identifier=roi_name,
        stat_preference=stat_preference,
        scope_preference=["roi"],
    )
    if not roi_series.empty:
        return roi_series

    if not roi_channels:
        return pd.Series(dtype=float)

    roi_set = set(roi_channels)
    for stat in stat_preference:
        cols = []
        for col in features_df.columns:
            parsed = NamingSchema.parse(str(col))
            if not parsed.get("valid"):
                continue
            if parsed.get("group") != group:
                continue
            if str(parsed.get("segment") or "") != str(segment):
                continue
            if str(parsed.get("band") or "") != str(band):
                continue
            if str(parsed.get("scope") or "") != "ch":
                continue
            if str(parsed.get("stat") or "") != str(stat):
                continue
            identifier = str(parsed.get("identifier") or "")
            if identifier and identifier in roi_set:
                cols.append(str(col))
        if cols:
            return features_df[cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)

    return pd.Series(dtype=float)


def plot_power_by_roi_band_condition(
    features_df: pd.DataFrame,
    events_df: pd.DataFrame,
    subject: str,
    save_dir: Path,
    logger: logging.Logger,
    config: Any,
) -> None:
    """Power by ROI, Band, and Condition (active timing).

    Creates a grid: rows = ROIs (9), cols = frequency bands (5).
    Each cell shows a 2-condition box+strip comparison.
    FDR-corrected significance markers applied across all tests.
    """

    if features_df is None or features_df.empty or events_df is None:
        return
    if len(features_df) != len(events_df):
        if logger:
            logger.warning(
                "Power ROI plot skipped: %d feature rows vs %d events",
                len(features_df),
                len(events_df),
            )
        return

    comp = _get_comparison_masks(events_df, config)
    if comp is None:
        return
    mask1, mask2, label1, label2 = comp
    mask1 = np.asarray(mask1, dtype=bool)
    mask2 = np.asarray(mask2, dtype=bool)
    if int(mask1.sum()) == 0 or int(mask2.sum()) == 0:
        return

    plot_cfg = get_plot_config(config)

    from eeg_pipeline.utils.config.loader import get_config_value
    active_window = get_config_value(config, "time_frequency_analysis.active_window", [3.0, 10.5])
    active_label = f"{active_window[0]:.1f}-{active_window[1]:.1f}s"

    segments = get_named_segments(features_df, group="power")
    if not segments:
        return
    segment = "active" if "active" in segments else segments[0]

    bands = get_named_bands(features_df, group="power", segment=segment)
    if not bands:
        return
    band_order = get_band_names(config)
    bands = [b for b in band_order if b in bands] + [b for b in bands if b not in band_order]

    rois = get_roi_definitions(config)
    data_rois = _get_named_identifiers(features_df, group="itpc", segment=segment, scope="roi")
    if rois:
        roi_names = list(rois.keys())
    else:
        roi_names = data_rois
    if not roi_names:
        if logger:
            logger.warning("No ROI names found for power ROI plot")
        return

    all_channels = extract_channels_from_columns(list(features_df.columns))

    _, band_colors, condition_colors = _get_bands_and_palettes(config)
    color1 = plot_cfg.get_color("blue")
    color2 = plot_cfg.get_color("red")
    tick1 = str(label1)
    tick2 = str(label2)
    if {label1, label2} == {"Non-pain", "Pain"}:
        color1 = condition_colors["nonpain"] if label1 == "Non-pain" else condition_colors["pain"]
        color2 = condition_colors["pain"] if label2 == "Pain" else condition_colors["nonpain"]
        tick1 = "NP" if label1 == "Non-pain" else "P"
        tick2 = "P" if label2 == "Pain" else "NP"
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
                if parsed.get("segment") != segment:
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

            vals_1 = roi_vals[mask1].dropna().values
            vals_2 = roi_vals[mask2].dropna().values

            plot_data[key] = (vals_1, vals_2)

            if len(vals_1) > 3 and len(vals_2) > 3:
                _, p = stats.mannwhitneyu(vals_1, vals_2)
                pooled_std = np.sqrt(
                    (
                        (len(vals_1) - 1) * np.var(vals_1, ddof=1)
                        + (len(vals_2) - 1) * np.var(vals_2, ddof=1)
                    )
                    / (len(vals_1) + len(vals_2) - 2)
                )
                d = (
                    (np.mean(vals_2) - np.mean(vals_1)) / pooled_std
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
    fig, axes = plt.subplots(
        n_rois,
        n_bands,
        figsize=(width_per_col * n_bands, height_per_row * n_rois),
        squeeze=False,
    )

    for row_idx, roi_name in enumerate(roi_names):
        for col_idx, band in enumerate(bands):
            ax = axes[row_idx, col_idx]
            key = (row_idx, col_idx)
            vals_1, vals_2 = plot_data[key]

            if len(vals_1) == 0 and len(vals_2) == 0:
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

            if len(vals_1) > 0 and len(vals_2) > 0:
                bp = ax.boxplot(
                    [vals_1, vals_2],
                    positions=[0, 1],
                    widths=0.4,
                    patch_artist=True,
                )
                bp["boxes"][0].set_facecolor(color1)
                bp["boxes"][0].set_alpha(0.6)
                bp["boxes"][1].set_facecolor(color2)
                bp["boxes"][1].set_alpha(0.6)

                ax.scatter(
                    np.random.uniform(-0.08, 0.08, len(vals_1)),
                    vals_1,
                    c=color1,
                    alpha=0.3,
                    s=6,
                )
                ax.scatter(
                    1 + np.random.uniform(-0.08, 0.08, len(vals_2)),
                    vals_2,
                    c=color2,
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
            ax.set_xticklabels([tick1, tick2], fontsize=plot_cfg.font.small)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            if row_idx == 0:
                ax.set_title(
                    band.capitalize(),
                    fontweight="bold",
                    color=band_colors.get(band) or "gray",
                    fontsize=plot_cfg.font.title,
                )
            if col_idx == 0:
                short_name = roi_name.replace("_", "\n").replace("Contra", "C").replace(
                    "Ipsi", "I"
                )
                ax.set_ylabel(short_name, fontsize=plot_cfg.font.medium)

    n_1 = int(mask1.sum())
    n_2 = int(mask2.sum())
    n_tests = len(all_pvalues)
    n_sig = sum(1 for k in qvalues if qvalues[k][3])

    title = (
        "Band Power by ROI: Condition Comparison\n"
        f"Active Phase ({active_label}), Baseline-Normalized dB\n"
        f"Subject: {subject} | N: {n_1} {label1}, {n_2} {label2} | "
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


def plot_complexity_by_roi_band_condition(
    features_df: pd.DataFrame,
    events_df: pd.DataFrame,
    subject: str,
    save_dir: Path,
    logger: logging.Logger,
    config: Any,
    metric: str = "lzc",
) -> None:
    """Complexity (LZC/PE) by ROI, Band, and Condition.

    Creates a grid: rows = ROIs (9), cols = frequency bands (5).
    FDR-corrected significance markers applied across all tests.
    """

    if features_df is None or features_df.empty or events_df is None:
        return

    comp = _get_comparison_masks(events_df, config)
    if comp is None:
        return
    mask1, mask2, label1, label2 = comp
    mask1 = np.asarray(mask1, dtype=bool)
    mask2 = np.asarray(mask2, dtype=bool)
    if int(mask1.sum()) == 0 or int(mask2.sum()) == 0:
        return

    plot_cfg = get_plot_config(config)

    from eeg_pipeline.utils.config.loader import get_config_value

    active_window = get_config_value(config, "time_frequency_analysis.active_window", [3.0, 10.5])
    active_label = f"{active_window[0]:.1f}-{active_window[1]:.1f}s"

    rois = get_roi_definitions(config)
    if not rois:
        return

    all_channels = extract_channels_from_columns(list(features_df.columns))

    bands, band_colors, condition_colors = _get_bands_and_palettes(config)
    color1 = plot_cfg.get_color("blue")
    color2 = plot_cfg.get_color("red")
    tick1 = str(label1)
    tick2 = str(label2)
    if {label1, label2} == {"Non-pain", "Pain"}:
        color1 = condition_colors["nonpain"] if label1 == "Non-pain" else condition_colors["pain"]
        color2 = condition_colors["pain"] if label2 == "Pain" else condition_colors["nonpain"]
        tick1 = "NP" if label1 == "Non-pain" else "P"
        tick2 = "P" if label2 == "Pain" else "NP"
    roi_names = list(rois.keys())
    n_rois = len(roi_names)
    n_bands = len(bands)

    metric_labels = {
        "lzc": "Lempel-Ziv Complexity",
        "pe": "Permutation Entropy",
    }
    metric_label = metric_labels.get(metric, metric.upper())

    segments = []
    for col in features_df.columns:
        parsed = NamingSchema.parse(str(col))
        if parsed.get("valid") and parsed.get("group") == "comp":
            segment = str(parsed.get("segment") or "")
            if segment:
                segments.append(segment)
    segment = "active" if "active" in segments else (segments[0] if segments else "active")

    plot_data: Dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    all_pvalues = []
    pvalue_keys = []

    for row_idx, roi_name in enumerate(roi_names):
        roi_patterns = rois[roi_name]
        roi_channels = get_roi_channels(roi_patterns, all_channels)

        for col_idx, band in enumerate(bands):
            key = (row_idx, col_idx)

            roi_vals = pd.Series([np.nan] * len(features_df))
            roi_set = set(roi_channels)
            cols: List[str] = []
            for c in features_df.columns:
                parsed = NamingSchema.parse(str(c))
                if not parsed.get("valid"):
                    continue
                if parsed.get("group") != "comp":
                    continue
                if str(parsed.get("segment") or "") != segment:
                    continue
                if str(parsed.get("band") or "") != str(band):
                    continue
                if str(parsed.get("scope") or "") != "ch":
                    continue
                if str(parsed.get("stat") or "") != str(metric):
                    continue
                if str(parsed.get("identifier") or "") not in roi_set:
                    continue
                cols.append(str(c))

            if cols:
                roi_vals = features_df[cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)

            vals_1 = roi_vals[mask1].dropna().values
            vals_2 = roi_vals[mask2].dropna().values
            plot_data[key] = (vals_1, vals_2)

            if len(vals_1) > 3 and len(vals_2) > 3:
                _, p = stats.mannwhitneyu(vals_1, vals_2)
                pooled_std = np.sqrt(
                    (
                        (len(vals_1) - 1) * np.var(vals_1, ddof=1)
                        + (len(vals_2) - 1) * np.var(vals_2, ddof=1)
                    )
                    / (len(vals_1) + len(vals_2) - 2)
                )
                d = (
                    (np.mean(vals_2) - np.mean(vals_1)) / pooled_std
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
    fig, axes = plt.subplots(
        n_rois,
        n_bands,
        figsize=(width_per_col * n_bands, height_per_row * n_rois),
        squeeze=False,
    )

    for row_idx, roi_name in enumerate(roi_names):
        for col_idx, band in enumerate(bands):
            ax = axes[row_idx, col_idx]
            key = (row_idx, col_idx)
            vals_1, vals_2 = plot_data[key]

            if len(vals_1) == 0 and len(vals_2) == 0:
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

            if len(vals_1) > 0 and len(vals_2) > 0:
                bp = ax.boxplot(
                    [vals_1, vals_2],
                    positions=[0, 1],
                    widths=0.4,
                    patch_artist=True,
                )
                bp["boxes"][0].set_facecolor(color1)
                bp["boxes"][0].set_alpha(0.6)
                bp["boxes"][1].set_facecolor(color2)
                bp["boxes"][1].set_alpha(0.6)

                ax.scatter(
                    np.random.uniform(-0.08, 0.08, len(vals_1)),
                    vals_1,
                    c=color1,
                    alpha=0.3,
                    s=6,
                )
                ax.scatter(
                    1 + np.random.uniform(-0.08, 0.08, len(vals_2)),
                    vals_2,
                    c=color2,
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
            ax.set_xticklabels([tick1, tick2], fontsize=plot_cfg.font.small)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            if row_idx == 0:
                ax.set_title(
                    band.capitalize(),
                    fontweight="bold",
                    color=band_colors.get(band) or "gray",
                    fontsize=plot_cfg.font.title,
                )
            if col_idx == 0:
                short_name = roi_name.replace("_", "\n").replace("Contra", "C").replace(
                    "Ipsi", "I"
                )
                ax.set_ylabel(short_name, fontsize=plot_cfg.font.medium)

    n_1 = int(mask1.sum())
    n_2 = int(mask2.sum())
    n_tests = len(all_pvalues)
    n_sig = sum(1 for k in qvalues if qvalues[k][3])

    title = (
        f"{metric_label} by ROI: Condition Comparison\n"
        f"Active Phase ({active_label})\n"
        f"Subject: {subject} | N: {n_1} {label1}, {n_2} {label2} | "
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

    comp = _get_comparison_masks(events_df, config)
    if comp is None:
        return
    mask1, mask2, label1, label2 = comp
    mask1 = np.asarray(mask1, dtype=bool)
    mask2 = np.asarray(mask2, dtype=bool)
    if int(mask1.sum()) == 0 or int(mask2.sum()) == 0:
        return

    from eeg_pipeline.utils.config.loader import get_config_value

    active_window = get_config_value(config, "time_frequency_analysis.active_window", [3.0, 10.5])
    active_label = f"{active_window[0]:.1f}-{active_window[1]:.1f}s"

    rois = get_roi_definitions(config)
    if not rois:
        return

    all_channels = extract_channels_from_columns(list(features_df.columns))

    roi_names = list(rois.keys())
    n_rois = len(roi_names)
    metrics = ["slope", "offset"]
    n_metrics = len(metrics)

    condition_colors = get_condition_colors(config)
    color1 = condition_colors.get("nonpain", "#4C72B0")
    color2 = condition_colors.get("pain", "#C44E52")
    tick1 = str(label1)
    tick2 = str(label2)
    if {label1, label2} == {"Non-pain", "Pain"}:
        color1 = condition_colors["nonpain"] if label1 == "Non-pain" else condition_colors["pain"]
        color2 = condition_colors["pain"] if label2 == "Pain" else condition_colors["nonpain"]
        tick1 = "NP" if label1 == "Non-pain" else "P"
        tick2 = "P" if label2 == "Pain" else "NP"

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
                if f"aperiodic_active_broadband_ch_" in col and f"_{metric}" in col:
                    for ch in roi_channels:
                        if f"_ch_{ch}_" in col:
                            cols.append(col)
                            break

            if cols:
                roi_vals = features_df[cols].mean(axis=1)
                vals_1 = roi_vals[mask1].dropna().values
                vals_2 = roi_vals[mask2].dropna().values
            else:
                vals_1 = np.array([])
                vals_2 = np.array([])

            plot_data[key] = (vals_1, vals_2)

            if len(vals_1) > 3 and len(vals_2) > 3:
                _, p = stats.mannwhitneyu(vals_1, vals_2)
                pooled_std = np.sqrt(
                    (
                        (len(vals_1) - 1) * np.var(vals_1, ddof=1)
                        + (len(vals_2) - 1) * np.var(vals_2, ddof=1)
                    )
                    / (len(vals_1) + len(vals_2) - 2)
                )
                d = (
                    (np.mean(vals_2) - np.mean(vals_1)) / pooled_std
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
            vals_1, vals_2 = plot_data[key]

            if len(vals_1) == 0 and len(vals_2) == 0:
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

            if len(vals_1) > 0 and len(vals_2) > 0:
                bp = ax.boxplot(
                    [vals_1, vals_2],
                    positions=[0, 1],
                    widths=0.4,
                    patch_artist=True,
                )
                bp["boxes"][0].set_facecolor(color1)
                bp["boxes"][0].set_alpha(0.6)
                bp["boxes"][1].set_facecolor(color2)
                bp["boxes"][1].set_alpha(0.6)

                ax.scatter(
                    np.random.uniform(-0.08, 0.08, len(vals_1)),
                    vals_1,
                    c=color1,
                    alpha=0.3,
                    s=6,
                )
                ax.scatter(
                    1 + np.random.uniform(-0.08, 0.08, len(vals_2)),
                    vals_2,
                    c=color2,
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
            ax.set_xticklabels([tick1, tick2], fontsize=plot_cfg.font.small)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            if row_idx == 0:
                ax.set_title(metric.capitalize(), fontweight="bold", fontsize=plot_cfg.font.title)
            if col_idx == 0:
                short_name = roi_name.replace("_", "\n").replace("Contra", "C").replace(
                    "Ipsi", "I"
                )
                ax.set_ylabel(short_name, fontsize=plot_cfg.font.medium)

    n_1 = int(mask1.sum())
    n_2 = int(mask2.sum())
    n_tests = len(all_pvalues)
    n_sig = sum(1 for k in qvalues if qvalues[k][3])

    title = (
        "Aperiodic 1/f Parameters by ROI: Condition Comparison\n"
        f"Active Phase ({active_label})\n"
        f"Subject: {subject} | N: {n_1} {label1}, {n_2} {label2} | "
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

    comp = _get_comparison_masks(events_df, config)
    if comp is None:
        return
    mask1, mask2, label1, label2 = comp
    mask1 = np.asarray(mask1, dtype=bool)
    mask2 = np.asarray(mask2, dtype=bool)
    if int(mask1.sum()) == 0 or int(mask2.sum()) == 0:
        return

    from eeg_pipeline.utils.config.loader import get_config_value

    active_window = get_config_value(config, "time_frequency_analysis.active_window", [3.0, 10.5])
    active_label = f"{active_window[0]:.1f}-{active_window[1]:.1f}s"

    rois = get_roi_definitions(config)
    if not rois:
        return

    all_pairs = extract_channel_pairs_from_columns(list(features_df.columns))
    all_channels = list(set([p[0] for p in all_pairs] + [p[1] for p in all_pairs]))

    bands, band_colors, condition_colors = _get_bands_and_palettes(config)
    color1 = condition_colors.get("nonpain", "#4C72B0")
    color2 = condition_colors.get("pain", "#C44E52")
    tick1 = str(label1)
    tick2 = str(label2)
    if {label1, label2} == {"Non-pain", "Pain"}:
        color1 = condition_colors["nonpain"] if label1 == "Non-pain" else condition_colors["pain"]
        color2 = condition_colors["pain"] if label2 == "Pain" else condition_colors["nonpain"]
        tick1 = "NP" if label1 == "Non-pain" else "P"
        tick2 = "P" if label2 == "Pain" else "NP"
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

            roi_vals = aggregate_connectivity_by_roi(features_df, f"conn_active_{band}_", roi_channels)

            if measure != "wpli":
                measure_cols = [
                    c
                    for c in features_df.columns
                    if f"conn_active_{band}_" in c and f"_{measure}" in c
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

            vals_1 = roi_vals[mask1].dropna().values
            vals_2 = roi_vals[mask2].dropna().values

            if len(vals_1) > 0 and len(vals_2) > 0:
                bp = ax.boxplot(
                    [vals_1, vals_2],
                    positions=[0, 1],
                    widths=0.4,
                    patch_artist=True,
                )
                bp["boxes"][0].set_facecolor(color1)
                bp["boxes"][0].set_alpha(0.6)
                bp["boxes"][1].set_facecolor(color2)
                bp["boxes"][1].set_alpha(0.6)

                ax.scatter(
                    np.random.uniform(-0.08, 0.08, len(vals_1)),
                    vals_1,
                    c=color1,
                    alpha=0.3,
                    s=6,
                )
                ax.scatter(
                    1 + np.random.uniform(-0.08, 0.08, len(vals_2)),
                    vals_2,
                    c=color2,
                    alpha=0.3,
                    s=6,
                )

                if len(vals_1) > 3 and len(vals_2) > 3:
                    _, p = stats.mannwhitneyu(vals_1, vals_2)
                    pooled_std = np.sqrt(
                        (
                            (len(vals_1) - 1) * np.var(vals_1, ddof=1)
                            + (len(vals_2) - 1) * np.var(vals_2, ddof=1)
                        )
                        / (len(vals_1) + len(vals_2) - 2)
                    )
                    d = (
                        (np.mean(vals_2) - np.mean(vals_1)) / pooled_std
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
            ax.set_xticklabels([tick1, tick2], fontsize=plot_cfg.font.small)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            if row_idx == 0:
                ax.set_title(
                    band.capitalize(),
                    fontweight="bold",
                    color=band_colors.get(band) or "gray",
                    fontsize=plot_cfg.font.title,
                )
            if col_idx == 0:
                short_name = roi_name.replace("_", "\n").replace("Contra", "C").replace(
                    "Ipsi", "I"
                )
                ax.set_ylabel(short_name, fontsize=plot_cfg.font.medium)

    n_1 = int(mask1.sum())
    n_2 = int(mask2.sum())
    fig.suptitle(
        f"Within-ROI {measure_label} by Band: Active ({active_label}) (sub-{subject})\nN: {n_1} {label1}, {n_2} {label2}",
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
    segment: str = "active",
) -> None:
    """ITPC by ROI, Band, and Condition.

    Creates a grid: rows = ROIs (9), cols = frequency bands (5).
    Each cell shows pain vs non-pain box+strip comparison.
    FDR-corrected significance markers applied across all tests.
    """

    if features_df is None or features_df.empty or events_df is None:
        return
    if len(features_df) != len(events_df):
        if logger:
            logger.warning(
                "ITPC ROI plot skipped: %d feature rows vs %d events",
                len(features_df),
                len(events_df),
            )
        return

    comp = _get_comparison_masks(events_df, config)
    if comp is None:
        return
    mask1, mask2, label1, label2 = comp
    mask1 = np.asarray(mask1, dtype=bool)
    mask2 = np.asarray(mask2, dtype=bool)
    if int(mask1.sum()) == 0 or int(mask2.sum()) == 0:
        return

    segments = get_named_segments(features_df, group="itpc")
    if not segments:
        return
    if segment not in segments:
        segment = "active" if "active" in segments else segments[0]
    segment_label = segment.replace("_", " ").title()

    bands = get_named_bands(features_df, group="itpc", segment=segment)
    if not bands:
        return
    band_order = get_band_names(config)
    bands = [b for b in band_order if b in bands] + [b for b in bands if b not in band_order]

    rois = get_roi_definitions(config)
    data_rois = _get_named_identifiers(features_df, group="itpc", segment=segment, scope="roi")
    if rois:
        roi_names = list(rois.keys())
    else:
        roi_names = data_rois
    if not roi_names:
        if logger:
            logger.warning("No ROI names found for ITPC plot")
        return

    all_channels = extract_channels_from_columns(list(features_df.columns))

    _, band_colors, condition_colors = _get_bands_and_palettes(config)
    color1 = condition_colors.get("nonpain", "#4C72B0")
    color2 = condition_colors.get("pain", "#C44E52")
    tick1 = str(label1)
    tick2 = str(label2)
    if {label1, label2} == {"Non-pain", "Pain"}:
        color1 = condition_colors["nonpain"] if label1 == "Non-pain" else condition_colors["pain"]
        color2 = condition_colors["pain"] if label2 == "Pain" else condition_colors["nonpain"]
        tick1 = "NP" if label1 == "Non-pain" else "P"
        tick2 = "P" if label2 == "Pain" else "NP"
    n_rois = len(roi_names)
    n_bands = len(bands)

    plot_data: Dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    all_pvalues = []
    pvalue_keys = []

    stat_preference = ["val", "mean", "avg", "value"]

    for row_idx, roi_name in enumerate(roi_names):
        roi_channels = []
        if rois and roi_name in rois:
            roi_channels = get_roi_channels(rois[roi_name], all_channels)

        for col_idx, band in enumerate(bands):
            key = (row_idx, col_idx)

            roi_vals = _collect_roi_series(
                features_df,
                group="itpc",
                segment=segment,
                band=band,
                roi_name=roi_name,
                roi_channels=roi_channels,
                stat_preference=stat_preference,
            )
            if roi_vals.empty:
                roi_vals = pd.Series([np.nan] * len(features_df), index=features_df.index)

            vals_1 = roi_vals[mask1].dropna().values
            vals_2 = roi_vals[mask2].dropna().values

            plot_data[key] = (vals_1, vals_2)

            if len(vals_1) > 3 and len(vals_2) > 3:
                _, p = stats.mannwhitneyu(vals_1, vals_2)
                pooled_std = np.sqrt(
                    (
                        (len(vals_1) - 1) * np.var(vals_1, ddof=1)
                        + (len(vals_2) - 1) * np.var(vals_2, ddof=1)
                    )
                    / (len(vals_1) + len(vals_2) - 2)
                )
                d = (
                    (np.mean(vals_2) - np.mean(vals_1)) / pooled_std
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
            vals_1, vals_2 = plot_data[key]

            if len(vals_1) == 0 and len(vals_2) == 0:
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

            if len(vals_1) > 0 and len(vals_2) > 0:
                bp = ax.boxplot(
                    [vals_1, vals_2],
                    positions=[0, 1],
                    widths=0.4,
                    patch_artist=True,
                )
                bp["boxes"][0].set_facecolor(color1)
                bp["boxes"][0].set_alpha(0.6)
                bp["boxes"][1].set_facecolor(color2)
                bp["boxes"][1].set_alpha(0.6)

                ax.scatter(
                    np.random.uniform(-0.08, 0.08, len(vals_1)),
                    vals_1,
                    c=color1,
                    alpha=0.3,
                    s=6,
                )
                ax.scatter(
                    1 + np.random.uniform(-0.08, 0.08, len(vals_2)),
                    vals_2,
                    c=color2,
                    alpha=0.3,
                    s=6,
                )

                all_vals = np.concatenate([vals_1, vals_2])
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
            ax.set_xticklabels([tick1, tick2], fontsize=plot_cfg.font.small)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            if row_idx == 0:
                ax.set_title(
                    band.capitalize(),
                    fontweight="bold",
                    color=band_colors.get(band) or "gray",
                    fontsize=plot_cfg.font.title,
                )
            if col_idx == 0:
                short_name = roi_name.replace("_", "\n").replace("Contra", "C").replace(
                    "Ipsi", "I"
                )
                ax.set_ylabel(short_name, fontsize=plot_cfg.font.medium)

    n_1 = int(mask1.sum())
    n_2 = int(mask2.sum())
    n_tests = len(all_pvalues)
    n_sig = sum(1 for k in qvalues if qvalues[k][3])

    title = (
        "Inter-Trial Phase Coherence by ROI: Condition Comparison\n"
        f"Segment ({segment_label}), LOO-ITPC\n"
        f"Subject: {subject} | N: {n_1} {label1}, {n_2} {label2} | "
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
    PAC columns: pac_active_{phase}_{amp}_ch_{channel}_val
    """

    if features_df is None or features_df.empty or events_df is None:
        return
    if len(features_df) != len(events_df):
        if logger:
            logger.warning(
                "PAC ROI plot skipped: %d feature rows vs %d events",
                len(features_df),
                len(events_df),
            )
        return

    comp = _get_comparison_masks(events_df, config)
    if comp is None:
        return
    mask1, mask2, label1, label2 = comp
    mask1 = np.asarray(mask1, dtype=bool)
    mask2 = np.asarray(mask2, dtype=bool)
    if int(mask1.sum()) == 0 or int(mask2.sum()) == 0:
        return

    segments = get_named_segments(features_df, group="pac")
    if not segments:
        return
    segment = "active" if "active" in segments else segments[0]

    pairs = get_named_bands(features_df, group="pac", segment=segment)
    if not pairs:
        return

    from eeg_pipeline.utils.config.loader import get_config_value

    cfg_pairs = get_config_value(config, "plotting.plots.features.pac_pairs", None)
    if cfg_pairs is None:
        cfg_pairs = get_config_value(config, "feature_engineering.pac.pairs", None)
    ordered_pairs = []
    for entry in cfg_pairs or []:
        if isinstance(entry, (list, tuple)) and len(entry) >= 2:
            ordered_pairs.append(f"{entry[0]}_{entry[1]}")
        elif isinstance(entry, str):
            ordered_pairs.append(entry)
    if ordered_pairs:
        pairs = [p for p in ordered_pairs if p in pairs] + [p for p in pairs if p not in ordered_pairs]

    rois = get_roi_definitions(config)
    data_rois = _get_named_identifiers(features_df, group="pac", segment=segment, scope="roi")
    if rois:
        roi_names = list(rois.keys())
    else:
        roi_names = data_rois
    if not roi_names:
        return

    all_channels = extract_channels_from_columns(list(features_df.columns))

    _, band_colors, condition_colors = _get_bands_and_palettes(config)
    color1 = condition_colors.get("nonpain", "#4C72B0")
    color2 = condition_colors.get("pain", "#C44E52")
    tick1 = str(label1)
    tick2 = str(label2)
    if {label1, label2} == {"Non-pain", "Pain"}:
        color1 = condition_colors["nonpain"] if label1 == "Non-pain" else condition_colors["pain"]
        color2 = condition_colors["pain"] if label2 == "Pain" else condition_colors["nonpain"]
        tick1 = "NP" if label1 == "Non-pain" else "P"
        tick2 = "P" if label2 == "Pain" else "NP"
    n_rois = len(roi_names)
    n_pairs = len(pairs)

    plot_data: Dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    all_pvalues = []
    pvalue_keys = []

    stat_preference = ["val", "mean", "avg", "value"]

    for row_idx, roi_name in enumerate(roi_names):
        roi_channels = []
        if rois and roi_name in rois:
            roi_channels = get_roi_channels(rois[roi_name], all_channels)

        for col_idx, pair in enumerate(pairs):
            key = (row_idx, col_idx)

            roi_vals = _collect_roi_series(
                features_df,
                group="pac",
                segment=segment,
                band=pair,
                roi_name=roi_name,
                roi_channels=roi_channels,
                stat_preference=stat_preference,
            )

            if roi_vals.empty:
                vals_1 = np.array([])
                vals_2 = np.array([])
            else:
                vals_1 = roi_vals[mask1].dropna().values
                vals_2 = roi_vals[mask2].dropna().values

            plot_data[key] = (vals_1, vals_2)

            if len(vals_1) > 3 and len(vals_2) > 3:
                _, p = stats.mannwhitneyu(vals_1, vals_2)
                pooled_std = np.sqrt(
                    (
                        (len(vals_1) - 1) * np.var(vals_1, ddof=1)
                        + (len(vals_2) - 1) * np.var(vals_2, ddof=1)
                    )
                    / (len(vals_1) + len(vals_2) - 2)
                )
                d = (
                    (np.mean(vals_2) - np.mean(vals_1)) / pooled_std
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
    fig, axes = plt.subplots(n_rois, n_pairs, figsize=(12, 2.5 * n_rois), squeeze=False)

    base_palette = (
        list(band_colors.values()) if band_colors else ["#440154", "#3b528b", "#21918c", "#fde725"]
    )
    pair_colors = [base_palette[i % len(base_palette)] for i in range(n_pairs)]

    for row_idx, roi_name in enumerate(roi_names):
        for col_idx, pair in enumerate(pairs):
            ax = axes[row_idx, col_idx]
            key = (row_idx, col_idx)
            vals_1, vals_2 = plot_data[key]

            if len(vals_1) == 0 and len(vals_2) == 0:
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

            if len(vals_1) > 0 and len(vals_2) > 0:
                bp = ax.boxplot(
                    [vals_1, vals_2],
                    positions=[0, 1],
                    widths=0.4,
                    patch_artist=True,
                )
                bp["boxes"][0].set_facecolor(color1)
                bp["boxes"][0].set_alpha(0.6)
                bp["boxes"][1].set_facecolor(color2)
                bp["boxes"][1].set_alpha(0.6)

                ax.scatter(
                    np.random.uniform(-0.08, 0.08, len(vals_1)),
                    vals_1,
                    c=color1,
                    alpha=0.3,
                    s=6,
                )
                ax.scatter(
                    1 + np.random.uniform(-0.08, 0.08, len(vals_2)),
                    vals_2,
                    c=color2,
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
            ax.set_xticklabels([tick1, tick2], fontsize=plot_cfg.font.small)
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

    n_1 = int(mask1.sum())
    n_2 = int(mask2.sum())
    n_tests = len(all_pvalues)
    n_sig = sum(1 for k in qvalues if qvalues[k][3])

    title = (
        "Phase-Amplitude Coupling by ROI: Condition Comparison\n"
        f"Segment ({segment.replace('_', ' ').title()}), MVL Method\n"
        f"Subject: {subject} | N: {n_1} {label1}, {n_2} {label2} | "
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
    segments: Optional[List[str]] = None,
    stat_preference: Optional[List[str]] = None,
    scope_preference: Optional[List[str]] = None,
) -> None:
    """Unified Band × Segment plot comparing Baseline vs Active.

    Creates a single row of plots, one per frequency band.
    Each cell shows Baseline vs Active comparison using paired Wilcoxon test.
    Uses the shared plot_paired_comparison helper for consistent styling.
    """
    from eeg_pipeline.plotting.features.utils import (
        plot_paired_comparison,
        get_named_segments,
        get_band_names,
        collect_named_series,
    )
    from eeg_pipeline.utils.config.loader import get_config_value

    if features_df is None or features_df.empty:
        return

    if segments is None:
        segments = ["baseline", "active"]
    available_segments = get_named_segments(features_df, group=feature_prefix)
    if available_segments:
        filtered = [seg for seg in segments if seg in available_segments]
        segments = filtered if filtered else available_segments

    if len(segments) < 2:
        if logger:
            logger.warning(
                f"Need at least 2 segments for {feature_label} baseline vs active comparison"
            )
        return

    baseline_seg = "baseline" if "baseline" in segments else segments[0]
    active_seg = "active" if "active" in segments else segments[-1]

    bands = get_band_names(config)

    if stat_preference is None:
        stat_preference = {
            "power": [
                "logratio",
                "logratio_mean",
                "baselined",
                "baselined_mean",
                "log10raw",
                "log10raw_mean",
                "mean",
                "val",
            ],
            "itpc": ["val"],
            "spectral": ["peak_freq", "peak_power", "center_freq", "bandwidth", "entropy"],
            "aperiodic": ["slope", "offset", "powcorr"],
            "ratios": ["power_ratio"],
            "asymmetry": ["index", "logdiff"],
            "comp": ["lzc", "pe"],
            "bursts": ["rate", "count", "duration_mean", "amp_mean", "fraction"],
        }.get(feature_prefix, [])
    if scope_preference is None:
        scope_preference = ["global", "roi", "ch", "chpair"]

    # Collect data for each band
    data_by_band = {}
    for band in bands:
        baseline_series, _, _ = collect_named_series(
            features_df,
            group=feature_prefix,
            segment=baseline_seg,
            band=band,
            stat_preference=stat_preference,
            scope_preference=scope_preference,
        )
        active_series, _, _ = collect_named_series(
            features_df,
            group=feature_prefix,
            segment=active_seg,
            band=band,
            stat_preference=stat_preference,
            scope_preference=scope_preference,
        )

        if not baseline_series.empty and not active_series.empty:
            valid_mask = baseline_series.notna() & active_series.notna()
            vals_baseline = baseline_series[valid_mask].values
            vals_active = active_series[valid_mask].values
            if len(vals_baseline) > 0:
                data_by_band[band] = (vals_baseline, vals_active)

    if not data_by_band:
        if logger:
            logger.warning(
                f"No {feature_label} data found for band × segment comparison plot"
            )
        return

    safe_name = feature_prefix.lower().replace(" ", "_")
    save_path = save_dir / f"sub-{subject}_{safe_name}_band_segment_condition"

    plot_paired_comparison(
        data_by_band=data_by_band,
        subject=subject,
        save_path=save_path,
        feature_label=feature_label,
        config=config,
        logger=logger,
        label1=baseline_seg.capitalize(),
        label2=active_seg.capitalize(),
    )




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
    """Temporal Evolution: Early vs Mid vs Late Active."""

    if features_df is None or features_df.empty or events_df is None:
        return

    comp = _get_comparison_masks(events_df, config)
    if comp is None:
        return
    mask1, mask2, label1, label2 = comp
    mask1 = np.asarray(mask1, dtype=bool)
    mask2 = np.asarray(mask2, dtype=bool)
    if int(mask1.sum()) == 0 or int(mask2.sum()) == 0:
        return

    from eeg_pipeline.utils.config.loader import get_config_value

    temporal_cfg = get_config_value(config, "plotting.plots.features.temporal", {})
    time_bins = temporal_cfg.get("time_bins", ["coarse_early", "coarse_mid", "coarse_late"])
    time_labels = temporal_cfg.get("time_labels", ["Early", "Mid", "Late"])

    available_segments = get_named_segments(features_df, group=feature_prefix)
    if available_segments:
        filtered_bins = [seg for seg in time_bins if seg in available_segments]
        if filtered_bins:
            time_bins = filtered_bins
        else:
            time_bins = available_segments
            time_labels = [seg.replace("_", " ").title() for seg in time_bins]

    bands, band_colors, condition_colors = _get_bands_and_palettes(config)
    color1 = condition_colors.get("nonpain", "#4C72B0")
    color2 = condition_colors.get("pain", "#C44E52")
    if {label1, label2} == {"Non-pain", "Pain"}:
        color1 = condition_colors["nonpain"] if label1 == "Non-pain" else condition_colors["pain"]
        color2 = condition_colors["pain"] if label2 == "Pain" else condition_colors["nonpain"]

    stat_preference = {
        "power": [
            "logratio",
            "logratio_mean",
            "baselined",
            "baselined_mean",
            "log10raw",
            "log10raw_mean",
            "mean",
            "val",
        ],
        "itpc": ["val"],
        "spectral": ["peak_freq", "peak_power", "center_freq", "bandwidth", "entropy"],
        "aperiodic": ["slope", "offset", "powcorr"],
        "ratios": ["power_ratio"],
        "asymmetry": ["index", "logdiff"],
        "comp": ["lzc", "pe"],
        "bursts": ["rate", "count", "duration_mean", "amp_mean", "fraction"],
    }.get(feature_prefix, [])
    scope_preference = ["global", "roi", "ch", "chpair"]

    plot_cfg = get_plot_config(config)
    fig, axes = plt.subplots(len(bands), 1, figsize=(8, 3 * len(bands)), sharex=True)

    for band_idx, band in enumerate(bands):
        ax = axes[band_idx]

        means1, means2 = [], []
        sems1, sems2 = [], []

        for time_bin in time_bins:
            series, _, _ = collect_named_series(
                features_df,
                group=feature_prefix,
                segment=time_bin,
                band=band,
                stat_preference=stat_preference,
                scope_preference=scope_preference,
            )
            if not series.empty:
                vals1 = series[mask1].dropna().values
                vals2 = series[mask2].dropna().values

                means1.append(np.mean(vals1) if len(vals1) > 0 else np.nan)
                means2.append(np.mean(vals2) if len(vals2) > 0 else np.nan)
                sems1.append(stats.sem(vals1) if len(vals1) > 1 else 0)
                sems2.append(stats.sem(vals2) if len(vals2) > 1 else 0)
            else:
                means1.append(np.nan)
                means2.append(np.nan)
                sems1.append(0)
                sems2.append(0)

        x = np.arange(len(time_bins))

        ax.errorbar(
            x - 0.1,
            means1,
            yerr=sems1,
            fmt="o-",
            color=color1,
            label=str(label1),
            capsize=3,
            markersize=8,
        )
        ax.errorbar(
            x + 0.1,
            means2,
            yerr=sems2,
            fmt="o-",
            color=color2,
            label=str(label2),
            capsize=3,
            markersize=8,
        )

        ax.set_xticks(x)
        ax.set_xticklabels(time_labels)
        ax.set_ylabel(band.capitalize(), fontweight="bold", color=band_colors.get(band) or "gray")
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
    "plot_complexity_by_roi_band_condition",
    "plot_aperiodic_by_roi_condition",
    "plot_connectivity_by_roi_band_condition",
    "plot_itpc_by_roi_band_condition",
    "plot_pac_by_roi_condition",
    "plot_band_segment_condition",
    "plot_temporal_evolution",
]
