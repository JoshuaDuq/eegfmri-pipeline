from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from eeg_pipeline.domain.features.naming import NamingSchema
from eeg_pipeline.utils.analysis.events import extract_comparison_mask
from eeg_pipeline.utils.config.loader import get_config_value
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
    safe_mannwhitneyu,
    compute_cohens_d,
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


###################################################################
# CONSTANTS
###################################################################

_MIN_SAMPLES_FOR_TEST = 3
_BOXPLOT_WIDTH = 0.4
_SCATTER_JITTER_RANGE = 0.08
_BOX_ALPHA = 0.6
_SCATTER_ALPHA = 0.3
_SCATTER_SIZE = 6
_Y_RANGE_PADDING_BOTTOM = 0.1
_Y_RANGE_PADDING_TOP = 0.25
_Y_ANNOTATION_OFFSET = 0.05


###################################################################
# HELPER FUNCTIONS
###################################################################


def _validate_dataframes(
    features_df: pd.DataFrame,
    events_df: pd.DataFrame,
    logger: Optional[logging.Logger] = None,
) -> bool:
    """Validate that dataframes are valid and aligned.
    
    Returns:
        True if valid, False otherwise
    """
    if features_df is None or features_df.empty or events_df is None:
        return False
    if len(features_df) != len(events_df):
        if logger:
            logger.warning(
                "Dataframe length mismatch: %d feature rows vs %d events",
                len(features_df),
                len(events_df),
            )
        return False
    return True


def _get_comparison_masks(events_df: pd.DataFrame, config: Any) -> Optional[tuple[np.ndarray, np.ndarray, str, str]]:
    """Extract comparison masks from events dataframe.
    
    Returns:
        Tuple of (mask1, mask2, label1, label2) or None if invalid
    """
    if events_df is None or events_df.empty:
        return None
    return extract_comparison_mask(events_df, config, require_enabled=False)


def _validate_masks(mask1: np.ndarray, mask2: np.ndarray) -> bool:
    """Validate that comparison masks have sufficient samples.
    
    Returns:
        True if both masks have at least one sample, False otherwise
    """
    mask1 = np.asarray(mask1, dtype=bool)
    mask2 = np.asarray(mask2, dtype=bool)
    return int(mask1.sum()) > 0 and int(mask2.sum()) > 0


def _get_mask_counts(mask1: np.ndarray, mask2: np.ndarray) -> Tuple[int, int]:
    """Convert masks to bool arrays and return sample counts.
    
    Returns:
        Tuple of (count1, count2)
    """
    mask1_bool = np.asarray(mask1, dtype=bool)
    mask2_bool = np.asarray(mask2, dtype=bool)
    return int(mask1_bool.sum()), int(mask2_bool.sum())


def _compute_statistics(
    vals_1: np.ndarray,
    vals_2: np.ndarray,
    min_samples: int = _MIN_SAMPLES_FOR_TEST,
) -> Optional[Tuple[float, float]]:
    """Compute Mann-Whitney U test and Cohen's d.
    
    Args:
        vals_1: First group values
        vals_2: Second group values
        min_samples: Minimum samples required for test
    
    Returns:
        Tuple of (p_value, cohens_d) or None if insufficient samples
    """
    if len(vals_1) < min_samples or len(vals_2) < min_samples:
        return None
    
    _, p_value = safe_mannwhitneyu(vals_1, vals_2, min_n=min_samples)
    cohens_d = compute_cohens_d(vals_1, vals_2)
    
    if not np.isfinite(p_value) or not np.isfinite(cohens_d):
        return None
    
    return float(p_value), float(cohens_d)


def _format_roi_name(roi_name: str) -> str:
    """Format ROI name for display (shorten common terms).
    
    Args:
        roi_name: Original ROI name
    
    Returns:
        Formatted name with line breaks and abbreviations
    """
    formatted = roi_name.replace("_", "\n")
    formatted = formatted.replace("Contra", "C")
    formatted = formatted.replace("Ipsi", "I")
    return formatted


def _plot_boxplot_with_scatter(
    ax: plt.Axes,
    vals_1: np.ndarray,
    vals_2: np.ndarray,
    color1: str,
    color2: str,
    label1: str,
    label2: str,
    plot_cfg: Optional[Any] = None,
) -> None:
    """Plot boxplot with overlaid scatter points for two groups.
    
    Args:
        ax: Matplotlib axes
        vals_1: First group values
        vals_2: Second group values
        color1: Color for first group
        color2: Color for second group
        label1: Label for first group
        label2: Label for second group
        plot_cfg: Plot configuration object (optional, for fontsize)
    """
    bp = ax.boxplot(
        [vals_1, vals_2],
        positions=[0, 1],
        widths=_BOXPLOT_WIDTH,
        patch_artist=True,
    )
    bp["boxes"][0].set_facecolor(color1)
    bp["boxes"][0].set_alpha(_BOX_ALPHA)
    bp["boxes"][1].set_facecolor(color2)
    bp["boxes"][1].set_alpha(_BOX_ALPHA)
    
    jitter_1 = np.random.uniform(-_SCATTER_JITTER_RANGE, _SCATTER_JITTER_RANGE, len(vals_1))
    jitter_2 = np.random.uniform(-_SCATTER_JITTER_RANGE, _SCATTER_JITTER_RANGE, len(vals_2))
    
    ax.scatter(jitter_1, vals_1, c=color1, alpha=_SCATTER_ALPHA, s=_SCATTER_SIZE)
    ax.scatter(1 + jitter_2, vals_2, c=color2, alpha=_SCATTER_ALPHA, s=_SCATTER_SIZE)
    
    fontsize = plot_cfg.font.small if plot_cfg else None
    ax.set_xticks([0, 1])
    ax.set_xticklabels([label1, label2], fontsize=fontsize)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _add_statistics_annotation(
    ax: plt.Axes,
    q_value: float,
    cohens_d: float,
    is_significant: bool,
    plot_cfg: Any,
    config: Optional[Any] = None,
    y_position: Optional[float] = None,
) -> None:
    """Add statistics annotation to plot.
    
    Args:
        ax: Matplotlib axes
        q_value: FDR-corrected q-value
        cohens_d: Cohen's d effect size
        is_significant: Whether result is significant
        plot_cfg: Plot configuration object
        config: Config object for color lookup (optional)
        y_position: Y position for annotation (if None, uses transform)
    """
    sig_marker = "†" if is_significant else ""
    annotation_text = f"q={q_value:.3f}{sig_marker}\nd={cohens_d:.2f}"
    sig_color = get_significance_color(is_significant, config)
    
    if y_position is not None:
        ax.annotate(
            annotation_text,
            xy=(0.5, y_position),
            ha="center",
            fontsize=plot_cfg.font.annotation,
            color=sig_color,
            fontweight="bold" if is_significant else "normal",
        )
    else:
        ax.text(
            0.5,
            0.95,
            annotation_text,
            transform=ax.transAxes,
            ha="center",
            fontsize=plot_cfg.font.annotation,
            va="top",
        )


def _get_active_window_label(config: Any) -> str:
    """Get formatted active window label from config.
    
    Returns:
        Formatted string like "3.0-10.5s"
    """
    active_window = get_config_value(config, "time_frequency_analysis.active_window", [3.0, 10.5])
    return f"{active_window[0]:.1f}-{active_window[1]:.1f}s"


def _filter_columns_by_criteria(
    columns: List[str],
    *,
    group: Optional[str] = None,
    segment: Optional[str] = None,
    band: Optional[str] = None,
    scope: Optional[str] = None,
    stat: Optional[str] = None,
    identifiers: Optional[set] = None,
) -> List[str]:
    """Filter feature columns by naming schema criteria.
    
    Args:
        columns: List of column names to filter
        group: Feature group (e.g., "power", "itpc")
        segment: Time segment (e.g., "active", "baseline")
        band: Frequency band
        scope: Feature scope (e.g., "ch", "roi", "global")
        stat: Statistic type (e.g., "val", "mean")
        identifiers: Set of channel/ROI identifiers to match
    
    Returns:
        List of matching column names
    """
    matching_cols = []
    for col in columns:
        parsed = NamingSchema.parse(str(col))
        if not parsed.get("valid"):
            continue
        if group and parsed.get("group") != group:
            continue
        if segment and str(parsed.get("segment") or "") != str(segment):
            continue
        if band and str(parsed.get("band") or "") != str(band):
            continue
        if scope and str(parsed.get("scope") or "") != str(scope):
            continue
        if stat and str(parsed.get("stat") or "") != str(stat):
            continue
        if identifiers is not None:
            identifier = str(parsed.get("identifier") or "")
            if identifier and identifier not in identifiers:
                continue
        matching_cols.append(str(col))
    return matching_cols


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
    if not _validate_dataframes(features_df, events_df, logger):
        return

    comp = _get_comparison_masks(events_df, config)
    if comp is None:
        return
    mask1, mask2, label1, label2 = comp
    if not _validate_masks(mask1, mask2):
        return

    plot_cfg = get_plot_config(config)
    active_label = _get_active_window_label(config)

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
    data_rois = _get_named_identifiers(features_df, group="power", segment=segment, scope="roi")
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
    color1 = condition_colors.get("condition_1", plot_cfg.get_color("blue"))
    color2 = condition_colors.get("condition_2", plot_cfg.get_color("red"))
    tick1 = str(label1)
    tick2 = str(label2)
    n_rois = len(roi_names)
    n_bands = len(bands)

    plot_data: Dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    all_pvalues = []
    pvalue_keys = []

    for row_idx, roi_name in enumerate(roi_names):
        roi_patterns = rois[roi_name]
        roi_channels = get_roi_channels(roi_patterns, all_channels)
        roi_set = set(roi_channels)

        for col_idx, band in enumerate(bands):
            key = (row_idx, col_idx)

            cols = _filter_columns_by_criteria(
                features_df.columns,
                group="power",
                segment=segment,
                band=band,
                scope="ch",
                identifiers=roi_set,
            )
            
            roi_vals = (
                features_df[cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
                if cols
                else pd.Series([np.nan] * len(features_df), index=features_df.index)
            )

            vals_1 = roi_vals[mask1].dropna().values
            vals_2 = roi_vals[mask2].dropna().values
            plot_data[key] = (vals_1, vals_2)

            stats_result = _compute_statistics(vals_1, vals_2)
            if stats_result is not None:
                p_value, cohens_d = stats_result
                all_pvalues.append(p_value)
                pvalue_keys.append((key, p_value, cohens_d))

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
                _plot_boxplot_with_scatter(ax, vals_1, vals_2, color1, color2, tick1, tick2, plot_cfg)

                if key in qvalues:
                    _, q_value, cohens_d, is_sig = qvalues[key]
                    _add_statistics_annotation(ax, q_value, cohens_d, is_sig, plot_cfg, config)

            if row_idx == 0:
                ax.set_title(
                    band.capitalize(),
                    fontweight="bold",
                    color=band_colors.get(band) or "gray",
                    fontsize=plot_cfg.font.title,
                )
            if col_idx == 0:
                ax.set_ylabel(_format_roi_name(roi_name), fontsize=plot_cfg.font.medium)

    n_1, n_2 = _get_mask_counts(mask1, mask2)
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
    if not _validate_dataframes(features_df, events_df, logger):
        return

    comp = _get_comparison_masks(events_df, config)
    if comp is None:
        return
    mask1, mask2, label1, label2 = comp
    if not _validate_masks(mask1, mask2):
        return

    plot_cfg = get_plot_config(config)
    active_label = _get_active_window_label(config)

    rois = get_roi_definitions(config)
    if not rois:
        return

    all_channels = extract_channels_from_columns(list(features_df.columns))

    bands, band_colors, condition_colors = _get_bands_and_palettes(config)
    color1 = condition_colors.get("condition_1", plot_cfg.get_color("blue"))
    color2 = condition_colors.get("condition_2", plot_cfg.get_color("red"))
    tick1 = str(label1)
    tick2 = str(label2)
    roi_names = list(rois.keys())
    n_rois = len(roi_names)
    n_bands = len(bands)

    metric_labels = {
        "lzc": "Lempel-Ziv Complexity",
        "pe": "Permutation Entropy",
    }
    metric_label = metric_labels.get(metric, metric.upper())

    segments = get_named_segments(features_df, group="comp")
    segment = "active" if "active" in segments else (segments[0] if segments else "active")

    plot_data: Dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    all_pvalues = []
    pvalue_keys = []

    for row_idx, roi_name in enumerate(roi_names):
        roi_patterns = rois[roi_name]
        roi_channels = get_roi_channels(roi_patterns, all_channels)

        for col_idx, band in enumerate(bands):
            key = (row_idx, col_idx)

            cols = _filter_columns_by_criteria(
                features_df.columns,
                group="comp",
                segment=segment,
                band=band,
                scope="ch",
                stat=metric,
                identifiers=set(roi_channels),
            )

            roi_vals = (
                features_df[cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
                if cols
                else pd.Series([np.nan] * len(features_df), index=features_df.index)
            )

            vals_1 = roi_vals[mask1].dropna().values
            vals_2 = roi_vals[mask2].dropna().values
            plot_data[key] = (vals_1, vals_2)

            stats_result = _compute_statistics(vals_1, vals_2)
            if stats_result is not None:
                p_value, cohens_d = stats_result
                all_pvalues.append(p_value)
                pvalue_keys.append((key, p_value, cohens_d))

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
                _plot_boxplot_with_scatter(ax, vals_1, vals_2, color1, color2, tick1, tick2, plot_cfg)

                if key in qvalues:
                    _, q_value, cohens_d, is_sig = qvalues[key]
                    _add_statistics_annotation(ax, q_value, cohens_d, is_sig, plot_cfg, config)

            if row_idx == 0:
                ax.set_title(
                    band.capitalize(),
                    fontweight="bold",
                    color=band_colors.get(band) or "gray",
                    fontsize=plot_cfg.font.title,
                )
            if col_idx == 0:
                ax.set_ylabel(_format_roi_name(roi_name), fontsize=plot_cfg.font.medium)

    n_1, n_2 = _get_mask_counts(mask1, mask2)
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
    if not _validate_dataframes(features_df, events_df, logger):
        return

    comp = _get_comparison_masks(events_df, config)
    if comp is None:
        return
    mask1, mask2, label1, label2 = comp
    if not _validate_masks(mask1, mask2):
        return

    active_label = _get_active_window_label(config)

    rois = get_roi_definitions(config)
    if not rois:
        return

    all_channels = extract_channels_from_columns(list(features_df.columns))

    roi_names = list(rois.keys())
    n_rois = len(roi_names)
    metrics = ["slope", "offset"]
    n_metrics = len(metrics)

    condition_colors = get_condition_colors(config)
    color1 = condition_colors.get("condition_1", "#4C72B0")
    color2 = condition_colors.get("condition_2", "#C44E52")
    tick1 = str(label1)
    tick2 = str(label2)

    plot_data: Dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    all_pvalues = []
    pvalue_keys = []

    for row_idx, roi_name in enumerate(roi_names):
        roi_patterns = rois[roi_name]
        roi_channels = get_roi_channels(roi_patterns, all_channels)

        for col_idx, metric in enumerate(metrics):
            key = (row_idx, col_idx)

            cols = []
            pattern_prefix = f"aperiodic_active_broadband_ch_"
            pattern_suffix = f"_{metric}"
            for col in features_df.columns:
                if pattern_prefix in col and pattern_suffix in col:
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

            stats_result = _compute_statistics(vals_1, vals_2)
            if stats_result is not None:
                p_value, cohens_d = stats_result
                all_pvalues.append(p_value)
                pvalue_keys.append((key, p_value, cohens_d))

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
                _plot_boxplot_with_scatter(ax, vals_1, vals_2, color1, color2, tick1, tick2, plot_cfg)

                if key in qvalues:
                    _, q_value, cohens_d, is_sig = qvalues[key]
                    _add_statistics_annotation(ax, q_value, cohens_d, is_sig, plot_cfg, config)

            if row_idx == 0:
                ax.set_title(metric.capitalize(), fontweight="bold", fontsize=plot_cfg.font.title)
            if col_idx == 0:
                ax.set_ylabel(_format_roi_name(roi_name), fontsize=plot_cfg.font.medium)

    n_1, n_2 = _get_mask_counts(mask1, mask2)
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
    if not _validate_dataframes(features_df, events_df, logger):
        return

    comp = _get_comparison_masks(events_df, config)
    if comp is None:
        return
    mask1, mask2, label1, label2 = comp
    if not _validate_masks(mask1, mask2):
        return

    active_label = _get_active_window_label(config)

    rois = get_roi_definitions(config)
    if not rois:
        return

    all_pairs = extract_channel_pairs_from_columns(list(features_df.columns))
    all_channels = list(set([p[0] for p in all_pairs] + [p[1] for p in all_pairs]))

    bands, band_colors, condition_colors = _get_bands_and_palettes(config)
    color1 = condition_colors.get("condition_1", "#4C72B0")
    color2 = condition_colors.get("condition_2", "#C44E52")
    tick1 = str(label1)
    tick2 = str(label2)
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
                _plot_boxplot_with_scatter(ax, vals_1, vals_2, color1, color2, tick1, tick2, plot_cfg)

                stats_result = _compute_statistics(vals_1, vals_2)
                if stats_result is not None:
                    p_value, cohens_d = stats_result
                    sig_marker = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                    ax.text(
                        0.5,
                        0.95,
                        f"p={p_value:.3f}{sig_marker}\nd={cohens_d:.2f}",
                        transform=ax.transAxes,
                        ha="center",
                        fontsize=plot_cfg.font.annotation,
                        va="top",
                    )

            if row_idx == 0:
                ax.set_title(
                    band.capitalize(),
                    fontweight="bold",
                    color=band_colors.get(band) or "gray",
                    fontsize=plot_cfg.font.title,
                )
            if col_idx == 0:
                ax.set_ylabel(_format_roi_name(roi_name), fontsize=plot_cfg.font.medium)

    n_1, n_2 = _get_mask_counts(mask1, mask2)
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
    if not _validate_dataframes(features_df, events_df, logger):
        return

    comp = _get_comparison_masks(events_df, config)
    if comp is None:
        return
    mask1, mask2, label1, label2 = comp
    if not _validate_masks(mask1, mask2):
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
    color1 = condition_colors.get("condition_1", "#4C72B0")
    color2 = condition_colors.get("condition_2", "#C44E52")
    tick1 = str(label1)
    tick2 = str(label2)
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

            stats_result = _compute_statistics(vals_1, vals_2)
            if stats_result is not None:
                p_value, cohens_d = stats_result
                all_pvalues.append(p_value)
                pvalue_keys.append((key, p_value, cohens_d))

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
                _plot_boxplot_with_scatter(ax, vals_1, vals_2, color1, color2, tick1, tick2, plot_cfg)

                all_vals = np.concatenate([vals_1, vals_2])
                ymin, ymax = np.nanmin(all_vals), np.nanmax(all_vals)
                yrange = ymax - ymin if ymax > ymin else 0.1
                ax.set_ylim(ymin - _Y_RANGE_PADDING_BOTTOM * yrange, ymax + _Y_RANGE_PADDING_TOP * yrange)

                if key in qvalues:
                    _, q_value, cohens_d, is_sig = qvalues[key]
                    y_position = ymax + _Y_ANNOTATION_OFFSET * yrange
                    _add_statistics_annotation(ax, q_value, cohens_d, is_sig, plot_cfg, config, y_position=y_position)

            if row_idx == 0:
                ax.set_title(
                    band.capitalize(),
                    fontweight="bold",
                    color=band_colors.get(band) or "gray",
                    fontsize=plot_cfg.font.title,
                )
            if col_idx == 0:
                ax.set_ylabel(_format_roi_name(roi_name), fontsize=plot_cfg.font.medium)

    n_1, n_2 = _get_mask_counts(mask1, mask2)
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
    if not _validate_dataframes(features_df, events_df, logger):
        return

    comp = _get_comparison_masks(events_df, config)
    if comp is None:
        return
    mask1, mask2, label1, label2 = comp
    if not _validate_masks(mask1, mask2):
        return

    segments = get_named_segments(features_df, group="pac")
    if not segments:
        return
    segment = "active" if "active" in segments else segments[0]

    pairs = get_named_bands(features_df, group="pac", segment=segment)
    if not pairs:
        return

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
    color1 = condition_colors.get("condition_1", "#4C72B0")
    color2 = condition_colors.get("condition_2", "#C44E52")
    tick1 = str(label1)
    tick2 = str(label2)
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

            stats_result = _compute_statistics(vals_1, vals_2)
            if stats_result is not None:
                p_value, cohens_d = stats_result
                all_pvalues.append(p_value)
                pvalue_keys.append((key, p_value, cohens_d))

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
                _plot_boxplot_with_scatter(ax, vals_1, vals_2, color1, color2, tick1, tick2, plot_cfg)

                if key in qvalues:
                    _, q_value, cohens_d, is_sig = qvalues[key]
                    _add_statistics_annotation(ax, q_value, cohens_d, is_sig, plot_cfg, config)

            if row_idx == 0:
                pair_label = pair.replace("_", "→")
                ax.set_title(
                    pair_label.capitalize(),
                    fontweight="bold",
                    color=pair_colors[col_idx],
                    fontsize=plot_cfg.font.title,
                )
            if col_idx == 0:
                ax.set_ylabel(_format_roi_name(roi_name), fontsize=plot_cfg.font.medium)

    n_1, n_2 = _get_mask_counts(mask1, mask2)
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
    from eeg_pipeline.plotting.features.utils import plot_paired_comparison

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
    color1 = condition_colors.get("condition_1", "#4C72B0")
    color2 = condition_colors.get("condition_2", "#C44E52")

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

    title = f"{feature_label} Temporal Evolution: {label2} vs {label1}\nSubject: {subject}"
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
