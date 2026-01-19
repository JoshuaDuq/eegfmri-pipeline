"""
Band Ratio Visualization
========================

Plots for band power ratios computed from precomputed data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from eeg_pipeline.domain.features.naming import NamingSchema
from eeg_pipeline.infra.paths import ensure_dir
from eeg_pipeline.plotting.config import get_plot_config
from eeg_pipeline.plotting.features.roi import get_roi_definitions
from eeg_pipeline.plotting.features.utils import (
    collect_named_series,
    compute_or_load_column_stats,
    get_named_bands,
    get_named_segments,
    plot_paired_comparison,
)
from eeg_pipeline.plotting.io.figures import log_if_present, save_fig
from eeg_pipeline.utils.analysis.events import extract_comparison_mask
from eeg_pipeline.utils.config.loader import get_config_value


def _select_preferred_segment(segments: List[str], preferred: str = "active") -> Optional[str]:
    """Select segment from list, preferring specified segment."""
    if not segments:
        return None
    if preferred in segments:
        return preferred
    return segments[0]


def _order_ratio_pairs_by_config(pairs: List[str], config: Any) -> List[str]:
    """Order ratio pairs according to config, preserving config order."""
    config_pairs = get_config_value(config, "feature_engineering.spectral.ratio_pairs", [])
    if not config_pairs:
        return pairs
    
    ordered = [
        f"{entry[0]}_{entry[1]}"
        for entry in config_pairs
        if isinstance(entry, (list, tuple)) and len(entry) >= 2
    ]
    
    if not ordered:
        return pairs
    
    config_ordered = [p for p in ordered if p in pairs]
    remaining = [p for p in pairs if p not in ordered]
    return config_ordered + remaining


def _extract_ratio_pairs_from_dataframe(features_df: pd.DataFrame) -> List[str]:
    """Extract unique ratio pair names from feature dataframe columns."""
    pairs = set()
    for column in features_df.columns:
        parsed = NamingSchema.parse(str(column))
        is_valid_ratio = parsed.get("valid") and parsed.get("group") == "ratios"
        if is_valid_ratio:
            band = parsed.get("band")
            if band:
                pairs.add(str(band))
    return sorted(list(pairs))


def _normalize_roi_identifier(name: str) -> str:
    """Normalize ROI identifier for comparison (case-insensitive, ignore separators)."""
    return name.lower().replace("_", "").replace("-", "")


def _get_ratio_columns_for_segment_pair_roi(
    features_df: pd.DataFrame,
    segment: str,
    pair: str,
    roi_name: str,
) -> List[str]:
    """Get ratio columns matching segment, pair, and ROI criteria."""
    matching_columns = []
    
    for column in features_df.columns:
        parsed = NamingSchema.parse(str(column))
        if not parsed.get("valid") or parsed.get("group") != "ratios":
            continue
        
        parsed_segment = str(parsed.get("segment") or "")
        parsed_band = str(parsed.get("band") or "")
        if parsed_segment != segment or parsed_band != pair:
            continue
        
        scope = parsed.get("scope") or ""
        
        if roi_name == "all":
            if scope == "global":
                matching_columns.append(column)
        elif scope == "roi":
            roi_identifier = str(parsed.get("identifier") or "")
            if _normalize_roi_identifier(roi_identifier) == _normalize_roi_identifier(roi_name):
                matching_columns.append(column)
    
    return matching_columns


def _normalize_roi_name_for_filename(roi_name: str) -> str:
    """Convert ROI name to safe filename suffix."""
    from eeg_pipeline.utils.formatting import sanitize_label
    if roi_name.lower() == "all":
        return ""
    return sanitize_label(roi_name).lower()


def _get_comparison_segments(config: Any, features_df: pd.DataFrame, logger: Any) -> List[str]:
    """Get segments for comparison from config or auto-detect from data."""
    config_segments = get_config_value(config, "plotting.comparisons.comparison_windows", [])
    if config_segments and len(config_segments) >= 2:
        return config_segments
    
    detected_segments = get_named_segments(features_df, group="ratios")
    if len(detected_segments) >= 2:
        selected = detected_segments[:2]
        log_if_present(logger, "info", f"Auto-detected segments for ratios comparison: {selected}")
        return selected
    
    return []


def _get_comparison_roi_names(config: Any) -> List[str]:
    """Get ROI names for comparison from config."""
    roi_definitions = get_roi_definitions(config)
    config_roi_names = list(roi_definitions.keys()) if roi_definitions else []
    
    configured_rois = get_config_value(config, "plotting.comparisons.comparison_rois", [])
    if not configured_rois:
        return ["all"] + config_roi_names
    
    roi_names = []
    for requested_roi in configured_rois:
        if requested_roi.lower() == "all":
            if "all" not in roi_names:
                roi_names.append("all")
        else:
            requested_normalized = _normalize_roi_identifier(requested_roi)
            for config_roi in config_roi_names:
                if _normalize_roi_identifier(config_roi) == requested_normalized:
                    roi_names.append(config_roi)
                    break
    
    return roi_names if roi_names else ["all"]


def plot_ratios_by_pair(
    features_df: pd.DataFrame,
    save_path: Path,
    *,
    config: Any = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> plt.Figure:
    """Plot distributions of power ratios by band-pair."""
    if features_df is None or features_df.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No ratio data", ha="center", va="center")
        return fig
    
    plot_cfg = get_plot_config(config)
    segments = get_named_segments(features_df, group="ratios")
    segment = _select_preferred_segment(segments, preferred="active")
    
    if segment is None:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No ratio data", ha="center", va="center")
        return fig
    
    pairs = get_named_bands(features_df, group="ratios", segment=segment)
    if not pairs:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No ratio data", ha="center", va="center")
        return fig
    
    pairs = _order_ratio_pairs_by_config(pairs, config)
    
    if figsize is None:
        width = max(8.0, len(pairs) * 1.2)
        figsize = (width, 5.0)
    
    fig, ax = plt.subplots(figsize=figsize)
    ratio_data = []
    positions = []
    
    for position, pair in enumerate(pairs):
        series, _, _ = collect_named_series(
            features_df,
            group="ratios",
            segment=segment,
            band=pair,
            stat_preference=["power_ratio"],
            scope_preference=["global", "roi", "ch"],
        )
        values = series.dropna().values
        if values.size == 0:
            continue
        ratio_data.append(values)
        positions.append(position)
    
    if ratio_data:
        violin_parts = ax.violinplot(ratio_data, positions=positions, showmedians=True, widths=0.7)
        for body in violin_parts.get("bodies", []):
            body.set_facecolor("#2563EB")
            body.set_alpha(0.6)
        
        pair_labels = [p.replace("_", "/") for p in pairs]
        ax.set_xticks(range(len(pairs)))
        ax.set_xticklabels(pair_labels, rotation=30, ha="right")
    else:
        ax.text(0.5, 0.5, "No ratio data", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
    
    ax.set_xlabel("Band Ratio")
    ax.set_ylabel("Power Ratio")
    segment_label = segment if segment is not None else "unknown"
    ax.set_title(f"Band Power Ratios ({segment_label})", fontweight="bold")
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


def _plot_window_comparison_ratios(
    features_df: pd.DataFrame,
    segments: List[str],
    pairs: List[str],
    roi_names: List[str],
    subject: str,
    save_dir: Path,
    config: Any,
    logger: Any,
    stats_dir: Optional[Path],
) -> None:
    """Plot paired window comparison for ratios."""
    segment1, segment2 = segments[0], segments[1]
    
    for roi_name in roi_names:
        data_by_band = {}
        for pair in pairs:
            columns1 = _get_ratio_columns_for_segment_pair_roi(features_df, segment1, pair, roi_name)
            columns2 = _get_ratio_columns_for_segment_pair_roi(features_df, segment2, pair, roi_name)
            
            if not columns1 or not columns2:
                continue
            
            series1 = features_df[columns1].apply(pd.to_numeric, errors="coerce").mean(axis=1)
            series2 = features_df[columns2].apply(pd.to_numeric, errors="coerce").mean(axis=1)
            
            valid_mask = series1.notna() & series2.notna()
            values1 = series1[valid_mask].values
            values2 = series2[valid_mask].values
            
            if len(values1) > 0:
                data_by_band[pair] = (values1, values2)
        
        if data_by_band:
            roi_suffix = _normalize_roi_name_for_filename(roi_name)
            filename_suffix = f"_roi-{roi_suffix}" if roi_suffix else ""
            save_path = save_dir / f"sub-{subject}_ratios_by_condition{filename_suffix}_window"
            
            plot_paired_comparison(
                data_by_band=data_by_band,
                subject=subject,
                save_path=save_path,
                feature_label="Band Ratios",
                config=config,
                logger=logger,
                label1=segment1.capitalize(),
                label2=segment2.capitalize(),
                roi_name=roi_name,
                stats_dir=stats_dir,
            )
    
    log_if_present(logger, "info", f"Saved ratios paired comparison plots for {len(roi_names)} ROIs")


def _plot_column_comparison_ratios(
    features_df: pd.DataFrame,
    events_df: pd.DataFrame,
    pairs: List[str],
    roi_names: List[str],
    subject: str,
    save_dir: Path,
    config: Any,
    logger: Any,
    stats_dir: Optional[Path],
) -> None:
    """Plot unpaired column comparison for ratios."""
    comparison_info = extract_comparison_mask(events_df, config, require_enabled=False)
    if not comparison_info:
        log_if_present(
            logger, "warning",
            "Column comparison enabled but config incomplete. "
            "Set plotting.comparisons.comparison_column and comparison_values."
        )
        return
    
    mask1, mask2, label1, label2 = comparison_info
    segment_name = get_config_value(config, "plotting.comparisons.comparison_segment", "active")
    plot_cfg = get_plot_config(config)
    
    condition_colors = {"v1": "#5a7d9a", "v2": "#c44e52"}
    n_pairs = len(pairs)
    n_trials = len(features_df)
    
    for roi_name in roi_names:
        cell_data = _collect_ratio_cell_data(
            features_df, pairs, segment_name, roi_name, mask1, mask2
        )
        
        qvalues, n_significant, use_precomputed = compute_or_load_column_stats(
            stats_dir=stats_dir,
            feature_type="ratios",
            feature_keys=pairs,
            cell_data=cell_data,
            config=config,
            logger=logger,
        )
        
        fig, axes = plt.subplots(1, n_pairs, figsize=(3 * n_pairs, 5), squeeze=False)
        
        for pair_idx, pair in enumerate(pairs):
            ax = axes.flatten()[pair_idx]
            data = cell_data.get(pair_idx)
            
            if not _has_valid_data(data):
                ax.text(
                    0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=plot_cfg.font.title, color="gray"
                )
                ax.set_xticks([])
                continue
            
            values1, values2 = data["v1"], data["v2"]
            _plot_ratio_boxplot_with_scatter(ax, values1, values2, condition_colors, plot_cfg)
            
            y_min, y_max = _compute_y_limits(values1, values2)
            ax.set_ylim(y_min, y_max)
            
            if pair_idx in qvalues:
                data_max = np.nanmax(np.concatenate([values1, values2]))
                y_range = y_max - y_min
                _annotate_statistics(ax, qvalues[pair_idx], data_max, y_range, plot_cfg)
            
            pair_label = pair.replace("_", "/").capitalize()
            ax.set_xticks([0, 1])
            ax.set_xticklabels([label1, label2], fontsize=9)
            ax.set_title(pair_label, fontweight="bold")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        
        _add_column_comparison_title(
            fig, subject, roi_name, label1, label2, n_trials,
            n_significant, len(qvalues), use_precomputed, plot_cfg
        )
        
        plt.tight_layout()
        
        roi_suffix = _normalize_roi_name_for_filename(roi_name)
        filename_suffix = f"_roi-{roi_suffix}" if roi_suffix else ""
        filename = f"sub-{subject}_ratios_by_condition{filename_suffix}_column"
        
        save_fig(
            fig, save_dir / filename,
            formats=plot_cfg.formats, dpi=plot_cfg.dpi,
            bbox_inches=plot_cfg.bbox_inches, pad_inches=plot_cfg.pad_inches
        )
        plt.close(fig)
    
    log_if_present(logger, "info", f"Saved ratios column comparison plots for {len(roi_names)} ROIs")


def _collect_ratio_cell_data(
    features_df: pd.DataFrame,
    pairs: List[str],
    segment: str,
    roi_name: str,
    mask1: pd.Series,
    mask2: pd.Series,
) -> dict:
    """Collect ratio data for each pair into cell_data dictionary."""
    cell_data = {}
    for pair_idx, pair in enumerate(pairs):
        columns = _get_ratio_columns_for_segment_pair_roi(features_df, segment, pair, roi_name)
        
        if not columns:
            cell_data[pair_idx] = None
            continue
        
        value_series = features_df[columns].apply(pd.to_numeric, errors="coerce").mean(axis=1)
        values1 = value_series[mask1].dropna().values
        values2 = value_series[mask2].dropna().values
        
        cell_data[pair_idx] = {"v1": values1, "v2": values2}
    
    return cell_data


def _has_valid_data(data: Optional[dict]) -> bool:
    """Check if cell data contains valid values."""
    if data is None:
        return False
    values1 = data.get("v1", [])
    values2 = data.get("v2", [])
    return len(values1) > 0 and len(values2) > 0


def _plot_ratio_boxplot_with_scatter(
    ax: plt.Axes,
    values1: np.ndarray,
    values2: np.ndarray,
    colors: dict,
    plot_cfg: Any,
) -> None:
    """Plot boxplot with overlaid scatter points for ratio comparison."""
    boxplot = ax.boxplot([values1, values2], positions=[0, 1], widths=0.4, patch_artist=True)
    boxplot["boxes"][0].set_facecolor(colors["v1"])
    boxplot["boxes"][0].set_alpha(0.6)
    boxplot["boxes"][1].set_facecolor(colors["v2"])
    boxplot["boxes"][1].set_alpha(0.6)
    
    jitter_range = 0.08
    jitter1 = np.random.uniform(-jitter_range, jitter_range, len(values1))
    jitter2 = np.random.uniform(-jitter_range, jitter_range, len(values2))
    ax.scatter(jitter1, values1, c=colors["v1"], alpha=0.3, s=6)
    ax.scatter(1 + jitter2, values2, c=colors["v2"], alpha=0.3, s=6)


def _compute_y_limits(values1: np.ndarray, values2: np.ndarray) -> Tuple[float, float]:
    """Compute y-axis limits with padding for ratio plot."""
    all_values = np.concatenate([values1, values2])
    y_min = np.nanmin(all_values)
    y_max = np.nanmax(all_values)
    y_range = y_max - y_min if y_max > y_min else 0.1
    return y_min - 0.1 * y_range, y_max + 0.3 * y_range


def _annotate_statistics(
    ax: plt.Axes,
    stats_tuple: Tuple,
    data_max: float,
    y_range: float,
    plot_cfg: Any,
) -> None:
    """Annotate statistical results on plot."""
    _, q_value, d_value, is_significant = stats_tuple
    significance_marker = "†" if is_significant else ""
    color = "#d62728" if is_significant else "#333333"
    fontweight = "bold" if is_significant else "normal"
    
    annotation_text = f"q={q_value:.3f}{significance_marker}\nd={d_value:.2f}"
    annotation_y = data_max + 0.05 * y_range
    ax.annotate(
        annotation_text,
        xy=(0.5, annotation_y),
        ha="center",
        fontsize=plot_cfg.font.medium,
        color=color,
        fontweight=fontweight,
    )


def _add_column_comparison_title(
    fig: plt.Figure,
    subject: str,
    roi_name: str,
    label1: str,
    label2: str,
    n_trials: int,
    n_significant: int,
    n_tests: int,
    use_precomputed: bool,
    plot_cfg: Any,
) -> None:
    """Add title to column comparison figure."""
    roi_display = roi_name.replace("_", " ").title() if roi_name.lower() != "all" else "All Channels"
    stats_source = "pre-computed" if use_precomputed else "Mann-Whitney U"
    title = (
        f"Ratios: {label1} vs {label2} (Column Comparison)\n"
        f"Subject: {subject} | ROI: {roi_display} | N: {n_trials} trials | {stats_source} | "
        f"FDR: {n_significant}/{n_tests} significant (†=q<0.05)"
    )
    fig.suptitle(title, fontsize=plot_cfg.font.suptitle, fontweight="bold", y=1.02)


def plot_ratios_by_condition(
    features_df: pd.DataFrame,
    events_df: pd.DataFrame,
    subject: str,
    save_dir: Path,
    logger: Any = None,
    config: Any = None,
    stats_dir: Optional[Path] = None,
) -> None:
    """Compare band power ratios between conditions.
    
    For window comparisons (paired): Uses the unified plot_paired_comparison helper.
    For column comparisons (unpaired): Uses Mann-Whitney U test with consistent styling.
    Creates one figure per ROI.
    
    If stats_dir is provided, uses pre-computed statistics from the behavior pipeline.
    """
    if features_df is None or features_df.empty or events_df is None:
        return
    
    compare_windows = get_config_value(config, "plotting.comparisons.compare_windows", True)
    compare_columns = get_config_value(config, "plotting.comparisons.compare_columns", False)
    
    segments = _get_comparison_segments(config, features_df, logger)
    pairs = _extract_ratio_pairs_from_dataframe(features_df)
    if not pairs:
        return
    
    pairs = _order_ratio_pairs_by_config(pairs, config)
    roi_names = _get_comparison_roi_names(config)
    
    log_if_present(
        logger, "info",
        f"Ratios comparison: segments={segments}, ROIs={roi_names}, "
        f"compare_windows={compare_windows}, compare_columns={compare_columns}"
    )
    
    ensure_dir(save_dir)
    
    if compare_windows and len(segments) >= 2:
        _plot_window_comparison_ratios(
            features_df, segments, pairs, roi_names, subject, save_dir,
            config, logger, stats_dir
        )
    
    if compare_columns:
        _plot_column_comparison_ratios(
            features_df, events_df, pairs, roi_names, subject, save_dir,
            config, logger, stats_dir
        )


__all__ = [
    "plot_ratios_by_pair",
    "plot_ratios_by_condition",
]
