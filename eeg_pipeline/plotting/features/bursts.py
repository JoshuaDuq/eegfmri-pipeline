"""
Burst Feature Visualization
===========================

Plots for oscillatory burst metrics across bands and conditions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from eeg_pipeline.plotting.config import get_plot_config
from eeg_pipeline.plotting.io.figures import save_fig, log_if_present
from eeg_pipeline.plotting.features.utils import (
    get_named_segments,
    get_named_bands,
    get_band_names,
    collect_named_series,
    get_band_color,
    plot_paired_comparison,
    compute_or_load_column_stats,
)
from eeg_pipeline.utils.config.loader import get_config_value, require_config_value
from eeg_pipeline.domain.features.naming import NamingSchema


# Plotting constants
_VIOLIN_WIDTH = 0.7
_VIOLIN_COLOR = "#D97706"
_VIOLIN_ALPHA = 0.6
_SCATTER_JITTER_RANGE = 0.08
_SCATTER_ALPHA = 0.3
_SCATTER_SIZE = 6
_BOX_WIDTH = 0.4
_Y_PADDING_FACTOR = 0.1
_Y_TOP_PADDING_FACTOR = 0.3
_FIGURE_WIDTH_PER_BAND = 1.2
_FIGURE_HEIGHT_PER_METRIC = 3.0
_MIN_FIGURE_WIDTH = 8.0
_MIN_FIGURE_HEIGHT = 4.5
_COLUMN_FIGURE_WIDTH_PER_BAND = 3.0
_COLUMN_FIGURE_HEIGHT = 5.0

# Color constants
_COLUMN_COMPARISON_COLOR_COND1 = "#5a7d9a"
_COLUMN_COMPARISON_COLOR_COND2 = "#c44e52"
_SIGNIFICANT_COLOR = "#d62728"
_NON_SIGNIFICANT_COLOR = "#333333"


def _select_segment(segments: List[str], preferred: str = "active") -> Optional[str]:
    """Select segment from list, preferring specified segment."""
    if not segments:
        return None
    if preferred in segments:
        return preferred
    return segments[0]


def _normalize_roi_name(name: str) -> str:
    """Normalize ROI name for comparison by removing separators."""
    return name.lower().replace("_", "").replace("-", "")


def _match_roi_name(candidate: str, target: str) -> bool:
    """Check if candidate ROI name matches target after normalization."""
    return _normalize_roi_name(candidate) == _normalize_roi_name(target)


def _format_metric_label(metric: str) -> str:
    """Format metric name for display (replace underscores, title case)."""
    return metric.replace("_", " ").title()


def _format_roi_display_name(roi_name: str) -> str:
    """Format ROI name for display."""
    if roi_name.lower() == "all":
        return "All Channels"
    return roi_name.replace("_", " ").title()


def _format_roi_filename_suffix(roi_name: str) -> str:
    """Format ROI name for filename suffix."""
    if roi_name.lower() == "all":
        return ""
    return f"_roi-{roi_name.replace(' ', '_').lower()}"


def _get_burst_columns(
    features_df: pd.DataFrame,
    segment: str,
    band: str,
    metric: str,
    roi_name: str,
) -> List[str]:
    """Get burst columns filtered by segment, band, metric, and ROI."""
    matching_columns = []
    
    for col in features_df.columns:
        parsed = NamingSchema.parse(str(col))
        if not parsed.get("valid") or parsed.get("group") != "bursts":
            continue
        if str(parsed.get("segment") or "") != segment:
            continue
        if str(parsed.get("band") or "") != band:
            continue
        if str(parsed.get("stat") or "") != metric:
            continue
        
        scope = parsed.get("scope") or ""
        if roi_name == "all":
            if scope == "global":
                matching_columns.append(col)
        else:
            if scope == "roi":
                roi_id = str(parsed.get("identifier") or "")
                if _match_roi_name(roi_id, roi_name):
                    matching_columns.append(col)
    
    return matching_columns


def _extract_roi_names_from_config(
    config: Any,
    config_roi_names: List[str],
) -> List[str]:
    """Extract and validate ROI names from configuration."""
    comp_rois = get_config_value(config, "plotting.comparisons.comparison_rois", [])
    
    if not comp_rois:
        roi_names = ["all"]
        roi_names.extend(config_roi_names)
        return roi_names
    
    roi_names = []
    for requested_roi in comp_rois:
        if requested_roi.lower() == "all":
            if "all" not in roi_names:
                roi_names.append("all")
        else:
            for config_roi in config_roi_names:
                if _match_roi_name(requested_roi, config_roi):
                    roi_names.append(config_roi)
                    break
    
    return roi_names if roi_names else ["all"]


def _detect_bands_from_data(features_df: pd.DataFrame) -> List[str]:
    """Detect frequency bands from burst feature columns."""
    detected_bands = set()
    for col in features_df.columns:
        parsed = NamingSchema.parse(str(col))
        if parsed.get("valid") and parsed.get("group") == "bursts":
            band = parsed.get("band")
            if band:
                detected_bands.add(str(band))
    return sorted(list(detected_bands))


def _get_comparison_segments(
    features_df: pd.DataFrame,
    config: Any,
    logger: Any,
) -> List[str]:
    """Get segments for comparison from config (no auto-detection)."""
    segments = require_config_value(config, "plotting.comparisons.comparison_windows")
    if not isinstance(segments, (list, tuple)) or len(segments) < 2:
        raise ValueError(
            "plotting.comparisons.comparison_windows must be a list/tuple with at least 2 window names "
            f"(got {segments!r})"
        )
    return [str(s) for s in segments]


def _get_burst_metrics(config: Any) -> List[str]:
    """Get burst metrics from config with fallback defaults."""
    metrics = get_config_value(
        config,
        "plotting.plots.features.bursts.comparison_metrics",
        None,
    ) or get_config_value(
        config,
        "plotting.plots.features.bursts.metrics",
        ["rate", "duration_mean", "amp_mean", "fraction"],
    )
    return list(metrics) if metrics else ["rate"]


def _prepare_window_comparison_data(
    features_df: pd.DataFrame,
    segments: List[str],
    bands: List[str],
    metric: str,
    roi_name: str,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Prepare data for window comparison (paired, 2 segments)."""
    data_by_band = {}
    seg1, seg2 = segments[0], segments[1]
    
    for band in bands:
        cols1 = _get_burst_columns(features_df, seg1, band, metric, roi_name)
        cols2 = _get_burst_columns(features_df, seg2, band, metric, roi_name)
        
        if not cols1 or not cols2:
            continue
        
        series1 = features_df[cols1].apply(pd.to_numeric, errors="coerce").mean(axis=1)
        series2 = features_df[cols2].apply(pd.to_numeric, errors="coerce").mean(axis=1)
        
        valid_mask = series1.notna() & series2.notna()
        values1 = series1[valid_mask].values
        values2 = series2[valid_mask].values
        
        if len(values1) > 0:
            data_by_band[band] = (values1, values2)
    
    return data_by_band


def _prepare_multi_window_comparison_data(
    features_df: pd.DataFrame,
    segments: List[str],
    bands: List[str],
    metric: str,
    roi_name: str,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Prepare data for multi-window comparison (3+ segments)."""
    data_by_band: Dict[str, Dict[str, np.ndarray]] = {}
    
    for band in bands:
        segment_series = {}
        for seg in segments:
            cols = _get_burst_columns(features_df, seg, band, metric, roi_name)
            if cols:
                segment_series[seg] = features_df[cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
        
        if len(segment_series) < 2:
            continue
        
        valid_mask = pd.Series(True, index=features_df.index)
        for series in segment_series.values():
            valid_mask &= series.notna()
        
        segment_values = {}
        for seg, series in segment_series.items():
            vals = series[valid_mask].values
            if len(vals) > 0:
                segment_values[seg] = vals
        
        if len(segment_values) >= 2:
            data_by_band[band] = segment_values
    
    return data_by_band


def _create_column_comparison_plot(
    cell_data: Dict[int, Optional[Dict[str, np.ndarray]]],
    qvalues: Dict[int, Tuple[float, float, float, bool]],
    bands: List[str],
    labels: Tuple[str, str],
    metric_label: str,
    subject: str,
    roi_display: str,
    n_trials: int,
    n_significant: int,
    n_tests: int,
    use_precomputed: bool,
    config: Any,
) -> plt.Figure:
    """Create column comparison plot with box plots per band."""
    plot_cfg = get_plot_config(config)
    n_bands = len(bands)
    label1, label2 = labels
    
    segment_colors = {
        "v1": _COLUMN_COMPARISON_COLOR_COND1,
        "v2": _COLUMN_COMPARISON_COLOR_COND2,
    }
    band_colors = {band: get_band_color(band, config) for band in bands}
    
    fig, axes = plt.subplots(
        1, n_bands,
        figsize=(_COLUMN_FIGURE_WIDTH_PER_BAND * n_bands, _COLUMN_FIGURE_HEIGHT),
        squeeze=False,
    )
    axes = axes.flatten()
    
    for col_idx, band in enumerate(bands):
        ax = axes[col_idx]
        data = cell_data.get(col_idx)
        
        if data is None or len(data.get("v1", [])) == 0 or len(data.get("v2", [])) == 0:
            ax.text(
                0.5, 0.5, "No data",
                ha="center", va="center",
                transform=ax.transAxes,
                fontsize=plot_cfg.font.title,
                color="gray",
            )
            ax.set_xticks([])
            continue
        
        values1, values2 = data["v1"], data["v2"]
        
        # Create box plot
        box_plot = ax.boxplot(
            [values1, values2],
            positions=[0, 1],
            widths=_BOX_WIDTH,
            patch_artist=True,
        )
        box_plot["boxes"][0].set_facecolor(segment_colors["v1"])
        box_plot["boxes"][0].set_alpha(_VIOLIN_ALPHA)
        box_plot["boxes"][1].set_facecolor(segment_colors["v2"])
        box_plot["boxes"][1].set_alpha(_VIOLIN_ALPHA)
        
        # Add scatter points with jitter
        jitter1 = np.random.uniform(-_SCATTER_JITTER_RANGE, _SCATTER_JITTER_RANGE, len(values1))
        jitter2 = np.random.uniform(-_SCATTER_JITTER_RANGE, _SCATTER_JITTER_RANGE, len(values2))
        ax.scatter(jitter1, values1, c=segment_colors["v1"], alpha=_SCATTER_ALPHA, s=_SCATTER_SIZE)
        ax.scatter(1 + jitter2, values2, c=segment_colors["v2"], alpha=_SCATTER_ALPHA, s=_SCATTER_SIZE)
        
        # Set y-axis limits with padding
        all_values = np.concatenate([values1, values2])
        y_min, y_max = np.nanmin(all_values), np.nanmax(all_values)
        y_range = y_max - y_min if y_max > y_min else 0.1
        ax.set_ylim(
            y_min - _Y_PADDING_FACTOR * y_range,
            y_max + _Y_TOP_PADDING_FACTOR * y_range,
        )
        
        # Add statistics annotation
        if col_idx in qvalues:
            _, q_value, effect_size, is_significant = qvalues[col_idx]
            significance_marker = "†" if is_significant else ""
            annotation_color = _SIGNIFICANT_COLOR if is_significant else _NON_SIGNIFICANT_COLOR
            annotation_text = f"q={q_value:.3f}{significance_marker}\nd={effect_size:.2f}"
            annotation_y = y_max + _Y_PADDING_FACTOR * y_range
            
            ax.annotate(
                annotation_text,
                xy=(0.5, annotation_y),
                ha="center",
                fontsize=plot_cfg.font.medium,
                color=annotation_color,
                fontweight="bold" if is_significant else "normal",
            )
        
        ax.set_xticks([0, 1])
        ax.set_xticklabels([label1, label2], fontsize=9)
        ax.set_title(
            band.capitalize(),
            fontweight="bold",
            color=band_colors.get(band, "#7C3AED"),
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    
    # Set figure title
    stats_source = "pre-computed" if use_precomputed else "Mann-Whitney U"
    title = (
        f"Bursts ({metric_label}): {label1} vs {label2} (Column Comparison)\n"
        f"Subject: {subject} | ROI: {roi_display} | N: {n_trials} trials | "
        f"{stats_source} | FDR: {n_significant}/{n_tests} significant (†=q<0.05)"
    )
    fig.suptitle(
        title,
        fontsize=plot_cfg.font.suptitle,
        fontweight="bold",
        y=1.02,
    )
    
    plt.tight_layout()
    return fig


def _plot_window_comparison(
    features_df: pd.DataFrame,
    segments: List[str],
    bands: List[str],
    metric: str,
    roi_names: List[str],
    subject: str,
    save_dir: Path,
    config: Any,
    logger: Any,
    stats_dir: Optional[Path],
) -> None:
    """Plot window comparison (paired) for burst metrics.
    
    Supports both 2-window comparison (simple paired) and multi-window comparison
    (3+ windows with all pairwise brackets and significance asterisks).
    """
    from eeg_pipeline.plotting.features.utils import plot_multi_window_comparison
    
    metric_label = _format_metric_label(metric)
    use_multi_window = len(segments) > 2
    
    for roi_name in roi_names:
        suffix = _format_roi_filename_suffix(roi_name)
        
        if use_multi_window:
            data_by_band_multi = _prepare_multi_window_comparison_data(
                features_df, segments, bands, metric, roi_name,
            )
            
            if data_by_band_multi:
                save_path = save_dir / f"sub-{subject}_bursts_{metric}_by_condition{suffix}_multiwindow"
                plot_multi_window_comparison(
                    data_by_band=data_by_band_multi,
                    subject=subject,
                    save_path=save_path,
                    feature_label=f"Bursts ({metric_label})",
                    segments=segments,
                    config=config,
                    logger=logger,
                    roi_name=roi_name,
                    stats_dir=stats_dir,
                )
        else:
            seg1, seg2 = segments[0], segments[1]
            data_by_band = _prepare_window_comparison_data(
                features_df, segments, bands, metric, roi_name,
            )
            
            if data_by_band:
                save_path = save_dir / f"sub-{subject}_bursts_{metric}_by_condition{suffix}_window"
                plot_paired_comparison(
                    data_by_band=data_by_band,
                    subject=subject,
                    save_path=save_path,
                    feature_label=f"Bursts ({metric_label})",
                    config=config,
                    logger=logger,
                    label1=seg1.capitalize(),
                    label2=seg2.capitalize(),
                    roi_name=roi_name,
                    stats_dir=stats_dir,
                )


def _plot_column_comparison(
    features_df: pd.DataFrame,
    events_df: pd.DataFrame,
    bands: List[str],
    metric: str,
    roi_names: List[str],
    subject: str,
    save_dir: Path,
    config: Any,
    logger: Any,
    stats_dir: Optional[Path],
) -> None:
    """Plot column comparison (unpaired) for burst metrics.
    
    Supports both 2-group comparison (simple unpaired) and multi-group comparison
    (3+ groups with all pairwise brackets and significance asterisks).
    """
    from eeg_pipeline.utils.analysis.events import extract_comparison_mask, extract_multi_group_masks
    from eeg_pipeline.plotting.features.utils import plot_multi_group_column_comparison
    
    values_spec = get_config_value(config, "plotting.comparisons.comparison_values", [])
    use_multi_group = isinstance(values_spec, (list, tuple)) and len(values_spec) > 2
    
    if use_multi_group:
        multi_group_info = extract_multi_group_masks(events_df, config, require_enabled=True)
        if not multi_group_info:
            raise ValueError("Multi-group column comparison requested but could not resolve group masks.")
        
        masks_dict, group_labels = multi_group_info
        segment_name = str(require_config_value(config, "plotting.comparisons.comparison_segment")).strip()
        metric_label = _format_metric_label(metric)
        
        from eeg_pipeline.plotting.features.utils import load_multigroup_stats
        multigroup_stats = load_multigroup_stats(stats_dir) if stats_dir else None
        
        for roi_name in roi_names:
            data_by_band: Dict[str, Dict[str, np.ndarray]] = {}
            for band in bands:
                cols = _get_burst_columns(features_df, segment_name, band, metric, roi_name)
                if not cols:
                    continue
                
                val_series = features_df[cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
                
                group_values = {}
                for label, mask in masks_dict.items():
                    vals = val_series[mask].dropna().values
                    if len(vals) > 0:
                        group_values[label] = vals
                
                if len(group_values) >= 2:
                    data_by_band[band] = group_values
            
            if data_by_band:
                suffix = _format_roi_filename_suffix(roi_name)
                save_path = save_dir / f"sub-{subject}_bursts_{metric}_by_condition{suffix}_multigroup"
                
                plot_multi_group_column_comparison(
                    data_by_band=data_by_band,
                    subject=subject,
                    save_path=save_path,
                    feature_label=f"Bursts ({metric_label})",
                    groups=group_labels,
                    config=config,
                    logger=logger,
                    roi_name=roi_name,
                    stats_dir=stats_dir,
                    multigroup_stats=multigroup_stats,
                )
        
        log_if_present(logger, "info", f"Saved bursts multi-group column comparison for {len(roi_names)} ROIs")
        return
    
    comp_mask_info = extract_comparison_mask(events_df, config, require_enabled=True)
    if not comp_mask_info:
        raise ValueError("Column comparison requested but could not resolve comparison masks.")
    
    mask1, mask2, label1, label2 = comp_mask_info
    segment_name = str(require_config_value(config, "plotting.comparisons.comparison_segment")).strip()
    metric_label = _format_metric_label(metric)
    plot_cfg = get_plot_config(config)
    n_trials = len(features_df)
    
    for roi_name in roi_names:
        # Collect data for all bands
        cell_data = {}
        for col_idx, band in enumerate(bands):
            cols = _get_burst_columns(features_df, segment_name, band, metric, roi_name)
            
            if not cols:
                cell_data[col_idx] = None
                continue
            
            value_series = features_df[cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
            values1 = value_series[mask1].dropna().values
            values2 = value_series[mask2].dropna().values
            
            cell_data[col_idx] = {"v1": values1, "v2": values2}
        
        # Compute or load statistics
        qvalues, n_significant, use_precomputed = compute_or_load_column_stats(
            stats_dir=stats_dir,
            feature_type="bursts",
            feature_keys=bands,
            cell_data=cell_data,
            config=config,
            logger=logger,
        )
        
        # Create plot
        roi_display = _format_roi_display_name(roi_name)
        fig = _create_column_comparison_plot(
            cell_data=cell_data,
            qvalues=qvalues,
            bands=bands,
            labels=(label1, label2),
            metric_label=metric_label,
            subject=subject,
            roi_display=roi_display,
            n_trials=n_trials,
            n_significant=n_significant,
            n_tests=len(qvalues),
            use_precomputed=use_precomputed,
            config=config,
        )
        
        # Save figure
        suffix = _format_roi_filename_suffix(roi_name)
        filename = f"sub-{subject}_bursts_{metric}_by_condition{suffix}_column"
        save_fig(
            fig,
            save_dir / filename,
            formats=plot_cfg.formats,
            dpi=plot_cfg.dpi,
            bbox_inches=plot_cfg.bbox_inches,
            pad_inches=plot_cfg.pad_inches,
            config=config,
        )
        plt.close(fig)


def plot_bursts_by_condition(
    features_df: pd.DataFrame,
    events_df: pd.DataFrame,
    subject: str,
    save_dir: Path,
    logger: Any = None,
    config: Any = None,
    stats_dir: Optional[Path] = None,
) -> None:
    """Compare burst metrics between conditions across bands.
    
    For window comparisons (paired): Uses the unified plot_paired_comparison helper.
    For column comparisons (unpaired): Uses Mann-Whitney U test with consistent styling.
    Creates one figure per ROI per metric.
    
    If stats_dir is provided, uses pre-computed statistics from the behavior pipeline.
    """
    from eeg_pipeline.infra.paths import ensure_dir
    from eeg_pipeline.plotting.features.roi import get_roi_definitions
    
    if features_df is None or features_df.empty or events_df is None:
        return

    compare_windows = get_config_value(
        config,
        "plotting.comparisons.compare_windows",
        True,
    )
    compare_columns = get_config_value(
        config,
        "plotting.comparisons.compare_columns",
        False,
    )
    
    segments = _get_comparison_segments(features_df, config, logger)
    bands = get_band_names(config) or _detect_bands_from_data(features_df)
    
    if not bands:
        return
    
    metrics = _get_burst_metrics(config)
    rois = get_roi_definitions(config)
    config_roi_names = list(rois.keys()) if rois else []
    roi_names = _extract_roi_names_from_config(config, config_roi_names)
    
    if logger:
        log_if_present(
            logger,
            "info",
            f"Burst comparison: segments={segments}, ROIs={roi_names}, "
            f"metrics={metrics}, compare_windows={compare_windows}, "
            f"compare_columns={compare_columns}",
        )
    
    ensure_dir(save_dir)
    
    for metric in metrics:
        if compare_windows and len(segments) >= 2:
            _plot_window_comparison(
                features_df=features_df,
                segments=segments,
                bands=bands,
                metric=metric,
                roi_names=roi_names,
                subject=subject,
                save_dir=save_dir,
                config=config,
                logger=logger,
                stats_dir=stats_dir,
            )
        
        if compare_columns:
            _plot_column_comparison(
                features_df=features_df,
                events_df=events_df,
                bands=bands,
                metric=metric,
                roi_names=roi_names,
                subject=subject,
                save_dir=save_dir,
                config=config,
                logger=logger,
                stats_dir=stats_dir,
            )


__all__ = [
    "plot_bursts_by_condition",
]
