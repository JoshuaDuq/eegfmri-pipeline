"""
Asymmetry Visualization
=======================

Plots for hemispheric asymmetry indices across bands.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from eeg_pipeline.domain.features.naming import NamingSchema
from eeg_pipeline.infra.paths import ensure_dir
from eeg_pipeline.plotting.config import get_plot_config
from eeg_pipeline.plotting.features.roi import get_roi_definitions, get_roi_channels
from eeg_pipeline.plotting.features.utils import (
    get_band_names,
    plot_paired_comparison,
    compute_or_load_column_stats,
)
from eeg_pipeline.plotting.io.figures import log_if_present, save_fig
from eeg_pipeline.utils.analysis.events import extract_comparison_mask
from eeg_pipeline.utils.config.loader import get_config_value, require_config_value


# Plotting constants
_VIOLIN_WIDTH = 0.7
_VIOLIN_ALPHA = 0.6
_VIOLIN_COLOR = "#0EA5E9"
_BOXPLOT_WIDTH = 0.4
_BOXPLOT_ALPHA = 0.6
_SCATTER_JITTER_RANGE = 0.08
_SCATTER_ALPHA = 0.3
_SCATTER_SIZE = 6
_Y_PADDING_LOWER = 0.1
_Y_PADDING_UPPER = 0.3
_ANNOTATION_Y_OFFSET = 0.05
_MIN_FIGURE_WIDTH = 8.0
_FIGURE_WIDTH_PER_BAND = 1.2
_FIGURE_HEIGHT = 5.0
_SUBPLOT_WIDTH_PER_BAND = 3.0

# Condition colors
_CONDITION_COLOR_1 = "#5a7d9a"
_CONDITION_COLOR_2 = "#c44e52"


def _sanitize_roi_name_for_path(roi_name: str) -> str:
    """Convert ROI name to filesystem-safe string."""
    from eeg_pipeline.utils.formatting import sanitize_label
    if roi_name.lower() == "all":
        return ""
    return sanitize_label(roi_name).lower()


def _roi_has_asymmetry_data(
    features_df: pd.DataFrame,
    roi_name: str,
    rois: Dict[str, List[str]],
) -> bool:
    """Check if ROI has any asymmetry channel pairs in the data.
    
    Asymmetry requires left-right hemispheric pairs. Midline ROIs and
    single-hemisphere ROIs without cross-hemispheric pairs return False.
    """
    if roi_name.lower() == "all":
        return True
    
    roi_patterns = rois.get(roi_name, [])
    if not roi_patterns:
        return False
    
    for col in features_df.columns:
        parsed = NamingSchema.parse(str(col))
        if not parsed.get("valid") or parsed.get("group") != "asymmetry":
            continue
        
        scope = parsed.get("scope") or ""
        if scope != "chpair":
            continue
        
        identifier = str(parsed.get("identifier") or "")
        if "-" not in identifier:
            continue
        
        channel1, channel2 = identifier.split("-", 1)
        roi_channels = get_roi_channels(roi_patterns, [channel1, channel2])
        if channel1 in roi_channels or channel2 in roi_channels:
            return True
    
    return False


def _get_asymmetry_columns(
    features_df: pd.DataFrame,
    segment: str,
    band: str,
    metric: str,
    roi_name: str,
    rois: Dict[str, List[str]],
) -> List[str]:
    """Get asymmetry columns filtered by segment, band, metric, and ROI."""
    columns = []
    roi_name_lower = roi_name.lower()
    
    for col in features_df.columns:
        parsed = NamingSchema.parse(str(col))
        if not parsed.get("valid") or parsed.get("group") != "asymmetry":
            continue
        if str(parsed.get("segment") or "") != segment:
            continue
        if str(parsed.get("band") or "") != band:
            continue
        if str(parsed.get("stat") or "") != metric:
            continue
        
        scope = parsed.get("scope") or ""
        if roi_name_lower == "all":
            columns.append(col)
        else:
            identifier = str(parsed.get("identifier") or "")
            if scope == "roi":
                identifier_normalized = identifier.lower().replace("_", "")
                roi_normalized = roi_name_lower.replace("_", "")
                if identifier_normalized == roi_normalized:
                    columns.append(col)
            elif scope == "chpair":
                if "-" in identifier:
                    channel1, channel2 = identifier.split("-", 1)
                    roi_channels = get_roi_channels(rois.get(roi_name, []), [channel1, channel2])
                    if channel1 in roi_channels or channel2 in roi_channels:
                        columns.append(col)
    
    return columns


def plot_asymmetry_by_condition(
    features_df: pd.DataFrame,
    events_df: pd.DataFrame,
    subject: str,
    save_dir: Path,
    logger: Any = None,
    config: Any = None,
    stats_dir: Optional[Path] = None,
) -> None:
    """Compare asymmetry indices between conditions.
    
    For window comparisons (paired): Uses the unified plot_paired_comparison helper.
    For column comparisons (unpaired): Uses Mann-Whitney U test with consistent styling.
    Creates one figure per ROI.
    
    If stats_dir is provided, uses pre-computed statistics from the behavior pipeline.
    """
    if features_df is None or features_df.empty or events_df is None:
        return
    
    comparison_config = _get_comparison_config(config)
    segments = _get_comparison_segments(features_df, config, logger)
    bands = _get_comparison_bands(features_df, config)
    metrics = _get_comparison_metrics(config)
    rois = get_roi_definitions(config)
    roi_names_raw = _get_comparison_roi_names(config, rois)
    
    if not bands:
        return
    
    roi_names = [
        r for r in roi_names_raw
        if _roi_has_asymmetry_data(features_df, r, rois)
    ]
    skipped_rois = set(roi_names_raw) - set(roi_names)
    if skipped_rois and logger:
        log_if_present(
            logger, "info",
            f"Skipping ROIs without asymmetry pairs: {sorted(skipped_rois)}"
        )
    
    if not roi_names:
        if logger:
            log_if_present(logger, "warning", "No ROIs with asymmetry data found")
        return
    
    if logger:
        log_if_present(
            logger, "info",
            f"Asymmetry comparison: segments={segments}, ROIs={roi_names}, "
            f"compare_windows={comparison_config['compare_windows']}, "
            f"compare_columns={comparison_config['compare_columns']}"
        )
    
    get_plot_config(config)
    ensure_dir(save_dir)
    
    for metric in metrics:
        metric_label = metric.replace("_", " ").title()
        
        if comparison_config["compare_windows"] and len(segments) >= 2:
            _plot_window_comparison(
                features_df, subject, save_dir, segments, bands, metric,
                metric_label, roi_names, rois, config, logger, stats_dir
            )
        
        if comparison_config["compare_columns"]:
            _plot_column_comparison(
                features_df, events_df, subject, save_dir, bands, metric,
                metric_label, roi_names, rois, config, logger, stats_dir
            )


def _get_comparison_config(config: Any) -> Dict[str, bool]:
    """Extract comparison configuration flags."""
    return {
        "compare_windows": get_config_value(
            config, "plotting.comparisons.compare_windows", True
        ),
        "compare_columns": get_config_value(
            config, "plotting.comparisons.compare_columns", False
        ),
    }


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


def _get_comparison_bands(
    features_df: pd.DataFrame,
    config: Any,
) -> List[str]:
    """Get bands for comparison from config or auto-detect."""
    bands = get_band_names(config)
    
    if not bands:
        detected_bands = set()
        for col in features_df.columns:
            parsed = NamingSchema.parse(str(col))
            if parsed.get("valid") and parsed.get("group") == "asymmetry":
                band = parsed.get("band")
                if band:
                    detected_bands.add(str(band))
        bands = sorted(list(detected_bands))
    
    return bands


def _get_comparison_metrics(config: Any) -> List[str]:
    """Get metrics for comparison from config."""
    raw = require_config_value(config, "plotting.plots.features.asymmetry.comparison_metrics")
    if isinstance(raw, str):
        metrics = [raw]
    elif isinstance(raw, (list, tuple)):
        metrics = [str(x).strip() for x in raw if str(x).strip()]
    else:
        raise TypeError(
            "plotting.plots.features.asymmetry.comparison_metrics must be a string or list of strings"
        )
    if not metrics:
        raise ValueError("plotting.plots.features.asymmetry.comparison_metrics must be non-empty")
    return metrics


def _get_comparison_roi_names(
    config: Any,
    rois: Dict[str, List[str]],
) -> List[str]:
    """Get ROI names for comparison from config."""
    config_roi_names = list(rois.keys()) if rois else []
    comp_rois = get_config_value(
        config, "plotting.comparisons.comparison_rois", []
    )
    
    if not comp_rois:
        return ["all"] + config_roi_names
    
    roi_names = []
    for requested_roi in comp_rois:
        if requested_roi.lower() == "all":
            if "all" not in roi_names:
                roi_names.append("all")
        else:
            matched = _match_roi_name(requested_roi, config_roi_names)
            if matched:
                roi_names.append(matched)
    
    return roi_names


def _match_roi_name(requested: str, available: List[str]) -> Optional[str]:
    """Match requested ROI name to available ROI names (case/underscore insensitive)."""
    requested_normalized = requested.lower().replace("_", "").replace("-", "")
    
    for available_roi in available:
        available_normalized = available_roi.lower().replace("_", "").replace("-", "")
        if requested_normalized == available_normalized:
            return available_roi
    
    return None


def _plot_window_comparison(
    features_df: pd.DataFrame,
    subject: str,
    save_dir: Path,
    segments: List[str],
    bands: List[str],
    metric: str,
    metric_label: str,
    roi_names: List[str],
    rois: Dict[str, List[str]],
    config: Any,
    logger: Any,
    stats_dir: Optional[Path],
) -> None:
    """Plot window comparison (paired) for asymmetry.
    
    Supports both 2-window comparison (simple paired) and multi-window comparison
    (3+ windows with all pairwise brackets and significance asterisks).
    """
    from eeg_pipeline.plotting.features.utils import plot_multi_window_comparison
    
    use_multi_window = len(segments) > 2
    
    for roi_name in roi_names:
        roi_suffix = _sanitize_roi_name_for_path(roi_name)
        suffix = f"_roi-{roi_suffix}" if roi_suffix else ""
        
        if use_multi_window:
            data_by_band_multi = _collect_multi_window_comparison_data(
                features_df, segments, bands, metric, roi_name, rois
            )
            
            if data_by_band_multi:
                save_path = save_dir / f"sub-{subject}_asymmetry_{metric}_by_condition{suffix}_multiwindow"
                plot_multi_window_comparison(
                    data_by_band=data_by_band_multi,
                    subject=subject,
                    save_path=save_path,
                    feature_label=f"Asymmetry ({metric_label})",
                    segments=segments,
                    config=config,
                    logger=logger,
                    roi_name=roi_name,
                    stats_dir=stats_dir,
                )
        else:
            segment1, segment2 = segments[0], segments[1]
            data_by_band = _collect_window_comparison_data(
                features_df, segment1, segment2, bands, metric, roi_name, rois
            )
            
            if data_by_band:
                save_path = save_dir / f"sub-{subject}_asymmetry_{metric}_by_condition{suffix}_window"
                plot_paired_comparison(
                    data_by_band=data_by_band,
                    subject=subject,
                    save_path=save_path,
                    feature_label=f"Asymmetry ({metric_label})",
                    config=config,
                    logger=logger,
                    label1=segment1.capitalize(),
                    label2=segment2.capitalize(),
                    roi_name=roi_name,
                    stats_dir=stats_dir,
                )


def _collect_window_comparison_data(
    features_df: pd.DataFrame,
    segment1: str,
    segment2: str,
    bands: List[str],
    metric: str,
    roi_name: str,
    rois: Dict[str, List[str]],
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Collect data for window comparison across bands (2 segments)."""
    data_by_band = {}
    
    for band in bands:
        cols1 = _get_asymmetry_columns(
            features_df, segment1, band, metric, roi_name, rois
        )
        cols2 = _get_asymmetry_columns(
            features_df, segment2, band, metric, roi_name, rois
        )
        
        if not cols1 or not cols2:
            continue
        
        series1 = features_df[cols1].apply(
            pd.to_numeric, errors="coerce"
        ).mean(axis=1)
        series2 = features_df[cols2].apply(
            pd.to_numeric, errors="coerce"
        ).mean(axis=1)
        
        valid_mask = series1.notna() & series2.notna()
        values1 = series1[valid_mask].values
        values2 = series2[valid_mask].values
        
        if len(values1) > 0:
            data_by_band[band] = (values1, values2)
    
    return data_by_band


def _collect_multi_window_comparison_data(
    features_df: pd.DataFrame,
    segments: List[str],
    bands: List[str],
    metric: str,
    roi_name: str,
    rois: Dict[str, List[str]],
) -> Dict[str, Dict[str, np.ndarray]]:
    """Collect data for multi-window comparison across bands (3+ segments)."""
    data_by_band: Dict[str, Dict[str, np.ndarray]] = {}
    
    for band in bands:
        segment_series = {}
        for seg in segments:
            cols = _get_asymmetry_columns(features_df, seg, band, metric, roi_name, rois)
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


def _plot_column_comparison(
    features_df: pd.DataFrame,
    events_df: pd.DataFrame,
    subject: str,
    save_dir: Path,
    bands: List[str],
    metric: str,
    metric_label: str,
    roi_names: List[str],
    rois: Dict[str, List[str]],
    config: Any,
    logger: Any,
    stats_dir: Optional[Path],
) -> None:
    """Plot column comparison (unpaired) for asymmetry.
    
    Supports both 2-group comparison (simple unpaired) and multi-group comparison
    (3+ groups with all pairwise brackets and significance asterisks).
    """
    from eeg_pipeline.utils.analysis.events import extract_multi_group_masks
    from eeg_pipeline.plotting.features.utils import plot_multi_group_column_comparison
    
    values_spec = get_config_value(config, "plotting.comparisons.comparison_values", [])
    use_multi_group = isinstance(values_spec, (list, tuple)) and len(values_spec) > 2
    
    if use_multi_group:
        multi_group_info = extract_multi_group_masks(events_df, config, require_enabled=True)
        if not multi_group_info:
            raise ValueError("Multi-group column comparison requested but could not resolve group masks.")
        
        masks_dict, group_labels = multi_group_info
        segment_name = str(require_config_value(config, "plotting.comparisons.comparison_segment")).strip()
        
        from eeg_pipeline.plotting.features.utils import load_multigroup_stats
        multigroup_stats = load_multigroup_stats(stats_dir) if stats_dir else None
        
        for roi_name in roi_names:
            data_by_band: Dict[str, Dict[str, np.ndarray]] = {}
            for band in bands:
                cols = _get_asymmetry_columns(features_df, segment_name, band, metric, roi_name, rois)
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
                roi_suffix = _sanitize_roi_name_for_path(roi_name)
                suffix = f"_roi-{roi_suffix}" if roi_suffix else ""
                save_path = save_dir / f"sub-{subject}_asymmetry_{metric}_by_condition{suffix}_multigroup"
                
                plot_multi_group_column_comparison(
                    data_by_band=data_by_band,
                    subject=subject,
                    save_path=save_path,
                    feature_label=f"Asymmetry ({metric_label})",
                    groups=group_labels,
                    config=config,
                    logger=logger,
                    roi_name=roi_name,
                    stats_dir=stats_dir,
                    multigroup_stats=multigroup_stats,
                )
        
        log_if_present(logger, "info", f"Saved asymmetry multi-group column comparison for {len(roi_names)} ROIs")
        return
    
    comparison_mask_info = extract_comparison_mask(events_df, config, require_enabled=True)
    if not comparison_mask_info:
        raise ValueError("Column comparison requested but could not resolve comparison masks.")
    
    mask1, mask2, label1, label2 = comparison_mask_info
    segment_name = str(require_config_value(config, "plotting.comparisons.comparison_segment")).strip()
    
    plot_cfg = get_plot_config(config)
    n_trials = len(features_df)
    
    for roi_name in roi_names:
        cell_data = _collect_column_comparison_data(
            features_df, segment_name, bands, metric, roi_name,
            rois, mask1, mask2
        )
        
        qvalues, n_significant, use_precomputed = compute_or_load_column_stats(
            stats_dir=stats_dir,
            feature_type="asymmetry",
            feature_keys=bands,
            cell_data=cell_data,
            config=config,
            logger=logger,
        )
        
        fig = _create_column_comparison_figure(
            bands, cell_data, qvalues, label1, label2, plot_cfg
        )
        
        _add_column_comparison_title(
            fig, metric_label, label1, label2, subject, roi_name,
            n_trials, n_significant, len(qvalues), use_precomputed, plot_cfg
        )
        
        plt.tight_layout()
        
        roi_suffix = _sanitize_roi_name_for_path(roi_name)
        suffix = f"_roi-{roi_suffix}" if roi_suffix else ""
        filename = f"sub-{subject}_asymmetry_{metric}_by_condition{suffix}_column"
        
        save_fig(
            fig, save_dir / filename,
            formats=plot_cfg.formats,
            dpi=plot_cfg.dpi,
            bbox_inches=plot_cfg.bbox_inches,
            pad_inches=plot_cfg.pad_inches,
            config=config,
        )
        plt.close(fig)


def _collect_column_comparison_data(
    features_df: pd.DataFrame,
    segment_name: str,
    bands: List[str],
    metric: str,
    roi_name: str,
    rois: Dict[str, List[str]],
    mask1: pd.Series,
    mask2: pd.Series,
) -> Dict[int, Optional[Dict[str, np.ndarray]]]:
    """Collect data for column comparison across bands."""
    cell_data = {}
    
    for col_idx, band in enumerate(bands):
        cols = _get_asymmetry_columns(
            features_df, segment_name, band, metric, roi_name, rois
        )
        
        if not cols:
            cell_data[col_idx] = None
            continue
        
        value_series = features_df[cols].apply(
            pd.to_numeric, errors="coerce"
        ).mean(axis=1)
        
        values1 = value_series[mask1].dropna().values
        values2 = value_series[mask2].dropna().values
        
        cell_data[col_idx] = {"v1": values1, "v2": values2}
    
    return cell_data


def _create_column_comparison_figure(
    bands: List[str],
    cell_data: Dict[int, Optional[Dict[str, np.ndarray]]],
    qvalues: Dict[int, Tuple[float, float, float, bool]],
    label1: str,
    label2: str,
    plot_cfg: Any,
) -> plt.Figure:
    """Create figure with subplots for column comparison."""
    n_bands = len(bands)
    fig, axes = plt.subplots(
        1, n_bands,
        figsize=(n_bands * _SUBPLOT_WIDTH_PER_BAND, _FIGURE_HEIGHT),
        squeeze=False,
    )
    
    for col_idx, band in enumerate(bands):
        ax = axes.flatten()[col_idx]
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
        _plot_column_comparison_subplot(
            ax, values1, values2, qvalues.get(col_idx), plot_cfg
        )
        
        ax.set_xticks([0, 1])
        ax.set_xticklabels([label1, label2], fontsize=9)
        ax.set_title(band.capitalize(), fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    
    return fig


def _plot_column_comparison_subplot(
    ax: plt.Axes,
    values1: np.ndarray,
    values2: np.ndarray,
    qvalue_info: Optional[Tuple[float, float, float, bool]],
    plot_cfg: Any,
) -> None:
    """Plot a single subplot for column comparison."""
    boxplot = ax.boxplot(
        [values1, values2],
        positions=[0, 1],
        widths=_BOXPLOT_WIDTH,
        patch_artist=True,
    )
    boxplot["boxes"][0].set_facecolor(_CONDITION_COLOR_1)
    boxplot["boxes"][0].set_alpha(_BOXPLOT_ALPHA)
    boxplot["boxes"][1].set_facecolor(_CONDITION_COLOR_2)
    boxplot["boxes"][1].set_alpha(_BOXPLOT_ALPHA)
    
    jitter1 = np.random.uniform(
        -_SCATTER_JITTER_RANGE, _SCATTER_JITTER_RANGE, len(values1)
    )
    jitter2 = np.random.uniform(
        -_SCATTER_JITTER_RANGE, _SCATTER_JITTER_RANGE, len(values2)
    )
    
    ax.scatter(
        jitter1, values1,
        c=_CONDITION_COLOR_1,
        alpha=_SCATTER_ALPHA,
        s=_SCATTER_SIZE,
    )
    ax.scatter(
        1 + jitter2, values2,
        c=_CONDITION_COLOR_2,
        alpha=_SCATTER_ALPHA,
        s=_SCATTER_SIZE,
    )
    
    all_values = np.concatenate([values1, values2])
    ymin, ymax = np.nanmin(all_values), np.nanmax(all_values)
    yrange = ymax - ymin if ymax > ymin else 0.1
    
    y_lower = ymin - _Y_PADDING_LOWER * yrange
    y_upper = ymax + _Y_PADDING_UPPER * yrange
    ax.set_ylim(y_lower, y_upper)
    
    if qvalue_info is not None:
        _, qvalue, effect_size, significant = qvalue_info
        significance_marker = "†" if significant else ""
        significance_color = "#d62728" if significant else "#333333"
        annotation_y = ymax + _ANNOTATION_Y_OFFSET * yrange
        
        ax.annotate(
            f"q={qvalue:.3f}{significance_marker}\nd={effect_size:.2f}",
            xy=(0.5, annotation_y),
            ha="center",
            fontsize=plot_cfg.font.medium,
            color=significance_color,
            fontweight="bold" if significant else "normal",
        )


def _add_column_comparison_title(
    fig: plt.Figure,
    metric_label: str,
    label1: str,
    label2: str,
    subject: str,
    roi_name: str,
    n_trials: int,
    n_significant: int,
    n_tests: int,
    use_precomputed: bool,
    plot_cfg: Any,
) -> None:
    """Add title to column comparison figure."""
    roi_display = (
        roi_name.replace("_", " ").title()
        if roi_name.lower() != "all"
        else "All Pairs"
    )
    
    stats_source = "pre-computed" if use_precomputed else "Mann-Whitney U"
    
    title = (
        f"Asymmetry ({metric_label}): {label1} vs {label2} (Column Comparison)\n"
        f"Subject: {subject} | ROI: {roi_display} | N: {n_trials} trials | "
        f"{stats_source} | FDR: {n_significant}/{n_tests} significant (†=q<0.05)"
    )
    
    fig.suptitle(
        title,
        fontsize=plot_cfg.font.suptitle,
        fontweight="bold",
        y=1.02,
    )


__all__ = [
    "plot_asymmetry_by_condition",
]
