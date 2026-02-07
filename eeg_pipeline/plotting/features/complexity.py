"""
Complexity Feature Visualization
=================================

Clean, publication-quality visualizations for nonlinear dynamics features.
Uses violin/strip plots for distributions, shows individual data points.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from eeg_pipeline.domain.features.naming import NamingSchema
from eeg_pipeline.infra.paths import ensure_dir
from eeg_pipeline.utils.config.loader import get_config_value, require_config_value
from eeg_pipeline.utils.analysis.events import extract_comparison_mask
from eeg_pipeline.plotting.io.figures import save_fig, log_if_present, get_band_color
from eeg_pipeline.plotting.config import get_plot_config
from eeg_pipeline.plotting.features.utils import (
    get_band_names,
    get_band_colors,
    plot_paired_comparison,
    compute_or_load_column_stats,
)
from eeg_pipeline.plotting.features.roi import get_roi_definitions


# Constants
COMPLEXITY_GROUP = "comp"
DEFAULT_METRICS = ["lzc", "pe", "sampen"]
METRIC_LABELS = {"lzc": "LZC", "pe": "PE", "sampen": "SampEn"}
MIN_SEGMENTS_FOR_COMPARISON = 2
VIOLIN_WIDTH = 0.7
VIOLIN_ALPHA = 0.6
SCATTER_ALPHA = 0.2
SCATTER_SIZE = 5
JITTER_RANGE = 0.1
BOXPLOT_WIDTH = 0.4
BOXPLOT_ALPHA = 0.6
BOXPLOT_SCATTER_ALPHA = 0.3
BOXPLOT_SCATTER_SIZE = 6
BOXPLOT_JITTER_RANGE = 0.08
Y_RANGE_PADDING_BOTTOM = 0.1
Y_RANGE_PADDING_TOP = 0.3
Y_ANNOTATION_OFFSET = 0.05
SUPTITLE_Y_OFFSET = 1.02
FIG_WIDTH_PER_BAND = 3
FIG_HEIGHT = 5
SIGNIFICANCE_MARKER = "†"
SIGNIFICANT_COLOR = "#d62728"
NON_SIGNIFICANT_COLOR = "#333333"
SEGMENT_COLORS = {"v1": "#5a7d9a", "v2": "#c44e52"}


def _metric_label(metric: str) -> str:
    """Map metric token to readable label."""
    metric = str(metric)
    if metric in METRIC_LABELS:
        return METRIC_LABELS[metric]
    m = re.fullmatch(r"mse(\d+)", metric.lower())
    if m:
        return f"MSE (scale {int(m.group(1))})"
    return metric.upper()


def _extract_segments_from_data(features_df: pd.DataFrame) -> set[str]:
    """Extract unique segment names from complexity feature columns."""
    segments = set()
    for col in features_df.columns:
        parsed = NamingSchema.parse(str(col))
        if parsed.get("valid") and parsed.get("group") == COMPLEXITY_GROUP:
            segment = parsed.get("segment")
            if segment:
                segments.add(str(segment))
    return segments


def _extract_bands_from_data(features_df: pd.DataFrame) -> List[str]:
    """Extract unique frequency band names from complexity feature columns."""
    bands = set()
    for col in features_df.columns:
        parsed = NamingSchema.parse(str(col))
        if parsed.get("valid") and parsed.get("group") == COMPLEXITY_GROUP:
            band = parsed.get("band")
            if band:
                bands.add(str(band))
    return sorted(bands)


def _extract_metrics_from_data(features_df: pd.DataFrame) -> List[str]:
    """Extract unique metric names from complexity feature columns."""
    metrics = set()
    for col in features_df.columns:
        parsed = NamingSchema.parse(str(col))
        if parsed.get("valid") and parsed.get("group") == COMPLEXITY_GROUP:
            stat = parsed.get("stat")
            if stat:
                metrics.add(str(stat))
    return sorted(metrics) if metrics else DEFAULT_METRICS


def _normalize_roi_name(name: str) -> str:
    """Normalize ROI name for comparison (case-insensitive, ignore separators)."""
    return name.lower().replace("_", "").replace("-", "")


def _match_roi_name(roi_id: str, roi_name: str) -> bool:
    """Check if ROI identifier matches the target ROI name."""
    return _normalize_roi_name(roi_id) == _normalize_roi_name(roi_name)


def _get_complexity_columns(
    features_df: pd.DataFrame,
    segment: str,
    band: str,
    metric: str,
    roi_name: str,
) -> List[str]:
    """Get complexity columns filtered by segment, band, metric, and ROI."""
    columns = []
    
    for col in features_df.columns:
        parsed = NamingSchema.parse(str(col))
        if not parsed.get("valid"):
            continue
        if parsed.get("group") != COMPLEXITY_GROUP:
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
                columns.append(col)
        else:
            if scope == "roi":
                roi_id = str(parsed.get("identifier") or "")
                if _match_roi_name(roi_id, roi_name):
                    columns.append(col)
    
    if roi_name == "all" and not columns:
        for col in features_df.columns:
            parsed = NamingSchema.parse(str(col))
            if not parsed.get("valid"):
                continue
            if parsed.get("group") != COMPLEXITY_GROUP:
                continue
            if str(parsed.get("segment") or "") != segment:
                continue
            if str(parsed.get("band") or "") != band:
                continue
            if str(parsed.get("stat") or "") != metric:
                continue
            if parsed.get("scope") == "roi":
                columns.append(col)
    
    return columns


def _determine_segments(config: Any, features_df: pd.DataFrame, logger: Any) -> List[str]:
    """Determine segments to compare from config (no fallbacks)."""
    segments = require_config_value(config, "plotting.comparisons.comparison_windows")
    if not isinstance(segments, (list, tuple)) or len(segments) < MIN_SEGMENTS_FOR_COMPARISON:
        raise ValueError(
            "plotting.comparisons.comparison_windows must be a list/tuple with at least "
            f"{MIN_SEGMENTS_FOR_COMPARISON} window names (got {segments!r})"
        )
    return [str(s) for s in segments]


def _determine_bands(config: Any, features_df: pd.DataFrame) -> List[str]:
    """Determine frequency bands from config or data."""
    bands = get_band_names(config)
    if not bands:
        bands = _extract_bands_from_data(features_df)
    return bands


def _determine_roi_names(config: Any, features_df: pd.DataFrame) -> List[str]:
    """Determine ROI names to plot from config or defaults."""
    rois = get_roi_definitions(config)
    config_roi_names = list(rois.keys()) if rois else []
    
    comp_rois = get_config_value(config, "plotting.comparisons.comparison_rois", [])
    if comp_rois:
        roi_names = []
        for roi in comp_rois:
            if roi.lower() == "all":
                if "all" not in roi_names:
                    roi_names.append("all")
            else:
                for config_roi in config_roi_names:
                    if _match_roi_name(roi, config_roi):
                        roi_names.append(config_roi)
                        break
    else:
        roi_names = ["all"]
        roi_names.extend(config_roi_names)
    
    return roi_names


def _create_roi_suffix(roi_name: str) -> str:
    """Create filename suffix for ROI name."""
    from eeg_pipeline.utils.formatting import sanitize_label
    if roi_name.lower() == "all":
        return ""
    roi_safe = sanitize_label(roi_name).lower()
    return f"_roi-{roi_safe}"


def _prepare_window_comparison_data(
    features_df: pd.DataFrame,
    segments: List[str],
    bands: List[str],
    metric: str,
    roi_name: str,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Prepare data for window comparison (paired, 2 segments)."""
    data_by_band = {}
    segment1, segment2 = segments[0], segments[1]
    
    for band in bands:
        cols1 = _get_complexity_columns(features_df, segment1, band, metric, roi_name)
        cols2 = _get_complexity_columns(features_df, segment2, band, metric, roi_name)
        
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
            cols = _get_complexity_columns(features_df, seg, band, metric, roi_name)
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


def _plot_window_comparison(
    features_df: pd.DataFrame,
    segments: List[str],
    bands: List[str],
    metrics: List[str],
    roi_names: List[str],
    subject: str,
    save_dir: Path,
    config: Any,
    logger: Any,
    stats_dir: Optional[Path],
) -> None:
    """Create paired window comparison plots.
    
    Supports both 2-window comparison (simple paired) and multi-window comparison
    (3+ windows with all pairwise brackets and significance asterisks).
    """
    from eeg_pipeline.plotting.features.utils import plot_multi_window_comparison
    
    use_multi_window = len(segments) > 2
    
    for metric in metrics:
        metric_label = _metric_label(metric)
        
        for roi_name in roi_names:
            suffix = _create_roi_suffix(roi_name)
            
            if use_multi_window:
                data_by_band_multi = _prepare_multi_window_comparison_data(
                    features_df, segments, bands, metric, roi_name
                )
                
                if data_by_band_multi:
                    save_path = save_dir / f"sub-{subject}_complexity_{metric}_by_condition{suffix}_multiwindow"
                    plot_multi_window_comparison(
                        data_by_band=data_by_band_multi,
                        subject=subject,
                        save_path=save_path,
                        feature_label=f"Complexity ({metric_label})",
                        segments=segments,
                        config=config,
                        logger=logger,
                        roi_name=roi_name,
                        stats_dir=stats_dir,
                    )
            else:
                data_by_band = _prepare_window_comparison_data(
                    features_df, segments, bands, metric, roi_name
                )
                
                if data_by_band:
                    save_path = save_dir / f"sub-{subject}_complexity_{metric}_by_condition{suffix}_window"
                    plot_paired_comparison(
                        data_by_band=data_by_band,
                        subject=subject,
                        save_path=save_path,
                        feature_label=f"Complexity ({metric_label})",
                        config=config,
                        logger=logger,
                        label1=segments[0].capitalize(),
                        label2=segments[1].capitalize(),
                        roi_name=roi_name,
                        stats_dir=stats_dir,
                    )
        
        plot_type = "multi-window" if use_multi_window else "paired"
        if logger:
            log_if_present(logger, "info", f"Saved complexity {metric_label} {plot_type} comparison plots for {len(roi_names)} ROIs")


def _prepare_column_comparison_data(
    features_df: pd.DataFrame,
    segment: str,
    bands: List[str],
    metric: str,
    roi_name: str,
    mask1: pd.Series,
    mask2: pd.Series,
) -> Dict[int, Dict[str, np.ndarray]]:
    """Prepare data for column comparison (unpaired)."""
    cell_data = {}
    
    for col_idx, band in enumerate(bands):
        cols = _get_complexity_columns(features_df, segment, band, metric, roi_name)
        
        if not cols:
            cell_data[col_idx] = None
            continue
        
        value_series = features_df[cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
        values1 = value_series[mask1].dropna().values
        values2 = value_series[mask2].dropna().values
        
        cell_data[col_idx] = {"v1": values1, "v2": values2}
    
    return cell_data


def _plot_single_band_comparison(
    ax: plt.Axes,
    data: Dict[str, np.ndarray],
    band: str,
    band_color: str,
    label1: str,
    label2: str,
    plot_cfg: Any,
    qvalue_info: Optional[Tuple[float, float, bool]],
) -> None:
    """Plot single band comparison on given axes."""
    values1 = data["v1"]
    values2 = data["v2"]
    
    if len(values1) == 0 or len(values2) == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
               transform=ax.transAxes, fontsize=plot_cfg.font.title, color="gray")
        ax.set_xticks([])
        return
    
    boxplot = ax.boxplot([values1, values2], positions=[0, 1], widths=BOXPLOT_WIDTH, patch_artist=True)
    boxplot["boxes"][0].set_facecolor(SEGMENT_COLORS["v1"])
    boxplot["boxes"][0].set_alpha(BOXPLOT_ALPHA)
    boxplot["boxes"][1].set_facecolor(SEGMENT_COLORS["v2"])
    boxplot["boxes"][1].set_alpha(BOXPLOT_ALPHA)
    
    jitter1 = np.random.uniform(-BOXPLOT_JITTER_RANGE, BOXPLOT_JITTER_RANGE, len(values1))
    jitter2 = np.random.uniform(-BOXPLOT_JITTER_RANGE, BOXPLOT_JITTER_RANGE, len(values2))
    ax.scatter(jitter1, values1, c=SEGMENT_COLORS["v1"], alpha=BOXPLOT_SCATTER_ALPHA, s=BOXPLOT_SCATTER_SIZE)
    ax.scatter(1 + jitter2, values2, c=SEGMENT_COLORS["v2"], alpha=BOXPLOT_SCATTER_ALPHA, s=BOXPLOT_SCATTER_SIZE)
    
    all_values = np.concatenate([values1, values2])
    ymin = np.nanmin(all_values)
    ymax = np.nanmax(all_values)
    yrange = ymax - ymin if ymax > ymin else 0.1
    ax.set_ylim(ymin - Y_RANGE_PADDING_BOTTOM * yrange, ymax + Y_RANGE_PADDING_TOP * yrange)
    
    if qvalue_info is not None:
        _, qvalue, cohens_d, is_significant = qvalue_info
        marker = SIGNIFICANCE_MARKER if is_significant else ""
        color = SIGNIFICANT_COLOR if is_significant else NON_SIGNIFICANT_COLOR
        fontweight = "bold" if is_significant else "normal"
        annotation_y = ymax + Y_ANNOTATION_OFFSET * yrange
        ax.annotate(f"q={qvalue:.3f}{marker}\nd={cohens_d:.2f}",
                   xy=(0.5, annotation_y), ha="center",
                   fontsize=plot_cfg.font.medium, color=color, fontweight=fontweight)
    
    ax.set_xticks([0, 1])
    ax.set_xticklabels([label1, label2], fontsize=9)
    ax.set_title(band.capitalize(), fontweight="bold", color=band_color)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _format_roi_display_name(roi_name: str) -> str:
    """Format ROI name for display."""
    if roi_name.lower() == "all":
        return "All Channels"
    return roi_name.replace("_", " ").title()


def _plot_column_comparison(
    features_df: pd.DataFrame,
    events_df: pd.DataFrame,
    bands: List[str],
    metrics: List[str],
    roi_names: List[str],
    subject: str,
    save_dir: Path,
    config: Any,
    logger: Any,
    stats_dir: Optional[Path],
) -> None:
    """Create unpaired column comparison plots.
    
    Supports both 2-group comparison (simple unpaired) and multi-group comparison
    (3+ groups with all pairwise brackets and significance asterisks).
    """
    from eeg_pipeline.utils.analysis.events import extract_multi_group_masks
    from eeg_pipeline.plotting.features.utils import plot_multi_group_column_comparison
    from eeg_pipeline.utils.formatting import sanitize_label
    
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
        
        for metric in metrics:
            metric_label = _metric_label(metric)
            
            for roi_name in roi_names:
                data_by_band: Dict[str, Dict[str, np.ndarray]] = {}
                for band in bands:
                    cols = _get_complexity_columns(features_df, segment_name, band, metric, roi_name)
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
                    roi_safe = sanitize_label(roi_name).lower() if roi_name != "all" else ""
                    suffix = f"_roi-{roi_safe}" if roi_safe else ""
                    save_path = save_dir / f"sub-{subject}_complexity_{metric}_by_condition{suffix}_multigroup"
                    
                    plot_multi_group_column_comparison(
                        data_by_band=data_by_band,
                        subject=subject,
                        save_path=save_path,
                        feature_label=f"Complexity ({metric_label})",
                        groups=group_labels,
                        config=config,
                        logger=logger,
                        roi_name=roi_name,
                        stats_dir=stats_dir,
                        multigroup_stats=multigroup_stats,
                    )
        
        log_if_present(logger, "info", f"Saved complexity multi-group column comparison for {len(roi_names)} ROIs")
        return
    
    comp_mask_info = extract_comparison_mask(events_df, config, require_enabled=True)
    if not comp_mask_info:
        raise ValueError("Column comparison requested but could not resolve comparison masks.")
    
    mask1, mask2, label1, label2 = comp_mask_info
    segment_name = str(require_config_value(config, "plotting.comparisons.comparison_segment")).strip()
    plot_cfg = get_plot_config(config)
    band_colors = {band: get_band_color(band, config) for band in bands}
    n_bands = len(bands)
    n_trials = len(features_df)
    
    for metric in metrics:
        metric_label = _metric_label(metric)
        
        for roi_name in roi_names:
            cell_data = _prepare_column_comparison_data(
                features_df, segment_name, bands, metric, roi_name, mask1, mask2
            )
            
            qvalues, n_significant, use_precomputed = compute_or_load_column_stats(
                stats_dir=stats_dir,
                feature_type="complexity",
                feature_keys=bands,
                cell_data=cell_data,
                config=config,
                logger=logger,
            )
            
            fig, axes = plt.subplots(1, n_bands, figsize=(FIG_WIDTH_PER_BAND * n_bands, FIG_HEIGHT), squeeze=False)
            
            for col_idx, band in enumerate(bands):
                ax = axes.flatten()[col_idx]
                data = cell_data.get(col_idx)
                qvalue_info = qvalues.get(col_idx) if col_idx in qvalues else None
                
                if data is None:
                    ax.text(0.5, 0.5, "No data", ha="center", va="center",
                           transform=ax.transAxes, fontsize=plot_cfg.font.title, color="gray")
                    ax.set_xticks([])
                    continue
                
                _plot_single_band_comparison(
                    ax, data, band, band_colors.get(band, "gray"),
                    label1, label2, plot_cfg, qvalue_info
                )
            
            n_tests = len(qvalues)
            roi_display = _format_roi_display_name(roi_name)
            stats_source = "pre-computed" if use_precomputed else "Mann-Whitney U"
            title = (f"Complexity ({metric_label}): {label1} vs {label2} (Column Comparison)\n"
                    f"Subject: {subject} | ROI: {roi_display} | N: {n_trials} trials | {stats_source} | "
                    f"FDR: {n_significant}/{n_tests} significant ({SIGNIFICANCE_MARKER}=q<0.05)")
            fig.suptitle(title, fontsize=plot_cfg.font.suptitle, fontweight="bold", y=SUPTITLE_Y_OFFSET)
            
            plt.tight_layout()
            
            suffix = _create_roi_suffix(roi_name)
            filename = f"sub-{subject}_complexity_{metric}_by_condition{suffix}_column"
            
            save_fig(fig, save_dir / filename, formats=plot_cfg.formats, dpi=plot_cfg.dpi,
                    bbox_inches=plot_cfg.bbox_inches, pad_inches=plot_cfg.pad_inches, config=config)
            plt.close(fig)
        
        if logger:
            log_if_present(logger, "info", f"Saved complexity {metric_label} column comparison plots for {len(roi_names)} ROIs")


def plot_complexity_by_condition(
    features_df: pd.DataFrame,
    events_df: pd.DataFrame,
    subject: str,
    save_dir: Path,
    logger: Any = None,
    config: Any = None,
    stats_dir: Optional[Path] = None,
) -> None:
    """Compare complexity metrics between conditions per band.
    
    Complexity uses frequency bands (delta, theta, alpha, beta, gamma) with
    metrics like lzc (Lempel-Ziv), pe (Permutation Entropy), sampen (Sample Entropy),
    and mseXX (Multiscale Entropy at scale XX).
    
    For window comparisons (paired): Uses the unified plot_paired_comparison helper.
    For column comparisons (unpaired): Uses Mann-Whitney U test with consistent styling.
    Creates one figure per ROI per metric.
    
    If stats_dir is provided, uses pre-computed statistics from the behavior pipeline.
    """
    if features_df is None or features_df.empty or events_df is None:
        return
    
    compare_windows = get_config_value(config, "plotting.comparisons.compare_windows", True)
    compare_columns = get_config_value(config, "plotting.comparisons.compare_columns", False)
    
    segments = _determine_segments(config, features_df, logger)
    bands = _determine_bands(config, features_df)
    if not bands:
        return
    
    metrics = _extract_metrics_from_data(features_df)
    roi_names = _determine_roi_names(config, features_df)
    
    if logger:
        log_if_present(logger, "info",
                      f"Complexity comparison: segments={segments}, ROIs={roi_names}, "
                      f"bands={bands}, metrics={metrics}, compare_windows={compare_windows}, "
                      f"compare_columns={compare_columns}")
    
    ensure_dir(save_dir)
    
    if compare_windows and len(segments) >= MIN_SEGMENTS_FOR_COMPARISON:
        _plot_window_comparison(
            features_df, segments, bands, metrics, roi_names,
            subject, save_dir, config, logger, stats_dir
        )
    
    if compare_columns:
        _plot_column_comparison(
            features_df, events_df, bands, metrics, roi_names,
            subject, save_dir, config, logger, stats_dir
        )


__all__ = [
    "plot_complexity_by_condition",
]
