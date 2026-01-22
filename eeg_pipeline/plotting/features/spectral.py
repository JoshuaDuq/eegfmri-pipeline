"""
Spectral Feature Visualization
==============================

Plots for spectral peak metrics and edge frequencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from eeg_pipeline.domain.features.naming import NamingSchema
from eeg_pipeline.plotting.config import get_plot_config
from eeg_pipeline.plotting.io.figures import save_fig
from eeg_pipeline.plotting.features.utils import get_band_names
from eeg_pipeline.utils.config.loader import get_config_value


# Constants
_VIOLIN_WIDTH = 0.7
_VIOLIN_ALPHA = 0.6
_VIOLIN_COLOR = "#7C3AED"
_HISTOGRAM_BINS = 25
_HISTOGRAM_ALPHA = 0.7
_HISTOGRAM_EDGECOLOR = "white"
_MEAN_LINEWIDTH = 1
_MEAN_LINESTYLE = "--"
_MEAN_COLOR = "black"

_MIN_FIG_WIDTH = 8.0
_MIN_FIG_HEIGHT = 4.5
_BAND_WIDTH_FACTOR = 1.2
_ROW_HEIGHT_FACTOR = 3.0

_BOX_WIDTH = 0.4
_BOX_ALPHA = 0.6
_SCATTER_JITTER_RANGE = 0.08
_SCATTER_ALPHA = 0.3
_SCATTER_SIZE = 6
_Y_PADDING_BOTTOM = 0.1
_Y_PADDING_TOP = 0.3
_Y_ANNOTATION_OFFSET = 0.05

_SEGMENT_COLOR_V1 = "#5a7d9a"
_SEGMENT_COLOR_V2 = "#c44e52"
_SIGNIFICANT_COLOR = "#d62728"
_NONSIGNIFICANT_COLOR = "#333333"
_GRAY_COLOR = "gray"

_DEFAULT_METRICS = [
    "peak_freq",
    "center_freq",
    "bandwidth",
    "entropy",
    "peak_power",
    "logratio_mean",
    "logratio_std",
    "slope",
]

_COMPARISON_METRICS_DEFAULT = [
    "peak_freq",
    "peak_power",
    "center_freq",
    "bandwidth",
    "entropy",
]

_MIN_SEGMENTS_FOR_COMPARISON = 2
_MAX_METRICS_FALLBACK = 4
_MAX_METRICS_COMPARISON_FALLBACK = 3
_SUBTITLE_Y_OFFSET = 1.02


@dataclass
class SpectralColumn:
    name: str
    segment: str
    band: str
    scope: str
    stat: str


def _create_empty_plot(message: str) -> plt.Figure:
    """Create an empty plot with a centered message."""
    fig, ax = plt.subplots()
    ax.text(0.5, 0.5, message, ha="center", va="center")
    return fig


def _parse_spectral_column(col: str) -> Optional[SpectralColumn]:
    """Parse a spectral column name into structured components."""
    parsed = NamingSchema.parse(str(col))
    if parsed.get("valid") and parsed.get("group") == "spectral":
        return SpectralColumn(
            name=str(col),
            segment=str(parsed.get("segment") or ""),
            band=str(parsed.get("band") or ""),
            scope=str(parsed.get("scope") or ""),
            stat=str(parsed.get("stat") or ""),
        )
    return None


def _collect_spectral_columns(features_df: pd.DataFrame) -> List[SpectralColumn]:
    cols: List[SpectralColumn] = []
    for col in features_df.columns:
        entry = _parse_spectral_column(col)
        if entry is not None:
            cols.append(entry)
    return cols


def _select_segment(segments: List[str], preferred: str = "active") -> Optional[str]:
    if not segments:
        return None
    if preferred in segments:
        return preferred
    return segments[0]


def _select_columns(
    entries: List[SpectralColumn],
    *,
    segment: str,
    band: str,
    stat_preference: List[str],
    scope_preference: List[str],
) -> Tuple[List[str], Optional[str], Optional[str]]:
    for scope in scope_preference:
        for stat in stat_preference:
            cols = [
                e.name
                for e in entries
                if e.segment == segment and e.band == band and e.scope == scope and e.stat == stat
            ]
            if cols:
                return cols, scope, stat
    return [], None, None


def _get_metric_values(features_df: pd.DataFrame, columns: List[str]) -> np.ndarray:
    """Extract numeric values from feature columns, averaging if multiple."""
    if len(columns) == 1:
        series = pd.to_numeric(features_df[columns[0]], errors="coerce")
    else:
        series = features_df[columns].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    return series.dropna().values


def _format_metric_label(metric: str) -> str:
    """Format metric name for display."""
    return metric.replace("_", " ").title()


def _plot_violin_for_metric(
    ax: plt.Axes,
    data_list: List[np.ndarray],
    positions: List[int],
    band_labels: List[str],
) -> None:
    """Plot violin plot for a single metric."""
    if not data_list:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        return

    parts = ax.violinplot(
        data_list,
        positions=positions,
        showmedians=True,
        widths=_VIOLIN_WIDTH,
    )
    for pc in parts.get("bodies", []):
        pc.set_facecolor(_VIOLIN_COLOR)
        pc.set_alpha(_VIOLIN_ALPHA)
    
    ax.set_xticks(range(len(band_labels)))
    ax.set_xticklabels([b.capitalize() for b in band_labels])


def plot_spectral_summary(
    features_df: pd.DataFrame,
    save_path: Path,
    *,
    config: Any = None,
    metrics: Optional[List[str]] = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> plt.Figure:
    """Plot spectral feature distributions by band."""
    plot_cfg = get_plot_config(config)
    entries = _collect_spectral_columns(features_df)
    
    if not entries:
        return _create_empty_plot("No spectral data")

    segments = sorted({e.segment for e in entries if e.segment})
    segment = _select_segment(segments, preferred="active")
    
    if segment is None:
        return _create_empty_plot("No spectral data")

    bands = sorted({e.band for e in entries if e.segment == segment and e.band})
    if not bands:
        return _create_empty_plot("No spectral data")

    if metrics is None:
        metrics = get_config_value(
            config,
            "plotting.plots.features.spectral.metrics",
            _DEFAULT_METRICS,
        )
    
    metrics = list(metrics or [])
    available_metrics = [m for m in metrics if any(e.stat == m for e in entries)]
    
    if not available_metrics:
        all_stats = sorted({e.stat for e in entries if e.stat})
        available_metrics = all_stats[:_MAX_METRICS_FALLBACK]

    band_order = get_band_names(config)
    ordered_bands = [b for b in band_order if b in bands]
    remaining_bands = [b for b in bands if b not in band_order]
    bands = ordered_bands + remaining_bands

    n_rows = len(available_metrics)
    if figsize is None:
        width = max(_MIN_FIG_WIDTH, len(bands) * _BAND_WIDTH_FACTOR)
        height = max(_MIN_FIG_HEIGHT, n_rows * _ROW_HEIGHT_FACTOR)
        figsize = (width, height)

    fig, axes = plt.subplots(n_rows, 1, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    scope_preference = ["global", "roi", "ch"]
    
    for ax, metric in zip(axes, available_metrics):
        data_list = []
        positions = []
        
        for i, band in enumerate(bands):
            cols, _, _ = _select_columns(
                entries,
                segment=segment,
                band=band,
                stat_preference=[metric],
                scope_preference=scope_preference,
            )
            if not cols:
                continue
            
            values = _get_metric_values(features_df, cols)
            if values.size == 0:
                continue
            
            data_list.append(values)
            positions.append(i)

        _plot_violin_for_metric(ax, data_list, positions, bands)
        
        metric_label = _format_metric_label(metric)
        ax.set_ylabel(metric_label)
        ax.set_title(metric_label, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        f"Spectral Features by Band ({segment})",
        fontsize=plot_cfg.font.figure_title,
        fontweight="bold",
        y=_SUBTITLE_Y_OFFSET,
    )
    plt.tight_layout()
    save_fig(
        fig,
        save_path,
        formats=plot_cfg.formats,
        dpi=plot_cfg.dpi,
        bbox_inches=plot_cfg.bbox_inches,
        pad_inches=plot_cfg.pad_inches,
        config=config,
    )
    plt.close(fig)
    return fig


def plot_spectral_edge_frequency(
    features_df: pd.DataFrame,
    save_path: Path,
    *,
    config: Any = None,
    figsize: Optional[Tuple[float, float]] = None,
) -> plt.Figure:
    """Plot broadband spectral edge frequency distribution."""
    plot_cfg = get_plot_config(config)
    entries = _collect_spectral_columns(features_df)
    
    if not entries:
        return _create_empty_plot("No spectral data")

    edge_cols = [
        e.name for e in entries
        if e.band == "broadband" and "edge" in e.stat
    ]
    
    if not edge_cols:
        return _create_empty_plot("No edge frequency data")

    values = _get_metric_values(features_df, edge_cols)
    
    if values.size == 0:
        return _create_empty_plot("No edge frequency data")

    if figsize is None:
        figsize = (_MIN_FIG_WIDTH, _MIN_FIG_HEIGHT)

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(
        values,
        bins=_HISTOGRAM_BINS,
        color=_VIOLIN_COLOR,
        alpha=_HISTOGRAM_ALPHA,
        edgecolor=_HISTOGRAM_EDGECOLOR,
    )
    ax.set_xlabel("Edge Frequency (Hz)")
    ax.set_ylabel("Count")
    ax.set_title("Spectral Edge Frequency", fontweight="bold")
    
    mean_value = np.nanmean(values)
    ax.axvline(
        mean_value,
        color=_MEAN_COLOR,
        linestyle=_MEAN_LINESTYLE,
        linewidth=_MEAN_LINEWIDTH,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Broadband Spectral Edge Frequency",
        fontsize=plot_cfg.font.figure_title,
        fontweight="bold",
        y=_SUBTITLE_Y_OFFSET,
    )
    plt.tight_layout()
    save_fig(
        fig,
        save_path,
        formats=plot_cfg.formats,
        dpi=plot_cfg.dpi,
        bbox_inches=plot_cfg.bbox_inches,
        pad_inches=plot_cfg.pad_inches,
        config=config,
    )
    plt.close(fig)
    return fig


def _get_roi_id_from_column_name(column_name: str) -> Optional[str]:
    """Extract ROI identifier from spectral column name."""
    parsed = NamingSchema.parse(str(column_name))
    if parsed.get("valid") and parsed.get("scope") == "roi":
        return parsed.get("identifier")
    return None


def _normalize_roi_name(name: str) -> str:
    """Normalize ROI name for comparison by removing separators."""
    return name.lower().replace("_", "").replace("-", "")


def _get_spectral_columns_for_roi(
    entries: List[SpectralColumn],
    segment: str,
    band: str,
    metric: str,
    roi_name: str,
) -> List[str]:
    """Get spectral column names filtered by segment, band, metric, and ROI."""
    columns = []
    
    for entry in entries:
        if entry.segment != segment or entry.band != band or entry.stat != metric:
            continue
        
        if roi_name == "all":
            if entry.scope == "global":
                columns.append(entry.name)
        elif entry.scope == "roi":
            roi_id = _get_roi_id_from_column_name(entry.name)
            if roi_id and _normalize_roi_name(roi_id) == _normalize_roi_name(roi_name):
                columns.append(entry.name)
    
    if roi_name == "all" and not columns:
        for entry in entries:
            if (entry.segment == segment and entry.band == band and
                entry.stat == metric and entry.scope == "roi"):
                columns.append(entry.name)
    
    return columns


def _get_comparison_configuration(
    entries: List[SpectralColumn],
    config: Any,
) -> Tuple[List[str], List[str], List[str]]:
    """Extract segments, bands, and metrics from config or data."""
    segment_set = sorted({e.segment for e in entries if e.segment})
    
    segments = get_config_value(config, "plotting.comparisons.comparison_windows", [])
    if not segments or len(segments) < _MIN_SEGMENTS_FOR_COMPARISON:
        if len(segment_set) >= _MIN_SEGMENTS_FOR_COMPARISON:
            segments = segment_set[:_MIN_SEGMENTS_FOR_COMPARISON]
    
    bands = get_band_names(config)
    if not bands:
        bands = sorted({
            e.band for e in entries
            if e.band and e.band != "broadband"
        })
    
    metrics = get_config_value(
        config,
        "plotting.plots.features.spectral.comparison_metrics",
        None,
    )
    if not metrics:
        metrics = get_config_value(
            config,
            "plotting.plots.features.spectral.metrics",
            _COMPARISON_METRICS_DEFAULT,
        )
    
    metrics = [m for m in metrics if any(e.stat == m for e in entries)]
    if not metrics:
        all_stats = sorted({e.stat for e in entries if e.stat})
        metrics = all_stats[:_MAX_METRICS_COMPARISON_FALLBACK]
    
    return segments, bands, metrics


def _get_roi_names(config: Any) -> List[str]:
    """Get ROI names from config, including 'all' option."""
    from eeg_pipeline.plotting.features.roi import get_roi_definitions
    
    rois = get_roi_definitions(config)
    config_roi_names = list(rois.keys()) if rois else []
    
    comp_rois = get_config_value(config, "plotting.comparisons.comparison_rois", [])
    if not comp_rois:
        return ["all"] + config_roi_names
    
    roi_names = []
    for r in comp_rois:
        if r.lower() == "all":
            if "all" not in roi_names:
                roi_names.append("all")
        else:
            normalized_r = _normalize_roi_name(r)
            for config_roi in config_roi_names:
                if _normalize_roi_name(config_roi) == normalized_r:
                    roi_names.append(config_roi)
                    break
    
    return roi_names


def _create_roi_suffix(roi_name: str) -> str:
    """Create filename suffix for ROI."""
    from eeg_pipeline.utils.formatting import sanitize_label
    if roi_name.lower() == "all":
        return ""
    roi_safe = sanitize_label(roi_name).lower()
    return f"_roi-{roi_safe}"


def _plot_window_comparison(
    features_df: pd.DataFrame,
    entries: List[SpectralColumn],
    segments: List[str],
    bands: List[str],
    metric: str,
    metric_label: str,
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
    from eeg_pipeline.plotting.features.utils import plot_paired_comparison, plot_multi_window_comparison
    from eeg_pipeline.plotting.io.figures import log_if_present
    
    use_multi_window = len(segments) > 2
    
    for roi_name in roi_names:
        suffix = _create_roi_suffix(roi_name)
        
        if use_multi_window:
            data_by_band_multi: Dict[str, Dict[str, np.ndarray]] = {}
            for band in bands:
                segment_series = {}
                for seg in segments:
                    cols = _get_spectral_columns_for_roi(entries, seg, band, metric, roi_name)
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
                    data_by_band_multi[band] = segment_values
            
            if data_by_band_multi:
                save_path = save_dir / f"sub-{subject}_spectral_{metric}_by_condition{suffix}_multiwindow"
                plot_multi_window_comparison(
                    data_by_band=data_by_band_multi,
                    subject=subject,
                    save_path=save_path,
                    feature_label=f"Spectral ({metric_label})",
                    segments=segments,
                    config=config,
                    logger=logger,
                    roi_name=roi_name,
                    stats_dir=stats_dir,
                )
        else:
            seg1, seg2 = segments[0], segments[1]
            data_by_band = {}
            
            for band in bands:
                cols1 = _get_spectral_columns_for_roi(entries, seg1, band, metric, roi_name)
                cols2 = _get_spectral_columns_for_roi(entries, seg2, band, metric, roi_name)
                
                if not cols1 or not cols2:
                    continue
                
                series1 = features_df[cols1].apply(pd.to_numeric, errors="coerce").mean(axis=1)
                series2 = features_df[cols2].apply(pd.to_numeric, errors="coerce").mean(axis=1)
                
                valid_mask = series1.notna() & series2.notna()
                values1 = series1[valid_mask].values
                values2 = series2[valid_mask].values
                
                if len(values1) > 0:
                    data_by_band[band] = (values1, values2)
            
            if data_by_band:
                save_path = save_dir / f"sub-{subject}_spectral_{metric}_by_condition{suffix}_window"
                plot_paired_comparison(
                    data_by_band=data_by_band,
                    subject=subject,
                    save_path=save_path,
                    feature_label=f"Spectral ({metric_label})",
                    config=config,
                    logger=logger,
                    label1=seg1.capitalize(),
                    label2=seg2.capitalize(),
                    roi_name=roi_name,
                    stats_dir=stats_dir,
                )
    
    plot_type = "multi-window" if use_multi_window else "paired"
    if logger:
        log_if_present(
            logger,
            "info",
            f"Saved spectral {metric_label} {plot_type} comparison plots for {len(roi_names)} ROIs",
        )


def _plot_column_comparison_single_band(
    ax: plt.Axes,
    values1: np.ndarray,
    values2: np.ndarray,
    band: str,
    band_color: str,
    label1: str,
    label2: str,
    qvalue_info: Optional[Tuple[float, float, float, bool]],
    plot_cfg: Any,
) -> None:
    """Plot single band column comparison with boxplot and statistics."""
    if len(values1) == 0 or len(values2) == 0:
        ax.text(
            0.5, 0.5, "No data",
            ha="center", va="center",
            transform=ax.transAxes,
            fontsize=plot_cfg.font.title,
            color=_GRAY_COLOR,
        )
        ax.set_xticks([])
        return
    
    box_plot = ax.boxplot(
        [values1, values2],
        positions=[0, 1],
        widths=_BOX_WIDTH,
        patch_artist=True,
    )
    box_plot["boxes"][0].set_facecolor(_SEGMENT_COLOR_V1)
    box_plot["boxes"][0].set_alpha(_BOX_ALPHA)
    box_plot["boxes"][1].set_facecolor(_SEGMENT_COLOR_V2)
    box_plot["boxes"][1].set_alpha(_BOX_ALPHA)
    
    jitter1 = np.random.uniform(-_SCATTER_JITTER_RANGE, _SCATTER_JITTER_RANGE, len(values1))
    jitter2 = np.random.uniform(-_SCATTER_JITTER_RANGE, _SCATTER_JITTER_RANGE, len(values2))
    ax.scatter(jitter1, values1, c=_SEGMENT_COLOR_V1, alpha=_SCATTER_ALPHA, s=_SCATTER_SIZE)
    ax.scatter(1 + jitter2, values2, c=_SEGMENT_COLOR_V2, alpha=_SCATTER_ALPHA, s=_SCATTER_SIZE)
    
    all_values = np.concatenate([values1, values2])
    y_min = np.nanmin(all_values)
    y_max = np.nanmax(all_values)
    y_range = y_max - y_min if y_max > y_min else 0.1
    
    y_padding_bottom = _Y_PADDING_BOTTOM * y_range
    y_padding_top = _Y_PADDING_TOP * y_range
    ax.set_ylim(y_min - y_padding_bottom, y_max + y_padding_top)
    
    if qvalue_info:
        _, q, d, is_significant = qvalue_info
        sig_marker = "†" if is_significant else ""
        sig_color = _SIGNIFICANT_COLOR if is_significant else _NONSIGNIFICANT_COLOR
        annotation_y = y_max + _Y_ANNOTATION_OFFSET * y_range
        ax.annotate(
            f"q={q:.3f}{sig_marker}\nd={d:.2f}",
            xy=(0.5, annotation_y),
            ha="center",
            fontsize=plot_cfg.font.medium,
            color=sig_color,
            fontweight="bold" if is_significant else "normal",
        )
    
    ax.set_xticks([0, 1])
    ax.set_xticklabels([label1, label2], fontsize=9)
    ax.set_title(band.capitalize(), fontweight="bold", color=band_color)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_column_comparison(
    features_df: pd.DataFrame,
    entries: List[SpectralColumn],
    events_df: pd.DataFrame,
    bands: List[str],
    metric: str,
    metric_label: str,
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
    from eeg_pipeline.utils.analysis.events import extract_comparison_mask, extract_multi_group_masks
    from eeg_pipeline.plotting.features.utils import (
        compute_or_load_column_stats,
        get_band_color,
        plot_multi_group_column_comparison,
    )
    from eeg_pipeline.plotting.io.figures import log_if_present
    from eeg_pipeline.utils.formatting import sanitize_label
    
    values_spec = get_config_value(config, "plotting.comparisons.comparison_values", [])
    use_multi_group = isinstance(values_spec, (list, tuple)) and len(values_spec) > 2
    
    if use_multi_group:
        multi_group_info = extract_multi_group_masks(events_df, config, require_enabled=False)
        if not multi_group_info:
            log_if_present(logger, "warning", "Multi-group column comparison enabled but config incomplete.")
            return
        
        masks_dict, group_labels = multi_group_info
        segment_name = get_config_value(config, "plotting.comparisons.comparison_segment", "active")
        
        from eeg_pipeline.plotting.features.utils import load_multigroup_stats
        multigroup_stats = load_multigroup_stats(stats_dir) if stats_dir else None
        
        for roi_name in roi_names:
            data_by_band: Dict[str, Dict[str, np.ndarray]] = {}
            for band in bands:
                cols = _get_spectral_columns_for_roi(entries, segment_name, band, metric, roi_name)
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
                roi_safe = sanitize_label(roi_name).lower() if roi_name.lower() != "all" else ""
                suffix = f"_roi-{roi_safe}" if roi_safe else ""
                save_path = save_dir / f"sub-{subject}_spectral_{metric}_by_condition{suffix}_multigroup"
                
                plot_multi_group_column_comparison(
                    data_by_band=data_by_band,
                    subject=subject,
                    save_path=save_path,
                    feature_label=f"Spectral ({metric_label})",
                    groups=group_labels,
                    multigroup_stats=multigroup_stats,
                    config=config,
                    logger=logger,
                    roi_name=roi_name,
                    stats_dir=stats_dir,
                )
        
        log_if_present(logger, "info", f"Saved spectral {metric_label} multi-group column comparison for {len(roi_names)} ROIs")
        return
    
    comp_mask_info = extract_comparison_mask(events_df, config, require_enabled=False)
    if not comp_mask_info:
        if logger:
            log_if_present(
                logger, "warning",
                "Column comparison enabled but config incomplete. "
                "Set plotting.comparisons.comparison_column and comparison_values."
            )
        return
    
    mask1, mask2, label1, label2 = comp_mask_info
    segment_name = get_config_value(
        config,
        "plotting.comparisons.comparison_segment",
        "active",
    )
    
    plot_cfg = get_plot_config(config)
    band_colors = {band: get_band_color(band, config) for band in bands}
    n_bands = len(bands)
    n_trials = len(features_df)
    
    for roi_name in roi_names:
        cell_data = {}
        
        for col_idx, band in enumerate(bands):
            cols = _get_spectral_columns_for_roi(
                entries, segment_name, band, metric, roi_name
            )
            
            if not cols:
                cell_data[col_idx] = None
                continue
            
            val_series = features_df[cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
            values1 = val_series[mask1].dropna().values
            values2 = val_series[mask2].dropna().values
            cell_data[col_idx] = {"v1": values1, "v2": values2}
        
        qvalues, n_significant, use_precomputed = compute_or_load_column_stats(
            stats_dir=stats_dir,
            feature_type="spectral",
            feature_keys=bands,
            cell_data=cell_data,
            config=config,
            logger=logger,
        )
        
        fig, axes = plt.subplots(1, n_bands, figsize=(3 * n_bands, 5), squeeze=False)
        
        for col_idx, band in enumerate(bands):
            ax = axes.flatten()[col_idx]
            data = cell_data.get(col_idx)
            
            if data is None:
                values1 = np.array([])
                values2 = np.array([])
                qvalue_info = None
            else:
                values1 = data.get("v1", [])
                values2 = data.get("v2", [])
                qvalue_info = qvalues.get(col_idx) if col_idx in qvalues else None
            
            _plot_column_comparison_single_band(
                ax,
                values1,
                values2,
                band,
                band_colors.get(band, _VIOLIN_COLOR),
                label1,
                label2,
                qvalue_info,
                plot_cfg,
            )
        
        n_tests = len(qvalues)
        roi_display = (
            roi_name.replace("_", " ").title()
            if roi_name.lower() != "all"
            else "All Channels"
        )
        
        stats_source = "pre-computed" if use_precomputed else "Mann-Whitney U"
        title = (
            f"Spectral ({metric_label}): {label1} vs {label2} (Column Comparison)\n"
            f"Subject: {subject} | ROI: {roi_display} | N: {n_trials} trials | "
            f"{stats_source} | FDR: {n_significant}/{n_tests} significant (†=q<0.05)"
        )
        fig.suptitle(
            title,
            fontsize=plot_cfg.font.suptitle,
            fontweight="bold",
            y=_SUBTITLE_Y_OFFSET,
        )
        
        plt.tight_layout()
        
        suffix = _create_roi_suffix(roi_name)
        filename = f"sub-{subject}_spectral_{metric}_by_condition{suffix}_column"
        
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
    
    if logger:
        log_if_present(
            logger,
            "info",
            f"Saved spectral {metric_label} column comparison plots for {len(roi_names)} ROIs",
        )


def plot_spectral_by_condition(
    features_df: pd.DataFrame,
    events_df: pd.DataFrame,
    subject: str,
    save_dir: Path,
    logger: Any = None,
    config: Any = None,
    stats_dir: Optional[Path] = None,
) -> None:
    """Compare spectral features between conditions per band.
    
    For window comparisons (paired): Uses the unified plot_paired_comparison helper.
    For column comparisons (unpaired): Uses Mann-Whitney U test with consistent styling.
    Creates one figure per ROI per metric.
    
    If stats_dir is provided, uses pre-computed statistics from the behavior pipeline.
    """
    from eeg_pipeline.infra.paths import ensure_dir
    from eeg_pipeline.plotting.io.figures import log_if_present
    
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
    
    entries = _collect_spectral_columns(features_df)
    if not entries:
        return
    
    segments, bands, metrics = _get_comparison_configuration(entries, config)
    
    if not bands:
        return
    
    if logger and len(segments) >= _MIN_SEGMENTS_FOR_COMPARISON:
        log_if_present(
            logger,
            "info",
            f"Auto-detected segments for spectral comparison: {segments}",
        )
    
    metric_labels = {m: _format_metric_label(m) for m in metrics}
    roi_names = _get_roi_names(config)
    
    if logger:
        log_if_present(
            logger,
            "info",
            f"Spectral comparison: segments={segments}, ROIs={roi_names}, "
            f"bands={bands}, metrics={metrics}, "
            f"compare_windows={compare_windows}, compare_columns={compare_columns}",
        )
    
    ensure_dir(save_dir)
    
    for metric in metrics:
        metric_label = metric_labels.get(metric, metric.upper())
        
        if compare_windows and len(segments) >= _MIN_SEGMENTS_FOR_COMPARISON:
            _plot_window_comparison(
                features_df,
                entries,
                segments,
                bands,
                metric,
                metric_label,
                roi_names,
                subject,
                save_dir,
                config,
                logger,
                stats_dir,
            )
        
        if compare_columns:
            _plot_column_comparison(
                features_df,
                entries,
                events_df,
                bands,
                metric,
                metric_label,
                roi_names,
                subject,
                save_dir,
                config,
                logger,
                stats_dir,
            )


__all__ = [
    "plot_spectral_summary",
    "plot_spectral_edge_frequency",
    "plot_spectral_by_condition",
]

