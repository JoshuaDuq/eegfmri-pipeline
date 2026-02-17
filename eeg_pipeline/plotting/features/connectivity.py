import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import re
import networkx as nx
from mne_connectivity.viz import plot_connectivity_circle

from eeg_pipeline.infra.paths import ensure_dir
from eeg_pipeline.plotting.io.figures import log_if_present, save_fig
from eeg_pipeline.utils.config.loader import (
    get_config_value,
    get_frequency_band_names,
    require_config_value,
)
from ..config import get_plot_config
from eeg_pipeline.utils.analysis.connectivity import (
    build_adjacency_from_edges,
    compute_significant_edges,
    parse_connectivity_columns,
)


@lru_cache(maxsize=256)
def _parse_connectivity_columns_cached(
    columns: Tuple[str, ...],
    measure: str,
    band: str,
    segment: Optional[str] = None,
) -> Tuple[Tuple[str, ...], Tuple[Tuple[str, str], ...]]:
    """Parse and cache connectivity columns for a measure and band."""
    cols, edges, _ = parse_connectivity_columns(list(columns), measure, band, segment=segment)
    return tuple(cols), tuple(edges)


def _filter_non_self_edges(
    columns: List[str],
    edges: List[Tuple[str, str]],
) -> Tuple[List[str], List[Tuple[str, str]]]:
    """Filter out self-connections (ch1 == ch2) from connectivity data."""
    filtered_columns = [col for col, edge in zip(columns, edges) if edge[0] != edge[1]]
    filtered_edges = [edge for edge in edges if edge[0] != edge[1]]
    return filtered_columns, filtered_edges


def _extract_unique_nodes(edges: List[Tuple[str, str]]) -> List[str]:
    """Extract sorted unique channel names from edge list."""
    unique_nodes = set()
    for ch1, ch2 in edges:
        unique_nodes.update([ch1, ch2])
    return sorted(list(unique_nodes))


def _get_channel_order(
    edges: List[Tuple[str, str]],
    info: Optional[mne.Info],
) -> Optional[List[str]]:
    """Get ordered channel list from edges, using info if available."""
    unique_nodes = _extract_unique_nodes(edges)
    if not unique_nodes:
        return None
    
    if info is not None:
        channel_order = [ch for ch in info.ch_names if ch in unique_nodes]
        return channel_order if channel_order else None
    
    return unique_nodes


def _build_connectivity_matrix(
    mean_connectivity: np.ndarray,
    edges: List[Tuple[str, str]],
    node_names: List[str],
    threshold: float,
) -> Tuple[np.ndarray, int]:
    """Build symmetric connectivity matrix from edge values above threshold."""
    n_nodes = len(node_names)
    node_indices = {name: idx for idx, name in enumerate(node_names)}
    matrix = np.zeros((n_nodes, n_nodes))
    n_significant = 0
    
    for value, (ch1, ch2) in zip(mean_connectivity, edges):
        if abs(value) >= threshold and ch1 in node_indices and ch2 in node_indices:
            idx1 = node_indices[ch1]
            idx2 = node_indices[ch2]
            matrix[idx1, idx2] = value
            matrix[idx2, idx1] = value
            n_significant += 1
    
    return matrix, n_significant


def _get_connectivity_colormap_and_range(measure: str) -> Tuple[str, Optional[float], Optional[float]]:
    """Get appropriate colormap and value range for connectivity measure."""
    measure_lower = measure.lower()
    if "wpli" in measure_lower or "pli" in measure_lower or "coherence" in measure_lower:
        return "viridis", 0.0, 1.0
    return "RdBu", None, None


def _filter_connectivity_columns_by_roi(
    columns: List[str],
    roi_name: str,
    roi_definitions: Dict[str, Any],
    all_features_columns: List[str],
) -> List[str]:
    """Filter connectivity columns to within-ROI edges."""
    if roi_name == "all" or roi_name not in roi_definitions:
        return columns
    
    from eeg_pipeline.plotting.features.roi import get_roi_channels
    
    channel_pattern = re.compile(r'_chpair_([^_]+)_([^_]+)_')
    all_channel_names = set()
    for col in all_features_columns:
        match = channel_pattern.search(str(col))
        if match:
            all_channel_names.add(match.group(1))
            all_channel_names.add(match.group(2))
    
    roi_channels = set(get_roi_channels(roi_definitions[roi_name], list(all_channel_names)))
    
    filtered_columns = []
    for col in columns:
        match = channel_pattern.search(str(col))
        if match:
            ch1, ch2 = match.group(1), match.group(2)
            if ch1 in roi_channels and ch2 in roi_channels:
                filtered_columns.append(col)
    
    return filtered_columns if filtered_columns else columns


def _get_roi_names_for_comparison(
    config: Any,
    roi_definitions: Dict[str, Any],
) -> List[str]:
    """Get list of ROI names to use for comparison plots."""
    comparison_rois = get_config_value(config, "plotting.comparisons.comparison_rois", [])
    if comparison_rois:
        roi_names = []
        for roi in comparison_rois:
            if roi.lower() == "all":
                if "all" not in roi_names:
                    roi_names.append("all")
            elif roi in roi_definitions:
                roi_names.append(roi)
        return roi_names
    
    roi_names = ["all"]
    if roi_definitions:
        roi_names.extend(list(roi_definitions.keys()))
    return roi_names


def _detect_segments_from_data(
    features_df: pd.DataFrame,
    config: Any,
    logger: Optional[logging.Logger],
) -> List[str]:
    """Resolve segments from config (no auto-detection)."""
    segments = require_config_value(config, "plotting.comparisons.comparison_windows")
    if not isinstance(segments, (list, tuple)) or len(segments) < 2:
        raise ValueError(
            "plotting.comparisons.comparison_windows must be a list/tuple with at least 2 window names "
            f"(got {segments!r})"
        )
    return [str(s) for s in segments]


def _plot_window_comparison_connectivity(
    features_df: pd.DataFrame,
    segments: List[str],
    measures: List[str],
    bands: List[str],
    roi_names: List[str],
    roi_definitions: Dict[str, Any],
    subject: str,
    save_dir: Path,
    config: Any,
    logger: logging.Logger,
    stats_dir: Optional[Path],
) -> None:
    """Plot paired window comparison for connectivity measures.
    
    Supports both 2-window comparison (simple paired) and multi-window comparison
    (3+ windows with all pairwise brackets and significance asterisks).
    """
    from eeg_pipeline.plotting.features.utils import plot_paired_comparison, plot_multi_window_comparison
    from eeg_pipeline.utils.formatting import sanitize_label
    
    use_multi_window = len(segments) > 2
    
    for roi_name in roi_names:
        for measure in measures:
            measure_safe = measure.lower()
            roi_safe = sanitize_label(roi_name).lower() if roi_name != "all" else ""
            suffix = f"_roi-{roi_safe}" if roi_safe else ""
            
            if use_multi_window:
                # Multi-window comparison: extract data for all segments
                data_by_band_multi: Dict[str, Dict[str, np.ndarray]] = {}
                for band in bands:
                    segment_series = {}
                    for seg in segments:
                        cols, _, _ = parse_connectivity_columns(
                            list(features_df.columns), measure, band, segment=seg
                        )
                        cols = _filter_connectivity_columns_by_roi(
                            cols, roi_name, roi_definitions, list(features_df.columns)
                        )
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
                    save_path = save_dir / (
                        f"sub-{subject}_connectivity_{measure_safe}_by_condition{suffix}_multiwindow"
                    )
                    plot_multi_window_comparison(
                        data_by_band=data_by_band_multi,
                        subject=subject,
                        save_path=save_path,
                        feature_label=f"Connectivity ({measure.upper()})",
                        segments=segments,
                        config=config,
                        logger=logger,
                        roi_name=roi_name,
                        stats_dir=stats_dir,
                    )
            else:
                # 2-window comparison
                segment1, segment2 = segments[0], segments[1]
                data_by_band = {}
                for band in bands:
                    cols1, _, _ = parse_connectivity_columns(
                        list(features_df.columns), measure, band, segment=segment1
                    )
                    cols2, _, _ = parse_connectivity_columns(
                        list(features_df.columns), measure, band, segment=segment2
                    )
                    
                    cols1 = _filter_connectivity_columns_by_roi(
                        cols1, roi_name, roi_definitions, list(features_df.columns)
                    )
                    cols2 = _filter_connectivity_columns_by_roi(
                        cols2, roi_name, roi_definitions, list(features_df.columns)
                    )
                    
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
                    save_path = save_dir / (
                        f"sub-{subject}_connectivity_{measure_safe}_by_condition{suffix}_window"
                    )
                    plot_paired_comparison(
                        data_by_band=data_by_band,
                        subject=subject,
                        save_path=save_path,
                        feature_label=f"Connectivity ({measure.upper()})",
                        config=config,
                        logger=logger,
                        label1=segment1.capitalize(),
                        label2=segment2.capitalize(),
                        roi_name=roi_name,
                        stats_dir=stats_dir,
                    )
    
    plot_type = "multi-window" if use_multi_window else "paired"
    log_if_present(logger, "info", 
                  f"Saved connectivity {plot_type} comparison plots for "
                  f"{len(measures)} measures × {len(roi_names)} ROIs")


def plot_connectivity_circle_by_condition(
    features_df: pd.DataFrame,
    events_df: pd.DataFrame,
    info: mne.Info,
    subject: str,
    save_dir: Path,
    logger: logging.Logger,
    config: Any,
    measure: str = "wpli",
    band: str = "alpha",
    n_lines: Optional[int] = None,
    significance_threshold: Optional[float] = None,
) -> None:
    """Plot connectivity circle diagrams for conditions (supports 2+ groups)."""
    if features_df is None or features_df.empty or events_df is None:
        log_if_present(logger, "warning", "No feature data for connectivity plot")
        return

    from eeg_pipeline.utils.analysis.events import extract_multi_group_masks
    
    multi_group_info = extract_multi_group_masks(events_df, config, require_enabled=True)
    if not multi_group_info:
        raise ValueError("Connectivity circle plot requested but could not resolve group masks.")
    masks_dict, group_labels = multi_group_info
    conditions = [(label, mask) for label, mask in masks_dict.items()]
    
    plot_cfg = get_plot_config(config)
    group_colors = plt.cm.Set2(np.linspace(0, 1, max(len(conditions), 3)))
    condition_colors = {label: group_colors[i] for i, (label, _) in enumerate(conditions)}

    columns_tuple, edges_tuple = _parse_connectivity_columns_cached(
        tuple(features_df.columns), measure, band
    )
    columns, edges = _filter_non_self_edges(list(columns_tuple), list(edges_tuple))
    
    if not columns:
        log_if_present(logger, "debug", 
                      f"No channel-pair connectivity columns found for {measure} {band}")
        return
    
    node_names = _extract_unique_nodes(edges)
    n_nodes = len(node_names)
    n_edges = len(edges)
    
    default_top_fraction = float(get_config_value(
        config, "plotting.plots.features.connectivity.circle_top_fraction", 0.1
    ))
    top_fraction = (significance_threshold if significance_threshold is not None 
                   else default_top_fraction)
    
    pooled_connectivity = features_df[columns].mean(axis=0).values
    absolute_connectivity = np.abs(pooled_connectivity)
    threshold = np.percentile(absolute_connectivity, (1 - top_fraction) * 100)
    
    def build_matrix_for_condition(condition_mask: np.ndarray) -> Tuple[np.ndarray, int]:
        """Build connectivity matrix for a specific condition."""
        mean_connectivity = features_df.loc[condition_mask, columns].mean(axis=0).values
        return _build_connectivity_matrix(mean_connectivity, edges, node_names, threshold)
    
    colormap, vmin, vmax = _get_connectivity_colormap_and_range(measure)
    if vmin is None:
        vmin, vmax = 0.0, 1.0
        colormap = "viridis"
    
    min_lines_config = int(get_config_value(
        config, "plotting.plots.features.connectivity.circle_min_lines", 20
    ))
    
    width_per_circle = float(plot_cfg.plot_type_configs.get("connectivity", {})
                             .get("width_per_circle", 9.0))
    
    title_base = (
        f"{measure.upper()} Connectivity: {band.capitalize()} Band\n"
        f"Subject: {subject} | Top {int(top_fraction*100)}% connections "
        f"(threshold ≥ {threshold:.3f})"
    )
    
    footer_text = (
        f"{n_nodes} nodes | {n_edges} total edges | "
        f"Showing connections ≥ {threshold:.3f}"
    )
    
    from eeg_pipeline.utils.formatting import sanitize_label
    
    def save_condition_circle(condition_matrix, condition_label, condition_color, n_trials, n_sig_edges):
        """Create and save a single connectivity circle for one condition."""
        n_lines_to_show = (n_lines if n_lines is not None 
                           else max(min_lines_config, n_sig_edges))
        
        fig, ax = plt.subplots(figsize=(width_per_circle, width_per_circle), 
                              subplot_kw=dict(polar=True))
        
        try:
            plot_connectivity_circle(
                condition_matrix, node_names, n_lines=n_lines_to_show, ax=ax,
                title="", show=False,
                vmin=vmin, vmax=vmax, colorbar=True, colormap=colormap
            )
            ax.set_title(
                f"{condition_label}\n(n={n_trials} trials, {n_sig_edges} edges)",
                fontsize=plot_cfg.font.suptitle,
                fontweight="bold",
                color=condition_color,
            )
        except Exception as e:
            log_if_present(logger, "error", f"Failed to plot {condition_label}: {e}")
            plt.close(fig)
            return
        
        fig.suptitle(title_base, fontsize=plot_cfg.font.figure_title, 
                    fontweight="bold", y=0.98)
        fig.text(0.5, 0.02, footer_text, ha='center', va='bottom', 
                fontsize=plot_cfg.font.large, color='gray')
        
        condition_safe = sanitize_label(condition_label).lower().replace(" ", "_")
        output_name = f"sub-{subject}_connectivity_{measure}_{band}_circle_{condition_safe}"
        save_fig(
            fig, save_dir / output_name,
            formats=plot_cfg.formats, dpi=plot_cfg.dpi,
            bbox_inches=plot_cfg.bbox_inches, pad_inches=plot_cfg.pad_inches, config=config
        )
        plt.close(fig)
    
    for condition_label, condition_mask in conditions:
        n_samples = min(len(features_df), len(condition_mask))
        if n_samples <= 0:
            continue
        
        condition_mask_array = np.asarray(condition_mask[:n_samples], dtype=bool)
        
        if int(condition_mask_array.sum()) == 0:
            continue
        
        condition_matrix, n_sig_edges = build_matrix_for_condition(condition_mask_array)
        n_trials = int(condition_mask_array.sum())
        condition_color = condition_colors.get(condition_label, plot_cfg.get_color("blue"))
        
        save_condition_circle(
            condition_matrix, condition_label, condition_color,
            n_trials, n_sig_edges
        )
    
    log_if_present(logger, "info", 
                  f"Saved {measure} {band} connectivity circles by condition")


def _plot_column_comparison_connectivity(
    features_df: pd.DataFrame,
    events_df: pd.DataFrame,
    measures: List[str],
    bands: List[str],
    roi_names: List[str],
    roi_definitions: Dict[str, Any],
    subject: str,
    save_dir: Path,
    config: Any,
    logger: logging.Logger,
    stats_dir: Optional[Path],
) -> None:
    """Plot unpaired column comparison for connectivity measures.
    
    Supports both 2-group comparison (simple unpaired) and multi-group comparison
    (3+ groups with all pairwise brackets and significance asterisks).
    """
    from eeg_pipeline.utils.analysis.events import extract_comparison_mask, extract_multi_group_masks
    from eeg_pipeline.plotting.features.utils import compute_or_load_column_stats, get_band_color, plot_multi_group_column_comparison
    from eeg_pipeline.utils.formatting import sanitize_label
    
    values_spec = get_config_value(config, "plotting.comparisons.comparison_values", [])
    use_multi_group = isinstance(values_spec, (list, tuple)) and len(values_spec) > 2
    
    if use_multi_group:
        multi_group_info = extract_multi_group_masks(events_df, config, require_enabled=True)
        if not multi_group_info:
            raise ValueError("Multi-group column comparison requested but could not resolve group masks.")
        
        masks_dict, group_labels = multi_group_info
        segment = str(require_config_value(config, "plotting.comparisons.comparison_segment")).strip()
        
        from eeg_pipeline.plotting.features.utils import load_multigroup_stats
        multigroup_stats = load_multigroup_stats(stats_dir) if stats_dir else None
        
        for roi_name in roi_names:
            for measure in measures:
                data_by_band: Dict[str, Dict[str, np.ndarray]] = {}
                for band in bands:
                    columns, _, _ = parse_connectivity_columns(
                        list(features_df.columns), measure, band, segment=segment
                    )
                    columns = _filter_connectivity_columns_by_roi(
                        columns, roi_name, roi_definitions, list(features_df.columns)
                    )
                    
                    if not columns:
                        continue
                    
                    value_series = features_df[columns].apply(pd.to_numeric, errors="coerce").mean(axis=1)
                    
                    group_values = {}
                    for label, mask in masks_dict.items():
                        vals = value_series[mask].dropna().values
                        if len(vals) > 0:
                            group_values[label] = vals
                    
                    if len(group_values) >= 2:
                        data_by_band[band] = group_values
                
                if data_by_band:
                    roi_safe = sanitize_label(roi_name).lower() if roi_name != "all" else ""
                    suffix = f"_roi-{roi_safe}" if roi_safe else ""
                    save_path = save_dir / f"sub-{subject}_connectivity_{measure}_by_condition{suffix}_multigroup"
                    
                    plot_multi_group_column_comparison(
                        data_by_band=data_by_band,
                        subject=subject,
                        save_path=save_path,
                        feature_label=f"Connectivity ({measure.upper()})",
                        groups=group_labels,
                        config=config,
                        logger=logger,
                        roi_name=roi_name,
                        stats_dir=stats_dir,
                        multigroup_stats=multigroup_stats,
                    )
        
        log_if_present(logger, "info", f"Saved connectivity multi-group column comparison for {len(roi_names)} ROIs")
        return
    
    comparison_info = extract_comparison_mask(events_df, config, require_enabled=True)
    if not comparison_info:
        raise ValueError("Column comparison requested but could not resolve comparison masks.")
    
    mask1, mask2, label1, label2 = comparison_info
    segment = str(require_config_value(config, "plotting.comparisons.comparison_segment")).strip()
    
    plot_cfg = get_plot_config(config)
    segment_colors = {"v1": "#5a7d9a", "v2": "#c44e52"}
    band_colors = {band: get_band_color(band, config) for band in bands}
    n_bands = len(bands)
    n_trials = len(features_df)
    
    for roi_name in roi_names:
        for measure in measures:
            cell_data = {}
            for band_idx, band in enumerate(bands):
                columns, _, _ = parse_connectivity_columns(
                    list(features_df.columns), measure, band, segment=segment
                )
                columns = _filter_connectivity_columns_by_roi(
                    columns, roi_name, roi_definitions, list(features_df.columns)
                )
                
                if not columns:
                    cell_data[band_idx] = None
                    continue
                
                value_series = features_df[columns].apply(
                    pd.to_numeric, errors="coerce"
                ).mean(axis=1)
                values1 = value_series[mask1].dropna().values
                values2 = value_series[mask2].dropna().values
                
                cell_data[band_idx] = {"v1": values1, "v2": values2}
            
            qvalues, n_significant, use_precomputed = compute_or_load_column_stats(
                stats_dir=stats_dir,
                feature_type="connectivity",
                feature_keys=bands,
                cell_data=cell_data,
                config=config,
                logger=logger,
            )
            
            fig, axes = plt.subplots(1, n_bands, figsize=(3 * n_bands, 5), squeeze=False)
            
            for band_idx, band in enumerate(bands):
                ax = axes.flatten()[band_idx]
                data = cell_data.get(band_idx)
                
                if data is None or len(data.get("v1", [])) == 0 or len(data.get("v2", [])) == 0:
                    ax.text(0.5, 0.5, "No data", ha="center", va="center",
                           transform=ax.transAxes, fontsize=plot_cfg.font.title, color="gray")
                    ax.set_xticks([])
                    continue
                
                values1, values2 = data["v1"], data["v2"]
                
                boxplot = ax.boxplot([values1, values2], positions=[0, 1], 
                                    widths=0.4, patch_artist=True)
                boxplot["boxes"][0].set_facecolor(segment_colors["v1"])
                boxplot["boxes"][0].set_alpha(0.6)
                boxplot["boxes"][1].set_facecolor(segment_colors["v2"])
                boxplot["boxes"][1].set_alpha(0.6)
                
                jitter_range = 0.08
                rng = np.random.default_rng(42)
                ax.scatter(rng.uniform(-jitter_range, jitter_range, len(values1)), 
                          values1, c=segment_colors["v1"], alpha=0.3, s=6)
                ax.scatter(1 + rng.uniform(-jitter_range, jitter_range, len(values2)), 
                          values2, c=segment_colors["v2"], alpha=0.3, s=6)
                
                all_values = np.concatenate([values1, values2])
                ymin, ymax = np.nanmin(all_values), np.nanmax(all_values)
                yrange = ymax - ymin if ymax > ymin else 0.1
                ax.set_ylim(ymin - 0.1 * yrange, ymax + 0.3 * yrange)
                
                if band_idx in qvalues:
                    _, qvalue, effect_size, is_significant = qvalues[band_idx]
                    sig_marker = "†" if is_significant else ""
                    sig_color = "#d62728" if is_significant else "#333333"
                    annotation_text = f"q={qvalue:.3f}{sig_marker}\nd={effect_size:.2f}"
                    ax.annotate(annotation_text, xy=(0.5, ymax + 0.05 * yrange),
                               ha="center", fontsize=plot_cfg.font.medium, color=sig_color,
                               fontweight="bold" if is_significant else "normal")
                
                ax.set_xticks([0, 1])
                ax.set_xticklabels([label1, label2], fontsize=9)
                ax.set_title(band.capitalize(), fontweight="bold", 
                           color=band_colors.get(band, "gray"))
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
            
            n_tests = len(qvalues)
            roi_display = roi_name.replace("_", " ").title() if roi_name != "all" else "All Edges"
            stats_source = "pre-computed" if use_precomputed else "Mann-Whitney U"
            title_text = (
                f"Connectivity ({measure.upper()}): {label1} vs {label2} (Column Comparison)\n"
                f"Subject: {subject} | ROI: {roi_display} | N: {n_trials} trials | "
                f"{stats_source} | FDR: {n_significant}/{n_tests} significant (†=q<0.05)"
            )
            fig.suptitle(title_text, fontsize=plot_cfg.font.suptitle, 
                        fontweight="bold", y=1.02)
            
            plt.tight_layout()
            
            from eeg_pipeline.utils.formatting import sanitize_label
            measure_safe = measure.lower()
            roi_safe = sanitize_label(roi_name).lower() if roi_name != "all" else ""
            suffix = f"_roi-{roi_safe}" if roi_safe else ""
            filename = f"sub-{subject}_connectivity_{measure_safe}_by_condition{suffix}_column"
            
            save_fig(fig, save_dir / filename, formats=plot_cfg.formats, dpi=plot_cfg.dpi,
                     bbox_inches=plot_cfg.bbox_inches, pad_inches=plot_cfg.pad_inches, config=config)
            plt.close(fig)
    
    log_if_present(logger, "info", 
                  f"Saved connectivity column comparison plots for "
                  f"{len(measures)} measures × {len(roi_names)} ROIs")


def plot_connectivity_by_condition(
    features_df: pd.DataFrame,
    events_df: pd.DataFrame,
    subject: str,
    save_dir: Path,
    logger: logging.Logger,
    config: Any,
    stats_dir: Optional[Path] = None,
) -> None:
    """Connectivity comparison by Measure × Band.
    
    For window comparisons (paired): Uses the unified plot_paired_comparison helper.
    For column comparisons (unpaired): Uses Mann-Whitney U test with consistent styling.
    Creates one figure per connectivity measure per ROI.
    
    If stats_dir is provided, uses pre-computed statistics from the behavior pipeline.
    """
    from eeg_pipeline.plotting.features.roi import get_roi_definitions
    
    if features_df is None or features_df.empty:
        return

    compare_windows = get_config_value(config, "plotting.comparisons.compare_windows", True)
    compare_columns = get_config_value(config, "plotting.comparisons.compare_columns", False)
    
    segments = _detect_segments_from_data(features_df, config, logger)
    measures = get_config_value(
        config, "plotting.plots.features.connectivity.measures", 
        ["aec", "wpli", "pli", "plv", "coherence"]
    )
    bands = list(get_frequency_band_names(config) or ['theta', 'alpha', 'beta', 'gamma'])
    
    roi_definitions = get_roi_definitions(config)
    roi_names = _get_roi_names_for_comparison(config, roi_definitions)
    
    if logger:
        logger.info(
            f"Connectivity comparison: segments={segments}, ROIs={roi_names}, "
            f"measures={measures}, compare_windows={compare_windows}, "
            f"compare_columns={compare_columns}"
        )
    
    if compare_windows and segments and len(segments) >= 2:
        _plot_window_comparison_connectivity(
            features_df, segments, measures, bands, roi_names, roi_definitions,
            subject, save_dir, config, logger, stats_dir
        )

    if compare_columns:
        _plot_column_comparison_connectivity(
            features_df, events_df, measures, bands, roi_names, roi_definitions,
            subject, save_dir, config, logger, stats_dir
        )


def plot_connectivity_heatmap(
    features_df: pd.DataFrame,
    info: mne.Info,
    subject: str,
    save_dir: Path,
    logger: logging.Logger,
    config: Any,
    measure: str = "wpli",
    band: str = "alpha",
    events_df: Optional[pd.DataFrame] = None,
) -> None:
    """Plot connectivity heatmap for a given measure and band."""
    columns_all, edges_all, _ = parse_connectivity_columns(
        list(features_df.columns), measure, band
    )
    columns, edges = _filter_non_self_edges(columns_all, edges_all)
    
    if not columns:
        log_if_present(logger, "debug", 
                      f"No channel-pair connectivity columns for {measure} {band} heatmap")
        return

    channel_order = _get_channel_order(edges, info)
    if not channel_order:
        log_if_present(logger, "debug", 
                      f"No channel names found for {measure} {band}")
        return
        
    adjacency_matrix = build_adjacency_from_edges(
        features_df, columns, channel_order, edges=edges
    )
    if not np.any(np.isfinite(adjacency_matrix)):
        return

    significant_edges = compute_significant_edges(features_df, columns, events_df, config)
    plot_cfg = get_plot_config(config)
    vmax = (float(np.nanmax(np.abs(adjacency_matrix))) 
            if np.any(np.isfinite(adjacency_matrix)) else 1.0)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(adjacency_matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(channel_order)))
    ax.set_yticks(range(len(channel_order)))
    ax.set_xticklabels(channel_order, rotation=90, fontsize=plot_cfg.font.annotation)
    ax.set_yticklabels(channel_order, fontsize=plot_cfg.font.annotation)
    ax.set_title(f"{measure.upper()} {band.capitalize()} mean connectivity (sub-{subject})")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Connectivity")

    if significant_edges:
        significant_mask = np.zeros_like(adjacency_matrix, dtype=bool)
        edge_to_indices = {
            (ch1, ch2): (channel_order.index(ch1), channel_order.index(ch2))
            for ch1, ch2 in edges 
            if ch1 in channel_order and ch2 in channel_order
        }
        for col in significant_edges:
            if col in columns:
                col_index = columns.index(col)
                edge = edges[col_index]
                if edge in edge_to_indices:
                    i, j = edge_to_indices[edge]
                    significant_mask[i, j] = significant_mask[j, i] = True
        if np.any(significant_mask):
            ax.contour(significant_mask, colors="k", levels=[0.5], linewidths=0.5)

    ensure_dir(save_dir)
    output_name = save_dir / f"sub-{subject}_connectivity_heatmap_{measure}_{band}"
    save_fig(
        fig,
        output_name,
        formats=plot_cfg.formats,
        dpi=plot_cfg.dpi,
        bbox_inches=plot_cfg.bbox_inches,
        pad_inches=plot_cfg.pad_inches,
        config=config,
    )
    plt.close(fig)
    log_if_present(logger, "info", f"Saved connectivity heatmap for {measure} {band}")


def plot_connectivity_network(
    features_df: pd.DataFrame,
    info: mne.Info,
    subject: str,
    save_dir: Path,
    logger: logging.Logger,
    config: Any,
    measure: str = "wpli",
    band: str = "alpha",
    events_df: Optional[pd.DataFrame] = None,
) -> None:
    """Plot connectivity network graph for a given measure and band."""
    if events_df is not None:
        plot_connectivity_network_by_condition(
            features_df, events_df, info, subject, save_dir, logger, config,
            measure=measure, band=band
        )
        return
    
    columns_all, edges_all, _ = parse_connectivity_columns(
        list(features_df.columns), measure, band
    )
    columns, edges = _filter_non_self_edges(columns_all, edges_all)
    
    if not columns:
        log_if_present(logger, "debug", 
                      f"No channel-pair connectivity columns for {measure} {band} network")
        return

    channel_order = _get_channel_order(edges, info)
    if not channel_order:
        log_if_present(logger, "debug", 
                      f"No channel names found for {measure} {band}")
        return
        
    adjacency_matrix = build_adjacency_from_edges(
        features_df, columns, channel_order, edges=edges
    )
    if not np.any(np.isfinite(adjacency_matrix)):
        return

    default_top_fraction = float(get_config_value(
        config, "plotting.plots.features.connectivity.network_top_fraction", 0.0
    ))
    if default_top_fraction > 0:
        absolute_weights = np.abs(adjacency_matrix)
        upper_triangle = np.triu(absolute_weights, k=1)
        finite_weights = upper_triangle[np.isfinite(upper_triangle)]
        if len(finite_weights) > 0:
            threshold = np.percentile(finite_weights, (1 - default_top_fraction) * 100)
        else:
            threshold = 0.0
    else:
        threshold = 0.0

    significant_edges = compute_significant_edges(features_df, columns, events_df, config)
    significant_set = significant_edges if isinstance(significant_edges, set) else set()

    plot_cfg = get_plot_config(config)
    graph = nx.Graph()
    for i, channel_i in enumerate(channel_order):
        graph.add_node(channel_i)
        for j in range(i + 1, len(channel_order)):
            weight = adjacency_matrix[i, j]
            if np.isfinite(weight) and np.abs(weight) >= threshold:
                graph.add_edge(channel_i, channel_order[j], weight=float(weight))

    if graph.number_of_edges() == 0:
        return

    node_positions = nx.spring_layout(graph, seed=42)
    edge_weights = np.array(
        [data["weight"] for _, _, data in graph.edges(data=True)], dtype=float
    )
    vmax = float(np.nanmax(np.abs(edge_weights))) if edge_weights.size else 1.0

    fig, ax = plt.subplots(figsize=(6, 5))
    graph_edges = list(graph.edges())
    edge_colors = [graph[u][v]["weight"] for u, v in graph_edges]
    node_color = plot_cfg.get_color("network_node")
    
    nx.draw_networkx_nodes(graph, node_positions, node_size=120, 
                          node_color=node_color, alpha=0.8, ax=ax)
    nx.draw_networkx_edges(
        graph,
        node_positions,
        edgelist=graph_edges,
        edge_color=edge_colors,
        edge_cmap=plt.cm.RdBu_r,
        edge_vmin=-vmax,
        edge_vmax=vmax,
        width=2.0,
        alpha=0.7,
        ax=ax,
    )
    nx.draw_networkx_labels(graph, node_positions, font_size=7, 
                           font_weight="bold", ax=ax)

    if significant_set:
        column_to_edge = {col: edges[i] for i, col in enumerate(columns)}
        highlight_edges = []
        for col in significant_set:
            if col in column_to_edge:
                ch1, ch2 = column_to_edge[col]
                if graph.has_edge(ch1, ch2):
                    highlight_edges.append((ch1, ch2))
                elif graph.has_edge(ch2, ch1):
                    highlight_edges.append((ch2, ch1))
        if highlight_edges:
            nx.draw_networkx_edges(
                graph, node_positions, edgelist=highlight_edges, 
                edge_color="lime", width=3.0, alpha=0.9, ax=ax
            )

    scalar_mappable = plt.cm.ScalarMappable(
        cmap="RdBu_r", norm=plt.Normalize(vmin=-vmax, vmax=vmax)
    )
    scalar_mappable.set_array([])
    cbar = plt.colorbar(scalar_mappable, ax=ax)
    cbar.set_label("Connectivity")
    
    title_text = f"Connectivity network ({measure.upper()} {band.capitalize()}, sub-{subject})"
    if default_top_fraction > 0:
        title_text += f" | Top {int(default_top_fraction*100)}% (threshold ≥ {threshold:.3f})"
    ax.set_title(title_text)
    ax.axis("off")

    ensure_dir(save_dir)
    output_name = save_dir / f"sub-{subject}_connectivity_network_{measure}_{band}"
    save_fig(
        fig,
        output_name,
        formats=plot_cfg.formats,
        config=config,
        dpi=plot_cfg.dpi,
        bbox_inches=plot_cfg.bbox_inches,
        pad_inches=plot_cfg.pad_inches,
    )
    plt.close(fig)
    log_if_present(logger, "info", f"Saved connectivity network for {measure} {band}")


def plot_connectivity_network_by_condition(
    features_df: pd.DataFrame,
    events_df: pd.DataFrame,
    info: mne.Info,
    subject: str,
    save_dir: Path,
    logger: logging.Logger,
    config: Any,
    measure: str = "wpli",
    band: str = "alpha",
) -> None:
    """Plot connectivity network graphs for conditions (supports 2+ groups)."""
    if features_df is None or features_df.empty or events_df is None:
        log_if_present(logger, "warning", "No feature data for connectivity network plot")
        return

    from eeg_pipeline.utils.analysis.events import extract_multi_group_masks
    
    multi_group_info = extract_multi_group_masks(events_df, config, require_enabled=True)
    if not multi_group_info:
        raise ValueError("Connectivity network plot requested but could not resolve group masks.")
    masks_dict, group_labels = multi_group_info
    conditions = [(label, mask) for label, mask in masks_dict.items()]
    
    plot_cfg = get_plot_config(config)
    group_colors = plt.cm.Set2(np.linspace(0, 1, max(len(conditions), 3)))
    condition_colors = {label: group_colors[i] for i, (label, _) in enumerate(conditions)}

    columns_all, edges_all, _ = parse_connectivity_columns(
        list(features_df.columns), measure, band
    )
    columns, edges = _filter_non_self_edges(columns_all, edges_all)
    
    if not columns:
        log_if_present(logger, "debug", 
                      f"No channel-pair connectivity columns for {measure} {band} network")
        return

    channel_order = _get_channel_order(edges, info)
    if not channel_order:
        log_if_present(logger, "debug", 
                      f"No channel names found for {measure} {band}")
        return
    
    default_top_fraction = float(get_config_value(
        config, "plotting.plots.features.connectivity.network_top_fraction", 0.0
    ))
    
    pooled_connectivity = features_df[columns].mean(axis=0).values
    absolute_connectivity = np.abs(pooled_connectivity)
    if default_top_fraction > 0:
        threshold = np.percentile(absolute_connectivity, (1 - default_top_fraction) * 100)
    else:
        threshold = 0.0
    
    def build_network_for_condition(condition_mask: np.ndarray) -> Optional[nx.Graph]:
        """Build network graph for a specific condition."""
        condition_features = features_df.loc[condition_mask]
        if len(condition_features) == 0:
            return None
            
        adjacency_matrix = build_adjacency_from_edges(
            condition_features, columns, channel_order, edges=edges
        )
        if not np.any(np.isfinite(adjacency_matrix)):
            return None

        graph = nx.Graph()
        for i, channel_i in enumerate(channel_order):
            graph.add_node(channel_i)
            for j in range(i + 1, len(channel_order)):
                weight = adjacency_matrix[i, j]
                if np.isfinite(weight) and np.abs(weight) >= threshold:
                    graph.add_edge(channel_i, channel_order[j], weight=float(weight))
        
        return graph if graph.number_of_edges() > 0 else None
    
    from eeg_pipeline.utils.formatting import sanitize_label
    
    def save_condition_network(condition_graph, condition_label, condition_color, n_trials):
        """Create and save a single connectivity network for one condition."""
        if condition_graph is None:
            return
        
        node_positions = nx.spring_layout(condition_graph, seed=42)
        edge_weights = np.array(
            [data["weight"] for _, _, data in condition_graph.edges(data=True)], dtype=float
        )
        vmax = float(np.nanmax(np.abs(edge_weights))) if edge_weights.size else 1.0

        fig, ax = plt.subplots(figsize=(6, 5))
        graph_edges = list(condition_graph.edges())
        edge_colors = [condition_graph[u][v]["weight"] for u, v in graph_edges]
        node_color = plot_cfg.get_color("network_node")
        
        nx.draw_networkx_nodes(condition_graph, node_positions, node_size=120, 
                              node_color=node_color, alpha=0.8, ax=ax)
        nx.draw_networkx_edges(
            condition_graph,
            node_positions,
            edgelist=graph_edges,
            edge_color=edge_colors,
            edge_cmap=plt.cm.RdBu_r,
            edge_vmin=-vmax,
            edge_vmax=vmax,
            width=2.0,
            alpha=0.7,
            ax=ax,
        )
        nx.draw_networkx_labels(condition_graph, node_positions, font_size=7, 
                               font_weight="bold", ax=ax)

        scalar_mappable = plt.cm.ScalarMappable(
            cmap="RdBu_r", norm=plt.Normalize(vmin=-vmax, vmax=vmax)
        )
        scalar_mappable.set_array([])
        cbar = plt.colorbar(scalar_mappable, ax=ax)
        cbar.set_label("Connectivity")
        
        title_text = (
            f"Connectivity Network ({measure.upper()} {band.capitalize()})\n"
            f"Subject: {subject} | {condition_label} (n={n_trials} trials)"
        )
        if default_top_fraction > 0:
            title_text += f" | Top {int(default_top_fraction*100)}% (threshold ≥ {threshold:.3f})"
        ax.set_title(title_text, fontsize=plot_cfg.font.figure_title, 
                    fontweight="bold", color=condition_color)
        ax.axis("off")

        condition_safe = sanitize_label(condition_label).lower().replace(" ", "_")
        output_name = f"sub-{subject}_connectivity_network_{measure}_{band}_{condition_safe}"
        save_fig(
            fig, save_dir / output_name,
            formats=plot_cfg.formats, dpi=plot_cfg.dpi,
            bbox_inches=plot_cfg.bbox_inches, pad_inches=plot_cfg.pad_inches, config=config
        )
        plt.close(fig)
    
    for condition_label, condition_mask in conditions:
        n_samples = min(len(features_df), len(condition_mask))
        if n_samples <= 0:
            continue
        
        condition_mask_array = np.asarray(condition_mask[:n_samples], dtype=bool)
        
        if int(condition_mask_array.sum()) == 0:
            continue
        
        condition_graph = build_network_for_condition(condition_mask_array)
        n_trials = int(condition_mask_array.sum())
        condition_color = condition_colors.get(condition_label, plot_cfg.get_color("blue"))
        
        save_condition_network(
            condition_graph, condition_label, condition_color, n_trials
        )
    
    log_if_present(logger, "info", 
                  f"Saved {measure} {band} connectivity networks by condition")
