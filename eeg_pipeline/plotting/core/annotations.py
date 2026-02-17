"""
Core annotation utilities.

Significance markers, ROI annotation helpers, and p-value formatting functions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    import mne

from ..config import get_plot_config
from ...utils.analysis.stats import fdr_bh_values as _fdr_bh_values
from .utils import get_font_sizes


###################################################################
# Significance Formatting
###################################################################


def format_cluster_significance_info(
    roi_sig_chs: List[str],
    roi_sig_indices: List[int],
    p_ch: Optional[np.ndarray],
    cluster_p_min: float,
    cluster_k: Optional[int],
) -> str:
    """Format cluster significance information string.
    
    Args:
        roi_sig_chs: List of significant channel names
        roi_sig_indices: List of indices for significant channels
        p_ch: Optional array of per-channel p-values
        cluster_p_min: Minimum cluster p-value
        cluster_k: Optional cluster size
    
    Returns:
        Formatted significance information string
    """
    sig_parts = []
    cluster_info = f"Cluster: p={cluster_p_min:.3f}"
    if cluster_k is not None:
        cluster_info += f", k={cluster_k}"
    sig_parts.append(cluster_info)
    sig_parts.append(f"Channels: {', '.join(roi_sig_chs)}")
    if p_ch is not None:
        p_str = ', '.join(f"{p_ch[idx]:.3f}" for idx in roi_sig_indices)
        sig_parts.append(f"p-values: {p_str}")
    return " | ".join(sig_parts)


def format_ttest_significance_info(
    roi_sig_chs: List[str],
    roi_sig_indices: List[int],
    p_ch: np.ndarray,
) -> str:
    """Format t-test significance information string.
    
    Args:
        roi_sig_chs: List of significant channel names
        roi_sig_indices: List of indices for significant channels
        p_ch: Array of per-channel p-values
    
    Returns:
        Formatted significance information string
    """
    sig_parts = ["T-Test:"]
    roi_p_vals = [p_ch[idx] for idx in roi_sig_indices]
    for channel_name, p_val in zip(roi_sig_chs, roi_p_vals):
        sig_parts.append(f"{channel_name}: {p_val:.3f}")
    return " | ".join(sig_parts)


def build_significance_info(
    roi_sig_chs: List[str],
    roi_sig_indices: List[int],
    p_ch: Optional[np.ndarray],
    is_cluster: bool,
    cluster_p_min: Optional[float],
    cluster_k: Optional[int],
) -> Optional[str]:
    """Build significance information string.
    
    Args:
        roi_sig_chs: List of significant channel names
        roi_sig_indices: List of indices for significant channels
        p_ch: Optional array of per-channel p-values
        is_cluster: Whether cluster test was used
        cluster_p_min: Optional minimum cluster p-value
        cluster_k: Optional cluster size
    
    Returns:
        Formatted significance information string, or None if no significant channels
    """
    if not roi_sig_chs:
        return None
    
    if is_cluster and cluster_p_min is not None:
        return format_cluster_significance_info(
            roi_sig_chs, roi_sig_indices, p_ch, cluster_p_min, cluster_k
        )
    
    if p_ch is not None:
        return format_ttest_significance_info(roi_sig_chs, roi_sig_indices, p_ch)
    
    return None


###################################################################
# P-value Processing
###################################################################


def extract_valid_pvalues(
    roi_pvalues_raw: Dict[str, Optional[float]]
) -> Tuple[List[float], Dict[str, int]]:
    """Extract valid p-values from dictionary.
    
    Args:
        roi_pvalues_raw: Dictionary mapping ROI names to p-values (may be None)
    
    Returns:
        Tuple of (list of valid p-values, mapping from ROI name to index in list)
    """
    p_vals_list = []
    roi_to_idx_map = {}
    
    for roi, p_val in roi_pvalues_raw.items():
        if p_val is not None and np.isfinite(p_val):
            roi_to_idx_map[roi] = len(p_vals_list)
            p_vals_list.append(p_val)
    
    return p_vals_list, roi_to_idx_map


def apply_fdr_correction_to_roi_pvalues(
    roi_pvalues_raw: Dict[str, Optional[float]],
    apply_fdr_correction: bool,
    fdr_alpha: float,
    plot_cfg=None,
) -> Dict[str, Optional[float]]:
    """Apply FDR correction to ROI p-values.
    
    Args:
        roi_pvalues_raw: Dictionary mapping ROI names to raw p-values
        apply_fdr_correction: Whether to apply FDR correction
        fdr_alpha: FDR alpha threshold
        plot_cfg: Optional PlotConfig instance
    
    Returns:
        Dictionary mapping ROI names to corrected p-values (or original if not corrected)
    """
    if plot_cfg is None:
        plot_cfg = get_plot_config(None)
    plot_cfg.plot_type_configs.get("tfr", {})
    min_rois_for_fdr = plot_cfg.validation.get("min_rois_for_fdr", 1)
    min_pvalues_for_fdr = plot_cfg.validation.get("min_pvalues_for_fdr", 1)
    
    if not apply_fdr_correction or len(roi_pvalues_raw) <= min_rois_for_fdr:
        return roi_pvalues_raw.copy()
    
    p_vals_list, roi_to_idx_map = extract_valid_pvalues(roi_pvalues_raw)
    
    if len(p_vals_list) <= min_pvalues_for_fdr:
        return roi_pvalues_raw.copy()
    
    p_vals_array = np.array(p_vals_list)
    _, q_values = _fdr_bh_values(p_vals_array, alpha=fdr_alpha)
    
    if q_values is None:
        return roi_pvalues_raw.copy()
    
    roi_pvalues_corrected = roi_pvalues_raw.copy()
    for roi, idx in roi_to_idx_map.items():
        roi_pvalues_corrected[roi] = float(q_values[idx])
    
    return roi_pvalues_corrected


###################################################################
# Annotation Positioning
###################################################################


def _validate_group_data_shapes(
    data_group_a: np.ndarray,
    data_group_b: np.ndarray,
    num_channels: int
) -> bool:
    """Validate that group data arrays have correct channel dimension.
    
    Args:
        data_group_a: Group A data array
        data_group_b: Group B data array
        num_channels: Expected number of channels
    
    Returns:
        True if both groups have valid shapes, False otherwise
    """
    group_a_valid = data_group_a.shape[1] == num_channels
    group_b_valid = data_group_b.shape[1] == num_channels
    return group_a_valid and group_b_valid


def find_annotation_x_position(ax: plt.Axes, plot_cfg=None) -> float:
    """Find x position for annotations, avoiding overlap with adjacent axes.
    
    Args:
        ax: Matplotlib axes to place annotations on
        plot_cfg: Optional PlotConfig instance
    
    Returns:
        X position in axes coordinates (0-1)
    """
    if plot_cfg is None:
        plot_cfg = get_plot_config(None)
    tfr_config = plot_cfg.plot_type_configs.get("tfr", {})
    y_center_tolerance = tfr_config.get("y_center_tolerance", 0.1)
    x_min_distance = tfr_config.get("x_min_distance", 0.01)
    default_x_position = tfr_config.get("default_x_position", 0.95)
    max_x_position = tfr_config.get("max_x_position", 1.08)
    x_offset = tfr_config.get("x_offset", 0.02)
    
    fig = ax.figure
    ax_bbox = ax.get_position()
    current_ax_right_edge = ax_bbox.x1
    current_ax_y_center = (ax_bbox.y0 + ax_bbox.y1) / 2.0
    
    right_neighbor_left_edge = None
    for other_ax in fig.get_axes():
        if other_ax == ax:
            continue
        other_bbox = other_ax.get_position()
        other_ax_y_center = (other_bbox.y0 + other_bbox.y1) / 2.0
        y_centers_are_close = abs(other_ax_y_center - current_ax_y_center) < y_center_tolerance
        is_to_the_right = other_bbox.x0 > current_ax_right_edge + x_min_distance
        if y_centers_are_close and is_to_the_right:
            if right_neighbor_left_edge is None or other_bbox.x0 < right_neighbor_left_edge:
                right_neighbor_left_edge = other_bbox.x0
    
    if right_neighbor_left_edge is None:
        return default_x_position
    
    ax_width = ax_bbox.x1 - ax_bbox.x0
    max_x_in_axes_coords = (right_neighbor_left_edge - ax_bbox.x0) / ax_width
    return min(max_x_position, max_x_in_axes_coords - x_offset)


###################################################################
# Annotation Label Building
###################################################################


def build_roi_annotation_label(
    roi: str,
    pct: float,
    roi_pvalue: Optional[float],
    sig_info: Optional[str],
    use_fdr: bool,
    paired: bool = False,
) -> str:
    """Build ROI annotation label string.
    
    Args:
        roi: ROI name
        pct: Percentage change value
        roi_pvalue: Optional p-value or q-value
        sig_info: Optional significance information string
        use_fdr: Whether FDR correction was applied
        paired: Whether paired test was used
    
    Returns:
        Formatted annotation label string
    """
    label = f"{roi}: {pct:+.1f}%"
    if roi_pvalue is not None and np.isfinite(roi_pvalue):
        test_type = "paired t-test" if paired else "t-test"
        if use_fdr:
            label += f" (FDR-BH {test_type}: q={roi_pvalue:.3f})"
        else:
            label += f" ({test_type}: p={roi_pvalue:.3f})"
    if sig_info:
        label += f" ({sig_info})"
    return label


###################################################################
# Annotation Rendering
###################################################################


def render_roi_annotations(
    ax: plt.Axes,
    annotations: List[Tuple[str, float, Optional[str], Optional[float]]],
    apply_fdr_correction: bool,
    paired: bool = False,
    plot_cfg=None,
) -> None:
    """Render ROI annotations on axes.
    
    Args:
        ax: Matplotlib axes to render annotations on
        annotations: List of (roi, pct, sig_info, roi_pvalue) tuples
        apply_fdr_correction: Whether FDR correction was applied
        paired: Whether paired test was used
        plot_cfg: Optional PlotConfig instance
    """
    if plot_cfg is None:
        plot_cfg = get_plot_config(None)
    tfr_config = plot_cfg.plot_type_configs.get("tfr", {})
    font_sizes = get_font_sizes(plot_cfg)
    annotation_fontsize = font_sizes["annotation"]
    
    x_pos_ax = find_annotation_x_position(ax, plot_cfg)
    annotation_y_start = tfr_config.get("annotation_y_start", 0.98)
    y_pos_ax = annotation_y_start
    
    num_rois_with_pvals = sum(
        1 for _, _, _, pval in annotations
        if pval is not None and np.isfinite(pval)
    )
    min_rois_for_fdr = plot_cfg.validation.get("min_rois_for_fdr", 1)
    should_use_fdr = apply_fdr_correction and num_rois_with_pvals > min_rois_for_fdr
    
    annotation_line_height = tfr_config.get("annotation_line_height", 0.045)
    annotation_min_spacing = tfr_config.get("annotation_min_spacing", 0.03)
    annotation_spacing_multiplier = tfr_config.get("annotation_spacing_multiplier", 0.3)
    
    for i, annotation in enumerate(annotations):
        roi, pct, sig_info, roi_pvalue = annotation
        label = build_roi_annotation_label(roi, pct, roi_pvalue, sig_info, should_use_fdr, paired)
        
        ax.text(
            x_pos_ax, y_pos_ax, label,
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=annotation_fontsize
        )
        
        if i < len(annotations) - 1:
            extra_spacing = annotation_line_height * annotation_spacing_multiplier
            total_line_spacing = annotation_line_height + annotation_min_spacing + extra_spacing
            y_pos_ax -= total_line_spacing


###################################################################
# Significance Marker Text
###################################################################


def get_sig_marker_text(config=None) -> str:
    """Get significance marker text for figure titles.
    
    Args:
        config: Optional configuration dictionary
    
    Returns:
        Significance marker text string, or empty string if diff_annotation_enabled is False
    """
    from eeg_pipeline.plotting.io.figures import get_viz_params
    from eeg_pipeline.utils.config.loader import get_config_value, ensure_config
    
    viz_params = get_viz_params(config)
    if not viz_params["diff_annotation_enabled"]:
        return ""
    
    config = ensure_config(config)
    plot_cfg = get_plot_config(config)
    
    tfr_config = plot_cfg.plot_type_configs.get("tfr", {}) if plot_cfg else {}
    default_sig_alpha = tfr_config.get(
        "default_significance_alpha",
        get_config_value(config, "statistics.sig_alpha", 0.05)
    )
    default_cluster_n_perm = tfr_config.get(
        "default_cluster_n_perm",
        get_config_value(config, "statistics.cluster_n_perm", 100)
    )
    
    alpha = get_config_value(config, "statistics.sig_alpha", default_sig_alpha)
    n_perm = get_config_value(config, "statistics.cluster_n_perm", default_cluster_n_perm)
    method = f"cluster permutation (n={n_perm})"
    return f" | Green markers: p < {alpha:.2f} ({method})"


###################################################################
# High-Level ROI Annotation (from raw data)
###################################################################


def add_roi_annotations(
    ax: "plt.Axes",
    data: np.ndarray,
    info: mne.Info,
    config=None,
    roi_map=None,
    sig_mask: Optional[np.ndarray] = None,
    p_ch: Optional[np.ndarray] = None,
    cluster_p_min: Optional[float] = None,
    cluster_k: Optional[int] = None,
    is_cluster: Optional[bool] = None,
    data_format: Optional[str] = None,
    data_group_a: Optional[np.ndarray] = None,
    data_group_b: Optional[np.ndarray] = None,
    paired: bool = False,
    apply_fdr_correction: bool = True,
    fdr_alpha: Optional[float] = None,
) -> None:
    """Add ROI annotations to topomap axes from raw data.
    
    This is a high-level function that processes raw data and calls
    render_roi_annotations with the appropriate annotation tuples.
    
    Args:
        ax: Matplotlib axes
        data: Topomap data array
        info: MNE Info object
        config: Optional configuration object
        roi_map: Optional ROI map dictionary
        sig_mask: Optional significance mask
        p_ch: Optional per-channel p-values
        cluster_p_min: Optional minimum cluster p-value
        cluster_k: Optional cluster size
        is_cluster: Whether cluster test was used
        data_format: Optional data format string
        data_group_a: Optional data for group A
        data_group_b: Optional data for group B
        paired: Whether paired test was used
        apply_fdr_correction: Whether to apply FDR correction
        fdr_alpha: Optional FDR alpha threshold
    """
    from eeg_pipeline.utils.validation import detect_data_format as _detect_data_format
    
    if config is None and roi_map is None:
        return
    
    if roi_map is None and config is not None:
        from ...utils.analysis.tfr import build_rois_from_info
        roi_map = build_rois_from_info(info, config=config)
    if not roi_map:
        return
    
    ch_names = info["ch_names"]
    if len(data) != len(ch_names):
        return
    
    data_finite = data[np.isfinite(data)]
    if data_finite.size == 0:
        return
    
    if data_group_a is not None and data_group_b is not None:
        if not _validate_group_data_shapes(data_group_a, data_group_b, len(ch_names)):
            data_group_a = None
            data_group_b = None
    
    from eeg_pipeline.utils.config.loader import get_config_value, ensure_config
    
    config = ensure_config(config)
    plot_cfg = get_plot_config(config)
    tfr_config = plot_cfg.plot_type_configs.get("tfr", {}) if plot_cfg else {}
    
    if fdr_alpha is None:
        default_alpha = get_config_value(config, "statistics.sig_alpha", 0.05)
        fdr_alpha_fallback = get_config_value(config, "statistics.fdr_alpha", default_alpha)
        fdr_alpha = get_config_value(
            config, "behavior_analysis.statistics.fdr_alpha", fdr_alpha_fallback
        )
    
    percent_detection_threshold = tfr_config.get("percent_detection_threshold", 5.0)
    is_percent_format = _detect_data_format(
        data, data_format, percent_threshold=percent_detection_threshold
    )
    
    from ...utils.analysis.tfr import (
        build_roi_channel_mask,
        extract_significant_roi_channels,
    )
    from ...utils.analysis.stats import (
        compute_roi_percentage_change,
        compute_roi_pvalue,
    )
    
    roi_pvalues_raw = {}
    roi_data_dict = {}
    
    for roi, roi_chs in roi_map.items():
        mask_vec = build_roi_channel_mask(ch_names, roi_chs)
        if not mask_vec.any():
            continue
        
        roi_data = data[mask_vec]
        pct = compute_roi_percentage_change(roi_data, is_percent_format)
        roi_data_dict[roi] = (pct, mask_vec)
        
        roi_pvalue = compute_roi_pvalue(
            mask_vec, data_group_a, data_group_b, paired
        )
        roi_pvalues_raw[roi] = roi_pvalue
    
    roi_pvalues_corrected = apply_fdr_correction_to_roi_pvalues(
        roi_pvalues_raw, apply_fdr_correction, fdr_alpha, plot_cfg
    )
    
    annotations = []
    for roi, roi_chs in roi_map.items():
        if roi not in roi_data_dict:
            continue
        
        pct, mask_vec = roi_data_dict[roi]
        roi_pvalue = roi_pvalues_corrected.get(roi)
        
        sig_info = None
        if sig_mask is not None and sig_mask.any():
            roi_sig_indices, roi_sig_chs = extract_significant_roi_channels(
                ch_names, mask_vec, sig_mask
            )
            sig_info = build_significance_info(
                roi_sig_chs, roi_sig_indices, p_ch, is_cluster,
                cluster_p_min, cluster_k
            )
        
        annotations.append((roi, pct, sig_info, roi_pvalue))
    
    if annotations:
        render_roi_annotations(ax, annotations, apply_fdr_correction, paired, plot_cfg)
