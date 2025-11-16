import os
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import matplotlib.colors as mcolors
from scipy.stats import ttest_rel, ttest_ind, t as t_dist

from eeg_pipeline.utils.io_utils import (
    ensure_aligned_lengths,
    deriv_group_plots_path,
    deriv_group_stats_path,
)
from eeg_pipeline.utils.tfr_utils import (
    apply_baseline_and_crop,
    validate_baseline_indices,
    avg_alltrials_to_avg_tfr,
    avg_by_mask_to_avg_tfr,
    align_avg_tfrs,
    align_paired_avg_tfrs,
    get_rois,
    canonicalize_ch_name as _canonicalize_ch_name,
    find_roi_channels as _find_roi_channels,
    run_tfr_morlet,
    create_tfr_subset,
    apply_baseline_and_average,
    clip_time_range,
    create_time_windows_fixed_size,
    create_time_windows_fixed_count,
    clip_time_window as clip_time_window_tfr,
    create_time_mask_strict,
    create_time_mask_loose,
    extract_trial_band_power,
    extract_band_channel_means,
    get_bands_for_tfr,
    build_roi_channel_mask,
    extract_significant_roi_channels,
    extract_roi_from_tfr,
    extract_roi_contrast_data,
)
from eeg_pipeline.utils.stats_utils import (
    compute_roi_percentage_change,
    compute_roi_pvalue,
    format_correlation_text,
)
from eeg_pipeline.utils.data_loading import (
    extract_aligned_column_vector,
    extract_pain_vector,
    extract_pain_vector_array,
    extract_temperature_series,
    compute_aligned_data_length,
    create_temperature_masks,
    create_temperature_masks_from_range,
    get_temperature_range,
    extract_time_frequency_grid,
    extract_importance_column,
    build_epoch_query_string,
)
from eeg_pipeline.utils.io_utils import (
    get_viz_params,
    unwrap_figure,
    plot_topomap_on_ax,
    robust_sym_vlim,
    sanitize_label,
    get_column_from_config,
    get_pain_column_from_config,
    get_temperature_column_from_config,
    extract_plotting_constants,
    get_band_color,
    extract_eeg_picks,
    require_epochs_tfr,
    detect_data_format,
    format_baseline_window_string,
    logratio_to_pct,
)
from eeg_pipeline.utils.plotting_config import get_plot_config
from eeg_pipeline.utils.stats_utils import (
    fdr_bh_mask as _fdr_bh_mask,
    fdr_bh_values as _fdr_bh_values,
    cluster_test_epochs as _cluster_test_epochs,
    cluster_test_two_sample_arrays as _cluster_test_two_sample_arrays,
    format_cluster_ann,
)




###################################################################
# Utilities
###################################################################

def _get_font_sizes(plot_cfg=None):
    if plot_cfg is None:
        from eeg_pipeline.utils.plotting_config import get_plot_config
        plot_cfg = get_plot_config(None)
    return {
        "annotation": plot_cfg.font.annotation,
        "label": plot_cfg.font.label,
        "title": plot_cfg.font.title,
        "ylabel": plot_cfg.font.ylabel,
        "suptitle": plot_cfg.font.suptitle,
        "figure_title": plot_cfg.font.figure_title,
    }


def _log(msg, logger=None, level: str = "info"):
    if logger is None:
        logger = logging.getLogger(__name__)
    getattr(logger, level)(msg)






def _build_topomap_diff_label(diff_data, cluster_p_min, cluster_k, cluster_mass, config, viz_params, paired=False) -> str:
    diff_mu = float(np.nanmean(diff_data))
    pct = logratio_to_pct(diff_mu)
    
    cl_txt = ""
    if viz_params["diff_annotation_enabled"] and cluster_p_min is not None:
        test_type = "paired cluster perm" if paired else "cluster perm"
        tfr_config = None
        if config:
            from eeg_pipeline.utils.plotting_config import get_plot_config
            plot_cfg = get_plot_config(config)
            tfr_config = plot_cfg.plot_type_configs.get("tfr", {})
        default_cluster_n_perm = tfr_config.get("default_cluster_n_perm", 1024) if tfr_config else 1024
        n_perm = config.get("statistics.cluster_n_perm", default_cluster_n_perm) if config else default_cluster_n_perm
        cl_txt = format_cluster_ann(cluster_p_min, cluster_k, cluster_mass, config=config)
        if cl_txt:
            cl_txt = f"{test_type} (n={n_perm}): {cl_txt}"
    
    return f"Δ%={pct:+.1f}%" + (f" | {cl_txt}" if cl_txt else "")


def _average_tfr_band(tfr_avg, fmin, fmax, tmin, tmax) -> Optional[np.ndarray]:
    freqs = np.asarray(tfr_avg.freqs)
    times = np.asarray(tfr_avg.times)
    f_mask = (freqs >= float(fmin)) & (freqs <= float(fmax))
    t_mask = (times >= float(tmin)) & (times < float(tmax))
    
    if f_mask.sum() == 0 or t_mask.sum() == 0:
        return None
    
    sel = np.asarray(tfr_avg.data)[:, f_mask, :][:, :, t_mask]
    return sel.mean(axis=(1, 2))




def _get_sig_marker_text(config=None) -> str:
    viz_params = get_viz_params(config)
    if not viz_params["diff_annotation_enabled"]:
        return ""
    
    plot_cfg = get_plot_config(config) if config else None
    tfr_config = plot_cfg.plot_type_configs.get("tfr", {}) if plot_cfg else {}
    default_sig_alpha = tfr_config.get("default_significance_alpha", 0.05) if plot_cfg else 0.05
    default_cluster_n_perm = tfr_config.get("default_cluster_n_perm", 1024) if plot_cfg else 1024
    alpha = config.get("statistics.sig_alpha", default_sig_alpha) if config else default_sig_alpha
    n_perm = config.get("statistics.cluster_n_perm", default_cluster_n_perm) if config else default_cluster_n_perm
    method = f"cluster permutation (n={n_perm})"
    return f" | Green markers: p < {alpha:.2f} ({method})"








def _format_cluster_significance_info(roi_sig_chs, roi_sig_indices, p_ch, cluster_p_min, cluster_k):
    sig_parts = []
    cluster_info = f"Cluster: p={cluster_p_min:.3f}"
    if cluster_k is not None:
        cluster_info += f", k={cluster_k}"
    sig_parts.append(cluster_info)
    sig_parts.append(f"Channels: {', '.join(roi_sig_chs)}")
    if p_ch is not None:
        roi_p_vals = [p_ch[i] for i in roi_sig_indices]
        p_str = ', '.join([f"{p:.3f}" for p in roi_p_vals])
        sig_parts.append(f"p-values: {p_str}")
    return " | ".join(sig_parts)


def _format_ttest_significance_info(roi_sig_chs, roi_sig_indices, p_ch):
    sig_parts = ["T-Test:"]
    roi_p_vals = [p_ch[i] for i in roi_sig_indices]
    for ch, p_val in zip(roi_sig_chs, roi_p_vals):
        sig_parts.append(f"{ch}: {p_val:.3f}")
    return " | ".join(sig_parts)


def _build_significance_info(roi_sig_chs, roi_sig_indices, p_ch, is_cluster, cluster_p_min, cluster_k) -> Optional[str]:
    if not roi_sig_chs:
        return None
    
    if is_cluster and cluster_p_min is not None:
        return _format_cluster_significance_info(roi_sig_chs, roi_sig_indices, p_ch, cluster_p_min, cluster_k)
    
    if p_ch is not None:
        return _format_ttest_significance_info(roi_sig_chs, roi_sig_indices, p_ch)
    
    return None




def _collect_roi_annotations(data, ch_names, roi_map, sig_mask, p_ch, is_cluster, 
                              cluster_p_min, cluster_k, is_percent_format,
                              data_group_a=None, data_group_b=None, paired=False, 
                              apply_fdr_correction=True, fdr_alpha=0.05, config=None):
    has_sig_info = sig_mask is not None and sig_mask.any()
    
    roi_pvalues_raw = {}
    roi_data_dict = {}
    
    for roi, roi_chs in roi_map.items():
        mask_vec = build_roi_channel_mask(ch_names, roi_chs)
        if not mask_vec.any():
            continue
        
        roi_data = data[mask_vec]
        pct = compute_roi_percentage_change(roi_data, is_percent_format)
        roi_data_dict[roi] = (pct, mask_vec)
        
        roi_pvalue = compute_roi_pvalue(mask_vec, ch_names, p_ch, sig_mask, is_cluster, cluster_p_min,
                                         data_group_a, data_group_b, paired)
        roi_pvalues_raw[roi] = roi_pvalue
    
    plot_cfg = get_plot_config(config) if config else None
    roi_pvalues_corrected = _apply_fdr_correction_to_roi_pvalues(
        roi_pvalues_raw, apply_fdr_correction, fdr_alpha, plot_cfg
    )
    
    annotations = []
    for roi, roi_chs in roi_map.items():
        if roi not in roi_data_dict:
            continue
        
        pct, mask_vec = roi_data_dict[roi]
        roi_pvalue = roi_pvalues_corrected.get(roi)
        
        sig_info = None
        if has_sig_info:
            roi_sig_indices, roi_sig_chs = extract_significant_roi_channels(ch_names, mask_vec, sig_mask)
            sig_info = _build_significance_info(roi_sig_chs, roi_sig_indices, p_ch, is_cluster, 
                                                cluster_p_min, cluster_k)
        
        annotations.append((roi, pct, sig_info, roi_pvalue))
    
    return annotations


def _extract_valid_pvalues(roi_pvalues_raw: Dict[str, Optional[float]]) -> Tuple[List[float], Dict[str, int]]:
    p_vals_list = []
    roi_to_idx_map = {}
    
    for roi, p_val in roi_pvalues_raw.items():
        if p_val is not None and np.isfinite(p_val):
            roi_to_idx_map[roi] = len(p_vals_list)
            p_vals_list.append(p_val)
    
    return p_vals_list, roi_to_idx_map


def _apply_fdr_correction_to_roi_pvalues(roi_pvalues_raw, apply_fdr_correction, fdr_alpha, plot_cfg=None):
    if plot_cfg is None:
        from eeg_pipeline.utils.plotting_config import get_plot_config
        plot_cfg = get_plot_config(None)
    tfr_config = plot_cfg.plot_type_configs.get("tfr", {})
    min_rois_for_fdr = plot_cfg.validation.get("min_rois_for_fdr", 1)
    min_pvalues_for_fdr = plot_cfg.validation.get("min_pvalues_for_fdr", 1)
    
    if not apply_fdr_correction or len(roi_pvalues_raw) <= min_rois_for_fdr:
        return roi_pvalues_raw.copy()
    
    p_vals_list, roi_to_idx_map = _extract_valid_pvalues(roi_pvalues_raw)
    
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


def _find_annotation_x_position(ax, plot_cfg=None):
    if plot_cfg is None:
        from eeg_pipeline.utils.plotting_config import get_plot_config
        plot_cfg = get_plot_config(None)
    tfr_config = plot_cfg.plot_type_configs.get("tfr", {})
    y_center_tolerance = tfr_config.get("y_center_tolerance", 0.1)
    x_min_distance = tfr_config.get("x_min_distance", 0.01)
    default_x_position = tfr_config.get("default_x_position", 0.95)
    max_x_position = tfr_config.get("max_x_position", 1.08)
    x_offset = tfr_config.get("x_offset", 0.02)
    
    fig = ax.figure
    ax_bbox = ax.get_position()
    ax_x1 = ax_bbox.x1
    ax_y_center = (ax_bbox.y0 + ax_bbox.y1) / 2.0
    
    right_neighbor_x0 = None
    for other_ax in fig.get_axes():
        if other_ax == ax:
            continue
        other_bbox = other_ax.get_position()
        other_y_center = (other_bbox.y0 + other_bbox.y1) / 2.0
        is_same_row = abs(other_y_center - ax_y_center) < y_center_tolerance
        is_right_of_current = other_bbox.x0 > ax_x1 + x_min_distance
        if is_same_row and is_right_of_current:
            if right_neighbor_x0 is None or other_bbox.x0 < right_neighbor_x0:
                right_neighbor_x0 = other_bbox.x0
    
    if right_neighbor_x0 is None:
        return default_x_position
    
    ax_width = ax_bbox.x1 - ax_bbox.x0
    max_x_fig = (right_neighbor_x0 - ax_bbox.x0) / ax_width
    return min(max_x_position, max_x_fig - x_offset)


def _build_roi_annotation_label(roi, pct, roi_pvalue, sig_info, use_fdr, paired=False):
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


def _render_roi_annotations(ax, annotations, apply_fdr_correction, paired=False, plot_cfg=None):
    if plot_cfg is None:
        from eeg_pipeline.utils.plotting_config import get_plot_config
        plot_cfg = get_plot_config(None)
    tfr_config = plot_cfg.plot_type_configs.get("tfr", {})
    font_sizes = _get_font_sizes(plot_cfg)
    annotation_fontsize = font_sizes["annotation"]
    
    x_pos_ax = _find_annotation_x_position(ax, plot_cfg)
    annotation_y_start = tfr_config.get("annotation_y_start", 0.98)
    y_pos_ax = annotation_y_start
    
    num_rois_with_pvals = sum(1 for _, _, _, pval in annotations if pval is not None and np.isfinite(pval))
    min_rois_for_fdr = plot_cfg.validation.get("min_rois_for_fdr", 1)
    use_fdr = apply_fdr_correction and num_rois_with_pvals > min_rois_for_fdr
    
    annotation_line_height = tfr_config.get("annotation_line_height", 0.045)
    annotation_min_spacing = tfr_config.get("annotation_min_spacing", 0.03)
    annotation_spacing_multiplier = tfr_config.get("annotation_spacing_multiplier", 0.3)
    
    for i, annotation in enumerate(annotations):
        roi, pct, sig_info, roi_pvalue = annotation
        label = _build_roi_annotation_label(roi, pct, roi_pvalue, sig_info, use_fdr, paired)
        
        ax.text(x_pos_ax, y_pos_ax, label, transform=ax.transAxes, 
               ha="left", va="top", fontsize=annotation_fontsize)
        
        if i < len(annotations) - 1:
            spacing_ax = annotation_min_spacing + (annotation_line_height * annotation_spacing_multiplier)
            y_pos_ax -= (annotation_line_height + spacing_ax)


def _add_roi_annotations(ax, data, info, config=None, roi_map=None, sig_mask=None, p_ch=None, 
                         cluster_p_min=None, cluster_k=None, cluster_mass=None, is_cluster=None,
                         data_format=None, data_group_a=None, data_group_b=None, paired=False,
                         apply_fdr_correction=True, fdr_alpha=None):
    if config is None and roi_map is None:
        return
    
    if roi_map is None and config is not None:
        from eeg_pipeline.utils.tfr_utils import build_rois_from_info
        roi_map = build_rois_from_info(info, config=config)
    if not roi_map:
        return
    
    ch_names = info["ch_names"]
    if len(data) != len(ch_names):
        return
    
    data_finite = data[np.isfinite(data)]
    if data_finite.size == 0:
        return
    
    has_valid_groups = data_group_a is not None and data_group_b is not None
    if has_valid_groups:
        group_a_valid = data_group_a.shape[1] == len(ch_names)
        group_b_valid = data_group_b.shape[1] == len(ch_names)
        if not group_a_valid or not group_b_valid:
            data_group_a = None
            data_group_b = None
    
    plot_cfg = get_plot_config(config) if config else None
    tfr_config = plot_cfg.plot_type_configs.get("tfr", {}) if plot_cfg else {}
    
    if fdr_alpha is None:
        default_fdr_alpha = tfr_config.get("default_fdr_alpha", 0.05) if plot_cfg else 0.05
        fdr_alpha = config.get("statistics.sig_alpha", default_fdr_alpha) if config else default_fdr_alpha
    
    percent_detection_threshold = tfr_config.get("percent_detection_threshold", 5.0) if plot_cfg else 5.0
    is_percent_format = detect_data_format(data, data_format, percent_threshold=percent_detection_threshold)
    annotations = _collect_roi_annotations(data, ch_names, roi_map, sig_mask, p_ch, 
                                          is_cluster, cluster_p_min, cluster_k, is_percent_format,
                                          data_group_a, data_group_b, paired,
                                          apply_fdr_correction, fdr_alpha, config)
    
    if not annotations:
        return
    
    _render_roi_annotations(ax, annotations, apply_fdr_correction, paired, plot_cfg)


def _create_pain_masks_from_vector(pain_vec):
    if pain_vec is None:
        return None, None
    if hasattr(pain_vec, 'fillna'):
        pain_vec = pain_vec.fillna(0).astype(int).values
    pain_mask = np.asarray(pain_vec == 1, dtype=bool)
    non_mask = np.asarray(pain_vec == 0, dtype=bool)
    return pain_mask, non_mask


def _create_pain_masks_from_events(events_df, pain_col):
    if events_df is None or pain_col is None or pain_col not in events_df.columns:
        return None, None
    vals = pd.to_numeric(events_df[pain_col], errors="coerce").fillna(0).astype(int)
    return _create_pain_masks_from_vector(vals)




def _get_baseline_window(config, baseline=None):
    if baseline is not None:
        return baseline
    plot_cfg = get_plot_config(config) if config else None
    tfr_config = plot_cfg.plot_type_configs.get("tfr", {}) if plot_cfg else {}
    default_baseline_start = tfr_config.get("default_baseline_start", -5.0) if plot_cfg else -5.0
    default_baseline_end = tfr_config.get("default_baseline_end", -0.01) if plot_cfg else -0.01
    default_baseline_window = [default_baseline_start, default_baseline_end]
    return tuple(config.get("time_frequency_analysis.baseline_window", default_baseline_window)) if config else tuple(default_baseline_window)






def _get_strict_mode(config):
    return config.get("analysis.strict_mode", True) if config else True


def _plot_topomap_with_label(ax, data, info, vmin, vmax, label_text, config=None):
    plot_cfg = get_plot_config(config) if config else None
    font_sizes = _get_font_sizes(plot_cfg)
    tfr_config = plot_cfg.plot_type_configs.get("tfr", {}) if plot_cfg else {}
    label_x_position = tfr_config.get("label_x_position", 0.5) if plot_cfg else 0.5
    label_y_position = tfr_config.get("label_y_position", 1.02) if plot_cfg else 1.02
    plot_topomap_on_ax(ax, data, info, vmin=vmin, vmax=vmax, config=config)
    _add_roi_annotations(ax, data, info, config=config)
    ax.text(label_x_position, label_y_position, label_text, transform=ax.transAxes, ha="center", va="top", fontsize=font_sizes["title"])


def _plot_topomap_with_percentage_label(ax, data, info, vmin, vmax, config):
    mu = float(np.nanmean(data))
    pct = logratio_to_pct(mu)
    label_text = f"%Δ={pct:+.1f}%"
    _plot_topomap_with_label(ax, data, info, vmin, vmax, label_text, config)


def _create_scalpmean_tfr_from_existing(tfr_avg, eeg_picks):
    tfr_sm = tfr_avg.copy().pick(eeg_picks)
    data_sm = np.asarray(tfr_sm.data).mean(axis=0, keepdims=True)
    tfr_sm = tfr_sm.pick([0])
    tfr_sm.data = data_sm
    tfr_sm.comment = "Scalp-averaged"
    return tfr_sm


def _plot_scalpmean_tfr(tfr_sm, title, filename, vlim, out_dir, config, logger, baseline_used, subject, task):
    font_sizes = _get_font_sizes()
    ch_name = tfr_sm.info['ch_names'][0]
    plot_kwargs = {"picks": ch_name, "show": False}
    if vlim is not None:
        plot_kwargs["vlim"] = vlim
    fig = unwrap_figure(tfr_sm.plot(**plot_kwargs))
    fig.suptitle(title, fontsize=font_sizes["figure_title"])
    _save_fig(fig, out_dir, filename, config=config, logger=logger, baseline_used=baseline_used, subject=subject, task=task)


def _plot_topomap_with_diff_label(ax, diff_data, info, vmin, vmax, sig_mask, cluster_info, config=None,
                                   data_group_a=None, data_group_b=None, paired=False):
    viz_params = get_viz_params(config)
    plot_topomap_on_ax(
        ax,
        diff_data,
        info,
        mask=sig_mask if viz_params["diff_annotation_enabled"] else None,
        mask_params=viz_params["sig_mask_params"],
        vmin=vmin,
        vmax=vmax,
        config=config,
    )
    is_cluster = sig_mask is not None and cluster_info.get("cluster_p_min") is not None
    _add_roi_annotations(
        ax, diff_data, info, config=config,
        sig_mask=sig_mask,
        cluster_p_min=cluster_info.get("cluster_p_min"),
        cluster_k=cluster_info.get("cluster_k"),
        cluster_mass=cluster_info.get("cluster_mass"),
        is_cluster=is_cluster,
        data_group_a=data_group_a,
        data_group_b=data_group_b,
        paired=paired
    )
    
    label = _build_topomap_diff_label(
        diff_data,
        cluster_info.get("cluster_p_min"),
        cluster_info.get("cluster_k"),
        cluster_info.get("cluster_mass"),
        config,
        viz_params,
        paired=paired
    )
    font_sizes = _get_font_sizes()
    plot_cfg = get_plot_config(config) if config else None
    tfr_config = plot_cfg.plot_type_configs.get("tfr", {}) if plot_cfg else {}
    label_x_position = tfr_config.get("label_x_position", 0.5) if plot_cfg else 0.5
    label_y_position = tfr_config.get("label_y_position", 1.02) if plot_cfg else 1.02
    ax.text(label_x_position, label_y_position, label, transform=ax.transAxes, ha="center", va="top", fontsize=font_sizes["title"])


def _add_normalized_colorbar(fig, axes_list, vmin, vmax, cmap, config=None):
    plot_cfg = get_plot_config(config)
    tfr_config = plot_cfg.plot_type_configs.get("tfr", {})
    colorbar_config = tfr_config.get("colorbar", {})
    cbar_fraction = colorbar_config.get("fraction", 0.045)
    cbar_pad = colorbar_config.get("pad", 0.06)
    cbar_shrink = colorbar_config.get("shrink", 0.9)
    sm = ScalarMappable(norm=mcolors.Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=axes_list, fraction=cbar_fraction, pad=cbar_pad, shrink=cbar_shrink)


def _create_difference_colorbar(fig, axes, vabs, cmap, label=None, fraction=0.015, pad=0.01, shrink=0.8, aspect=30, fontsize=9):
    if vabs <= 0:
        return None
    sm = ScalarMappable(norm=mcolors.TwoSlopeNorm(vmin=-vabs, vcenter=0.0, vmax=vabs), cmap=cmap)
    sm.set_array([])
    axes_list = axes.ravel().tolist() if hasattr(axes, 'ravel') else axes if isinstance(axes, list) else [axes]
    cbar = fig.colorbar(sm, ax=axes_list, fraction=fraction, pad=pad, shrink=shrink, aspect=aspect)
    if label:
        cbar.set_label(label, fontsize=fontsize)
    return cbar


def _add_diff_colorbar(fig, ax, vabs, cmap, config=None):
    if vabs > 0:
        plot_cfg = get_plot_config(config)
        tfr_config = plot_cfg.plot_type_configs.get("tfr", {})
        colorbar_config = tfr_config.get("colorbar", {})
        cbar_fraction = colorbar_config.get("fraction", 0.045)
        cbar_pad = colorbar_config.get("pad", 0.06)
        cbar_shrink = colorbar_config.get("shrink", 0.9)
        sm = ScalarMappable(norm=mcolors.TwoSlopeNorm(vmin=-vabs, vcenter=0.0, vmax=vabs), cmap=cmap)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, fraction=cbar_fraction, pad=cbar_pad, shrink=cbar_shrink)


def _compute_band_diff_data(tfr_pain, tfr_non, tfr_max, tfr_min, fmin, fmax_eff, tmin, tmax):
    pain_data = _average_tfr_band(tfr_pain, fmin=fmin, fmax=fmax_eff, tmin=tmin, tmax=tmax)
    non_data = _average_tfr_band(tfr_non, fmin=fmin, fmax=fmax_eff, tmin=tmin, tmax=tmax)
    max_data = _average_tfr_band(tfr_max, fmin=fmin, fmax=fmax_eff, tmin=tmin, tmax=tmax)
    min_data = _average_tfr_band(tfr_min, fmin=fmin, fmax=fmax_eff, tmin=tmin, tmax=tmax)
    
    if pain_data is None or non_data is None or max_data is None or min_data is None:
        return None, None
    
    pain_diff_data = pain_data - non_data
    temp_diff_data = max_data - min_data
    return pain_diff_data, temp_diff_data


def _compute_band_diff_data_windows(tfr_pain, tfr_non, tfr_max, tfr_min, fmin, fmax_eff, 
                                    window_starts, window_ends, has_temp):
    pain_diff_data_windows = []
    temp_diff_data_windows = []
    
    for tmin_win, tmax_win in zip(window_starts, window_ends):
        pain_data = _average_tfr_band(tfr_pain, fmin=fmin, fmax=fmax_eff, tmin=tmin_win, tmax=tmax_win)
        non_data = _average_tfr_band(tfr_non, fmin=fmin, fmax=fmax_eff, tmin=tmin_win, tmax=tmax_win)
        
        if pain_data is not None and non_data is not None:
            pain_diff_data_windows.append(pain_data - non_data)
        else:
            pain_diff_data_windows.append(None)
        
        if has_temp and tfr_max is not None and tfr_min is not None:
            max_data = _average_tfr_band(tfr_max, fmin=fmin, fmax=fmax_eff, tmin=tmin_win, tmax=tmax_win)
            min_data = _average_tfr_band(tfr_min, fmin=fmin, fmax=fmax_eff, tmin=tmin_win, tmax=tmax_win)
            
            if max_data is not None and min_data is not None:
                temp_diff_data_windows.append(max_data - min_data)
            else:
                temp_diff_data_windows.append(None)
        else:
            temp_diff_data_windows.append(None)
    
    return pain_diff_data_windows, temp_diff_data_windows


def _plot_temporal_topomaps_for_bands(
    tfr_pain, tfr_non, tfr_sub, tfr_max, tfr_min,
    pain_mask, non_mask, mask_max, mask_min,
    window_starts, window_ends, has_temp, t_min, t_max,
    tmin_clip, tmax_clip, n_windows, baseline_used,
    window_label, filename_base, out_dir, config, logger
):
    fmax_available = float(np.max(tfr_pain.freqs))
    bands = get_bands_for_tfr(max_freq_available=fmax_available, config=config)

    valid_bands = {}
    all_band_pain_diff_data = {}
    all_band_temp_diff_data = {}

    for band_name, (fmin, fmax) in bands.items():
        fmax_eff = min(fmax, fmax_available)
        if fmin >= fmax_eff:
            continue

        pain_diff_data_windows, temp_diff_data_windows = _compute_band_diff_data_windows(
            tfr_pain, tfr_non, tfr_max, tfr_min, fmin, fmax_eff, window_starts, window_ends, has_temp
        )

        pain_diff_data_valid = [d for d in pain_diff_data_windows if d is not None]
        temp_diff_data_valid = [d for d in temp_diff_data_windows if d is not None]

        if len(pain_diff_data_valid) == 0:
            _log(f"No valid data found for {band_name} temporal topomaps; skipping this band.", logger, "warning")
            continue

        valid_bands[band_name] = (fmin, fmax_eff, pain_diff_data_windows, temp_diff_data_windows if has_temp else None)
        all_band_pain_diff_data[band_name] = pain_diff_data_valid
        if has_temp:
            all_band_temp_diff_data[band_name] = temp_diff_data_valid

    if len(valid_bands) == 0:
        _log("No valid bands found for temporal topomaps; skipping.", logger, "warning")
        return

    all_pain_diff_data = [d for data_list in all_band_pain_diff_data.values() for d in data_list]
    all_temp_diff_data = [d for data_list in all_band_temp_diff_data.values() for d in data_list] if has_temp else []
    all_diff_data = all_pain_diff_data + all_temp_diff_data
    vabs_diff = robust_sym_vlim(all_diff_data) if len(all_diff_data) > 0 else 1e-6

    viz_params = get_viz_params(config)
    font_sizes = _get_font_sizes()
    baseline_str = f"bl{abs(baseline_used[0]):.1f}to{abs(baseline_used[1]):.2f}" if baseline_used else "bl"
    sig_text = _get_sig_marker_text(config)
    
    _log(f"Creating separate figure for each of {len(valid_bands)} frequency bands...", logger)
    
    temporal_spacing = config.get("time_frequency_analysis.topomap.temporal.single_subject", {}) if config else {}
    hspace = temporal_spacing.get("hspace", 0.4)
    wspace = temporal_spacing.get("wspace", 0.3)
    
    for band_name, (fmin, fmax_eff, pain_diff_data_windows, temp_diff_data_windows) in valid_bands.items():
        freq_label = f"{band_name} ({fmin:.0f}-{fmax_eff:.0f}Hz)"
        n_rows = 2 if (has_temp and temp_diff_data_windows is not None) else 1
        
        topo_size = max(1.2, min(3.0, 1800.0 / n_windows))
        fig_width = min(1800.0, topo_size * n_windows)
        fig, axes = plt.subplots(
            n_rows, n_windows, figsize=(fig_width * 1.5, 3.5 * n_rows), squeeze=False,
            gridspec_kw={"hspace": hspace, "wspace": wspace}
        )
        
        row_pain = 0
        row_temp = 1 if n_rows == 2 else None
        
        axes[row_pain, 0].set_ylabel(f"Pain - Non\n{freq_label}", fontsize=font_sizes["ylabel"], labelpad=10)

        for col, (tmin_win, tmax_win) in enumerate(zip(window_starts, window_ends)):
            time_label = f"{tmin_win:.2f}s"
            axes[row_pain, col].set_title(time_label, fontsize=font_sizes["title"], pad=12, y=1.07)

            pain_diff_data = pain_diff_data_windows[col]
            _plot_single_topomap_window(
                axes[row_pain, col], pain_diff_data, tfr_pain.info, tfr_sub,
                pain_mask, non_mask, fmin, fmax_eff, tmin_win, tmax_win,
                vabs_diff, config, viz_params, paired=False, logger=logger
            )

            if n_rows == 2 and temp_diff_data_windows is not None:
                temp_diff_data = temp_diff_data_windows[col]
                _plot_single_topomap_window(
                    axes[row_temp, col], temp_diff_data, tfr_max.info, tfr_sub,
                    mask_max, mask_min, fmin, fmax_eff, tmin_win, tmax_win,
                    vabs_diff, config, viz_params, paired=False, logger=logger
                )
                
                if col == 0:
                    axes[row_temp, 0].set_ylabel(f"Max - Min temp\n{freq_label}", fontsize=font_sizes["ylabel"], labelpad=10)

        _create_difference_colorbar(
            fig, axes, vabs_diff, viz_params["topo_cmap"],
            label="log10(power/baseline) difference"
        )

        title_parts = [f"Temporal topomaps: Pain - Non-pain difference ({band_name}, {window_label})"]
        if n_rows == 2:
            title_parts.append(f"Max - Min temp ({t_min:.1f}-{t_max:.1f}°C)")
        title_parts.append(f"log10(power/baseline) difference, vlim ±{vabs_diff:.2f}")
        fig.suptitle(
            f"{' | '.join(title_parts)}{sig_text}\n",
            fontsize=font_sizes["suptitle"], y=0.995
        )

        filename = filename_base.format(band_name=band_name, tmin=tmin_clip, tmax=tmax_clip, n_windows=n_windows, baseline_str=baseline_str)
        _save_fig(fig, out_dir, filename, config=config, logger=logger)
        plt.close(fig)


def _plot_single_topomap_window(ax, diff_data, info, tfr_sub, mask_a, mask_b, fmin, fmax_eff, 
                                 tmin_win, tmax_win, vabs_diff, config, viz_params, paired=False, logger=None):
    if diff_data is None:
        ax.axis('off')
        return
    
    sig_mask = cluster_p_min = cluster_k = cluster_mass = None
    if viz_params["diff_annotation_enabled"]:
        diff_data_len = len(diff_data) if diff_data is not None else None
        sig_mask, cluster_p_min, cluster_k, cluster_mass = _compute_cluster_significance(
            tfr_sub, mask_a, mask_b, fmin, fmax_eff, tmin_win, tmax_win, config, diff_data_len=diff_data_len, logger=logger
        )
    
    plot_topomap_on_ax(
        ax, diff_data, info,
        vmin=-vabs_diff, vmax=+vabs_diff,
        mask=(sig_mask if viz_params["diff_annotation_enabled"] else None),
        mask_params=viz_params["sig_mask_params"],
        config=config
    )
    
    data_group_a = extract_trial_band_power(tfr_sub[mask_a], fmin, fmax_eff, tmin_win, tmax_win)
    data_group_b = extract_trial_band_power(tfr_sub[mask_b], fmin, fmax_eff, tmin_win, tmax_win)
    _add_roi_annotations(
        ax, diff_data, info, config=config,
        sig_mask=(sig_mask if viz_params["diff_annotation_enabled"] else None),
        cluster_p_min=cluster_p_min, cluster_k=cluster_k, cluster_mass=cluster_mass,
        is_cluster=(sig_mask is not None and cluster_p_min is not None),
        data_group_a=data_group_a,
        data_group_b=data_group_b,
        paired=paired
    )
    
    label = _build_topomap_diff_label(diff_data, cluster_p_min, cluster_k, cluster_mass, config, viz_params, paired=paired)
    font_sizes = _get_font_sizes()
    plot_cfg = get_plot_config(config) if config else None
    tfr_config = plot_cfg.plot_type_configs.get("tfr", {}) if plot_cfg else {}
    label_x_position = tfr_config.get("label_x_position", 0.5) if plot_cfg else 0.5
    label_y_position_bottom = tfr_config.get("label_y_position_bottom", 1.08) if plot_cfg else 1.08
    ax.text(label_x_position, label_y_position_bottom, label, transform=ax.transAxes, ha="center", va="bottom", fontsize=font_sizes["label"])


def _compute_cluster_significance(tfr, mask1, mask2, fmin, fmax_eff, tmin, tmax, config, diff_data_len=None, logger=None):
    try:
        sig_mask, cluster_p_min, cluster_k, cluster_mass = _cluster_test_epochs(
            tfr, mask1, mask2, fmin=fmin, fmax=fmax_eff, tmin=tmin, tmax=tmax, paired=False, config=config
        )
        if sig_mask is not None and diff_data_len is not None and len(sig_mask) != diff_data_len:
            if logger:
                _log(f"Cluster significance mask length mismatch: sig_mask length={len(sig_mask)}, diff_data_len={diff_data_len}. Discarding results.", logger, "warning")
            return None, None, None, None
        return sig_mask, cluster_p_min, cluster_k, cluster_mass
    except (ValueError, RuntimeError) as e:
        if logger:
            _log(f"Cluster test failed: {type(e).__name__}: {e}", logger, "warning")
        strict_mode = _get_strict_mode(config)
        if strict_mode:
            raise
        return None, None, None, None


def _compute_cluster_significance_from_combined(tfr1, tfr2, fmin, fmax_eff, tmin, tmax, config, diff_data_len, logger):
    if tfr1 is None or tfr2 is None:
        return None, None, None, None
    
    tfr_combined_all = _concatenate_epochs_tfr_group([tfr1, tfr2], logger)
    if tfr_combined_all is None:
        return None, None, None, None
    
    n1 = len(tfr1)
    n2 = len(tfr2)
    mask1_combined = np.concatenate([np.ones(n1, dtype=bool), np.zeros(n2, dtype=bool)])
    mask2_combined = np.concatenate([np.zeros(n1, dtype=bool), np.ones(n2, dtype=bool)])
    
    return _compute_cluster_significance(tfr_combined_all, mask1_combined, mask2_combined, fmin, fmax_eff, tmin, tmax, config, diff_data_len, logger=logger)


def _plot_diff_topomap_with_label(ax, diff_data, info, vabs, sig_mask, cluster_p_min, cluster_k, cluster_mass, config, viz_params,
                                   data_group_a=None, data_group_b=None, paired=False):
    if vabs > 0:
        vmin = -vabs
        vmax = +vabs
    else:
        if diff_data is not None:
            diff_data_arr = np.asarray(diff_data)
            diff_finite = diff_data_arr[np.isfinite(diff_data_arr)]
            has_nonzero = diff_finite.size > 0 and np.any(np.abs(diff_finite) > 0)
        else:
            has_nonzero = False
        if has_nonzero:
            vabs_effective = robust_sym_vlim(diff_data)
            vmin = -vabs_effective
            vmax = +vabs_effective
        else:
            vmin = None
            vmax = None
    plot_topomap_on_ax(
        ax,
        diff_data,
        info,
        mask=(sig_mask if viz_params["diff_annotation_enabled"] else None),
        mask_params=viz_params["sig_mask_params"],
        vmin=vmin,
        vmax=vmax,
        config=config,
    )
    _add_roi_annotations(
        ax, diff_data, info, config=config,
        sig_mask=(sig_mask if viz_params["diff_annotation_enabled"] else None),
        cluster_p_min=cluster_p_min, cluster_k=cluster_k, cluster_mass=cluster_mass,
        is_cluster=(sig_mask is not None and cluster_p_min is not None),
        data_group_a=data_group_a,
        data_group_b=data_group_b,
        paired=paired
    )
    label = _build_topomap_diff_label(diff_data, cluster_p_min, cluster_k, cluster_mass, config, viz_params, paired=paired)
    font_sizes = _get_font_sizes()
    plot_cfg = get_plot_config(config) if config else None
    tfr_config = plot_cfg.plot_type_configs.get("tfr", {}) if plot_cfg else {}
    label_x_position = tfr_config.get("label_x_position", 0.5) if plot_cfg else 0.5
    label_y_position = tfr_config.get("label_y_position", 1.02) if plot_cfg else 1.02
    ax.text(label_x_position, label_y_position, label, transform=ax.transAxes, ha="center", va="top", fontsize=font_sizes["title"])


def _create_colorbar_for_topomaps(fig, axes, vmin, vmax, cmap, colorbar_pad, colorbar_fraction, config=None):
    sm = ScalarMappable(norm=mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax), cmap=cmap)
    sm.set_array([])
    plot_cfg = get_plot_config(config) if config else None
    if plot_cfg:
        tfr_config = plot_cfg.plot_type_configs.get("tfr", {})
        colorbar_config = tfr_config.get("colorbar", {})
        colorbar_multiplier = colorbar_config.get("multiplier", 8.0)
    else:
        colorbar_multiplier = 8.0
    pad = colorbar_pad * colorbar_multiplier
    fig.colorbar(sm, ax=axes, fraction=colorbar_fraction, pad=pad)


def _compute_significance_mask(tfr_sub, mask_a, mask_b, fmin, fmax, tmin, tmax, config):
    viz_params = get_viz_params(config)
    if not viz_params["diff_annotation_enabled"]:
        return None, {}
    
    sig_mask, cluster_p_min, cluster_k, cluster_mass = _cluster_test_epochs(
        tfr_sub, mask_a, mask_b, fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax, paired=False, config=config
    )
    return sig_mask, {
        "cluster_p_min": cluster_p_min,
        "cluster_k": cluster_k,
        "cluster_mass": cluster_mass
    }


def _prepare_pain_contrast_data(tfr, events_df, pain_col, config, logger=None):
    if not require_epochs_tfr(tfr, "Contrast", logger):
        return None, None, None, None
    
    if pain_col is None:
        _log("Events with pain binary column required for contrast; skipping.", logger, "warning")
        return None, None, None, None
    
    n = compute_aligned_data_length(tfr, events_df)
    
    pain_vec = extract_pain_vector(tfr, events_df, pain_col, n)
    if pain_vec is None:
        return None, None, None, None
    
    pain_mask = pain_vec == 1
    non_mask = pain_vec == 0
    
    _log(f"Pain/non-pain counts (n={n}): pain={int(pain_mask.sum())}, non-pain={int(non_mask.sum())}.", logger)
    
    if pain_mask.sum() == 0 or non_mask.sum() == 0:
        _log("One of the groups has zero trials; skipping contrasts.", logger, "warning")
        return None, None, None, None
    
    tfr_sub = create_tfr_subset(tfr, n)
    try:
        ensure_aligned_lengths(
            tfr_sub, pain_mask, non_mask,
            context="Pain contrast",
            strict=_get_strict_mode(config),
            logger=logger
        )
    except ValueError as e:
        _log(f"{e}. Skipping contrast.", logger, "error")
        return None, None, None, None
    
    if len(pain_mask) != len(tfr_sub):
        pain_mask = pain_mask[:len(tfr_sub)]
        non_mask = non_mask[:len(tfr_sub)]
    
    return tfr_sub, pain_mask, non_mask, n


def _plot_single_tfr_figure(tfr, central_ch, vlim, title, filename, out_dir, config, logger, baseline_used, subject=None, task=None, band=None):
    font_sizes = _get_font_sizes()
    plot_kwargs = {"picks": central_ch, "show": False}
    if vlim is not None:
        plot_kwargs["vlim"] = vlim
    fig = unwrap_figure(tfr.plot(**plot_kwargs))
    fig.suptitle(title, fontsize=font_sizes["figure_title"])
    _save_fig(fig, out_dir, filename, config=config, logger=logger, baseline_used=baseline_used, subject=subject, task=task, band=band)


def _compute_plateau_statistics(arr, times, plateau_window, config, logger):
    tmin_req, tmax_req = plateau_window
    strict_mode = _get_strict_mode(config)
    if strict_mode:
        tmask = create_time_mask_strict(times, tmin_req, tmax_req)
    else:
        tmask = create_time_mask_loose(times, tmin_req, tmax_req, logger)
    mu = float(np.nanmean(arr[:, tmask]))
    pct = logratio_to_pct(mu)
    return mu, pct, tmask


def _plot_central_channel_contrast(tfr_pain, tfr_non, central_ch, plateau_window, baseline_used, out_dir, config, logger=None):
    ch_idx = tfr_pain.info["ch_names"].index(central_ch)
    arr_pain = np.asarray(tfr_pain.data[ch_idx])
    arr_non = np.asarray(tfr_non.data[ch_idx])
    vabs_pn = robust_sym_vlim([arr_pain, arr_non])
    
    times = np.asarray(tfr_pain.times)
    _, pct_pain, _ = _compute_plateau_statistics(arr_pain, times, plateau_window, config, logger)
    _, pct_non, _ = _compute_plateau_statistics(arr_non, times, plateau_window, config, logger)
    
    _plot_single_tfr_figure(
        tfr_pain, central_ch, (-vabs_pn, +vabs_pn),
        f"{central_ch} — Pain (baseline logratio)\nvlim ±{vabs_pn:.2f}; mean %Δ vs BL={pct_pain:+.0f}%",
        f"tfr_{central_ch}_pain_bl.png", out_dir, config, logger, baseline_used
    )
    
    _plot_single_tfr_figure(
        tfr_non, central_ch, (-vabs_pn, +vabs_pn),
        f"{central_ch} — Non-pain (baseline logratio)\nvlim ±{vabs_pn:.2f}; mean %Δ vs BL={pct_non:+.0f}%",
        f"tfr_{central_ch}_nonpain_bl.png", out_dir, config, logger, baseline_used
    )
    
    tfr_diff = tfr_pain.copy()
    tfr_diff.data = tfr_pain.data - tfr_non.data
    tfr_diff.comment = "pain-minus-nonpain"
    
    arr_diff = np.asarray(arr_pain) - np.asarray(arr_non)
    vabs_diff = robust_sym_vlim(arr_diff)
    _, pct_diff, _ = _compute_plateau_statistics(arr_diff, times, plateau_window, config, logger)
    
    _plot_single_tfr_figure(
        tfr_diff, central_ch, (-vabs_diff, +vabs_diff),
        f"{central_ch} — Pain minus Non (baseline logratio)\nvlim ±{vabs_diff:.2f}; Δ% vs BL={pct_diff:+.0f}%",
        f"tfr_{central_ch}_pain_minus_non_bl.png", out_dir, config, logger, baseline_used
    )


def _collect_pain_nonpain_avg_tfrs(powers, events_by_subj, config, baseline, logger):
    avg_pain = []
    avg_non = []
    
    for power, ev in zip(powers, events_by_subj):
        if ev is None:
            continue
        pain_col = get_pain_column_from_config(config, ev)
        if pain_col is None:
            continue
        pain_mask, non_mask = _create_pain_masks_from_events(ev, pain_col)
        if pain_mask is None or pain_mask.sum() == 0 or non_mask.sum() == 0:
            continue
        a_p = avg_by_mask_to_avg_tfr(power, pain_mask, baseline=baseline, logger=logger)
        a_n = avg_by_mask_to_avg_tfr(power, non_mask, baseline=baseline, logger=logger)
        if a_p is None or a_n is None:
            continue
        avg_pain.append(a_p)
        avg_non.append(a_n)
    
    return avg_pain, avg_non


def _create_group_roi_tfr(template_tfr, data, info, nave, comment):
    grp = template_tfr.copy()
    grp.data = data
    grp.info = info
    grp.nave = nave
    grp.comment = comment
    return grp


def _create_group_scalpmean_tfr(template_tfr, data, sfreq, nave, comment):
    tfr = template_tfr.copy()
    tfr.data = data
    tfr.info = mne.create_info(["AllEEG"], sfreq=sfreq, ch_types='eeg')
    tfr.nave = nave
    tfr.comment = comment
    return tfr




def _collect_roi_contrast_data(powers, events_by_subj, roi, roi_map, config, logger):
    roi_p_list = []
    roi_n_list = []
    
    for power, ev in zip(powers, events_by_subj):
        contrast_result = extract_roi_contrast_data(power, ev, roi, roi_map, config)
        r_p, r_n = (None, None) if contrast_result is None else contrast_result
        if r_p is not None and r_n is not None:
            roi_p_list.append(r_p)
            roi_n_list.append(r_n)
    
    if len(roi_p_list) < 1 or len(roi_n_list) < 1:
        if roi_map is not None:
            roi_p_list = []
            roi_n_list = []
            for power, ev in zip(powers, events_by_subj):
                contrast_result = extract_roi_contrast_data(power, ev, roi, None, config)
                r_p, r_n = (None, None) if contrast_result is None else contrast_result
                if r_p is not None and r_n is not None:
                    roi_p_list.append(r_p)
                    roi_n_list.append(r_n)
    
    return roi_p_list, roi_n_list








def _align_and_trim_masks(tfr_sub, masks_dict, config, logger):
    try:
        for context, masks in masks_dict.items():
            if masks is None:
                continue
            mask_list = masks if isinstance(masks, (list, tuple)) else [masks]
            ensure_aligned_lengths(
                tfr_sub, *mask_list,
                context=context,
                strict=_get_strict_mode(config),
                logger=logger
            )
        
        tfr_len = len(tfr_sub)
        trimmed = {}
        for key, masks in masks_dict.items():
            if masks is None:
                trimmed[key] = None
            elif isinstance(masks, (list, tuple)):
                trimmed[key] = [m[:tfr_len] if len(m) != tfr_len else m for m in masks]
            else:
                trimmed[key] = masks[:tfr_len] if len(masks) != tfr_len else masks
        
        return trimmed
    except ValueError as e:
        _log(f"{e}. Skipping alignment.", logger, "error")
        return None


def _pick_central_channel(info, preferred="Cz", logger=None):
    ch_names = info["ch_names"]
    if preferred in ch_names:
        return preferred
    
    for ch_name in ch_names:
        if ch_name.lower() == preferred.lower():
            return ch_name
    
    picks = extract_eeg_picks(info, exclude_bads=False)
    if len(picks) == 0:
        raise RuntimeError("No EEG channels available for plotting.")
    
    fallback = ch_names[picks[0]]
    _log(f"Channel '{preferred}' not found; using '{fallback}' instead.", logger, "warning")
    return fallback


def _pick_central_channel_group(info, preferred="Cz"):
    ch_names = info["ch_names"]
    if preferred in ch_names:
        return preferred
    
    for ch_name in ch_names:
        if ch_name.lower() == preferred.lower():
            return ch_name
    
    picks = extract_eeg_picks(info, exclude_bads=False)
    if len(picks) == 0:
        raise RuntimeError("No EEG channels available for plotting.")
    
    raise ValueError(
        f"Channel '{preferred}' not found in group-level analysis. Ensure all subjects have '{preferred}' channel."
    )




def _build_filename_stem(name, baseline_used, subject=None, task=None, band=None):
    stem, _ = (name.rsplit(".", 1) + [""])[:2]
    
    header_parts = []
    if subject:
        header_parts.append(f"sub-{subject}")
    if task:
        header_parts.append(f"task-{task}")
    if band:
        header_parts.append(f"band-{band}")
    
    baseline_str = format_baseline_window_string(baseline_used)
    if baseline_str not in stem:
        stem = f"{stem}_{baseline_str}"
    if header_parts:
        stem = f"{'_'.join(header_parts)}_{stem}"
    
    return stem


def _build_footer_text(config, baseline_used):
    default_footer_template = "tfr_baseline"
    baseline_decimal_places = 2
    
    if not hasattr(config, "get"):
        return None
    
    from eeg_pipeline.utils.io_utils import build_footer as _build_footer
    template_name = config.get("output.tfr_footer_template", default_footer_template)
    footer_kwargs = {
        "baseline_window": baseline_used,
        "baseline": f"[{float(baseline_used[0]):.{baseline_decimal_places}f}, {float(baseline_used[1]):.{baseline_decimal_places}f}] s",
    }
    return _build_footer(template_name, config, **footer_kwargs)




def _save_fig(
    fig_obj,
    out_dir: Path,
    name: str,
    config,
    formats=None,
    logger: Optional[logging.Logger] = None,
    baseline_used: Optional[Tuple[float, float]] = None,
    subject: Optional[str] = None,
    task: Optional[str] = None,
    band: Optional[str] = None,
):
    from eeg_pipeline.utils.io_utils import save_fig as _central_save_fig
    out_dir.mkdir(parents=True, exist_ok=True)

    if baseline_used is None:
        plot_cfg = get_plot_config(config)
        tfr_config = plot_cfg.plot_type_configs.get("tfr", {})
        default_baseline_start = tfr_config.get("default_baseline_start", -5.0)
        default_baseline_end = tfr_config.get("default_baseline_end", -0.01)
        default_baseline_window = [default_baseline_start, default_baseline_end]
        baseline_used = tuple(config.get("time_frequency_analysis.baseline_window", default_baseline_window))

    figs = fig_obj if isinstance(fig_obj, list) else [fig_obj]
    stem = _build_filename_stem(name, baseline_used, subject, task, band)
    
    plot_cfg = get_plot_config(config)
    exts = formats if formats else list(plot_cfg.formats) if plot_cfg.formats else ["png"]
    
    footer_text = _build_footer_text(config, baseline_used)

    for i, f in enumerate(figs):
        out_name = f"{stem}.{exts[0]}" if i == 0 else f"{stem}_{i+1}.{exts[0]}"
        out_path = out_dir / out_name
        _central_save_fig(
            f,
            out_path,
            logger=logger,
            footer=footer_text,
            formats=tuple(exts),
            dpi=plot_cfg.dpi,
            bbox_inches=plot_cfg.bbox_inches,
            pad_inches=plot_cfg.pad_inches,
        )


###################################################################
# Single-Subject Plots
###################################################################

def plot_cz_all_trials_raw(tfr, out_dir: Path, config, logger: Optional[logging.Logger] = None) -> None:
    tfr_avg = tfr.copy().average() if isinstance(tfr, mne.time_frequency.EpochsTFR) else tfr.copy()
    central_ch = _pick_central_channel(tfr_avg.info, preferred="Cz", logger=logger)
    fig = unwrap_figure(tfr_avg.plot(picks=central_ch, show=False))
    font_sizes = _get_font_sizes()
    fig.suptitle(f"{central_ch} TFR — all trials (raw, no baseline)", fontsize=font_sizes["figure_title"])
    _save_fig(fig, out_dir, f"tfr_{central_ch}_all_trials_raw.png", config=config, logger=logger)


def plot_cz_all_trials(
    tfr,
    out_dir: Path,
    config,
    baseline: Tuple[Optional[float], Optional[float]],
    plateau_window: Tuple[float, float],
    logger: Optional[logging.Logger] = None,
    subject: Optional[str] = None,
    task: Optional[str] = None,
) -> None:
    tfr_avg, baseline_used = apply_baseline_and_average(tfr, baseline, logger)

    central_ch = _pick_central_channel(tfr_avg.info, preferred="Cz", logger=logger)
    ch_idx = tfr_avg.info["ch_names"].index(central_ch)
    arr = np.asarray(tfr_avg.data[ch_idx])
    vabs = robust_sym_vlim(arr)
    times = np.asarray(tfr_avg.times)
    _, pct, _ = _compute_plateau_statistics(arr, times, plateau_window, config, logger)

    _plot_single_tfr_figure(
        tfr_avg, central_ch, (-vabs, +vabs),
        f"{central_ch} TFR — all trials (baseline logratio)\nvlim ±{vabs:.2f}; mean %Δ vs BL={pct:+.0f}%",
        f"tfr_{central_ch}_all_trials.png",
        out_dir, config, logger, baseline_used
    )


def plot_channels_all_trials(
    tfr,
    out_dir: Path,
    config,
    baseline: Tuple[Optional[float], Optional[float]],
    logger: Optional[logging.Logger] = None,
    subject: Optional[str] = None,
    task: Optional[str] = None,
    channels: Optional[List[str]] = None,
) -> None:
    tfr_avg, baseline_used = apply_baseline_and_average(tfr, baseline, logger)

    ch_names = tfr_avg.info["ch_names"]
    if channels is not None:
        channels_set = {ch.upper() for ch in channels}
        ch_names = [ch for ch in ch_names if ch.upper() in channels_set]
        if not ch_names:
            if logger:
                logger.warning(f"No matching channels found for specified channels: {channels}")
            return
    
    ch_dir = out_dir / "channels"
    ch_dir.mkdir(parents=True, exist_ok=True)

    fmax_available = float(np.max(tfr_avg.freqs))
    bands = get_bands_for_tfr(max_freq_available=fmax_available, config=config)

    for ch in ch_names:
        _plot_single_tfr_figure(
            tfr_avg, ch, None, f"{ch} — all trials (baseline logratio)",
            f"tfr_{ch}_all_trials.png", ch_dir, config, logger, baseline_used,
            subject=subject, task=task
        )

        for band, (fmin, fmax) in bands.items():
            fmax_eff = min(fmax, fmax_available)
            if fmin >= fmax_eff:
                continue
            band_dir = ch_dir / band
            band_dir.mkdir(parents=True, exist_ok=True)

            tfr_band = tfr_avg.copy()
            freq_mask = (np.asarray(tfr_band.freqs) >= fmin) & (np.asarray(tfr_band.freqs) <= fmax_eff)
            if not freq_mask.any():
                continue
            
            fig = unwrap_figure(tfr_band.plot(picks=ch, fmin=fmin, fmax=fmax_eff, show=False))
            font_sizes = _get_font_sizes()
            fig.suptitle(f"{ch} — {band} band (baseline logratio)", fontsize=font_sizes["figure_title"])
            _save_fig(
                fig, band_dir, f"tfr_{ch}_{band}_all_trials.png",
                config=config, logger=logger, baseline_used=baseline_used,
                subject=subject, task=task, band=band
            )


###################################################################
# Scalp-Averaged TFR (Single Subject)
###################################################################

def plot_scalpmean_all_trials(
    tfr,
    out_dir: Path,
    config,
    baseline: Tuple[Optional[float], Optional[float]],
    plateau_window: Tuple[float, float],
    logger: Optional[logging.Logger] = None,
    subject: Optional[str] = None,
    task: Optional[str] = None,
) -> None:
    tfr_avg, baseline_used = apply_baseline_and_average(tfr, baseline, logger)
    
    eeg_picks = extract_eeg_picks(tfr_avg, exclude_bads=False)
    if len(eeg_picks) == 0:
        _log("No EEG channels found for scalp-averaged plot", logger, "warning")
        return
    
    tfr_sm = _create_scalpmean_tfr_from_existing(tfr_avg, eeg_picks)
    
    times = np.asarray(tfr_sm.times)
    arr = np.asarray(tfr_sm.data[0])
    vabs = robust_sym_vlim(arr)
    _, pct, _ = _compute_plateau_statistics(arr, times, plateau_window, config, logger)
    
    _plot_scalpmean_tfr(
        tfr_sm,
        f"Scalp-averaged TFR — all trials (baseline logratio)\nvlim ±{vabs:.2f}; mean %Δ vs BL={pct:+.0f}%",
        "tfr_scalpmean_all_trials.png",
        (-vabs, +vabs),
        out_dir, config, logger, baseline_used, subject, task
    )


def contrast_scalpmean_pain_nonpain(
    tfr: "mne.time_frequency.EpochsTFR | mne.time_frequency.AverageTFR",
    events_df: Optional[pd.DataFrame],
    out_dir: Path,
    config,
    baseline: Tuple[Optional[float], Optional[float]],
    plateau_window: Tuple[float, float],
    logger: Optional[logging.Logger] = None,
    subject: Optional[str] = None,
    task: Optional[str] = None,
) -> None:
    pain_col = get_pain_column_from_config(config, events_df)
    tfr_sub, pain_mask, non_mask, n = _prepare_pain_contrast_data(tfr, events_df, pain_col, config, logger)
    if tfr_sub is None:
        return

    baseline_used = apply_baseline_and_crop(tfr_sub, baseline=baseline, mode="logratio", logger=logger)
    tfr_pain = tfr_sub[pain_mask].average()
    tfr_non = tfr_sub[non_mask].average()

    eeg_picks_p = extract_eeg_picks(tfr_pain, exclude_bads=False)
    eeg_picks_n = extract_eeg_picks(tfr_non, exclude_bads=False)
    
    if len(eeg_picks_p) == 0 or len(eeg_picks_n) == 0:
        _log("No EEG channels found for scalp-averaged contrast", logger, "warning")
        return
    
    tfr_pain_sm = _create_scalpmean_tfr_from_existing(tfr_pain, eeg_picks_p)
    tfr_non_sm = _create_scalpmean_tfr_from_existing(tfr_non, eeg_picks_n)
    tfr_diff_sm = tfr_pain_sm.copy()
    tfr_diff_sm.data = tfr_pain_sm.data - tfr_non_sm.data
    tfr_diff_sm.comment = "Pain-minus-Non"
    
    arr_pain = np.asarray(tfr_pain_sm.data[0])
    arr_non = np.asarray(tfr_non_sm.data[0])
    arr_diff = np.asarray(tfr_diff_sm.data[0])
    
    vabs_pn = robust_sym_vlim([arr_pain, arr_non])
    vabs_diff = robust_sym_vlim(arr_diff)

    times = np.asarray(tfr_pain.times)
    _, pct_pain, tmask = _compute_plateau_statistics(arr_pain, times, plateau_window, config, logger)
    _, pct_non, _ = _compute_plateau_statistics(arr_non, times, plateau_window, config, logger)
    _, pct_diff, _ = _compute_plateau_statistics(arr_diff, times, plateau_window, config, logger)

    _plot_scalpmean_tfr(
        tfr_pain_sm, f"Scalp-averaged TFR — Pain (baseline logratio)\nvlim ±{vabs_pn:.2f}; mean %Δ vs BL={pct_pain:+.0f}%",
        "tfr_scalpmean_pain_bl.png", (-vabs_pn, +vabs_pn), out_dir, config, logger, baseline_used, subject, task
    )
    _plot_scalpmean_tfr(
        tfr_non_sm, f"Scalp-averaged TFR — Non-pain (baseline logratio)\nvlim ±{vabs_pn:.2f}; mean %Δ vs BL={pct_non:+.0f}%",
        "tfr_scalpmean_nonpain_bl.png", (-vabs_pn, +vabs_pn), out_dir, config, logger, baseline_used, subject, task
    )
    _plot_scalpmean_tfr(
        tfr_diff_sm, f"Scalp-averaged TFR — Pain minus Non-pain (baseline logratio)\nvlim ±{vabs_diff:.2f}; mean %Δ vs BL={pct_diff:+.0f}%",
        "tfr_scalpmean_pain_minus_non_bl.png", (-vabs_diff, +vabs_diff), out_dir, config, logger, baseline_used, subject, task
    )


###################################################################
# Pain vs Non-pain (Subject)
###################################################################

def contrast_channels_pain_nonpain(
    tfr,
    events_df: Optional[pd.DataFrame],
    out_dir: Path,
    config,
    baseline: Tuple[Optional[float], Optional[float]],
    logger: Optional[logging.Logger] = None,
    subject: Optional[str] = None,
    channels: Optional[List[str]] = None,
) -> None:
    pain_col = get_pain_column_from_config(config, events_df)
    tfr_sub, pain_mask, non_mask, n = _prepare_pain_contrast_data(tfr, events_df, pain_col, config, logger)
    if tfr_sub is None:
        return

    baseline_used = apply_baseline_and_crop(tfr_sub, baseline=baseline, mode="logratio", logger=logger)
    tfr_pain = tfr_sub[pain_mask].average()
    tfr_non = tfr_sub[non_mask].average()
    tfr_diff = tfr_pain.copy()
    tfr_diff.data = tfr_pain.data - tfr_non.data

    ch_names = tfr_pain.info["ch_names"]
    if channels is not None:
        channels_set = {ch.upper() for ch in channels}
        ch_names = [ch for ch in ch_names if ch.upper() in channels_set]
        if not ch_names:
            if logger:
                logger.warning(f"No matching channels found for specified channels: {channels}")
            return
    
    ch_dir = out_dir / "channels"
    ch_dir.mkdir(parents=True, exist_ok=True)

    fmax_available = float(np.max(tfr_pain.freqs))
    bands = get_bands_for_tfr(max_freq_available=fmax_available, config=config)

    for ch in ch_names:
        _plot_single_tfr_figure(tfr_pain, ch, None, f"{ch} — Painful (baseline logratio)", f"tfr_{ch}_painful_bl.png", ch_dir, config, logger, baseline_used, subject=subject)
        _plot_single_tfr_figure(tfr_non, ch, None, f"{ch} — Non-pain (baseline logratio)", f"tfr_{ch}_nonpain_bl.png", ch_dir, config, logger, baseline_used, subject=subject)
        _plot_single_tfr_figure(tfr_diff, ch, None, f"{ch} — Pain minus Non-pain (baseline logratio)", f"tfr_{ch}_pain_minus_nonpain_bl.png", ch_dir, config, logger, baseline_used, subject=subject)

        for band, (fmin, fmax) in bands.items():
            fmax_eff = min(fmax, fmax_available)
            if fmin >= fmax_eff:
                continue
            band_dir = ch_dir / band
            band_dir.mkdir(parents=True, exist_ok=True)

            _plot_single_tfr_figure(tfr_pain, ch, None, f"{ch} — {band} Painful (baseline logratio)", f"tfr_{ch}_{band}_painful_bl.png", band_dir, config, logger, baseline_used, subject=subject, band=band)
            _plot_single_tfr_figure(tfr_non, ch, None, f"{ch} — {band} Non-pain (baseline logratio)", f"tfr_{ch}_{band}_nonpain_bl.png", band_dir, config, logger, baseline_used, subject=subject, band=band)
            _plot_single_tfr_figure(tfr_diff, ch, None, f"{ch} — {band} Pain minus Non-pain (baseline logratio)", f"tfr_{ch}_{band}_pain_minus_nonpain_bl.png", band_dir, config, logger, baseline_used, subject=subject, band=band)


def qc_baseline_plateau_power(
    tfr,
    out_dir: Path,
    config,
    baseline: Optional[Tuple[Optional[float], Optional[float]]] = None,
    plateau_window: Tuple[float, float] = (3.0, 10.5),
    logger: Optional[logging.Logger] = None,
) -> None:
    baseline = _get_baseline_window(config, baseline)
    qc_dir = out_dir / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)

    data = getattr(tfr, "data", None)
    if data is None or data.ndim not in [3, 4]:
        return

    if data.ndim == 3:
        data = data[None, ...]

    freqs = np.asarray(tfr.freqs)
    times = np.asarray(tfr.times)

    min_baseline_samples = config.get("tfr_topography_pipeline.min_baseline_samples", 5)
    b_start, b_end, tmask_base_idx = validate_baseline_indices(times, baseline, min_samples=min_baseline_samples, logger=logger)
    tmask_base = np.zeros(len(times), dtype=bool)
    tmask_base[tmask_base_idx] = True
    tmask_plat = (times >= plateau_window[0]) & (times < plateau_window[1])

    if not np.any(tmask_plat):
        _log(f"QC skipped: plateau samples={int(tmask_plat.sum())}", logger, "warning")
        return

    tfr_avg = tfr.average() if isinstance(tfr, mne.time_frequency.EpochsTFR) else tfr
    
    rows = []
    font_sizes = _get_font_sizes()

    band_bounds = config.get("time_frequency_analysis.bands") or config.frequency_bands
    band_bounds_dict = {k: tuple(v) for k, v in band_bounds.items()}
    for band, (fmin, fmax) in band_bounds_dict.items():
        fmask = (freqs >= float(fmin)) & (freqs <= (float(fmax) if fmax is not None else freqs.max()))
        if not np.any(fmask):
            continue

        base = data[:, :, fmask, :][:, :, :, tmask_base].mean(axis=(2, 3))
        plat = data[:, :, fmask, :][:, :, :, tmask_plat].mean(axis=(2, 3))

        base_flat = base.reshape(-1)
        plat_flat = plat.reshape(-1)
        plot_cfg = get_plot_config(config)
        tfr_config = plot_cfg.plot_type_configs.get("tfr", {})
        qc_config = tfr_config.get("qc", {})
        epsilon_for_division = qc_config.get("epsilon_for_division", 1e-20)
        percentage_multiplier = tfr_config.get("percentage_multiplier", 100.0)
        pct_change = ((plat_flat - base_flat) / (base_flat + epsilon_for_division)) * percentage_multiplier

        qc_fig_width = qc_config.get("fig_width", 8)
        qc_fig_height = qc_config.get("fig_height", 3)
        histogram_bins = qc_config.get("histogram_bins", 50)
        histogram_alpha = qc_config.get("histogram_alpha", 0.8)
        fig, axes = plt.subplots(1, 2, figsize=(qc_fig_width, qc_fig_height), constrained_layout=True)
        axes[0].hist(base_flat, bins=histogram_bins, color="tab:blue", alpha=histogram_alpha)
        axes[0].set_title(f"Baseline power — {band}")
        axes[0].set_xlabel("Power (a.u.)")
        axes[0].set_ylabel("Count")
        axes[1].hist(pct_change, bins=histogram_bins, color="tab:orange", alpha=histogram_alpha)
        axes[1].set_title(f"% signal change (plateau vs baseline) — {band}")
        axes[1].set_xlabel("% change")
        axes[1].set_ylabel("Count")
        fig.suptitle(
            f"Baseline vs Plateau QC — {band}\n(baseline={b_start:.2f}–{b_end:.2f}s; plateau={plateau_window[0]:.2f}–{plateau_window[1]:.2f}s)",
            fontsize=font_sizes["ylabel"],
        )
        _save_fig(fig, qc_dir, f"qc_baseline_plateau_hist_{band}.png", config=config, logger=logger)

        topo_vals = None
        if tfr_avg is not None:
            fmin_eff = float(fmin)
            fmax_eff = float(fmax) if fmax is not None else float(freqs.max())
            topo_plat = _average_tfr_band(
                tfr_avg,
                fmin=fmin_eff,
                fmax=fmax_eff,
                tmin=float(plateau_window[0]),
                tmax=float(plateau_window[1]),
            )
            topo_base = _average_tfr_band(
                tfr_avg,
                fmin=fmin_eff,
                fmax=fmax_eff,
                tmin=float(b_start),
                tmax=float(b_end),
            )
            if topo_plat is not None and topo_base is not None:
                plot_cfg_topo_pct = get_plot_config(config)
                tfr_config_topo_pct = plot_cfg_topo_pct.plot_type_configs.get("tfr", {})
                qc_config_topo_pct = tfr_config_topo_pct.get("qc", {})
                epsilon_for_division_topo_pct = qc_config_topo_pct.get("epsilon_for_division", 1e-20)
                percentage_multiplier_topo_pct = tfr_config_topo_pct.get("percentage_multiplier", 100.0)
                topo_vals = ((topo_plat - topo_base) / (topo_base + epsilon_for_division_topo_pct)) * percentage_multiplier_topo_pct

        row = {
            "band": band,
            "baseline_mean": float(np.nanmean(base_flat)),
            "baseline_median": float(np.nanmedian(base_flat)),
            "plateau_mean": float(np.nanmean(plat_flat)),
            "plateau_median": float(np.nanmedian(plat_flat)),
            "pct_change_mean": float(np.nanmean(pct_change)),
            "pct_change_median": float(np.nanmedian(pct_change)),
            "n_baseline_samples": int(tmask_base.sum()),
            "n_plateau_samples": int(tmask_plat.sum()),
        }
        if topo_vals is not None and np.isfinite(topo_vals).any():
            row["pct_change_mean_topomap"] = float(np.nanmean(topo_vals))
            row["pct_change_median_topomap"] = float(np.nanmedian(topo_vals))
        else:
            row["pct_change_mean_topomap"] = float("nan")
            row["pct_change_median_topomap"] = float("nan")
        rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        df_path = qc_dir / "qc_baseline_plateau_summary.tsv"
        df.to_csv(df_path, sep="\t", index=False)
        _log(f"Saved QC summary: {df_path}", logger)


def _turn_off_column_axes(axes, column_idx, n_rows):
    for r in range(n_rows):
        axes[r, column_idx].axis('off')


def _plot_pain_nonpain_topomaps(
    axes, column_idx, pain_data, non_data, diff_data, info,
    vabs_pn, diff_abs, sig_mask, cluster_info, config,
    data_group_a, data_group_b
):
    pain_mu = float(np.nanmean(pain_data))
    pain_pct = logratio_to_pct(pain_mu)
    _plot_topomap_with_label(
        axes[0, column_idx], pain_data, info, -vabs_pn, +vabs_pn,
        f"%Δ={pain_pct:+.1f}%", config
    )

    non_mu = float(np.nanmean(non_data))
    non_pct = logratio_to_pct(non_mu)
    _plot_topomap_with_label(
        axes[1, column_idx], non_data, info, -vabs_pn, +vabs_pn,
        f"%Δ={non_pct:+.1f}%", config
    )

    if diff_abs > 0:
        diff_vmin = -diff_abs
        diff_vmax = +diff_abs
    else:
        if diff_data is not None:
            diff_data_arr = np.asarray(diff_data)
            diff_finite = diff_data_arr[np.isfinite(diff_data_arr)]
            has_nonzero = diff_finite.size > 0 and np.any(np.abs(diff_finite) > 0)
        else:
            has_nonzero = False
        if has_nonzero:
            diff_abs_effective = robust_sym_vlim(diff_data)
            diff_vmin = -diff_abs_effective
            diff_vmax = +diff_abs_effective
        else:
            diff_vmin = None
            diff_vmax = None
    _plot_topomap_with_diff_label(
        axes[2, column_idx], diff_data, info, diff_vmin, diff_vmax,
        sig_mask, cluster_info, config,
        data_group_a=data_group_a,
        data_group_b=data_group_b,
        paired=False
    )


def _add_colorbars_for_column(fig, axes, column_idx, vabs_pn, diff_abs, viz_params, config=None):
    _create_colorbar_for_topomaps(
        fig, [axes[0, column_idx], axes[1, column_idx]], -vabs_pn, +vabs_pn,
        viz_params["topo_cmap"], viz_params["colorbar_pad"], viz_params["colorbar_fraction"], config
    )
    if diff_abs > 0:
        _create_colorbar_for_topomaps(
            fig, axes[2, column_idx], -diff_abs, +diff_abs,
            viz_params["topo_cmap"], viz_params["colorbar_pad"], viz_params["colorbar_fraction"], config
        )


def _finalize_topomap_figure(fig, axes, row_labels, tmin, tmax, config):
    font_sizes = _get_font_sizes()
    axes[0, 0].set_ylabel(row_labels[0], fontsize=font_sizes["ylabel"])
    axes[1, 0].set_ylabel(row_labels[1], fontsize=font_sizes["ylabel"])
    axes[2, 0].set_ylabel(row_labels[2], fontsize=font_sizes["ylabel"])
    sig_text = _get_sig_marker_text(config)
    fig.suptitle(
        f"Topomaps (baseline: logratio; t=[{tmin:.1f}, {tmax:.1f}] s){sig_text}",
        fontsize=font_sizes["figure_title"]
    )
    fig.supxlabel("Frequency bands", fontsize=font_sizes["ylabel"])
    plot_cfg_finalize = get_plot_config(config)
    tfr_config_finalize = plot_cfg_finalize.plot_type_configs.get("tfr", {})
    topomap_config_finalize = tfr_config_finalize.get("topomap", {})
    topo_subplots_right = topomap_config_finalize.get("subplots_right", 0.75)
    fig.subplots_adjust(right=topo_subplots_right)


def contrast_pain_nonpain(
    tfr: "mne.time_frequency.EpochsTFR | mne.time_frequency.AverageTFR",
    events_df: Optional[pd.DataFrame],
    out_dir: Path,
    config,
    baseline: Tuple[Optional[float], Optional[float]],
    plateau_window: Tuple[float, float],
    logger: Optional[logging.Logger] = None,
    subject: Optional[str] = None,
) -> None:
    pain_col = get_pain_column_from_config(config, events_df)
    tfr_sub, pain_mask, non_mask, n = _prepare_pain_contrast_data(tfr, events_df, pain_col, config, logger)
    if tfr_sub is None:
        return

    baseline_used = apply_baseline_and_crop(tfr_sub, baseline=baseline, mode="logratio", logger=logger)
    tfr_pain = tfr_sub[pain_mask].average()
    tfr_non = tfr_sub[non_mask].average()

    central_ch = _pick_central_channel(tfr_pain.info, preferred="Cz", logger=logger)
    _plot_central_channel_contrast(tfr_pain, tfr_non, central_ch, plateau_window, baseline_used, out_dir, config, logger)

    times = np.asarray(tfr_pain.times)
    tmin_eff = float(max(np.min(times), plateau_window[0]))
    tmax_eff = float(min(np.max(times), plateau_window[1]))
    fmax_available = float(np.max(tfr_pain.freqs))
    bands = get_bands_for_tfr(max_freq_available=fmax_available, config=config)
    tmin, tmax = tmin_eff, tmax_eff

    n_pain = int(pain_mask.sum())
    n_non = int(non_mask.sum())
    row_labels = [f"Pain (n={n_pain})", f"Non-pain (n={n_non})", "Pain - Non"]
    n_cols = len(bands)
    plot_cfg_contrast = get_plot_config(config)
    tfr_config_contrast = plot_cfg_contrast.plot_type_configs.get("tfr", {})
    topomap_config_contrast = tfr_config_contrast.get("topomap", {})
    topo_n_rows = topomap_config_contrast.get("n_rows", 3)
    topo_fig_size_per_col = topomap_config_contrast.get("fig_size_per_col", 7.0)
    topo_fig_size_per_row = topomap_config_contrast.get("fig_size_per_row", 7.0)
    topo_wspace = topomap_config_contrast.get("wspace", 1.2)
    topo_hspace = topomap_config_contrast.get("hspace", 0.25)
    fig, axes = plt.subplots(
        topo_n_rows,
        n_cols,
        figsize=(topo_fig_size_per_col * n_cols, topo_fig_size_per_row * topo_n_rows),
        squeeze=False,
        gridspec_kw={"wspace": topo_wspace, "hspace": topo_hspace},
    )
    viz_params = get_viz_params(config)
    for c, (band, (fmin, fmax)) in enumerate(bands.items()):
        fmax_eff = min(fmax, fmax_available)
        if fmin >= fmax_eff:
            _turn_off_column_axes(axes, c, topo_n_rows)
            continue

        pain_data = _average_tfr_band(tfr_pain, fmin=fmin, fmax=fmax_eff, tmin=tmin, tmax=tmax)
        non_data = _average_tfr_band(tfr_non, fmin=fmin, fmax=fmax_eff, tmin=tmin, tmax=tmax)
        if pain_data is None or non_data is None:
            _turn_off_column_axes(axes, c, topo_n_rows)
            continue

        diff_data = pain_data - non_data
        vabs_pn = robust_sym_vlim([pain_data, non_data])
        topo_default_diff_abs = topomap_config_contrast.get("default_diff_abs", 0.0)
        diff_abs = robust_sym_vlim(diff_data) if np.isfinite(diff_data).any() else topo_default_diff_abs

        sig_mask, cluster_info = _compute_significance_mask(
            tfr_sub, pain_mask, non_mask, fmin, fmax_eff, tmin, tmax, config
        )

        data_group_a = extract_trial_band_power(tfr_sub[pain_mask], fmin, fmax_eff, tmin, tmax)
        data_group_b = extract_trial_band_power(tfr_sub[non_mask], fmin, fmax_eff, tmin, tmax)

        _plot_pain_nonpain_topomaps(
            axes, c, pain_data, non_data, diff_data, tfr_pain.info,
            vabs_pn, diff_abs, sig_mask, cluster_info, config,
            data_group_a, data_group_b
        )

        font_sizes = _get_font_sizes()
        axes[0, c].set_title(
            f"{band} ({fmin:.0f}-{fmax_eff:.0f} Hz)",
            fontsize=font_sizes["title"], pad=topomap_config_contrast.get("title_pad", 4), y=topomap_config_contrast.get("title_y", 1.04)
        )

        if c == len(bands) - 1:
            _add_colorbars_for_column(
                fig, axes, c, vabs_pn, diff_abs, viz_params, config
            )
    
    _finalize_topomap_figure(
        fig, axes, row_labels, tmin, tmax, config
    )
    _save_fig(fig, out_dir, "topomap_grid_bands_pain_non_diff_bl.png", config=config)


###################################################################
# Temperature Contrasts (Subject)
###################################################################

def contrast_maxmin_temperature(
    tfr: "mne.time_frequency.EpochsTFR | mne.time_frequency.AverageTFR",
    events_df: Optional[pd.DataFrame],
    out_dir: Path,
    config,
    baseline: Optional[Tuple[Optional[float], Optional[float]]] = None,
    plateau_window: Tuple[float, float] = (3.0, 10.5),
    logger: Optional[logging.Logger] = None,
) -> None:
    baseline = _get_baseline_window(config, baseline)
    if not require_epochs_tfr(tfr, "Max-vs-min temperature contrast", logger):
        return
    if events_df is None:
        _log("Max-vs-min temperature contrast requires events_df; skipping.", logger)
        return
    temp_col = get_temperature_column_from_config(config, events_df)
    if temp_col is None:
        _log("Max-vs-min temperature contrast: no temperature column found; skipping.", logger)
        return

    n = compute_aligned_data_length(tfr, events_df)
    
    temp_series = extract_temperature_series(tfr, events_df, temp_col, n)
    if temp_series is None:
        _log("Max-vs-min temperature contrast: no temperature column found; skipping.", logger)
        return

    t_min, t_max = get_temperature_range(temp_series)
    if t_min is None or t_max is None:
        _log("Max-vs-min temperature contrast: need at least 2 temperature levels; skipping.", logger)
        return

    mask_min, mask_max = create_temperature_masks_from_range(temp_series, t_min, t_max)
    if mask_min.sum() == 0 or mask_max.sum() == 0:
        _log(f"Max-vs-min temperature contrast: zero trials in one group (min n={int(mask_min.sum())}, max n={int(mask_max.sum())}); skipping.", logger)
        return

    tfr_sub = create_tfr_subset(tfr, n)
    try:
        strict_mode = config.get("analysis.strict_mode", True)
        ensure_aligned_lengths(
            tfr_sub, mask_min, mask_max,
            context=f"Temperature contrast",
            strict=strict_mode,
            logger=logger
        )
    except ValueError as e:
        _log(f"{e}. Skipping contrast.", logger, "error")
        return
    if len(mask_min) != len(tfr_sub) or len(mask_max) != len(tfr_sub):
        mask_min = mask_min[:len(tfr_sub)]
        mask_max = mask_max[:len(tfr_sub)]

    baseline_used = apply_baseline_and_crop(tfr_sub, baseline=baseline, mode="logratio", logger=logger)
    tfr_min = tfr_sub[mask_min].average()
    tfr_max = tfr_sub[mask_max].average()

    times = np.asarray(tfr_max.times)
    tmin_req, tmax_req = plateau_window
    tmin_eff = float(max(times.min(), tmin_req))
    tmax_eff = float(min(times.max(), tmax_req))
    tmin, tmax = tmin_eff, tmax_eff

    fmax_available = float(np.max(tfr_max.freqs))
    bands = get_bands_for_tfr(max_freq_available=fmax_available, config=config)

    n_rows = len(bands)
    n_cols = 4
    plot_cfg_large = get_plot_config(config)
    fig_size_per_col_large = plot_cfg_large.get_figure_size("tfr_per_col_large", plot_type="tfr")[0]
    fig_size_per_row_medium = plot_cfg_large.get_figure_size("tfr_per_row_medium", plot_type="tfr")[1]
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_size_per_col_large * n_cols, fig_size_per_row_medium * n_rows),
        squeeze=False,
        gridspec_kw={"width_ratios": [1.0, 1.0, 0.25, 1.0], "wspace": 1.2, "hspace": 0.4},
    )
    for r, (band, (fmin, fmax)) in enumerate(bands.items()):
        fmax_eff = min(fmax, fmax_available)
        if fmin >= fmax_eff:
            for c in range(n_cols):
                axes[r, c].axis("off")
            continue
        max_data = _average_tfr_band(tfr_max, fmin=fmin, fmax=fmax_eff, tmin=tmin, tmax=tmax)
        min_data = _average_tfr_band(tfr_min, fmin=fmin, fmax=fmax_eff, tmin=tmin, tmax=tmax)
        if max_data is None or min_data is None:
            for c in range(n_cols):
                axes[r, c].axis("off")
            continue

        sig_mask = None
        cluster_p_min = cluster_k = cluster_mass = None
        if get_viz_params(config)["diff_annotation_enabled"]:
            sig_mask, cluster_p_min, cluster_k, cluster_mass = _cluster_test_epochs(
                tfr_sub, mask_max, mask_min, fmin=fmin, fmax=fmax_eff, tmin=tmin, tmax=tmax, paired=False, config=config
            )

        diff_data = max_data - min_data
        max_mu = float(np.nanmean(max_data))
        min_mu = float(np.nanmean(min_data))
        diff_mu = float(np.nanmean(diff_data))
        vabs_pn = robust_sym_vlim([max_data, min_data])
        diff_abs = robust_sym_vlim(diff_data) if np.isfinite(diff_data).any() else 0.0

        max_pct = logratio_to_pct(max_mu)
        _plot_topomap_with_label(
            axes[r, 0], max_data, tfr_max.info, -vabs_pn, +vabs_pn,
            f"%Δ={max_pct:+.1f}%", config
        )

        min_pct = logratio_to_pct(min_mu)
        _plot_topomap_with_label(
            axes[r, 1], min_data, tfr_min.info, -vabs_pn, +vabs_pn,
            f"%Δ={min_pct:+.1f}%", config
        )

        axes[r, 2].axis("off")

        diff_vmin = -diff_abs if diff_abs > 0 else None
        diff_vmax = +diff_abs if diff_abs > 0 else None
        cluster_info = {
            "cluster_p_min": cluster_p_min,
            "cluster_k": cluster_k,
            "cluster_mass": cluster_mass
        }
        _plot_topomap_with_diff_label(
            axes[r, 3], diff_data, tfr_max.info, diff_vmin, diff_vmax,
            sig_mask, cluster_info, config
        )

        font_sizes = _get_font_sizes()
        if r == 0:
            title_pad = 4
            title_y = 1.04
            axes[r, 0].set_title(f"Max {t_max:.1f}°C (n={int(mask_max.sum())})", fontsize=font_sizes["title"], pad=title_pad, y=title_y)
            axes[r, 1].set_title(f"Min {t_min:.1f}°C (n={int(mask_min.sum())})", fontsize=font_sizes["title"], pad=title_pad, y=title_y)
            axes[r, 3].set_title("Max - Min", fontsize=font_sizes["title"], pad=title_pad, y=title_y)
        axes[r, 0].set_ylabel(f"{band} ({fmin:.0f}-{fmax_eff:.0f} Hz)", fontsize=font_sizes["ylabel"])

        viz_params = get_viz_params(config)
        _add_normalized_colorbar(fig, [axes[r, 0], axes[r, 1]], -vabs_pn, +vabs_pn, viz_params["topo_cmap"], config)
        _add_diff_colorbar(fig, axes[r, 3], diff_abs, viz_params["topo_cmap"], config)

    sig_text = _get_sig_marker_text(config)
    font_sizes = _get_font_sizes()
    fig.suptitle(
        f"Topomaps: Max vs Min temperature (baseline: logratio; t=[{tmin:.1f}, {tmax:.1f}] s){sig_text}",
        fontsize=font_sizes["figure_title"],
    )
    fig.supylabel("Frequency bands", fontsize=font_sizes["ylabel"])
    _save_fig(
        fig,
        out_dir,
        "topomap_grid_bands_maxmin_temp_diff_bl.png",
        config,
    )


###################################################################
# Combined Pain and Temperature Contrasts (Subject)
###################################################################

def plot_bands_pain_temp_contrasts(
    tfr: "mne.time_frequency.EpochsTFR",
    events_df: Optional[pd.DataFrame],
    out_dir: Path,
    config,
    baseline: Optional[Tuple[Optional[float], Optional[float]]] = None,
    plateau_window: Tuple[float, float] = (3.0, 10.5),
    logger: Optional[logging.Logger] = None,
) -> None:
    baseline = _get_baseline_window(config, baseline)
    if not require_epochs_tfr(tfr, "Combined contrast", logger):
        return
    if events_df is None:
        _log("Combined contrast requires events_df; skipping.", logger, "warning")
        return
    
    pain_col = get_pain_column_from_config(config, events_df)
    temp_col = get_temperature_column_from_config(config, events_df)
    
    if pain_col is None:
        _log("Combined contrast: no pain column found; skipping.", logger, "warning")
        return
    if temp_col is None:
        _log("Combined contrast: no temperature column found; skipping.", logger, "warning")
        return

    n = compute_aligned_data_length(tfr, events_df)

    pain_vec = extract_pain_vector(tfr, events_df, pain_col, n)
    if pain_vec is None:
        _log("Combined contrast: could not extract pain vector; skipping.", logger, "warning")
        return
    
    pain_mask, non_mask = _create_pain_masks_from_vector(pain_vec)
    if pain_mask.sum() == 0 or non_mask.sum() == 0:
        _log("Combined contrast: one pain group has zero trials; skipping.", logger, "warning")
        return
    
    temp_series = extract_temperature_series(tfr, events_df, temp_col, n)
    if temp_series is None:
        _log("Combined contrast: no temperature column found; skipping.", logger, "warning")
        return

    t_min, t_max = get_temperature_range(temp_series)
    if t_min is None or t_max is None:
        _log("Combined contrast: need at least 2 temperature levels; skipping.", logger, "warning")
        return

    mask_min, mask_max = create_temperature_masks_from_range(temp_series, t_min, t_max)
    
    if mask_min.sum() == 0 or mask_max.sum() == 0:
        _log(f"Combined contrast: zero trials in one temp group (min n={int(mask_min.sum())}, max n={int(mask_max.sum())}); skipping.", logger, "warning")
        return

    tfr_sub = create_tfr_subset(tfr, n)
    aligned = _align_and_trim_masks(
        tfr_sub,
        {
            "Pain contrast": (pain_mask, non_mask),
            "Temperature contrast": (mask_min, mask_max)
        },
        config, logger
    )
    if aligned is None:
        return
    
    pain_mask, non_mask = aligned["Pain contrast"]
    mask_min, mask_max = aligned["Temperature contrast"]

    baseline_used = apply_baseline_and_crop(tfr_sub, baseline=baseline, mode="logratio", logger=logger)
    tfr_pain = tfr_sub[pain_mask].average()
    tfr_non = tfr_sub[non_mask].average()
    tfr_min = tfr_sub[mask_min].average()
    tfr_max = tfr_sub[mask_max].average()

    times = np.asarray(tfr_pain.times)
    tmin_req, tmax_req = plateau_window
    tmin_eff = float(max(times.min(), tmin_req))
    tmax_eff = float(min(times.max(), tmax_req))
    tmin, tmax = tmin_eff, tmax_eff

    fmax_available = float(np.max(tfr_pain.freqs))
    bands = get_bands_for_tfr(max_freq_available=fmax_available, config=config)

    n_rows = 2
    n_cols = len(bands)
    plot_cfg_large2_pain = get_plot_config(config)
    fig_size_per_col_large2_pain = plot_cfg_large2_pain.get_figure_size("tfr_per_col_large", plot_type="tfr")[0]
    fig_size_per_row_large_pain = plot_cfg_large2_pain.get_figure_size("tfr_per_row_large", plot_type="tfr")[1]
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_size_per_col_large2_pain * n_cols, fig_size_per_row_large_pain * n_rows),
        squeeze=False,
        gridspec_kw={"wspace": 1.2, "hspace": 0.25},
    )

    viz_params = get_viz_params(config)
    n_pain = int(pain_mask.sum())
    n_non = int(non_mask.sum())
    n_max = int(mask_max.sum())
    n_min = int(mask_min.sum())

    all_pain_diff = []
    all_temp_diff = []
    band_significance_data = {}
    
    for c, (band, (fmin, fmax)) in enumerate(bands.items()):
        fmax_eff = min(fmax, fmax_available)
        if fmin >= fmax_eff:
            for r in range(n_rows):
                axes[r, c].axis('off')
            continue

        pain_diff_data, temp_diff_data = _compute_band_diff_data(tfr_pain, tfr_non, tfr_max, tfr_min, fmin, fmax_eff, tmin, tmax)
        
        if pain_diff_data is None or temp_diff_data is None:
            for r in range(n_rows):
                axes[r, c].axis('off')
            continue
        
        all_pain_diff.append(pain_diff_data)
        all_temp_diff.append(temp_diff_data)

        pain_sig_mask = None
        pain_cluster_p_min = pain_cluster_k = pain_cluster_mass = None
        if viz_params["diff_annotation_enabled"]:
            pain_diff_data_len = len(pain_diff_data) if pain_diff_data is not None else None
            pain_sig_mask, pain_cluster_p_min, pain_cluster_k, pain_cluster_mass = _compute_cluster_significance(
                tfr_sub, pain_mask, non_mask, fmin, fmax_eff, tmin, tmax, config, diff_data_len=pain_diff_data_len, logger=logger
            )

        temp_sig_mask = None
        temp_cluster_p_min = temp_cluster_k = temp_cluster_mass = None
        if viz_params["diff_annotation_enabled"]:
            temp_diff_data_len = len(temp_diff_data) if temp_diff_data is not None else None
            temp_sig_mask, temp_cluster_p_min, temp_cluster_k, temp_cluster_mass = _compute_cluster_significance(
                tfr_sub, mask_max, mask_min, fmin, fmax_eff, tmin, tmax, config, diff_data_len=temp_diff_data_len, logger=logger
            )
        
        band_significance_data[band] = {
            "pain": (pain_sig_mask, pain_cluster_p_min, pain_cluster_k, pain_cluster_mass),
            "temp": (temp_sig_mask, temp_cluster_p_min, temp_cluster_k, temp_cluster_mass),
            "pain_diff": pain_diff_data,
            "temp_diff": temp_diff_data,
        }

    pain_diff_abs = robust_sym_vlim(all_pain_diff) if len(all_pain_diff) > 0 else 0.0
    temp_diff_abs = robust_sym_vlim(all_temp_diff) if len(all_temp_diff) > 0 else 0.0
    
    for c, (band, (fmin, fmax)) in enumerate(bands.items()):
        fmax_eff = min(fmax, fmax_available)
        if fmin >= fmax_eff:
            continue

        if band not in band_significance_data:
            continue

        band_data = band_significance_data[band]
        pain_diff_data = band_data["pain_diff"]
        temp_diff_data = band_data["temp_diff"]
        pain_sig_mask, pain_cluster_p_min, pain_cluster_k, pain_cluster_mass = band_data["pain"]
        temp_sig_mask, temp_cluster_p_min, temp_cluster_k, temp_cluster_mass = band_data["temp"]

        pain_data_group_a = extract_trial_band_power(tfr_sub[pain_mask], fmin, fmax_eff, tmin, tmax)
        pain_data_group_b = extract_trial_band_power(tfr_sub[non_mask], fmin, fmax_eff, tmin, tmax)
        temp_data_group_a = extract_trial_band_power(tfr_sub[mask_max], fmin, fmax_eff, tmin, tmax)
        temp_data_group_b = extract_trial_band_power(tfr_sub[mask_min], fmin, fmax_eff, tmin, tmax)

        _plot_diff_topomap_with_label(
            axes[0, c], pain_diff_data, tfr_pain.info, pain_diff_abs,
            pain_sig_mask, pain_cluster_p_min, pain_cluster_k, pain_cluster_mass, config, viz_params,
            data_group_a=pain_data_group_a,
            data_group_b=pain_data_group_b,
            paired=False
        )

        _plot_diff_topomap_with_label(
            axes[1, c], temp_diff_data, tfr_max.info, temp_diff_abs,
            temp_sig_mask, temp_cluster_p_min, temp_cluster_k, temp_cluster_mass, config, viz_params,
            data_group_a=temp_data_group_a,
            data_group_b=temp_data_group_b,
            paired=False
        )

        axes[0, c].set_title(f"{band} ({fmin:.0f}-{fmax_eff:.0f} Hz)", fontsize=9, pad=4, y=1.04)

    _add_diff_colorbar(fig, axes[0, :].ravel().tolist(), pain_diff_abs, viz_params["topo_cmap"], config)
    _add_diff_colorbar(fig, axes[1, :].ravel().tolist(), temp_diff_abs, viz_params["topo_cmap"], config)

    font_sizes = _get_font_sizes()
    axes[0, 0].set_ylabel(f"Pain - Non (n={n_pain}-{n_non})", fontsize=font_sizes["ylabel"])
    axes[1, 0].set_ylabel(f"Max - Min temp ({t_max:.1f}-{t_min:.1f}°C, n={n_max}-{n_min})", fontsize=font_sizes["ylabel"])
    sig_text = _get_sig_marker_text(config)
    fig.suptitle(f"Topomaps (baseline: logratio; t=[{tmin:.1f}, {tmax:.1f}] s){sig_text}", fontsize=font_sizes["figure_title"])
    fig.supxlabel("Frequency bands", fontsize=font_sizes["ylabel"])
    _save_fig(fig, out_dir, "topomap_grid_bands_pain_temp_contrasts_bl.png", config=config, logger=logger, baseline_used=baseline_used)


###################################################################
# ROI Processing (Subject)
###################################################################

def compute_roi_tfrs(
    epochs: mne.Epochs,
    freqs: np.ndarray,
    n_cycles: np.ndarray,
    config,
    roi_map: Optional[Dict[str, list[str]]] = None,
) -> Dict[str, mne.time_frequency.EpochsTFR]:
    if roi_map is None:
        from eeg_pipeline.utils.tfr_utils import build_rois_from_info as _build_rois
        roi_map = _build_rois(epochs.info, config=config)
    roi_tfrs = {}
    for roi, chs in roi_map.items():
        picks = mne.pick_channels(epochs.ch_names, include=chs, ordered=True)
        if len(picks) == 0:
            continue
        data = epochs.get_data()
        roi_data = np.nanmean(data[:, picks, :], axis=1, keepdims=True)
        info = mne.create_info([roi], sfreq=epochs.info['sfreq'], ch_types='eeg')
        epo_roi = mne.EpochsArray(
            roi_data,
            info,
            events=epochs.events,
            event_id=epochs.event_id,
            tmin=epochs.tmin,
            metadata=epochs.metadata,
            verbose=False,
        )
        power = run_tfr_morlet(
            epo_roi,
            freqs=freqs,
            n_cycles=n_cycles,
            decim=config.get("tfr_topography_pipeline.tfr.decim", 4),
            picks="eeg",
            config=config,
            logger=None,
        )
        roi_tfrs[roi] = power
    return roi_tfrs


def plot_rois_all_trials(
    roi_tfrs: Dict[str, mne.time_frequency.EpochsTFR],
    out_dir: Path,
    config,
    baseline: Tuple[Optional[float], Optional[float]],
    logger: Optional[logging.Logger] = None,
) -> None:
    rois_dir = out_dir / "rois"
    for roi, tfr in roi_tfrs.items():
        tfr_c = tfr.copy()
        baseline_used = apply_baseline_and_crop(tfr_c, baseline=baseline, mode="logratio", logger=logger)
        tfr_avg = tfr_c.average()
        ch = tfr_avg.info['ch_names'][0]
        roi_tag = sanitize_label(roi)
        roi_dir = rois_dir / roi_tag

        fig = unwrap_figure(tfr_avg.plot(picks=ch, show=False))
        font_sizes = _get_font_sizes()
        fig.suptitle(f"ROI: {roi} — all trials (baseline logratio)", fontsize=font_sizes["figure_title"])
        _save_fig(fig, roi_dir, "tfr_all_trials_bl.png", config=config, logger=logger, baseline_used=baseline_used)

        fmax_available = float(np.max(tfr_avg.freqs))
        bands = get_bands_for_tfr(max_freq_available=fmax_available, config=config)
        for band, (fmin, fmax) in bands.items():
            fmax_eff = min(fmax, fmax_available)
            if fmin >= fmax_eff:
                continue
            band_dir = roi_dir / band
            fig_b = tfr_avg.plot(picks=ch, fmin=fmin, fmax=fmax_eff, show=False)
            fig_b = fig_b[0] if isinstance(fig_b, list) else fig_b
            fig_b.suptitle(f"ROI: {roi} — {band} band (baseline logratio)", fontsize=12)
            _save_fig(fig_b, band_dir, f"tfr_{band}_all_trials_bl.png", config=config, logger=logger, baseline_used=baseline_used)


###################################################################
# Group Plots
###################################################################

def group_contrast_maxmin_temperature(
    powers: List["mne.time_frequency.EpochsTFR"],
    events_by_subj: List[Optional[pd.DataFrame]],
    out_dir: Path,
    config,
    baseline: Optional[Tuple[Optional[float], Optional[float]]] = None,
    plateau_window: Tuple[float, float] = (3.0, 10.5),
    logger: Optional[logging.Logger] = None,
) -> None:
    baseline = _get_baseline_window(config, baseline)
    if not powers:
        return
    temps = []
    for ev in events_by_subj:
        if ev is None:
            continue
        tcol = get_temperature_column_from_config(config, ev)
        if tcol is None:
            continue
        vals = pd.to_numeric(ev[tcol], errors="coerce").round(1)
        temps.extend(list(vals.dropna().unique()))
    temps = sorted(set(map(float, temps)))
    if len(temps) < 2:
        logger and logger.info("Group max/min: fewer than 2 temperature levels; skipping")
        return
    t_min, t_max = float(min(temps)), float(max(temps))

    avg_min: List["mne.time_frequency.AverageTFR"] = []
    avg_max: List["mne.time_frequency.AverageTFR"] = []
    for power, ev in zip(powers, events_by_subj):
        if ev is None:
            continue
        tcol = get_temperature_column_from_config(config, ev)
        if tcol is None:
            continue
        vals = pd.to_numeric(ev[tcol], errors="coerce").round(1)
        mask_min = np.asarray(vals == round(t_min, 1), dtype=bool)
        mask_max = np.asarray(vals == round(t_max, 1), dtype=bool)
        if mask_min.sum() == 0 or mask_max.sum() == 0:
            continue
        a_min = avg_by_mask_to_avg_tfr(power, mask_min)
        a_max = avg_by_mask_to_avg_tfr(power, mask_max)
        if a_min is not None and a_max is not None:
            avg_min.append(a_min)
            avg_max.append(a_max)

    info_common, data_min, data_max = align_paired_avg_tfrs(avg_min, avg_max, logger=logger)
    if info_common is None or data_min is None or data_max is None:
        logger and logger.info("Group max/min: could not align paired min/max TFRs; skipping")
        return

    mean_min = data_min.mean(axis=0)
    mean_max = data_max.mean(axis=0)
    freqs = np.asarray(avg_min[0].freqs if avg_min else avg_max[0].freqs)
    times = np.asarray(avg_min[0].times if avg_min else avg_max[0].times)
    fmax_available = float(freqs.max())
    bands = get_bands_for_tfr(max_freq_available=fmax_available, config=config)
    tmin_req, tmax_req = plateau_window
    tmin = float(max(times.min(), tmin_req))
    tmax = float(min(times.max(), tmax_req))

    n_rows, n_cols = 3, len(bands)
    row_labels = [f"Max {t_max:.1f}°C (n={data_max.shape[0]})", f"Min {t_min:.1f}°C (n={data_min.shape[0]})", "Max - Min"]
    plot_cfg_large2 = get_plot_config(config)
    fig_size_per_col_large2 = plot_cfg_large2.get_figure_size("tfr_per_col_large", plot_type="tfr")[0]
    fig_size_per_row_large = plot_cfg_large2.get_figure_size("tfr_per_row_large", plot_type="tfr")[1]
    tfr_config_large2 = plot_cfg_large2.plot_type_configs.get("tfr", {})
    topomap_config_large2 = tfr_config_large2.get("topomap", {})
    topo_wspace_large2 = topomap_config_large2.get("wspace", 1.2)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(fig_size_per_col_large2 * n_cols, fig_size_per_row_large * n_rows),
        squeeze=False,
        gridspec_kw={"wspace": topo_wspace_large2, "hspace": 0.35}
    )

    for c, (band, (fmin, fmax)) in enumerate(bands.items()):
        fmax_eff = min(fmax, fmax_available)
        if fmin >= fmax_eff:
            for r in range(n_rows):
                axes[r, c].axis('off')
            continue
        fmask = (freqs >= fmin) & (freqs <= fmax_eff)
        tmask = (times >= tmin) & (times < tmax)
        if fmask.sum() == 0 or tmask.sum() == 0:
            for r in range(n_rows):
                axes[r, c].axis('off')
            continue
        v_max = mean_max[:, fmask, :][:, :, tmask].mean(axis=(1, 2))
        v_min = mean_min[:, fmask, :][:, :, tmask].mean(axis=(1, 2))
        v_diff = v_max - v_min
        vabs_pn = robust_sym_vlim([v_max, v_min])
        vabs_diff = robust_sym_vlim(v_diff)

        _plot_topomap_with_percentage_label(axes[0, c], v_max, info_common, -vabs_pn, +vabs_pn, config)
        _plot_topomap_with_percentage_label(axes[1, c], v_min, info_common, -vabs_pn, +vabs_pn, config)
        sig_mask = cluster_p_min = cluster_k = cluster_mass = None
        fdr_txt = ""
        p_ch_used = None
        is_cluster_used = False
        if get_viz_params(config)["diff_annotation_enabled"]:
            subj_max = data_max[:, :, fmask, :][:, :, :, tmask].mean(axis=(2, 3))
            subj_min = data_min[:, :, fmask, :][:, :, :, tmask].mean(axis=(2, 3))
            sig_mask, cluster_p_min, cluster_k, cluster_mass = _cluster_test_two_sample_arrays(
                subj_max, subj_min, info_common, alpha=config.get("statistics.sig_alpha", 0.05), paired=True, n_permutations=config.get("statistics.cluster_n_perm", 1024), config=config
            )
            if sig_mask is not None and cluster_p_min is not None:
                is_cluster_used = True
            elif sig_mask is None:
                res = ttest_rel(subj_max, subj_min, axis=0, nan_policy="omit")
                p_ch_used = np.asarray(res.pvalue)
                sig_mask = _fdr_bh_mask(p_ch_used, alpha=config.get("statistics.sig_alpha", 0.05))
                rej, q = _fdr_bh_values(p_ch_used, alpha=config.get("statistics.sig_alpha", 0.05))
                k_rej = int(np.nansum(rej)) if rej is not None else 0
                q_min = float(np.nanmin(q)) if q is not None and np.isfinite(q).any() else None
                fdr_txt = format_cluster_ann(q_min, k_rej if k_rej > 0 else None, config=config)

        ax = axes[2, c]
        plot_topomap_on_ax(
            ax,
            v_diff,
            info_common,
            mask=(sig_mask if get_viz_params(config)["diff_annotation_enabled"] else None),
            mask_params=get_viz_params(config)["sig_mask_params"],
            vmin=-vabs_diff,
            vmax=+vabs_diff,
        )
        _add_roi_annotations(
            ax, v_diff, info_common, config=config,
            sig_mask=(sig_mask if get_viz_params(config)["diff_annotation_enabled"] else None),
            p_ch=p_ch_used,
            cluster_p_min=cluster_p_min, cluster_k=cluster_k, cluster_mass=cluster_mass,
            is_cluster=is_cluster_used
        )
        mu_d = float(np.nanmean(v_diff))
        pct_d = logratio_to_pct(mu_d)
        cl_txt = (format_cluster_ann(cluster_p_min, cluster_k, cluster_mass, config=config) or fdr_txt) if get_viz_params(config)["diff_annotation_enabled"] else ""
        label = f"Δ%={pct_d:+.1f}%" + (f" | {cl_txt}" if cl_txt else "")
        ax.text(0.5, 1.02, label, transform=ax.transAxes, ha="center", va="top", fontsize=9)
        axes[0, c].set_title(f"{band} ({fmin:.0f}-{fmax_eff:.0f} Hz)", fontsize=9, pad=4, y=1.04)

    viz_params = get_viz_params(config)
    _add_normalized_colorbar(fig, [axes[0, :].ravel().tolist(), axes[1, :].ravel().tolist()], -vabs_pn, +vabs_pn, viz_params["topo_cmap"], config)
    _add_diff_colorbar(fig, axes[2, :].ravel().tolist(), vabs_diff, viz_params["topo_cmap"], config)

    axes[0, 0].set_ylabel(row_labels[0], fontsize=10)
    axes[1, 0].set_ylabel(row_labels[1], fontsize=10)
    axes[2, 0].set_ylabel(row_labels[2], fontsize=10)
    fig.suptitle(
        f"Group Topomaps: Max vs Min temperature (baseline logratio; t=[{tmin:.1f}, {tmax:.1f}] s)",
        fontsize=12,
    )
    _save_fig(fig, out_dir, "group_topomap_grid_bands_maxmin_temp_diff_baseline_logratio.png", config=config, logger=logger)


def group_rois_all_trials(
    powers: List["mne.time_frequency.EpochsTFR"],
    out_dir: Path,
    config,
    baseline: Tuple[Optional[float], Optional[float]],
    logger: Optional[logging.Logger] = None,
    roi_map: Optional[Dict[str, list[str]]] = None,
) -> None:
    if not powers:
        return

    avg_list = []
    for p in powers:
        t = p.copy()
        baseline_used = apply_baseline_and_crop(t, baseline=baseline, mode="logratio", logger=logger)
        avg_list.append(t.average())

    if not avg_list:
        return

    if roi_map is not None:
        rois = list(roi_map.keys())
    else:
        if config is None:
            raise ValueError("Either roi_map or config is required for group_rois_all_trials")
        roi_defs = get_rois(config)
        rois = list(roi_defs.keys())

    for roi in rois:
        per_subj: List["mne.time_frequency.AverageTFR"] = []
        for a in avg_list:
            chs_all = None
            if roi_map is not None:
                chs_all = roi_map.get(roi)
            if chs_all is not None:
                subj_chs = a.info['ch_names']
                canon_subj = {_canonicalize_ch_name(ch).upper(): ch for ch in subj_chs}
                want = {_canonicalize_ch_name(ch).upper() for ch in chs_all}
                chs = [canon_subj[_canonicalize_ch_name(ch).upper()] for ch in subj_chs if _canonicalize_ch_name(ch).upper() in want]
            else:
                roi_defs = get_rois(config)
                pats = roi_defs.get(roi, [])
                chs = _find_roi_channels(a.info, pats)
            if len(chs) == 0:
                continue
            picks = mne.pick_channels(a.info['ch_names'], include=chs, exclude=[])
            if len(picks) == 0:
                continue
            data = np.nanmean(np.asarray(a.data)[picks, :, :], axis=0, keepdims=True)
            ra = a.copy()
            ra.data = data
            ra.info = mne.create_info([f"ROI:{roi}"], sfreq=a.info['sfreq'], ch_types='eeg')
            per_subj.append(ra)

        if len(per_subj) < 1 and roi_map is not None:
            for a in avg_list:
                roi_defs = get_rois(config)
                pats = roi_defs.get(roi, [])
                chs_rx = _find_roi_channels(a.info, pats)
                if chs_rx:
                    picks = mne.pick_channels(a.info['ch_names'], include=chs_rx, exclude=[])
                    if len(picks) == 0:
                        continue
                    data = np.nanmean(np.asarray(a.data)[picks, :, :], axis=0, keepdims=True)
                    ra = a.copy()
                    ra.data = data
                    ra.info = mne.create_info([f"ROI:{roi}"], sfreq=a.info['sfreq'], ch_types='eeg')
                    per_subj.append(ra)

        if len(per_subj) < 1:
            logger and logger.info(f"Group ROI all-trials: no subjects contributed to ROI '{roi}'")
            continue

        info_c, data_c = align_avg_tfrs(per_subj, logger=logger)
        if info_c is None or data_c is None:
            continue

        mean_roi = data_c.mean(axis=0)
        grp = per_subj[0].copy()
        grp.data = mean_roi
        grp.info = info_c
        grp.nave = int(data_c.shape[0])
        grp.comment = f"Group ROI:{roi}"
        ch = grp.info['ch_names'][0]
        title_fontsize = 12
        fig = unwrap_figure(grp.plot(picks=ch, show=False))
        fig.suptitle(f"Group ROI: {roi} — all trials (baseline logratio, n={data_c.shape[0]})", fontsize=title_fontsize)
        _save_fig(fig, out_dir, f"group_tfr_ROI-{sanitize_label(roi)}_all_trials_baseline_logratio.png", config=config, logger=logger, baseline_used=baseline)


def group_contrast_pain_nonpain_rois(
    powers: List["mne.time_frequency.EpochsTFR"],
    events_by_subj: List[Optional[pd.DataFrame]],
    out_dir: Path,
    config,
    baseline: Tuple[Optional[float], Optional[float]],
    logger: Optional[logging.Logger] = None,
    roi_map: Optional[Dict[str, list[str]]] = None,
) -> None:
    if not powers:
        return

    if roi_map is not None:
        rois = list(roi_map.keys())
    else:
        if config is None:
            raise ValueError("Either roi_map or config is required for group_contrast_pain_nonpain_rois")
        roi_defs = get_rois(config)
        rois = list(roi_defs.keys())
    for roi in rois:
        roi_p_list, roi_n_list = _collect_roi_contrast_data(powers, events_by_subj, roi, roi_map, config, logger)
        
        if len(roi_p_list) < 1 or len(roi_n_list) < 1:
            logger and logger.info(f"Group ROI pain/non: no subjects contributed to ROI '{roi}'")
            continue

        info_p, data_p = align_avg_tfrs(roi_p_list, logger=logger)
        info_n, data_n = align_avg_tfrs(roi_n_list, logger=logger)
        if info_p is None or info_n is None or data_p is None or data_n is None:
            continue

        mean_p = data_p.mean(axis=0)
        mean_n = data_n.mean(axis=0)

        grp_p = _create_group_roi_tfr(roi_p_list[0], mean_p, info_p, int(data_p.shape[0]), f"Group ROI:{roi} Pain")
        grp_n = _create_group_roi_tfr(roi_n_list[0], mean_n, info_n, int(data_n.shape[0]), f"Group ROI:{roi} Non")
        diff = mean_p - mean_n
        grp_d = _create_group_roi_tfr(roi_p_list[0], diff, info_p, int(min(data_p.shape[0], data_n.shape[0])), f"Group ROI:{roi} Diff")
        
        ch = grp_p.info['ch_names'][0]

        _plot_single_tfr_figure(
            grp_p, ch, None, f"Group ROI: {roi} — Pain (baseline logratio, n={data_p.shape[0]})",
            f"group_tfr_ROI-{sanitize_label(roi)}_pain_baseline_logratio.png", out_dir, config, logger, baseline
        )
        _plot_single_tfr_figure(
            grp_n, ch, None, f"Group ROI: {roi} — Non-pain (baseline logratio, n={data_n.shape[0]})",
            f"group_tfr_ROI-{sanitize_label(roi)}_nonpain_baseline_logratio.png", out_dir, config, logger, baseline
        )
        n_diff = min(data_p.shape[0], data_n.shape[0])
        _plot_single_tfr_figure(
            grp_d, ch, None, f"Group ROI: {roi} — Pain minus Non (baseline logratio, n={n_diff})",
            f"group_tfr_ROI-{sanitize_label(roi)}_pain_minus_non_baseline_logratio.png", out_dir, config, logger, baseline
        )


def group_contrast_pain_nonpain_scalpmean(
    powers: List["mne.time_frequency.EpochsTFR"],
    events_by_subj: List[Optional[pd.DataFrame]],
    out_dir: Path,
    config,
    baseline: Optional[Tuple[Optional[float], Optional[float]]] = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    baseline = _get_baseline_window(config, baseline)
    if not powers:
        return

    avg_pain, avg_non = _collect_pain_nonpain_avg_tfrs(powers, events_by_subj, config, baseline, logger)
    if len(avg_pain) < 1 or len(avg_non) < 1:
        return

    info_p, data_p, data_n = align_paired_avg_tfrs(avg_pain, avg_non, logger=logger)
    if info_p is None or data_p is None or data_n is None:
        return

    n_subj = int(min(data_p.shape[0], data_n.shape[0]))
    mean_p = data_p.mean(axis=0)
    mean_n = data_n.mean(axis=0)

    data_p_sm = np.asarray(mean_p).mean(axis=0, keepdims=True)
    data_n_sm = np.asarray(mean_n).mean(axis=0, keepdims=True)
    diff_sm = data_p_sm - data_n_sm

    tmpl = avg_pain[0].copy()
    sfreq = tmpl.info['sfreq']

    grp_p = _create_group_scalpmean_tfr(tmpl, data_p_sm, sfreq, n_subj, "Group AllEEG Pain")
    grp_n = _create_group_scalpmean_tfr(tmpl, data_n_sm, sfreq, n_subj, "Group AllEEG Non")
    grp_d = _create_group_scalpmean_tfr(tmpl, diff_sm, sfreq, n_subj, "Group AllEEG Diff")

    _plot_scalpmean_tfr(grp_p, f"Group TFR: All EEG — Pain (baseline logratio, n={n_subj})", "group_tfr_AllEEG_pain_baseline_logratio.png", None, out_dir, config, logger, baseline, None, None)
    _plot_scalpmean_tfr(grp_n, f"Group TFR: All EEG — Non-pain (baseline logratio, n={n_subj})", "group_tfr_AllEEG_nonpain_baseline_logratio.png", None, out_dir, config, logger, baseline, None, None)
    _plot_scalpmean_tfr(grp_d, "Group TFR: All EEG — Pain minus Non (baseline logratio)", "group_tfr_AllEEG_pain_minus_non_baseline_logratio.png", None, out_dir, config, logger, baseline, None, None)


def plot_topomap_grid_baseline_temps(
    tfr: "mne.time_frequency.EpochsTFR | mne.time_frequency.AverageTFR",
    events_df: Optional[pd.DataFrame],
    out_dir: Path,
    config,
    baseline: Optional[Tuple[Optional[float], Optional[float]]] = None,
    plateau_window: Tuple[float, float] = (3.0, 10.5),
    logger: Optional[logging.Logger] = None,
) -> None:
    baseline = _get_baseline_window(config, baseline)
    if events_df is None:
        _log("Temperature grid: events_df is None; skipping.")
        return
    temp_col = get_temperature_column_from_config(config, events_df)
    if temp_col is None:
        _log("Temperature grid: no temperature column found; skipping.")
        return

    tfr_corr = tfr.copy()
    baseline_used = apply_baseline_and_crop(tfr_corr, baseline=baseline, mode="percent", logger=logger)
    tfr_avg_all_corr = tfr_corr.average() if isinstance(tfr_corr, mne.time_frequency.EpochsTFR) else tfr_corr

    temps = (
        pd.to_numeric(events_df[temp_col], errors="coerce")
        .round(1)
        .dropna()
        .unique()
    )
    temps = sorted(map(float, temps))
    if len(temps) == 0:
        _log("Temperature grid: no temperature levels; skipping.")
        return

    times_corr = np.asarray(tfr_avg_all_corr.times)
    tmin_req, tmax_req = plateau_window
    tmin = float(max(times_corr.min(), tmin_req))
    tmax = float(min(times_corr.max(), tmax_req))

    fmax_available = float(np.max(tfr_avg_all_corr.freqs))
    bands = get_bands_for_tfr(max_freq_available=fmax_available, config=config)

    cond_tfrs: list[tuple[str, "mne.time_frequency.AverageTFR", int, float]] = []
    n_all = len(tfr_corr) if isinstance(tfr_corr, mne.time_frequency.EpochsTFR) else 1
    cond_tfrs.append(("All trials", tfr_avg_all_corr, n_all, np.nan))

    if isinstance(tfr_corr, mne.time_frequency.EpochsTFR):
        for tval in temps:
            temp_values = pd.to_numeric(events_df[temp_col], errors="coerce")
            mask = np.abs(temp_values - float(tval)) < 0.05
            mask = np.asarray(mask, dtype=bool)
            if mask.sum() == 0:
                continue
            tfr_temp = tfr_corr.copy()[mask].average()
            cond_tfrs.append((f"{tval:.1f}°C", tfr_temp, int(mask.sum()), float(tval)))
    else:
        _log("Temperature grid: input is AverageTFR; cannot split by temperature; showing only All trials.")

    plot_cfg = get_plot_config(config)
    tfr_config = plot_cfg.plot_type_configs.get("tfr", {})
    topomap_config = tfr_config.get("topomap", {})
    
    per_col = topomap_config.get("fig_size_per_col", 7.0)
    per_row = topomap_config.get("fig_size_per_row", 7.0)
    wspace = topomap_config.get("wspace", 1.2)
    hspace = topomap_config.get("hspace", 0.25)

    n_cols, n_rows = len(cond_tfrs), len(bands)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(per_col * n_cols, per_row * n_rows), squeeze=False,
        gridspec_kw={"wspace": wspace, "hspace": hspace},
    )

    for r, (band, (fmin, fmax)) in enumerate(bands.items()):
        fmax_eff = min(fmax, fmax_available)
        if fmin >= fmax_eff:
            for c in range(n_cols):
                axes[r, c].axis("off")
            continue

        diff_datas: list[Optional[np.ndarray]] = []
        for _, tfr_cond, _, _ in cond_tfrs:
            d = _average_tfr_band(tfr_cond, fmin=fmin, fmax=fmax_eff, tmin=tmin, tmax=tmax)
            diff_datas.append(d)

        vals = [v for v in diff_datas if v is not None and np.isfinite(v).any()]
        if len(vals) == 0:
            for c in range(n_cols):
                axes[r, c].axis("off")
            continue

        diff_abs = robust_sym_vlim(vals, cap=100.0)
        if not np.isfinite(diff_abs) or diff_abs == 0:
            diff_abs = 1e-6

        for idx, (label, tfr_cond, n_cond, _tval) in enumerate(cond_tfrs, start=0):
            ax = axes[r, idx]
            data = diff_datas[idx]
            if data is None:
                ax.axis("off")
                continue

            plot_topomap_on_ax(ax, data, tfr_cond.info, vmin=-diff_abs, vmax=+diff_abs)
            _add_roi_annotations(ax, data, tfr_cond.info, config=config, data_format="percent")
            eeg_picks = extract_eeg_picks(tfr_cond, exclude_bads=False)
            mu = float(np.nanmean(data[eeg_picks])) if len(eeg_picks) > 0 else float(np.nanmean(data))
            title_y = topomap_config.get("title_y", 1.04)
            title_pad = topomap_config.get("title_pad", 4)
            ax.text(0.5, 1.02, f"%Δ={mu:+.1f}%", transform=ax.transAxes, ha="center", va="top", fontsize=plot_cfg.font.title)
            if r == 0:
                ax.set_title(f"{label} (n={n_cond})", fontsize=plot_cfg.font.title, pad=title_pad, y=title_y)

        axes[r, 0].set_ylabel(f"{band} ({fmin:.0f}-{fmax_eff:.0f} Hz)", fontsize=plot_cfg.font.ylabel)

        colorbar_config = tfr_config.get("colorbar", {})
        cbar_fraction = colorbar_config.get("fraction", 0.045)
        cbar_pad = colorbar_config.get("pad", 0.06)
        cbar_shrink = colorbar_config.get("shrink", 0.9)

        sm_diff = ScalarMappable(norm=mcolors.TwoSlopeNorm(vmin=-diff_abs, vcenter=0.0, vmax=diff_abs), cmap=get_viz_params(config)["topo_cmap"])
        sm_diff.set_array([])
        cbar_d = fig.colorbar(sm_diff, ax=axes[r, :].ravel().tolist(), fraction=cbar_fraction, pad=cbar_pad, shrink=cbar_shrink)
        cbar_d.set_label("Percent change from baseline (%)", fontsize=plot_cfg.font.title)

    fig.suptitle(
        f"Topomaps by temperature: % change from baseline over plateau t=[{tmin:.1f}, {tmax:.1f}] s",
        fontsize=plot_cfg.font.figure_title,
    )
    _save_fig(fig, out_dir, "topomap_grid_bands_alltrials_plus_temperatures_baseline_percent.png", config=config, logger=logger, baseline_used=baseline_used)


def plot_pain_nonpain_temporal_topomaps_diff_allbands(
    tfr: "mne.time_frequency.EpochsTFR",
    events_df: Optional[pd.DataFrame],
    out_dir: Path,
    config,
    baseline: Optional[Tuple[Optional[float], Optional[float]]] = None,
    plateau_window: Tuple[float, float] = (3.0, 10.5),
    window_size_ms: float = 100.0,
    logger: Optional[logging.Logger] = None,
) -> None:
    if not require_epochs_tfr(tfr, "Temporal topomaps", logger):
        return

    baseline = _get_baseline_window(config, baseline)

    pain_col = get_pain_column_from_config(config, events_df)
    temp_col = get_temperature_column_from_config(config, events_df)
    
    if pain_col is None:
        _log(f"Events with pain binary column required for temporal topomaps; skipping.", logger, "warning")
        return

    n = compute_aligned_data_length(tfr, events_df)

    pain_vec = extract_pain_vector_array(tfr, events_df, pain_col, n)
    if pain_vec is None:
        _log(f"Events with pain binary column required for temporal topomaps; skipping.", logger, "warning")
        return
    
    pain_mask, non_mask = _create_pain_masks_from_vector(pain_vec)
    if pain_mask is None:
        _log(f"Could not create pain masks; skipping.", logger, "warning")
        return

    if pain_mask.sum() == 0 or non_mask.sum() == 0:
        _log("One of the groups has zero trials; skipping temporal topomaps.", logger, "warning")
        return

    _log(f"Temporal topomaps (diff, all bands): pain={int(pain_mask.sum())}, non-pain={int(non_mask.sum())} trials.", logger)

    tfr_sub = create_tfr_subset(tfr, n)
    aligned = _align_and_trim_masks(
        tfr_sub,
        {"Pain contrast": (pain_mask, non_mask)},
        config, logger
    )
    if aligned is None:
        return
    
    pain_mask, non_mask = aligned["Pain contrast"]

    baseline_used = apply_baseline_and_crop(tfr_sub, baseline=baseline, mode="logratio", logger=logger)

    tfr_pain = tfr_sub[pain_mask].average()
    tfr_non = tfr_sub[non_mask].average()

    has_temp = False
    tfr_max = None
    tfr_min = None
    mask_max = None
    mask_min = None
    t_min = None
    t_max = None
    
    temp_series = extract_temperature_series(tfr, events_df, temp_col, n)
    if temp_series is not None:
        temp_result = create_temperature_masks(temp_series)
        if temp_result[0] is not None:
            t_min, t_max, mask_min, mask_max = temp_result
            
            if mask_min.sum() > 0 and mask_max.sum() > 0:
                try:
                    ensure_aligned_lengths(
                        tfr_sub, mask_min, mask_max,
                        context=f"Temperature contrast",
                        strict=_get_strict_mode(config),
                        logger=logger
                    )
                    if len(mask_min) != len(tfr_sub) or len(mask_max) != len(tfr_sub):
                        mask_min = mask_min[:len(tfr_sub)]
                        mask_max = mask_max[:len(tfr_sub)]
                    tfr_min = tfr_sub[mask_min].average()
                    tfr_max = tfr_sub[mask_max].average()
                    has_temp = True
                    _log(f"Temporal topomaps: max temp={int(mask_max.sum())}, min temp={int(mask_min.sum())} trials.", logger)
                except ValueError as e:
                    _log(f"{e}. Skipping temperature contrast.", logger, "warning")

    times = np.asarray(tfr_pain.times)
    tmin_req, tmax_req = plateau_window
    clipped = clip_time_range(times, tmin_req, tmax_req)
    if clipped is None:
        _log(f"No valid time interval within data range; skipping temporal topomaps (available [{times.min():.2f}, {times.max():.2f}] s).", logger, "warning")
        return
    tmin_clip, tmax_clip = clipped
    
    if tmin_clip is None or tmax_clip is None:
        _log(f"No valid time interval within data range; skipping temporal topomaps (available [{times.min():.2f}, {times.max():.2f}] s).", logger, "warning")
        return

    window_starts, window_ends = create_time_windows_fixed_size(tmin_clip, tmax_clip, window_size_ms)
    n_windows = len(window_starts)
    _log(f"Creating temporal topomaps from {tmin_clip:.2f} to {tmax_clip:.2f} s using {n_windows} windows ({window_size_ms:.1f}ms each).", logger)

    if n_windows == 0:
        _log("No valid windows created; skipping.", logger, "warning")
        return

    window_label = f"{tmin_clip:.1f}–{tmax_clip:.1f}s; {n_windows} windows @ {window_size_ms:.1f}ms"
    filename_base = "temporal_topomaps_pain_minus_nonpain_{band_name}_{tmin:.0f}-{tmax:.0f}s_{n_windows}windows_{baseline_str}.png"
    
    _plot_temporal_topomaps_for_bands(
        tfr_pain, tfr_non, tfr_sub, tfr_max, tfr_min,
        pain_mask, non_mask, mask_max, mask_min,
        window_starts, window_ends, has_temp, t_min, t_max,
        tmin_clip, tmax_clip, n_windows, baseline_used,
        window_label, filename_base, out_dir, config, logger
    )


def plot_temporal_topomaps_allbands_plateau(
    tfr: "mne.time_frequency.EpochsTFR",
    events_df: Optional[pd.DataFrame],
    out_dir: Path,
    config,
    baseline: Optional[Tuple[Optional[float], Optional[float]]] = None,
    plateau_window: Tuple[float, float] = (3.0, 10.5),
    window_count: int = 5,
    logger: Optional[logging.Logger] = None,
) -> None:
    if not require_epochs_tfr(tfr, "Temporal topomaps", logger):
        return

    baseline = _get_baseline_window(config, baseline)

    pain_col = get_pain_column_from_config(config, events_df)
    temp_col = get_temperature_column_from_config(config, events_df)
    
    if pain_col is None:
        _log(f"Events with pain binary column required for temporal topomaps; skipping.", logger, "warning")
        return

    n = compute_aligned_data_length(tfr, events_df)

    pain_vec = extract_pain_vector_array(tfr, events_df, pain_col, n)
    if pain_vec is None:
        _log(f"Events with pain binary column required for temporal topomaps; skipping.", logger, "warning")
        return
    
    pain_mask, non_mask = _create_pain_masks_from_vector(pain_vec)
    if pain_mask is None:
        _log(f"Could not create pain masks; skipping.", logger, "warning")
        return

    if pain_mask.sum() == 0 or non_mask.sum() == 0:
        _log("One of the groups has zero trials; skipping temporal topomaps.", logger, "warning")
        return

    _log(f"Temporal topomaps (plateau, all bands): pain={int(pain_mask.sum())}, non-pain={int(non_mask.sum())} trials.", logger)

    tfr_sub = create_tfr_subset(tfr, n)
    aligned = _align_and_trim_masks(
        tfr_sub,
        {"Pain contrast": (pain_mask, non_mask)},
        config, logger
    )
    if aligned is None:
        return
    
    pain_mask, non_mask = aligned["Pain contrast"]

    baseline_used = apply_baseline_and_crop(tfr_sub, baseline=baseline, mode="logratio", logger=logger)

    tfr_pain = tfr_sub[pain_mask].average()
    tfr_non = tfr_sub[non_mask].average()

    has_temp = False
    tfr_max = None
    tfr_min = None
    mask_max = None
    mask_min = None
    t_min = None
    t_max = None
    
    temp_series = extract_temperature_series(tfr, events_df, temp_col, n)
    if temp_series is not None:
        temp_result = create_temperature_masks(temp_series)
        if temp_result[0] is not None:
            t_min, t_max, mask_min, mask_max = temp_result
            
            if mask_min.sum() > 0 and mask_max.sum() > 0:
                try:
                    ensure_aligned_lengths(
                        tfr_sub, mask_min, mask_max,
                        context=f"Temperature contrast",
                        strict=_get_strict_mode(config),
                        logger=logger
                    )
                    if len(mask_min) != len(tfr_sub) or len(mask_max) != len(tfr_sub):
                        mask_min = mask_min[:len(tfr_sub)]
                        mask_max = mask_max[:len(tfr_sub)]
                    tfr_min = tfr_sub[mask_min].average()
                    tfr_max = tfr_sub[mask_max].average()
                    has_temp = True
                    _log(f"Temporal topomaps: max temp={int(mask_max.sum())}, min temp={int(mask_min.sum())} trials.", logger)
                except ValueError as e:
                    _log(f"{e}. Skipping temperature contrast.", logger, "warning")

    times = np.asarray(tfr_pain.times)
    tmin_req, tmax_req = plateau_window
    clipped = clip_time_range(times, tmin_req, tmax_req)
    if clipped is None:
        _log(f"No valid time interval within data range; skipping temporal topomaps (available [{times.min():.2f}, {times.max():.2f}] s).", logger, "warning")
        return
    tmin_clip, tmax_clip = clipped
    
    if tmin_clip is None or tmax_clip is None:
        _log(f"No valid time interval within data range; skipping temporal topomaps (available [{times.min():.2f}, {times.max():.2f}] s).", logger, "warning")
        return

    window_starts, window_ends = create_time_windows_fixed_count(tmin_clip, tmax_clip, window_count)
    n_windows = len(window_starts)
    window_size_eff = float((tmax_clip - tmin_clip) / n_windows) if n_windows > 0 else 0.0
    _log(f"Creating temporal topomaps over plateau [{tmin_clip:.2f}, {tmax_clip:.2f}] s using {n_windows} windows (~{window_size_eff:.2f}s each).", logger)

    if n_windows == 0:
        _log("No valid windows created; skipping.", logger, "warning")
        return

    window_label = f"plateau {tmin_clip:.0f}–{tmax_clip:.0f}s; {n_windows} windows @ {window_size_eff:.2f}s"
    filename_base = "temporal_topomaps_plateau_{band_name}_{tmin:.0f}-{tmax:.0f}s_{n_windows}windows_{baseline_str}.png"
    
    _plot_temporal_topomaps_for_bands(
        tfr_pain, tfr_non, tfr_sub, tfr_max, tfr_min,
        pain_mask, non_mask, mask_max, mask_min,
        window_starts, window_ends, has_temp, t_min, t_max,
        tmin_clip, tmax_clip, n_windows, baseline_used,
        window_label, filename_base, out_dir, config, logger
    )


def contrast_pain_nonpain_rois(
    roi_tfrs: Dict[str, mne.time_frequency.EpochsTFR],
    events_df: Optional[pd.DataFrame],
    out_dir: Path,
    config,
    baseline: Tuple[Optional[float], Optional[float]],
    logger: Optional[logging.Logger] = None,
) -> None:
    pain_col = get_pain_column_from_config(config, events_df)
    if pain_col is None:
        _log(f"Events with pain binary column required for ROI contrasts; skipping.", logger=logger)
        return

    rois_dir = out_dir / "rois"
    for roi, tfr in roi_tfrs.items():
        try:
            n_epochs = tfr.data.shape[0]
            n_meta = len(events_df) if events_df is not None else n_epochs
            n = compute_aligned_data_length(tfr, events_df)
            if n_epochs != n_meta:
                _log(f"ROI {roi}: trimming to {n} epochs to match events.")

            pain_vec = extract_pain_vector_array(tfr, events_df, pain_col, n)
            if pain_vec is None:
                continue
            
            pain_mask, non_mask = _create_pain_masks_from_vector(pain_vec)
            if pain_mask is None:
                continue
            
            if pain_mask.sum() == 0 or non_mask.sum() == 0:
                _log(f"ROI {roi}: one group has zero trials; skipping.")
                continue

            tfr_sub = create_tfr_subset(tfr, n)
            aligned = _align_and_trim_masks(
                tfr_sub,
                {f"ROI {roi}": (pain_mask, non_mask)},
                config, logger
            )
            if aligned is None:
                continue
            
            pain_mask, non_mask = aligned[f"ROI {roi}"]
            baseline_used = apply_baseline_and_crop(tfr_sub, baseline=baseline, mode="logratio", logger=logger)
            tfr_pain = tfr_sub[pain_mask].average()
            tfr_non = tfr_sub[non_mask].average()

            ch = tfr_pain.info['ch_names'][0]
            roi_tag = sanitize_label(roi)
            roi_dir = rois_dir / roi_tag

            fig = unwrap_figure(tfr_pain.plot(picks=ch, show=False))
            font_sizes = _get_font_sizes()
            fig.suptitle(f"ROI: {roi} — Painful (baseline logratio)", fontsize=font_sizes["figure_title"])
            _save_fig(fig, roi_dir, "tfr_painful_bl.png", config=config, logger=logger, baseline_used=baseline_used)

            fig = unwrap_figure(tfr_non.plot(picks=ch, show=False))
            font_sizes = _get_font_sizes()
            fig.suptitle(f"ROI: {roi} — Non-pain (baseline logratio)", fontsize=font_sizes["figure_title"])
            _save_fig(fig, roi_dir, "tfr_nonpain_bl.png", config=config, logger=logger, baseline_used=baseline_used)

            tfr_diff = tfr_pain.copy()
            tfr_diff.data = tfr_pain.data - tfr_non.data
            fig = unwrap_figure(tfr_diff.plot(picks=ch, show=False))
            font_sizes = _get_font_sizes()
            fig.suptitle(f"ROI: {roi} — Pain minus Non-pain (baseline logratio)", fontsize=font_sizes["figure_title"])
            _save_fig(fig, roi_dir, "tfr_pain_minus_nonpain_bl.png", config=config, logger=logger, baseline_used=baseline_used)

            fmax_available = float(np.max(tfr_pain.freqs))
            bands = get_bands_for_tfr(max_freq_available=fmax_available, config=config)
            for band, (fmin, fmax) in bands.items():
                fmax_eff = min(fmax, fmax_available)
                if fmin >= fmax_eff:
                    continue
                band_dir = roi_dir / band
                fig_b = unwrap_figure(tfr_pain.plot(picks=ch, fmin=fmin, fmax=fmax_eff, show=False))
                fig_b.suptitle(f"ROI: {roi} — {band} Painful (baseline logratio)", fontsize=12)
                _save_fig(fig_b, band_dir, f"tfr_{band}_painful_bl.png", config=config, logger=logger, baseline_used=baseline_used)

                fig_b = unwrap_figure(tfr_non.plot(picks=ch, fmin=fmin, fmax=fmax_eff, show=False))
                fig_b.suptitle(f"ROI: {roi} — {band} Non-pain (baseline logratio)", fontsize=12)
                _save_fig(fig_b, band_dir, f"tfr_{band}_nonpain_bl.png", config=config, logger=logger, baseline_used=baseline_used)

                fig_b = unwrap_figure(tfr_diff.plot(picks=ch, fmin=fmin, fmax=fmax_eff, show=False))
                fig_b.suptitle(f"ROI: {roi} — {band} Pain minus Non-pain (baseline logratio)", fontsize=12)
                _save_fig(fig_b, band_dir, f"tfr_{band}_pain_minus_nonpain_bl.png", config=config, logger=logger, baseline_used=baseline_used)
        except (FileNotFoundError, ValueError, RuntimeError, KeyError, IndexError) as exc:
            _log(f"ROI {roi}: error while computing ROI contrasts ({exc})", logger, "error")
            continue


###################################################################
# Group TF correlation (moved for completeness)
###################################################################

def _discover_subjects_with_tf(roi_suffix, method_suffix, config, allowed_subjects=None):
    subs = []
    for sd in sorted(config.deriv_root.glob("sub-*")):
        if not sd.is_dir():
            continue
        sub = sd.name[4:]
        if allowed_subjects is not None and sub not in allowed_subjects:
            continue
        cand = sd / "eeg" / "stats" / f"tf_corr_stats{roi_suffix}{method_suffix}.tsv"
        if cand.exists():
            subs.append(sub)
    return subs


def _load_subject_tf(sub, roi_suffix, method_suffix, config):
    p = config.deriv_root / f"sub-{sub}" / "eeg" / "stats" / f"tf_corr_stats{roi_suffix}{method_suffix}.tsv"
    return pd.read_csv(p, sep="\t") if p.exists() else None




def _annotate_tf_correlation_figure(fig, config, alpha):
    try:
        default_baseline = [-0.5, -0.01]
        bwin = config.get("time_frequency_analysis.baseline_window", default_baseline) if config else default_baseline
        corr_txt = f"FDR BH α={alpha}"
        text = (
            f"Group TF correlation | Baseline: [{float(bwin[0]):.2f}, {float(bwin[1]):.2f}] s | "
            f"{corr_txt}"
        )
        plot_cfg_annotate = get_plot_config(config)
        font_size_label = plot_cfg_annotate.font.label
        fig.text(0.01, 0.01, text, fontsize=font_size_label, alpha=0.8)
    except Exception:
        pass


def _select_tf_correlation_method(method, roi_suffix, min_subjects, config, allowed_subjects, subjects_param, logger):
    if method == "auto":
        for method_name in ("_spearman", "_pearson"):
            subjects_found = _discover_subjects_with_tf(roi_suffix, method_name, config, allowed_subjects)
            if len(subjects_found) >= min_subjects:
                return method_name, subjects_found
        _log(f"Group TF correlation skipped for ROI '{roi_suffix or 'all'}' — insufficient subject heatmaps.", logger, "warning")
        return None, None
    else:
        method_suffix = f"_{method.lower()}"
        subjects_found = subjects_param or _discover_subjects_with_tf(roi_suffix, method_suffix, config, allowed_subjects)
        return method_suffix, subjects_found


def group_tf_correlation(subjects=None, roi=None, method="auto", alpha=None, min_subjects=None, config=None, logger=None):
    if alpha is None:
        plot_cfg_alpha = get_plot_config(config) if config else None
        if plot_cfg_alpha:
            tfr_config_alpha = plot_cfg_alpha.plot_type_configs.get("tfr", {})
            default_significance_alpha = tfr_config_alpha.get("default_significance_alpha", 0.05)
        else:
            default_significance_alpha = 0.05
        alpha = config.get("statistics.sig_alpha", default_significance_alpha) if config else default_significance_alpha
    if min_subjects is None:
        min_subjects = int(config.get("analysis.min_subjects_for_topomaps", 3)) if config else 3
    
    roi_raw = roi.lower() if isinstance(roi, str) else None
    roi_suffix = f"_{re.sub(r'[^A-Za-z0-9._-]+', '_', roi_raw)}" if roi_raw else ""
    allowed_subjects = set(subjects) if subjects else None

    method_suffix, subjects_to_use = _select_tf_correlation_method(method, roi_suffix, min_subjects, config, allowed_subjects, subjects, logger)
    if method_suffix is None or not subjects_to_use:
        if not subjects_to_use:
            _log(f"Group TF correlation skipped for ROI '{roi or 'all'}' — no subject files for method '{method}'.", logger, "warning")
        return None

    dfs = []
    used_subjects = []
    for sub in subjects_to_use:
        df = _load_subject_tf(sub, roi_suffix, method_suffix, config)
        if df is None or df.empty or df.dropna(subset=["correlation", "frequency", "time"]).empty:
            continue
        dfs.append(df.dropna(subset=["correlation", "frequency", "time"]))
        used_subjects.append(sub)

    if len(dfs) < min_subjects:
        _log(f"Group TF correlation skipped for ROI '{roi or 'all'}' — fewer than {min_subjects} subjects with valid data.", logger, "warning")
        return None

    f_common, t_common = extract_time_frequency_grid(dfs[0])
    for df in dfs[1:]:
        f, t = extract_time_frequency_grid(df)
        f_common = np.intersect1d(f_common, f)
        t_common = np.intersect1d(t_common, t)

    if f_common.size == 0 or t_common.size == 0:
        _log(f"Group TF correlation skipped for ROI '{roi or 'all'}' — unable to find common TF grid.", logger, "warning")
        return None

    mats: list[np.ndarray] = []
    for df in dfs:
        df_use = df.copy()
        df_use["frequency"] = np.round(df_use["frequency"].astype(float), 6)
        df_use["time"] = np.round(df_use["time"].astype(float), 6)
        pivot = df_use.pivot_table(index="frequency", columns="time", values="correlation", aggfunc="mean")
        pivot = pivot.reindex(index=f_common, columns=t_common)
        mats.append(pivot.to_numpy())

    Z = np.stack([np.arctanh(np.clip(m, -0.999999, 0.999999)) for m in mats], axis=0)
    z_mean = np.nanmean(Z, axis=0)
    z_sd = np.nanstd(Z, axis=0, ddof=1)
    n = np.sum(np.isfinite(Z), axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        denom = z_sd / np.sqrt(np.maximum(n, 1))
        denom[denom == 0] = np.nan
        t_stat = z_mean / denom

    p_vals = np.full_like(t_stat, np.nan, dtype=float)
    finite = np.isfinite(t_stat) & (n > 1)
    if np.any(finite):
        df = np.maximum(n[finite] - 1, 1)
        t_abs = np.abs(t_stat[finite])
        p_vals[finite] = 2.0 * t_dist.sf(t_abs, df=df)

    rej, q_flat = _fdr_bh_values(p_vals[np.isfinite(p_vals)], alpha=alpha)
    q_vals = np.full_like(p_vals, np.nan)
    if q_flat is not None:
        q_vals[np.isfinite(p_vals)] = q_flat

    sig_mask = np.zeros_like(p_vals, dtype=bool)
    if rej is not None:
        sig_mask[np.isfinite(p_vals)] = rej.astype(bool)
    sig_mask &= (n >= min_subjects)
    r_mean = np.tanh(z_mean)

    stats_dir = deriv_group_stats_path(config.deriv_root)
    plots_dir = deriv_group_plots_path(config.deriv_root, "tf_corr")
    stats_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    out_tsv = stats_dir / f"tf_corr_group{roi_suffix}{method_suffix}.tsv"
    df_out = pd.DataFrame(
        {
            "frequency": np.repeat(f_common, len(t_common)),
            "time": np.tile(t_common, len(f_common)),
            "r_mean": r_mean.flatten(),
            "z_mean": z_mean.flatten(),
            "n": n.flatten(),
            "p": p_vals.flatten(),
            "q": q_vals.flatten(),
            "significant": sig_mask.flatten(),
        }
    )
    df_out.to_csv(out_tsv, sep="\t", index=False)

    extent = [t_common[0], t_common[-1], f_common[0], f_common[-1]]
    cmap = "RdBu_r"
    vmin = -0.6
    vmax = 0.6
    figure_paths = []

    plot_cfg_small = get_plot_config(config)
    fig_size_small = plot_cfg_small.get_figure_size("small", plot_type="tfr")
    fig1, ax1 = plt.subplots(figsize=fig_size_small)
    im1 = ax1.imshow(
        r_mean,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax1.axvline(0.0, color="k", linestyle="--", alpha=0.6)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Frequency (Hz)")
    title_roi = roi or "All channels"
    title_method = method_suffix.strip("_").title()
    ax1.set_title(f"Group TF correlation — mean r ({title_method}, {title_roi})")
    cb1 = plt.colorbar(im1, ax=ax1)
    cb1.set_label("r")
    plt.tight_layout()
    _annotate_tf_correlation_figure(fig1, config, alpha)
    save_formats = config.get("output.save_formats", ["png"]) if config else ["png"]
    _save_fig(
        fig1,
        plots_dir,
        f"tf_corr_group_rmean{roi_suffix}{method_suffix}",
        config,
        
        logger=logger,
    )
    for ext in save_formats:
        figure_paths.append(
            plots_dir / f"tf_corr_group_rmean{roi_suffix}{method_suffix}.{ext}"
        )

    fig2, ax2 = plt.subplots(figsize=fig_size_small)
    im2 = ax2.imshow(
        np.where(sig_mask, r_mean, np.nan),
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax2.axvline(0.0, color="k", linestyle="--", alpha=0.6)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Frequency (Hz)")
    sig_title = f"Group TF correlation — FDR<{alpha:g} ({title_method}, {title_roi})"
    ax2.set_title(sig_title)
    cb2 = plt.colorbar(im2, ax=ax2)
    cb2.set_label("r (significant)")
    plt.tight_layout()
    _annotate_tf_correlation_figure(fig2, config, alpha)
    save_formats = config.get("output.save_formats", ["png"]) if config else ["png"]
    _save_fig(
        fig2,
        plots_dir,
        f"tf_corr_group_sig{roi_suffix}{method_suffix}",
        config,
        
        logger=logger,
    )
    for ext in save_formats:
        figure_paths.append(
            plots_dir / f"tf_corr_group_sig{roi_suffix}{method_suffix}.{ext}"
        )

    _log(
        f"Group TF correlation saved (ROI={roi or 'all'}, method={method_suffix.strip('_')}): {out_tsv}",
        logger,
        "info"
    )
    return out_tsv, figure_paths


###################################################################
# Group TFR Plotting Functions
###################################################################

def _combine_multiple_tfr_groups(tfr_lists_dict, min_count, logger):
    combined = {}
    for key, tfr_list in tfr_lists_dict.items():
        if len(tfr_list) >= min_count:
            combined[key] = _combine_avg_tfrs_group(tfr_list, logger)
        else:
            combined[key] = None
    return combined


def _combine_avg_tfrs_group(
    avg_tfr_list: List["mne.time_frequency.AverageTFR"],
    logger: Optional[logging.Logger] = None,
) -> Optional["mne.time_frequency.AverageTFR"]:
    if not avg_tfr_list:
        return None
    avg_tfr_list = [t for t in avg_tfr_list if t is not None]
    if not avg_tfr_list:
        return None
    
    base = avg_tfr_list[0]
    base_times = np.asarray(base.times)
    base_freqs = np.asarray(base.freqs)
    base_chs = list(base.info["ch_names"])
    
    valid_tfrs = [base]
    for tfr in avg_tfr_list[1:]:
        if np.allclose(tfr.times, base_times) and np.allclose(tfr.freqs, base_freqs):
            valid_tfrs.append(tfr)
        elif logger:
            logger.warning(f"Skipping TFR: times/freqs mismatch for group alignment")
    
    if len(valid_tfrs) == 0:
        return None
    
    ch_sets = [set(t.info["ch_names"]) for t in valid_tfrs]
    common_chs = list(sorted(set.intersection(*ch_sets))) if ch_sets else []
    if len(common_chs) == 0:
        if logger:
            logger.warning("No common channels across subjects; cannot combine TFRs")
        return None
    
    data_list = []
    for tfr in valid_tfrs:
        ch_indices = [tfr.info["ch_names"].index(ch) for ch in common_chs]
        data_list.append(np.asarray(tfr.data)[ch_indices, :, :])
    
    combined_data = np.mean(np.stack(data_list, axis=0), axis=0)
    
    pick_inds = [base_chs.index(ch) for ch in common_chs]
    info_common = mne.pick_info(base.info, pick_inds)
    
    combined_tfr = mne.time_frequency.AverageTFR(
        info=info_common,
        data=combined_data,
        times=base_times,
        freqs=base_freqs,
        nave=len(valid_tfrs),
        method=None,
        comment=base.comment if hasattr(base, 'comment') else 'grand_average'
    )
    return combined_tfr


def _combine_multiple_epochs_tfr_groups(tfr_lists_dict, min_count, logger):
    combined = {}
    for key, tfr_list in tfr_lists_dict.items():
        if len(tfr_list) >= min_count:
            combined[key] = _concatenate_epochs_tfr_group(tfr_list, logger)
        else:
            combined[key] = None
    return combined


def _concatenate_epochs_tfr_group(
    tfr_list: List["mne.time_frequency.EpochsTFR"],
    logger: Optional[logging.Logger] = None,
) -> Optional["mne.time_frequency.EpochsTFR"]:
    if not tfr_list:
        return None
    tfr_list = [t for t in tfr_list if t is not None]
    if not tfr_list:
        return None
    
    base = tfr_list[0]
    base_times = np.asarray(base.times)
    base_freqs = np.asarray(base.freqs)
    base_chs = list(base.info["ch_names"])
    
    valid_tfrs = [base]
    for tfr in tfr_list[1:]:
        if np.allclose(tfr.times, base_times) and np.allclose(tfr.freqs, base_freqs):
            valid_tfrs.append(tfr)
        elif logger:
            logger.warning(f"Skipping TFR: times/freqs mismatch for group concatenation")
    
    if len(valid_tfrs) == 0:
        return None
    
    ch_sets = [set(t.info["ch_names"]) for t in valid_tfrs]
    common_chs = list(sorted(set.intersection(*ch_sets))) if ch_sets else []
    if len(common_chs) == 0:
        if logger:
            logger.warning("No common channels across subjects; cannot concatenate TFRs")
        return None
    
    data_list = []
    events_list = []
    for tfr in valid_tfrs:
        ch_indices = [tfr.info["ch_names"].index(ch) for ch in common_chs]
        data_subj = np.asarray(tfr.data)[:, ch_indices, :, :]
        data_list.append(data_subj)
        if hasattr(tfr, 'events') and tfr.events is not None:
            events_list.append(tfr.events)
    
    combined_data = np.concatenate(data_list, axis=0)
    combined_events = np.concatenate(events_list, axis=0) if events_list and all(e is not None for e in events_list) else base.events
    
    pick_inds = [base_chs.index(ch) for ch in common_chs]
    info_common = mne.pick_info(base.info, pick_inds)
    
    combined_tfr = mne.time_frequency.EpochsTFR(
        info_common,
        combined_data,
        base_times,
        base_freqs,
        events=combined_events,
        event_id=base.event_id if hasattr(base, 'event_id') else None,
        metadata=base.metadata if hasattr(base, 'metadata') and base.metadata is not None else None,
        method=None,
        verbose=False
    )
    return combined_tfr


def group_plot_bands_pain_temp_contrasts(
    powers: List["mne.time_frequency.EpochsTFR"],
    events_list: List[Optional[pd.DataFrame]],
    out_dir: Path,
    config,
    baseline: Optional[Tuple[Optional[float], Optional[float]]] = None,
    plateau_window: Tuple[float, float] = (3.0, 10.5),
    logger: Optional[logging.Logger] = None,
) -> None:
    baseline = _get_baseline_window(config, baseline)
    
    if len(powers) < 2:
        _log("Group contrast requires at least 2 subjects; skipping.", logger, "warning")
        return
    
    pain_col = get_pain_column_from_config(config)
    temp_col = get_temperature_column_from_config(config)
    
    if pain_col is None or temp_col is None:
        _log("Group contrast: missing pain or temperature column; skipping.", logger, "warning")
        return
    
    tfr_pain_avg_list = []
    tfr_non_avg_list = []
    tfr_max_avg_list = []
    tfr_min_avg_list = []
    tfr_pain_epochs_list = []
    tfr_non_epochs_list = []
    tfr_max_epochs_list = []
    tfr_min_epochs_list = []
    
    for tfr, events_df in zip(powers, events_list):
        if events_df is None:
            continue
        
        n = compute_aligned_data_length(tfr, events_df)
        
        pain_vec = extract_pain_vector_array(tfr, events_df, pain_col, n)
        if pain_vec is None:
            continue
        
        pain_mask, non_mask = _create_pain_masks_from_vector(pain_vec)
        if pain_mask is None:
            continue
        
        temp_series = extract_temperature_series(tfr, events_df, temp_col, n)
        if temp_series is None:
            continue
        
        temp_result = create_temperature_masks(temp_series)
        if temp_result[0] is None:
            continue
        
        t_min, t_max, mask_min, mask_max = temp_result
        
        if pain_mask.sum() == 0 or non_mask.sum() == 0 or mask_min.sum() == 0 or mask_max.sum() == 0:
            continue
        
        tfr_sub = create_tfr_subset(tfr, n)
        aligned = _align_and_trim_masks(
            tfr_sub,
            {
                "Pain contrast": (pain_mask, non_mask),
                "Temperature contrast": (mask_min, mask_max)
            },
            config, logger
        )
        if aligned is None:
            continue
        
        pain_mask, non_mask = aligned["Pain contrast"]
        mask_min, mask_max = aligned["Temperature contrast"]
        
        baseline_used = apply_baseline_and_crop(tfr_sub, baseline=baseline, mode="logratio", logger=logger)
        tfr_pain_avg_list.append(tfr_sub[pain_mask].average())
        tfr_non_avg_list.append(tfr_sub[non_mask].average())
        tfr_max_avg_list.append(tfr_sub[mask_max].average())
        tfr_min_avg_list.append(tfr_sub[mask_min].average())
        tfr_pain_epochs_list.append(tfr_sub[pain_mask])
        tfr_non_epochs_list.append(tfr_sub[non_mask])
        tfr_max_epochs_list.append(tfr_sub[mask_max])
        tfr_min_epochs_list.append(tfr_sub[mask_min])
    
    if len(tfr_pain_avg_list) < 2:
        _log("Group contrast: insufficient subjects with valid data; skipping.", logger, "warning")
        return
    
    combined_avg = _combine_multiple_tfr_groups({
        "pain": tfr_pain_avg_list,
        "non": tfr_non_avg_list,
        "max": tfr_max_avg_list,
        "min": tfr_min_avg_list
    }, 2, logger)
    
    tfr_pain = combined_avg["pain"]
    tfr_non = combined_avg["non"]
    tfr_max = combined_avg["max"]
    tfr_min = combined_avg["min"]
    
    if tfr_pain is None or tfr_non is None or tfr_max is None or tfr_min is None:
        _log("Group contrast: failed to combine TFRs; skipping.", logger, "warning")
        return
    
    combined_epochs = _combine_multiple_epochs_tfr_groups({
        "pain": tfr_pain_epochs_list,
        "non": tfr_non_epochs_list,
        "max": tfr_max_epochs_list,
        "min": tfr_min_epochs_list
    }, 2, logger)
    
    tfr_pain_combined = combined_epochs["pain"]
    tfr_non_combined = combined_epochs["non"]
    tfr_max_combined = combined_epochs["max"]
    tfr_min_combined = combined_epochs["min"]
    
    times = np.asarray(tfr_pain.times)
    tmin_req, tmax_req = plateau_window
    tmin_eff = float(max(times.min(), tmin_req))
    tmax_eff = float(min(times.max(), tmax_req))
    tmin, tmax = tmin_eff, tmax_eff
    
    fmax_available = float(np.max(tfr_pain.freqs))
    bands = get_bands_for_tfr(max_freq_available=fmax_available, config=config)
    
    n_rows = 2
    n_cols = len(bands)
    plot_cfg_large2_pain = get_plot_config(config)
    fig_size_per_col_large2_pain = plot_cfg_large2_pain.get_figure_size("tfr_per_col_large", plot_type="tfr")[0]
    fig_size_per_row_large_pain = plot_cfg_large2_pain.get_figure_size("tfr_per_row_large", plot_type="tfr")[1]
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_size_per_col_large2_pain * n_cols, fig_size_per_row_large_pain * n_rows),
        squeeze=False,
        gridspec_kw={"wspace": 1.2, "hspace": 0.25},
    )
    
    viz_params = get_viz_params(config)
    n_pain = len(tfr_pain_avg_list)
    n_non = len(tfr_non_avg_list)
    n_max = len(tfr_max_avg_list)
    n_min = len(tfr_min_avg_list)
    
    all_pain_diff = []
    all_temp_diff = []
    
    for c, (band, (fmin, fmax)) in enumerate(bands.items()):
        fmax_eff = min(fmax, fmax_available)
        if fmin >= fmax_eff:
            for r in range(n_rows):
                axes[r, c].axis('off')
            continue
        
        pain_data = _average_tfr_band(tfr_pain, fmin=fmin, fmax=fmax_eff, tmin=tmin, tmax=tmax)
        non_data = _average_tfr_band(tfr_non, fmin=fmin, fmax=fmax_eff, tmin=tmin, tmax=tmax)
        max_data = _average_tfr_band(tfr_max, fmin=fmin, fmax=fmax_eff, tmin=tmin, tmax=tmax)
        min_data = _average_tfr_band(tfr_min, fmin=fmin, fmax=fmax_eff, tmin=tmin, tmax=tmax)
        
        if pain_data is None or non_data is None or max_data is None or min_data is None:
            for r in range(n_rows):
                axes[r, c].axis('off')
            continue
        
        pain_diff_data = pain_data - non_data
        temp_diff_data = max_data - min_data
        
        all_pain_diff.append(pain_diff_data)
        all_temp_diff.append(temp_diff_data)
    
    pain_diff_abs = robust_sym_vlim(all_pain_diff) if len(all_pain_diff) > 0 else 0.0
    temp_diff_abs = robust_sym_vlim(all_temp_diff) if len(all_temp_diff) > 0 else 0.0
    
    for c, (band, (fmin, fmax)) in enumerate(bands.items()):
        fmax_eff = min(fmax, fmax_available)
        if fmin >= fmax_eff:
            continue
        
        pain_diff_data, temp_diff_data = _compute_band_diff_data(tfr_pain, tfr_non, tfr_max, tfr_min, fmin, fmax_eff, tmin, tmax)
        
        if pain_diff_data is None or temp_diff_data is None:
            continue
        
        pain_sig_mask = None
        pain_cluster_p_min = pain_cluster_k = pain_cluster_mass = None
        if viz_params["diff_annotation_enabled"]:
            pain_sig_mask, pain_cluster_p_min, pain_cluster_k, pain_cluster_mass = _compute_cluster_significance_from_combined(
                tfr_pain_combined, tfr_non_combined, fmin, fmax_eff, tmin, tmax, config, len(pain_diff_data), logger
            )
        
        temp_sig_mask = None
        temp_cluster_p_min = temp_cluster_k = temp_cluster_mass = None
        if viz_params["diff_annotation_enabled"]:
            temp_sig_mask, temp_cluster_p_min, temp_cluster_k, temp_cluster_mass = _compute_cluster_significance_from_combined(
                tfr_max_combined, tfr_min_combined, fmin, fmax_eff, tmin, tmax, config, len(temp_diff_data), logger
            )
        
        ax = axes[0, c]
        plot_topomap_on_ax(
            ax,
            pain_diff_data,
            tfr_pain.info,
            mask=(pain_sig_mask if viz_params["diff_annotation_enabled"] else None),
            mask_params=viz_params["sig_mask_params"],
            vmin=(-pain_diff_abs if pain_diff_abs > 0 else None),
            vmax=(+pain_diff_abs if pain_diff_abs > 0 else None),
            config=config,
        )
        _add_roi_annotations(
            ax, pain_diff_data, tfr_pain.info, config=config,
            sig_mask=(pain_sig_mask if viz_params["diff_annotation_enabled"] else None),
            cluster_p_min=pain_cluster_p_min, cluster_k=pain_cluster_k, cluster_mass=pain_cluster_mass,
            is_cluster=(pain_sig_mask is not None and pain_cluster_p_min is not None)
        )
        pain_diff_mu = float(np.nanmean(pain_diff_data))
        pain_pct_mu = logratio_to_pct(pain_diff_mu)
        pain_cl_txt = format_cluster_ann(pain_cluster_p_min, pain_cluster_k, pain_cluster_mass, config=config) if viz_params["diff_annotation_enabled"] else ""
        pain_label = f"Δ%={pain_pct_mu:+.1f}%" + (f" | {pain_cl_txt}" if pain_cl_txt else "")
        ax.text(0.5, 1.02, pain_label, transform=ax.transAxes, ha="center", va="top", fontsize=9)
        
        ax = axes[1, c]
        plot_topomap_on_ax(
            ax,
            temp_diff_data,
            tfr_max.info,
            mask=(temp_sig_mask if viz_params["diff_annotation_enabled"] else None),
            mask_params=viz_params["sig_mask_params"],
            vmin=(-temp_diff_abs if temp_diff_abs > 0 else None),
            vmax=(+temp_diff_abs if temp_diff_abs > 0 else None),
            config=config,
        )
        _add_roi_annotations(
            ax, temp_diff_data, tfr_max.info, config=config,
            sig_mask=(temp_sig_mask if viz_params["diff_annotation_enabled"] else None),
            cluster_p_min=temp_cluster_p_min, cluster_k=temp_cluster_k, cluster_mass=temp_cluster_mass,
            is_cluster=(temp_sig_mask is not None and temp_cluster_p_min is not None)
        )
        temp_diff_mu = float(np.nanmean(temp_diff_data))
        temp_pct_mu = logratio_to_pct(temp_diff_mu)
        temp_cl_txt = format_cluster_ann(temp_cluster_p_min, temp_cluster_k, temp_cluster_mass, config=config) if viz_params["diff_annotation_enabled"] else ""
        temp_label = f"Δ%={temp_pct_mu:+.1f}%" + (f" | {temp_cl_txt}" if temp_cl_txt else "")
        ax.text(0.5, 1.02, temp_label, transform=ax.transAxes, ha="center", va="top", fontsize=9)
        
        axes[0, c].set_title(f"{band} ({fmin:.0f}-{fmax_eff:.0f} Hz)", fontsize=9, pad=4, y=1.04)
    
    if pain_diff_abs > 0:
        sm_pain = ScalarMappable(norm=mcolors.TwoSlopeNorm(vmin=-pain_diff_abs, vcenter=0.0, vmax=pain_diff_abs), cmap=viz_params["topo_cmap"])
        sm_pain.set_array([])
        fig.colorbar(sm_pain, ax=axes[0, :].ravel().tolist(), fraction=0.045, pad=0.06, shrink=0.9)
    
    if temp_diff_abs > 0:
        sm_temp = ScalarMappable(norm=mcolors.TwoSlopeNorm(vmin=-temp_diff_abs, vcenter=0.0, vmax=temp_diff_abs), cmap=viz_params["topo_cmap"])
        sm_temp.set_array([])
        fig.colorbar(sm_temp, ax=axes[1, :].ravel().tolist(), fraction=0.045, pad=0.06, shrink=0.9)
    
    font_sizes = _get_font_sizes()
    axes[0, 0].set_ylabel(f"Pain - Non (N={n_pain} subjects)", fontsize=font_sizes["ylabel"])
    axes[1, 0].set_ylabel(f"Max - Min temp (N={n_max} subjects)", fontsize=font_sizes["ylabel"])
    sig_text = _get_sig_marker_text(config)
    fig.suptitle(f"Group Topomaps (baseline: logratio; t=[{tmin:.1f}, {tmax:.1f}] s, N={n_pain} subjects){sig_text}", fontsize=font_sizes["figure_title"])
    fig.supxlabel("Frequency bands", fontsize=font_sizes["ylabel"])
    _save_fig(fig, out_dir, "group_topomap_grid_bands_pain_temp_contrasts_bl.png", config=config, logger=logger, baseline_used=baseline)


def group_plot_topomap_grid_baseline_temps(
    powers: List["mne.time_frequency.EpochsTFR"],
    events_list: List[Optional[pd.DataFrame]],
    out_dir: Path,
    config,
    baseline: Optional[Tuple[Optional[float], Optional[float]]] = None,
    plateau_window: Tuple[float, float] = (3.0, 10.5),
    logger: Optional[logging.Logger] = None,
) -> None:
    baseline = _get_baseline_window(config, baseline)
    
    if len(powers) < 2:
        _log("Group temperature grid requires at least 2 subjects; skipping.", logger, "warning")
        return
    
    temp_col = next((c for c in config.get("event_columns.temperature", []) if c), None)
    if temp_col is None:
        _log("Group temperature grid: no temperature column found; skipping.", logger, "warning")
        return
    
    all_temps = set()
    tfr_all_list = []
    tfr_by_temp: Dict[float, List["mne.time_frequency.AverageTFR"]] = {}
    
    for tfr, events_df in zip(powers, events_list):
        if events_df is None:
            continue
        
        tfr_corr = tfr.copy()
        baseline_used = apply_baseline_and_crop(tfr_corr, baseline=baseline, mode="percent", logger=logger)
        tfr_avg_all = tfr_corr.average()
        tfr_all_list.append(tfr_avg_all)
        
        temps = (
            pd.to_numeric(events_df[temp_col], errors="coerce")
            .round(1)
            .dropna()
            .unique()
        )
        temps = sorted(map(float, temps))
        all_temps.update(temps)
        
        for tval in temps:
            temp_values = pd.to_numeric(events_df[temp_col], errors="coerce")
            mask = np.abs(temp_values - float(tval)) < 0.05
            mask = np.asarray(mask, dtype=bool)
            if mask.sum() == 0:
                continue
            tfr_temp = tfr_corr.copy()[mask].average()
            tfr_by_temp.setdefault(float(tval), []).append(tfr_temp)
    
    if len(tfr_all_list) < 2:
        _log("Group temperature grid: insufficient subjects; skipping.", logger, "warning")
        return
    
    tfr_avg_all_combined = _combine_avg_tfrs_group(tfr_all_list, logger)
    if tfr_avg_all_combined is None:
        _log("Group temperature grid: failed to combine TFRs; skipping.", logger, "warning")
        return
    
    times_corr = np.asarray(tfr_avg_all_combined.times)
    tmin_req, tmax_req = plateau_window
    tmin = float(max(times_corr.min(), tmin_req))
    tmax = float(min(times_corr.max(), tmax_req))
    
    fmax_available = float(np.max(tfr_avg_all_combined.freqs))
    bands = get_bands_for_tfr(max_freq_available=fmax_available, config=config)
    
    cond_tfrs: list[tuple[str, "mne.time_frequency.AverageTFR", int, float]] = []
    cond_tfrs.append(("All trials", tfr_avg_all_combined, len(tfr_all_list), np.nan))
    
    sorted_temps = sorted(all_temps)
    for tval in sorted_temps:
        if tval in tfr_by_temp and len(tfr_by_temp[tval]) >= 2:
            tfr_combined = _combine_avg_tfrs_group(tfr_by_temp[tval], logger)
            if tfr_combined is not None:
                cond_tfrs.append((f"{tval:.1f}°C", tfr_combined, len(tfr_by_temp[tval]), float(tval)))
    
    n_cols, n_rows = len(cond_tfrs), len(bands)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(10.0 * n_cols, 10.0 * n_rows), squeeze=False,
        gridspec_kw={"wspace": 1.2, "hspace": 0.50},
    )
    
    for r, (band, (fmin, fmax)) in enumerate(bands.items()):
        fmax_eff = min(fmax, fmax_available)
        if fmin >= fmax_eff:
            for c in range(n_cols):
                axes[r, c].axis("off")
            continue
        
        diff_datas: list[Optional[np.ndarray]] = []
        for _, tfr_cond, _, _ in cond_tfrs:
            d = _average_tfr_band(tfr_cond, fmin=fmin, fmax=fmax_eff, tmin=tmin, tmax=tmax)
            diff_datas.append(d)
        
        vals = [v for v in diff_datas if v is not None and np.isfinite(v).any()]
        if len(vals) == 0:
            for c in range(n_cols):
                axes[r, c].axis("off")
            continue
        
        diff_abs = robust_sym_vlim(vals, cap=100.0)
        if not np.isfinite(diff_abs) or diff_abs == 0:
            diff_abs = 1e-6
        
        for idx, (label, tfr_cond, n_cond, _tval) in enumerate(cond_tfrs, start=0):
            ax = axes[r, idx]
            data = diff_datas[idx]
            if data is None:
                ax.axis("off")
                continue
            
            plot_topomap_on_ax(ax, data, tfr_cond.info, vmin=-diff_abs, vmax=+diff_abs)
            _add_roi_annotations(ax, data, tfr_cond.info, config=config, data_format="percent")
            eeg_picks = extract_eeg_picks(tfr_cond, exclude_bads=False)
            mu = float(np.nanmean(data[eeg_picks])) if len(eeg_picks) > 0 else float(np.nanmean(data))
            ax.text(0.5, 1.02, f"%Δ={mu:+.1f}%", transform=ax.transAxes, ha="center", va="top", fontsize=9)
            if r == 0:
                ax.set_title(f"{label} (N={n_cond} subjects)", fontsize=9, pad=4, y=1.04)
        
        axes[r, 0].set_ylabel(f"{band} ({fmin:.0f}-{fmax_eff:.0f} Hz)", fontsize=10)
        
        sm_diff = ScalarMappable(norm=mcolors.TwoSlopeNorm(vmin=-diff_abs, vcenter=0.0, vmax=diff_abs), cmap=get_viz_params(config)["topo_cmap"])
        sm_diff.set_array([])
        cbar_d = fig.colorbar(sm_diff, ax=axes[r, :].ravel().tolist(), fraction=0.045, pad=0.06, shrink=0.9)
        cbar_d.set_label("Percent change from baseline (%)")
    
    fig.suptitle(
        f"Group Topomaps by temperature: % change from baseline over plateau t=[{tmin:.1f}, {tmax:.1f}] s (N={len(tfr_all_list)} subjects)",
        fontsize=12,
    )
    _save_fig(fig, out_dir, "group_topomap_grid_bands_alltrials_plus_temperatures_baseline_percent.png", config=config, logger=logger, baseline_used=baseline)


def group_plot_pain_nonpain_temporal_topomaps_diff_allbands(
    powers: List["mne.time_frequency.EpochsTFR"],
    events_list: List[Optional[pd.DataFrame]],
    out_dir: Path,
    config,
    baseline: Optional[Tuple[Optional[float], Optional[float]]] = None,
    plateau_window: Tuple[float, float] = (3.0, 10.5),
    window_size_ms: float = 100.0,
    logger: Optional[logging.Logger] = None,
) -> None:
    baseline = _get_baseline_window(config, baseline)
    
    if len(powers) < 2:
        _log("Group temporal topomaps require at least 2 subjects; skipping.", logger, "warning")
        return
    
    pain_col = next((c for c in config.get("event_columns.pain_binary", []) if c), None)
    temp_col = next((c for c in config.get("event_columns.temperature", []) if c), None)
    
    if pain_col is None:
        _log("Group temporal topomaps: pain column required; skipping.", logger, "warning")
        return
    
    tfr_pain_avg_list = []
    tfr_non_avg_list = []
    tfr_max_avg_list = []
    tfr_min_avg_list = []
    tfr_pain_epochs_list = []
    tfr_non_epochs_list = []
    tfr_max_epochs_list = []
    tfr_min_epochs_list = []
    has_temp = False
    baseline_used = None
    
    for tfr, events_df in zip(powers, events_list):
        if events_df is None:
            continue
        
        n = compute_aligned_data_length(tfr, events_df)
        
        pain_vec = extract_pain_vector_array(tfr, events_df, pain_col, n)
        if pain_vec is None:
            continue
        
        pain_mask, non_mask = _create_pain_masks_from_vector(pain_vec)
        if pain_mask is None:
            continue
        
        if pain_mask.sum() == 0 or non_mask.sum() == 0:
            continue
        
        tfr_sub = create_tfr_subset(tfr, n)
        aligned = _align_and_trim_masks(
            tfr_sub,
            {"Pain contrast": (pain_mask, non_mask)},
            config, logger
        )
        if aligned is None:
            continue
        
        pain_mask, non_mask = aligned["Pain contrast"]
        
        baseline_used = apply_baseline_and_crop(tfr_sub, baseline=baseline, mode="logratio", logger=logger)
        tfr_pain_avg_list.append(tfr_sub[pain_mask].average())
        tfr_non_avg_list.append(tfr_sub[non_mask].average())
        tfr_pain_epochs_list.append(tfr_sub[pain_mask])
        tfr_non_epochs_list.append(tfr_sub[non_mask])
        
        temp_series = extract_temperature_series(tfr, events_df, temp_col, n)
        if temp_series is not None:
            temp_result = create_temperature_masks(temp_series)
            if temp_result[0] is not None:
                t_min, t_max, mask_min, mask_max = temp_result
                
                if mask_min.sum() > 0 and mask_max.sum() > 0:
                    try:
                        ensure_aligned_lengths(
                            tfr_sub, mask_min, mask_max,
                            context=f"Temperature contrast",
                            strict=_get_strict_mode(config),
                            logger=logger
                        )
                        if len(mask_min) != len(tfr_sub) or len(mask_max) != len(tfr_sub):
                            mask_min = mask_min[:len(tfr_sub)]
                            mask_max = mask_max[:len(tfr_sub)]
                        tfr_min_avg_list.append(tfr_sub[mask_min].average())
                        tfr_max_avg_list.append(tfr_sub[mask_max].average())
                        tfr_min_epochs_list.append(tfr_sub[mask_min])
                        tfr_max_epochs_list.append(tfr_sub[mask_max])
                        has_temp = True
                    except ValueError:
                        pass
    
    if len(tfr_pain_avg_list) < 2:
        _log("Group temporal topomaps: insufficient subjects; skipping.", logger, "warning")
        return
    
    combined_avg = _combine_multiple_tfr_groups({
        "pain": tfr_pain_avg_list,
        "non": tfr_non_avg_list,
        "max": tfr_max_avg_list if has_temp else [],
        "min": tfr_min_avg_list if has_temp else []
    }, 2, logger)
    
    tfr_pain = combined_avg["pain"]
    tfr_non = combined_avg["non"]
    tfr_max = combined_avg["max"] if has_temp else None
    tfr_min = combined_avg["min"] if has_temp else None
    
    if tfr_pain is None or tfr_non is None:
        _log("Group temporal topomaps: failed to combine TFRs; skipping.", logger, "warning")
        return
    
    combined_epochs = _combine_multiple_epochs_tfr_groups({
        "pain": tfr_pain_epochs_list,
        "non": tfr_non_epochs_list,
        "max": tfr_max_epochs_list if has_temp else [],
        "min": tfr_min_epochs_list if has_temp else []
    }, 2, logger)
    
    tfr_pain_combined = combined_epochs["pain"]
    tfr_non_combined = combined_epochs["non"]
    tfr_max_combined = combined_epochs["max"] if has_temp else None
    tfr_min_combined = combined_epochs["min"] if has_temp else None
    
    times = np.asarray(tfr_pain.times)
    tmin_req, tmax_req = plateau_window
    tmin_start = float(times.min())
    tmax_clip = float(min(times.max(), tmax_req))
    
    if not np.isfinite(tmin_start) or not np.isfinite(tmax_clip) or (tmax_clip <= tmin_start):
        _log("Group temporal topomaps: no valid time interval; skipping.", logger, "warning")
        return
    
    window_starts, window_ends = create_time_windows_fixed_size(tmin_start, tmax_clip, window_size_ms)
    n_windows = len(window_starts)
    if n_windows == 0:
        _log("Group temporal topomaps: no valid windows; skipping.", logger, "warning")
        return
    
    fmax_available = float(np.max(tfr_pain.freqs))
    bands = get_bands_for_tfr(max_freq_available=fmax_available, config=config)
    
    valid_bands = {}
    all_band_pain_diff_data = {}
    all_band_temp_diff_data = {}
    
    for band_name, (fmin, fmax) in bands.items():
        fmax_eff = min(fmax, fmax_available)
        if fmin >= fmax_eff:
            continue
        
        pain_diff_data_windows = []
        temp_diff_data_windows = []
        
        for tmin_win, tmax_win in zip(window_starts, window_ends):
            pain_data = _average_tfr_band(tfr_pain, fmin=fmin, fmax=fmax_eff, tmin=tmin_win, tmax=tmax_win)
            non_data = _average_tfr_band(tfr_non, fmin=fmin, fmax=fmax_eff, tmin=tmin_win, tmax=tmax_win)
            
            if pain_data is not None and non_data is not None:
                diff_data = pain_data - non_data
                pain_diff_data_windows.append(diff_data)
            else:
                pain_diff_data_windows.append(None)
            
            if has_temp and tfr_max is not None and tfr_min is not None:
                max_data = _average_tfr_band(tfr_max, fmin=fmin, fmax=fmax_eff, tmin=tmin_win, tmax=tmax_win)
                min_data = _average_tfr_band(tfr_min, fmin=fmin, fmax=fmax_eff, tmin=tmin_win, tmax=tmax_win)
                
                if max_data is not None and min_data is not None:
                    temp_diff_data = max_data - min_data
                    temp_diff_data_windows.append(temp_diff_data)
                else:
                    temp_diff_data_windows.append(None)
            else:
                temp_diff_data_windows.append(None)
        
        pain_diff_data_valid = [d for d in pain_diff_data_windows if d is not None]
        if len(pain_diff_data_valid) == 0:
            continue
        
        valid_bands[band_name] = (fmin, fmax_eff, pain_diff_data_windows, temp_diff_data_windows if has_temp else None)
        all_band_pain_diff_data[band_name] = pain_diff_data_valid
        if has_temp:
            all_band_temp_diff_data[band_name] = [d for d in temp_diff_data_windows if d is not None]
    
    if len(valid_bands) == 0:
        _log("Group temporal topomaps: no valid bands; skipping.", logger, "warning")
        return
    
    all_pain_diff_data = [d for data_list in all_band_pain_diff_data.values() for d in data_list]
    all_temp_diff_data = [d for data_list in all_band_temp_diff_data.values() for d in data_list] if has_temp else []
    all_diff_data = all_pain_diff_data + all_temp_diff_data
    vabs_diff = robust_sym_vlim(all_diff_data) if len(all_diff_data) > 0 else 1e-6
    
    n_bands = len(valid_bands)
    n_rows = n_bands * 2 if has_temp else n_bands
    temporal_spacing = config.get("time_frequency_analysis.topomap.temporal.group", {}) if config else {}
    hspace = temporal_spacing.get("hspace", 0.4)
    wspace = temporal_spacing.get("wspace", 1.4)
    topo_size = max(1.2, min(3.0, 1800.0 / n_windows))
    fig_width = min(1800.0, topo_size * n_windows)
    fig, axes = plt.subplots(
        n_rows, n_windows, figsize=(fig_width * 1.5, 10.0 * n_rows), squeeze=False,
        gridspec_kw={"hspace": hspace, "wspace": wspace}
    )
    
    for band_idx, (band_name, (fmin, fmax_eff, pain_diff_data_windows, temp_diff_data_windows)) in enumerate(valid_bands.items()):
        freq_label = f"{band_name} ({fmin:.0f}-{fmax_eff:.0f}Hz)"
        row_pain = band_idx * 2 if has_temp else band_idx
        row_temp = band_idx * 2 + 1 if has_temp else None
        
        font_sizes = _get_font_sizes()
        axes[row_pain, 0].set_ylabel(f"Pain - Non\n{freq_label}", fontsize=font_sizes["ylabel"], labelpad=10)
        
        for col, (tmin_win, tmax_win) in enumerate(zip(window_starts, window_ends)):
            if row_pain == 0:
                time_label = f"{tmin_win:.2f}s"
                font_sizes = _get_font_sizes()
                axes[row_pain, col].set_title(time_label, fontsize=font_sizes["title"], pad=12, y=1.07)
            
            pain_diff_data = pain_diff_data_windows[col]
            if pain_diff_data is not None:
                pain_sig_mask = pain_cluster_p_min = pain_cluster_k = pain_cluster_mass = None
                if get_viz_params(config)["diff_annotation_enabled"]:
                    pain_sig_mask, pain_cluster_p_min, pain_cluster_k, pain_cluster_mass = _compute_cluster_significance_from_combined(
                        tfr_pain_combined, tfr_non_combined, fmin, fmax_eff, tmin_win, tmax_win, config, len(pain_diff_data), logger
                    )
                
                plot_topomap_on_ax(
                    axes[row_pain, col], pain_diff_data, tfr_pain.info,
                    vmin=-vabs_diff, vmax=+vabs_diff,
                    mask=(pain_sig_mask if viz_params["diff_annotation_enabled"] else None), 
                    mask_params=viz_params["sig_mask_params"],
                    config=config
                )
                pain_data_group_a = extract_trial_band_power(tfr_sub[pain_mask], fmin, fmax_eff, tmin_win, tmax_win)
                pain_data_group_b = extract_trial_band_power(tfr_sub[non_mask], fmin, fmax_eff, tmin_win, tmax_win)
                _add_roi_annotations(
                    axes[row_pain, col], pain_diff_data, tfr_pain.info, config=config,
                    sig_mask=(pain_sig_mask if viz_params["diff_annotation_enabled"] else None),
                    cluster_p_min=pain_cluster_p_min, cluster_k=pain_cluster_k, cluster_mass=pain_cluster_mass,
                    is_cluster=(pain_sig_mask is not None and pain_cluster_p_min is not None),
                    data_group_a=pain_data_group_a,
                    data_group_b=pain_data_group_b,
                    paired=False
                )
                label = _build_topomap_diff_label(pain_diff_data, pain_cluster_p_min, pain_cluster_k, pain_cluster_mass, config, viz_params, paired=False)
                axes[row_pain, col].text(0.5, 1.08, label, transform=axes[row_pain, col].transAxes, ha="center", va="bottom", fontsize=font_sizes["label"])
            else:
                axes[row_pain, col].axis('off')
            
            if has_temp and row_temp is not None and temp_diff_data_windows is not None:
                temp_diff_data = temp_diff_data_windows[col]
                if temp_diff_data is not None:
                    temp_sig_mask = temp_cluster_p_min = temp_cluster_k = temp_cluster_mass = None
                    if get_viz_params(config)["diff_annotation_enabled"]:
                        temp_sig_mask, temp_cluster_p_min, temp_cluster_k, temp_cluster_mass = _compute_cluster_significance_from_combined(
                            tfr_max_combined, tfr_min_combined, fmin, fmax_eff, tmin_win, tmax_win, config, len(temp_diff_data), logger
                        )
                    
                    plot_topomap_on_ax(
                        axes[row_temp, col], temp_diff_data, tfr_max.info,
                        vmin=-vabs_diff, vmax=+vabs_diff,
                        mask=(temp_sig_mask if get_viz_params(config)["diff_annotation_enabled"] else None), 
                        mask_params=get_viz_params(config)["sig_mask_params"],
                        config=config
                    )
                    _add_roi_annotations(
                        axes[row_temp, col], temp_diff_data, tfr_max.info, config=config,
                        sig_mask=(temp_sig_mask if get_viz_params(config)["diff_annotation_enabled"] else None),
                        cluster_p_min=temp_cluster_p_min, cluster_k=temp_cluster_k, cluster_mass=temp_cluster_mass,
                        is_cluster=(temp_sig_mask is not None and temp_cluster_p_min is not None)
                    )
                    mu = float(np.nanmean(temp_diff_data))
                    pct = logratio_to_pct(mu)
                    cl_txt = format_cluster_ann(temp_cluster_p_min, temp_cluster_k, temp_cluster_mass, config=config) if get_viz_params(config)["diff_annotation_enabled"] else ""
                    label = f"Δ%={pct:+.1f}%" + (f" | {cl_txt}" if cl_txt else "")
                    axes[row_temp, col].text(0.5, 1.08, label, transform=axes[row_temp, col].transAxes, ha="center", va="bottom", fontsize=font_sizes["label"])
                else:
                    axes[row_temp, col].axis('off')
                
                if col == 0:
                    axes[row_temp, 0].set_ylabel(f"Max - Min temp\n{freq_label}", fontsize=font_sizes["ylabel"], labelpad=10)
    
    viz_params = get_viz_params(config)
    _create_difference_colorbar(
        fig, axes, vabs_diff, viz_params["topo_cmap"],
        label="log10(power/baseline) difference"
    )
    
    baseline_str = f"bl{abs(baseline_used[0]):.1f}to{abs(baseline_used[1]):.2f}" if baseline_used else "bl"
    sig_text = _get_sig_marker_text(config)
    title_parts = [f"Group Temporal topomaps: Pain - Non-pain difference (all bands, {tmin_start:.1f}–{tmax_clip:.1f}s; {n_windows} windows @ {window_size_ms}ms, N={len(tfr_pain_avg_list)} subjects)"]
    if has_temp:
        title_parts.append(f"Max - Min temp")
    title_parts.append(f"log10(power/baseline) difference, vlim ±{vabs_diff:.2f}")
    font_sizes = _get_font_sizes()
    fig.suptitle(
        f"{' | '.join(title_parts)}{sig_text}\n",
        fontsize=font_sizes["suptitle"], y=0.995
    )
    
    filename = f"group_temporal_topomaps_pain_minus_nonpain_diff_allbands_{tmin_start:.0f}-{tmax_clip:.0f}s_{n_windows}windows_{baseline_str}.png"
    _save_fig(fig, out_dir, filename, config=config, logger=logger, baseline_used=baseline_used)


def group_plot_temporal_topomaps_allbands_plateau(
    powers: List["mne.time_frequency.EpochsTFR"],
    events_list: List[Optional[pd.DataFrame]],
    out_dir: Path,
    config,
    baseline: Optional[Tuple[Optional[float], Optional[float]]] = None,
    plateau_window: Tuple[float, float] = (3.0, 10.5),
    window_count: int = 5,
    logger: Optional[logging.Logger] = None,
) -> None:
    baseline = _get_baseline_window(config, baseline)
    
    if len(powers) < 2:
        _log("Group temporal topomaps require at least 2 subjects; skipping.", logger, "warning")
        return
    
    pain_col = next((c for c in config.get("event_columns.pain_binary", []) if c), None)
    temp_col = next((c for c in config.get("event_columns.temperature", []) if c), None)
    
    if pain_col is None:
        _log("Group temporal topomaps: pain column required; skipping.", logger, "warning")
        return
    
    tfr_pain_avg_list = []
    tfr_non_avg_list = []
    tfr_max_avg_list = []
    tfr_min_avg_list = []
    tfr_pain_epochs_list = []
    tfr_non_epochs_list = []
    tfr_max_epochs_list = []
    tfr_min_epochs_list = []
    has_temp = False
    baseline_used = None
    
    for tfr, events_df in zip(powers, events_list):
        if events_df is None:
            continue
        
        n = compute_aligned_data_length(tfr, events_df)
        
        pain_vec = extract_pain_vector_array(tfr, events_df, pain_col, n)
        if pain_vec is None:
            continue
        
        pain_mask, non_mask = _create_pain_masks_from_vector(pain_vec)
        if pain_mask is None:
            continue
        
        if pain_mask.sum() == 0 or non_mask.sum() == 0:
            continue
        
        tfr_sub = create_tfr_subset(tfr, n)
        aligned = _align_and_trim_masks(
            tfr_sub,
            {"Pain contrast": (pain_mask, non_mask)},
            config, logger
        )
        if aligned is None:
            continue
        
        pain_mask, non_mask = aligned["Pain contrast"]
        
        baseline_used = apply_baseline_and_crop(tfr_sub, baseline=baseline, mode="logratio", logger=logger)
        tfr_pain_avg_list.append(tfr_sub[pain_mask].average())
        tfr_non_avg_list.append(tfr_sub[non_mask].average())
        tfr_pain_epochs_list.append(tfr_sub[pain_mask])
        tfr_non_epochs_list.append(tfr_sub[non_mask])
        
        temp_series = extract_temperature_series(tfr, events_df, temp_col, n)
        if temp_series is not None:
            temp_result = create_temperature_masks(temp_series)
            if temp_result[0] is not None:
                t_min, t_max, mask_min, mask_max = temp_result
                
                if mask_min.sum() > 0 and mask_max.sum() > 0:
                    try:
                        ensure_aligned_lengths(
                            tfr_sub, mask_min, mask_max,
                            context=f"Temperature contrast",
                            strict=_get_strict_mode(config),
                            logger=logger
                        )
                        if len(mask_min) != len(tfr_sub) or len(mask_max) != len(tfr_sub):
                            mask_min = mask_min[:len(tfr_sub)]
                            mask_max = mask_max[:len(tfr_sub)]
                        tfr_min_avg_list.append(tfr_sub[mask_min].average())
                        tfr_max_avg_list.append(tfr_sub[mask_max].average())
                        tfr_min_epochs_list.append(tfr_sub[mask_min])
                        tfr_max_epochs_list.append(tfr_sub[mask_max])
                        has_temp = True
                    except ValueError:
                        pass
    
    if len(tfr_pain_avg_list) < 2:
        _log("Group temporal topomaps: insufficient subjects; skipping.", logger, "warning")
        return
    
    combined_avg = _combine_multiple_tfr_groups({
        "pain": tfr_pain_avg_list,
        "non": tfr_non_avg_list,
        "max": tfr_max_avg_list if has_temp else [],
        "min": tfr_min_avg_list if has_temp else []
    }, 2, logger)
    
    tfr_pain = combined_avg["pain"]
    tfr_non = combined_avg["non"]
    tfr_max = combined_avg["max"] if has_temp else None
    tfr_min = combined_avg["min"] if has_temp else None
    
    if tfr_pain is None or tfr_non is None:
        _log("Group temporal topomaps: failed to combine TFRs; skipping.", logger, "warning")
        return
    
    combined_epochs = _combine_multiple_epochs_tfr_groups({
        "pain": tfr_pain_epochs_list,
        "non": tfr_non_epochs_list,
        "max": tfr_max_epochs_list if has_temp else [],
        "min": tfr_min_epochs_list if has_temp else []
    }, 2, logger)
    
    tfr_pain_combined = combined_epochs["pain"]
    tfr_non_combined = combined_epochs["non"]
    tfr_max_combined = combined_epochs["max"] if has_temp else None
    tfr_min_combined = combined_epochs["min"] if has_temp else None
    
    times = np.asarray(tfr_pain.times)
    tmin_req, tmax_req = plateau_window
    clipped = clip_time_range(times, tmin_req, tmax_req)
    if clipped is None:
        return None
    tmin_clip, tmax_clip = clipped
    
    if tmin_clip is None or tmax_clip is None:
        _log("Group temporal topomaps: no valid time interval; skipping.", logger, "warning")
        return
    
    window_starts, window_ends = create_time_windows_fixed_count(tmin_clip, tmax_clip, window_count)
    n_windows = len(window_starts)
    
    if n_windows == 0:
        _log("Group temporal topomaps: no valid windows; skipping.", logger, "warning")
        return
    
    fmax_available = float(np.max(tfr_pain.freqs))
    bands = get_bands_for_tfr(max_freq_available=fmax_available, config=config)
    
    valid_bands = {}
    all_band_pain_diff_data = {}
    all_band_temp_diff_data = {}
    
    for band_name, (fmin, fmax) in bands.items():
        fmax_eff = min(fmax, fmax_available)
        if fmin >= fmax_eff:
            continue
        
        pain_diff_data_windows = []
        temp_diff_data_windows = []
        
        for tmin_win, tmax_win in zip(window_starts, window_ends):
            pain_data = _average_tfr_band(tfr_pain, fmin=fmin, fmax=fmax_eff, tmin=tmin_win, tmax=tmax_win)
            non_data = _average_tfr_band(tfr_non, fmin=fmin, fmax=fmax_eff, tmin=tmin_win, tmax=tmax_win)
            
            if pain_data is not None and non_data is not None:
                diff_data = pain_data - non_data
                pain_diff_data_windows.append(diff_data)
            else:
                pain_diff_data_windows.append(None)
            
            if has_temp and tfr_max is not None and tfr_min is not None:
                max_data = _average_tfr_band(tfr_max, fmin=fmin, fmax=fmax_eff, tmin=tmin_win, tmax=tmax_win)
                min_data = _average_tfr_band(tfr_min, fmin=fmin, fmax=fmax_eff, tmin=tmin_win, tmax=tmax_win)
                
                if max_data is not None and min_data is not None:
                    temp_diff_data = max_data - min_data
                    temp_diff_data_windows.append(temp_diff_data)
                else:
                    temp_diff_data_windows.append(None)
            else:
                temp_diff_data_windows.append(None)
        
        pain_diff_data_valid = [d for d in pain_diff_data_windows if d is not None]
        if len(pain_diff_data_valid) == 0:
            continue
        
        valid_bands[band_name] = (fmin, fmax_eff, pain_diff_data_windows, temp_diff_data_windows if has_temp else None)
        all_band_pain_diff_data[band_name] = pain_diff_data_valid
        if has_temp:
            all_band_temp_diff_data[band_name] = [d for d in temp_diff_data_windows if d is not None]
    
    if len(valid_bands) == 0:
        _log("Group temporal topomaps: no valid bands; skipping.", logger, "warning")
        return
    
    all_pain_diff_data = [d for data_list in all_band_pain_diff_data.values() for d in data_list]
    all_temp_diff_data = [d for data_list in all_band_temp_diff_data.values() for d in data_list] if has_temp else []
    all_diff_data = all_pain_diff_data + all_temp_diff_data
    vabs_diff = robust_sym_vlim(all_diff_data) if len(all_diff_data) > 0 else 1e-6
    
    n_bands = len(valid_bands)
    n_rows = n_bands * 2 if has_temp else n_bands
    temporal_spacing = config.get("time_frequency_analysis.topomap.temporal.group", {}) if config else {}
    hspace = temporal_spacing.get("hspace", 0.4)
    wspace = temporal_spacing.get("wspace", 1.4)
    topo_size = max(1.2, min(3.0, 1800.0 / n_windows))
    fig_width = min(1800.0, topo_size * n_windows)
    fig, axes = plt.subplots(
        n_rows, n_windows, figsize=(fig_width * 1.5, 10.0 * n_rows), squeeze=False,
        gridspec_kw={"hspace": hspace, "wspace": wspace}
    )
    
    for band_idx, (band_name, (fmin, fmax_eff, pain_diff_data_windows, temp_diff_data_windows)) in enumerate(valid_bands.items()):
        freq_label = f"{band_name} ({fmin:.0f}-{fmax_eff:.0f}Hz)"
        row_pain = band_idx * 2 if has_temp else band_idx
        row_temp = band_idx * 2 + 1 if has_temp else None
        
        font_sizes = _get_font_sizes()
        axes[row_pain, 0].set_ylabel(f"Pain - Non\n{freq_label}", fontsize=font_sizes["ylabel"], labelpad=10)
        
        for col, (tmin_win, tmax_win) in enumerate(zip(window_starts, window_ends)):
            if row_pain == 0:
                time_label = f"{tmin_win:.2f}s"
                font_sizes = _get_font_sizes()
                axes[row_pain, col].set_title(time_label, fontsize=font_sizes["title"], pad=12, y=1.07)
            
            pain_diff_data = pain_diff_data_windows[col]
            if pain_diff_data is not None:
                pain_sig_mask = pain_cluster_p_min = pain_cluster_k = pain_cluster_mass = None
                if get_viz_params(config)["diff_annotation_enabled"]:
                    pain_sig_mask, pain_cluster_p_min, pain_cluster_k, pain_cluster_mass = _compute_cluster_significance_from_combined(
                        tfr_pain_combined, tfr_non_combined, fmin, fmax_eff, tmin_win, tmax_win, config, len(pain_diff_data), logger
                    )
                
                plot_topomap_on_ax(
                    axes[row_pain, col], pain_diff_data, tfr_pain.info,
                    vmin=-vabs_diff, vmax=+vabs_diff,
                    mask=(pain_sig_mask if viz_params["diff_annotation_enabled"] else None), 
                    mask_params=viz_params["sig_mask_params"],
                    config=config
                )
                pain_data_group_a = extract_trial_band_power(tfr_sub[pain_mask], fmin, fmax_eff, tmin_win, tmax_win)
                pain_data_group_b = extract_trial_band_power(tfr_sub[non_mask], fmin, fmax_eff, tmin_win, tmax_win)
                _add_roi_annotations(
                    axes[row_pain, col], pain_diff_data, tfr_pain.info, config=config,
                    sig_mask=(pain_sig_mask if viz_params["diff_annotation_enabled"] else None),
                    cluster_p_min=pain_cluster_p_min, cluster_k=pain_cluster_k, cluster_mass=pain_cluster_mass,
                    is_cluster=(pain_sig_mask is not None and pain_cluster_p_min is not None),
                    data_group_a=pain_data_group_a,
                    data_group_b=pain_data_group_b,
                    paired=False
                )
                label = _build_topomap_diff_label(pain_diff_data, pain_cluster_p_min, pain_cluster_k, pain_cluster_mass, config, viz_params, paired=False)
                axes[row_pain, col].text(0.5, 1.08, label, transform=axes[row_pain, col].transAxes, ha="center", va="bottom", fontsize=font_sizes["label"])
            else:
                axes[row_pain, col].axis('off')
            
            if has_temp and row_temp is not None and temp_diff_data_windows is not None:
                temp_diff_data = temp_diff_data_windows[col]
                if temp_diff_data is not None:
                    temp_sig_mask = temp_cluster_p_min = temp_cluster_k = temp_cluster_mass = None
                    if get_viz_params(config)["diff_annotation_enabled"]:
                        temp_sig_mask, temp_cluster_p_min, temp_cluster_k, temp_cluster_mass = _compute_cluster_significance_from_combined(
                            tfr_max_combined, tfr_min_combined, fmin, fmax_eff, tmin_win, tmax_win, config, len(temp_diff_data), logger
                        )
                    
                    plot_topomap_on_ax(
                        axes[row_temp, col], temp_diff_data, tfr_max.info,
                        vmin=-vabs_diff, vmax=+vabs_diff,
                        mask=(temp_sig_mask if get_viz_params(config)["diff_annotation_enabled"] else None), 
                        mask_params=get_viz_params(config)["sig_mask_params"],
                        config=config
                    )
                    _add_roi_annotations(
                        axes[row_temp, col], temp_diff_data, tfr_max.info, config=config,
                        sig_mask=(temp_sig_mask if get_viz_params(config)["diff_annotation_enabled"] else None),
                        cluster_p_min=temp_cluster_p_min, cluster_k=temp_cluster_k, cluster_mass=temp_cluster_mass,
                        is_cluster=(temp_sig_mask is not None and temp_cluster_p_min is not None)
                    )
                    mu = float(np.nanmean(temp_diff_data))
                    pct = logratio_to_pct(mu)
                    cl_txt = format_cluster_ann(temp_cluster_p_min, temp_cluster_k, temp_cluster_mass, config=config) if get_viz_params(config)["diff_annotation_enabled"] else ""
                    label = f"Δ%={pct:+.1f}%" + (f" | {cl_txt}" if cl_txt else "")
                    axes[row_temp, col].text(0.5, 1.08, label, transform=axes[row_temp, col].transAxes, ha="center", va="bottom", fontsize=font_sizes["label"])
                else:
                    axes[row_temp, col].axis('off')
                
                if col == 0:
                    axes[row_temp, 0].set_ylabel(f"Max - Min temp\n{freq_label}", fontsize=font_sizes["ylabel"], labelpad=10)
    
    viz_params = get_viz_params(config)
    _create_difference_colorbar(
        fig, axes, vabs_diff, viz_params["topo_cmap"],
        label="log10(power/baseline) difference"
    )
    
    baseline_str = f"bl{abs(baseline_used[0]):.1f}to{abs(baseline_used[1]):.2f}" if baseline_used else "bl"
    sig_text = _get_sig_marker_text(config)
    title_parts = [f"Group Temporal topomaps: Pain - Non-pain difference (all bands, plateau {tmin_clip:.0f}–{tmax_clip:.0f}s; {n_windows} windows, N={len(tfr_pain_avg_list)} subjects)"]
    if has_temp:
        title_parts.append(f"Max - Min temp")
    title_parts.append(f"log10(power/baseline) difference, vlim ±{vabs_diff:.2f}")
    font_sizes = _get_font_sizes()
    fig.suptitle(
        f"{' | '.join(title_parts)}{sig_text}\n",
        fontsize=font_sizes["suptitle"], y=0.995
    )
    
    filename = f"group_temporal_topomaps_allbands_plateau_{tmin_clip:.0f}-{tmax_clip:.0f}s_{n_windows}windows_{baseline_str}.png"
    _save_fig(fig, out_dir, filename, config=config, logger=logger, baseline_used=baseline_used)


