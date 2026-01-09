"""
TFR contrast plotting functions.

Functions for creating pain and temperature contrast visualizations,
including max-min temperature contrasts and combined pain/temperature plots.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt

from eeg_pipeline.plotting.io.figures import (
    robust_sym_vlim,
    logratio_to_pct,
    get_viz_params,
    plot_topomap_on_ax,
)
from eeg_pipeline.utils.analysis.events import extract_comparison_mask
from eeg_pipeline.utils.data.columns import get_temperature_column_from_config
from eeg_pipeline.utils.validation import require_epochs_tfr, ensure_aligned_lengths
from ...utils.config.loader import get_config_value, ensure_config
from ...utils.analysis.tfr import (
    apply_baseline_and_crop,
    create_tfr_subset,
    get_bands_for_tfr,
    average_tfr_band,
    extract_trial_band_power,
)
from ...utils.analysis.stats import (
    cluster_test_epochs,
)
from ...utils.data.tfr_alignment import (
    compute_aligned_data_length,
    extract_temperature_series,
    get_temperature_range,
    create_temperature_masks_from_range,
)
from ..config import get_plot_config
from ..core.utils import get_font_sizes, log
from ..core.statistics import get_strict_mode, compute_cluster_significance
from ..core.colorbars import add_normalized_colorbar, add_diff_colorbar
from ..core.topomaps import build_topomap_diff_label
from ..core.annotations import add_roi_annotations, get_sig_marker_text


###################################################################
# Helper Functions
###################################################################


def _get_label_position(config) -> Tuple[float, float]:
    """Get label position from configuration.
    
    Args:
        config: Configuration object
        
    Returns:
        Tuple of (x_position, y_position)
    """
    plot_cfg = get_plot_config(config) if config else None
    if plot_cfg:
        tfr_config = plot_cfg.plot_type_configs.get("tfr", {})
        x_position = tfr_config.get("label_x_position", 0.5)
        y_position = tfr_config.get("label_y_position", 1.02)
    else:
        x_position = 0.5
        y_position = 1.02
    return x_position, y_position


def _get_aligned_events_df_for_tfr(tfr, events_df: Optional[pd.DataFrame], n: int) -> Optional[pd.DataFrame]:
    meta = getattr(tfr, "metadata", None)
    if isinstance(meta, pd.DataFrame) and not meta.empty:
        return meta.iloc[:n]
    if events_df is not None and not events_df.empty:
        return events_df.iloc[:n]
    return None


def _prepare_comparison_contrast_data(
    tfr,
    events_df: Optional[pd.DataFrame],
    config,
    logger: Optional[logging.Logger] = None,
    *,
    context: str = "Condition contrast",
) -> tuple[
    Optional[mne.time_frequency.EpochsTFR],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[str],
    Optional[str],
    Optional[int],
]:
    if not require_epochs_tfr(tfr, context, logger):
        return None, None, None, None, None, None

    n = compute_aligned_data_length(tfr, events_df)
    aligned_events = _get_aligned_events_df_for_tfr(tfr, events_df, n)
    if aligned_events is None:
        log("Events metadata required for contrast; skipping.", logger, "warning")
        return None, None, None, None, None, None

    comp = extract_comparison_mask(aligned_events, config, require_enabled=False)
    if comp is None:
        log("No comparison column/values available for contrast; skipping.", logger, "warning")
        return None, None, None, None, None, None

    mask1, mask2, label1, label2 = comp
    mask1 = np.asarray(mask1[:n], dtype=bool)
    mask2 = np.asarray(mask2[:n], dtype=bool)
    if int(mask1.sum()) == 0 or int(mask2.sum()) == 0:
        log("One of the groups has zero trials; skipping contrasts.", logger, "warning")
        return None, None, None, None, None, None

    tfr_sub = create_tfr_subset(tfr, n)
    try:
        ensure_aligned_lengths(
            tfr_sub,
            mask1,
            mask2,
            context=context,
            strict=get_strict_mode(config),
            logger=logger,
        )
    except ValueError as e:
        log(f"{e}. Skipping contrast.", logger, "error")
        return None, None, None, None, None, None

    if len(mask1) != len(tfr_sub):
        mask1 = mask1[: len(tfr_sub)]
        mask2 = mask2[: len(tfr_sub)]

    return tfr_sub, mask1, mask2, label1, label2, n


def _get_baseline_window(config, baseline: Optional[Tuple[Optional[float], Optional[float]]] = None) -> Tuple[Optional[float], Optional[float]]:
    """Get baseline window from config or use provided value.
    
    Args:
        config: Configuration object
        baseline: Optional baseline window tuple
        
    Returns:
        Baseline window tuple
    """
    if baseline is not None:
        return baseline
    if not config:
        return -5.0, -0.01

    override = config.get("plotting.tfr.default_baseline_window", None)
    if isinstance(override, (list, tuple)) and len(override) == 2:
        return tuple(override)

    return tuple(config.get("time_frequency_analysis.baseline_window", [-5.0, -0.01]))


def _create_pain_masks_from_vector(pain_vec) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Create pain and non-pain masks from pain vector.
    
    Args:
        pain_vec: Condition indicator vector (pandas Series or numpy array)
        
    Returns:
        Tuple of (pain_mask, non_mask)
    """
    if pain_vec is None:
        return None, None
    if hasattr(pain_vec, 'fillna'):
        pain_vec = pain_vec.fillna(0).astype(int).values
    pain_mask = np.asarray(pain_vec == 1, dtype=bool)
    non_mask = np.asarray(pain_vec == 0, dtype=bool)
    return pain_mask, non_mask


def _create_pain_masks_from_events(events_df: Optional[pd.DataFrame], pain_col: Optional[str]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Create pain and non-pain masks from events DataFrame.
    
    Args:
        events_df: Events DataFrame
        pain_col: Condition indicator column name
        
    Returns:
        Tuple of (pain_mask, non_mask)
    """
    if events_df is None or pain_col is None or pain_col not in events_df.columns:
        return None, None
    vals = pd.to_numeric(events_df[pain_col], errors="coerce").fillna(0).astype(int)
    return _create_pain_masks_from_vector(vals)


def _plot_topomap_with_label(
    ax: plt.Axes,
    data: np.ndarray,
    info: mne.Info,
    vmin: float,
    vmax: float,
    label_text: str,
    config=None,
) -> None:
    """Plot topomap with label text.
    
    Args:
        ax: Matplotlib axes
        data: Topomap data array
        info: MNE Info object
        vmin: Minimum value for colormap
        vmax: Maximum value for colormap
        label_text: Label text to display
        config: Optional configuration object
    """
    plot_cfg = get_plot_config(config) if config else None
    font_sizes = get_font_sizes(plot_cfg)
    x_position, y_position = _get_label_position(config)
    plot_topomap_on_ax(ax, data, info, vmin=vmin, vmax=vmax, config=config)
    add_roi_annotations(ax, data, info, config=config)
    ax.text(x_position, y_position, label_text, transform=ax.transAxes, ha="center", va="top", fontsize=font_sizes["title"])


def _plot_topomap_with_percentage_label(
    ax: plt.Axes,
    data: np.ndarray,
    info: mne.Info,
    vmin: float,
    vmax: float,
    config=None,
) -> None:
    """Plot topomap with percentage change label.
    
    Args:
        ax: Matplotlib axes
        data: Topomap data array
        info: MNE Info object
        vmin: Minimum value for colormap
        vmax: Maximum value for colormap
        config: Optional configuration object
    """
    from ..core.topomaps import build_topomap_percentage_label
    label_text = build_topomap_percentage_label(data)
    _plot_topomap_with_label(ax, data, info, vmin, vmax, label_text, config)


def _plot_topomap_with_diff_label(
    ax: plt.Axes,
    diff_data: np.ndarray,
    info: mne.Info,
    vmin: Optional[float],
    vmax: Optional[float],
    sig_mask: Optional[np.ndarray],
    cluster_info: Dict,
    config=None,
    data_group_a: Optional[np.ndarray] = None,
    data_group_b: Optional[np.ndarray] = None,
    paired: bool = False,
) -> None:
    """Plot topomap with difference label and significance mask.
    
    Args:
        ax: Matplotlib axes
        diff_data: Difference data array
        info: MNE Info object
        vmin: Minimum value for colormap
        vmax: Maximum value for colormap
        sig_mask: Optional significance mask
        cluster_info: Dictionary with cluster information
        config: Optional configuration object
        data_group_a: Optional data for group A
        data_group_b: Optional data for group B
        paired: Whether paired test was used
    """
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
    add_roi_annotations(
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
    
    label = build_topomap_diff_label(
        diff_data,
        cluster_info.get("cluster_p_min"),
        cluster_info.get("cluster_k"),
        cluster_info.get("cluster_mass"),
        config,
        viz_params,
        paired=paired
    )
    font_sizes = get_font_sizes()
    x_position, y_position = _get_label_position(config)
    ax.text(x_position, y_position, label, transform=ax.transAxes, ha="center", va="top", fontsize=font_sizes["title"])


def _compute_diff_vlims_from_vabs(diff_data: np.ndarray, vabs: float) -> Tuple[Optional[float], Optional[float]]:
    """Compute vmin and vmax for difference plot from absolute value.
    
    Args:
        diff_data: Difference data array
        vabs: Absolute value for symmetric range
        
    Returns:
        Tuple of (vmin, vmax) or (None, None) if no valid data
    """
    if vabs > 0:
        return -vabs, +vabs
    
    if diff_data is not None:
        diff_data_arr = np.asarray(diff_data)
        diff_finite = diff_data_arr[np.isfinite(diff_data_arr)]
        has_nonzero = diff_finite.size > 0 and np.any(np.abs(diff_finite) > 0)
    else:
        has_nonzero = False
    
    if has_nonzero:
        vabs_effective = robust_sym_vlim(diff_data)
        return -vabs_effective, +vabs_effective
    
    return None, None


def _plot_diff_topomap_with_label(
    ax: plt.Axes,
    diff_data: np.ndarray,
    info: mne.Info,
    vabs: float,
    sig_mask: Optional[np.ndarray],
    cluster_p_min: Optional[float],
    cluster_k: Optional[int],
    cluster_mass: Optional[float],
    config,
    viz_params: Dict,
    data_group_a: Optional[np.ndarray] = None,
    data_group_b: Optional[np.ndarray] = None,
    paired: bool = False,
) -> None:
    """Plot difference topomap with label and significance.
    
    Args:
        ax: Matplotlib axes
        diff_data: Difference data array
        info: MNE Info object
        vabs: Absolute value for symmetric range
        sig_mask: Optional significance mask
        cluster_p_min: Optional minimum cluster p-value
        cluster_k: Optional cluster size
        cluster_mass: Optional cluster mass
        config: Configuration object
        viz_params: Visualization parameters dictionary
        data_group_a: Optional data for group A
        data_group_b: Optional data for group B
        paired: Whether paired test was used
    """
    vmin, vmax = _compute_diff_vlims_from_vabs(diff_data, vabs)
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
    add_roi_annotations(
        ax, diff_data, info, config=config,
        sig_mask=(sig_mask if viz_params["diff_annotation_enabled"] else None),
        cluster_p_min=cluster_p_min, cluster_k=cluster_k, cluster_mass=cluster_mass,
        is_cluster=(sig_mask is not None and cluster_p_min is not None),
        data_group_a=data_group_a,
        data_group_b=data_group_b,
        paired=paired
    )
    label = build_topomap_diff_label(diff_data, cluster_p_min, cluster_k, cluster_mass, config, viz_params, paired=paired)
    font_sizes = get_font_sizes()
    x_position, y_position = _get_label_position(config)
    ax.text(x_position, y_position, label, transform=ax.transAxes, ha="center", va="top", fontsize=font_sizes["title"])


def _compute_band_diff_data(
    tfr_condition_2,
    tfr_condition_1,
    tfr_max_temp,
    tfr_min_temp,
    fmin: float,
    fmax_effective: float,
    tmin: float,
    tmax: float,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Compute band difference data for condition and temperature contrasts.
    
    Args:
        tfr_condition_2: TFR for condition 2
        tfr_condition_1: TFR for condition 1
        tfr_max_temp: TFR for maximum temperature
        tfr_min_temp: TFR for minimum temperature
        fmin: Minimum frequency
        fmax_effective: Effective maximum frequency
        tmin: Minimum time
        tmax: Maximum time
        
    Returns:
        Tuple of (condition_diff_data, temperature_diff_data) or (None, None) on failure
    """
    condition_2_data = average_tfr_band(tfr_condition_2, fmin=fmin, fmax=fmax_effective, tmin=tmin, tmax=tmax)
    condition_1_data = average_tfr_band(tfr_condition_1, fmin=fmin, fmax=fmax_effective, tmin=tmin, tmax=tmax)
    max_temp_data = average_tfr_band(tfr_max_temp, fmin=fmin, fmax=fmax_effective, tmin=tmin, tmax=tmax)
    min_temp_data = average_tfr_band(tfr_min_temp, fmin=fmin, fmax=fmax_effective, tmin=tmin, tmax=tmax)
    
    if condition_2_data is None or condition_1_data is None or max_temp_data is None or min_temp_data is None:
        return None, None
    
    condition_diff_data = condition_2_data - condition_1_data
    temperature_diff_data = max_temp_data - min_temp_data
    return condition_diff_data, temperature_diff_data


def _trim_mask_to_length(mask, target_length: int):
    """Trim mask to target length if needed.
    
    Args:
        mask: Mask array
        target_length: Target length
        
    Returns:
        Trimmed mask
    """
    if len(mask) != target_length:
        return mask[:target_length]
    return mask


def _align_and_trim_masks(
    tfr_sub,
    masks_dict: Dict[str, Tuple],
    config,
    logger: Optional[logging.Logger] = None,
) -> Optional[Dict]:
    """Align and trim masks to match TFR length.
    
    Args:
        tfr_sub: TFR subset object
        masks_dict: Dictionary mapping context names to mask tuples
        config: Configuration object
        logger: Optional logger instance
        
    Returns:
        Dictionary of trimmed masks or None on failure
    """
    try:
        for context, masks in masks_dict.items():
            if masks is None:
                continue
            mask_list = masks if isinstance(masks, (list, tuple)) else [masks]
            ensure_aligned_lengths(
                tfr_sub, *mask_list,
                context=context,
                strict=get_strict_mode(config),
                logger=logger
            )
        
        tfr_len = len(tfr_sub)
        trimmed = {}
        for key, masks in masks_dict.items():
            if masks is None:
                trimmed[key] = None
            elif isinstance(masks, (list, tuple)):
                trimmed[key] = [_trim_mask_to_length(m, tfr_len) for m in masks]
            else:
                trimmed[key] = _trim_mask_to_length(masks, tfr_len)
        
        return trimmed
    except ValueError as e:
        log(f"{e}. Skipping alignment.", logger, "error")
        return None




def _extract_subject_means(
    tfr1_list: List["mne.time_frequency.EpochsTFR"],
    tfr2_list: List["mne.time_frequency.EpochsTFR"],
    fmin: float,
    fmax_effective: float,
    tmin: float,
    tmax: float,
) -> Tuple[List[np.ndarray], List[np.ndarray], List["mne.time_frequency.EpochsTFR"], List["mne.time_frequency.EpochsTFR"], Optional[mne.Info]]:
    """Extract subject-level means from TFR lists.
    
    Args:
        tfr1_list: List of EpochsTFR objects for condition 1
        tfr2_list: List of EpochsTFR objects for condition 2
        fmin: Minimum frequency
        fmax_effective: Effective maximum frequency
        tmin: Minimum time
        tmax: Maximum time
        
    Returns:
        Tuple of (subject_1_means, subject_2_means, valid_tfr1_list, valid_tfr2_list, reference_info)
    """
    subject_1_means = []
    subject_2_means = []
    valid_tfr1_list = []
    valid_tfr2_list = []
    reference_info = None
    
    for tfr_1, tfr_2 in zip(tfr1_list, tfr2_list):
        if not isinstance(tfr_1, mne.time_frequency.EpochsTFR) or not isinstance(tfr_2, mne.time_frequency.EpochsTFR):
            continue
        
        data_1 = extract_trial_band_power(tfr_1, fmin, fmax_effective, tmin, tmax)
        data_2 = extract_trial_band_power(tfr_2, fmin, fmax_effective, tmin, tmax)
        
        if data_1 is None or data_2 is None:
            continue
        
        if data_1.shape[0] == 0 or data_2.shape[0] == 0:
            continue
        
        mean_1 = np.nanmean(data_1, axis=0)
        mean_2 = np.nanmean(data_2, axis=0)
        
        if mean_1.ndim == 0:
            mean_1 = mean_1.reshape(1)
        if mean_2.ndim == 0:
            mean_2 = mean_2.reshape(1)
        
        if reference_info is None:
            reference_info = tfr_1.info
        
        subject_1_means.append(mean_1)
        subject_2_means.append(mean_2)
        valid_tfr1_list.append(tfr_1)
        valid_tfr2_list.append(tfr_2)
    
    return subject_1_means, subject_2_means, valid_tfr1_list, valid_tfr2_list, reference_info


def _find_common_channels(tfr_list: List["mne.time_frequency.EpochsTFR"]) -> List[str]:
    """Find common channels across TFR objects.
    
    Args:
        tfr_list: List of EpochsTFR objects
        
    Returns:
        List of common channel names
    """
    if not tfr_list:
        return []
    
    ch_sets = [set(tfr.info["ch_names"]) for tfr in tfr_list]
    common_chs = list(sorted(set.intersection(*ch_sets))) if ch_sets else []
    return common_chs


def _align_subject_data_to_common_channels(
    subject_1_means: List[np.ndarray],
    subject_2_means: List[np.ndarray],
    valid_tfr1_list: List["mne.time_frequency.EpochsTFR"],
    valid_tfr2_list: List["mne.time_frequency.EpochsTFR"],
    common_channels: List[str],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Align subject data to common channels.
    
    Args:
        subject_1_means: List of mean arrays for condition 1
        subject_2_means: List of mean arrays for condition 2
        valid_tfr1_list: List of valid TFR objects for condition 1
        valid_tfr2_list: List of valid TFR objects for condition 2
        common_channels: List of common channel names
        
    Returns:
        Tuple of (group_1_array, group_2_array) or (None, None) on failure
    """
    group_1_array = []
    group_2_array = []
    
    for mean_1, mean_2, tfr_1, tfr_2 in zip(subject_1_means, subject_2_means, valid_tfr1_list, valid_tfr2_list):
        ch_indices_1 = [tfr_1.info["ch_names"].index(ch) for ch in common_channels if ch in tfr_1.info["ch_names"]]
        ch_indices_2 = [tfr_2.info["ch_names"].index(ch) for ch in common_channels if ch in tfr_2.info["ch_names"]]
        
        if len(ch_indices_1) != len(common_channels) or len(ch_indices_2) != len(common_channels):
            continue
        
        group_1_array.append(mean_1[ch_indices_1])
        group_2_array.append(mean_2[ch_indices_2])
    
    if not group_1_array or not group_2_array:
        return None, None
    
    return np.stack(group_1_array, axis=0), np.stack(group_2_array, axis=0)


def _compute_cluster_significance_from_combined(
    tfr1_list: List["mne.time_frequency.EpochsTFR"],
    tfr2_list: List["mne.time_frequency.EpochsTFR"],
    fmin: float,
    fmax_effective: float,
    tmin: float,
    tmax: float,
    config,
    diff_data_len: Optional[int],
    logger: Optional[logging.Logger] = None,
) -> Tuple[Optional[np.ndarray], Optional[float], Optional[int], Optional[float]]:
    """Compute cluster significance from combined subject-level TFRs.
    
    Args:
        tfr1_list: List of EpochsTFR objects for condition 1 (one per subject)
        tfr2_list: List of EpochsTFR objects for condition 2 (one per subject)
        fmin: Minimum frequency
        fmax_effective: Effective maximum frequency
        tmin: Minimum time
        tmax: Maximum time
        config: Configuration object
        diff_data_len: Expected length of difference data (for validation)
        logger: Optional logger instance
        
    Returns:
        Tuple of (sig_mask, cluster_p_min, cluster_k, cluster_mass)
    """
    from ...utils.analysis.stats import cluster_test_two_sample_arrays as _cluster_test_two_sample_arrays
    
    min_subjects_required = 2
    if not tfr1_list or not tfr2_list:
        return None, None, None, None
    
    if len(tfr1_list) < min_subjects_required or len(tfr2_list) < min_subjects_required:
        log("Cluster test requires at least 2 subjects per condition; skipping", logger, "warning")
        return None, None, None, None
    
    subject_1_means, subject_2_means, valid_tfr1_list, valid_tfr2_list, reference_info = _extract_subject_means(
        tfr1_list, tfr2_list, fmin, fmax_effective, tmin, tmax
    )
    
    if len(subject_1_means) < min_subjects_required:
        log("Insufficient subjects with valid data for cluster test; skipping", logger, "warning")
        return None, None, None, None
    
    all_tfr_list = valid_tfr1_list + valid_tfr2_list
    common_channels = _find_common_channels(all_tfr_list)
    
    if len(common_channels) == 0:
        log("No common channels across subjects for cluster test; skipping", logger, "warning")
        return None, None, None, None
    
    group_1_array, group_2_array = _align_subject_data_to_common_channels(
        subject_1_means, subject_2_means, valid_tfr1_list, valid_tfr2_list, common_channels
    )
    
    if group_1_array is None or group_2_array is None:
        log("Insufficient subjects after channel alignment for cluster test; skipping", logger, "warning")
        return None, None, None, None
    
    if len(group_1_array) < min_subjects_required or len(group_2_array) < min_subjects_required:
        log("Insufficient subjects after channel alignment for cluster test; skipping", logger, "warning")
        return None, None, None, None
    
    info_common = mne.pick_info(reference_info, [reference_info["ch_names"].index(ch) for ch in common_channels])
    
    sig_mask_full, cluster_p_min, cluster_k, cluster_mass = _cluster_test_two_sample_arrays(
        group_1_array, group_2_array, info_common,
        alpha=get_config_value(ensure_config(config), "statistics.sig_alpha", 0.05),
        paired=False,
        n_permutations=get_config_value(ensure_config(config), "statistics.cluster_n_perm", 100),
        config=config
    )
    
    if sig_mask_full is None:
        return None, None, None, None
    
    if diff_data_len is not None and len(sig_mask_full) != diff_data_len:
        log(f"Cluster significance mask length mismatch: sig_mask length={len(sig_mask_full)}, diff_data_len={diff_data_len}. Discarding results.", logger, "warning")
        return None, None, None, None
    
    return sig_mask_full, cluster_p_min, cluster_k, cluster_mass


def _save_fig(
    fig_obj,
    out_dir: Path,
    name: str,
    config,
    formats=None,
    logger: Optional[logging.Logger] = None,
    baseline_used: Optional[Tuple[float, float]] = None,
) -> None:
    """Save figure with proper formatting.
    
    Args:
        fig_obj: Matplotlib figure or list of figures
        out_dir: Output directory path
        name: Base filename
        config: Configuration object
        formats: Optional list of file formats
        logger: Optional logger instance
        baseline_used: Optional baseline window tuple
    """
    from eeg_pipeline.plotting.io.figures import save_fig as central_save_fig, build_footer
    from eeg_pipeline.utils.formatting import format_baseline_window_string
    
    out_dir.mkdir(parents=True, exist_ok=True)

    if baseline_used is None:
        override = config.get("plotting.tfr.default_baseline_window", None)
        if isinstance(override, (list, tuple)) and len(override) == 2:
            baseline_used = tuple(override)
        else:
            baseline_used = tuple(config.get("time_frequency_analysis.baseline_window", [-5.0, -0.01]))

    figs = fig_obj if isinstance(fig_obj, list) else [fig_obj]
    stem, _ = (name.rsplit(".", 1) + [""])[:2]
    baseline_str = format_baseline_window_string(baseline_used)
    if baseline_str not in stem:
        stem = f"{stem}_{baseline_str}"
    
    plot_cfg = get_plot_config(config)
    exts = formats if formats else list(plot_cfg.formats) if plot_cfg.formats else ["png"]
    
    default_footer_template = "tfr_baseline"
    baseline_decimal_places = 2
    footer_text = None
    template_name = config.get("output.tfr_footer_template", default_footer_template)
    footer_kwargs = {
        "baseline_window": baseline_used,
        "baseline": f"[{float(baseline_used[0]):.{baseline_decimal_places}f}, {float(baseline_used[1]):.{baseline_decimal_places}f}] s",
    }
    footer_text = build_footer(template_name, config, **footer_kwargs)

    for i, f in enumerate(figs):
        out_name = f"{stem}.{exts[0]}" if i == 0 else f"{stem}_{i+1}.{exts[0]}"
        out_path = out_dir / out_name
        central_save_fig(
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
# Contrast Plotting Functions
###################################################################


def contrast_maxmin_temperature(
    tfr: "mne.time_frequency.EpochsTFR | mne.time_frequency.AverageTFR",
    events_df: Optional[pd.DataFrame],
    out_dir: Path,
    config,
    baseline: Optional[Tuple[Optional[float], Optional[float]]] = None,
    active_window: Tuple[float, float] = (3.0, 10.5),
    logger: Optional[logging.Logger] = None,
) -> None:
    """Plot max vs min temperature contrast topomaps.
    
    Creates topomap grid showing max temperature, min temperature, and their difference
    across frequency bands.
    
    Args:
        tfr: MNE TFR object (EpochsTFR or AverageTFR)
        events_df: Optional events DataFrame with temperature column
        out_dir: Output directory path
        config: Configuration object
        baseline: Optional baseline window tuple (defaults to config)
        active_window: Active window tuple for statistics
        logger: Optional logger instance
    """
    baseline = _get_baseline_window(config, baseline)
    if not require_epochs_tfr(tfr, "Max-vs-min temperature contrast", logger):
        return
    if events_df is None:
        log("Max-vs-min temperature contrast requires events_df; skipping.", logger)
        return
    temp_col = get_temperature_column_from_config(config, events_df)
    if temp_col is None:
        log("Max-vs-min temperature contrast: no temperature column found; skipping.", logger)
        return

    n = compute_aligned_data_length(tfr, events_df)
    
    temp_series = extract_temperature_series(tfr, events_df, temp_col, n)
    if temp_series is None:
        log("Max-vs-min temperature contrast: no temperature column found; skipping.", logger)
        return

    t_min, t_max = get_temperature_range(temp_series)
    if t_min is None or t_max is None:
        log("Max-vs-min temperature contrast: need at least 2 temperature levels; skipping.", logger)
        return

    mask_min, mask_max = create_temperature_masks_from_range(temp_series, t_min, t_max)
    if mask_min.sum() == 0 or mask_max.sum() == 0:
        log(f"Max-vs-min temperature contrast: zero trials in one group (min n={int(mask_min.sum())}, max n={int(mask_max.sum())}); skipping.", logger)
        return

    tfr_sub = create_tfr_subset(tfr, n)
    try:
        strict_mode = get_strict_mode(config)
        ensure_aligned_lengths(
            tfr_sub, mask_min, mask_max,
            context=f"Temperature contrast",
            strict=strict_mode,
            logger=logger
        )
    except ValueError as e:
        log(f"{e}. Skipping contrast.", logger, "error")
        return
    if len(mask_min) != len(tfr_sub) or len(mask_max) != len(tfr_sub):
        mask_min = mask_min[:len(tfr_sub)]
        mask_max = mask_max[:len(tfr_sub)]

    tfr_sub_stats = tfr_sub.copy()
    baseline_used = apply_baseline_and_crop(tfr_sub_stats, baseline=baseline, mode="logratio", logger=logger)

    tfr_min = tfr_sub[mask_min].average()
    tfr_max = tfr_sub[mask_max].average()

    apply_baseline_and_crop(tfr_min, baseline=baseline_used, mode="logratio", logger=logger)
    apply_baseline_and_crop(tfr_max, baseline=baseline_used, mode="logratio", logger=logger)

    times = np.asarray(tfr_max.times)
    tmin_required, tmax_required = active_window
    tmin_effective = float(max(times.min(), tmin_required))
    tmax_effective = float(min(times.max(), tmax_required))
    tmin, tmax = tmin_effective, tmax_effective

    fmax_available = float(np.max(tfr_max.freqs))
    bands = get_bands_for_tfr(max_freq_available=fmax_available, config=config)

    n_rows = len(bands)
    n_cols = 4
    plot_cfg = get_plot_config(config)
    fig_size_per_col = plot_cfg.get_figure_size("tfr_per_col_large", plot_type="tfr")[0]
    fig_size_per_row = plot_cfg.get_figure_size("tfr_per_row_medium", plot_type="tfr")[1]
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_size_per_col * n_cols, fig_size_per_row * n_rows),
        squeeze=False,
        gridspec_kw={"width_ratios": [1.0, 1.0, 0.25, 1.0], "wspace": 1.2, "hspace": 0.4},
    )
    for r, (band, (fmin, fmax)) in enumerate(bands.items()):
        fmax_effective = min(fmax, fmax_available)
        if fmin >= fmax_effective:
            for c in range(n_cols):
                axes[r, c].axis("off")
            continue
        max_data = average_tfr_band(tfr_max, fmin=fmin, fmax=fmax_effective, tmin=tmin, tmax=tmax)
        min_data = average_tfr_band(tfr_min, fmin=fmin, fmax=fmax_effective, tmin=tmin, tmax=tmax)
        if max_data is None or min_data is None:
            for c in range(n_cols):
                axes[r, c].axis("off")
            continue

        sig_mask = None
        cluster_p_min = cluster_k = cluster_mass = None
        viz_params = get_viz_params(config)
        if viz_params["diff_annotation_enabled"]:
            sig_mask, cluster_p_min, cluster_k, cluster_mass = cluster_test_epochs(
                tfr_sub_stats, mask_max, mask_min, fmin=fmin, fmax=fmax_effective, tmin=tmin, tmax=tmax, paired=False, config=config
            )

        diff_data = max_data - min_data
        vabs_symmetric = robust_sym_vlim([max_data, min_data])
        diff_abs = robust_sym_vlim(diff_data) if np.isfinite(diff_data).any() else 0.0

        _plot_topomap_with_percentage_label(
            axes[r, 0], max_data, tfr_max.info, -vabs_symmetric, +vabs_symmetric, config
        )

        _plot_topomap_with_percentage_label(
            axes[r, 1], min_data, tfr_min.info, -vabs_symmetric, +vabs_symmetric, config
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

        font_sizes = get_font_sizes()
        if r == 0:
            title_pad = 4
            title_y = 1.04
            axes[r, 0].set_title(f"Max {t_max:.1f}°C (n={int(mask_max.sum())})", fontsize=font_sizes["title"], pad=title_pad, y=title_y)
            axes[r, 1].set_title(f"Min {t_min:.1f}°C (n={int(mask_min.sum())})", fontsize=font_sizes["title"], pad=title_pad, y=title_y)
            axes[r, 3].set_title("Max - Min", fontsize=font_sizes["title"], pad=title_pad, y=title_y)
        axes[r, 0].set_ylabel(f"{band} ({fmin:.0f}-{fmax_effective:.0f} Hz)", fontsize=font_sizes["ylabel"])

        add_normalized_colorbar(fig, [axes[r, 0], axes[r, 1]], -vabs_symmetric, +vabs_symmetric, viz_params["topo_cmap"], config)
        add_diff_colorbar(fig, axes[r, 3], diff_abs, viz_params["topo_cmap"], config)

    sig_text = get_sig_marker_text(config)
    font_sizes = get_font_sizes()
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
        baseline_used=baseline_used,
        logger=logger,
    )


def plot_bands_pain_temp_contrasts(
    tfr: "mne.time_frequency.EpochsTFR",
    events_df: Optional[pd.DataFrame],
    out_dir: Path,
    config,
    baseline: Optional[Tuple[Optional[float], Optional[float]]] = None,
    active_window: Tuple[float, float] = (3.0, 10.5),
    logger: Optional[logging.Logger] = None,
) -> None:
    """Plot combined pain and temperature contrast topomaps.
    
    Creates topomap grid showing pain-nonpain and max-min temperature differences
    across frequency bands.
    
    Args:
        tfr: MNE EpochsTFR object
        events_df: Optional events DataFrame with pain and temperature columns
        out_dir: Output directory path
        config: Configuration object
        baseline: Optional baseline window tuple (defaults to config)
        active_window: Active window tuple for statistics
        logger: Optional logger instance
    """
    baseline = _get_baseline_window(config, baseline)
    if not require_epochs_tfr(tfr, "Combined contrast", logger):
        return
    tfr_sub, mask1, mask2, label1, label2, n = _prepare_comparison_contrast_data(
        tfr, events_df, config, logger, context="Combined contrast"
    )
    if tfr_sub is None:
        return

    aligned_events = _get_aligned_events_df_for_tfr(tfr, events_df, int(n))
    if aligned_events is None:
        log("Combined contrast: no events metadata found; skipping.", logger, "warning")
        return

    temp_col = get_temperature_column_from_config(config, aligned_events)
    if temp_col is None:
        log("Combined contrast: no temperature column found; skipping.", logger, "warning")
        return

    temp_series = extract_temperature_series(tfr, aligned_events, temp_col, int(n))
    if temp_series is None:
        log("Combined contrast: no temperature column found; skipping.", logger, "warning")
        return

    t_min, t_max = get_temperature_range(temp_series)
    if t_min is None or t_max is None:
        log("Combined contrast: need at least 2 temperature levels; skipping.", logger, "warning")
        return

    mask_min, mask_max = create_temperature_masks_from_range(temp_series, t_min, t_max)
    
    if mask_min.sum() == 0 or mask_max.sum() == 0:
        log(f"Combined contrast: zero trials in one temp group (min n={int(mask_min.sum())}, max n={int(mask_max.sum())}); skipping.", logger, "warning")
        return

    aligned = _align_and_trim_masks(
        tfr_sub,
        {
            "Condition contrast": (mask2, mask1),
            "Temperature contrast": (mask_min, mask_max)
        },
        config, logger
    )
    if aligned is None:
        return
    
    mask2, mask1 = aligned["Condition contrast"]
    mask_min, mask_max = aligned["Temperature contrast"]

    tfr_2 = tfr_sub[mask2].average()
    tfr_1 = tfr_sub[mask1].average()
    tfr_min = tfr_sub[mask_min].average()
    tfr_max = tfr_sub[mask_max].average()

    baseline_used = apply_baseline_and_crop(tfr_2, baseline=baseline, mode="logratio", logger=logger)
    apply_baseline_and_crop(tfr_1, baseline=baseline, mode="logratio", logger=logger)
    apply_baseline_and_crop(tfr_min, baseline=baseline, mode="logratio", logger=logger)
    apply_baseline_and_crop(tfr_max, baseline=baseline, mode="logratio", logger=logger)

    times = np.asarray(tfr_2.times)
    tmin_required, tmax_required = active_window
    tmin_effective = float(max(times.min(), tmin_required))
    tmax_effective = float(min(times.max(), tmax_required))
    tmin, tmax = tmin_effective, tmax_effective

    fmax_available = float(np.max(tfr_2.freqs))
    bands = get_bands_for_tfr(max_freq_available=fmax_available, config=config)

    n_rows = 2
    n_cols = len(bands)
    plot_cfg = get_plot_config(config)
    fig_size_per_col = plot_cfg.get_figure_size("tfr_per_col_large", plot_type="tfr")[0]
    fig_size_per_row = plot_cfg.get_figure_size("tfr_per_row_large", plot_type="tfr")[1]
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_size_per_col * n_cols, fig_size_per_row * n_rows),
        squeeze=False,
        gridspec_kw={"wspace": 1.2, "hspace": 0.25},
    )

    viz_params = get_viz_params(config)
    n_2 = int(mask2.sum())
    n_1 = int(mask1.sum())
    n_max = int(mask_max.sum())
    n_min = int(mask_min.sum())

    all_cond_diff = []
    all_temp_diff = []
    band_significance_data = {}
    
    for c, (band, (fmin, fmax)) in enumerate(bands.items()):
        fmax_effective = min(fmax, fmax_available)
        if fmin >= fmax_effective:
            for r in range(n_rows):
                axes[r, c].axis('off')
            continue

        cond_diff_data, temp_diff_data = _compute_band_diff_data(tfr_2, tfr_1, tfr_max, tfr_min, fmin, fmax_effective, tmin, tmax)
        
        if cond_diff_data is None or temp_diff_data is None:
            for r in range(n_rows):
                axes[r, c].axis('off')
            continue
        
        all_cond_diff.append(cond_diff_data)
        all_temp_diff.append(temp_diff_data)

        cond_sig_mask = None
        cond_cluster_p_min = cond_cluster_k = cond_cluster_mass = None
        if viz_params["diff_annotation_enabled"]:
            cond_diff_data_len = len(cond_diff_data) if cond_diff_data is not None else None
            cond_sig_mask, cond_cluster_p_min, cond_cluster_k, cond_cluster_mass = compute_cluster_significance(
                tfr_sub, mask2, mask1, fmin, fmax_effective, tmin, tmax, config, diff_data_len=cond_diff_data_len, logger=logger
            )

        temp_sig_mask = None
        temp_cluster_p_min = temp_cluster_k = temp_cluster_mass = None
        if viz_params["diff_annotation_enabled"]:
            temp_diff_data_len = len(temp_diff_data) if temp_diff_data is not None else None
            temp_sig_mask, temp_cluster_p_min, temp_cluster_k, temp_cluster_mass = compute_cluster_significance(
                tfr_sub, mask_max, mask_min, fmin, fmax_effective, tmin, tmax, config, diff_data_len=temp_diff_data_len, logger=logger
            )
        
        band_significance_data[band] = {
            "cond": (cond_sig_mask, cond_cluster_p_min, cond_cluster_k, cond_cluster_mass),
            "temp": (temp_sig_mask, temp_cluster_p_min, temp_cluster_k, temp_cluster_mass),
            "cond_diff": cond_diff_data,
            "temp_diff": temp_diff_data,
        }

    cond_diff_abs = robust_sym_vlim(all_cond_diff) if len(all_cond_diff) > 0 else 0.0
    temp_diff_abs = robust_sym_vlim(all_temp_diff) if len(all_temp_diff) > 0 else 0.0
    
    for c, (band, (fmin, fmax)) in enumerate(bands.items()):
        fmax_effective = min(fmax, fmax_available)
        if fmin >= fmax_effective:
            continue

        if band not in band_significance_data:
            continue

        band_data = band_significance_data[band]
        cond_diff_data = band_data["cond_diff"]
        temp_diff_data = band_data["temp_diff"]
        cond_sig_mask, cond_cluster_p_min, cond_cluster_k, cond_cluster_mass = band_data["cond"]
        temp_sig_mask, temp_cluster_p_min, temp_cluster_k, temp_cluster_mass = band_data["temp"]

        cond_data_group_a = extract_trial_band_power(tfr_sub[mask2], fmin, fmax_effective, tmin, tmax)
        cond_data_group_b = extract_trial_band_power(tfr_sub[mask1], fmin, fmax_effective, tmin, tmax)
        temp_data_group_a = extract_trial_band_power(tfr_sub[mask_max], fmin, fmax_effective, tmin, tmax)
        temp_data_group_b = extract_trial_band_power(tfr_sub[mask_min], fmin, fmax_effective, tmin, tmax)

        _plot_diff_topomap_with_label(
            axes[0, c], cond_diff_data, tfr_2.info, cond_diff_abs,
            cond_sig_mask, cond_cluster_p_min, cond_cluster_k, cond_cluster_mass, config, viz_params,
            data_group_a=cond_data_group_a,
            data_group_b=cond_data_group_b,
            paired=False
        )

        _plot_diff_topomap_with_label(
            axes[1, c], temp_diff_data, tfr_max.info, temp_diff_abs,
            temp_sig_mask, temp_cluster_p_min, temp_cluster_k, temp_cluster_mass, config, viz_params,
            data_group_a=temp_data_group_a,
            data_group_b=temp_data_group_b,
            paired=False
        )

        axes[0, c].set_title(f"{band} ({fmin:.0f}-{fmax_effective:.0f} Hz)", fontsize=9, pad=4, y=1.04)

    add_diff_colorbar(fig, axes[0, :].ravel().tolist(), cond_diff_abs, viz_params["topo_cmap"], config)
    add_diff_colorbar(fig, axes[1, :].ravel().tolist(), temp_diff_abs, viz_params["topo_cmap"], config)

    font_sizes = get_font_sizes()
    axes[0, 0].set_ylabel(f"{label2} - {label1} (n={n_2}-{n_1})", fontsize=font_sizes["ylabel"])
    axes[1, 0].set_ylabel(f"Max - Min temp ({t_max:.1f}-{t_min:.1f}°C, n={n_max}-{n_min})", fontsize=font_sizes["ylabel"])
    sig_text = get_sig_marker_text(config)
    fig.suptitle(f"Topomaps (baseline: logratio; t=[{tmin:.1f}, {tmax:.1f}] s){sig_text}", fontsize=font_sizes["figure_title"])
    fig.supxlabel("Frequency bands", fontsize=font_sizes["ylabel"])
    _save_fig(fig, out_dir, "topomap_grid_bands_pain_temp_contrasts_bl.png", config=config, logger=logger, baseline_used=baseline_used)


def contrast_pain_nonpain(
    tfr: "mne.time_frequency.EpochsTFR | mne.time_frequency.AverageTFR",
    events_df: Optional[pd.DataFrame],
    out_dir: Path,
    config,
    baseline: Tuple[Optional[float], Optional[float]],
    active_window: Tuple[float, float],
    logger: Optional[logging.Logger] = None,
    subject: Optional[str] = None,
) -> None:
    """Plot pain vs non-pain contrast with central channel and topomaps.
    
    Creates a comprehensive contrast visualization including:
    - Central channel TFR plots (pain, non-pain, difference)
    - Topomap grid showing pain, non-pain, and difference across frequency bands
    
    Args:
        tfr: MNE TFR object (EpochsTFR or AverageTFR)
        events_df: Optional events DataFrame with pain column
        out_dir: Output directory path
        config: Configuration object
        baseline: Baseline window tuple (tmin, tmax)
        active_window: Active window tuple for statistics
        logger: Optional logger instance
        subject: Optional subject identifier
    """
    from .channels import _pick_central_channel, _plot_single_tfr_figure, _compute_active_statistics

    tfr_sub, mask1, mask2, label1, label2, _ = _prepare_comparison_contrast_data(
        tfr, events_df, config, logger, context="Condition contrast"
    )
    if tfr_sub is None:
        return

    tfr_sub_stats = tfr_sub.copy()
    baseline_used = apply_baseline_and_crop(tfr_sub_stats, baseline=baseline, mode="logratio", logger=logger)

    tfr_1 = tfr_sub[mask1].average()
    tfr_2 = tfr_sub[mask2].average()

    apply_baseline_and_crop(tfr_2, baseline=baseline_used, mode="logratio", logger=logger)
    apply_baseline_and_crop(tfr_1, baseline=baseline_used, mode="logratio", logger=logger)

    central_ch = _pick_central_channel(tfr_2.info, preferred="Cz", logger=logger)
    
    ch_idx = tfr_2.info["ch_names"].index(central_ch)
    arr_condition_2 = np.asarray(tfr_2.data[ch_idx])
    arr_condition_1 = np.asarray(tfr_1.data[ch_idx])
    vabs_symmetric = robust_sym_vlim([arr_condition_2, arr_condition_1])
    
    times = np.asarray(tfr_2.times)
    _, pct_condition_2, _ = _compute_active_statistics(arr_condition_2, times, active_window, config, logger)
    _, pct_condition_1, _ = _compute_active_statistics(arr_condition_1, times, active_window, config, logger)
    
    _plot_single_tfr_figure(
        tfr_2, central_ch, (-vabs_symmetric, +vabs_symmetric),
        f"{central_ch} — {label2} (baseline logratio)\nvlim ±{vabs_symmetric:.2f}; mean %Δ vs BL={pct_condition_2:+.0f}%",
        f"tfr_{central_ch}_pain_bl.png", out_dir, config, logger, baseline_used
    )
    
    _plot_single_tfr_figure(
        tfr_1, central_ch, (-vabs_symmetric, +vabs_symmetric),
        f"{central_ch} — {label1} (baseline logratio)\nvlim ±{vabs_symmetric:.2f}; mean %Δ vs BL={pct_condition_1:+.0f}%",
        f"tfr_{central_ch}_nonpain_bl.png", out_dir, config, logger, baseline_used
    )
    
    tfr_diff = tfr_2.copy()
    tfr_diff.data = tfr_2.data - tfr_1.data
    tfr_diff.comment = "cond2-minus-cond1"
    
    arr_diff = np.asarray(arr_condition_2) - np.asarray(arr_condition_1)
    vabs_diff = robust_sym_vlim(arr_diff)
    _, pct_diff, _ = _compute_active_statistics(arr_diff, times, active_window, config, logger)
    
    _plot_single_tfr_figure(
        tfr_diff, central_ch, (-vabs_diff, +vabs_diff),
        f"{central_ch} — {label2} minus {label1} (baseline logratio)\nvlim ±{vabs_diff:.2f}; Δ% vs BL={pct_diff:+.0f}%",
        f"tfr_{central_ch}_pain_minus_non_bl.png", out_dir, config, logger, baseline_used
    )

    times = np.asarray(tfr_2.times)
    tmin_effective = float(max(np.min(times), active_window[0]))
    tmax_effective = float(min(np.max(times), active_window[1]))
    fmax_available = float(np.max(tfr_2.freqs))
    bands = get_bands_for_tfr(max_freq_available=fmax_available, config=config)
    tmin, tmax = tmin_effective, tmax_effective

    n_condition_2 = int(mask2.sum())
    n_condition_1 = int(mask1.sum())
    row_labels = [f"{label2} (n={n_condition_2})", f"{label1} (n={n_condition_1})", f"{label2} - {label1}"]
    n_cols = len(bands)
    n_rows = 3
    plot_cfg = get_plot_config(config)
    tfr_config = plot_cfg.plot_type_configs.get("tfr", {})
    topomap_config = tfr_config.get("topomap", {})
    fig_size_per_col = plot_cfg.get_figure_size("tfr_per_col_large", plot_type="tfr")[0]
    fig_size_per_row = plot_cfg.get_figure_size("tfr_per_row_large", plot_type="tfr")[1]
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_size_per_col * n_cols, fig_size_per_row * n_rows),
        squeeze=False,
        gridspec_kw={"wspace": 1.2, "hspace": 0.25},
    )
    viz_params = get_viz_params(config)
    
    def _turn_off_column_axes(axes, column_idx, n_rows):
        for r in range(n_rows):
            axes[r, column_idx].axis('off')
    
    def _compute_diff_vlims(diff_data, diff_abs):
        """Compute vmin and vmax for difference plot.
        
        Args:
            diff_data: Difference data array
            diff_abs: Absolute value for symmetric range
            
        Returns:
            Tuple of (vmin, vmax) or (None, None) if no valid data
        """
        if diff_abs > 0:
            return -diff_abs, +diff_abs
        
        if diff_data is not None:
            diff_data_arr = np.asarray(diff_data)
            diff_finite = diff_data_arr[np.isfinite(diff_data_arr)]
            has_nonzero = diff_finite.size > 0 and np.any(np.abs(diff_finite) > 0)
        else:
            has_nonzero = False
        
        if has_nonzero:
            diff_abs_effective = robust_sym_vlim(diff_data)
            return -diff_abs_effective, +diff_abs_effective
        
        return None, None
    
    def _plot_pain_nonpain_topomaps(
        axes, column_idx, condition_2_data, condition_1_data, diff_data, info,
        vabs_symmetric, diff_abs, sig_mask, cluster_info, config,
        data_group_a, data_group_b
    ):
        condition_2_mean = float(np.nanmean(condition_2_data))
        condition_2_pct = logratio_to_pct(condition_2_mean)
        _plot_topomap_with_label(
            axes[0, column_idx], condition_2_data, info, -vabs_symmetric, +vabs_symmetric,
            f"%Δ={condition_2_pct:+.1f}%", config
        )

        condition_1_mean = float(np.nanmean(condition_1_data))
        condition_1_pct = logratio_to_pct(condition_1_mean)
        _plot_topomap_with_label(
            axes[1, column_idx], condition_1_data, info, -vabs_symmetric, +vabs_symmetric,
            f"%Δ={condition_1_pct:+.1f}%", config
        )

        diff_vmin, diff_vmax = _compute_diff_vlims(diff_data, diff_abs)
        _plot_topomap_with_diff_label(
            axes[2, column_idx], diff_data, info, diff_vmin, diff_vmax,
            sig_mask, cluster_info, config,
            data_group_a=data_group_a,
            data_group_b=data_group_b,
            paired=False
        )
    
    def _add_colorbars_for_column(fig, axes, column_idx, vabs_symmetric, diff_abs, viz_params, config=None):
        from ..core.colorbars import add_normalized_colorbar, add_diff_colorbar
        add_normalized_colorbar(fig, [axes[0, column_idx], axes[1, column_idx]], -vabs_symmetric, +vabs_symmetric, viz_params["topo_cmap"], config)
        if diff_abs > 0:
            add_diff_colorbar(fig, axes[2, column_idx], diff_abs, viz_params["topo_cmap"], config)
    
    def _finalize_topomap_figure(fig, axes, row_labels, tmin, tmax, config):
        font_sizes = get_font_sizes()
        axes[0, 0].set_ylabel(row_labels[0], fontsize=font_sizes["ylabel"])
        axes[1, 0].set_ylabel(row_labels[1], fontsize=font_sizes["ylabel"])
        axes[2, 0].set_ylabel(row_labels[2], fontsize=font_sizes["ylabel"])
        from ..core.annotations import get_sig_marker_text
        sig_text = get_sig_marker_text(config)
        fig.suptitle(
            f"Topomaps (baseline: logratio; t=[{tmin:.1f}, {tmax:.1f}] s){sig_text}",
            fontsize=font_sizes["figure_title"]
        )
        fig.supxlabel("Frequency bands", fontsize=font_sizes["ylabel"])
        plot_cfg = get_plot_config(config)
        tfr_config = plot_cfg.plot_type_configs.get("tfr", {})
        topomap_config = tfr_config.get("topomap", {})
        subplots_right = topomap_config.get("subplots_right", 0.75)
        fig.subplots_adjust(right=subplots_right)
    
    for c, (band, (fmin, fmax)) in enumerate(bands.items()):
        fmax_effective = min(fmax, fmax_available)
        if fmin >= fmax_effective:
            _turn_off_column_axes(axes, c, n_rows)
            continue

        data_condition_2 = average_tfr_band(tfr_2, fmin=fmin, fmax=fmax_effective, tmin=tmin, tmax=tmax)
        data_condition_1 = average_tfr_band(tfr_1, fmin=fmin, fmax=fmax_effective, tmin=tmin, tmax=tmax)
        if data_condition_2 is None or data_condition_1 is None:
            _turn_off_column_axes(axes, c, n_rows)
            continue

        diff_data = data_condition_2 - data_condition_1
        vabs_symmetric = robust_sym_vlim([data_condition_2, data_condition_1])
        default_diff_abs = topomap_config.get("default_diff_abs", 0.0)
        diff_abs = robust_sym_vlim(diff_data) if np.isfinite(diff_data).any() else default_diff_abs

        sig_mask, cluster_p_min, cluster_k, cluster_mass = compute_cluster_significance(
            tfr_sub_stats, mask2, mask1, fmin, fmax_effective, tmin, tmax, config, diff_data_len=len(diff_data), logger=logger
        )
        cluster_info_dict = {
            "cluster_p_min": cluster_p_min,
            "cluster_k": cluster_k,
            "cluster_mass": cluster_mass
        }

        data_group_a = extract_trial_band_power(tfr_sub_stats[mask2], fmin, fmax_effective, tmin, tmax)
        data_group_b = extract_trial_band_power(tfr_sub_stats[mask1], fmin, fmax_effective, tmin, tmax)

        _plot_pain_nonpain_topomaps(
            axes, c, data_condition_2, data_condition_1, diff_data, tfr_2.info,
            vabs_symmetric, diff_abs, sig_mask, cluster_info_dict, config,
            data_group_a, data_group_b
        )

        font_sizes = get_font_sizes()
        axes[0, c].set_title(
            f"{band} ({fmin:.0f}-{fmax_effective:.0f} Hz)",
            fontsize=font_sizes["title"], pad=topomap_config.get("title_pad", 4), y=topomap_config.get("title_y", 1.04)
        )

        if c == len(bands) - 1:
            _add_colorbars_for_column(
                fig, axes, c, vabs_symmetric, diff_abs, viz_params, config
            )
    
    _finalize_topomap_figure(
        fig, axes, row_labels, tmin, tmax, config
    )
    _save_fig(fig, out_dir, "topomap_grid_bands_pain_non_diff_bl.png", config=config, logger=logger, baseline_used=baseline_used)
