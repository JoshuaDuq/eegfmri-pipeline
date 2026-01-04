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
    extract_eeg_picks,
    logratio_to_pct,
    get_viz_params,
    plot_topomap_on_ax,
)
from eeg_pipeline.utils.data.columns import get_pain_column_from_config, get_temperature_column_from_config
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
    extract_pain_vector,
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
        pain_vec: Pain vector (pandas Series or numpy array)
        
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
        pain_col: Pain column name
        
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
    tfr_config = plot_cfg.plot_type_configs.get("tfr", {}) if plot_cfg else {}
    label_x_position = tfr_config.get("label_x_position", 0.5) if plot_cfg else 0.5
    label_y_position = tfr_config.get("label_y_position", 1.02) if plot_cfg else 1.02
    plot_topomap_on_ax(ax, data, info, vmin=vmin, vmax=vmax, config=config)
    add_roi_annotations(ax, data, info, config=config)
    ax.text(label_x_position, label_y_position, label_text, transform=ax.transAxes, ha="center", va="top", fontsize=font_sizes["title"])


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
    plot_cfg = get_plot_config(config) if config else None
    tfr_config = plot_cfg.plot_type_configs.get("tfr", {}) if plot_cfg else {}
    label_x_position = tfr_config.get("label_x_position", 0.5) if plot_cfg else 0.5
    label_y_position = tfr_config.get("label_y_position", 1.02) if plot_cfg else 1.02
    ax.text(label_x_position, label_y_position, label, transform=ax.transAxes, ha="center", va="top", fontsize=font_sizes["title"])


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
    plot_cfg = get_plot_config(config) if config else None
    tfr_config = plot_cfg.plot_type_configs.get("tfr", {}) if plot_cfg else {}
    label_x_position = tfr_config.get("label_x_position", 0.5) if plot_cfg else 0.5
    label_y_position = tfr_config.get("label_y_position", 1.02) if plot_cfg else 1.02
    ax.text(label_x_position, label_y_position, label, transform=ax.transAxes, ha="center", va="top", fontsize=font_sizes["title"])


def _compute_band_diff_data(
    tfr_pain,
    tfr_non,
    tfr_max,
    tfr_min,
    fmin: float,
    fmax_eff: float,
    tmin: float,
    tmax: float,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Compute band difference data for pain and temperature contrasts.
    
    Args:
        tfr_pain: Pain condition TFR
        tfr_non: Non-pain condition TFR
        tfr_max: Max temperature TFR
        tfr_min: Min temperature TFR
        fmin: Minimum frequency
        fmax_eff: Effective maximum frequency
        tmin: Minimum time
        tmax: Maximum time
        
    Returns:
        Tuple of (pain_diff_data, temp_diff_data) or (None, None) on failure
    """
    pain_data = average_tfr_band(tfr_pain, fmin=fmin, fmax=fmax_eff, tmin=tmin, tmax=tmax)
    non_data = average_tfr_band(tfr_non, fmin=fmin, fmax=fmax_eff, tmin=tmin, tmax=tmax)
    max_data = average_tfr_band(tfr_max, fmin=fmin, fmax=fmax_eff, tmin=tmin, tmax=tmax)
    min_data = average_tfr_band(tfr_min, fmin=fmin, fmax=fmax_eff, tmin=tmin, tmax=tmax)
    
    if pain_data is None or non_data is None or max_data is None or min_data is None:
        return None, None
    
    pain_diff_data = pain_data - non_data
    temp_diff_data = max_data - min_data
    return pain_diff_data, temp_diff_data


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
                trimmed[key] = [m[:tfr_len] if len(m) != tfr_len else m for m in masks]
            else:
                trimmed[key] = masks[:tfr_len] if len(masks) != tfr_len else masks
        
        return trimmed
    except ValueError as e:
        log(f"{e}. Skipping alignment.", logger, "error")
        return None




def _compute_cluster_significance_from_combined(
    tfr1_list: List["mne.time_frequency.EpochsTFR"],
    tfr2_list: List["mne.time_frequency.EpochsTFR"],
    fmin: float,
    fmax_eff: float,
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
        fmax_eff: Effective maximum frequency
        tmin: Minimum time
        tmax: Maximum time
        config: Configuration object
        diff_data_len: Expected length of difference data (for validation)
        logger: Optional logger instance
        
    Returns:
        Tuple of (sig_mask, cluster_p_min, cluster_k, cluster_mass)
    """
    from ...utils.analysis.stats import cluster_test_two_sample_arrays as _cluster_test_two_sample_arrays
    
    if not tfr1_list or not tfr2_list:
        return None, None, None, None
    
    min_subjects = 2
    if len(tfr1_list) < min_subjects or len(tfr2_list) < min_subjects:
        log("Cluster test requires at least 2 subjects per condition; skipping", logger, "warning")
        return None, None, None, None
    
    subject_a_means = []
    subject_b_means = []
    tfr_a_list_valid = []
    tfr_b_list_valid = []
    reference_info = None
    
    for tfr_a, tfr_b in zip(tfr1_list, tfr2_list):
        if not isinstance(tfr_a, mne.time_frequency.EpochsTFR) or not isinstance(tfr_b, mne.time_frequency.EpochsTFR):
            continue
        
        data_a = extract_trial_band_power(tfr_a, fmin, fmax_eff, tmin, tmax)
        data_b = extract_trial_band_power(tfr_b, fmin, fmax_eff, tmin, tmax)
        
        if data_a is None or data_b is None:
            continue
        
        if data_a.shape[0] == 0 or data_b.shape[0] == 0:
            continue
        
        mean_a = np.nanmean(data_a, axis=0)
        mean_b = np.nanmean(data_b, axis=0)
        
        if mean_a.ndim == 0:
            mean_a = mean_a.reshape(1)
        if mean_b.ndim == 0:
            mean_b = mean_b.reshape(1)
        
        if reference_info is None:
            reference_info = tfr_a.info
        
        subject_a_means.append(mean_a)
        subject_b_means.append(mean_b)
        tfr_a_list_valid.append(tfr_a)
        tfr_b_list_valid.append(tfr_b)
    
    if len(subject_a_means) < min_subjects:
        log("Insufficient subjects with valid data for cluster test; skipping", logger, "warning")
        return None, None, None, None
    
    ch_sets = [set(tfr.info["ch_names"]) for tfr in tfr_a_list_valid]
    ch_sets.extend([set(tfr.info["ch_names"]) for tfr in tfr_b_list_valid])
    common_chs = list(sorted(set.intersection(*ch_sets))) if ch_sets else []
    
    if len(common_chs) == 0:
        log("No common channels across subjects for cluster test; skipping", logger, "warning")
        return None, None, None, None
    
    group_a_array = []
    group_b_array = []
    
    for mean_a, mean_b, tfr_a, tfr_b in zip(subject_a_means, subject_b_means, tfr_a_list_valid, tfr_b_list_valid):
        ch_indices_a = [tfr_a.info["ch_names"].index(ch) for ch in common_chs if ch in tfr_a.info["ch_names"]]
        ch_indices_b = [tfr_b.info["ch_names"].index(ch) for ch in common_chs if ch in tfr_b.info["ch_names"]]
        
        if len(ch_indices_a) != len(common_chs) or len(ch_indices_b) != len(common_chs):
            continue
        
        group_a_array.append(mean_a[ch_indices_a])
        group_b_array.append(mean_b[ch_indices_b])
    
    if len(group_a_array) < min_subjects or len(group_b_array) < min_subjects:
        log("Insufficient subjects after channel alignment for cluster test; skipping", logger, "warning")
        return None, None, None, None
    
    group_a_subjects = np.stack(group_a_array, axis=0)
    group_b_subjects = np.stack(group_b_array, axis=0)
    
    info_common = mne.pick_info(reference_info, [reference_info["ch_names"].index(ch) for ch in common_chs])
    
    sig_mask_full, cluster_p_min, cluster_k, cluster_mass = _cluster_test_two_sample_arrays(
        group_a_subjects, group_b_subjects, info_common,
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
    tmin_req, tmax_req = active_window
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
        max_data = average_tfr_band(tfr_max, fmin=fmin, fmax=fmax_eff, tmin=tmin, tmax=tmax)
        min_data = average_tfr_band(tfr_min, fmin=fmin, fmax=fmax_eff, tmin=tmin, tmax=tmax)
        if max_data is None or min_data is None:
            for c in range(n_cols):
                axes[r, c].axis("off")
            continue

        sig_mask = None
        cluster_p_min = cluster_k = cluster_mass = None
        if get_viz_params(config)["diff_annotation_enabled"]:
            sig_mask, cluster_p_min, cluster_k, cluster_mass = cluster_test_epochs(
                tfr_sub_stats, mask_max, mask_min, fmin=fmin, fmax=fmax_eff, tmin=tmin, tmax=tmax, paired=False, config=config
            )

        diff_data = max_data - min_data
        max_mu = float(np.nanmean(max_data))
        min_mu = float(np.nanmean(min_data))
        diff_mu = float(np.nanmean(diff_data))
        vabs_pn = robust_sym_vlim([max_data, min_data])
        diff_abs = robust_sym_vlim(diff_data) if np.isfinite(diff_data).any() else 0.0

        _plot_topomap_with_percentage_label(
            axes[r, 0], max_data, tfr_max.info, -vabs_pn, +vabs_pn, config
        )

        _plot_topomap_with_percentage_label(
            axes[r, 1], min_data, tfr_min.info, -vabs_pn, +vabs_pn, config
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
        axes[r, 0].set_ylabel(f"{band} ({fmin:.0f}-{fmax_eff:.0f} Hz)", fontsize=font_sizes["ylabel"])

        viz_params = get_viz_params(config)
        add_normalized_colorbar(fig, [axes[r, 0], axes[r, 1]], -vabs_pn, +vabs_pn, viz_params["topo_cmap"], config)
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
    if events_df is None:
        log("Combined contrast requires events_df; skipping.", logger, "warning")
        return
    
    pain_col = get_pain_column_from_config(config, events_df)
    temp_col = get_temperature_column_from_config(config, events_df)
    
    if pain_col is None:
        log("Combined contrast: no pain column found; skipping.", logger, "warning")
        return
    if temp_col is None:
        log("Combined contrast: no temperature column found; skipping.", logger, "warning")
        return

    n = compute_aligned_data_length(tfr, events_df)

    pain_vec = extract_pain_vector(tfr, events_df, pain_col, n)
    if pain_vec is None:
        log("Combined contrast: could not extract pain vector; skipping.", logger, "warning")
        return
    
    pain_mask, non_mask = _create_pain_masks_from_vector(pain_vec)
    if pain_mask.sum() == 0 or non_mask.sum() == 0:
        log("Combined contrast: one pain group has zero trials; skipping.", logger, "warning")
        return
    
    temp_series = extract_temperature_series(tfr, events_df, temp_col, n)
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

    tfr_pain = tfr_sub[pain_mask].average()
    tfr_non = tfr_sub[non_mask].average()
    tfr_min = tfr_sub[mask_min].average()
    tfr_max = tfr_sub[mask_max].average()

    baseline_used = apply_baseline_and_crop(tfr_pain, baseline=baseline, mode="logratio", logger=logger)
    apply_baseline_and_crop(tfr_non, baseline=baseline, mode="logratio", logger=logger)
    apply_baseline_and_crop(tfr_min, baseline=baseline, mode="logratio", logger=logger)
    apply_baseline_and_crop(tfr_max, baseline=baseline, mode="logratio", logger=logger)

    times = np.asarray(tfr_pain.times)
    tmin_req, tmax_req = active_window
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
            pain_sig_mask, pain_cluster_p_min, pain_cluster_k, pain_cluster_mass = compute_cluster_significance(
                tfr_sub, pain_mask, non_mask, fmin, fmax_eff, tmin, tmax, config, diff_data_len=pain_diff_data_len, logger=logger
            )

        temp_sig_mask = None
        temp_cluster_p_min = temp_cluster_k = temp_cluster_mass = None
        if viz_params["diff_annotation_enabled"]:
            temp_diff_data_len = len(temp_diff_data) if temp_diff_data is not None else None
            temp_sig_mask, temp_cluster_p_min, temp_cluster_k, temp_cluster_mass = compute_cluster_significance(
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

    add_diff_colorbar(fig, axes[0, :].ravel().tolist(), pain_diff_abs, viz_params["topo_cmap"], config)
    add_diff_colorbar(fig, axes[1, :].ravel().tolist(), temp_diff_abs, viz_params["topo_cmap"], config)

    font_sizes = get_font_sizes()
    axes[0, 0].set_ylabel(f"Pain - Non (n={n_pain}-{n_non})", fontsize=font_sizes["ylabel"])
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
    from .scalpmean import _prepare_pain_contrast_data
    from .channels import _pick_central_channel, _plot_single_tfr_figure, _compute_active_statistics
    from .topomaps import _add_roi_annotations
    
    pain_col = get_pain_column_from_config(config, events_df)
    tfr_sub, pain_mask, non_mask, n = _prepare_pain_contrast_data(tfr, events_df, pain_col, config, logger)
    if tfr_sub is None:
        return

    tfr_sub_stats = tfr_sub.copy()
    baseline_used = apply_baseline_and_crop(tfr_sub_stats, baseline=baseline, mode="logratio", logger=logger)

    tfr_pain = tfr_sub[pain_mask].average()
    tfr_non = tfr_sub[non_mask].average()

    apply_baseline_and_crop(tfr_pain, baseline=baseline_used, mode="logratio", logger=logger)
    apply_baseline_and_crop(tfr_non, baseline=baseline_used, mode="logratio", logger=logger)

    central_ch = _pick_central_channel(tfr_pain.info, preferred="Cz", logger=logger)
    
    ch_idx = tfr_pain.info["ch_names"].index(central_ch)
    arr_pain = np.asarray(tfr_pain.data[ch_idx])
    arr_non = np.asarray(tfr_non.data[ch_idx])
    vabs_pn = robust_sym_vlim([arr_pain, arr_non])
    
    times = np.asarray(tfr_pain.times)
    _, pct_pain, _ = _compute_active_statistics(arr_pain, times, active_window, config, logger)
    _, pct_non, _ = _compute_active_statistics(arr_non, times, active_window, config, logger)
    
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
    _, pct_diff, _ = _compute_active_statistics(arr_diff, times, active_window, config, logger)
    
    _plot_single_tfr_figure(
        tfr_diff, central_ch, (-vabs_diff, +vabs_diff),
        f"{central_ch} — Pain minus Non (baseline logratio)\nvlim ±{vabs_diff:.2f}; Δ% vs BL={pct_diff:+.0f}%",
        f"tfr_{central_ch}_pain_minus_non_bl.png", out_dir, config, logger, baseline_used
    )

    times = np.asarray(tfr_pain.times)
    tmin_eff = float(max(np.min(times), active_window[0]))
    tmax_eff = float(min(np.max(times), active_window[1]))
    fmax_available = float(np.max(tfr_pain.freqs))
    bands = get_bands_for_tfr(max_freq_available=fmax_available, config=config)
    tmin, tmax = tmin_eff, tmax_eff

    n_pain = int(pain_mask.sum())
    n_non = int(non_mask.sum())
    row_labels = [f"Pain (n={n_pain})", f"Non-pain (n={n_non})", "Pain - Non"]
    n_cols = len(bands)
    topo_n_rows = 3
    plot_cfg_contrast = get_plot_config(config)
    tfr_config_contrast = plot_cfg_contrast.plot_type_configs.get("tfr", {})
    topomap_config_contrast = tfr_config_contrast.get("topomap", {})
    fig_size_per_col_large = plot_cfg_contrast.get_figure_size("tfr_per_col_large", plot_type="tfr")[0]
    fig_size_per_row_large = plot_cfg_contrast.get_figure_size("tfr_per_row_large", plot_type="tfr")[1]
    fig, axes = plt.subplots(
        topo_n_rows,
        n_cols,
        figsize=(fig_size_per_col_large * n_cols, fig_size_per_row_large * topo_n_rows),
        squeeze=False,
        gridspec_kw={"wspace": 1.2, "hspace": 0.25},
    )
    viz_params = get_viz_params(config)
    
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
        from ..core.colorbars import add_normalized_colorbar, add_diff_colorbar
        add_normalized_colorbar(fig, [axes[0, column_idx], axes[1, column_idx]], -vabs_pn, +vabs_pn, viz_params["topo_cmap"], config)
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
        plot_cfg_finalize = get_plot_config(config)
        tfr_config_finalize = plot_cfg_finalize.plot_type_configs.get("tfr", {})
        topomap_config_finalize = tfr_config_finalize.get("topomap", {})
        topo_subplots_right = topomap_config_finalize.get("subplots_right", 0.75)
        fig.subplots_adjust(right=topo_subplots_right)
    
    for c, (band, (fmin, fmax)) in enumerate(bands.items()):
        fmax_eff = min(fmax, fmax_available)
        if fmin >= fmax_eff:
            _turn_off_column_axes(axes, c, topo_n_rows)
            continue

        pain_data = average_tfr_band(tfr_pain, fmin=fmin, fmax=fmax_eff, tmin=tmin, tmax=tmax)
        non_data = average_tfr_band(tfr_non, fmin=fmin, fmax=fmax_eff, tmin=tmin, tmax=tmax)
        if pain_data is None or non_data is None:
            _turn_off_column_axes(axes, c, topo_n_rows)
            continue

        diff_data = pain_data - non_data
        vabs_pn = robust_sym_vlim([pain_data, non_data])
        topo_default_diff_abs = topomap_config_contrast.get("default_diff_abs", 0.0)
        diff_abs = robust_sym_vlim(diff_data) if np.isfinite(diff_data).any() else topo_default_diff_abs

        sig_mask, cluster_p_min, cluster_k, cluster_mass = compute_cluster_significance(
            tfr_sub_stats, pain_mask, non_mask, fmin, fmax_eff, tmin, tmax, config, diff_data_len=len(diff_data), logger=logger
        )
        cluster_info_dict = {
            "cluster_p_min": cluster_p_min,
            "cluster_k": cluster_k,
            "cluster_mass": cluster_mass
        }

        data_group_a = extract_trial_band_power(tfr_sub_stats[pain_mask], fmin, fmax_eff, tmin, tmax)
        data_group_b = extract_trial_band_power(tfr_sub_stats[non_mask], fmin, fmax_eff, tmin, tmax)

        _plot_pain_nonpain_topomaps(
            axes, c, pain_data, non_data, diff_data, tfr_pain.info,
            vabs_pn, diff_abs, sig_mask, cluster_info_dict, config,
            data_group_a, data_group_b
        )

        font_sizes = get_font_sizes()
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
    _save_fig(fig, out_dir, "topomap_grid_bands_pain_non_diff_bl.png", config=config, logger=logger, baseline_used=baseline_used)
