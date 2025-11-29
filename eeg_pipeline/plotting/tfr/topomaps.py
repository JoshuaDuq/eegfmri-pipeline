"""
TFR-specific topomap plotting functions.

Functions for creating topomap visualizations for time-frequency representations,
including temperature grids and temporal topomaps.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt

from ...utils.io.general import (
    robust_sym_vlim,
    extract_eeg_picks,
    get_pain_column_from_config,
    get_temperature_column_from_config,
    require_epochs_tfr,
    ensure_aligned_lengths,
    get_viz_params,
    plot_topomap_on_ax,
    detect_data_format,
)
from ...utils.analysis.tfr import (
    apply_baseline_and_crop,
    create_tfr_subset,
    get_bands_for_tfr,
    average_tfr_band,
    extract_trial_band_power,
    clip_time_range,
    create_time_windows_fixed_size,
    create_time_windows_fixed_count,
)
from ...utils.data.loading import (
    compute_aligned_data_length,
    extract_pain_vector_array,
    extract_temperature_series,
    create_temperature_masks,
)
from ..config import get_plot_config
from ..core.utils import get_font_sizes, log
from ..core.statistics import get_strict_mode, compute_cluster_significance, build_statistical_title
from ..core.colorbars import create_difference_colorbar
from ..core.topomaps import build_topomap_diff_label, build_topomap_percentage_label
from ..core.annotations import (
    add_roi_annotations,
    apply_fdr_correction_to_roi_pvalues,
    build_significance_info,
    render_roi_annotations,
)
from ..core.annotations import get_sig_marker_text
from .contrasts import (
    _get_baseline_window,
    _create_pain_masks_from_vector,
    _align_and_trim_masks,
)
from .channels import _save_fig as _channels_save_fig


###################################################################
# Helper Functions
###################################################################


def _compute_band_diff_data_windows(
    tfr_pain,
    tfr_non,
    tfr_max,
    tfr_min,
    fmin: float,
    fmax_eff: float,
    window_starts: np.ndarray,
    window_ends: np.ndarray,
    has_temp: bool,
) -> Tuple[List[Optional[np.ndarray]], List[Optional[np.ndarray]]]:
    """Compute band difference data for multiple time windows.
    
    Args:
        tfr_pain: Pain condition TFR
        tfr_non: Non-pain condition TFR
        tfr_max: Max temperature TFR (optional)
        tfr_min: Min temperature TFR (optional)
        fmin: Minimum frequency
        fmax_eff: Effective maximum frequency
        window_starts: Array of window start times
        window_ends: Array of window end times
        has_temp: Whether temperature data is available
        
    Returns:
        Tuple of (pain_diff_data_windows, temp_diff_data_windows)
    """
    pain_diff_data_windows = []
    temp_diff_data_windows = []
    
    for tmin_win, tmax_win in zip(window_starts, window_ends):
        pain_data = average_tfr_band(tfr_pain, fmin=fmin, fmax=fmax_eff, tmin=tmin_win, tmax=tmax_win)
        non_data = average_tfr_band(tfr_non, fmin=fmin, fmax=fmax_eff, tmin=tmin_win, tmax=tmax_win)
        
        if pain_data is not None and non_data is not None:
            pain_diff_data_windows.append(pain_data - non_data)
        else:
            pain_diff_data_windows.append(None)
        
        if has_temp and tfr_max is not None and tfr_min is not None:
            max_data = average_tfr_band(tfr_max, fmin=fmin, fmax=fmax_eff, tmin=tmin_win, tmax=tmax_win)
            min_data = average_tfr_band(tfr_min, fmin=fmin, fmax=fmax_eff, tmin=tmin_win, tmax=tmax_win)
            
            if max_data is not None and min_data is not None:
                temp_diff_data_windows.append(max_data - min_data)
            else:
                temp_diff_data_windows.append(None)
        else:
            temp_diff_data_windows.append(None)
    
    return pain_diff_data_windows, temp_diff_data_windows


def _plot_single_topomap_window(
    ax: plt.Axes,
    diff_data: np.ndarray,
    info: mne.Info,
    tfr_sub,
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    fmin: float,
    fmax_eff: float,
    tmin_win: float,
    tmax_win: float,
    vabs_diff: float,
    config,
    viz_params: Dict,
    paired: bool = False,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Plot a single topomap window with optional statistical overlay.
    
    Args:
        ax: Matplotlib axes
        diff_data: Difference data array (n_channels,)
        info: MNE Info object
        tfr_sub: Trial-level TFR object for statistics
        mask_a: Boolean mask for condition A
        mask_b: Boolean mask for condition B
        fmin: Minimum frequency
        fmax_eff: Effective maximum frequency
        tmin_win: Window start time
        tmax_win: Window end time
        vabs_diff: Absolute value for symmetric range
        config: Configuration object
        viz_params: Visualization parameters dictionary
        paired: Whether paired test was used
        logger: Optional logger instance
    """
    if diff_data is None:
        ax.axis('off')
        return
    
    sig_mask = cluster_p_min = cluster_k = cluster_mass = None
    if viz_params["diff_annotation_enabled"] and tfr_sub is not None:
        diff_data_len = len(diff_data) if diff_data is not None else None
        sig_mask, cluster_p_min, cluster_k, cluster_mass = compute_cluster_significance(
            tfr_sub, mask_a, mask_b, fmin, fmax_eff, tmin_win, tmax_win, config, diff_data_len=diff_data_len, logger=logger
        )
    
    plot_topomap_on_ax(
        ax, diff_data, info,
        vmin=-vabs_diff, vmax=+vabs_diff,
        mask=(sig_mask if viz_params["diff_annotation_enabled"] else None),
        mask_params=viz_params["sig_mask_params"],
        config=config
    )
    
    data_group_a = None
    data_group_b = None
    if tfr_sub is not None:
        data_group_a = extract_trial_band_power(tfr_sub[mask_a], fmin, fmax_eff, tmin_win, tmax_win)
        data_group_b = extract_trial_band_power(tfr_sub[mask_b], fmin, fmax_eff, tmin_win, tmax_win)
    
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
    label_y_position_bottom = tfr_config.get("label_y_position_bottom", 1.08) if plot_cfg else 1.08
    ax.text(label_x_position, label_y_position_bottom, label, transform=ax.transAxes, ha="center", va="bottom", fontsize=font_sizes["label"])


def _plot_temporal_topomaps_for_bands(
    tfr_pain,
    tfr_non,
    tfr_sub,
    tfr_max,
    tfr_min,
    pain_mask: np.ndarray,
    non_mask: np.ndarray,
    mask_max: Optional[np.ndarray],
    mask_min: Optional[np.ndarray],
    window_starts: np.ndarray,
    window_ends: np.ndarray,
    has_temp: bool,
    t_min: Optional[float],
    t_max: Optional[float],
    tmin_clip: float,
    tmax_clip: float,
    n_windows: int,
    baseline_used: Tuple[float, float],
    window_label: str,
    filename_base: str,
    out_dir: Path,
    config,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Plot temporal topomaps for all frequency bands.
    
    Args:
        tfr_pain: Pain condition TFR
        tfr_non: Non-pain condition TFR
        tfr_sub: Trial-level TFR for statistics
        tfr_max: Max temperature TFR (optional)
        tfr_min: Min temperature TFR (optional)
        pain_mask: Boolean mask for pain condition
        non_mask: Boolean mask for non-pain condition
        mask_max: Boolean mask for max temperature (optional)
        mask_min: Boolean mask for min temperature (optional)
        window_starts: Array of window start times
        window_ends: Array of window end times
        has_temp: Whether temperature data is available
        t_min: Minimum temperature value (optional)
        t_max: Maximum temperature value (optional)
        tmin_clip: Clipped minimum time
        tmax_clip: Clipped maximum time
        n_windows: Number of time windows
        baseline_used: Baseline window tuple
        window_label: Label for time windows
        filename_base: Base filename template
        out_dir: Output directory path
        config: Configuration object
        logger: Optional logger instance
    """
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
        temp_diff_data_valid = [d for d in temp_diff_data_windows if d is not None] if has_temp else []

        if len(pain_diff_data_valid) == 0:
            log(f"No valid data found for {band_name} temporal topomaps; skipping this band.", logger, "warning")
            continue

        valid_bands[band_name] = (fmin, fmax_eff, pain_diff_data_windows, temp_diff_data_windows if has_temp else None)
        all_band_pain_diff_data[band_name] = pain_diff_data_valid
        if has_temp:
            all_band_temp_diff_data[band_name] = temp_diff_data_valid

    if len(valid_bands) == 0:
        log("No valid bands found for temporal topomaps; skipping.", logger, "warning")
        return

    all_pain_diff_data = [d for data_list in all_band_pain_diff_data.values() for d in data_list]
    all_temp_diff_data = [d for data_list in all_band_temp_diff_data.values() for d in data_list] if has_temp else []
    all_diff_data = all_pain_diff_data + all_temp_diff_data
    vabs_diff = robust_sym_vlim(all_diff_data) if len(all_diff_data) > 0 else 1e-6

    viz_params = get_viz_params(config)
    font_sizes = get_font_sizes()
    baseline_str = f"bl{abs(baseline_used[0]):.1f}to{abs(baseline_used[1]):.2f}" if baseline_used else "bl"
    sig_text = get_sig_marker_text(config)
    
    log(f"Creating separate figure for each of {len(valid_bands)} frequency bands...", logger)
    
    temporal_spacing = config.get("time_frequency_analysis.topomap.temporal.single_subject", {}) if config else {}
    hspace = temporal_spacing.get("hspace", 0.15)
    wspace = temporal_spacing.get("wspace", 0.8)
    
    plot_cfg = get_plot_config(config)
    fig_size_per_col_large = plot_cfg.get_figure_size("tfr_per_col_large", plot_type="tfr")[0]
    fig_size_per_row_large = plot_cfg.get_figure_size("tfr_per_row_large", plot_type="tfr")[1]
    
    for band_name, (fmin, fmax_eff, pain_diff_data_windows, temp_diff_data_windows) in valid_bands.items():
        freq_label = f"{band_name} ({fmin:.0f}-{fmax_eff:.0f}Hz)"
        n_rows = 2 if (has_temp and temp_diff_data_windows is not None) else 1
        
        fig, axes = plt.subplots(
            n_rows, n_windows, 
            figsize=(fig_size_per_col_large * n_windows, fig_size_per_row_large * n_rows), 
            squeeze=False,
            gridspec_kw={"hspace": hspace, "wspace": wspace}
        )
        
        row_pain = 0
        row_temp = 1 if n_rows == 2 else None
        
        axes[row_pain, 0].set_ylabel(f"Pain - Non\n{freq_label}", fontsize=font_sizes["ylabel"], labelpad=10)

        for col, (tmin_win, tmax_win) in enumerate(zip(window_starts, window_ends)):
            time_label = f"{tmin_win:.2f}s"
            axes[row_pain, col].set_title(time_label, fontsize=9, pad=12, y=1.07)

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

        create_difference_colorbar(
            fig, axes, vabs_diff, viz_params["topo_cmap"],
            label="log10(power/baseline) difference",
            config=config
        )

        n_trials_pain = int(pain_mask.sum()) if pain_mask is not None else None
        n_trials_non = int(non_mask.sum()) if non_mask is not None else None
        
        title_parts = [f"Temporal topomaps: Pain - Non-pain difference ({band_name}, {window_label})"]
        if n_rows == 2:
            title_parts.append(f"Max - Min temp ({t_min:.1f}-{t_max:.1f}°C)")
        title_parts.append(f"log10(power/baseline) difference, vlim ±{vabs_diff:.2f}")
        
        stat_title = build_statistical_title(
            config, baseline_used, paired=False,
            n_trials_pain=n_trials_pain, n_trials_non=n_trials_non, is_group=False
        )
        
        full_title = f"{' | '.join(title_parts)}"
        if stat_title:
            full_title += f"\n{stat_title}"
        if sig_text:
            full_title += sig_text
        
        fig.suptitle(full_title, fontsize=font_sizes["figure_title"], y=0.995)

        filename = filename_base.format(band_name=band_name, tmin=tmin_clip, tmax=tmax_clip, n_windows=n_windows, baseline_str=baseline_str)
        _channels_save_fig(fig, out_dir, filename, config=config, logger=logger)
        plt.close(fig)


###################################################################
# Topomap Plotting Functions
###################################################################


def plot_topomap_grid_baseline_temps(
    tfr: "mne.time_frequency.EpochsTFR | mne.time_frequency.AverageTFR",
    events_df: Optional[pd.DataFrame],
    out_dir: Path,
    config,
    baseline: Optional[Tuple[Optional[float], Optional[float]]] = None,
    plateau_window: Tuple[float, float] = (3.0, 10.5),
    logger: Optional[logging.Logger] = None,
) -> None:
    """Plot topomap grid showing baseline percent change by temperature.
    
    Creates a grid of topomaps showing percent change from baseline across
    frequency bands and temperature conditions.
    
    Args:
        tfr: MNE TFR object (EpochsTFR or AverageTFR)
        events_df: Optional events DataFrame with temperature column
        out_dir: Output directory path
        config: Configuration object
        baseline: Optional baseline window tuple (defaults to config)
        plateau_window: Plateau window tuple for statistics
        logger: Optional logger instance
    """
    baseline = _get_baseline_window(config, baseline)
    if events_df is None:
        log("Temperature grid: events_df is None; skipping.", logger)
        return
    temp_col = get_temperature_column_from_config(config, events_df)
    if temp_col is None:
        log("Temperature grid: no temperature column found; skipping.", logger)
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
        log("Temperature grid: no temperature levels; skipping.", logger)
        return

    times_corr = np.asarray(tfr_avg_all_corr.times)
    tmin_req, tmax_req = plateau_window
    tmin = float(max(times_corr.min(), tmin_req))
    tmax = float(min(times_corr.max(), tmax_req))

    fmax_available = float(np.max(tfr_avg_all_corr.freqs))
    bands = get_bands_for_tfr(max_freq_available=fmax_available, config=config)

    cond_tfrs: List[Tuple[str, mne.time_frequency.AverageTFR, int, float]] = []
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
        log("Temperature grid: input is AverageTFR; cannot split by temperature; showing only All trials.", logger)

    plot_cfg = get_plot_config(config)
    tfr_config = plot_cfg.plot_type_configs.get("tfr", {})
    topomap_config = tfr_config.get("topomap", {})
    fig_size_per_col_large = plot_cfg.get_figure_size("tfr_per_col_large", plot_type="tfr")[0]
    fig_size_per_row_large = plot_cfg.get_figure_size("tfr_per_row_large", plot_type="tfr")[1]

    n_cols, n_rows = len(cond_tfrs), len(bands)
    fig, axes = plt.subplots(
        n_rows, n_cols, 
        figsize=(fig_size_per_col_large * n_cols, fig_size_per_row_large * n_rows), 
        squeeze=False,
        gridspec_kw={"wspace": 1.2, "hspace": 0.25},
    )

    for r, (band, (fmin, fmax)) in enumerate(bands.items()):
        fmax_eff = min(fmax, fmax_available)
        if fmin >= fmax_eff:
            for c in range(n_cols):
                axes[r, c].axis("off")
            continue

        diff_datas: List[Optional[np.ndarray]] = []
        for _, tfr_cond, _, _ in cond_tfrs:
            d = average_tfr_band(tfr_cond, fmin=fmin, fmax=fmax_eff, tmin=tmin, tmax=tmax)
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
            add_roi_annotations(ax, data, tfr_cond.info, config=config, data_format="percent")
            eeg_picks = extract_eeg_picks(tfr_cond, exclude_bads=False)
            data_for_label = data[eeg_picks] if len(eeg_picks) > 0 else data
            label_text = build_topomap_percentage_label(data_for_label)
            title_y = topomap_config.get("title_y", 1.04)
            title_pad = topomap_config.get("title_pad", 4)
            ax.text(0.5, 1.02, label_text, transform=ax.transAxes, ha="center", va="top", fontsize=plot_cfg.font.title)
            if r == 0:
                ax.set_title(f"{label} (n={n_cond})", fontsize=plot_cfg.font.title, pad=title_pad, y=title_y)

        axes[r, 0].set_ylabel(f"{band} ({fmin:.0f}-{fmax_eff:.0f} Hz)", fontsize=plot_cfg.font.ylabel)

        create_difference_colorbar(
            fig, axes[r, :].ravel().tolist(), diff_abs, get_viz_params(config)["topo_cmap"],
            label="Percent change from baseline (%)",
            fontsize=plot_cfg.font.title,
            config=config
        )

    fig.suptitle(
        f"Topomaps by temperature: % change from baseline over plateau t=[{tmin:.1f}, {tmax:.1f}] s",
        fontsize=plot_cfg.font.figure_title,
    )
    _channels_save_fig(fig, out_dir, "topomap_grid_bands_alltrials_plus_temperatures_baseline_percent.png", config=config, logger=logger, baseline_used=baseline_used)


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
    """Plot temporal topomaps showing pain-nonpain difference across time windows.
    
    Creates temporal topomap sequences showing pain-nonpain differences across
    multiple time windows for each frequency band.
    
    Args:
        tfr: MNE EpochsTFR object
        events_df: Optional events DataFrame with pain column
        out_dir: Output directory path
        config: Configuration object
        baseline: Optional baseline window tuple (defaults to config)
        plateau_window: Plateau window tuple for statistics
        window_size_ms: Size of each time window in milliseconds
        logger: Optional logger instance
    """
    if not require_epochs_tfr(tfr, "Temporal topomaps", logger):
        return

    baseline = _get_baseline_window(config, baseline)

    pain_col = get_pain_column_from_config(config, events_df)
    temp_col = get_temperature_column_from_config(config, events_df)
    
    if pain_col is None:
        log("Events with pain binary column required for temporal topomaps; skipping.", logger, "warning")
        return

    n = compute_aligned_data_length(tfr, events_df)

    pain_vec = extract_pain_vector_array(tfr, events_df, pain_col, n)
    if pain_vec is None:
        log("Events with pain binary column required for temporal topomaps; skipping.", logger, "warning")
        return
    
    pain_mask, non_mask = _create_pain_masks_from_vector(pain_vec)
    if pain_mask is None:
        log("Could not create pain masks; skipping.", logger, "warning")
        return

    if pain_mask.sum() == 0 or non_mask.sum() == 0:
        log("One of the groups has zero trials; skipping temporal topomaps.", logger, "warning")
        return

    log(f"Temporal topomaps (diff, all bands): pain={int(pain_mask.sum())}, non-pain={int(non_mask.sum())} trials.", logger)

    tfr_sub = create_tfr_subset(tfr, n)
    aligned = _align_and_trim_masks(
        tfr_sub,
        {"Pain contrast": (pain_mask, non_mask)},
        config, logger
    )
    if aligned is None:
        return
    
    pain_mask, non_mask = aligned["Pain contrast"]

    tfr_sub_stats = tfr_sub.copy()
    baseline_used = apply_baseline_and_crop(tfr_sub_stats, baseline=baseline, mode="logratio", logger=logger)

    tfr_pain = tfr_sub[pain_mask].average()
    tfr_non = tfr_sub[non_mask].average()
    
    apply_baseline_and_crop(tfr_pain, baseline=baseline_used, mode="logratio", logger=logger)
    apply_baseline_and_crop(tfr_non, baseline=baseline_used, mode="logratio", logger=logger)

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
                        strict=get_strict_mode(config),
                        logger=logger
                    )
                    if len(mask_min) != len(tfr_sub) or len(mask_max) != len(tfr_sub):
                        mask_min = mask_min[:len(tfr_sub)]
                        mask_max = mask_max[:len(tfr_sub)]
                    
                    tfr_min = tfr_sub[mask_min].average()
                    tfr_max = tfr_sub[mask_max].average()
                    apply_baseline_and_crop(tfr_min, baseline=baseline_used, mode="logratio", logger=logger)
                    apply_baseline_and_crop(tfr_max, baseline=baseline_used, mode="logratio", logger=logger)
                    
                    has_temp = True
                    log(f"Temporal topomaps: max temp={int(mask_max.sum())}, min temp={int(mask_min.sum())} trials.", logger)
                except ValueError as e:
                    log(f"{e}. Skipping temperature contrast.", logger, "warning")

    times = np.asarray(tfr_pain.times)
    tmin_req, tmax_req = plateau_window
    clipped = clip_time_range(times, tmin_req, tmax_req)
    if clipped is None:
        log(f"No valid time interval within data range; skipping temporal topomaps (available [{times.min():.2f}, {times.max():.2f}] s).", logger, "warning")
        return
    tmin_clip, tmax_clip = clipped
    
    if tmin_clip is None or tmax_clip is None:
        log(f"No valid time interval within data range; skipping temporal topomaps (available [{times.min():.2f}, {times.max():.2f}] s).", logger, "warning")
        return

    window_starts, window_ends = create_time_windows_fixed_size(tmin_clip, tmax_clip, window_size_ms)
    n_windows = len(window_starts)
    log(f"Creating temporal topomaps from {tmin_clip:.2f} to {tmax_clip:.2f} s using {n_windows} windows ({window_size_ms:.1f}ms each).", logger)

    if n_windows == 0:
        log("No valid windows created; skipping.", logger, "warning")
        return

    window_label = f"{tmin_clip:.1f}–{tmax_clip:.1f}s; {n_windows} windows @ {window_size_ms:.1f}ms"
    filename_base = "temporal_topomaps_pain_minus_nonpain_{band_name}_{tmin:.0f}-{tmax:.0f}s_{n_windows}windows_{baseline_str}.png"
    
    _plot_temporal_topomaps_for_bands(
        tfr_pain, tfr_non, tfr_sub_stats, tfr_max, tfr_min,
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
    """Plot temporal topomaps over plateau window with fixed window count.
    
    Creates temporal topomap sequences showing pain-nonpain differences across
    a fixed number of time windows over the plateau period.
    
    Args:
        tfr: MNE EpochsTFR object
        events_df: Optional events DataFrame with pain column
        out_dir: Output directory path
        config: Configuration object
        baseline: Optional baseline window tuple (defaults to config)
        plateau_window: Plateau window tuple for statistics
        window_count: Number of time windows to create
        logger: Optional logger instance
    """
    if not require_epochs_tfr(tfr, "Temporal topomaps", logger):
        return

    baseline = _get_baseline_window(config, baseline)

    pain_col = get_pain_column_from_config(config, events_df)
    temp_col = get_temperature_column_from_config(config, events_df)
    
    if pain_col is None:
        log("Events with pain binary column required for temporal topomaps; skipping.", logger, "warning")
        return

    n = compute_aligned_data_length(tfr, events_df)

    pain_vec = extract_pain_vector_array(tfr, events_df, pain_col, n)
    if pain_vec is None:
        log("Events with pain binary column required for temporal topomaps; skipping.", logger, "warning")
        return
    
    pain_mask, non_mask = _create_pain_masks_from_vector(pain_vec)
    if pain_mask is None:
        log("Could not create pain masks; skipping.", logger, "warning")
        return

    if pain_mask.sum() == 0 or non_mask.sum() == 0:
        log("One of the groups has zero trials; skipping temporal topomaps.", logger, "warning")
        return

    log(f"Temporal topomaps (plateau, all bands): pain={int(pain_mask.sum())}, non-pain={int(non_mask.sum())} trials.", logger)

    tfr_sub = create_tfr_subset(tfr, n)
    aligned = _align_and_trim_masks(
        tfr_sub,
        {"Pain contrast": (pain_mask, non_mask)},
        config, logger
    )
    if aligned is None:
        return
    
    pain_mask, non_mask = aligned["Pain contrast"]

    tfr_sub_stats = tfr_sub.copy()
    baseline_used = apply_baseline_and_crop(tfr_sub_stats, baseline=baseline, mode="logratio", logger=logger)

    tfr_pain = tfr_sub[pain_mask].average()
    tfr_non = tfr_sub[non_mask].average()
    
    apply_baseline_and_crop(tfr_pain, baseline=baseline_used, mode="logratio", logger=logger)
    apply_baseline_and_crop(tfr_non, baseline=baseline_used, mode="logratio", logger=logger)

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
                        strict=get_strict_mode(config),
                        logger=logger
                    )
                    if len(mask_min) != len(tfr_sub) or len(mask_max) != len(tfr_sub):
                        mask_min = mask_min[:len(tfr_sub)]
                        mask_max = mask_max[:len(tfr_sub)]
                    
                    tfr_min = tfr_sub[mask_min].average()
                    tfr_max = tfr_sub[mask_max].average()
                    apply_baseline_and_crop(tfr_min, baseline=baseline_used, mode="logratio", logger=logger)
                    apply_baseline_and_crop(tfr_max, baseline=baseline_used, mode="logratio", logger=logger)
                    
                    has_temp = True
                    log(f"Temporal topomaps: max temp={int(mask_max.sum())}, min temp={int(mask_min.sum())} trials.", logger)
                except ValueError as e:
                    log(f"{e}. Skipping temperature contrast.", logger, "warning")

    times = np.asarray(tfr_pain.times)
    tmin_req, tmax_req = plateau_window
    clipped = clip_time_range(times, tmin_req, tmax_req)
    if clipped is None:
        log(f"No valid time interval within data range; skipping temporal topomaps (available [{times.min():.2f}, {times.max():.2f}] s).", logger, "warning")
        return
    tmin_clip, tmax_clip = clipped
    
    if tmin_clip is None or tmax_clip is None:
        log(f"No valid time interval within data range; skipping temporal topomaps (available [{times.min():.2f}, {times.max():.2f}] s).", logger, "warning")
        return

    window_starts, window_ends = create_time_windows_fixed_count(tmin_clip, tmax_clip, window_count)
    n_windows = len(window_starts)
    window_size_eff = float((tmax_clip - tmin_clip) / n_windows) if n_windows > 0 else 0.0
    log(f"Creating temporal topomaps over plateau [{tmin_clip:.2f}, {tmax_clip:.2f}] s using {n_windows} windows (~{window_size_eff:.2f}s each).", logger)

    if n_windows == 0:
        log("No valid windows created; skipping.", logger, "warning")
        return

    window_label = f"plateau {tmin_clip:.0f}–{tmax_clip:.0f}s; {n_windows} windows @ {window_size_eff:.2f}s"
    filename_base = "temporal_topomaps_plateau_{band_name}_{tmin:.0f}-{tmax:.0f}s_{n_windows}windows_{baseline_str}.png"
    
    _plot_temporal_topomaps_for_bands(
        tfr_pain, tfr_non, tfr_sub_stats, tfr_max, tfr_min,
        pain_mask, non_mask, mask_max, mask_min,
        window_starts, window_ends, has_temp, t_min, t_max,
        tmin_clip, tmax_clip, n_windows, baseline_used,
        window_label, filename_base, out_dir, config, logger
    )

