"""
TFR-specific topomap plotting functions.

Functions for creating topomap visualizations for time-frequency representations,
including temperature grids and temporal topomaps.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, NamedTuple

import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt

from eeg_pipeline.plotting.io.figures import (
    robust_sym_vlim,
    extract_eeg_picks,
    get_viz_params,
    plot_topomap_on_ax,
)
from eeg_pipeline.utils.data.columns import get_temperature_column_from_config
from eeg_pipeline.utils.validation import require_epochs_tfr, ensure_aligned_lengths
from ...utils.analysis.tfr import (
    apply_baseline_and_crop,
    create_tfr_subset,
    get_bands_for_tfr,
    average_tfr_band,
    extract_trial_band_power,
    clip_time_range,
)
from ...utils.analysis.windowing import (
    build_time_windows_fixed_count,
    build_time_windows_fixed_size_clamped,
)
from ...utils.data.tfr_alignment import (
    extract_temperature_series,
    create_temperature_masks,
)
from ..config import get_plot_config
from ..core.utils import get_font_sizes, log
from ..core.statistics import get_strict_mode, compute_cluster_significance, build_statistical_title
from ..core.colorbars import create_difference_colorbar
from ..core.topomaps import build_topomap_diff_label, build_topomap_percentage_label
from ..core.annotations import add_roi_annotations, get_sig_marker_text
from .contrasts import _get_baseline_window, _align_and_trim_masks, _get_aligned_events_df_for_tfr
from .channels import _save_fig as _channels_save_fig


TEMPERATURE_TOLERANCE = 0.05
PERCENT_CAP = 100.0
DEFAULT_VLIM_FALLBACK = 1e-6


class TemperatureData(NamedTuple):
    """Container for temperature-related TFR data and masks."""
    has_temperature: bool
    tfr_max: Optional[mne.time_frequency.AverageTFR]
    tfr_min: Optional[mne.time_frequency.AverageTFR]
    mask_max: Optional[np.ndarray]
    mask_min: Optional[np.ndarray]
    t_min: Optional[float]
    t_max: Optional[float]


def _prepare_temperature_data(
    tfr_sub: mne.time_frequency.EpochsTFR,
    tfr: mne.time_frequency.EpochsTFR,
    events_df: Optional[pd.DataFrame],
    n_trials: int,
    baseline_used: Tuple[float, float],
    config,
    logger: Optional[logging.Logger] = None,
) -> TemperatureData:
    """Extract and prepare temperature-related TFR data.
    
    Args:
        tfr_sub: Subset TFR for statistics
        tfr: Full TFR object
        events_df: Events DataFrame
        n_trials: Number of trials
        baseline_used: Baseline window tuple
        config: Configuration object
        logger: Optional logger instance
        
    Returns:
        TemperatureData named tuple with all temperature-related data
    """
    aligned_events = _get_aligned_events_df_for_tfr(tfr, events_df, n_trials)
    temp_col = get_temperature_column_from_config(config, aligned_events) if aligned_events is not None else None
    temp_series = extract_temperature_series(tfr, aligned_events, temp_col, n_trials) if temp_col else None
    
    if temp_series is None:
        return TemperatureData(False, None, None, None, None, None, None)
    
    temp_result = create_temperature_masks(temp_series)
    if temp_result[0] is None:
        return TemperatureData(False, None, None, None, None, None, None)
    
    t_min, t_max, mask_min, mask_max = temp_result
    
    if mask_min.sum() == 0 or mask_max.sum() == 0:
        return TemperatureData(False, None, None, None, None, None, None)
    
    try:
        ensure_aligned_lengths(
            tfr_sub, mask_min, mask_max,
            context="Temperature contrast",
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
        
        log(f"Temperature contrast: max temp={int(mask_max.sum())}, min temp={int(mask_min.sum())} trials.", logger)
        return TemperatureData(True, tfr_max, tfr_min, mask_max, mask_min, t_min, t_max)
    except ValueError as e:
        log(f"{e}. Skipping temperature contrast.", logger, "warning")
        return TemperatureData(False, None, None, None, None, None, None)


def _compute_band_diff_data_windows(
    tfr_condition_2: mne.time_frequency.AverageTFR,
    tfr_condition_1: mne.time_frequency.AverageTFR,
    temp_data: TemperatureData,
    fmin: float,
    fmax_eff: float,
    window_starts: np.ndarray,
    window_ends: np.ndarray,
) -> Tuple[List[Optional[np.ndarray]], List[Optional[np.ndarray]]]:
    """Compute band difference data for multiple time windows.
    
    Returns:
        Tuple of (condition_diff_windows, temperature_diff_windows)
    """
    condition_diff_windows = []
    temperature_diff_windows = []
    
    for tmin_win, tmax_win in zip(window_starts, window_ends):
        condition_2_data = average_tfr_band(
            tfr_condition_2, fmin=fmin, fmax=fmax_eff, tmin=tmin_win, tmax=tmax_win
        )
        condition_1_data = average_tfr_band(
            tfr_condition_1, fmin=fmin, fmax=fmax_eff, tmin=tmin_win, tmax=tmax_win
        )
        
        if condition_2_data is not None and condition_1_data is not None:
            condition_diff_windows.append(condition_2_data - condition_1_data)
        else:
            condition_diff_windows.append(None)
        
        if temp_data.has_temperature and temp_data.tfr_max is not None and temp_data.tfr_min is not None:
            max_data = average_tfr_band(
                temp_data.tfr_max, fmin=fmin, fmax=fmax_eff, tmin=tmin_win, tmax=tmax_win
            )
            min_data = average_tfr_band(
                temp_data.tfr_min, fmin=fmin, fmax=fmax_eff, tmin=tmin_win, tmax=tmax_win
            )
            
            if max_data is not None and min_data is not None:
                temperature_diff_windows.append(max_data - min_data)
            else:
                temperature_diff_windows.append(None)
        else:
            temperature_diff_windows.append(None)
    
    return condition_diff_windows, temperature_diff_windows


def _compute_statistical_mask(
    tfr_sub: Optional[mne.time_frequency.EpochsTFR],
    condition_mask_a: np.ndarray,
    condition_mask_b: np.ndarray,
    fmin: float,
    fmax_eff: float,
    tmin_win: float,
    tmax_win: float,
    diff_data: np.ndarray,
    config,
    viz_params: Dict,
    logger: Optional[logging.Logger] = None,
) -> Tuple[Optional[np.ndarray], Optional[float], Optional[int], Optional[float]]:
    """Compute statistical significance mask for topomap.
    
    Returns:
        Tuple of (sig_mask, cluster_p_min, cluster_k, cluster_mass)
    """
    if not viz_params["diff_annotation_enabled"] or tfr_sub is None:
        return None, None, None, None
    
    diff_data_len = len(diff_data) if diff_data is not None else None
    return compute_cluster_significance(
        tfr_sub, condition_mask_a, condition_mask_b,
        fmin, fmax_eff, tmin_win, tmax_win,
        config, diff_data_len=diff_data_len, logger=logger
    )


def _extract_trial_power_data(
    tfr_sub: Optional[mne.time_frequency.EpochsTFR],
    condition_mask_a: np.ndarray,
    condition_mask_b: np.ndarray,
    fmin: float,
    fmax_eff: float,
    tmin_win: float,
    tmax_win: float,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Extract trial-level band power for two conditions."""
    if tfr_sub is None:
        return None, None
    
    data_group_a = extract_trial_band_power(tfr_sub[condition_mask_a], fmin, fmax_eff, tmin_win, tmax_win)
    data_group_b = extract_trial_band_power(tfr_sub[condition_mask_b], fmin, fmax_eff, tmin_win, tmax_win)
    return data_group_a, data_group_b


def _get_label_position(config) -> Tuple[float, float]:
    """Get label position from config with defaults."""
    plot_cfg = get_plot_config(config) if config else None
    if plot_cfg is None:
        return 0.5, 1.08
    
    tfr_config = plot_cfg.plot_type_configs.get("tfr", {})
    label_x = tfr_config.get("label_x_position", 0.5)
    label_y = tfr_config.get("label_y_position_bottom", 1.08)
    return label_x, label_y


def _plot_single_topomap_window(
    ax: plt.Axes,
    diff_data: np.ndarray,
    info: mne.Info,
    tfr_sub: Optional[mne.time_frequency.EpochsTFR],
    condition_mask_a: np.ndarray,
    condition_mask_b: np.ndarray,
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
    """Plot a single topomap window with optional statistical overlay."""
    if diff_data is None:
        ax.axis('off')
        return
    
    sig_mask, cluster_p_min, cluster_k, cluster_mass = _compute_statistical_mask(
        tfr_sub, condition_mask_a, condition_mask_b,
        fmin, fmax_eff, tmin_win, tmax_win,
        diff_data, config, viz_params, logger
    )
    
    annotation_mask = sig_mask if viz_params["diff_annotation_enabled"] else None
    plot_topomap_on_ax(
        ax, diff_data, info,
        vmin=-vabs_diff, vmax=+vabs_diff,
        mask=annotation_mask,
        mask_params=viz_params["sig_mask_params"],
        config=config
    )
    
    data_group_a, data_group_b = _extract_trial_power_data(
        tfr_sub, condition_mask_a, condition_mask_b,
        fmin, fmax_eff, tmin_win, tmax_win
    )
    
    is_cluster = sig_mask is not None and cluster_p_min is not None
    add_roi_annotations(
        ax, diff_data, info, config=config,
        sig_mask=annotation_mask,
        cluster_p_min=cluster_p_min,
        cluster_k=cluster_k,
        cluster_mass=cluster_mass,
        is_cluster=is_cluster,
        data_group_a=data_group_a,
        data_group_b=data_group_b,
        paired=paired
    )
    
    label = build_topomap_diff_label(
        diff_data, cluster_p_min, cluster_k, cluster_mass,
        config, viz_params, paired=paired
    )
    font_sizes = get_font_sizes()
    label_x, label_y = _get_label_position(config)
    ax.text(
        label_x, label_y, label,
        transform=ax.transAxes,
        ha="center", va="bottom",
        fontsize=font_sizes["label"]
    )


def _collect_valid_bands(
    tfr_condition_2: mne.time_frequency.AverageTFR,
    tfr_condition_1: mne.time_frequency.AverageTFR,
    temp_data: TemperatureData,
    window_starts: np.ndarray,
    window_ends: np.ndarray,
    config,
    logger: Optional[logging.Logger] = None,
) -> Tuple[Dict[str, Tuple[float, float, List[Optional[np.ndarray]], Optional[List[Optional[np.ndarray]]]]], List[np.ndarray]]:
    """Collect valid frequency bands and compute difference data.
    
    Returns:
        Tuple of (valid_bands dict, all_diff_data list)
    """
    fmax_available = float(np.max(tfr_condition_2.freqs))
    bands = get_bands_for_tfr(max_freq_available=fmax_available, config=config)

    valid_bands = {}
    all_diff_data = []

    for band_name, (fmin, fmax) in bands.items():
        fmax_eff = min(fmax, fmax_available)
        if fmin >= fmax_eff:
            continue

        condition_diff_windows, temp_diff_windows = _compute_band_diff_data_windows(
            tfr_condition_2, tfr_condition_1, temp_data,
            fmin, fmax_eff, window_starts, window_ends
        )

        condition_diff_valid = [d for d in condition_diff_windows if d is not None]
        if len(condition_diff_valid) == 0:
            log(f"No valid data found for {band_name} temporal topomaps; skipping this band.", logger, "warning")
            continue

        valid_bands[band_name] = (fmin, fmax_eff, condition_diff_windows, temp_diff_windows)
        all_diff_data.extend(condition_diff_valid)
        
        if temp_data.has_temperature and temp_diff_windows is not None:
            temp_diff_valid = [d for d in temp_diff_windows if d is not None]
            all_diff_data.extend(temp_diff_valid)

    return valid_bands, all_diff_data


def _build_figure_title(
    condition_label_2: str,
    condition_label_1: str,
    band_name: str,
    window_label: str,
    temp_data: TemperatureData,
    vabs_diff: float,
    baseline_used: Tuple[float, float],
    condition_mask_2: np.ndarray,
    condition_mask_1: np.ndarray,
    config,
) -> str:
    """Build figure title with statistical information."""
    title_parts = [
        f"Temporal topomaps: {condition_label_2} - {condition_label_1} difference ({band_name}, {window_label})"
    ]
    
    if temp_data.has_temperature and temp_data.t_min is not None and temp_data.t_max is not None:
        title_parts.append(f"Max - Min temp ({temp_data.t_min:.1f}-{temp_data.t_max:.1f}°C)")
    
    title_parts.append(f"log10(power/baseline) difference, vlim ±{vabs_diff:.2f}")
    
    n_trials_condition_2 = int(condition_mask_2.sum()) if condition_mask_2 is not None else None
    n_trials_condition_1 = int(condition_mask_1.sum()) if condition_mask_1 is not None else None
    
    stat_title = build_statistical_title(
        config, baseline_used, paired=False,
        n_trials_pain=n_trials_condition_2,
        n_trials_non=n_trials_condition_1,
        is_group=False
    )
    
    full_title = " | ".join(title_parts)
    if stat_title:
        full_title += f"\n{stat_title}"
    
    sig_text = get_sig_marker_text(config)
    if sig_text:
        full_title += sig_text
    
    return full_title


def _plot_temporal_topomaps_for_bands(
    tfr_condition_2: mne.time_frequency.AverageTFR,
    tfr_condition_1: mne.time_frequency.AverageTFR,
    tfr_sub: mne.time_frequency.EpochsTFR,
    temp_data: TemperatureData,
    condition_mask_2: np.ndarray,
    condition_mask_1: np.ndarray,
    window_starts: np.ndarray,
    window_ends: np.ndarray,
    tmin_clip: float,
    tmax_clip: float,
    n_windows: int,
    baseline_used: Tuple[float, float],
    window_label: str,
    filename_base: str,
    out_dir: Path,
    config,
    logger: Optional[logging.Logger] = None,
    *,
    condition_label_2: str = "Condition 2",
    condition_label_1: str = "Condition 1",
) -> None:
    """Plot temporal topomaps for all frequency bands."""
    valid_bands, all_diff_data = _collect_valid_bands(
        tfr_condition_2, tfr_condition_1, temp_data,
        window_starts, window_ends, config, logger
    )

    if len(valid_bands) == 0:
        log("No valid bands found for temporal topomaps; skipping.", logger, "warning")
        return

    vabs_diff = robust_sym_vlim(all_diff_data) if len(all_diff_data) > 0 else DEFAULT_VLIM_FALLBACK
    viz_params = get_viz_params(config)
    font_sizes = get_font_sizes()
    baseline_str = f"bl{abs(baseline_used[0]):.1f}to{abs(baseline_used[1]):.2f}" if baseline_used else "bl"
    
    log(f"Creating separate figure for each of {len(valid_bands)} frequency bands...", logger)
    
    temporal_spacing = config.get("time_frequency_analysis.topomap.temporal.single_subject", {}) if config else {}
    hspace = temporal_spacing.get("hspace", 0.15)
    wspace = temporal_spacing.get("wspace", 0.8)
    
    plot_cfg = get_plot_config(config)
    fig_size_per_col_large = plot_cfg.get_figure_size("tfr_per_col_large", plot_type="tfr")[0]
    fig_size_per_row_large = plot_cfg.get_figure_size("tfr_per_row_large", plot_type="tfr")[1]
    
    for band_name, (fmin, fmax_eff, condition_diff_windows, temp_diff_windows) in valid_bands.items():
        freq_label = f"{band_name} ({fmin:.0f}-{fmax_eff:.0f}Hz)"
        n_rows = 2 if (temp_data.has_temperature and temp_diff_windows is not None) else 1
        
        fig, axes = plt.subplots(
            n_rows, n_windows,
            figsize=(fig_size_per_col_large * n_windows, fig_size_per_row_large * n_rows),
            squeeze=False,
            gridspec_kw={"hspace": hspace, "wspace": wspace}
        )
        
        row_condition = 0
        row_temperature = 1 if n_rows == 2 else None
        
        axes[row_condition, 0].set_ylabel(
            f"{condition_label_2} - {condition_label_1}\n{freq_label}",
            fontsize=font_sizes["ylabel"],
            labelpad=10
        )

        for col, (tmin_win, tmax_win) in enumerate(zip(window_starts, window_ends)):
            time_label = f"{tmin_win:.2f}s"
            axes[row_condition, col].set_title(time_label, fontsize=9, pad=12, y=1.07)

            condition_diff_data = condition_diff_windows[col]
            _plot_single_topomap_window(
                axes[row_condition, col], condition_diff_data, tfr_condition_2.info, tfr_sub,
                condition_mask_2, condition_mask_1, fmin, fmax_eff, tmin_win, tmax_win,
                vabs_diff, config, viz_params, paired=False, logger=logger
            )

            if n_rows == 2 and temp_diff_windows is not None:
                temp_diff_data = temp_diff_windows[col]
                _plot_single_topomap_window(
                    axes[row_temperature, col], temp_diff_data, temp_data.tfr_max.info, tfr_sub,
                    temp_data.mask_max, temp_data.mask_min, fmin, fmax_eff, tmin_win, tmax_win,
                    vabs_diff, config, viz_params, paired=False, logger=logger
                )
                
                if col == 0:
                    axes[row_temperature, 0].set_ylabel(
                        f"Max - Min temp\n{freq_label}",
                        fontsize=font_sizes["ylabel"],
                        labelpad=10
                    )

        create_difference_colorbar(
            fig, axes, vabs_diff, viz_params["topo_cmap"],
            label="log10(power/baseline) difference",
            config=config
        )

        full_title = _build_figure_title(
            condition_label_2, condition_label_1, band_name, window_label,
            temp_data, vabs_diff, baseline_used,
            condition_mask_2, condition_mask_1, config
        )
        
        fig.suptitle(full_title, fontsize=font_sizes["figure_title"], y=0.995)

        filename = filename_base.format(
            band_name=band_name, tmin=tmin_clip, tmax=tmax_clip,
            n_windows=n_windows, baseline_str=baseline_str
        )
        _channels_save_fig(fig, out_dir, filename, config=config, logger=logger)
        plt.close(fig)


def plot_topomap_grid_baseline_temps(
    tfr: "mne.time_frequency.EpochsTFR | mne.time_frequency.AverageTFR",
    events_df: Optional[pd.DataFrame],
    out_dir: Path,
    config,
    baseline: Optional[Tuple[Optional[float], Optional[float]]] = None,
    active_window: Tuple[float, float] = (3.0, 10.5),
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
        active_window: Active window tuple for statistics
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
    tmin_req, tmax_req = active_window
    tmin = float(max(times_corr.min(), tmin_req))
    tmax = float(min(times_corr.max(), tmax_req))

    fmax_available = float(np.max(tfr_avg_all_corr.freqs))
    bands = get_bands_for_tfr(max_freq_available=fmax_available, config=config)

    cond_tfrs: List[Tuple[str, mne.time_frequency.AverageTFR, int, float]] = []
    n_all = len(tfr_corr) if isinstance(tfr_corr, mne.time_frequency.EpochsTFR) else 1
    cond_tfrs.append(("All trials", tfr_avg_all_corr, n_all, np.nan))

    if isinstance(tfr_corr, mne.time_frequency.EpochsTFR):
        for temp_value in temps:
            temp_values = pd.to_numeric(events_df[temp_col], errors="coerce")
            mask = np.abs(temp_values - float(temp_value)) < TEMPERATURE_TOLERANCE
            mask = np.asarray(mask, dtype=bool)
            if mask.sum() == 0:
                continue
            tfr_temp = tfr_corr.copy()[mask].average()
            cond_tfrs.append((f"{temp_value:.1f}°C", tfr_temp, int(mask.sum()), float(temp_value)))
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

        diff_abs = robust_sym_vlim(vals, cap=PERCENT_CAP)
        if not np.isfinite(diff_abs) or diff_abs == 0:
            diff_abs = DEFAULT_VLIM_FALLBACK

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
        f"Topomaps by temperature: % change from baseline over active window t=[{tmin:.1f}, {tmax:.1f}] s",
        fontsize=plot_cfg.font.figure_title,
    )
    _channels_save_fig(fig, out_dir, "topomap_grid_bands_alltrials_plus_temperatures_baseline_percent.png", config=config, logger=logger, baseline_used=baseline_used)


def plot_pain_nonpain_temporal_topomaps_diff_allbands(
    tfr: "mne.time_frequency.EpochsTFR",
    events_df: Optional[pd.DataFrame],
    out_dir: Path,
    config,
    baseline: Optional[Tuple[Optional[float], Optional[float]]] = None,
    active_window: Tuple[float, float] = (3.0, 10.5),
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
        active_window: Active window tuple for statistics
        window_size_ms: Size of each time window in milliseconds
        logger: Optional logger instance
    """
    if not require_epochs_tfr(tfr, "Temporal topomaps", logger):
        return

    baseline = _get_baseline_window(config, baseline)
    from .contrasts import _prepare_comparison_contrast_data, _get_aligned_events_df_for_tfr

    tfr_sub, mask1, mask2, label1, label2, n = _prepare_comparison_contrast_data(
        tfr, events_df, config, logger, context="Temporal topomaps"
    )
    if tfr_sub is None:
        return

    log(f"Temporal topomaps (diff, all bands): {label2}={int(mask2.sum())}, {label1}={int(mask1.sum())} trials.", logger)

    aligned = _align_and_trim_masks(
        tfr_sub,
        {"Condition contrast": (mask2, mask1)},
        config,
        logger,
    )
    if aligned is None:
        return

    mask2, mask1 = aligned["Condition contrast"]

    tfr_sub_stats = tfr_sub.copy()
    baseline_used = apply_baseline_and_crop(tfr_sub_stats, baseline=baseline, mode="logratio", logger=logger)

    tfr_condition_2 = tfr_sub[mask2].average()
    tfr_condition_1 = tfr_sub[mask1].average()
    
    apply_baseline_and_crop(tfr_condition_2, baseline=baseline_used, mode="logratio", logger=logger)
    apply_baseline_and_crop(tfr_condition_1, baseline=baseline_used, mode="logratio", logger=logger)

    temp_data = _prepare_temperature_data(
        tfr_sub, tfr, events_df, int(n), baseline_used, config, logger
    )

    times = np.asarray(tfr_condition_2.times)
    tmin_req, tmax_req = active_window
    clipped = clip_time_range(times, tmin_req, tmax_req)
    if clipped is None:
        log(f"No valid time interval within data range; skipping temporal topomaps (available [{times.min():.2f}, {times.max():.2f}] s).", logger, "warning")
        return
    tmin_clip, tmax_clip = clipped
    
    if tmin_clip is None or tmax_clip is None:
        log(f"No valid time interval within data range; skipping temporal topomaps (available [{times.min():.2f}, {times.max():.2f}] s).", logger, "warning")
        return

    window_starts, window_ends = build_time_windows_fixed_size_clamped(
        tmin_clip,
        tmax_clip,
        window_size_ms / 1000.0,
    )
    n_windows = len(window_starts)
    log(f"Creating temporal topomaps from {tmin_clip:.2f} to {tmax_clip:.2f} s using {n_windows} windows ({window_size_ms:.1f}ms each).", logger)

    if n_windows == 0:
        log("No valid windows created; skipping.", logger, "warning")
        return

    window_label = f"{tmin_clip:.1f}–{tmax_clip:.1f}s; {n_windows} windows @ {window_size_ms:.1f}ms"
    filename_base = "temporal_topomaps_pain_minus_nonpain_{band_name}_{tmin:.0f}-{tmax:.0f}s_{n_windows}windows_{baseline_str}.png"
    
    _plot_temporal_topomaps_for_bands(
        tfr_condition_2, tfr_condition_1, tfr_sub_stats, temp_data,
        mask2, mask1, window_starts, window_ends,
        tmin_clip, tmax_clip, n_windows, baseline_used,
        window_label, filename_base, out_dir, config, logger,
        condition_label_2=str(label2),
        condition_label_1=str(label1),
    )


def plot_temporal_topomaps_allbands_active(
    tfr: "mne.time_frequency.EpochsTFR",
    events_df: Optional[pd.DataFrame],
    out_dir: Path,
    config,
    baseline: Optional[Tuple[Optional[float], Optional[float]]] = None,
    active_window: Tuple[float, float] = (3.0, 10.5),
    window_count: int = 5,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Plot temporal topomaps over active window with fixed window count.
    
    Creates temporal topomap sequences showing pain-nonpain differences across
    a fixed number of time windows over the active period.
    
    Args:
        tfr: MNE EpochsTFR object
        events_df: Optional events DataFrame with pain column
        out_dir: Output directory path
        config: Configuration object
        baseline: Optional baseline window tuple (defaults to config)
        active_window: Active window tuple for statistics
        window_count: Number of time windows to create
        logger: Optional logger instance
    """
    if not require_epochs_tfr(tfr, "Temporal topomaps", logger):
        return

    baseline = _get_baseline_window(config, baseline)
    from .contrasts import _prepare_comparison_contrast_data, _get_aligned_events_df_for_tfr

    tfr_sub, mask1, mask2, label1, label2, n = _prepare_comparison_contrast_data(
        tfr, events_df, config, logger, context="Temporal topomaps"
    )
    if tfr_sub is None:
        return

    log(f"Temporal topomaps (active, all bands): {label2}={int(mask2.sum())}, {label1}={int(mask1.sum())} trials.", logger)

    aligned = _align_and_trim_masks(
        tfr_sub,
        {"Condition contrast": (mask2, mask1)},
        config,
        logger,
    )
    if aligned is None:
        return

    mask2, mask1 = aligned["Condition contrast"]

    tfr_sub_stats = tfr_sub.copy()
    baseline_used = apply_baseline_and_crop(tfr_sub_stats, baseline=baseline, mode="logratio", logger=logger)

    tfr_condition_2 = tfr_sub[mask2].average()
    tfr_condition_1 = tfr_sub[mask1].average()
    
    apply_baseline_and_crop(tfr_condition_2, baseline=baseline_used, mode="logratio", logger=logger)
    apply_baseline_and_crop(tfr_condition_1, baseline=baseline_used, mode="logratio", logger=logger)

    temp_data = _prepare_temperature_data(
        tfr_sub, tfr, events_df, int(n), baseline_used, config, logger
    )

    times = np.asarray(tfr_condition_2.times)
    tmin_req, tmax_req = active_window
    clipped = clip_time_range(times, tmin_req, tmax_req)
    if clipped is None:
        log(f"No valid time interval within data range; skipping temporal topomaps (available [{times.min():.2f}, {times.max():.2f}] s).", logger, "warning")
        return
    tmin_clip, tmax_clip = clipped
    
    if tmin_clip is None or tmax_clip is None:
        log(f"No valid time interval within data range; skipping temporal topomaps (available [{times.min():.2f}, {times.max():.2f}] s).", logger, "warning")
        return

    window_starts, window_ends = build_time_windows_fixed_count(tmin_clip, tmax_clip, window_count)
    n_windows = len(window_starts)
    window_size_eff = float((tmax_clip - tmin_clip) / n_windows) if n_windows > 0 else 0.0
    log(f"Creating temporal topomaps over active window [{tmin_clip:.2f}, {tmax_clip:.2f}] s using {n_windows} windows (~{window_size_eff:.2f}s each).", logger)

    if n_windows == 0:
        log("No valid windows created; skipping.", logger, "warning")
        return

    window_label = f"active {tmin_clip:.0f}–{tmax_clip:.0f}s; {n_windows} windows @ {window_size_eff:.2f}s"
    filename_base = "temporal_topomaps_active_{band_name}_{tmin:.0f}-{tmax:.0f}s_{n_windows}windows_{baseline_str}.png"
    
    _plot_temporal_topomaps_for_bands(
        tfr_condition_2, tfr_condition_1, tfr_sub_stats, temp_data,
        mask2, mask1, window_starts, window_ends,
        tmin_clip, tmax_clip, n_windows, baseline_used,
        window_label, filename_base, out_dir, config, logger,
        condition_label_2=str(label2),
        condition_label_1=str(label1),
    )
