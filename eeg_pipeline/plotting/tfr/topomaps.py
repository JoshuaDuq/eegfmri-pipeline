"""
TFR-specific topomap plotting functions.

Functions for creating topomap visualizations for time-frequency representations,
including predictor grids and temporal topomaps.
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
    get_viz_params,
    plot_topomap_on_ax,
)
from eeg_pipeline.utils.data.columns import get_predictor_column_from_config
from eeg_pipeline.utils.validation import require_epochs_tfr, ensure_aligned_lengths
from ...utils.analysis.tfr import (
    apply_baseline_and_crop,
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
    extract_predictor_series,
    create_predictor_masks,
)
from ..config import get_plot_config
from ..core.utils import get_font_sizes, log
from ..core.statistics import get_strict_mode, compute_cluster_significance, build_statistical_title
from ..core.colorbars import create_difference_colorbar
from ..core.topomaps import build_topomap_diff_label
from ..core.annotations import add_roi_annotations, get_sig_marker_text
from .contrasts import (
    _get_baseline_window,
    _align_and_trim_masks,
    _get_aligned_events_df_for_tfr,
    _prepare_comparison_contrast_data,
)
from .channels import _save_fig as _channels_save_fig
from eeg_pipeline.utils.config.loader import require_config_value


TEMPERATURE_TOLERANCE = 0.05
PERCENT_CAP = 100.0
DEFAULT_VLIM_FALLBACK = 1e-6


class PredictorData(NamedTuple):
    """Container for predictor-related TFR data and masks."""
    has_predictor: bool
    tfr_max: Optional[mne.time_frequency.AverageTFR]
    tfr_min: Optional[mne.time_frequency.AverageTFR]
    mask_max: Optional[np.ndarray]
    mask_min: Optional[np.ndarray]
    t_min: Optional[float]
    t_max: Optional[float]


def _prepare_predictor_data(
    tfr_sub: mne.time_frequency.EpochsTFR,
    tfr: mne.time_frequency.EpochsTFR,
    events_df: Optional[pd.DataFrame],
    n_trials: int,
    baseline_used: Tuple[float, float],
    config,
    logger: Optional[logging.Logger] = None,
) -> PredictorData:
    """Extract and prepare predictor-related TFR data.
    
    Args:
        tfr_sub: Subset TFR for statistics
        tfr: Full TFR object
        events_df: Events DataFrame
        n_trials: Number of trials
        baseline_used: Baseline window tuple
        config: Configuration object
        logger: Optional logger instance
        
    Returns:
        PredictorData named tuple with all predictor-related data
    """
    aligned_events = _get_aligned_events_df_for_tfr(tfr, events_df, n_trials)
    temp_col = get_predictor_column_from_config(config, aligned_events) if aligned_events is not None else None
    pred_series = extract_predictor_series(tfr, aligned_events, temp_col, n_trials) if temp_col else None

    if pred_series is None:
        return PredictorData(False, None, None, None, None, None, None)

    pred_result = create_predictor_masks(pred_series)
    if pred_result[0] is None:
        return PredictorData(False, None, None, None, None, None, None)

    t_min, t_max, mask_min, mask_max = pred_result
    
    if mask_min.sum() == 0 or mask_max.sum() == 0:
        return PredictorData(False, None, None, None, None, None, None)
    
    try:
        ensure_aligned_lengths(
            tfr_sub, mask_min, mask_max,
            context="Predictor contrast",
            strict=get_strict_mode(config),
            logger=logger
        )
        # tfr_sub is expected to be baseline-corrected at the epoch level.
        tfr_min = tfr_sub[mask_min].average()
        tfr_max = tfr_sub[mask_max].average()
        
        log(
            f"Predictor contrast: max n={int(mask_max.sum())}, min n={int(mask_min.sum())} trials.",
            logger,
        )
        return PredictorData(True, tfr_max, tfr_min, mask_max, mask_min, t_min, t_max)
    except ValueError as e:
        log(f"{e}. Skipping predictor contrast.", logger, "warning")
        return PredictorData(False, None, None, None, None, None, None)


def _compute_band_diff_data_windows(
    tfr_condition_2: mne.time_frequency.AverageTFR,
    tfr_condition_1: mne.time_frequency.AverageTFR,
    temp_data: PredictorData,
    fmin: float,
    fmax_eff: float,
    window_starts: np.ndarray,
    window_ends: np.ndarray,
) -> Tuple[List[Optional[np.ndarray]], List[Optional[np.ndarray]]]:
    """Compute band difference data for multiple time windows.
    
    Returns:
        Tuple of (condition_diff_windows, predictor_diff_windows)
    """
    condition_diff_windows = []
    predictor_diff_windows = []
    
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
        
        if temp_data.has_predictor and temp_data.tfr_max is not None and temp_data.tfr_min is not None:
            max_data = average_tfr_band(
                temp_data.tfr_max, fmin=fmin, fmax=fmax_eff, tmin=tmin_win, tmax=tmax_win
            )
            min_data = average_tfr_band(
                temp_data.tfr_min, fmin=fmin, fmax=fmax_eff, tmin=tmin_win, tmax=tmax_win
            )
            
            if max_data is not None and min_data is not None:
                predictor_diff_windows.append(max_data - min_data)
            else:
                predictor_diff_windows.append(None)
        else:
            predictor_diff_windows.append(None)
    
    return condition_diff_windows, predictor_diff_windows


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
    
    return compute_cluster_significance(
        tfr_sub, condition_mask_a, condition_mask_b,
        fmin, fmax_eff, tmin_win, tmax_win,
        config, diff_data_len=len(diff_data), logger=logger
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
    temp_data: PredictorData,
    window_starts: np.ndarray,
    window_ends: np.ndarray,
    config,
    logger: Optional[logging.Logger] = None,
) -> Tuple[Dict[str, Tuple[float, float, List[Optional[np.ndarray]], Optional[List[Optional[np.ndarray]]]]], List[np.ndarray]]:
    """Collect valid frequency bands and compute difference data.
    
    Returns:
        Tuple of (valid_bands dict, all_diff_data list)
    """
    from eeg_pipeline.utils.config.loader import get_config_value
    
    fmax_available = float(np.max(tfr_condition_2.freqs))
    bands = get_bands_for_tfr(max_freq_available=fmax_available, config=config)
    
    selected_bands = get_config_value(config, "time_frequency_analysis.selected_bands", None)
    if selected_bands and isinstance(selected_bands, (list, tuple)) and len(selected_bands) > 0:
        bands = {k: v for k, v in bands.items() if k in selected_bands}

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
        
        if temp_data.has_predictor and temp_diff_windows is not None:
            temp_diff_valid = [d for d in temp_diff_windows if d is not None]
            all_diff_data.extend(temp_diff_valid)

    return valid_bands, all_diff_data


def _build_figure_title(
    condition_label_2: str,
    condition_label_1: str,
    band_name: str,
    window_label: str,
    temp_data: PredictorData,
    vabs_diff: float,
    baseline_used: Tuple[float, float],
    condition_mask_2: np.ndarray,
    condition_mask_1: np.ndarray,
    config,
) -> str:
    """Build figure title with statistical information."""
    title_parts = [
        f"{condition_label_2} - {condition_label_1} ({band_name}, {window_label})"
    ]
    
    title_parts.append(f"log10(power/baseline) difference, vlim ±{vabs_diff:.2f}")
    
    n_trials_condition_2 = int(condition_mask_2.sum()) if condition_mask_2 is not None else None
    n_trials_condition_1 = int(condition_mask_1.sum()) if condition_mask_1 is not None else None
    
    stat_title = build_statistical_title(
        config, baseline_used, paired=False,
        n_trials_condition_2=n_trials_condition_2,
        n_trials_condition_1=n_trials_condition_1,
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
    temp_data: PredictorData,
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
        n_rows = 1
        
        fig, axes = plt.subplots(
            n_rows, n_windows,
            figsize=(fig_size_per_col_large * n_windows, fig_size_per_row_large * n_rows),
            squeeze=False,
            gridspec_kw={"hspace": hspace, "wspace": wspace}
        )
        
        axes[0, 0].set_ylabel(
            f"{condition_label_2} - {condition_label_1}\n{freq_label}",
            fontsize=font_sizes["ylabel"],
            labelpad=10
        )

        for col, (tmin_win, tmax_win) in enumerate(zip(window_starts, window_ends)):
            time_label = f"{tmin_win:.2f}s"
            axes[0, col].set_title(time_label, fontsize=9, pad=12, y=1.07)

            condition_diff_data = condition_diff_windows[col]
            _plot_single_topomap_window(
                axes[0, col], condition_diff_data, tfr_condition_2.info, tfr_sub,
                condition_mask_2, condition_mask_1, fmin, fmax_eff, tmin_win, tmax_win,
                vabs_diff, config, viz_params, paired=False, logger=logger
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

        from .channels import _sanitize_label_for_filename
        label_2_sanitized = _sanitize_label_for_filename(condition_label_2)
        label_1_sanitized = _sanitize_label_for_filename(condition_label_1)
        filename = filename_base.format(
            label_2=label_2_sanitized, label_1=label_1_sanitized,
            band_name=band_name, tmin=tmin_clip, tmax=tmax_clip,
            n_windows=n_windows, baseline_str=baseline_str
        )
        _channels_save_fig(fig, out_dir, filename, config=config, logger=logger)
        plt.close(fig)


def _prepare_temporal_topomap_data(
    tfr: "mne.time_frequency.EpochsTFR",
    events_df: Optional[pd.DataFrame],
    config,
    baseline: Optional[Tuple[Optional[float], Optional[float]]],
    active_window: Tuple[float, float],
    logger: Optional[logging.Logger],
    log_context: str,
) -> Optional[Tuple[
    mne.time_frequency.AverageTFR,
    mne.time_frequency.AverageTFR,
    mne.time_frequency.EpochsTFR,
    PredictorData,
    np.ndarray,
    np.ndarray,
    Tuple[float, float],
    float,
    float,
    str,
    str,
]]:
    """Prepare common data for temporal topomap plotting.
    
    Returns:
        Tuple of (tfr_condition_2, tfr_condition_1, tfr_sub_stats, temp_data,
                 mask2, mask1, baseline_used, tmin_clip, tmax_clip, label2, label1)
        or None if preparation fails
    """
    if not require_epochs_tfr(tfr, "Temporal topomaps", logger):
        return None

    baseline = _get_baseline_window(config, baseline)
    tfr_sub, mask1, mask2, label1, label2, n = _prepare_comparison_contrast_data(
        tfr, events_df, config, logger, context="Temporal topomaps"
    )
    if tfr_sub is None:
        return None

    log(f"{log_context}: {label2}={int(mask2.sum())}, {label1}={int(mask1.sum())} trials.", logger)

    aligned = _align_and_trim_masks(
        tfr_sub,
        {"Condition contrast": (mask2, mask1)},
        config,
        logger,
    )
    if aligned is None:
        return None

    mask2, mask1 = aligned["Condition contrast"]

    tfr_sub_stats = tfr_sub.copy()
    baseline_used = apply_baseline_and_crop(tfr_sub_stats, baseline=baseline, mode="logratio", logger=logger)

    tfr_condition_2 = tfr_sub_stats[mask2].average()
    tfr_condition_1 = tfr_sub_stats[mask1].average()

    temp_data = _prepare_predictor_data(
        tfr_sub_stats, tfr, events_df, int(n), baseline_used, config, logger
    )

    times = np.asarray(tfr_condition_2.times)
    tmin_req, tmax_req = active_window
    clipped = clip_time_range(times, tmin_req, tmax_req)
    if clipped is None:
        log(f"No valid time interval within data range; skipping temporal topomaps (available [{times.min():.2f}, {times.max():.2f}] s).", logger, "warning")
        return None
    tmin_clip, tmax_clip = clipped

    return (
        tfr_condition_2, tfr_condition_1, tfr_sub_stats, temp_data,
        mask2, mask1, baseline_used, tmin_clip, tmax_clip, str(label2), str(label1)
    )


def plot_temporal_topomaps_diff_allbands(
    tfr: "mne.time_frequency.EpochsTFR",
    events_df: Optional[pd.DataFrame],
    out_dir: Path,
    config,
    baseline: Optional[Tuple[Optional[float], Optional[float]]] = None,
    active_window: Optional[Tuple[float, float]] = None,
    window_size_ms: Optional[float] = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Plot temporal topomaps showing condition difference across time windows.
    
    Creates temporal topomap sequences showing condition differences across
    multiple time windows for each frequency band.
    
    Args:
        tfr: MNE EpochsTFR object
        events_df: Optional events DataFrame with condition column
        out_dir: Output directory path
        config: Configuration object
        baseline: Optional baseline window tuple (defaults to config)
        active_window: Optional active window tuple (defaults to config: time_frequency_analysis.active_window)
        window_size_ms: Optional size of each time window in milliseconds (defaults to config: time_frequency_analysis.topomap.temporal.window_size_ms)
        logger: Optional logger instance
    """
    if active_window is None:
        active_raw = require_config_value(config, "time_frequency_analysis.active_window")
        if not isinstance(active_raw, (list, tuple)) or len(active_raw) < 2:
            raise ValueError(
                "time_frequency_analysis.active_window must be a list/tuple of length 2 "
                f"(got {active_raw!r})"
            )
        active_window = (float(active_raw[0]), float(active_raw[1]))
    if window_size_ms is None:
        window_size_ms = float(require_config_value(config, "time_frequency_analysis.topomap.temporal.window_size_ms"))
    
    prepared = _prepare_temporal_topomap_data(
        tfr, events_df, config, baseline, active_window, logger,
        "Temporal topomaps (diff, all bands)"
    )
    if prepared is None:
        return

    tfr_condition_2, tfr_condition_1, tfr_sub_stats, temp_data, mask2, mask1, baseline_used, tmin_clip, tmax_clip, label2, label1 = prepared

    window_starts, window_ends = build_time_windows_fixed_size_clamped(
        tmin_clip, tmax_clip, window_size_ms / 1000.0
    )
    n_windows = len(window_starts)
    log(f"Creating temporal topomaps from {tmin_clip:.2f} to {tmax_clip:.2f} s using {n_windows} windows ({window_size_ms:.1f}ms each).", logger)

    if n_windows == 0:
        log("No valid windows created; skipping.", logger, "warning")
        return

    window_label = f"{tmin_clip:.1f}–{tmax_clip:.1f}s; {n_windows} windows @ {window_size_ms:.1f}ms"
    from .channels import _sanitize_label_for_filename
    _sanitize_label_for_filename(label2)
    _sanitize_label_for_filename(label1)
    filename_base = "temporal_topomaps_{label_2}_minus_{label_1}_{band_name}_{tmin:.0f}-{tmax:.0f}s_{n_windows}windows_{baseline_str}.png"
    
    _plot_temporal_topomaps_for_bands(
        tfr_condition_2, tfr_condition_1, tfr_sub_stats, temp_data,
        mask2, mask1, window_starts, window_ends,
        tmin_clip, tmax_clip, n_windows, baseline_used,
        window_label, filename_base, out_dir, config, logger,
        condition_label_2=label2,
        condition_label_1=label1,
    )


def plot_temporal_topomaps_allbands_active(
    tfr: "mne.time_frequency.EpochsTFR",
    events_df: Optional[pd.DataFrame],
    out_dir: Path,
    config,
    baseline: Optional[Tuple[Optional[float], Optional[float]]] = None,
    active_window: Optional[Tuple[float, float]] = None,
    window_count: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Plot temporal topomaps over active window with fixed window count.
    
    Creates temporal topomap sequences showing condition differences across
    a fixed number of time windows over the active period.
    
    Args:
        tfr: MNE EpochsTFR object
        events_df: Optional events DataFrame with condition column
        out_dir: Output directory path
        config: Configuration object
        baseline: Optional baseline window tuple (defaults to config)
        active_window: Optional active window tuple (defaults to config: time_frequency_analysis.active_window)
        window_count: Optional number of time windows to create (defaults to config: time_frequency_analysis.topomap.temporal.window_count)
        logger: Optional logger instance
    """
    if active_window is None:
        active_raw = require_config_value(config, "time_frequency_analysis.active_window")
        if not isinstance(active_raw, (list, tuple)) or len(active_raw) < 2:
            raise ValueError(
                "time_frequency_analysis.active_window must be a list/tuple of length 2 "
                f"(got {active_raw!r})"
            )
        active_window = (float(active_raw[0]), float(active_raw[1]))
    if window_count is None:
        window_count = int(require_config_value(config, "time_frequency_analysis.topomap.temporal.window_count"))
    
    prepared = _prepare_temporal_topomap_data(
        tfr, events_df, config, baseline, active_window, logger,
        "Temporal topomaps (active, all bands)"
    )
    if prepared is None:
        return

    tfr_condition_2, tfr_condition_1, tfr_sub_stats, temp_data, mask2, mask1, baseline_used, tmin_clip, tmax_clip, label2, label1 = prepared

    window_starts, window_ends = build_time_windows_fixed_count(tmin_clip, tmax_clip, window_count)
    n_windows = len(window_starts)
    window_size_eff = float((tmax_clip - tmin_clip) / n_windows) if n_windows > 0 else 0.0
    log(f"Creating temporal topomaps over active window [{tmin_clip:.2f}, {tmax_clip:.2f}] s using {n_windows} windows (~{window_size_eff:.2f}s each).", logger)

    if n_windows == 0:
        log("No valid windows created; skipping.", logger, "warning")
        return

    window_label = f"active {tmin_clip:.0f}–{tmax_clip:.0f}s; {n_windows} windows @ {window_size_eff:.2f}s"
    from .channels import _sanitize_label_for_filename
    _sanitize_label_for_filename(label2)
    _sanitize_label_for_filename(label1)
    filename_base = "temporal_topomaps_{label_2}_minus_{label_1}_active_{band_name}_{tmin:.0f}-{tmax:.0f}s_{n_windows}windows_{baseline_str}.png"
    
    _plot_temporal_topomaps_for_bands(
        tfr_condition_2, tfr_condition_1, tfr_sub_stats, temp_data,
        mask2, mask1, window_starts, window_ends,
        tmin_clip, tmax_clip, n_windows, baseline_used,
        window_label, filename_base, out_dir, config, logger,
        condition_label_2=label2,
        condition_label_1=label1,
    )
