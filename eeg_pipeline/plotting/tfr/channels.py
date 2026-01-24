"""
Channel-level TFR plotting functions.

Functions for creating time-frequency representations (TFR) plots at the channel level,
including single-channel and multi-channel visualizations.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from eeg_pipeline.plotting.io.figures import (
    unwrap_figure,
    extract_eeg_picks,
    logratio_to_pct,
    save_fig as central_save_fig,
)
from eeg_pipeline.utils.formatting import format_baseline_window_string, sanitize_label
from eeg_pipeline.utils.config.loader import require_config_value
from ...utils.analysis.windowing import time_mask_loose, time_mask_strict
from ...utils.analysis.tfr import (
    apply_baseline_and_average,
    apply_baseline_and_crop,
)
from ..config import get_plot_config
from ..core.utils import get_font_sizes, log
from ..core.statistics import get_strict_mode


###################################################################
# Helper Functions
###################################################################


def _filter_channels_by_names(
    available_channels: List[str],
    requested_channels: List[str],
    logger: Optional[logging.Logger] = None,
) -> List[str]:
    """Filter available channels by requested channel names (case-insensitive).
    
    Args:
        available_channels: List of available channel names
        requested_channels: List of requested channel names
        logger: Optional logger instance
        
    Returns:
        Filtered list of channel names that match requested channels
    """
    requested_set = {ch.upper() for ch in requested_channels}
    filtered = [ch for ch in available_channels if ch.upper() in requested_set]
    if not filtered and logger:
        logger.warning(
            f"No matching channels found for specified channels: {requested_channels}"
        )
    return filtered


def _pick_central_channel(info, preferred: str = "Cz", logger: Optional[logging.Logger] = None) -> str:
    """Pick a central channel for plotting, preferring the specified channel.
    
    Args:
        info: MNE Info object containing channel information
        preferred: Preferred channel name (default: "Cz")
        logger: Optional logger instance
        
    Returns:
        Channel name to use for plotting
        
    Raises:
        RuntimeError: If no EEG channels are available
    """
    ch_names = info["ch_names"]
    if preferred in ch_names:
        return preferred
    
    for ch_name in ch_names:
        if ch_name.lower() == preferred.lower():
            return ch_name
    
    picks = extract_eeg_picks(info, exclude_bads=True)
    if len(picks) == 0:
        raise RuntimeError("No EEG channels available for plotting.")
    
    fallback = ch_names[picks[0]]
    log(f"Channel '{preferred}' not found; using '{fallback}' instead.", logger, "warning")
    return fallback


def _compute_active_statistics(
    arr: np.ndarray,
    times: np.ndarray,
    active_window: Tuple[float, float],
    config,
    logger: Optional[logging.Logger] = None,
) -> Tuple[float, float, np.ndarray]:
    """Compute statistics for an active window in TFR data.
    
    Args:
        arr: TFR data array (freqs x times)
        times: Time points array
        active_window: Tuple of (tmin, tmax) for active window
        config: Configuration object
        logger: Optional logger instance
        
    Returns:
        Tuple of (mean, percentage_change, time_mask)
    """
    tmin_req, tmax_req = active_window
    strict_mode = get_strict_mode(config)
    if strict_mode:
        tmask = time_mask_strict(times, tmin_req, tmax_req)
    else:
        tmask = time_mask_loose(times, tmin_req, tmax_req, logger)
    mu = float(np.nanmean(arr[:, tmask]))
    pct = logratio_to_pct(mu)
    return mu, pct, tmask


def _build_filename_stem(
    name: str,
    baseline_used: Tuple[float, float],
    subject: Optional[str] = None,
    task: Optional[str] = None,
    band: Optional[str] = None,
) -> str:
    """Build filename stem with subject, task, band, and baseline information.
    
    Args:
        name: Base filename
        baseline_used: Baseline window tuple
        subject: Optional subject identifier
        task: Optional task identifier
        band: Optional frequency band identifier
        
    Returns:
        Formatted filename stem
    """
    stem = Path(name).stem
    
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


def _sanitize_label_for_filename(label: str) -> str:
    """Sanitize label for use in filenames.
    
    Args:
        label: Label string (e.g., "Condition 1", "pain", "non-pain")
        
    Returns:
        Sanitized, lowercase label suitable for filenames
    """
    sanitized = sanitize_label(label).lower()
    sanitized = sanitized.replace(" ", "_")
    return sanitized


def _get_baseline_window(config) -> Tuple[float, float]:
    """Get baseline window from config (required).
    
    Args:
        config: Configuration object
        
    Returns:
        Baseline window tuple (tmin, tmax)
    """
    baseline_window = require_config_value(config, "time_frequency_analysis.baseline_window")
    if not isinstance(baseline_window, (list, tuple)) or len(baseline_window) < 2:
        raise ValueError(
            "time_frequency_analysis.baseline_window must be a list/tuple of length 2 "
            f"(got {baseline_window!r})"
        )
    return float(baseline_window[0]), float(baseline_window[1])


def _get_output_formats(formats, config) -> List[str]:
    """Get output file formats with fallback logic.
    
    Args:
        formats: Optional list of formats
        config: Configuration object
        
    Returns:
        List of format strings
    """
    if formats is not None:
        return formats
    plot_cfg = get_plot_config(config)
    if plot_cfg.formats:
        return list(plot_cfg.formats)
    return ["png"]


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
) -> None:
    """Save figure with proper formatting and footer.
    
    Args:
        fig_obj: Matplotlib figure or list of figures
        out_dir: Output directory path
        name: Base filename
        config: Configuration object
        formats: Optional list of file formats (defaults to config formats)
        logger: Optional logger instance
        baseline_used: Optional baseline window tuple
        subject: Optional subject identifier
        task: Optional task identifier
        band: Optional frequency band identifier
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    if baseline_used is None:
        baseline_used = _get_baseline_window(config)

    figs = fig_obj if isinstance(fig_obj, list) else [fig_obj]
    stem = _build_filename_stem(name, baseline_used, subject, task, band)
    
    plot_cfg = get_plot_config(config)
    exts = _get_output_formats(formats, config)
    footer_text = None

    for i, f in enumerate(figs):
        suffix = "" if i == 0 else f"_{i+1}"
        out_name = f"{stem}{suffix}.{exts[0]}"
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
            config=config,
        )


def _plot_single_tfr_figure(
    tfr,
    central_ch: str,
    vlim: Optional[Tuple[float, float]],
    title: str,
    filename: str,
    out_dir: Path,
    config,
    logger: Optional[logging.Logger],
    baseline_used: Tuple[float, float],
    subject: Optional[str] = None,
    task: Optional[str] = None,
    band: Optional[str] = None,
) -> None:
    """Plot a single TFR figure for a specific channel.
    
    Args:
        tfr: MNE TFR object (AverageTFR)
        central_ch: Channel name to plot
        vlim: Optional value limits tuple (vmin, vmax)
        title: Figure title
        filename: Output filename
        out_dir: Output directory path
        config: Configuration object
        logger: Optional logger instance
        baseline_used: Baseline window tuple
        subject: Optional subject identifier
        task: Optional task identifier
        band: Optional frequency band identifier
    """
    font_sizes = get_font_sizes()
    plot_kwargs = {"picks": central_ch, "show": False}
    if vlim is not None:
        plot_kwargs["vlim"] = vlim
    fig = unwrap_figure(tfr.plot(**plot_kwargs))
    fig.suptitle(title, fontsize=font_sizes["figure_title"])
    _save_fig(
        fig,
        out_dir,
        filename,
        config=config,
        logger=logger,
        baseline_used=baseline_used,
        subject=subject,
        task=task,
        band=band,
    )


###################################################################
# Channel-Level Plotting Functions
###################################################################


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
    """Plot TFR for multiple channels with baseline correction.
    
    Creates plots for all specified channels (or all channels if None).
    
    Args:
        tfr: MNE TFR object (EpochsTFR or AverageTFR)
        out_dir: Output directory path
        config: Configuration object
        baseline: Baseline window tuple (tmin, tmax)
        logger: Optional logger instance
        subject: Optional subject identifier
        task: Optional task identifier
        channels: Optional list of channel names to plot (case-insensitive)
    """
    tfr_avg, baseline_used = apply_baseline_and_average(tfr, baseline, logger)

    ch_names = tfr_avg.info["ch_names"]
    if channels is not None:
        ch_names = _filter_channels_by_names(ch_names, channels, logger)
        if not ch_names:
            return
    
    ch_dir = out_dir / "channels"
    ch_dir.mkdir(parents=True, exist_ok=True)

    for ch in ch_names:
        title = f"{ch} — all trials (baseline logratio)"
        filename = f"tfr_{ch}_all_trials.png"
        _plot_single_tfr_figure(
            tfr_avg,
            ch,
            None,
            title,
            filename,
            ch_dir,
            config,
            logger,
            baseline_used,
            subject=subject,
            task=task,
        )


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
    """Plot channel-level TFR contrast between pain and non-pain conditions.
    
    Creates plots for each channel showing pain condition, non-pain condition,
    and their difference.
    
    Args:
        tfr: MNE TFR object (EpochsTFR or AverageTFR)
        events_df: Optional events DataFrame with pain column
        out_dir: Output directory path
        config: Configuration object
        baseline: Baseline window tuple (tmin, tmax)
        logger: Optional logger instance
        subject: Optional subject identifier
        channels: Optional list of channel names to plot (case-insensitive)
    """
    from .contrasts import _prepare_comparison_contrast_data

    tfr_sub, mask1, mask2, label1, label2, _ = _prepare_comparison_contrast_data(
        tfr, events_df, config, logger, context="Channel contrast"
    )
    if tfr_sub is None:
        return

    # Baseline at the epoch level before averaging to avoid bias with nonlinear modes.
    tfr_sub_stats = tfr_sub.copy()
    baseline_used = apply_baseline_and_crop(tfr_sub_stats, baseline=baseline, mode="logratio", logger=logger)

    tfr_1 = tfr_sub_stats[mask1].average()
    tfr_2 = tfr_sub_stats[mask2].average()

    tfr_diff = tfr_2.copy()
    tfr_diff.data = tfr_2.data - tfr_1.data

    ch_names = tfr_2.info["ch_names"]
    if channels is not None:
        ch_names = _filter_channels_by_names(ch_names, channels, logger)
        if not ch_names:
            return
    
    ch_dir = out_dir / "channels"
    ch_dir.mkdir(parents=True, exist_ok=True)

    label_2_sanitized = _sanitize_label_for_filename(label2)
    label_1_sanitized = _sanitize_label_for_filename(label1)

    for ch in ch_names:
        _plot_single_tfr_figure(
            tfr_2,
            ch,
            None,
            f"{ch} — {label2} (baseline logratio)",
            f"tfr_{ch}_{label_2_sanitized}.png",
            ch_dir,
            config,
            logger,
            baseline_used,
            subject=subject,
        )
        _plot_single_tfr_figure(
            tfr_1,
            ch,
            None,
            f"{ch} — {label1} (baseline logratio)",
            f"tfr_{ch}_{label_1_sanitized}.png",
            ch_dir,
            config,
            logger,
            baseline_used,
            subject=subject,
        )
        _plot_single_tfr_figure(
            tfr_diff,
            ch,
            None,
            f"{ch} — {label2} vs {label1} (logratio difference)",
            f"tfr_{ch}_{label_2_sanitized}_minus_{label_1_sanitized}.png",
            ch_dir,
            config,
            logger,
            baseline_used,
            subject=subject,
        )
