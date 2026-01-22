"""
Scalp-mean TFR plotting functions.

Functions for creating scalp-averaged time-frequency representations (TFR) plots,
including single-subject and pain contrast visualizations.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import mne

from eeg_pipeline.plotting.io.figures import (
    unwrap_figure,
    robust_sym_vlim,
    extract_eeg_picks,
    logratio_to_pct,
    save_fig as central_save_fig,
)
from eeg_pipeline.utils.formatting import format_baseline_window_string, sanitize_label
from ...utils.analysis.windowing import time_mask_loose, time_mask_strict
from ...utils.analysis.tfr import (
    apply_baseline_and_average,
    apply_baseline_and_crop,
)
from ..config import get_plot_config
from ..core.utils import get_font_sizes, log
from ..core.statistics import get_strict_mode
from ..core.topomaps import create_scalpmean_tfr_from_existing


DEFAULT_FORMAT = "png"


###################################################################
# Helper Functions
###################################################################


def _compute_active_statistics(
    tfr_data: np.ndarray,
    times: np.ndarray,
    active_window: Tuple[float, float],
    config,
    logger: Optional[logging.Logger] = None,
) -> Tuple[float, float, np.ndarray]:
    """Compute statistics for an active window in TFR data.
    
    Args:
        tfr_data: TFR data array (freqs x times)
        times: Time points array
        active_window: Tuple of (tmin, tmax) for active window
        config: Configuration object
        logger: Optional logger instance
        
    Returns:
        Tuple of (mean_logratio, percentage_change, time_mask)
    """
    tmin_required, tmax_required = active_window
    strict_mode = get_strict_mode(config)
    
    if strict_mode:
        time_mask = time_mask_strict(times, tmin_required, tmax_required)
    else:
        time_mask = time_mask_loose(times, tmin_required, tmax_required, logger)
    
    mean_logratio = float(np.nanmean(tfr_data[:, time_mask]))
    percentage_change = logratio_to_pct(mean_logratio)
    
    return mean_logratio, percentage_change, time_mask


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
    stem, _ = (name.rsplit(".", 1) + [""])[:2]
    
    header_parts = []
    if subject:
        header_parts.append(f"sub-{subject}")
    if task:
        header_parts.append(f"task-{task}")
    if band:
        header_parts.append(f"band-{band}")
    
    baseline_string = format_baseline_window_string(baseline_used)
    if baseline_string not in stem:
        stem = f"{stem}_{baseline_string}"
    
    if header_parts:
        stem = f"{'_'.join(header_parts)}_{stem}"
    
    return stem




def _save_fig(
    fig_obj,
    out_dir: Path,
    name: str,
    config,
    baseline_used: Tuple[float, float],
    logger: Optional[logging.Logger] = None,
    subject: Optional[str] = None,
    task: Optional[str] = None,
    band: Optional[str] = None,
    formats: Optional[list] = None,
    label_1: Optional[str] = None,
    label_2: Optional[str] = None,
) -> None:
    """Save figure with proper formatting and footer.
    
    Args:
        fig_obj: Matplotlib figure or list of figures
        out_dir: Output directory path
        name: Base filename
        config: Configuration object
        baseline_used: Baseline window tuple
        logger: Optional logger instance
        subject: Optional subject identifier
        task: Optional task identifier
        band: Optional frequency band identifier
        formats: Optional list of file formats (defaults to config formats)
        label_1: Optional condition label 1 (for contrast plots)
        label_2: Optional condition label 2 (for contrast plots)
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    figures = fig_obj if isinstance(fig_obj, list) else [fig_obj]
    filename_stem = _build_filename_stem(name, baseline_used, subject, task, band)
    
    plot_config = get_plot_config(config)
    file_extensions = (
        formats
        if formats
        else list(plot_config.formats) if plot_config.formats else [DEFAULT_FORMAT]
    )
    
    footer_text = None

    for index, figure in enumerate(figures):
        if index == 0:
            output_name = f"{filename_stem}.{file_extensions[0]}"
        else:
            output_name = f"{filename_stem}_{index + 1}.{file_extensions[0]}"
        
        output_path = out_dir / output_name
        central_save_fig(
            figure,
            output_path,
            logger=logger,
            footer=footer_text,
            formats=tuple(file_extensions),
            dpi=plot_config.dpi,
            bbox_inches=plot_config.bbox_inches,
            pad_inches=plot_config.pad_inches,
            config=config,
        )


def _plot_scalpmean_tfr(
    tfr_scalpmean,
    title: str,
    filename: str,
    vlim: Optional[Tuple[float, float]],
    out_dir: Path,
    config,
    baseline_used: Tuple[float, float],
    logger: Optional[logging.Logger] = None,
    subject: Optional[str] = None,
    task: Optional[str] = None,
    label_1: Optional[str] = None,
    label_2: Optional[str] = None,
) -> None:
    """Plot a scalp-mean TFR figure.
    
    Args:
        tfr_scalpmean: Scalp-mean TFR object (AverageTFR)
        title: Figure title
        filename: Output filename
        vlim: Optional value limits tuple (vmin, vmax)
        out_dir: Output directory path
        config: Configuration object
        baseline_used: Baseline window tuple
        logger: Optional logger instance
        subject: Optional subject identifier
        task: Optional task identifier
        label_1: Optional condition label 1 (for contrast plots)
        label_2: Optional condition label 2 (for contrast plots)
    """
    font_sizes = get_font_sizes()
    channel_name = tfr_scalpmean.info["ch_names"][0]
    
    plot_kwargs = {"picks": channel_name, "show": False}
    if vlim is not None:
        plot_kwargs["vlim"] = vlim
    
    figure = unwrap_figure(tfr_scalpmean.plot(**plot_kwargs))
    figure.suptitle(title, fontsize=font_sizes["figure_title"])
    
    _save_fig(
        figure,
        out_dir,
        filename,
        config,
        baseline_used,
        logger=logger,
        subject=subject,
        task=task,
        label_1=label_1,
        label_2=label_2,
    )


###################################################################
# Scalp-Mean Plotting Functions
###################################################################


def plot_scalpmean_all_trials(
    tfr,
    out_dir: Path,
    config,
    baseline: Tuple[Optional[float], Optional[float]],
    active_window: Tuple[float, float],
    logger: Optional[logging.Logger] = None,
    subject: Optional[str] = None,
    task: Optional[str] = None,
) -> None:
    """Plot scalp-averaged TFR for all trials with baseline correction.
    
    Args:
        tfr: MNE TFR object (EpochsTFR or AverageTFR)
        out_dir: Output directory path
        config: Configuration object
        baseline: Baseline window tuple (tmin, tmax)
        active_window: Active window tuple for statistics
        logger: Optional logger instance
        subject: Optional subject identifier
        task: Optional task identifier
    """
    tfr_averaged, baseline_used = apply_baseline_and_average(tfr, baseline, logger)
    
    eeg_picks = extract_eeg_picks(tfr_averaged, exclude_bads=True)
    if len(eeg_picks) == 0:
        log("No EEG channels found for scalp-averaged plot", logger, "warning")
        return
    
    tfr_scalpmean = create_scalpmean_tfr_from_existing(tfr_averaged, eeg_picks)
    
    times = np.asarray(tfr_scalpmean.times)
    tfr_data = np.asarray(tfr_scalpmean.data[0])
    absolute_vlim = robust_sym_vlim(tfr_data)
    
    _, percentage_change, _ = _compute_active_statistics(
        tfr_data, times, active_window, config, logger
    )
    
    title = (
        f"Scalp-averaged TFR — all trials (baseline logratio)\n"
        f"vlim ±{absolute_vlim:.2f}; mean %Δ vs BL={percentage_change:+.0f}%"
    )
    vlim = (-absolute_vlim, absolute_vlim)
    
    _plot_scalpmean_tfr(
        tfr_scalpmean,
        title,
        "tfr_scalpmean_all_trials.png",
        vlim,
        out_dir,
        config,
        baseline_used,
        logger=logger,
        subject=subject,
        task=task,
    )


def _create_scalpmean_contrast_plots(
    tfr_condition_1,
    tfr_condition_2,
    tfr_difference,
    label_1: str,
    label_2: str,
    times: np.ndarray,
    active_window: Tuple[float, float],
    baseline_used: Tuple[float, float],
    out_dir: Path,
    config,
    logger: Optional[logging.Logger] = None,
    subject: Optional[str] = None,
    task: Optional[str] = None,
) -> None:
    """Create and save three scalp-mean contrast plots.
    
    Args:
        tfr_condition_1: Scalp-mean TFR for condition 1
        tfr_condition_2: Scalp-mean TFR for condition 2
        tfr_difference: Scalp-mean TFR for difference (condition 2 - condition 1)
        label_1: Label for condition 1
        label_2: Label for condition 2
        times: Time points array
        active_window: Active window tuple for statistics
        baseline_used: Baseline window tuple
        out_dir: Output directory path
        config: Configuration object
        logger: Optional logger instance
        subject: Optional subject identifier
        task: Optional task identifier
    """
    tfr_data_1 = np.asarray(tfr_condition_1.data[0])
    tfr_data_2 = np.asarray(tfr_condition_2.data[0])
    tfr_data_diff = np.asarray(tfr_difference.data[0])
    
    absolute_vlim_conditions = robust_sym_vlim([tfr_data_1, tfr_data_2])
    absolute_vlim_difference = robust_sym_vlim(tfr_data_diff)
    
    _, percentage_change_1, _ = _compute_active_statistics(
        tfr_data_1, times, active_window, config, logger
    )
    _, percentage_change_2, _ = _compute_active_statistics(
        tfr_data_2, times, active_window, config, logger
    )
    _, percentage_change_diff, _ = _compute_active_statistics(
        tfr_data_diff, times, active_window, config, logger
    )
    
    vlim_conditions = (-absolute_vlim_conditions, absolute_vlim_conditions)
    vlim_difference = (-absolute_vlim_difference, absolute_vlim_difference)
    
    title_condition_2 = (
        f"Scalp-averaged TFR — {label_2} (baseline logratio)\n"
        f"vlim ±{absolute_vlim_conditions:.2f}; mean %Δ vs BL={percentage_change_2:+.0f}%"
    )
    title_condition_1 = (
        f"Scalp-averaged TFR — {label_1} (baseline logratio)\n"
        f"vlim ±{absolute_vlim_conditions:.2f}; mean %Δ vs BL={percentage_change_1:+.0f}%"
    )
    title_difference = (
        f"Scalp-averaged TFR — {label_2} vs {label_1} (logratio difference)\n"
        f"vlim ±{absolute_vlim_difference:.2f}; mean %Δ {label_2} vs {label_1}={percentage_change_diff:+.0f}%"
    )
    
    label_2_sanitized = _sanitize_label_for_filename(label_2)
    label_1_sanitized = _sanitize_label_for_filename(label_1)
    
    _plot_scalpmean_tfr(
        tfr_condition_2,
        title_condition_2,
        f"tfr_scalpmean_{label_2_sanitized}.png",
        vlim_conditions,
        out_dir,
        config,
        baseline_used,
        logger=logger,
        subject=subject,
        task=task,
        label_1=label_1,
        label_2=label_2,
    )
    _plot_scalpmean_tfr(
        tfr_condition_1,
        title_condition_1,
        f"tfr_scalpmean_{label_1_sanitized}.png",
        vlim_conditions,
        out_dir,
        config,
        baseline_used,
        logger=logger,
        subject=subject,
        task=task,
        label_1=label_1,
        label_2=label_2,
    )
    _plot_scalpmean_tfr(
        tfr_difference,
        title_difference,
        f"tfr_scalpmean_{label_2_sanitized}_minus_{label_1_sanitized}.png",
        vlim_difference,
        out_dir,
        config,
        baseline_used,
        logger=logger,
        subject=subject,
        task=task,
        label_1=label_1,
        label_2=label_2,
    )


def contrast_scalpmean_pain_nonpain(
    tfr: "mne.time_frequency.EpochsTFR | mne.time_frequency.AverageTFR",
    events_df: Optional[pd.DataFrame],
    out_dir: Path,
    config,
    baseline: Tuple[Optional[float], Optional[float]],
    active_window: Tuple[float, float],
    logger: Optional[logging.Logger] = None,
    subject: Optional[str] = None,
    task: Optional[str] = None,
) -> None:
    """Plot scalp-averaged TFR contrast between pain and non-pain conditions.
    
    Creates three plots: pain condition, non-pain condition, and their difference.
    
    Args:
        tfr: MNE TFR object (EpochsTFR or AverageTFR)
        events_df: Optional events DataFrame with pain column
        out_dir: Output directory path
        config: Configuration object
        baseline: Baseline window tuple (tmin, tmax)
        active_window: Active window tuple for statistics
        logger: Optional logger instance
        subject: Optional subject identifier
        task: Optional task identifier
    """
    from .contrasts import _prepare_comparison_contrast_data

    tfr_subset, mask_1, mask_2, label_1, label_2, _ = _prepare_comparison_contrast_data(
        tfr, events_df, config, logger, context="Scalpmean contrast"
    )
    if tfr_subset is None:
        return

    # Baseline at the epoch level before averaging to avoid bias with nonlinear baseline modes.
    tfr_epochs_1 = tfr_subset[mask_1].copy()
    tfr_epochs_2 = tfr_subset[mask_2].copy()

    baseline_used = apply_baseline_and_crop(
        tfr_epochs_1, baseline=baseline, mode="logratio", logger=logger
    )
    apply_baseline_and_crop(
        tfr_epochs_2, baseline=baseline_used, mode="logratio", logger=logger
    )

    tfr_condition_1 = tfr_epochs_1.average()
    tfr_condition_2 = tfr_epochs_2.average()

    eeg_picks_1 = extract_eeg_picks(tfr_condition_1, exclude_bads=True)
    eeg_picks_2 = extract_eeg_picks(tfr_condition_2, exclude_bads=True)
    
    if len(eeg_picks_1) == 0 or len(eeg_picks_2) == 0:
        log("No EEG channels found for scalp-averaged contrast", logger, "warning")
        return
    
    tfr_scalpmean_1 = create_scalpmean_tfr_from_existing(tfr_condition_1, eeg_picks_1)
    tfr_scalpmean_2 = create_scalpmean_tfr_from_existing(tfr_condition_2, eeg_picks_2)
    
    tfr_difference = tfr_scalpmean_2.copy()
    tfr_difference.data = tfr_scalpmean_2.data - tfr_scalpmean_1.data
    tfr_difference.comment = "cond2-minus-cond1"
    
    times = np.asarray(tfr_condition_1.times)
    
    _create_scalpmean_contrast_plots(
        tfr_scalpmean_1,
        tfr_scalpmean_2,
        tfr_difference,
        label_1,
        label_2,
        times,
        active_window,
        baseline_used,
        out_dir,
        config,
        logger=logger,
        subject=subject,
        task=task,
    )
