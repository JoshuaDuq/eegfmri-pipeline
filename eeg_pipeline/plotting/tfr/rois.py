"""
ROI TFR plotting functions.

Functions for computing and plotting time-frequency representations (TFR) 
at the region of interest (ROI) level.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import mne

from eeg_pipeline.plotting.io.figures import unwrap_figure
from eeg_pipeline.utils.formatting import sanitize_label
from eeg_pipeline.utils.config.loader import get_config_value
from ...utils.analysis.tfr import (
    apply_baseline_and_crop,
    resolve_tfr_workers,
)
from ..core.utils import get_font_sizes, log
from .channels import _save_fig
from .contrasts import _prepare_comparison_contrast_data


###################################################################
# ROI Computation and Plotting Functions
###################################################################


def _filter_rois_by_config(
    roi_tfrs: Dict[str, mne.time_frequency.EpochsTFR],
    config,
) -> Dict[str, mne.time_frequency.EpochsTFR]:
    """Filter ROI TFRs based on comparison_rois configuration.
    
    Args:
        roi_tfrs: Dictionary mapping ROI names to EpochsTFR objects
        config: Configuration object
    
    Returns:
        Filtered dictionary containing only requested ROIs, or original
        if no specific ROIs are requested.
    """
    comparison_rois = get_config_value(
        config, "plotting.comparisons.comparison_rois", []
    )
    has_specific_rois = any(roi.lower() != "all" for roi in comparison_rois)
    
    if not comparison_rois or not has_specific_rois:
        return roi_tfrs
    
    filtered_rois = {}
    for roi_name in comparison_rois:
        if roi_name.lower() == "all":
            continue
        if roi_name in roi_tfrs:
            filtered_rois[roi_name] = roi_tfrs[roi_name]
    
    return filtered_rois


def compute_roi_tfrs(
    epochs: mne.Epochs,
    freqs: np.ndarray,
    n_cycles: np.ndarray,
    config,
    roi_map: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, mne.time_frequency.EpochsTFR]:
    """Compute TFR for each ROI by averaging channels within each ROI.
    
    Args:
        epochs: MNE Epochs object
        freqs: Frequency array for TFR computation
        n_cycles: Number of cycles array for TFR computation
        config: Configuration object
        roi_map: Optional ROI map dictionary mapping ROI names to channel lists.
                 If None, builds ROIs from config.
    
    Returns:
        Dictionary mapping ROI names to EpochsTFR objects
    """
    if roi_map is None:
        from ...utils.analysis.tfr import build_rois_from_info as _build_rois
        roi_map = _build_rois(epochs.info, config=config)
    
    roi_tfrs = {}
    for roi, channel_names in roi_map.items():
        picks = mne.pick_channels(epochs.ch_names, include=channel_names, ordered=True)
        if len(picks) == 0:
            continue
        
        epochs_data = epochs.get_data()
        roi_data = np.nanmean(epochs_data[:, picks, :], axis=1, keepdims=True)
        
        roi_info = mne.create_info(
            [roi], sfreq=epochs.info['sfreq'], ch_types='eeg'
        )
        roi_epochs = mne.EpochsArray(
            roi_data,
            roi_info,
            events=epochs.events,
            event_id=epochs.event_id,
            tmin=epochs.tmin,
            metadata=epochs.metadata,
            verbose=False,
        )
        
        workers_default = (
            int(config.get("time_frequency_analysis.tfr.workers", -1))
            if config else -1
        )
        workers = resolve_tfr_workers(workers_default=workers_default)
        decim = (
            config.get("time_frequency_analysis.tfr.decim", 4) if config else 4
        )
        
        power = roi_epochs.compute_tfr(
            method="morlet",
            freqs=freqs,
            n_cycles=n_cycles,
            decim=decim,
            picks="eeg",
            use_fft=True,
            return_itc=False,
            average=False,
            n_jobs=workers,
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
    """Plot TFR for all ROIs with baseline correction.
    
    Creates plots for each ROI showing the averaged TFR across all trials
    with baseline correction applied.
    
    Args:
        roi_tfrs: Dictionary mapping ROI names to EpochsTFR objects
        out_dir: Output directory path
        config: Configuration object
        baseline: Baseline window tuple (tmin, tmax)
        logger: Optional logger instance
    """
    roi_tfrs = _filter_rois_by_config(roi_tfrs, config)
    rois_dir = out_dir / "rois"
    font_sizes = get_font_sizes()
    
    for roi, tfr in roi_tfrs.items():
        tfr_averaged = tfr.copy().average()
        baseline_used = apply_baseline_and_crop(
            tfr_averaged, baseline=baseline, mode="logratio", logger=logger
        )
        
        channel_name = tfr_averaged.info['ch_names'][0]
        roi_tag = sanitize_label(roi)
        roi_dir = rois_dir / roi_tag

        figure = unwrap_figure(tfr_averaged.plot(picks=channel_name, show=False))
        figure.suptitle(
            f"ROI: {roi} — all trials (baseline logratio)",
            fontsize=font_sizes["figure_title"]
        )
        _save_fig(
            figure, roi_dir, "tfr_all_trials_bl.png",
            config=config, logger=logger, baseline_used=baseline_used
        )


def _plot_roi_tfr_figure(
    tfr: mne.time_frequency.AverageTFR,
    roi_name: str,
    title_suffix: str,
    output_dir: Path,
    filename: str,
    config,
    baseline_used: Tuple[Optional[float], Optional[float]],
    font_sizes: Dict[str, int],
    logger: Optional[logging.Logger] = None,
) -> None:
    """Plot a single ROI TFR figure and save it.
    
    Args:
        tfr: AverageTFR object to plot
        roi_name: Name of the ROI
        title_suffix: Suffix for the figure title
        output_dir: Directory to save the figure
        filename: Output filename
        config: Configuration object
        baseline_used: Baseline window tuple used
        font_sizes: Dictionary of font sizes
        logger: Optional logger instance
    """
    channel_name = tfr.info['ch_names'][0]
    figure = unwrap_figure(tfr.plot(picks=channel_name, show=False))
    figure.suptitle(
        f"ROI: {roi_name} — {title_suffix} (baseline logratio)",
        fontsize=font_sizes["figure_title"]
    )
    _save_fig(
        figure, output_dir, filename,
        config=config, logger=logger, baseline_used=baseline_used
    )


def contrast_pain_nonpain_rois(
    roi_tfrs: Dict[str, mne.time_frequency.EpochsTFR],
    events_df: Optional[pd.DataFrame],
    out_dir: Path,
    config,
    baseline: Tuple[Optional[float], Optional[float]],
    logger: Optional[logging.Logger] = None,
) -> None:
    """Plot ROI-level TFR contrast between pain and non-pain conditions.
    
    Creates plots for each ROI showing pain condition, non-pain condition,
    and their difference.
    
    Args:
        roi_tfrs: Dictionary mapping ROI names to EpochsTFR objects
        events_df: Optional events DataFrame with pain column
        out_dir: Output directory path
        config: Configuration object
        baseline: Baseline window tuple (tmin, tmax)
        logger: Optional logger instance
    """
    roi_tfrs = _filter_rois_by_config(roi_tfrs, config)
    rois_dir = out_dir / "rois"
    font_sizes = get_font_sizes()
    
    for roi, tfr in roi_tfrs.items():
        try:
            tfr_subset, mask1, mask2, label1, label2, _ = (
                _prepare_comparison_contrast_data(
                    tfr, events_df, config, logger,
                    context=f"ROI {roi} contrast"
                )
            )
            if tfr_subset is None:
                continue

            baseline_used = apply_baseline_and_crop(
                tfr_subset, baseline=baseline, mode="logratio", logger=logger
            )
            tfr_condition2 = tfr_subset[mask2].average()
            tfr_condition1 = tfr_subset[mask1].average()

            roi_tag = sanitize_label(roi)
            roi_dir = rois_dir / roi_tag

            _plot_roi_tfr_figure(
                tfr_condition2, roi, label2, roi_dir,
                "tfr_painful_bl.png", config, baseline_used,
                font_sizes, logger
            )

            _plot_roi_tfr_figure(
                tfr_condition1, roi, label1, roi_dir,
                "tfr_nonpain_bl.png", config, baseline_used,
                font_sizes, logger
            )

            tfr_difference = tfr_condition2.copy()
            tfr_difference.data = tfr_condition2.data - tfr_condition1.data
            difference_title = f"{label2} minus {label1}"
            _plot_roi_tfr_figure(
                tfr_difference, roi, difference_title, roi_dir,
                "tfr_pain_minus_nonpain_bl.png", config, baseline_used,
                font_sizes, logger
            )
        except (FileNotFoundError, ValueError, RuntimeError, KeyError, IndexError) as exc:
            log(
                f"ROI {roi}: error while computing ROI contrasts ({exc})",
                logger, "error"
            )
            continue
