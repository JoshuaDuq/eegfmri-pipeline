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
from ..config import get_plot_config
from ..core.utils import get_font_sizes, log
from ..core.statistics import get_strict_mode
from .channels import _save_fig
from .contrasts import _prepare_comparison_contrast_data


###################################################################
# ROI Computation and Plotting Functions
###################################################################


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
        workers_default = int(config.get("time_frequency_analysis.tfr.workers", -1)) if config else -1
        workers = resolve_tfr_workers(workers_default=workers_default)
        power = epo_roi.compute_tfr(
            method="morlet",
            freqs=freqs,
            n_cycles=n_cycles,
            decim=config.get("time_frequency_analysis.tfr.decim", 4) if config else 4,
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
    # Filter ROIs based on config
    comp_rois = get_config_value(config, "plotting.comparisons.comparison_rois", [])
    has_specific = any(r.lower() != "all" for r in comp_rois)
    if comp_rois and has_specific:
        filtered_rois = {}
        for r in comp_rois:
            if r.lower() == "all": continue
            if r in roi_tfrs:
                filtered_rois[r] = roi_tfrs[r]
        roi_tfrs = filtered_rois

    rois_dir = out_dir / "rois"
    for roi, tfr in roi_tfrs.items():
        tfr_c = tfr.copy()
        tfr_avg = tfr_c.average()
        baseline_used = apply_baseline_and_crop(tfr_avg, baseline=baseline, mode="logratio", logger=logger)
        
        ch = tfr_avg.info['ch_names'][0]
        roi_tag = sanitize_label(roi)
        roi_dir = rois_dir / roi_tag

        fig = unwrap_figure(tfr_avg.plot(picks=ch, show=False))
        font_sizes = get_font_sizes()
        fig.suptitle(f"ROI: {roi} — all trials (baseline logratio)", fontsize=font_sizes["figure_title"])
        _save_fig(fig, roi_dir, "tfr_all_trials_bl.png", config=config, logger=logger, baseline_used=baseline_used)


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
    # Filter ROIs based on config
    comp_rois = get_config_value(config, "plotting.comparisons.comparison_rois", [])
    has_specific = any(r.lower() != "all" for r in comp_rois)
    if comp_rois and has_specific:
        filtered_rois = {}
        for r in comp_rois:
            if r.lower() == "all": continue
            if r in roi_tfrs:
                filtered_rois[r] = roi_tfrs[r]
        roi_tfrs = filtered_rois

    rois_dir = out_dir / "rois"
    for roi, tfr in roi_tfrs.items():
        try:
            tfr_sub, mask1, mask2, label1, label2, _ = _prepare_comparison_contrast_data(
                tfr, events_df, config, logger, context=f"ROI {roi} contrast"
            )
            if tfr_sub is None:
                continue

            baseline_used = apply_baseline_and_crop(tfr_sub, baseline=baseline, mode="logratio", logger=logger)
            tfr_2 = tfr_sub[mask2].average()
            tfr_1 = tfr_sub[mask1].average()

            ch = tfr_2.info['ch_names'][0]
            roi_tag = sanitize_label(roi)
            roi_dir = rois_dir / roi_tag

            fig = unwrap_figure(tfr_2.plot(picks=ch, show=False))
            font_sizes = get_font_sizes()
            fig.suptitle(f"ROI: {roi} — {label2} (baseline logratio)", fontsize=font_sizes["figure_title"])
            _save_fig(fig, roi_dir, "tfr_painful_bl.png", config=config, logger=logger, baseline_used=baseline_used)

            fig = unwrap_figure(tfr_1.plot(picks=ch, show=False))
            font_sizes = get_font_sizes()
            fig.suptitle(f"ROI: {roi} — {label1} (baseline logratio)", fontsize=font_sizes["figure_title"])
            _save_fig(fig, roi_dir, "tfr_nonpain_bl.png", config=config, logger=logger, baseline_used=baseline_used)

            tfr_diff = tfr_2.copy()
            tfr_diff.data = tfr_2.data - tfr_1.data
            fig = unwrap_figure(tfr_diff.plot(picks=ch, show=False))
            font_sizes = get_font_sizes()
            fig.suptitle(f"ROI: {roi} — {label2} minus {label1} (baseline logratio)", fontsize=font_sizes["figure_title"])
            _save_fig(fig, roi_dir, "tfr_pain_minus_nonpain_bl.png", config=config, logger=logger, baseline_used=baseline_used)
        except (FileNotFoundError, ValueError, RuntimeError, KeyError, IndexError) as exc:
            log(f"ROI {roi}: error while computing ROI contrasts ({exc})", logger, "error")
            continue
