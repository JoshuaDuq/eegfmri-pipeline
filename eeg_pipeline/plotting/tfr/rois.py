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

from ...utils.io.general import (
    unwrap_figure,
    sanitize_label,
    get_pain_column_from_config,
    ensure_aligned_lengths,
)
from ...utils.analysis.tfr import (
    apply_baseline_and_crop,
    create_tfr_subset,
    resolve_tfr_workers,
)
from ...utils.data.loading import (
    compute_aligned_data_length,
    extract_pain_vector_array,
)
from ..config import get_plot_config
from ..core.utils import get_font_sizes, log
from ..core.statistics import get_strict_mode
from .channels import _save_fig
from .contrasts import _create_pain_masks_from_vector, _align_and_trim_masks


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
        workers_default = int(config.get("tfr_topography_pipeline.tfr.workers", -1)) if config else -1
        workers = resolve_tfr_workers(workers_default=workers_default)
        power = epo_roi.compute_tfr(
            method="morlet",
            freqs=freqs,
            n_cycles=n_cycles,
            decim=config.get("tfr_topography_pipeline.tfr.decim", 4) if config else 4,
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
    pain_col = get_pain_column_from_config(config, events_df)
    if pain_col is None:
        log(f"Events with pain binary column required for ROI contrasts; skipping.", logger, "warning")
        return

    rois_dir = out_dir / "rois"
    for roi, tfr in roi_tfrs.items():
        try:
            n_epochs = tfr.data.shape[0]
            n_meta = len(events_df) if events_df is not None else n_epochs
            n = compute_aligned_data_length(tfr, events_df)
            if n_epochs != n_meta:
                log(f"ROI {roi}: trimming to {n} epochs to match events.", logger)

            pain_vec = extract_pain_vector_array(tfr, events_df, pain_col, n)
            if pain_vec is None:
                continue
            
            pain_mask, non_mask = _create_pain_masks_from_vector(pain_vec)
            if pain_mask is None:
                continue
            
            if pain_mask.sum() == 0 or non_mask.sum() == 0:
                log(f"ROI {roi}: one group has zero trials; skipping.", logger, "warning")
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
            font_sizes = get_font_sizes()
            fig.suptitle(f"ROI: {roi} — Painful (baseline logratio)", fontsize=font_sizes["figure_title"])
            _save_fig(fig, roi_dir, "tfr_painful_bl.png", config=config, logger=logger, baseline_used=baseline_used)

            fig = unwrap_figure(tfr_non.plot(picks=ch, show=False))
            font_sizes = get_font_sizes()
            fig.suptitle(f"ROI: {roi} — Non-pain (baseline logratio)", fontsize=font_sizes["figure_title"])
            _save_fig(fig, roi_dir, "tfr_nonpain_bl.png", config=config, logger=logger, baseline_used=baseline_used)

            tfr_diff = tfr_pain.copy()
            tfr_diff.data = tfr_pain.data - tfr_non.data
            fig = unwrap_figure(tfr_diff.plot(picks=ch, show=False))
            font_sizes = get_font_sizes()
            fig.suptitle(f"ROI: {roi} — Pain minus Non-pain (baseline logratio)", fontsize=font_sizes["figure_title"])
            _save_fig(fig, roi_dir, "tfr_pain_minus_nonpain_bl.png", config=config, logger=logger, baseline_used=baseline_used)
        except (FileNotFoundError, ValueError, RuntimeError, KeyError, IndexError) as exc:
            log(f"ROI {roi}: error while computing ROI contrasts ({exc})", logger, "error")
            continue

