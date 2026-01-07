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
    build_footer,
    save_fig as central_save_fig,
)
from eeg_pipeline.utils.formatting import format_baseline_window_string
from ...utils.analysis.windowing import time_mask_loose, time_mask_strict
from ...utils.analysis.tfr import (
    apply_baseline_and_average,
    apply_baseline_and_crop,
)
from ..config import get_plot_config
from ..core.utils import get_font_sizes, log
from ..core.statistics import get_strict_mode
from ..core.topomaps import create_scalpmean_tfr_from_existing


###################################################################
# Helper Functions
###################################################################


def _compute_active_statistics(
    arr: np.ndarray,
    times: np.ndarray,
    active_window: Tuple[float, float],
    config,
    logger: Optional[logging.Logger] = None,
) -> Tuple[float, float, np.ndarray]:
    """Compute statistics for a active window in TFR data.
    
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
    stem, _ = (name.rsplit(".", 1) + [""])[:2]
    
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


def _build_footer_text(config, baseline_used: Tuple[float, float]) -> Optional[str]:
    """Build footer text for TFR plots with baseline information.
    
    Args:
        config: Configuration object
        baseline_used: Baseline window tuple
        
    Returns:
        Footer text string or None if config doesn't support it
    """
    default_footer_template = "tfr_baseline"
    baseline_decimal_places = 2
    
    if not hasattr(config, "get"):
        return None
    
    template_name = config.get("output.tfr_footer_template", default_footer_template)
    footer_kwargs = {
        "baseline_window": baseline_used,
        "baseline": f"[{float(baseline_used[0]):.{baseline_decimal_places}f}, {float(baseline_used[1]):.{baseline_decimal_places}f}] s",
    }
    return build_footer(template_name, config, **footer_kwargs)


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
        override = config.get("plotting.tfr.default_baseline_window", None)
        if isinstance(override, (list, tuple)) and len(override) == 2:
            baseline_used = tuple(override)
        else:
            baseline_used = tuple(config.get("time_frequency_analysis.baseline_window", [-5.0, -0.01]))

    figs = fig_obj if isinstance(fig_obj, list) else [fig_obj]
    stem = _build_filename_stem(name, baseline_used, subject, task, band)
    
    plot_cfg = get_plot_config(config)
    exts = formats if formats else list(plot_cfg.formats) if plot_cfg.formats else ["png"]
    
    footer_text = _build_footer_text(config, baseline_used)

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


def _plot_scalpmean_tfr(
    tfr_sm,
    title: str,
    filename: str,
    vlim: Optional[Tuple[float, float]],
    out_dir: Path,
    config,
    logger: Optional[logging.Logger],
    baseline_used: Tuple[float, float],
    subject: Optional[str] = None,
    task: Optional[str] = None,
) -> None:
    """Plot a scalp-mean TFR figure.
    
    Args:
        tfr_sm: Scalp-mean TFR object (AverageTFR)
        title: Figure title
        filename: Output filename
        vlim: Optional value limits tuple (vmin, vmax)
        out_dir: Output directory path
        config: Configuration object
        logger: Optional logger instance
        baseline_used: Baseline window tuple
        subject: Optional subject identifier
        task: Optional task identifier
    """
    font_sizes = get_font_sizes()
    ch_name = tfr_sm.info['ch_names'][0]
    plot_kwargs = {"picks": ch_name, "show": False}
    if vlim is not None:
        plot_kwargs["vlim"] = vlim
    fig = unwrap_figure(tfr_sm.plot(**plot_kwargs))
    fig.suptitle(title, fontsize=font_sizes["figure_title"])
    _save_fig(fig, out_dir, filename, config=config, logger=logger, baseline_used=baseline_used, subject=subject, task=task)


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
    tfr_avg, baseline_used = apply_baseline_and_average(tfr, baseline, logger)
    
    eeg_picks = extract_eeg_picks(tfr_avg, exclude_bads=False)
    if len(eeg_picks) == 0:
        log("No EEG channels found for scalp-averaged plot", logger, "warning")
        return
    
    tfr_sm = create_scalpmean_tfr_from_existing(tfr_avg, eeg_picks)
    
    times = np.asarray(tfr_sm.times)
    arr = np.asarray(tfr_sm.data[0])
    vabs = robust_sym_vlim(arr)
    _, pct, _ = _compute_active_statistics(arr, times, active_window, config, logger)
    
    _plot_scalpmean_tfr(
        tfr_sm,
        f"Scalp-averaged TFR — all trials (baseline logratio)\nvlim ±{vabs:.2f}; mean %Δ vs BL={pct:+.0f}%",
        "tfr_scalpmean_all_trials.png",
        (-vabs, +vabs),
        out_dir, config, logger, baseline_used, subject, task
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

    tfr_sub, mask1, mask2, label1, label2, _ = _prepare_comparison_contrast_data(
        tfr, events_df, config, logger, context="Scalpmean contrast"
    )
    if tfr_sub is None:
        return

    tfr_1 = tfr_sub[mask1].average()
    tfr_2 = tfr_sub[mask2].average()

    baseline_used = apply_baseline_and_crop(tfr_1, baseline=baseline, mode="logratio", logger=logger)
    apply_baseline_and_crop(tfr_2, baseline=baseline, mode="logratio", logger=logger)

    eeg_picks_1 = extract_eeg_picks(tfr_1, exclude_bads=False)
    eeg_picks_2 = extract_eeg_picks(tfr_2, exclude_bads=False)
    
    if len(eeg_picks_1) == 0 or len(eeg_picks_2) == 0:
        log("No EEG channels found for scalp-averaged contrast", logger, "warning")
        return
    
    tfr_1_sm = create_scalpmean_tfr_from_existing(tfr_1, eeg_picks_1)
    tfr_2_sm = create_scalpmean_tfr_from_existing(tfr_2, eeg_picks_2)
    tfr_diff_sm = tfr_2_sm.copy()
    tfr_diff_sm.data = tfr_2_sm.data - tfr_1_sm.data
    tfr_diff_sm.comment = "cond2-minus-cond1"
    
    arr_1 = np.asarray(tfr_1_sm.data[0])
    arr_2 = np.asarray(tfr_2_sm.data[0])
    arr_diff = np.asarray(tfr_diff_sm.data[0])
    
    vabs_pn = robust_sym_vlim([arr_1, arr_2])
    vabs_diff = robust_sym_vlim(arr_diff)

    times = np.asarray(tfr_1.times)
    _, pct_1, _ = _compute_active_statistics(arr_1, times, active_window, config, logger)
    _, pct_2, _ = _compute_active_statistics(arr_2, times, active_window, config, logger)
    _, pct_diff, _ = _compute_active_statistics(arr_diff, times, active_window, config, logger)

    _plot_scalpmean_tfr(
        tfr_2_sm, f"Scalp-averaged TFR — {label2} (baseline logratio)\nvlim ±{vabs_pn:.2f}; mean %Δ vs BL={pct_2:+.0f}%",
        "tfr_scalpmean_pain_bl.png", (-vabs_pn, +vabs_pn), out_dir, config, logger, baseline_used, subject, task
    )
    _plot_scalpmean_tfr(
        tfr_1_sm, f"Scalp-averaged TFR — {label1} (baseline logratio)\nvlim ±{vabs_pn:.2f}; mean %Δ vs BL={pct_1:+.0f}%",
        "tfr_scalpmean_nonpain_bl.png", (-vabs_pn, +vabs_pn), out_dir, config, logger, baseline_used, subject, task
    )
    _plot_scalpmean_tfr(
        tfr_diff_sm, f"Scalp-averaged TFR — {label2} minus {label1} (baseline logratio)\nvlim ±{vabs_diff:.2f}; mean %Δ vs BL={pct_diff:+.0f}%",
        "tfr_scalpmean_pain_minus_non_bl.png", (-vabs_diff, +vabs_diff), out_dir, config, logger, baseline_used, subject, task
    )
