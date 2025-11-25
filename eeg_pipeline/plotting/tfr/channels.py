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
import mne

from ...utils.io.general import (
    unwrap_figure,
    robust_sym_vlim,
    extract_eeg_picks,
    format_baseline_window_string,
    logratio_to_pct,
    build_footer,
    save_fig as central_save_fig,
    get_pain_column_from_config,
    require_epochs_tfr,
    ensure_aligned_lengths,
)
from ...utils.analysis.tfr import (
    apply_baseline_and_average,
    apply_baseline_and_crop,
    create_tfr_subset,
    get_bands_for_tfr,
    create_time_mask_strict,
    create_time_mask_loose,
)
from ...utils.data.loading import (
    compute_aligned_data_length,
    extract_pain_vector,
)
from ..config import get_plot_config
from ..core.utils import get_font_sizes, log
from ..core.statistics import get_strict_mode


###################################################################
# Helper Functions
###################################################################


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
    
    picks = extract_eeg_picks(info, exclude_bads=False)
    if len(picks) == 0:
        raise RuntimeError("No EEG channels available for plotting.")
    
    fallback = ch_names[picks[0]]
    log(f"Channel '{preferred}' not found; using '{fallback}' instead.", logger, "warning")
    return fallback


def _compute_plateau_statistics(
    arr: np.ndarray,
    times: np.ndarray,
    plateau_window: Tuple[float, float],
    config,
    logger: Optional[logging.Logger] = None,
) -> Tuple[float, float, np.ndarray]:
    """Compute statistics for a plateau window in TFR data.
    
    Args:
        arr: TFR data array (freqs x times)
        times: Time points array
        plateau_window: Tuple of (tmin, tmax) for plateau window
        config: Configuration object
        logger: Optional logger instance
        
    Returns:
        Tuple of (mean, percentage_change, time_mask)
    """
    tmin_req, tmax_req = plateau_window
    strict_mode = get_strict_mode(config)
    if strict_mode:
        tmask = create_time_mask_strict(times, tmin_req, tmax_req)
    else:
        tmask = create_time_mask_loose(times, tmin_req, tmax_req, logger)
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
        plot_cfg = get_plot_config(config)
        tfr_config = plot_cfg.plot_type_configs.get("tfr", {})
        default_baseline_start = tfr_config.get("default_baseline_start", -5.0)
        default_baseline_end = tfr_config.get("default_baseline_end", -0.01)
        default_baseline_window = [default_baseline_start, default_baseline_end]
        baseline_used = tuple(config.get("time_frequency_analysis.baseline_window", default_baseline_window))

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
    _save_fig(fig, out_dir, filename, config=config, logger=logger, baseline_used=baseline_used, subject=subject, task=task, band=band)


###################################################################
# Channel-Level Plotting Functions
###################################################################


def plot_cz_all_trials_raw(
    tfr,
    out_dir: Path,
    config,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Plot raw TFR for central channel (Cz) without baseline correction.
    
    Args:
        tfr: MNE TFR object (EpochsTFR or AverageTFR)
        out_dir: Output directory path
        config: Configuration object
        logger: Optional logger instance
    """
    tfr_avg = tfr.copy().average() if isinstance(tfr, mne.time_frequency.EpochsTFR) else tfr.copy()
    central_ch = _pick_central_channel(tfr_avg.info, preferred="Cz", logger=logger)
    fig = unwrap_figure(tfr_avg.plot(picks=central_ch, show=False))
    font_sizes = get_font_sizes()
    fig.suptitle(f"{central_ch} TFR — all trials (raw, no baseline)", fontsize=font_sizes["figure_title"])
    _save_fig(fig, out_dir, f"tfr_{central_ch}_all_trials_raw.png", config=config, logger=logger)


def plot_cz_all_trials(
    tfr,
    out_dir: Path,
    config,
    baseline: Tuple[Optional[float], Optional[float]],
    plateau_window: Tuple[float, float],
    logger: Optional[logging.Logger] = None,
    subject: Optional[str] = None,
    task: Optional[str] = None,
) -> None:
    """Plot TFR for central channel (Cz) with baseline correction.
    
    Args:
        tfr: MNE TFR object (EpochsTFR or AverageTFR)
        out_dir: Output directory path
        config: Configuration object
        baseline: Baseline window tuple (tmin, tmax)
        plateau_window: Plateau window tuple for statistics
        logger: Optional logger instance
        subject: Optional subject identifier
        task: Optional task identifier
    """
    tfr_avg, baseline_used = apply_baseline_and_average(tfr, baseline, logger)

    central_ch = _pick_central_channel(tfr_avg.info, preferred="Cz", logger=logger)
    ch_idx = tfr_avg.info["ch_names"].index(central_ch)
    arr = np.asarray(tfr_avg.data[ch_idx])
    vabs = robust_sym_vlim(arr)
    times = np.asarray(tfr_avg.times)
    _, pct, _ = _compute_plateau_statistics(arr, times, plateau_window, config, logger)

    _plot_single_tfr_figure(
        tfr_avg, central_ch, (-vabs, +vabs),
        f"{central_ch} TFR — all trials (baseline logratio)\nvlim ±{vabs:.2f}; mean %Δ vs BL={pct:+.0f}%",
        f"tfr_{central_ch}_all_trials.png",
        out_dir, config, logger, baseline_used
    )


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
    
    Creates plots for all specified channels (or all channels if None) and generates
    frequency band-specific plots for each channel.
    
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
        channels_set = {ch.upper() for ch in channels}
        ch_names = [ch for ch in ch_names if ch.upper() in channels_set]
        if not ch_names:
            if logger:
                logger.warning(f"No matching channels found for specified channels: {channels}")
            return
    
    ch_dir = out_dir / "channels"
    ch_dir.mkdir(parents=True, exist_ok=True)

    fmax_available = float(np.max(tfr_avg.freqs))
    bands = get_bands_for_tfr(max_freq_available=fmax_available, config=config)

    for ch in ch_names:
        _plot_single_tfr_figure(
            tfr_avg, ch, None, f"{ch} — all trials (baseline logratio)",
            f"tfr_{ch}_all_trials.png", ch_dir, config, logger, baseline_used,
            subject=subject, task=task
        )

        for band, (fmin, fmax) in bands.items():
            fmax_eff = min(fmax, fmax_available)
            if fmin >= fmax_eff:
                continue
            band_dir = ch_dir / band
            band_dir.mkdir(parents=True, exist_ok=True)

            tfr_band = tfr_avg.copy()
            freq_mask = (np.asarray(tfr_band.freqs) >= fmin) & (np.asarray(tfr_band.freqs) <= fmax_eff)
            if not freq_mask.any():
                continue
            
            fig = unwrap_figure(tfr_band.plot(picks=ch, fmin=fmin, fmax=fmax_eff, show=False))
            font_sizes = get_font_sizes()
            fig.suptitle(f"{ch} — {band} band (baseline logratio)", fontsize=font_sizes["figure_title"])
            _save_fig(
                fig, band_dir, f"tfr_{ch}_{band}_all_trials.png",
                config=config, logger=logger, baseline_used=baseline_used,
                subject=subject, task=task, band=band
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
    and their difference, with optional frequency band-specific plots.
    
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
    from .scalpmean import _prepare_pain_contrast_data
    
    pain_col = get_pain_column_from_config(config, events_df)
    tfr_sub, pain_mask, non_mask, n = _prepare_pain_contrast_data(tfr, events_df, pain_col, config, logger)
    if tfr_sub is None:
        return

    tfr_pain = tfr_sub[pain_mask].average()
    tfr_non = tfr_sub[non_mask].average()

    baseline_used = apply_baseline_and_crop(tfr_pain, baseline=baseline, mode="logratio", logger=logger)
    apply_baseline_and_crop(tfr_non, baseline=baseline, mode="logratio", logger=logger)

    tfr_diff = tfr_pain.copy()
    tfr_diff.data = tfr_pain.data - tfr_non.data

    ch_names = tfr_pain.info["ch_names"]
    if channels is not None:
        channels_set = {ch.upper() for ch in channels}
        ch_names = [ch for ch in ch_names if ch.upper() in channels_set]
        if not ch_names:
            if logger:
                logger.warning(f"No matching channels found for specified channels: {channels}")
            return
    
    ch_dir = out_dir / "channels"
    ch_dir.mkdir(parents=True, exist_ok=True)

    fmax_available = float(np.max(tfr_pain.freqs))
    bands = get_bands_for_tfr(max_freq_available=fmax_available, config=config)

    for ch in ch_names:
        _plot_single_tfr_figure(tfr_pain, ch, None, f"{ch} — Painful (baseline logratio)", f"tfr_{ch}_painful_bl.png", ch_dir, config, logger, baseline_used, subject=subject)
        _plot_single_tfr_figure(tfr_non, ch, None, f"{ch} — Non-pain (baseline logratio)", f"tfr_{ch}_nonpain_bl.png", ch_dir, config, logger, baseline_used, subject=subject)
        _plot_single_tfr_figure(tfr_diff, ch, None, f"{ch} — Pain minus Non-pain (baseline logratio)", f"tfr_{ch}_pain_minus_nonpain_bl.png", ch_dir, config, logger, baseline_used, subject=subject)

        for band, (fmin, fmax) in bands.items():
            fmax_eff = min(fmax, fmax_available)
            if fmin >= fmax_eff:
                continue
            band_dir = ch_dir / band
            band_dir.mkdir(parents=True, exist_ok=True)

            _plot_single_tfr_figure(tfr_pain, ch, None, f"{ch} — {band} Painful (baseline logratio)", f"tfr_{ch}_{band}_painful_bl.png", band_dir, config, logger, baseline_used, subject=subject, band=band)
            _plot_single_tfr_figure(tfr_non, ch, None, f"{ch} — {band} Non-pain (baseline logratio)", f"tfr_{ch}_{band}_nonpain_bl.png", band_dir, config, logger, baseline_used, subject=subject, band=band)
            _plot_single_tfr_figure(tfr_diff, ch, None, f"{ch} — {band} Pain minus Non-pain (baseline logratio)", f"tfr_{ch}_{band}_pain_minus_nonpain_bl.png", band_dir, config, logger, baseline_used, subject=subject, band=band)

