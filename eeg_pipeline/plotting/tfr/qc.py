"""
TFR quality control plotting functions.

Functions for creating quality control visualizations for time-frequency
representations, including baseline vs active comparisons.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt

from ...utils.config.loader import get_config_value, ensure_config, get_frequency_bands
from ...utils.analysis.tfr import (
    validate_baseline_indices,
    average_tfr_band,
)
from ..config import get_plot_config
from ..core.utils import get_font_sizes, log
from .contrasts import _get_baseline_window
from .channels import _save_fig


def _get_qc_config_values(config):
    """Extract QC configuration values from config object.
    
    Args:
        config: Configuration object
        
    Returns:
        Dictionary with QC configuration values
    """
    plot_cfg = get_plot_config(config)
    tfr_config = plot_cfg.plot_type_configs.get("tfr", {}) if plot_cfg else {}
    qc_config = tfr_config.get("qc", {})
    
    return {
        "epsilon_for_division": qc_config.get("epsilon_for_division", 1e-20),
        "percentage_multiplier": tfr_config.get("percentage_multiplier", 100.0),
        "fig_width": qc_config.get("fig_width", 8),
        "fig_height": qc_config.get("fig_height", 3),
        "histogram_bins": qc_config.get("histogram_bins", 50),
        "histogram_alpha": qc_config.get("histogram_alpha", 0.8),
    }


def _compute_percentage_change(
    active_values: np.ndarray,
    baseline_values: np.ndarray,
    epsilon: float,
    multiplier: float,
) -> np.ndarray:
    """Compute percentage change from baseline to active values.
    
    Args:
        active_values: Active period values
        baseline_values: Baseline period values
        epsilon: Small value to prevent division by zero
        multiplier: Multiplier for percentage (typically 100.0)
        
    Returns:
        Array of percentage change values
    """
    denominator = baseline_values + epsilon
    raw_change = (active_values - baseline_values) / denominator
    return raw_change * multiplier


def _create_histogram_plot(
    baseline_values: np.ndarray,
    percentage_change: np.ndarray,
    band_name: str,
    baseline_window: Tuple[float, float],
    active_window: Tuple[float, float],
    config_values: dict,
) -> plt.Figure:
    """Create histogram plot comparing baseline and percentage change.
    
    Args:
        baseline_values: Baseline power values
        percentage_change: Percentage change values
        band_name: Frequency band name
        baseline_window: Baseline time window (start, end)
        active_window: Active time window (start, end)
        config_values: QC configuration values
        
    Returns:
        Matplotlib figure object
    """
    fig, axes = plt.subplots(
        1, 2,
        figsize=(config_values["fig_width"], config_values["fig_height"]),
        constrained_layout=True,
    )
    
    axes[0].hist(
        baseline_values,
        bins=config_values["histogram_bins"],
        color="tab:blue",
        alpha=config_values["histogram_alpha"],
    )
    axes[0].set_title(f"Baseline power — {band_name}")
    axes[0].set_xlabel("Power (a.u.)")
    axes[0].set_ylabel("Count")
    
    axes[1].hist(
        percentage_change,
        bins=config_values["histogram_bins"],
        color="tab:orange",
        alpha=config_values["histogram_alpha"],
    )
    axes[1].set_title(f"% signal change (active vs baseline) — {band_name}")
    axes[1].set_xlabel("% change")
    axes[1].set_ylabel("Count")
    
    font_sizes = get_font_sizes()
    baseline_start, baseline_end = baseline_window
    active_start, active_end = active_window
    fig.suptitle(
        f"Baseline vs Active QC — {band_name}\n"
        f"(baseline={baseline_start:.2f}–{baseline_end:.2f}s; "
        f"active={active_start:.2f}–{active_end:.2f}s)",
        fontsize=font_sizes["ylabel"],
    )
    
    return fig


def _compute_topomap_percentage_change(
    tfr_avg: mne.time_frequency.AverageTFR,
    band_name: str,
    fmin: float,
    fmax: float,
    baseline_window: Tuple[float, float],
    active_window: Tuple[float, float],
    config_values: dict,
) -> Optional[np.ndarray]:
    """Compute percentage change for topomap visualization.
    
    Args:
        tfr_avg: Averaged TFR object
        band_name: Frequency band name
        fmin: Minimum frequency
        fmax: Maximum frequency
        baseline_window: Baseline time window (start, end)
        active_window: Active time window (start, end)
        config_values: QC configuration values
        
    Returns:
        Array of percentage change values per channel, or None if computation fails
    """
    baseline_start, baseline_end = baseline_window
    active_start, active_end = active_window
    
    active_topomap = average_tfr_band(
        tfr_avg,
        fmin=fmin,
        fmax=fmax,
        tmin=float(active_start),
        tmax=float(active_end),
    )
    baseline_topomap = average_tfr_band(
        tfr_avg,
        fmin=fmin,
        fmax=fmax,
        tmin=float(baseline_start),
        tmax=float(baseline_end),
    )
    
    if active_topomap is None or baseline_topomap is None:
        return None
    
    return _compute_percentage_change(
        active_topomap,
        baseline_topomap,
        config_values["epsilon_for_division"],
        config_values["percentage_multiplier"],
    )


def _process_single_band(
    data: np.ndarray,
    freqs: np.ndarray,
    times: np.ndarray,
    band_name: str,
    fmin: float,
    fmax: float,
    baseline_mask: np.ndarray,
    active_mask: np.ndarray,
    baseline_window: Tuple[float, float],
    active_window: Tuple[float, float],
    tfr_avg: Optional[mne.time_frequency.AverageTFR],
    config_values: dict,
    qc_dir: Path,
    config,
    logger: Optional[logging.Logger],
) -> Optional[dict]:
    """Process a single frequency band for QC analysis.
    
    Args:
        data: TFR data array (epochs, channels, frequencies, times)
        freqs: Frequency array
        times: Time array
        band_name: Frequency band name
        fmin: Minimum frequency for band
        fmax: Maximum frequency for band
        baseline_mask: Boolean mask for baseline time points
        active_mask: Boolean mask for active time points
        baseline_window: Baseline time window (start, end)
        active_window: Active time window (start, end)
        tfr_avg: Averaged TFR object for topomap
        config_values: QC configuration values
        qc_dir: Output directory for QC plots
        config: Configuration object
        logger: Optional logger instance
        
    Returns:
        Dictionary with summary statistics, or None if processing fails
    """
    fmax_effective = float(fmax) if fmax is not None else float(freqs.max())
    frequency_mask = (freqs >= float(fmin)) & (freqs <= fmax_effective)
    if not np.any(frequency_mask):
        return None
    
    baseline_data = data[:, :, frequency_mask, :][:, :, :, baseline_mask]
    active_data = data[:, :, frequency_mask, :][:, :, :, active_mask]
    
    baseline_power = baseline_data.mean(axis=(2, 3))
    active_power = active_data.mean(axis=(2, 3))
    
    baseline_flat = baseline_power.reshape(-1)
    active_flat = active_power.reshape(-1)
    
    percentage_change = _compute_percentage_change(
        active_flat,
        baseline_flat,
        config_values["epsilon_for_division"],
        config_values["percentage_multiplier"],
    )
    
    fig = _create_histogram_plot(
        baseline_flat,
        percentage_change,
        band_name,
        baseline_window,
        active_window,
        config_values,
    )
    _save_fig(
        fig,
        qc_dir,
        f"qc_baseline_active_hist_{band_name}.png",
        config=config,
        logger=logger,
    )
    
    topomap_percentage_change = None
    if tfr_avg is not None:
        fmin_effective = float(fmin)
        topomap_percentage_change = _compute_topomap_percentage_change(
            tfr_avg,
            band_name,
            fmin_effective,
            fmax_effective,
            baseline_window,
            active_window,
            config_values,
        )
    
    summary = {
        "band": band_name,
        "baseline_mean": float(np.nanmean(baseline_flat)),
        "baseline_median": float(np.nanmedian(baseline_flat)),
        "active_mean": float(np.nanmean(active_flat)),
        "active_median": float(np.nanmedian(active_flat)),
        "pct_change_mean": float(np.nanmean(percentage_change)),
        "pct_change_median": float(np.nanmedian(percentage_change)),
        "n_baseline_samples": int(baseline_mask.sum()),
        "n_active_samples": int(active_mask.sum()),
    }
    
    if topomap_percentage_change is not None and np.isfinite(topomap_percentage_change).any():
        summary["pct_change_mean_topomap"] = float(np.nanmean(topomap_percentage_change))
        summary["pct_change_median_topomap"] = float(np.nanmedian(topomap_percentage_change))
    else:
        summary["pct_change_mean_topomap"] = float("nan")
        summary["pct_change_median_topomap"] = float("nan")
    
    return summary


def qc_baseline_active_power(
    tfr,
    out_dir: Path,
    config,
    baseline: Optional[Tuple[Optional[float], Optional[float]]] = None,
    active_window: Tuple[float, float] = (3.0, 10.5),
    logger: Optional[logging.Logger] = None,
) -> None:
    """Create quality control plots comparing baseline vs active power.
    
    Generates histograms and summary statistics comparing power during baseline
    and active periods across frequency bands. Creates topomap visualizations
    showing percentage change from baseline to active.
    
    Args:
        tfr: MNE TFR object (EpochsTFR or AverageTFR)
        out_dir: Output directory path
        config: Configuration object
        baseline: Optional baseline window tuple (defaults to config)
        active_window: Active window tuple for statistics
        logger: Optional logger instance
    """
    baseline_window = _get_baseline_window(config, baseline)
    qc_dir = out_dir / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)

    data = getattr(tfr, "data", None)
    if data is None or data.ndim not in [3, 4]:
        return

    if data.ndim == 3:
        data = data[None, ...]

    freqs = np.asarray(tfr.freqs)
    times = np.asarray(tfr.times)

    config = ensure_config(config)
    min_baseline_samples = int(
        get_config_value(
            config,
            "time_frequency_analysis.constants.min_samples_for_baseline_validation",
            5,
        )
    )
    baseline_start, baseline_end, baseline_indices = validate_baseline_indices(
        times, baseline_window, min_samples=min_baseline_samples, logger=logger
    )
    baseline_mask = np.zeros(len(times), dtype=bool)
    baseline_mask[baseline_indices] = True
    
    active_mask = (times >= active_window[0]) & (times < active_window[1])
    if not np.any(active_mask):
        log(f"QC skipped: active samples={int(active_mask.sum())}", logger, "warning")
        return

    tfr_avg = tfr.average() if isinstance(tfr, mne.time_frequency.EpochsTFR) else tfr
    
    band_bounds = get_frequency_bands(config)
    if not band_bounds:
        log("QC skipped: no frequency bands found in config", logger, "warning")
        return
    
    config_values = _get_qc_config_values(config)
    baseline_window_tuple = (baseline_start, baseline_end)
    summary_rows = []
    
    for band_name, band_range in band_bounds.items():
        fmin, fmax = tuple(band_range)
        summary = _process_single_band(
            data,
            freqs,
            times,
            band_name,
            fmin,
            fmax,
            baseline_mask,
            active_mask,
            baseline_window_tuple,
            active_window,
            tfr_avg,
            config_values,
            qc_dir,
            config,
            logger,
        )
        if summary is not None:
            summary_rows.append(summary)

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = qc_dir / "qc_baseline_active_summary.tsv"
        summary_df.to_csv(summary_path, sep="\t", index=False)
        log(f"Saved QC summary: {summary_path}", logger)


__all__ = [
    "qc_baseline_active_power",
]

