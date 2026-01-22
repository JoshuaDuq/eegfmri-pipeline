"""
TFR Band Power Evolution Plots
==============================

Visualizations showing how power in each frequency band evolves across time,
for user-specified conditions (all trials, condition_1/condition_2).

Each plot answers a specific scientific question about temporal dynamics.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

from eeg_pipeline.infra.paths import ensure_dir
from eeg_pipeline.plotting.features.roi import get_roi_channels, get_roi_definitions
from eeg_pipeline.plotting.io.figures import save_fig
from eeg_pipeline.utils.analysis.events import extract_comparison_mask

from ...utils.analysis.tfr import get_bands_for_tfr
from ..config import get_plot_config


# =============================================================================
# Constants
# =============================================================================

# Default band colors (fallback for user-defined bands)
_DEFAULT_BAND_COLORS = {
    "delta": "#1f77b4",
    "theta": "#2ca02c",
    "alpha": "#ff7f0e",
    "beta": "#d62728",
    "gamma": "#9467bd",
}

def _get_bands_from_config(tfr: mne.time_frequency.EpochsTFR, config: Any) -> List[str]:
    """Get frequency band names from config, respecting user selection.
    
    If selected_bands is specified, returns only those bands in that order.
    Otherwise, returns all bands from config in sorted order.
    
    Args:
        tfr: EpochsTFR object
        config: Configuration object
        
    Returns:
        List of band names
    """
    from eeg_pipeline.utils.config.loader import get_frequency_bands, get_config_value
    
    selected_bands = get_config_value(config, "time_frequency_analysis.selected_bands", None)
    if selected_bands and isinstance(selected_bands, (list, tuple)) and len(selected_bands) > 0:
        return [b for b in selected_bands if b]
    
    config_bands = get_frequency_bands(config)
    if config_bands:
        return sorted(config_bands.keys())
    
    bands_dict = get_bands_for_tfr(tfr=tfr, config=config)
    return sorted(bands_dict.keys())


def _get_band_color(band: str) -> str:
    """Get color for a frequency band, with fallback for user-defined bands.
    
    Args:
        band: Band name
        
    Returns:
        Hex color code
    """
    return _DEFAULT_BAND_COLORS.get(band, "#666666")


def _get_band_range_label(band: str, bands_dict: Dict[str, Tuple[float, float]]) -> str:
    """Get frequency range label for a band.
    
    Args:
        band: Band name
        bands_dict: Dictionary mapping band names to (fmin, fmax) tuples or [fmin, fmax] lists
        
    Returns:
        Formatted range string (e.g., "8.0-12.9 Hz")
    """
    if band in bands_dict:
        band_range = bands_dict[band]
        if isinstance(band_range, (list, tuple)) and len(band_range) >= 2:
            fmin = float(band_range[0])
            fmax = float(band_range[1])
            return f"{fmin:.1f}-{fmax:.1f} Hz"
        elif isinstance(band_range, tuple) and len(band_range) == 2:
            fmin, fmax = band_range
            return f"{fmin:.1f}-{fmax:.1f} Hz"
    return f"{band} Hz"

CONDITION_COLORS = {
    "all": "#333333",
    "condition_1": "#4C72B0",
    "condition_2": "#C42847",
}

# Plotting style constants
MIN_TRIALS_FOR_PLOT = 3
FILL_ALPHA = 0.3
FILL_ALPHA_OVERLAY = 0.2
REFERENCE_LINE_ALPHA = 0.7
REFERENCE_LINE_WIDTH = 0.5
MAIN_LINE_WIDTH = 2.0
SUBPLOT_LINE_WIDTH = 1.5
BAR_WIDTH = 0.35
BAR_ALPHA = 0.8
BAR_CAPSIZE = 3


# =============================================================================
# Helper Functions
# =============================================================================

def _compute_mean_sem(power_data: np.ndarray, n_trials: int) -> Tuple[np.ndarray, np.ndarray]:
    """Compute mean and standard error of the mean across trials.
    
    Args:
        power_data: Power array (n_trials, n_times)
        n_trials: Number of trials for SEM calculation
        
    Returns:
        mean_power: Mean across trials (n_times,)
        sem_power: Standard error of the mean (n_times,)
    """
    mean_power = np.nanmean(power_data, axis=0)
    sem_power = np.nanstd(power_data, axis=0) / np.sqrt(n_trials)
    return mean_power, sem_power


def _plot_mean_sem_with_reference_lines(
    ax: plt.Axes,
    times: np.ndarray,
    mean_power: np.ndarray,
    sem_power: np.ndarray,
    color: str,
    linewidth: float = MAIN_LINE_WIDTH,
    fill_alpha: float = FILL_ALPHA,
) -> None:
    """Plot mean ± SEM with reference lines at zero.
    
    Args:
        ax: Matplotlib axes
        times: Time array
        mean_power: Mean power array
        sem_power: SEM array
        color: Line color
        linewidth: Line width for main plot
        fill_alpha: Alpha for fill_between
    """
    ax.fill_between(
        times,
        mean_power - sem_power,
        mean_power + sem_power,
        alpha=fill_alpha,
        color=color,
    )
    ax.plot(times, mean_power, color=color, linewidth=linewidth)
    ax.axhline(0, color="gray", linestyle="--", linewidth=REFERENCE_LINE_WIDTH, alpha=REFERENCE_LINE_ALPHA)
    ax.axvline(0, color="black", linestyle="-", linewidth=REFERENCE_LINE_WIDTH, alpha=0.5)


def _get_condition_title_labels(label1: str, label2: str) -> Dict[str, str]:
    """Get condition title labels for plots.
    
    Args:
        label1: Label for condition 1
        label2: Label for condition 2
        
    Returns:
        Dictionary mapping condition keys to display labels
    """
    return {
        "all": "All Trials",
        "condition_2": f"{label2} Trials",
        "condition_1": f"{label1} Trials",
    }


def _get_condition_header_labels(label1: str, label2: str, n_trials: int) -> Dict[str, str]:
    """Get condition header labels with trial counts.
    
    Args:
        label1: Label for condition 1
        label2: Label for condition 2
        n_trials: Number of trials
        
    Returns:
        Dictionary mapping condition keys to header labels
    """
    return {
        "all": f"All Trials\n(n={n_trials})",
        "condition_2": f"{label2}\n(n={n_trials})",
        "condition_1": f"{label1}\n(n={n_trials})",
    }


def _get_available_rois(tfr: mne.time_frequency.EpochsTFR, config: Any) -> List[str]:
    """Get list of ROI names that have matching channels in the TFR.
    
    Args:
        tfr: EpochsTFR object
        config: Configuration object
        
    Returns:
        List of ROI names with available channels
    """
    rois = get_roi_definitions(config)
    available_rois = []
    for roi_name in rois.keys():
        indices = _get_roi_channel_indices(tfr, roi_name, config)
        if len(indices) > 0:
            available_rois.append(roi_name)
    return available_rois


def _get_band_power_timecourse(
    tfr: mne.time_frequency.EpochsTFR,
    band: str,
    config: Any,
    channel_indices: Optional[List[int]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract mean power timecourse for a frequency band.
    
    Args:
        tfr: EpochsTFR object
        band: Frequency band name
        config: Configuration object
        channel_indices: Optional list of channel indices to average
        
    Returns:
        times: Time array
        power: Power array (n_trials, n_times)
    """
    from eeg_pipeline.utils.config.loader import get_frequency_bands
    
    config_bands = get_frequency_bands(config)
    if band not in config_bands:
        return np.array([]), np.array([])
    
    band_range = config_bands[band]
    if isinstance(band_range, (list, tuple)) and len(band_range) >= 2:
        fmin = float(band_range[0])
        fmax = float(band_range[1]) if band_range[1] is not None else float(np.max(tfr.freqs))
    else:
        return np.array([]), np.array([])
    
    fmax_effective = min(fmax, float(np.max(tfr.freqs)))
    freqs = tfr.freqs
    freq_mask = (freqs >= fmin) & (freqs <= fmax_effective)
    
    if not np.any(freq_mask):
        return np.array([]), np.array([])
    
    # Get data: (n_epochs, n_channels, n_freqs, n_times)
    data = tfr.data
    
    # Average over frequencies in band
    band_data = data[:, :, freq_mask, :].mean(axis=2)  # (n_epochs, n_channels, n_times)
    
    # Average over channels
    if channel_indices is not None and len(channel_indices) > 0:
        band_data = band_data[:, channel_indices, :].mean(axis=1)  # (n_epochs, n_times)
    else:
        band_data = band_data.mean(axis=1)  # (n_epochs, n_times)
    
    return tfr.times, band_data


def _get_roi_channel_indices(tfr: mne.time_frequency.EpochsTFR, roi_name: str, config: Any) -> List[int]:
    """Get channel indices for a ROI from config-defined ROIs.
    
    Args:
        tfr: EpochsTFR object with channel info
        roi_name: Name of the ROI to get channels for
        config: Configuration object containing ROI definitions
        
    Returns:
        List of channel indices matching the ROI
    """
    ch_names = tfr.ch_names
    
    # Get ROI definitions from config (user-defined in global setup)
    rois = get_roi_definitions(config)
    if not rois or roi_name not in rois:
        return []
    
    # Get channel names for this ROI using the regex patterns
    roi_patterns = rois[roi_name]
    roi_channels = get_roi_channels(roi_patterns, ch_names)
    
    # Convert channel names to indices
    indices = [i for i, ch in enumerate(ch_names) if ch in roi_channels]
    return indices


def _create_condition_masks(
    events_df: pd.DataFrame,
    config: Any,
) -> Tuple[Dict[str, np.ndarray], str, str]:
    """Create masks for different conditions based on user-specified comparison.
    
    Only creates masks for what the user explicitly specifies in the comparison configuration.
    No automatic temperature detection - only uses the comparison_column, comparison_values, 
    and comparison_labels from the config.
    
    Args:
        events_df: Events DataFrame
        config: Configuration object
        
    Returns:
        Tuple of (masks dict, label1, label2)
        Masks dict has keys: 'all', 'condition_1', 'condition_2'
    """
    n_trials = len(events_df)
    masks = {"all": np.ones(n_trials, dtype=bool)}

    label1 = "Condition 1"
    label2 = "Condition 2"
    comp = extract_comparison_mask(events_df, config, require_enabled=False)

    if comp is not None:
        mask1, mask2, label1, label2 = comp
        masks["condition_1"] = np.asarray(mask1, dtype=bool)
        masks["condition_2"] = np.asarray(mask2, dtype=bool)

    return masks, label1, label2


def _apply_baseline(
    power: np.ndarray,
    times: np.ndarray,
    baseline: Tuple[float, float],
) -> np.ndarray:
    """Apply baseline correction (percent change from baseline)."""
    baseline_mask = (times >= baseline[0]) & (times <= baseline[1])
    if not np.any(baseline_mask):
        return power
    
    baseline_mean = power[:, baseline_mask].mean(axis=1, keepdims=True)
    baseline_mean = np.where(baseline_mean > 0, baseline_mean, np.nan)
    return (power - baseline_mean) / baseline_mean * 100


# =============================================================================
# Main Plotting Functions
# =============================================================================

def plot_band_power_evolution_all_conditions(
    tfr: mne.time_frequency.EpochsTFR,
    events_df: pd.DataFrame,
    save_dir: Path,
    config: Any,
    baseline: Tuple[float, float] = (-2.0, 0.0),
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Path]:
    """Plot how each band's power evolves across time for all conditions.
    
    Creates a grid: rows = bands, columns = conditions
    Question: "How does power in each frequency band evolve across the trial?"
    
    Args:
        tfr: EpochsTFR object
        events_df: Events DataFrame with condition/temperature columns
        save_dir: Directory to save plots
        config: Configuration object
        baseline: Baseline window for correction
        logger: Logger instance
        
    Returns:
        Dictionary of saved file paths
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    ensure_dir(save_dir / "evolution")
    saved = {}

    masks, label1, label2 = _create_condition_masks(events_df, config)
    condition_order = ["all", "condition_2", "condition_1"]
    conditions = [c for c in condition_order if c in masks and masks[c].sum() > 0]

    if len(conditions) == 0:
        logger.warning("No valid conditions found")
        return saved

    plot_cfg = get_plot_config(config)
    primary_ext = plot_cfg.formats[0] if plot_cfg.formats else "png"

    bands = _get_bands_from_config(tfr, config)
    from eeg_pipeline.utils.config.loader import get_frequency_bands
    bands_dict = get_frequency_bands(config)
    n_bands = len(bands)
    n_conds = len(conditions)
    fig, axes = plt.subplots(n_bands, n_conds, figsize=(4 * n_conds, 3 * n_bands), squeeze=False)

    for i, band in enumerate(bands):
        times, power = _get_band_power_timecourse(tfr, band, config)
        if len(times) == 0:
            continue

        power_bl = _apply_baseline(power, times, baseline)

        for j, cond in enumerate(conditions):
            ax = axes[i, j]
            mask = masks[cond]

            if mask.sum() == 0:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                continue

            cond_power = power_bl[mask]
            mean_power, sem_power = _compute_mean_sem(cond_power, mask.sum())
            _plot_mean_sem_with_reference_lines(ax, times, mean_power, sem_power, _get_band_color(band))

            if i == 0:
                header_labels = _get_condition_header_labels(label1, label2, mask.sum())
                ax.set_title(
                    header_labels.get(cond, cond),
                    fontsize=11,
                    fontweight="bold",
                    color=CONDITION_COLORS.get(cond, "black"),
                )

            if j == 0:
                band_range = _get_band_range_label(band, bands_dict)
                ax.set_ylabel(
                    f"{band.upper()}\n({band_range})\n% change",
                    fontsize=10,
                    color=_get_band_color(band),
                )

            if i == n_bands - 1:
                ax.set_xlabel("Time (s)", fontsize=10)

            ax.set_xlim(times[0], times[-1])

    fig.suptitle(
        "How does power in each frequency band evolve across the trial?",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    plt.tight_layout()

    path = save_dir / "evolution" / f"band_power_evolution_all_conditions.{primary_ext}"
    save_fig(
        fig,
        path,
        logger=logger,
        formats=plot_cfg.formats,
        dpi=plot_cfg.savefig_dpi,
        bbox_inches=plot_cfg.bbox_inches,
        pad_inches=plot_cfg.pad_inches,
        config=config,
    )

    saved["band_power_evolution"] = path

    return saved


def plot_band_power_by_roi(
    tfr: mne.time_frequency.EpochsTFR,
    events_df: pd.DataFrame,
    save_dir: Path,
    config: Any,
    baseline: Tuple[float, float] = (-2.0, 0.0),
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Path]:
    """Plot power evolution for each band in each ROI.
    
    Creates one figure per ROI showing all bands × conditions.
    Each ROI gets its own subfolder.
    Question: "How does power evolve in different brain regions?"
    
    Args:
        tfr: EpochsTFR object
        events_df: Events DataFrame
        save_dir: Directory to save plots
        config: Configuration object
        baseline: Baseline window
        logger: Logger instance
        
    Returns:
        Dictionary of saved file paths
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    ensure_dir(save_dir / "evolution")
    saved = {}

    masks, label1, label2 = _create_condition_masks(events_df, config)
    plot_cfg = get_plot_config(config)
    primary_ext = plot_cfg.formats[0] if plot_cfg.formats else "png"

    available_rois = _get_available_rois(tfr, config)
    if not available_rois:
        logger.warning("No ROIs found in TFR channels")
        return saved

    bands = _get_bands_from_config(tfr, config)
    from eeg_pipeline.utils.config.loader import get_frequency_bands
    bands_dict = get_frequency_bands(config)
    cond_titles = _get_condition_title_labels(label1, label2)
    condition_order = ["all", "condition_2", "condition_1"]
    conditions = [c for c in condition_order if c in masks and masks[c].sum() > 0]

    for roi_name in available_rois:
        roi_indices = _get_roi_channel_indices(tfr, roi_name, config)
        if len(roi_indices) == 0:
            continue

        roi_dir = save_dir / "evolution" / "rois" / roi_name
        ensure_dir(roi_dir)

        n_bands = len(bands)
        n_conds = len(conditions)
        fig, axes = plt.subplots(n_bands, n_conds, figsize=(4 * n_conds, 3 * n_bands), squeeze=False)

        for i, band in enumerate(bands):
            times, power = _get_band_power_timecourse(tfr, band, config, roi_indices)
            if len(times) == 0:
                continue

            power_bl = _apply_baseline(power, times, baseline)

            for j, cond in enumerate(conditions):
                ax = axes[i, j]
                mask = masks[cond]

                if mask.sum() < MIN_TRIALS_FOR_PLOT:
                    ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                    continue

                cond_power = power_bl[mask]
                mean_power, sem_power = _compute_mean_sem(cond_power, mask.sum())
                _plot_mean_sem_with_reference_lines(
                    ax, times, mean_power, sem_power, _get_band_color(band), SUBPLOT_LINE_WIDTH
                )

                if i == 0:
                    header_labels = _get_condition_header_labels(label1, label2, mask.sum())
                    ax.set_title(
                        header_labels.get(cond, cond),
                        fontsize=11,
                        fontweight="bold",
                        color=CONDITION_COLORS.get(cond, "black"),
                    )

                if j == 0:
                    band_range = _get_band_range_label(band, bands_dict)
                    ax.set_ylabel(
                        f"{band.upper()}\n({band_range})\n% change",
                        fontsize=10,
                        color=_get_band_color(band),
                    )

                if i == n_bands - 1:
                    ax.set_xlabel("Time (s)", fontsize=10)

                ax.tick_params(labelsize=8)
                ax.set_xlim(times[0], times[-1])

        fig.suptitle(
            f"{roi_name} — Band Power Evolution Across Conditions",
            fontsize=14,
            fontweight="bold",
            y=1.02,
        )

        plt.tight_layout()

        path = roi_dir / f"band_power_evolution.{primary_ext}"
        save_fig(
            fig,
            path,
            logger=logger,
            formats=plot_cfg.formats,
            dpi=plot_cfg.savefig_dpi,
            bbox_inches=plot_cfg.bbox_inches,
            pad_inches=plot_cfg.pad_inches,
        )

        saved[f"band_power_roi_{roi_name}"] = path

    return saved


def _plot_condition_overlay(
    ax: plt.Axes,
    times: np.ndarray,
    power_bl: np.ndarray,
    masks: Dict[str, np.ndarray],
    condition_specs: List[Tuple[str, str, str]],
) -> None:
    """Plot multiple conditions overlaid on the same axes.
    
    Args:
        ax: Matplotlib axes
        times: Time array
        power_bl: Baseline-corrected power (n_trials, n_times)
        masks: Dictionary of condition masks
        condition_specs: List of (condition_key, color, label) tuples
    """
    for cond, color, label in condition_specs:
        if cond not in masks or masks[cond].sum() < MIN_TRIALS_FOR_PLOT:
            continue

        mask = masks[cond]
        cond_power = power_bl[mask]
        mean_power, sem_power = _compute_mean_sem(cond_power, mask.sum())

        ax.fill_between(
            times, mean_power - sem_power, mean_power + sem_power, alpha=FILL_ALPHA_OVERLAY, color=color
        )
        ax.plot(times, mean_power, color=color, linewidth=MAIN_LINE_WIDTH, label=f"{label} (n={mask.sum()})")

    ax.axhline(0, color="gray", linestyle="--", linewidth=REFERENCE_LINE_WIDTH)
    ax.axvline(0, color="black", linestyle="-", linewidth=REFERENCE_LINE_WIDTH)


def plot_condition_comparison_per_band(
    tfr: mne.time_frequency.EpochsTFR,
    events_df: pd.DataFrame,
    save_dir: Path,
    config: Any,
    baseline: Tuple[float, float] = (-2.0, 0.0),
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Path]:
    """Plot condition comparisons overlaid for each band.
    
    Creates one figure per band showing all conditions overlaid.
    Question: "Do conditions differ in their power dynamics?"
    
    Args:
        tfr: EpochsTFR object
        events_df: Events DataFrame
        save_dir: Directory to save plots
        config: Configuration object
        baseline: Baseline window
        logger: Logger instance
        
    Returns:
        Dictionary of saved file paths
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    ensure_dir(save_dir / "evolution")
    saved = {}

    masks, label1, label2 = _create_condition_masks(events_df, config)
    plot_cfg = get_plot_config(config)
    primary_ext = plot_cfg.formats[0] if plot_cfg.formats else "png"

    has_conditions = "condition_1" in masks and "condition_2" in masks
    
    if not has_conditions:
        logger.warning("No valid conditions found for comparison plots")
        return saved
    
    bands = _get_bands_from_config(tfr, config)
    from eeg_pipeline.utils.config.loader import get_frequency_bands
    bands_dict = get_frequency_bands(config)
    fig, axes = plt.subplots(1, len(bands), figsize=(3 * len(bands), 3), squeeze=False)

    for j, band in enumerate(bands):
        times, power = _get_band_power_timecourse(tfr, band, config)
        if len(times) == 0:
            continue

        power_bl = _apply_baseline(power, times, baseline)

        ax = axes[0, j]
        condition_specs = [
            ("condition_2", CONDITION_COLORS["condition_2"], label2),
            ("condition_1", CONDITION_COLORS["condition_1"], label1),
        ]
        _plot_condition_overlay(ax, times, power_bl, masks, condition_specs)
        band_range = _get_band_range_label(band, bands_dict)
        ax.set_title(f"{band.upper()}\n({band_range})", fontsize=10, fontweight="bold", color=_get_band_color(band))
        if j == 0:
            ax.set_ylabel(f"{label2} vs {label1}\n% change", fontsize=10)
            ax.legend(fontsize=8, loc="upper right")
        ax.set_xlabel("Time (s)", fontsize=10)
        ax.set_xlim(times[0], times[-1])

    fig.suptitle("Do conditions differ in their power dynamics?", fontsize=12, fontweight="bold", y=1.02)

    plt.tight_layout()

    path = save_dir / "evolution" / f"condition_comparison_per_band.{primary_ext}"
    save_fig(
        fig,
        path,
        logger=logger,
        formats=plot_cfg.formats,
        dpi=plot_cfg.savefig_dpi,
        bbox_inches=plot_cfg.bbox_inches,
        pad_inches=plot_cfg.pad_inches,
        config=config,
    )

    saved["condition_comparison"] = path

    return saved


def plot_roi_condition_comparison(
    tfr: mne.time_frequency.EpochsTFR,
    events_df: pd.DataFrame,
    save_dir: Path,
    config: Any,
    baseline: Tuple[float, float] = (-2.0, 0.0),
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Path]:
    """Plot condition comparisons for each ROI using all configured bands.
    
    Creates one figure per ROI in its own subfolder.
    Question: "Which brain regions show the largest condition differences?"
    
    Args:
        tfr: EpochsTFR object
        events_df: Events DataFrame
        save_dir: Directory to save plots
        config: Configuration object
        baseline: Baseline window
        logger: Logger instance
        
    Returns:
        Dictionary of saved file paths
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    ensure_dir(save_dir / "evolution")
    saved = {}

    masks, label1, label2 = _create_condition_masks(events_df, config)
    plot_cfg = get_plot_config(config)
    primary_ext = plot_cfg.formats[0] if plot_cfg.formats else "png"

    bands = _get_bands_from_config(tfr, config)
    from eeg_pipeline.utils.config.loader import get_frequency_bands
    bands_dict = get_frequency_bands(config)
    
    available_rois = _get_available_rois(tfr, config)
    if not available_rois:
        return saved

    has_conditions = "condition_1" in masks and "condition_2" in masks
    
    if not has_conditions:
        logger.warning("No valid conditions found for ROI comparison plots")
        return saved

    n_bands = len(bands)
    
    for roi_name in available_rois:
        roi_indices = _get_roi_channel_indices(tfr, roi_name, config)
        if len(roi_indices) == 0:
            continue

        roi_dir = save_dir / "evolution" / "rois" / roi_name
        ensure_dir(roi_dir)

        fig, axes = plt.subplots(1, n_bands, figsize=(4 * n_bands, 3), squeeze=False)

        col_idx = 0
        for band in bands:
            times, power = _get_band_power_timecourse(tfr, band, config, roi_indices)
            if len(times) == 0:
                col_idx += 1
                continue

            power_bl = _apply_baseline(power, times, baseline)
            
            ax = axes[0, col_idx]
            condition_specs = [
                ("condition_2", CONDITION_COLORS["condition_2"], label2),
                ("condition_1", CONDITION_COLORS["condition_1"], label1),
            ]
            _plot_condition_overlay(ax, times, power_bl, masks, condition_specs)

            band_range = _get_band_range_label(band, bands_dict)
            ax.set_title(
                f"{band.upper()} ({band_range}) - {label2} vs {label1}",
                fontsize=10,
                fontweight="bold",
                color=_get_band_color(band),
            )
            if col_idx == 0:
                ax.set_ylabel(f"{roi_name}\n% change", fontsize=9)
                ax.legend(fontsize=7, loc="upper right")
            ax.set_xlabel("Time (s)", fontsize=9)
            ax.set_xlim(times[0], times[-1])
            ax.tick_params(labelsize=8)
            col_idx += 1

        band_names_str = "/".join([b.upper() for b in bands])
        fig.suptitle(
            f"{roi_name} — Condition Comparison ({band_names_str})",
            fontsize=12,
            fontweight="bold",
            y=1.02,
        )

        plt.tight_layout()

        path = roi_dir / f"condition_comparison.{primary_ext}"
        save_fig(
            fig,
            path,
            logger=logger,
            formats=plot_cfg.formats,
            dpi=plot_cfg.savefig_dpi,
            bbox_inches=plot_cfg.bbox_inches,
            pad_inches=plot_cfg.pad_inches,
            config=config,
        )

        saved[f"roi_condition_comparison_{roi_name}"] = path

    return saved


def _extract_band_values_from_dataframe(df: pd.DataFrame, band: str, value_col: str) -> float:
    """Extract value for a specific band from filtered DataFrame.
    
    Args:
        df: Filtered DataFrame
        band: Band name
        value_col: Column name to extract
        
    Returns:
        Value for the band, or 0.0 if not found
    """
    band_data = df[df["band"] == band]
    if len(band_data) > 0:
        return band_data[value_col].values[0]
    return 0.0


def _plot_summary_bars(
    ax: plt.Axes,
    df: pd.DataFrame,
    condition_keys: List[str],
    condition_labels: Dict[str, str],
    title: str,
) -> None:
    """Plot grouped bar chart for summary data.
    
    Args:
        ax: Matplotlib axes
        df: DataFrame with summary data
        condition_keys: List of condition keys to plot
        condition_labels: Dictionary mapping condition keys to display labels
        title: Plot title
    """
    filtered_df = df[df["condition"].isin(condition_keys)]
    if len(filtered_df) == 0:
        return

    bands = list(set(filtered_df["band"].unique()))
    bands = sorted(bands)
    x_positions = np.arange(len(bands))
    n_conditions = len(condition_keys)

    for i, cond in enumerate(condition_keys):
        cond_data = filtered_df[filtered_df["condition"] == cond]
        if len(cond_data) == 0:
            continue

        means = [_extract_band_values_from_dataframe(cond_data, band, "mean") for band in bands]
        sems = [_extract_band_values_from_dataframe(cond_data, band, "sem") for band in bands]

        offset = (i - 0.5 * (n_conditions - 1)) * BAR_WIDTH
        label = condition_labels.get(cond, cond.replace("_", " ").title())
        ax.bar(
            x_positions + offset,
            means,
            BAR_WIDTH,
            yerr=sems,
            label=label,
            color=CONDITION_COLORS.get(cond, "gray"),
            capsize=BAR_CAPSIZE,
            alpha=BAR_ALPHA,
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels([b.upper() for b in bands], fontsize=10)
    ax.axhline(0, color="gray", linestyle="--", linewidth=REFERENCE_LINE_WIDTH)
    ax.set_ylabel("Mean Active Power (% change)", fontsize=10)
    ax.set_title(title, fontweight="bold", loc="left", fontsize=11)
    ax.legend(fontsize=9)


def plot_band_power_summary(
    tfr: mne.time_frequency.EpochsTFR,
    events_df: pd.DataFrame,
    save_dir: Path,
    config: Any,
    baseline: Tuple[float, float] = (-2.0, 0.0),
    active_window: Tuple[float, float] = (3.0, 10.5),
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Path]:
    """Create summary bar plot of mean active power by band and condition.
    
    Question: "What is the overall pattern of power changes across bands and conditions?"
    
    Args:
        tfr: EpochsTFR object
        events_df: Events DataFrame
        save_dir: Directory to save plots
        config: Configuration object
        baseline: Baseline window
        active_window: Active window for averaging
        logger: Logger instance
        
    Returns:
        Dictionary of saved file paths
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    ensure_dir(save_dir / "evolution")
    saved = {}

    masks, label1, label2 = _create_condition_masks(events_df, config)
    plot_cfg = get_plot_config(config)
    primary_ext = plot_cfg.formats[0] if plot_cfg.formats else "png"

    bands = _get_bands_from_config(tfr, config)
    from eeg_pipeline.utils.config.loader import get_frequency_bands
    bands_dict = get_frequency_bands(config)
    summary_data = []

    for band in bands:
        times, power = _get_band_power_timecourse(tfr, band, config)
        if len(times) == 0:
            continue

        power_bl = _apply_baseline(power, times, baseline)

        active_mask = (times >= active_window[0]) & (times <= active_window[1])
        if not np.any(active_mask):
            continue

        for cond, mask in masks.items():
            if mask.sum() < MIN_TRIALS_FOR_PLOT:
                continue

            cond_power = power_bl[mask][:, active_mask]
            mean_active = np.nanmean(cond_power)
            sem_active = np.nanstd(cond_power.mean(axis=1)) / np.sqrt(mask.sum())

            summary_data.append(
                {
                    "band": band,
                    "condition": cond,
                    "mean": mean_active,
                    "sem": sem_active,
                    "n": mask.sum(),
                }
            )

    if not summary_data:
        return saved

    df = pd.DataFrame(summary_data)

    has_conditions = "condition_1" in masks and "condition_2" in masks
    
    if not has_conditions:
        logger.warning("No valid conditions found for summary plot")
        return saved

    fig, axes = plt.subplots(1, 1, figsize=(7, 5), squeeze=False)
    ax = axes[0, 0]

    cond_keys = ["condition_2", "condition_1"]
    cond_labels = {"condition_2": label2, "condition_1": label1}
    _plot_summary_bars(ax, df, cond_keys, cond_labels, f"{label2} vs {label1}")

    fig.suptitle(
        "What is the overall pattern of power changes across bands and conditions?",
        fontsize=12,
        fontweight="bold",
        y=1.02,
    )

    plt.tight_layout()

    path = save_dir / "evolution" / f"band_power_summary.{primary_ext}"
    save_fig(
        fig,
        path,
        logger=logger,
        formats=plot_cfg.formats,
        dpi=plot_cfg.savefig_dpi,
        bbox_inches=plot_cfg.bbox_inches,
        pad_inches=plot_cfg.pad_inches,
        config=config,
    )

    saved["band_power_summary"] = path

    return saved


# =============================================================================
# Main Entry Point
# =============================================================================

def visualize_band_evolution(
    tfr: mne.time_frequency.EpochsTFR,
    events_df: pd.DataFrame,
    save_dir: Path,
    config: Any,
    baseline: Tuple[float, float] = (-2.0, 0.0),
    active_window: Tuple[float, float] = (3.0, 10.5),
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Path]:
    """Generate all band power evolution visualizations.
    
    Args:
        tfr: EpochsTFR object
        events_df: Events DataFrame
        save_dir: Directory to save plots
        config: Configuration object
        baseline: Baseline window
        active_window: Active window
        logger: Logger instance
        
    Returns:
        Dictionary of all saved file paths
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info("Generating band power evolution plots...")
    
    all_saved = {}
    
    # 1. Band power evolution for all conditions
    saved = plot_band_power_evolution_all_conditions(
        tfr, events_df, save_dir, config, baseline, logger
    )
    all_saved.update(saved)
    
    # 2. Band power by ROI for each condition
    saved = plot_band_power_by_roi(
        tfr, events_df, save_dir, config, baseline, logger
    )
    all_saved.update(saved)
    
    # 3. Condition comparison per band
    saved = plot_condition_comparison_per_band(
        tfr, events_df, save_dir, config, baseline, logger
    )
    all_saved.update(saved)
    
    # 4. ROI condition comparison (all bands)
    saved = plot_roi_condition_comparison(
        tfr, events_df, save_dir, config, baseline, logger
    )
    all_saved.update(saved)
    
    # 5. Summary bar plot
    saved = plot_band_power_summary(
        tfr, events_df, save_dir, config, baseline, active_window, logger
    )
    all_saved.update(saved)
    
    logger.info(f"Generated {len(all_saved)} band evolution plots")
    
    return all_saved


__all__ = [
    "visualize_band_evolution",
    "plot_band_power_evolution_all_conditions",
    "plot_band_power_by_roi",
    "plot_condition_comparison_per_band",
    "plot_roi_condition_comparison",
    "plot_band_power_summary",
]
