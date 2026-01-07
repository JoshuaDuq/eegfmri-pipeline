"""
TFR Band Power Evolution Plots
==============================

Visualizations showing how power in each frequency band evolves across time,
for different conditions (all trials, condition_1/condition_2, high/low temperature).

Each plot answers a specific scientific question about temporal dynamics.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne

from eeg_pipeline.infra.paths import ensure_dir
from eeg_pipeline.plotting.io.figures import save_fig
from eeg_pipeline.plotting.features.roi import get_roi_definitions, get_roi_channels
from eeg_pipeline.utils.data.columns import get_temperature_column_from_config
from ...utils.analysis.tfr import get_bands_for_tfr, build_rois_from_info
from ..config import get_plot_config


# =============================================================================
# Constants
# =============================================================================

BANDS = ["delta", "theta", "alpha", "beta", "gamma"]
BAND_COLORS = {
    "delta": "#1f77b4",
    "theta": "#2ca02c", 
    "alpha": "#ff7f0e",
    "beta": "#d62728",
    "gamma": "#9467bd",
}
BAND_RANGES = {
    "delta": "1-4 Hz",
    "theta": "4-8 Hz",
    "alpha": "8-13 Hz",
    "beta": "13-30 Hz",
    "gamma": "30-100 Hz",
}

# ROI definitions are now loaded from config via get_roi_definitions()

# Condition colors (for display when per-plot style isn't provided)
CONDITION_COLORS = {
    "all": "#333333",
    "condition_1": "#4C72B0",
    "condition_2": "#C42847",
    "high_temp": "#D62728",
    "low_temp": "#1F77B4",
}


# =============================================================================
# Helper Functions
# =============================================================================

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
    bands = get_bands_for_tfr(tfr=tfr, config=config)
    if band not in bands:
        return np.array([]), np.array([])
    
    fmin, fmax = bands[band]
    freqs = tfr.freqs
    freq_mask = (freqs >= fmin) & (freqs <= fmax)
    
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
) -> tuple[Dict[str, np.ndarray], str, str]:
    """Create masks for different conditions.
    
    Returns dict with keys: 'all', 'condition_1', 'condition_2', 'high_temp', 'low_temp'
    """
    n_trials = len(events_df)
    masks = {"all": np.ones(n_trials, dtype=bool)}

    label1 = "Condition 1"
    label2 = "Condition 2"
    try:
        from eeg_pipeline.utils.analysis.events import extract_comparison_mask

        comp = extract_comparison_mask(events_df, config, require_enabled=False)
    except Exception:
        comp = None

    if comp is not None:
        mask1, mask2, label1, label2 = comp
        masks["condition_1"] = np.asarray(mask1, dtype=bool)
        masks["condition_2"] = np.asarray(mask2, dtype=bool)
    
    # High/low temperature (median split) using configured temperature column candidates.
    temp_col = get_temperature_column_from_config(config, events_df)
    
    if temp_col:
        temps = pd.to_numeric(events_df[temp_col], errors="coerce")
        valid_temps = temps.dropna()
        if len(valid_temps) > 0:
            median_temp = valid_temps.median()
            masks["high_temp"] = (temps >= median_temp).fillna(False).values
            masks["low_temp"] = (temps < median_temp).fillna(False).values
    
    return masks, str(label1), str(label2)


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
    
    # Get condition masks
    masks, label1, label2 = _create_condition_masks(events_df, config)
    conditions = ["all", "condition_2", "condition_1", "high_temp", "low_temp"]
    conditions = [c for c in conditions if c in masks and masks[c].sum() > 0]
    
    if len(conditions) == 0:
        logger.warning("No valid conditions found")
        return saved
    
    plot_cfg = get_plot_config(config)
    primary_ext = plot_cfg.formats[0] if plot_cfg.formats else "png"
    
    # Create figure
    n_bands = len(BANDS)
    n_conds = len(conditions)
    fig, axes = plt.subplots(n_bands, n_conds, figsize=(4*n_conds, 3*n_bands), squeeze=False)
    
    for i, band in enumerate(BANDS):
        times, power = _get_band_power_timecourse(tfr, band, config)
        if len(times) == 0:
            continue
        
        # Apply baseline correction
        power_bl = _apply_baseline(power, times, baseline)
        
        for j, cond in enumerate(conditions):
            ax = axes[i, j]
            mask = masks[cond]
            
            if mask.sum() == 0:
                ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
                continue
            
            cond_power = power_bl[mask]
            mean_power = np.nanmean(cond_power, axis=0)
            sem_power = np.nanstd(cond_power, axis=0) / np.sqrt(mask.sum())
            
            # Plot mean ± SEM
            ax.fill_between(times, mean_power - sem_power, mean_power + sem_power,
                           alpha=0.3, color=BAND_COLORS[band])
            ax.plot(times, mean_power, color=BAND_COLORS[band], linewidth=2)
            
            # Add reference lines
            ax.axhline(0, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
            ax.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
            
            # Labels
            if i == 0:
                cond_labels = {
                    "all": f"All Trials\n(n={mask.sum()})",
                    "condition_2": f"{label2}\n(n={mask.sum()})",
                    "condition_1": f"{label1}\n(n={mask.sum()})",
                    "high_temp": f"High Temp\n(n={mask.sum()})",
                    "low_temp": f"Low Temp\n(n={mask.sum()})",
                }
                ax.set_title(cond_labels.get(cond, cond), fontsize=11, fontweight='bold',
                            color=CONDITION_COLORS.get(cond, 'black'))
            
            if j == 0:
                ax.set_ylabel(f"{band.upper()}\n({BAND_RANGES[band]})\n% change",
                             fontsize=10, color=BAND_COLORS[band])
            
            if i == n_bands - 1:
                ax.set_xlabel("Time (s)", fontsize=10)
            
            ax.set_xlim(times[0], times[-1])
    
    fig.suptitle("How does power in each frequency band evolve across the trial?",
                fontsize=14, fontweight='bold', y=1.02)
    
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
    
    Creates one figure per condition showing all bands × ROIs.
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
    
    # Get available ROIs from config
    rois = get_roi_definitions(config)
    available_rois = []
    for roi_name in rois.keys():
        indices = _get_roi_channel_indices(tfr, roi_name, config)
        if len(indices) > 0:
            available_rois.append(roi_name)
    
    if not available_rois:
        logger.warning("No ROIs found in TFR channels")
        return saved
    
    # Create one figure per condition
    for cond, mask in masks.items():
        if mask.sum() < 3:
            continue
        
        n_rois = len(available_rois)
        n_bands = len(BANDS)
        
        fig, axes = plt.subplots(n_rois, n_bands, figsize=(3*n_bands, 2.5*n_rois), squeeze=False)
        
        for i, roi_name in enumerate(available_rois):
            roi_indices = _get_roi_channel_indices(tfr, roi_name, config)
            
            for j, band in enumerate(BANDS):
                ax = axes[i, j]
                
                times, power = _get_band_power_timecourse(tfr, band, config, roi_indices)
                if len(times) == 0:
                    ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
                    continue
                
                power_bl = _apply_baseline(power, times, baseline)
                cond_power = power_bl[mask]
                
                mean_power = np.nanmean(cond_power, axis=0)
                sem_power = np.nanstd(cond_power, axis=0) / np.sqrt(mask.sum())
                
                ax.fill_between(times, mean_power - sem_power, mean_power + sem_power,
                               alpha=0.3, color=BAND_COLORS[band])
                ax.plot(times, mean_power, color=BAND_COLORS[band], linewidth=1.5)
                
                ax.axhline(0, color='gray', linestyle='--', linewidth=0.5, alpha=0.7)
                ax.axvline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
                
                if i == 0:
                    ax.set_title(f"{band.upper()}", fontsize=10, fontweight='bold',
                                color=BAND_COLORS[band])
                
                if j == 0:
                    ax.set_ylabel(f"{roi_name}\n% change", fontsize=9)
                
                if i == n_rois - 1:
                    ax.set_xlabel("Time (s)", fontsize=9)
                
                ax.tick_params(labelsize=8)
                ax.set_xlim(times[0], times[-1])
        
        cond_titles = {
            "all": "All Trials",
            "condition_2": f"{label2} Trials",
            "condition_1": f"{label1} Trials",
            "high_temp": "High Temperature",
            "low_temp": "Low Temperature",
        }
        fig.suptitle(f"How does power evolve in different brain regions?\n{cond_titles.get(cond, cond)} (n={mask.sum()})",
                    fontsize=12, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        
        path = save_dir / "evolution" / f"band_power_by_roi_{cond}.{primary_ext}"
        save_fig(
            fig,
            path,
            logger=logger,
            formats=plot_cfg.formats,
            dpi=plot_cfg.savefig_dpi,
            bbox_inches=plot_cfg.bbox_inches,
            pad_inches=plot_cfg.pad_inches,
        )
        
        saved[f"band_power_roi_{cond}"] = path
    
    return saved


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
    
    # Create figure with 2 rows: condition comparison, temperature comparison
    fig, axes = plt.subplots(2, len(BANDS), figsize=(3*len(BANDS), 6), squeeze=False)
    
    for j, band in enumerate(BANDS):
        times, power = _get_band_power_timecourse(tfr, band, config)
        if len(times) == 0:
            continue
        
        power_bl = _apply_baseline(power, times, baseline)
        
        # Row 1: Condition 2 vs Condition 1
        ax = axes[0, j]
        for cond, color, label in [
            ("condition_2", CONDITION_COLORS["condition_2"], label2),
            ("condition_1", CONDITION_COLORS["condition_1"], label1),
        ]:
            if cond not in masks or masks[cond].sum() < 3:
                continue
            
            mask = masks[cond]
            cond_power = power_bl[mask]
            mean_power = np.nanmean(cond_power, axis=0)
            sem_power = np.nanstd(cond_power, axis=0) / np.sqrt(mask.sum())
            
            ax.fill_between(times, mean_power - sem_power, mean_power + sem_power,
                           alpha=0.2, color=color)
            ax.plot(times, mean_power, color=color, linewidth=2, label=f"{label} (n={mask.sum()})")
        
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
        ax.axvline(0, color='black', linestyle='-', linewidth=0.5)
        ax.set_title(f"{band.upper()}\n({BAND_RANGES[band]})", fontsize=10, 
                    fontweight='bold', color=BAND_COLORS[band])
        if j == 0:
            ax.set_ylabel(f"{label2} vs {label1}\n% change", fontsize=10)
            ax.legend(fontsize=8, loc='upper right')
        ax.set_xlim(times[0], times[-1])
        
        # Row 2: High vs Low Temperature
        ax = axes[1, j]
        for cond, color, label in [("high_temp", CONDITION_COLORS["high_temp"], "High Temp"),
                                    ("low_temp", CONDITION_COLORS["low_temp"], "Low Temp")]:
            if cond not in masks or masks[cond].sum() < 3:
                continue
            
            mask = masks[cond]
            cond_power = power_bl[mask]
            mean_power = np.nanmean(cond_power, axis=0)
            sem_power = np.nanstd(cond_power, axis=0) / np.sqrt(mask.sum())
            
            ax.fill_between(times, mean_power - sem_power, mean_power + sem_power,
                           alpha=0.2, color=color)
            ax.plot(times, mean_power, color=color, linewidth=2, label=f"{label} (n={mask.sum()})")
        
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
        ax.axvline(0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xlabel("Time (s)", fontsize=10)
        if j == 0:
            ax.set_ylabel("High vs Low Temp\n% change", fontsize=10)
            ax.legend(fontsize=8, loc='upper right')
        ax.set_xlim(times[0], times[-1])
    
    fig.suptitle("Do conditions differ in their power dynamics?",
                fontsize=12, fontweight='bold', y=1.02)
    
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
    """Plot condition comparisons for each ROI (focusing on alpha/beta).
    
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
    
    # Focus on alpha and beta (common choices for summary figures)
    focus_bands = ["alpha", "beta"]
    
    # Get available ROIs from config
    rois = get_roi_definitions(config)
    available_rois = []
    for roi_name in rois.keys():
        indices = _get_roi_channel_indices(tfr, roi_name, config)
        if len(indices) > 0:
            available_rois.append(roi_name)
    
    if not available_rois:
        return saved
    
    # Create figure: rows = ROIs, columns = bands, overlaid conditions
    n_rois = len(available_rois)
    n_bands = len(focus_bands)
    
    fig, axes = plt.subplots(n_rois, n_bands * 2, figsize=(4*n_bands*2, 2.5*n_rois), squeeze=False)
    
    for i, roi_name in enumerate(available_rois):
        roi_indices = _get_roi_channel_indices(tfr, roi_name, config)
        
        col_idx = 0
        for band in focus_bands:
            times, power = _get_band_power_timecourse(tfr, band, config, roi_indices)
            if len(times) == 0:
                col_idx += 2
                continue
            
            power_bl = _apply_baseline(power, times, baseline)
            
            # Condition comparison
            ax = axes[i, col_idx]
            for cond, color, label in [
                ("condition_2", CONDITION_COLORS["condition_2"], label2),
                ("condition_1", CONDITION_COLORS["condition_1"], label1),
            ]:
                if cond not in masks or masks[cond].sum() < 3:
                    continue
                
                mask = masks[cond]
                mean_power = np.nanmean(power_bl[mask], axis=0)
                sem_power = np.nanstd(power_bl[mask], axis=0) / np.sqrt(mask.sum())
                
                ax.fill_between(times, mean_power - sem_power, mean_power + sem_power,
                               alpha=0.2, color=color)
                ax.plot(times, mean_power, color=color, linewidth=1.5, label=label)
            
            ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
            ax.axvline(0, color='black', linestyle='-', linewidth=0.5)
            
            if i == 0:
                ax.set_title(f"{band.upper()} - {label2} vs {label1}", fontsize=10, fontweight='bold',
                            color=BAND_COLORS[band])
            if col_idx == 0:
                ax.set_ylabel(f"{roi_name}\n% change", fontsize=9)
            if i == 0 and col_idx == 0:
                ax.legend(fontsize=7, loc='upper right')
            if i == n_rois - 1:
                ax.set_xlabel("Time (s)", fontsize=9)
            ax.set_xlim(times[0], times[-1])
            ax.tick_params(labelsize=8)
            
            # Temperature comparison
            ax = axes[i, col_idx + 1]
            for cond, color, label in [("high_temp", CONDITION_COLORS["high_temp"], "High"),
                                        ("low_temp", CONDITION_COLORS["low_temp"], "Low")]:
                if cond not in masks or masks[cond].sum() < 3:
                    continue
                
                mask = masks[cond]
                mean_power = np.nanmean(power_bl[mask], axis=0)
                sem_power = np.nanstd(power_bl[mask], axis=0) / np.sqrt(mask.sum())
                
                ax.fill_between(times, mean_power - sem_power, mean_power + sem_power,
                               alpha=0.2, color=color)
                ax.plot(times, mean_power, color=color, linewidth=1.5, label=label)
            
            ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
            ax.axvline(0, color='black', linestyle='-', linewidth=0.5)
            
            if i == 0:
                ax.set_title(f"{band.upper()} - Temp", fontsize=10, fontweight='bold',
                            color=BAND_COLORS[band])
            if i == 0 and col_idx == 0:
                ax.legend(fontsize=7, loc='upper right')
            if i == n_rois - 1:
                ax.set_xlabel("Time (s)", fontsize=9)
            ax.set_xlim(times[0], times[-1])
            ax.tick_params(labelsize=8)
            
            col_idx += 2
    
    fig.suptitle("Which brain regions show the largest condition differences?",
                fontsize=12, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    path = save_dir / "evolution" / f"roi_condition_comparison.{primary_ext}"
    save_fig(
        fig,
        path,
        logger=logger,
        formats=plot_cfg.formats,
        dpi=plot_cfg.savefig_dpi,
        bbox_inches=plot_cfg.bbox_inches,
        pad_inches=plot_cfg.pad_inches,
    )
    
    saved["roi_condition_comparison"] = path
    
    return saved


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
    
    # Compute mean active power for each band and condition
    summary_data = []
    
    for band in BANDS:
        times, power = _get_band_power_timecourse(tfr, band, config)
        if len(times) == 0:
            continue
        
        power_bl = _apply_baseline(power, times, baseline)
        
        # Get active indices
        active_mask = (times >= active_window[0]) & (times <= active_window[1])
        if not np.any(active_mask):
            continue
        
        for cond, mask in masks.items():
            if mask.sum() < 3:
                continue
            
            cond_power = power_bl[mask][:, active_mask]
            mean_active = np.nanmean(cond_power)
            sem_active = np.nanstd(cond_power.mean(axis=1)) / np.sqrt(mask.sum())
            
            summary_data.append({
                'band': band,
                'condition': cond,
                'mean': mean_active,
                'sem': sem_active,
                'n': mask.sum(),
            })
    
    if not summary_data:
        return saved
    
    df = pd.DataFrame(summary_data)
    
    # Create grouped bar plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Panel A: Condition 2 vs Condition 1
    ax = axes[0]
    cond_keys = ['condition_2', 'condition_1']
    cond_df = df[df['condition'].isin(cond_keys)]
    
    if len(cond_df) > 0:
        x = np.arange(len(BANDS))
        width = 0.35
        cond_display = {"condition_2": label2, "condition_1": label1}
        
        for i, cond in enumerate(cond_keys):
            key_df = cond_df[cond_df['condition'] == cond]
            if len(key_df) > 0:
                means = [key_df[key_df['band'] == b]['mean'].values[0] if len(key_df[key_df['band'] == b]) > 0 else 0 for b in BANDS]
                sems = [key_df[key_df['band'] == b]['sem'].values[0] if len(key_df[key_df['band'] == b]) > 0 else 0 for b in BANDS]
                
                offset = (i - 0.5) * width
                bars = ax.bar(x + offset, means, width, yerr=sems, 
                             label=str(cond_display.get(cond, cond)).replace('_', ' ').title(),
                             color=CONDITION_COLORS.get(cond, 'gray'),
                             capsize=3, alpha=0.8)
        
        ax.set_xticks(x)
        ax.set_xticklabels([b.upper() for b in BANDS], fontsize=10)
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
        ax.set_ylabel('Mean Active Power (% change)', fontsize=10)
        ax.set_title(f'A. {label2} vs {label1}', fontweight='bold', loc='left', fontsize=11)
        ax.legend(fontsize=9)
    
    # Panel B: High vs Low Temperature
    ax = axes[1]
    temp_conds = ['high_temp', 'low_temp']
    temp_df = df[df['condition'].isin(temp_conds)]
    
    if len(temp_df) > 0:
        x = np.arange(len(BANDS))
        width = 0.35
        
        for i, cond in enumerate(temp_conds):
            cond_df = temp_df[temp_df['condition'] == cond]
            if len(cond_df) > 0:
                means = [cond_df[cond_df['band'] == b]['mean'].values[0] if len(cond_df[cond_df['band'] == b]) > 0 else 0 for b in BANDS]
                sems = [cond_df[cond_df['band'] == b]['sem'].values[0] if len(cond_df[cond_df['band'] == b]) > 0 else 0 for b in BANDS]
                
                offset = (i - 0.5) * width
                bars = ax.bar(x + offset, means, width, yerr=sems,
                             label=cond.replace('_', ' ').title(),
                             color=CONDITION_COLORS.get(cond, 'gray'),
                             capsize=3, alpha=0.8)
        
        ax.set_xticks(x)
        ax.set_xticklabels([b.upper() for b in BANDS], fontsize=10)
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.5)
        ax.set_ylabel('Mean Active Power (% change)', fontsize=10)
        ax.set_title('B. High vs Low Temperature', fontweight='bold', loc='left', fontsize=11)
        ax.legend(fontsize=9)
    
    fig.suptitle("What is the overall pattern of power changes across bands and conditions?",
                fontsize=12, fontweight='bold', y=1.02)
    
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
    
    # 4. ROI condition comparison (alpha/beta focus)
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
