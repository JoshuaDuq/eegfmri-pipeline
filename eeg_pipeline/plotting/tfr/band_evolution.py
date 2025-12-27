"""
TFR Band Power Evolution Plots
==============================

Visualizations showing how power in each frequency band evolves across time,
for different conditions (all trials, pain/non-pain, high/low temperature).

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
from ...utils.analysis.tfr import get_bands_for_tfr
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

# ROI definitions (simplified for visualization)
ROI_CHANNELS = {
    "Frontal": ["Fp1", "Fp2", "Fpz", "AF3", "AF4", "F3", "F4", "Fz", "F7", "F8"],
    "Central": ["C3", "C4", "Cz", "FC3", "FC4", "CP3", "CP4"],
    "Parietal": ["P3", "P4", "Pz", "P7", "P8", "POz"],
    "Occipital": ["O1", "O2", "Oz", "PO3", "PO4", "PO7", "PO8"],
    "Temporal": ["T7", "T8", "TP7", "TP8", "FT7", "FT8"],
}

# Condition colors
CONDITION_COLORS = {
    "all": "#333333",
    "pain": "#C42847",
    "nonpain": "#4C72B0",
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


def _get_roi_channel_indices(tfr: mne.time_frequency.EpochsTFR, roi_name: str) -> List[int]:
    """Get channel indices for a ROI."""
    ch_names = tfr.ch_names
    roi_channels = ROI_CHANNELS.get(roi_name, [])
    indices = [i for i, ch in enumerate(ch_names) if ch in roi_channels]
    return indices


def _find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Find first matching column from candidates."""
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _create_condition_masks(
    events_df: pd.DataFrame,
    config: Any,
) -> Dict[str, np.ndarray]:
    """Create masks for different conditions.
    
    Returns dict with keys: 'all', 'pain', 'nonpain', 'high_temp', 'low_temp'
    """
    n_trials = len(events_df)
    masks = {"all": np.ones(n_trials, dtype=bool)}
    
    # Pain/non-pain - try multiple column names
    pain_candidates = ["pain_binary_coded", "pain_binary", "pain", "painful"]
    pain_col = _find_column(events_df, pain_candidates)
    
    if pain_col:
        pain_vec = pd.to_numeric(events_df[pain_col], errors="coerce").fillna(0).astype(int)
        masks["pain"] = (pain_vec == 1).values
        masks["nonpain"] = (pain_vec == 0).values
    
    # High/low temperature - try multiple column names
    temp_candidates = ["stimulus_temp", "temperature", "temp", "stim_temp"]
    temp_col = _find_column(events_df, temp_candidates)
    
    if temp_col:
        temps = pd.to_numeric(events_df[temp_col], errors="coerce")
        valid_temps = temps.dropna()
        if len(valid_temps) > 0:
            median_temp = valid_temps.median()
            masks["high_temp"] = (temps >= median_temp).fillna(False).values
            masks["low_temp"] = (temps < median_temp).fillna(False).values
    
    return masks


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
        events_df: Events DataFrame with pain/temperature columns
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
    masks = _create_condition_masks(events_df, config)
    conditions = ["all", "pain", "nonpain", "high_temp", "low_temp"]
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
                    "pain": f"Pain\n(n={mask.sum()})",
                    "nonpain": f"Non-Pain\n(n={mask.sum()})",
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
    
    masks = _create_condition_masks(events_df, config)
    plot_cfg = get_plot_config(config)
    primary_ext = plot_cfg.formats[0] if plot_cfg.formats else "png"
    
    # Get available ROIs
    available_rois = []
    for roi_name in ROI_CHANNELS.keys():
        indices = _get_roi_channel_indices(tfr, roi_name)
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
            roi_indices = _get_roi_channel_indices(tfr, roi_name)
            
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
            "pain": "Pain Trials",
            "nonpain": "Non-Pain Trials",
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
    Question: "Do pain and temperature conditions differ in their power dynamics?"
    
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
    
    masks = _create_condition_masks(events_df, config)
    plot_cfg = get_plot_config(config)
    primary_ext = plot_cfg.formats[0] if plot_cfg.formats else "png"
    
    # Create figure with 2 rows: pain comparison, temperature comparison
    fig, axes = plt.subplots(2, len(BANDS), figsize=(3*len(BANDS), 6), squeeze=False)
    
    for j, band in enumerate(BANDS):
        times, power = _get_band_power_timecourse(tfr, band, config)
        if len(times) == 0:
            continue
        
        power_bl = _apply_baseline(power, times, baseline)
        
        # Row 1: Pain vs Non-Pain
        ax = axes[0, j]
        for cond, color, label in [("pain", CONDITION_COLORS["pain"], "Pain"),
                                    ("nonpain", CONDITION_COLORS["nonpain"], "Non-Pain")]:
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
            ax.set_ylabel("Pain vs Non-Pain\n% change", fontsize=10)
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
    
    fig.suptitle("Do pain and temperature conditions differ in their power dynamics?",
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
    
    masks = _create_condition_masks(events_df, config)
    plot_cfg = get_plot_config(config)
    primary_ext = plot_cfg.formats[0] if plot_cfg.formats else "png"
    
    # Focus on alpha and beta (most relevant for pain)
    focus_bands = ["alpha", "beta"]
    
    available_rois = []
    for roi_name in ROI_CHANNELS.keys():
        indices = _get_roi_channel_indices(tfr, roi_name)
        if len(indices) > 0:
            available_rois.append(roi_name)
    
    if not available_rois:
        return saved
    
    # Create figure: rows = ROIs, columns = bands, overlaid conditions
    n_rois = len(available_rois)
    n_bands = len(focus_bands)
    
    fig, axes = plt.subplots(n_rois, n_bands * 2, figsize=(4*n_bands*2, 2.5*n_rois), squeeze=False)
    
    for i, roi_name in enumerate(available_rois):
        roi_indices = _get_roi_channel_indices(tfr, roi_name)
        
        col_idx = 0
        for band in focus_bands:
            times, power = _get_band_power_timecourse(tfr, band, config, roi_indices)
            if len(times) == 0:
                col_idx += 2
                continue
            
            power_bl = _apply_baseline(power, times, baseline)
            
            # Pain comparison
            ax = axes[i, col_idx]
            for cond, color, label in [("pain", CONDITION_COLORS["pain"], "Pain"),
                                        ("nonpain", CONDITION_COLORS["nonpain"], "Non-Pain")]:
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
                ax.set_title(f"{band.upper()} - Pain", fontsize=10, fontweight='bold',
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
    
    masks = _create_condition_masks(events_df, config)
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
    
    # Panel A: Pain vs Non-Pain
    ax = axes[0]
    pain_conds = ['pain', 'nonpain']
    pain_df = df[df['condition'].isin(pain_conds)]
    
    if len(pain_df) > 0:
        x = np.arange(len(BANDS))
        width = 0.35
        
        for i, cond in enumerate(pain_conds):
            cond_df = pain_df[pain_df['condition'] == cond]
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
        ax.set_title('A. Pain vs Non-Pain', fontweight='bold', loc='left', fontsize=11)
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
