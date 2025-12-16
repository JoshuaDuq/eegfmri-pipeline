"""
Microstate visualization plotting functions.

Functions for creating microstate plots including templates, coverage, transitions,
GFP sequences, and group-level analyses.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne

from eeg_pipeline.plotting.io.figures import (
    save_fig,
    extract_eeg_picks,
    log_if_present as _log_if_present,
    validate_picks as _validate_picks,
)
from eeg_pipeline.infra.paths import ensure_dir
from eeg_pipeline.utils.data.columns import find_temperature_column_in_events
from ...utils.data.epochs_loading import resolve_columns
from ...utils.data.alignment import get_aligned_events, align_events_to_epochs
from ...utils.data.alignment import validate_alignment as validate_aligned_events_length
from ...utils.analysis.stats import (
    compute_coverage_statistics,
    compute_consensus_labels,
    format_correlation_text,
)
from scipy.stats import mannwhitneyu
from ...analysis.features.microstates import (
    compute_gfp_with_floor as _compute_gfp,
    label_timecourse as _label_timecourse,
    extract_templates_from_trials as _extract_templates_from_trials,
)
from ..config import get_plot_config
from eeg_pipeline.utils.analysis.events import extract_pain_mask


###################################################################
# Helper Functions
###################################################################


def _create_state_letters(n_states: int) -> List[str]:
    """Create state letter labels (A, B, C, ...).
    
    Args:
        n_states: Number of microstates
    
    Returns:
        List of state letter labels
    """
    return [chr(65 + i) for i in range(n_states)]


def _setup_template_grid(n_states: int, plot_cfg) -> Tuple[plt.Figure, List[plt.Axes]]:
    """Setup subplot grid for microstate templates.
    
    Args:
        n_states: Number of microstates
        plot_cfg: PlotConfig instance
    
    Returns:
        Tuple of (figure, list of axes)
    """
    n_cols = min(4, n_states)
    n_rows = int(np.ceil(n_states / n_cols))
    microstate_config = plot_cfg.plot_type_configs.get("microstate", {})
    
    width_value = plot_cfg.figure_sizes.get("microstate_width_per_state", 3.6)
    width_per_state = float(width_value[0] if isinstance(width_value, tuple) else width_value)
    
    height_value = plot_cfg.figure_sizes.get("microstate_height_per_state", 3.2)
    height_per_state = float(height_value[0] if isinstance(height_value, tuple) else height_value)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(width_per_state * n_cols, height_per_state * n_rows)
    )
    
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    return fig, axes


def _prepare_pvalue_matrix(
    pval_df: Optional[pd.DataFrame],
    metric_labels: List[str],
    state_labels: List[str],
    corr_matrix_shape: Tuple[int, int]
) -> np.ndarray:
    """Prepare p-value matrix for correlation heatmap.
    
    Args:
        pval_df: Optional DataFrame with p-values
        metric_labels: List of metric labels
        state_labels: List of state labels
        corr_matrix_shape: Shape of correlation matrix
    
    Returns:
        P-value matrix array
    """
    if pval_df is not None and not pval_df.empty:
        return pval_df.reindex(index=metric_labels, columns=state_labels).to_numpy(dtype=float)
    return np.full(corr_matrix_shape, np.nan, dtype=float)


def _compute_gfp_and_labels_for_condition(
    epoch_data_condition: np.ndarray,
    templates: np.ndarray,
    time_mask: np.ndarray
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Compute GFP and labels for a condition.
    
    Args:
        epoch_data_condition: Epoch data for condition (n_epochs, n_channels, n_times)
        templates: Microstate templates array (n_states, n_channels)
        time_mask: Boolean mask for time points (n_times,)
    
    Returns:
        Tuple of (list of GFP arrays, list of label arrays)
    """
    gfp_all_trials = []
    labels_all_trials = []
    
    # Validate shapes before processing
    if epoch_data_condition.ndim != 3:
        raise ValueError(f"epoch_data_condition must be 3D, got shape {epoch_data_condition.shape}")
    
    n_epochs, n_channels, n_times = epoch_data_condition.shape
    n_states, n_template_channels = templates.shape
    
    if n_channels != n_template_channels:
        raise ValueError(
            f"Channel mismatch: epoch data has {n_channels} channels, "
            f"templates have {n_template_channels} channels"
        )
    
    if len(time_mask) != n_times:
        raise ValueError(
            f"Time mask length ({len(time_mask)}) does not match epoch time points ({n_times})"
        )
    
    for epoch in epoch_data_condition:
        gfp = _compute_gfp(epoch)
        state_labels, _ = _label_timecourse(epoch, templates)
        gfp_all_trials.append(gfp[time_mask])
        labels_all_trials.append(state_labels[time_mask])
    return gfp_all_trials, labels_all_trials


def _plot_topomap_row(
    fig: plt.Figure,
    gs: Any,
    templates: np.ndarray,
    info_eeg: mne.Info,
    state_letters: List[str],
    colors: np.ndarray,
    n_states: int,
    plot_cfg: Any
) -> None:
    """Plot row of topomaps for microstate templates.
    
    Args:
        fig: Matplotlib figure
        gs: GridSpec object
        templates: Microstate templates array
        info_eeg: MNE Info object
        state_letters: List of state letter labels
        colors: Array of colors for each state
        n_states: Number of microstates
    """
    for state_idx in range(n_states):
        ax_topo = fig.add_subplot(gs[0, state_idx])
        mne.viz.plot_topomap(
            templates[state_idx], info_eeg, axes=ax_topo,
            show=False, contours=6, cmap="RdBu_r"
        )
        ax_topo.set_title(
            f"State {state_letters[state_idx]}", fontsize=plot_cfg.font.suptitle, weight='bold', color=colors[state_idx]
        )


def _plot_gfp_sequence(
    ax_gfp: plt.Axes,
    times: np.ndarray,
    gfp_mean: np.ndarray,
    labels_consensus: np.ndarray,
    colors: np.ndarray,
    plateau_start: Optional[float],
    plot_cfg: Any
) -> None:
    """Plot GFP sequence colored by microstate.
    
    Args:
        ax_gfp: Matplotlib axes for GFP plot
        times: Time array
        gfp_mean: Mean GFP values
        labels_consensus: Consensus state labels
        colors: Array of colors for each state
        plateau_start: Optional plateau start time
        plot_cfg: PlotConfig instance
    """
    microstate_config = plot_cfg.plot_type_configs.get("microstate", {})
    stimulus_start_time = microstate_config.get("stimulus_start_time", 0.0)
    for time_idx in range(len(times) - 1):
        state = labels_consensus[time_idx]
        ax_gfp.fill_between(
            [times[time_idx], times[time_idx + 1]], 0, gfp_mean[time_idx],
            color=colors[state], alpha=plot_cfg.style.alpha_fill, linewidth=0
        )
    ax_gfp.plot(times, gfp_mean, 'k-', linewidth=plot_cfg.style.line.width_thick, alpha=plot_cfg.style.line.alpha_standard)
    ax_gfp.axvline(stimulus_start_time, color='gray', linestyle='--',
                   linewidth=plot_cfg.style.line.width_thin, alpha=plot_cfg.style.line.alpha_dim)
    if plateau_start is not None:
        ax_gfp.axvline(plateau_start, color='gray', linestyle=':',
                       linewidth=plot_cfg.style.line.width_thin, alpha=plot_cfg.style.line.alpha_dim)
    ax_gfp.set_xlabel("Time (s)", fontsize=plot_cfg.font.large)
    ax_gfp.set_ylabel("GFP (μV)", fontsize=plot_cfg.font.large)
    ax_gfp.set_xlim([times[0], times[-1]])
    ax_gfp.grid(True, alpha=plot_cfg.style.alpha_grid, axis='y')


def _validate_gfp_plotting_inputs(
    epochs: mne.Epochs,
    templates: Optional[np.ndarray],
    events_df: Optional[pd.DataFrame],
    logger: logging.Logger
) -> bool:
    """Validate inputs for GFP plotting.
    
    Args:
        epochs: MNE Epochs object
        templates: Optional microstate templates
        events_df: Optional events DataFrame
        logger: Logger instance
    
    Returns:
        True if inputs are valid, False otherwise
    """
    if templates is None or events_df is None or events_df.empty:
        _log_if_present(logger, "warning", "Missing data for GFP microstate plot")
        return False
    return True


def _prepare_gfp_plotting_data(
    epochs: mne.Epochs,
    events_df: pd.DataFrame,
    config: Any,
    logger: logging.Logger
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[List[float]]]:
    """Prepare data for GFP plotting.
    
    Args:
        epochs: MNE Epochs object
        events_df: Events DataFrame
        config: Configuration object
        logger: Logger instance
    
    Returns:
        Tuple of (epoch_data, times_stimulus, nonpain_mask, pain_mask, plateau_window)
        Returns (None, None, None, None, None) on failure
    """
    pain_col, _, _ = resolve_columns(events_df, config=config)
    if pain_col is None:
        _log_if_present(logger, "warning", "No pain binary column found")
        return None, None, None, None, None
    
    aligned_events = align_events_to_epochs(events_df, epochs, logger=logger)
    if not validate_aligned_events_length(aligned_events, epochs, logger):
        return None, None, None, None, None
    
    picks = extract_eeg_picks(epochs)
    if not _validate_picks(picks, logger):
        return None, None, None, None, None
    
    pain_values = pd.to_numeric(aligned_events[pain_col], errors="coerce")
    epoch_data = epochs.get_data()[:, picks, :]
    times = epochs.times
    plateau_window = config.get("time_frequency_analysis.plateau_window", [3.0, 10.5]) if config else [3.0, 10.5]
    plateau_end = float(plateau_window[1])
    stimulus_mask = (times >= 0.0) & (times <= plateau_end)
    
    if not stimulus_mask.any():
        _log_if_present(logger, "warning", "No timepoints in stimulus window")
        return None, None, None, None, None
    
    times_stimulus = times[stimulus_mask]
    nonpain_mask = (pain_values == 0).to_numpy()
    pain_mask = (pain_values == 1).to_numpy()
    
    n_nonpain_trials = nonpain_mask.sum()
    n_pain_trials = pain_mask.sum()
    plot_cfg = get_plot_config(config)
    min_trials_for_comparison = plot_cfg.validation.get("min_trials_for_comparison", 1)
    if n_nonpain_trials < min_trials_for_comparison or n_pain_trials < min_trials_for_comparison:
        _log_if_present(logger, "warning", "Insufficient trials for pain comparison")
        return None, None, None, None, None
    
    return epoch_data, times_stimulus, nonpain_mask, pain_mask, plateau_window


def _create_gfp_sequence_figure(
    templates: np.ndarray,
    info_eeg: mne.Info,
    state_letters: List[str],
    colors: np.ndarray,
    n_states: int,
    times_stimulus: np.ndarray,
    gfp_mean: np.ndarray,
    labels_consensus: np.ndarray,
    plateau_start: float,
    plot_cfg: Any
) -> plt.Figure:
    """Create GFP sequence figure with topomaps and GFP plot.
    
    Args:
        templates: Microstate templates array
        info_eeg: MNE Info object
        state_letters: List of state letter labels
        colors: Array of colors for each state
        n_states: Number of microstates
        times_stimulus: Time array for stimulus period
        gfp_mean: Mean GFP values
        labels_consensus: Consensus state labels
        plateau_start: Plateau start time
        plot_cfg: PlotConfig instance
    
    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=(14, 6))
    microstate_config = plot_cfg.plot_type_configs.get("microstate", {})
    grid_height_ratio_topomap = microstate_config.get("grid_height_ratio_topomap", 1.0)
    grid_height_ratio_gfp = microstate_config.get("grid_height_ratio_gfp", 1.5)
    grid_hspace = microstate_config.get("grid_hspace", 0.15)
    grid_wspace = microstate_config.get("grid_wspace", 0.3)
    gs = fig.add_gridspec(
        2, n_states,
        height_ratios=[grid_height_ratio_topomap, grid_height_ratio_gfp],
        hspace=grid_hspace, wspace=grid_wspace
    )
    
    _plot_topomap_row(fig, gs, templates, info_eeg, state_letters, colors, n_states, plot_cfg)
    
    ax_gfp = fig.add_subplot(gs[1, :])
    _plot_gfp_sequence(ax_gfp, times_stimulus, gfp_mean, labels_consensus, colors, plateau_start, plot_cfg)
    
    return fig


def _parse_temporal_bin_config(
    bin_config: Any,
    config: Optional[Any] = None
) -> Optional[Tuple[float, float, str]]:
    """Parse temporal bin configuration.
    
    Args:
        bin_config: Bin configuration (dict or list/tuple)
        config: Optional configuration object
    
    Returns:
        Tuple of (start_time, end_time, label) or None if invalid
    """
    if isinstance(bin_config, dict):
        if config is not None:
            plot_cfg = get_plot_config(config)
            microstate_config = plot_cfg.plot_type_configs.get("microstate", {})
            stimulus_start_time = microstate_config.get("stimulus_start_time", 0.0)
        else:
            stimulus_start_time = 0.0
        start_time = float(bin_config.get("start", stimulus_start_time))
        end_time = float(bin_config.get("end", stimulus_start_time))
        label = str(bin_config.get("label", "unknown"))
        return start_time, end_time, label
    
    if isinstance(bin_config, (list, tuple)) and len(bin_config) >= 3:
        start_time = float(bin_config[0])
        end_time = float(bin_config[1])
        label = str(bin_config[2])
        return start_time, end_time, label
    
    return None


def _compute_gfp_and_labels_for_bin(
    epoch_data_condition: np.ndarray,
    templates: np.ndarray,
    bin_mask: np.ndarray
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Compute GFP and labels for temporal bin.
    
    Args:
        epoch_data_condition: Epoch data for condition
        templates: Microstate templates array
        bin_mask: Boolean mask for time bin
    
    Returns:
        Tuple of (list of GFP arrays, list of label arrays)
    """
    gfp_all_trials = []
    labels_all_trials = []
    for epoch in epoch_data_condition:
        gfp = _compute_gfp(epoch)
        state_labels, _ = _label_timecourse(epoch, templates)
        gfp_all_trials.append(gfp[bin_mask])
        labels_all_trials.append(state_labels[bin_mask])
    return gfp_all_trials, labels_all_trials


def _plot_gfp_sequence_for_bin(
    templates: np.ndarray,
    info_eeg: mne.Info,
    times_bin: np.ndarray,
    gfp_mean: np.ndarray,
    labels_consensus: np.ndarray,
    state_letters: List[str],
    colors: np.ndarray,
    n_states: int,
    condition_title: str,
    temporal_label: str,
    plot_cfg: Any
) -> plt.Figure:
    """Plot GFP sequence for temporal bin.
    
    Args:
        templates: Microstate templates array
        info_eeg: MNE Info object
        times_bin: Time array for bin
        gfp_mean: Mean GFP values
        labels_consensus: Consensus state labels
        state_letters: List of state letter labels
        colors: Array of colors for each state
        n_states: Number of microstates
        condition_title: Condition title string
        temporal_label: Temporal bin label
        plot_cfg: PlotConfig instance
    
    Returns:
        Matplotlib figure
    """
    microstate_config = plot_cfg.plot_type_configs.get("microstate", {})
    grid_height_ratio_topomap = microstate_config.get("grid_height_ratio_topomap", 1.0)
    grid_height_ratio_gfp = microstate_config.get("grid_height_ratio_gfp", 1.5)
    grid_hspace = microstate_config.get("grid_hspace", 0.15)
    grid_wspace = microstate_config.get("grid_wspace", 0.3)
    stimulus_start_time = microstate_config.get("stimulus_start_time", 0.0)
    
    fig = plt.figure(figsize=(14, 6))
    gs = fig.add_gridspec(
        2, n_states,
        height_ratios=[grid_height_ratio_topomap, grid_height_ratio_gfp],
        hspace=grid_hspace, wspace=grid_wspace
    )
    
    _plot_topomap_row(fig, gs, templates, info_eeg, state_letters, colors, n_states, plot_cfg)
    
    ax_gfp = fig.add_subplot(gs[1, :])
    plateau_start = times_bin[0] if times_bin[0] > 0 else None
    _plot_gfp_sequence(ax_gfp, times_bin, gfp_mean, labels_consensus, colors, plateau_start, plot_cfg)
    if times_bin[0] <= stimulus_start_time <= times_bin[-1]:
        ax_gfp.axvline(
            stimulus_start_time, color='gray', linestyle='--',
            linewidth=plot_cfg.style.line.width_thin, alpha=plot_cfg.style.line.alpha_dim
        )
    
    fig.suptitle(
        f"Microstate Sequence - {condition_title} Trials ({temporal_label.capitalize()} Period)",
        fontsize=13, weight='bold'
    )
    return fig


###################################################################
# Microstate Template Plotting
###################################################################


def plot_microstate_templates(
    templates: np.ndarray,
    info: mne.Info,
    subject: str,
    save_dir: Path,
    n_states: int,
    logger: logging.Logger,
    config: Any
) -> None:
    """Plot microstate templates as topomaps.
    
    Args:
        templates: Microstate templates array (n_states, n_channels)
        info: MNE Info object
        subject: Subject identifier
        save_dir: Directory to save plot
        n_states: Number of microstates
        logger: Logger instance
        config: Configuration object
    """
    if templates is None or len(templates) == 0:
        _log_if_present(logger, "warning", "No templates to plot")
        return
    
    plot_cfg = get_plot_config(config)
    fig, axes = _setup_template_grid(n_states, plot_cfg)
    state_letters = _create_state_letters(n_states)
    
    for i in range(n_states):
        mne.viz.plot_topomap(templates[i], info, axes=axes[i], show=False, contours=6, cmap="RdBu_r")
        axes[i].set_title(f"State {state_letters[i]}", fontsize=plot_cfg.font.title)
    
    for j in range(n_states, len(axes)):
        axes[j].set_visible(False)
    
    plt.suptitle(f"Microstate Templates (K={n_states})", fontsize=plot_cfg.font.figure_title)
    microstate_config = plot_cfg.plot_type_configs.get("microstate", {})
    tight_rect = microstate_config.get("tight_rect_microstate", plot_cfg.get_layout_rect("tight_rect_microstate"))
    plt.tight_layout(rect=tight_rect)
    save_fig(fig, save_dir / f"sub-{subject}_microstate_templates", 
             formats=plot_cfg.formats, dpi=plot_cfg.dpi, 
             bbox_inches=plot_cfg.bbox_inches, pad_inches=plot_cfg.pad_inches)
    plt.close(fig)
    _log_if_present(logger, "info", "Saved microstate templates")


def plot_microstate_templates_by_pain(
    epochs: mne.Epochs,
    events_df: pd.DataFrame,
    subject: str,
    task: str,
    save_dir: Path,
    n_states: int,
    logger: logging.Logger,
    config: Any
) -> None:
    """Plot microstate templates separately for pain and non-pain conditions.
    
    Args:
        epochs: MNE Epochs object
        events_df: Events DataFrame
        subject: Subject identifier
        task: Task name
        save_dir: Directory to save plots
        n_states: Number of microstates
        logger: Logger instance
        config: Configuration object
    """
    if events_df is None or events_df.empty:
        _log_if_present(logger, "warning", "Missing events for pain-specific templates")
        return
    
    aligned_events = get_aligned_events(
        epochs, subject, task, strict=True, config=config, logger=logger
    )
    if aligned_events is None:
        _log_if_present(logger, "error", "Alignment failed for plotting function")
        return
    
    pain_col, _, _ = resolve_columns(aligned_events, config=config)
    if pain_col is None:
        _log_if_present(logger, "warning", "No pain binary column found")
        return
    
    picks = extract_eeg_picks(epochs)
    if not _validate_picks(picks, logger):
        _log_if_present(logger, "warning", "No EEG channels for pain-specific templates")
        return
    
    pain_vals = pd.to_numeric(aligned_events[pain_col], errors="coerce")
    nonpain_mask = (pain_vals == 0).to_numpy()
    pain_mask = (pain_vals == 1).to_numpy()
    n_nonpain = int(nonpain_mask.sum())
    n_pain = int(pain_mask.sum())
    
    plot_cfg = get_plot_config(config)
    min_trials_for_templates = plot_cfg.validation.get("min_trials_for_templates", 5)
    if (n_nonpain < min_trials_for_templates or
            n_pain < min_trials_for_templates):
        _log_if_present(logger, "warning", "Insufficient trials for pain-specific templates")
        return
    
    X = epochs.get_data()[:, picks, :]
    sfreq = float(epochs.info["sfreq"])
    templates_nonpain = _extract_templates_from_trials(
        X[nonpain_mask], sfreq, n_states
    )
    templates_pain = _extract_templates_from_trials(
        X[pain_mask], sfreq, n_states
    )
    
    if templates_nonpain is None or templates_pain is None:
        _log_if_present(logger, "warning", "Could not compute pain-specific templates")
        return
    
    plot_cfg = get_plot_config(config)
    info_eeg = mne.pick_info(epochs.info, picks)
    state_letters = _create_state_letters(n_states)
    width_value = plot_cfg.figure_sizes.get("microstate_width_per_state", 3.6)
    width_per_state = float(width_value[0] if isinstance(width_value, tuple) else width_value)
    height_value = plot_cfg.figure_sizes.get("microstate_height_templates", 7.0)
    height_templates = float(height_value[0] if isinstance(height_value, tuple) else height_value)
    fig, axes = plt.subplots(
        2, n_states,
        figsize=(width_per_state * n_states, height_templates)
    )
    if n_states == 1:
        axes = axes.reshape(2, 1)
    
    for i in range(n_states):
        mne.viz.plot_topomap(
            templates_nonpain[i], info_eeg, axes=axes[0, i],
            show=False, contours=6, cmap="RdBu_r"
        )
        axes[0, i].set_title(f"State {state_letters[i]}", fontsize=plot_cfg.font.title)
        mne.viz.plot_topomap(
            templates_pain[i], info_eeg, axes=axes[1, i],
            show=False, contours=6, cmap="RdBu_r"
        )
    
    microstate_config = plot_cfg.plot_type_configs.get("microstate", {})
    label_x_offset = microstate_config.get("label_offset_x", -0.3)
    label_y_position = microstate_config.get("label_y_position", 0.5)
    axes[0, 0].text(
        label_x_offset, label_y_position, f"Non-pain (n={n_nonpain})",
        transform=axes[0, 0].transAxes,
        fontsize=plot_cfg.font.large, rotation=90, va='center', weight='bold'
    )
    axes[1, 0].text(
        label_x_offset, label_y_position, f"Pain (n={n_pain})",
        transform=axes[1, 0].transAxes,
        fontsize=plot_cfg.font.large, rotation=90, va='center', weight='bold'
    )
    
    plot_cfg = get_plot_config(config)
    tight_rect = plot_cfg.plot_type_configs.get("microstate", {}).get("tight_rect_microstate", plot_cfg.get_layout_rect("tight_rect_microstate"))
    plt.suptitle(f"Microstate Templates by Pain Condition (K={n_states})", fontsize=plot_cfg.font.figure_title)
    plt.tight_layout(rect=tight_rect)
    save_fig(
        fig,
        save_dir / f"sub-{subject}_microstate_templates_by_pain",
        formats=plot_cfg.formats, dpi=plot_cfg.dpi,
        bbox_inches=plot_cfg.bbox_inches, pad_inches=plot_cfg.pad_inches
    )
    plt.close(fig)
    _log_if_present(logger, "info", "Saved microstate templates by pain condition")


def plot_microstate_templates_by_temperature(
    epochs: mne.Epochs,
    events_df: pd.DataFrame,
    subject: str,
    task: str,
    save_dir: Path,
    n_states: int,
    logger: logging.Logger,
    config: Any
) -> None:
    """Plot microstate templates separately for different temperature levels.
    
    Args:
        epochs: MNE Epochs object
        events_df: Events DataFrame
        subject: Subject identifier
        task: Task name
        save_dir: Directory to save plots
        n_states: Number of microstates
        logger: Logger instance
        config: Configuration object
    """
    if events_df is None or events_df.empty:
        _log_if_present(logger, "warning", "Missing events for temperature-specific templates")
        return
    
    aligned_events = get_aligned_events(
        epochs, subject, task, strict=True, config=config, logger=logger
    )
    if aligned_events is None:
        _log_if_present(logger, "error", "Alignment failed for plotting function")
        return
    
    temp_col = find_temperature_column_in_events(aligned_events)
    if temp_col is None:
        _log_if_present(logger, "warning", "No temperature column found")
        return
    
    picks = extract_eeg_picks(epochs)
    if not _validate_picks(picks, logger):
        _log_if_present(logger, "warning", "No EEG channels for temperature-specific templates")
        return
    
    temps = pd.to_numeric(aligned_events[temp_col], errors="coerce")
    unique_temps = sorted(temps.dropna().unique())
    if len(unique_temps) < 2:
        _log_if_present(logger, "warning", "Insufficient temperature levels for comparison")
        return
    
    X = epochs.get_data()[:, picks, :]
    sfreq = float(epochs.info["sfreq"])
    templates_by_temp = {}
    
    for temp in unique_temps:
        temp_mask = (temps == temp).to_numpy()
        plot_cfg = get_plot_config(config)
        min_trials_for_templates = plot_cfg.validation.get("min_trials_for_templates", 5)
        if temp_mask.sum() < min_trials_for_templates:
            continue
        templates = _extract_templates_from_trials(X[temp_mask], sfreq, n_states)
        if templates is not None:
            templates_by_temp[temp] = templates
    
    if len(templates_by_temp) < 2:
        _log_if_present(logger, "warning", "Could not compute templates for multiple temperatures")
        return
    
    info_eeg = mne.pick_info(epochs.info, picks)
    state_letters = _create_state_letters(n_states)
    plot_cfg = get_plot_config(config)
    sorted_temps = sorted(templates_by_temp.keys())
    n_temps = len(sorted_temps)
    width_value = plot_cfg.figure_sizes.get("microstate_width_per_state", 3.6)
    width_per_state = float(width_value[0] if isinstance(width_value, tuple) else width_value)
    height_value = plot_cfg.figure_sizes.get("microstate_height_per_state", 3.2)
    height_per_state = float(height_value[0] if isinstance(height_value, tuple) else height_value)
    fig, axes = plt.subplots(
        n_temps, n_states,
        figsize=(width_per_state * n_states, height_per_state * n_temps)
    )
    if n_temps == 1 or n_states == 1:
        axes = axes.reshape(n_temps, n_states)
    
    for row_idx, temp in enumerate(sorted_temps):
        templates = templates_by_temp[temp]
        for col_idx in range(n_states):
            mne.viz.plot_topomap(
                templates[col_idx], info_eeg, axes=axes[row_idx, col_idx],
                show=False, contours=6, cmap="RdBu_r"
            )
            if row_idx == 0:
                axes[row_idx, col_idx].set_title(
                    f"State {state_letters[col_idx]}", fontsize=plot_cfg.font.title
                )
        microstate_config = plot_cfg.plot_type_configs.get("microstate", {})
        temp_label_x = microstate_config.get("temperature_label_x", -0.35)
        label_y_position = microstate_config.get("label_y_position", 0.5)
        axes[row_idx, 0].text(
            temp_label_x, label_y_position, f"{temp:.1f}°C",
            transform=axes[row_idx, 0].transAxes,
            fontsize=plot_cfg.font.large, rotation=90, va='center', weight='bold'
        )
    
    plot_cfg = get_plot_config(config)
    tight_rect = plot_cfg.plot_type_configs.get("microstate", {}).get("tight_rect_microstate", plot_cfg.get_layout_rect("tight_rect_microstate"))
    plt.suptitle(f"Microstate Templates by Temperature (K={n_states})", fontsize=plot_cfg.font.figure_title)
    plt.tight_layout(rect=tight_rect)
    save_fig(
        fig,
        save_dir / f"sub-{subject}_microstate_templates_by_temperature",
        formats=plot_cfg.formats, dpi=plot_cfg.dpi,
        bbox_inches=plot_cfg.bbox_inches, pad_inches=plot_cfg.pad_inches
    )
    plt.close(fig)
    _log_if_present(logger, "info", "Saved microstate templates by temperature")


###################################################################
# Microstate Coverage and Correlation Plotting
###################################################################


def plot_microstate_coverage_by_pain(
    ms_df: pd.DataFrame,
    events_df: pd.DataFrame,
    subject: str,
    save_dir: Path,
    n_states: int,
    logger: logging.Logger,
    config: Any
) -> None:
    """Plot microstate coverage by pain condition.
    
    Args:
        ms_df: Microstate DataFrame
        events_df: Events DataFrame
        subject: Subject identifier
        save_dir: Directory to save plot
        n_states: Number of microstates
        logger: Logger instance
        config: Configuration object
    """
    if ms_df is None or ms_df.empty or events_df is None or events_df.empty:
        _log_if_present(logger, "warning", "Missing data for coverage by pain plot")
        return
    
    pain_col, _, _ = resolve_columns(events_df, config=config)
    if pain_col is None:
        _log_if_present(logger, "warning", "No pain binary column found")
        return
    
    if len(ms_df) != len(events_df):
        raise ValueError(
            f"Microstate dataframe ({len(ms_df)} rows) and events "
            f"({len(events_df)} rows) length mismatch for subject {subject}"
        )
    
    pain_values = pd.to_numeric(events_df[pain_col], errors="coerce")
    valid_mask = pain_values.notna()
    nonpain_mask = valid_mask & (pain_values == 0)
    pain_mask = valid_mask & (pain_values == 1)
    n_nonpain = int(nonpain_mask.sum())
    n_pain = int(pain_mask.sum())
    state_letters = _create_state_letters(n_states)
    means_nonpain, means_pain, sems_nonpain, sems_pain = [], [], [], []
    
    for state_idx in range(n_states):
        coverage_column = f"ms_coverage_{state_idx}"
        if coverage_column not in ms_df.columns:
            means_nonpain.append(0.0)
            means_pain.append(0.0)
            sems_nonpain.append(0.0)
            sems_pain.append(0.0)
            continue
        
        coverage_values = pd.to_numeric(ms_df[coverage_column], errors="coerce")
        mean_np, mean_p, sem_np, sem_p = compute_coverage_statistics(
            coverage_values, nonpain_mask, pain_mask
        )
        means_nonpain.append(mean_np)
        means_pain.append(mean_p)
        sems_nonpain.append(sem_np)
        sems_pain.append(sem_p)
    
    plot_cfg = get_plot_config(config)
    fig_size = plot_cfg.get_figure_size("medium", plot_type="features")
    x_positions = np.arange(n_states)
    fig, ax = plt.subplots(figsize=fig_size)
    
    nonpain_color = plot_cfg.get_color("nonpain", plot_type="features")
    pain_color = plot_cfg.get_color("pain", plot_type="features")
    
    ax.bar(
        x_positions - plot_cfg.style.bar.width/2, means_nonpain, plot_cfg.style.bar.width, 
        yerr=sems_nonpain, label=f'Non-pain (n={n_nonpain})', color=nonpain_color, 
        alpha=plot_cfg.style.bar.alpha, capsize=plot_cfg.style.bar.capsize
    )
    ax.bar(
        x_positions + plot_cfg.style.bar.width/2, means_pain, plot_cfg.style.bar.width, 
        yerr=sems_pain, label=f'Pain (n={n_pain})', color=pain_color, 
        alpha=plot_cfg.style.bar.alpha, capsize=plot_cfg.style.bar.capsize
    )
    ax.set_xticks(x_positions)
    ax.set_xticklabels(state_letters)
    ax.set_xlabel("Microstate", fontsize=plot_cfg.font.ylabel)
    ax.set_ylabel("Coverage (fraction of time)", fontsize=plot_cfg.font.ylabel)
    ax.set_title("Microstate Coverage by Pain Condition", fontsize=plot_cfg.font.figure_title)
    ax.legend(fontsize=plot_cfg.font.title)
    ax.grid(True, alpha=plot_cfg.style.alpha_grid, axis='y')
    
    footer_text = f"K={n_states} states | n={len(ms_df)} total trials"
    fig.text(
        0.99, 0.01, footer_text,
        ha='right', va='bottom',
        fontsize=plot_cfg.font.small,
        color='gray', alpha=0.8
    )
    
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    save_fig(
        fig,
        save_dir / f"sub-{subject}_microstate_coverage_by_pain",
        formats=plot_cfg.formats, dpi=plot_cfg.dpi,
        bbox_inches=plot_cfg.bbox_inches, pad_inches=plot_cfg.pad_inches
    )
    plt.close(fig)
    _log_if_present(logger, "info", "Saved microstate coverage by pain condition")





###################################################################
# Microstate Temporal Evolution Plotting
###################################################################


def plot_microstate_temporal_evolution(
    epochs: mne.Epochs,
    templates: np.ndarray,
    events_df: pd.DataFrame,
    subject: str,
    task: str,
    save_dir: Path,
    n_states: int,
    logger: logging.Logger,
    config: Any
) -> None:
    """Plot temporal evolution of microstate probabilities.
    
    Args:
        epochs: MNE Epochs object
        templates: Microstate templates array
        events_df: Events DataFrame
        subject: Subject identifier
        task: Task name
        save_dir: Directory to save plot
        n_states: Number of microstates
        logger: Logger instance
        config: Configuration object
    """
    if templates is None or events_df is None or events_df.empty:
        _log_if_present(logger, "warning", "Missing data for temporal evolution plot")
        return
    
    aligned_events = get_aligned_events(
        epochs, subject, task, strict=True, config=config, logger=logger
    )
    if aligned_events is None:
        _log_if_present(logger, "error", "Alignment failed for plotting function")
        return
    
    pain_col, _, _ = resolve_columns(aligned_events, config=config)
    if pain_col is None:
        _log_if_present(logger, "warning", "No pain binary column found")
        return
    
    picks = extract_eeg_picks(epochs)
    if not _validate_picks(picks, logger):
        return
    
    epoch_data = epochs.get_data()[:, picks, :]
    
    # Validate templates match epoch channel count
    n_epoch_channels = epoch_data.shape[1]
    n_template_channels = templates.shape[1]
    if n_epoch_channels != n_template_channels:
        _log_if_present(
            logger, "warning",
            f"Channel mismatch: epoch data has {n_epoch_channels} channels, "
            f"templates have {n_template_channels}. Skipping temporal evolution plot."
        )
        return
    
    times = epochs.times
    pain_values = pd.to_numeric(aligned_events[pain_col], errors="coerce")
    nonpain_mask = (pain_values == 0).to_numpy()
    pain_mask = (pain_values == 1).to_numpy()
    state_probabilities_nonpain = np.zeros((n_states, len(times)))
    state_probabilities_pain = np.zeros((n_states, len(times)))
    
    for trial_idx in range(len(epoch_data)):
        epoch = epoch_data[trial_idx]
        state_labels, _ = _label_timecourse(epoch, templates)
        
        if nonpain_mask[trial_idx]:
            for time_idx, state_idx in enumerate(state_labels):
                state_probabilities_nonpain[state_idx, time_idx] += 1
        elif pain_mask[trial_idx]:
            for time_idx, state_idx in enumerate(state_labels):
                state_probabilities_pain[state_idx, time_idx] += 1
    
    n_nonpain_trials = max(1, nonpain_mask.sum())
    n_pain_trials = max(1, pain_mask.sum())
    state_probabilities_nonpain /= n_nonpain_trials
    state_probabilities_pain /= n_pain_trials
    
    plot_cfg = get_plot_config(config)
    state_letters = _create_state_letters(n_states)
    colors = plt.cm.Set2(np.linspace(0, 1, n_states))
    microstate_config = plot_cfg.plot_type_configs.get("microstate", {})
    stimulus_start_time = microstate_config.get("stimulus_start_time", 0.0)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for state_idx in range(n_states):
        axes[0].plot(
            times, state_probabilities_nonpain[state_idx],
            label=f"State {state_letters[state_idx]}",
            color=colors[state_idx], linewidth=plot_cfg.style.line.width_thick
        )
        axes[1].plot(
            times, state_probabilities_pain[state_idx],
            label=f"State {state_letters[state_idx]}",
            color=colors[state_idx], linewidth=plot_cfg.style.line.width_thick
        )
    
    axes[0].axvline(stimulus_start_time, color='k', linestyle='--',
                    linewidth=plot_cfg.style.line.width_thin, alpha=plot_cfg.style.line.alpha_dim)
    axes[1].axvline(stimulus_start_time, color='k', linestyle='--',
                    linewidth=plot_cfg.style.line.width_thin, alpha=plot_cfg.style.line.alpha_dim)
    axes[0].set_ylabel("Probability", fontsize=plot_cfg.font.ylabel)
    axes[0].set_title("Non-pain Trials", fontsize=plot_cfg.font.large)
    axes[0].legend(loc='upper right', fontsize=plot_cfg.font.medium, ncol=n_states)
    axes[0].grid(True, alpha=plot_cfg.style.alpha_grid)
    axes[0].set_ylim([0, 1])
    axes[1].set_ylabel("Probability", fontsize=plot_cfg.font.ylabel)
    axes[1].set_xlabel("Time (s)", fontsize=plot_cfg.font.label)
    axes[1].set_title("Pain Trials", fontsize=plot_cfg.font.large)
    axes[1].legend(loc='upper right', fontsize=plot_cfg.font.medium, ncol=n_states)
    axes[1].grid(True, alpha=plot_cfg.style.alpha_grid)
    axes[1].set_ylim([0, 1])
    
    tight_rect = microstate_config.get("tight_rect_microstate", plot_cfg.get_layout_rect("tight_rect_microstate"))
    plt.suptitle("Temporal Evolution of Microstate Probabilities", fontsize=plot_cfg.font.figure_title)
    plt.tight_layout(rect=tight_rect)
    save_fig(
        fig,
        save_dir / f"sub-{subject}_microstate_temporal_evolution",
        formats=plot_cfg.formats, dpi=plot_cfg.dpi,
        bbox_inches=plot_cfg.bbox_inches, pad_inches=plot_cfg.pad_inches
    )
    plt.close(fig)
    _log_if_present(logger, "info", "Saved microstate temporal evolution")


###################################################################
# Microstate GFP Sequence Plotting
###################################################################


def plot_microstate_gfp_colored_by_state(
    epochs: mne.Epochs,
    templates: np.ndarray,
    events_df: pd.DataFrame,
    subject: str,
    save_dir: Path,
    n_states: int,
    logger: logging.Logger,
    config: Any
) -> None:
    """Plot GFP sequence colored by microstate for pain and non-pain conditions.
    
    Args:
        epochs: MNE Epochs object
        templates: Microstate templates array
        events_df: Events DataFrame
        subject: Subject identifier
        save_dir: Directory to save plots
        n_states: Number of microstates
        logger: Logger instance
        config: Configuration object
    """
    if not _validate_gfp_plotting_inputs(epochs, templates, events_df, logger):
        return
    
    result = _prepare_gfp_plotting_data(epochs, events_df, config, logger)
    if result[0] is None:
        return
    
    epoch_data, times_stimulus, nonpain_mask, pain_mask, plateau_window = result
    
    picks = extract_eeg_picks(epochs)
    
    # Validate templates match epoch channel count
    n_epoch_channels = epoch_data.shape[1]
    n_template_channels = templates.shape[1]
    if n_epoch_channels != n_template_channels:
        _log_if_present(
            logger, "warning",
            f"Channel mismatch: epoch data has {n_epoch_channels} channels, "
            f"templates have {n_template_channels}. Skipping GFP colored plot."
        )
        return
    info_eeg = mne.pick_info(epochs.info, picks)
    state_letters = _create_state_letters(n_states)
    colors = plt.cm.Set2(np.linspace(0, 1, n_states))
    plateau_start = float(plateau_window[0])
    plateau_end = float(plateau_window[1])
    plot_cfg = get_plot_config(config)
    microstate_config = plot_cfg.plot_type_configs.get("microstate", {})
    stimulus_start_time = microstate_config.get("stimulus_start_time", 0.0)
    gfp_scale_factor_value = microstate_config.get("gfp_scale_factor", 1e6)
    gfp_scale_factor = float(gfp_scale_factor_value)
    min_trials_for_comparison = plot_cfg.validation.get("min_trials_for_comparison", 1)
    
    times = epochs.times
    stimulus_mask = (times >= stimulus_start_time) & (times <= plateau_end)
    
    for condition_mask, condition_label in [(nonpain_mask, "nonpain"), (pain_mask, "pain")]:
        if condition_mask.sum() < min_trials_for_comparison:
            continue
        
        epoch_data_condition = epoch_data[condition_mask]
        gfp_all_trials, labels_all_trials = _compute_gfp_and_labels_for_condition(
            epoch_data_condition, templates, stimulus_mask
        )
        
        gfp_mean = np.mean(gfp_all_trials, axis=0) * gfp_scale_factor
        labels_consensus = compute_consensus_labels(labels_all_trials, len(times_stimulus))
        
        fig = _create_gfp_sequence_figure(
            templates, info_eeg, state_letters, colors, n_states,
            times_stimulus, gfp_mean, labels_consensus, plateau_start, plot_cfg
        )
        
        condition_title = "Non-pain" if condition_label == "nonpain" else "Pain"
        fig.suptitle(
            f"Microstate Sequence - {condition_title} Trials (Stimulus Period)",
            fontsize=13,
            weight='bold'
        )
        plot_cfg = get_plot_config(config)
        tight_rect = plot_cfg.plot_type_configs.get("microstate", {}).get("tight_rect_microstate", plot_cfg.get_layout_rect("tight_rect_microstate"))
        plt.tight_layout(rect=tight_rect)
        save_fig(
            fig,
            save_dir / f"sub-{subject}_microstate_gfp_sequence_{condition_label}",
            formats=plot_cfg.formats, dpi=plot_cfg.dpi,
            bbox_inches=plot_cfg.bbox_inches, pad_inches=plot_cfg.pad_inches
        )
        plt.close(fig)
    
    _log_if_present(logger, "info", "Saved microstate GFP sequence plots")


def plot_microstate_gfp_by_temporal_bins(
    epochs: mne.Epochs,
    templates: np.ndarray,
    events_df: pd.DataFrame,
    subject: str,
    task: str,
    save_dir: Path,
    n_states: int,
    logger: logging.Logger,
    config: Any
) -> None:
    """Plot GFP sequence by temporal bins for pain and non-pain conditions.
    
    Args:
        epochs: MNE Epochs object
        templates: Microstate templates array
        events_df: Events DataFrame
        subject: Subject identifier
        task: Task name
        save_dir: Directory to save plots
        n_states: Number of microstates
        logger: Logger instance
        config: Configuration object
    """
    if templates is None or events_df is None or events_df.empty:
        _log_if_present(logger, "warning", "Missing data for temporal bin GFP plots")
        return
    
    aligned_events = get_aligned_events(
        epochs, subject, task, strict=True, config=config, logger=logger
    )
    if aligned_events is None:
        _log_if_present(logger, "error", "Alignment failed for plotting function")
        return
    
    pain_col, _, _ = resolve_columns(aligned_events, config=config)
    if pain_col is None:
        _log_if_present(logger, "warning", "No pain binary column found")
        return
    
    picks = extract_eeg_picks(epochs)
    if not _validate_picks(picks, logger):
        return
    
    pain_values = pd.to_numeric(aligned_events[pain_col], errors="coerce")
    epoch_data = epochs.get_data()[:, picks, :]
    
    # Validate templates match epoch channel count
    n_epoch_channels = epoch_data.shape[1]
    n_template_channels = templates.shape[1]
    if n_epoch_channels != n_template_channels:
        _log_if_present(
            logger, "warning",
            f"Channel mismatch: epoch data has {n_epoch_channels} channels, "
            f"templates have {n_template_channels}. Skipping temporal bin GFP plot."
        )
        return
    
    times = epochs.times
    nonpain_mask = (pain_values == 0).to_numpy()
    pain_mask = (pain_values == 1).to_numpy()
    
    if nonpain_mask.sum() < 1 or pain_mask.sum() < 1:
        _log_if_present(logger, "warning", "Insufficient trials for pain comparison")
        return
    
    info_eeg = mne.pick_info(epochs.info, picks)
    state_letters = _create_state_letters(n_states)
    colors = plt.cm.Set2(np.linspace(0, 1, n_states))
    temporal_bins = config.get("feature_engineering.features.temporal_bins", []) if config else []
    
    for bin_config in temporal_bins:
        bin_params = _parse_temporal_bin_config(bin_config, config)
        if bin_params is None:
            _log_if_present(logger, "warning", f"Invalid temporal bin configuration: {bin_config}; skipping")
            continue
        
        t_start, t_end, temporal_label = bin_params
        bin_mask = (times >= t_start) & (times <= t_end)
        if not bin_mask.any():
            _log_if_present(logger, "warning", f"No timepoints in {temporal_label} bin")
            continue
        
        times_bin = times[bin_mask]
        
        for condition_mask, condition_label in [(nonpain_mask, "nonpain"), (pain_mask, "pain")]:
            if condition_mask.sum() < 1:
                continue
            
            epoch_data_condition = epoch_data[condition_mask]
            gfp_all, labels_all = _compute_gfp_and_labels_for_bin(
                epoch_data_condition, templates, bin_mask
            )
            plot_cfg = get_plot_config(config)
            microstate_config = plot_cfg.plot_type_configs.get("microstate", {})
            gfp_scale_factor_value = microstate_config.get("gfp_scale_factor", 1e6)
            gfp_scale_factor = float(gfp_scale_factor_value)
            gfp_mean = np.mean(gfp_all, axis=0) * gfp_scale_factor
            labels_consensus = compute_consensus_labels(labels_all, len(times_bin))
            
            condition_title = "Non-pain" if condition_label == "nonpain" else "Pain"
            fig = _plot_gfp_sequence_for_bin(
                templates, info_eeg, times_bin, gfp_mean, labels_consensus,
                state_letters, colors, n_states, condition_title, temporal_label, plot_cfg
            )
            
            tight_rect = microstate_config.get("tight_rect_microstate", plot_cfg.get_layout_rect("tight_rect_microstate"))
            plt.tight_layout(rect=tight_rect)
            output_path = save_dir / f"sub-{subject}_microstate_gfp_sequence_{condition_label}_{temporal_label}"
            save_fig(fig, output_path, formats=plot_cfg.formats, dpi=plot_cfg.dpi,
                     bbox_inches=plot_cfg.bbox_inches, pad_inches=plot_cfg.pad_inches)
            plt.close(fig)
    
    _log_if_present(logger, "info", "Saved microstate GFP sequence plots by temporal bins")


###################################################################
# Microstate Transition Plotting
###################################################################


def plot_microstate_transition_network(
    transitions: Any,
    subject: str,
    save_dir: Path,
    logger: logging.Logger,
    config: Any,
    n_nonpain: Optional[int] = None,
    n_pain: Optional[int] = None,
) -> None:
    """Plot microstate transition network matrices.
    
    Args:
        transitions: MicrostateTransitionStats object
        subject: Subject identifier
        save_dir: Directory to save plot
        logger: Logger instance
        config: Configuration object
        n_nonpain: Optional number of non-pain trials
        n_pain: Optional number of pain trials
    """
    if transitions is None:
        _log_if_present(logger, "warning", "No microstate transition data provided; skipping plot")
        return

    state_labels = transitions.state_labels
    n_states = len(state_labels)
    if n_states == 0:
        _log_if_present(logger, "warning", "Empty transition matrices; skipping plot")
        return

    trans_nonpain = transitions.nonpain
    trans_pain = transitions.pain
    from eeg_pipeline.utils.config.loader import get_config_value
    from eeg_pipeline.plotting.features.utils import get_fdr_alpha

    epsilon_amp = float(get_config_value(config, "feature_engineering.constants.epsilon_amp", 1e-10))
    vmax = max(float(np.max(trans_nonpain)), float(np.max(trans_pain)), epsilon_amp)
    q_mat = getattr(transitions, "q_values", None)
    alpha = get_fdr_alpha(config)
    cmap = plt.cm.get_cmap("YlOrRd")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    condition_titles = [
        f"Non-pain (n={n_nonpain})" if n_nonpain else "Non-pain",
        f"Pain (n={n_pain})" if n_pain else "Pain"
    ]
    for ax, matrix, title in zip(axes, [trans_nonpain, trans_pain], condition_titles):
        im = ax.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=vmax, aspect="auto")
        ax.set_xticks(np.arange(n_states))
        ax.set_yticks(np.arange(n_states))
        ax.set_xticklabels(state_labels)
        ax.set_yticklabels(state_labels)
        ax.set_xlabel("To State", fontsize=plot_cfg.font.title)
        ax.set_ylabel("From State", fontsize=plot_cfg.font.title)
        ax.set_title(f"{title} Transitions", fontsize=plot_cfg.font.suptitle)
        for i in range(n_states):
            for j in range(n_states):
                value = matrix[i, j]
                if value <= 0:
                    continue
                text_color = "white" if value > 0.5 * vmax else "black"
                label = f"{value:.2f}"
                if q_mat is not None:
                    q_val = q_mat[i, j]
                    if np.isfinite(q_val):
                        if q_val < alpha:
                            label += f"\nq={q_val:.3f}"
                        else:
                            text_color = "gray"
                            label += "\nns"
                ax.text(j, i, label, ha="center", va="center", color=text_color, fontsize=plot_cfg.font.medium)
        plt.colorbar(im, ax=ax, label="Probability", shrink=0.8)

    plot_cfg = get_plot_config(config)
    suffix = " (q<alpha shown)" if q_mat is not None else ""
    plt.suptitle(f"Microstate Transition Probabilities by Condition{suffix}", fontsize=plot_cfg.font.figure_title)
    
    total_trials = (n_nonpain or 0) + (n_pain or 0)
    if total_trials > 0:
        footer_text = f"K={n_states} states | n={total_trials} total trials"
        fig.text(
            0.99, 0.01, footer_text,
            ha='right', va='bottom',
            fontsize=plot_cfg.font.small,
            color='gray', alpha=0.8
        )
    
    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    save_fig(
        fig,
        save_dir / f"sub-{subject}_microstate_transitions",
        formats=plot_cfg.formats, dpi=plot_cfg.dpi,
        bbox_inches=plot_cfg.bbox_inches, pad_inches=plot_cfg.pad_inches
    )
    plt.close(fig)
    _log_if_present(logger, "info", "Saved microstate transition network")


def plot_microstate_duration_distributions(
    duration_stats: List[Any],
    subject: str,
    save_dir: Path,
    logger: logging.Logger,
    config: Any,
    n_nonpain: Optional[int] = None,
    n_pain: Optional[int] = None,
) -> None:
    """Plot microstate duration distributions by pain condition.
    
    Args:
        duration_stats: List of MicrostateDurationStat objects
        subject: Subject identifier
        save_dir: Directory to save plot
        logger: Logger instance
        config: Configuration object
        n_nonpain: Optional number of non-pain trials
        n_pain: Optional number of pain trials
    """
    if not duration_stats:
        _log_if_present(logger, "warning", "No microstate duration statistics provided; skipping violin plot")
        return

    n_states = len(duration_stats)
    fig, axes = plt.subplots(n_states, 1, figsize=(10, 2.5 * n_states), sharex=True)
    if n_states == 1:
        axes = [axes]

    for ax, stat in zip(axes, duration_stats):
        nonpain_data = stat.nonpain
        pain_data = stat.pain
        if nonpain_data.size == 0 and pain_data.size == 0:
            ax.set_visible(False)
            continue

        data_to_plot: List[np.ndarray] = []
        labels: List[str] = []
        colors_to_use: List[str] = []

        if nonpain_data.size:
            data_to_plot.append(nonpain_data)
            n_label = f" (n={n_nonpain})" if n_nonpain else ""
            labels.append(f"Non-pain{n_label}")
            colors_to_use.append("steelblue")
        if pain_data.size:
            data_to_plot.append(pain_data)
            n_label = f" (n={n_pain})" if n_pain else ""
            labels.append(f"Pain{n_label}")
            colors_to_use.append("orangered")

        if data_to_plot:
            parts = ax.violinplot(data_to_plot, positions=range(len(data_to_plot)), showmeans=True, showmedians=True)
            for pc, color in zip(parts['bodies'], colors_to_use):
                pc.set_facecolor(color)
                pc.set_alpha(0.6)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_ylabel("Duration (s)", fontsize=plot_cfg.font.title)
        ax.set_title(f"State {stat.state}", fontsize=plot_cfg.font.suptitle)
        ax.grid(True, alpha=0.3, axis='y')

        q_val = getattr(stat, "q_value", np.nan)
        from eeg_pipeline.plotting.features.utils import get_fdr_alpha

        alpha = get_fdr_alpha(config)
        if nonpain_data.size and pain_data.size and np.isfinite(q_val):
            if q_val < alpha:
                y_min, y_max = ax.get_ylim()
                ax.plot([0, 1], [y_max * 0.95, y_max * 0.95], 'k-', linewidth=1)
                sig_text = "**" if q_val < (alpha / 5) else "*"
                ax.text(0.5, y_max * 0.97, f"{sig_text} (q={q_val:.3f}, alpha={alpha:.2f})", ha='center', fontsize=plot_cfg.font.title)

    if n_states > 0:
        axes[-1].set_xlabel("Condition", fontsize=plot_cfg.font.title)

    plot_cfg = get_plot_config(config)
    plt.suptitle("Microstate Duration Distributions by Pain Condition", fontsize=plot_cfg.font.figure_title)
    
    total_trials = (n_nonpain or 0) + (n_pain or 0)
    if total_trials > 0:
        footer_text = f"K={n_states} states | n={total_trials} total trials"
        fig.text(
            0.99, 0.01, footer_text,
            ha='right', va='bottom',
            fontsize=plot_cfg.font.small,
            color='gray', alpha=0.8
        )
    
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    save_fig(
        fig,
        save_dir / f"sub-{subject}_microstate_duration_distributions",
        formats=plot_cfg.formats, dpi=plot_cfg.dpi,
        bbox_inches=plot_cfg.bbox_inches, pad_inches=plot_cfg.pad_inches
    )
    plt.close(fig)
    _log_if_present(logger, "info", "Saved microstate duration distributions")



def plot_microstate_by_condition(
    microstate_df: pd.DataFrame,
    events_df: pd.DataFrame,
    subject: str,
    save_dir: Path,
    logger: logging.Logger,
    config: Any,
) -> None:
    """Compare microstate metrics between Pain and Non-pain using box+strip plots.
    
    Statistical improvements:
    - Shows both raw p-value and FDR-corrected q-value
    - Includes bootstrap 95% CI for mean difference
    - Reports Cohen's d effect size
    - Footer shows total tests and correction method
    """
    if microstate_df is None or microstate_df.empty or events_df is None:
        return

    pain_mask = extract_pain_mask(events_df, config)
    if pain_mask is None:
        return
    
    from eeg_pipeline.plotting.features.utils import (
        compute_condition_stats,
        apply_fdr_correction,
        format_stats_annotation,
        format_footer_annotation,
    )
        
    metrics = ['coverage', 'duration', 'occurrence']
    states = []
    
    for c in microstate_df.columns:
        if c.startswith('ms_coverage_State'):
            states.append(c.replace('ms_coverage_State', ''))
    states = sorted(list(set(states)))
    
    if not states:
        return
    
    condition_colors = {"pain": "#d62728", "nonpain": "#1f77b4"}
    plot_cfg = get_plot_config(config)
    
    for metric in metrics:
        all_stats = []
        all_pvals = []
        metric_data = []
        
        for state in states:
            col = f"ms_{metric}_State{state}"
            if col not in microstate_df.columns:
                continue
            vals = pd.to_numeric(microstate_df[col], errors='coerce')
            if len(vals) != len(pain_mask):
                continue
            vals_pain = vals[pain_mask].dropna()
            vals_nonpain = vals[~pain_mask].dropna()
            if len(vals_pain) >= 3 and len(vals_nonpain) >= 3:
                stats_result = compute_condition_stats(vals_nonpain.values, vals_pain.values, n_boot=1000, config=config)
                all_stats.append(stats_result)
                all_pvals.append(stats_result["p_raw"])
                metric_data.append({
                    'state': state,
                    'pain': vals_pain.values,
                    'nonpain': vals_nonpain.values,
                    'stats': stats_result,
                    'stats_idx': len(all_stats) - 1,
                })
        
        if not metric_data:
            continue
        
        if all_pvals:
            valid_pvals = [p for p in all_pvals if np.isfinite(p)]
            if valid_pvals:
                rejected, qvals, _ = apply_fdr_correction(valid_pvals, config=config)
                q_idx = 0
                for i, p in enumerate(all_pvals):
                    if np.isfinite(p):
                        all_stats[i]["q_fdr"] = qvals[q_idx]
                        all_stats[i]["fdr_significant"] = rejected[q_idx]
                        q_idx += 1
                    else:
                        all_stats[i]["q_fdr"] = np.nan
                        all_stats[i]["fdr_significant"] = False
                n_significant = int(np.sum(rejected))
            else:
                n_significant = 0
        else:
            n_significant = 0
            
        n_states_with_data = len(metric_data)
        fig, axes = plt.subplots(1, n_states_with_data, figsize=(4 * n_states_with_data, 5), sharey=True)
        if n_states_with_data == 1:
            axes = [axes]
        
        for idx, data in enumerate(metric_data):
            ax = axes[idx]
            vals_nonpain = data['nonpain']
            vals_pain = data['pain']
            
            bp = ax.boxplot([vals_nonpain, vals_pain], 
                           positions=[0, 1], widths=0.4, patch_artist=True)
            bp["boxes"][0].set_facecolor(condition_colors["nonpain"])
            bp["boxes"][0].set_alpha(0.6)
            bp["boxes"][1].set_facecolor(condition_colors["pain"])
            bp["boxes"][1].set_alpha(0.6)
            
            ax.scatter(np.random.uniform(-0.1, 0.1, len(vals_nonpain)), 
                      vals_nonpain, c=condition_colors["nonpain"], alpha=0.3, s=8)
            ax.scatter(1 + np.random.uniform(-0.1, 0.1, len(vals_pain)), 
                      vals_pain, c=condition_colors["pain"], alpha=0.3, s=8)
            
            if data.get("stats") is not None:
                s = data["stats"]
                annotation = format_stats_annotation(
                    p_raw=s["p_raw"],
                    q_fdr=s.get("q_fdr"),
                    cohens_d=s["cohens_d"],
                    ci_low=s["ci_low"],
                    ci_high=s["ci_high"],
                    compact=True,
                )
                text_color = "#d62728" if s.get("fdr_significant", False) else "#333333"
                ax.text(0.5, 0.98, annotation, transform=ax.transAxes, 
                       ha="center", fontsize=7, va="top", color=text_color,
                       bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
            
            ax.set_xticks([0, 1])
            ax.set_xticklabels(["NP", "P"], fontsize=plot_cfg.font.large)
            ax.set_title(f"State {data['state']}", fontweight="bold")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            
            if idx == 0:
                ax.set_ylabel(metric.capitalize())
        
        n_pain = int(pain_mask.sum())
        n_nonpain = int((~pain_mask).sum())
        fig.suptitle(f"Microstate {metric.capitalize()} by Condition (sub-{subject})\nN: {n_nonpain} NP, {n_pain} P", 
                    fontsize=plot_cfg.font.figure_title, fontweight="bold", y=1.02)
        
        n_tests = len([p for p in all_pvals if np.isfinite(p)])
        footer = format_footer_annotation(
            n_tests=n_tests,
            correction_method="FDR-BH",
            alpha=0.05,
            n_significant=n_significant,
            additional_info="Mann-Whitney U | Bootstrap 95% CI | †=FDR significant"
        )
        fig.text(0.5, 0.01, footer, ha="center", va="bottom", fontsize=8, color="gray")
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.98])
        save_fig(fig, save_dir / f"sub-{subject}_microstate_{metric}_by_condition",
                 formats=plot_cfg.formats, dpi=plot_cfg.dpi,
                 bbox_inches=plot_cfg.bbox_inches, pad_inches=plot_cfg.pad_inches)
        plt.close(fig)
        
    if logger:
        logger.info("Saved microstate condition comparisons")


def plot_microstate_transition_matrix(
    microstate_df: pd.DataFrame,
    subject: str,
    save_dir: Path,
    logger: logging.Logger,
    config: Any,
) -> None:
    """Plot microstate transition probability matrix as heatmap.
    
    Shows A→B, A→C, B→A, B→C etc. transition probabilities.
    Handles column patterns: trans_{i}_to_{j} or transition_{i}_{j}
    """
    if microstate_df is None or microstate_df.empty:
        return
    
    # Look for both 'transition' and 'trans_' patterns
    trans_cols = [c for c in microstate_df.columns 
                  if 'transition' in c.lower() or 'trans_' in c.lower()]
    if not trans_cols:
        if logger:
            logger.info("No transition columns found for matrix")
        return
    
    # Extract state indices from column names
    states = set()
    for c in trans_cols:
        # Pattern: trans_{i}_to_{j} or microstates_..._trans_{i}_to_{j}
        if 'trans_' in c.lower() and '_to_' in c.lower():
            try:
                parts = c.lower().split('trans_')[1].split('_to_')
                if len(parts) == 2:
                    from_state = parts[0]
                    to_state = parts[1].split('_')[0]  # Handle trailing parts
                    states.add(from_state)
                    states.add(to_state)
            except (IndexError, ValueError):
                pass
        else:
            # Legacy patterns
            parts = c.split('_')
            for p in parts:
                if len(p) == 1 and p.isalpha() and p.isupper():
                    states.add(p)
                elif p.startswith('State') or p.startswith('state'):
                    states.add(p.replace('State', '').replace('state', ''))
    
    states = sorted(list(states))
    n_states = len(states)
    
    if n_states < 2:
        if logger:
            logger.info("Not enough states for transition matrix")
        return
    
    trans_matrix = np.zeros((n_states, n_states))
    
    for i, from_state in enumerate(states):
        for j, to_state in enumerate(states):
            # Match patterns
            patterns = [
                f"trans_{from_state}_to_{to_state}",
                f"transition_{from_state}_{to_state}",
                f"ms_transition_{from_state}_{to_state}",
                f"transition_State{from_state}_State{to_state}",
                f"microstates_transition_{from_state}_{to_state}",
            ]
            for pat in patterns:
                matching = [c for c in trans_cols if pat.lower() in c.lower()]
                if matching:
                    val = microstate_df[matching[0]].mean()
                    if not np.isnan(val):
                        trans_matrix[i, j] = val
                    break
    
    if trans_matrix.sum() == 0:
        if logger:
            logger.info("Empty transition matrix")
        return
    
    plot_cfg = get_plot_config(config)
    fig, ax = plt.subplots(figsize=(6, 5))
    
    state_labels = [f"State {s}" for s in states]
    
    im = ax.imshow(trans_matrix, cmap='Blues', vmin=0)
    
    ax.set_xticks(range(n_states))
    ax.set_yticks(range(n_states))
    ax.set_xticklabels(state_labels, fontsize=plot_cfg.font.large)
    ax.set_yticklabels(state_labels, fontsize=plot_cfg.font.large)
    ax.set_xlabel("To State", fontsize=plot_cfg.font.title)
    ax.set_ylabel("From State", fontsize=plot_cfg.font.title)
    
    for i in range(n_states):
        for j in range(n_states):
            val = trans_matrix[i, j]
            if val > 0:
                color = 'white' if val > trans_matrix.max() * 0.6 else 'black'
                ax.text(j, i, f"{val:.2f}", ha='center', va='center', 
                       fontsize=plot_cfg.font.medium, color=color)
    
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Transition Probability", fontsize=plot_cfg.font.large)
    
    ax.set_title(f"Microstate Transition Matrix (sub-{subject})", 
                fontsize=plot_cfg.font.suptitle, fontweight="bold")
    
    plt.tight_layout()
    save_fig(fig, save_dir / f"sub-{subject}_microstate_transition_matrix",
             formats=plot_cfg.formats, dpi=plot_cfg.dpi,
             bbox_inches=plot_cfg.bbox_inches, pad_inches=plot_cfg.pad_inches)
    plt.close(fig)
    
    if logger:
        logger.info("Saved microstate transition matrix")



