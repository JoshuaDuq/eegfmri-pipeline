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

from ...utils.io.general import (
    save_fig,
    ensure_dir,
    find_temperature_column_in_events,
    extract_eeg_picks,
    log_if_present as _log_if_present,
    validate_picks as _validate_picks,
)
from ...utils.data.loading import (
    resolve_columns,
    get_aligned_events,
    align_events_with_policy,
    validate_aligned_events_length,
)
from ...utils.analysis.stats import (
    compute_coverage_statistics,
    compute_consensus_labels,
    format_correlation_text,
)
from ...analysis.features.microstates import (
    compute_gfp as _compute_gfp,
    label_timecourse as _label_timecourse,
    extract_templates_from_trials as _extract_templates_from_trials,
    MicrostateDurationStat,
    MicrostateTransitionStats,
)
from ..config import get_plot_config


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
        templates: Microstate templates array
        time_mask: Boolean mask for time points
    
    Returns:
        Tuple of (list of GFP arrays, list of label arrays)
    """
    gfp_all_trials = []
    labels_all_trials = []
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
    n_states: int
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
            f"State {state_letters[state_idx]}", fontsize=11, weight='bold', color=colors[state_idx]
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
    
    aligned_events = align_events_with_policy(events_df, epochs, config=config, logger=logger)
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
    
    _plot_topomap_row(fig, gs, templates, info_eeg, state_letters, colors, n_states)
    
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
    
    _plot_topomap_row(fig, gs, templates, info_eeg, state_letters, colors, n_states)
    
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
        X[nonpain_mask], sfreq, n_states, config
    )
    templates_pain = _extract_templates_from_trials(
        X[pain_mask], sfreq, n_states, config
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
        axes[0, i].set_title(f"State {state_letters[i]}", fontsize=10)
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
        templates = _extract_templates_from_trials(X[temp_mask], sfreq, n_states, config)
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
                    f"State {state_letters[col_idx]}", fontsize=10
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


def plot_microstate_pain_correlation_heatmap(
    corr_df: pd.DataFrame,
    pval_df: Optional[pd.DataFrame],
    subject: str,
    save_dir: Path,
    logger: logging.Logger,
    config: Any
) -> None:
    """Plot microstate-pain correlation heatmap.
    
    Args:
        corr_df: Correlation DataFrame (metrics x states)
        pval_df: Optional p-value DataFrame
        subject: Subject identifier
        save_dir: Directory to save plot
        logger: Logger instance
        config: Configuration object
    """
    if corr_df is None or corr_df.empty:
        _log_if_present(logger, "warning", "No microstate correlation data provided; skipping heatmap")
        return

    plot_cfg = get_plot_config(config)
    features_config = plot_cfg.plot_type_configs.get("features", {})
    correlation_config = features_config.get("correlation", {})

    metric_labels = list(corr_df.index)
    state_labels = list(corr_df.columns)
    n_states = len(state_labels)
    corr_matrix = corr_df.to_numpy(dtype=float)
    p_matrix = _prepare_pvalue_matrix(pval_df, metric_labels, state_labels, corr_matrix.shape)

    width_value = plot_cfg.figure_sizes.get("microstate_width_per_column", 1.2)
    width_per_col = float(width_value[0] if isinstance(width_value, tuple) else width_value)
    height_value = plot_cfg.figure_sizes.get("microstate_height_per_row", 1.0)
    height_per_row = float(height_value[0] if isinstance(height_value, tuple) else height_value)

    fig, ax = plt.subplots(
        figsize=(
            max(6, n_states * width_per_col),
            max(5, len(metric_labels) * height_per_row)
        )
    )
    
    vmin = correlation_config.get("vmin", -0.6)
    vmax = correlation_config.get("vmax", 0.6)
    threshold_text = correlation_config.get("threshold_text", 0.4)
    
    im = ax.imshow(
        corr_matrix, cmap="RdBu_r",
        vmin=vmin, vmax=vmax, aspect="auto"
    )
    ax.set_xticks(np.arange(n_states))
    ax.set_yticks(np.arange(len(metric_labels)))
    ax.set_xticklabels(state_labels)
    ax.set_yticklabels(metric_labels)
    ax.set_xlabel("Microstate", fontsize=plot_cfg.font.ylabel)
    ax.set_ylabel("Metric", fontsize=plot_cfg.font.ylabel)
    ax.set_title("Microstate-Pain Rating Correlations (Spearman r)", fontsize=plot_cfg.font.figure_title)

    for i, metric in enumerate(metric_labels):
        for j, state in enumerate(state_labels):
            value = corr_matrix[i, j]
            if not np.isfinite(value):
                continue
            
            text_color = "white" if abs(value) > threshold_text else "black"
            text = format_correlation_text(value, p_matrix[i, j])
            ax.text(j, i, text, ha="center", va="center", color=text_color, fontsize=plot_cfg.font.medium)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Spearman r", fontsize=plot_cfg.font.title)
    plt.tight_layout()
    save_fig(
        fig,
        save_dir / f"sub-{subject}_microstate_pain_correlation",
        formats=plot_cfg.formats, dpi=plot_cfg.dpi,
        bbox_inches=plot_cfg.bbox_inches, pad_inches=plot_cfg.pad_inches
    )
    plt.close(fig)
    _log_if_present(logger, "info", "Saved microstate-pain correlation heatmap")


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
    transitions: MicrostateTransitionStats,
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
    vmax = max(float(np.max(trans_nonpain)), float(np.max(trans_pain)), 1e-6)
    q_mat = getattr(transitions, "q_values", None)
    alpha = float(config.get("behavior_analysis.statistics.fdr_alpha", 0.05)) if hasattr(config, "get") else 0.05
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
        ax.set_xlabel("To State", fontsize=10)
        ax.set_ylabel("From State", fontsize=10)
        ax.set_title(f"{title} Transitions", fontsize=11)
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
                ax.text(j, i, label, ha="center", va="center", color=text_color, fontsize=8)
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
    duration_stats: List[MicrostateDurationStat],
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
        ax.set_ylabel("Duration (s)", fontsize=10)
        ax.set_title(f"State {stat.state}", fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')

        q_val = getattr(stat, "q_value", np.nan)
        alpha = float(config.get("behavior_analysis.statistics.fdr_alpha", 0.05)) if hasattr(config, "get") else 0.05
        if nonpain_data.size and pain_data.size and np.isfinite(q_val):
            if q_val < alpha:
                y_min, y_max = ax.get_ylim()
                ax.plot([0, 1], [y_max * 0.95, y_max * 0.95], 'k-', linewidth=1)
                sig_text = "**" if q_val < (alpha / 5) else "*"
                ax.text(0.5, y_max * 0.97, f"{sig_text} (q={q_val:.3f}, alpha={alpha:.2f})", ha='center', fontsize=10)

    if n_states > 0:
        axes[-1].set_xlabel("Condition", fontsize=10)

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


###################################################################
# Group Microstate Plotting
###################################################################


def plot_group_microstate_template_stability(
    group_template_path: Path,
    save_dir: Path,
    logger: logging.Logger,
    config: Any,
) -> None:
    """Plot group microstate template stability correlation matrix.
    
    Args:
        group_template_path: Path to group template file
        save_dir: Directory to save plot
        logger: Logger instance
        config: Configuration object
    """
    if not group_template_path.exists():
        _log_if_present(logger, "warning", f"Group microstate template file not found: {group_template_path}")
        return
    try:
        data = np.load(group_template_path, allow_pickle=True)
        templates = data.get("templates")
    except Exception as exc:
        _log_if_present(logger, "warning", f"Failed to load group templates: {exc}")
        return
    if templates is None or templates.size == 0:
        _log_if_present(logger, "warning", "Group templates empty; skipping stability plot")
        return

    corr_mat = np.corrcoef(templates)
    plot_cfg = get_plot_config(config)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(corr_mat, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_title("Group microstate template correlations")
    ax.set_xlabel("Template")
    ax.set_ylabel("Template")
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Correlation")
    output_path = save_dir / "group_microstate_template_correlation"
    save_fig(fig, output_path, formats=plot_cfg.formats, dpi=plot_cfg.dpi,
             bbox_inches=plot_cfg.bbox_inches, pad_inches=plot_cfg.pad_inches)
    plt.close(fig)
    _log_if_present(logger, "info", "Saved group microstate template correlation heatmap")


def plot_group_microstate_transition_summary(
    transitions: MicrostateTransitionStats,
    save_dir: Path,
    logger: logging.Logger,
    config: Any,
) -> None:
    """Plot group microstate transition summary with difference plot.
    
    Args:
        transitions: MicrostateTransitionStats object
        save_dir: Directory to save plot
        logger: Logger instance
        config: Configuration object
    """
    if transitions is None:
        _log_if_present(logger, "warning", "No group microstate transitions to plot")
        return

    state_labels = transitions.state_labels
    n_states = len(state_labels)
    if n_states == 0:
        _log_if_present(logger, "warning", "No states found for microstate transitions")
        return

    ensure_dir(save_dir)
    plot_cfg = get_plot_config(config)
    alpha = float(config.get("behavior_analysis.statistics.fdr_alpha", 0.05)) if hasattr(config, "get") else 0.05
    q_mat = getattr(transitions, "q_values", None)
    sig_mask = (np.isfinite(q_mat) & (q_mat < alpha)) if q_mat is not None else None

    diff = transitions.pain - transitions.nonpain
    vmax = np.nanmax([transitions.nonpain, transitions.pain, 1e-6])
    diff_lim = np.nanmax(np.abs(diff))
    diff_lim = diff_lim if np.isfinite(diff_lim) and diff_lim > 0 else 1.0

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    matrices = [
        (transitions.nonpain, "Non-pain", "YlOrRd", (0, vmax)),
        (transitions.pain, "Pain", "YlOrRd", (0, vmax)),
        (diff, "Pain - Non-pain", "RdBu_r", (-diff_lim, diff_lim)),
    ]

    ims = []
    for ax, (mat, title, cmap, vlim) in zip(axes, matrices):
        im = ax.imshow(mat, cmap=cmap, vmin=vlim[0], vmax=vlim[1], aspect="auto")
        ax.set_xticks(np.arange(n_states))
        ax.set_yticks(np.arange(n_states))
        ax.set_xticklabels(state_labels)
        ax.set_yticklabels(state_labels)
        ax.set_xlabel("To State")
        ax.set_ylabel("From State")
        ax.set_title(title)
        if sig_mask is not None and title != "Pain - Non-pain":
            ax.contour(sig_mask, levels=[0.5], colors="k", linewidths=0.8)
        ims.append(im)

    cbar = fig.colorbar(ims[0], ax=axes[:2].tolist(), shrink=0.8, pad=0.02)
    cbar.set_label("Transition rate (per s)")
    cbar_diff = fig.colorbar(ims[2], ax=[axes[2]], shrink=0.8, pad=0.02)
    cbar_diff.set_label("Δ transition rate (per s)")

    fig.suptitle("Group microstate transition summary", fontsize=plot_cfg.font.figure_title)
    output_path = save_dir / "group_microstate_transitions"
    save_fig(
        fig,
        output_path,
        formats=plot_cfg.formats,
        dpi=plot_cfg.dpi,
        bbox_inches=plot_cfg.bbox_inches,
        pad_inches=plot_cfg.pad_inches,
    )
    plt.close(fig)
    _log_if_present(logger, "info", f"Saved group microstate transitions to {output_path}")

