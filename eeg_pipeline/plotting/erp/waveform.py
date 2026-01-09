"""ERP waveform visualizations.

This module provides functions for plotting event-related potentials (ERPs),
including butterfly plots, ROI-based waveforms, and condition contrasts.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

from eeg_pipeline.plotting.config import get_plot_config
from eeg_pipeline.plotting.io.figures import save_fig
from eeg_pipeline.plotting.style import use_style, get_color
from eeg_pipeline.utils.analysis.channels import build_roi_map, pick_eeg_channels
from eeg_pipeline.utils.analysis.spatial import get_roi_definitions
from eeg_pipeline.utils.config.loader import get_config_value
from eeg_pipeline.utils.analysis.events import resolve_comparison_spec


_MICROVOLTS_PER_VOLT = 1e6
_MILLISECONDS_PER_SECOND = 1000
_DEFAULT_BASELINE_WINDOW = [-0.2, 0.0]


def _build_metadata_query(column: str, value: Any) -> str:
    """Build a metadata query string for a column and value.
    
    Parameters
    ----------
    column : str
        Column name in metadata.
    value : Any
        Value to query for.
    
    Returns
    -------
    str
        Query string in pandas/MNE syntax.
    """
    column_expression = f"`{column}`"
    try:
        numeric_value = pd.to_numeric(str(value), errors="coerce")
        if not np.isnan(numeric_value):
            if float(numeric_value).is_integer():
                return f"{column_expression} == {int(numeric_value)}"
            return f"{column_expression} == {float(numeric_value)}"
    except (ValueError, TypeError):
        pass
    return f"{column_expression} == {repr(str(value))}"


def _filter_roi_map_by_config(roi_map: Dict[str, List[int]], config: Any) -> Dict[str, List[int]]:
    """Filter ROI map based on comparison_rois configuration.
    
    Parameters
    ----------
    roi_map : dict
        Mapping of ROI names to channel indices.
    config : Any
        Pipeline configuration.
    
    Returns
    -------
    dict
        Filtered ROI map containing only requested ROIs.
    """
    comparison_rois = get_config_value(
        config, "plotting.comparisons.comparison_rois", []
    )
    has_specific_rois = any(roi.lower() != "all" for roi in comparison_rois)
    
    if not comparison_rois or not has_specific_rois:
        return roi_map
    
    filtered_map = {}
    for roi_name in comparison_rois:
        if roi_name.lower() == "all":
            continue
        if roi_name in roi_map:
            filtered_map[roi_name] = roi_map[roi_name]
    
    return filtered_map


def _compute_roi_waveform_statistics(
    data: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute mean waveform and SEM for ROI data.
    
    Parameters
    ----------
    data : np.ndarray
        Shape (n_trials, n_channels, n_times) or (n_trials, n_times).
        If 3D, averages across channels first.
    
    Returns
    -------
    mean_waveform : np.ndarray
        Mean waveform in microvolts, shape (n_times,).
    sem_waveform : np.ndarray
        Standard error of the mean in microvolts, shape (n_times,).
    """
    if data.ndim == 3:
        channel_averaged = np.mean(data, axis=1)
    else:
        channel_averaged = data
    
    n_trials = len(channel_averaged)
    mean_waveform = np.mean(channel_averaged, axis=0) * _MICROVOLTS_PER_VOLT
    sem_waveform = (
        np.std(channel_averaged, axis=0) / np.sqrt(n_trials)
    ) * _MICROVOLTS_PER_VOLT
    
    return mean_waveform, sem_waveform


def _get_baseline_window(config: Any) -> Tuple[float, float]:
    """Extract baseline window from configuration.
    
    Parameters
    ----------
    config : Any
        Pipeline configuration.
    
    Returns
    -------
    tuple
        (start_time, end_time) in seconds.
    """
    erp_config = config.get("feature_engineering.erp", {})
    return tuple(erp_config.get("baseline_window", _DEFAULT_BASELINE_WINDOW))


def _plot_roi_waveform_with_error(
    ax: plt.Axes,
    time_vector: np.ndarray,
    mean_waveform: np.ndarray,
    sem_waveform: np.ndarray,
    label: str,
    color: Optional[str] = None,
) -> None:
    """Plot ROI waveform with shaded error region.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to plot on.
    time_vector : np.ndarray
        Time points in milliseconds.
    mean_waveform : np.ndarray
        Mean waveform in microvolts.
    sem_waveform : np.ndarray
        SEM in microvolts.
    label : str
        Label for the waveform.
    color : str, optional
        Color for the plot. If None, uses default color scheme.
    """
    ax.plot(time_vector, mean_waveform, label=label, color=color, linewidth=2)
    ax.fill_between(
        time_vector,
        mean_waveform - sem_waveform,
        mean_waveform + sem_waveform,
        alpha=0.2,
        color=color,
    )


def plot_butterfly_erp(
    epochs: mne.Epochs,
    subject: str,
    save_dir: Path,
    config: Any,
    logger: logging.Logger,
    conditions: Optional[Dict[str, str]] = None,
) -> List[Path]:
    """Plot butterfly ERP for all trials or specific conditions.
    
    Parameters
    ----------
    epochs : mne.Epochs
        The epochs data.
    subject : str
        Subject ID for labeling and filenames.
    save_dir : Path
        Directory to save plots.
    config : Any
        Pipeline configuration.
    logger : logging.Logger
        Logger instance.
    conditions : dict, optional
        Conditions to plot separately (name -> query).
        If None, plots all trials.
    """
    saved_paths = []
    save_dir.mkdir(parents=True, exist_ok=True)
    plot_cfg = get_plot_config(config)
    primary_ext = plot_cfg.formats[0] if plot_cfg.formats else "png"
    
    with use_style(context="paper"):
        # 1. Overall butterfly (all trials)
        evoked = epochs.average()
        fig = evoked.plot(
            picks="eeg",
            spatial_colors=True,
            gfp=True,
            show=False,
            window_title=f"sub-{subject} Butterfly ERP (All Trials)"
        )
        fig.suptitle(f"sub-{subject}: Butterfly ERP (All Trials)", fontsize=14)
        
        path = save_dir / f"sub-{subject}_erp_butterfly_all.{primary_ext}"
        save_fig(
            fig,
            path,
            logger=logger,
            formats=plot_cfg.formats,
            dpi=plot_cfg.savefig_dpi,
            bbox_inches=plot_cfg.bbox_inches,
            pad_inches=plot_cfg.pad_inches,
        )
        saved_paths.append(path)
        
        # 2. Condition-specific butterflies
        if conditions:
            for cond_name, query in conditions.items():
                try:
                    cond_epochs = epochs[query]
                    if len(cond_epochs) == 0:
                        continue
                        
                    evoked_cond = cond_epochs.average()
                    fig = evoked_cond.plot(
                        picks="eeg",
                        spatial_colors=True,
                        gfp=True,
                        show=False,
                        window_title=f"sub-{subject} Butterfly ERP ({cond_name})"
                    )
                    fig.suptitle(f"sub-{subject}: Butterfly ERP ({cond_name})", fontsize=14)
                    
                    path = save_dir / f"sub-{subject}_erp_butterfly_{cond_name}.{primary_ext}"
                    save_fig(
                        fig,
                        path,
                        logger=logger,
                        formats=plot_cfg.formats,
                        dpi=plot_cfg.savefig_dpi,
                        bbox_inches=plot_cfg.bbox_inches,
                        pad_inches=plot_cfg.pad_inches,
                    )
                    saved_paths.append(path)
                except (KeyError, ValueError, IndexError) as e:
                    logger.warning(f"Failed to plot butterfly for condition {cond_name}: {e}")

    return saved_paths


def plot_roi_erp(
    epochs: mne.Epochs,
    subject: str,
    save_dir: Path,
    config: Any,
    logger: logging.Logger,
    conditions: Optional[Dict[str, str]] = None,
) -> List[Path]:
    """Plot ROI-based ERP waveforms with shaded error bars.
    
    Parameters
    ----------
    epochs : mne.Epochs
        The epochs data.
    subject : str
        Subject ID for labeling and filenames.
    save_dir : Path
        Directory to save plots.
    config : Any
        Pipeline configuration.
    logger : logging.Logger
        Logger instance.
    conditions : dict, optional
        Mapping of display name -> metadata query string (pandas/MNE syntax).
        If None, plots all trials.
    """
    saved_paths = []
    save_dir.mkdir(parents=True, exist_ok=True)
    plot_cfg = get_plot_config(config)
    primary_ext = plot_cfg.formats[0] if plot_cfg.formats else "png"
    
    roi_defs = get_roi_definitions(config)
    if not roi_defs:
        logger.warning("No ROI definitions found in config; skipping ROI ERP plot.")
        return saved_paths

    picks, channel_names = pick_eeg_channels(epochs)
    roi_map = build_roi_map(channel_names, roi_defs)
    
    if not roi_map:
        logger.warning("No ROIs matched available channels.")
        return saved_paths

    time_vector_ms = epochs.times * _MILLISECONDS_PER_SECOND
    roi_map = _filter_roi_map_by_config(roi_map, config)
    baseline_window = _get_baseline_window(config)

    with use_style(context="paper"):
        for roi_name, channel_indices in roi_map.items():
            if not channel_indices:
                continue
                
            fig, ax = plt.subplots(figsize=(8, 5))
            
            if conditions:
                for condition_name, query in conditions.items():
                    try:
                        condition_data = epochs[query].get_data(picks=channel_indices)
                        if condition_data.size == 0:
                            continue
                            
                        mean_waveform, sem_waveform = _compute_roi_waveform_statistics(
                            condition_data
                        )
                        color = get_color(condition_name, default=None)
                        _plot_roi_waveform_with_error(
                            ax,
                            time_vector_ms,
                            mean_waveform,
                            sem_waveform,
                            condition_name,
                            color,
                        )
                    except (KeyError, ValueError, IndexError) as e:
                        logger.warning(
                            f"Failed to plot ROI {roi_name} for condition {condition_name}: {e}"
                        )
            else:
                all_trials_data = epochs.get_data(picks=channel_indices)
                mean_waveform, sem_waveform = _compute_roi_waveform_statistics(
                    all_trials_data
                )
                _plot_roi_waveform_with_error(
                    ax,
                    time_vector_ms,
                    mean_waveform,
                    sem_waveform,
                    "All Trials",
                    "#333333",
                )

            ax.axvline(0, color="black", linestyle="--", alpha=0.5)
            ax.axhline(0, color="black", linestyle="-", alpha=0.3)
            ax.set_xlabel("Time (ms)")
            ax.set_ylabel("Amplitude (μV)")
            ax.set_title(f"sub-{subject}: {roi_name} ERP")
            ax.legend()
            
            baseline_start_ms = baseline_window[0] * _MILLISECONDS_PER_SECOND
            baseline_end_ms = baseline_window[1] * _MILLISECONDS_PER_SECOND
            ax.axvspan(
                baseline_start_ms,
                baseline_end_ms,
                color="gray",
                alpha=0.1,
                label="Baseline",
            )
            
            path = save_dir / f"sub-{subject}_erp_roi_{roi_name.lower()}.{primary_ext}"
            save_fig(
                fig,
                path,
                logger=logger,
                formats=plot_cfg.formats,
                dpi=plot_cfg.savefig_dpi,
                bbox_inches=plot_cfg.bbox_inches,
                pad_inches=plot_cfg.pad_inches,
            )
            saved_paths.append(path)
            
    return saved_paths


def plot_erp_contrast(
    epochs: mne.Epochs,
    subject: str,
    save_dir: Path,
    config: Any,
    logger: logging.Logger,
    cond_a: Optional[str] = None,
    cond_b: Optional[str] = None,
    label_a: Optional[str] = None,
    label_b: Optional[str] = None,
) -> List[Path]:
    """Plot ERP contrast (A - B) for all channels.
    
    Parameters
    ----------
    epochs : mne.Epochs
        The epochs data.
    subject : str
        Subject ID for labeling and filenames.
    save_dir : Path
        Directory to save plots.
    config : Any
        Pipeline configuration.
    logger : logging.Logger
        Logger instance.
    cond_a : str, optional
        Metadata query string for condition A (minuend in A - B).
        If None, resolved from config.
    cond_b : str, optional
        Metadata query string for condition B (subtrahend in A - B).
        If None, resolved from config.
    label_a : str, optional
        Display label for condition A. If None, resolved from config.
    label_b : str, optional
        Display label for condition B. If None, resolved from config.
    """
    saved_paths = []
    save_dir.mkdir(parents=True, exist_ok=True)
    plot_cfg = get_plot_config(config)
    primary_ext = plot_cfg.formats[0] if plot_cfg.formats else "png"

    if cond_a is None or cond_b is None or label_a is None or label_b is None:
        metadata = getattr(epochs, "metadata", None)
        comparison_spec = (
            resolve_comparison_spec(metadata, config, require_enabled=False)
            if metadata is not None
            else None
        )
        if comparison_spec is None:
            logger.warning(
                "No configured comparison found in metadata; skipping ERP contrast."
            )
            return saved_paths
        
        column, value_b, value_a, auto_label_b, auto_label_a = comparison_spec
        cond_a = _build_metadata_query(column, value_a)
        cond_b = _build_metadata_query(column, value_b)
        label_a = auto_label_a
        label_b = auto_label_b
    
    try:
        evoked_a = epochs[cond_a].average()
        evoked_b = epochs[cond_b].average()
        
        contrast_evoked = mne.combine_evoked([evoked_a, evoked_b], weights=[1, -1])
        
        with use_style(context="paper"):
            fig = contrast_evoked.plot(
                picks="eeg",
                spatial_colors=True,
                gfp=True,
                show=False,
                window_title=f"sub-{subject} ERP Contrast ({label_a} - {label_b})"
            )
            fig.suptitle(
                f"sub-{subject}: ERP Contrast ({label_a} - {label_b})",
                fontsize=14
            )
            
            path = save_dir / (
                f"sub-{subject}_erp_contrast_"
                f"{label_a.lower()}_{label_b.lower()}.{primary_ext}"
            )
            save_fig(
                fig,
                path,
                logger=logger,
                formats=plot_cfg.formats,
                dpi=plot_cfg.savefig_dpi,
                bbox_inches=plot_cfg.bbox_inches,
                pad_inches=plot_cfg.pad_inches,
            )
            saved_paths.append(path)
            
    except (KeyError, ValueError, IndexError) as e:
        logger.warning(f"Failed to plot ERP contrast: {e}")
        
    return saved_paths
