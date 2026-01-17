"""ERP topomap visualizations.

This module provides functions for plotting the spatial distribution of ERP
amplitudes at specific time windows.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import mne

from eeg_pipeline.plotting.config import get_plot_config
from eeg_pipeline.plotting.io.figures import save_fig
from eeg_pipeline.plotting.style import use_style


_TIME_WINDOW_SECONDS = 0.05
_TITLE_FONTSIZE = 8
_MIN_CONDITIONS_FOR_CONTRAST = 2
_DEFAULT_COMPONENTS = [
    {"name": "N2", "start": 0.15, "end": 0.25},
    {"name": "P2", "start": 0.3, "end": 0.45},
]


def _extract_component_times(components: List[Dict[str, Any]]) -> List[float]:
    """Extract midpoint times from component definitions.
    
    Args:
        components: List of component dictionaries with 'start' and 'end' keys.
    
    Returns:
        List of midpoint times in seconds.
    """
    return [(component["start"] + component["end"]) / 2.0 for component in components]


def _get_primary_extension(plot_config: Any) -> str:
    """Get primary file extension from plot configuration.
    
    Args:
        plot_config: Plot configuration object.
    
    Returns:
        Primary file extension (default: "png").
    """
    return plot_config.formats[0] if plot_config.formats else "png"


def _set_title_fontsize(figure: plt.Figure, fontsize: int = _TITLE_FONTSIZE) -> None:
    """Set fontsize for all axes titles in figure.
    
    Args:
        figure: Matplotlib figure object.
        fontsize: Font size to apply to titles.
    """
    for axis in figure.axes:
        if axis.get_title():
            axis.set_title(axis.get_title(), fontsize=fontsize)


def _create_and_save_topomap(
    evoked: mne.Evoked,
    times: List[float],
    save_path: Path,
    plot_config: Any,
    logger: logging.Logger,
) -> Path:
    """Create topomap figure and save to disk.
    
    Args:
        evoked: MNE Evoked object to plot.
        times: List of time points to plot.
        save_path: Path where figure should be saved.
        plot_config: Plot configuration object.
        logger: Logger instance.
    
    Returns:
        Path where figure was saved.
    """
    figure = evoked.plot_topomap(
        times=times,
        average=_TIME_WINDOW_SECONDS,
        show=False,
    )
    _set_title_fontsize(figure)
    
    save_fig(
        figure,
        save_path,
        logger=logger,
        formats=plot_config.formats,
        dpi=plot_config.savefig_dpi,
        bbox_inches=plot_config.bbox_inches,
        pad_inches=plot_config.pad_inches,
    )
    return save_path


def _plot_overall_topomaps(
    epochs: mne.Epochs,
    subject: str,
    times: List[float],
    save_dir: Path,
    plot_config: Any,
    logger: logging.Logger,
) -> Path:
    """Plot topomaps for all trials combined.
    
    Args:
        epochs: MNE Epochs object.
        subject: Subject identifier.
        times: List of time points to plot.
        save_dir: Directory to save figure.
        plot_config: Plot configuration object.
        logger: Logger instance.
    
    Returns:
        Path where figure was saved.
    """
    evoked = epochs.average()
    primary_ext = _get_primary_extension(plot_config)
    save_path = save_dir / f"sub-{subject}_erp_topomaps_all.{primary_ext}"
    return _create_and_save_topomap(evoked, times, save_path, plot_config, logger)


def _plot_condition_topomaps(
    epochs: mne.Epochs,
    subject: str,
    times: List[float],
    conditions: Dict[str, str],
    save_dir: Path,
    plot_config: Any,
    logger: logging.Logger,
) -> List[Path]:
    """Plot topomaps for each condition separately.
    
    Args:
        epochs: MNE Epochs object.
        subject: Subject identifier.
        times: List of time points to plot.
        conditions: Dictionary mapping condition names to query strings.
        save_dir: Directory to save figures.
        plot_config: Plot configuration object.
        logger: Logger instance.
    
    Returns:
        List of paths where figures were saved.
    """
    saved_paths = []
    primary_ext = _get_primary_extension(plot_config)
    
    for condition_name, query in conditions.items():
        try:
            condition_epochs = epochs[query]
            if len(condition_epochs) == 0:
                logger.debug(f"Skipping condition '{condition_name}': no epochs found")
                continue
            
            evoked_condition = condition_epochs.average()
            save_path = save_dir / f"sub-{subject}_erp_topomaps_{condition_name}.{primary_ext}"
            saved_path = _create_and_save_topomap(
                evoked_condition, times, save_path, plot_config, logger
            )
            saved_paths.append(saved_path)
        except (KeyError, ValueError, IndexError) as error:
            logger.warning(
                f"Failed to plot topomap for condition '{condition_name}': {error}"
            )
    
    return saved_paths


def _plot_contrast_topomaps(
    epochs: mne.Epochs,
    subject: str,
    times: List[float],
    conditions: Dict[str, str],
    save_dir: Path,
    plot_config: Any,
    logger: logging.Logger,
) -> Optional[Path]:
    """Plot contrast topomaps between first two conditions.
    
    Args:
        epochs: MNE Epochs object.
        subject: Subject identifier.
        times: List of time points to plot.
        conditions: Dictionary mapping condition names to query strings.
        save_dir: Directory to save figure.
        plot_config: Plot configuration object.
        logger: Logger instance.
    
    Returns:
        Path where figure was saved, or None if contrast could not be computed.
    """
    condition_keys = list(conditions.keys())
    if len(condition_keys) < _MIN_CONDITIONS_FOR_CONTRAST:
        return None
    
    try:
        evoked_a = epochs[conditions[condition_keys[0]]].average()
        evoked_b = epochs[conditions[condition_keys[1]]].average()
        difference = mne.combine_evoked([evoked_a, evoked_b], weights=[1, -1])
        
        primary_ext = _get_primary_extension(plot_config)
        save_path = save_dir / f"sub-{subject}_erp_topomaps_contrast.{primary_ext}"
        return _create_and_save_topomap(difference, times, save_path, plot_config, logger)
    except (KeyError, ValueError, IndexError) as error:
        logger.warning(f"Failed to plot contrast topomaps: {error}")
        return None


def plot_erp_topomaps(
    epochs: mne.Epochs,
    subject: str,
    save_dir: Path,
    config: Any,
    logger: logging.Logger,
    conditions: Optional[Dict[str, str]] = None,
) -> List[Path]:
    """Plot ERP topomaps for defined components.
    
    Creates three types of topomaps:
    1. Overall topomaps using all trials
    2. Condition-specific topomaps (if conditions provided)
    3. Contrast topomaps between first two conditions (if 2+ conditions provided)
    
    Args:
        epochs: Epochs object containing ERP data.
        subject: Subject identifier for file naming.
        save_dir: Directory where figures will be saved.
        config: Configuration object containing ERP component definitions.
        logger: Logger instance for warnings and errors.
        conditions: Optional dictionary mapping condition names to query strings.
    
    Returns:
        List of paths where figures were saved.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    plot_config = get_plot_config(config)
    
    erp_config = config.get("feature_engineering.erp", {})
    components = erp_config.get("components", [])
    if not components:
        logger.warning("No ERP components defined in config; using defaults for topomaps.")
        components = _DEFAULT_COMPONENTS
    
    times = _extract_component_times(components)
    saved_paths = []
    
    with use_style(context="paper"):
        overall_path = _plot_overall_topomaps(
            epochs, subject, times, save_dir, plot_config, logger
        )
        saved_paths.append(overall_path)
        
        if conditions:
            condition_paths = _plot_condition_topomaps(
                epochs, subject, times, conditions, save_dir, plot_config, logger
            )
            saved_paths.extend(condition_paths)
            
            contrast_path = _plot_contrast_topomaps(
                epochs, subject, times, conditions, save_dir, plot_config, logger
            )
            if contrast_path is not None:
                saved_paths.append(contrast_path)
    
    return saved_paths
