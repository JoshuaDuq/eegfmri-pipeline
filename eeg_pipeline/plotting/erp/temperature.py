"""
ERP temperature analysis plotting functions.

Functions for creating ERP plots analyzing responses by temperature levels.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mne

from eeg_pipeline.utils.io.plotting import (
    unwrap_figure as _unwrap_figure,
    log_if_present as _log_if_present,
    build_footer,
    save_fig,
)
from eeg_pipeline.utils.io.formatting import format_baseline_string as _format_baseline_string, sanitize_label as _sanitize_label
from eeg_pipeline.utils.validation import validate_epochs_for_plotting
from eeg_pipeline.utils.io.paths import ensure_dir
from eeg_pipeline.utils.io.columns import find_temperature_column_in_metadata
from ...utils.data.loading import (
    process_temperature_levels,
    build_epoch_query_string,
)
from ..config import get_plot_config
from .contrasts import _apply_baseline_correction, _save_erp_figure


###################################################################
# Helper Functions
###################################################################


def _find_temperature_column(epochs: mne.Epochs, config) -> Optional[str]:
    """Find temperature column in epochs metadata.
    
    Args:
        epochs: MNE Epochs object
        config: Config dictionary
    
    Returns:
        Temperature column name if found, None otherwise
    """
    return find_temperature_column_in_metadata(epochs, config)


def _plot_evoked_with_agg_backend(evoked: mne.Evoked, picks: str, title: str):
    """Plot evoked response with aggregation backend.
    
    Args:
        evoked: MNE Evoked object
        picks: Channel picks string
        title: Plot title
    
    Returns:
        Matplotlib figure object
    """
    figure = evoked.plot(picks=picks, spatial_colors=True, show=False)
    figure.suptitle(title)
    return figure


def _plot_compare_evokeds_silent(evokeds, picks, combine):
    """Plot comparison of evoked responses with warnings suppressed.
    
    Args:
        evokeds: Dictionary or list of Evoked objects
        picks: Channel picks string
        combine: Optional method to combine channels
    
    Returns:
        MNE figure object
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mne.set_log_level('ERROR')
        return mne.viz.plot_compare_evokeds(
            evokeds, picks=picks, combine=combine, show=False
        )




###################################################################
# ERP Temperature Analysis Functions
###################################################################


def erp_by_temperature(
    epochs: mne.Epochs,
    output_dir: Path,
    config,
    baseline_window: Tuple[float, float],
    erp_picks: str,
    erp_combine: str,
    erp_output_files: Dict[str, str],
    logger: Optional[object] = None,
    subject: Optional[str] = None
) -> None:
    """Create ERP plots analyzing responses by temperature levels.
    
    Args:
        epochs: MNE Epochs object
        output_dir: Directory to save output plots
        config: Config dictionary
        baseline_window: Tuple of (baseline_start, baseline_end) in seconds
        erp_picks: Channel picks string for ERP analysis
        erp_combine: Method to combine channels (e.g., 'gfp')
        erp_output_files: Dictionary mapping plot types to output filenames
        logger: Optional logger instance
        subject: Optional subject identifier
    """
    if not validate_epochs_for_plotting(epochs, logger):
        return
    
    temperature_column = _find_temperature_column(epochs, config)
    if temperature_column is None:
        _log_if_present(logger, "warning", "ERP by temperature: No temperature column found in metadata.")
        return

    epochs_baselined = _apply_baseline_correction(epochs, baseline_window)
    temperature_levels, temperature_labels, is_numeric = process_temperature_levels(
        epochs_baselined, temperature_column
    )
    evokeds_by_temperature: Dict[str, mne.Evoked] = {}
    for level in temperature_levels:
        query, label = build_epoch_query_string(
            temperature_column, level, is_numeric, temperature_labels
        )
        epochs_at_level = epochs_baselined[query]
        if len(epochs_at_level) > 0:
            evokeds_by_temperature[label] = epochs_at_level.average(picks=erp_picks)
    
    if not evokeds_by_temperature:
        _log_if_present(logger, "warning", "ERP by temperature: No evokeds computed.")
        return
    
    subject_prefix = f"sub-{subject}_" if subject else ""
    baseline_str = _format_baseline_string(baseline_window)
    template_name = erp_output_files.get(
        "temp_butterfly_template",
        "erp_by_temperature_butterfly_{label}.png"
    )
    
    for label, evoked in evokeds_by_temperature.items():
        figure = _plot_evoked_with_agg_backend(evoked, erp_picks, f"ERP - Temperature {label}")
        safe_label = _sanitize_label(label)
        output_name = template_name.format(label=safe_label)
        output_name = subject_prefix + output_name.replace(".png", ".svg")
        output_path = output_dir / output_name
        plot_cfg = get_plot_config(config)
        footer_text = build_footer(
            "erp_complete", config=config, baseline=baseline_str, method=erp_combine
        )
        ensure_dir(output_path.parent)
        save_fig(
            figure,
            output_path,
            logger=logger,
            footer=footer_text,
            tight_layout_rect=plot_cfg.get_layout_rect("tight_rect"),
            formats=plot_cfg.formats,
            dpi=plot_cfg.dpi,
            bbox_inches=plot_cfg.bbox_inches,
            pad_inches=plot_cfg.pad_inches
        )
    
    if len(evokeds_by_temperature) >= 2:
        butterfly_name = erp_output_files.get("temp_butterfly", "erp_by_temperature_butterfly.png")
        butterfly_name = subject_prefix + butterfly_name.replace(".png", ".svg")
        
        figure = _plot_compare_evokeds_silent(
            evokeds_by_temperature, erp_picks, None
        )
        _save_erp_figure(
            figure, output_dir / butterfly_name, config, baseline_str, erp_combine, logger
        )



