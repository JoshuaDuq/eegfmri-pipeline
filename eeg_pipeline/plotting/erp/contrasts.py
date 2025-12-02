"""
ERP pain contrast plotting functions.

Functions for creating ERP plots comparing painful vs non-painful conditions.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mne

from ...utils.io.general import (
    unwrap_figure as _unwrap_figure,
    format_baseline_string as _format_baseline_string,
    log_if_present as _log_if_present,
    validate_epochs_for_plotting,
    build_footer,
    save_fig,
    ensure_dir,
    find_pain_column_in_metadata,
)
from ...utils.data.loading import select_epochs_by_value
from ..config import get_plot_config


###################################################################
# Helper Functions
###################################################################


def _apply_baseline_correction(epochs: mne.Epochs, baseline_window: Tuple[float, float]) -> mne.Epochs:
    """Apply baseline correction to epochs.
    
    Args:
        epochs: MNE Epochs object
        baseline_window: Tuple of (baseline_start, baseline_end) in seconds
    
    Returns:
        Baseline-corrected Epochs object
    """
    baseline_start = float(baseline_window[0])
    baseline_end = min(float(baseline_window[1]), 0.0)
    return epochs.copy().apply_baseline((baseline_start, baseline_end))


def _find_pain_column(epochs: mne.Epochs, config) -> Optional[str]:
    """Find pain column in epochs metadata.
    
    Args:
        epochs: MNE Epochs object
        config: Config dictionary
    
    Returns:
        Pain column name if found, None otherwise
    """
    return find_pain_column_in_metadata(epochs, config)


def _save_erp_figure(
    figure, output_path: Path, config, baseline_str: str, method: str, logger,
    n_epochs_info: Optional[str] = None
) -> None:
    """Save ERP figure with footer and formatting.
    
    Args:
        figure: Matplotlib figure or MNE figure object
        output_path: Path to save figure
        config: Config dictionary
        baseline_str: Baseline string for footer
        method: ERP combination method
        logger: Optional logger instance
        n_epochs_info: Optional string with epoch counts (e.g., "Pain: n=30, Non-pain: n=45")
    """
    plot_cfg = get_plot_config(config)
    footer_text = build_footer("erp_complete", config=config, baseline=baseline_str, method=method)
    if n_epochs_info:
        footer_text = f"{footer_text} | {n_epochs_info}"
    ensure_dir(output_path.parent)
    save_fig(
        _unwrap_figure(figure),
        output_path,
        logger=logger,
        footer=footer_text,
        tight_layout_rect=plot_cfg.get_layout_rect("tight_rect"),
        formats=plot_cfg.formats,
        dpi=plot_cfg.dpi,
        bbox_inches=plot_cfg.bbox_inches,
        pad_inches=plot_cfg.pad_inches
    )


def _create_evoked_comparison_plot(
    evokeds_dict: Dict[str, mne.Evoked],
    picks: str,
    combine_method: Optional[str],
    colors: Dict[str, str]
):
    """Create comparison plot for multiple evoked responses.
    
    Args:
        evokeds_dict: Dictionary mapping condition names to Evoked objects
        picks: Channel picks string
        combine_method: Optional method to combine channels (e.g., 'gfp')
        colors: Dictionary mapping condition names to colors
    
    Returns:
        MNE figure object
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mne.set_log_level('ERROR')
        return mne.viz.plot_compare_evokeds(
            evokeds_dict, picks=picks, combine=combine_method, show=False, colors=colors
        )


###################################################################
# ERP Pain Contrast Functions
###################################################################


def erp_contrast_pain(
    epochs: mne.Epochs,
    output_dir: Path,
    config,
    baseline_window: Tuple[float, float],
    erp_picks: str,
    pain_color: str,
    nonpain_color: str,
    erp_combine: str,
    erp_output_files: Dict[str, str],
    logger: Optional[object] = None,
    subject: Optional[str] = None
) -> None:
    """Create ERP contrast plot comparing painful vs non-painful conditions.
    
    Args:
        epochs: MNE Epochs object
        output_dir: Directory to save output plots
        config: Config dictionary
        baseline_window: Tuple of (baseline_start, baseline_end) in seconds
        erp_picks: Channel picks string for ERP analysis
        pain_color: Color for painful condition
        nonpain_color: Color for non-painful condition
        erp_combine: Method to combine channels (e.g., 'gfp')
        erp_output_files: Dictionary mapping plot types to output filenames
        logger: Optional logger instance
        subject: Optional subject identifier
    """
    if not validate_epochs_for_plotting(epochs, logger):
        return
    
    pain_column = _find_pain_column(epochs, config)
    if pain_column is None:
        _log_if_present(logger, "warning", "ERP pain contrast: No pain column found in metadata.")
        return

    epochs_baselined = _apply_baseline_correction(epochs, baseline_window)
    epochs_pain = select_epochs_by_value(epochs_baselined, pain_column, 1)
    epochs_nonpain = select_epochs_by_value(epochs_baselined, pain_column, 0)
    n_pain = len(epochs_pain)
    n_nonpain = len(epochs_nonpain)
    if n_pain == 0 or n_nonpain == 0:
        _log_if_present(logger, "warning", "ERP pain contrast: one of the groups has zero trials.")
        return

    evoked_pain = epochs_pain.average(picks=erp_picks)
    evoked_nonpain = epochs_nonpain.average(picks=erp_picks)
    colors = {"painful": pain_color, "non-painful": nonpain_color}
    
    subject_prefix = f"sub-{subject}_" if subject else ""
    baseline_str = _format_baseline_string(baseline_window)
    n_epochs_info = f"Pain: n={n_pain}, Non-pain: n={n_nonpain}"
    
    butterfly_name = erp_output_files.get("pain_butterfly", "erp_pain_binary_butterfly.png")
    butterfly_name = subject_prefix + butterfly_name.replace(".png", ".svg")
    evokeds_dict = {"painful": evoked_pain, "non-painful": evoked_nonpain}
    figure = _create_evoked_comparison_plot(evokeds_dict, erp_picks, None, colors)
    output_path = output_dir / butterfly_name
    _save_erp_figure(figure, output_path, config, baseline_str, erp_combine, logger, n_epochs_info)



