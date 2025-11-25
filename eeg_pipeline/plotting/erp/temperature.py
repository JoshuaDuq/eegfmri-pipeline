"""
ERP temperature analysis plotting functions.

Functions for creating ERP plots analyzing responses by temperature levels.
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
    find_temperature_column_in_metadata,
    sanitize_label as _sanitize_label,
)
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


def _collect_evokeds_by_temperature(
    all_epochs: List[mne.Epochs],
    config,
    erp_picks: str
) -> Dict[str, List[mne.Evoked]]:
    """Collect evoked responses grouped by temperature level across subjects.
    
    Args:
        all_epochs: List of MNE Epochs objects from multiple subjects
        config: Config dictionary
        erp_picks: Channel picks string for ERP analysis
    
    Returns:
        Dictionary mapping temperature labels to lists of Evoked objects
    """
    evokeds_by_temperature: Dict[str, List[mne.Evoked]] = {}
    for epochs in all_epochs:
        temperature_column = _find_temperature_column(epochs, config)
        if temperature_column is None:
            continue
        temperature_levels, temperature_labels, is_numeric = (
            process_temperature_levels(epochs, temperature_column)
        )
        for level in temperature_levels:
            query, label = build_epoch_query_string(
                temperature_column, level, is_numeric, temperature_labels
            )
            epochs_at_level = epochs[query]
            if len(epochs_at_level) > 0:
                evoked = epochs_at_level.average(picks=erp_picks)
                evokeds_by_temperature.setdefault(label, []).append(evoked)
    return evokeds_by_temperature


def _save_group_erp_individual_temperature_plots(
    grand_averages: Dict[str, mne.Evoked],
    output_dir: Path,
    erp_picks: str,
    erp_combine: str,
    erp_output_files: Dict[str, str],
    baseline_str: str,
    config,
    logger
) -> None:
    """Save individual temperature plots for group ERP analysis.
    
    Args:
        grand_averages: Dictionary mapping temperature labels to grand average Evoked objects
        output_dir: Directory to save output plots
        erp_picks: Channel picks string for ERP analysis
        erp_combine: Method to combine channels
        erp_output_files: Dictionary mapping plot types to output filenames
        baseline_str: Baseline string for footer
        config: Config dictionary
        logger: Optional logger instance
    """
    for label, evoked in grand_averages.items():
        figure = _plot_evoked_with_agg_backend(
            evoked, erp_picks, f"Group ERP - Temperature {label}"
        )
        safe_label = _sanitize_label(label)
        template_name = erp_output_files.get(
            "temp_butterfly_template",
            "erp_by_temperature_butterfly_{label}.png"
        )
        output_name = "group_" + template_name.format(label=safe_label)
        output_path = output_dir / output_name
        plot_cfg = get_plot_config(config)
        footer_text = build_footer(
            "erp_complete", config=config, baseline=baseline_str, method=erp_combine
        )
        ensure_dir(output_path.parent)
        save_fig(
            figure, output_path, logger=logger, footer=footer_text,
            tight_layout_rect=plot_cfg.get_layout_rect("tight_rect"),
            formats=plot_cfg.formats,
            dpi=plot_cfg.dpi,
            bbox_inches=plot_cfg.bbox_inches,
            pad_inches=plot_cfg.pad_inches
        )


def _save_group_erp_comparison_plot(
    grand_averages: Dict[str, mne.Evoked],
    evokeds_by_temperature: Dict[str, List[mne.Evoked]],
    output_dir: Path,
    erp_picks: str,
    erp_combine: str,
    erp_output_files: Dict[str, str],
    baseline_str: str,
    config,
    logger,
    combine_method: Optional[str],
    output_key: str,
    title_template: str
) -> None:
    """Save group ERP comparison plot for temperature analysis.
    
    Args:
        grand_averages: Dictionary mapping temperature labels to grand average Evoked objects
        evokeds_by_temperature: Dictionary mapping temperature labels to lists of Evoked objects
        output_dir: Directory to save output plots
        erp_picks: Channel picks string for ERP analysis
        erp_combine: Method to combine channels
        erp_output_files: Dictionary mapping plot types to output filenames
        baseline_str: Baseline string for footer
        config: Config dictionary
        logger: Optional logger instance
        combine_method: Optional method to combine channels for this plot
        output_key: Key in erp_output_files for output filename
        title_template: Template string for plot title
    """
    figure = _plot_compare_evokeds_silent(grand_averages, erp_picks, combine_method)
    unwrapped_figure = _unwrap_figure(figure)
    subject_info = ", ".join([
        f"{label}: N={len(evokeds_by_temperature[label])}"
        for label in grand_averages.keys()
    ])
    unwrapped_figure.suptitle(title_template.format(subject_info), fontsize=14, fontweight='bold')
    plot_cfg = get_plot_config(config)
    footer_text = build_footer(
        "erp_complete", config=config, baseline=baseline_str, method=erp_combine
    )
    output_name = "group_" + erp_output_files.get(output_key, f"erp_by_temperature_{output_key}.png")
    output_path = output_dir / output_name
    ensure_dir(output_path.parent)
    save_fig(
        unwrapped_figure, output_path, logger=logger, footer=footer_text,
        tight_layout_rect=plot_cfg.get_layout_rect("tight_rect"),
        formats=plot_cfg.formats,
        dpi=plot_cfg.dpi,
        bbox_inches=plot_cfg.bbox_inches,
        pad_inches=plot_cfg.pad_inches
    )


def _save_group_erp_comparison_plots(
    grand_averages: Dict[str, mne.Evoked],
    evokeds_by_temperature: Dict[str, List[mne.Evoked]],
    output_dir: Path,
    erp_picks: str,
    erp_combine: str,
    erp_output_files: Dict[str, str],
    baseline_str: str,
    config,
    logger
) -> None:
    """Save multiple group ERP comparison plots for temperature analysis.
    
    Args:
        grand_averages: Dictionary mapping temperature labels to grand average Evoked objects
        evokeds_by_temperature: Dictionary mapping temperature labels to lists of Evoked objects
        output_dir: Directory to save output plots
        erp_picks: Channel picks string for ERP analysis
        erp_combine: Method to combine channels
        erp_output_files: Dictionary mapping plot types to output filenames
        baseline_str: Baseline string for footer
        config: Config dictionary
        logger: Optional logger instance
    """
    _save_group_erp_comparison_plot(
        grand_averages, evokeds_by_temperature, output_dir,
        erp_picks, erp_combine, erp_output_files, baseline_str, config, logger,
        None, "temp_butterfly", "Group ERP by Temperature (Butterfly) — {}"
    )
    _save_group_erp_comparison_plot(
        grand_averages, evokeds_by_temperature, output_dir,
        erp_picks, erp_combine, erp_output_files, baseline_str, config, logger,
        erp_combine, "temp_gfp", "Group ERP by Temperature ({} subjects)"
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


def group_erp_by_temperature(
    all_epochs: List[mne.Epochs],
    output_dir: Path,
    config,
    baseline_window: Tuple[float, float],
    erp_picks: str,
    erp_combine: str,
    erp_output_files: Dict[str, str],
    logger: Optional[object] = None
) -> None:
    """Create group-level ERP plots analyzing responses by temperature levels.
    
    Args:
        all_epochs: List of MNE Epochs objects from multiple subjects
        output_dir: Directory to save output plots
        config: Config dictionary
        baseline_window: Tuple of (baseline_start, baseline_end) in seconds
        erp_picks: Channel picks string for ERP analysis
        erp_combine: Method to combine channels (e.g., 'gfp')
        erp_output_files: Dictionary mapping plot types to output filenames
        logger: Optional logger instance
    """
    if not all_epochs:
        return
    
    evokeds_by_temperature = _collect_evokeds_by_temperature(
        all_epochs, config, erp_picks
    )
    
    if not evokeds_by_temperature:
        _log_if_present(logger, "warning", "Group ERP by temperature: No evokeds computed across subjects")
        return
    
    grand_averages = {
        label: mne.grand_average(evokeds, interpolate_bads=True)
        for label, evokeds in evokeds_by_temperature.items()
        if evokeds
    }
    
    baseline_str = _format_baseline_string(baseline_window)
    
    _save_group_erp_individual_temperature_plots(
        grand_averages, output_dir, erp_picks, erp_combine,
        erp_output_files, baseline_str, config, logger
    )
    
    if len(grand_averages) >= 2:
        _save_group_erp_comparison_plots(
            grand_averages, evokeds_by_temperature, output_dir,
            erp_picks, erp_combine, erp_output_files, baseline_str, config, logger
        )

