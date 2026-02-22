"""Definition mapping helpers for plotting CLI command."""

from __future__ import annotations

from typing import Any, List, Optional, Set, Dict

from eeg_pipeline.cli.commands.plotting_catalog import PLOT_BY_ID
from eeg_pipeline.cli.commands.plotting_selection import unique_in_order


def map_plot_id_to_plotters(plot_id: str, feature_categories: List[str]) -> Optional[List[str]]:
    """Map a plot ID to feature plotter tokens in ``category.plotter`` format."""
    plot_id_to_plotter = {
        "power_by_condition": "plot_power_condition_comparison",
        "power_spectral_density": "plot_psd_visualization",
        "band_power_topomaps": "plot_power_summary",
        "cross_frequency_power_correlation": "plot_power_condition_comparison",
        "connectivity_by_condition": "plot_connectivity_condition",
        "connectivity_circle_condition": "plot_connectivity_mne_suite",
        "connectivity_heatmap": "plot_connectivity_mne_suite",
        "connectivity_network": "plot_connectivity_mne_suite",
        "aperiodic_topomaps": "aperiodic_suite",
        "aperiodic_by_condition": "aperiodic_suite",
        "itpc_topomaps": "itpc_suite",
        "itpc_by_condition": "itpc_suite",
        "pac_by_condition": "pac_suite",
        "erds_by_condition": "plot_erds",
        "complexity_by_condition": "plot_complexity",
        "spectral_by_condition": "plot_spectral",
        "ratios_by_condition": "plot_ratios",
        "asymmetry_by_condition": "plot_asymmetry",
        "bursts_by_condition": "plot_bursts",
        "erp_butterfly": "erp_suite",
        "erp_roi": "erp_suite",
        "erp_contrast": "erp_suite",
    }

    plotter_name = plot_id_to_plotter.get(plot_id)
    if plotter_name is None:
        return None
    result = [f"{category}.{plotter_name}" for category in feature_categories]
    return result if result else None


def collect_plot_definitions(
    plot_ids: List[str],
) -> tuple[Set[str], Set[str], List[str], List[str], List[str], List[str]]:
    """Collect categories and plot modes from selected plot IDs."""
    feature_categories: Set[str] = set()
    feature_plot_patterns: Set[str] = set()
    feature_plotters: Set[str] = set()
    behavior_plots: List[str] = []
    tfr_plots: List[str] = []
    erp_plots: List[Any] = []

    for plot_id in plot_ids:
        definition = PLOT_BY_ID.get(plot_id)
        if definition is None:
            continue
        if definition.feature_categories:
            feature_categories.update(definition.feature_categories)
            if definition.feature_plot_patterns:
                feature_plot_patterns.update(str(p) for p in definition.feature_plot_patterns)
            else:
                feature_plot_patterns.add(plot_id)
            plotter_names = map_plot_id_to_plotters(plot_id, list(definition.feature_categories))
            if plotter_names:
                feature_plotters.update(plotter_names)
        if definition.behavior_plots:
            behavior_plots.extend(definition.behavior_plots)
        if definition.tfr_plots:
            tfr_plots.extend(definition.tfr_plots)
        if definition.erp_plots:
            erp_plots.append(definition.erp_plots)

    flat_erp_plots: List[str] = []
    for plot in erp_plots:
        if isinstance(plot, list):
            flat_erp_plots.extend(plot)
        else:
            flat_erp_plots.append(plot)

    return (
        feature_categories,
        feature_plot_patterns,
        unique_in_order(behavior_plots),
        unique_in_order(tfr_plots),
        unique_in_order(flat_erp_plots),
        sorted(feature_plotters) if feature_plotters else [],
    )
