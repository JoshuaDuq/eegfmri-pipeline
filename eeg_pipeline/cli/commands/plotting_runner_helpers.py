"""Rendering helpers for plotting CLI command."""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Set

from eeg_pipeline.cli.commands.plotting_catalog import PLOT_BY_ID
from eeg_pipeline.cli.commands.plotting_definition_helpers import map_plot_id_to_plotters
from eeg_pipeline.cli.commands.plotting_item_overrides import apply_plot_item_overrides
from eeg_pipeline.cli.commands.plotting_selection import unique_in_order
from eeg_pipeline.utils.config.loader import ConfigDict


SHARED_COMPARISON_OVERRIDE_KEYS = {
    "compare_windows",
    "comparison_windows",
    "compare_columns",
    "comparison_segment",
    "comparison_column",
    "comparison_values",
    "comparison_labels",
    "comparison_rois",
}


def _collect_shared_group_overrides(
    plot_ids: List[str],
    plot_item_configs: Dict[str, Dict[str, List[str]]],
) -> Dict[str, Dict[str, List[str]]]:
    """Collect group-wide comparison overrides and reject conflicts."""
    shared_by_group: Dict[str, Dict[str, List[str]]] = {}

    for plot_id in plot_ids:
        definition = PLOT_BY_ID.get(plot_id)
        if definition is None or not definition.group:
            continue

        overrides = plot_item_configs.get(plot_id, {})
        if not overrides:
            continue

        group_shared = shared_by_group.setdefault(definition.group, {})
        for key in SHARED_COMPARISON_OVERRIDE_KEYS:
            if key not in overrides:
                continue

            values = list(overrides[key])
            if key in group_shared and group_shared[key] != values:
                raise ValueError(
                    "Conflicting --plot-item-config values for shared comparison setting "
                    f"{key!r} in plot group {definition.group!r}."
                )
            group_shared[key] = values

    return shared_by_group


def _resolve_plot_overrides(
    plot_id: str,
    plot_item_configs: Dict[str, Dict[str, List[str]]],
    shared_group_overrides: Dict[str, Dict[str, List[str]]],
) -> Dict[str, List[str]]:
    """Merge group-shared comparison overrides with plot-specific overrides."""
    definition = PLOT_BY_ID.get(plot_id)
    if definition is None or not definition.group:
        return dict(plot_item_configs.get(plot_id, {}))

    resolved = dict(shared_group_overrides.get(definition.group, {}))
    resolved.update(plot_item_configs.get(plot_id, {}))
    return resolved


def render_plots_with_per_plot_config(
    plot_ids: List[str],
    plot_item_configs: Dict[str, Dict[str, List[str]]],
    subjects: List[str],
    task: str,
    config: Any,
    selected_feature_plotters: Optional[List[str]],
    progress: Any,
) -> None:
    """Render plots with per-plot configuration overrides."""
    from eeg_pipeline.plotting.orchestration.features import visualize_features_for_subjects
    from eeg_pipeline.plotting.orchestration.behavior import visualize_behavior_for_subjects
    from eeg_pipeline.plotting.orchestration.tfr import visualize_tfr_for_subjects
    from eeg_pipeline.plotting.orchestration.erp import visualize_erp_for_subjects

    shared_group_overrides = _collect_shared_group_overrides(plot_ids, plot_item_configs)
    total = len(plot_ids)
    for idx, plot_id in enumerate(plot_ids, start=1):
        definition = PLOT_BY_ID.get(plot_id)
        if definition is None:
            continue

        plot_config = ConfigDict(copy.deepcopy(dict(config)))
        overrides = _resolve_plot_overrides(plot_id, plot_item_configs, shared_group_overrides)
        if overrides:
            apply_plot_item_overrides(plot_config, overrides)

        progress.step(f"Rendering {definition.label or plot_id}", current=idx, total=total)

        if definition.feature_categories:
            patterns = (
                list(definition.feature_plot_patterns)
                if definition.feature_plot_patterns
                else [plot_id]
            )
            
            # Map plot ID to plotter names if feature_plotters not explicitly provided
            plotter_names = selected_feature_plotters
            if plotter_names is None:
                plotter_names = map_plot_id_to_plotters(plot_id, definition.feature_categories)
            
            visualize_features_for_subjects(
                subjects=subjects,
                task=task,
                config=plot_config,
                visualize_categories=sorted(definition.feature_categories),
                feature_plotters=plotter_names,
                plot_name_patterns=patterns,
            )
        if definition.behavior_plots:
            visualize_behavior_for_subjects(
                subjects=subjects,
                task=task,
                config=plot_config,
                plots=unique_in_order(list(definition.behavior_plots)),
            )
        if definition.tfr_plots:
            visualize_tfr_for_subjects(
                subjects=subjects,
                task=task,
                config=plot_config,
                plots=unique_in_order(list(definition.tfr_plots)),
            )
        if definition.erp_plots:
            plots_list = definition.erp_plots if isinstance(definition.erp_plots, list) else [definition.erp_plots]
            visualize_erp_for_subjects(
                subjects=subjects,
                task=task,
                config=plot_config,
                plots=unique_in_order(list(plots_list)),
            )


def render_plots_without_per_plot_config(
    feature_categories: Set[str],
    feature_plot_patterns: Set[str],
    behavior_plots: List[str],
    tfr_plots: List[str],
    erp_plots: List[str],
    subjects: List[str],
    task: str,
    config: Any,
    selected_feature_plotters: Optional[List[str]],
    progress: Any,
) -> None:
    """Render plots without per-plot configuration overrides."""
    from eeg_pipeline.plotting.orchestration.features import visualize_features_for_subjects
    from eeg_pipeline.plotting.orchestration.behavior import visualize_behavior_for_subjects
    from eeg_pipeline.plotting.orchestration.tfr import visualize_tfr_for_subjects
    from eeg_pipeline.plotting.orchestration.erp import visualize_erp_for_subjects

    steps = sum([
        bool(feature_categories),
        bool(behavior_plots),
        bool(tfr_plots),
        bool(erp_plots),
    ])
    step_idx = 0

    if feature_categories:
        step_idx += 1
        progress.step("Rendering feature plots", current=step_idx, total=steps)
        visualize_features_for_subjects(
            subjects=subjects,
            task=task,
            config=config,
            visualize_categories=sorted(feature_categories),
            feature_plotters=selected_feature_plotters,
            plot_name_patterns=sorted(feature_plot_patterns) if feature_plot_patterns else None,
        )

    if behavior_plots:
        step_idx += 1
        progress.step("Rendering behavior plots", current=step_idx, total=steps)
        visualize_behavior_for_subjects(
            subjects=subjects,
            task=task,
            config=config,
            plots=behavior_plots,
        )

    if tfr_plots:
        step_idx += 1
        progress.step("Rendering TFR plots", current=step_idx, total=steps)
        visualize_tfr_for_subjects(
            subjects=subjects,
            task=task,
            config=config,
            plots=tfr_plots,
        )

    if erp_plots:
        step_idx += 1
        progress.step("Rendering ERP plots", current=step_idx, total=steps)
        visualize_erp_for_subjects(
            subjects=subjects,
            task=task,
            config=config,
            plots=erp_plots,
        )


def run_group_plotting(
    *,
    subjects: List[str],
    task: str,
    config: Any,
    plot_ids: List[str],
    plot_item_configs: Dict[str, Dict[str, List[str]]],
    progress: Any,
) -> None:
    """Run group-level plotting for selected plot IDs."""
    import logging

    from eeg_pipeline.plotting.orchestration.features import (
        visualize_band_power_topomaps_for_group,
        visualize_power_by_condition_for_group,
        visualize_power_timecourse_for_group,
        visualize_power_spectral_density_for_group,
    )

    logger = logging.getLogger(__name__)
    group_plotters = {
        "band_power_topomaps": visualize_band_power_topomaps_for_group,
        "power_by_condition": visualize_power_by_condition_for_group,
        "power_timecourse": visualize_power_timecourse_for_group,
        "power_spectral_density": visualize_power_spectral_density_for_group,
    }
    supported = set(group_plotters.keys())
    shared_group_overrides = _collect_shared_group_overrides(plot_ids, plot_item_configs)

    total = len(plot_ids)
    for idx, plot_id in enumerate(plot_ids, start=1):
        definition = PLOT_BY_ID.get(plot_id)
        label = (definition.label if definition else "") or plot_id
        progress.step(f"Rendering {label} (group)", current=idx, total=total)

        if plot_id not in supported:
            logger.warning(
                "Group plotting currently supports only %s; skipping %s",
                ", ".join(sorted(supported)),
                plot_id,
            )
            continue

        plot_config = ConfigDict(copy.deepcopy(dict(config)))
        overrides = _resolve_plot_overrides(plot_id, plot_item_configs, shared_group_overrides)
        if overrides:
            apply_plot_item_overrides(plot_config, overrides)

        group_plotters[plot_id](
            subjects=subjects,
            task=task,
            config=plot_config,
            logger=logger,
        )
