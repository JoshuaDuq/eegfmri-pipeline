"""Execution orchestration for plotting CLI command."""

from __future__ import annotations

import argparse
from typing import Any, List

from eeg_pipeline.cli.common import create_progress_reporter, resolve_task
from eeg_pipeline.cli.commands.plotting_catalog import PLOT_BY_ID, PLOT_GROUPS
from eeg_pipeline.cli.commands.plotting_config_overrides import apply_all_config_overrides
from eeg_pipeline.cli.commands.plotting_definition_helpers import collect_plot_definitions
from eeg_pipeline.cli.commands.plotting_item_overrides import (
    parse_plot_item_configs,
    validate_plot_item_configs,
)
from eeg_pipeline.cli.commands.plotting_runner_helpers import (
    render_plots_with_per_plot_config,
    render_plots_without_per_plot_config,
    run_group_plotting,
)
from eeg_pipeline.cli.commands.plotting_selection import resolve_plot_ids
from eeg_pipeline.cli.commands.plotting_tfr_mode import run_tfr_mode


def run_plotting(args: argparse.Namespace, subjects: List[str], config: Any) -> None:
    """Execute the plotting command."""
    if args.mode == "tfr":
        if getattr(args, "analysis_scope", "subject") == "group":
            raise ValueError("Group analysis scope is not supported for mode='tfr'.")
        return run_tfr_mode(
            args=args,
            subjects=subjects,
            config=config,
            create_progress_reporter=create_progress_reporter,
        )

    if getattr(args, "bids_root", None):
        config.setdefault("paths", {})["bids_root"] = args.bids_root
    if getattr(args, "deriv_root", None):
        config.setdefault("paths", {})["deriv_root"] = args.deriv_root

    plot_ids = resolve_plot_ids(args, plot_ids=set(PLOT_BY_ID.keys()), plot_groups=PLOT_GROUPS)
    if not plot_ids:
        raise ValueError("No plots selected")

    plot_item_configs = parse_plot_item_configs(getattr(args, "plot_item_config", None))
    if plot_item_configs:
        validate_plot_item_configs(plot_item_configs, set(PLOT_BY_ID.keys()))

    selected_feature_plotters = getattr(args, "feature_plotters", None)
    apply_all_config_overrides(args, config)

    task = resolve_task(args.task, config)
    progress = create_progress_reporter(args)

    if getattr(args, "analysis_scope", "subject") == "group":
        progress.start("plotting_group", subjects)
        run_group_plotting(
            subjects=subjects,
            task=task,
            config=config,
            plot_ids=plot_ids,
            plot_item_configs=plot_item_configs,
            progress=progress,
        )
        progress.complete(success=True)
        return

    if plot_item_configs:
        progress.start("plotting", subjects)
        render_plots_with_per_plot_config(
            plot_ids=plot_ids,
            plot_item_configs=plot_item_configs,
            subjects=subjects,
            task=task,
            config=config,
            selected_feature_plotters=selected_feature_plotters,
            progress=progress,
        )
        progress.complete(success=True)
        return

    (
        feature_categories,
        feature_plot_patterns,
        behavior_plots,
        tfr_plots,
        erp_plots,
        computed_feature_plotters,
    ) = collect_plot_definitions(plot_ids)

    if not any([feature_categories, behavior_plots, tfr_plots, erp_plots]):
        raise ValueError("No plots resolved from selection")

    effective_feature_plotters = (
        computed_feature_plotters if computed_feature_plotters else selected_feature_plotters
    )

    progress.start("plotting", subjects)
    render_plots_without_per_plot_config(
        feature_categories=feature_categories,
        feature_plot_patterns=feature_plot_patterns,
        behavior_plots=behavior_plots,
        tfr_plots=tfr_plots,
        erp_plots=erp_plots,
        subjects=subjects,
        task=task,
        config=config,
        selected_feature_plotters=effective_feature_plotters,
        progress=progress,
    )
    progress.complete(success=True)
