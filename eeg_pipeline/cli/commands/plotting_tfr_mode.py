"""TFR mode helpers for plotting CLI command."""

from __future__ import annotations

import argparse
from typing import Any, Dict, List, Optional

from eeg_pipeline.cli.commands.plotting_catalog import PLOT_BY_ID
from eeg_pipeline.cli.commands.plotting_config_overrides import apply_all_config_overrides
from eeg_pipeline.cli.commands.plotting_definition_helpers import collect_plot_definitions
from eeg_pipeline.cli.commands.plotting_item_overrides import (
    apply_plot_item_overrides,
    parse_plot_item_configs,
    validate_plot_item_configs,
)


def update_tfr_config(
    config: Any,
    bands: Optional[List[str]] = None,
    tmin: Optional[float] = None,
    tmax: Optional[float] = None,
) -> None:
    """Update config with TFR analysis parameters."""
    tfr_section = config.setdefault("time_frequency_analysis", {})
    if bands is not None:
        tfr_section["selected_bands"] = bands
    if tmin is not None:
        tfr_section["tmin"] = tmin
    if tmax is not None:
        tfr_section["tmax"] = tmax


def validate_time_range(tmin: Optional[float], tmax: Optional[float]) -> None:
    """Validate that time range is logically consistent."""
    if tmin is not None and tmax is not None and tmin >= tmax:
        raise ValueError(f"tmin ({tmin}) must be less than tmax ({tmax})")


def run_tfr_mode(
    *,
    args: argparse.Namespace,
    subjects: List[str],
    config: Any,
    create_progress_reporter: Any,
) -> None:
    """Execute TFR visualization mode."""
    from eeg_pipeline.plotting.orchestration.tfr import visualize_tfr_for_subjects

    validate_time_range(args.tmin, args.tmax)
    update_tfr_config(config, args.bands, args.tmin, args.tmax)

    apply_all_config_overrides(args, config)

    plot_item_configs = parse_plot_item_configs(args.plot_item_config)
    if plot_item_configs:
        validate_plot_item_configs(plot_item_configs, set(PLOT_BY_ID.keys()))
        for plot_id, overrides in plot_item_configs.items():
            definition = PLOT_BY_ID.get(plot_id)
            if definition and definition.tfr_plots:
                apply_plot_item_overrides(config, overrides)

    progress = create_progress_reporter(args)
    progress.start("tfr_visualize", subjects)
    progress.step("Rendering TFR plots", current=1, total=2)

    tfr_plots = None
    if args.plots:
        _, _, _, tfr_plots, _, _ = collect_plot_definitions(args.plots)
        tfr_plots = tfr_plots if tfr_plots else None

    visualize_tfr_for_subjects(
        subjects=subjects,
        task=args.task,
        tfr_roi_only=args.tfr_roi,
        tfr_topomaps_only=args.tfr_topomaps_only,
        plots=tfr_plots,
        n_jobs=args.n_jobs,
        config=config,
    )

    progress.step("Finalizing", current=2, total=2)
    progress.complete(success=True)
