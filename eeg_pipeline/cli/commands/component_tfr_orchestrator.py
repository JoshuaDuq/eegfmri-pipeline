"""Execution orchestrator for ICA-component TFR analysis."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from eeg_pipeline.cli.common import resolve_task
from eeg_pipeline.utils.config.overrides import apply_set_overrides
from eeg_pipeline.utils.config.roots import resolve_eeg_deriv_root


def run_component_tfr(
    args: argparse.Namespace,
    subjects: list[str],
    config: Any,
) -> None:
    """Run component TFR analysis for the selected subjects."""
    from eeg_pipeline.pipelines.component_tfr import ComponentTFRPipeline

    apply_set_overrides(config, getattr(args, "set_overrides", None))
    if args.condition_column is not None:
        condition_column = str(args.condition_column).strip()
        if not condition_column:
            raise ValueError("--condition-column must not be empty.")
        config["component_time_frequency.condition_column"] = condition_column
    if args.n_jobs is not None:
        config["component_time_frequency.n_jobs"] = args.n_jobs

    task = resolve_task(args.task, config)
    deriv_root = Path(resolve_eeg_deriv_root(config))
    pipeline = ComponentTFRPipeline(config=config, deriv_root=deriv_root)
    pipeline.run_batch(
        subjects=subjects,
        task=task,
        overwrite=bool(args.overwrite),
        dry_run=bool(args.dry_run),
    )
