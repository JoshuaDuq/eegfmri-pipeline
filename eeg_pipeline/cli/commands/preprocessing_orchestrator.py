"""Execution orchestrator for preprocessing CLI command."""

from __future__ import annotations

import argparse
from typing import Any, List

from eeg_pipeline.cli.common import create_progress_reporter, resolve_task
from eeg_pipeline.cli.commands.preprocessing_overrides import (
    _resolve_n_jobs,
    _update_alignment_event_config,
    _update_epochs_config,
    _update_ica_config,
    _update_icalabel_config,
    _update_path_config,
    _update_preprocessing_config,
    _update_pyprep_config,
    _validate_epoch_parameters,
)
from eeg_pipeline.utils.config.overrides import apply_set_overrides


def run_preprocessing(args: argparse.Namespace, subjects: List[str], config: Any) -> None:
    """Execute the preprocessing command."""
    from eeg_pipeline.pipelines.preprocessing import PreprocessingPipeline

    _validate_epoch_parameters(args)

    progress = create_progress_reporter(args)

    _update_path_config(args, config)
    _update_preprocessing_config(args, config)
    _update_pyprep_config(args, config)
    _update_ica_config(args, config)
    _update_icalabel_config(args, config)
    _update_epochs_config(args, config)
    _update_alignment_event_config(args, config)
    apply_set_overrides(config, getattr(args, "set_overrides", None))
    task = resolve_task(args.task, config)

    pipeline = PreprocessingPipeline(config=config)
    n_jobs = _resolve_n_jobs(args, config)

    pipeline.run_batch(
        subjects=subjects,
        task=task,
        mode=args.mode,
        use_pyprep=args.use_pyprep,
        use_icalabel=args.use_icalabel,
        n_jobs=n_jobs,
        progress=progress,
    )
