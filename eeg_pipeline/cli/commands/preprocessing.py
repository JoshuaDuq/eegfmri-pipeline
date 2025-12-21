"""Preprocessing CLI command."""

from __future__ import annotations

import argparse
from typing import Any, List

from eeg_pipeline.cli.common import (
    add_common_subject_args,
    add_task_arg,
    add_output_format_args,
    resolve_task,
    create_progress_reporter,
)


def setup_preprocessing(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """Configure the preprocessing command parser."""
    parser = subparsers.add_parser(
        "preprocessing",
        help="EEG preprocessing: bad channels, ICA, epochs",
        description="Run EEG preprocessing pipeline: detect bad channels, fit ICA, create epochs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "mode",
        choices=["full", "bad-channels", "ica", "epochs"],
        help="Preprocessing mode: full (all steps), bad-channels, ica, or epochs",
    )
    add_common_subject_args(parser)
    add_task_arg(parser)
    add_output_format_args(parser)
    
    prep_group = parser.add_argument_group("Preprocessing options")
    prep_group.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel jobs for bad channel detection",
    )
    prep_group.add_argument(
        "--use-icalabel",
        action="store_true",
        default=True,
        help="Use mne-icalabel for ICA component classification (default: True)",
    )
    prep_group.add_argument(
        "--no-icalabel",
        dest="use_icalabel",
        action="store_false",
        help="Disable mne-icalabel, use MNE-BIDS pipeline ICA detection",
    )
    prep_group.add_argument(
        "--use-pyprep",
        action="store_true",
        default=True,
        help="Use PyPREP for bad channel detection (default: True)",
    )
    prep_group.add_argument(
        "--no-pyprep",
        dest="use_pyprep",
        action="store_false",
        help="Disable PyPREP bad channel detection",
    )
    
    return parser


def run_preprocessing(args: argparse.Namespace, subjects: List[str], config: Any) -> None:
    """Execute the preprocessing command."""
    from eeg_pipeline.pipelines.preprocessing import PreprocessingPipeline
    
    progress = create_progress_reporter(args)
    task = resolve_task(args.task, config)
    
    pipeline = PreprocessingPipeline(config=config)
    
    pipeline.run_batch(
        subjects=subjects,
        task=task,
        mode=args.mode,
        use_pyprep=args.use_pyprep,
        use_icalabel=args.use_icalabel,
        n_jobs=args.n_jobs,
        progress=progress,
    )
