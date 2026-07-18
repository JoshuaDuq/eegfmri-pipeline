"""Parser for the ICA-component TFR command."""

from __future__ import annotations

import argparse

from eeg_pipeline.cli.common import (
    add_common_subject_args,
    add_output_format_args,
    add_path_args,
    add_task_arg,
)


def setup_component_tfr(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    """Configure the component-tfr command parser."""
    parser = subparsers.add_parser(
        "component-tfr",
        help="Fit 6-14 Hz ICA and compute condition-separated component TFRs",
        description=(
            "Fit one pooled 6-14 Hz analysis ICA per subject/task, apply its "
            "unmixing weights to broadband clean epochs, and compute 1-30 Hz "
            "DPSS component power separately for every condition."
        ),
    )
    add_common_subject_args(parser)
    add_task_arg(parser)
    add_output_format_args(parser)
    add_path_args(parser)
    parser.add_argument(
        "--condition-column",
        default=None,
        help=(
            "Clean-events metadata column used to split TFRs "
            "(default: component_time_frequency.condition_column)"
        ),
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=None,
        help="Parallel workers for filtering and TFR computation (default from config)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Atomically replace existing component-TFR outputs for each subject/task",
    )
    return parser
