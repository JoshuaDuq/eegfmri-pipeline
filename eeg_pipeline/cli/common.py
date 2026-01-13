"""
Common CLI Utilities
====================

Shared argument parsing utilities and helper functions for CLI subcommands.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, List, Optional

from eeg_pipeline.cli.progress import (
    ProgressEvent,
    ProgressReporter,
    create_progress_reporter,
)
from eeg_pipeline.infra.paths import resolve_deriv_root


DEFAULT_TASK_KEY = "project.task"
MIN_SUBJECTS_KEY = "analysis.min_subjects_for_group"
MIN_SUBJECTS_FOR_ML = 2


def add_common_subject_args(parser: argparse.ArgumentParser) -> None:
    """Add mutually exclusive subject selection arguments to parser."""
    subject_group = parser.add_mutually_exclusive_group()
    subject_group.add_argument(
        "--group", type=str,
        help="Group of subjects: 'all' or comma-separated list"
    )
    subject_group.add_argument(
        "--subject", "-s", type=str, action="append",
        help="Subject label(s) without 'sub-' prefix"
    )
    subject_group.add_argument(
        "--all-subjects", action="store_true",
        help="Process all available subjects"
    )


def add_task_arg(parser: argparse.ArgumentParser) -> None:
    """Add task label argument to parser."""
    parser.add_argument(
        "--task", "-t", type=str, default=None,
        help="Task label (default from config)"
    )


def add_path_args(parser: argparse.ArgumentParser) -> None:
    """Add BIDS and derivatives path override arguments."""
    path_group = parser.add_argument_group("Path overrides")
    path_group.add_argument(
        "--bids-root",
        type=str,
        default=None,
        help="Override BIDS root path (default from config)",
    )
    path_group.add_argument(
        "--bids-fmri-root",
        type=str,
        default=None,
        help="Override BIDS fMRI root path (default from config)",
    )
    path_group.add_argument(
        "--deriv-root",
        type=str,
        default=None,
        help="Override derivatives root path (default from config)",
    )


def add_output_format_args(parser: argparse.ArgumentParser) -> None:
    """Add JSON/format output arguments to a parser."""
    parser.add_argument(
        "--json",
        action="store_true",
        dest="output_json",
        help="Output in JSON format (for TUI/scripting)"
    )
    parser.add_argument(
        "--progress-json",
        action="store_true",
        dest="progress_json",
        help="Emit progress events as JSON lines"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without executing"
    )


def resolve_task(task: Optional[str], config: Any) -> str:
    """Resolve task label from argument or config."""
    task_label = task or config.get(DEFAULT_TASK_KEY)
    if task_label is None:
        raise ValueError(f"Missing required config value: {DEFAULT_TASK_KEY}")
    return task_label


def validate_subjects_not_empty(subjects: List[str], operation: str) -> None:
    """Validate that subjects list is not empty."""
    if not subjects:
        raise ValueError(f"No subjects specified for {operation}")


def validate_min_subjects(
    subjects: List[str],
    min_count: int,
    operation: str
) -> None:
    """Validate that subjects list meets minimum count requirement."""
    if len(subjects) < min_count:
        raise ValueError(
            f"{operation} requires at least {min_count} subjects, "
            f"got {len(subjects)}"
        )


def get_deriv_root(config: Any) -> Path:
    """Get derivatives root path from config."""
    return resolve_deriv_root(config=config)


def output_json(data: Any) -> None:
    """Print data as formatted JSON."""
    print(json.dumps(data, indent=2, default=str))


def output_text(data: Any) -> None:
    """Print data as plain text."""
    print(data)


def output_formatted_text(data: Any, formatter: Callable[[Any], str]) -> None:
    """Print data formatted by the provided formatter function."""
    print(formatter(data))


def output_result(args: argparse.Namespace, data: Any, text_formatter: Optional[Callable[[Any], str]] = None) -> None:
    """Output result as JSON or formatted text based on args."""
    if getattr(args, "output_json", False):
        output_json(data)
    elif text_formatter is not None:
        output_formatted_text(data, text_formatter)
    else:
        output_text(data)
