"""
Common CLI Utilities
====================

Shared argument parsing utilities and helper functions for CLI subcommands.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Tuple

from eeg_pipeline.infra.paths import resolve_deriv_root
from eeg_pipeline.cli.progress import create_progress_reporter, ProgressReporter


DEFAULT_TASK_KEY = "project.task"
MIN_SUBJECTS_KEY = "analysis.min_subjects_for_group"
MIN_SUBJECTS_FOR_ML = 2

__all__ = [
    "ProgressReporter",
    "create_progress_reporter",
    "add_common_subject_args",
    "add_task_arg",
    "add_path_args",
    "add_output_format_args",
    "resolve_task",
    "validate_subjects_not_empty",
    "validate_min_subjects",
    "get_deriv_root",
    "apply_arg_overrides",
    "MIN_SUBJECTS_KEY",
    "MIN_SUBJECTS_FOR_ML",
]


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
    parser.add_argument(
        "--set",
        dest="set_overrides",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help=(
            "Override config key at runtime (repeatable). "
            "Example: --set project.task=mytask"
        ),
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


def apply_arg_overrides(
    args: argparse.Namespace,
    config: Any,
    mapping: Sequence[Tuple[str, str, Callable[[Any], Any]]],
) -> None:
    """Apply CLI argument overrides to config from a declarative mapping.

    Each entry is ``(arg_attr, config_key, cast_fn)``. The override is applied
    only when ``getattr(args, arg_attr)`` is not ``None``.
    """
    for arg_attr, config_key, cast_fn in mapping:
        value = getattr(args, arg_attr, None)
        if value is not None:
            config[config_key] = cast_fn(value)
