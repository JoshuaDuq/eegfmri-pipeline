"""
Common CLI Utilities
====================

Shared argument parsing utilities and helper functions for CLI subcommands.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, List, Optional


DEFAULT_TASK_KEY = "project.task"
MIN_SUBJECTS_KEY = "analysis.min_subjects_for_group"
MIN_SUBJECTS_FOR_DECODING = 2


def add_common_subject_args(parser: argparse.ArgumentParser) -> None:
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
    parser.add_argument(
        "--task", "-t", type=str, default=None,
        help="Task label (default from config)"
    )


def resolve_task(task: Optional[str], config: Any) -> str:
    resolved = task or config.get(DEFAULT_TASK_KEY)
    if resolved is None:
        raise ValueError(f"Missing required config value: {DEFAULT_TASK_KEY}")
    return resolved


def validate_subjects_not_empty(subjects: List[str], operation: str) -> None:
    if not subjects:
        raise ValueError(f"No subjects specified for {operation}")


def validate_min_subjects(
    subjects: List[str],
    min_count: int,
    operation: str
) -> None:
    if len(subjects) < min_count:
        raise ValueError(
            f"{operation} requires at least {min_count} subjects, "
            f"got {len(subjects)}"
        )


def get_deriv_root(config: Any) -> Path:
    return Path(config.deriv_root)
