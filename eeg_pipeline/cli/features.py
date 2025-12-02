"""
Features CLI Subcommand
=======================

Parser setup and run function for feature extraction.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, List

from eeg_pipeline.cli.common import add_common_subject_args, add_task_arg
from eeg_pipeline.context.features import FEATURE_CATEGORIES


def setup_features_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "features",
        help="Features analysis: extract features or visualize",
        description="Features pipeline: extract features or visualize",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "mode",
        choices=["compute", "visualize"],
        help="Pipeline mode"
    )
    
    add_common_subject_args(parser)
    add_task_arg(parser)
    parser.add_argument(
        "--fixed-templates", type=str,
        help="Path to .npz file containing fixed microstate templates"
    )
    parser.add_argument(
        "--feature-categories", nargs="+",
        choices=FEATURE_CATEGORIES,
        default=None,
        help=f"Specific feature categories to compute (default: all). Choices: {', '.join(FEATURE_CATEGORIES)}"
    )


def run_features(args, subjects: List[str], config: Any) -> None:
    from eeg_pipeline.pipelines.features import extract_features_for_subjects
    from eeg_pipeline.plotting.features import visualize_features_for_subjects
    
    if args.mode == "compute":
        extract_features_for_subjects(
            subjects=subjects,
            task=args.task,
            fixed_templates_path=Path(args.fixed_templates) if args.fixed_templates else None,
            feature_categories=args.feature_categories,
        )
    elif args.mode == "visualize":
        visualize_features_for_subjects(subjects=subjects, task=args.task)
    else:
        raise ValueError(f"Invalid mode: {args.mode}. Must be 'compute' or 'visualize'.")
