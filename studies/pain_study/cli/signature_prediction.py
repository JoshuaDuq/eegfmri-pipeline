"""CLI command for Study 1 signature prediction."""

from __future__ import annotations

import argparse
from typing import Any, List

from eeg_pipeline.cli.common import (
    add_common_subject_args,
    add_output_format_args,
    add_path_args,
    add_task_arg,
    resolve_task,
)
from eeg_pipeline.utils.config.overrides import apply_set_overrides
from studies.pain_study.study1 import SignaturePredictionRunner
from studies.pain_study.study1.config import apply_study1_config_defaults


def setup_signature_prediction(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    """Configure the Study 1 signature prediction parser."""
    parser = subparsers.add_parser(
        "signature-prediction",
        help="Study 1 EEG-to-signature prediction",
        description="Run Study 1 target preparation, feature benchmark, deep regression, or report.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "mode",
        choices=[
            "prepare-targets",
            "prepare-features",
            "feature-benchmark",
            "deep-regression",
            "report",
        ],
        help="Study 1 stage to run",
    )
    add_common_subject_args(parser)
    add_task_arg(parser)
    add_output_format_args(parser)
    add_path_args(parser)
    parser.add_argument(
        "--study1-config",
        type=str,
        default=None,
        help="Path to the Study 1 YAML config.",
    )
    return parser


def run_signature_prediction(
    args: argparse.Namespace,
    subjects: List[str],
    config: Any,
) -> None:
    """Execute a Study 1 stage."""
    apply_study1_config_defaults(config, config_path=getattr(args, "study1_config", None))
    apply_set_overrides(config, getattr(args, "set_overrides", None))
    task = resolve_task(getattr(args, "task", None), config)

    runner = SignaturePredictionRunner(config=config)
    runner.run(
        mode=str(args.mode),
        subjects=list(subjects),
        task=task,
    )


__all__ = ["run_signature_prediction", "setup_signature_prediction"]
