"""CLI command for Study 2 source interpretation."""

from __future__ import annotations

import argparse
import logging
from typing import Any, List

from eeg_pipeline.cli.common import (
    add_common_subject_args,
    add_output_format_args,
    add_path_args,
    add_task_arg,
    resolve_task,
)
from eeg_pipeline.utils.config.overrides import apply_set_overrides
from studies.pain_study.study2.config import apply_study2_config_defaults
from studies.pain_study.study2.runner import STUDY2_STAGES, Study2Runner


def _stage_modes() -> list[str]:
    return [stage.name for stage in STUDY2_STAGES] + ["all"]


def setup_source_interpretation(
    subparsers: argparse._SubParsersAction,
) -> argparse.ArgumentParser:
    """Configure the Study 2 source-interpretation parser."""
    parser = subparsers.add_parser(
        "source-interpretation",
        help="Study 2 source interpretation of NPS-predictive EEG activity",
        description="Run Study 2 source-interpretation stages or the full sequence.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "mode",
        choices=_stage_modes(),
        help="Study 2 stage to run, or 'all' for the full dependency-ordered sequence",
    )
    add_common_subject_args(parser)
    add_task_arg(parser)
    add_output_format_args(parser)
    add_path_args(parser)
    parser.add_argument(
        "--study2-config",
        type=str,
        default=None,
        help="Path to the Study 2 YAML config.",
    )
    return parser


def run_source_interpretation(
    args: argparse.Namespace,
    subjects: List[str],
    config: Any,
) -> None:
    """Execute a Study 2 source-interpretation stage."""
    apply_study2_config_defaults(config, config_path=getattr(args, "study2_config", None))
    apply_set_overrides(config, getattr(args, "set_overrides", None))
    task = resolve_task(getattr(args, "task", None), config)
    mode = str(args.mode)
    if bool(getattr(args, "dry_run", False)):
        logging.info(
            "[dry-run] Study 2 source-interpretation would run mode=%s, task=%s, subjects=%s",
            mode,
            task,
            subjects,
        )
        return

    runner = Study2Runner(config=config)
    runner.run(mode=mode, subjects=list(subjects), task=task)


__all__ = ["run_source_interpretation", "setup_source_interpretation"]
