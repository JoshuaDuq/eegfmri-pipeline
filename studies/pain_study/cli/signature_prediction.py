"""CLI command for Study 1 signature prediction."""

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
from eeg_pipeline.utils.config.roots import resolve_fmri_bids_root
from eeg_pipeline.utils.config.overrides import apply_set_overrides
from studies.pain_study.study1 import SignaturePredictionRunner
from studies.pain_study.study1.config import apply_study1_config_defaults


def _normalize_subject_label(subject: str) -> str:
    label = str(subject).strip()
    if not label:
        raise ValueError("Subject identifiers must be non-empty.")
    return label if label.startswith("sub-") else f"sub-{label}"


def _resolve_all_subjects_for_study1(
    *,
    mode: str,
    subjects: list[str],
    config: Any,
) -> list[str]:
    if mode != "prepare-targets":
        return []

    bids_fmri_root = resolve_fmri_bids_root(config, task_is_rest=False)
    eligible_subjects: list[str] = []
    excluded_subjects: list[str] = []
    for subject in subjects:
        subject_label = _normalize_subject_label(subject)
        if (bids_fmri_root / subject_label / "func").is_dir():
            eligible_subjects.append(subject)
        else:
            excluded_subjects.append(subject_label)

    if not eligible_subjects:
        raise FileNotFoundError(
            "Study 1 --all-subjects found no subjects with fMRI func directories "
            f"under {bids_fmri_root}."
        )
    if excluded_subjects:
        logging.info(
            "Study 1 --all-subjects excluded %d EEG-only subject(s): %s",
            len(excluded_subjects),
            ", ".join(sorted(excluded_subjects)),
        )
    return eligible_subjects


def _resolve_subjects_for_study1(
    *,
    args: argparse.Namespace,
    subjects: list[str],
    config: Any,
) -> list[str]:
    if not bool(getattr(args, "all_subjects", False)):
        return subjects
    return _resolve_all_subjects_for_study1(
        mode=str(args.mode),
        subjects=subjects,
        config=config,
    )


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
    mode = str(args.mode)
    resolved_subjects = _resolve_subjects_for_study1(
        args=args,
        subjects=list(subjects),
        config=config,
    )
    if bool(getattr(args, "dry_run", False)):
        logging.info(
            "[dry-run] Study 1 signature-prediction would run mode=%s, task=%s, subjects=%s",
            mode,
            task,
            resolved_subjects,
        )
        return

    runner = SignaturePredictionRunner(config=config)
    runner.run(
        mode=mode,
        subjects=resolved_subjects,
        task=task,
    )


__all__ = ["run_signature_prediction", "setup_signature_prediction"]
