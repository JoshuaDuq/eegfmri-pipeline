"""CLI command for trial-wise cortical EEG-BOLD coupling."""

from __future__ import annotations

import argparse
from typing import Any, List

from eeg_pipeline.cli.common import (
    add_common_subject_args,
    add_output_format_args,
    add_path_args,
    add_task_arg,
    create_progress_reporter,
    resolve_task,
)
from studies.pain_study.pipelines.eeg_bold_coupling import (
    EEGBOLDCouplingPipeline,
)
from studies.pain_study.config.eeg_bold_coupling_loader import (
    apply_eeg_bold_coupling_config_defaults,
)
from eeg_pipeline.utils.config.overrides import apply_set_overrides


def setup_coupling(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """Configure the EEG-BOLD coupling parser."""
    parser = subparsers.add_parser(
        "coupling",
        help="Trial-wise cortical EEG-BOLD coupling analysis",
        description=(
            "Run the Study 2 EEG-BOLD coupling pipeline using subject-specific "
            "source localization, trial-wise LSS betas, predefined cortical ROIs, "
            "and group aggregation."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("mode", choices=["compute"], help="Operation to run")
    add_common_subject_args(parser)
    add_task_arg(parser)
    add_output_format_args(parser)
    add_path_args(parser)
    parser.add_argument(
        "--coupling-config",
        type=str,
        default=None,
        help="Path to the EEG-BOLD coupling YAML config",
    )

    parser.add_argument(
        "--bands",
        nargs="+",
        default=None,
        help="Override eeg_bold_coupling.eeg.bands",
    )
    parser.add_argument(
        "--source-method",
        choices=["lcmv", "eloreta"],
        default=None,
        help="Override eeg_bold_coupling.eeg.method",
    )
    parser.add_argument(
        "--source-subjects-dir",
        type=str,
        default=None,
        help="Override eeg_bold_coupling.eeg.subjects_dir",
    )
    parser.add_argument(
        "--selection-column",
        type=str,
        default=None,
        help="Override eeg_bold_coupling.fmri.selection_column",
    )
    parser.add_argument(
        "--selection-values",
        nargs="+",
        default=None,
        help="Override eeg_bold_coupling.fmri.selection_values",
    )
    parser.add_argument(
        "--contrast-name",
        type=str,
        default=None,
        help="Override eeg_bold_coupling.fmri.contrast_name",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override eeg_bold_coupling.output_dir",
    )
    return parser


def _apply_coupling_overrides(args: argparse.Namespace, config: Any) -> None:
    if getattr(args, "bands", None) is not None:
        config["eeg_bold_coupling.eeg.bands"] = list(args.bands)
    if getattr(args, "source_method", None) is not None:
        config["eeg_bold_coupling.eeg.method"] = str(args.source_method)
    if getattr(args, "source_subjects_dir", None) is not None:
        config["eeg_bold_coupling.eeg.subjects_dir"] = str(args.source_subjects_dir)
    if getattr(args, "selection_column", None) is not None:
        config["eeg_bold_coupling.fmri.selection_column"] = str(args.selection_column)
    if getattr(args, "selection_values", None) is not None:
        config["eeg_bold_coupling.fmri.selection_values"] = list(args.selection_values)
    if getattr(args, "contrast_name", None) is not None:
        config["eeg_bold_coupling.fmri.contrast_name"] = str(args.contrast_name)
    if getattr(args, "output_dir", None) is not None:
        config["eeg_bold_coupling.output_dir"] = str(args.output_dir)


def run_coupling(args: argparse.Namespace, subjects: List[str], config: Any) -> None:
    """Execute the EEG-BOLD coupling pipeline."""
    apply_eeg_bold_coupling_config_defaults(
        config,
        config_path=getattr(args, "coupling_config", None),
    )
    _apply_coupling_overrides(args, config)
    apply_set_overrides(config, getattr(args, "set_overrides", None))

    task = resolve_task(getattr(args, "task", None), config)
    progress = create_progress_reporter(args)

    pipeline = EEGBOLDCouplingPipeline(config=config)
    ledger = pipeline.run_batch(
        subjects=subjects,
        task=task,
        progress=progress,
    )
    failed = [
        str(entry.get("subject"))
        for entry in ledger
        if str(entry.get("status", "")).strip().lower() == "failed"
    ]
    if failed:
        raise RuntimeError(
            f"{len(failed)}/{len(subjects)} subjects failed; see batch ledger for details."
        )


__all__ = ["setup_coupling", "run_coupling"]
