"""Execution orchestrator for stats CLI command."""

from __future__ import annotations

import argparse
from typing import Any, List

from eeg_pipeline.cli.common import resolve_task
from eeg_pipeline.cli.commands.stats_helpers import (
    _collect_all_subjects,
    _collect_fmri_analysis_subjects,
    _handle_features_mode,
    _handle_storage_mode,
    _handle_subjects_mode,
    _handle_summary_mode,
    _handle_timeline_mode,
)
from eeg_pipeline.infra.paths import resolve_deriv_root


def run_stats(args: argparse.Namespace, subjects: List[str], config: Any) -> None:
    """Execute the stats command."""
    task = resolve_task(args.task, config)
    deriv_root = resolve_deriv_root(config=config)

    bids_subjects, epochs_subjects, features_subjects, eeg_prep_subjects, fmri_prep_subjects = _collect_all_subjects(
        deriv_root,
        task,
        config,
    )

    fmri_first_level, fmri_beta_series, fmri_lss = _collect_fmri_analysis_subjects(deriv_root)

    if args.mode == "summary":
        _handle_summary_mode(
            args,
            task,
            deriv_root,
            bids_subjects,
            epochs_subjects,
            features_subjects,
            eeg_prep_subjects,
            fmri_prep_subjects,
            fmri_first_level,
            fmri_beta_series,
            fmri_lss,
        )
    elif args.mode == "subjects":
        _handle_subjects_mode(args, bids_subjects, epochs_subjects, features_subjects)
    elif args.mode == "features":
        _handle_features_mode(args, deriv_root, features_subjects)
    elif args.mode == "storage":
        _handle_storage_mode(args, deriv_root, features_subjects)
    elif args.mode == "timeline":
        _handle_timeline_mode(args, deriv_root, features_subjects)
