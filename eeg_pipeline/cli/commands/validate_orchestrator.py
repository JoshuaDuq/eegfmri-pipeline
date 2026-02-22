"""Execution orchestrator for validate CLI command."""

from __future__ import annotations

import argparse
import json as json_module
from typing import Any, Dict, List

from eeg_pipeline.cli.common import resolve_task
from eeg_pipeline.cli.commands.validate_checks import (
    _collect_subjects_to_validate,
    _output_json_report,
    _output_text_report,
    _should_validate_mode,
    _validate_behavior,
    _validate_bids,
    _validate_epochs,
    _validate_features,
    _validate_structure,
)


def run_validate(args: argparse.Namespace, subjects: List[str], config: Any) -> None:
    """Execute the validate command."""
    from eeg_pipeline.infra.paths import resolve_deriv_root

    task = resolve_task(args.task, config)
    deriv_root = resolve_deriv_root(config=config)

    subjects_to_validate = _collect_subjects_to_validate(
        args.subjects,
        deriv_root,
        task,
        config,
    )

    if not subjects_to_validate:
        if args.output_json:
            print(
                json_module.dumps(
                    {
                        "status": "no_subjects",
                        "issues": [],
                        "warnings": [],
                    }
                )
            )
        else:
            print("No subjects found to validate")
        return

    issues: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []
    passed: List[str] = []

    if _should_validate_mode(args.mode, "quick"):
        _validate_structure(deriv_root, issues, warnings, passed)

    if _should_validate_mode(args.mode, "epochs"):
        _validate_epochs(deriv_root, subjects_to_validate, issues, warnings, passed)

    if _should_validate_mode(args.mode, "features"):
        _validate_features(deriv_root, subjects_to_validate, issues, warnings, passed)

    if _should_validate_mode(args.mode, "behavior"):
        _validate_behavior(deriv_root, subjects_to_validate, issues, warnings, passed)

    if _should_validate_mode(args.mode, "bids"):
        _validate_bids(config, issues, warnings, passed)

    if args.output_json:
        _output_json_report(subjects_to_validate, issues, warnings, passed)
    else:
        _output_text_report(args.mode, subjects_to_validate, issues, warnings, passed)
