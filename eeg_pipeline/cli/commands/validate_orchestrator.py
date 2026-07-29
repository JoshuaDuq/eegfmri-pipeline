"""Execution orchestrator for validate CLI command."""

from __future__ import annotations

import argparse
import json as json_module
from typing import Any, Dict, List

from eeg_pipeline.cli.common import get_deriv_root, resolve_task


def _collect_config_issues(
    config: Any,
    issues: List[Dict[str, Any]],
    warnings: List[Dict[str, Any]],
    passed: List[str],
) -> None:
    """Fold the config coherence report into the validate command's own vocabulary.

    Reported here as well as raised by the pipeline, because this is where someone looks
    *before* committing to a run: the point of the check is to be cheap enough to ask
    first.
    """
    from eeg_pipeline.utils.config.coherence import check_config_coherence

    report = check_config_coherence(config)
    for issue in report.errors:
        issues.append({"type": "config", "message": str(issue)})
    for warning in report.warnings:
        warnings.append({"type": "config", "message": str(warning)})
    if report.ok:
        passed.append("Configuration is self-consistent")


def run_validate(args: argparse.Namespace, subjects: List[str], config: Any) -> None:
    """Execute the validate command."""
    mode = "config" if getattr(args, "config_only", False) else args.mode

    if mode == "config":
        config_issues: List[Dict[str, Any]] = []
        config_warnings: List[Dict[str, Any]] = []
        config_passed: List[str] = []
        _collect_config_issues(config, config_issues, config_warnings, config_passed)

        from eeg_pipeline.cli.commands.validate_checks import (
            _output_json_report,
            _output_text_report,
        )

        if args.output_json:
            _output_json_report([], config_issues, config_warnings, config_passed)
        else:
            _output_text_report(mode, [], config_issues, config_warnings, config_passed)
        return

    task = resolve_task(args.task, config)
    deriv_root = get_deriv_root(config, command="validate")

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
        # Cheapest check there is, and the one most likely to explain what follows.
        _collect_config_issues(config, issues, warnings, passed)
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
