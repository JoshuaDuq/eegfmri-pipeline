"""Validation checks and report helpers for validate CLI command."""

from __future__ import annotations

import json as json_module
from pathlib import Path
from typing import Any, Dict, List, Optional

import mne
import pandas as pd

_MAX_SUBJECTS_TO_VALIDATE = 10
_MAX_ISSUES_TO_DISPLAY = 10
_MAX_WARNINGS_TO_DISPLAY = 10
_MAX_ERROR_MESSAGE_LENGTH = 50
_MAX_READ_ERROR_LENGTH = 60


def _collect_subjects_to_validate(
    subjects_arg: Optional[List[str]],
    deriv_root: Path,
    task: str,
    config: Any,
) -> List[str]:
    """Collect subjects to validate from arguments or discover from filesystem."""
    from eeg_pipeline.utils.data.subjects import (
        _collect_subjects_from_derivatives_epochs,
        _collect_subjects_from_features,
    )

    if subjects_arg:
        return [subject.replace("sub-", "") for subject in subjects_arg]

    epochs_subjects = set(_collect_subjects_from_derivatives_epochs(deriv_root, task, config))
    features_subjects = set(_collect_subjects_from_features(deriv_root))
    return list(epochs_subjects | features_subjects)


def _validate_table_schema(
    path: Path,
    *,
    required: List[str],
    any_of: Optional[List[str]] = None,
) -> Optional[str]:
    """Validate table has required columns (TSV or parquet)."""
    try:
        if path.suffix == ".parquet":
            from eeg_pipeline.infra.tsv import read_parquet

            df = read_parquet(path)
            df = df.head(5) if df is not None else pd.DataFrame()
        else:
            df = pd.read_csv(path, sep="\t", nrows=5)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError) as exc:
        error_message = str(exc)[:_MAX_READ_ERROR_LENGTH]
        return f"Read error: {error_message}"

    available_columns = set(df.columns.astype(str).tolist())
    missing_required = [col for col in required if col not in available_columns]

    if missing_required:
        return f"Missing columns: {', '.join(missing_required)}"

    if any_of:
        has_any_column = any(col in available_columns for col in any_of)
        if not has_any_column:
            return f"Missing one of: {', '.join(any_of)}"

    return None


def _validate_structure(
    deriv_root: Path,
    issues: List[Dict[str, str]],
    warnings: List[Dict[str, str]],
    passed: List[str],
) -> None:
    """Validate basic directory structure."""
    epochs_dir = deriv_root / "epochs"
    if epochs_dir.exists():
        passed.append("Epochs directory exists")
    else:
        issues.append({"type": "structure", "message": "Epochs directory not found"})

    features_root = deriv_root / "features"
    if features_root.exists():
        passed.append("Features directory exists")
    else:
        warnings.append(
            {
                "type": "structure",
                "message": "No features directory (run features compute first)",
            }
        )


def _validate_epochs(
    deriv_root: Path,
    subjects: List[str],
    issues: List[Dict[str, Any]],
    warnings: List[Dict[str, Any]],
    passed: List[str],
) -> None:
    """Validate epoch files for subjects."""
    subjects_to_check = subjects[:_MAX_SUBJECTS_TO_VALIDATE]

    for subject in subjects_to_check:
        subject_epochs_dir = deriv_root / "epochs" / f"sub-{subject}"

        if not subject_epochs_dir.exists():
            warnings.append(
                {
                    "type": "epochs",
                    "subject": subject,
                    "message": "No epochs directory",
                }
            )
            continue

        epoch_files = list(subject_epochs_dir.glob("*-epo.fif"))
        if not epoch_files:
            warnings.append(
                {
                    "type": "epochs",
                    "subject": subject,
                    "message": "No epoch files found",
                }
            )
            continue

        try:
            epochs = mne.read_epochs(epoch_files[0], preload=False, verbose=False)
            num_epochs = len(epochs)
            passed.append(f"sub-{subject}: {num_epochs} epochs valid")
        except (OSError, ValueError, RuntimeError) as exc:
            error_message = str(exc)[:_MAX_ERROR_MESSAGE_LENGTH]
            issues.append(
                {
                    "type": "epochs",
                    "subject": subject,
                    "message": f"Corrupt epoch file: {error_message}",
                }
            )


def _validate_features(
    deriv_root: Path,
    subjects: List[str],
    issues: List[Dict[str, Any]],
    warnings: List[Dict[str, Any]],
    passed: List[str],
) -> None:
    """Validate feature files for subjects."""
    from eeg_pipeline.infra.paths import deriv_features_path

    subjects_to_check = subjects[:_MAX_SUBJECTS_TO_VALIDATE]

    for subject in subjects_to_check:
        features_dir = deriv_features_path(deriv_root, subject)
        if not features_dir.exists():
            continue

        tsv_files = list(features_dir.glob("features_*.tsv"))
        if not tsv_files:
            warnings.append(
                {
                    "type": "features",
                    "subject": subject,
                    "message": "No feature files",
                }
            )
            continue

        for tsv_file in tsv_files:
            try:
                df = pd.read_csv(tsv_file, sep="\t", nrows=5)

                has_epoch_or_trial = "epoch" in df.columns or "trial" in df.columns
                if not has_epoch_or_trial:
                    warnings.append(
                        {
                            "type": "features",
                            "subject": subject,
                            "file": tsv_file.name,
                            "message": "Missing epoch/trial column",
                        }
                    )

                columns_with_nan = df.columns[df.isna().any()].tolist()
                if columns_with_nan:
                    nan_columns_preview = ", ".join(columns_with_nan[:3])
                    warnings.append(
                        {
                            "type": "features",
                            "subject": subject,
                            "file": tsv_file.name,
                            "message": f"NaN in columns: {nan_columns_preview}",
                        }
                    )
                else:
                    passed.append(f"sub-{subject}: {tsv_file.name} valid")

            except (pd.errors.EmptyDataError, pd.errors.ParserError, OSError) as exc:
                error_message = str(exc)[:_MAX_READ_ERROR_LENGTH]
                issues.append(
                    {
                        "type": "features",
                        "subject": subject,
                        "file": tsv_file.name,
                        "message": f"Read error: {error_message}",
                    }
                )


def _validate_behavior(
    deriv_root: Path,
    subjects: List[str],
    issues: List[Dict[str, Any]],
    warnings: List[Dict[str, Any]],
    passed: List[str],
) -> None:
    """Validate behavior analysis files for subjects."""
    from eeg_pipeline.infra.paths import deriv_stats_path

    behavior_checks = [
        (
            "correlations",
            ["correlations*/*/correlations*.tsv", "correlations*/*/correlations*.parquet"],
            ["feature", "target", "p_primary"],
            ["r_primary", "r"],
        ),
        (
            "pain_sensitivity",
            ["pain_sensitivity*/*/pain_sensitivity*.tsv", "pain_sensitivity*/*/pain_sensitivity*.parquet"],
            ["feature", "p_primary"],
            None,
        ),
        (
            "regression",
            [
                "trialwise_regression*/*/regression_feature_effects*.tsv",
                "trialwise_regression*/*/regression_feature_effects*.parquet",
            ],
            ["feature", "target", "beta_feature", "p_primary"],
            None,
        ),
        (
            "models",
            [
                "feature_models*/*/models_feature_effects*.tsv",
                "feature_models*/*/models_feature_effects*.parquet",
            ],
            ["feature", "target", "model_family", "beta_feature", "p_primary"],
            None,
        ),
        (
            "condition_effects",
            ["condition_effects*/*/condition_effects*.tsv", "condition_effects*/*/condition_effects*.parquet"],
            ["feature", "p_primary"],
            None,
        ),
    ]

    subjects_to_check = subjects[:_MAX_SUBJECTS_TO_VALIDATE]

    for subject in subjects_to_check:
        stats_dir = deriv_stats_path(deriv_root, subject)
        if not stats_dir.exists():
            warnings.append(
                {
                    "type": "behavior",
                    "subject": subject,
                    "message": "No stats directory (run behavior compute first)",
                }
            )
            continue

        from eeg_pipeline.utils.data.trial_table import (
            discover_trial_table_candidates,
            select_preferred_trial_tables,
        )

        trials_files = select_preferred_trial_tables(discover_trial_table_candidates(stats_dir))
        if trials_files:
            passed.append(f"sub-{subject}: trial table present ({trials_files[0].name})")
        else:
            warnings.append(
                {
                    "type": "behavior",
                    "subject": subject,
                    "message": "Missing canonical trial table (trial table not found)",
                }
            )

        for check_name, patterns, required_cols, any_of_cols in behavior_checks:
            for pattern in patterns:
                for path in stats_dir.glob(pattern):
                    error = _validate_table_schema(path, required=required_cols, any_of=any_of_cols)
                    if error:
                        warnings.append(
                            {
                                "type": "behavior",
                                "subject": subject,
                                "file": path.name,
                                "message": f"{check_name}: {error}",
                            }
                        )
                    else:
                        passed.append(f"sub-{subject}: {path.name} schema OK")


def _validate_bids(
    config: Any,
    issues: List[Dict[str, str]],
    warnings: List[Dict[str, str]],
    passed: List[str],
) -> None:
    """Validate BIDS structure."""
    bids_root = getattr(config, "bids_root", None)

    if not bids_root or not Path(bids_root).exists():
        warnings.append(
            {
                "type": "bids",
                "message": "BIDS root not configured or not found",
            }
        )
        return

    bids_path = Path(bids_root)
    description_file = bids_path / "dataset_description.json"
    if description_file.exists():
        passed.append("dataset_description.json exists")
    else:
        issues.append(
            {
                "type": "bids",
                "message": "Missing dataset_description.json",
            }
        )

    participants_file = bids_path / "participants.tsv"
    if participants_file.exists():
        passed.append("participants.tsv exists")
    else:
        warnings.append({"type": "bids", "message": "Missing participants.tsv"})


def _determine_status(issues: List[Any], warnings: List[Any]) -> str:
    """Determine overall validation status."""
    if issues:
        return "error"
    if warnings:
        return "warning"
    return "ok"


def _format_subject_prefix(issue_or_warning: Dict[str, Any]) -> str:
    """Format subject prefix for display."""
    subject = issue_or_warning.get("subject", "")
    return f"sub-{subject}: " if subject else ""


def _output_json_report(
    subjects: List[str],
    issues: List[Dict[str, Any]],
    warnings: List[Dict[str, Any]],
    passed: List[str],
) -> None:
    """Output validation results as JSON."""
    status = _determine_status(issues, warnings)
    report = {
        "status": status,
        "subjects_checked": len(subjects),
        "issues": issues,
        "warnings": warnings,
        "passed": len(passed),
    }
    print(json_module.dumps(report, indent=2))


def _output_text_report(
    mode: str,
    subjects: List[str],
    issues: List[Dict[str, Any]],
    warnings: List[Dict[str, Any]],
    passed: List[str],
) -> None:
    """Output validation results as formatted text."""
    print("=" * 50)
    print("       DATA VALIDATION REPORT")
    print("=" * 50)
    print()
    print(f"  Mode: {mode}")
    print(f"  Subjects checked: {len(subjects)}")
    print()

    if issues:
        print("  ✗ ERRORS")
        print("  " + "-" * 30)
        issues_to_display = issues[:_MAX_ISSUES_TO_DISPLAY]
        for issue in issues_to_display:
            subject_prefix = _format_subject_prefix(issue)
            print(f"    {subject_prefix}{issue['message']}")
        remaining_issues = len(issues) - _MAX_ISSUES_TO_DISPLAY
        if remaining_issues > 0:
            print(f"    ... and {remaining_issues} more")
        print()

    if warnings:
        print("  ⚠ WARNINGS")
        print("  " + "-" * 30)
        warnings_to_display = warnings[:_MAX_WARNINGS_TO_DISPLAY]
        for warning in warnings_to_display:
            subject_prefix = _format_subject_prefix(warning)
            print(f"    {subject_prefix}{warning['message']}")
        remaining_warnings = len(warnings) - _MAX_WARNINGS_TO_DISPLAY
        if remaining_warnings > 0:
            print(f"    ... and {remaining_warnings} more")
        print()

    print("  SUMMARY")
    print("  " + "-" * 30)
    status = _determine_status(issues, warnings)
    status_text = "✗ FAILED" if status == "error" else ("⚠ WARNINGS" if status == "warning" else "✓ PASSED")
    status_color_code = "\033[91m" if status == "error" else ("\033[93m" if status == "warning" else "\033[92m")
    print(f"  Status: {status_color_code}{status_text}\033[0m")
    print(f"  Passed: {len(passed)}")
    print(f"  Issues: {len(issues)}")
    print(f"  Warnings: {len(warnings)}")
    print()


def _should_validate_mode(mode: str, target_mode: str) -> bool:
    """Check if validation mode should run for given target mode."""
    return mode in [target_mode, "all"]
