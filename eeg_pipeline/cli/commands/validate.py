"""Data validation CLI command."""

from __future__ import annotations

import argparse
import json as json_module
from pathlib import Path
from typing import Any, List

from eeg_pipeline.cli.common import add_task_arg, resolve_task


def setup_validate(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """Configure the validate command parser."""
    parser = subparsers.add_parser(
        "validate",
        help="Validate data integrity: epochs, features, BIDS",
        description="Check data files for integrity and consistency issues",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "mode",
        choices=["all", "epochs", "features", "behavior", "bids", "quick"],
        nargs="?",
        default="quick",
        help="What to validate (default: quick)",
    )
    add_task_arg(parser)
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=None,
        help="Specific subjects to validate",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="output_json",
        help="Output in JSON format",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Attempt to fix minor issues (e.g., missing metadata)",
    )
    return parser


def run_validate(args: argparse.Namespace, subjects: List[str], config: Any) -> None:
    """Execute the validate command."""
    from eeg_pipeline.utils.data.subjects import (
        _collect_subjects_from_derivatives_epochs,
        _collect_subjects_from_features,
    )
    from eeg_pipeline.infra.paths import resolve_deriv_root, deriv_features_path, deriv_stats_path
    
    task = resolve_task(args.task, config)
    deriv_root = resolve_deriv_root(config=config)
    
    issues = []
    warnings = []
    passed = []
    
    if args.subjects:
        validate_subjects = [s.replace("sub-", "") for s in args.subjects]
    else:
        epochs_subjects = set(_collect_subjects_from_derivatives_epochs(deriv_root, task, config))
        features_subjects = set(_collect_subjects_from_features(deriv_root))
        validate_subjects = list(epochs_subjects | features_subjects)
    
    if not validate_subjects:
        if args.output_json:
            print(json_module.dumps({"status": "no_subjects", "issues": [], "warnings": []}))
        else:
            print("No subjects found to validate")
        return
    
    if args.mode in ["quick", "all"]:
        epochs_dir = deriv_root / "epochs"
        if not epochs_dir.exists():
            issues.append({"type": "structure", "message": "Epochs directory not found"})
        else:
            passed.append("Epochs directory exists")
        
        features_root = deriv_root / "features"
        if not features_root.exists():
            warnings.append({"type": "structure", "message": "No features directory (run features compute first)"})
        else:
            passed.append("Features directory exists")
    
    if args.mode in ["epochs", "all"]:
        import mne
        
        for subj in validate_subjects[:10]:
            subj_epochs_dir = deriv_root / "epochs" / f"sub-{subj}"
            if not subj_epochs_dir.exists():
                warnings.append({"type": "epochs", "subject": subj, "message": "No epochs directory"})
                continue
            
            epoch_files = list(subj_epochs_dir.glob("*-epo.fif"))
            if not epoch_files:
                warnings.append({"type": "epochs", "subject": subj, "message": "No epoch files found"})
                continue
            
            try:
                epochs = mne.read_epochs(epoch_files[0], preload=False, verbose=False)
                n_epochs = len(epochs)
                passed.append(f"sub-{subj}: {n_epochs} epochs valid")
            except Exception as e:
                issues.append({"type": "epochs", "subject": subj, "message": f"Corrupt epoch file: {str(e)[:50]}"})
    
    if args.mode in ["features", "all"]:
        import pandas as pd
        
        for subj in validate_subjects[:10]:
            features_dir = deriv_features_path(deriv_root, subj)
            if not features_dir.exists():
                continue
            
            tsv_files = list(features_dir.glob("features_*.tsv"))
            if not tsv_files:
                warnings.append({"type": "features", "subject": subj, "message": "No feature files"})
                continue
            
            for tsv in tsv_files:
                try:
                    df = pd.read_csv(tsv, sep="\t", nrows=5)
                    
                    if "epoch" not in df.columns and "trial" not in df.columns:
                        warnings.append({
                            "type": "features",
                            "subject": subj,
                            "file": tsv.name,
                            "message": "Missing epoch/trial column"
                        })
                    
                    nan_cols = df.columns[df.isna().any()].tolist()
                    if nan_cols:
                        warnings.append({
                            "type": "features",
                            "subject": subj,
                            "file": tsv.name,
                            "message": f"NaN in columns: {', '.join(nan_cols[:3])}"
                        })
                    else:
                        passed.append(f"sub-{subj}: {tsv.name} valid")
                        
                except Exception as e:
                    issues.append({
                        "type": "features",
                        "subject": subj,
                        "file": tsv.name,
                        "message": f"Read error: {str(e)[:40]}"
                    })

    if args.mode in ["behavior", "all"]:
        import pandas as pd

        def _validate_tsv_schema(path: Path, *, required: List[str], any_of: Optional[List[str]] = None) -> Optional[str]:
            try:
                df = pd.read_csv(path, sep="\t", nrows=5)
            except Exception as exc:
                return f"Read error: {str(exc)[:60]}"
            cols = set(df.columns.astype(str).tolist())
            missing = [c for c in required if c not in cols]
            if missing:
                return f"Missing columns: {', '.join(missing)}"
            if any_of:
                if not any(c in cols for c in any_of):
                    return f"Missing one of: {', '.join(any_of)}"
            return None

        behavior_checks = [
            ("correlations", "correlations*.tsv", ["feature", "target", "p_primary"], ["r_primary", "r"]),
            ("pain_sensitivity", "pain_sensitivity*.tsv", ["feature", "p_primary"], None),
            ("regression", "regression_feature_effects*.tsv", ["feature", "target", "beta_feature", "p_primary"], None),
            ("models", "models_feature_effects*.tsv", ["feature", "target", "model_family", "beta_feature", "p_primary"], None),
            ("condition_effects", "condition_effects*.tsv", ["feature", "p_primary"], None),
        ]

        for subj in validate_subjects[:10]:
            stats_dir = deriv_stats_path(deriv_root, subj)
            if not stats_dir.exists():
                warnings.append({"type": "behavior", "subject": subj, "message": "No stats directory (run behavior compute first)"})
                continue

            trials_files = list(stats_dir.glob("trials*.parquet"))
            if not trials_files:
                warnings.append({"type": "behavior", "subject": subj, "message": "Missing trials*.parquet (trial table not found)"})
            else:
                passed.append(f"sub-{subj}: trial table present ({trials_files[0].name})")

            for check_name, pattern, required_cols, any_of_cols in behavior_checks:
                for path in stats_dir.glob(pattern):
                    err = _validate_tsv_schema(path, required=required_cols, any_of=any_of_cols)
                    if err:
                        warnings.append({
                            "type": "behavior",
                            "subject": subj,
                            "file": path.name,
                            "message": f"{check_name}: {err}",
                        })
                    else:
                        passed.append(f"sub-{subj}: {path.name} schema OK")
    
    if args.mode in ["bids", "all"]:
        bids_root = config.bids_root if hasattr(config, "bids_root") else None
        
        if bids_root and Path(bids_root).exists():
            desc_file = Path(bids_root) / "dataset_description.json"
            if desc_file.exists():
                passed.append("dataset_description.json exists")
            else:
                issues.append({"type": "bids", "message": "Missing dataset_description.json"})
            
            participants_file = Path(bids_root) / "participants.tsv"
            if participants_file.exists():
                passed.append("participants.tsv exists")
            else:
                warnings.append({"type": "bids", "message": "Missing participants.tsv"})
        else:
            warnings.append({"type": "bids", "message": "BIDS root not configured or not found"})
    
    if args.output_json:
        print(json_module.dumps({
            "status": "error" if issues else ("warning" if warnings else "ok"),
            "subjects_checked": len(validate_subjects),
            "issues": issues,
            "warnings": warnings,
            "passed": len(passed),
        }, indent=2))
    else:
        print("=" * 50)
        print("       DATA VALIDATION REPORT")
        print("=" * 50)
        print()
        print(f"  Mode: {args.mode}")
        print(f"  Subjects checked: {len(validate_subjects)}")
        print()
        
        if issues:
            print("  ✗ ERRORS")
            print("  " + "-" * 30)
            for issue in issues[:10]:
                subj = issue.get("subject", "")
                subj_str = f"sub-{subj}: " if subj else ""
                print(f"    {subj_str}{issue['message']}")
            if len(issues) > 10:
                print(f"    ... and {len(issues) - 10} more")
            print()
        
        if warnings:
            print("  ⚠ WARNINGS")
            print("  " + "-" * 30)
            for warn in warnings[:10]:
                subj = warn.get("subject", "")
                subj_str = f"sub-{subj}: " if subj else ""
                print(f"    {subj_str}{warn['message']}")
            if len(warnings) > 10:
                print(f"    ... and {len(warnings) - 10} more")
            print()
        
        print("  SUMMARY")
        print("  " + "-" * 30)
        summary_status = "✗ FAILED" if issues else ("⚠ WARNINGS" if warnings else "✓ PASSED")
        status_color = "\033[91m" if issues else ("\033[93m" if warnings else "\033[92m")
        print(f"  Status: {status_color}{summary_status}\033[0m")
        print(f"  Passed: {len(passed)}")
        print(f"  Issues: {len(issues)}")
        print(f"  Warnings: {len(warnings)}")
        print()
