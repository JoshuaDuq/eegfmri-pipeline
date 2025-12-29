"""Info and discovery CLI command."""

from __future__ import annotations

import argparse
import json as json_module
from typing import Any, List

from eeg_pipeline.cli.common import add_task_arg, resolve_task
from eeg_pipeline.cli.commands.base import detect_available_bands, detect_feature_availability, _empty_feature_availability


def setup_info(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """Configure the info command parser."""
    parser = subparsers.add_parser(
        "info",
        help="Discovery and status: list subjects, features, config",
        description="Show information about subjects, features, and configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "mode",
        choices=["subjects", "features", "config", "version"],
        help="What to show: subjects, features for a subject, config summary, or version",
    )
    parser.add_argument(
        "target",
        nargs="?",
        default=None,
        help="Subject ID (for features mode) or config key (for config mode)",
    )
    add_task_arg(parser)
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show processing status for each subject (subjects mode only)",
    )
    parser.add_argument(
        "--source",
        choices=["bids", "epochs", "features", "source_data", "all"],
        default="epochs",
        help="Discovery source for subjects (default: epochs)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="output_json",
        help="Output in JSON format",
    )
    parser.add_argument(
        "--keys",
        nargs="*",
        default=None,
        help="Config keys to fetch (config mode only)",
    )
    return parser


def run_info(args: argparse.Namespace, subjects: List[str], config: Any) -> None:
    """Execute the info command."""
    from eeg_pipeline.utils.data.subjects import (
        get_available_subjects,
        _collect_subjects_from_bids,
        _collect_subjects_from_derivatives_epochs,
        _collect_subjects_from_features,
        get_epoch_metadata,
    )
    from eeg_pipeline.utils.data.feature_io import load_feature_bundle
    from eeg_pipeline.infra.paths import resolve_deriv_root, deriv_features_path, deriv_stats_path
    from eeg_pipeline.infra.logging import get_logger
    
    # Suppress logging when JSON output requested to keep output clean for parsing
    if args.output_json:
        import logging
        logging.getLogger("mne").setLevel(logging.ERROR)
        logging.getLogger("nilearn").setLevel(logging.ERROR)
        
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.CRITICAL)  # Suppress all but critical
    else:
        logger = get_logger(__name__)
    
    task = resolve_task(args.task, config)
    deriv_root = resolve_deriv_root(config=config)
    
    if args.mode == "subjects":
        if args.source == "all":
            sources = ["bids", "derivatives_epochs", "features", "source_data"]
        elif args.source == "bids":
            sources = ["bids"]
        elif args.source == "features":
            sources = ["features"]
        elif args.source == "source_data":
            sources = ["source_data"]
        else:
            sources = ["derivatives_epochs"]
        
        source_map = {
            "bids": _collect_subjects_from_bids,
            "derivatives_epochs": lambda: _collect_subjects_from_derivatives_epochs(deriv_root, task, config),
            "features": lambda: _collect_subjects_from_features(deriv_root),
        }
        
        discovered = get_available_subjects(
            config=config,
            deriv_root=deriv_root,
            task=task,
            discovery_sources=sources,
            logger=logger,
        )
        
        if args.status:
            epochs_subjects = set(_collect_subjects_from_derivatives_epochs(deriv_root, task, config))
            features_subjects = set(_collect_subjects_from_features(deriv_root))
            
            results = []
            for subj in discovered:
                status = {
                    "subject": f"sub-{subj}",
                    "epochs": subj in epochs_subjects,
                    "features": subj in features_subjects,
                }
                results.append(status)
            
            if args.output_json:
                json_results = []
                
                # Try to get epoch metadata from the first valid subject to show generic bounds
                global_epoch_metadata = {}
                for subj in discovered:
                    if subj in epochs_subjects:
                        global_epoch_metadata = get_epoch_metadata(subj, task, deriv_root, config=config)
                        if global_epoch_metadata:
                            break

                for r in results:
                    subj_id = r["subject"].replace("sub-", "")
                    available_bands = []
                    has_stats = False
                    
                    if r["features"] or r["epochs"]:
                        features_dir = deriv_features_path(deriv_root, subj_id)
                        # Even if features dir doesn't exist, we might have stats
                        feature_availability = detect_feature_availability(features_dir)
                        if features_dir.exists():
                            available_bands = detect_available_bands(features_dir)
                    else:
                        # No features or epochs - return explicit unavailable status for all
                        feature_availability = _empty_feature_availability()

                    stats_dir = deriv_stats_path(deriv_root, subj_id)
                    if stats_dir.exists():
                        for pattern in ("*.tsv", "*.npz", "*.csv", "*.json"):
                            if any(stats_dir.glob(pattern)):
                                has_stats = True
                                break
                    
                    # Get metadata for this subject if available
                    metadata = {}
                    if r["epochs"]:
                        metadata = get_epoch_metadata(subj_id, task, deriv_root, config=config)
                        # If subject metadata failed, fallback to global first successful one
                        if not metadata and global_epoch_metadata:
                            metadata = global_epoch_metadata

                    json_results.append({
                        "id": subj_id,
                        "has_epochs": r["epochs"],
                        "has_features": r["features"],
                        "has_stats": has_stats,
                        "epoch_metadata": metadata,
                        "available_bands": available_bands,
                        "feature_availability": feature_availability,
                    })
                print(json_module.dumps({"subjects": json_results, "count": len(json_results)}, indent=2))
            else:
                for r in results:
                    epoch_mark = "x" if r["epochs"] else " "
                    feat_mark = "x" if r["features"] else " "
                    print(f"{r['subject']}  [{epoch_mark}]epochs  [{feat_mark}]features")
                print(f"\nTotal: {len(results)} subjects")
        else:
            if args.output_json:
                json_results = [{"id": s, "has_epochs": False, "has_features": False} for s in discovered]
                print(json_module.dumps({"subjects": json_results, "count": len(discovered)}, indent=2))
            else:
                for subj in discovered:
                    print(f"sub-{subj}")
                print(f"\nTotal: {len(discovered)} subjects")
    
    elif args.mode == "features":
        if not args.target:
            print("Error: subject ID required for features mode")
            print("Usage: eeg-pipeline info features <SUBJECT_ID>")
            return
        
        subject = args.target.replace("sub-", "")
        features_dir = deriv_features_path(deriv_root, subject)
        
        if not features_dir.exists():
            print(f"No features directory found for sub-{subject}")
            return
        
        feature_files = list(features_dir.glob("features_*.tsv"))
        results = []
        
        for fpath in sorted(feature_files):
            import pandas as pd
            try:
                df = pd.read_csv(fpath, sep="\t", nrows=1)
                n_cols = len(df.columns)
                results.append({
                    "file": fpath.name,
                    "columns": n_cols,
                })
            except Exception:
                results.append({"file": fpath.name, "columns": "error"})
        
        if args.output_json:
            print(json_module.dumps({"subject": f"sub-{subject}", "features": results}))
        else:
            print(f"Features for sub-{subject}:")
            for r in results:
                print(f"  {r['file']}: {r['columns']} columns")
            print(f"\nTotal: {len(results)} feature files")
    
    elif args.mode == "config":
        if args.keys:
            values = {key: config.get(key) for key in args.keys}
            if args.output_json:
                print(json_module.dumps(values, indent=2))
            else:
                for key, value in values.items():
                    print(f"{key}: {value}")
        elif args.target:
            value = config.get(args.target)
            if args.output_json:
                print(json_module.dumps({args.target: value}))
            else:
                print(f"{args.target}: {value}")
        else:
            summary = {
                "bids_root": str(config.bids_root) if hasattr(config, "bids_root") else None,
                "deriv_root": str(deriv_root),
                "source_root": config.get("paths.source_data"),
                "task": task,
                "preprocessing_n_jobs": config.get("preprocessing.n_jobs", 1),
                "n_subjects": len(config.subjects) if hasattr(config, "subjects") and config.subjects else 0,
            }
            if args.output_json:
                print(json_module.dumps(summary, indent=2))
            else:
                print("Configuration Summary:")
                for k, v in summary.items():
                    print(f"  {k}: {v}")
    
    elif args.mode == "version":
        import eeg_pipeline
        version = getattr(eeg_pipeline, "__version__", "unknown")
        if args.output_json:
            print(json_module.dumps({"version": version}))
        else:
            print(f"eeg-pipeline version: {version}")
