"""Execution orchestrator for features extraction CLI command."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, List

from eeg_pipeline.cli.common import create_progress_reporter, resolve_task
from eeg_pipeline.cli.commands.features_helpers import _apply_feature_config_overrides
from eeg_pipeline.utils.config.overrides import apply_set_overrides

def run_features(args: argparse.Namespace, subjects: List[str], config: Any) -> None:
    """Execute the features command."""
    import sys
    from eeg_pipeline.pipelines.features import FeaturePipeline
    from eeg_pipeline.plotting.orchestration.features import visualize_features_for_subjects
    
    # Capture CLI command for reproducibility (stored in extraction config)
    cli_command = " ".join(sys.argv)
    
    categories = getattr(args, "categories", None)
    progress = create_progress_reporter(args)
    
    if getattr(args, "bids_root", None):
        config.setdefault("paths", {})["bids_root"] = args.bids_root
    if getattr(args, "bids_rest_root", None):
        config.setdefault("paths", {})["bids_rest_root"] = args.bids_rest_root
    if getattr(args, "deriv_root", None):
        config.setdefault("paths", {})["deriv_root"] = args.deriv_root
    if getattr(args, "deriv_rest_root", None):
        config.setdefault("paths", {})["deriv_rest_root"] = args.deriv_rest_root
    if getattr(args, "freesurfer_dir", None):
        config.setdefault("paths", {})["freesurfer_dir"] = args.freesurfer_dir
    
    if args.mode == "compute":
        _apply_feature_config_overrides(args, config)
        apply_set_overrides(config, getattr(args, "set_overrides", None))
        task = resolve_task(args.task, config)

        time_ranges = []
        if getattr(args, "time_range", None):
            for name, tmin, tmax in args.time_range:
                time_ranges.append({
                    "name": name,
                    "tmin": float(tmin) if tmin.lower() != "none" and tmin != "" else None,
                    "tmax": float(tmax) if tmax.lower() != "none" and tmax != "" else None,
                })
        
        pipeline = FeaturePipeline(config=config)
        ledger = pipeline.run_batch(
            subjects=subjects,
            task=task,
            feature_categories=categories,
            bands=getattr(args, "bands", None),
            spatial_modes=getattr(args, "spatial", None),
            tmin=getattr(args, "tmin", None),
            tmax=getattr(args, "tmax", None),
            time_ranges=time_ranges or None,
            aggregation_method=getattr(args, "aggregation_method", "mean"),
            fixed_templates_path=Path(args.fixed_templates_path) if getattr(args, "fixed_templates_path", None) else None,
            progress=progress,
            cli_command=cli_command,
        )
        failed_subjects = [
            str(entry.get("subject"))
            for entry in (ledger or [])
            if str(entry.get("status", "")).strip().lower() == "failed"
        ]
        if failed_subjects:
            raise RuntimeError(
                f"{len(failed_subjects)}/{len(subjects)} subjects failed; see batch ledger for details."
            )
    elif args.mode == "visualize":
        apply_set_overrides(config, getattr(args, "set_overrides", None))
        task = resolve_task(args.task, config)
        visualize_features_for_subjects(
            subjects=subjects,
            task=task,
            config=config,
            visualize_categories=categories,
        )
