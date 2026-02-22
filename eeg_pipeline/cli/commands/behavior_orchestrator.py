"""Execution orchestrator for behavior analysis CLI command."""

from __future__ import annotations

import argparse
from typing import Any, List

from eeg_pipeline.cli.common import create_progress_reporter, resolve_task
from eeg_pipeline.cli.commands.behavior_config import (
    _build_computation_features,
    _configure_behavior_compute_mode,
)

def run_behavior(args: argparse.Namespace, subjects: List[str], config: Any) -> None:
    """Execute the behavior command."""
    from eeg_pipeline.pipelines.behavior import BehaviorPipeline
    from eeg_pipeline.plotting.orchestration.behavior import visualize_behavior_for_subjects
    from eeg_pipeline.analysis.behavior.orchestration import StageRegistry, config_to_stage_names
    
    # Handle discoverability options first
    if getattr(args, "list_stages", False):
        stages = StageRegistry.list_stages()
        print("\n=== Available Behavior Pipeline Stages ===\n")
        for stage in stages:
            print(f"  {stage['name']}")
            print(f"    Description: {stage['description']}")
            print(f"    Group: {stage['group']}")
            if stage['requires']:
                print(f"    Requires: {', '.join(stage['requires'])}")
            if stage['produces']:
                print(f"    Produces: {', '.join(stage['produces'])}")
            print()
        return
    
    if getattr(args, "dry_run", False):
        # Build a mock pipeline config to resolve stages
        from eeg_pipeline.pipelines.behavior import BehaviorPipelineConfig
        pipeline_config = BehaviorPipelineConfig()
        # Apply CLI args to config
        if hasattr(args, "computations") and args.computations:
            for comp in args.computations:
                setattr(pipeline_config, f"run_{comp}", True)
        
        stages = config_to_stage_names(pipeline_config)
        dry_run_result = StageRegistry.dry_run(stages)
        
        print("\n=== Dry Run: Behavior Pipeline ===\n")
        print(f"Requested stages: {', '.join(dry_run_result['requested'])}")
        print(f"Resolved stages ({dry_run_result['n_stages']} total):")
        for i, stage in enumerate(dry_run_result['execution_order'], 1):
            print(f"  {i}. {stage}")
        print(f"\nExpected outputs: {', '.join(dry_run_result['expected_outputs'])}")
        return
    
    categories = getattr(args, "categories", None)
    progress = create_progress_reporter(args)
    task = resolve_task(args.task, config)
    
    if getattr(args, "bids_root", None):
        config.setdefault("paths", {})["bids_root"] = args.bids_root
    if getattr(args, "deriv_root", None):
        config.setdefault("paths", {})["deriv_root"] = args.deriv_root
    
    if args.mode == "compute":
        _configure_behavior_compute_mode(args, config)

        computation_features = _build_computation_features(args)
        
        pipeline = BehaviorPipeline(
            config=config,
            computations=args.computations,
            feature_categories=categories,
            feature_files=getattr(args, "feature_files", None),
            computation_features=computation_features,
        )
        
        pipeline.run_batch(
            subjects=subjects,
            task=task,
            bands=getattr(args, "bands", None),
            validate_only=bool(getattr(args, "validate_only", False)),
            progress=progress,
        )
    elif args.mode == "visualize":
        selected_plots = getattr(args, "plots", None)
        run_all_plots = bool(getattr(args, "all_plots", False))
        skip_scatter = bool(getattr(args, "skip_scatter", False))

        if selected_plots is not None:
            visualize_categories = None
            plots = selected_plots
        elif skip_scatter:
            visualize_categories = None
            plots = [
                "psychometrics",
                "temporal_topomaps",
                "dose_response",
            ]
        elif run_all_plots:
            visualize_categories = None
            plots = []
        else:
            visualize_categories = categories
            plots = None

        visualize_behavior_for_subjects(
            subjects=subjects,
            task=task,
            config=config,
            visualize_categories=visualize_categories,
            plots=plots,
        )
