"""Execution orchestrator for behavior analysis CLI command."""

from __future__ import annotations

import argparse
from typing import Any, List

from eeg_pipeline.cli.common import create_progress_reporter, resolve_task
from eeg_pipeline.cli.commands.behavior_config import (
    _build_computation_features,
    _configure_behavior_compute_mode,
)
from eeg_pipeline.utils.config.behavior_loader import apply_behavior_config_defaults
from eeg_pipeline.utils.config.overrides import apply_set_overrides


def run_behavior(args: argparse.Namespace, subjects: List[str], config: Any) -> None:
    """Execute the behavior command."""
    from eeg_pipeline.pipelines.behavior import BehaviorPipeline
    from eeg_pipeline.analysis.behavior.orchestration import StageRegistry, config_to_stage_names

    # Handle discoverability options first
    if getattr(args, "list_stages", False):
        stages = StageRegistry.list_stages()
        print("\n=== Available Behavior Pipeline Stages ===\n")
        for stage in stages:
            print(f"  {stage['name']}")
            print(f"    Description: {stage['description']}")
            print(f"    Group: {stage['group']}")
            if stage["requires"]:
                print(f"    Requires: {', '.join(stage['requires'])}")
            if stage["produces"]:
                print(f"    Produces: {', '.join(stage['produces'])}")
            print()
        return

    if getattr(args, "dry_run", False):
        from eeg_pipeline.pipelines.behavior import BehaviorPipelineConfig

        apply_behavior_config_defaults(config)
        pipeline_config = BehaviorPipelineConfig.from_config(config)
        if hasattr(args, "computations") and args.computations:
            for comp in args.computations:
                setattr(pipeline_config, f"run_{comp}", True)

        stages = config_to_stage_names(pipeline_config)
        dry_run_result = StageRegistry.dry_run(stages)

        print("\n=== Dry Run: Behavior Pipeline ===\n")
        print(f"Requested stages: {', '.join(dry_run_result['requested'])}")
        print(f"Resolved stages ({dry_run_result['n_stages']} total):")
        for i, stage in enumerate(dry_run_result["execution_order"], 1):
            print(f"  {i}. {stage}")
        print(f"\nExpected outputs: {', '.join(dry_run_result['expected_outputs'])}")
        return

    apply_behavior_config_defaults(config)

    categories = getattr(args, "categories", None)
    progress = create_progress_reporter(args)

    if getattr(args, "bids_root", None):
        config.setdefault("paths", {})["bids_root"] = args.bids_root
    if getattr(args, "deriv_root", None):
        config.setdefault("paths", {})["deriv_root"] = args.deriv_root

    if args.mode == "compute":
        _configure_behavior_compute_mode(args, config)
        apply_set_overrides(config, getattr(args, "set_overrides", None))
        task = resolve_task(args.task, config)

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
            progress=progress,
        )
