"""
Behavior CLI Subcommand
=======================

Parser setup and run function for behavior correlation analysis.
"""

from __future__ import annotations

import argparse
from typing import Any, List

from eeg_pipeline.cli.common import add_common_subject_args, add_task_arg, resolve_task

BEHAVIOR_COMPUTATIONS = [
    "power_roi", "connectivity_roi", "connectivity_heatmaps", "sliding_connectivity",
    "time_frequency", "temporal_correlations", "cluster_test", 
    "precomputed_correlations", "condition_correlations", "exports"
]


def setup_behavior_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "behavior",
        help="Behavior analysis: compute correlations or visualize",
        description="Behavior pipeline: compute correlations or visualize",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "mode",
        choices=["compute", "visualize"],
        help="Pipeline mode"
    )
    
    add_common_subject_args(parser)
    add_task_arg(parser)
    
    compute_group = parser.add_argument_group("Compute mode options")
    compute_group.add_argument(
        "--correlation-method", choices=["spearman", "pearson"], default=None,
        help="Correlation method (default from config)"
    )
    compute_group.add_argument(
        "--bootstrap", type=int, default=0,
        help="Number of bootstrap iterations (default: 0)"
    )
    compute_group.add_argument(
        "--n-perm", type=int, default=None,
        help="Number of permutations (default from config)"
    )
    compute_group.add_argument(
        "--rng-seed", type=int, default=None,
        help="Random seed (default: project.random_state in config)"
    )
    compute_group.add_argument(
        "--computations", nargs="+",
        choices=BEHAVIOR_COMPUTATIONS,
        default=None,
        help=f"Specific behavior computations to run (default: all). Choices: {', '.join(BEHAVIOR_COMPUTATIONS)}"
    )
    
    visualize_group = parser.add_argument_group("Visualize mode options")
    plot_group = visualize_group.add_mutually_exclusive_group()
    plot_group.add_argument(
        "--plots", nargs="+",
        metavar="PLOT",
        help="Specific plot types to generate"
    )
    plot_group.add_argument(
        "--all-plots", action="store_true",
        help="Generate all available plots (default behavior)"
    )
    visualize_group.add_argument(
        "--skip-scatter", action="store_true",
        help="Skip ROI scatter plot generation"
    )


def run_behavior(args, subjects: List[str], config: Any) -> None:
    from eeg_pipeline.pipelines.behavior import compute_behavior_correlations_for_subjects
    from eeg_pipeline.plotting.behavioral.viz import visualize_behavior_for_subjects
    
    if args.mode == "compute":
        rng_seed = args.rng_seed if args.rng_seed is not None else config.get("project.random_state")
        compute_behavior_correlations_for_subjects(
            subjects=subjects,
            task=args.task,
            correlation_method=args.correlation_method,
            bootstrap=args.bootstrap,
            n_perm=args.n_perm,
            rng_seed=rng_seed,
            computations=args.computations,
        )
    elif args.mode == "visualize":
        task = resolve_task(args.task, config)
        visualize_behavior_for_subjects(
            subjects=subjects,
            task=task,
            config=config,
            scatter_only=False,
            temporal_only=False,
        )
    else:
        raise ValueError(f"Invalid mode: {args.mode}. Must be 'compute' or 'visualize'.")
