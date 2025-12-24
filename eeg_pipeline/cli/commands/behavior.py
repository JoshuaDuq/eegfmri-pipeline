"""Behavior analysis CLI command."""

from __future__ import annotations

import argparse
from typing import Any, List

from eeg_pipeline.cli.common import (
    add_common_subject_args,
    add_task_arg,
    add_output_format_args,
    add_path_args,
    resolve_task,
    create_progress_reporter,
)
from eeg_pipeline.cli.commands.base import (
    BEHAVIOR_COMPUTATIONS,
    BEHAVIOR_VISUALIZE_CATEGORIES,
)
from eeg_pipeline.utils.data.feature_discovery import STANDARD_FEATURE_FILES

FEATURE_FILE_CHOICES = list(STANDARD_FEATURE_FILES.keys())


def setup_behavior(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """Configure the behavior command parser."""
    parser = subparsers.add_parser(
        "behavior",
        help="Behavior analysis: compute correlations or visualize",
        description="Behavior pipeline: compute correlations or visualize",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("mode", choices=["compute", "visualize"], help="Pipeline mode")
    add_common_subject_args(parser)
    add_task_arg(parser)
    add_output_format_args(parser)
    parser.add_argument(
        "--categories",
        nargs="+",
        choices=BEHAVIOR_VISUALIZE_CATEGORIES,
        default=None,
        metavar="CATEGORY",
        help="Feature categories to process (e.g., power, connectivity, itpc)",
    )
    
    compute_group = parser.add_argument_group("Compute mode options")
    compute_group.add_argument("--correlation-method", choices=["spearman", "pearson"], default=None)
    compute_group.add_argument("--bootstrap", type=int, default=0)
    compute_group.add_argument("--n-perm", type=int, default=None)
    compute_group.add_argument("--rng-seed", type=int, default=None)
    compute_group.add_argument("--control-temperature", action="store_true", default=None)
    compute_group.add_argument("--no-control-temperature", action="store_false", dest="control_temperature")
    compute_group.add_argument("--control-trial-order", action="store_true", default=None)
    compute_group.add_argument("--no-control-trial-order", action="store_false", dest="control_trial_order")
    compute_group.add_argument("--fdr-alpha", type=float, default=None)
    compute_group.add_argument("--computations", nargs="+", choices=BEHAVIOR_COMPUTATIONS, default=None)
    compute_group.add_argument(
        "--validate-only",
        action="store_true",
        help="Load and validate data only (no statistics). Still writes metadata and outputs manifest.",
    )
    
    feature_choices = [
        "power", "connectivity", "aperiodic", "erp", "bursts", "itpc", "pac",
        "complexity", "quality", "erds", "spectral", "ratios", "asymmetry",
    ]
    compute_group.add_argument(
        "--correlations-features", nargs="+", choices=feature_choices, default=None,
        help="Feature categories for correlations analysis"
    )
    compute_group.add_argument(
        "--pain_sensitivity-features", nargs="+", choices=feature_choices, default=None,
        help="Feature categories for pain sensitivity analysis"
    )
    compute_group.add_argument(
        "--condition-features", nargs="+", choices=feature_choices, default=None,
        help="Feature categories for condition comparison"
    )
    compute_group.add_argument(
        "--temporal-features", nargs="+", choices=feature_choices, default=None,
        help="Feature categories for temporal analysis"
    )
    compute_group.add_argument(
        "--cluster-features", nargs="+", choices=feature_choices, default=None,
        help="Feature categories for cluster permutation tests"
    )
    compute_group.add_argument(
        "--mediation-features", nargs="+", choices=feature_choices, default=None,
        help="Feature categories for mediation analysis"
    )
    compute_group.add_argument(
        "--feature-files",
        nargs="+",
        choices=FEATURE_FILE_CHOICES,
        default=None,
        metavar="FILE",
        help="Specific feature files to load (e.g., power, aperiodic, connectivity). Default: all available",
    )
    compute_group.add_argument(
        "--bands",
        nargs="+",
        choices=["delta", "theta", "alpha", "beta", "gamma"],
        default=None,
        help="Frequency bands to use for analysis (default: all)",
    )
    
    # Cluster-specific options
    cluster_group = parser.add_argument_group("Cluster permutation options")
    cluster_group.add_argument("--cluster-threshold", type=float, default=None, help="Cluster forming threshold")
    cluster_group.add_argument("--cluster-min-size", type=int, default=None, help="Minimum cluster size")
    cluster_group.add_argument("--cluster-tail", type=int, choices=[-1, 0, 1], default=None, help="Test tail: 0=two-tailed, 1=upper, -1=lower")
    
    # Mediation-specific options
    mediation_group = parser.add_argument_group("Mediation analysis options")
    mediation_group.add_argument("--mediation-bootstrap", type=int, default=None, help="Bootstrap iterations for mediation")
    mediation_group.add_argument("--mediation-max-mediators", type=int, default=None, help="Maximum mediators to test")
    
    # Mixed effects-specific options
    mixed_group = parser.add_argument_group("Mixed effects options")
    mixed_group.add_argument("--mixed-max-features", type=int, default=None, help="Maximum features for mixed effects")
    
    # Condition-specific options
    condition_group = parser.add_argument_group("Condition comparison options")
    condition_group.add_argument("--condition-effect-threshold", type=float, default=None, help="Minimum effect size (Cohen's d) to report")
    
    visualize_group = parser.add_argument_group("Visualize mode options")
    plot_group = visualize_group.add_mutually_exclusive_group()
    plot_group.add_argument("--plots", nargs="+", metavar="PLOT")
    plot_group.add_argument("--all-plots", action="store_true")
    visualize_group.add_argument("--skip-scatter", action="store_true")

    add_path_args(parser)

    return parser


def run_behavior(args: argparse.Namespace, subjects: List[str], config: Any) -> None:
    """Execute the behavior command."""
    from eeg_pipeline.pipelines.behavior import BehaviorPipeline
    from eeg_pipeline.plotting.orchestration.behavior import visualize_behavior_for_subjects
    
    categories = getattr(args, "categories", None)
    progress = create_progress_reporter(args)
    task = resolve_task(args.task, config)
    
    if getattr(args, "bids_root", None):
        config.setdefault("paths", {})["bids_root"] = args.bids_root
    if getattr(args, "deriv_root", None):
        config.setdefault("paths", {})["deriv_root"] = args.deriv_root
    
    if args.mode == "compute":
        rng_seed = args.rng_seed if args.rng_seed is not None else config.get("project.random_state")
        if args.correlation_method:
            config.setdefault("behavior_analysis", {}).setdefault("statistics", {})["correlation_method"] = args.correlation_method
        if args.bootstrap and args.bootstrap > 0:
            config.setdefault("behavior_analysis", {})["bootstrap"] = args.bootstrap
        if args.n_perm and args.n_perm > 0:
            config.setdefault("behavior_analysis", {}).setdefault("statistics", {})["n_permutations"] = args.n_perm
        if rng_seed is not None:
            config.setdefault("project", {})["random_state"] = rng_seed
        if getattr(args, "control_temperature", None) is not None:
            config.setdefault("behavior_analysis", {})["control_temperature"] = args.control_temperature
        if getattr(args, "control_trial_order", None) is not None:
            config.setdefault("behavior_analysis", {})["control_trial_order"] = args.control_trial_order
        if getattr(args, "fdr_alpha", None) is not None:
            config.setdefault("behavior_analysis", {}).setdefault("statistics", {})["fdr_alpha"] = args.fdr_alpha

        # Cluster-specific options
        if getattr(args, "cluster_threshold", None) is not None:
            config.setdefault("behavior_analysis", {}).setdefault("cluster", {})["forming_threshold"] = args.cluster_threshold
        if getattr(args, "cluster_min_size", None) is not None:
            config.setdefault("behavior_analysis", {}).setdefault("cluster", {})["min_cluster_size"] = args.cluster_min_size
        if getattr(args, "cluster_tail", None) is not None:
            config.setdefault("behavior_analysis", {}).setdefault("cluster", {})["tail"] = args.cluster_tail
        
        # Mediation-specific options
        if getattr(args, "mediation_bootstrap", None) is not None:
            config.setdefault("behavior_analysis", {}).setdefault("mediation", {})["n_bootstrap"] = args.mediation_bootstrap
        if getattr(args, "mediation_max_mediators", None) is not None:
            config.setdefault("behavior_analysis", {}).setdefault("mediation", {})["max_mediators"] = args.mediation_max_mediators
        
        # Mixed effects-specific options
        if getattr(args, "mixed_max_features", None) is not None:
            config.setdefault("behavior_analysis", {}).setdefault("mixed_effects", {})["max_features"] = args.mixed_max_features
        
        # Condition-specific options
        if getattr(args, "condition_effect_threshold", None) is not None:
            config.setdefault("behavior_analysis", {}).setdefault("condition", {})["effect_size_threshold"] = args.condition_effect_threshold
        
        computation_features = {}
        if getattr(args, "correlations_features", None):
            computation_features["correlations"] = args.correlations_features
        if getattr(args, "pain_sensitivity_features", None):
            computation_features["pain_sensitivity"] = args.pain_sensitivity_features
        if getattr(args, "condition_features", None):
            computation_features["condition"] = args.condition_features
        if getattr(args, "temporal_features", None):
            computation_features["temporal"] = args.temporal_features
        if getattr(args, "cluster_features", None):
            computation_features["cluster"] = args.cluster_features
        if getattr(args, "mediation_features", None):
            computation_features["mediation"] = args.mediation_features
        
        pipeline = BehaviorPipeline(
            config=config,
            computations=args.computations,
            feature_categories=categories,
            feature_files=getattr(args, "feature_files", None),
            computation_features=computation_features if computation_features else None,
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
                "pain_clusters",
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
            scatter_only=False,
            temporal_only=False,
            visualize_categories=visualize_categories,
            plots=plots,
        )
