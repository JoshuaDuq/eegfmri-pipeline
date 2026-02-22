"""Execution orchestrator for machine learning CLI command."""

from __future__ import annotations

import argparse
from typing import Any, Dict, List

from eeg_pipeline.cli.commands.machine_learning_overrides import (
    _update_fmri_signature_target_config,
    _update_ml_plot_config,
    _update_model_config,
    _update_path_config,
)
from eeg_pipeline.cli.commands.machine_learning_parser import ML_STAGES
from eeg_pipeline.cli.common import (
    MIN_SUBJECTS_FOR_ML,
    MIN_SUBJECTS_KEY,
    create_progress_reporter,
    resolve_task,
    validate_min_subjects,
    validate_subjects_not_empty,
)


def _print_stage_list() -> None:
    """Print available ML stages and exit."""
    print("\nAvailable ML stages:")
    print("=" * 50)
    for stage, description in ML_STAGES.items():
        print(f"  {stage:22s} - {description}")
    print("\nUsage: eeg ml <stage> --subjects 0001 0002 ...")


def _print_dry_run_info(args: argparse.Namespace, subjects: List[str]) -> None:
    """Print dry-run information and exit."""
    print(f"\n[DRY RUN] Would execute ML stage: {args.mode}")
    print(f"  Subjects: {subjects}")
    print(f"  CV scope: {args.cv_scope}")
    print(f"  Model: {args.model}")
    print(f"  n_perm: {args.n_perm}")
    print(f"  inner_splits: {args.inner_splits}")
    if args.target:
        print(f"  Target: {args.target}")
    if args.feature_families:
        print(f"  Feature families: {args.feature_families}")
    if args.feature_bands:
        print(f"  Feature bands: {args.feature_bands}")
    if args.feature_segments:
        print(f"  Feature segments: {args.feature_segments}")
    if args.feature_scopes:
        print(f"  Feature scopes: {args.feature_scopes}")
    if args.feature_stats:
        print(f"  Feature stats: {args.feature_stats}")
    if args.covariates:
        print(f"  Covariates: {args.covariates}")
    if args.require_trial_ml_safe:
        print("  Require trial_ml_safe: True")
    if args.ml_plots is not None:
        print(f"  ML plots enabled: {bool(args.ml_plots)}")
    if args.ml_plot_formats:
        print(f"  ML plot formats: {args.ml_plot_formats}")


def _validate_ml_requirements(subjects: List[str], cv_scope: str, config: Any) -> None:
    """Validate subject requirements for ML pipeline."""
    validate_subjects_not_empty(subjects, "ml")

    if cv_scope == "group":
        min_subjects = config.get(MIN_SUBJECTS_KEY, MIN_SUBJECTS_FOR_ML)
        validate_min_subjects(subjects, min_subjects, "ML (group scope)")


def _build_pipeline_kwargs(args: argparse.Namespace, config: Any) -> Dict[str, Any]:
    """Build keyword arguments for pipeline.run_batch."""
    rng_seed = args.rng_seed if args.rng_seed is not None else config.get("project.random_state")

    return {
        "cv_scope": args.cv_scope,
        "n_perm": args.n_perm,
        "inner_splits": args.inner_splits,
        "outer_jobs": args.outer_jobs,
        "rng_seed": rng_seed,
        "model": args.model,
        "uncertainty_alpha": args.uncertainty_alpha,
        "perm_n_repeats": args.perm_n_repeats,
        "classification_model": args.classification_model,
        "target": args.target,
        "binary_threshold": args.binary_threshold,
        "feature_families": args.feature_families,
        "feature_bands": args.feature_bands,
        "feature_segments": args.feature_segments,
        "feature_scopes": args.feature_scopes,
        "feature_stats": args.feature_stats,
        "feature_harmonization": args.feature_harmonization,
        "baseline_predictors": args.baseline_predictors,
        "covariates": args.covariates,
    }


def run_ml(args: argparse.Namespace, subjects: List[str], config: Any) -> None:
    """Execute the ml command."""
    if args.list_stages:
        _print_stage_list()
        return

    from eeg_pipeline.pipelines.machine_learning import MLPipeline

    if args.dry_run:
        _print_dry_run_info(args, subjects)
        return

    _validate_ml_requirements(subjects, args.cv_scope, config)

    progress = create_progress_reporter(args)
    task = resolve_task(args.task, config)

    _update_path_config(args, config)
    _update_model_config(args, config)
    _update_ml_plot_config(args, config)
    _update_fmri_signature_target_config(args, config)
    # Enforce trial-level ML-safe feature computation by default to prevent CV leakage.
    # --require-trial-ml-safe is retained for backward-compatible CLI ergonomics.
    config["machine_learning.data.require_trial_ml_safe"] = True
    config["feature_engineering.analysis_mode"] = "trial_ml_safe"

    pipeline_kwargs = _build_pipeline_kwargs(args, config)
    pipeline = MLPipeline(config=config)

    pipeline.run_batch(
        subjects=subjects,
        task=task,
        mode=args.mode,
        progress=progress,
        **pipeline_kwargs,
    )
