#!/usr/bin/env python
"""Run one Study 1 feature-benchmark cell.

This is intended for Slurm arrays when confirmatory permutation counts are too
large for the full sequential Study 1 benchmark to finish within one job.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

from eeg_pipeline.analysis.machine_learning.orchestration import run_model_comparison_ml
from eeg_pipeline.cli.main import update_config_from_args
from eeg_pipeline.utils.config.loader import get_config_value, load_config
from eeg_pipeline.utils.config.overrides import apply_set_overrides
from eeg_pipeline.utils.config.roots import resolve_eeg_deriv_root
from studies.pain_study.study1.cohort import (
    study1_feature_root,
    study1_temporal_control_feature_root,
)
from studies.pain_study.study1.config import apply_study1_config_defaults
from studies.pain_study.study1.feature_benchmark import (
    EXPLORATORY_BAND_PRESETS,
    PRIMARY_BAND_PRESETS,
    PRIMARY_FEATURE_SCOPES,
    PRIMARY_FEATURE_SEGMENTS,
    PRIMARY_FEATURE_STATS,
    feature_benchmark_config,
    feature_results_root,
)
from studies.pain_study.study1.feature_spec import (
    PRIMARY_FEATURE_FAMILY,
    resolve_exploratory_feature_families,
)
from studies.pain_study.study1.temporal_controls import (
    TEMPORAL_CONTROL_BANDS,
    TEMPORAL_CONTROL_FEATURE_STATS,
    TEMPORAL_CONTROL_PARTITION,
    resolve_temporal_control_windows,
    temporal_control_feature_spec,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--partition",
        required=True,
        choices=["primary", "exploratory", TEMPORAL_CONTROL_PARTITION],
    )
    parser.add_argument("--target", required=True, choices=["NPS", "SIIPS1"])
    parser.add_argument("--spec", required=True)
    parser.add_argument("--subject", action="append", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--study1-config", required=True)
    parser.add_argument("--bids-root", required=True)
    parser.add_argument("--bids-fmri-root", required=True)
    parser.add_argument("--deriv-root", required=True)
    parser.add_argument(
        "--set",
        dest="set_overrides",
        action="append",
        default=[],
        help="Config override as KEY=VALUE. May be repeated.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> Any:
    config = load_config()
    runtime_args = argparse.Namespace(
        task=args.task,
        task_is_rest=None,
        source_root=None,
        bids_root=args.bids_root,
        bids_rest_root=None,
        bids_fmri_root=args.bids_fmri_root,
        deriv_root=args.deriv_root,
        deriv_rest_root=None,
        set_overrides=[],
    )
    update_config_from_args(config, runtime_args)
    apply_study1_config_defaults(config, config_path=args.study1_config)
    apply_set_overrides(config, args.set_overrides)
    return config


def run_primary(args: argparse.Namespace, config: Any, logger: logging.Logger) -> Path:
    if args.spec not in PRIMARY_BAND_PRESETS:
        raise ValueError(f"Unknown primary Study 1 feature preset: {args.spec}")
    return run_power_bands(
        args=args,
        config=config,
        logger=logger,
        bands=PRIMARY_BAND_PRESETS[args.spec],
        feature_input_root=study1_feature_root(config),
        partition="primary",
        segments=list(PRIMARY_FEATURE_SEGMENTS),
        stats=list(PRIMARY_FEATURE_STATS),
    )


def run_exploratory(args: argparse.Namespace, config: Any, logger: logging.Logger) -> Path:
    if args.spec in EXPLORATORY_BAND_PRESETS:
        return run_power_bands(
            args=args,
            config=config,
            logger=logger,
            bands=EXPLORATORY_BAND_PRESETS[args.spec],
            feature_input_root=study1_feature_root(config),
            partition="exploratory",
            segments=list(PRIMARY_FEATURE_SEGMENTS),
            stats=list(PRIMARY_FEATURE_STATS),
        )

    exploratory_families = resolve_exploratory_feature_families(config)
    if args.spec not in exploratory_families:
        raise ValueError(f"Unknown exploratory Study 1 feature family: {args.spec}")

    return run_model_comparison_ml(
        subjects=list(args.subject),
        task=args.task,
        deriv_root=resolve_eeg_deriv_root(config),
        config=feature_benchmark_config(config, target_name=args.target),
        n_perm=feature_benchmark_permutations(config),
        inner_splits=feature_benchmark_inner_splits(config),
        outer_jobs=feature_benchmark_outer_jobs(config),
        rng_seed=rng_seed(config),
        results_root=feature_results_root(
            config,
            partition="exploratory",
            target_name=args.target,
            feature_spec=args.spec,
        ),
        logger=logger,
        target="fmri_signature",
        feature_families=[args.spec],
        feature_input_root=study1_feature_root(config),
        feature_bands=None,
        feature_harmonization=feature_harmonization(config),
        model_names=["elasticnet", "ridge"],
    )


def run_temporal_control(args: argparse.Namespace, config: Any, logger: logging.Logger) -> Path:
    windows = {
        temporal_control_feature_spec(window.name): window
        for window in resolve_temporal_control_windows(config)
    }
    if args.spec not in windows:
        raise ValueError(f"Unknown temporal-control Study 1 feature spec: {args.spec}")
    return run_power_bands(
        args=args,
        config=config,
        logger=logger,
        bands=list(TEMPORAL_CONTROL_BANDS),
        feature_input_root=study1_temporal_control_feature_root(config),
        partition=TEMPORAL_CONTROL_PARTITION,
        segments=[windows[args.spec].name],
        stats=list(TEMPORAL_CONTROL_FEATURE_STATS),
    )


def run_power_bands(
    *,
    args: argparse.Namespace,
    config: Any,
    logger: logging.Logger,
    bands: list[str],
    feature_input_root: Path,
    partition: str,
    segments: list[str],
    stats: list[str],
) -> Path:
    return run_model_comparison_ml(
        subjects=list(args.subject),
        task=args.task,
        deriv_root=resolve_eeg_deriv_root(config),
        config=feature_benchmark_config(config, target_name=args.target),
        n_perm=feature_benchmark_permutations(config),
        inner_splits=feature_benchmark_inner_splits(config),
        outer_jobs=feature_benchmark_outer_jobs(config),
        rng_seed=rng_seed(config),
        results_root=feature_results_root(
            config,
            partition=partition,
            target_name=args.target,
            feature_spec=args.spec,
        ),
        logger=logger,
        target="fmri_signature",
        feature_families=[PRIMARY_FEATURE_FAMILY],
        feature_input_root=feature_input_root,
        feature_bands=list(bands),
        feature_segments=segments,
        feature_scopes=list(PRIMARY_FEATURE_SCOPES),
        feature_stats=stats,
        feature_harmonization=feature_harmonization(config),
        model_names=["elasticnet", "ridge"],
    )


def feature_benchmark_permutations(config: Any) -> int:
    n_perm = int(get_config_value(config, "study1.feature_benchmark.n_perm", 0))
    if n_perm <= 0:
        raise ValueError("study1.feature_benchmark.n_perm must be > 0.")
    return n_perm


def feature_benchmark_inner_splits(config: Any) -> int:
    return int(get_config_value(config, "study1.feature_benchmark.inner_splits", 5))


def feature_benchmark_outer_jobs(config: Any) -> int:
    return int(get_config_value(config, "study1.feature_benchmark.outer_jobs", 1))


def rng_seed(config: Any) -> int:
    return int(get_config_value(config, "project.random_state", 42))


def feature_harmonization(config: Any) -> str:
    return str(
        get_config_value(
            config,
            "study1.feature_benchmark.feature_harmonization",
            "intersection",
        )
    ).strip()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = parse_args()
    config = build_config(args)
    logger = logging.getLogger("study1_benchmark_cell")

    if args.partition == "primary":
        output = run_primary(args, config, logger)
    elif args.partition == "exploratory":
        output = run_exploratory(args, config, logger)
    elif args.partition == TEMPORAL_CONTROL_PARTITION:
        output = run_temporal_control(args, config, logger)
    else:
        raise ValueError(f"Unsupported Study 1 partition: {args.partition}")

    print(output)


if __name__ == "__main__":
    main()
