#!/usr/bin/env python
"""List Study 1 feature-benchmark cells whose summary JSON is missing."""

from __future__ import annotations

import argparse
from typing import Any

from eeg_pipeline.cli.main import update_config_from_args
from eeg_pipeline.utils.config.loader import load_config
from eeg_pipeline.utils.config.overrides import apply_set_overrides
from studies.pain_study.study1.config import apply_study1_config_defaults
from studies.pain_study.study1.feature_benchmark import (
    EXPLORATORY_BAND_PRESETS,
    PRIMARY_BAND_PRESETS,
    feature_results_root,
)
from studies.pain_study.study1.feature_spec import resolve_exploratory_feature_families
from studies.pain_study.study1.targets import PRIMARY_SIGNATURES
from studies.pain_study.study1.temporal_controls import (
    TEMPORAL_CONTROL_PARTITION,
    resolve_temporal_control_windows,
    temporal_control_feature_spec,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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


def iter_expected_cells(config: Any) -> list[tuple[str, str, str]]:
    cells: list[tuple[str, str, str]] = []
    for target_name in PRIMARY_SIGNATURES:
        cells.extend(("primary", target_name, spec) for spec in PRIMARY_BAND_PRESETS)
        cells.extend(
            ("exploratory", target_name, spec)
            for spec in sorted(resolve_exploratory_feature_families(config))
        )
        cells.extend(("exploratory", target_name, spec) for spec in EXPLORATORY_BAND_PRESETS)
        cells.extend(
            (
                TEMPORAL_CONTROL_PARTITION,
                target_name,
                temporal_control_feature_spec(window.name),
            )
            for window in resolve_temporal_control_windows(config)
        )
    return cells


def summary_exists(
    config: Any,
    *,
    partition: str,
    target_name: str,
    feature_spec: str,
) -> bool:
    summary_path = (
        feature_results_root(
            config,
            partition=partition,
            target_name=target_name,
            feature_spec=feature_spec,
        )
        / "model_comparison"
        / "metrics"
        / "model_comparison_summary.json"
    )
    return summary_path.exists()


def main() -> None:
    args = parse_args()
    config = build_config(args)
    for partition, target_name, feature_spec in iter_expected_cells(config):
        if not summary_exists(
            config,
            partition=partition,
            target_name=target_name,
            feature_spec=feature_spec,
        ):
            print(partition, target_name, feature_spec)


if __name__ == "__main__":
    main()
