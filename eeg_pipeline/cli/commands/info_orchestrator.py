"""Execution orchestrator for info and discovery CLI command."""

from __future__ import annotations

import argparse
from typing import Any, List

from eeg_pipeline.cli.common import resolve_task
from eeg_pipeline.cli.commands.info_helpers import (
    MODE_CONFIG,
    MODE_DISCOVER,
    MODE_FEATURES,
    MODE_FMRI_COLUMNS,
    MODE_FMRI_CONDITIONS,
    MODE_ML_FEATURE_SPACE,
    MODE_MULTIGROUP_STATS,
    MODE_PLOTTERS,
    MODE_ROIS,
    MODE_SUBJECTS,
    MODE_VERSION,
    _extract_subject_id,
    _get_logger,
    _handle_config_mode,
    _handle_discover_mode,
    _handle_features_mode,
    _handle_fmri_columns_mode,
    _handle_fmri_conditions_mode,
    _handle_ml_feature_space_mode,
    _handle_multigroup_stats_mode,
    _handle_plotters_mode,
    _handle_rois_mode,
    _handle_subjects_mode,
    _handle_version_mode,
)

def run_info(args: argparse.Namespace, subjects: List[str], config: Any) -> None:
    """Execute the info command."""
    from eeg_pipeline.infra.paths import resolve_deriv_root

    logger = _get_logger(args.output_json)
    task = resolve_task(args.task, config)
    deriv_root = resolve_deriv_root(config=config)

    if args.mode == MODE_PLOTTERS:
        _handle_plotters_mode(args.output_json)
    elif args.mode == MODE_FMRI_CONDITIONS:
        _handle_fmri_conditions_mode(args, config)
    elif args.mode == MODE_FMRI_COLUMNS:
        _handle_fmri_columns_mode(args, config)
    elif args.mode == MODE_ROIS:
        _handle_rois_mode(args, subjects, config)
    elif args.mode == MODE_DISCOVER:
        _handle_discover_mode(args, subjects, config)
    elif args.mode == MODE_MULTIGROUP_STATS:
        _handle_multigroup_stats_mode(args, subjects, config)
    elif args.mode == MODE_SUBJECTS:
        _handle_subjects_mode(args, deriv_root, task, config, logger)
    elif args.mode == MODE_FEATURES:
        if not args.target:
            print("Error: subject ID required for features mode")
            print("Usage: eeg-pipeline info features <SUBJECT_ID>")
            return
        subject_id = _extract_subject_id(args.target)
        _handle_features_mode(subject_id, deriv_root, args.output_json)
    elif args.mode == MODE_CONFIG:
        _handle_config_mode(args, config, deriv_root, task)
    elif args.mode == MODE_VERSION:
        _handle_version_mode(args.output_json)
    elif args.mode == MODE_ML_FEATURE_SPACE:
        _handle_ml_feature_space_mode(args, config, task)
