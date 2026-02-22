"""Parser construction for info and discovery CLI command."""

from __future__ import annotations

import argparse

from eeg_pipeline.cli.common import add_task_arg
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
    SOURCE_ALL,
    SOURCE_BIDS,
    SOURCE_BIDS_FMRI,
    SOURCE_EPOCHS,
    SOURCE_FEATURES,
    SOURCE_SOURCE_DATA,
)

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
        choices=[MODE_SUBJECTS, MODE_FEATURES, MODE_CONFIG, MODE_VERSION, MODE_PLOTTERS, MODE_DISCOVER, MODE_ROIS, MODE_FMRI_CONDITIONS, MODE_FMRI_COLUMNS, MODE_MULTIGROUP_STATS, MODE_ML_FEATURE_SPACE],
        help="What to show: subjects, features, config, version, discover columns, rois, fmri-conditions, fmri-columns, or multigroup-stats",
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
        "--cache",
        action="store_true",
        dest="subjects_cache",
        help="Cache subject discovery/status results on disk (subjects mode only; speeds up the TUI)",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        dest="subjects_refresh",
        help="Force refresh of cached subject discovery/status (subjects mode only)",
    )
    parser.add_argument(
        "--source",
        choices=[SOURCE_BIDS, SOURCE_BIDS_FMRI, SOURCE_EPOCHS, SOURCE_FEATURES, SOURCE_SOURCE_DATA, SOURCE_ALL],
        default=SOURCE_EPOCHS,
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

    discover_group = parser.add_argument_group("Discover options (mode: discover)")
    discover_group.add_argument(
        "--discover-source",
        choices=["events", "trial-table", "condition-effects", "all"],
        default="all",
        help="Where to discover columns from (default: all)",
    )
    discover_group.add_argument(
        "--subject",
        default=None,
        help="Specific subject to scan (default: auto-detect from first available)",
    )
    discover_group.add_argument(
        "--column",
        default=None,
        help="Get values for a specific column only",
    )

    ml_group = parser.add_argument_group("ML feature-space options (mode: ml-feature-space)")
    ml_group.add_argument(
        "--subjects",
        nargs="+",
        default=None,
        help="Subjects to scan (e.g., 0001 0002). Defaults to first available subjects with features.",
    )
    ml_group.add_argument(
        "--max-subjects",
        type=int,
        default=3,
        help="Max subjects to scan when --subjects is omitted (default: 3).",
    )
    ml_group.add_argument(
        "--feature-families",
        nargs="+",
        default=None,
        help="Feature families to scan (e.g., power connectivity). Defaults to config machine_learning.data.feature_families.",
    )

    return parser


