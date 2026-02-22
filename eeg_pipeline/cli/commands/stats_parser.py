"""Parser construction for stats CLI command."""

from __future__ import annotations

import argparse

from eeg_pipeline.cli.common import add_task_arg


def setup_stats(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """Configure the stats command parser."""
    parser = subparsers.add_parser(
        "stats",
        help="Project statistics: subjects, features, storage",
        description="Show project-wide statistics and analytics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "mode",
        choices=["summary", "subjects", "features", "storage", "timeline"],
        nargs="?",
        default="summary",
        help="What to show (default: summary)",
    )
    add_task_arg(parser)
    parser.add_argument(
        "--json",
        action="store_true",
        dest="output_json",
        help="Output in JSON format",
    )
    return parser
