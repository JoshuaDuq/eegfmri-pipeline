"""Parser construction for validate CLI command."""

from __future__ import annotations

import argparse

from eeg_pipeline.cli.common import add_task_arg


def setup_validate(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """Configure the validate command parser."""
    parser = subparsers.add_parser(
        "validate",
        help="Validate data integrity: epochs, features, BIDS",
        description="Check data files for integrity and consistency issues",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "mode",
        choices=["all", "epochs", "features", "behavior", "bids", "quick"],
        nargs="?",
        default="quick",
        help="What to validate (default: quick)",
    )
    add_task_arg(parser)
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=None,
        help="Specific subjects to validate",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="output_json",
        help="Output in JSON format",
    )
    return parser
