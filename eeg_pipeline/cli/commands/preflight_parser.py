"""Parser construction for the preflight CLI command."""

from __future__ import annotations

import argparse

from eeg_pipeline.cli.common import add_path_args, add_task_arg


def setup_preflight(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """Configure the preflight command parser."""
    parser = subparsers.add_parser(
        "preflight",
        help="Inspect a BIDS dataset before processing it",
        description=(
            "Report the shape of the configured dataset without processing or writing "
            "anything: subjects, tasks and runs, whether every recording has events, "
            "sampling rate and channel layout across runs, the declared ECG/EOG "
            "channels, the line frequency, and completeness of the required event "
            "columns. Reads BIDS metadata only, never the recordings."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_task_arg(parser)
    add_path_args(parser)
    parser.add_argument(
        "--json",
        action="store_true",
        dest="output_json",
        help="Output in JSON format",
    )
    return parser
