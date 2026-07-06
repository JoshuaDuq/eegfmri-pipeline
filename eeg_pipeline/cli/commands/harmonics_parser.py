"""Parser construction for scanner-harmonic QC command."""

from __future__ import annotations

import argparse

from eeg_pipeline.analysis.qc.scanner_harmonics import (
    DEFAULT_GAMMA_EXCLUSIONS,
    DEFAULT_GAMMA_WINDOW,
    DEFAULT_HARMONIC_WINDOWS,
)
from eeg_pipeline.cli.common import add_common_subject_args


def setup_harmonics(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """Configure the scanner-harmonic QC command parser."""
    parser = subparsers.add_parser(
        "harmonics",
        help="Scanner-harmonic QC for gamma EEG analyses",
        description="Compute harmonic-aware gamma QC reports from BrainVision EEG files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input-root",
        required=True,
        help="Directory containing BrainVision .vhdr files to scan recursively.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where scanner_harmonic_gamma_qc.tsv/json will be written.",
    )
    parser.add_argument(
        "--pattern",
        default="*.vhdr",
        help="Recursive filename pattern under --input-root (default: *.vhdr).",
    )
    add_common_subject_args(parser)
    parser.add_argument(
        "--channels",
        nargs="+",
        default=None,
        help=(
            "Channel names to summarize. If omitted, the validated representative "
            "scanner-harmonic QC channel set is used."
        ),
    )
    parser.add_argument(
        "--gamma-band",
        nargs=2,
        type=float,
        default=(DEFAULT_GAMMA_WINDOW.low_hz, DEFAULT_GAMMA_WINDOW.high_hz),
        metavar=("LOW", "HIGH"),
        help="Gamma range in Hz before scanner-harmonic exclusions (default: 30.1 80).",
    )
    parser.add_argument(
        "--exclude-band",
        nargs=2,
        type=float,
        action="append",
        default=None,
        metavar=("LOW", "HIGH"),
        help=(
            "Gamma exclusion window in Hz. Repeatable. Defaults to "
            f"{_format_defaults(DEFAULT_GAMMA_EXCLUSIONS)}."
        ),
    )
    parser.add_argument(
        "--harmonic-window",
        nargs=2,
        type=float,
        action="append",
        default=None,
        metavar=("LOW", "HIGH"),
        help=(
            "Harmonic QC window in Hz. Repeatable. Defaults to "
            f"{_format_defaults(DEFAULT_HARMONIC_WINDOWS)}."
        ),
    )
    parser.add_argument(
        "--nperseg",
        type=int,
        default=16_384,
        help="Welch segment length in samples (default: 16384).",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=4_096,
        help="Minimum samples required per file for PSD QC (default: 4096).",
    )
    return parser


def _format_defaults(windows: tuple) -> str:
    return ", ".join(f"{window.low_hz:g}-{window.high_hz:g}" for window in windows)
