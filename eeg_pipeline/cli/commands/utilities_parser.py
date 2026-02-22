"""Parser construction for utilities CLI command."""

from __future__ import annotations

import argparse

from eeg_pipeline.cli.common import (
    add_common_subject_args,
    add_task_arg,
    add_output_format_args,
)

def setup_utilities(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """Configure the utilities command parser."""
    parser = subparsers.add_parser(
        "utilities",
        help="Utilities: raw-to-bids conversion, merge behavior, clean up disk space",
        description="Utilities pipeline: convert raw EEG to BIDS, merge behavioral data, or clean up disk space",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "mode",
        choices=["raw-to-bids", "fmri-raw-to-bids", "merge-psychopy", "clean"],
        help="Utility operation mode"
    )
    add_common_subject_args(parser)
    add_task_arg(parser)
    add_output_format_args(parser)

    parser.add_argument(
        "--source-root",
        type=str,
        default=None,
        help="Override source data root path (default from config)"
    )
    parser.add_argument(
        "--bids-root",
        type=str,
        default=None,
        help="Override BIDS root path (default from config)"
    )
    parser.add_argument(
        "--bids-fmri-root",
        type=str,
        default=None,
        help="Override BIDS fMRI root path (default from config)"
    )

    raw_group = parser.add_argument_group("raw-to-bids options")
    raw_group.add_argument("--montage", type=str, default="easycap-M1")
    raw_group.add_argument("--line-freq", type=float, default=60.0)
    raw_group.add_argument("--overwrite", action="store_true")
    raw_group.add_argument("--trim-to-first-volume", action="store_true")
    raw_group.add_argument("--event-prefix", action="append")
    raw_group.add_argument("--keep-all-annotations", action="store_true")

    fmri_group = parser.add_argument_group("fmri-raw-to-bids options")
    fmri_group.add_argument("--session", type=str, default=None, help="Optional BIDS session label (without 'ses-')")
    fmri_group.add_argument("--rest-task", type=str, default="rest", help="Task label to use for resting-state BOLD")
    fmri_group.add_argument(
        "--no-rest",
        action="store_true",
        help="Do not convert resting-state series (if present)",
    )
    fmri_group.add_argument(
        "--no-fieldmaps",
        action="store_true",
        help="Do not convert fieldmaps (if present)",
    )
    fmri_group.add_argument(
        "--dicom-mode",
        choices=["symlink", "copy", "skip"],
        default="symlink",
        help="How to store original DICOMs under <bids_fmri_root>/sourcedata (default: symlink)",
    )
    fmri_group.add_argument(
        "--no-events",
        action="store_true",
        help="Do not generate BIDS *_events.tsv from PsychoPy TrialSummary.csv",
    )
    fmri_group.add_argument(
        "--event-granularity",
        choices=["trial", "phases"],
        default="phases",
        help="Events granularity for stimulation (default: phases)",
    )
    fmri_group.add_argument(
        "--onset-reference",
        choices=["as_is", "first_iti_start", "first_stim_start"],
        default="as_is",
        help="How to zero event onsets within each run (default: as_is)",
    )
    fmri_group.add_argument(
        "--onset-offset-s",
        type=float,
        default=0.0,
        help="Additive offset applied after onset-reference (seconds)",
    )
    fmri_group.add_argument(
        "--dcm2niix-path",
        type=str,
        default=None,
        help="Path to dcm2niix binary (defaults to PATH lookup)",
    )
    fmri_group.add_argument(
        "--dcm2niix-arg",
        action="append",
        default=None,
        help="Extra argument passed to dcm2niix (repeatable, inserted after executable)",
    )

    merge_group = parser.add_argument_group("merge-psychopy options")
    merge_group.add_argument("--event-type", action="append")
    merge_group.add_argument(
        "--qc-column",
        action="append",
        default=None,
        help="Cross-modal EEG↔fMRI QC column to compare (repeatable). If omitted, cross-modal column QC is skipped.",
    )
    merge_group.add_argument(
        "--allow-misaligned-trim",
        action="store_true",
        help="Allow behavioral/events count mismatch (trims/pads). Not recommended.",
    )

    clean_group = parser.add_argument_group("clean options")
    clean_group.add_argument(
        "--target",
        choices=["plots", "cache", "logs", "old-features", "all", "preview"],
        default="preview",
        help="What to clean (default: preview shows what would be removed)",
    )
    clean_group.add_argument(
        "--older-than",
        type=int,
        default=None,
        metavar="DAYS",
        help="Only remove files older than N days",
    )
    clean_group.add_argument(
        "--subjects",
        nargs="+",
        default=None,
        help="Only clean specific subjects",
    )
    clean_group.add_argument(
        "--force",
        action="store_true",
        help="Skip confirmation prompt",
    )

    return parser
