"""Parser construction for the cohort preprocessing QC report."""

from __future__ import annotations

import argparse

from eeg_pipeline.preprocessing.report.cohort.aggregate import DEFAULT_GATES


def setup_cohort_report(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """Configure the cohort-report command parser."""
    parser = subparsers.add_parser(
        "cohort-report",
        help="Aggregate per-subject preprocessing QC into one cohort report",
        description=(
            "Build a cohort preprocessing QC report from the per-subject QC sidecars "
            "written beside each subject report. Reads recorded measurements rather than "
            "recomputing them, so it is cheap to re-run whenever a participant is added, "
            "and a cohort figure is the same measurement as the subject figure beneath it."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=None,
        help=(
            "Participant labels to aggregate, without the sub- prefix. Defaults to every "
            "participant with a sidecar. A requested participant with no subject report is "
            "an error rather than an absence, so a typo cannot quietly shrink the cohort."
        ),
    )
    parser.add_argument(
        "--task",
        default=None,
        help=(
            "Restrict the cohort to one task, and name the outputs after it. Left unset, "
            "whatever is on disk is aggregated and the report states which tasks that "
            "turned out to be."
        ),
    )
    parser.add_argument(
        "--deriv-root",
        default=None,
        help=(
            "Derivatives root to search. Defaults to the configured deriv_root, under "
            "which subject reports are looked for in preprocessed/eeg."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Where the report, its audit tables and its log are written. Defaults to "
            "<deriv-root>/preprocessed/eeg/group."
        ),
    )
    parser.add_argument(
        "--min-subjects-for-median",
        type=int,
        default=None,
        help=(
            "Participants required before a panel draws a median and interquartile band. "
            "Overrides report.thresholds.min_subjects_for_median, which defaults to "
            f"{DEFAULT_GATES.min_subjects_for_median}. A value that would extrapolate a "
            "quartile beyond the observed participants is rejected."
        ),
    )
    parser.add_argument(
        "--min-subjects-for-outer-band",
        type=int,
        default=None,
        help=(
            "Participants required before a panel additionally draws the 10th-to-90th "
            "band. Overrides report.thresholds.min_subjects_for_outer_band, which "
            f"defaults to {DEFAULT_GATES.min_subjects_for_outer_band}."
        ),
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Title for the document. Defaults to one naming the task.",
    )
    return parser
