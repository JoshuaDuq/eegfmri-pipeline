"""``eeg-pipeline line-comb`` -- diagnose and remove the scanner room's line comb.

The stages run in the order the modes are listed. ``diagnose`` measures which narrowband
lines are present and which of them belong to the comb; ``plot`` draws that measurement;
``benchmark`` checks the removal against stated preservation criteria before anything is
written; ``apply`` writes the cleaned BIDS copy; ``verify`` re-measures it; ``report``
turns the whole thing into the tables behind ``docs/scanner_harmonic_removal.md``.

Run ``benchmark`` before ``apply``. The criteria are stated before the measurement, so a
failure means the settings are wrong, not that the criteria should move.
"""

from __future__ import annotations

import argparse
from typing import Any, List

MODES = ("diagnose", "plot", "benchmark", "apply", "verify", "report")


def setup_line_comb(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """Configure the line-comb parser."""
    parser = subparsers.add_parser(
        "line-comb",
        help="Diagnose and remove the scanner room's narrowband line comb",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("mode", choices=list(MODES), help="Stage to run, in the order listed")
    parser.add_argument(
        "--subjects",
        nargs="*",
        default=None,
        help="Restrict to these subjects (default: every subject found)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Workflow YAML (default: studies/pain_study/scripts/line_comb/config.yaml)",
    )
    parser.add_argument("--bids-root", type=str, default=None, help="Override the BIDS EEG root")
    parser.add_argument(
        "--output-root", type=str, default=None, help="apply: where the cleaned BIDS copy goes"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None, help="diagnose/plot: where the catalogue goes"
    )
    parser.add_argument(
        "--report-dir", type=str, default=None, help="benchmark/verify/report: where tables go"
    )
    parser.add_argument("--deriv-root", type=str, default=None, help="diagnose: preprocessed EEG")
    parser.add_argument("--source-root", type=str, default=None, help="diagnose: source data root")
    parser.add_argument(
        "--limit", type=int, default=None, help="benchmark: number of runs to sample"
    )
    parser.add_argument(
        "--fundamental-scope",
        choices=("session", "run"),
        default="session",
        help="Pool the frequency estimate over a session (default) or use each run's own",
    )
    parser.add_argument(
        "--stage",
        choices=("cache", "analyse", "all"),
        default="all",
        help="diagnose: read the recordings, rework the statistics from cache, or both",
    )
    parser.add_argument("--filter-length", default=None, help="Override the removal filter length")
    parser.add_argument(
        "--mt-bandwidth", type=float, default=None, help="Override the multitaper bandwidth"
    )
    return parser


def run_line_comb(args: argparse.Namespace, subjects: List[str], config: Any) -> None:
    """Dispatch one stage to the module that implements it."""
    from studies.pain_study.scripts.line_comb import diagnose, plot, remove, report

    if subjects and not args.subjects:
        args.subjects = list(subjects)

    if args.mode == "diagnose":
        diagnose.run(args)
    elif args.mode == "plot":
        args.input_dir = args.output_dir
        plot.run(args)
    elif args.mode == "report":
        args.removal_dir = args.report_dir
        args.diagnosis_dir = args.output_dir
        report.run(args)
    else:
        args.stage = args.mode
        remove.run(args)
