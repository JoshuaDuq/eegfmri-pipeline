"""``eeg-pipeline line-comb`` -- diagnose and remove the scanner room's line comb.

The stages run in the order the modes are listed. ``diagnose`` measures which narrowband
lines are present and which of them belong to the comb; ``plot`` draws that measurement;
``benchmark`` checks the removal against stated preservation criteria before anything is
written; ``apply`` writes the cleaned BIDS copy; ``verify`` re-measures it; ``report``
turns the whole thing into the tables behind ``docs/scanner_harmonic_removal.md``.

``notch`` is the optional last stage. It reads what ``apply`` wrote and takes out bands
that are clusters rather than resolvable lines, which no amount of sinusoid subtraction
can reach. It writes its own BIDS root, so the two transforms stay separable.

``psd`` draws the before-and-after spectra, from MNE Welch estimates of the source and of
every derivative that exists. It needs only ``apply`` to have run.

Run ``benchmark`` before ``apply``. The criteria are stated before the measurement, so a
failure means the settings are wrong, not that the criteria should move.
"""

from __future__ import annotations

import argparse
from typing import Any, List

MODES = ("diagnose", "plot", "benchmark", "apply", "verify", "report", "notch", "psd")


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
        help="diagnose: restrict to these subjects (default: every subject found)",
    )
    parser.add_argument(
        # Not --config: the top-level CLI strips that out of argv before argparse runs, so
        # a subcommand declaring it advertises an option it can never be given. The path
        # went to the core loader instead and the run died on an unrelated deriv_root error.
        "--workflow-config",
        dest="config",
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
    from studies.pain_study.scripts.line_comb import diagnose, notch, plot, psd, remove, report

    if subjects and not args.subjects:
        args.subjects = list(subjects)
    if args.mode in {"benchmark", "apply", "verify", "notch"} and args.subjects:
        raise ValueError(
            f"line-comb {args.mode} must use all recordings; subject subsets cannot certify "
            "or transform the cohort."
        )

    if args.mode == "diagnose":
        diagnose.run(args)
    elif args.mode == "plot":
        args.input_dir = args.output_dir
        plot.run(args)
    elif args.mode == "report":
        args.removal_dir = args.report_dir
        report.run(args)
    elif args.mode == "notch":
        notch.run(args)
    elif args.mode == "psd":
        psd.run(args)
    else:
        args.stage = args.mode
        remove.run(args)
