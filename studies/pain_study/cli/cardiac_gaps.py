"""``eeg-pipeline cardiac-gaps`` -- fill BrainVision Analyzer's pulse-marker gaps.

Analyzer's own correction is kept wherever it marked a beat, because it measurably beats
ours there. What it never marked is untouched artifact, and that is all this workflow
changes: beats are recovered by QRS template matching, correction is confined to the
recovered stretches, and the result is scored on removal and preservation together.

``report`` measures the gaps without changing anything; ``markers`` writes recordings
carrying the recovered R markers so Analyzer can be pointed at them; ``benchmark`` scores
methods and ranks before anything is written; ``apply`` writes the corrected recordings;
``verify`` re-reads the written binaries and scores them.

Choose on both benchmark arms together. Removal alone improves monotonically with rank
while the signal is being destroyed.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, List

MODES = ("report", "markers", "benchmark", "apply", "verify")


def setup_cardiac_gaps(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    """Configure the cardiac-gaps parser."""
    parser = subparsers.add_parser(
        "cardiac-gaps",
        help="Recover Analyzer's unmarked heartbeats and correct only where it left gaps",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("mode", choices=list(MODES), help="Stage to run, in the order listed")
    parser.add_argument(
        "--subjects",
        nargs="*",
        default=None,
        help="Restrict to these subjects (default: every paired recording found)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Workflow YAML (default: studies/pain_study/scripts/cardiac_gaps/config.yaml)",
    )
    parser.add_argument(
        "--uncorrected-root",
        type=str,
        default=None,
        help="Analyzer export with pulse markers and no BCG correction",
    )
    parser.add_argument(
        "--corrected-root",
        type=str,
        default=None,
        help="Analyzer export with its correction applied",
    )
    parser.add_argument(
        "--output-root", type=str, default=None, help="Where corrected recordings are written"
    )
    parser.add_argument(
        "--output", type=str, default=None, help="Destination TSV (default: named for the mode)"
    )
    parser.add_argument(
        "--method", choices=["obs", "aas"], default=None, help="Correction method (default: config)"
    )
    parser.add_argument(
        "--n-components", type=int, default=None, help="OBS component count (default: config)"
    )
    parser.add_argument("--limit", type=int, default=None, help="Cap the number of recordings")
    return parser


#: Options the workflow treats as paths: it calls ``destination.parent`` and builds
#: ``output_root / name``, so a string reaches those as an ``AttributeError`` only after
#: the run has finished and the results are about to be written.
_PATH_OPTIONS = ("output", "output_root", "config", "uncorrected_root", "corrected_root")


def run_cardiac_gaps(args: argparse.Namespace, subjects: List[str], config: Any) -> None:
    """Dispatch one stage to the correction module."""
    from studies.pain_study.scripts.cardiac_gaps import correct

    if subjects and not args.subjects:
        args.subjects = list(subjects)
    args.command = args.mode
    # None is meaningful -- it means "take this from the config" -- so it is left alone.
    for option in _PATH_OPTIONS:
        value = getattr(args, option, None)
        if value is not None:
            setattr(args, option, Path(value))
    correct.run(args)
