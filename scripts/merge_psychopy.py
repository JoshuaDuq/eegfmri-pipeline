#!/usr/bin/env python3
"""Script: merge PsychoPy TrialSummary.csv into BIDS events.tsv.

Thin wrapper around `eeg_pipeline.pipelines.merge_psychopy.MergePsychopyPipeline`.
Prefer the main CLI (`eeg-pipeline utilities merge-psychopy ...`) for TUI integration.
"""

from __future__ import annotations

import argparse
from typing import Any

from eeg_pipeline.utils.config.loader import load_config
from eeg_pipeline.pipelines.merge_psychopy import MergePsychopyPipeline


def _update_config(config: dict[str, Any], args: argparse.Namespace) -> None:
    paths = config.setdefault("paths", {})
    if args.source_root:
        paths["source_data"] = args.source_root
    if args.bids_root:
        paths["bids_root"] = args.bids_root
    if args.task:
        config.setdefault("project", {})["task"] = args.task


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge PsychoPy TrialSummary → BIDS events.tsv")
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--subject", action="append", required=True, help="Subject label without 'sub-' (repeatable)")
    parser.add_argument("--source-root", type=str, default=None)
    parser.add_argument("--bids-root", type=str, default=None)
    parser.add_argument("--event-prefix", action="append", default=None)
    parser.add_argument("--event-type", action="append", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-misaligned-trim",
        action="store_true",
        help="Allow behavioral/events count mismatch (trims/pads). Not recommended.",
    )
    args = parser.parse_args()

    config = load_config()
    _update_config(config, args)

    pipeline = MergePsychopyPipeline(config=config)
    pipeline.run_batch(
        subjects=args.subject,
        task=args.task,
        event_prefixes=args.event_prefix,
        event_types=args.event_type,
        dry_run=args.dry_run,
        allow_misaligned_trim=args.allow_misaligned_trim,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
