#!/usr/bin/env python3
"""Script: merge PsychoPy TrialSummary.csv into BIDS events.tsv.

Thin wrapper around `eeg_pipeline.pipelines.merge_psychopy.MergePsychopyPipeline`.
Prefer the main CLI (`eeg-pipeline utilities merge-psychopy ...`) for TUI integration.
"""

from __future__ import annotations

import argparse
from eeg_pipeline.utils.config.loader import load_config
from eeg_pipeline.utils.config.overrides import apply_runtime_overrides
from eeg_pipeline.pipelines.merge_psychopy import MergePsychopyPipeline

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
    apply_runtime_overrides(
        config,
        task=args.task,
        source_root=args.source_root,
        bids_root=args.bids_root,
    )

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
