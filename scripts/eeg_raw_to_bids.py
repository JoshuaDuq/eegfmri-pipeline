#!/usr/bin/env python3
"""Script: convert raw EEG (BrainVision) to BIDS.

This is a thin wrapper around `eeg_pipeline.pipelines.eeg_raw_to_bids.EEGRawToBidsPipeline`.
Prefer the main CLI (`eeg-pipeline utilities raw-to-bids ...`) for TUI integration.
"""

from __future__ import annotations

import argparse
from typing import Any

from eeg_pipeline.utils.config.loader import load_config
from eeg_pipeline.pipelines.eeg_raw_to_bids import EEGRawToBidsPipeline


def _update_config(config: dict[str, Any], args: argparse.Namespace) -> None:
    paths = config.setdefault("paths", {})
    if args.source_root:
        paths["source_data"] = args.source_root
    if args.bids_root:
        paths["bids_root"] = args.bids_root
    if args.task:
        config.setdefault("project", {})["task"] = args.task


def main() -> int:
    parser = argparse.ArgumentParser(description="EEG raw (BrainVision) → BIDS converter")
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--subject", action="append", required=True, help="Subject label without 'sub-' (repeatable)")
    parser.add_argument("--source-root", type=str, default=None)
    parser.add_argument("--bids-root", type=str, default=None)
    parser.add_argument("--montage", type=str, default="easycap-M1")
    parser.add_argument("--line-freq", type=float, default=60.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--trim-to-first-volume", action="store_true")
    parser.add_argument("--event-prefix", action="append", default=None)
    parser.add_argument("--keep-all-annotations", action="store_true")
    args = parser.parse_args()

    config = load_config()
    _update_config(config, args)

    pipeline = EEGRawToBidsPipeline(config=config)
    pipeline.run_batch(
        subjects=args.subject,
        task=args.task,
        montage=args.montage,
        line_freq=args.line_freq,
        overwrite=args.overwrite,
        do_trim_to_first_volume=args.trim_to_first_volume,
        event_prefixes=args.event_prefix,
        keep_all_annotations=args.keep_all_annotations,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

