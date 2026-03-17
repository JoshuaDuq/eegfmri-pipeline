"""CLI-only entrypoint for paradigm-specific conversion/merge scripts."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from eeg_raw_to_bids import run_raw_to_bids
from fmri_raw_to_bids import run_fmri_raw_to_bids
from merge_psychopy import run_merge_psychopy


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Paradigm-specific preprocessing helpers (CLI only).",
    )
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    sub = p.add_subparsers(dest="command", required=True)

    eeg = sub.add_parser("eeg-raw-to-bids", help="Convert BrainVision EEG raw data to BIDS")
    eeg.add_argument("--source-root", required=True)
    eeg.add_argument("--bids-root", required=True)
    eeg.add_argument("--task", required=True)
    eeg.add_argument("--subject", action="append", default=None)
    eeg.add_argument("--montage", default="easycap-M1")
    eeg.add_argument("--line-freq", type=float, default=60.0)
    eeg.add_argument("--overwrite", action="store_true")
    eeg.add_argument("--trim-to-first-volume", action="store_true")
    eeg.add_argument("--event-prefix", action="append", default=None)
    eeg.add_argument("--keep-all-annotations", action="store_true")

    fmri = sub.add_parser("fmri-raw-to-bids", help="Convert fMRI DICOM raw data to BIDS")
    fmri.add_argument("--source-root", required=True)
    fmri.add_argument("--bids-fmri-root", required=True)
    fmri.add_argument("--task", required=True)
    fmri.add_argument("--subject", action="append", default=None)
    fmri.add_argument("--session", default=None)
    fmri.add_argument("--rest-task", default="rest")
    fmri.add_argument("--no-rest", action="store_true")
    fmri.add_argument("--no-fieldmaps", action="store_true")
    fmri.add_argument("--dicom-mode", choices=["symlink", "copy", "skip"], default="symlink")
    fmri.add_argument("--overwrite", action="store_true")
    fmri.add_argument("--no-events", action="store_true")
    fmri.add_argument("--event-granularity", choices=["trial", "phases"], default="phases")
    fmri.add_argument("--onset-reference", choices=["as_is", "first_iti_start", "first_stim_start"], default="first_iti_start")
    fmri.add_argument("--onset-offset-s", type=float, default=0.0)
    fmri.add_argument("--dcm2niix-path", default=None)
    fmri.add_argument("--dcm2niix-arg", action="append", default=None)

    merge = sub.add_parser("merge-psychopy", help="Merge PsychoPy TrialSummary into BIDS EEG events.tsv")
    merge.add_argument("--source-root", required=True)
    merge.add_argument("--bids-root", required=True)
    merge.add_argument("--task", required=True)
    merge.add_argument("--subject", action="append", default=None)
    merge.add_argument("--event-prefix", action="append", default=None)
    merge.add_argument("--event-type", action="append", default=None)
    merge.add_argument("--dry-run", action="store_true")
    merge.add_argument("--allow-misaligned-trim", action="store_true")

    return p


def main() -> int:
    args = _parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s %(name)s: %(message)s")

    if args.command == "eeg-raw-to-bids":
        n = run_raw_to_bids(
            source_root=Path(args.source_root),
            bids_root=Path(args.bids_root),
            task=args.task,
            subjects=args.subject,
            montage=args.montage,
            line_freq=float(args.line_freq),
            overwrite=bool(args.overwrite),
            do_trim_to_first_volume=bool(args.trim_to_first_volume),
            event_prefixes=args.event_prefix,
            keep_all_annotations=bool(args.keep_all_annotations),
        )
        print(f"Converted EEG files: {n}")
        return 0

    if args.command == "fmri-raw-to-bids":
        n = run_fmri_raw_to_bids(
            source_root=Path(args.source_root),
            bids_fmri_root=Path(args.bids_fmri_root),
            task=args.task,
            subjects=args.subject,
            session=args.session,
            rest_task=args.rest_task,
            include_rest=not args.no_rest,
            include_fieldmaps=not args.no_fieldmaps,
            dicom_mode=args.dicom_mode,
            overwrite=bool(args.overwrite),
            create_events=not args.no_events,
            event_granularity=args.event_granularity,
            onset_reference=args.onset_reference,
            onset_offset_s=float(args.onset_offset_s),
            dcm2niix_path=args.dcm2niix_path,
            dcm2niix_extra_args=args.dcm2niix_arg,
        )
        print(f"Converted fMRI series: {n}")
        return 0

    n = run_merge_psychopy(
        bids_root=Path(args.bids_root),
        source_root=Path(args.source_root),
        task=args.task,
        subjects=args.subject,
        event_prefixes=args.event_prefix,
        event_types=args.event_type,
        dry_run=bool(args.dry_run),
        allow_misaligned_trim=bool(args.allow_misaligned_trim),
    )
    print(f"Merged events files: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
