"""Preprocessing orchestration (analysis-layer).

This module contains reusable, non-CLI orchestration for converting raw EEG to BIDS and
merging behavioral data into BIDS events files.

The pipeline layer (`eeg_pipeline.pipelines.preprocessing`) should delegate to these
functions to keep pipeline modules thin.
"""

from __future__ import annotations

import inspect
import logging
import re
from pathlib import Path
from typing import Any, List, Optional

import mne
import pandas as pd

from eeg_pipeline.infra.tsv import read_tsv

from eeg_pipeline.utils.data.preprocessing import (
    combine_runs_for_subject,
    create_event_mask,
    ensure_dataset_description,
    extract_run_number,
    filter_annotations,
    find_behavior_csv_for_run,
    find_brainvision_vhdrs,
    get_run_index,
    normalize_event_filters,
    normalize_string,
    parse_subject_id,
    set_channel_types,
    set_montage,
    trim_to_first_volume,
)

logger = logging.getLogger(__name__)


def merge_behavior_to_events(
    events_tsv: Path,
    source_root: Path,
    event_prefixes: Optional[List[str]] = None,
    event_types: Optional[List[str]] = None,
    dry_run: bool = False,
    *,
    _logger: Optional[logging.Logger] = None,
) -> bool:
    """Merge behavioral data into a single events.tsv file."""
    log = _logger or logger

    from eeg_pipeline.utils.data.alignment import trim_behavioral_to_events_strict

    m = re.search(r"sub-([A-Za-z0-9]+)", str(events_tsv))
    if not m:
        log.warning("Could not parse subject from: %s", events_tsv)
        return False
    sub_label = m.group(1)

    run_num = extract_run_number(events_tsv)
    beh_csv = find_behavior_csv_for_run(source_root / f"sub-{sub_label}", run=run_num)

    if not beh_csv or not beh_csv.exists():
        if run_num is None:
            log.warning(
                "No TrialSummary.csv found for sub-%s under %s/sub-%s/PsychoPy_Data",
                sub_label,
                source_root,
                sub_label,
            )
        else:
            log.warning(
                "No TrialSummary.csv matching run %d found for sub-%s under %s/sub-%s/PsychoPy_Data",
                run_num,
                sub_label,
                source_root,
                sub_label,
            )
        return False

    try:
        ev_df = read_tsv(events_tsv)
    except (pd.errors.ParserError, OSError) as exc:
        log.error("Failed reading events: %s -> %s", events_tsv, exc)
        return False

    try:
        beh_df = pd.read_csv(beh_csv)
    except (pd.errors.ParserError, OSError) as exc:
        log.error("Failed reading behavior: %s -> %s", beh_csv, exc)
        return False

    if "trial_type" not in ev_df.columns:
        log.warning("'trial_type' column missing in events: %s", events_tsv)
        return False

    normalized_trial_types = ev_df["trial_type"].map(normalize_string)

    normalized_prefixes = normalize_event_filters(event_prefixes)
    normalized_types = normalize_event_filters(event_types)

    if not normalized_prefixes and not normalized_types:
        normalized_prefixes = [normalize_string("Trig_therm/T  1")]

    event_mask = create_event_mask(normalized_trial_types, normalized_prefixes, normalized_types)
    target_indices = ev_df.index[event_mask].tolist()

    if len(target_indices) == 0:
        criteria_parts = []
        if normalized_prefixes:
            criteria_parts.append(f"prefixes={normalized_prefixes}")
        if normalized_types:
            criteria_parts.append(f"types={normalized_types}")
        criteria_description = "; ".join(criteria_parts) if criteria_parts else "<none>"
        log.warning("No target events in: %s (criteria: %s)", events_tsv, criteria_description)
        return False

    target_events_df = ev_df.iloc[target_indices].copy()
    try:
        behavioral_subset = trim_behavioral_to_events_strict(beh_df, target_events_df)
    except ValueError as exc:
        run_text = f"run-{run_num} " if run_num is not None else ""
        log.error("Behavioral/events mismatch for sub-%s %s: %s", sub_label, run_text, exc)
        return False

    n_matched = len(behavioral_subset)
    event_rows_to_update = target_indices[:n_matched]

    for column in behavioral_subset.columns:
        if column not in ev_df.columns:
            ev_df[column] = pd.NA
        ev_df.loc[event_rows_to_update, column] = behavioral_subset[column].values

    if dry_run:
        log.info(
            "[dry-run] Would update: %s with columns: %s from %s",
            events_tsv,
            list(behavioral_subset.columns),
            beh_csv.name,
        )
        return True

    try:
        ev_df.to_csv(events_tsv, sep="\t", index=False)
        run_text = f" run-{run_num}" if run_num is not None else ""
        log.info(
            "Merged behavior -> events for sub-%s%s: %s using %s",
            sub_label,
            run_text,
            events_tsv,
            beh_csv.name,
        )
        return True
    except OSError as exc:
        log.error("Failed writing events: %s -> %s", events_tsv, exc)
        return False


def run_raw_to_bids(
    source_root: Path,
    bids_root: Path,
    task: str,
    subjects: Optional[List[str]] = None,
    montage: str = "easycap-M1",
    line_freq: float = 60.0,
    overwrite: bool = False,
    zero_base_onsets: bool = False,
    do_trim_to_first_volume: bool = False,
    event_prefixes: Optional[List[str]] = None,
    keep_all_annotations: bool = False,
    *,
    _logger: Optional[logging.Logger] = None,
) -> int:
    """Convert raw BrainVision files to BIDS format."""
    log = _logger or logger

    from mne_bids import BIDSPath, write_raw_bids

    log.info("Scanning for BrainVision files in: %s", source_root)
    vhdrs = find_brainvision_vhdrs(source_root)
    if not vhdrs:
        log.error("No .vhdr files found under sub-*/eeg/. Nothing to convert.")
        return 0

    if subjects:
        subj_set = set(subjects)
        vhdrs = [p for p in vhdrs if parse_subject_id(p) in subj_set]
        if not vhdrs:
            log.error("No matching .vhdr files for subjects: %s", sorted(subj_set))
            return 0

    ensure_dataset_description(bids_root, name=f"{task} EEG")
    montage_name = montage if montage else None

    for i, vhdr in enumerate(vhdrs, 1):
        subject_label = parse_subject_id(vhdr)
        run_index = get_run_index(vhdr)

        raw = mne.io.read_raw_brainvision(vhdr, preload=False, verbose=False)
        set_channel_types(raw)

        if montage_name:
            set_montage(raw, montage_name)

        raw.info["line_freq"] = line_freq

        was_trimmed = False
        if do_trim_to_first_volume:
            was_trimmed = trim_to_first_volume(raw)

        if was_trimmed and not raw.preload:
            raw.load_data()

        filter_annotations(raw, event_prefixes, keep_all_annotations, zero_base_onsets)

        bids_path = BIDSPath(
            subject=subject_label,
            task=task,
            run=run_index,
            datatype="eeg",
            suffix="eeg",
            root=bids_root,
        )

        signature = inspect.signature(write_raw_bids)
        parameters = signature.parameters
        kwargs: dict = {}
        if "allow_preload" in parameters:
            kwargs["allow_preload"] = bool(getattr(raw, "preload", False))
        if "format" in parameters:
            kwargs["format"] = "BrainVision"
        if "verbose" in parameters:
            kwargs["verbose"] = False

        write_raw_bids(
            raw=raw,
            bids_path=bids_path,
            overwrite=overwrite,
            **kwargs,
        )

        log.info("[%d/%d] Wrote: sub-%s", i, len(vhdrs), subject_label)

    log.info("Done. Converted %d file(s) to BIDS in: %s", len(vhdrs), bids_root)
    return len(vhdrs)


def run_merge_behavior(
    bids_root: Path,
    source_root: Path,
    task: str,
    event_prefixes: Optional[List[str]] = None,
    event_types: Optional[List[str]] = None,
    dry_run: bool = False,
    *,
    _logger: Optional[logging.Logger] = None,
) -> int:
    """Merge behavioral data into BIDS events files."""
    log = _logger or logger

    pattern_run = f"sub-*/eeg/*_task-{task}_run-*_events.tsv"
    ev_paths = sorted(bids_root.glob(pattern_run))
    if not ev_paths:
        pattern = f"sub-*/eeg/*_task-{task}_events.tsv"
        ev_paths = sorted(bids_root.glob(pattern))
        if not ev_paths:
            log.info("No events found under %s for task '%s'", bids_root, task)
            return 0

    n_ok = 0
    eeg_dirs = []
    for ev in ev_paths:
        ok = merge_behavior_to_events(
            ev,
            source_root=source_root,
            event_prefixes=event_prefixes,
            event_types=event_types,
            dry_run=dry_run,
            _logger=log,
        )
        n_ok += int(ok)
        eeg_dirs.append(ev.parent)

    if not dry_run:
        seen = set()
        for d in eeg_dirs:
            if d in seen:
                continue
            seen.add(d)
            combine_runs_for_subject(d, task=task)

    log.info("Done. Processed %d event file(s), merged successfully: %d.", len(ev_paths), n_ok)
    return n_ok
