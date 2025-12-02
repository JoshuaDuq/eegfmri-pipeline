"""
Preprocessing Pipeline (Canonical)
==================================

High-level preprocessing orchestration functions:
- run_raw_to_bids: Convert raw BrainVision files to BIDS format
- run_merge_behavior: Merge behavioral data into BIDS events files

Helper functions for file discovery, run indexing, and data manipulation
are in eeg_pipeline.utils.data.preprocessing.
"""

from __future__ import annotations

import re
import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd
import mne

from eeg_pipeline.utils.data.preprocessing import (
    find_brainvision_vhdrs,
    parse_subject_id,
    extract_run_number,
    get_run_index,
    normalize_string,
    normalize_event_filters,
    find_behavior_csv_for_run,
    create_event_mask,
    combine_runs_for_subject,
    trim_to_first_volume,
    filter_annotations,
    set_channel_types,
    set_montage,
    ensure_dataset_description,
)

logger = logging.getLogger(__name__)


###################################################################
# Merge Functions
###################################################################


def merge_behavior_to_events(
    events_tsv: Path,
    source_root: Path,
    event_prefixes: Optional[List[str]] = None,
    event_types: Optional[List[str]] = None,
    dry_run: bool = False,
) -> bool:
    from eeg_pipeline.utils.data.loading import trim_behavioral_to_events_strict
    
    m = re.search(r"sub-([A-Za-z0-9]+)", str(events_tsv))
    if not m:
        logger.warning("Could not parse subject from: %s", events_tsv)
        return False
    sub_label = m.group(1)

    run_num = extract_run_number(events_tsv)
    beh_csv = find_behavior_csv_for_run(source_root / f"sub-{sub_label}", run=run_num)

    if not beh_csv or not beh_csv.exists():
        if run_num is None:
            logger.warning(
                "No TrialSummary.csv found for sub-%s under %s/sub-%s/PsychoPy_Data",
                sub_label, source_root, sub_label
            )
        else:
            logger.warning(
                "No TrialSummary.csv matching run %d found for sub-%s under %s/sub-%s/PsychoPy_Data",
                run_num, sub_label, source_root, sub_label
            )
        return False

    try:
        ev_df = pd.read_csv(events_tsv, sep="\t")
    except (pd.errors.ParserError, OSError) as e:
        logger.error("Failed reading events: %s -> %s", events_tsv, e)
        return False

    try:
        beh_df = pd.read_csv(beh_csv)
    except (pd.errors.ParserError, OSError) as e:
        logger.error("Failed reading behavior: %s -> %s", beh_csv, e)
        return False

    if "trial_type" not in ev_df.columns:
        logger.warning("'trial_type' column missing in events: %s", events_tsv)
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
        logger.warning("No target events in: %s (criteria: %s)", events_tsv, criteria_description)
        return False

    target_events_df = ev_df.iloc[target_indices].copy()
    try:
        behavioral_subset = trim_behavioral_to_events_strict(beh_df, target_events_df)
    except ValueError as e:
        run_text = f"run-{run_num} " if run_num is not None else ""
        logger.error("Behavioral/events mismatch for sub-%s %s: %s", sub_label, run_text, e)
        return False

    n_matched = len(behavioral_subset)
    event_rows_to_update = target_indices[:n_matched]

    for column in behavioral_subset.columns:
        if column not in ev_df.columns:
            ev_df[column] = pd.NA
        ev_df.loc[event_rows_to_update, column] = behavioral_subset[column].values

    if dry_run:
        logger.info(
            "[dry-run] Would update: %s with columns: %s from %s",
            events_tsv, list(behavioral_subset.columns), beh_csv.name
        )
        return True

    try:
        ev_df.to_csv(events_tsv, sep="\t", index=False)
        run_text = f" run-{run_num}" if run_num is not None else ""
        logger.info(
            "Merged behavior -> events for sub-%s%s: %s using %s",
            sub_label, run_text, events_tsv, beh_csv.name
        )
        return True
    except OSError as e:
        logger.error("Failed writing events: %s -> %s", events_tsv, e)
        return False


###################################################################
# High-Level Preprocessing Functions
###################################################################


def run_raw_to_bids(
    source_root: Path,
    bids_root: Path,
    task: str,
    subjects: Optional[List[str]] = None,
    montage: str = "easycap-M1",
    line_freq: float = 60.0,
    overwrite: bool = False,
    zero_base_onsets: bool = False,
    trim_to_first_volume: bool = False,
    event_prefixes: Optional[List[str]] = None,
    keep_all_annotations: bool = False,
) -> int:
    import inspect
    from mne_bids import BIDSPath, write_raw_bids
    
    logger.info("Scanning for BrainVision files in: %s", source_root)
    vhdrs = find_brainvision_vhdrs(source_root)
    if not vhdrs:
        logger.error("No .vhdr files found under sub-*/eeg/. Nothing to convert.")
        return 0
    
    if subjects:
        subj_set = set(subjects)
        vhdrs = [p for p in vhdrs if parse_subject_id(p) in subj_set]
        if not vhdrs:
            logger.error("No matching .vhdr files for subjects: %s", sorted(subj_set))
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
        if trim_to_first_volume:
            from eeg_pipeline.utils.data.preprocessing import trim_to_first_volume as _trim
            was_trimmed = _trim(raw)
        
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
        kwargs = {}
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
        
        logger.info("[%d/%d] Wrote: sub-%s", i, len(vhdrs), subject_label)
    
    logger.info("Done. Converted %d file(s) to BIDS in: %s", len(vhdrs), bids_root)
    return len(vhdrs)


def run_merge_behavior(
    bids_root: Path,
    source_root: Path,
    task: str,
    event_prefixes: Optional[List[str]] = None,
    event_types: Optional[List[str]] = None,
    dry_run: bool = False,
) -> int:
    pattern_run = f"sub-*/eeg/*_task-{task}_run-*_events.tsv"
    ev_paths = sorted(bids_root.glob(pattern_run))
    if not ev_paths:
        pattern = f"sub-*/eeg/*_task-{task}_events.tsv"
        ev_paths = sorted(bids_root.glob(pattern))
        if not ev_paths:
            logger.info("No events found under %s for task '%s'", bids_root, task)
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
    
    logger.info("Done. Processed %d event file(s), merged successfully: %d.", len(ev_paths), n_ok)
    return n_ok


###################################################################
# Exports
###################################################################

__all__ = [
    "run_raw_to_bids",
    "run_merge_behavior",
    "merge_behavior_to_events",
    "find_brainvision_vhdrs",
    "parse_subject_id",
    "extract_run_number",
    "get_run_index",
    "normalize_string",
    "normalize_event_filters",
    "find_behavior_csv_for_run",
    "create_event_mask",
    "combine_runs_for_subject",
    "set_channel_types",
    "set_montage",
    "ensure_dataset_description",
]
