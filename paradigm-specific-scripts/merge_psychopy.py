"""Merge PsychoPy TrialSummary.csv into BIDS *_events.tsv (paradigm-specific)."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List, Optional

import pandas as pd

from eeg_pipeline.analysis.utilities.bids_metadata import ensure_events_sidecar, ensure_task_events_json
from eeg_pipeline.infra.tsv import read_tsv
from eeg_pipeline.utils.data.preprocessing import (
    combine_runs_for_subject,
    create_event_mask,
    extract_run_number,
    find_behavior_csv_for_run,
    normalize_event_filters,
    normalize_string,
)

logger = logging.getLogger(__name__)


def _validate_and_trim_psychopy_rows(
    psychopy_df: pd.DataFrame,
    events_df: pd.DataFrame,
    *,
    allow_misaligned_trim: bool,
    log: logging.Logger,
) -> pd.DataFrame:
    n_psychopy_rows = len(psychopy_df)
    n_event_rows = len(events_df)

    if n_psychopy_rows == n_event_rows:
        return psychopy_df.reset_index(drop=True)

    msg = (
        f"PsychoPy/events count mismatch: psychopy={n_psychopy_rows} "
        f"events={n_event_rows}. This can happen if trigger filtering "
        "misses trials, triggers are duplicated, or the wrong behavioral file was selected."
    )
    if not allow_misaligned_trim:
        raise ValueError(msg)

    log.warning("%s Proceeding because allow_misaligned_trim=True.", msg)
    if n_psychopy_rows > n_event_rows:
        return psychopy_df.iloc[:n_event_rows].reset_index(drop=True)
    return psychopy_df.reindex(range(n_event_rows)).reset_index(drop=True)


def _qc_compare_trial_intervals(
    target_events_df: pd.DataFrame,
    psychopy_df: pd.DataFrame,
    *,
    log: logging.Logger,
    warn_threshold_s: float = 0.25,
) -> None:
    """QC: compare inter-trial intervals between EEG triggers and PsychoPy times."""
    if "onset" not in target_events_df.columns:
        return
    if "stim_start_time" not in psychopy_df.columns:
        return

    try:
        eeg_onsets = pd.to_numeric(target_events_df["onset"], errors="coerce").to_numpy()
        beh_onsets = pd.to_numeric(psychopy_df["stim_start_time"], errors="coerce").to_numpy()
    except Exception:
        return

    if len(eeg_onsets) < 3 or len(beh_onsets) < 3:
        return
    if not (pd.notna(eeg_onsets).all() and pd.notna(beh_onsets).all()):
        return

    eeg_d = pd.Series(eeg_onsets).diff().dropna().to_numpy()
    beh_d = pd.Series(beh_onsets).diff().dropna().to_numpy()
    if len(eeg_d) != len(beh_d) or len(eeg_d) < 2:
        return

    abs_err = abs(eeg_d - beh_d)
    median_abs_err = float(pd.Series(abs_err).median())
    p95_abs_err = float(pd.Series(abs_err).quantile(0.95))

    if median_abs_err > warn_threshold_s:
        log.warning(
            "QC: inter-trial interval mismatch (median abs err=%.3fs, p95=%.3fs). "
            "This suggests the selected trigger may not correspond to PsychoPy stim_start_time, "
            "or run synchronization differs.",
            median_abs_err,
            p95_abs_err,
        )
    else:
        log.info(
            "QC: inter-trial intervals match well (median abs err=%.3fs, p95=%.3fs).",
            median_abs_err,
            p95_abs_err,
        )


def merge_psychopy_to_events(
    events_tsv: Path,
    source_root: Path,
    event_prefixes: Optional[List[str]] = None,
    event_types: Optional[List[str]] = None,
    dry_run: bool = False,
    allow_misaligned_trim: bool = False,
    *,
    _logger: Optional[logging.Logger] = None,
) -> bool:
    """Merge behavioral data into a single events.tsv file."""
    log = _logger or logger

    m = re.search(r"sub-([A-Za-z0-9]+)", str(events_tsv))
    if not m:
        log.warning("Could not parse subject from: %s", events_tsv)
        return False
    sub_label = m.group(1)

    run_num = extract_run_number(events_tsv)
    beh_csv = find_behavior_csv_for_run(
        source_root / f"sub-{sub_label}",
        run=run_num,
        behavior_dir_name="PsychoPy_Data",
        glob_pattern="*TrialSummary.csv",
    )

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
        psychopy_df = pd.read_csv(beh_csv)
    except (pd.errors.ParserError, OSError) as exc:
        log.error("Failed reading behavior: %s -> %s", beh_csv, exc)
        return False

    if run_num is not None and "run_id" in psychopy_df.columns:
        unique_runs = sorted(
            {int(r) for r in pd.to_numeric(psychopy_df["run_id"], errors="coerce").dropna().unique()}
        )
        if unique_runs and unique_runs != [int(run_num)]:
            message = f"Behavior run_id mismatch for {beh_csv.name}: found {unique_runs}, expected {run_num}"
            if not allow_misaligned_trim:
                log.error(message)
                return False
            log.warning("%s (allow_misaligned_trim=True)", message)

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
        psychopy_subset = _validate_and_trim_psychopy_rows(
            psychopy_df,
            target_events_df,
            allow_misaligned_trim=allow_misaligned_trim,
            log=log,
        )
    except ValueError as exc:
        run_text = f"run-{run_num} " if run_num is not None else ""
        log.error("PsychoPy/events mismatch for sub-%s %s: %s", sub_label, run_text, exc)
        return False

    _qc_compare_trial_intervals(target_events_df, psychopy_subset, log=log)

    n_matched = len(psychopy_subset)
    event_rows_to_update = target_indices[:n_matched]

    for column in psychopy_subset.columns:
        if column not in ev_df.columns:
            ev_df[column] = pd.NA
        ev_df.loc[event_rows_to_update, column] = psychopy_subset[column].values

    if run_num is not None:
        if "run_id" not in ev_df.columns:
            ev_df["run_id"] = int(run_num)
        else:
            ev_df["run_id"] = pd.to_numeric(ev_df["run_id"], errors="coerce").fillna(int(run_num))

    if "onset" in ev_df.columns:
        sort_cols = ["onset"]
        if "sample" in ev_df.columns:
            sort_cols.append("sample")
        ev_df = ev_df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    if dry_run:
        log.info(
            "[dry-run] Would update: %s with columns: %s from %s",
            events_tsv,
            list(psychopy_subset.columns),
            beh_csv.name,
        )
        return True

    try:
        ev_df.to_csv(events_tsv, sep="\t", index=False)
        ensure_events_sidecar(events_tsv, list(ev_df.columns))
        run_text = f" run-{run_num}" if run_num is not None else ""
        log.info(
            "Merged PsychoPy -> events for sub-%s%s: %s using %s",
            sub_label,
            run_text,
            events_tsv,
            beh_csv.name,
        )
        return True
    except OSError as exc:
        log.error("Failed writing events: %s -> %s", events_tsv, exc)
        return False


def run_merge_psychopy(
    bids_root: Path,
    source_root: Path,
    task: str,
    subjects: Optional[List[str]] = None,
    event_prefixes: Optional[List[str]] = None,
    event_types: Optional[List[str]] = None,
    dry_run: bool = False,
    allow_misaligned_trim: bool = False,
    *,
    _logger: Optional[logging.Logger] = None,
) -> int:
    """Merge PsychoPy data into BIDS events files."""
    log = _logger or logger

    pattern_run = f"sub-*/eeg/*_task-{task}_run-*_events.tsv"
    ev_paths = sorted(p for p in bids_root.glob(pattern_run) if not p.name.startswith("._"))
    if not ev_paths:
        pattern = f"sub-*/eeg/*_task-{task}_events.tsv"
        ev_paths = sorted(p for p in bids_root.glob(pattern) if not p.name.startswith("._"))
        if not ev_paths:
            log.info("No events found under %s for task '%s'", bids_root, task)
            return 0

    if subjects:
        subj_set = {f"sub-{s}" if not s.startswith("sub-") else s for s in subjects}
        ev_paths = [p for p in ev_paths if p.parts[-3] in subj_set]
        if not ev_paths:
            log.info("No matching events found for subjects: %s", sorted(subj_set))
            return 0

    n_ok = 0
    eeg_dirs = []
    for ev in ev_paths:
        ok = merge_psychopy_to_events(
            ev,
            source_root=source_root,
            event_prefixes=event_prefixes,
            event_types=event_types,
            dry_run=dry_run,
            allow_misaligned_trim=allow_misaligned_trim,
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
            combined = combine_runs_for_subject(d, task=task)
            if combined is not None:
                ensure_events_sidecar(combined, list(read_tsv(combined).columns))

        ensure_task_events_json(bids_root, task=task)

    log.info("Done. Processed %d event file(s), merged successfully: %d.", len(ev_paths), n_ok)
    return n_ok
