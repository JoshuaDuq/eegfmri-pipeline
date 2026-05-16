#!/usr/bin/env python3
"""One-off repair for restart artifacts in EEG events.tsv.

Use this when PsychoPy restarted mid-run and EEG continued recording, leaving
extra task trigger rows in BIDS events.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()


def _find_behavior_csv_for_run(source_sub_dir: Path, run: int) -> Optional[Path]:
    psychopy_dir = source_sub_dir / "PsychoPy_Data"
    if not psychopy_dir.exists():
        return None

    csvs = sorted(
        path
        for path in psychopy_dir.glob("*TrialSummary.csv")
        if path.is_file() and not path.name.startswith("._")
    )
    if not csvs:
        return None

    pat = re.compile(rf"run-?{run}(?:[^0-9]|$)", flags=re.IGNORECASE)
    candidates = [c for c in csvs if pat.search(c.name)]
    if not candidates:
        return None

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _find_best_window(
    event_onsets: list[float],
    behavior_onsets: list[float],
) -> Optional[tuple[int, int, float, float]]:
    n_events = len(event_onsets)
    n_behavior = len(behavior_onsets)
    if n_behavior < 3 or n_events <= n_behavior:
        return None

    beh_d = pd.Series(behavior_onsets).diff().dropna().to_numpy()
    if len(beh_d) < 2:
        return None

    best: Optional[tuple[int, int, float, float]] = None
    for start in range(0, n_events - n_behavior + 1):
        stop = start + n_behavior
        window = event_onsets[start:stop]
        win_d = pd.Series(window).diff().dropna().to_numpy()
        if len(win_d) != len(beh_d):
            continue
        abs_err = abs(win_d - beh_d)
        median_abs_err = float(pd.Series(abs_err).median())
        p95_abs_err = float(pd.Series(abs_err).quantile(0.95))
        candidate = (start, stop - 1, median_abs_err, p95_abs_err)
        if best is None or (candidate[2], candidate[3], candidate[0]) < (best[2], best[3], best[0]):
            best = candidate
    return best


def _load_numeric(series: pd.Series) -> Optional[list[float]]:
    values = pd.to_numeric(series, errors="coerce")
    if values.isna().any():
        return None
    return [float(v) for v in values.tolist()]


def _run_number_from_events_path(events_path: Path) -> int:
    match = re.search(r"_run-([0-9]+)_events\.tsv$", events_path.name)
    if match is None:
        raise ValueError(f"Cannot parse run number from events filename: {events_path.name}")
    return int(match.group(1))


def _target_indices_from_events(
    trial_types: pd.Series,
    event_prefix: Optional[str],
) -> tuple[list[int], str]:
    normalized = trial_types.map(_normalize)

    prefix = _normalize(event_prefix or "")
    if prefix:
        return trial_types.index[normalized.str.startswith(prefix)].tolist(), prefix

    excluded_prefixes = (
        "Volume",
        "Pulse",
        "SyncStatus",
        "New Segment",
        "Bad",
        "EDGE",
        "Response",
    )

    mask = normalized.ne("") & normalized.str.lower().ne("n/a")
    for excluded in excluded_prefixes:
        mask &= ~normalized.str.startswith(excluded)
    return trial_types.index[mask].tolist(), "auto(non-housekeeping)"


def repair_events_file(
    events_path: Path,
    behavior_path: Path,
    *,
    event_prefix: Optional[str],
    behavior_onset_col: str,
    bad_prefix: str,
    median_err_threshold_s: float,
    write_backup: bool,
) -> str:
    ev_df = pd.read_csv(events_path, sep="\t")
    beh_df = pd.read_csv(behavior_path)

    if "trial_type" not in ev_df.columns:
        return f"skip {events_path.name}: missing trial_type"
    if "onset" not in ev_df.columns:
        return f"skip {events_path.name}: missing onset"
    if behavior_onset_col not in beh_df.columns:
        return f"skip {events_path.name}: behavior missing {behavior_onset_col}"

    target_indices, target_label = _target_indices_from_events(
        ev_df["trial_type"],
        event_prefix,
    )
    n_events = len(target_indices)
    n_behavior = int(len(beh_df))

    if n_events == 0:
        return f"skip {events_path.name}: no events matching {target_label}"
    if n_events == n_behavior:
        return f"ok   {events_path.name}: already aligned ({n_events})"
    if n_events < n_behavior:
        return f"skip {events_path.name}: fewer events ({n_events}) than behavior ({n_behavior})"

    event_onsets = _load_numeric(ev_df.loc[target_indices, "onset"])
    behavior_onsets = _load_numeric(beh_df[behavior_onset_col])
    if event_onsets is None or behavior_onsets is None:
        return f"skip {events_path.name}: non-numeric onsets prevent interval alignment"

    best = _find_best_window(event_onsets, behavior_onsets)
    if best is None:
        return f"skip {events_path.name}: could not find alignment window"

    start, stop, median_err, p95_err = best
    if median_err > median_err_threshold_s:
        return (
            f"skip {events_path.name}: best window median error {median_err:.3f}s exceeds "
            f"threshold {median_err_threshold_s:.3f}s"
        )

    keep = set(target_indices[start : stop + 1])
    extras = [idx for idx in target_indices if idx not in keep]
    if not extras:
        return f"ok   {events_path.name}: no extra triggers"

    if write_backup:
        backup_path = events_path.with_suffix(".tsv.bak")
        ev_df.to_csv(backup_path, sep="\t", index=False)

    ev_df.loc[extras, "trial_type"] = ev_df.loc[extras, "trial_type"].map(
        lambda s: f"{bad_prefix}{s}"
    )
    ev_df.to_csv(events_path, sep="\t", index=False)

    return (
        f"fix  {events_path.name}: relabeled {len(extras)} extra trigger(s) "
        f"(window {start + 1}-{stop + 1}, median={median_err:.3f}s, p95={p95_err:.3f}s)"
    )


def _discover_runs(eeg_dir: Path, task: str, run: Optional[int]) -> Iterable[Path]:
    if run is not None:
        p = eeg_dir / f"{eeg_dir.parent.name}_task-{task}_run-{run}_events.tsv"
        if p.exists():
            yield p
        return
    yield from sorted(eeg_dir.glob(f"{eeg_dir.parent.name}_task-{task}_run-*_events.tsv"))


def _get_union_columns(frames: list[tuple[int, pd.DataFrame, Path]]) -> list[str]:
    columns: list[str] = []
    for _, dataframe, _ in frames:
        for column in dataframe.columns:
            if column not in columns:
                columns.append(column)
    return columns


def _add_run_id_column(dataframe: pd.DataFrame, run_number: int) -> None:
    if "run_id" not in dataframe.columns and "run" not in dataframe.columns:
        dataframe.insert(0, "run_id", run_number)
    elif "run_id" in dataframe.columns and dataframe["run_id"].isna().any():
        dataframe["run_id"] = dataframe["run_id"].fillna(run_number)
    elif "run" in dataframe.columns and "run_id" not in dataframe.columns:
        if dataframe["run"].isna().any():
            dataframe["run"] = dataframe["run"].fillna(run_number)


def _update_sample_indices(dataframe: pd.DataFrame, cumulative_offset: int) -> int:
    if "sample" not in dataframe.columns:
        return cumulative_offset

    sample_numeric = pd.to_numeric(dataframe["sample"], errors="coerce")
    if not sample_numeric.notna().any():
        return cumulative_offset

    adjusted_sample = sample_numeric + cumulative_offset
    dataframe["sample"] = adjusted_sample
    return int(adjusted_sample.max()) + 1


def _write_combined_events_from_runs(eeg_dir: Path, task: str, *, write_backup: bool) -> Path:
    run_files = list(_discover_runs(eeg_dir, task, run=None))
    if not run_files:
        raise FileNotFoundError(f"No run-level events files found under {eeg_dir}")

    frames = [
        (_run_number_from_events_path(path), pd.read_csv(path, sep="\t"), path)
        for path in run_files
    ]
    frames.sort(key=lambda frame: frame[0])
    union_columns = _get_union_columns(frames)

    parts: list[pd.DataFrame] = []
    cumulative_sample_offset = 0
    for run_number, dataframe, _ in frames:
        for column in union_columns:
            if column not in dataframe.columns:
                dataframe[column] = pd.NA
        dataframe = dataframe[union_columns].copy()
        _add_run_id_column(dataframe, run_number)
        cumulative_sample_offset = _update_sample_indices(dataframe, cumulative_sample_offset)
        parts.append(dataframe)

    combined = pd.concat(parts, axis=0, ignore_index=True)
    combined_path = eeg_dir / f"{eeg_dir.parent.name}_task-{task}_events.tsv"
    if write_backup and combined_path.exists():
        shutil.copy2(combined_path, combined_path.with_suffix(".tsv.bak"))
    combined.to_csv(combined_path, sep="\t", index=False)
    return combined_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repair extra restart triggers in BIDS events.tsv")
    parser.add_argument("--source-root", required=True, help="Path to source_data root")
    parser.add_argument("--bids-root", required=True, help="Path to BIDS EEG root")
    parser.add_argument(
        "--subject", required=True, help="Subject without 'sub-' prefix (e.g., 0002)"
    )
    parser.add_argument("--task", required=True, help="Task label")
    parser.add_argument(
        "--run", type=int, default=None, help="Run number to repair (default: all runs)"
    )
    parser.add_argument(
        "--event-prefix",
        default=None,
        help="Optional trial_type prefix to align (default: auto-detect non-housekeeping events)",
    )
    parser.add_argument(
        "--behavior-onset-col", default="stim_start_time", help="Behavior onset column"
    )
    parser.add_argument(
        "--bad-prefix", default="BAD_restart/", help="Prefix used to relabel extra triggers"
    )
    parser.add_argument(
        "--median-err-threshold-s",
        type=float,
        default=0.25,
        help="Max median interval error allowed for automatic repair",
    )
    parser.add_argument("--no-backup", action="store_true", help="Do not write .tsv.bak backup")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    subject = args.subject if str(args.subject).startswith("sub-") else f"sub-{args.subject}"
    source_sub_dir = Path(args.source_root) / subject
    eeg_dir = Path(args.bids_root) / subject / "eeg"

    if not eeg_dir.exists():
        print(f"error: missing EEG directory: {eeg_dir}")
        return 2

    paths = list(_discover_runs(eeg_dir, args.task, args.run))
    if not paths:
        print(f"error: no run events files found for {subject} task-{args.task}")
        return 2

    repaired = 0
    for events_path in paths:
        run_num = _run_number_from_events_path(events_path)
        behavior_path = _find_behavior_csv_for_run(source_sub_dir, run_num)
        if behavior_path is None:
            print(f"skip {events_path.name}: no TrialSummary for run {run_num}")
            continue

        msg = repair_events_file(
            events_path=events_path,
            behavior_path=behavior_path,
            event_prefix=args.event_prefix,
            behavior_onset_col=args.behavior_onset_col,
            bad_prefix=args.bad_prefix,
            median_err_threshold_s=float(args.median_err_threshold_s),
            write_backup=not bool(args.no_backup),
        )
        print(msg)
        if msg.startswith("fix  "):
            repaired += 1

    combined_path = _write_combined_events_from_runs(
        eeg_dir,
        args.task,
        write_backup=not bool(args.no_backup),
    )
    print(f"sync {combined_path.name}: regenerated from run-level events")

    return 0


if __name__ == "__main__":
    sys.exit(main())
