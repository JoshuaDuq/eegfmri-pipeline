import os
import re
import sys
from pathlib import Path
from typing import Optional, List
import pandas as pd

os.environ["PYTHONUTF8"] = "1"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from eeg_pipeline.utils.config_loader import EEGConfig
from eeg_pipeline.utils.io_utils import trim_behavioral_to_events_strict

config = EEGConfig()
PROJECT_ROOT = config.project.root
BIDS_ROOT = config.project.bids_root
SOURCE_ROOT = config.project.source_root
TASK = config.project.task


###################################################################
# Helper Functions
###################################################################

def _norm_trial_type(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()


def _extract_run_number_from_path(p: Path) -> Optional[int]:
    s = str(p)
    m = re.search(r"run-?(\d+)", s, flags=re.IGNORECASE)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return None
    return None


def _find_behavior_csv_for_run(source_sub_dir: Path, run: Optional[int]) -> Optional[Path]:
    psychopy_dir = source_sub_dir / "PsychoPy_Data"
    if not psychopy_dir.exists():
        return None

    csvs: List[Path] = sorted(psychopy_dir.glob("*TrialSummary.csv"))
    if not csvs:
        return None
    if run is None:
        return csvs[0]

    candidates: List[Path] = []
    pat = re.compile(rf"run-?{run}(?:[^0-9]|$)", flags=re.IGNORECASE)
    for c in csvs:
        if pat.search(c.name):
            candidates.append(c)

    if not candidates:
        return None

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


###################################################################
# Main Merge Functions
###################################################################

def merge_one_subject_events(
    events_tsv: Path,
    source_root: Path,
    event_prefixes: Optional[List[str]] = None,
    event_types: Optional[List[str]] = None,
    dry_run: bool = False,
) -> bool:
    m = re.search(r"sub-([A-Za-z0-9]+)", str(events_tsv))
    if not m:
        print(f"[skip] Could not parse subject from: {events_tsv}")
        return False
    sub_label = m.group(1)

    run_num = _extract_run_number_from_path(events_tsv)
    beh_csv = _find_behavior_csv_for_run(source_root / f"sub-{sub_label}", run=run_num)

    if not beh_csv or not beh_csv.exists():
        if run_num is None:
            print(
                f"[warn] No TrialSummary.csv found for sub-{sub_label} under "
                f"{source_root}/sub-{sub_label}/PsychoPy_Data"
            )
        else:
            print(
                f"[warn] No TrialSummary.csv matching run {run_num} found for sub-{sub_label} under "
                f"{source_root}/sub-{sub_label}/PsychoPy_Data"
            )
        return False

    try:
        ev_df = pd.read_csv(events_tsv, sep="\t")
    except Exception as e:
        print(f"[error] Failed reading events: {events_tsv} -> {e}")
        return False

    try:
        beh_df = pd.read_csv(beh_csv)
    except Exception as e:
        print(f"[error] Failed reading behavior: {beh_csv} -> {e}")
        return False

    if "trial_type" not in ev_df.columns:
        print(f"[warn] 'trial_type' column missing in events: {events_tsv}")
        return False

    norm_tt = ev_df["trial_type"].map(_norm_trial_type)
    prefixes = None if event_prefixes in (None, [], [None]) else [
        _norm_trial_type(p) for p in event_prefixes if str(p).strip() != ""
    ]
    types = None if event_types in (None, [], [None]) else [
        _norm_trial_type(t) for t in event_types if str(t).strip() != ""
    ]
    if not prefixes and not types:
        prefixes = ["Trig_therm/T  1"]

    mask = pd.Series(False, index=ev_df.index)
    if prefixes:
        for p in prefixes:
            mask = mask | norm_tt.str.startswith(p)
    if types:
        mask = mask | norm_tt.isin(types)

    target_idx = ev_df.index[mask].tolist()
    if len(target_idx) == 0:
        sel_desc = []
        if prefixes:
            sel_desc.append(f"prefixes={prefixes}")
        if types:
            sel_desc.append(f"types={types}")
        crit = "; ".join(sel_desc) if sel_desc else "<none>"
        print(f"[warn] No target events in: {events_tsv} (criteria: {crit})")
        return False

    ev_target_df = ev_df.iloc[target_idx].copy()
    try:
        beh_sub = trim_behavioral_to_events_strict(beh_df, ev_target_df)
    except ValueError as e:
        run_txt = f"run-{run_num} " if run_num is not None else ""
        print(f"[error] Behavioral/events mismatch for sub-{sub_label} {run_txt}: {e}")
        return False

    n = len(beh_sub)
    ev_target_rows = target_idx[:n]

    for col in beh_sub.columns:
        if col not in ev_df.columns:
            ev_df[col] = pd.NA
        ev_df.loc[ev_target_rows, col] = beh_sub[col].values

    if dry_run:
        print(
            f"[dry-run] Would update: {events_tsv} with columns: {list(beh_sub.columns)} "
            f"from {beh_csv.name}"
        )
        return True

    try:
        ev_df.to_csv(events_tsv, sep="\t", index=False)
        run_txt = f" run-{run_num}" if run_num is not None else ""
        print(
            f"[ok] Merged behavior -> events for sub-{sub_label}{run_txt}: {events_tsv} "
            f"using {beh_csv.name}"
        )
        return True
    except Exception as e:
        print(f"[error] Failed writing events: {events_tsv} -> {e}")
        return False


def _combine_runs_for_subject(sub_eeg_dir: Path, task: str) -> Optional[Path]:
    run_files = sorted(sub_eeg_dir.glob(f"*_task-{task}_run-*_events.tsv"))
    if not run_files:
        return None

    frames: List[tuple[int, pd.DataFrame, Path]] = []
    for p in run_files:
        r = _extract_run_number_from_path(p)
        if r is None:
            continue
        try:
            df = pd.read_csv(p, sep="\t")
        except Exception as e:
            print(f"[warn] Skipping run file due to read error: {p} -> {e}")
            continue
        if "onset" in df.columns:
            df = df.sort_values("onset", kind="mergesort")
        frames.append((r, df, p))

    if not frames:
        return None

    frames.sort(key=lambda t: t[0])
    run_set = {r for r, _, _ in frames}
    n_runs = len(run_set)

    union_cols: List[str] = []
    for _, df, _ in frames:
        for c in list(df.columns):
            if c not in union_cols:
                union_cols.append(c)

    dfs: List[pd.DataFrame] = []
    for r, df, _ in frames:
        for c in union_cols:
            if c not in df.columns:
                df[c] = pd.NA
        df = df[union_cols]
        
        if "run_id" not in df.columns and "run" not in df.columns:
            df.insert(0, "run_id", r)
        elif "run_id" in df.columns and df["run_id"].isna().any():
            df["run_id"] = df["run_id"].fillna(r)
        elif "run" in df.columns and "run_id" not in df.columns:
            if df["run"].isna().any():
                df["run"] = df["run"].fillna(r)
        
        dfs.append(df)

    combined = pd.concat(dfs, axis=0, ignore_index=True)
    if "onset" in combined.columns:
        sort_by = ["run_id", "onset"] if "run_id" in combined.columns else ["run", "onset"] if "run" in combined.columns else ["onset"]
        combined = combined.sort_values(sort_by, kind="mergesort").reset_index(drop=True)
    else:
        sort_by = ["run_id"] if "run_id" in combined.columns else ["run"] if "run" in combined.columns else []
        if sort_by:
            combined = combined.sort_values(sort_by, kind="mergesort").reset_index(drop=True)

    sub_prefix = sub_eeg_dir.parent.name
    out_path = sub_eeg_dir / f"{sub_prefix}_task-{task}_events.tsv"

    try:
        combined.to_csv(out_path, sep="\t", index=False)
        print(f"[ok] Wrote combined events ({n_runs} run(s), {len(combined)} rows): {out_path}")
        return out_path
    except Exception as e:
        print(f"[error] Failed writing combined events for {sub_prefix}: {e}")
        return None


###################################################################
# Main Entry Point
###################################################################

def main():
    import argparse

    bids_root = BIDS_ROOT
    source_root = SOURCE_ROOT
    task = TASK

    parser = argparse.ArgumentParser(
        description="Merge behavioral TrialSummary.csv into BIDS events.tsv for each subject"
    )
    parser.add_argument(
        "--bids_root",
        type=str,
        default=str(bids_root),
        help="BIDS root containing sub-*/eeg/*_events.tsv"
    )
    parser.add_argument(
        "--source_root",
        type=str,
        default=str(source_root),
        help="Source root containing sub-*/PsychoPy_Data/*TrialSummary.csv"
    )
    parser.add_argument("--task", type=str, default=task, help="Task label used in events filenames")
    parser.add_argument(
        "--event_prefix",
        action="append",
        default=None,
        help=(
            "Repeatable. Keep only events whose normalized trial_type startswith any provided prefix. "
            "Examples: --event_prefix Trig_therm/T  1 --event_prefix 'Trig_therm/T  1'"
        ),
    )
    parser.add_argument(
        "--event_type",
        action="append",
        default=None,
        help=(
            "Repeatable. Keep only events whose normalized trial_type equals any provided exact value. "
            "Examples: --event_type 'Trig_therm/T  1' --event_type 'Trig_mech/T  2'"
        ),
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Do not write files; just report planned changes"
    )

    args = parser.parse_args()

    bids_root = Path(args.bids_root).resolve()
    source_root = Path(args.source_root).resolve()
    task = args.task

    pattern_run = f"sub-*/eeg/*_task-{task}_run-*_events.tsv"
    ev_paths = sorted(bids_root.glob(pattern_run))
    if not ev_paths:
        pattern = f"sub-*/eeg/*_task-{task}_events.tsv"
        ev_paths = sorted(bids_root.glob(pattern))
        if not ev_paths:
            print(
                f"[info] No events found under {bids_root} for task '{task}' "
                f"with patterns {pattern_run} or {pattern}"
            )
            sys.exit(0)

    n_ok = 0
    eeg_dirs: List[Path] = []
    for ev in ev_paths:
        ok = merge_one_subject_events(
            ev,
            source_root=source_root,
            event_prefixes=args.event_prefix,
            event_types=args.event_type,
            dry_run=args.dry_run,
        )
        n_ok += int(ok)
        eeg_dirs.append(ev.parent)

    if not args.dry_run:
        seen = set()
        for d in eeg_dirs:
            if d in seen:
                continue
            seen.add(d)
            _combine_runs_for_subject(d, task=task)

    print(f"Done. Processed {len(ev_paths)} event file(s), merged successfully: {n_ok}.")


if __name__ == "__main__":
    main()
