import os
import re
import sys
import logging
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

from eeg_pipeline.utils.config.loader import load_settings
from eeg_pipeline.utils.data.loading import trim_behavioral_to_events_strict

logger = logging.getLogger(__name__)

config = load_settings()
BIDS_ROOT = config.bids_root
SOURCE_ROOT = Path(config.get("paths.source_data", "data/source_data"))
TASK = config.get("project.task", "thermalactive")


###################################################################
# Helper Functions
###################################################################

def _norm_trial_type(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()


def _normalize_event_filters(filters: Optional[List[str]]) -> Optional[List[str]]:
    if filters in (None, [], [None]):
        return None
    normalized = [_norm_trial_type(f) for f in filters if str(f).strip() != ""]
    return normalized if normalized else None


def _create_event_mask(
    normalized_trial_types: pd.Series,
    prefixes: Optional[List[str]],
    types: Optional[List[str]]
) -> pd.Series:
    mask = pd.Series(False, index=normalized_trial_types.index)
    if prefixes:
        for prefix in prefixes:
            mask = mask | normalized_trial_types.str.startswith(prefix)
    if types:
        mask = mask | normalized_trial_types.isin(types)
    return mask


def _format_selection_criteria(
    prefixes: Optional[List[str]],
    types: Optional[List[str]]
) -> str:
    criteria_parts = []
    if prefixes:
        criteria_parts.append(f"prefixes={prefixes}")
    if types:
        criteria_parts.append(f"types={types}")
    return "; ".join(criteria_parts) if criteria_parts else "<none>"


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
        logger.warning("Could not parse subject from: %s", events_tsv)
        return False
    sub_label = m.group(1)

    run_num = _extract_run_number_from_path(events_tsv)
    beh_csv = _find_behavior_csv_for_run(source_root / f"sub-{sub_label}", run=run_num)

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

    normalized_trial_types = ev_df["trial_type"].map(_norm_trial_type)
    
    normalized_prefixes = _normalize_event_filters(event_prefixes)
    normalized_types = _normalize_event_filters(event_types)
    
    if not normalized_prefixes and not normalized_types:
        normalized_prefixes = [_norm_trial_type("Trig_therm/T  1")]

    event_mask = _create_event_mask(normalized_trial_types, normalized_prefixes, normalized_types)
    target_indices = ev_df.index[event_mask].tolist()
    
    if len(target_indices) == 0:
        criteria_description = _format_selection_criteria(normalized_prefixes, normalized_types)
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


def _load_run_files(run_files: List[Path]) -> List[tuple[int, pd.DataFrame, Path]]:
    frames = []
    for file_path in run_files:
        run_number = _extract_run_number_from_path(file_path)
        if run_number is None:
            continue
        try:
            dataframe = pd.read_csv(file_path, sep="\t")
        except (pd.errors.ParserError, OSError) as e:
            logger.warning("Skipping run file due to read error: %s -> %s", file_path, e)
            continue
        if "onset" in dataframe.columns:
            dataframe = dataframe.sort_values("onset", kind="mergesort")
        frames.append((run_number, dataframe, file_path))
    return frames


def _get_union_columns(frames: List[tuple[int, pd.DataFrame, Path]]) -> List[str]:
    union_columns = []
    for _, dataframe, _ in frames:
        for column in dataframe.columns:
            if column not in union_columns:
                union_columns.append(column)
    return union_columns


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
    if sample_numeric.notna().any():
        if cumulative_offset > 0:
            dataframe["sample"] = sample_numeric + cumulative_offset
        max_sample = int(sample_numeric.max()) if cumulative_offset == 0 else int((sample_numeric + cumulative_offset).max())
        return max_sample + 1
    
    if "onset" in dataframe.columns:
        onset_numeric = pd.to_numeric(dataframe["onset"], errors="coerce")
        if onset_numeric.notna().any():
            max_onset = float(onset_numeric.max())
            sample_numeric = pd.to_numeric(dataframe["sample"], errors="coerce")
            if sample_numeric.notna().any() and onset_numeric.notna().any() and max_onset > 0:
                sampling_rate_estimate = float(sample_numeric.max() / max_onset)
                if sampling_rate_estimate > 0:
                    return int(max_onset * sampling_rate_estimate) + 1
    
    return cumulative_offset


def _get_sort_columns(combined_df: pd.DataFrame) -> List[str]:
    if "onset" in combined_df.columns:
        if "run_id" in combined_df.columns:
            return ["run_id", "onset"]
        if "run" in combined_df.columns:
            return ["run", "onset"]
        return ["onset"]
    
    if "run_id" in combined_df.columns:
        return ["run_id"]
    if "run" in combined_df.columns:
        return ["run"]
    return []


def _combine_runs_for_subject(sub_eeg_dir: Path, task: str) -> Optional[Path]:
    run_files = sorted(sub_eeg_dir.glob(f"*_task-{task}_run-*_events.tsv"))
    if not run_files:
        return None

    frames = _load_run_files(run_files)
    if not frames:
        return None

    frames.sort(key=lambda t: t[0])
    n_runs = len({r for r, _, _ in frames})
    union_columns = _get_union_columns(frames)

    dataframes = []
    cumulative_sample_offset = 0
    
    for run_number, dataframe, _ in frames:
        for column in union_columns:
            if column not in dataframe.columns:
                dataframe[column] = pd.NA
        dataframe = dataframe[union_columns]
        
        _add_run_id_column(dataframe, run_number)
        cumulative_sample_offset = _update_sample_indices(dataframe, cumulative_sample_offset)
        
        dataframes.append(dataframe)

    combined = pd.concat(dataframes, axis=0, ignore_index=True)
    sort_columns = _get_sort_columns(combined)
    if sort_columns:
        combined = combined.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)

    sub_prefix = sub_eeg_dir.parent.name
    out_path = sub_eeg_dir / f"{sub_prefix}_task-{task}_events.tsv"

    try:
        combined.to_csv(out_path, sep="\t", index=False)
        logger.info("Wrote combined events (%d run(s), %d rows): %s", n_runs, len(combined), out_path)
        return out_path
    except OSError as e:
        logger.error("Failed writing combined events for %s: %s", sub_prefix, e)
        return None


###################################################################
# Main Entry Point
###################################################################

def main():
    import argparse
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

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
            logger.info(
                "No events found under %s for task '%s' with patterns %s or %s",
                bids_root, task, pattern_run, pattern
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

    logger.info("Done. Processed %d event file(s), merged successfully: %d.", len(ev_paths), n_ok)


if __name__ == "__main__":
    main()
