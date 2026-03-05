"""
Preprocessing Data Utilities
============================

Helper functions for preprocessing operations:
- File discovery (BrainVision files)
- Run index extraction
- Behavioral data helpers
- Run combination functions
"""

from __future__ import annotations

import re
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import mne

from eeg_pipeline.infra.tsv import read_tsv

logger = logging.getLogger(__name__)


###################################################################
# File Discovery
###################################################################


def find_brainvision_vhdrs(source_root: Path) -> List[Path]:
    vhdrs = sorted(source_root.glob("sub-*/eeg/*.vhdr"))
    return [p for p in vhdrs if p.is_file() and not p.name.startswith("._")]


def parse_subject_id(path: Path) -> str:
    m = re.search(r"sub-([A-Za-z0-9]+)", str(path))
    if not m:
        raise ValueError(f"Could not parse subject from path: {path}")
    return m.group(1)


###################################################################
# Run Index Extraction
###################################################################


def extract_run_number(path: Path) -> Optional[int]:
    match = re.search(r"run[-_]?(\d+)", path.stem, flags=re.IGNORECASE)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    return None


def get_run_index(path: Path) -> Optional[int]:
    run_index = extract_run_number(path)
    if run_index is not None:
        return run_index
    
    all_runs = sorted(path.parent.glob("*.vhdr"))
    if len(all_runs) <= 1:
        return None
    
    inferred_run = all_runs.index(path) + 1
    logger.warning(
        "No explicit run found in filename '%s'. "
        "Inferring run=%d by alphabetical order among %d files. "
        "Prefer 'run-01' style filenames to guarantee correct run IDs.",
        path.name, inferred_run, len(all_runs)
    )
    return inferred_run


###################################################################
# Behavioral Data Helpers
###################################################################


def normalize_string(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()


def normalize_event_filters(filters: Optional[List[str]]) -> Optional[List[str]]:
    if filters in (None, [], [None]):
        return None
    normalized = [normalize_string(f) for f in filters if str(f).strip() != ""]
    return normalized if normalized else None


def find_behavior_csv_for_run(
    source_sub_dir: Path,
    run: Optional[int] = None,
    *,
    behavior_dir_name: str = "behavior",
    glob_pattern: str = "*.csv",
) -> Optional[Path]:
    behavior_dir = source_sub_dir / behavior_dir_name
    if not behavior_dir.exists():
        return None

    csvs: List[Path] = sorted(
        p for p in behavior_dir.glob(glob_pattern) if p.is_file() and not p.name.startswith("._")
    )
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


def create_event_mask(
    normalized_trial_types: pd.Series,
    prefixes: Optional[List[str]],
    types: Optional[List[str]],
) -> pd.Series:
    mask = pd.Series(False, index=normalized_trial_types.index)
    if prefixes:
        for prefix in prefixes:
            mask = mask | normalized_trial_types.str.startswith(prefix)
    if types:
        mask = mask | normalized_trial_types.isin(types)
    return mask


###################################################################
# Run Combination Functions
###################################################################


def load_run_files(run_files: List[Path]) -> List[Tuple[int, pd.DataFrame, Path]]:
    frames = []
    for file_path in run_files:
        run_number = extract_run_number(file_path)
        if run_number is None:
            continue
        try:
            dataframe = read_tsv(file_path)
        except (pd.errors.ParserError, OSError) as e:
            logger.warning("Skipping run file due to read error: %s -> %s", file_path, e)
            continue
        if "onset" in dataframe.columns:
            dataframe = dataframe.sort_values("onset", kind="mergesort")
        frames.append((run_number, dataframe, file_path))
    return frames


def get_union_columns(frames: List[Tuple[int, pd.DataFrame, Path]]) -> List[str]:
    union_columns = []
    for _, dataframe, _ in frames:
        for column in dataframe.columns:
            if column not in union_columns:
                union_columns.append(column)
    return union_columns


def add_run_id_column(dataframe: pd.DataFrame, run_number: int) -> None:
    if "run_id" not in dataframe.columns and "run" not in dataframe.columns:
        dataframe.insert(0, "run_id", run_number)
    elif "run_id" in dataframe.columns and dataframe["run_id"].isna().any():
        dataframe["run_id"] = dataframe["run_id"].fillna(run_number)
    elif "run" in dataframe.columns and "run_id" not in dataframe.columns:
        if dataframe["run"].isna().any():
            dataframe["run"] = dataframe["run"].fillna(run_number)


def update_sample_indices(dataframe: pd.DataFrame, cumulative_offset: int) -> int:
    if "sample" not in dataframe.columns:
        return cumulative_offset
    
    sample_numeric = pd.to_numeric(dataframe["sample"], errors="coerce")
    if not sample_numeric.notna().any():
        return cumulative_offset
    
    if cumulative_offset > 0:
        dataframe["sample"] = sample_numeric + cumulative_offset
    
    max_sample = int((sample_numeric + cumulative_offset).max())
    return max_sample + 1


def get_sort_columns(combined_df: pd.DataFrame) -> List[str]:
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


def combine_runs_for_subject(sub_eeg_dir: Path, task: str) -> Optional[Path]:
    run_files = sorted(p for p in sub_eeg_dir.glob(f"*_task-{task}_run-*_events.tsv") if not p.name.startswith("._"))
    if not run_files:
        return None

    frames = load_run_files(run_files)
    if not frames:
        return None

    frames.sort(key=lambda t: t[0])
    n_runs = len({r for r, _, _ in frames})
    union_columns = get_union_columns(frames)

    dataframes = []
    cumulative_sample_offset = 0
    
    for run_number, dataframe, _ in frames:
        for column in union_columns:
            if column not in dataframe.columns:
                dataframe[column] = pd.NA
        dataframe = dataframe[union_columns]
        
        add_run_id_column(dataframe, run_number)
        cumulative_sample_offset = update_sample_indices(dataframe, cumulative_sample_offset)
        
        dataframes.append(dataframe)

    combined = pd.concat(dataframes, axis=0, ignore_index=True)
    sort_columns = get_sort_columns(combined)
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
# Raw-to-BIDS Helpers
###################################################################




def trim_to_first_volume(raw: mne.io.BaseRaw) -> bool:
    if len(raw.annotations) == 0:
        return False
    
    volume_pattern = re.compile(r"(^|[/,])V\s*1(\D|$)")
    volume_indices = [
        idx
        for idx, description in enumerate(raw.annotations.description)
        if normalize_string(description).startswith("Volume/V") 
        or volume_pattern.search(normalize_string(description)) is not None
    ]
    
    if not volume_indices:
        return False
    
    first_onset = min(raw.annotations.onset[idx] for idx in volume_indices)
    if not isinstance(first_onset, (int, float)) or first_onset <= 0:
        return False
    
    logger.info(
        "Trimming raw to first volume trigger at %.3fs relative to recording start.",
        first_onset
    )
    raw.crop(tmin=float(first_onset), tmax=None)
    return True


def filter_annotations(
    raw: mne.io.BaseRaw,
    event_prefixes: Optional[List[str]],
    keep_all: bool,
    zero_base: bool,
) -> None:
    if len(raw.annotations) == 0:
        return

    if keep_all:
        if not zero_base:
            return
        onsets = [float(o) for o in raw.annotations.onset]
        if not onsets:
            return
        base = min(onsets)
        if base == 0.0:
            return
        shifted = mne.Annotations(
            onset=[float(o) - base for o in raw.annotations.onset],
            duration=list(raw.annotations.duration),
            description=list(raw.annotations.description),
            orig_time=raw.annotations.orig_time,
        )
        raw.set_annotations(shifted)
        return
    
    if event_prefixes is None:
        # Default: keep both task triggers and fMRI volume triggers for
        # simultaneous EEG-fMRI alignment/QC.
        normalized_prefixes = ["Trig_", "Volume"]
    else:
        normalized_prefixes = [normalize_string(p) for p in event_prefixes if str(p).strip() != ""]
    
    keep_indices = [
        idx
        for idx, description in enumerate(raw.annotations.description)
        if any(normalize_string(description).startswith(prefix) for prefix in normalized_prefixes)
    ]
    
    if not keep_indices:
        logger.warning(
            "No annotations matched provided prefixes. "
            "Prefixes=%s. Found %d annotations but will drop all, resulting in no events.tsv. "
            "Use --keep_all_annotations or adjust --event_prefix to keep the desired events.",
            normalized_prefixes, len(raw.annotations)
        )
        raw.set_annotations(mne.Annotations([], [], [], orig_time=raw.annotations.orig_time))
        return
    
    new_onsets = [raw.annotations.onset[idx] for idx in keep_indices]
    new_durations = [raw.annotations.duration[idx] for idx in keep_indices]
    new_descriptions = [raw.annotations.description[idx] for idx in keep_indices]

    if zero_base and new_onsets:
        base = float(min(float(o) for o in new_onsets))
        if base != 0.0:
            new_onsets = [float(onset) - base for onset in new_onsets]
    
    filtered_annotations = mne.Annotations(
        onset=new_onsets,
        duration=new_durations,
        description=new_descriptions,
        orig_time=raw.annotations.orig_time,
    )
    raw.set_annotations(filtered_annotations)


def set_channel_types(raw: mne.io.BaseRaw) -> None:
    non_eeg_channel_types = {"HEOG": "eog", "VEOG": "eog", "ECG": "ecg"}
    present_channel_types = {
        name: channel_type
        for name, channel_type in non_eeg_channel_types.items()
        if name in raw.ch_names
    }
    if present_channel_types:
        raw.set_channel_types(present_channel_types)


def set_montage(raw: mne.io.BaseRaw, montage_name: str) -> None:
    montage = mne.channels.make_standard_montage(montage_name)
    if "FPz" in raw.ch_names and "Fpz" not in raw.ch_names:
        raw.rename_channels({"FPz": "Fpz"})
    raw.set_montage(montage, on_missing="warn")


def ensure_dataset_description(bids_root: Path, name: str = "EEG BIDS dataset") -> None:
    from mne_bids import make_dataset_description
    bids_root.mkdir(parents=True, exist_ok=True)
    dataset_description = bids_root / "dataset_description.json"
    if dataset_description.exists():
        return
    make_dataset_description(
        path=bids_root,
        name=name,
        dataset_type="raw",
        overwrite=False,
    )


###################################################################
# Clean (Post-Rejection) Events TSV
###################################################################


def _build_epoch_event_mask(
    events_df: pd.DataFrame,
    conditions: List[str],
    condition_columns: Optional[List[str]] = None,
) -> tuple[pd.Series, str]:
    condition_column = _resolve_epoch_condition_column(
        events_df,
        condition_columns=condition_columns,
    )
    trial_type_norm = events_df[condition_column].astype(str).map(normalize_string)
    cond_norm = [normalize_string(c) for c in conditions if str(c).strip() != ""]

    mask = pd.Series(False, index=events_df.index)
    for cond in cond_norm:
        # Support both exact-match and prefix-match conditions.
        mask = mask | (trial_type_norm == cond) | trial_type_norm.str.startswith(cond)
    return mask, condition_column


def _resolve_epoch_condition_column(
    events_df: pd.DataFrame,
    *,
    condition_columns: Optional[List[str]] = None,
) -> str:
    candidates: list[str] = []
    if condition_columns:
        candidates.extend(str(col).strip() for col in condition_columns if str(col).strip())
    candidates.extend(["condition", "trial_type"])

    deduped: list[str] = []
    seen: set[str] = set()
    for col in candidates:
        key = col.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(col)

    column_lookup = {str(col).strip().lower(): str(col) for col in events_df.columns}
    for col in deduped:
        resolved = column_lookup.get(col.lower())
        if resolved is not None:
            return resolved

    raise ValueError(
        "events.tsv is missing a usable condition column for epoch alignment. "
        f"Tried: {deduped}; available columns: {list(events_df.columns)}"
    )


def _load_subject_events_for_epochs(bids_sub_eeg_dir: Path, subject_label: str, task: str) -> pd.DataFrame:
    """Load a subject/task events table suitable for epoch alignment.

    Prefers the combined ``*_task-<task>_events.tsv`` when present; otherwise
    concatenates per-run ``run-*_events.tsv`` in run order.
    """
    combined = bids_sub_eeg_dir / f"{subject_label}_task-{task}_events.tsv"
    if combined.exists():
        df = read_tsv(combined)
        if "onset" in df.columns:
            sort_cols = ["onset"]
            if "run_id" in df.columns:
                sort_cols.insert(0, "run_id")
            if "sample" in df.columns:
                sort_cols.append("sample")
            df = df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
        return df

    run_files = sorted(bids_sub_eeg_dir.glob(f"{subject_label}_task-{task}_run-*_events.tsv"))
    if not run_files:
        raise FileNotFoundError(
            f"No events.tsv found for {subject_label}, task-{task} under {bids_sub_eeg_dir}"
        )

    frames = load_run_files(run_files)
    if not frames:
        raise FileNotFoundError(
            f"Found {len(run_files)} run events files but none could be read for {subject_label}, task-{task}"
        )

    frames.sort(key=lambda t: t[0])
    union_columns = get_union_columns(frames)

    parts: list[pd.DataFrame] = []
    for run_number, df, _path in frames:
        for col in union_columns:
            if col not in df.columns:
                df[col] = pd.NA
        df = df[union_columns].copy()
        add_run_id_column(df, run_number)
        parts.append(df)

    out = pd.concat(parts, axis=0, ignore_index=True)
    sort_cols = ["run_id"] if "run_id" in out.columns else []
    if "onset" in out.columns:
        sort_cols.append("onset")
    if "sample" in out.columns:
        sort_cols.append("sample")
    if sort_cols:
        out = out.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    return out


def _derive_clean_events_path_from_epochs(epochs_path: Path) -> Path:
    name = epochs_path.name
    if name.endswith("_proc-clean_epo.fif"):
        return epochs_path.with_name(name.replace("_proc-clean_epo.fif", "_proc-clean_events.tsv"))
    if name.endswith("_proc-cleaned_epo.fif"):
        return epochs_path.with_name(name.replace("_proc-cleaned_epo.fif", "_proc-cleaned_events.tsv"))
    if name.endswith("_clean_epo.fif"):
        return epochs_path.with_name(name.replace("_clean_epo.fif", "_clean_events.tsv"))
    if name.endswith("_epo.fif"):
        return epochs_path.with_name(name.replace("_epo.fif", "_events.tsv"))
    return epochs_path.with_suffix(".tsv")


def write_clean_events_tsv_for_epochs(
    *,
    subject: str,
    task: str,
    bids_root: Path,
    epochs_path: Path,
    conditions: Optional[List[str]] = None,
    condition_columns: Optional[List[str]] = None,
    overwrite: bool = True,
    _logger: Optional[logging.Logger] = None,
) -> Path:
    """Write a clean, epoch-aligned events.tsv that excludes rejected epochs.

    Output is written next to the clean epochs file (derivatives), using the
    same naming stem (e.g., ``*_proc-clean_events.tsv``).
    """
    from eeg_pipeline.analysis.utilities.bids_metadata import ensure_events_sidecar

    log = _logger or logger

    subject_label = subject if subject.startswith("sub-") else f"sub-{subject}"
    bids_sub_eeg_dir = bids_root / subject_label / "eeg"
    if not bids_sub_eeg_dir.exists():
        raise FileNotFoundError(f"Missing BIDS EEG directory: {bids_sub_eeg_dir}")
    if not epochs_path.exists():
        raise FileNotFoundError(f"Missing clean epochs file: {epochs_path}")

    out_path = _derive_clean_events_path_from_epochs(epochs_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not overwrite:
        return out_path

    epochs = mne.read_epochs(epochs_path, preload=False, verbose=False)

    if conditions is None:
        # Fallback to event_id keys if explicit conditions were not provided.
        conditions = list(getattr(epochs, "event_id", {}).keys())
    if not conditions:
        raise ValueError(
            f"No epoching conditions provided and epochs.event_id is empty for {subject_label}, task-{task}"
        )

    events_df = _load_subject_events_for_epochs(bids_sub_eeg_dir, subject_label, task)
    mask, condition_column = _build_epoch_event_mask(
        events_df,
        conditions,
        condition_columns=condition_columns,
    )
    target = events_df.loc[mask].copy().reset_index(drop=True)

    if len(target) == 0:
        raise ValueError(
            f"No events matched conditions={conditions} in {subject_label}, task-{task}. "
            f"Available {condition_column} values: "
            f"{sorted(set(events_df.get(condition_column, pd.Series(dtype=str)).dropna().astype(str)))}"
        )

    # Track pre-rejection index within the condition-filtered target set.
    target.insert(0, "event_index", range(len(target)))

    n_epochs = len(epochs)
    if n_epochs == 0:
        kept = target.iloc[0:0].copy().reset_index(drop=True)
        kept.insert(0, "epoch_index", [])
        kept.to_csv(out_path, sep="\t", index=False)
        ensure_events_sidecar(out_path, list(kept.columns))
        log.warning("All epochs were rejected; wrote empty clean events: %s", out_path)
        return out_path

    if hasattr(epochs, "selection"):
        sel = list(getattr(epochs, "selection"))
        if len(sel) == n_epochs and sel and max(sel) < len(target):
            kept = target.iloc[sel].copy().reset_index(drop=True)
        elif len(target) == n_epochs:
            kept = target.copy().reset_index(drop=True)
        else:
            raise ValueError(
                f"Cannot map kept epochs to events for {subject_label}, task-{task}: "
                f"epochs={n_epochs}, target_events={len(target)}, selection_len={len(sel)}."
            )
    elif len(target) == n_epochs:
        kept = target.copy().reset_index(drop=True)
    else:
        raise ValueError(
            f"Cannot map kept epochs to events for {subject_label}, task-{task}: "
            f"epochs={n_epochs}, target_events={len(target)} and epochs.selection unavailable."
        )

    kept.insert(0, "epoch_index", range(len(kept)))

    kept.to_csv(out_path, sep="\t", index=False)
    ensure_events_sidecar(out_path, list(kept.columns))
    log.info("Wrote clean events (n=%d): %s", len(kept), out_path)
    return out_path


__all__ = [
    "find_brainvision_vhdrs",
    "parse_subject_id",
    "extract_run_number",
    "get_run_index",
    "normalize_string",
    "normalize_event_filters",
    "find_behavior_csv_for_run",
    "create_event_mask",
    "load_run_files",
    "get_union_columns",
    "add_run_id_column",
    "update_sample_indices",
    "get_sort_columns",
    "combine_runs_for_subject",
    "trim_to_first_volume",
    "filter_annotations",
    "set_channel_types",
    "set_montage",
    "ensure_dataset_description",
    "write_clean_events_tsv_for_epochs",
]
