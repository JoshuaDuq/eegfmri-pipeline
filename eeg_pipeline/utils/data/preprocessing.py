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

import numpy as np
import pandas as pd
import mne

logger = logging.getLogger(__name__)


###################################################################
# File Discovery
###################################################################


def find_brainvision_vhdrs(source_root: Path) -> List[Path]:
    vhdrs = sorted(source_root.glob("sub-*/eeg/*.vhdr"))
    return [p for p in vhdrs if p.is_file()]


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


def infer_run_number(vhdr_path: Path) -> Optional[int]:
    all_runs = sorted(vhdr_path.parent.glob("*.vhdr"))
    if len(all_runs) <= 1:
        return None
    inferred_run = all_runs.index(vhdr_path) + 1
    logger.warning(
        "No explicit run found in filename '%s'. "
        "Inferring run=%d by alphabetical order among %d files. "
        "Prefer 'run-01' style filenames to guarantee correct run IDs.",
        vhdr_path.name, inferred_run, len(all_runs)
    )
    return inferred_run


def get_run_index(path: Path) -> Optional[int]:
    run_index = extract_run_number(path)
    if run_index is None:
        run_index = infer_run_number(path)
    return run_index


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
) -> Optional[Path]:
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
            dataframe = pd.read_csv(file_path, sep="\t")
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
    run_files = sorted(sub_eeg_dir.glob(f"*_task-{task}_run-*_events.tsv"))
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


def find_first_volume_trigger(annotations: mne.Annotations) -> Optional[float]:
    if len(annotations) == 0:
        return None
    
    volume_pattern = re.compile(r"(^|[/,])V\s*1(\D|$)")
    volume_indices = [
        idx
        for idx, description in enumerate(annotations.description)
        if normalize_string(description).startswith("Volume/V") 
        or volume_pattern.search(normalize_string(description)) is not None
    ]
    
    if not volume_indices:
        return None
    
    first_onset = min(annotations.onset[idx] for idx in volume_indices)
    if isinstance(first_onset, (int, float)) and first_onset > 0:
        return float(first_onset)
    return None


def trim_to_first_volume(raw: mne.io.BaseRaw) -> bool:
    first_volume_time = find_first_volume_trigger(raw.annotations)
    if first_volume_time is None:
        return False
    
    logger.info(
        "Trimming raw to first volume trigger at %.3fs relative to recording start.",
        first_volume_time
    )
    raw.crop(tmin=first_volume_time, tmax=None)
    return True


def normalize_event_prefixes(prefixes: Optional[List[str]]) -> List[str]:
    if prefixes is None:
        return ["Trig_therm"]
    return [normalize_string(p) for p in prefixes if str(p).strip() != ""]


def filter_annotations_by_prefixes(
    annotations: mne.Annotations,
    normalized_prefixes: List[str]
) -> List[int]:
    return [
        idx
        for idx, description in enumerate(annotations.description)
        if any(normalize_string(description).startswith(prefix) for prefix in normalized_prefixes)
    ]


def zero_base_onsets(onsets: List[float]) -> List[float]:
    if not onsets:
        return onsets
    base = onsets[0]
    return [onset - base for onset in onsets]


def filter_annotations(
    raw: mne.io.BaseRaw,
    event_prefixes: Optional[List[str]],
    keep_all: bool,
    zero_base: bool
) -> None:
    if len(raw.annotations) == 0 or keep_all:
        return
    
    normalized_prefixes = normalize_event_prefixes(event_prefixes)
    keep_indices = filter_annotations_by_prefixes(raw.annotations, normalized_prefixes)
    
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
    
    if zero_base:
        new_onsets = zero_base_onsets(new_onsets)
    
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


def events_from_raw_annotations(raw: mne.io.BaseRaw) -> Tuple[Optional[np.ndarray], Optional[dict]]:
    events, event_id = mne.events_from_annotations(raw, event_id=None)
    if events is None or len(events) == 0 or not event_id:
        return None, None
    return events, event_id


def ensure_dataset_description(bids_root: Path, name: str = "EEG BIDS dataset") -> None:
    from mne_bids import make_dataset_description
    bids_root.mkdir(parents=True, exist_ok=True)
    make_dataset_description(
        path=bids_root,
        name=name,
        dataset_type="raw",
        overwrite=True,
    )


__all__ = [
    "find_brainvision_vhdrs",
    "parse_subject_id",
    "extract_run_number",
    "infer_run_number",
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
    "find_first_volume_trigger",
    "trim_to_first_volume",
    "normalize_event_prefixes",
    "filter_annotations_by_prefixes",
    "zero_base_onsets",
    "filter_annotations",
    "set_channel_types",
    "set_montage",
    "events_from_raw_annotations",
    "ensure_dataset_description",
]
