import os
import re
import sys
import argparse
import inspect
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
import mne
from mne_bids import BIDSPath, write_raw_bids, make_dataset_description

os.environ["PYTHONUTF8"] = "1"
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from eeg_pipeline.utils.config.loader import load_settings

config = load_settings(script_name=Path(__file__).name)

PROJECT_ROOT = config.project_root
BIDS_ROOT = config.bids_root
TASK = config.task
MONTAGE_NAME = config.get("raw_to_bids.default_montage", "easycap-M1")
LINE_FREQ = float(config.get("raw_to_bids.default_line_freq", 60.0))


###################################################################
# Helper Functions
###################################################################

def find_brainvision_vhdrs(source_root: Path) -> List[Path]:
    vhdrs = sorted(source_root.glob("sub-*/eeg/*.vhdr"))
    return [p for p in vhdrs if p.is_file()]


def parse_subject_id(path: Path) -> str:
    m = re.search(r"sub-([A-Za-z0-9]+)", str(path))
    if not m:
        raise ValueError(f"Could not parse subject from path: {path}")
    return m.group(1)


def _extract_run_label(name: str) -> Optional[int]:
    m = re.search(r"run-?(\d+)", name, flags=re.IGNORECASE)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return None
    return None


def events_from_raw_annotations(raw: mne.io.BaseRaw) -> Tuple[Optional[np.ndarray], Optional[dict]]:
    events, event_id = mne.events_from_annotations(raw, event_id=None)
    if events is None or len(events) == 0 or not event_id:
        return None, None
    return events, event_id


def ensure_dataset_description(bids_root: Path, name: str = "EEG BIDS dataset") -> None:
    bids_root.mkdir(parents=True, exist_ok=True)
    make_dataset_description(
        path=bids_root,
        name=name,
        dataset_type="raw",
        overwrite=True,
    )


###################################################################
# Conversion Functions
###################################################################

def _normalize_string(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()


def _extract_run_number(vhdr_path: Path) -> Optional[int]:
    match = re.search(r"run[-_]?(\d+)", vhdr_path.stem, flags=re.IGNORECASE)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            return None
    return None


def _infer_run_number(vhdr_path: Path) -> Optional[int]:
    all_runs = sorted(vhdr_path.parent.glob("*.vhdr"))
    if len(all_runs) <= 1:
        return None
    inferred_run = all_runs.index(vhdr_path) + 1
    print(
        f"Warning: No explicit run found in filename '{vhdr_path.name}'. "
        f"Inferring run={inferred_run} by alphabetical order among {len(all_runs)} files. "
        f"Prefer 'run-01' style filenames to guarantee correct run IDs."
    )
    return inferred_run


def _get_run_index(vhdr_path: Path) -> Optional[int]:
    run_index = _extract_run_number(vhdr_path)
    if run_index is None:
        run_index = _infer_run_number(vhdr_path)
    return run_index


def _set_channel_types(raw: mne.io.BaseRaw) -> None:
    non_eeg_channel_types = {"HEOG": "eog", "VEOG": "eog", "ECG": "ecg"}
    present_channel_types = {
        name: channel_type
        for name, channel_type in non_eeg_channel_types.items()
        if name in raw.ch_names
    }
    if present_channel_types:
        raw.set_channel_types(present_channel_types)


def _set_montage(raw: mne.io.BaseRaw, montage_name: str) -> None:
    montage = mne.channels.make_standard_montage(montage_name)
    if "FPz" in raw.ch_names and "Fpz" not in raw.ch_names:
        raw.rename_channels({"FPz": "Fpz"})
    raw.set_montage(montage, on_missing="warn")


def _find_first_volume_trigger(annotations: mne.Annotations) -> Optional[float]:
    if len(annotations) == 0:
        return None
    
    volume_pattern = re.compile(r"(^|[/,])V\s*1(\D|$)")
    volume_indices = [
        idx
        for idx, description in enumerate(annotations.description)
        if _normalize_string(description).startswith("Volume/V") 
        or volume_pattern.search(_normalize_string(description)) is not None
    ]
    
    if not volume_indices:
        return None
    
    first_onset = min(annotations.onset[idx] for idx in volume_indices)
    if isinstance(first_onset, (int, float)) and first_onset > 0:
        return float(first_onset)
    return None


def _trim_to_first_volume(raw: mne.io.BaseRaw) -> bool:
    first_volume_time = _find_first_volume_trigger(raw.annotations)
    if first_volume_time is None:
        return False
    
    print(
        f"Trimming raw to first volume trigger at {first_volume_time:.3f}s "
        f"relative to recording start."
    )
    raw.crop(tmin=first_volume_time, tmax=None)
    return True


def _normalize_event_prefixes(prefixes: Optional[List[str]]) -> List[str]:
    if prefixes is None:
        return ["Trig_therm"]
    return [_normalize_string(p) for p in prefixes if str(p).strip() != ""]


def _filter_annotations_by_prefixes(
    annotations: mne.Annotations,
    normalized_prefixes: List[str]
) -> List[int]:
    return [
        idx
        for idx, description in enumerate(annotations.description)
        if any(_normalize_string(description).startswith(prefix) for prefix in normalized_prefixes)
    ]


def _zero_base_onsets(onsets: List[float]) -> List[float]:
    if not onsets:
        return onsets
    base = onsets[0]
    return [onset - base for onset in onsets]


def _filter_annotations(
    raw: mne.io.BaseRaw,
    event_prefixes: Optional[List[str]],
    keep_all: bool,
    zero_base: bool
) -> None:
    if len(raw.annotations) == 0 or keep_all:
        return
    
    normalized_prefixes = _normalize_event_prefixes(event_prefixes)
    keep_indices = _filter_annotations_by_prefixes(raw.annotations, normalized_prefixes)
    
    if not keep_indices:
        print(
            "Warning: No annotations matched provided prefixes. "
            f"Prefixes={normalized_prefixes}. Found {len(raw.annotations)} annotations "
            "but will drop all, resulting in no events.tsv. "
            "Use --keep_all_annotations or adjust --event_prefix to keep the desired events."
        )
        raw.set_annotations(mne.Annotations([], [], [], orig_time=raw.annotations.orig_time))
        return
    
    new_onsets = [raw.annotations.onset[idx] for idx in keep_indices]
    new_durations = [raw.annotations.duration[idx] for idx in keep_indices]
    new_descriptions = [raw.annotations.description[idx] for idx in keep_indices]
    
    if zero_base:
        new_onsets = _zero_base_onsets(new_onsets)
    
    filtered_annotations = mne.Annotations(
        onset=new_onsets,
        duration=new_durations,
        description=new_descriptions,
        orig_time=raw.annotations.orig_time,
    )
    raw.set_annotations(filtered_annotations)


def _prepare_write_kwargs(raw: mne.io.BaseRaw) -> dict:
    signature = inspect.signature(write_raw_bids)
    parameters = signature.parameters
    kwargs = {}
    if "allow_preload" in parameters:
        kwargs["allow_preload"] = bool(getattr(raw, "preload", False))
    if "format" in parameters:
        kwargs["format"] = "BrainVision"
    if "verbose" in parameters:
        kwargs["verbose"] = False
    return kwargs


def convert_one(
    vhdr_path: Path,
    bids_root: Path,
    task: str,
    montage_name: Optional[str],
    line_freq: Optional[float],
    overwrite: bool = False,
    zero_base_onsets: bool = False,
    trim_to_first_volume: bool = False,
    event_prefixes: Optional[List[str]] = None,
    keep_all_annotations: bool = False,
) -> BIDSPath:
    subject_label = parse_subject_id(vhdr_path)
    run_index = _get_run_index(vhdr_path)

    raw = mne.io.read_raw_brainvision(vhdr_path, preload=False, verbose=False)

    _set_channel_types(raw)

    if montage_name:
        _set_montage(raw, montage_name)

    raw.info["line_freq"] = line_freq

    was_trimmed = False
    if trim_to_first_volume:
        was_trimmed = _trim_to_first_volume(raw)

    if was_trimmed and not raw.preload:
        raw.load_data()

    _filter_annotations(raw, event_prefixes, keep_all_annotations, zero_base_onsets)

    bids_path = BIDSPath(
        subject=subject_label,
        task=task,
        run=run_index,
        datatype="eeg",
        suffix="eeg",
        root=bids_root,
    )

    write_kwargs = _prepare_write_kwargs(raw)
    write_raw_bids(
        raw=raw,
        bids_path=bids_path,
        overwrite=overwrite,
        **write_kwargs,
    )

    _postprocess_bids_files(bids_path, raw, task, subject_label, run_index, vhdr_path)
    return bids_path


def _add_sample_column_to_events(events_df: pd.DataFrame, sampling_freq: float) -> None:
    if "onset" in events_df.columns and "sample" not in events_df.columns:
        sample_values = pd.to_numeric(events_df["onset"], errors="coerce") * sampling_freq
        sample_values = sample_values.round().astype("Int64")
        events_df["sample"] = sample_values


def _get_subject_label_from_path(events_path: Path, fallback_label: str) -> str:
    try:
        return parse_subject_id(events_path)
    except ValueError:
        return fallback_label


def _get_run_label(events_path: Path, events_df: pd.DataFrame, run_idx: Optional[int]) -> Optional[int]:
    if run_idx is not None:
        return run_idx
    if 'run' in events_df.columns:
        return _extract_run_label(events_path.name)
    return None


def _postprocess_bids_files(
    bids_path: BIDSPath,
    raw: mne.io.BaseRaw,
    task: str,
    sub_label: str,
    run_idx: Optional[int],
    vhdr_path: Path
) -> None:
    sampling_freq = float(raw.info.get("sfreq", 0.0))
    if sampling_freq <= 0:
        return

    events_bids_path = bids_path.copy().update(suffix="events", extension=".tsv")
    events_path = events_bids_path.fpath
    if events_path is None or not events_path.exists():
        return

    events_df = pd.read_csv(events_path, sep="\t")
    
    _add_sample_column_to_events(events_df, sampling_freq)
    if "sample" in events_df.columns:
        events_df.to_csv(events_path, sep="\t", index=False)

    subject_label = _get_subject_label_from_path(events_path, sub_label)
    run_label = _get_run_label(events_path, events_df, run_idx)

    _write_events_integrity(events_path, events_df, subject_label, task, run_label)
    _write_events_sidecar(events_bids_path)
    _fix_channels_file(bids_path)


def _count_missing_values(events_df: pd.DataFrame, column: str, default: int) -> int:
    if column not in events_df.columns:
        return default
    return int(events_df[column].isna().sum())


def _get_unique_trial_types(events_df: pd.DataFrame) -> List[str]:
    if "trial_type" not in events_df.columns:
        return []
    unique_types = events_df["trial_type"].dropna().unique().tolist()
    return sorted({str(x) for x in unique_types})


def _check_onset_monotonicity(events_df: pd.DataFrame) -> bool:
    if "onset" not in events_df.columns:
        return False
    onset_values = pd.to_numeric(events_df["onset"], errors="coerce").to_numpy()
    differences = np.diff(onset_values)
    return bool(np.any(differences < -1e-9))


def _count_negative_onsets(events_df: pd.DataFrame) -> int:
    if "onset" not in events_df.columns:
        return 0
    onset_numeric = pd.to_numeric(events_df["onset"], errors="coerce")
    return int((onset_numeric < 0).sum())


def _write_events_integrity(
    ev_path: Path,
    ev: pd.DataFrame,
    sub: str,
    task: str,
    run_label: Optional[int]
) -> None:
    n_rows = len(ev)
    has_sample = int("sample" in ev.columns)
    
    n_missing_onset = _count_missing_values(ev, "onset", n_rows)
    n_missing_duration = _count_missing_values(ev, "duration", n_rows)
    n_missing_trial_type = _count_missing_values(ev, "trial_type", n_rows)
    
    unique_trial_types = _get_unique_trial_types(ev)
    
    onset_min = float(pd.to_numeric(ev["onset"], errors="coerce").min()) if "onset" in ev.columns else np.nan
    onset_max = float(pd.to_numeric(ev["onset"], errors="coerce").max()) if "onset" in ev.columns else np.nan
    
    is_nonmonotonic = _check_onset_monotonicity(ev)
    n_negative_onsets = _count_negative_onsets(ev)

    integrity_path = ev_path.parent / f"sub-{sub}_task-{task}_events_integrity.tsv"
    integrity_row = pd.DataFrame([{
        "subject": sub,
        "run": run_label,
        "n_rows": n_rows,
        "has_sample": has_sample,
        "n_missing_onset": n_missing_onset,
        "n_missing_duration": n_missing_duration,
        "n_missing_trial_type": n_missing_trial_type,
        "unique_trial_types": "|".join(unique_trial_types),
        "onset_min": onset_min,
        "onset_max": onset_max,
        "onset_nonmonotonic": int(is_nonmonotonic),
        "negative_onsets": n_negative_onsets,
    }])
    write_header = not integrity_path.exists()
    integrity_row.to_csv(
        integrity_path,
        sep="\t",
        index=False,
        mode=("w" if write_header else "a"),
        header=write_header
    )


def _write_events_sidecar(ev_bp: BIDSPath) -> None:
    import json

    ev_json_bp = ev_bp.copy().update(extension=".json")
    ev_json_path = ev_json_bp.fpath
    if ev_json_path is None:
        return

    ev_desc = {
        "onset": {
            "LongName": "Event onset",
            "Description": "Onset of the event relative to the start of the recording",
            "Units": "s",
        },
        "duration": {
            "LongName": "Event duration",
            "Description": "Duration of the event",
            "Units": "s",
        },
        "trial_type": {
            "LongName": "Event label",
            "Description": "Categorical event label indicating experimental condition",
        },
        "sample": {
            "LongName": "Sample index",
            "Description": "Sample index corresponding to the event onset",
            "Units": "samples",
        },
    }
    with open(ev_json_path, "w", encoding="utf-8") as f:
        json.dump(ev_desc, f, indent=2)


def _normalize_channel_types(channels_df: pd.DataFrame) -> None:
    if "type" not in channels_df.columns:
        return
    
    channel_types = channels_df["type"].astype(str).str.upper()
    name_is_gsr = (
        channels_df.get("name", pd.Series([None] * len(channels_df)))
        .astype(str)
        .str.upper()
        .eq("GSR")
    )
    channel_types = channel_types.where(~name_is_gsr, other="MISC")
    channel_types = channel_types.replace({
        "EEG": "EEG",
        "EOG": "EOG",
        "EMG": "EMG",
        "ECG": "ECG",
        "GSR": "MISC",
        "MISC": "MISC",
    })
    channels_df["type"] = channel_types


def _normalize_channel_status(channels_df: pd.DataFrame) -> None:
    if "status" in channels_df.columns:
        status = channels_df["status"].astype(str).str.lower()
        status = status.where(~status.isin(["", "nan", "none"]), other="good")
        status = status.replace({"ok": "good"})
        channels_df["status"] = status
    else:
        channels_df["status"] = "good"


def _fix_channels_file(bids_path: BIDSPath) -> None:
    channels_bids_path = bids_path.copy().update(suffix="channels", extension=".tsv")
    channels_path = channels_bids_path.fpath
    if channels_path is None or not channels_path.exists():
        return

    channels_df = pd.read_csv(channels_path, sep="\t")
    
    _normalize_channel_types(channels_df)
    _normalize_channel_status(channels_df)
    
    channels_df.to_csv(channels_path, sep="\t", index=False)


###################################################################
# Main Entry Point
###################################################################

def main():
    parser = argparse.ArgumentParser(
        description="Convert BrainVision EEG to BIDS using MNE-BIDS"
    )
    parser.add_argument(
        "--source_root",
        type=str,
        default=str(PROJECT_ROOT / "data" / "source_data"),
        help="Path to source_data root containing sub-*/eeg"
    )
    parser.add_argument(
        "--bids_root",
        type=str,
        default=str(BIDS_ROOT),
        help="Output BIDS root directory"
    )
    parser.add_argument("--task", type=str, default=TASK, help="BIDS task label")
    parser.add_argument(
        "--subjects",
        type=str,
        nargs="*",
        default=None,
        help="Optional list of subject labels to include (e.g., 001 002). "
             "If omitted, all found are used."
    )
    parser.add_argument(
        "--montage",
        type=str,
        default=MONTAGE_NAME,
        help="Standard montage name to set on raw (e.g., easycap-M1). Use '' to skip."
    )
    parser.add_argument(
        "--line_freq",
        type=float,
        default=LINE_FREQ,
        help="Line noise frequency (Hz) metadata for sidecar."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing BIDS files"
    )
    parser.add_argument(
        "--zero_base_onsets",
        action="store_true",
        help="Zero-base kept annotation onsets so events start at 0.0"
    )
    parser.add_argument(
        "--trim_to_first_volume",
        action="store_true",
        help=(
            "Crop raw to start at the first MRI volume trigger "
            "(e.g., 'Volume/V 1', 'Volume,V 1', or bare 'V  1') "
            "to remove the initial dummy-scan period where the scanner may not send triggers."
        ),
    )
    parser.add_argument(
        "--event_prefix",
        action="append",
        default=None,
        help=(
            "Keep only annotations whose normalized label starts with this prefix. "
            "Repeat to keep multiple prefixes, e.g., --event_prefix Trig_therm --event_prefix Reward_on. "
            "If omitted, defaults to Trig_therm. Use --keep_all_annotations to keep all annotations."
        ),
    )
    parser.add_argument(
        "--keep_all_annotations",
        action="store_true",
        help="If set, do not filter annotations at all; write whatever exists."
    )

    args = parser.parse_args()

    source_root = Path(args.source_root).resolve()
    bids_root = Path(args.bids_root).resolve()
    task = args.task
    montage_name = args.montage if args.montage else None

    print(f"Scanning for BrainVision files in: {source_root}")
    vhdrs = find_brainvision_vhdrs(source_root)
    if not vhdrs:
        print("No .vhdr files found under sub-*/eeg/. Nothing to convert.")
        sys.exit(1)

    if args.subjects:
        subj_set = set(args.subjects)
        vhdrs = [p for p in vhdrs if parse_subject_id(p) in subj_set]
        if not vhdrs:
            print(f"No matching .vhdr files for subjects: {sorted(subj_set)}")
            sys.exit(1)

    ensure_dataset_description(bids_root, name=f"{task} EEG")

    written: List[BIDSPath] = []
    for i, vhdr in enumerate(vhdrs, 1):
        bp = convert_one(
            vhdr_path=vhdr,
            bids_root=bids_root,
            task=task,
            montage_name=montage_name,
            line_freq=args.line_freq,
            overwrite=args.overwrite,
            zero_base_onsets=args.zero_base_onsets,
            trim_to_first_volume=args.trim_to_first_volume,
            event_prefixes=args.event_prefix,
            keep_all_annotations=args.keep_all_annotations,
        )
        written.append(bp)
        rel = (
            str(bp.fpath).replace(str(bids_root) + os.sep, "")
            if bp.fpath else str(bp)
        )
        print(f"[{i}/{len(vhdrs)}] Wrote: {rel}")

    by_sub = {}
    for bp in written:
        if bp is None:
            continue
        s = bp.subject
        if not s:
            continue
        by_sub.setdefault(s, True)
    
    for s in sorted(by_sub.keys()):
        subj_eeg = bids_root / f"sub-{s}" / "eeg"
        integ = subj_eeg / f"sub-{s}_task-{task}_events_integrity.tsv"
        if integ.exists():
            print(f"Integrity summary for sub-{s}: {integ}")

    print(f"Done. Converted {len(written)} file(s) to BIDS in: {bids_root}")


if __name__ == "__main__":
    main()
