"""EEG raw data to BIDS conversion for the pain study."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Literal, Optional

import mne
import numpy as np

from eeg_pipeline.analysis.utilities.bids_metadata import (
    ensure_task_events_json,
)
from eeg_pipeline.utils.data.preprocessing import (
    ensure_dataset_description,
    filter_annotations,
    find_brainvision_vhdrs,
    get_run_index,
    parse_subject_id,
    set_channel_types,
    set_montage,
    trim_to_first_volume,
)

logger = logging.getLogger(__name__)

SourceFormat = Literal["brainvision", "native-fif"]


def _find_native_corrected_fifs(source_root: Path, task: str) -> list[Path]:
    pattern = f"sub-*/eeg/sub-*_task-{task}_run-*_desc-mriartifactclean_raw.fif"
    return sorted(
        path
        for path in source_root.glob(pattern)
        if path.is_file() and not path.name.startswith("._")
    )


def _find_source_files(
    source_root: Path,
    source_format: SourceFormat,
    task: str,
) -> list[Path]:
    if source_format == "brainvision":
        source_files = find_brainvision_vhdrs(source_root)
    elif source_format == "native-fif":
        source_files = _find_native_corrected_fifs(source_root, task)
    else:
        raise ValueError(f"Unsupported EEG source format: {source_format}")

    if not source_files:
        raise FileNotFoundError(
            f"No {source_format} EEG files found under {source_root}"
        )
    return source_files


def _read_raw(source_file: Path, source_format: SourceFormat) -> mne.io.BaseRaw:
    if source_format == "brainvision":
        return mne.io.read_raw_brainvision(source_file, preload=False, verbose=False)
    if source_format == "native-fif":
        return mne.io.read_raw_fif(source_file, preload=True, verbose=False)
    raise ValueError(f"Unsupported EEG source format: {source_format}")


def _has_volume_triggers(raw: mne.io.BaseRaw) -> bool:
    if len(raw.annotations) == 0:
        return False
    for desc in raw.annotations.description:
        s = str(desc).strip()
        if s.startswith("Volume/") or s.startswith("Volume"):
            return True
        if s.startswith("Volume/V"):
            return True
    return False


def _discard_unrecorded_terminal_volumes(
    raw: mne.io.BaseRaw,
    log: logging.Logger,
) -> None:
    if len(raw.annotations) == 0:
        return

    sample_indices = raw.time_as_index(
        raw.annotations.onset,
        use_rounding=True,
        origin=raw.annotations.orig_time,
    )
    valid_mask = (sample_indices >= 0) & (sample_indices < raw.n_times)
    if valid_mask.all():
        return

    invalid_indices = np.flatnonzero(~valid_mask)
    expected_indices = np.arange(
        len(raw.annotations) - invalid_indices.size,
        len(raw.annotations),
    )
    sampling_period = 1.0 / float(raw.info["sfreq"])
    final_sample_time = float(raw.times[-1])
    invalid_descriptions = raw.annotations.description[invalid_indices]
    invalid_onsets = raw.annotations.onset[invalid_indices]
    invalid_durations = raw.annotations.duration[invalid_indices]

    removable = (
        np.array_equal(invalid_indices, expected_indices)
        and np.all(invalid_descriptions == "Volume/V  1")
        and np.all(invalid_durations == 0.0)
        and np.all(invalid_onsets > final_sample_time)
        and np.all(invalid_onsets <= final_sample_time + sampling_period)
    )
    if not removable:
        details = ", ".join(
            f"{description!r} at {onset:.9f}s"
            for description, onset in zip(
                invalid_descriptions,
                invalid_onsets,
                strict=True,
            )
        )
        raise ValueError(f"Annotations fall outside recorded EEG data: {details}")

    log.info(
        "Discarding %d terminal volume marker(s) without a recorded output sample.",
        invalid_indices.size,
    )
    raw.set_annotations(raw.annotations[valid_mask])


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
    source_format: SourceFormat = "brainvision",
    _logger: Optional[logging.Logger] = None,
) -> int:
    """Convert one explicitly selected EEG source format to BIDS."""
    log = _logger or logger

    from mne_bids import BIDSPath, write_raw_bids

    log.info("Scanning for %s EEG files in: %s", source_format, source_root)
    source_files = _find_source_files(source_root, source_format, task)

    if subjects:
        subj_set = set(subjects)
        source_files = [
            path for path in source_files if parse_subject_id(path) in subj_set
        ]
        if not source_files:
            raise FileNotFoundError(
                f"No matching {source_format} files for subjects: {sorted(subj_set)}"
            )

    ensure_dataset_description(bids_root, name=f"{task} EEG")
    ensure_task_events_json(bids_root, task=task)

    for index, source_file in enumerate(source_files, 1):
        subject_label = parse_subject_id(source_file)
        run_index = get_run_index(source_file)
        if run_index is None:
            raise ValueError(f"Run number is required in EEG filename: {source_file}")

        raw = _read_raw(source_file, source_format)
        set_channel_types(raw)

        if montage:
            set_montage(raw, montage)

        raw.info["line_freq"] = line_freq

        has_vol = _has_volume_triggers(raw)
        if do_trim_to_first_volume and not has_vol:
            log.warning(
                "trim_to_first_volume requested but no volume triggers detected in %s. "
                "EEG↔fMRI temporal anchoring will be limited.",
                source_file.name,
            )
        if (not do_trim_to_first_volume) and has_vol:
            log.info(
                "Volume triggers detected in %s. For EEG↔fMRI alignment, consider enabling "
                "--trim-to-first-volume and --zero-base-onsets.",
                source_file.name,
            )

        was_trimmed = False
        if do_trim_to_first_volume:
            was_trimmed = trim_to_first_volume(raw)

        if was_trimmed and not raw.preload:
            raw.load_data()

        filter_annotations(raw, event_prefixes, keep_all_annotations, zero_base_onsets)
        _discard_unrecorded_terminal_volumes(raw, log)

        bids_path = BIDSPath(
            subject=subject_label,
            task=task,
            run=run_index,
            datatype="eeg",
            suffix="eeg",
            root=bids_root,
        )

        write_raw_bids(
            raw=raw,
            bids_path=bids_path,
            overwrite=overwrite,
            allow_preload=raw.preload,
            format="BrainVision",
            verbose=False,
        )

        log.info(
            "[%d/%d] Wrote: sub-%s run-%d",
            index,
            len(source_files),
            subject_label,
            run_index,
        )

    log.info("Done. Converted %d file(s) to BIDS in: %s", len(source_files), bids_root)
    return len(source_files)
