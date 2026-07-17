from datetime import datetime, timezone
import logging
from pathlib import Path

import mne
import numpy as np
import pandas as pd

from studies.pain_study.scripts.eeg_raw_to_bids import (
    _discard_unrecorded_terminal_volumes,
    run_raw_to_bids,
)


def test_native_fif_conversion_lets_mne_bids_create_participant_rows(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "corrected"
    source_eeg = source_root / "sub-0001" / "eeg"
    source_eeg.mkdir(parents=True)
    source_file = source_eeg / "sub-0001_task-thermalactive_run-1_desc-mriartifactclean_raw.fif"

    info = mne.create_info(["Fz", "ECG"], sfreq=100.0, ch_types=["eeg", "ecg"])
    raw = mne.io.RawArray(np.zeros((2, 1_000)), info, verbose=False)
    raw.set_annotations(mne.Annotations([1.0], [0.0], ["Trig_therm/T  1"]))
    raw.save(source_file, overwrite=False, verbose=False)

    bids_root = tmp_path / "bids"
    converted = run_raw_to_bids(
        source_root=source_root,
        bids_root=bids_root,
        task="thermalactive",
        montage="",
        source_format="native-fif",
    )

    assert converted == 1
    participants = pd.read_csv(bids_root / "participants.tsv", sep="\t")
    assert participants["participant_id"].tolist() == ["sub-0001"]
    assert "age" in participants.columns


def test_native_fif_conversion_discards_terminal_volume_without_data(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "corrected"
    source_eeg = source_root / "sub-0002" / "eeg"
    source_eeg.mkdir(parents=True)
    source_file = source_eeg / "sub-0002_task-thermalactive_run-1_desc-mriartifactclean_raw.fif"

    info = mne.create_info(["Fz", "ECG"], sfreq=100.0, ch_types=["eeg", "ecg"])
    raw = mne.io.RawArray(np.zeros((2, 1_000)), info, verbose=False)
    raw.set_annotations(
        mne.Annotations(
            [1.0, 9.996],
            [0.0, 0.0],
            ["Trig_therm/T  1", "Volume/V  1"],
        )
    )
    raw.save(source_file, overwrite=False, verbose=False)

    bids_root = tmp_path / "bids"
    converted = run_raw_to_bids(
        source_root=source_root,
        bids_root=bids_root,
        task="thermalactive",
        montage="",
        source_format="native-fif",
    )

    assert converted == 1
    events_path = bids_root / "sub-0002" / "eeg" / "sub-0002_task-thermalactive_run-1_events.tsv"
    events = pd.read_csv(events_path, sep="\t")
    assert events["trial_type"].tolist() == ["Trig_therm/T  1"]


def test_terminal_volume_validation_uses_annotation_time_after_crop() -> None:
    info = mne.create_info(["Fz", "ECG"], sfreq=100.0, ch_types=["eeg", "ecg"])
    raw = mne.io.RawArray(np.zeros((2, 1_000)), info, verbose=False)
    raw.set_meas_date(datetime(2026, 1, 1, tzinfo=timezone.utc))
    raw.set_annotations(
        mne.Annotations(
            [1.0, 9.996],
            [0.0, 0.0],
            ["Volume/V  1", "Volume/V  1"],
            orig_time=raw.info["meas_date"],
        )
    )
    raw.crop(tmin=1.0)

    _discard_unrecorded_terminal_volumes(raw, logging.getLogger(__name__))

    assert raw.annotations.description.tolist() == ["Volume/V  1"]
