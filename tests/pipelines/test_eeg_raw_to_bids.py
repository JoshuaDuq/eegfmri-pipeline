from datetime import datetime, timezone
import logging
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import pytest

from studies.pain_study.scripts.conversion.eeg_raw_to_bids import (
    _canonicalize_thermode_annotations,
    _discard_unrecorded_terminal_volumes,
    _find_source_files,
    run_raw_to_bids,
    trim_to_first_event,
    trim_to_volume_bounds,
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


def test_thermode_canonicalization_rewrites_exactly_eleven_stim_on_markers() -> None:
    raw = _raw_with_annotations(
        ["Stim_on/S  1"] * 11 + ["Iti_start/I  1"],
        [float(index) for index in range(12)],
    )

    _canonicalize_thermode_annotations(raw, logging.getLogger(__name__))

    assert raw.annotations.description.tolist() == [
        "Trig_therm/T  1"
    ] * 11 + ["Iti_start/I  1"]


def test_thermode_canonicalization_preserves_existing_canonical_markers() -> None:
    descriptions = ["Stim_on/S  1", "Trig_therm/T  1"] * 11
    raw = _raw_with_annotations(
        descriptions,
        [float(index) for index in range(len(descriptions))],
    )

    rewritten = _canonicalize_thermode_annotations(
        raw,
        logging.getLogger(__name__),
    )

    assert rewritten is False
    assert raw.annotations.description.tolist() == descriptions


def test_thermode_canonicalization_rejects_incomplete_stim_on_markers() -> None:
    raw = _raw_with_annotations(
        ["Stim_on/S  1"] * 10,
        [float(index) for index in range(10)],
    )

    with pytest.raises(ValueError, match="expected 11 Stim_on/S  1 markers, found 10"):
        _canonicalize_thermode_annotations(raw, logging.getLogger(__name__))


def test_brainvision_conversion_reads_the_requested_source_layout(tmp_path: Path) -> None:
    source_eeg = tmp_path / "sub-0001" / "eeg"
    original = source_eeg / "original_untrimmed_5khz" / "run1.vhdr"
    processed = source_eeg / "analyzer_brainvision_processed_1khz" / "run1.vhdr"
    original.parent.mkdir(parents=True)
    processed.parent.mkdir(parents=True)
    original.touch()
    processed.touch()

    assert _find_source_files(
        tmp_path, "brainvision", "thermalactive", "original_untrimmed_5khz"
    ) == [original]


def _raw_with_annotations(descriptions, onsets) -> mne.io.BaseRaw:
    info = mne.create_info(["Fz", "ECG"], sfreq=100.0, ch_types=["eeg", "ecg"])
    raw = mne.io.RawArray(np.zeros((2, 6_000)), info, verbose=False)
    # Every recording this converter reads carries a measurement date, and annotation
    # onsets are anchored to it. Without one, MNE re-anchors them to first_samp on every
    # set_annotations, so a cropped recording would read back shifted.
    raw.set_meas_date(datetime(2026, 6, 23, 10, 56, tzinfo=timezone.utc))
    raw.set_annotations(mne.Annotations(onsets, [0.0] * len(onsets), descriptions))
    return raw


def test_trim_to_first_event_starts_the_recording_at_the_matching_annotation() -> None:
    raw = _raw_with_annotations(
        ["Rec_start/R  1", "Iti_start/I  1", "Trig_therm/T  1", "Iti_start/I  1"],
        [0.5, 15.0, 35.0, 50.0],
    )

    onset = trim_to_first_event(raw, "Iti_start")

    assert onset == 15.0
    assert raw.first_time == 15.0
    assert list(raw.annotations.description) == [
        "Iti_start/I  1",
        "Trig_therm/T  1",
        "Iti_start/I  1",
    ]


def test_trim_to_first_event_rejects_a_recording_without_the_marker() -> None:
    raw = _raw_with_annotations(["Rec_start/R  1"], [0.5])

    with pytest.raises(ValueError, match="Iti_start"):
        trim_to_first_event(raw, "Iti_start")


def test_conversion_trims_the_recording_to_the_first_requested_event(tmp_path: Path) -> None:
    source_root = tmp_path / "sources"
    source_eeg = source_root / "sub-0003" / "eeg"
    source_eeg.mkdir(parents=True)
    source_file = source_eeg / "sub-0003_task-thermalactive_run-1_desc-mriartifactclean_raw.fif"

    raw = _raw_with_annotations(
        ["Rec_start/R  1", "Iti_start/I  1", "Trig_therm/T  1"],
        [0.5, 15.0, 35.0],
    )
    raw.save(source_file, overwrite=False, verbose=False)

    bids_root = tmp_path / "bids"
    run_raw_to_bids(
        source_root=source_root,
        bids_root=bids_root,
        task="thermalactive",
        montage="",
        event_prefixes=["Rec_start", "Iti_start", "Trig_therm"],
        source_format="native-fif",
        trim_to_first_event_prefix="Iti_start",
    )

    events = pd.read_csv(
        bids_root / "sub-0003" / "eeg" / "sub-0003_task-thermalactive_run-1_events.tsv",
        sep="\t",
    )
    assert events["trial_type"].tolist() == ["Iti_start/I  1", "Trig_therm/T  1"]
    assert events["onset"].tolist() == [0.0, 20.0]


def _fif_source_without_meas_date(source_root: Path, subject: str) -> None:
    """A source recording that never had a ``meas_date``, so ``orig_time`` is None."""
    source_eeg = source_root / f"sub-{subject}" / "eeg"
    source_eeg.mkdir(parents=True)
    info = mne.create_info(["Fz", "ECG"], sfreq=100.0, ch_types=["eeg", "ecg"])
    raw = mne.io.RawArray(np.zeros((2, 1_000)), info, verbose=False)
    assert raw.info["meas_date"] is None
    raw.set_annotations(
        mne.Annotations(
            onset=[1.0, 2.0, 3.0, 5.0, 8.0],
            duration=[0.0] * 5,
            description=[
                "Trig_therm/T  1",
                "Volume/V  1",
                "Trig_therm/T  1",
                "Volume/V  1",
                "Trig_therm/T  1",
            ],
        )
    )
    raw.save(
        source_eeg / f"sub-{subject}_task-thermalactive_run-1_desc-mriartifactclean_raw.fif",
        verbose=False,
    )


def test_volume_bound_trim_without_meas_date_keeps_event_onsets(tmp_path: Path) -> None:
    source_root = tmp_path / "corrected"
    _fif_source_without_meas_date(source_root, "0003")

    bids_root = tmp_path / "bids"
    run_raw_to_bids(
        source_root=source_root,
        bids_root=bids_root,
        task="thermalactive",
        montage="",
        source_format="native-fif",
        do_trim_to_volume_bounds=True,
    )

    events_path = bids_root / "sub-0003" / "eeg" / "sub-0003_task-thermalactive_run-1_events.tsv"
    events = pd.read_csv(events_path, sep="\t")

    # Volumes at 2.0 s and 5.0 s (TR = 3.0 s). The crop keeps one TR after the
    # last marker, so the thermode at 8.0 s survives as 6.0 s.
    assert events["trial_type"].tolist() == [
        "Volume/V  1",
        "Trig_therm/T  1",
        "Volume/V  1",
        "Trig_therm/T  1",
    ]
    assert events["onset"].tolist() == [0.0, 1.0, 3.0, 6.0]


def _raw_with_volume_train(
    volume_onsets: list[float],
    *,
    duration_s: float = 12.0,
    sfreq: float = 100.0,
    extra_onsets: list[tuple[float, str]] | None = None,
) -> mne.io.RawArray:
    info = mne.create_info(["Fz"], sfreq=sfreq, ch_types=["eeg"])
    raw = mne.io.RawArray(np.zeros((1, int(duration_s * sfreq))), info, verbose=False)
    onsets = list(volume_onsets)
    descriptions = ["Volume/V  1"] * len(volume_onsets)
    if extra_onsets:
        for onset, description in extra_onsets:
            onsets.append(onset)
            descriptions.append(description)
    raw.set_annotations(mne.Annotations(onset=onsets, duration=[0.0] * len(onsets), description=descriptions))
    return raw


def test_trim_to_volume_bounds_keeps_one_tr_after_the_last_marker() -> None:
    raw = _raw_with_volume_train([2.0, 2.9, 3.8], duration_s=8.0)
    assert trim_to_volume_bounds(raw) is True
    assert raw.times[-1] == pytest.approx(3.8 + 0.9 - 2.0)


def test_trim_to_volume_bounds_clips_when_eeg_stops_mid_last_volume() -> None:
    raw = _raw_with_volume_train([2.0, 2.9, 3.8], duration_s=4.2)
    assert trim_to_volume_bounds(raw) is True
    assert raw.times[-1] < 3.8 + 0.9 - 2.0
    assert raw.times[-1] == pytest.approx(4.2 - 2.0 - 1 / raw.info["sfreq"])


def test_trim_to_volume_bounds_drops_a_later_scanner_restart() -> None:
    raw = _raw_with_volume_train([1.0, 1.9, 2.8, 8.0, 8.9], duration_s=12.0)
    assert trim_to_volume_bounds(raw) is True
    volume_onsets = [
        onset
        for onset, description in zip(raw.annotations.onset, raw.annotations.description)
        if description == "Volume/V  1"
    ]
    assert volume_onsets == pytest.approx([1.0, 1.9, 2.8])
    assert raw.times[-1] == pytest.approx(2.8 + 0.9 - 1.0)
