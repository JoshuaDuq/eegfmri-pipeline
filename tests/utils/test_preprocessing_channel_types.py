from unittest.mock import Mock

import mne
import numpy as np

from eeg_pipeline.utils.data.preprocessing import filter_annotations, set_channel_types


def test_set_channel_types_explicitly_accepts_expected_unit_changes() -> None:
    raw = Mock()
    raw.ch_names = ["Cz", "ECG"]

    set_channel_types(raw)

    raw.set_channel_types.assert_called_once_with(
        {"ECG": "ecg"},
        on_unit_change="ignore",
    )


def test_default_annotation_filter_preserves_analyzer_pulse_markers() -> None:
    info = mne.create_info(["Cz", "ECG"], sfreq=100.0, ch_types=["eeg", "ecg"])
    raw = mne.io.RawArray(np.zeros((2, 1_000)), info, verbose=False)
    raw.set_annotations(
        mne.Annotations(
            onset=[1.0, 2.0, 3.0, 4.0],
            duration=[0.0] * 4,
            description=[
                "Volume/V  1",
                "Pulse Artifact/R",
                "Trig_therm/T  1",
                "SyncStatus/Sync On",
            ],
        )
    )

    filter_annotations(raw, event_prefixes=None, keep_all=False, zero_base=False)

    assert raw.annotations.description.tolist() == [
        "Volume/V  1",
        "Pulse Artifact/R",
        "Trig_therm/T  1",
    ]


def test_custom_annotation_filter_still_preserves_analyzer_pulse_markers() -> None:
    info = mne.create_info(["Cz"], sfreq=100.0, ch_types=["eeg"])
    raw = mne.io.RawArray(np.zeros((1, 1_000)), info, verbose=False)
    raw.set_annotations(
        mne.Annotations(
            onset=[1.0, 2.0, 3.0],
            duration=[0.0] * 3,
            description=[
                "Pulse Artifact/R",
                "Trig_therm/T  1",
                "SyncStatus/Sync On",
            ],
        )
    )

    filter_annotations(
        raw,
        event_prefixes=["Trig_therm"],
        keep_all=False,
        zero_base=False,
    )

    assert raw.annotations.description.tolist() == [
        "Pulse Artifact/R",
        "Trig_therm/T  1",
    ]
