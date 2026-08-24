"""Rebuilt annotations must land where they were read from on a cropped raw.

``raw.annotations.onset`` is reported on the raw's absolute timeline, but
``set_annotations`` treats an incoming ``orig_time=None`` object as relative to
the first sample: it crops against that frame and then adds ``raw.first_time``.
Reading onsets off a cropped raw and setting them straight back therefore drops
the annotations past the end and shifts the survivors by the crop offset.
"""

from datetime import datetime, timezone

import mne
import numpy as np

from eeg_pipeline.utils.data.preprocessing import filter_annotations


def _cropped_raw_without_meas_date() -> mne.io.BaseRaw:
    info = mne.create_info(["Cz"], sfreq=100.0, ch_types=["eeg"])
    raw = mne.io.RawArray(np.zeros((1, 1_000)), info, verbose=False)
    raw.set_annotations(
        mne.Annotations(
            onset=[1.0, 3.0, 5.0],
            duration=[0.0] * 3,
            description=["Volume/V  1", "Trig_therm/T  1", "Volume/V  1"],
        )
    )
    assert raw.info["meas_date"] is None
    raw.crop(tmin=2.0)
    assert raw.first_time == 2.0
    return raw


def test_filter_annotations_keeps_onsets_anchored_on_a_cropped_raw() -> None:
    raw = _cropped_raw_without_meas_date()

    filter_annotations(raw, event_prefixes=None, keep_all=False, zero_base=False)

    assert raw.annotations.description.tolist() == ["Trig_therm/T  1", "Volume/V  1"]
    assert raw.annotations.onset.tolist() == [3.0, 5.0]


def test_filter_annotations_zero_bases_the_first_kept_event_to_the_data_start() -> None:
    raw = _cropped_raw_without_meas_date()

    filter_annotations(raw, event_prefixes=None, keep_all=False, zero_base=True)

    # first_time is 2.0, so the first kept event lands on the first sample.
    assert raw.annotations.onset.tolist() == [2.0, 4.0]
    assert (raw.annotations.onset - raw.first_time).tolist() == [0.0, 2.0]


def test_keep_all_zero_base_anchors_to_the_data_start_on_a_cropped_raw() -> None:
    raw = _cropped_raw_without_meas_date()

    filter_annotations(raw, event_prefixes=None, keep_all=True, zero_base=True)

    assert raw.annotations.onset.tolist() == [2.0, 4.0]


def test_zero_base_still_anchors_to_absolute_zero_on_an_uncropped_raw() -> None:
    info = mne.create_info(["Cz"], sfreq=100.0, ch_types=["eeg"])
    raw = mne.io.RawArray(np.zeros((1, 1_000)), info, verbose=False)
    raw.set_annotations(mne.Annotations([1.0, 3.0], [0.0] * 2, ["Trig_therm/T  1", "Volume/V  1"]))

    filter_annotations(raw, event_prefixes=None, keep_all=False, zero_base=True)

    assert raw.annotations.onset.tolist() == [0.0, 2.0]


def test_zero_base_keeps_events_on_a_cropped_raw_that_has_a_meas_date() -> None:
    info = mne.create_info(["Cz"], sfreq=100.0, ch_types=["eeg"])
    raw = mne.io.RawArray(np.zeros((1, 1_000)), info, verbose=False)
    raw.set_meas_date(datetime(2026, 1, 1, tzinfo=timezone.utc))
    raw.set_annotations(
        mne.Annotations(
            onset=[1.0, 3.0, 5.0],
            duration=[0.0] * 3,
            description=["Volume/V  1", "Trig_therm/T  1", "Volume/V  1"],
            orig_time=raw.info["meas_date"],
        )
    )
    raw.crop(tmin=2.0)

    filter_annotations(raw, event_prefixes=None, keep_all=False, zero_base=True)

    # Anchoring to absolute zero would push both events off the front of the
    # recording, which starts at first_time, and MNE would silently drop them.
    assert raw.annotations.description.tolist() == ["Trig_therm/T  1", "Volume/V  1"]
    assert raw.annotations.onset.tolist() == [2.0, 4.0]
