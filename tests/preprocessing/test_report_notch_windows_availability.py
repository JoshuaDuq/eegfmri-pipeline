from __future__ import annotations

import numpy as np

from eeg_pipeline.preprocessing.report.filtering import in_notch, notch_windows


def test_no_line_frequency_and_no_intervals_owns_nothing() -> None:
    assert notch_windows(None, fmax=100.0) == ()


def test_line_frequency_alone_is_the_harmonic_grid() -> None:
    windows = notch_windows(60.0, fmax=100.0, half_width=2.0)

    assert windows == ((58.0, 62.0),)


def test_recorded_intervals_own_their_bands_without_a_configured_notch() -> None:
    """A Decomb manifest sets notch_freq to null, so the intervals are the only source."""
    windows = notch_windows(
        None,
        fmax=100.0,
        unavailable_intervals=((59.8, 60.2), (57.0, 57.5)),
    )

    assert windows == ((57.0, 57.5), (59.8, 60.2))


def test_recorded_intervals_are_used_as_measured_not_widened() -> None:
    # The manifest interval already spans the stopband and its FIR transitions, so
    # padding it by the fixed half-width would claim bins the filter never touched.
    windows = notch_windows(None, fmax=100.0, half_width=2.0, unavailable_intervals=((59.8, 60.2),))

    assert windows == ((59.8, 60.2),)


def test_intervals_above_the_ceiling_are_dropped_and_partial_ones_clipped() -> None:
    windows = notch_windows(
        None,
        fmax=100.0,
        unavailable_intervals=((95.0, 120.0), (140.0, 150.0)),
    )

    assert windows == ((95.0, 100.0),)


def test_configured_notch_and_recorded_intervals_combine() -> None:
    windows = notch_windows(
        60.0,
        fmax=100.0,
        half_width=2.0,
        unavailable_intervals=((30.0, 31.0),),
    )

    assert windows == ((30.0, 31.0), (58.0, 62.0))


def test_recorded_intervals_mask_the_frequencies_they_cover() -> None:
    windows = notch_windows(None, fmax=100.0, unavailable_intervals=((59.8, 60.2),))
    frequencies = np.array([59.0, 59.9, 60.0, 60.5])

    assert in_notch(frequencies, windows).tolist() == [False, True, True, False]


class _Config:
    def __init__(self, values: dict) -> None:
        self._values = values

    def get(self, key: str, default=None):
        return self._values.get(key, default)


def test_settings_carry_no_intervals_without_a_configured_manifest() -> None:
    from eeg_pipeline.preprocessing.report.settings import ReportSettings

    settings = ReportSettings.from_config(_Config({"preprocessing.notch_freq": 60}))

    assert settings.unavailable_intervals_by_recording == {}
    assert settings.spectra_line_frequency == 60


def test_settings_key_manifest_intervals_by_recording_id(tmp_path) -> None:
    import json

    from eeg_pipeline.preprocessing.report.settings import ReportSettings

    (tmp_path / "dataset_description.json").write_text(
        json.dumps({"Name": "d", "GeneratedBy": [{"Name": "decomb", "Version": "1"}]}),
        encoding="utf-8",
    )
    rows = [
        ("recording", "unavailable_low_hz", "unavailable_high_hz", "outcome", "removal_round"),
        ("sub-0000_task-thermalactive_run-1_eeg", "59.8", "60.2", "line_detected", "1"),
        ("sub-0000_task-thermalactive_run-1_eeg", "", "", "no_line_detected", ""),
    ]
    manifest = tmp_path / "line_notch_manifest.tsv"
    manifest.write_text("\n".join("\t".join(row) for row in rows) + "\n", encoding="utf-8")

    settings = ReportSettings.from_config(
        _Config({"paths.decomb_manifest": str(manifest), "preprocessing.notch_freq": None})
    )

    # The key matches the recording id the report derives from the filtered raw filename.
    assert settings.unavailable_intervals_by_recording == {
        "sub-0000_task-thermalactive_run-1": ((59.8, 60.2),)
    }
    assert settings.spectra_line_frequency is None
