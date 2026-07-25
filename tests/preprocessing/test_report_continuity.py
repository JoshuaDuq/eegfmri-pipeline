"""Time-resolved quality must localise a bad stretch inside a good run."""

from __future__ import annotations

import matplotlib
import mne
import numpy as np
import pytest

matplotlib.use("Agg")

from eeg_pipeline.preprocessing.report.continuity import (  # noqa: E402
    add_continuity_section,
    compute_run_continuity,
    continuity_html,
    plot_run_continuity,
)

SFREQ = 250.0
DURATION = 120.0
TR = 0.9


def _raw(*, disturbance=None, bad_span=None, volume_onsets=None, n_channels=8):
    rng = np.random.default_rng(0)
    info = mne.create_info([f"C{index}" for index in range(n_channels)], SFREQ, "eeg")
    n_samples = int(DURATION * SFREQ)
    data = rng.normal(0, 1e-5, (n_channels, n_samples))
    if disturbance is not None:
        start, stop, factor = disturbance
        data[:, int(start * SFREQ) : int(stop * SFREQ)] *= factor
    raw = mne.io.RawArray(data, info, verbose="ERROR")

    annotations = mne.Annotations([], [], [])
    if bad_span is not None:
        annotations += mne.Annotations(
            onset=[bad_span[0]], duration=[bad_span[1]], description=["BAD_movement"]
        )
    if volume_onsets is not None:
        annotations += mne.Annotations(
            onset=volume_onsets,
            duration=0.0,
            description=["Volume/V  1"] * len(volume_onsets),
        )
    if len(annotations):
        raw.set_annotations(annotations)
    return raw


def test_a_disturbance_is_localised_to_when_it_happened() -> None:
    run = compute_run_continuity(_raw(disturbance=(60.0, 70.0, 5.0)), recording_id="run-1")

    assert 55.0 < run.worst_window_s < 75.0
    assert run.worst_excursion_db > 10.0


def test_a_steady_run_has_no_excursion() -> None:
    run = compute_run_continuity(_raw(), recording_id="run-1")

    assert run.worst_excursion_db < 3.0


def test_amplitude_is_relative_to_each_channel_so_a_loud_channel_is_not_an_event() -> None:
    """A uniformly noisy channel is the bad-channel panel's business, not this one."""
    raw = _raw()
    raw._data[2] *= 20.0

    run = compute_run_continuity(raw, recording_id="run-1")

    assert run.worst_excursion_db < 3.0
    assert float(np.max(np.abs(run.relative_db[2]))) < 3.0


def test_bad_spans_are_reported_as_a_fraction_of_the_run() -> None:
    run = compute_run_continuity(_raw(bad_span=(30.0, 12.0)), recording_id="run-1")

    assert len(run.bad_spans) == 1
    assert run.bad_fraction == pytest.approx(12.0 / DURATION, rel=0.01)


def test_an_interruption_in_the_volume_train_is_found() -> None:
    onsets = list(np.arange(0.0, 50.0, TR)) + list(np.arange(70.0, 110.0, TR))

    run = compute_run_continuity(
        _raw(volume_onsets=onsets),
        recording_id="run-1",
        volume_description="Volume/V  1",
    )

    assert len(run.volume_gaps) == 1
    gap_onset, gap_duration = run.volume_gaps[0]
    assert gap_onset == pytest.approx(49.5, abs=1.0)
    assert gap_duration == pytest.approx(20.0, abs=1.0)


def test_an_uninterrupted_volume_train_has_no_gaps() -> None:
    run = compute_run_continuity(
        _raw(volume_onsets=list(np.arange(0.0, 110.0, TR))),
        recording_id="run-1",
        volume_description="Volume/V  1",
    )

    assert run.volume_gaps == ()


def test_bad_channels_are_excluded_from_the_map() -> None:
    raw = _raw()
    raw.info["bads"] = ["C1"]

    run = compute_run_continuity(raw, recording_id="run-1")

    assert "C1" not in run.channel_names
    assert len(run.channel_names) == 7


def test_a_run_shorter_than_two_windows_is_rejected() -> None:
    raw = _raw()
    short = raw.copy().crop(tmax=1.5)

    with pytest.raises(ValueError, match="shorter than two"):
        compute_run_continuity(short, recording_id="run-1", window_seconds=1.0)


def test_a_non_positive_window_is_rejected() -> None:
    with pytest.raises(ValueError, match="window must be positive"):
        compute_run_continuity(_raw(), recording_id="run-1", window_seconds=0.0)


def test_the_section_renders_and_replaces_on_rebuild() -> None:
    run = compute_run_continuity(
        _raw(disturbance=(60.0, 70.0, 5.0), bad_span=(30.0, 5.0)), recording_id="run-1"
    )
    report = mne.Report(title="continuity", verbose="ERROR")

    add_continuity_section(report=report, runs=[run])
    add_continuity_section(report=report, runs=[run])

    assert len(report._content) == 2
    assert all("run-continuity" in element.tags for element in report._content)
    assert "run-1" in continuity_html([run])
    assert plot_run_continuity(run).axes


def test_an_empty_section_is_an_error_rather_than_a_blank_panel() -> None:
    report = mne.Report(title="continuity", verbose="ERROR")

    with pytest.raises(ValueError, match="at least one run"):
        add_continuity_section(report=report, runs=[])
