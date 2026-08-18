"""Residual gradient evidence must find a focal comb the channel median cannot see."""

from __future__ import annotations

import mne
import numpy as np
import pytest

from studies.pain_study.analysis.gradient.comb import (
    CombNotMeasured,
    CombResidual,
    MINIMUM_VOLUMES,
    VOLUME_MARKER_DESCRIPTION,
    VolumeTiming,
    compute_comb_residual,
    measure_volume_timing,
)
from studies.pain_study.analysis.gradient.locked import compute_volume_locked_average

SFREQ = 500.0
TR = 0.9
DURATION = 120.0


def _raw(*, comb_channels=(0, 1), comb_amplitude=3e-6, n_channels=12, with_markers=True):
    rng = np.random.default_rng(0)
    names = [f"C{index}" for index in range(n_channels)]
    info = mne.create_info(names, SFREQ, "eeg")
    n_samples = int(DURATION * SFREQ)
    times = np.arange(n_samples) / SFREQ
    data = rng.normal(0, 1e-5, (n_channels, n_samples))
    for channel in comb_channels:
        for order in range(18, 60):
            data[channel] += comb_amplitude * np.sin(
                2 * np.pi * (order / TR) * times + rng.uniform(0, 6)
            )
    raw = mne.io.RawArray(data, info, verbose="ERROR")
    if with_markers:
        onsets = np.arange(0.0, DURATION - TR, TR)
        raw.set_annotations(
            mne.Annotations(
                onset=onsets,
                duration=0.0,
                description=[VOLUME_MARKER_DESCRIPTION] * len(onsets),
            )
        )
    return raw


def test_volume_rate_is_measured_from_the_markers() -> None:
    timing = measure_volume_timing(_raw())

    assert timing is not None
    assert timing.repetition_time_s == pytest.approx(TR, abs=1e-6)
    assert timing.fundamental_hz == pytest.approx(1.0 / TR, abs=1e-6)
    assert timing.n_volumes > MINIMUM_VOLUMES


def test_a_recording_without_volume_markers_has_no_timing() -> None:
    """A dataset recorded outside a scanner gets no section, not an empty one."""
    assert measure_volume_timing(_raw(with_markers=False)) is None


def test_harmonics_fall_inside_the_requested_band() -> None:
    timing = measure_volume_timing(_raw())

    harmonics = timing.harmonics(fmin=15.0, fmax=90.0)

    assert harmonics
    assert min(harmonics) >= 15.0
    assert max(harmonics) <= 90.0
    spacing = np.diff(harmonics)
    assert np.allclose(spacing, 1.0 / TR, atol=1e-6)


def test_a_reversed_harmonic_band_is_rejected() -> None:
    timing = VolumeTiming(
        n_volumes=100,
        repetition_time_s=TR,
        interval_jitter_s=0.0,
    )

    with pytest.raises(ValueError, match="empty or reversed"):
        timing.harmonics(fmin=90.0, fmax=15.0)


def test_a_focal_comb_is_found_in_the_worst_channel_not_the_median() -> None:
    """This is the whole point of the section: gradient residual is focal.

    A comb confined to two of twelve sensors leaves the across-channel median flat. A
    measurement that collapses channels before scoring the comb reports a clean run.
    """
    raw = _raw(comb_channels=(0, 1))
    timing = measure_volume_timing(raw)

    comb = compute_comb_residual(raw, raw.copy(), timing=timing, recording_id="sub-01_run-1")

    assert comb is not None
    assert comb.before_excess_db.shape == (12, comb.harmonic_frequencies_hz.size)
    assert float(np.max(comb.before_worst_db)) > 20.0
    assert float(np.max(comb.before_typical_db)) < 6.0
    assert comb.worst_channel in {"C0", "C1"}


def _comb_residual(recording_id: str = "run-1") -> CombResidual:
    # Hand-built rather than measured, so a test can pin exact channel/harmonic values.
    timing = VolumeTiming(n_volumes=100, repetition_time_s=1.0, interval_jitter_s=0.0)
    return CombResidual(
        recording_id=recording_id,
        timing=timing,
        harmonic_frequencies_hz=np.asarray([20.0, 21.0, 22.0]),
        channel_names=("global-maximum", "persistent-comb"),
        before_excess_db=np.zeros((2, 3)),
        after_excess_db=np.asarray([[20.0, -10.0, -10.0], [8.0, 8.0, 8.0]]),
    )


def test_largest_line_value_frequency_and_channel_share_one_index() -> None:
    """A measurement must not combine two different definitions of "worst"."""
    comb = _comb_residual()

    assert comb.worst_excess_db == pytest.approx(20.0)
    assert comb.worst_harmonic_hz == pytest.approx(20.0)
    assert comb.worst_channel == "global-maximum"
    assert comb.persistent_worst_channel == "persistent-comb"


def test_a_clean_recording_shows_no_comb() -> None:
    raw = _raw(comb_channels=())
    timing = measure_volume_timing(raw)

    comb = compute_comb_residual(raw, raw.copy(), timing=timing, recording_id="sub-01_run-1")

    assert comb is not None
    assert abs(comb.median_before_excess_db) < 3.0
    assert float(np.max(comb.before_worst_db)) < 12.0


def test_a_resolution_too_coarse_for_the_comb_reports_no_measurement() -> None:
    """Below a few bins per harmonic the excess is set by the window, not the data."""
    raw = _raw()
    timing = measure_volume_timing(raw)

    comb = compute_comb_residual(
        raw,
        raw.copy(),
        timing=timing,
        recording_id="sub-01_run-1",
        welch_seconds=1.0,
    )

    assert isinstance(comb, CombNotMeasured)
    assert "resolution" in comb.reason


def test_a_run_whose_every_harmonic_was_removed_upstream_says_so() -> None:
    """The case that made four of sub-0001's six runs disappear from the table.

    Upstream line removal had notched every integer multiple of the volume rate, so there
    was no comb line left to score. Dropping the run silently leaves the reader with a
    table of the two runs that still had a comb and no reason to think the others differ.
    """
    raw = _raw(comb_channels=(0, 1))
    timing = measure_volume_timing(raw)

    comb = compute_comb_residual(
        raw,
        raw.copy(),
        timing=timing,
        recording_id="sub-01_run-1",
        unavailable_intervals=[(0.0, 200.0)],
    )

    assert isinstance(comb, CombNotMeasured)
    assert comb.recording_id == "sub-01_run-1"
    assert comb.n_harmonics > 0
    assert comb.n_notched == comb.n_harmonics
    assert "stopband" in comb.reason


def _notched_raw(*, line_frequency=60.0, depth=1e-3, stopband_half_width=0.25):
    """A run whose comb is intact except at the line frequency, which is notched out.

    The stopband is narrow on purpose, matching what MNE's notch actually applies
    (``notch_widths`` defaults to ``freq / 200``, so 0.3 Hz at 60 Hz). Width is what
    makes the trough visible at all: the comb excess is a ratio of the power at a
    harmonic to the background a third of a harmonic-spacing away, so a stopband wide
    enough to cover both attenuates them together and the ratio barely moves. A real
    notch is narrower than that gap, kills the peak alone, and the harmonic scores tens
    of decibels below its own background.
    """
    raw = _raw(comb_channels=tuple(range(12)))
    data = raw.get_data()
    n_samples = data.shape[1]
    spectrum = np.fft.rfft(data, axis=1)
    frequencies = np.fft.rfftfreq(n_samples, d=1.0 / SFREQ)
    stopband = np.abs(frequencies - line_frequency) <= stopband_half_width
    spectrum[:, stopband] *= depth
    notched = mne.io.RawArray(
        np.fft.irfft(spectrum, n=n_samples, axis=1), raw.info.copy(), verbose="ERROR"
    )
    notched.set_annotations(raw.annotations)
    return notched


def test_a_harmonic_inside_the_notch_is_marked_and_left_out_of_the_statistics() -> None:
    """A notch drives its band to the numerical floor, so a comb line landing in one
    scores tens of decibels *below* its own background: the pipeline's own filter,
    measured as though the gradient correction had removed the artifact.

    The reported median barely moves — it is a median over thousands of values, and a
    handful of extreme ones do not shift it, which is what a median is for. What the mask
    buys is that no reported figure is ever taken from a stopband, and that the figure
    stops drawing the filter as the deepest feature in the run.
    """
    raw = _notched_raw()
    timing = measure_volume_timing(raw)

    scored = compute_comb_residual(
        raw, raw.copy(), timing=timing, recording_id="run-1", line_frequency=60.0
    )
    unmasked = compute_comb_residual(
        raw, raw.copy(), timing=timing, recording_id="run-1", line_frequency=None
    )

    assert scored.notched.any()
    # Every masked harmonic really is inside the stopband, and the line itself is masked.
    assert np.all(np.abs(scored.harmonic_frequencies_hz[scored.notched] - 60.0) <= 2.0)
    assert scored.notched[int(np.argmin(np.abs(scored.harmonic_frequencies_hz - 60.0)))]
    # The trough is in the data and still drawn, and it is no longer inside the scored set.
    trough_db = float(np.min(unmasked.after_excess_db))
    assert trough_db < -10.0
    assert float(np.min(scored.after_excess_db[:, scored.scored])) > trough_db


def test_an_unnotched_recording_scores_every_harmonic() -> None:
    """Passing no line frequency is the EEG-only case, not a reason to drop lines."""
    raw = _raw(comb_channels=(0, 1))
    timing = measure_volume_timing(raw)

    comb = compute_comb_residual(raw, raw.copy(), timing=timing, recording_id="run-1")

    assert not comb.notched.any()
    assert comb.scored.all()


def test_a_channel_offset_does_not_enter_the_locked_residual() -> None:
    """The locked residual names a waveform, and a peak-to-peak on a non-negative RMS
    trace is not blind to the level that trace sits at: adding a constant to a channel
    raises its squared contribution unevenly across latencies and inflates the range.

    On sub-0015 this accounted for roughly 40% of every reported figure. A constant per
    epoch cannot be phase-locked to the volume marker, so removing it changes the number
    without changing the artifact.
    """
    raw = _raw(comb_channels=tuple(range(12)), comb_amplitude=5e-6)
    timing = measure_volume_timing(raw)

    plain = compute_volume_locked_average(
        raw, raw.copy(), timing=timing, recording_id="run-1"
    )

    offset = raw.copy()
    offset._data = offset.get_data() + 4e-5
    shifted = compute_volume_locked_average(
        offset, offset.copy(), timing=timing, recording_id="run-1"
    )

    assert shifted.before_peak_to_peak_uv == pytest.approx(
        plain.before_peak_to_peak_uv, rel=1e-6
    )


def test_cleaning_lowers_the_measured_comb() -> None:
    raw = _raw(comb_channels=(0, 1))
    cleaned = _raw(comb_channels=())
    timing = measure_volume_timing(raw)

    comb = compute_comb_residual(raw, cleaned, timing=timing, recording_id="sub-01_run-1")

    assert float(np.max(comb.after_worst_db)) < float(np.max(comb.before_worst_db))


def test_the_volume_locked_average_recovers_the_periodic_residual() -> None:
    raw = _raw(comb_channels=tuple(range(12)), comb_amplitude=5e-6)
    cleaned = _raw(comb_channels=())
    timing = measure_volume_timing(raw)

    locked = compute_volume_locked_average(raw, cleaned, timing=timing, recording_id="sub-01_run-1")

    assert locked is not None
    assert locked.times_s[-1] < TR
    assert locked.before_peak_to_peak_uv > locked.after_peak_to_peak_uv


def test_the_volume_locked_average_needs_markers() -> None:
    raw = _raw(with_markers=False)
    timing = VolumeTiming(
        n_volumes=100,
        repetition_time_s=TR,
        interval_jitter_s=0.0,
    )

    assert (
        compute_volume_locked_average(raw, raw.copy(), timing=timing, recording_id="sub-01_run-1")
        is None
    )


def test_the_locked_average_measures_the_agreement_between_its_halves() -> None:
    """Carried from the estimator rather than recomputed, so the table and the floor
    cannot disagree about which epochs went into which half."""
    raw = _raw(comb_channels=tuple(range(12)), comb_amplitude=5e-6)
    timing = measure_volume_timing(raw)

    locked = compute_volume_locked_average(
        raw, raw.copy(), timing=timing, recording_id="run-1"
    )

    assert locked.before_half_correlation > 0.9
    assert locked.after_half_correlation > 0.9
