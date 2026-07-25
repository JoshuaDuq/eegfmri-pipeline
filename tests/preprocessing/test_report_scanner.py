"""Residual gradient evidence must find a focal comb the channel median cannot see."""

from __future__ import annotations

import matplotlib
import mne
import numpy as np
import pytest

matplotlib.use("Agg")

from eeg_pipeline.preprocessing.report.scanner import (  # noqa: E402
    MINIMUM_VOLUMES,
    VOLUME_MARKER_DESCRIPTION,
    VolumeTiming,
    add_scanner_residual_section,
    compute_comb_residual,
    compute_volume_locked_average,
    measure_volume_timing,
    plot_comb_residual,
    plot_volume_locked_average,
    scanner_residual_html,
)

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

    assert comb is None


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


def test_the_section_reports_both_measurements() -> None:
    raw = _raw(comb_channels=(0, 1))
    timing = measure_volume_timing(raw)
    comb = compute_comb_residual(raw, raw.copy(), timing=timing, recording_id="run-1")
    locked = compute_volume_locked_average(raw, raw.copy(), timing=timing, recording_id="run-1")
    report = mne.Report(title="scanner", verbose="ERROR")

    add_scanner_residual_section(report=report, combs=[comb], averages=[locked])

    assert len(report._content) == 3
    assert all("scanner-residual" in element.tags for element in report._content)

    document = scanner_residual_html([comb], [locked])
    assert f"{TR:.4f}" in document
    assert comb.worst_channel in document
    assert plot_comb_residual([comb]).axes
    assert plot_volume_locked_average([locked]).axes


def test_rebuilding_the_section_replaces_rather_than_accumulates() -> None:
    raw = _raw()
    timing = measure_volume_timing(raw)
    comb = compute_comb_residual(raw, raw.copy(), timing=timing, recording_id="run-1")
    report = mne.Report(title="scanner", verbose="ERROR")

    add_scanner_residual_section(report=report, combs=[comb], averages=[])
    add_scanner_residual_section(report=report, combs=[comb], averages=[])

    assert len(report._content) == 2


def test_an_empty_section_is_an_error_rather_than_a_blank_panel() -> None:
    report = mne.Report(title="scanner", verbose="ERROR")

    with pytest.raises(ValueError, match="at least one measured run"):
        add_scanner_residual_section(report=report, combs=[], averages=[])
