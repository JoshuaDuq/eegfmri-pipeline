"""Sensor spectra must show the worst channel, stop at the low-pass, and drop bads."""

from __future__ import annotations

import matplotlib
import mne
import numpy as np
import pytest

matplotlib.use("Agg")

from eeg_pipeline.preprocessing.report.spectra import (  # noqa: E402
    SPREAD_PERCENTILES,
    compute_run_spectra,
    plot_run_spectra,
    spectra_summary_html,
)

SFREQ = 500.0
DURATION = 60.0


def _raw(*, focal_channels=(), focal_amplitude=0.0, n_channels=10, exponent=1.0):
    rng = np.random.default_rng(0)
    info = mne.create_info([f"C{index}" for index in range(n_channels)], SFREQ, "eeg")
    n_samples = int(DURATION * SFREQ)
    times = np.arange(n_samples) / SFREQ
    frequencies = np.fft.rfftfreq(n_samples, 1 / SFREQ)
    data = np.empty((n_channels, n_samples))
    for index in range(n_channels):
        spectrum = np.fft.rfft(rng.standard_normal(n_samples))
        spectrum[1:] /= frequencies[1:] ** (exponent / 2.0)
        data[index] = np.fft.irfft(spectrum, n_samples)[:n_samples] * 1e-5
    for channel in focal_channels:
        data[channel] += focal_amplitude * np.sin(2 * np.pi * 37.0 * times)
    return mne.io.RawArray(data, info, verbose="ERROR")


def test_a_focal_peak_appears_in_the_maximum_not_the_median() -> None:
    """The median across channels is why a two-channel failure went unseen."""
    raw = _raw(focal_channels=(0,), focal_amplitude=4e-5)

    spectra = compute_run_spectra(raw, raw.copy(), recording_id="run-1")

    # Prominence over the neighbouring frequencies, not over the whole trace: the 1/f
    # slope alone makes any single frequency differ from the all-frequency median.
    peak = np.abs(spectra.frequencies - 37.0) <= 0.5
    neighbourhood = (np.abs(spectra.frequencies - 37.0) > 2.0) & (
        np.abs(spectra.frequencies - 37.0) <= 7.0
    )

    def prominence(trace: np.ndarray) -> float:
        return float(np.max(trace[peak]) - np.median(trace[neighbourhood]))

    assert prominence(spectra.before.max_db) > 20.0
    assert prominence(spectra.before.median_db) < 5.0


def test_the_spread_band_brackets_the_median() -> None:
    spectra = compute_run_spectra(_raw(), _raw(), recording_id="run-1")

    low, high = (spectra.before.spread_low_db, spectra.before.spread_high_db)
    assert np.all(low <= spectra.before.median_db)
    assert np.all(spectra.before.median_db <= high)
    assert np.all(high <= spectra.before.max_db)
    assert SPREAD_PERCENTILES == (10.0, 90.0)


def test_the_axis_stops_at_the_configured_low_pass() -> None:
    """Above the low-pass the filter sets the trace, so plotting further is roll-off."""
    raw = _raw()

    spectra = compute_run_spectra(raw, raw.copy(), recording_id="run-1", fmax=100.0)

    assert spectra.frequencies[-1] <= 100.0
    assert "low-pass" in spectra.fmax_reason


def test_without_a_low_pass_the_axis_stops_below_nyquist() -> None:
    raw = _raw()

    spectra = compute_run_spectra(raw, raw.copy(), recording_id="run-1")

    assert spectra.frequencies[-1] < SFREQ / 2
    assert "Nyquist" in spectra.fmax_reason


def test_a_low_pass_above_nyquist_falls_back_to_nyquist() -> None:
    raw = _raw()

    spectra = compute_run_spectra(raw, raw.copy(), recording_id="run-1", fmax=10_000.0)

    assert spectra.frequencies[-1] < SFREQ / 2
    assert "Nyquist" in spectra.fmax_reason


def test_bad_channels_do_not_enter_the_spread() -> None:
    raw = _raw()
    raw.info["bads"] = ["C0"]

    spectra = compute_run_spectra(raw, raw.copy(), recording_id="run-1")

    assert spectra.n_channels == 9


def test_the_aperiodic_fit_recovers_the_generated_slope() -> None:
    raw = _raw(exponent=1.6)

    spectra = compute_run_spectra(raw, raw.copy(), recording_id="run-1", fmax=100.0)

    assert spectra.before.aperiodic is not None
    assert spectra.before.aperiodic.exponent == pytest.approx(1.6, abs=0.15)
    assert spectra.exponent_change == pytest.approx(0.0, abs=0.01)


def test_cleaning_that_flattens_the_background_shows_as_an_exponent_change() -> None:
    before = _raw(exponent=1.6)
    after = _raw(exponent=0.8)

    spectra = compute_run_spectra(before, after, recording_id="run-1", fmax=100.0)

    assert spectra.exponent_change is not None
    assert spectra.exponent_change < -0.5


def test_an_empty_band_is_rejected() -> None:
    raw = _raw()

    with pytest.raises(ValueError, match="no frequency range"):
        compute_run_spectra(raw, raw.copy(), recording_id="run-1", fmin=200.0, fmax=100.0)


def test_the_summary_states_the_exponents_and_the_worst_channel_gap() -> None:
    raw = _raw(focal_channels=(0,), focal_amplitude=4e-5)
    spectra = compute_run_spectra(raw, raw.copy(), recording_id="run-1", fmax=100.0)

    document = spectra_summary_html([spectra])

    assert "run-1" in document
    assert "exponent" in document
    assert "Worst channel above median" in document
    assert plot_run_spectra(spectra, line_frequency=60.0).axes
