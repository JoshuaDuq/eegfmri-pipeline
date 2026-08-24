"""Sensor spectra show focal maxima, stop at the low-pass, and exclude marked bads."""

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


def test_the_summary_names_the_fit_range_that_was_actually_used() -> None:
    raw = _raw(exponent=1.6)
    spectra = compute_run_spectra(
        raw,
        raw.copy(),
        recording_id="run-1",
        fmax=100.0,
        aperiodic_fit_range_hz=(5.0, 35.0),
    )

    document = spectra_summary_html([spectra])

    assert "5-35 Hz" in document
    assert "2-45 Hz" not in document


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


def test_the_summary_states_the_exponents_and_the_maximum_envelope_gap() -> None:
    raw = _raw(focal_channels=(0,), focal_amplitude=4e-5)
    spectra = compute_run_spectra(raw, raw.copy(), recording_id="run-1", fmax=100.0)

    document = spectra_summary_html([spectra])

    assert "run-1" in document
    assert "exponent" in document
    assert "Frequency-wise maximum above median" in document
    figure = plot_run_spectra(spectra, line_frequency=60.0)
    labels = [line.get_label() for axis in figure.axes for line in axis.lines]
    assert "maximum envelopes (not a paired sensor)" in labels
    assert "worst channel" not in " ".join(labels).lower()


def test_the_summary_exposes_aperiodic_fit_quality() -> None:
    spectra = compute_run_spectra(_raw(), _raw(), recording_id="run-1", fmax=100.0)
    fit = spectra.before.aperiodic

    document = spectra_summary_html([spectra])

    assert fit is not None
    assert "Fit quality" in document
    assert "R²" in document
    assert "RMS residual" in document
    assert f"{fit.n_bins_used}/{fit.n_bins_available}" in document


def test_the_power_axis_names_its_reference() -> None:
    """ "PSD (dB)" alone is not a unit: dB is a ratio and needs a denominator.

    MNE returns power in V^2/Hz, so the plotted values sit near -100 dB and a reader
    without the reference cannot tell whether that is an amplitude, a power, or a change.
    """
    raw = _raw()
    spectra = compute_run_spectra(raw, raw.copy(), recording_id="run-1")

    figure = plot_run_spectra(spectra)

    ylabel = figure.axes[0].get_ylabel()
    assert "V" in ylabel and "Hz" in ylabel


def test_power_is_referenced_to_the_unit_eeg_is_read_in() -> None:
    """Microvolts squared, not volts squared.

    Referenced to 1 V^2/Hz an ordinary EEG spectrum runs from about -150 to -120 dB, and
    the offset of the aperiodic fit is reported as something like -104. Those numbers are
    correct and unreadable: no EEG reference values are quoted in that scale, so a
    reviewer cannot tell a normal background from a loud one, and the two reports they
    might compare are both in a scale neither was published in. The conversion is a fixed
    120 dB, so nothing about the shape of the spectrum changes — only the ladder it is
    read against.

    Anchored on Parseval rather than a plausible-looking range: a sinusoid of amplitude A
    carries A^2/2 of power, so integrating the spectrum across the peak has exactly one
    right answer in microvolts squared.
    """
    amplitude_uv = 20.0
    n_samples = int(DURATION * SFREQ)
    times = np.arange(n_samples) / SFREQ
    tone = amplitude_uv * 1e-6 * np.sin(2 * np.pi * 37.0 * times)
    info = mne.create_info([f"C{index}" for index in range(4)], SFREQ, "eeg")
    raw = mne.io.RawArray(np.tile(tone, (4, 1)), info, verbose="ERROR")

    spectra = compute_run_spectra(raw, raw.copy(), recording_id="run-1")

    peak = np.abs(spectra.frequencies - 37.0) <= 2.0
    integrated = np.trapz(
        10.0 ** (spectra.before.median_db[peak] / 10.0),
        spectra.frequencies[peak],
    )
    assert integrated == pytest.approx(amplitude_uv**2 / 2.0, rel=1e-3)


def test_the_aperiodic_offset_is_reported_in_the_same_unit_as_the_spectrum() -> None:
    """The offset is the fitted level at 1 Hz, so it moves with the reference or misleads."""
    raw = _raw()
    spectra = compute_run_spectra(raw, raw.copy(), recording_id="run-1")

    document = spectra_summary_html([spectra])

    assert "µV" in document
    # The V^2-referenced offset for this fixture sits below -100; the µV^2 one cannot.
    assert spectra.before.aperiodic.offset_db > -100.0


def test_a_notch_does_not_set_the_power_axis() -> None:
    """A notch filter drives its band to the numerical floor, tens of dB below the data.

    Left in the limits it stretched the axis by 40 dB and squeezed the spectrum the panel
    exists to show into the top third of it.
    """
    raw = _raw()
    cleaned = raw.copy()
    spectra = compute_run_spectra(raw, cleaned, recording_id="run-1")
    # Drive one bin far below the rest, as a notch does at the line frequency.
    notch_bin = int(np.argmin(np.abs(spectra.frequencies - 60.0)))
    for stage in (spectra.before, spectra.after):
        stage.median_db[notch_bin] -= 60.0
        stage.spread_low_db[notch_bin] -= 60.0

    figure = plot_run_spectra(spectra, line_frequency=60.0)

    lower, _ = figure.axes[0].get_ylim()
    outside = np.delete(spectra.before.median_db, notch_bin)
    assert lower > float(np.min(outside)) - 25.0
    assert lower > float(spectra.before.median_db[notch_bin])


def test_the_marker_note_names_every_kind_of_line_it_draws() -> None:
    raw = _raw()
    spectra = compute_run_spectra(raw, raw.copy(), recording_id="run-1")

    figure = plot_run_spectra(spectra, line_frequency=60.0, marked_frequencies=(1.111, 2.222))

    notes = " ".join(text.get_text() for axis in figure.axes for text in axis.texts)
    assert "line-noise harmonics" in notes
    assert "configured frequencies of interest" in notes
    # Core draws marks from a line frequency and a configured list, and knows of no other
    # kind. Naming one it cannot produce sends a reader looking for a mark that is absent.
    assert "gradient" not in notes.lower()


# --------------------------------------------------------------------------------------
# A named comb must not be fitted through
# --------------------------------------------------------------------------------------


def test_a_named_comb_reaches_the_fit() -> None:
    """A spectrum with teeth on it must fit the background, not the teeth."""
    sfreq, seconds, fundamental = 500.0, 60.0, 1.111
    rng = np.random.default_rng(7)
    n_samples = int(sfreq * seconds)
    times = np.arange(n_samples) / sfreq
    data = rng.normal(0, 1e-5, (4, n_samples))
    # A comb of narrow lines across the fit range, as a gradient artifact leaves behind.
    for order in range(2, 41):
        data += 6e-6 * np.sin(2 * np.pi * order * fundamental * times)
    info = mne.create_info(["C1", "C2", "C3", "C4"], sfreq, "eeg")
    raw = mne.io.RawArray(data, info, verbose="ERROR")
    # A tooth plus the 1.5-bin skirt either side, on this run's 0.25 Hz Welch grid.
    comb = tuple(
        (order * fundamental - 0.375, order * fundamental + 0.375) for order in range(1, 54)
    )

    without = compute_run_spectra(raw, raw, recording_id="r", fmax=60.0)
    with_comb = compute_run_spectra(
        raw, raw, recording_id="r", fmax=60.0, aperiodic_exclude_hz=comb
    )

    assert without.before.aperiodic is not None
    assert with_comb.before.aperiodic is not None
    # Withholding the teeth leaves the line describing the background it is meant to.
    assert with_comb.before.aperiodic.r_squared >= without.before.aperiodic.r_squared


def test_aperiodic_exclude_hz_withholds_a_named_window():
    raw = _raw(focal_channels=(), n_channels=8, exponent=1.0)
    baseline = compute_run_spectra(raw, raw, recording_id="sub-01_run-1", fmax=100.0)
    excluded = compute_run_spectra(
        raw,
        raw,
        recording_id="sub-01_run-1",
        fmax=100.0,
        aperiodic_exclude_hz=((20.0, 25.0),),
    )
    assert excluded.before.aperiodic.exponent != baseline.before.aperiodic.exponent


def test_aperiodic_exclude_hz_empty_leaves_the_fit_untouched():
    raw = _raw(focal_channels=(), n_channels=8, exponent=1.0)
    baseline = compute_run_spectra(raw, raw, recording_id="sub-01_run-1", fmax=100.0)
    empty = compute_run_spectra(
        raw, raw, recording_id="sub-01_run-1", fmax=100.0, aperiodic_exclude_hz=()
    )
    assert empty.before.aperiodic.exponent == baseline.before.aperiodic.exponent


def test_compute_run_spectra_no_longer_takes_a_volume_rate():
    import inspect

    from eeg_pipeline.preprocessing.report import spectra

    assert (
        "gradient_fundamental_hz" not in inspect.signature(spectra.compute_run_spectra).parameters
    )
    assert not hasattr(spectra, "gradient_windows")


def test_aperiodic_exclude_hz_composes_with_notch_windows():
    # Both sources must apply together, not one in place of the other. The default fit
    # range (2-45 Hz) sits below a 60 Hz notch, so that window alone would never touch the
    # fit and "composes" would pass whether or not the two sources actually combined.
    # Widening the fit range to hold both windows makes "both excluded" separable from
    # "either excluded alone", so a later edit that made one source replace the other
    # changes the result to match a single-source fit instead of the combined one.
    raw = _raw(focal_channels=(), n_channels=8, exponent=1.0)
    shared = dict(recording_id="sub-01_run-1", fmax=100.0, aperiodic_fit_range_hz=(2.0, 90.0))
    both = compute_run_spectra(
        raw, raw, line_frequency=60.0, aperiodic_exclude_hz=((20.0, 25.0),), **shared
    )
    notch_only = compute_run_spectra(raw, raw, line_frequency=60.0, **shared)
    exclude_only = compute_run_spectra(raw, raw, aperiodic_exclude_hz=((20.0, 25.0),), **shared)
    assert both.before.aperiodic.exponent != notch_only.before.aperiodic.exponent
    assert both.before.aperiodic.exponent != exclude_only.before.aperiodic.exponent
