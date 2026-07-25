"""Preservation evidence must separate a cleaned recording from an emptied one."""

from __future__ import annotations

import matplotlib
import mne
import numpy as np
import pytest

matplotlib.use("Agg")

from eeg_pipeline.preprocessing.report.preservation import (  # noqa: E402
    MINIMUM_TRIALS_FOR_SPLIT_HALF,
    add_rest_preservation_review,
    add_task_preservation_review,
    compute_posterior_alpha,
    compute_split_half_reliability,
    plot_preservation,
    preservation_html,
)

SFREQ = 250.0
POSTERIOR = ["Pz", "POz", "Oz", "O1", "O2", "PO3", "PO4"]
ANTERIOR = ["Fz", "Cz", "C3", "C4"]


def _continuous(*, alpha_amplitude: float, seconds: float = 120.0):
    """Continuous data carrying posterior alpha at 10.5 Hz and nothing else."""
    rng = np.random.default_rng(1)
    names = POSTERIOR + ANTERIOR
    info = mne.create_info(names, SFREQ, "eeg")
    n_samples = int(seconds * SFREQ)
    times = np.arange(n_samples) / SFREQ
    data = np.empty((len(names), n_samples))
    frequencies = np.fft.rfftfreq(n_samples, 1 / SFREQ)
    for index, name in enumerate(names):
        spectrum = np.fft.rfft(rng.standard_normal(n_samples))
        spectrum[1:] /= frequencies[1:] ** 0.5
        data[index] = np.fft.irfft(spectrum, n_samples)[:n_samples] * 1e-5
        if name in POSTERIOR:
            data[index] += alpha_amplitude * np.sin(2 * np.pi * 10.5 * times + rng.uniform(0, 6))
    raw = mne.io.RawArray(data, info, verbose="ERROR")
    raw.set_montage("standard_1020", verbose="ERROR")
    return raw


def _evoked_epochs(*, response_amplitude: float, n_trials: int = 40):
    """Epochs carrying a stimulus-locked deflection of a given size."""
    rng = np.random.default_rng(2)
    names = POSTERIOR + ANTERIOR
    info = mne.create_info(names, SFREQ, "eeg")
    n_times = int(SFREQ)
    times = np.arange(n_times) / SFREQ - 0.2
    response = response_amplitude * np.exp(-(((times - 0.15) / 0.05) ** 2))
    data = rng.normal(0, 1e-5, (n_trials, len(names), n_times)) + response
    return mne.EpochsArray(data, info, tmin=-0.2, verbose="ERROR")


def test_split_half_reliability_rises_with_the_surviving_response() -> None:
    """The measurement has to order recordings by how much response is left in them.

    Absolute values are modest even for a sound dataset — with a 6 µV response against
    10 µV single-trial noise, theory puts r near 0.3 — so what matters is that the
    measurement separates a recording that kept its response from one that lost it.
    """
    scores = [
        compute_split_half_reliability(_evoked_epochs(response_amplitude=amplitude)).correlation
        for amplitude in (0.0, 6e-6, 20e-6)
    ]

    assert scores == sorted(scores)
    assert scores[0] < 0.1
    assert scores[-1] > 0.8


def test_spearman_brown_steps_up_to_the_full_trial_count() -> None:
    reliability = compute_split_half_reliability(_evoked_epochs(response_amplitude=6e-6))

    assert reliability is not None
    assert reliability.corrected_correlation > reliability.correlation


def test_split_half_reliability_collapses_when_the_response_is_gone() -> None:
    """An emptied recording scores well on every removal metric; it must fail here."""
    epochs = _evoked_epochs(response_amplitude=0.0)

    reliability = compute_split_half_reliability(epochs)

    assert reliability is not None
    assert abs(reliability.correlation) < 0.3


def test_a_long_epoch_does_not_dilute_the_score_with_baseline() -> None:
    """This pipeline epochs -7 to +15 s; correlating all of it would score near zero.

    The response occupies about a second. Averaging it into twenty seconds of baseline
    would report a sound dataset as having lost its signal, which is the failure this
    section exists to detect and so is the one it must not manufacture.
    """
    rng = np.random.default_rng(4)
    names = POSTERIOR + ANTERIOR
    info = mne.create_info(names, SFREQ, "eeg")
    n_times = int(SFREQ * 22.0)
    times = np.arange(n_times) / SFREQ - 7.0
    response = 12e-6 * np.exp(-(((times - 0.15) / 0.05) ** 2))
    data = rng.normal(0, 1e-5, (40, len(names), n_times)) + response
    epochs = mne.EpochsArray(data, info, tmin=-7.0, verbose="ERROR")

    windowed = compute_split_half_reliability(epochs)
    whole_epoch = compute_split_half_reliability(epochs, response_window_s=(-7.0, 15.0))

    assert windowed is not None and whole_epoch is not None
    assert windowed.response_window_s == (0.0, 1.0)
    # Correlating the whole epoch buries the response in baseline and reports the same
    # sound recording as having lost its signal.
    assert whole_epoch.correlation < 0.2
    assert windowed.correlation > 0.5


def test_a_window_outside_the_epoch_yields_no_measurement() -> None:
    epochs = _evoked_epochs(response_amplitude=6e-6)

    assert compute_split_half_reliability(epochs, response_window_s=(30.0, 40.0)) is None


def test_too_few_trials_yields_no_measurement() -> None:
    epochs = _evoked_epochs(response_amplitude=6e-6, n_trials=MINIMUM_TRIALS_FOR_SPLIT_HALF - 1)

    assert compute_split_half_reliability(epochs) is None


def test_trials_are_split_by_alternation_not_by_half() -> None:
    """A midpoint split would confound reliability with drift over the session.

    The response latency here shifts steadily from 150 to 450 ms, as it would with
    fatigue or habituation. A midpoint split then correlates an early-latency average
    against a late-latency one and reports the reliable response as absent. Alternating
    trials puts the same range of latencies in both halves.
    """
    rng = np.random.default_rng(5)
    names = POSTERIOR + ANTERIOR
    info = mne.create_info(names, SFREQ, "eeg")
    n_trials, n_times = 40, int(SFREQ)
    times = np.arange(n_times) / SFREQ - 0.2
    data = rng.normal(0, 1e-5, (n_trials, len(names), n_times))
    for trial in range(n_trials):
        latency = 0.15 + 0.3 * (trial / (n_trials - 1))
        data[trial] += 12e-6 * np.exp(-(((times - latency) / 0.05) ** 2))
    drifting = mne.EpochsArray(data, info, tmin=-0.2, verbose="ERROR")

    alternating = compute_split_half_reliability(drifting).correlation

    half = n_trials // 2
    window = drifting.copy().crop(tmin=0.0, tmax=drifting.times[-1])
    first = window[:half].average().get_data()
    second = window[half:].average().get_data()
    midpoint = float(np.corrcoef(first.ravel(), second.ravel())[0, 1])

    assert alternating > 0.2
    assert midpoint < 0.1
    assert alternating > midpoint


def test_posterior_alpha_is_found_where_it_was_injected() -> None:
    alpha = compute_posterior_alpha(_continuous(alpha_amplitude=8e-6))

    assert alpha is not None
    assert set(alpha.channel_names) == set(POSTERIOR)
    assert alpha.peak_frequency_hz == pytest.approx(10.5, abs=0.6)
    assert alpha.prominence_db > 10.0
    assert alpha.has_peak


def test_a_recording_without_alpha_has_a_low_prominence() -> None:
    alpha = compute_posterior_alpha(_continuous(alpha_amplitude=0.0))

    assert alpha is not None
    assert alpha.prominence_db < 5.0


def test_prominence_does_not_follow_absolute_loudness() -> None:
    """Scaling a recording must not change whether it is judged to contain a rhythm."""
    quiet = _continuous(alpha_amplitude=8e-6)
    loud = quiet.copy()
    loud._data = loud._data * 10.0

    assert compute_posterior_alpha(quiet).prominence_db == pytest.approx(
        compute_posterior_alpha(loud).prominence_db, abs=0.01
    )


def test_a_montage_without_posterior_channels_yields_no_alpha() -> None:
    info = mne.create_info(ANTERIOR, SFREQ, "eeg")
    raw = mne.io.RawArray(
        np.random.default_rng(3).normal(0, 1e-5, (len(ANTERIOR), int(SFREQ * 60))),
        info,
        verbose="ERROR",
    )
    raw.set_montage("standard_1020", verbose="ERROR")

    assert compute_posterior_alpha(raw) is None


def test_resting_state_reports_alpha_without_split_half() -> None:
    """Rest has no stimulus, so evoked reliability is undefined rather than merely weak."""
    epochs = mne.make_fixed_length_epochs(
        _continuous(alpha_amplitude=8e-6), duration=4.0, preload=True, verbose="ERROR"
    )
    report = mne.Report(title="rest", verbose="ERROR")

    alpha = add_rest_preservation_review(report=report, epochs=epochs)

    assert alpha is not None
    assert len(report._content) == 2
    assert all("signal-preservation" in element.tags for element in report._content)


def test_the_section_reports_both_measurements_for_a_task() -> None:
    epochs = _evoked_epochs(response_amplitude=6e-6)
    epochs.set_montage("standard_1020", verbose="ERROR")
    report = mne.Report(title="task", verbose="ERROR")

    reliability, alpha = add_task_preservation_review(report=report, epochs=epochs)

    assert reliability is not None
    document = preservation_html(reliability=reliability, alpha=alpha)
    assert "Spearman-Brown" in document
    assert plot_preservation(reliability=reliability, alpha=alpha).axes


def test_an_empty_preservation_panel_is_an_error() -> None:
    with pytest.raises(ValueError, match="at least one measurement"):
        preservation_html()
