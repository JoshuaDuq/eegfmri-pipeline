"""Preservation evidence must separate a cleaned recording from an emptied one."""

from __future__ import annotations

from dataclasses import replace

import matplotlib
import mne
import numpy as np
import pytest

matplotlib.use("Agg")

from eeg_pipeline.preprocessing.report.preservation import (  # noqa: E402
    MINIMUM_TRIALS_FOR_SPLIT_HALF,
    add_rest_preservation_review,
    add_task_preservation_review,
    _stratified_half_indices,
    compute_posterior_alpha,
    compute_split_half_reliability,
    plot_preservation,
    preservation_html,
    resolvable_prominence_threshold,
)
from eeg_pipeline.preprocessing.report.settings import ReportSettings  # noqa: E402

SFREQ = 250.0
POSTERIOR = ["Pz", "POz", "Oz", "O1", "O2", "PO3", "PO4"]
ANTERIOR = ["Fz", "Cz", "C3", "C4"]


def _continuous(*, alpha_amplitude: float, seconds: float = 120.0, seed: int = 1):
    """Continuous data carrying posterior alpha at 10.5 Hz and nothing else."""
    rng = np.random.default_rng(seed)
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


def test_split_halves_are_balanced_within_each_event_code() -> None:
    """Blocked conditions must not become a systematic difference between halves."""
    event_codes = np.asarray([1] * 21 + [2] * 21 + [3] * 20)

    even, odd = _stratified_half_indices(event_codes)

    assert len(even) == len(odd) == 30
    for code in (1, 2, 3):
        assert np.sum(event_codes[even] == code) == 10
        assert np.sum(event_codes[odd] == code) == 10


def test_posterior_alpha_is_found_where_it_was_injected() -> None:
    alpha = compute_posterior_alpha(_continuous(alpha_amplitude=8e-6))

    assert alpha is not None
    assert set(alpha.channel_names) == set(POSTERIOR)
    assert alpha.peak_frequency_hz == pytest.approx(10.5, abs=0.6)
    assert alpha.prominence_db > 10.0
    assert alpha.has_peak


def test_the_posterior_spectrum_uses_the_same_power_reference_as_the_sensor_spectra() -> None:
    """One report must not quote power in two scales.

    The sensor-spectra panel is referenced to 1 µV²/Hz. This panel drew a bare "PSD (dB)"
    against V²/Hz, so the same recording read about 120 dB quieter here than a few
    sections earlier, and neither axis said so. Prominence is a difference and is
    unaffected either way; the level is what a reader cannot otherwise place.
    """
    from eeg_pipeline.preprocessing.report.spectra import POWER_UNIT_LABEL

    alpha = compute_posterior_alpha(_continuous(alpha_amplitude=8e-6))

    assert alpha is not None
    # A µV²-referenced EEG spectrum does not sit below -100 dB; a V²-referenced one does.
    assert alpha.power_db.max() > -60.0
    axis = plot_preservation(alpha=alpha).axes[0]
    assert POWER_UNIT_LABEL in axis.get_ylabel()


def test_the_alpha_panel_draws_the_background_its_prominence_is_measured_against() -> None:
    """The panel printed "8.3 dB over background" and never drew the background.

    Prominence here is the height of the peak over a line fitted to the surrounding
    spectrum, and that line is the whole basis of the number: absolute alpha power varies
    by an order of magnitude between participants, so the level says nothing on its own.
    The fit was computed, used, and discarded, leaving a reader with a decibel figure and
    no way to see what it was taken from.

    ``aperiodic_line_db`` exists to draw this line -- its own docstring says so.
    """
    alpha = compute_posterior_alpha(_continuous(alpha_amplitude=8e-6))

    assert alpha is not None
    assert alpha.background_db.shape == alpha.power_db.shape

    axis = plot_preservation(alpha=alpha).axes[0]
    drawn = [line.get_ydata() for line in axis.lines]
    assert any(
        np.allclose(trace, alpha.background_db) for trace in drawn
    ), "the fitted background is not drawn under the spectrum"

    # The drawn gap at the peak is the reported number, so the figure and the caption
    # cannot describe two different measurements.
    peak = int(np.argmin(np.abs(alpha.frequencies_hz - alpha.peak_frequency_hz)))
    gap = float(alpha.power_db[peak] - alpha.background_db[peak])
    assert gap == pytest.approx(alpha.prominence_db, abs=1e-6)


def test_the_alpha_panel_states_the_roughness_the_peak_had_to_clear() -> None:
    """A prominence means nothing without the scatter of the background beneath it.

    The pipeline measures that scatter and corrects it for the width of the search, and
    the panel reported neither -- so a peak 8 dB over a background that wanders 4 dB read
    the same as one over a background that wanders 0.2 dB.

    Reported as the measured threshold, not as a verdict: the reader is given the height,
    the roughness and the bar, and draws their own conclusion.
    """
    alpha = compute_posterior_alpha(_continuous(alpha_amplitude=8e-6))
    axis = plot_preservation(alpha=alpha).axes[0]

    annotations = " ".join(text.get_text().lower() for text in axis.texts)
    assert "roughness" in annotations
    assert f"{alpha.background_residual_db:.1f}" in annotations


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


def test_task_review_adds_mne_joint_evoked_plots_over_the_response_window(
    monkeypatch,
) -> None:
    """Show waveform and scalp maps, using the scientifically configured window."""
    epochs = _evoked_epochs(response_amplitude=6e-6)
    epochs.set_montage("standard_1020", verbose="ERROR")
    report = mne.Report(title="task", verbose="ERROR")
    plotted_times = []

    def fake_plot_joint(self, *, times, picks, show):
        plotted_times.append(np.asarray(times))
        assert picks == "eeg"
        assert show is False
        return matplotlib.pyplot.figure()

    monkeypatch.setattr(mne.Evoked, "plot_joint", fake_plot_joint)

    add_task_preservation_review(
        report=report,
        epochs=epochs,
        settings=ReportSettings(response_window_s=(0.0, 0.2)),
    )

    assert plotted_times[0] == pytest.approx(np.linspace(0.0, 0.2, 4))
    assert any(element.section == "Evoked responses" for element in report._content)


def test_preservation_review_uses_the_configured_window_band_and_channels() -> None:
    epochs = _evoked_epochs(response_amplitude=6e-6)
    epochs.set_montage("standard_1020", verbose="ERROR")
    report = mne.Report(title="task", verbose="ERROR")
    settings = ReportSettings(
        response_window_s=(0.1, 0.2),
        alpha_band_hz=(9.0, 11.0),
        posterior_channel_pattern="^O",
    )

    reliability, alpha = add_task_preservation_review(
        report=report,
        epochs=epochs,
        settings=settings,
    )

    assert reliability.response_window_s == pytest.approx((0.1, 0.2))
    assert alpha is not None
    assert all(name.startswith("O") for name in alpha.channel_names)
    assert 9.0 <= alpha.peak_frequency_hz <= 11.0
    assert alpha.band_hz == (9.0, 11.0)

    figure = plot_preservation(alpha=alpha)
    band = figure.axes[0].patches[0]
    assert band.get_x() == pytest.approx(9.0)
    assert band.get_x() + band.get_width() == pytest.approx(11.0)


def test_an_empty_preservation_panel_is_an_error() -> None:
    with pytest.raises(ValueError, match="at least one measurement"):
        preservation_html()


def test_the_panel_names_the_epochs_the_measurement_came_from() -> None:
    """A pre-rejection number and a post-rejection number are not interchangeable.

    The panel is worth showing at ICA-review time, before 16 of 22 components are
    approved for exclusion, because every other panel at that point measures removal.
    But the epochs it is measured on then still include the ones autoreject will drop,
    so the figure has to say which set it used or the two readings are indistinguishable.
    """
    epochs = _evoked_epochs(response_amplitude=6e-6)
    epochs.set_montage("standard_1020", verbose="ERROR")
    reliability = compute_split_half_reliability(epochs)

    document = preservation_html(
        reliability=reliability,
        analysis_status="Provisional — all task epochs",
    )

    assert "Provisional — all task epochs" in document


def test_the_panel_states_its_basis_even_when_it_is_the_final_one() -> None:
    """The default has to be stated too, or only provisional panels carry provenance."""
    epochs = _evoked_epochs(response_amplitude=6e-6)
    epochs.set_montage("standard_1020", verbose="ERROR")
    reliability = compute_split_half_reliability(epochs)

    document = preservation_html(reliability=reliability)

    assert "retained epochs" in document


def test_a_provisional_review_is_superseded_rather_than_duplicated() -> None:
    """The final pass must replace the provisional panel, not sit beside it."""
    epochs = _evoked_epochs(response_amplitude=6e-6)
    epochs.set_montage("standard_1020", verbose="ERROR")
    report = mne.Report(title="task", verbose="ERROR")

    add_task_preservation_review(
        report=report,
        epochs=epochs,
        analysis_status="Provisional — all task epochs",
    )
    add_task_preservation_review(report=report, epochs=epochs)

    assert len(report._content) == 4
    rendered = "".join(str(element.html) for element in report._content)
    assert "Provisional" not in rendered


def _posterior_evoked_epochs(*, response_amplitude: float, n_trials: int = 40):
    """Epochs whose stimulus-locked deflection is confined to posterior channels.

    :func:`_evoked_epochs` adds the same deflection to every channel. That response has
    no topography, so average referencing removes all of it, which makes those epochs
    unusable for any test about what survives referencing.
    """
    rng = np.random.default_rng(6)
    names = POSTERIOR + ANTERIOR
    info = mne.create_info(names, SFREQ, "eeg")
    n_times = int(SFREQ)
    times = np.arange(n_times) / SFREQ - 0.2
    response = response_amplitude * np.exp(-(((times - 0.15) / 0.05) ** 2))
    data = rng.normal(0, 1e-5, (n_trials, len(names), n_times))
    for index, name in enumerate(names):
        if name in POSTERIOR:
            data[:, index, :] += response
    return mne.EpochsArray(data, info, tmin=-0.2, verbose="ERROR")


def test_the_plotted_traces_survive_average_referencing() -> None:
    """The panel plotted the across-channel mean, which average referencing zeroes out.

    Average referencing subtracts the across-channel mean from every channel, so that
    mean is zero by construction afterwards and the traces were floating-point
    cancellation noise around 1e-16 µV — drawn beneath a correlation computed on the
    real data, which invited reading a reliable dataset as empty. Global field power is
    a spatial standard deviation and the reference does not remove it.
    """
    epochs = _posterior_evoked_epochs(response_amplitude=20e-6)
    epochs.set_eeg_reference("average", projection=False, verbose="ERROR")

    reliability = compute_split_half_reliability(epochs)

    assert reliability is not None
    for trace in (reliability.odd_gfp_uv, reliability.even_gfp_uv):
        # Microvolts, not the 1e-10 µV that the cancelling mean produced.
        assert trace.max() > 1.0
        assert np.all(trace >= 0.0)


def test_the_plotted_traces_follow_the_response() -> None:
    """Whatever the panel draws has to grow with the response that survived.

    This is what the across-channel mean could not do: it returned the same
    floating-point noise whether the response was intact or absent.
    """
    peaks = []
    for amplitude in (0.0, 20e-6):
        epochs = _posterior_evoked_epochs(response_amplitude=amplitude)
        epochs.set_eeg_reference("average", projection=False, verbose="ERROR")
        reliability = compute_split_half_reliability(epochs)
        peaks.append(float(reliability.odd_gfp_uv.max()))

    assert peaks[1] > 2.0 * peaks[0]


def test_spatial_agreement_is_resolved_at_each_latency() -> None:
    """The pooled reliability must be accompanied by evidence from the same halves."""
    epochs = _posterior_evoked_epochs(response_amplitude=20e-6)
    epochs.set_eeg_reference("average", projection=False, verbose="ERROR")

    reliability = compute_split_half_reliability(epochs)

    assert reliability is not None
    assert reliability.spatial_correlation.shape == reliability.times_s.shape
    assert np.all(np.isfinite(reliability.spatial_correlation))
    assert np.all(np.abs(reliability.spatial_correlation) <= 1.0)


def test_spatial_agreement_panel_has_a_fixed_correlation_scale() -> None:
    epochs = _posterior_evoked_epochs(response_amplitude=20e-6)
    epochs.set_eeg_reference("average", projection=False, verbose="ERROR")
    reliability = compute_split_half_reliability(epochs)

    figure = plot_preservation(reliability=reliability)
    spatial_axis = figure.axes[1]

    assert spatial_axis.get_ylabel() == "Pearson r across channels"
    assert spatial_axis.get_ylim() == pytest.approx((-1.0, 1.0))
    np.testing.assert_allclose(
        spatial_axis.lines[0].get_ydata(),
        reliability.spatial_correlation,
    )


# --------------------------------------------------------------------------------------
# Resolvability: the peak is a maximum, and has to be judged as one
# --------------------------------------------------------------------------------------


def test_the_search_width_sets_the_bar_a_peak_has_to_clear() -> None:
    """A wider search finds a larger maximum by chance, so it must demand a larger one.

    The prominence is the largest excess over the fitted background across every bin in the
    alpha band. Whether that maximum is remarkable depends on how many bins it was the
    maximum of, which a criterion in fixed multiples of the residual cannot express: the
    factor of two used previously sat below the two-and-a-half standard deviations a
    seventy-bin maximum reaches by chance, before any allowance for the residual and the
    excess being measured on opposite sides of the excluded window.
    """
    narrow = resolvable_prominence_threshold(10)
    wide = resolvable_prominence_threshold(200)

    assert wide > narrow
    assert narrow > 2.0
    # In multiples of the aperiodic fit residual. A real recording with an unambiguous
    # 11 dB alpha peak measures about 14 on the same scale, so the bar separates a rhythm
    # from noise by a wide margin rather than sitting between them.
    assert 3.0 < narrow < 8.0
    assert 3.0 < wide < 8.0


def test_a_recording_with_no_rhythm_is_rarely_credited_with_one() -> None:
    """The test exists to keep fabricated peaks out of the cohort's peak-frequency panel.

    Scored against the fit residual and a fixed factor of two, rhythm-free recordings
    passed about eighty per cent of the time: the residual is measured where the aperiodic
    line was fitted and so is a smaller scale than the excess where that line is
    extrapolated across the alpha window. Judging the maximum against the scatter of the
    very bins it was the maximum of puts both on one footing.
    """
    resolvable = 0
    for seed in range(30):
        alpha = compute_posterior_alpha(_continuous(alpha_amplitude=0.0, seed=seed))
        assert alpha is not None
        resolvable += int(alpha.is_resolvable())

    assert resolvable <= 6


def test_a_real_rhythm_is_still_resolved_and_located() -> None:
    """Specificity bought at the cost of finding nothing would be no improvement."""
    alpha = compute_posterior_alpha(_continuous(alpha_amplitude=4e-6))

    assert alpha is not None
    assert alpha.is_resolvable()
    assert alpha.peak_frequency_hz == pytest.approx(10.5, abs=1.0)


def test_the_search_width_is_recorded_so_the_peak_can_be_scored_as_a_maximum() -> None:
    """Without it the test cannot tell a search over ten bins from one over a hundred."""
    alpha = compute_posterior_alpha(_continuous(alpha_amplitude=0.0))

    assert alpha is not None
    assert alpha.n_search_bins > 1


def test_a_band_with_two_comparable_bumps_reports_the_contest() -> None:
    """A bistable argmax must not enter a cohort histogram as a located rhythm.

    Seen on real data: one participant's peak sat at 13.4 Hz before cleaning and 10.0 Hz
    after, with the two bumps half a decibel apart. That is a coin toss reported as a
    three-hertz shift in someone's alpha frequency.
    """
    rng = np.random.default_rng(11)
    names = POSTERIOR + ANTERIOR
    n_samples = int(SFREQ * 180.0)
    times = np.arange(n_samples) / SFREQ
    data = np.empty((len(names), n_samples))
    frequencies = np.fft.rfftfreq(n_samples, 1 / SFREQ)
    for index, name in enumerate(names):
        spectrum = np.fft.rfft(rng.standard_normal(n_samples))
        spectrum[1:] /= frequencies[1:] ** 0.5
        data[index] = np.fft.irfft(spectrum, n_samples)[:n_samples] * 1e-5
        if name in POSTERIOR:
            # Two rhythms of near-equal size, three hertz apart.
            data[index] += 6e-6 * np.sin(2 * np.pi * 9.5 * times + rng.uniform(0, 6))
            data[index] += 6e-6 * np.sin(2 * np.pi * 12.5 * times + rng.uniform(0, 6))
    raw = mne.io.RawArray(data, mne.create_info(names, SFREQ, "eeg"), verbose="ERROR")

    alpha = compute_posterior_alpha(raw)

    assert alpha is not None
    # Both injected rhythms are found, as two separate candidates rather than as one bump.
    assert np.isfinite(alpha.runner_up_frequency_hz)
    assert abs(alpha.runner_up_frequency_hz - alpha.peak_frequency_hz) >= 1.0
    found = sorted((alpha.peak_frequency_hz, alpha.runner_up_frequency_hz))
    assert found[0] == pytest.approx(9.5, abs=0.6)
    assert found[1] == pytest.approx(12.5, abs=0.6)
    # And they are close enough in height that the winner is a near thing.
    assert alpha.runner_up_gap_db < 0.5 * alpha.prominence_db


def test_a_contest_is_judged_against_the_spectrum_s_own_roughness() -> None:
    """Two bumps closer than the per-bin scatter are not separated by the data.

    On a real recording the winner led by 0.50 dB against a residual of 0.59 -- and the
    frequency duly moved 3.4 Hz when cleaning nudged the two. Scored against the spectrum's
    own noise rather than an absolute decibel figure, so the criterion travels between
    recordings of different quality.
    """
    alpha = compute_posterior_alpha(_continuous(alpha_amplitude=8e-6))
    assert alpha is not None

    near = replace(alpha, runner_up_gap_db=0.5, background_residual_db=0.59)
    clear = replace(alpha, runner_up_gap_db=4.0, background_residual_db=0.59)

    assert near.peak_is_contested
    assert not clear.peak_is_contested


def test_a_single_clear_rhythm_is_not_contested() -> None:
    alpha = compute_posterior_alpha(_continuous(alpha_amplitude=8e-6))

    assert alpha is not None
    assert not alpha.peak_is_contested


def test_a_band_with_one_bump_has_no_runner_up_at_all() -> None:
    """An uncontested peak reports no rival rather than the shoulder of itself."""
    alpha = compute_posterior_alpha(_continuous(alpha_amplitude=8e-6))

    assert alpha is not None
    if np.isfinite(alpha.runner_up_frequency_hz):
        # Whatever it found is a separate bump, never a neighbouring bin of the winner.
        assert abs(alpha.runner_up_frequency_hz - alpha.peak_frequency_hz) >= 1.0


def test_a_maximum_on_the_band_edge_is_not_a_peak() -> None:
    """The highest point of a window is not a peak unless the spectrum comes back down.

    Outside the window it may go on rising, so an edge maximum locates the edge of the
    search rather than a rhythm.
    """
    alpha = compute_posterior_alpha(_continuous(alpha_amplitude=8e-6))
    assert alpha is not None

    on_edge = replace(alpha, is_interior=False)

    assert alpha.is_resolvable()
    assert not on_edge.is_resolvable()


def test_spearman_brown_is_withheld_at_a_non_positive_correlation() -> None:
    """The formula steps up a reliability, and a non-positive split-half correlation is
    not one: the halves share no response to have more of. Applied anyway, 2r/(1+r) leaves
    the correlation range below r = -1/3 -- and short of that it still inflates a
    meaningless number, which is how -0.300 was recorded and reported as -0.858."""
    import mne
    import numpy as np

    from eeg_pipeline.preprocessing.report.preservation import compute_split_half_reliability

    rng = np.random.default_rng(3)
    info = mne.create_info([f"C{i}" for i in range(8)], 100.0, "eeg")
    # Anti-correlated halves: odd trials carry a bump, even trials carry its inverse.
    n = 40
    times = np.arange(60) / 100.0
    bump = np.exp(-((times - 0.2) ** 2) / 0.002) * 1e-5
    data = rng.normal(0, 1e-7, (n, 8, times.size))
    data[0::2] += bump
    data[1::2] -= bump
    epochs = mne.EpochsArray(data, info, tmin=0.0, verbose="ERROR")

    result = compute_split_half_reliability(epochs, response_window_s=(0.0, 0.5))

    assert result is not None
    assert result.correlation < 0.0
    assert result.corrected_correlation is None


def test_spearman_brown_is_reported_where_it_applies() -> None:
    """A positive correlation still gets its step-up, above the raw value."""
    import mne
    import numpy as np

    from eeg_pipeline.preprocessing.report.preservation import compute_split_half_reliability

    rng = np.random.default_rng(4)
    info = mne.create_info([f"C{i}" for i in range(8)], 100.0, "eeg")
    times = np.arange(60) / 100.0
    bump = np.exp(-((times - 0.2) ** 2) / 0.002) * 1e-5
    data = rng.normal(0, 3e-6, (40, 8, times.size)) + bump
    epochs = mne.EpochsArray(data, info, tmin=0.0, verbose="ERROR")

    result = compute_split_half_reliability(epochs, response_window_s=(0.0, 0.5))

    assert result is not None and result.correlation > 0.0
    assert result.corrected_correlation is not None
    assert result.corrected_correlation > result.correlation
    assert abs(result.corrected_correlation) <= 1.0
