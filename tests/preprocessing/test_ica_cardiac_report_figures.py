"""The cardiac review panels have to make a bad R-peak detection visible.

Detection quality is the thing this section is read for: every downstream cardiac
measurement inherits it. These tests pin the choices that decide whether a reviewer can
see a misplaced or doubled detection at a glance.
"""

from __future__ import annotations

import matplotlib
import mne
import numpy as np

matplotlib.use("Agg")

from eeg_pipeline.preprocessing.ica_cardiac_report import (  # noqa: E402
    _plot_component_cardiac_review,
    _plot_run_cardiac_review,
)
from eeg_pipeline.preprocessing.ica_cardiac_review import (  # noqa: E402
    ComponentCardiacReview,
    RunCardiacReview,
)

SFREQ = 250.0
CHANNELS = ["Fp1", "Fp2", "Cz", "Pz", "O1", "O2"]


def _ica_info() -> mne.Info:
    info = mne.create_info(CHANNELS, SFREQ, "eeg")
    info.set_montage("standard_1020", verbose="ERROR")
    return info


class _Ica:
    """The plotting code needs only a montage and a component matrix."""

    def __init__(self) -> None:
        self.info = _ica_info()
        self.n_components_ = 2

    def get_components(self) -> np.ndarray:
        return np.tile(np.linspace(-1.0, 1.0, len(CHANNELS))[:, None], (1, 2))


def _run_review(
    *,
    heart_rate_bpm: np.ndarray | None = None,
    average_pulse_bpm: float = 60.0,
) -> RunCardiacReview:
    times = np.arange(0.0, 10.0, 1.0 / SFREQ)
    peak_times = np.arange(0.5, 10.0, 1.0)
    # A synthetic ECG whose R peaks are exactly where the detections claim they are.
    ecg = np.zeros_like(times)
    for peak in peak_times:
        ecg += np.exp(-(((times - peak) / 0.01) ** 2))
    rate = np.full(peak_times.size - 1, 60.0) if heart_rate_bpm is None else heart_rate_bpm
    locked = np.linspace(-0.4, 0.6, 50)
    return RunCardiacReview(
        recording_id="sub-01_task-x_run-1",
        representative_times=times,
        representative_ecg_mv=ecg,
        representative_peak_times=peak_times,
        rr_times=np.arange(rate.size, dtype=float),
        heart_rate_bpm=rate,
        locked_times=locked,
        before_gfp_uv=np.ones_like(locked) * 2.0,
        after_gfp_uv=np.ones_like(locked),
        r_locked_epoch_count=int(peak_times.size),
        average_pulse_bpm=average_pulse_bpm,
        events=np.zeros((peak_times.size, 3), dtype=int),
        before_topography_uv=np.linspace(-2.0, 2.0, len(CHANNELS)),
        after_topography_uv=np.linspace(-1.0, 1.0, len(CHANNELS)),
        topography_time=0.17,
    )


def _ecg_axis(figure):
    return next(axis for axis in figure.axes if "Representative ECG" in axis.get_title())


def _heart_rate_axis(figure):
    return next(axis for axis in figure.axes if "heart rate" in axis.get_title())


def test_detected_peaks_are_marked_on_the_ecg_trace() -> None:
    """Full-height rules beside an R peak look identical to ones on top of it.

    A marker placed at the amplitude the ECG actually had makes a detection that missed
    the peak fall visibly off the waveform, which a vertical line cannot do.
    """
    review = _run_review()

    figure = _plot_run_cardiac_review(review, ica=_Ica())

    axis = _ecg_axis(figure)
    assert not axis.get_lines()[1:], "peaks should not be drawn as full-height rules"
    offsets = np.concatenate([collection.get_offsets() for collection in axis.collections])
    np.testing.assert_allclose(np.sort(offsets[:, 0]), review.representative_peak_times)
    # Every marker sits on the waveform, which is what makes a stray one obvious.
    assert offsets[:, 1].min() > 0.5


def test_the_heart_rate_panel_is_not_bounded_by_a_single_implausible_beat() -> None:
    """One 170 bpm interval must not compress three hundred real beats into a band."""
    rate = np.full(300, 60.0)
    rate[10] = 170.0
    rate[20] = 20.0

    figure = _plot_run_cardiac_review(_run_review(heart_rate_bpm=rate), ica=_Ica())

    lower, upper = _heart_rate_axis(figure).get_ylim()
    assert upper < 120.0
    assert lower > 30.0


def test_beats_left_outside_the_drawn_range_are_counted_not_hidden() -> None:
    rate = np.full(300, 60.0)
    rate[10] = 170.0

    figure = _plot_run_cardiac_review(_run_review(heart_rate_bpm=rate), ica=_Ica())

    annotations = [text.get_text() for text in _heart_rate_axis(figure).texts]
    assert any("outside this range" in text for text in annotations)


def test_the_two_rates_are_labelled_by_what_each_one_measures() -> None:
    """The figure showed "MNE average pulse 57.1 bpm" beside its own "median 68 bpm".

    They are not two estimates of one quantity. MNE's ``average_pulse`` is beats divided
    by recording minutes, so a detector that misses beats lowers it; the median is the
    typical instantaneous ``60/RR``. Presenting the first as *the* pulse next to a
    contradicting median invited the reader to treat one of them as wrong.
    """
    rate = np.full(300, 68.0)

    figure = _plot_run_cardiac_review(
        _run_review(heart_rate_bpm=rate, average_pulse_bpm=57.1), ica=_Ica()
    )

    suptitle = figure.get_suptitle()
    assert "average pulse" not in suptitle.lower()
    # Whatever it is called, the title must say the number is a yield over the recording.
    assert "57.1" in suptitle and "per recording minute" in suptitle
    labels = [text.get_text() for text in _heart_rate_axis(figure).get_legend().get_texts()]
    assert any("median" in label and "68" in label for label in labels)


def test_the_ratio_between_the_two_rates_is_stated_on_the_panel() -> None:
    """The two rates agree only when every beat was detected, so the ratio is the useful
    quantity. It is a measurement and is shown unconditionally: gating it on a threshold
    would make the figure assert a verdict about the detector instead of reporting."""
    figure = _plot_run_cardiac_review(
        _run_review(heart_rate_bpm=np.full(300, 68.0), average_pulse_bpm=57.1), ica=_Ica()
    )

    annotations = " ".join(text.get_text() for text in _heart_rate_axis(figure).texts)
    assert "84%" in annotations

    agreeing = _plot_run_cardiac_review(
        _run_review(heart_rate_bpm=np.full(300, 60.0), average_pulse_bpm=60.0), ica=_Ica()
    )
    assert "100%" in " ".join(text.get_text() for text in _heart_rate_axis(agreeing).texts)


def test_the_panel_draws_no_verdict_about_the_detector() -> None:
    """A ratio below one has several causes and the figure cannot tell them apart."""
    figure = _plot_run_cardiac_review(
        _run_review(heart_rate_bpm=np.full(300, 68.0), average_pulse_bpm=40.0), ica=_Ica()
    )

    text = " ".join(
        [figure.get_suptitle()] + [artist.get_text() for artist in _heart_rate_axis(figure).texts]
    ).lower()
    for verdict in ("fail", "undetected", "missed", "poor", "bad", "warning"):
        assert verdict not in text


def test_successive_beats_are_not_joined_into_a_continuous_trace() -> None:
    """Joining them turns an alternating detection pattern into a block of ink."""
    figure = _plot_run_cardiac_review(_run_review(), ica=_Ica())

    axis = _heart_rate_axis(figure)
    # Asserted against the beats rather than against a line count, which the half- and
    # double-rate guides legitimately raised. Every line on this panel is a horizontal
    # guide spanning the axis; none of them traces the beat series.
    beat_count = _run_review().heart_rate_bpm.size
    for line in axis.get_lines():
        assert len(line.get_xdata()) < beat_count
    assert "median" in axis.get_legend().get_texts()[0].get_text()


def test_the_half_rate_line_names_where_a_missed_beat_lands() -> None:
    """A missed beat spans two intervals and halves the instantaneous rate, so dropout
    appears as a second cloud parallel to the median rather than as scatter. sub-0015's
    run-1 carries a clear one that the panel drew without naming."""
    review = _run_review()
    figure = _plot_run_cardiac_review(review, ica=_Ica())

    axis = _heart_rate_axis(figure)
    median_bpm = float(np.median(review.heart_rate_bpm))
    guides = [float(line.get_ydata()[0]) for line in axis.get_lines()]

    assert any(abs(value - median_bpm * 0.5) < 1e-6 for value in guides)
    assert any(abs(value - median_bpm * 2.0) < 1e-6 for value in guides)


def _component_review() -> ComponentCardiacReview:
    times = np.linspace(-0.4, 0.6, 40)
    runs = 2
    return ComponentCardiacReview(
        run_ids=("sub-01_task-x_run-1", "sub-01_task-x_run-2"),
        times=times,
        run_mean_z=np.zeros((runs, 2, times.size)),
        correlation_scores=np.array([[0.05, 0.01], [0.06, 0.02]]),
        ctps_scores=np.array([[0.08, 0.01], [0.07, 0.02]]),
        correlation_flags=np.array([[True, False], [False, False]]),
        ctps_flags=np.zeros((runs, 2), dtype=bool),
        r_locked_epoch_counts=np.array([10, 10]),
        run_ecg_z=np.zeros((runs, times.size)),
    )


def test_the_score_panel_scales_to_the_scores_it_shows() -> None:
    """A fixed +/-0.3 floor flattened every score onto a line through zero."""
    figure = _plot_component_cardiac_review(
        _component_review(),
        ica=_Ica(),
        component=0,
        status="RETAINED",
        status_description="",
    )

    axis = next(a for a in figure.axes if a.get_title() == "ECG correlation (r)")
    lower, upper = axis.get_ylim()
    assert upper < 0.2
    assert lower > -0.1


def test_a_flag_ring_never_wears_a_run_colour() -> None:
    """The ring sits on a run-coloured point, so a hue would be ambiguous."""
    from eeg_pipeline.preprocessing.report.style import MARK_COLOR, RUN_COLORS

    figure = _plot_component_cardiac_review(
        _component_review(),
        ica=_Ica(),
        component=0,
        status="RETAINED",
        status_description="",
    )

    handle = next(
        handle
        for handle in figure.legends[0].legend_handles
        if "find_bads_ecg" in handle.get_label()
    )
    ring_colour = handle.get_color()
    assert matplotlib.colors.to_hex(ring_colour) == matplotlib.colors.to_hex(MARK_COLOR)
    assert ring_colour not in RUN_COLORS


def test_the_run_key_is_given_once_for_the_whole_figure() -> None:
    """Both data panels are keyed by the same run colours; two legends sat on the data."""
    figure = _plot_component_cardiac_review(
        _component_review(),
        ica=_Ica(),
        component=0,
        status="RETAINED",
        status_description="",
    )

    assert len(figure.legends) == 1
    labels = {text.get_text() for text in figure.legends[0].get_texts()}
    assert {"run-1", "run-2", "Median", "ECG median"} <= labels
    waveform_axis = next(a for a in figure.axes if "R-locked ICA waveform" in a.get_title())
    assert waveform_axis.get_legend() is None


def _score_axes(figure):
    return [axis for axis in figure.axes if "score" in axis.get_ylabel().lower()]


def test_correlation_and_ctps_get_an_axis_each() -> None:
    """They are different quantities and cannot share a scale.

    ``find_bads_ecg`` returns a signed Pearson correlation against the ECG channel and a
    CTPS kappa, which is non-negative by construction. Drawn against one axis, a
    correlation of -0.13 and a kappa of 0.15 look like mirror images of one effect; they
    are two unrelated numbers that happen to be similar in magnitude.
    """
    figure = _plot_component_cardiac_review(
        _component_review(),
        ica=_Ica(),
        component=0,
        status="RETAINED",
        status_description="",
    )

    titles = [axis.get_title() for axis in figure.axes]
    assert "ECG correlation (r)" in titles
    assert "CTPS (kappa)" in titles


def test_each_score_axis_scales_to_its_own_quantity() -> None:
    """A large kappa must not stretch the correlation axis, or the reverse."""
    review = _component_review()
    review = ComponentCardiacReview(
        run_ids=review.run_ids,
        times=review.times,
        run_mean_z=review.run_mean_z,
        correlation_scores=np.array([[0.01, 0.0], [0.02, 0.0]]),
        ctps_scores=np.array([[0.90, 0.0], [0.95, 0.0]]),
        correlation_flags=np.zeros((2, 2), dtype=bool),
        ctps_flags=np.zeros((2, 2), dtype=bool),
        r_locked_epoch_counts=review.r_locked_epoch_counts,
        run_ecg_z=review.run_ecg_z,
    )

    figure = _plot_component_cardiac_review(
        review, ica=_Ica(), component=0, status="RETAINED", status_description=""
    )

    correlation_axis = next(a for a in figure.axes if a.get_title() == "ECG correlation (r)")
    ctps_axis = next(a for a in figure.axes if a.get_title() == "CTPS (kappa)")
    assert correlation_axis.get_ylim()[1] < 0.2
    assert ctps_axis.get_ylim()[1] > 0.9


def test_the_ctps_axis_never_descends_below_zero() -> None:
    """CTPS kappa cannot be negative, so an axis that shows negative space misleads."""
    figure = _plot_component_cardiac_review(
        _component_review(),
        ica=_Ica(),
        component=0,
        status="RETAINED",
        status_description="",
    )

    ctps_axis = next(a for a in figure.axes if a.get_title() == "CTPS (kappa)")
    assert ctps_axis.get_ylim()[0] >= 0.0


def test_the_screening_panel_covers_every_component_and_both_detectors() -> None:
    """The per-component slides show one component at a time, which cannot say which
    components stand out against the rest of the decomposition. That is the only way to
    reach a cardiac component the classifier labelled brain and kept, because the slides
    require you to already suspect it.

    Both detectors are drawn because they disagree: CTPS responds to phase locking with
    the beat and correlation to waveform similarity.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    from eeg_pipeline.preprocessing.ica_cardiac_report import _plot_component_cardiac_scores
    from eeg_pipeline.preprocessing.ica_cardiac_review import ComponentCardiacReview

    runs, components = 3, 6
    rng = np.random.default_rng(0)
    ctps = rng.uniform(0.0, 0.2, (runs, components))
    ctps[:, 4] = 0.8
    review = ComponentCardiacReview(
        run_ids=tuple(f"run-{index}" for index in range(runs)),
        times=np.linspace(-0.2, 0.6, 10),
        run_mean_z=rng.normal(0, 1, (runs, components, 10)),
        correlation_scores=rng.uniform(0.0, 0.3, (runs, components)),
        ctps_scores=ctps,
        correlation_flags=np.zeros((runs, components), dtype=bool),
        ctps_flags=np.tile(np.arange(components) == 4, (runs, 1)),
        r_locked_epoch_counts=np.full(runs, 100),
        run_ecg_z=rng.normal(0, 1, (runs, 10)),
    )

    figure = _plot_component_cardiac_scores(review, excluded=[4])

    axis = figure.axes[0]
    labels = " ".join(text.get_text() for text in axis.get_legend().get_texts())
    assert "CTPS" in labels
    assert "correlation" in labels.lower()
    # Every component is on the axis, not only the flagged one.
    plotted = max(
        collection.get_offsets().shape[0]
        for collection in axis.collections
        if collection.get_offsets().shape[0] > 1
    )
    assert plotted == components
    plt.close(figure)
