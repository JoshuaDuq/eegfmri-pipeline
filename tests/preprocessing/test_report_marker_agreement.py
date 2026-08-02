"""Reconciling the two beat detectors that drive the cardiac evidence.

Analyzer's pulse correction is driven by its own R markers; this pipeline's cardiac review
detects R peaks from the ECG signal. When they disagree the report currently shows both
without ever putting them on the same axis, and the two sections sit far enough apart that
the disagreement is easy to miss. On sub-0015 the Analyzer marker train collapsed in runs
2, 4 and 6 — fewer than three markers where the signal detector found ~500 beats — which
means the pulse-artifact correction for half the session was driven by almost nothing.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from eeg_pipeline.preprocessing.report.analyzer_qc import (  # noqa: E402
    add_marker_agreement_section,
    compute_marker_agreement,
    compute_run_marker_agreement,
    marker_agreement_html,
    plot_marker_agreement,
)


def _agreement(marker_onsets, detected_onsets, **overrides):
    return compute_marker_agreement(
        recording_id="run-1",
        marker_onsets_s=np.asarray(marker_onsets, dtype=float),
        detected_onsets_s=np.asarray(detected_onsets, dtype=float),
        **overrides,
    )


def test_identical_trains_agree_completely() -> None:
    beats = np.arange(0.0, 60.0, 0.85)

    agreement = _agreement(beats, beats)

    assert agreement.n_markers == len(beats)
    assert agreement.n_detected == len(beats)
    assert agreement.matched_fraction == pytest.approx(1.0)


def test_a_collapsed_marker_train_beside_a_healthy_detection_is_reported_not_raised() -> None:
    """This is the case the panel exists for, so it must produce a number, not an error.

    Analyzer wrote fewer than three markers for sub-0015 runs 2, 4 and 6 while the signal
    detector found roughly 500 beats in each. A function that raised here would remove the
    evidence exactly where it matters most.
    """
    detected = np.arange(0.0, 60.0, 0.85)

    agreement = _agreement([1.2], detected)

    assert agreement.n_markers == 1
    assert agreement.n_detected == len(detected)
    assert agreement.matched_fraction < 0.05


def test_no_markers_at_all_is_zero_agreement_rather_than_undefined() -> None:
    agreement = _agreement([], np.arange(0.0, 60.0, 0.85))

    assert agreement.n_markers == 0
    assert agreement.matched_fraction == 0.0


def test_no_beats_detected_leaves_the_fraction_undefined() -> None:
    """The fraction is of detected beats, so with none there is nothing to take it over."""
    agreement = _agreement(np.arange(0.0, 60.0, 0.85), [])

    assert agreement.n_detected == 0
    assert agreement.matched_fraction is None


def test_a_marker_within_tolerance_counts_as_the_same_beat() -> None:
    """Analyzer marks the R peak it modelled, which need not be the sample MNE picks."""
    detected = np.array([10.0, 20.0, 30.0])

    close = _agreement(detected + 0.04, detected, tolerance_s=0.1)
    far = _agreement(detected + 0.40, detected, tolerance_s=0.1)

    assert close.matched_fraction == pytest.approx(1.0)
    assert far.matched_fraction == pytest.approx(0.0)


def test_one_marker_cannot_account_for_several_detected_beats() -> None:
    """Without this, a single marker beside a burst of beats reads as full agreement."""
    agreement = _agreement([10.0], [9.98, 10.0, 10.02], tolerance_s=0.1)

    assert agreement.n_matched == 1
    assert agreement.matched_fraction == pytest.approx(1.0 / 3.0)


def test_sensitivity_and_precision_keep_their_distinct_denominators() -> None:
    agreement = _agreement([10.0, 20.0, 20.02], [10.0, 20.0], tolerance_s=0.05)

    assert agreement.matched_fraction == 1.0
    assert agreement.marker_precision == pytest.approx(2.0 / 3.0)

    document = marker_agreement_html([agreement])
    assert "Beat sensitivity" in document
    assert "Marker precision" in document


def _healthy_and_collapsed():
    """One run where both detectors agree, and one where the marker train collapsed."""
    beats = np.arange(0.0, 60.0, 0.85)
    return [
        compute_marker_agreement(
            recording_id="sub-01_run-1",
            marker_onsets_s=beats,
            detected_onsets_s=beats,
        ),
        compute_marker_agreement(
            recording_id="sub-01_run-2",
            marker_onsets_s=np.array([1.2]),
            detected_onsets_s=beats,
        ),
    ]


def test_a_constant_lag_is_measured_rather_than_reported_as_disagreement() -> None:
    """Two trains offset by a fixed delay describe the same heartbeat.

    On sub-0012 run-5, Analyzer wrote 545 markers, the ECG gave 571 beats, and *none*
    matched: every marker sat 303 ms ahead of its beat, with an interquartile range of
    19 ms. The panel reported "0.0% of beats marked", which reads as "the marker train
    does not describe this heartbeat" — the opposite of what the data says.

    The matched fraction is correct and stays as it is. What was missing is the
    measurement that makes it readable: a tight lag distribution says offset, a broad one
    says disagreement, and the share alone cannot tell them apart.
    """
    beats = np.arange(1.0, 60.0, 0.8)
    lagged = compute_marker_agreement(
        recording_id="run-1",
        marker_onsets_s=beats - 0.303,
        detected_onsets_s=beats,
    )

    assert lagged.n_matched == 0
    assert lagged.median_lag_s == pytest.approx(0.303, abs=1e-6)
    # Tight: every beat carries the same offset, so the spread is what separates this
    # from two detectors that genuinely disagree.
    assert lagged.lag_iqr_s == pytest.approx(0.0, abs=1e-6)


def test_genuine_disagreement_has_no_tight_lag_to_report() -> None:
    """Markers scattered against the beats must not read as a clean offset."""
    rng = np.random.default_rng(0)
    beats = np.arange(1.0, 60.0, 0.8)
    scattered = compute_marker_agreement(
        recording_id="run-1",
        marker_onsets_s=np.sort(rng.uniform(1.0, 60.0, beats.size)),
        detected_onsets_s=beats,
    )

    assert scattered.lag_iqr_s > 0.1


def test_a_run_with_nothing_to_compare_reports_no_lag() -> None:
    """One empty train leaves no pairs, which is absence of evidence, not a lag of zero."""
    empty = compute_marker_agreement(
        recording_id="run-1",
        marker_onsets_s=np.array([]),
        detected_onsets_s=np.arange(1.0, 10.0, 0.8),
    )

    assert empty.median_lag_s is None
    assert empty.lag_iqr_s is None


def test_the_table_puts_both_detectors_counts_side_by_side() -> None:
    document = marker_agreement_html(_healthy_and_collapsed())

    assert "run-1" in document and "run-2" in document
    # The collapsed run's two counts are the comparison the panel exists to make.
    assert "71" in document and "1" in document


def test_the_table_reports_the_disagreement_without_grading_it() -> None:
    """Analyzer's own status is Analyzer's to give; this panel measures, it does not judge."""
    document = marker_agreement_html(_healthy_and_collapsed())

    for verdict in ("fail", "pass", "&#9888;", "unusable", "invalid"):
        assert verdict not in document.lower()


def test_an_undefined_fraction_is_not_drawn_as_zero_agreement() -> None:
    """No detected beats means nothing to compare, which is not total disagreement."""
    nothing_detected = compute_marker_agreement(
        recording_id="sub-01_run-3",
        marker_onsets_s=np.arange(0.0, 60.0, 0.85),
        detected_onsets_s=np.array([]),
    )

    document = marker_agreement_html([nothing_detected])

    # Sensitivity is undefined without detected beats; precision is zero because none of
    # the recorded Analyzer markers can be supported by a detected beat.
    assert document.count("0.0%") == 1
    assert "&mdash;" in document


def test_the_figure_draws_both_trains_for_every_run() -> None:
    """Seeing when the marker train stopped is the point; a total alone cannot show it."""
    figure = plot_marker_agreement(_healthy_and_collapsed())

    assert len(figure.axes) == 2
    for axis in figure.axes:
        labels = {line.get_label() for line in axis.get_lines()}
        assert any("Analyzer" in str(label) for label in labels)
        assert any("ECG" in str(label) for label in labels)
    # The panels share one time axis, so only the bottom one carries its label.
    assert figure.axes[-1].get_xlabel()


def test_beats_are_binned_into_a_rate_rather_than_drawn_as_one_tick_each() -> None:
    """Roughly 500 beats over an 8-minute run is more events than the axis has pixels.

    Drawn one tick per beat they alias, and the interference banding reads as structure in
    the marker train that is not in the data. Binning to a rate keeps the quantity the
    panel is actually about — were beats being marked at this moment — and cannot
    manufacture a pattern from the sampling of the axis.
    """
    beats = np.arange(0.0, 600.0, 0.85)
    agreement = compute_marker_agreement(
        recording_id="sub-01_run-1",
        marker_onsets_s=beats,
        detected_onsets_s=beats,
    )

    axis = plot_marker_agreement([agreement]).axes[0]

    for line in axis.get_lines():
        assert len(line.get_xdata()) < len(beats)
    assert "min" in axis.get_ylabel()


def test_a_train_that_stops_partway_falls_to_zero_at_that_moment() -> None:
    """A partial collapse and a train that never started are different faults."""
    beats = np.arange(0.0, 600.0, 0.85)
    agreement = compute_marker_agreement(
        recording_id="sub-01_run-1",
        marker_onsets_s=beats[beats < 200.0],
        detected_onsets_s=beats,
    )

    axis = plot_marker_agreement([agreement]).axes[0]
    marker_line = next(line for line in axis.get_lines() if "Analyzer" in str(line.get_label()))
    times = np.asarray(marker_line.get_xdata(), dtype=float)
    rate = np.asarray(marker_line.get_ydata(), dtype=float)

    assert rate[times < 150.0].mean() > 50.0
    assert rate[times > 300.0].max() == 0.0


def test_the_last_window_is_not_drawn_as_a_collapse_it_did_not_have() -> None:
    """A run rarely ends on a window boundary, so the final bin is short.

    Counting a partial window at the full window's rate turns two seconds of recording
    into a rate near zero, and the trace plunges to the floor at the right edge of every
    panel. That is an artefact of where the run stopped, and it sat in the same figure
    that exists to show when a marker train really did stop.
    """
    # 605 s of beats against 10 s windows: the last window holds 5 s of recording.
    beats = np.arange(0.0, 605.0, 0.85)
    agreement = compute_marker_agreement(
        recording_id="sub-01_run-1",
        marker_onsets_s=beats,
        detected_onsets_s=beats,
    )

    axis = plot_marker_agreement([agreement]).axes[0]
    marker_line = next(line for line in axis.get_lines() if "Analyzer" in str(line.get_label()))
    rate = np.asarray(marker_line.get_ydata(), dtype=float)

    # The train is uniform throughout, so no window may report a collapse.
    assert rate.min() > 50.0


def test_the_rate_axis_is_not_anchored_to_a_rate_no_heart_reaches() -> None:
    """Half the panel sat below 40 bpm, which no run in this study spends time at.

    The comparison the panel exists for is between two traces a few beats per minute
    apart, and anchoring at zero spent most of the height on rates neither trace visits.
    The axis still has to reach zero when a train actually stops, which is the one case
    where the floor carries information.
    """
    beats = np.arange(0.0, 600.0, 0.85)
    healthy = compute_marker_agreement(
        recording_id="sub-01_run-1",
        marker_onsets_s=beats,
        detected_onsets_s=beats,
    )

    axis = plot_marker_agreement([healthy]).axes[0]

    assert axis.get_ylim()[0] > 20.0


def test_a_train_that_stops_still_shows_its_floor() -> None:
    """Zero belongs on the axis exactly when a detector reached it."""
    beats = np.arange(0.0, 600.0, 0.85)
    collapsing = compute_marker_agreement(
        recording_id="sub-01_run-1",
        marker_onsets_s=beats[beats < 200.0],
        detected_onsets_s=beats,
    )

    axis = plot_marker_agreement([collapsing]).axes[0]

    assert axis.get_ylim()[0] == pytest.approx(0.0)


def test_the_figure_refuses_an_empty_set_rather_than_drawing_a_blank() -> None:
    with pytest.raises(ValueError, match="at least one run"):
        plot_marker_agreement([])


def test_the_section_is_added_when_a_marker_train_exists_to_reconcile() -> None:
    import mne

    report = mne.Report(title="subject", verbose="ERROR")

    add_marker_agreement_section(report=report, agreements=_healthy_and_collapsed())

    assert report._content
    assert all("marker-agreement" in element.tags for element in report._content)


def _raw_with_ecg(beat_onsets, *, marker_onsets, sfreq=250.0, duration=60.0):
    """A run carrying a real ECG channel whose R peaks sit at ``beat_onsets``."""
    import mne

    n_samples = int(duration * sfreq)
    times = np.arange(n_samples) / sfreq
    ecg = np.zeros(n_samples)
    for onset in beat_onsets:
        # A narrow positive deflection is enough for MNE's detector to lock onto.
        ecg += 1e-3 * np.exp(-0.5 * ((times - onset) / 0.012) ** 2)
    eeg = np.random.default_rng(0).normal(0, 1e-6, (2, n_samples))
    info = mne.create_info(["C0", "C1", "ECG"], sfreq, ["eeg", "eeg", "ecg"])
    raw = mne.io.RawArray(np.vstack([eeg, ecg[None, :]]), info, verbose="ERROR")
    if len(marker_onsets):
        raw.set_annotations(
            mne.Annotations(
                onset=np.asarray(marker_onsets, dtype=float),
                duration=0.0,
                description=["Pulse Artifact/R"] * len(marker_onsets),
            )
        )
    return raw


def test_a_run_is_reconciled_against_its_own_ecg_channel() -> None:
    beats = np.arange(2.0, 58.0, 0.85)
    raw = _raw_with_ecg(beats, marker_onsets=beats)

    agreement = compute_run_marker_agreement(raw, recording_id="sub-01_run-1")

    assert agreement is not None
    assert agreement.n_markers == len(beats)
    # The detector need not find every beat, but it must be reconciling against the ECG.
    assert agreement.n_detected > 0.8 * len(beats)


def test_a_run_without_an_ecg_channel_is_not_reconciled_against_a_synthesized_one() -> None:
    """MNE will build a surrogate ECG from EEG when no ECG exists.

    That surrogate is a different detector from the one whose beats appear elsewhere in
    the report, so comparing Analyzer's markers against it would answer a question nobody
    asked and put a number in the table that no other panel corroborates.
    """

    beats = np.arange(2.0, 58.0, 0.85)
    raw = _raw_with_ecg(beats, marker_onsets=beats)
    raw.drop_channels(["ECG"])

    assert compute_run_marker_agreement(raw, recording_id="sub-01_run-1") is None


def test_a_dataset_with_no_analyzer_markers_gets_no_reconciliation_section() -> None:
    """An EEG-only lab has no Analyzer stage, so there is no second detector to compare."""
    import mne

    beats = np.arange(0.0, 60.0, 0.85)
    report = mne.Report(title="subject", verbose="ERROR")

    add_marker_agreement_section(
        report=report,
        agreements=[
            compute_marker_agreement(
                recording_id="sub-01_run-1",
                marker_onsets_s=np.array([]),
                detected_onsets_s=beats,
            )
        ],
    )

    assert report._content == []
