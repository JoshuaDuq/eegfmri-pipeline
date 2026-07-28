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


def _montaged_raw(channel_names, *, n_samples=None):
    """A run whose channels carry real scalp positions, in deliberately scrambled order."""
    rng = np.random.default_rng(0)
    n_samples = n_samples or int(DURATION * SFREQ)
    info = mne.create_info(list(channel_names), SFREQ, "eeg")
    raw = mne.io.RawArray(
        rng.normal(0, 1e-5, (len(channel_names), n_samples)), info, verbose="ERROR"
    )
    raw.set_montage("standard_1020", verbose="ERROR")
    return raw


def test_channels_are_ordered_down_the_head_rather_than_by_file_order() -> None:
    """A spatially coherent excursion must read as a band, not as scattered rows.

    Acquisition order on a 63-channel cap spirals around the head, so consecutive rows of
    the map are not neighbours on the scalp. A blink, a movement, or a lead-loop artifact
    is spatially smooth, and drawn in that order it breaks into stripes separated by rows
    from the other side of the head — which looks like several unrelated channels
    misbehaving rather than one region moving.
    """
    # Front-to-back on the scalp: Fpz, Fz, Cz, Pz, Oz. Passed in scrambled order.
    raw = _montaged_raw(["Pz", "Fpz", "Oz", "Cz", "Fz"])

    run = compute_run_continuity(raw, recording_id="run-1")

    assert list(run.channel_names) == ["Fpz", "Fz", "Cz", "Pz", "Oz"]


def test_a_run_without_positions_keeps_the_order_it_arrived_in() -> None:
    """Without a montage there is no scalp order to sort by, and inventing one misleads."""
    raw = _raw(n_channels=4)

    run = compute_run_continuity(raw, recording_id="run-1")

    assert list(run.channel_names) == ["C0", "C1", "C2", "C3"]


def test_the_channel_axis_says_how_it_is_ordered() -> None:
    """A silent reorder is worse than none: it misleads whoever knows the file order.

    Only a third of the channels get a tick label at 63 channels, so the rows between them
    are read by position. That reading depends entirely on what the axis is sorted by, and
    the axis is the only place that can say.
    """
    raw = _montaged_raw(["Pz", "Fpz", "Oz", "Cz", "Fz"])
    run = compute_run_continuity(raw, recording_id="run-1")

    ylabel = plot_run_continuity(run).axes[0].get_ylabel()

    assert "anterior" in ylabel.lower() or "front" in ylabel.lower()


def test_the_most_anterior_channel_is_drawn_at_the_top() -> None:
    """The axis must run the way its label says it runs.

    ``pcolormesh`` puts row 0 at the bottom, so sorting anterior-first and drawing it
    unchanged renders the head upside down relative to the "anterior → posterior" label
    and relative to every topography in the report.
    """
    raw = _montaged_raw(["Pz", "Fpz", "Oz", "Cz", "Fz"])
    run = compute_run_continuity(raw, recording_id="run-1")

    axis = plot_run_continuity(run).axes[0]

    assert run.channel_names[0] == "Fpz"
    bottom, top = axis.get_ylim()
    assert bottom > top, "row 0 is the most anterior channel and must render at the top"


def test_reordering_moves_each_channel_row_with_its_name() -> None:
    """The rows and the tick labels must not come apart."""
    raw = _montaged_raw(["Pz", "Fpz", "Oz", "Cz", "Fz"])
    # Make one channel unmistakable: a loud second half on Fpz only.
    data = raw.get_data()
    data[raw.ch_names.index("Fpz"), int(60.0 * SFREQ) :] *= 8.0
    raw._data = data

    run = compute_run_continuity(raw, recording_id="run-1")

    loud = run.relative_db[:, run.times_s > 60.0].max(axis=1)
    assert run.channel_names[int(np.argmax(loud))] == "Fpz"


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


def test_the_onset_transient_does_not_become_the_reported_excursion() -> None:
    """All six runs of one subject reported "+25 dB at 0.0 min", which is the high-pass
    filter settling rather than anything that happened to the participant. It also hid
    every real excursion, because nothing later in the run could beat it."""
    rng = np.random.default_rng(0)
    data = rng.normal(0, 1e-5, (8, int(SFREQ * DURATION)))
    # A settling transient at the run start, and a smaller real disturbance later.
    data[:, : int(SFREQ * 2)] *= 40.0
    data[:, int(SFREQ * 60) : int(SFREQ * 62)] *= 6.0
    info = mne.create_info([f"C{index}" for index in range(8)], SFREQ, "eeg")
    raw = mne.io.RawArray(data, info, verbose="ERROR")
    with raw.info._unlock():
        raw.info["highpass"] = 0.1

    run = compute_run_continuity(raw, recording_id="run-1")

    assert run.worst_window_s > 30.0
    # The transient is still drawn: it is excluded from the statistic, not from the data.
    assert float(np.max(run.excursion_db)) > run.worst_excursion_db
    assert run.settling_s > 0.0


def test_the_excursion_axis_is_not_scaled_by_the_transient_it_excludes() -> None:
    """Excluding the transient from the statistic while letting it set the axis leaves the
    figure disagreeing with its own caption: the caption reports the worst settled moment,
    and the panel devotes four fifths of its height to the span that moment is not in."""
    from eeg_pipeline.preprocessing.report.continuity import plot_run_continuity

    rng = np.random.default_rng(0)
    data = rng.normal(0, 1e-5, (8, int(SFREQ * DURATION)))
    data[:, : int(SFREQ * 2)] *= 40.0
    data[:, int(SFREQ * 60) : int(SFREQ * 62)] *= 6.0
    info = mne.create_info([f"C{index}" for index in range(8)], SFREQ, "eeg")
    raw = mne.io.RawArray(data, info, verbose="ERROR")
    with raw.info._unlock():
        raw.info["highpass"] = 0.1
    run = compute_run_continuity(raw, recording_id="run-1")

    figure = plot_run_continuity(run)
    trace_axis = figure.axes[1]
    low, high = trace_axis.get_ylim()

    # The settled range fits, with the transient left to run off the top of the axis.
    assert high < float(np.max(run.excursion_db))
    assert high >= run.worst_excursion_db
    assert low <= float(np.min(run.excursion_db[run.settled_mask]))
    # Zero is the reference every value on this axis is measured against.
    assert low <= 0.0 <= high


def test_task_events_are_carried_so_an_excursion_can_be_read_against_trials() -> None:
    """"A +12 dB excursion at 5.5 min" is a fact about the recording; "it covers four
    trials" is the one that decides what to do about it. Scanner and artifact marks are
    not events and must not pad the rug."""
    raw = _raw()
    raw.set_annotations(
        mne.Annotations(
            onset=[5.0, 25.0, 45.0],
            duration=[0.0, 0.0, 0.0],
            description=["Trig_therm/T  1", "Trig_therm/T  1", "Trig_therm/T  1"],
        )
        + mne.Annotations(onset=[10.0], duration=[2.0], description=["BAD_break"])
    )

    run = compute_run_continuity(raw, recording_id="run-1")

    assert run.event_onsets == (5.0, 25.0, 45.0)


def test_a_resting_state_run_has_no_event_rug() -> None:
    """No events to draw, so the rug is absent rather than an empty strip of axis."""
    from eeg_pipeline.preprocessing.report.continuity import plot_run_continuity

    raw = _raw()
    raw.set_annotations(mne.Annotations(onset=[10.0], duration=[2.0], description=["BAD_break"]))

    run = compute_run_continuity(raw, recording_id="rest")
    figure = plot_run_continuity(run)

    assert run.event_onsets == ()
    assert figure.axes[1].get_legend() is None


def test_the_settling_window_follows_the_recording_highpass() -> None:
    """A slower high-pass takes longer to settle, so the excluded span is not a constant."""
    info = mne.create_info(["C0", "C1"], SFREQ, "eeg")
    raw = mne.io.RawArray(
        np.random.default_rng(0).normal(0, 1e-5, (2, int(SFREQ * DURATION))),
        info,
        verbose="ERROR",
    )
    with raw.info._unlock():
        raw.info["highpass"] = 0.1
    slow = compute_run_continuity(raw, recording_id="run-1")
    with raw.info._unlock():
        raw.info["highpass"] = 1.0
    fast = compute_run_continuity(raw, recording_id="run-1")

    assert slow.settling_s > fast.settling_s


def test_a_run_without_a_highpass_excludes_nothing() -> None:
    """Unfiltered data has no settling transient to attribute the onset to."""
    info = mne.create_info(["C0", "C1"], SFREQ, "eeg")
    raw = mne.io.RawArray(
        np.random.default_rng(0).normal(0, 1e-5, (2, int(SFREQ * DURATION))),
        info,
        verbose="ERROR",
    )
    with raw.info._unlock():
        raw.info["highpass"] = 0.0

    run = compute_run_continuity(raw, recording_id="run-1")

    assert run.settling_s == 0.0
    assert run.worst_excursion_db == float(np.max(run.excursion_db))


def test_the_section_sits_with_the_other_raw_input_evidence() -> None:
    """This panel is measured from the raw run and says nothing about the ICA.

    It anchored before the ICA component review, which dropped it between the ocular
    review and the decomposition summary and split the ICA sections into two blocks with
    a raw-data panel wedged in the middle.
    """
    run = compute_run_continuity(_raw(), recording_id="sub-01_task-x_run-1")
    report = mne.Report(title="continuity", verbose="ERROR")
    figure = plot_run_continuity(run)
    report.add_figure(fig=figure, title="Raw thing", section="Raw (original)", tags=("raw",))
    report.add_figure(
        fig=figure,
        title="An ICA thing",
        section="ICA decomposition quality",
        tags=("ica", "ica-component-review"),
    )

    add_continuity_section(report=report, runs=[run])

    sections = [element.section for element in report._content]
    assert sections.index("Data quality over time") < sections.index("Raw (original)")


def _run_continuity(*, volume_markers, gaps=()):
    """Build a RunContinuity directly, so the scanner-dependence can be varied."""
    from eeg_pipeline.preprocessing.report.continuity import RunContinuity

    return RunContinuity(
        recording_id="sub-01_task-x_run-1",
        window_seconds=1.0,
        times_s=np.arange(60.0),
        channel_names=("C0", "C1"),
        relative_db=np.zeros((2, 60)),
        bad_spans=(),
        volume_gaps=tuple(gaps),
        duration_s=60.0,
        has_volume_markers=volume_markers,
    )


def test_the_volume_marker_column_is_absent_without_a_scanner() -> None:
    """An EEG-only recording has no volume markers, so the column is structurally zero.

    A column that can only ever read 0 is not evidence; it invites the reader to wonder
    what would have made it non-zero, and answers a question about equipment they do not
    have.
    """
    from eeg_pipeline.preprocessing.report.continuity import continuity_html

    document = continuity_html([_run_continuity(volume_markers=False)])

    assert "Volume-marker gaps" not in document
    assert "scanner" not in document.lower()


def test_the_volume_marker_column_is_present_with_a_scanner() -> None:
    from eeg_pipeline.preprocessing.report.continuity import continuity_html

    document = continuity_html([_run_continuity(volume_markers=True)])

    assert "Volume-marker gaps" in document
    assert "scanner" in document.lower()


def test_a_scanner_run_with_no_gaps_still_gets_the_column() -> None:
    """Zero gaps is a measurement when markers exist; absent markers are not."""
    from eeg_pipeline.preprocessing.report.continuity import continuity_html

    document = continuity_html([_run_continuity(volume_markers=True, gaps=())])

    assert "Volume-marker gaps" in document


def test_volume_markers_are_recorded_when_a_description_is_configured() -> None:
    """The flag follows the configuration, not whether any gap happened to be found."""
    from eeg_pipeline.preprocessing.report.continuity import compute_run_continuity

    raw = _raw(volume_onsets=np.arange(0.0, DURATION, 0.9))

    with_markers = compute_run_continuity(
        raw, recording_id="run-1", volume_description="Volume/V  1"
    )
    without_markers = compute_run_continuity(raw, recording_id="run-1")

    assert with_markers.has_volume_markers
    assert not without_markers.has_volume_markers
