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


def test_overlapping_bad_annotations_count_their_union_once() -> None:
    raw = _raw()
    raw.set_annotations(
        mne.Annotations(
            onset=[20.0, 25.0],
            duration=[10.0, 10.0],
            description=["BAD_movement", "BAD_gradient"],
        )
    )

    run = compute_run_continuity(raw, recording_id="run-1")

    assert run.bad_fraction == pytest.approx(15.0 / DURATION)


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


def test_continuity_keeps_mne_raw_waveforms_for_visual_qc() -> None:
    """A summary heatmap complements, but cannot replace, the sensor waveform."""
    raw = _raw(n_channels=2)
    run = compute_run_continuity(raw, recording_id="run-1")
    report = mne.Report(title="continuity", verbose="ERROR")
    report.add_raw(raw, title="Raw (original)", psd=False, butterfly=True)

    add_continuity_section(report=report, runs=[run])

    assert any(
        element.section == "Raw (original)" and element.name == "Time series"
        for element in report._content
    )


def test_an_empty_section_is_an_error_rather_than_a_blank_panel() -> None:
    report = mne.Report(title="continuity", verbose="ERROR")

    with pytest.raises(ValueError, match="at least one run"):
        add_continuity_section(report=report, runs=[])


def test_the_onset_transient_does_not_become_the_reported_excursion() -> None:
    """All six runs of one subject reported "+25 dB at 0.0 min", which is the high-pass
    boundary response rather than anything that happened to the participant. It also
    hid every real excursion, because nothing later in the run could beat it."""
    rng = np.random.default_rng(0)
    data = rng.normal(0, 1e-5, (8, int(SFREQ * DURATION)))
    # A filter-edge response at the run start, and a smaller real disturbance later.
    data[:, : int(SFREQ * 2)] *= 40.0
    data[:, int(SFREQ * 60) : int(SFREQ * 62)] *= 6.0
    info = mne.create_info([f"C{index}" for index in range(8)], SFREQ, "eeg")
    raw = mne.io.RawArray(data, info, verbose="ERROR")
    run = compute_run_continuity(
        raw,
        recording_id="run-1",
        edge_support_seconds=5.0,
    )

    assert run.worst_window_s > 30.0
    # The transient is still drawn: it is excluded from the statistic, not from the data.
    assert float(np.max(run.excursion_db)) > run.worst_excursion_db
    assert run.edge_support_s > 0.0


def test_the_terminal_edge_response_is_also_excluded() -> None:
    rng = np.random.default_rng(0)
    data = rng.normal(0, 1e-5, (8, int(SFREQ * DURATION)))
    data[:, int(SFREQ * 60) : int(SFREQ * 62)] *= 6.0
    data[:, -int(SFREQ * 2) :] *= 40.0
    info = mne.create_info([f"C{index}" for index in range(8)], SFREQ, "eeg")
    raw = mne.io.RawArray(data, info, verbose="ERROR")

    run = compute_run_continuity(
        raw,
        recording_id="run-1",
        edge_support_seconds=5.0,
    )

    assert run.worst_window_s < DURATION - 5.0
    assert not run.full_support_mask[-1]
    assert float(np.max(run.excursion_db)) > run.worst_excursion_db


def test_the_excursion_axis_is_not_scaled_by_the_transient_it_excludes() -> None:
    """Excluding the transient from the statistic while letting it set the axis leaves the
    figure disagreeing with its own caption: the caption reports the worst full-support
    moment, and the panel devotes four fifths of its height to the edge span."""
    from eeg_pipeline.preprocessing.report.continuity import plot_run_continuity

    rng = np.random.default_rng(0)
    data = rng.normal(0, 1e-5, (8, int(SFREQ * DURATION)))
    data[:, : int(SFREQ * 2)] *= 40.0
    data[:, int(SFREQ * 60) : int(SFREQ * 62)] *= 6.0
    info = mne.create_info([f"C{index}" for index in range(8)], SFREQ, "eeg")
    raw = mne.io.RawArray(data, info, verbose="ERROR")
    run = compute_run_continuity(
        raw,
        recording_id="run-1",
        edge_support_seconds=5.0,
    )

    figure = plot_run_continuity(run)
    trace_axis = figure.axes[1]
    low, high = trace_axis.get_ylim()

    # The full-support range fits, with the edge response left to run off the top.
    assert high < float(np.max(run.excursion_db))
    assert high >= run.worst_excursion_db
    assert low <= float(np.min(run.excursion_db[run.full_support_mask]))
    # Zero is the reference every value on this axis is measured against.
    assert low <= 0.0 <= high


def test_task_events_are_carried_so_an_excursion_can_be_read_against_trials() -> None:
    """ "A +12 dB excursion at 5.5 min" is a fact about the recording; "it covers four
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


def test_the_filter_edge_exclusion_is_supplied_by_the_realised_filter() -> None:
    """A cutoff alone does not define a filter's impulse-response support."""
    info = mne.create_info(["C0", "C1"], SFREQ, "eeg")
    raw = mne.io.RawArray(
        np.random.default_rng(0).normal(0, 1e-5, (2, int(SFREQ * DURATION))),
        info,
        verbose="ERROR",
    )
    with raw.info._unlock():
        raw.info["highpass"] = 0.1
    run = compute_run_continuity(
        raw,
        recording_id="run-1",
        edge_support_seconds=16.5,
    )

    assert run.edge_support_s == 16.5


def test_no_filter_description_means_no_edge_exclusion() -> None:
    """The report must not infer an impulse response from a scalar cutoff."""
    info = mne.create_info(["C0", "C1"], SFREQ, "eeg")
    raw = mne.io.RawArray(
        np.random.default_rng(0).normal(0, 1e-5, (2, int(SFREQ * DURATION))),
        info,
        verbose="ERROR",
    )
    with raw.info._unlock():
        raw.info["highpass"] = 0.1

    run = compute_run_continuity(raw, recording_id="run-1")

    assert run.edge_support_s == 0.0
    assert run.worst_excursion_db == float(np.max(run.excursion_db))


@pytest.mark.parametrize("edge_support_seconds", [-1.0, float("nan")])
def test_invalid_filter_edge_exclusions_are_rejected(edge_support_seconds) -> None:
    with pytest.raises(ValueError, match="edge-support span"):
        compute_run_continuity(
            _raw(),
            recording_id="run-1",
            edge_support_seconds=edge_support_seconds,
        )


def test_filter_support_cannot_consume_both_ends_of_the_run() -> None:
    with pytest.raises(ValueError, match="leaves no continuity windows"):
        compute_run_continuity(
            _raw(),
            recording_id="run-1",
            edge_support_seconds=DURATION / 2.0,
        )


def test_the_section_sits_with_the_other_raw_input_evidence() -> None:
    """This panel is measured from the raw run and says nothing about the ICA.

    It once anchored itself before the ICA component review, which dropped it between the
    ocular review and the decomposition summary and split the ICA sections into two blocks
    with a raw-data panel wedged in the middle. Its position is now stated in the
    document's section order rather than negotiated by the section itself.
    """
    from eeg_pipeline.preprocessing.report.organize import order_sections

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
    order_sections(report)

    sections = [element.section for element in report._content]
    assert sections.index("Data quality over time") < sections.index("Raw (original)")
    assert sections.index("Raw (original)") < sections.index("ICA decomposition quality")


def test_a_beat_label_is_a_search_instruction_not_an_assumption() -> None:
    """Naming the beat label excludes that train from the rug; naming none searches for none.

    The rug exists to show the trials a bad stretch covers. A beat train is roughly one
    marker per second, so drawing it as events buries exactly what the panel is read for.
    """
    beats = np.arange(0.5, DURATION, 0.9)
    raw = _raw()
    raw.set_annotations(
        mne.Annotations(
            onset=[*beats, 20.0],
            duration=[0.0] * (len(beats) + 1),
            description=["Cardiac/R"] * len(beats) + ["stimulus/heat"],
        )
    )

    named = compute_run_continuity(raw, recording_id="run-1", pulse_description="Cardiac/R")
    unnamed = compute_run_continuity(raw, recording_id="run-1")

    assert named.event_onsets == (20.0,)
    # Unnamed, the beats are indistinguishable from trials and the rug is unreadable.
    assert len(unnamed.event_onsets) == len(beats) + 1


def test_acquisition_markers_are_not_drawn_as_task_events() -> None:
    """Bookkeeping a site writes is excluded by prefix, which is the configurable route."""
    raw = _raw()
    raw.set_annotations(
        mne.Annotations(
            onset=[5.0, 10.0, 20.0],
            duration=[0.0, 0.0, 0.0],
            description=["Acq/Trigger", "Cardiac/R", "stimulus/heat"],
        )
    )

    run = compute_run_continuity(
        raw,
        recording_id="run-1",
        pulse_description="Cardiac/R",
        non_event_prefixes=("BAD", "EDGE", "NEW SEGMENT", "Acq/"),
    )

    assert run.event_onsets == (20.0,)


def test_exact_event_descriptions_define_the_trial_rug() -> None:
    """A configured condition list is stronger evidence than guessing by exclusion."""
    raw = _raw()
    raw.set_annotations(
        mne.Annotations(
            onset=[5.0, 10.0, 15.0, 20.0],
            duration=[0.0] * 4,
            description=["stimulus/heat", "Response/R", "Volume/V  1", "Cardiac/R"],
        )
    )

    run = compute_run_continuity(
        raw,
        recording_id="run-1",
        event_descriptions=("stimulus/heat",),
    )

    assert run.event_onsets == (5.0,)


def _clean_epochs(n_epochs=8, n_channels=6):
    import mne
    import numpy as np

    rng = np.random.default_rng(0)
    info = mne.create_info([f"C{index}" for index in range(n_channels)], 100.0, "eeg")
    info.set_montage(
        mne.channels.make_dig_montage(
            ch_pos={
                name: pos
                for name, pos in zip(
                    info["ch_names"], rng.normal(0, 0.05, (n_channels, 3)), strict=True
                )
            },
            coord_frame="head",
        )
    )
    data = rng.normal(0, 1e-5, (n_epochs, n_channels, 100))
    # One epoch loud across the montage, one channel loud throughout: the two failures
    # this panel exists to separate from a retention count.
    data[3] *= 6.0
    data[:, 1, :] *= 5.0
    return mne.EpochsArray(data, info, verbose="ERROR")


def test_the_epoch_panel_scores_each_channel_against_its_own_median() -> None:
    """A constitutionally noisy sensor must not paint its whole row, while a sensor that
    failed for a few trials must show those trials."""
    import matplotlib.pyplot as plt
    import numpy as np

    from eeg_pipeline.preprocessing.report.continuity import plot_epoch_channel_amplitude

    figure = plot_epoch_channel_amplitude(_clean_epochs())

    mesh = [c for c in figure.axes[0].collections if hasattr(c, "get_array")][0]
    # The mesh is drawn transposed: rows are channels, columns are epochs.
    values = np.asarray(mesh.get_array()).reshape(6, 8)
    # Channel 1 is loud in every epoch, so against its own median it is unremarkable.
    assert abs(float(np.median(values[1, :]))) < 1.0
    # Epoch 3 is loud across the montage and has to stand out.
    assert float(np.median(values[:, 3])) > 3.0
    plt.close(figure)


def test_the_epoch_panel_runs_anterior_to_posterior_like_the_run_panel() -> None:
    """pcolormesh draws row 0 at the bottom, so without inverting, the axis runs the
    opposite way to its own label and to the run-level panel beside it."""
    import matplotlib.pyplot as plt

    from eeg_pipeline.preprocessing.report.continuity import plot_epoch_channel_amplitude

    figure = plot_epoch_channel_amplitude(_clean_epochs())

    bottom, top = figure.axes[0].get_ylim()
    assert bottom > top, "the channel axis must be inverted so anterior sits at the top"
    plt.close(figure)


def test_the_epoch_panel_refuses_a_recording_with_no_eeg() -> None:
    import mne
    import numpy as np
    import pytest as _pytest

    from eeg_pipeline.preprocessing.report.continuity import plot_epoch_channel_amplitude

    info = mne.create_info(["M1", "M2"], 100.0, "misc")
    epochs = mne.EpochsArray(np.zeros((3, 2, 50)), info, verbose="ERROR")

    with _pytest.raises(ValueError, match="at least one EEG channel"):
        plot_epoch_channel_amplitude(epochs)
