"""Beat detection evidence must separate variability from detector dropout."""

from __future__ import annotations

import matplotlib
import mne
import numpy as np
import pytest

matplotlib.use("Agg")

from eeg_pipeline.preprocessing.report.rr_intervals import (  # noqa: E402
    add_rr_interval_section,
    compute_rr_intervals,
    plot_rr_intervals,
    rr_intervals_html,
)

SFREQ = 250.0
DURATION = 180.0


def _raw_with_beats(beat_onsets):
    info = mne.create_info(["C0", "C1"], SFREQ, "eeg")
    raw = mne.io.RawArray(
        np.random.default_rng(0).normal(0, 1e-5, (2, int(DURATION * SFREQ))),
        info,
        verbose="ERROR",
    )
    if len(beat_onsets):
        raw.set_annotations(
            mne.Annotations(
                onset=beat_onsets,
                duration=0.0,
                description=[BEAT_MARKER] * len(beat_onsets),
            )
        )
    return raw


#: The label this file's fixtures write. Core defaults to none, so a caller that wants an
#: interval series must name it, the way a study config does.
BEAT_MARKER = "Pulse Artifact/R"


def _regular_beats(period=0.85, jitter=0.0, seed=0):
    rng = np.random.default_rng(seed)
    onsets = np.cumsum(rng.normal(period, jitter, 200))
    return onsets[onsets < DURATION]


def test_the_median_rate_is_recovered() -> None:
    series = compute_rr_intervals(
        _raw_with_beats(_regular_beats(period=0.75)), recording_id="run-1", description=BEAT_MARKER
    )

    assert series is not None
    assert series.median_bpm == pytest.approx(80.0, abs=1.0)


def test_variability_alone_is_not_counted_as_dropout() -> None:
    """A participant with a variable rate must not be reported as a failing detector."""
    series = compute_rr_intervals(
        _raw_with_beats(_regular_beats(jitter=0.08)), recording_id="run-1", description=BEAT_MARKER
    )

    assert series.dropout_count == 0


def test_a_missed_beat_is_counted() -> None:
    beats = _regular_beats()
    # Drop every twentieth beat, which doubles the interval that spans it.
    kept = np.array([beat for index, beat in enumerate(beats) if index % 20 != 0])

    series = compute_rr_intervals(_raw_with_beats(kept), recording_id="run-1", description=BEAT_MARKER)

    assert series.dropout_count >= 8


def test_a_detector_that_fails_partway_shows_in_the_interval_series() -> None:
    beats = _regular_beats()
    survived = np.concatenate([beats[beats < 90.0], beats[beats >= 90.0][::3]])

    series = compute_rr_intervals(_raw_with_beats(survived), recording_id="run-1", description=BEAT_MARKER)

    late = series.intervals_s[series.beat_times_s >= 90.0]
    early = series.intervals_s[series.beat_times_s < 90.0]
    assert float(np.median(late)) > 2.0 * float(np.median(early))
    assert series.dropout_count > 0


def test_a_run_without_markers_yields_no_series() -> None:
    assert compute_rr_intervals(_raw_with_beats([]), recording_id="run-1", description=BEAT_MARKER) is None


def test_too_few_markers_yield_no_series() -> None:
    assert compute_rr_intervals(_raw_with_beats([1.0, 2.0]), recording_id="run-1", description=BEAT_MARKER) is None


def test_the_section_renders_and_replaces_on_rebuild() -> None:
    series = compute_rr_intervals(_raw_with_beats(_regular_beats()), recording_id="run-1", description=BEAT_MARKER)
    report = mne.Report(title="analyzer", verbose="ERROR")

    add_rr_interval_section(report=report, series=[series])
    add_rr_interval_section(report=report, series=[series])

    # Counted by distinct title rather than by a total, so adding a panel to the section
    # does not require editing a magic number in a test about accumulation.
    titles = [element.name for element in report._content]
    assert len(titles) == len(set(titles))
    assert all("rr-intervals" in element.tags for element in report._content)
    assert "run-1" in rr_intervals_html([series])
    assert plot_rr_intervals([series]).axes


def test_an_empty_section_is_an_error_rather_than_a_blank_panel() -> None:
    report = mne.Report(title="analyzer", verbose="ERROR")

    with pytest.raises(ValueError, match="at least one run"):
        add_rr_interval_section(report=report, series=[])


def _series_for(count: int, *, period=0.85):
    return [
        compute_rr_intervals(
            _raw_with_beats(_regular_beats(period=period, seed=index)),
            recording_id=f"sub-01_task-x_run-{index + 1}", description=BEAT_MARKER,
        )
        for index in range(count)
    ]


def test_the_panels_share_one_interval_scale() -> None:
    """A run whose detector collapsed reaches intervals of two minutes.

    On independent axes that run gets the same panel height as the usable ones and every
    other panel is compressed into a flat line, so the figure spends its space on the run
    that is already known to be broken.
    """
    series = _series_for(3)
    collapsed = compute_rr_intervals(
        _raw_with_beats(np.array([1.0, 30.0, 95.0, 160.0])),
        recording_id="sub-01_task-x_run-4", description=BEAT_MARKER,
    )

    figure = plot_rr_intervals([*series, collapsed])

    assert len({axis.get_ylim() for axis in figure.axes}) == 1


def test_a_collapsed_run_does_not_flatten_the_readable_ones() -> None:
    """The shared scale is physiological, not driven by the worst run in the set.

    Letting a run with a failed detector set the limits spent the axis on the two orders
    of magnitude between a real interval and a two-minute gap, which left the ordinary
    beat-to-beat variation of every good run inside a band a few pixels tall. The window
    is a fixed plausible range instead, so what the panels resolve does not depend on
    which runs happen to share the figure.
    """
    from eeg_pipeline.preprocessing.report.rr_intervals import DRAWN_RR_RANGE_S

    series = _series_for(3)
    collapsed = compute_rr_intervals(
        _raw_with_beats(np.array([1.0, 30.0, 95.0, 160.0])),
        recording_id="sub-01_task-x_run-4", description=BEAT_MARKER,
    )

    with_collapsed = plot_rr_intervals([*series, collapsed])
    without_collapsed = plot_rr_intervals(series)

    assert with_collapsed.axes[0].get_ylim() == without_collapsed.axes[0].get_ylim()
    lower, upper = with_collapsed.axes[0].get_ylim()
    assert (lower, upper) == pytest.approx(DRAWN_RR_RANGE_S)


def test_a_long_interval_is_drawn_where_it_falls_rather_than_on_the_rail() -> None:
    """A linear 0.3-2 s window censored the magnitude of every interval above it.

    The count survived in the title, but the value did not: on sub-0012 run-1, 82 of 84
    long intervals were drawn stacked on the 2 s boundary, so a 2.1 s gap and a 60 s one
    were the same mark. That is the measurement the panel exists to show.

    A logarithmic axis over a wider window keeps ordinary beat-to-beat variation legible
    -- 0.3-2 s still occupies more than half the height -- while putting a collapsed
    detector's intervals at a readable position instead of against the rail.
    """
    from eeg_pipeline.preprocessing.report.rr_intervals import DRAWN_RR_RANGE_S

    # Intervals of 5 s: far above any plausible rhythm, well inside the drawn window.
    lapsed = compute_rr_intervals(
        _raw_with_beats(np.array([1.0, 6.0, 11.0, 16.0])),
        recording_id="sub-01_task-x_run-1", description=BEAT_MARKER,
    )

    figure = plot_rr_intervals([lapsed])
    axis = figure.axes[0]

    assert axis.get_yscale() == "log"
    assert DRAWN_RR_RANGE_S[1] > 5.0
    drawn = np.concatenate([np.asarray(line.get_ydata(), dtype=float) for line in axis.lines])
    finite = drawn[np.isfinite(drawn)]
    # The 5 s intervals are plotted at 5 s, not flattened onto the top of the window.
    assert np.any(np.isclose(finite, 5.0))
    assert "outside" not in axis.get_title().lower()


def test_the_physiological_band_stays_visible_on_the_widened_axis() -> None:
    """Widening the window must not cost the ordinary rhythm its resolution.

    The whole point of the fixed window was that beat-to-beat variation stays readable.
    On a log axis the plausible band still has to own most of the panel, or the change
    has traded one censoring for another.
    """
    from eeg_pipeline.preprocessing.report.rr_intervals import (
        DRAWN_RR_RANGE_S,
        PLAUSIBLE_RR_RANGE_S,
    )

    drawn_span = np.log10(DRAWN_RR_RANGE_S[1]) - np.log10(DRAWN_RR_RANGE_S[0])
    plausible_span = np.log10(PLAUSIBLE_RR_RANGE_S[1]) - np.log10(PLAUSIBLE_RR_RANGE_S[0])

    assert plausible_span / drawn_span > 0.5


def test_intervals_outside_the_window_are_counted_on_the_panel() -> None:
    """Clipping without saying so would turn a failed detector into a tidy panel."""
    collapsed = compute_rr_intervals(
        _raw_with_beats(np.array([1.0, 30.0, 95.0, 160.0])),
        recording_id="sub-01_task-x_run-4", description=BEAT_MARKER,
    )

    figure = plot_rr_intervals([collapsed])

    title = figure.axes[0].get_title()
    assert "3" in title
    assert "outside" in title.lower()


def test_a_run_entirely_inside_the_window_says_nothing_about_clipping() -> None:
    """A clean run must not carry a note about a failure mode it did not have."""
    figure = plot_rr_intervals(_series_for(1))

    assert "outside" not in figure.axes[0].get_title().lower()


def test_the_poincare_plot_separates_missed_from_double_detections() -> None:
    """RR_n against RR_n+1, where the two failure modes land in different places.

    A missed beat produces one interval near twice the median followed by a normal one,
    so it sits on the 2x reference line; a double detection produces the mirror pair on
    the 0.5x line. Neither is distinguishable from ordinary variability in the time
    series, where both are simply "a tall point".
    """
    from eeg_pipeline.preprocessing.report.rr_intervals import plot_rr_poincare

    figure = plot_rr_poincare(_series_for(2))

    axis = figure.axes[0]
    assert "RR" in axis.get_xlabel()
    assert "RR" in axis.get_ylabel()
    labels = " ".join(text.get_text() for text in axis.texts)
    legend = axis.get_legend()
    if legend is not None:
        labels += " ".join(text.get_text() for text in legend.get_texts())
    assert "2" in labels


def test_a_poincare_reference_line_never_wears_a_run_colour() -> None:
    """The 2x and 0.5x guides were drawn in the flag colour, which is also the colour
    RUN_COLORS hands to the second run. In a four-run session run-2's cloud and the "one
    beat missed" reference were the same vermillion, so the figure said a run was flagged
    when nothing had flagged it."""
    from matplotlib.colors import to_hex

    from eeg_pipeline.preprocessing.report.rr_intervals import plot_rr_poincare

    axis = plot_rr_poincare(_series_for(4)).axes[0]

    run_colours = {to_hex(collection.get_facecolor()[0]) for collection in axis.collections}
    guide_colours = {to_hex(line.get_color()) for line in axis.lines}
    assert run_colours.isdisjoint(guide_colours)


def test_the_two_poincare_failure_modes_are_told_apart_without_the_legend_text() -> None:
    """Both dotted guides shared one colour and one dash pattern, so the only thing
    separating "one beat missed" from "one beat counted twice" was reading which of two
    identical entries sat higher in the legend."""
    from eeg_pipeline.preprocessing.report.rr_intervals import plot_rr_poincare

    axis = plot_rr_poincare(_series_for(2)).axes[0]

    styles = {
        (line.get_linestyle(), line.get_color())
        for line in axis.lines
        if "beat" in str(line.get_label())
    }
    assert len(styles) == 2


def test_the_poincare_plot_needs_at_least_one_run() -> None:
    from eeg_pipeline.preprocessing.report.rr_intervals import plot_rr_poincare

    with pytest.raises(ValueError):
        plot_rr_poincare([])


def test_a_run_with_no_usable_markers_is_named_rather_than_omitted() -> None:
    """Runs 2 and 4 simply vanished from this figure, and the reader had to cross-read the
    Analyzer table to discover that a panel was missing rather than merely empty."""
    figure = plot_rr_intervals(
        _series_for(1),
        missing=("sub-01_task-x_run-2", "sub-01_task-x_run-4"),
    )

    text = " ".join(
        [figure.get_suptitle(), *(axis.get_title() for axis in figure.axes)]
        + [artist.get_text() for axis in figure.axes for artist in axis.texts]
    )
    assert "run-2" in text and "run-4" in text


def test_the_panels_name_the_run_without_the_full_recording_id() -> None:
    figure = plot_rr_intervals(_series_for(2))

    assert "run-1" in figure.axes[0].get_title()
    assert "sub-01_task-x" not in figure.axes[0].get_title()


def test_the_caption_fits_inside_the_figure() -> None:
    """The suptitle was one long line wider than the axes, so it was clipped mid-word."""
    figure = plot_rr_intervals(_series_for(3))
    figure.canvas.draw()

    title = figure._suptitle.get_window_extent()
    assert title.x0 >= 0.0
    assert title.x1 <= figure.get_window_extent().x1


def test_rr_section_lives_outside_the_analyzer_module():
    from eeg_pipeline.preprocessing.report import rr_intervals

    assert hasattr(rr_intervals, "add_rr_interval_section")
