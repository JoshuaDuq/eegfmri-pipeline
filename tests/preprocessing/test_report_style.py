"""The shared palette has to keep one colour meaning one thing across the report.

Every figure in the subject report draws through these constants, so a colour that
carries two meanings carries them everywhere at once. These tests pin the separations
that a reader depends on when moving between panels of the same figure.
"""

from __future__ import annotations

from eeg_pipeline.preprocessing.report.style import (
    AFTER_COLOR,
    BEFORE_COLOR,
    EXCLUDED_COLOR,
    FLAG_COLOR,
    GUIDE_COLOR,
    MARK_COLOR,
    RETAINED_COLOR,
    RUN_COLORS,
    separated_labels,
)


def test_a_processing_stage_is_never_the_flag_colour() -> None:
    """ "Before correction" and "flagged" appeared side by side in one analyzer figure.

    They shared vermillion, so the left panel's pre-ICA markers and the right panel's
    fallback-detection bars were the same ink with unrelated meanings.
    """
    assert BEFORE_COLOR != FLAG_COLOR
    assert AFTER_COLOR != FLAG_COLOR


def test_an_exclusion_is_never_a_detector_flag() -> None:
    """Most excluded components were never flagged by a detector, and vice versa."""
    assert EXCLUDED_COLOR != FLAG_COLOR
    assert EXCLUDED_COLOR != BEFORE_COLOR
    assert EXCLUDED_COLOR != AFTER_COLOR


def test_status_is_a_neutral_ramp_rather_than_a_hue() -> None:
    """Hue is reserved for measured quantities so that no hue means "a decision"."""
    for value in (EXCLUDED_COLOR, RETAINED_COLOR):
        assert not value.startswith("#")
    assert float(EXCLUDED_COLOR) < float(RETAINED_COLOR)


def test_a_mark_drawn_over_runs_owns_no_run_colour() -> None:
    """Flag rings sit on top of run-coloured points, so any hue would collide."""
    assert MARK_COLOR not in RUN_COLORS


def test_guides_are_distinct_from_the_status_ramp() -> None:
    assert GUIDE_COLOR not in (EXCLUDED_COLOR, RETAINED_COLOR)


def test_a_panel_wash_stays_lighter_than_the_mark_it_is_paired_with() -> None:
    """The wash sits behind a topography, so it cannot be the dark end of the ramp."""
    from eeg_pipeline.preprocessing.report.style import EXCLUDED_PANEL_FILL

    assert not EXCLUDED_PANEL_FILL.startswith("#")
    assert float(EXCLUDED_PANEL_FILL) > float(EXCLUDED_COLOR)


def test_report_css_constrains_figure_svgs_without_touching_ui_icons() -> None:
    """Matplotlib writes a hard ``width`` in points; nothing else caps it.

    MNE gives every ``<img>`` Bootstrap's ``img-fluid``, but inline SVGs are dropped in
    raw, so a wide vector figure (the volume-locked residual is 1814 pt) overflows the
    column instead of scaling. The selector has to stay scoped to figures: a bare ``svg``
    rule would also catch the accordion chevrons, which are sized by their own rule.
    """
    from eeg_pipeline.preprocessing.report.style import REPORT_CSS

    assert "max-width: 100%" in REPORT_CSS
    assert "figure svg" in REPORT_CSS
    for line in REPORT_CSS.splitlines():
        selector = line.split("{")[0].strip()
        if selector.endswith("svg"):
            assert selector.startswith("figure"), f"unscoped svg selector: {selector!r}"


def test_report_css_is_applied_once_across_repeated_open_and_save_cycles() -> None:
    """``add_custom_css`` appends, and the pipeline opens the same report six times."""
    from types import SimpleNamespace

    from eeg_pipeline.preprocessing.report.style import apply_report_css

    report = SimpleNamespace(include="")

    def add_custom_css(css: str) -> None:
        report.include += f"\n<style>{css}</style>"

    report.add_custom_css = add_custom_css

    for _ in range(6):
        apply_report_css(report)

    assert report.include.count("figure svg") == 1


def test_crowded_labels_are_nudged_apart_without_reordering() -> None:
    """Two identifiers printed over each other cost a panel the point of labelling it.

    The marker carries the value and the label only says which line it belongs to, so
    moving a label to keep it readable costs nothing a reader relies on. Reordering would:
    the label would then name the wrong trace.
    """
    placed = separated_labels([(0.0, "low"), (0.05, "high")], minimum_gap=1.0)

    assert [label for _, label in placed] == ["low", "high"]
    assert placed[1][0] - placed[0][0] >= 1.0


def test_labels_already_far_apart_are_left_where_they_were() -> None:
    placed = separated_labels([(0.0, "low"), (10.0, "high")], minimum_gap=1.0)

    assert placed == [(0.0, "low"), (10.0, "high")]


def test_labels_are_placed_in_value_order_whatever_order_they_arrived_in() -> None:
    """A label's position is a property of its value, not of the dict it was built from."""
    placed = separated_labels([(10.0, "high"), (0.0, "low")], minimum_gap=1.0)

    assert [label for _, label in placed] == ["low", "high"]


def test_a_recording_with_a_run_entity_is_labelled_by_its_run() -> None:
    from eeg_pipeline.preprocessing.report.style import run_label

    assert run_label("sub-0001_task-thermalactive_run-3") == "run-3"
    assert run_label("sub-0001_task-thermalactive_run-3", bare=True) == "3"


def test_a_label_that_is_already_a_label_passes_through() -> None:
    """Cohort tables carry a bare ``run-2`` rather than a full recording id.

    Pinned because collapsing those onto one value put six runs in a single column of the
    cohort grid, which read as a participant with one run.
    """
    from eeg_pipeline.preprocessing.report.style import run_label

    assert run_label("run-2") == "run-2"


def test_a_runless_recording_is_not_labelled_with_the_whole_recording_id() -> None:
    """A baseline or single-session acquisition carries no ``run-`` entity.

    The whole id came back, so every row of every per-run table read
    ``sub-0001_task-baseline`` — the subject and task the report is already titled with,
    repeated once per row and identical in all of them.
    """
    from eeg_pipeline.preprocessing.report.style import run_label

    label = run_label("sub-0001_task-baseline")

    assert "sub-0001" not in label
    assert label == "task-baseline"


def test_a_runless_recording_keeps_whatever_does_distinguish_it() -> None:
    """Without runs, the session is what tells two recordings apart."""
    from eeg_pipeline.preprocessing.report.style import run_label

    assert run_label("sub-0001_ses-02_task-rest") == "ses-02_task-rest"


def test_a_recording_id_with_nothing_left_to_name_it_says_so() -> None:
    """Rather than an empty cell, which reads as a missing value."""
    from eeg_pipeline.preprocessing.report.style import run_label

    assert run_label("sub-0001") == "recording"
