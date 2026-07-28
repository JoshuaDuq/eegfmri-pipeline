from __future__ import annotations

import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")

from eeg_pipeline.preprocessing.report.rejection import (  # noqa: E402
    plot_rejection,
    rejection_summary_html,
    retention_by_group,
    summarize_rejection,
)


def _drop_log(dropped: set[int], total: int = 10):
    return [("AUTOREJECT",) if index in dropped else () for index in range(total)]


def test_summarize_counts_reasons_and_positions() -> None:
    summary = summarize_rejection(_drop_log({0, 1, 7}))

    assert summary.total == 10
    assert summary.kept == 7
    assert summary.dropped == 3
    assert summary.dropped_fraction == pytest.approx(0.3)
    assert summary.reasons == {"AUTOREJECT": 3}
    assert summary.dropped_positions == (0, 1, 7)


def test_summarize_handles_a_fully_retained_set() -> None:
    summary = summarize_rejection(_drop_log(set()))

    assert summary.dropped == 0
    assert summary.dropped_fraction == 0.0
    assert summary.reasons == {}


def test_multiple_reasons_on_one_epoch_are_all_counted() -> None:
    summary = summarize_rejection([("AUTOREJECT", "BAD_break"), (), ("BAD_break",)])

    assert summary.reasons == {"AUTOREJECT": 1, "BAD_break": 2}
    assert summary.dropped == 2


def test_retention_reports_only_columns_that_exist() -> None:
    events = pd.DataFrame({"run_id": [1, 1, 2], "unrelated": [0, 0, 0]})

    counts = retention_by_group(events, total=5)

    assert set(counts) == {"run_id"}
    assert counts["run_id"].to_dict() == {1: 2, 2: 1}


def test_retention_rejects_more_clean_trials_than_originals() -> None:
    events = pd.DataFrame({"run_id": [1, 2, 3]})

    with pytest.raises(ValueError, match="cannot contain more trials"):
        retention_by_group(events, total=2)


def test_summary_html_states_the_dropped_fraction_and_grouping() -> None:
    summary = summarize_rejection(_drop_log({0, 1}))
    counts = retention_by_group(pd.DataFrame({"run_id": [1, 1, 2, 2, 2]}), total=10)

    document = rejection_summary_html(summary, group_counts=counts)

    assert "20.0%" in document
    assert "run_id" in document
    assert "no longer a random sample" in document


def test_uniform_single_group_is_not_reported_as_a_distribution() -> None:
    """A single-valued grouping carries no information about uneven loss."""
    summary = summarize_rejection(_drop_log({0}))
    counts = retention_by_group(pd.DataFrame({"trial_type": ["a", "a", "a"]}), total=10)

    document = rejection_summary_html(summary, group_counts=counts)
    figure = plot_rejection(summary, group_counts=counts)

    assert "random sample" not in document
    assert len(figure.axes) == 1


def test_plot_marks_every_dropped_epoch() -> None:
    summary = summarize_rejection(_drop_log({0, 1, 2}, total=8))
    counts = retention_by_group(pd.DataFrame({"run_id": [1, 2, 2]}), total=8)

    figure = plot_rejection(summary, group_counts=counts)

    assert len(figure.axes) == 2
    assert len(figure.axes[0].patches) == 8
    assert "3 of 8 epochs dropped" in figure.axes[0].get_title()


def test_run_labels_are_carried_back_onto_pre_cleaning_positions() -> None:
    """Dropped positions are unlabelled; retained ones inherit their run in order."""
    import numpy as np

    from eeg_pipeline.preprocessing.report.rejection import run_of_position

    summary = summarize_rejection(_drop_log({1, 4}, total=6))
    events = pd.DataFrame({"run_id": [1, 1, 2, 2]})

    assignment = run_of_position(summary, events)

    np.testing.assert_array_equal(assignment[[0, 2, 3, 5]], [1, 1, 2, 2])
    assert np.isnan(assignment[1]) and np.isnan(assignment[4])


def test_run_mapping_refuses_a_misaligned_events_table() -> None:
    from eeg_pipeline.preprocessing.report.rejection import run_of_position

    summary = summarize_rejection(_drop_log({0}, total=6))

    with pytest.raises(ValueError, match="cannot be aligned"):
        run_of_position(summary, pd.DataFrame({"run_id": [1, 2]}))


def test_run_mapping_is_absent_without_a_run_column() -> None:
    from eeg_pipeline.preprocessing.report.rejection import run_of_position

    summary = summarize_rejection(_drop_log(set(), total=3))

    assert run_of_position(summary, pd.DataFrame({"trial_type": ["a", "b", "c"]})) is None


def test_the_retention_strip_is_a_strip_not_a_wall() -> None:
    """One bit per epoch must not be drawn floor to ceiling.

    Full-height bars in a saturated colour turned sixty-odd epochs into a block of ink
    that dominated the figure without making the few dropped epochs any easier to find.
    """
    summary = summarize_rejection(_drop_log({0, 1, 7}))

    figure = plot_rejection(summary)

    position_axis = figure.axes[0]
    bottom, top = position_axis.get_ylim()
    heights = [patch.get_height() for patch in position_axis.patches]
    assert heights, "the retention strip should draw one patch per epoch"
    assert max(heights) < 0.5 * (top - bottom)


def test_the_retention_strip_says_which_colour_means_dropped() -> None:
    """The strip is the only place the drop positions appear, so it carries its own key."""
    summary = summarize_rejection(_drop_log({0, 1, 7}))

    figure = plot_rejection(summary)

    legend = figure.axes[0].get_legend()
    assert legend is not None
    assert {text.get_text() for text in legend.get_texts()} == {"Kept", "Dropped"}


def test_dropped_and_kept_epochs_are_drawn_in_different_colours() -> None:
    summary = summarize_rejection(_drop_log({0, 1, 7}))

    figure = plot_rejection(summary)

    colours = [patch.get_facecolor() for patch in figure.axes[0].patches]
    assert colours[0] == colours[1] == colours[7]
    assert colours[0] != colours[2]
