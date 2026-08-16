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


def test_ignored_events_are_not_epochs_in_the_rejection_denominator() -> None:
    """MNE logs non-selected events, but they never entered epoch cleaning."""
    summary = summarize_rejection(
        [("IGNORED",), (), ("AUTOREJECT",), ("IGNORED",), (), ("BAD_break",)]
    )

    assert summary.total == 4
    assert summary.kept == 2
    assert summary.dropped == 2
    assert summary.dropped_fraction == 0.5
    assert summary.reasons == {"AUTOREJECT": 1, "BAD_break": 1}
    assert summary.dropped_positions == (1, 3)


def test_retention_reports_only_columns_that_exist() -> None:
    events = pd.DataFrame({"run_id": [1, 1, 2], "unrelated": [0, 0, 0]})

    retention = retention_by_group(events, total=5)

    assert set(retention) == {"run_id"}
    assert retention["run_id"].retained.to_dict() == {1: 2, 2: 1}


def test_retention_rejects_more_clean_trials_than_originals() -> None:
    events = pd.DataFrame({"run_id": [1, 2, 3]})

    with pytest.raises(ValueError, match="cannot contain more trials"):
        retention_by_group(events, total=2)


def test_summary_html_states_the_dropped_fraction_and_grouping() -> None:
    summary = summarize_rejection(_drop_log({0, 1}))
    retention = retention_by_group(pd.DataFrame({"run_id": [1, 1, 2, 2, 2]}), total=10)

    document = rejection_summary_html(summary, group_retention=retention)

    assert "20.0%" in document
    assert "run_id" in document
    assert "cannot distinguish the original design from differential rejection" in document
    assert "may overlap" in document


def test_uniform_single_group_is_not_reported_as_a_distribution() -> None:
    """A single-valued grouping carries no information about uneven loss."""
    summary = summarize_rejection(_drop_log({0}))
    retention = retention_by_group(
        pd.DataFrame({"trial_type": ["a", "a", "a"]}), total=10
    )

    document = rejection_summary_html(summary, group_retention=retention)
    figure = plot_rejection(summary, group_retention=retention)

    assert "random sample" not in document
    assert len(figure.axes) == 1


def test_plot_marks_every_dropped_epoch() -> None:
    summary = summarize_rejection(_drop_log({0, 1, 2}, total=8))
    retention = retention_by_group(pd.DataFrame({"run_id": [1, 2, 2]}), total=8)

    figure = plot_rejection(summary, group_retention=retention)

    assert len(figure.axes) == 2
    assert len(figure.axes[0].patches) == 8
    assert "3 of 8 epochs dropped" in figure.axes[0].get_title()


def test_presented_counts_produce_per_group_retention_rates() -> None:
    presented = pd.DataFrame({"trial_type": ["a"] * 5 + ["b"] * 5})
    retained = pd.DataFrame({"trial_type": ["a"] * 4 + ["b"]})
    summary = summarize_rejection(_drop_log({4, 6, 7, 8, 9}, total=10))

    groups = retention_by_group(retained, presented_events=presented, total=10)
    document = rejection_summary_html(summary, group_retention=groups)
    figure = plot_rejection(summary, group_retention=groups)

    assert groups["trial_type"].rates.to_dict() == {"a": 0.8, "b": 0.2}
    assert "4 / 5 (80.0%)" in document
    assert "1 / 5 (20.0%)" in document
    assert "Retention rate by trial_type" in figure.axes[1].get_title()
    assert figure.axes[1].get_ylim() == pytest.approx((0.0, 1.0))


def test_run_labels_are_carried_back_onto_pre_cleaning_positions() -> None:
    """Every position inherits its run from the full presented-event table."""
    import numpy as np

    from eeg_pipeline.preprocessing.report.rejection import run_of_position

    summary = summarize_rejection(_drop_log({1, 4}, total=6))
    events = pd.DataFrame({"run_id": [1, 1, 1, 2, 2, 2]})

    assignment = run_of_position(summary, events)

    np.testing.assert_array_equal(assignment, [1, 1, 1, 2, 2, 2])


def test_run_mapping_refuses_a_misaligned_events_table() -> None:
    from eeg_pipeline.preprocessing.report.rejection import run_of_position

    summary = summarize_rejection(_drop_log({0}, total=6))

    with pytest.raises(ValueError, match="contain 2 rows"):
        run_of_position(summary, pd.DataFrame({"run_id": [1, 2]}))


def test_run_mapping_is_absent_without_a_run_column() -> None:
    from eeg_pipeline.preprocessing.report.rejection import run_of_position

    summary = summarize_rejection(_drop_log(set(), total=3))

    assert run_of_position(summary, pd.DataFrame({"trial_type": ["a", "b", "c"]})) is None


def test_run_boundary_is_not_inferred_across_dropped_epochs() -> None:
    """Drops on both sides of a boundary must not move that boundary to a neighbour."""
    import numpy as np

    from eeg_pipeline.preprocessing.report.rejection import run_of_position

    summary = summarize_rejection(_drop_log({2, 3}, total=6))
    presented = pd.DataFrame({"run_id": [1, 1, 1, 2, 2, 2]})
    assignment = run_of_position(summary, presented)

    figure = plot_rejection(summary, run_assignment=assignment)
    boundary_lines = [
        line
        for line in figure.axes[0].lines
        if np.allclose(np.asarray(line.get_xdata(), dtype=float), 2.5)
    ]

    assert len(boundary_lines) == 1


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


def test_a_session_that_dropped_nothing_gets_no_retention_breakdown() -> None:
    """Bars at 1.0 in every group answer 'did rejection fall unevenly' with the same 'no'
    the header already gave, and spend two panels of the figure doing it.

    The position strip stays: an empty strip is the difference between nothing dropped
    and nothing measured.
    """
    import matplotlib.pyplot as plt

    from eeg_pipeline.preprocessing.report.rejection import (
        GroupRetention,
        RejectionSummary,
        plot_rejection,
    )

    import pandas as pd

    counts = pd.Series({"44.3": 33, "48.3": 33})
    retention = {"stimulus_temp": GroupRetention(retained=counts, presented=counts)}

    kept_everything = plot_rejection(
        RejectionSummary(total=66, kept=66, reasons={}, dropped_positions=()),
        group_retention=retention,
    )
    dropped_some = plot_rejection(
        RejectionSummary(total=66, kept=64, reasons={"PTP": 2}, dropped_positions=(3, 9)),
        group_retention=retention,
    )

    assert len(kept_everything.axes) == 1
    assert len(dropped_some.axes) > 1
    plt.close(kept_everything)
    plt.close(dropped_some)
