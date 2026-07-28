"""The section that stops a destroyed cohort from looking like a clean one.

Its failure modes are the subtle ones. A peak frequency recorded where there is no rhythm.
A reliability compared across participants measured on different numbers of trials. An
ordinate mechanically coupled to the abscissa, so that the scatter draws an arithmetic
relationship and presents it as an empirical one.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from eeg_pipeline.preprocessing.report.cohort.collect import Cohort  # noqa: E402
from eeg_pipeline.preprocessing.report.cohort.preservation import (  # noqa: E402
    MAX_LABELLED_SCATTER,
    alpha_frame,
    alpha_table,
    cleaning_versus_signal,
    plot_alpha_prominence,
    plot_cleaning_versus_signal,
    reliability_frame,
    reliability_table,
)
from eeg_pipeline.preprocessing.report.cohort.sidecar import (  # noqa: E402
    AcquisitionContext,
    Paradigm,
    SubjectSidecar,
)

FREQUENCIES = np.round(np.arange(3.0, 25.25, 0.25), 4)


def _runs(n_runs: int = 2) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "run": [f"run-{index + 1}" for index in range(n_runs)],
            "n_channels": [63] * n_runs,
            "duration_s": [600.0] * n_runs,
            "flagged_fraction": [0.01] * n_runs,
            "continuity_median_db": [0.2] * n_runs,
            "continuity_max_db": [4.0] * n_runs,
        }
    )


def _participant(
    subject: str,
    *,
    prominence_before: float = 8.0,
    prominence_after: float = 6.0,
    resolvable: bool = True,
    peak_hz: float | None = 10.2,
    variance_removed: float | None = 0.86,
    reliability: float | None = 0.72,
    n_trials: int | None = 80,
    paradigm: Paradigm = Paradigm.TASK,
    frequencies: np.ndarray = FREQUENCIES,
) -> SubjectSidecar:
    measurements: dict = {
        "alpha_prominence_db_before": prominence_before,
        "alpha_prominence_db_after": prominence_after,
        "alpha_peak_resolvable_before": resolvable,
        "alpha_peak_resolvable_after": resolvable,
    }
    if resolvable and peak_hz is not None:
        measurements["alpha_peak_frequency_hz_before"] = peak_hz
        measurements["alpha_peak_frequency_hz_after"] = peak_hz
    if variance_removed is not None:
        measurements["variance_removed"] = variance_removed
    if reliability is not None and n_trials is not None:
        measurements["split_half_r"] = reliability
        measurements["split_half_n_trials"] = n_trials
    return SubjectSidecar(
        subject=subject,
        task="thermalactive",
        context=AcquisitionContext.OUT_OF_SCANNER,
        paradigm=paradigm,
        measurements=measurements,
        runs=_runs(),
    )


def _cohort(*participants) -> Cohort:
    return Cohort(participants=tuple(participants))


# --------------------------------------------------------------------------------------
# Alpha
# --------------------------------------------------------------------------------------






def test_a_resolvable_peak_contributes_its_frequency() -> None:
    frame = alpha_frame(_cohort(_participant("0014", resolvable=True, peak_hz=10.4)))

    assert frame.loc[0, "peak_hz_after"] == pytest.approx(10.4)
    assert bool(frame.loc[0, "resolvable_after"])


def test_an_unresolvable_peak_contributes_no_frequency() -> None:
    """The argmax of a band is still the largest bin when there is no rhythm there."""
    frame = alpha_frame(_cohort(_participant("0014", resolvable=False, peak_hz=None)))

    assert np.isnan(frame.loc[0, "peak_hz_after"])
    assert not bool(frame.loc[0, "resolvable_after"])


def test_the_alpha_table_counts_participants_without_a_resolvable_peak() -> None:
    frame = alpha_frame(
        _cohort(
            _participant("0014", resolvable=True),
            _participant("0015", resolvable=False, peak_hz=None),
        )
    )

    html = alpha_table(frame)

    assert "1 participant(s) carry no peak frequency" in html


def test_the_alpha_table_reports_the_within_participant_shift() -> None:
    frame = alpha_frame(_cohort(_participant("0014", prominence_before=8.0, prominence_after=2.0)))

    html = alpha_table(frame)

    assert "-6.0" in html


def test_the_prominence_figure_pairs_each_participant() -> None:
    frame = alpha_frame(_cohort(_participant("0014"), _participant("0015")))

    figure = plot_alpha_prominence(frame)

    assert [label.get_text() for label in figure.axes[0].get_xticklabels()] == [
        "Before ICA",
        "After ICA",
    ]
    matplotlib.pyplot.close(figure)



# --------------------------------------------------------------------------------------
# Reliability
# --------------------------------------------------------------------------------------


def test_reliability_is_projected_to_the_smallest_retained_trial_count() -> None:
    frame = reliability_frame(
        _cohort(
            _participant("0014", reliability=0.80, n_trials=120),
            _participant("0015", reliability=0.80, n_trials=40),
        )
    )

    assert frame["reference_trials"].tolist() == [40, 40]
    # Equal observed reliability, unequal trial counts: the one measured on three times
    # the trials is the weaker recording once both are compared at the same length.
    projected = dict(zip(frame["subject"], frame["projected"]))
    assert projected["0014"] < projected["0015"]
    assert projected["0015"] == pytest.approx(0.80)


def test_a_resting_participant_contributes_no_reliability() -> None:
    """Rest has no evoked response to split in half."""
    frame = reliability_frame(_cohort(_participant("0014", paradigm=Paradigm.REST)))

    assert frame.empty


def test_a_participant_without_a_trial_count_cannot_be_projected() -> None:
    frame = reliability_frame(_cohort(_participant("0014", n_trials=None)))

    assert frame.empty


def test_the_reliability_table_names_the_reference_it_projected_to() -> None:
    frame = reliability_frame(
        _cohort(
            _participant("0014", n_trials=120),
            _participant("0015", n_trials=44),
        )
    )

    html = reliability_table(frame)

    assert "At 44 trials" in html
    assert "smallest retained count" in html


# --------------------------------------------------------------------------------------
# Cleaning against signal
# --------------------------------------------------------------------------------------


def test_the_scatter_uses_prominence_rather_than_power() -> None:
    """Absolute power falls mechanically as variance is removed; prominence does not."""
    frame = cleaning_versus_signal(_cohort(_participant("0014", prominence_after=6.0)))

    assert "prominence_db" in frame.columns
    assert frame.loc[0, "prominence_db"] == pytest.approx(6.0)


def test_a_participant_missing_either_axis_is_absent_from_the_scatter() -> None:
    frame = cleaning_versus_signal(
        _cohort(_participant("0014"), _participant("0015", variance_removed=None))
    )

    assert frame["subject"].tolist() == ["0014"]


def test_the_scatter_distinguishes_participants_without_a_resolvable_peak() -> None:
    frame = cleaning_versus_signal(
        _cohort(
            _participant("0014", resolvable=True),
            _participant("0015", resolvable=False, peak_hz=None),
        )
    )

    figure = plot_cleaning_versus_signal(frame)

    labels = {text.get_text() for text in figure.axes[0].get_legend().get_texts()}
    assert "No resolvable peak" in labels
    matplotlib.pyplot.close(figure)


def test_the_scatter_fits_no_line_and_reports_no_coefficient() -> None:
    """A trend line through a screening plot invites a finding to be read out of it."""
    frame = cleaning_versus_signal(
        _cohort(*(_participant(f"{index:04d}", variance_removed=0.5 + index / 40) for index in range(12)))
    )

    figure = plot_cleaning_versus_signal(frame)

    # The only line on the axis is the definitional guide at zero prominence.
    assert len(figure.axes[0].lines) == 1
    assert figure.axes[0].lines[0].get_ydata()[0] == pytest.approx(0.0)
    assert "r =" not in figure.axes[0].get_title()
    matplotlib.pyplot.close(figure)


def test_the_scatter_keeps_naming_points_well_past_a_handful() -> None:
    """A screening plot that cannot name its outliers has failed at what it is for.

    Points are separated in two dimensions, so they crowd far later than endpoints in a
    column. The limit is correspondingly higher than the line panels'.
    """
    frame = cleaning_versus_signal(
        _cohort(*(_participant(f"{index:04d}") for index in range(20)))
    )

    figure = plot_cleaning_versus_signal(frame)

    assert len(figure.axes[0].texts) == 20
    matplotlib.pyplot.close(figure)


def test_a_cohort_beyond_the_scatter_limit_leaves_identities_to_the_table() -> None:
    frame = cleaning_versus_signal(
        _cohort(*(_participant(f"{index:04d}") for index in range(MAX_LABELLED_SCATTER + 5)))
    )

    figure = plot_cleaning_versus_signal(frame)

    assert not figure.axes[0].texts
    matplotlib.pyplot.close(figure)


def test_a_small_cohort_names_every_point() -> None:
    frame = cleaning_versus_signal(_cohort(_participant("0014"), _participant("0015")))

    figure = plot_cleaning_versus_signal(frame)

    labelled = {text.get_text() for text in figure.axes[0].texts}
    assert {"0014", "0015"} <= labelled
    matplotlib.pyplot.close(figure)


# --------------------------------------------------------------------------------------
# A bistable peak frequency must not read as a located rhythm
# --------------------------------------------------------------------------------------


def _alpha_row(subject: str, **values) -> dict:
    row = {
        "subject": subject,
        "prominence_before": 6.0,
        "prominence_after": 6.5,
        "resolvable_before": True,
        "resolvable_after": True,
        "peak_hz_before": 10.5,
        "peak_hz_after": 10.5,
        "contested_before": False,
        "contested_after": False,
        "runner_up_hz_before": np.nan,
        "runner_up_hz_after": np.nan,
        "runner_up_gap_db_before": np.nan,
        "runner_up_gap_db_after": np.nan,
    }
    row.update(values)
    return row


def test_a_contest_at_either_end_is_reported() -> None:
    """A contest before cleaning is what makes an apparent shift through it meaningless.

    Checking only the final stage would stay quiet about exactly the participant whose
    frequency is unsettled.
    """
    frame = pd.DataFrame([_alpha_row("0015", contested_before=True)])

    assert "contested for 0015" in alpha_table(frame)


def test_a_shift_between_contested_stages_is_named_as_an_argmax_crossing() -> None:
    """Seen on real data: 13.4 Hz before, 9.9 Hz after, the two bumps 0.5 dB apart."""
    frame = pd.DataFrame(
        [
            _alpha_row(
                "0015",
                peak_hz_before=13.4,
                peak_hz_after=9.9,
                contested_before=True,
                runner_up_hz_before=9.9,
                runner_up_gap_db_before=0.5,
            )
        ]
    )

    html = alpha_table(frame)

    assert "13.4 to 9.9 Hz" in html
    assert "crossed from one bump to the other" in html


def test_a_settled_shift_is_not_explained_away() -> None:
    """A frequency that moved with one clear peak at each end is a real change."""
    frame = pd.DataFrame([_alpha_row("0014", peak_hz_before=13.4, peak_hz_after=9.9)])

    html = alpha_table(frame)

    assert "crossed from one bump" not in html
    assert "contested for" not in html


def test_a_contested_peak_that_did_not_move_is_still_flagged() -> None:
    """The frequency being unsettled is worth knowing whether or not it happened to move."""
    frame = pd.DataFrame([_alpha_row("0015", contested_after=True)])

    html = alpha_table(frame)

    assert "contested for 0015" in html
    assert "crossed from one bump" not in html
