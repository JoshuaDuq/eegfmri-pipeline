"""The cohort spectrum has two ways of quietly lying and one of being unreadable.

It lies if it resamples participants onto a common grid, which smears the narrow features
the figure exists to show, or if it marks a line-noise frequency that only some of the
contributing recordings were filtered at. It becomes unreadable if a notch band -- tens of
decibels below anything else -- is allowed into the power axis.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from eeg_pipeline.preprocessing.report.cohort.collect import Cohort  # noqa: E402
from eeg_pipeline.preprocessing.report.cohort.sidecar import (  # noqa: E402
    Paradigm,
    SubjectSidecar,
)
from eeg_pipeline.preprocessing.report.cohort.spectra import (  # noqa: E402
    MINIMUM_EXPONENT_SPAN,
    aperiodic_frame,
    aperiodic_table,
    cohort_spectra,
    participant_spectrum,
    plot_aperiodic_shift,
    plot_cohort_spectra,
    spectra_audit,
)

FREQUENCIES = np.round(np.logspace(np.log10(1.0), np.log10(100.0), 60), 4)


def _spectrum_curves(
    *, n_runs: int, level: float, frequencies: np.ndarray = FREQUENCIES, notch_at: float | None = None
) -> pd.DataFrame:
    frames = []
    for run in range(n_runs):
        for stage, offset in (("before", 4.0), ("after", 0.0)):
            median = level + offset - 10.0 * np.log10(frequencies)
            if notch_at is not None:
                # A notch drives its band to the numerical floor, far below the spectrum.
                median = np.where(np.abs(frequencies - notch_at) < 2.0, level - 90.0, median)
            frames.append(
                pd.DataFrame(
                    {
                        "run": f"run-{run + 1}",
                        "stage": stage,
                        "freq_hz": frequencies,
                        "median_db": median,
                        "max_db": median + 6.0,
                    }
                )
            )
    return pd.concat(frames, ignore_index=True)


def _runs(n_runs: int, *, exponent_before: float = 1.4, exponent_after: float = 1.35):
    return pd.DataFrame(
        {
            "run": [f"run-{index + 1}" for index in range(n_runs)],
            "n_channels": [63] * n_runs,
            "duration_s": [600.0] * n_runs,
            "flagged_fraction": [0.01] * n_runs,
            "continuity_median_db": [0.2] * n_runs,
            "continuity_max_db": [4.0] * n_runs,
            "aperiodic_exponent_before": [exponent_before] * n_runs,
            "aperiodic_exponent_after": [exponent_after] * n_runs,
            "aperiodic_offset_db_before": [22.0] * n_runs,
            "aperiodic_offset_db_after": [21.0] * n_runs,
            "aperiodic_r_squared_before": [0.98] * n_runs,
            "aperiodic_r_squared_after": [0.97] * n_runs,
        }
    )


def _participant(
    subject: str,
    *,
    level: float = 20.0,
    n_runs: int = 2,
    frequencies: np.ndarray = FREQUENCIES,
    line_frequency: float | None = 60.0,
    notch_at: float | None = None,
    exponent_after: float = 1.35,
) -> SubjectSidecar:
    return SubjectSidecar(
        subject=subject,
        task="thermalactive",
        paradigm=Paradigm.REST,
        settings=({} if line_frequency is None else {"spectra_line_frequency": line_frequency}),
        runs=_runs(n_runs, exponent_after=exponent_after),
        spectrum_curves=_spectrum_curves(
            n_runs=n_runs, level=level, frequencies=frequencies, notch_at=notch_at
        ),
    )


def _cohort(*participants) -> Cohort:
    return Cohort(participants=tuple(participants))


# --------------------------------------------------------------------------------------
# Pooling
# --------------------------------------------------------------------------------------


def test_a_participants_runs_are_pooled_before_it_joins_the_cohort() -> None:
    series = participant_spectrum(_participant("0014", n_runs=6), stage="after")

    assert series is not None
    assert series.index.to_numpy() == pytest.approx(FREQUENCIES)


def test_both_stages_are_pooled_separately() -> None:
    spectra = cohort_spectra(_cohort(_participant("0014"), _participant("0015")))

    assert spectra is not None
    # Before sits 4 dB above after by construction.
    assert spectra.before.per_subject["0014"][0] - spectra.after.per_subject["0014"][0] == (
        pytest.approx(4.0)
    )


def test_incompatible_bins_are_raised_rather_than_resampled() -> None:
    """Resampling would smear exactly the narrow features the figure exists to show.

    A different bin *spacing* is not a narrower range: there is no shared grid to restrict
    to, and any pooling would have to invent bins. That is raised, naming the participants,
    rather than silently interpolated.
    """
    different_spacing = np.round(np.logspace(np.log10(1.0), np.log10(100.0), 91), 4)
    cohort = _cohort(_participant("0014"), _participant("0015", frequencies=different_spacing))

    with pytest.raises(ValueError, match="0015"):
        cohort_spectra(cohort)


def test_a_narrower_participant_is_named_as_setting_the_range() -> None:
    shared = FREQUENCIES[FREQUENCIES <= 45.0]
    spectra = cohort_spectra(
        _cohort(_participant("0014"), _participant("0015", frequencies=shared))
    )

    assert spectra is not None
    assert spectra.limiting_subjects == ("0015",)
    assert spectra.frequencies[-1] == pytest.approx(shared[-1])


def test_a_cohort_with_no_spectra_has_no_panel() -> None:
    bare = SubjectSidecar(
        subject="0014",
        task="rest",
        paradigm=Paradigm.REST,
    )

    assert cohort_spectra(_cohort(bare)) is None


def test_two_participants_get_no_cohort_median() -> None:
    spectra = cohort_spectra(_cohort(_participant("0014"), _participant("0015")))

    assert spectra.after.median is None
    assert set(spectra.after.per_subject) == {"0014", "0015"}


# --------------------------------------------------------------------------------------
# The figure
# --------------------------------------------------------------------------------------


def test_a_notch_floor_does_not_squeeze_the_spectra_off_the_axis() -> None:
    """The failure this guards: 90 dB of filter stretching the panel it belongs to."""
    spectra = cohort_spectra(
        _cohort(
            _participant("0014", notch_at=60.0),
            _participant("0015", notch_at=60.0),
        )
    )

    figure = plot_cohort_spectra(spectra, line_frequency=60.0)

    low, high = figure.axes[0].get_ylim()
    # The spectra live within roughly 20 dB; the notch sits 90 dB below them and must not
    # be allowed to set the axis.
    assert high - low < 60.0
    matplotlib.pyplot.close(figure)


def test_the_frequency_axis_is_logarithmic() -> None:
    spectra = cohort_spectra(_cohort(_participant("0014"), _participant("0015")))

    figure = plot_cohort_spectra(spectra)

    assert figure.axes[0].get_xscale() == "log"
    matplotlib.pyplot.close(figure)


def test_a_small_cohort_names_every_trace() -> None:
    spectra = cohort_spectra(_cohort(_participant("0014"), _participant("0015")))

    figure = plot_cohort_spectra(spectra)

    labelled = {text.get_text() for text in figure.axes[0].texts}
    assert {"0014", "0015"} <= labelled
    matplotlib.pyplot.close(figure)


def test_a_line_frequency_all_participants_share_is_marked() -> None:
    spectra = cohort_spectra(_cohort(_participant("0014"), _participant("0015")))

    figure = plot_cohort_spectra(spectra, line_frequency=60.0)

    assert any("60 Hz line" in text.get_text() for text in figure.axes[0].texts)
    matplotlib.pyplot.close(figure)


def test_the_title_names_the_participant_that_narrowed_the_range() -> None:
    shared = FREQUENCIES[FREQUENCIES <= 45.0]
    spectra = cohort_spectra(
        _cohort(_participant("0014"), _participant("0015", frequencies=shared))
    )

    figure = plot_cohort_spectra(spectra)

    assert "range set by 0015" in figure.axes[0].get_title()
    matplotlib.pyplot.close(figure)


def test_the_audit_holds_every_plotted_trace() -> None:
    spectra = cohort_spectra(_cohort(_participant("0014"), _participant("0015")))

    audit = spectra_audit(spectra)

    assert "after_0014_db" in audit.columns
    assert "before_0015_db" in audit.columns
    assert len(audit) == spectra.frequencies.size


# --------------------------------------------------------------------------------------
# Aperiodic
# --------------------------------------------------------------------------------------


def test_the_aperiodic_frame_pools_runs_within_a_participant() -> None:
    frame = aperiodic_frame(_cohort(_participant("0014", n_runs=6)))

    assert frame.loc[0, "exponent_before"] == pytest.approx(1.4)
    assert frame.loc[0, "exponent_after"] == pytest.approx(1.35)


def test_a_participant_without_a_fit_is_absent_rather_than_zero() -> None:
    participant = _participant("0014")
    participant.runs.drop(
        columns=[column for column in participant.runs.columns if column.startswith("aperiodic")],
        inplace=True,
    )

    assert aperiodic_frame(_cohort(participant)).empty


def test_the_aperiodic_figure_pairs_each_participant() -> None:
    """Two boxplots would carry the between-participant spread instead."""
    frame = aperiodic_frame(_cohort(_participant("0014"), _participant("0015")))

    figure = plot_aperiodic_shift(frame)

    assert [label.get_text() for label in figure.axes[0].get_xticklabels()] == [
        "Before ICA",
        "After ICA",
    ]
    matplotlib.pyplot.close(figure)


def test_the_aperiodic_figure_draws_the_exponent_alone() -> None:
    """The offset falls whenever variance is removed, so a panel of it restates a column."""
    frame = aperiodic_frame(_cohort(_participant("0014"), _participant("0015")))

    figure = plot_aperiodic_shift(frame)

    assert len(figure.axes) == 1
    assert figure.axes[0].get_ylabel() == "1/f exponent"
    matplotlib.pyplot.close(figure)


def test_the_frequency_axis_is_labelled_where_a_reader_reads_it() -> None:
    """Decade-only ticks leave the alpha band somewhere between the first two labels."""
    spectra = cohort_spectra(_cohort(_participant("0014"), _participant("0015")))

    figure = plot_cohort_spectra(spectra)

    labels = {text.get_text() for text in figure.axes[0].get_xticklabels()}
    assert {"1", "10", "100"} <= labels
    assert "20" in labels and "50" in labels
    matplotlib.pyplot.close(figure)


def test_a_negligible_aperiodic_shift_is_not_magnified_into_a_cliff() -> None:
    """A paired axis autoscales to its own change, which flatters a trivial one."""
    frame = aperiodic_frame(
        _cohort(
            _participant("0014", exponent_after=1.399),
            _participant("0015", exponent_after=1.401),
        )
    )

    figure = plot_aperiodic_shift(frame)

    low, high = figure.axes[0].get_ylim()
    assert high - low == pytest.approx(MINIMUM_EXPONENT_SPAN)
    matplotlib.pyplot.close(figure)


def test_a_real_aperiodic_shift_still_fills_the_axis() -> None:
    frame = aperiodic_frame(
        _cohort(_participant("0014", exponent_after=1.35), _participant("0015", exponent_after=0.6))
    )

    figure = plot_aperiodic_shift(frame)

    low, high = figure.axes[0].get_ylim()
    assert high - low > MINIMUM_EXPONENT_SPAN
    matplotlib.pyplot.close(figure)


def test_participants_with_the_same_value_do_not_have_their_labels_overprinted() -> None:
    """Two identifiers drawn on top of each other cost the panel its point."""
    frame = aperiodic_frame(
        _cohort(
            _participant("0014", exponent_after=1.20),
            _participant("0015", exponent_after=1.20),
        )
    )

    figure = plot_aperiodic_shift(frame)

    positions = sorted(text.xy[1] for text in figure.axes[0].texts)
    assert len(positions) == 2
    assert positions[1] - positions[0] > 0.0
    matplotlib.pyplot.close(figure)


def test_the_aperiodic_table_reports_the_within_participant_shift() -> None:
    frame = aperiodic_frame(
        _cohort(
            _participant("0014", exponent_after=1.35),
            _participant("0015", exponent_after=0.90),
        )
    )

    html = aperiodic_table(frame)

    # 0.90 - 1.40: a participant whose background flattened through cleaning.
    assert "-0.50" in html
    assert "1/f exponent" in html
