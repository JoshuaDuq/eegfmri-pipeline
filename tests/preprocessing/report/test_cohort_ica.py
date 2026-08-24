"""Decomposition quality, and the one place this report is allowed to say "violation".

Variance removed is the pipeline's headline number, pooled over the whole cohort now that
the acquisition-context axis is gone. The rank check is the opposite case -- an algebraic
fact that holds whatever the recording, and therefore the only thing here stated as a
fault.
"""

from __future__ import annotations

import matplotlib
import pandas as pd

matplotlib.use("Agg")

from eeg_pipeline.preprocessing.report.cohort.collect import Cohort  # noqa: E402
from eeg_pipeline.preprocessing.report.cohort.ica import (  # noqa: E402
    decomposition_frame,
    decomposition_table,
    plot_label_composition,
    rank_violations,
    variance_pooled,
    variance_summary_html,
)
from eeg_pipeline.preprocessing.report.cohort.record import component_label_counts  # noqa: E402
from eeg_pipeline.preprocessing.report.cohort.sidecar import (  # noqa: E402
    Paradigm,
    SubjectSidecar,
)


def _runs() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "run": ["run-1"],
            "n_channels": [63],
            "duration_s": [600.0],
            "flagged_fraction": [0.01],
            "continuity_median_db": [0.2],
            "continuity_max_db": [4.0],
        }
    )


def _participant(
    subject: str,
    *,
    n_components: int = 62,
    data_rank: int = 62,
    variance_removed: float = 0.86,
    n_excluded: int = 29,
    labels: dict[str, int] | None = None,
) -> SubjectSidecar:
    measurements: dict = {
        "n_channels": 63,
        "n_components": n_components,
        "data_rank": data_rank,
        "n_excluded": n_excluded,
        "retained_dimensions": n_components - n_excluded,
        "variance_removed": variance_removed,
        "samples_per_squared_component": 188.0,
        "condition_number": 245.0,
    }
    for label, count in (labels or {}).items():
        measurements[f"n_excluded_{label}"] = count
    return SubjectSidecar(
        subject=subject,
        task="thermalactive",
        paradigm=Paradigm.TASK,
        measurements=measurements,
        runs=_runs(),
    )


def _cohort(*participants) -> Cohort:
    return Cohort(participants=tuple(participants))


# --------------------------------------------------------------------------------------
# Label counting, at write time
# --------------------------------------------------------------------------------------


def _components(descriptions, statuses=None) -> pd.DataFrame:
    statuses = statuses or ["bad"] * len(descriptions)
    return pd.DataFrame(
        {
            "component": range(len(descriptions)),
            "status": statuses,
            "status_description": descriptions,
        }
    )


def test_detector_prose_is_counted_under_a_class() -> None:
    counts = component_label_counts(
        _components(
            [
                "Auto-detected eye blink (MNE-ICALabel)",
                "Auto-detected heart beat (MNE-ICALabel)",
                "Auto-detected ECG artifact (MNE)",
                "Auto-detected muscle artifact (MNE-ICALabel)",
            ]
        )
    )

    assert counts["eye"] == 1
    assert counts["heart"] == 2
    assert counts["muscle"] == 1


def test_only_excluded_components_are_counted() -> None:
    counts = component_label_counts(
        _components(
            ["Auto-detected eye blink (MNE-ICALabel)", "brain"],
            statuses=["bad", "good"],
        )
    )

    assert counts["eye"] == 1
    assert sum(counts.values()) == 1


def test_an_unrecognised_description_is_counted_rather_than_dropped() -> None:
    """A count that silently loses components would understate what was removed."""
    counts = component_label_counts(_components(["something a future detector wrote"]))

    assert counts["other"] == 1


def test_an_exclusion_with_no_recorded_reason_is_counted_as_such() -> None:
    counts = component_label_counts(_components([""]))

    assert counts["unrecorded"] == 1


def test_channel_noise_is_not_swallowed_by_line_noise() -> None:
    """Both contain 'noise', so the more specific reading has to win."""
    counts = component_label_counts(
        _components(["Auto-detected channel noise", "Auto-detected line noise"])
    )

    assert counts["channel"] == 1
    assert counts["line"] == 1


def test_a_participant_with_no_component_table_counts_nothing() -> None:
    assert sum(component_label_counts(None).values()) == 0


# --------------------------------------------------------------------------------------
# The rank check
# --------------------------------------------------------------------------------------


def test_more_components_than_rank_is_a_violation() -> None:
    """The one algebraic claim in the report: a rank-r dataset spans r directions."""
    frame = decomposition_frame(
        _cohort(
            _participant("0014", n_components=62, data_rank=62),
            _participant("0015", n_components=63, data_rank=61),
        )
    )

    assert rank_violations(frame) == ["0015"]


def test_a_well_posed_cohort_reports_no_violation() -> None:
    frame = decomposition_frame(_cohort(_participant("0014"), _participant("0015")))

    assert rank_violations(frame) == []


def test_a_participant_without_a_recorded_rank_is_not_accused() -> None:
    participant = _participant("0014")
    del participant.measurements["data_rank"]

    assert rank_violations(decomposition_frame(_cohort(participant))) == []


# --------------------------------------------------------------------------------------
# Variance removed
# --------------------------------------------------------------------------------------


def test_variance_removed_is_pooled_across_the_whole_cohort() -> None:
    """Every participant with a measurement contributes to one denominator."""
    frame = decomposition_frame(
        _cohort(
            *(
                _participant(
                    f"{index:04d}",
                    variance_removed=0.86,
                )
                for index in range(6)
            ),
            *(
                _participant(
                    f"{index:04d}",
                    variance_removed=0.30,
                )
                for index in range(10, 16)
            ),
        )
    )

    pooled = variance_pooled(frame)

    # One population: every participant contributes to the same denominator.
    assert pooled.denominator.n_subjects == 12


def test_a_cohort_with_no_measured_variance_pools_nothing() -> None:
    """An absent measurement must not read as a measured zero."""
    frame = decomposition_frame(_cohort(_participant("0014", variance_removed=float("nan"))))

    assert variance_pooled(frame) is None


def test_below_the_gate_the_participants_are_listed_instead_of_summarised() -> None:
    frame = decomposition_frame(
        _cohort(
            _participant("0014", variance_removed=0.86),
            _participant("0015", variance_removed=0.91),
        )
    )

    html = variance_summary_html(variance_pooled(frame))

    assert "0014 86.0%" in html
    assert "0015 91.0%" in html


# --------------------------------------------------------------------------------------
# Figures and table
# --------------------------------------------------------------------------------------


def test_the_composition_bar_shows_only_classes_that_occurred() -> None:
    frame = decomposition_frame(
        _cohort(
            _participant("0014", labels={"eye": 3, "heart": 9, "muscle": 0}),
            _participant("0015", labels={"eye": 1, "heart": 4, "muscle": 0}),
        )
    )

    figure = plot_label_composition(frame)

    labels = {text.get_text() for text in figure.axes[0].get_legend().get_texts()}
    assert labels == {"eye", "heart"}
    matplotlib.pyplot.close(figure)


def test_the_composition_bar_names_every_participant() -> None:
    frame = decomposition_frame(
        _cohort(
            _participant("0014", labels={"muscle": 20}),
            _participant("0015", labels={"muscle": 2}),
        )
    )

    figure = plot_label_composition(frame)

    assert [label.get_text() for label in figure.axes[0].get_yticklabels()] == ["0014", "0015"]
    matplotlib.pyplot.close(figure)


def test_the_table_is_sorted_by_variance_removed() -> None:
    """Sorting is the only emphasis: no row is marked as good or bad."""
    frame = decomposition_frame(
        _cohort(
            _participant("0014", variance_removed=0.70),
            _participant("0015", variance_removed=0.95),
        )
    )

    html = decomposition_table(frame)

    assert html.index("0015") < html.index("0014")
    assert "no row is marked" in html


def test_a_cohort_without_decomposition_measurements_has_no_frame() -> None:
    bare = SubjectSidecar(
        subject="0014",
        task="rest",
        paradigm=Paradigm.REST,
    )

    assert decomposition_frame(_cohort(bare)).empty
