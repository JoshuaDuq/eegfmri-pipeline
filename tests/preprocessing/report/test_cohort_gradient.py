"""The gradient section has to survive the two ways it can read backwards.

Pooling harmonics by frequency across participants scanned at different repetition times
silently mixes one participant's third harmonic with another's fourth. And a participant
whose volume markers are irregular has its comb smeared across neighbouring bins, which
lowers every measured excess and makes poor timing look like a clean correction.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from eeg_pipeline.preprocessing.report.cohort.collect import Cohort  # noqa: E402
from eeg_pipeline.preprocessing.report.cohort.gradient import (  # noqa: E402
    attenuation_table,
    cohort_comb,
    comb_attenuation,
    comb_audit,
    participant_comb,
    plot_cohort_comb,
    timing_table,
)
from eeg_pipeline.preprocessing.report.cohort.sidecar import (  # noqa: E402
    AcquisitionContext,
    Paradigm,
    SubjectSidecar,
)

INDICES = np.asarray([30, 31, 32, 33])


def _comb_curves(
    *,
    n_runs: int,
    before_db: float,
    after_db: float,
    repetition_time_s: float,
    indices: np.ndarray = INDICES,
) -> pd.DataFrame:
    frames = []
    for run in range(n_runs):
        frames.append(
            pd.DataFrame(
                {
                    "run": f"run-{run + 1}",
                    "harmonic_index": indices,
                    "harmonic_hz": indices / repetition_time_s,
                    "notched": False,
                    "before_excess_db_median": np.full(indices.size, before_db),
                    "before_excess_db_max": np.full(indices.size, before_db + 6.0),
                    "after_excess_db_median": np.full(indices.size, after_db),
                    "after_excess_db_max": np.full(indices.size, after_db + 3.0),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _runs(*, n_runs: int, repetition_time_s: float, jitter_s: float = 0.002) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "run": [f"run-{index + 1}" for index in range(n_runs)],
            "n_channels": [63] * n_runs,
            "duration_s": [600.0] * n_runs,
            "flagged_fraction": [0.01] * n_runs,
            "continuity_median_db": [0.2] * n_runs,
            "continuity_max_db": [4.0] * n_runs,
            "n_volumes": [300] * n_runs,
            "repetition_time_s": [repetition_time_s] * n_runs,
            "volume_jitter_s": [jitter_s] * n_runs,
            "volume_locked_corrected_before_uv": [2.0] * n_runs,
            "volume_locked_corrected_uv": [0.8] * n_runs,
            "volume_locked_noise_floor_uv": [0.3] * n_runs,
        }
    )


def _participant(
    subject: str,
    *,
    before_db: float = 14.0,
    after_db: float = 2.0,
    repetition_time_s: float = 2.0,
    n_runs: int = 2,
    jitter_s: float = 0.002,
    indices: np.ndarray = INDICES,
    with_comb: bool = True,
) -> SubjectSidecar:
    return SubjectSidecar(
        subject=subject,
        task="thermalactive",
        context=AcquisitionContext.IN_SCANNER,
        paradigm=Paradigm.TASK,
        runs=_runs(n_runs=n_runs, repetition_time_s=repetition_time_s, jitter_s=jitter_s),
        comb_curves=(
            _comb_curves(
                n_runs=n_runs,
                before_db=before_db,
                after_db=after_db,
                repetition_time_s=repetition_time_s,
                indices=indices,
            )
            if with_comb
            else pd.DataFrame()
        ),
    )


def _cohort(*participants) -> Cohort:
    return Cohort(participants=tuple(participants))


# --------------------------------------------------------------------------------------
# Pooling
# --------------------------------------------------------------------------------------


def test_a_participants_runs_are_pooled_before_it_joins_the_cohort() -> None:
    pooled = participant_comb(_participant("0014", n_runs=6))

    assert len(pooled) == INDICES.size
    assert pooled["after_excess_db_median"].tolist() == [2.0] * INDICES.size


def test_a_participant_without_a_resolved_comb_contributes_nothing() -> None:
    assert participant_comb(_participant("0014", with_comb=False)) is None


def test_a_cohort_where_nobody_resolved_a_comb_has_no_panel() -> None:
    """An ordinary outcome: the measurement declines on frequency resolution."""
    assert cohort_comb(_cohort(_participant("0014", with_comb=False))) is None


def test_a_shared_repetition_time_is_drawn_against_frequency() -> None:
    comb = cohort_comb(_cohort(_participant("0014"), _participant("0015")))

    assert comb is not None
    assert comb.on_frequency_axis
    assert comb.harmonic_hz == pytest.approx(INDICES / 2.0)


def test_mixed_repetition_times_fall_back_to_the_harmonic_index() -> None:
    """The failure this guards: pooling one participant's third harmonic with another's."""
    comb = cohort_comb(
        _cohort(
            _participant("0014", repetition_time_s=2.0),
            _participant("0015", repetition_time_s=1.5),
        )
    )

    assert comb is not None
    assert not comb.on_frequency_axis
    assert comb.harmonic_hz is None
    assert comb.harmonic_index == pytest.approx(INDICES)


def test_participants_covering_different_harmonics_are_restricted_to_the_shared_set() -> None:
    comb = cohort_comb(
        _cohort(
            _participant("0014", indices=np.asarray([30, 31, 32, 33])),
            _participant("0015", indices=np.asarray([32, 33, 34, 35])),
        )
    )

    assert comb is not None
    assert comb.harmonic_index == pytest.approx(np.asarray([32.0, 33.0]))


def test_every_participant_is_kept_beneath_the_summary() -> None:
    comb = cohort_comb(_cohort(_participant("0014"), _participant("0015")))

    assert set(comb.after.per_subject) == {"0014", "0015"}


def test_two_participants_get_no_cohort_median() -> None:
    comb = cohort_comb(_cohort(_participant("0014"), _participant("0015")))

    assert comb.after.median is None
    assert comb.after.denominator.n_subjects == 2


# --------------------------------------------------------------------------------------
# Attenuation
# --------------------------------------------------------------------------------------


def test_attenuation_is_a_within_participant_difference() -> None:
    attenuation = comb_attenuation(
        cohort_comb(
            _cohort(
                _participant("0014", before_db=14.0, after_db=2.0),
                _participant("0015", before_db=20.0, after_db=11.0),
            )
        )
    )

    assert attenuation["0014"] == pytest.approx(12.0)
    assert attenuation["0015"] == pytest.approx(9.0)


def test_a_participant_whose_comb_survived_shows_little_attenuation() -> None:
    attenuation = comb_attenuation(
        cohort_comb(_cohort(_participant("0014", before_db=14.0, after_db=13.0)))
    )

    assert attenuation["0014"] == pytest.approx(1.0)


def test_the_attenuation_table_names_the_pairing_it_used() -> None:
    html = attenuation_table(cohort_comb(_cohort(_participant("0014"), _participant("0015"))))

    assert "within-participant difference" in html
    assert "0014" in html and "0015" in html


# --------------------------------------------------------------------------------------
# Timing, which works against the panels above
# --------------------------------------------------------------------------------------


def test_the_timing_table_reports_jitter_in_milliseconds() -> None:
    html = timing_table(_cohort(_participant("0014", jitter_s=0.004)))

    assert "4.0" in html
    assert "Worst jitter (ms)" in html


def test_the_timing_table_says_why_jitter_matters() -> None:
    """Without this, poor timing reads as a clean correction."""
    html = timing_table(_cohort(_participant("0014")))

    assert "smear" in html
    assert "cleanest comb" in html


def test_the_timing_table_reports_the_floor_beside_the_residual() -> None:
    html = timing_table(_cohort(_participant("0014")))

    assert "Noise floor" in html
    assert "does not fall merely because a participant was scanned for longer" in html


def test_an_eeg_only_cohort_has_no_timing_table() -> None:
    participant = SubjectSidecar(
        subject="0020",
        task="rest",
        context=AcquisitionContext.OUT_OF_SCANNER,
        paradigm=Paradigm.REST,
        runs=pd.DataFrame(
            {
                "run": ["run-1"],
                "n_channels": [63],
                "duration_s": [600.0],
                "flagged_fraction": [0.0],
                "continuity_median_db": [0.0],
                "continuity_max_db": [1.0],
            }
        ),
    )

    assert timing_table(_cohort(participant)) == ""


# --------------------------------------------------------------------------------------
# Figure and audit
# --------------------------------------------------------------------------------------


def test_the_audit_holds_every_plotted_value() -> None:
    comb = cohort_comb(_cohort(_participant("0014"), _participant("0015")))

    audit = comb_audit(comb)

    assert len(audit) == INDICES.size
    assert "before_0014_db" in audit.columns
    assert "after_0015_db" in audit.columns
    assert audit["n_subjects"].tolist() == [2] * INDICES.size


def test_the_audit_records_no_median_where_none_was_drawn() -> None:
    comb = cohort_comb(_cohort(_participant("0014"), _participant("0015")))

    assert comb_audit(comb)["after_median_db"].isna().all()


def test_the_figure_labels_the_axis_it_actually_used() -> None:
    mixed = cohort_comb(
        _cohort(
            _participant("0014", repetition_time_s=2.0),
            _participant("0015", repetition_time_s=1.5),
        )
    )

    figure = plot_cohort_comb(mixed)

    assert "Harmonic index" in figure.axes[0].get_xlabel()
    matplotlib.pyplot.close(figure)


def test_a_small_cohort_names_every_trace_on_the_figure() -> None:
    """An unlabelled trace shows that a correction failed without saying whose."""
    comb = cohort_comb(
        _cohort(
            _participant("0014", after_db=1.5),
            _participant("0015", after_db=9.0),
        )
    )

    figure = plot_cohort_comb(comb)

    labelled = {text.get_text() for text in figure.axes[0].texts}
    assert {"0014", "0015"} <= labelled
    matplotlib.pyplot.close(figure)


def test_a_large_cohort_drops_the_trace_labels_rather_than_colliding_them() -> None:
    comb = cohort_comb(_cohort(*(_participant(f"{index:04d}") for index in range(12))))

    figure = plot_cohort_comb(comb)

    assert not figure.axes[0].texts
    # The cohort median carries the panel instead.
    assert comb.after.median is not None
    matplotlib.pyplot.close(figure)


def test_individual_traces_fade_as_the_cohort_grows() -> None:
    """Two traces must read as two recordings; forty must read as a band."""
    small = plot_cohort_comb(cohort_comb(_cohort(_participant("0014"), _participant("0015"))))
    large = plot_cohort_comb(
        cohort_comb(_cohort(*(_participant(f"{index:04d}") for index in range(30))))
    )

    small_alpha = min(line.get_alpha() or 1.0 for line in small.axes[0].lines)
    large_alpha = min(line.get_alpha() or 1.0 for line in large.axes[0].lines)
    assert large_alpha < small_alpha
    matplotlib.pyplot.close(small)
    matplotlib.pyplot.close(large)


def test_the_figure_names_the_frequency_axis_when_it_is_legitimate() -> None:
    comb = cohort_comb(_cohort(_participant("0014"), _participant("0015")))

    figure = plot_cohort_comb(comb)

    assert "Harmonic frequency" in figure.axes[0].get_xlabel()
    assert "2 participant(s)" in figure.axes[0].get_title()
    matplotlib.pyplot.close(figure)


def test_a_participant_is_named_once_rather_than_at_both_stages() -> None:
    """Two labels per participant at one x is the collision the spectrum panel avoids.

    Colour already carries the stage, so a label on each of the before and after traces
    prints the same identifier twice at the same right edge. The spectrum panel labels the
    cleaned trace only, which is the one a reader tracks.
    """
    comb = cohort_comb(
        _cohort(
            _participant("0014", after_db=1.5),
            _participant("0015", after_db=9.0),
        )
    )

    figure = plot_cohort_comb(comb)

    assert sorted(text.get_text() for text in figure.axes[0].texts) == ["0014", "0015"]
    matplotlib.pyplot.close(figure)


def test_crowded_trace_labels_are_nudged_apart() -> None:
    """Two participants whose combs land together must still be individually readable."""
    comb = cohort_comb(
        _cohort(
            _participant("0014", after_db=2.00),
            _participant("0015", after_db=2.01),
        )
    )

    figure = plot_cohort_comb(comb)

    # ``xy`` is the annotated point in data coordinates; ``get_position`` would return the
    # constant offset in points that every label shares.
    positions = sorted(text.xy[1] for text in figure.axes[0].texts)
    assert positions[1] - positions[0] > 0.1
    matplotlib.pyplot.close(figure)


def test_the_timing_table_pairs_the_locked_residual_before_and_after() -> None:
    """The pre-ICA locked amplitude is measured and recorded, and was never reported.

    Reporting the residual after cleaning alone leaves a reader unable to tell a recording
    that never had a volume-locked artifact from one whose correction removed it.
    """
    table = timing_table(_cohort(_participant("0014")))

    assert "Before ICA" in table
    assert "2.00" in table  # the pre-ICA amplitude
    assert "1.20" in table  # what the exclusions removed, distinct from either side


def test_a_sidecar_without_the_before_column_still_reports_its_timing() -> None:
    """A sidecar written before the pre-ICA amplitude was recorded is not a broken one.

    Only the required columns are guaranteed. ``DataFrame.get`` returns ``None`` for an
    absent one and ``pd.to_numeric(None)`` returns a bare float, so reading an optional
    column that way fails on exactly the sidecars that predate it.
    """
    participant = _participant("0014")
    participant = SubjectSidecar(
        subject=participant.subject,
        task=participant.task,
        context=participant.context,
        paradigm=participant.paradigm,
        runs=participant.runs.drop(columns=["volume_locked_corrected_before_uv"]),
        comb_curves=participant.comb_curves,
    )

    table = timing_table(_cohort(participant))

    assert "0014" in table
    assert "0.80" in table  # the after-ICA amplitude, which is present
