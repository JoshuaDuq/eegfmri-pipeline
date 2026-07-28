"""The pooling rules a cohort figure rests on, pinned without rendering anything.

Every scientific claim the cohort report makes is a claim about how participants and runs
were combined. Those claims are testable as arithmetic, and this file is where they are
tested, so that a broken aggregation fails here rather than in a figure a reader has to
disbelieve on sight.
"""

from __future__ import annotations

import numpy as np
import pytest

from eeg_pipeline.preprocessing.report.cohort.aggregate import (
    BandGates,
    BandRegime,
    Contribution,
    align_grids,
    paired_differences,
    pool_participants_curve,
    pool_participants_scalar,
    pool_runs_curve,
    pool_runs_rate,
    pool_runs_union,
    project_reliability,
    quantile_is_supported,
)


def _scalars(values, *, run_counts=None):
    """Contributions from a mapping of subject to value."""
    counts = run_counts or {}
    return [
        Contribution(
            subject=subject,
            value=np.asarray([value], dtype=float),
            n_runs=counts.get(subject, 1),
        )
        for subject, value in values.items()
    ]


# --------------------------------------------------------------------------------------
# Run to subject
# --------------------------------------------------------------------------------------


def test_a_participant_with_many_runs_does_not_outvote_one_with_few() -> None:
    """The unit of inference is the participant, so run count cannot buy influence."""
    values = {"0001": 0.0, "0002": 1.0, "0003": 2.0, "0004": 3.0, "0005": 4.0}
    # One participant sat in the scanner twenty times longer than anyone else.
    pooled = pool_participants_scalar(_scalars(values, run_counts={"0005": 100}))

    # The median is the middle participant. Weighting by run count would drag it to the
    # extreme, and the cohort figure would be describing one long session.
    assert pooled.median == pytest.approx(2.0)
    assert pooled.denominator.n_subjects == 5
    assert pooled.denominator.n_runs == 104


def test_rates_pool_over_the_whole_session_rather_than_across_run_medians() -> None:
    """A 30-second run must not carry the weight of a 12-minute one."""
    # One long clean run and one short run that was entirely flagged.
    numerators = (0.0, 30.0)
    denominators = (720.0, 30.0)

    pooled = pool_runs_rate(numerators, denominators)

    # Sum over sum: 30 of 750 seconds were flagged.
    assert pooled == pytest.approx(0.04)
    # The unweighted median of the per-run fractions would claim half the session was
    # flagged, which is the failure this rule exists to prevent.
    assert np.median([0.0, 1.0]) == pytest.approx(0.5)


def test_curves_pool_across_runs_with_equal_weight() -> None:
    """Each run is an independent estimate of the same spectrum."""
    runs = [np.asarray([1.0, 2.0]), np.asarray([3.0, 4.0]), np.asarray([11.0, 12.0])]

    assert pool_runs_curve(runs) == pytest.approx(np.asarray([3.0, 4.0]))


def test_bad_channels_pool_as_a_union_across_runs() -> None:
    """A channel bad in any run is bad for the session under a union policy."""
    assert pool_runs_union([("Fp1", "Oz"), (), ("Fp1", "T7")]) == ("Fp1", "Oz", "T7")


# --------------------------------------------------------------------------------------
# Pooling scale
# --------------------------------------------------------------------------------------


def test_median_pooling_is_transform_invariant_at_odd_participant_counts() -> None:
    """The median commutes with the decibel transform, so the scale cannot bias it."""
    linear = {"a": 1.0, "b": 10.0, "c": 100.0, "d": 1000.0, "e": 10000.0}

    pooled_linear = pool_participants_scalar(_scalars(linear)).median
    in_decibels = {name: 10.0 * np.log10(value) for name, value in linear.items()}
    pooled_decibels = pool_participants_scalar(_scalars(in_decibels)).median

    assert pooled_linear == pytest.approx(100.0)
    assert 10.0 * np.log10(pooled_linear) == pytest.approx(pooled_decibels)


def test_even_participant_counts_average_the_central_pair_in_the_pooling_scale() -> None:
    """The one place the median does depend on scale, pinned rather than glossed over.

    With an even count the two central order statistics are averaged, and an arithmetic
    mean does not commute with a logarithm. The deviation is small and bounded by the gap
    between the central pair; the mean, by contrast, is scale-dependent at every count.
    This test exists so the caveat cannot be quietly forgotten.
    """
    linear = {"a": 1.0, "b": 10.0, "c": 100.0, "d": 1000.0, "e": 10000.0, "f": 100000.0}

    pooled_linear = pool_participants_scalar(_scalars(linear)).median
    in_decibels = {name: 10.0 * np.log10(value) for name, value in linear.items()}
    pooled_decibels = pool_participants_scalar(_scalars(in_decibels)).median

    # The two central participants are averaged, and an arithmetic mean of powers is not
    # the arithmetic mean of their decibels.
    assert pooled_linear == pytest.approx(550.0)
    assert pooled_decibels == pytest.approx(25.0)
    assert 10.0 * np.log10(pooled_linear) == pytest.approx(27.4, abs=0.1)


# --------------------------------------------------------------------------------------
# Band gating
# --------------------------------------------------------------------------------------


def test_quantile_support_follows_the_weibull_plotting_position() -> None:
    """A quantile is supported when it lies inside the observed order statistics."""
    # The k-th of n order statistics estimates the k/(n+1) quantile, so p is interior
    # when 1/(n+1) <= p <= n/(n+1).
    assert not quantile_is_supported(0.25, 2)
    assert quantile_is_supported(0.25, 3)
    assert not quantile_is_supported(0.10, 8)
    assert quantile_is_supported(0.10, 9)


def test_gates_that_would_extrapolate_a_quantile_are_rejected() -> None:
    """A configurable gate must still respect what the data can support."""
    with pytest.raises(ValueError, match="quartile"):
        BandGates(min_subjects_for_median=2, min_subjects_for_outer_band=10)

    with pytest.raises(ValueError, match="decile"):
        BandGates(min_subjects_for_median=5, min_subjects_for_outer_band=8)


def test_below_the_median_gate_only_individual_participants_are_reported() -> None:
    """At four participants there is no cohort statistic to state."""
    pooled = pool_participants_scalar(_scalars({str(i): float(i) for i in range(4)}))

    assert pooled.regime is BandRegime.INDIVIDUALS
    assert pooled.median is None
    assert pooled.quartiles is None
    assert pooled.deciles is None
    assert len(pooled.per_subject) == 4


def test_the_interquartile_band_appears_at_the_median_gate() -> None:
    pooled = pool_participants_scalar(_scalars({str(i): float(i) for i in range(5)}))

    assert pooled.regime is BandRegime.MEDIAN_IQR
    assert pooled.median == pytest.approx(2.0)
    assert pooled.quartiles is not None
    assert pooled.deciles is None


def test_the_outer_band_appears_only_at_the_outer_gate() -> None:
    nine = pool_participants_scalar(_scalars({str(i): float(i) for i in range(9)}))
    ten = pool_participants_scalar(_scalars({str(i): float(i) for i in range(10)}))

    assert nine.regime is BandRegime.MEDIAN_IQR
    assert nine.deciles is None
    assert ten.regime is BandRegime.MEDIAN_IQR_DECILES
    assert ten.deciles is not None


def test_individual_participants_survive_every_regime() -> None:
    """Summary statistics are added to the participants, never substituted for them."""
    pooled = pool_participants_scalar(_scalars({str(i): float(i) for i in range(40)}))

    assert pooled.regime is BandRegime.MEDIAN_IQR_DECILES
    assert len(pooled.per_subject) == 40


def test_a_single_participant_reports_itself_without_summary_language() -> None:
    pooled = pool_participants_scalar(_scalars({"0014": 3.0}))

    assert pooled.regime is BandRegime.INDIVIDUALS
    assert pooled.median is None
    assert pooled.denominator.n_subjects == 1


# --------------------------------------------------------------------------------------
# Denominators
# --------------------------------------------------------------------------------------


def test_every_aggregate_carries_the_participants_behind_it() -> None:
    """A panel cannot print a number without the denominator that produced it."""
    pooled = pool_participants_scalar(
        _scalars({"0014": 1.0, "0015": 2.0}, run_counts={"0014": 6, "0015": 4})
    )

    assert pooled.denominator.subjects == ("0014", "0015")
    assert pooled.denominator.n_subjects == 2
    assert pooled.denominator.n_runs == 10


def test_pooling_nothing_is_an_error_rather_than_an_empty_panel() -> None:
    with pytest.raises(ValueError, match="at least one participant"):
        pool_participants_scalar([])


def test_a_duplicated_participant_is_an_error() -> None:
    """Two rows for one participant would weight them twice."""
    duplicated = [
        Contribution(subject="0014", value=np.asarray([1.0]), n_runs=1),
        Contribution(subject="0014", value=np.asarray([2.0]), n_runs=1),
    ]

    with pytest.raises(ValueError, match="0014"):
        pool_participants_scalar(duplicated)


# --------------------------------------------------------------------------------------
# Curves and grids
# --------------------------------------------------------------------------------------


def test_curves_pool_pointwise_across_participants() -> None:
    contributions = [
        Contribution(subject=name, value=np.asarray([scale, 10.0 * scale]), n_runs=1)
        for name, scale in zip("abcde", (1.0, 2.0, 3.0, 4.0, 5.0))
    ]

    pooled = pool_participants_curve(contributions, grid=np.asarray([1.0, 2.0]))

    assert pooled.median == pytest.approx(np.asarray([3.0, 30.0]))
    assert pooled.grid == pytest.approx(np.asarray([1.0, 2.0]))


def test_curves_of_differing_length_are_an_error_naming_the_participant() -> None:
    contributions = [
        Contribution(subject="0014", value=np.asarray([1.0, 2.0]), n_runs=1),
        Contribution(subject="0015", value=np.asarray([1.0]), n_runs=1),
    ]

    with pytest.raises(ValueError, match="0015"):
        pool_participants_curve(contributions, grid=np.asarray([1.0, 2.0]))


def test_grids_are_restricted_to_the_shared_range_rather_than_interpolated() -> None:
    """Interpolating a spectrum onto a foreign grid smears the peaks it exists to show."""
    grids = {
        "0014": np.arange(0.0, 50.0, 0.5),
        "0015": np.arange(2.0, 40.0, 0.5),
    }

    alignment = align_grids(grids)

    assert alignment.grid[0] == pytest.approx(2.0)
    assert alignment.grid[-1] == pytest.approx(39.5)
    assert alignment.limiting_subjects == ("0015",)
    assert grids["0014"][alignment.masks["0014"]] == pytest.approx(alignment.grid)


def test_incompatible_bin_spacing_is_an_error_naming_the_participants() -> None:
    """Two different frequency resolutions cannot be pooled without inventing bins."""
    grids = {
        "0014": np.arange(0.0, 40.0, 0.5),
        "0015": np.arange(0.0, 40.0, 0.25),
    }

    with pytest.raises(ValueError, match="0015"):
        align_grids(grids)


# --------------------------------------------------------------------------------------
# Paired comparisons
# --------------------------------------------------------------------------------------


def test_paired_differences_are_taken_within_participant() -> None:
    """Before and after ICA are two measurements of one participant, not two samples."""
    before = {"0014": 12.0, "0015": 4.0}
    after = {"0014": 10.0, "0015": 3.0}

    differences = paired_differences(before, after)

    assert differences == {"0014": -2.0, "0015": -1.0}


def test_a_participant_measured_on_only_one_side_breaks_the_pairing() -> None:
    with pytest.raises(ValueError, match="0015"):
        paired_differences({"0014": 1.0, "0015": 2.0}, {"0014": 1.0})


# --------------------------------------------------------------------------------------
# Reliability projection
# --------------------------------------------------------------------------------------


def test_reliability_is_projected_to_a_common_trial_count() -> None:
    """Reliability grows with test length, so raw values compare different scales."""
    # A participant with twice the reference trial count, measured at 0.8.
    projected = project_reliability(0.8, observed_trials=80, reference_trials=40)

    # Spearman-Brown stepped down by a factor of one half.
    assert projected == pytest.approx(0.5 * 0.8 / (1 + (0.5 - 1) * 0.8))
    assert projected < 0.8


def test_projection_at_the_reference_count_changes_nothing() -> None:
    assert project_reliability(0.73, observed_trials=40, reference_trials=40) == pytest.approx(0.73)


def test_a_participant_with_more_trials_projects_below_one_with_fewer_at_equal_reliability() -> None:
    """The confound the projection exists to remove, stated as a test."""
    sparse = project_reliability(0.7, observed_trials=30, reference_trials=30)
    prolific = project_reliability(0.7, observed_trials=120, reference_trials=30)

    assert prolific < sparse


def test_a_non_positive_trial_count_is_an_error() -> None:
    with pytest.raises(ValueError, match="trial count"):
        project_reliability(0.5, observed_trials=0, reference_trials=40)


# --------------------------------------------------------------------------------------
# The estimator has to be the one the gate was argued from
# --------------------------------------------------------------------------------------


def test_the_quantile_estimator_matches_the_rule_that_gates_it() -> None:
    """The gate is justified by Weibull plotting positions; the bands must use them.

    Under NumPy's default the p = 0.10 "quantile" of ten participants is an interpolation a
    tenth of the way from the smallest to the second smallest, and exists for as few as
    two. Gating on one convention while estimating with another means the gate does not
    describe what the band actually is.
    """
    contributions = [
        Contribution(subject=f"{index:04d}", value=np.asarray([float(index)]), n_runs=1)
        for index in range(10)
    ]

    pooled = pool_participants_scalar(contributions, gates=BandGates())

    assert pooled.deciles is not None
    expected = np.quantile(np.arange(10.0), (0.10, 0.90), method="weibull")
    assert pooled.deciles == pytest.approx(tuple(expected))


def test_a_missing_value_is_refused_rather_than_erasing_the_pooled_curve() -> None:
    """One gap would blank the cohort median there while the denominator still counted it.

    ``np.median`` over a stack containing a missing value returns a missing value at that
    point, so a single participant with a hole at one frequency erases the cohort curve at
    that frequency -- and the denominator printed beside the figure would be unchanged.
    """
    contributions = [
        Contribution(subject="0014", value=np.asarray([1.0, 2.0, 3.0]), n_runs=1),
        Contribution(subject="0015", value=np.asarray([2.0, np.nan, 4.0]), n_runs=1),
    ]

    with pytest.raises(ValueError, match="0015"):
        pool_participants_curve(contributions, grid=np.asarray([1.0, 2.0, 3.0]))


def test_the_refusal_names_the_participant_and_the_remedy() -> None:
    contributions = [Contribution(subject="0016", value=np.asarray([np.inf]), n_runs=1)]

    with pytest.raises(ValueError, match="shrinking the denominator"):
        pool_participants_scalar(contributions)
