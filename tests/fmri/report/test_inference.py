from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from fmri_pipeline.analysis.report import inference


# --- empirical null -------------------------------------------------------


def test_the_empirical_null_recovers_a_standard_normal() -> None:
    values = np.random.default_rng(0).standard_normal(200_000)
    fitted = inference.empirical_null(values)
    assert fitted.centre == pytest.approx(0.0, abs=0.02)
    assert fitted.scale == pytest.approx(1.0, abs=0.02)
    assert fitted.n == 200_000


def test_the_empirical_null_recovers_a_shifted_and_widened_null() -> None:
    values = -0.6 + 1.5 * np.random.default_rng(1).standard_normal(200_000)
    fitted = inference.empirical_null(values)
    assert fitted.centre == pytest.approx(-0.6, abs=0.03)
    assert fitted.scale == pytest.approx(1.5, abs=0.03)


def test_the_empirical_null_ignores_a_realistic_signal_tail() -> None:
    # The point of a robust estimator: a real activation tail must not be read as
    # evidence that the null itself is wide. Five percent of voxels strongly active is
    # already a vigorous contrast, and the estimate has to stay on the null.
    rng = np.random.default_rng(2)
    values = np.concatenate([rng.standard_normal(190_000), 6.0 + rng.standard_normal(10_000)])
    assert inference.empirical_null(values).scale == pytest.approx(1.0, abs=0.1)


def test_the_empirical_null_errs_wide_rather_than_narrow_under_heavy_contamination() -> None:
    # Beyond its breakdown the estimator does move, and the direction matters: an
    # over-wide null makes the applied threshold look weaker than it is, so a reader is
    # made more cautious rather than less. Narrowing would be the dangerous failure.
    rng = np.random.default_rng(3)
    values = np.concatenate([rng.standard_normal(150_000), 6.0 + rng.standard_normal(50_000)])
    assert inference.empirical_null(values).scale > 1.0


def test_the_empirical_null_refuses_a_degenerate_input() -> None:
    with pytest.raises(ValueError, match="spread"):
        inference.empirical_null(np.zeros(100))


def test_the_empirical_null_requires_finite_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        inference.empirical_null(np.array([np.nan, np.inf]))


# --- p-values -------------------------------------------------------------


def test_two_sided_p_values_double_the_upper_tail() -> None:
    values = np.array([-1.96, 0.0, 1.96])
    p = inference.p_values(values, two_sided=True)
    assert p == pytest.approx([0.05, 1.0, 0.05], abs=1e-3)


def test_one_sided_p_values_read_only_the_upper_tail() -> None:
    # A one-sided test never examined the negative half, so a large negative z is
    # the least significant thing on the map, not the most.
    p = inference.p_values(np.array([-1.64, 0.0, 1.64]), two_sided=False)
    assert p == pytest.approx([0.95, 0.5, 0.05], abs=1e-3)


# --- Benjamini-Hochberg ---------------------------------------------------


def test_the_fdr_threshold_matches_a_hand_computed_benjamini_hochberg() -> None:
    # p sorted: .001 .008 .039 .041 .042; k/m*q at q=.05 is .01 .02 .03 .04 .05.
    # The largest k passing is 4 (.041 <= .04 is false; .042 <= .05 is true at k=5),
    # so BH steps up to k=5 and rejects everything at or below p = .042.
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042])
    assert inference.fdr_p_cutoff(p, q=0.05) == pytest.approx(0.042)


def test_the_fdr_cutoff_rejects_nothing_when_no_p_value_passes() -> None:
    assert inference.fdr_p_cutoff(np.array([0.6, 0.7, 0.8]), q=0.05) is None


def test_the_fdr_threshold_is_stricter_than_the_uncorrected_one() -> None:
    rng = np.random.default_rng(3)
    values = np.concatenate([rng.standard_normal(20_000), 5.0 + rng.standard_normal(500)])
    threshold = inference.fdr_threshold(values, q=0.05, two_sided=True)
    assert threshold is not None
    assert threshold > 1.96


def test_the_fdr_threshold_is_none_when_nothing_survives() -> None:
    values = np.random.default_rng(4).standard_normal(5_000)
    assert inference.fdr_threshold(values, q=1e-6, two_sided=True) is None


def test_the_fdr_threshold_controls_the_false_discovery_proportion() -> None:
    # Under a pure null, BH at q must reject essentially nothing; any rejection is a
    # false discovery, so the realised proportion is the whole of it.
    values = np.random.default_rng(5).standard_normal(100_000)
    threshold = inference.fdr_threshold(values, q=0.05, two_sided=True)
    survivors = 0 if threshold is None else int((np.abs(values) > threshold).sum())
    assert survivors <= 0.05 * 100_000


# --- Bonferroni and expected false positives ------------------------------


def test_the_bonferroni_threshold_splits_alpha_across_the_voxels() -> None:
    threshold = inference.bonferroni_threshold(n=50_000, alpha=0.05, two_sided=True)
    assert threshold == pytest.approx(stats.norm.isf(0.05 / (2 * 50_000)))


def test_the_one_sided_bonferroni_threshold_is_less_strict() -> None:
    two = inference.bonferroni_threshold(n=1_000, alpha=0.05, two_sided=True)
    one = inference.bonferroni_threshold(n=1_000, alpha=0.05, two_sided=False)
    assert one < two


def test_expected_false_positives_counts_the_null_tail() -> None:
    expected = inference.expected_false_positives(n=50_000, threshold=2.3, two_sided=True)
    assert expected == pytest.approx(50_000 * 2 * stats.norm.sf(2.3))


def test_expected_false_positives_needs_a_positive_threshold() -> None:
    with pytest.raises(ValueError, match="> 0"):
        inference.expected_false_positives(n=10, threshold=0.0, two_sided=True)


# --- the assembled summary ------------------------------------------------


def _summary(values: np.ndarray, **kwargs):
    params = dict(applied_threshold=2.3, fdr_q=0.05, alpha=0.05, two_sided=True)
    params.update(kwargs)
    return inference.threshold_context(values, **params)


def test_the_summary_orders_the_thresholds_from_permissive_to_strict() -> None:
    rng = np.random.default_rng(6)
    values = np.concatenate([rng.standard_normal(40_000), 5.0 + rng.standard_normal(800)])
    summary = _summary(values)
    assert summary.applied < summary.fdr < summary.bonferroni


def test_the_summary_counts_survivors_at_each_threshold() -> None:
    rng = np.random.default_rng(7)
    values = np.concatenate([rng.standard_normal(40_000), 5.0 + rng.standard_normal(800)])
    summary = _summary(values)
    assert summary.applied_survivors > summary.fdr_survivors > summary.bonferroni_survivors


def test_the_summary_survives_an_fdr_that_rejects_nothing() -> None:
    # A contrast with no signal is a result, not a fault: the panel still has to draw.
    summary = _summary(np.random.default_rng(8).standard_normal(20_000))
    assert summary.fdr is None
    assert summary.fdr_survivors == 0
    assert summary.bonferroni > 0


def test_the_summary_carries_the_empirical_null() -> None:
    values = -0.6 + 1.5 * np.random.default_rng(9).standard_normal(50_000)
    summary = _summary(values)
    assert summary.null.scale == pytest.approx(1.5, abs=0.05)


def test_the_summary_reports_the_applied_threshold_in_empirical_null_units() -> None:
    # The whole point: |z| > 2.3 against a null of width 1.5 is not 2.3 sigma.
    values = 1.5 * np.random.default_rng(10).standard_normal(50_000)
    summary = _summary(values)
    assert summary.applied_in_null_units == pytest.approx(2.3 / 1.5, abs=0.05)


def test_the_summary_tolerates_a_null_it_cannot_fit() -> None:
    # An all-constant map has no spread to fit. The corrected thresholds are still
    # well defined, so the summary must not be lost with the null.
    summary = _summary(np.concatenate([np.zeros(999), np.array([9.0])]))
    assert summary.null is None
    assert summary.applied_in_null_units is None
    assert summary.bonferroni > 0
