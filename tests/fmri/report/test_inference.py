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


# --- random field theory --------------------------------------------------


def test_the_rft_threshold_rises_with_the_search_volume() -> None:
    small = inference.rft_voxel_threshold(n_resels=100.0, alpha=0.05, two_sided=True)
    large = inference.rft_voxel_threshold(n_resels=10_000.0, alpha=0.05, two_sided=True)
    assert small < large


def test_the_one_sided_rft_threshold_is_less_strict() -> None:
    two = inference.rft_voxel_threshold(n_resels=5_000.0, alpha=0.05, two_sided=True)
    one = inference.rft_voxel_threshold(n_resels=5_000.0, alpha=0.05, two_sided=False)
    assert one < two


def test_the_rft_threshold_beats_bonferroni_only_on_a_smooth_field() -> None:
    # Which of the two valid bounds is tighter is a property of the smoothness, not a
    # ranking that holds in general -- and assuming the ranking is how a report ends up
    # quoting the looser one as its correction.
    #
    # Heavily smoothed: 32,768 voxels over 318 resels, about a hundred voxels each.
    # Bonferroni charges for a family that is not there and RFT wins.
    assert inference.rft_voxel_threshold(
        n_resels=318.0, alpha=0.05, two_sided=False
    ) < inference.bonferroni_threshold(n=32_768, alpha=0.05, two_sided=False)

    # This study's own regime: 50,626 voxels over 5,037 resels, about ten voxels each.
    # The Euler-characteristic approximation overshoots and Bonferroni is tighter.
    assert inference.rft_voxel_threshold(
        n_resels=5_037.0, alpha=0.05, two_sided=True
    ) > inference.bonferroni_threshold(n=50_626, alpha=0.05, two_sided=True)


def test_the_rft_threshold_solves_the_euler_characteristic_equation() -> None:
    # The defining property: at the returned height, the expected Euler
    # characteristic of the excursion set equals alpha. Computed here from the
    # Worsley (1996) 3D density directly, so the test does not restate the solver.
    resels, alpha = 5_037.0, 0.05
    height = inference.rft_voxel_threshold(
        n_resels=resels, alpha=alpha, two_sided=False
    )
    density = (
        (4.0 * np.log(2.0)) ** 1.5
        / (2.0 * np.pi) ** 2
        * (height**2 - 1.0)
        * np.exp(-(height**2) / 2.0)
    )
    assert resels * density == pytest.approx(alpha, rel=1e-6)


def test_the_rft_threshold_controls_the_familywise_error_rate() -> None:
    # The property the height is sold on, measured rather than assumed: over
    # realisations of a smooth null field, the share in which any voxel clears the
    # height must land near alpha. A formula error shows up here as a rate off by
    # orders of magnitude, so the bound is deliberately loose -- RFT is an
    # approximation and its exactness is not what is being asserted.
    from scipy import ndimage

    rng = np.random.default_rng(7)
    sigma, pad, core, trials = 2.0, 12, 32, 300

    def realisation() -> np.ndarray:
        # Smoothed on a padded grid and cropped back. ``gaussian_filter`` reflects at
        # the boundary, which leaves edge voxels with more variance than the interior;
        # normalising the whole volume by one standard deviation then pushes the
        # maximum onto that rim. Uncropped, this test measures an edge artefact and
        # reads a 56% familywise error rate off a correct height.
        field = ndimage.gaussian_filter(
            rng.standard_normal((core + 2 * pad,) * 3), sigma=sigma
        )
        field = field[pad : pad + core, pad : pad + core, pad : pad + core]
        return field / field.std()

    def fwhm_of(field: np.ndarray) -> float:
        variance = field.var()
        widths = []
        for axis in range(3):
            ratio = np.diff(field, axis=axis).var() / (2.0 * variance)
            widths.append(
                np.sqrt(-1.0 / (4.0 * np.log(1.0 - ratio))) * np.sqrt(8.0 * np.log(2.0))
            )
        return float(np.mean(widths))

    fields = [realisation() for _ in range(trials)]
    resels = core**3 / float(np.mean([fwhm_of(field) for field in fields]) ** 3)
    height = inference.rft_voxel_threshold(
        n_resels=resels, alpha=0.05, two_sided=False
    )

    rate = np.mean([field.max() > height for field in fields])
    # Conservative but not by an order of magnitude: measured at 0.031 against a
    # nominal 0.05, which is the mild overshoot the approximation is known for.
    assert 0.005 <= rate <= 0.05


def test_the_rft_threshold_validates_its_inputs() -> None:
    with pytest.raises(ValueError, match="resel"):
        inference.rft_voxel_threshold(n_resels=0.0, alpha=0.05, two_sided=True)
    with pytest.raises(ValueError, match="alpha"):
        inference.rft_voxel_threshold(n_resels=100.0, alpha=0.0, two_sided=True)


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


def test_the_summary_carries_a_random_field_height_when_given_a_search_volume() -> None:
    values = np.random.default_rng(10).standard_normal(50_000)
    summary = _summary(values, n_resels=5_037.0)
    assert summary.n_resels == pytest.approx(5_037.0)
    assert summary.rft == pytest.approx(
        inference.rft_voxel_threshold(n_resels=5_037.0, alpha=0.05, two_sided=True)
    )
    assert summary.rft_survivors == 0


def test_the_summary_has_no_random_field_height_without_a_search_volume() -> None:
    # Smoothness cannot always be estimated, and a report that loses its whole
    # threshold table over a missing resel count is worse than one without the row.
    summary = _summary(np.random.default_rng(11).standard_normal(20_000))
    assert summary.n_resels is None
    assert summary.rft is None
    assert summary.rft_survivors == 0


def test_the_summary_survives_a_search_volume_too_small_for_random_field_theory() -> None:
    summary = _summary(
        np.random.default_rng(12).standard_normal(20_000), n_resels=0.5
    )
    assert summary.rft is None
    assert summary.bonferroni > 0


def test_the_summary_carries_the_empirical_null() -> None:
    values = -0.6 + 1.5 * np.random.default_rng(9).standard_normal(50_000)
    summary = _summary(values)
    assert summary.null.scale == pytest.approx(1.5, abs=0.05)


def test_the_summary_tolerates_a_null_it_cannot_fit() -> None:
    # An all-constant map has no spread to fit. The corrected thresholds are still
    # well defined, so the summary must not be lost with the null.
    summary = _summary(np.concatenate([np.zeros(999), np.array([9.0])]))
    assert summary.null is None
    assert summary.calibration is None
    assert summary.bonferroni > 0


# --- the map's own null, applied ------------------------------------------


def test_expected_false_positives_under_a_wide_null_exceed_the_theoretical_count() -> None:
    # The single most consequential number in the panel. A null of width 1.5 puts far
    # more voxels past 2.3 than N(0, 1) does, and it is the larger count the observed
    # survivors have to be judged against.
    null = inference.EmpiricalNull(centre=0.0, scale=1.5, n=50_000)
    theoretical = inference.expected_false_positives(
        n=50_000, threshold=2.3, two_sided=True
    )
    empirical = inference.expected_false_positives_under(
        n=50_000, threshold=2.3, null=null, two_sided=True
    )
    assert empirical > 4 * theoretical


def test_expected_false_positives_under_a_standard_null_match_the_theoretical_count() -> None:
    null = inference.EmpiricalNull(centre=0.0, scale=1.0, n=1_000)
    assert inference.expected_false_positives_under(
        n=1_000, threshold=2.3, null=null, two_sided=True
    ) == pytest.approx(
        inference.expected_false_positives(n=1_000, threshold=2.3, two_sided=True)
    )


def test_expected_false_positives_under_a_shifted_null_count_both_tails() -> None:
    # A shifted null loads one tail and empties the other; the total is what the
    # observed survivor count is compared against, so both have to be in it.
    null = inference.EmpiricalNull(centre=-0.6, scale=1.5, n=50_000)
    total = inference.expected_false_positives_under(
        n=50_000, threshold=2.3, null=null, two_sided=True
    )
    upper_only = 50_000 * stats.norm.sf((2.3 + 0.6) / 1.5)
    assert total > upper_only


def test_a_shifted_null_makes_a_symmetric_threshold_asymmetric_in_evidence() -> None:
    # The finding this replaced "N x the null's width" to express: on a null centred
    # below zero, |z| > 2.3 is a far weaker claim downward than upward.
    values = -0.6 + 1.5 * np.random.default_rng(20).standard_normal(200_000)
    calibration = _summary(values).calibration
    assert calibration is not None
    assert calibration.lower_tail_p > 3 * calibration.upper_tail_p


def test_a_centred_null_makes_the_two_tails_agree() -> None:
    values = 1.5 * np.random.default_rng(21).standard_normal(200_000)
    calibration = _summary(values).calibration
    assert calibration.lower_tail_p == pytest.approx(calibration.upper_tail_p, rel=0.05)


def test_one_sided_inference_reports_no_lower_tail() -> None:
    # The lower tail was never examined, so a probability for it would describe a test
    # that did not run.
    values = 1.5 * np.random.default_rng(22).standard_normal(50_000)
    calibration = _summary(values, two_sided=False).calibration
    assert calibration.upper_tail_p is not None
    assert calibration.lower_tail_p is None
    assert calibration.fdr_lower is None


def test_the_empirical_null_fdr_is_stricter_than_the_theoretical_one_on_a_wide_null() -> None:
    # Efron's correction, applied. Under an over-dispersed null the theoretical-null
    # FDR rejects voxels that are ordinary noise for this map; the empirical-null FDR
    # is what does not.
    values = 1.5 * np.random.default_rng(23).standard_normal(100_000)
    summary = _summary(values)
    assert summary.fdr_survivors > summary.calibration.fdr_survivors


def test_the_empirical_null_fdr_rejects_essentially_nothing_under_pure_noise() -> None:
    # Whatever the null's width, a map with no signal has no discoveries in it. That is
    # exactly the guarantee the theoretical-null FDR loses when the null is misfitted.
    values = -0.6 + 1.5 * np.random.default_rng(24).standard_normal(100_000)
    calibration = _summary(values).calibration
    assert calibration.fdr_survivors <= 0.05 * 100_000 * 0.01


def test_the_empirical_null_fdr_still_finds_a_real_signal() -> None:
    # Strictness must not become blindness: a genuine tail well past the fitted null
    # has to survive, or the correction would only ever remove findings.
    rng = np.random.default_rng(25)
    values = np.concatenate(
        [1.5 * rng.standard_normal(100_000), 12.0 + rng.standard_normal(500)]
    )
    calibration = _summary(values).calibration
    assert calibration.fdr_survivors > 400


def test_the_empirical_null_fdr_region_is_asymmetric_on_a_shifted_null() -> None:
    # It is a rejection region in raw z, not a height: the two bounds sit at different
    # distances from zero, which is precisely what a single "|z| >" figure cannot say.
    rng = np.random.default_rng(26)
    values = np.concatenate(
        [-0.6 + 1.5 * rng.standard_normal(100_000), 12.0 + rng.standard_normal(500)]
    )
    calibration = _summary(values).calibration
    assert calibration.fdr_upper is not None and calibration.fdr_lower is not None
    assert abs(calibration.fdr_lower) > abs(calibration.fdr_upper)


# --- an unthresholded report still needs the corrected heights ------------


def test_the_summary_accepts_no_applied_threshold() -> None:
    # threshold_mode: none. The corrected thresholds are what the map would have been
    # cut at, and an unthresholded report is the one whose reader most needs them.
    values = np.random.default_rng(11).standard_normal(20_000)
    summary = _summary(values, applied_threshold=None)
    assert summary.applied is None
    assert summary.applied_survivors is None
    assert summary.expected_null_survivors is None
    assert summary.bonferroni > 0
    assert summary.null is not None
    # The empirical-null FDR does not depend on an applied height, and an
    # unthresholded report is the one whose reader most needs it.
    assert summary.calibration is not None
    assert summary.calibration.expected_survivors is None
    assert summary.calibration.upper_tail_p is None
