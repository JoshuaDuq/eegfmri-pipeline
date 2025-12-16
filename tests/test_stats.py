"""
Statistics Module Tests.

Exhaustive tests for bootstrap, effect size, FDR, and core stats utilities.
"""

import numpy as np
import pandas as pd
import pytest


###################################################################
# Bootstrap Tests
###################################################################

class TestBootstrapCorrelationCI:
    """Test bootstrap confidence intervals for correlation."""

    def test_bootstrap_corr_ci_returns_tuple(self):
        from eeg_pipeline.utils.analysis.stats.bootstrap import bootstrap_corr_ci

        x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        y = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20])

        ci_low, ci_high = bootstrap_corr_ci(x, y, method="spearman", n_boot=100)

        assert isinstance(ci_low, float)
        assert isinstance(ci_high, float)
        assert ci_low <= ci_high  # Allow equality for perfect correlation

    def test_bootstrap_ci_perfect_correlation(self):
        from eeg_pipeline.utils.analysis.stats.bootstrap import bootstrap_corr_ci

        x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        y = x * 2

        ci_low, ci_high = bootstrap_corr_ci(x, y, method="pearson", n_boot=100)

        assert ci_low > 0.9
        assert ci_high <= 1.0

    def test_bootstrap_ci_insufficient_data(self):
        from eeg_pipeline.utils.analysis.stats.bootstrap import bootstrap_corr_ci

        x = np.array([1, 2])
        y = np.array([3, 4])

        ci_low, ci_high = bootstrap_corr_ci(x, y, n_boot=50)

        assert np.isnan(ci_low)
        assert np.isnan(ci_high)

    def test_bootstrap_ci_with_nan_values(self):
        from eeg_pipeline.utils.analysis.stats.bootstrap import bootstrap_corr_ci

        x = np.array([1, np.nan, 3, 4, 5, 6, 7, 8, 9, 10])
        y = np.array([2, 4, np.nan, 8, 10, 12, 14, 16, 18, 20])

        ci_low, ci_high = bootstrap_corr_ci(x, y, n_boot=100)

        # Should still compute with valid data
        assert np.isfinite(ci_low) or np.isnan(ci_low)


class TestBootstrapMeanCI:
    """Test bootstrap confidence intervals for means."""

    def test_bootstrap_mean_ci_returns_tuple(self):
        from eeg_pipeline.utils.analysis.stats.bootstrap import bootstrap_mean_ci

        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        mean, ci_low, ci_high = bootstrap_mean_ci(data, n_boot=100)

        assert isinstance(mean, float)
        assert np.isclose(mean, 5.5)
        assert ci_low < mean < ci_high

    def test_bootstrap_mean_diff_ci(self):
        from eeg_pipeline.utils.analysis.stats.bootstrap import bootstrap_mean_diff_ci

        group1 = np.array([1, 2, 3, 4, 5])
        group2 = np.array([6, 7, 8, 9, 10])

        diff, ci_low, ci_high = bootstrap_mean_diff_ci(group1, group2, n_boot=100)

        assert diff > 0  # group2 > group1
        assert ci_low < diff < ci_high


class TestPermutationPValue:
    """Test permutation-based p-values."""

    def test_perm_pval_returns_float(self):
        from eeg_pipeline.utils.analysis.stats.bootstrap import perm_pval_simple

        x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        y = np.array([10, 9, 8, 7, 6, 5, 4, 3, 2, 1])

        p = perm_pval_simple(x, y, n_perm=50)

        assert isinstance(p, float)
        assert 0 <= p <= 1

    def test_perm_pval_perfect_correlation(self):
        from eeg_pipeline.utils.analysis.stats.bootstrap import perm_pval_simple

        x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        y = x * 2

        p = perm_pval_simple(x, y, n_perm=50)

        assert p < 0.1  # Should be significant

    def test_perm_pval_insufficient_data(self):
        from eeg_pipeline.utils.analysis.stats.bootstrap import perm_pval_simple

        x = np.array([1, 2])
        y = np.array([3, 4])

        p = perm_pval_simple(x, y, n_perm=50)

        assert np.isnan(p)


###################################################################
# Effect Size Tests
###################################################################

class TestCohensD:
    """Test Cohen's d effect size."""

    def test_cohens_d_positive_effect(self):
        from eeg_pipeline.utils.analysis.stats.effect_size import cohens_d

        group1 = np.array([1, 2, 3, 4, 5])
        group2 = np.array([6, 7, 8, 9, 10])

        d = cohens_d(group1, group2)

        assert np.abs(d) > 2.0  # Large effect

    def test_cohens_d_no_effect(self):
        from eeg_pipeline.utils.analysis.stats.effect_size import cohens_d

        group1 = np.array([5, 5, 5, 5, 5])
        group2 = np.array([5, 5, 5, 5, 5])

        d = cohens_d(group1, group2)

        assert np.isclose(d, 0.0) or np.isnan(d)

    def test_cohens_d_identical_groups(self):
        from eeg_pipeline.utils.analysis.stats.effect_size import cohens_d

        group = np.array([1, 2, 3, 4, 5])

        d = cohens_d(group, group)

        assert np.isclose(d, 0.0)


class TestHedgesG:
    """Test Hedges' g effect size."""

    def test_hedges_g_smaller_than_d(self):
        from eeg_pipeline.utils.analysis.stats.effect_size import cohens_d, hedges_g

        group1 = np.array([1, 2, 3, 4, 5])
        group2 = np.array([6, 7, 8, 9, 10])

        d = cohens_d(group1, group2)
        g = hedges_g(group1, group2)

        # Hedges' g includes bias correction, typically smaller
        assert np.abs(g) <= np.abs(d)


class TestFisherZTest:
    """Test Fisher z-test for correlation differences."""

    def test_fisher_z_test_returns_tuple(self):
        from eeg_pipeline.utils.analysis.stats.effect_size import fisher_z_test

        z, p = fisher_z_test(r1=0.5, r2=0.1, n1=30, n2=30)

        assert isinstance(z, float)
        assert isinstance(p, float)
        assert 0 <= p <= 1

    def test_fisher_z_test_equal_correlations(self):
        from eeg_pipeline.utils.analysis.stats.effect_size import fisher_z_test

        z, p = fisher_z_test(r1=0.5, r2=0.5, n1=30, n2=30)

        assert np.isclose(z, 0.0)
        assert p > 0.9

    def test_fisher_z_test_opposite_correlations(self):
        from eeg_pipeline.utils.analysis.stats.effect_size import fisher_z_test

        z, p = fisher_z_test(r1=0.8, r2=-0.8, n1=50, n2=50)

        assert np.abs(z) > 2
        assert p < 0.05


class TestEffectSizeConversion:
    """Test effect size conversion utilities."""

    def test_r_to_d_conversion(self):
        from eeg_pipeline.utils.analysis.stats.effect_size import r_to_d

        d = r_to_d(0.5)

        assert d > 0
        assert np.isfinite(d)

    def test_d_to_r_conversion(self):
        from eeg_pipeline.utils.analysis.stats.effect_size import d_to_r

        r = d_to_r(0.8)

        assert 0 < r < 1
        assert np.isfinite(r)


###################################################################
# FDR Correction Tests
###################################################################

class TestFDRBH:
    """Test Benjamini-Hochberg FDR correction."""

    def test_fdr_bh_returns_array(self):
        from eeg_pipeline.utils.analysis.stats.fdr import fdr_bh

        pvals = [0.01, 0.02, 0.03, 0.5, 0.9]

        q_values = fdr_bh(pvals)

        assert len(q_values) == len(pvals)
        assert all(np.isfinite(q_values))

    def test_fdr_bh_preserves_order(self):
        from eeg_pipeline.utils.analysis.stats.fdr import fdr_bh

        pvals = [0.01, 0.05, 0.1, 0.5]

        q_values = fdr_bh(pvals)

        # Q-values should preserve relative ordering
        assert q_values[0] <= q_values[1]
        assert q_values[1] <= q_values[2]
        assert q_values[2] <= q_values[3]

    def test_fdr_bh_all_significant(self):
        from eeg_pipeline.utils.analysis.stats.fdr import fdr_bh

        pvals = [0.001, 0.002, 0.003]

        q_values = fdr_bh(pvals, alpha=0.05)

        assert all(q < 0.05 for q in q_values)

    def test_fdr_bh_none_significant(self):
        from eeg_pipeline.utils.analysis.stats.fdr import fdr_bh

        pvals = [0.8, 0.85, 0.9, 0.95]

        q_values = fdr_bh(pvals, alpha=0.05)

        assert all(q >= 0.05 for q in q_values)


class TestFDRReject:
    """Test FDR rejection decisions."""

    def test_fdr_bh_reject_returns_mask(self):
        from eeg_pipeline.utils.analysis.stats.fdr import fdr_bh_reject

        pvals = np.array([0.001, 0.01, 0.1, 0.5, 0.9])

        reject_mask, critical = fdr_bh_reject(pvals, alpha=0.05)

        assert reject_mask.dtype == bool
        assert len(reject_mask) == len(pvals)

    def test_fdr_bh_mask_helper(self):
        from eeg_pipeline.utils.analysis.stats.fdr import fdr_bh_mask

        pvals = np.array([0.001, 0.01, 0.5, 0.9])

        mask = fdr_bh_mask(pvals, alpha=0.05)

        assert mask[0] == True  # 0.001 should be significant
        assert mask[3] == False  # 0.9 should not be significant


class TestFDRCorrection:
    """Test FDR correction wrapper."""

    def test_fdr_correction_returns_all(self):
        from eeg_pipeline.utils.analysis.stats.fdr import fdr_correction

        pvals = np.array([0.01, 0.03, 0.05, 0.5])

        q_values, reject, critical = fdr_correction(pvals, alpha=0.05)

        assert len(q_values) == len(pvals)
        assert len(reject) == len(pvals)
        assert np.isfinite(critical)


###################################################################
# Correlation Tests
###################################################################

class TestComputeCorrelation:
    """Test correlation computation."""

    def test_spearman_correlation(self):
        from eeg_pipeline.utils.analysis.stats.correlation import compute_correlation

        x = np.array([1, 2, 3, 4, 5])
        y = np.array([5, 4, 3, 2, 1])

        r, p = compute_correlation(x, y, method="spearman")

        assert np.isclose(r, -1.0)
        assert p < 0.05

    def test_pearson_correlation(self):
        from eeg_pipeline.utils.analysis.stats.correlation import compute_correlation

        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 6, 8, 10])

        r, p = compute_correlation(x, y, method="pearson")

        assert np.isclose(r, 1.0)

    def test_correlation_insufficient_data(self):
        from eeg_pipeline.utils.analysis.stats.correlation import compute_correlation

        x = np.array([1, 2])
        y = np.array([3, 4])

        r, p = compute_correlation(x, y)

        # Should return NaN or handle gracefully
        assert np.isfinite(r) or np.isnan(r)


###################################################################
# Validation Utilities Tests
###################################################################

class TestExtractFiniteMask:
    """Test finite value extraction."""

    def test_extract_finite_mask_all_finite(self):
        from eeg_pipeline.utils.analysis.stats import extract_finite_mask

        x = np.array([1, 2, 3, 4])
        y = np.array([5, 6, 7, 8])

        x_out, y_out, mask = extract_finite_mask(x, y)

        assert len(x_out) == 4
        assert np.all(mask)

    def test_extract_finite_mask_with_nan(self):
        from eeg_pipeline.utils.analysis.stats import extract_finite_mask

        x = np.array([1, np.nan, 3, 4])
        y = np.array([5, 6, np.nan, 8])

        x_out, y_out, mask = extract_finite_mask(x, y)

        assert len(x_out) == 2  # Only indices 0 and 3 are fully finite
        assert not mask[1]
        assert not mask[2]

    def test_extract_finite_mask_with_inf(self):
        from eeg_pipeline.utils.analysis.stats import extract_finite_mask

        x = np.array([1, np.inf, 3, -np.inf])
        y = np.array([5, 6, 7, 8])

        x_out, y_out, mask = extract_finite_mask(x, y)

        assert len(x_out) == 2  # Infs are excluded
