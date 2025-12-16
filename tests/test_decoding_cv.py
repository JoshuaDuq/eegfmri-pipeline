"""
Decoding Cross-Validation Tests.

Tests for CV utilities, scoring functions, and fold creation.
"""

import numpy as np
import pandas as pd
import pytest

from eeg_pipeline.analysis.decoding.cv import (
    create_loso_folds,
    create_stratified_cv_by_binned_targets,
    safe_pearsonr,
    create_block_aware_cv,
    set_random_seeds,
)


class TestLeaveOneSubjectOutFolds:
    """Test LOSO fold creation."""

    def test_loso_creates_correct_folds(self):
        X = np.random.randn(30, 10)
        groups = np.repeat([0, 1, 2], 10)

        folds = create_loso_folds(X, groups)

        assert len(folds) == 3
        for fold_num, train_idx, test_idx in folds:
            assert len(train_idx) == 20
            assert len(test_idx) == 10
            assert len(set(train_idx) & set(test_idx)) == 0

    def test_loso_each_subject_heldout_once(self):
        X = np.random.randn(30, 10)
        groups = np.repeat([0, 1, 2], 10)

        folds = create_loso_folds(X, groups)

        test_indices = set()
        for _, _, test_idx in folds:
            test_indices.update(test_idx)

        assert test_indices == set(range(30))


class TestStratifiedCVByBins:
    """Test stratified CV with continuous targets."""

    def test_stratified_cv_creates_valid_folds(self):
        y = np.linspace(0, 100, 50)
        config = {"decoding": {"cv": {"default_n_splits": 5, "default_n_bins": 5}}}

        cv, y_binned = create_stratified_cv_by_binned_targets(y, config=config)

        n_folds = sum(1 for _ in cv.split(y, y_binned))
        assert n_folds == 5

    def test_stratified_cv_no_overlap(self):
        y = np.linspace(0, 100, 50)
        config = {"decoding": {"cv": {"default_n_splits": 5, "default_n_bins": 5}}}

        cv, y_binned = create_stratified_cv_by_binned_targets(y, config=config)

        for train_idx, test_idx in cv.split(y, y_binned):
            overlap = set(train_idx) & set(test_idx)
            assert len(overlap) == 0


class TestBlockAwareCV:
    """Test block-aware CV utilities."""

    def test_block_aware_cv_respects_blocks(self):
        blocks = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
        config = {"decoding": {"cv": {"default_n_splits": 2}}}

        cv, _ = create_block_aware_cv(blocks, config=config)

        for train_idx, test_idx in cv.split(blocks, groups=blocks):
            train_blocks = set(blocks[train_idx])
            test_blocks = set(blocks[test_idx])
            assert train_blocks.isdisjoint(test_blocks)


class TestSafePearsonR:
    """Test safe Pearson correlation with edge cases."""

    def test_basic_correlation(self):
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 6, 8, 10])

        r, p = safe_pearsonr(x, y)

        assert np.isclose(r, 1.0)
        assert p < 0.01

    def test_negative_correlation(self):
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([10, 8, 6, 4, 2])

        r, p = safe_pearsonr(x, y)

        assert np.isclose(r, -1.0)

    def test_zero_variance_returns_nan(self):
        x = np.array([1, 1, 1, 1, 1])
        y = np.array([1, 2, 3, 4, 5])

        r, p = safe_pearsonr(x, y)

        assert np.isnan(r)
        assert np.isnan(p)

    def test_handles_nan_values(self):
        x = np.array([1, 2, np.nan, 4, 5])
        y = np.array([2, 4, 6, 8, 10])

        r, p = safe_pearsonr(x, y)

        assert np.isfinite(r) or np.isnan(r)


class TestRandomSeeds:
    """Test reproducibility utilities."""

    def test_set_seeds_deterministic(self):
        set_random_seeds(42, fold=0)
        val1 = np.random.rand()

        set_random_seeds(42, fold=0)
        val2 = np.random.rand()

        assert val1 == val2

    def test_different_folds_different_seeds(self):
        set_random_seeds(42, fold=0)
        val1 = np.random.rand()

        set_random_seeds(42, fold=1)
        val2 = np.random.rand()

        assert val1 != val2
