from __future__ import annotations

import numpy as np
import pytest


def test_compute_haufe_pattern_returns_covariance_weight_product() -> None:
    from studies.pain_study.study2.haufe import compute_haufe_pattern

    X_train = np.asarray(
        [
            [1.0, 2.0, 4.0],
            [2.0, 3.0, 6.0],
            [4.0, 5.0, 8.0],
            [7.0, 9.0, 11.0],
        ],
        dtype=float,
    )
    coefficients = np.asarray([0.5, -0.25, 1.5], dtype=float)

    result = compute_haufe_pattern(X_train, coefficients)

    expected_covariance = np.cov(X_train, rowvar=False, ddof=1)
    np.testing.assert_allclose(result.feature_covariance, expected_covariance)
    np.testing.assert_allclose(result.pattern, expected_covariance @ coefficients)
    assert result.n_observations == X_train.shape[0]
    assert result.n_features == X_train.shape[1]


def test_compute_haufe_pattern_uses_correlation_for_standardized_features() -> None:
    from studies.pain_study.study2.haufe import compute_haufe_pattern

    raw = np.asarray(
        [
            [1.0, 1.0, 6.0],
            [2.0, 3.0, 5.0],
            [3.0, 2.0, 4.0],
            [4.0, 6.0, 2.0],
            [5.0, 5.0, 1.0],
        ],
        dtype=float,
    )
    X_train = (raw - np.mean(raw, axis=0)) / np.std(raw, axis=0, ddof=1)
    coefficients = np.asarray([1.0, 0.5, -0.25], dtype=float)

    result = compute_haufe_pattern(X_train, coefficients)

    expected_correlation = np.corrcoef(X_train, rowvar=False)
    np.testing.assert_allclose(result.feature_covariance, expected_correlation)
    np.testing.assert_allclose(result.pattern, expected_correlation @ coefficients)


def test_compute_haufe_pattern_rejects_mismatched_coefficients() -> None:
    from studies.pain_study.study2.haufe import compute_haufe_pattern

    with pytest.raises(ValueError, match="coefficients length must match"):
        compute_haufe_pattern(
            np.ones((4, 3), dtype=float),
            np.ones(2, dtype=float),
        )


def test_compute_haufe_pattern_rejects_non_finite_training_features() -> None:
    from studies.pain_study.study2.haufe import compute_haufe_pattern

    X_train = np.asarray([[1.0, 2.0], [np.nan, 3.0], [4.0, 5.0]], dtype=float)

    with pytest.raises(ValueError, match="contains non-finite"):
        compute_haufe_pattern(X_train, np.ones(2, dtype=float))


def test_compute_haufe_pattern_rejects_constant_training_columns() -> None:
    from studies.pain_study.study2.haufe import compute_haufe_pattern

    X_train = np.asarray(
        [
            [1.0, 2.0],
            [1.0, 3.0],
            [1.0, 4.0],
        ],
        dtype=float,
    )

    with pytest.raises(ValueError, match="zero variance"):
        compute_haufe_pattern(X_train, np.ones(2, dtype=float))
