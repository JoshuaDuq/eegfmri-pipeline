"""Haufe-transformed sensor-pattern helpers for Study 2."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HaufePattern:
    """Sensor-feature activation pattern from a linear backward model."""

    pattern: np.ndarray
    feature_covariance: np.ndarray
    n_observations: int
    n_features: int


def compute_haufe_pattern(
    X_train: np.ndarray,
    coefficients: np.ndarray,
) -> HaufePattern:
    """Compute the Haufe activation pattern from training features and weights.

    The Study 2 protocol interprets linear sensor patterns as
    ``cov(X_train) @ coefficients``. When Study 1 features are standardized in
    the training fold, this covariance is the training-fold correlation matrix.
    """
    feature_matrix = np.asarray(X_train, dtype=float)
    weight_vector = np.asarray(coefficients, dtype=float)
    _validate_inputs(feature_matrix, weight_vector)

    centered = feature_matrix - np.mean(feature_matrix, axis=0, keepdims=True)
    feature_covariance = centered.T @ centered / float(feature_matrix.shape[0] - 1)
    pattern = feature_covariance @ weight_vector
    return HaufePattern(
        pattern=pattern,
        feature_covariance=feature_covariance,
        n_observations=int(feature_matrix.shape[0]),
        n_features=int(feature_matrix.shape[1]),
    )


def _validate_inputs(feature_matrix: np.ndarray, weight_vector: np.ndarray) -> None:
    if feature_matrix.ndim != 2:
        raise ValueError(
            f"Study 2 Haufe X_train must be 2D, got shape {feature_matrix.shape}."
        )
    if weight_vector.ndim != 1:
        raise ValueError(
            f"Study 2 Haufe coefficients must be 1D, got shape {weight_vector.shape}."
        )
    if feature_matrix.shape[0] < 2:
        raise ValueError("Study 2 Haufe pattern requires at least two training observations.")
    if feature_matrix.shape[1] != weight_vector.shape[0]:
        raise ValueError(
            "Study 2 Haufe coefficients length must match X_train columns: "
            f"{weight_vector.shape[0]} != {feature_matrix.shape[1]}."
        )
    if not np.all(np.isfinite(feature_matrix)):
        raise ValueError("Study 2 Haufe X_train contains non-finite values.")
    if not np.all(np.isfinite(weight_vector)):
        raise ValueError("Study 2 Haufe coefficients contain non-finite values.")

    column_variance = np.var(feature_matrix, axis=0, ddof=1)
    if np.any(column_variance <= 0.0):
        raise ValueError("Study 2 Haufe X_train contains a zero variance feature column.")


__all__ = ["HaufePattern", "compute_haufe_pattern"]
