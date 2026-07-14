"""Deterministic participant-level bootstrap statistics."""

from __future__ import annotations

import numpy as np

BOOTSTRAP_BATCH_SIZE = 256


def paired_participant_bootstrap(
    values: np.ndarray,
    *,
    iterations: int,
    confidence_level: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return cohort medians and paired participant-bootstrap intervals."""
    value_matrix = np.asarray(values, dtype=float)
    if value_matrix.ndim != 2 or value_matrix.shape[0] < 1:
        raise ValueError("Participant bootstrap requires a non-empty two-dimensional matrix.")
    if not np.isfinite(value_matrix).all():
        raise ValueError("Participant bootstrap values must be finite.")
    if iterations < 1:
        raise ValueError("Participant bootstrap iterations must be positive.")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("Participant bootstrap confidence_level must be between 0 and 1.")

    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0,
        value_matrix.shape[0],
        size=(iterations, value_matrix.shape[0]),
    )
    estimates = np.empty((iterations, value_matrix.shape[1]), dtype=float)
    for start in range(0, iterations, BOOTSTRAP_BATCH_SIZE):
        stop = min(start + BOOTSTRAP_BATCH_SIZE, iterations)
        estimates[start:stop] = np.median(value_matrix[indices[start:stop]], axis=1)
    alpha = (1.0 - confidence_level) / 2.0
    return (
        np.median(value_matrix, axis=0),
        np.quantile(estimates, alpha, axis=0),
        np.quantile(estimates, 1.0 - alpha, axis=0),
    )


__all__ = ["paired_participant_bootstrap"]
