"""BrainSMASH surrogate generation for Study 2 surface maps."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np


def generate_brainsmash_surrogates(
    *,
    target_map: np.ndarray,
    distance_matrix: np.ndarray,
    n_surrogates: int,
    seed: int,
    generator_type: Callable[..., Any] | None = None,
) -> np.ndarray:
    """Generate an exact number of spatial-autocorrelation-matched maps."""
    target = _finite_target(target_map)
    distances = _distance_matrix(distance_matrix, n_vertices=target.size)
    if not isinstance(n_surrogates, int) or isinstance(n_surrogates, bool) or n_surrogates < 1:
        raise ValueError("Study 2 BrainSMASH n_surrogates must be a positive integer.")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("Study 2 BrainSMASH seed must be an integer.")

    if generator_type is None:
        try:
            from brainsmash.mapgen.base import Base
        except ImportError as error:
            raise ImportError(
                "BrainSMASH is required to generate Study 2 spatial surrogates."
            ) from error
        generator_type = Base

    generator = generator_type(x=target, D=distances, seed=seed)
    surrogates = np.asarray(generator(n=n_surrogates), dtype=float)
    expected_shape = (n_surrogates, target.size)
    if surrogates.shape != expected_shape:
        raise ValueError(
            f"BrainSMASH returned shape {surrogates.shape}; expected {expected_shape}."
        )
    if not np.all(np.isfinite(surrogates)):
        raise ValueError("BrainSMASH returned non-finite surrogate values.")
    return surrogates


def _finite_target(values: np.ndarray) -> np.ndarray:
    target = np.asarray(values, dtype=float)
    if target.ndim != 1 or target.size < 2:
        raise ValueError("Study 2 BrainSMASH target_map must contain at least two vertices.")
    if not np.all(np.isfinite(target)):
        raise ValueError("Study 2 BrainSMASH target_map contains non-finite values.")
    if float(np.ptp(target)) <= 0.0:
        raise ValueError("Study 2 BrainSMASH target_map must have nonzero variance.")
    return target


def _distance_matrix(values: np.ndarray, *, n_vertices: int) -> np.ndarray:
    distances = np.asarray(values, dtype=float)
    if distances.shape != (n_vertices, n_vertices):
        raise ValueError("Study 2 BrainSMASH distance_matrix must be square and match target_map.")
    if not np.all(np.isfinite(distances)) or np.any(distances < 0.0):
        raise ValueError("Study 2 BrainSMASH distances must be finite and nonnegative.")
    if not np.allclose(distances, distances.T):
        raise ValueError("Study 2 BrainSMASH distance_matrix must be symmetric.")
    if not np.allclose(np.diag(distances), 0.0):
        raise ValueError("Study 2 BrainSMASH distance_matrix diagonal must be zero.")
    return distances


__all__ = ["generate_brainsmash_surrogates"]
