"""Point-spread full-width half-maximum summaries for Study 2."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PointSpreadFWHMReport:
    vertex_fwhm_mm: np.ndarray
    median_fwhm_mm: float
    min_fwhm_mm: float
    max_fwhm_mm: float
    q1_fwhm_mm: float
    q3_fwhm_mm: float
    n_vertices: int


def compute_point_spread_fwhm(
    *,
    resolution_matrix: np.ndarray,
    distances_mm: np.ndarray,
) -> PointSpreadFWHMReport:
    """Compute vertex-wise point-spread FWHM from a resolution matrix and distances."""
    resolution = np.asarray(resolution_matrix, dtype=float)
    distances = np.asarray(distances_mm, dtype=float)
    _validate_inputs(resolution, distances)

    vertex_fwhm = np.asarray(
        [
            _single_vertex_fwhm(resolution[:, vertex], distances)
            for vertex in range(resolution.shape[1])
        ],
        dtype=float,
    )
    return PointSpreadFWHMReport(
        vertex_fwhm_mm=vertex_fwhm,
        median_fwhm_mm=float(np.median(vertex_fwhm)),
        min_fwhm_mm=float(np.min(vertex_fwhm)),
        max_fwhm_mm=float(np.max(vertex_fwhm)),
        q1_fwhm_mm=float(np.quantile(vertex_fwhm, 0.25)),
        q3_fwhm_mm=float(np.quantile(vertex_fwhm, 0.75)),
        n_vertices=int(vertex_fwhm.size),
    )


def _validate_inputs(resolution: np.ndarray, distances: np.ndarray) -> None:
    if resolution.ndim != 2 or resolution.shape[0] != resolution.shape[1]:
        raise ValueError("Study 2 resolution_matrix must be square.")
    if distances.shape != resolution.shape:
        raise ValueError("Study 2 distances_mm must match resolution_matrix shape.")
    if resolution.shape[0] == 0:
        raise ValueError("Study 2 point-spread inputs must not be empty.")
    if not np.all(np.isfinite(resolution)):
        raise ValueError("Study 2 resolution_matrix contains non-finite values.")
    if not np.all(np.isfinite(distances)):
        raise ValueError("Study 2 distances_mm contains non-finite values.")
    if np.any(distances < 0.0):
        raise ValueError("Study 2 distances_mm must be non-negative.")
    if not np.allclose(distances, distances.T):
        raise ValueError("Study 2 distances_mm must be symmetric.")


def _single_vertex_fwhm(psf: np.ndarray, distances: np.ndarray) -> float:
    magnitude = np.abs(psf)
    peak = float(np.max(magnitude))
    if peak <= 0.0:
        raise ValueError("Study 2 point-spread function has zero peak.")
    half_max_mask = magnitude >= peak / 2.0
    half_max_distances = distances[np.ix_(half_max_mask, half_max_mask)]
    return float(np.max(half_max_distances))


__all__ = ["PointSpreadFWHMReport", "compute_point_spread_fwhm"]
