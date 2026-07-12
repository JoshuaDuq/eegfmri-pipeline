from __future__ import annotations

import numpy as np
import pytest


def test_compute_point_spread_fwhm_uses_half_max_extent() -> None:
    from studies.pain_study.study2.point_spread import compute_point_spread_fwhm

    resolution_matrix = np.asarray(
        [
            [1.0, 0.6, 0.1],
            [0.6, 1.0, 0.6],
            [0.1, 0.6, 1.0],
        ],
        dtype=float,
    )
    distances_mm = np.asarray(
        [
            [0.0, 10.0, 20.0],
            [10.0, 0.0, 10.0],
            [20.0, 10.0, 0.0],
        ],
        dtype=float,
    )

    result = compute_point_spread_fwhm(
        resolution_matrix=resolution_matrix,
        distances_mm=distances_mm,
    )

    assert result.vertex_fwhm_mm.tolist() == [10.0, 20.0, 10.0]
    assert result.median_fwhm_mm == pytest.approx(10.0)
    assert result.max_fwhm_mm == pytest.approx(20.0)


def test_compute_point_spread_fwhm_rejects_distance_shape_mismatch() -> None:
    from studies.pain_study.study2.point_spread import compute_point_spread_fwhm

    with pytest.raises(ValueError, match="distances_mm"):
        compute_point_spread_fwhm(
            resolution_matrix=np.ones((3, 3), dtype=float),
            distances_mm=np.ones((2, 2), dtype=float),
        )


def test_compute_point_spread_fwhm_uses_resolution_matrix_columns() -> None:
    from studies.pain_study.study2.point_spread import compute_point_spread_fwhm

    resolution_matrix = np.asarray(
        [
            [1.0, 0.1, 0.1],
            [0.6, 1.0, 0.1],
            [0.1, 0.6, 1.0],
        ],
        dtype=float,
    )
    distances_mm = np.asarray(
        [
            [0.0, 10.0, 20.0],
            [10.0, 0.0, 10.0],
            [20.0, 10.0, 0.0],
        ],
        dtype=float,
    )

    result = compute_point_spread_fwhm(
        resolution_matrix=resolution_matrix,
        distances_mm=distances_mm,
    )

    assert result.vertex_fwhm_mm.tolist() == [10.0, 10.0, 0.0]
