from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage

from studies.pain_study.scripts.t1.t1_anatomical_reference import (
    AnatomicalReferenceParameters,
    estimate_anatomical_reference,
    fit_anatomically_constrained_scalp,
)


def _synthetic_anatomy() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    shape = (128, 150, 160)
    head_center = np.array([64.0, 68.0, 72.0])
    brain_center = np.array([64.0, 64.0, 76.0])
    head_radii = np.array([53.0, 64.0, 72.0])
    brain_radii = np.array([43.0, 50.0, 48.0])
    coordinates = np.moveaxis(np.indices(shape, dtype=float), 0, -1)
    head_distance = np.sum(((coordinates - head_center) / head_radii) ** 2, axis=-1)
    brain_distance = np.sum(((coordinates - brain_center) / brain_radii) ** 2, axis=-1)
    head_mask = head_distance <= 1.0
    brain_mask = brain_distance <= 1.0

    random = np.random.default_rng(14)
    data = random.normal(0.0, 0.8, shape)
    data[head_mask] += 48.0
    data[brain_mask] += 82.0
    nose = ((coordinates[..., 0] - head_center[0]) / 12.0) ** 2 + (
        (coordinates[..., 1] - 137.0) / 16.0
    ) ** 2 + ((coordinates[..., 2] - 61.0) / 18.0) ** 2 <= 1.0
    data[nose] += 48.0
    full_head = head_mask | nose
    surface = full_head & ~ndimage.binary_erosion(full_head)
    surface_points = np.argwhere(surface).astype(float)
    return data.astype(np.float32), np.eye(4), surface_points, brain_center


def _ellipsoid_surface(
    center: np.ndarray,
    radii: np.ndarray,
    polar_count: int = 80,
    azimuth_count: int = 160,
) -> np.ndarray:
    polar = np.linspace(0.05, np.pi / 2.0 + 0.30, polar_count)
    azimuth = np.linspace(-np.pi, np.pi, azimuth_count, endpoint=False)
    polar_grid, azimuth_grid = np.meshgrid(polar, azimuth, indexing="ij")
    directions = np.column_stack(
        [
            np.sin(polar_grid).ravel() * np.cos(azimuth_grid).ravel(),
            np.sin(polar_grid).ravel() * np.sin(azimuth_grid).ravel(),
            np.cos(polar_grid).ravel(),
        ]
    )
    return center + directions * radii


def test_native_t1_reference_recovers_synthetic_brain_center() -> None:
    data, affine, surface_points, expected_center = _synthetic_anatomy()

    reference = estimate_anatomical_reference(
        data,
        affine,
        surface_points,
        AnatomicalReferenceParameters(),
    )

    assert np.linalg.norm(reference.position_mri_mm - expected_center) < 8.0
    assert reference.candidate_volume_mm3 > 500_000.0
    assert np.all(reference.candidate_spans_mm > 80.0)


def test_scalp_fit_keeps_center_anchored_on_partial_cap() -> None:
    reference = np.array([1.0, -2.0, 3.0])
    expected_center = reference + np.array([0.0, 7.0, -5.0])
    expected_radii = np.array([78.0, 95.0, 84.0])
    surface_points = _ellipsoid_surface(expected_center, expected_radii)

    model = fit_anatomically_constrained_scalp(
        surface_points,
        reference,
        AnatomicalReferenceParameters(),
    )

    assert np.allclose(model.center_mri_mm, expected_center)
    assert np.allclose(model.radii_mm, expected_radii, atol=0.2)
    assert model.p95_residual_mm < 0.2


def test_scalp_fit_rejects_anatomically_incoherent_surface() -> None:
    reference = np.zeros(3)
    center = reference + np.array([0.0, 7.0, -5.0])
    first_surface = _ellipsoid_surface(center, np.array([78.0, 95.0, 84.0]))
    second_surface = _ellipsoid_surface(
        center + np.array([0.0, 0.0, 25.0]),
        np.array([62.0, 75.0, 60.0]),
    )
    surface_points = np.concatenate([first_surface, second_surface])

    with pytest.raises(RuntimeError, match="residual"):
        fit_anatomically_constrained_scalp(
            surface_points,
            reference,
            AnatomicalReferenceParameters(
                maximum_median_residual_mm=2.0,
                maximum_p95_residual_mm=4.0,
            ),
        )
