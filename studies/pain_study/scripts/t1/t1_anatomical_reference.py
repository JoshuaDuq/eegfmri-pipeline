"""Native-T1 anatomical reference and constrained scalp modeling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.optimize import least_squares

from studies.pain_study.scripts.t1.t1_electrode_localization import _otsu_threshold


@dataclass(frozen=True)
class AnatomicalReferenceParameters:
    """Validated assumptions for native-T1 head-center estimation."""

    superior_crop_depth_mm: float = 125.0
    anterior_crop_margin_mm: float = 50.0
    tissue_threshold_scale: float = 0.90
    head_center_offset_mm: tuple[float, float, float] = (0.0, 7.0, -5.0)
    reference_uncertainty_mm: float = 10.0
    head_center_offset_uncertainty_mm: float = 8.0
    minimum_candidate_volume_mm3: float = 500_000.0
    maximum_candidate_volume_mm3: float = 3_000_000.0
    minimum_candidate_spans_mm: tuple[float, float, float] = (80.0, 80.0, 90.0)
    minimum_scalp_radii_mm: tuple[float, float, float] = (60.0, 70.0, 65.0)
    maximum_scalp_radii_mm: tuple[float, float, float] = (100.0, 115.0, 110.0)
    minimum_superior_radius_ratio: float = 0.75
    maximum_superior_radius_ratio: float = 1.25
    maximum_median_residual_mm: float = 6.0
    maximum_p95_residual_mm: float = 15.0

    def __post_init__(self) -> None:
        positive_values = {
            "superior_crop_depth_mm": self.superior_crop_depth_mm,
            "anterior_crop_margin_mm": self.anterior_crop_margin_mm,
            "reference_uncertainty_mm": self.reference_uncertainty_mm,
            "head_center_offset_uncertainty_mm": self.head_center_offset_uncertainty_mm,
            "minimum_candidate_volume_mm3": self.minimum_candidate_volume_mm3,
            "maximum_candidate_volume_mm3": self.maximum_candidate_volume_mm3,
            "maximum_median_residual_mm": self.maximum_median_residual_mm,
            "maximum_p95_residual_mm": self.maximum_p95_residual_mm,
        }
        invalid = [name for name, value in positive_values.items() if value <= 0.0]
        if invalid:
            raise ValueError(f"Anatomical-reference parameters must be positive: {invalid}.")
        if not 0.0 < self.tissue_threshold_scale < 2.0:
            raise ValueError("tissue_threshold_scale must lie between zero and two.")
        if self.minimum_candidate_volume_mm3 >= self.maximum_candidate_volume_mm3:
            raise ValueError(
                "minimum_candidate_volume_mm3 must be smaller than " "maximum_candidate_volume_mm3."
            )
        _validate_vector(self.head_center_offset_mm, "head_center_offset_mm")
        minimum_spans = _validate_positive_vector(
            self.minimum_candidate_spans_mm,
            "minimum_candidate_spans_mm",
        )
        minimum_radii = _validate_positive_vector(
            self.minimum_scalp_radii_mm,
            "minimum_scalp_radii_mm",
        )
        maximum_radii = _validate_positive_vector(
            self.maximum_scalp_radii_mm,
            "maximum_scalp_radii_mm",
        )
        if np.any(minimum_radii >= maximum_radii):
            raise ValueError("Every minimum scalp radius must be smaller than its maximum.")
        if np.any(minimum_spans <= 0.0):
            raise ValueError("minimum_candidate_spans_mm must be positive.")
        if not 0.0 < self.minimum_superior_radius_ratio < self.maximum_superior_radius_ratio:
            raise ValueError("Superior-radius ratio bounds are invalid.")
        if self.maximum_median_residual_mm >= self.maximum_p95_residual_mm:
            raise ValueError(
                "maximum_median_residual_mm must be smaller than maximum_p95_residual_mm."
            )


@dataclass(frozen=True)
class AnatomicalReference:
    """Intensity-weighted native-T1 approximation of the brain centroid."""

    position_mri_mm: np.ndarray
    tissue_threshold: float
    candidate_volume_mm3: float
    candidate_spans_mm: np.ndarray


@dataclass(frozen=True)
class ConstrainedScalpModel:
    """Scalp ellipsoid fixed to an anatomically referenced head center."""

    center_mri_mm: np.ndarray
    radii_mm: np.ndarray
    median_residual_mm: float
    p95_residual_mm: float
    superior_radius_ratio: float


def _validate_vector(values: tuple[float, float, float], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain three finite values.")
    return array


def _validate_positive_vector(
    values: tuple[float, float, float],
    name: str,
) -> np.ndarray:
    array = _validate_vector(values, name)
    if np.any(array <= 0.0):
        raise ValueError(f"{name} must contain three positive values.")
    return array


def _world_axis_grid(shape: tuple[int, ...], affine: np.ndarray, axis: int) -> np.ndarray:
    voxel_axes = np.ogrid[tuple(slice(0, length) for length in shape)]
    world_axis = affine[axis, 3]
    for voxel_axis, coordinates in enumerate(voxel_axes):
        world_axis = world_axis + affine[axis, voxel_axis] * coordinates
    return world_axis


def _candidate_spans_mm(candidate: np.ndarray, affine: np.ndarray) -> np.ndarray:
    bounds = ndimage.find_objects(candidate.astype(np.uint8))
    if not bounds or bounds[0] is None:
        raise RuntimeError("Native-T1 anatomical-reference candidate is empty.")
    slices = bounds[0]
    limits = [(item.start, item.stop - 1) for item in slices]
    corners = np.array(
        [[x, y, z] for x in limits[0] for y in limits[1] for z in limits[2]],
        dtype=float,
    )
    world_corners = corners @ affine[:3, :3].T + affine[:3, 3]
    return np.ptp(world_corners, axis=0)


def estimate_anatomical_reference(
    data: np.ndarray,
    affine: np.ndarray,
    scalp_surface_mri_mm: np.ndarray,
    parameters: AnatomicalReferenceParameters,
) -> AnatomicalReference:
    """Estimate a reproducible intracranial intensity centroid from native T1w."""
    image = np.asarray(data, dtype=float)
    affine_array = np.asarray(affine, dtype=float)
    surface = np.asarray(scalp_surface_mri_mm, dtype=float)
    if image.ndim != 3 or not np.isfinite(image).all():
        raise ValueError("Native T1w data must be a finite three-dimensional array.")
    if affine_array.shape != (4, 4) or not np.isfinite(affine_array).all():
        raise ValueError("T1 affine must be a finite 4-by-4 matrix.")
    if surface.ndim != 2 or surface.shape[1] != 3 or surface.shape[0] < 1_000:
        raise ValueError("Anatomical-reference estimation requires a valid scalp surface.")

    positive = image[image > 0.0]
    if positive.size < 1_000:
        raise ValueError("Native T1w data contain too few positive voxels.")
    upper_intensity = np.percentile(positive, 99.8)
    tissue_threshold = parameters.tissue_threshold_scale * _otsu_threshold(
        positive[positive <= upper_intensity]
    )
    superior_limit = float(np.percentile(surface[:, 2], 99.0))
    anterior_limit = float(np.percentile(surface[:, 1], 99.0))
    world_y = _world_axis_grid(image.shape, affine_array, axis=1)
    world_z = _world_axis_grid(image.shape, affine_array, axis=2)
    anatomical_crop = (world_z > superior_limit - parameters.superior_crop_depth_mm) & (
        world_y < anterior_limit - parameters.anterior_crop_margin_mm
    )
    candidate = anatomical_crop & (image > tissue_threshold)
    voxel_volume_mm3 = abs(float(np.linalg.det(affine_array[:3, :3])))
    candidate_volume_mm3 = float(np.count_nonzero(candidate) * voxel_volume_mm3)
    if not (
        parameters.minimum_candidate_volume_mm3
        <= candidate_volume_mm3
        <= parameters.maximum_candidate_volume_mm3
    ):
        raise RuntimeError(
            "Native-T1 anatomical-reference candidate volume is implausible: "
            f"{candidate_volume_mm3:.0f} mm3."
        )
    candidate_spans = _candidate_spans_mm(candidate, affine_array)
    minimum_spans = np.asarray(parameters.minimum_candidate_spans_mm, dtype=float)
    if np.any(candidate_spans < minimum_spans):
        raise RuntimeError(
            "Native-T1 anatomical-reference candidate has implausible spans: "
            f"{candidate_spans.round(1).tolist()} mm."
        )

    weights = np.where(candidate, image - tissue_threshold, 0.0)
    centroid_voxel = np.asarray(ndimage.center_of_mass(weights), dtype=float)
    if not np.isfinite(centroid_voxel).all():
        raise RuntimeError("Native-T1 anatomical-reference centroid is non-finite.")
    centroid_mri_mm = affine_array[:3, :3] @ centroid_voxel + affine_array[:3, 3]
    return AnatomicalReference(
        position_mri_mm=centroid_mri_mm,
        tissue_threshold=float(tissue_threshold),
        candidate_volume_mm3=candidate_volume_mm3,
        candidate_spans_mm=candidate_spans,
    )


def _radial_residuals_mm(
    points: np.ndarray,
    center: np.ndarray,
    radii: np.ndarray,
) -> np.ndarray:
    vectors = points - center
    distances = np.linalg.norm(vectors, axis=1)
    directions = vectors / distances[:, np.newaxis]
    ellipsoid_distances = 1.0 / np.sqrt(np.sum((directions / radii) ** 2, axis=1))
    return np.abs(distances - ellipsoid_distances)


def fit_anatomically_constrained_scalp(
    scalp_surface_mri_mm: np.ndarray,
    anatomical_reference_mri_mm: np.ndarray,
    parameters: AnatomicalReferenceParameters,
) -> ConstrainedScalpModel:
    """Fit scalp radii while fixing the ill-conditioned center anatomically."""
    surface = np.asarray(scalp_surface_mri_mm, dtype=float)
    reference = np.asarray(anatomical_reference_mri_mm, dtype=float)
    if surface.ndim != 2 or surface.shape[1] != 3 or surface.shape[0] < 500:
        raise ValueError("Constrained scalp fitting requires at least 500 surface points.")
    if reference.shape != (3,) or not np.isfinite(reference).all():
        raise ValueError("Anatomical reference must contain three finite values.")

    center = reference + np.asarray(parameters.head_center_offset_mm, dtype=float)
    lower = np.percentile(surface, 1.0, axis=0)
    upper = np.percentile(surface, 99.0, axis=0)
    horizontal_half_spans = (upper[:2] - lower[:2]) / 2.0
    cap_threshold = center[2] - 0.25 * float(np.mean(horizontal_half_spans))
    cap_points = surface[surface[:, 2] >= cap_threshold]
    if cap_points.shape[0] < 500:
        raise RuntimeError("Could not isolate enough anatomically referenced scalp points.")

    minimum_radii = np.asarray(parameters.minimum_scalp_radii_mm, dtype=float)
    maximum_radii = np.asarray(parameters.maximum_scalp_radii_mm, dtype=float)
    initial_radii = np.array(
        [horizontal_half_spans[0], horizontal_half_spans[1], upper[2] - center[2]],
        dtype=float,
    )
    initial_radii = np.clip(initial_radii, minimum_radii, maximum_radii)

    def normalized_residuals(radii: np.ndarray) -> np.ndarray:
        normalized = (cap_points - center) / radii
        return np.sqrt(np.sum(normalized**2, axis=1)) - 1.0

    fit = least_squares(
        normalized_residuals,
        initial_radii,
        bounds=(minimum_radii, maximum_radii),
        loss="soft_l1",
        f_scale=0.02,
        max_nfev=300,
    )
    if not fit.success:
        raise RuntimeError(f"Anatomically constrained scalp fitting failed: {fit.message}")
    residuals_mm = _radial_residuals_mm(cap_points, center, fit.x)
    median_residual_mm = float(np.median(residuals_mm))
    p95_residual_mm = float(np.percentile(residuals_mm, 95.0))
    if (
        median_residual_mm > parameters.maximum_median_residual_mm
        or p95_residual_mm > parameters.maximum_p95_residual_mm
    ):
        raise RuntimeError(
            "Anatomically constrained scalp residuals exceed acceptance limits: "
            f"median {median_residual_mm:.2f} mm, p95 {p95_residual_mm:.2f} mm."
        )
    superior_ratio = float(fit.x[2] / np.mean(fit.x[:2]))
    if not (
        parameters.minimum_superior_radius_ratio
        <= superior_ratio
        <= parameters.maximum_superior_radius_ratio
    ):
        raise RuntimeError(
            "Anatomically constrained scalp has an implausible superior-radius ratio: "
            f"{superior_ratio:.3f}."
        )
    return ConstrainedScalpModel(
        center_mri_mm=center,
        radii_mm=fit.x,
        median_residual_mm=median_residual_mm,
        p95_residual_mm=p95_residual_mm,
        superior_radius_ratio=superior_ratio,
    )


__all__ = [
    "AnatomicalReference",
    "AnatomicalReferenceParameters",
    "ConstrainedScalpModel",
    "estimate_anatomical_reference",
    "fit_anatomically_constrained_scalp",
]
