"""Localize scalp EEG electrodes from full-head T1-weighted MRI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from scipy import ndimage
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class LocalizationParameters:
    """Numerical parameters for T1 electrode localization."""

    threshold_scale: float = 0.40
    closing_iterations: int = 2
    candidate_radius_mm: float = 18.0
    maximum_contact_displacement_mm: float = 12.0
    min_evidence_z: float = 2.0
    max_rotation_degrees: float = 25.0
    registration_iterations: int = 3
    minimum_artifact_separation_mm: float = 8.0
    minimum_artifact_prominence_mm: float = 1.5
    prominence_neighbor_count: int = 512
    footprint_neighbor_count: int = 64
    neighbor_query_chunk_size: int = 5_000
    minimum_head_spans_mm: tuple[float, float, float] = (105.0, 125.0, 125.0)

    def __post_init__(self) -> None:
        if not 0.0 < self.threshold_scale < 1.0:
            raise ValueError("threshold_scale must lie between zero and one.")
        if self.closing_iterations < 0:
            raise ValueError("closing_iterations cannot be negative.")
        if self.candidate_radius_mm <= 0.0:
            raise ValueError("candidate_radius_mm must be positive.")
        if not 0.0 < self.maximum_contact_displacement_mm <= self.candidate_radius_mm:
            raise ValueError(
                "maximum_contact_displacement_mm must be positive and cannot exceed "
                "candidate_radius_mm."
            )
        if self.min_evidence_z < 0.0:
            raise ValueError("min_evidence_z cannot be negative.")
        if not 0.0 < self.max_rotation_degrees <= 45.0:
            raise ValueError("max_rotation_degrees must lie between zero and 45 degrees.")
        if self.registration_iterations < 1:
            raise ValueError("registration_iterations must be positive.")
        if self.minimum_artifact_separation_mm <= 0.0:
            raise ValueError("minimum_artifact_separation_mm must be positive.")
        if self.minimum_artifact_prominence_mm <= 0.0:
            raise ValueError("minimum_artifact_prominence_mm must be positive.")
        if self.prominence_neighbor_count < 8:
            raise ValueError("prominence_neighbor_count must be at least eight.")
        if not 8 <= self.footprint_neighbor_count < self.prominence_neighbor_count:
            raise ValueError(
                "footprint_neighbor_count must be at least eight and smaller than "
                "prominence_neighbor_count."
            )
        if self.neighbor_query_chunk_size < 1:
            raise ValueError("neighbor_query_chunk_size must be positive.")
        spans = np.asarray(self.minimum_head_spans_mm, dtype=float)
        if spans.shape != (3,) or not np.isfinite(spans).all() or np.any(spans <= 0.0):
            raise ValueError("minimum_head_spans_mm must contain three positive values.")


@dataclass(frozen=True)
class ElectrodeLocalization:
    """Labeled MRI-space electrode contacts and detection diagnostics."""

    positions_mri_mm: dict[str, np.ndarray]
    artifact_positions_mri_mm: dict[str, np.ndarray]
    confidence: dict[str, float]
    evidence_z: dict[str, float]
    prominence_mm: dict[str, float]
    signed_prominence_mm: dict[str, float]
    template_displacement_mm: dict[str, float]
    cap_rotation_matrix: np.ndarray
    scalp_surface_mri_mm: np.ndarray
    scalp_center_mri_mm: np.ndarray
    scalp_radii_mm: np.ndarray


def _otsu_threshold(values: np.ndarray) -> float:
    histogram, edges = np.histogram(values, bins=512)
    centers = (edges[:-1] + edges[1:]) / 2.0
    weights = histogram.astype(float)
    cumulative_weight = np.cumsum(weights)
    cumulative_mean = np.cumsum(weights * centers)
    total_weight = cumulative_weight[-1]
    total_mean = cumulative_mean[-1]
    denominator = cumulative_weight * (total_weight - cumulative_weight)
    variance = np.zeros_like(denominator)
    valid = denominator > 0.0
    variance[valid] = (
        total_mean * cumulative_weight[valid] - cumulative_mean[valid] * total_weight
    ) ** 2 / denominator[valid]
    return float(centers[np.argmax(variance)])


def _largest_component(mask: np.ndarray) -> np.ndarray:
    labels, component_count = ndimage.label(mask)
    if component_count == 0:
        raise ValueError("T1 segmentation contains no foreground component.")
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == int(np.argmax(sizes))


def _segment_full_head(
    data: np.ndarray,
    parameters: LocalizationParameters,
) -> np.ndarray:
    if data.ndim != 3:
        raise ValueError(f"T1 data must be three-dimensional, got shape {data.shape}.")
    if not np.isfinite(data).all():
        raise ValueError("T1 data contain non-finite values.")
    positive = data[data > 0.0]
    if positive.size < 1_000:
        raise ValueError("T1 data do not contain enough positive full-head voxels.")

    upper_limit = np.percentile(positive, 99.8)
    clipped = positive[positive <= upper_limit]
    threshold = parameters.threshold_scale * _otsu_threshold(clipped)
    foreground = data > threshold
    if parameters.closing_iterations:
        foreground = ndimage.binary_closing(
            foreground,
            iterations=parameters.closing_iterations,
        )
    foreground = ndimage.binary_fill_holes(foreground)
    head = _largest_component(foreground)
    foreground_fraction = float(head.mean())
    if not 0.01 < foreground_fraction < 0.80:
        raise ValueError(
            "Full-head segmentation has an implausible foreground fraction: "
            f"{foreground_fraction:.4f}."
        )
    return head


def _surface_points_mri_mm(mask: np.ndarray, affine: np.ndarray) -> np.ndarray:
    boundary = mask & ~ndimage.binary_erosion(mask)
    voxel_points = np.argwhere(boundary)
    if voxel_points.shape[0] < 1_000:
        raise ValueError("Full-head segmentation has too few surface points.")
    rotation_scaling = affine[:3, :3]
    translation = affine[:3, 3]
    return np.einsum("ij,nj->ni", rotation_scaling, voxel_points) + translation


def _fit_scalp_ellipsoid(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lower = np.percentile(points, 1.0, axis=0)
    upper = np.percentile(points, 99.0, axis=0)
    half_span = (upper - lower) / 2.0
    center = (lower + upper) / 2.0
    center[2] = upper[2] - float(np.mean(half_span[:2]))
    radii = np.array(
        [half_span[0], half_span[1], float(np.mean(half_span[:2]))],
        dtype=float,
    )
    cap_points = points[points[:, 2] >= center[2] - 0.20 * radii[2]]
    if cap_points.shape[0] < 500:
        raise ValueError("Could not isolate enough superior scalp surface points.")

    initial = np.concatenate([center, radii])
    center_margin = np.maximum(radii * 0.35, 10.0)
    lower_bounds = np.concatenate([center - center_margin, radii * 0.65])
    upper_bounds = np.concatenate([center + center_margin, radii * 1.45])

    def residuals(parameters: np.ndarray) -> np.ndarray:
        fitted_center = parameters[:3]
        fitted_radii = parameters[3:]
        normalized = (cap_points - fitted_center) / fitted_radii
        return np.sqrt(np.sum(normalized**2, axis=1)) - 1.0

    fit = least_squares(
        residuals,
        initial,
        bounds=(lower_bounds, upper_bounds),
        loss="soft_l1",
        f_scale=0.02,
        max_nfev=300,
    )
    if not fit.success:
        raise RuntimeError(f"Scalp ellipsoid fitting failed: {fit.message}")
    return fit.x[:3], fit.x[3:]


def _validate_full_head_extent(
    surface_points: np.ndarray,
    minimum_spans_mm: tuple[float, float, float],
) -> None:
    spans = np.ptp(surface_points, axis=0)
    required = np.asarray(minimum_spans_mm, dtype=float)
    if np.any(spans < required):
        raise ValueError(
            "T1 segmentation does not have a plausible full-head extent in MRI RAS "
            f"coordinates; observed {spans.round(1).tolist()} mm, required at least "
            f"{required.tolist()} mm. The input may be brain-extracted or cropped."
        )


def _ellipsoid_contact(
    center: np.ndarray,
    radii: np.ndarray,
    direction: np.ndarray,
) -> np.ndarray:
    unit_direction = direction / np.linalg.norm(direction)
    radius = 1.0 / np.sqrt(np.sum((unit_direction / radii) ** 2))
    return center + radius * unit_direction


def _robust_z_score(value: float, reference: np.ndarray) -> float:
    median = float(np.median(reference))
    scale = 1.4826 * float(np.median(np.abs(reference - median)))
    if scale < 0.25:
        scale = 0.25
    return abs(value - median) / scale


def _detect_artifact_near_contact(
    predicted_contact: np.ndarray,
    center: np.ndarray,
    radii: np.ndarray,
    surface_points: np.ndarray,
    surface_prominence_mm: np.ndarray,
    surface_baseline_residual_mm: np.ndarray,
    surface_tree: cKDTree,
    candidate_radius_mm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    candidate_indices = surface_tree.query_ball_point(
        predicted_contact,
        candidate_radius_mm,
    )
    if len(candidate_indices) < 20:
        raise RuntimeError(
            "Too few scalp candidates within "
            f"{candidate_radius_mm:g} mm of a predicted electrode contact."
        )
    candidate_indices_array = np.asarray(candidate_indices, dtype=int)
    candidates = surface_points[candidate_indices_array]
    candidate_vectors = candidates - center
    candidate_directions = candidate_vectors / np.linalg.norm(
        candidate_vectors,
        axis=1,
        keepdims=True,
    )
    candidate_prominence = surface_prominence_mm[candidate_indices_array]
    strongest_index = int(np.argmax(np.abs(candidate_prominence)))
    artifact_position = candidates[strongest_index]
    artifact_direction = candidate_directions[strongest_index]
    surface_index = candidate_indices_array[strongest_index]
    smooth_scalp_radius = 1.0 / np.sqrt(np.sum((artifact_direction / radii) ** 2))
    smooth_scalp_radius += surface_baseline_residual_mm[surface_index]
    contact_position = center + smooth_scalp_radius * artifact_direction
    evidence_z = _robust_z_score(
        candidate_prominence[strongest_index],
        candidate_prominence,
    )
    return (
        artifact_position,
        artifact_direction,
        contact_position,
        evidence_z,
        float(candidate_prominence[strongest_index]),
    )


def _neighbor_median(
    values: np.ndarray,
    points: np.ndarray,
    tree: cKDTree,
    neighbor_count: int,
    chunk_size: int,
) -> np.ndarray:
    medians = np.empty(points.shape[0], dtype=float)
    for start in range(0, points.shape[0], chunk_size):
        stop = min(start + chunk_size, points.shape[0])
        _, indices = tree.query(
            points[start:stop],
            k=neighbor_count,
            workers=-1,
        )
        medians[start:stop] = np.median(values[indices], axis=1)
    return medians


def _surface_prominence_mm(
    surface_points: np.ndarray,
    center: np.ndarray,
    radii: np.ndarray,
    surface_tree: cKDTree,
    neighbor_count: int,
    footprint_neighbor_count: int,
    query_chunk_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    vectors = surface_points - center
    directions = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    ellipsoid_radii = 1.0 / np.sqrt(np.sum((directions / radii) ** 2, axis=1))
    radial_residuals = np.linalg.norm(vectors, axis=1) - ellipsoid_radii
    baseline_neighbors = min(neighbor_count, surface_points.shape[0])
    local_baseline = _neighbor_median(
        radial_residuals,
        surface_points,
        surface_tree,
        baseline_neighbors,
        query_chunk_size,
    )
    point_prominence = radial_residuals - local_baseline
    footprint_neighbors = min(footprint_neighbor_count, surface_points.shape[0])
    coherent_prominence = _neighbor_median(
        point_prominence,
        surface_points,
        surface_tree,
        footprint_neighbors,
        query_chunk_size,
    )
    return coherent_prominence, local_baseline


def _fit_cap_rotation(
    template_directions: np.ndarray,
    observed_directions: np.ndarray,
    maximum_degrees: float,
) -> np.ndarray:
    maximum_radians = np.deg2rad(maximum_degrees)

    def residuals(rotation_vector: np.ndarray) -> np.ndarray:
        rotated = Rotation.from_rotvec(rotation_vector).apply(template_directions)
        return (rotated - observed_directions).ravel()

    fit = least_squares(
        residuals,
        np.zeros(3),
        bounds=(-maximum_radians, maximum_radians),
        loss="soft_l1",
        f_scale=0.025,
        max_nfev=200,
    )
    if not fit.success:
        raise RuntimeError(f"Electrode-cap registration failed: {fit.message}")
    fitted_angle = float(np.linalg.norm(fit.x))
    if fitted_angle > maximum_radians:
        raise RuntimeError(
            "Electrode-cap registration exceeds the maximum cap rotation of "
            f"{maximum_degrees:g} degrees; fitted {np.rad2deg(fitted_angle):.2f} degrees."
        )
    return Rotation.from_rotvec(fit.x).as_matrix()


def _validate_unique_artifacts(
    artifacts: Mapping[str, np.ndarray],
    minimum_separation_mm: float,
) -> None:
    names = list(artifacts)
    if len(names) < 2:
        return
    coordinates = np.stack([artifacts[name] for name in names])
    distances = np.linalg.norm(
        coordinates[:, np.newaxis] - coordinates[np.newaxis, :],
        axis=2,
    )
    upper_triangle = np.triu_indices(len(names), k=1)
    collisions = np.flatnonzero(distances[upper_triangle] < minimum_separation_mm)
    if collisions.size == 0:
        return
    pairs = [
        (names[upper_triangle[0][index]], names[upper_triangle[1][index]]) for index in collisions
    ]
    raise RuntimeError(
        "Electrode detection is not one-to-one; artifact candidates are shared or "
        f"closer than {minimum_separation_mm:g} mm for channel pairs: {pairs}."
    )


def assess_electrodes_from_array(
    data: np.ndarray,
    affine: np.ndarray,
    template_positions: Mapping[str, np.ndarray],
    parameters: LocalizationParameters | None = None,
) -> ElectrodeLocalization:
    """Estimate labeled electrode candidates without applying acceptance gates."""
    options = parameters or LocalizationParameters()
    affine_array = np.asarray(affine, dtype=float)
    if affine_array.shape != (4, 4) or not np.isfinite(affine_array).all():
        raise ValueError("T1 affine must be a finite 4-by-4 matrix.")
    if not template_positions:
        raise ValueError("At least one template electrode position is required.")

    head_mask = _segment_full_head(np.asarray(data, dtype=float), options)
    surface_points = _surface_points_mri_mm(head_mask, affine_array)
    _validate_full_head_extent(surface_points, options.minimum_head_spans_mm)
    center, radii = _fit_scalp_ellipsoid(surface_points)
    surface_tree = cKDTree(surface_points)
    surface_prominence, surface_baseline_residual = _surface_prominence_mm(
        surface_points,
        center,
        radii,
        surface_tree,
        options.prominence_neighbor_count,
        options.footprint_neighbor_count,
        options.neighbor_query_chunk_size,
    )

    names = list(template_positions)
    template_directions = np.empty((len(names), 3), dtype=float)
    for index, name in enumerate(names):
        direction = np.asarray(template_positions[name], dtype=float)
        if direction.shape != (3,) or not np.isfinite(direction).all():
            raise ValueError(f"Template position for {name!r} must contain three finite values.")
        direction_norm = np.linalg.norm(direction)
        if direction_norm == 0.0:
            raise ValueError(f"Template position for {name!r} cannot be the origin.")
        template_directions[index] = direction / direction_norm

    cap_rotation = np.eye(3)
    for _ in range(options.registration_iterations):
        observed_directions = np.empty_like(template_directions)
        for index, template_direction in enumerate(template_directions):
            registered_direction = cap_rotation @ template_direction
            predicted_contact = _ellipsoid_contact(center, radii, registered_direction)
            _, observed_direction, _, _, _ = _detect_artifact_near_contact(
                predicted_contact,
                center,
                radii,
                surface_points,
                surface_prominence,
                surface_baseline_residual,
                surface_tree,
                options.candidate_radius_mm,
            )
            observed_directions[index] = observed_direction
        cap_rotation = _fit_cap_rotation(
            template_directions,
            observed_directions,
            options.max_rotation_degrees,
        )

    contacts: dict[str, np.ndarray] = {}
    artifacts: dict[str, np.ndarray] = {}
    confidence: dict[str, float] = {}
    evidence_scores: dict[str, float] = {}
    prominence_scores: dict[str, float] = {}
    signed_prominence_scores: dict[str, float] = {}
    template_displacements: dict[str, float] = {}

    for name, template_direction in zip(names, template_directions, strict=True):
        registered_direction = cap_rotation @ template_direction
        predicted_contact = _ellipsoid_contact(center, radii, registered_direction)
        artifact_position, artifact_direction, contact, evidence_z, signed_prominence = (
            _detect_artifact_near_contact(
                predicted_contact,
                center,
                radii,
                surface_points,
                surface_prominence,
                surface_baseline_residual,
                surface_tree,
                options.candidate_radius_mm,
            )
        )
        prominence = abs(signed_prominence)
        displacement = float(np.linalg.norm(contact - predicted_contact))
        distance_weight = np.exp(-0.5 * (displacement / options.candidate_radius_mm) ** 2)
        evidence_weight = 1.0 - np.exp(-0.5 * evidence_z)
        prominence_weight = 1.0 - np.exp(
            -0.5 * (prominence / options.minimum_artifact_prominence_mm) ** 2
        )

        contacts[name] = contact
        artifacts[name] = artifact_position
        confidence[name] = float(
            np.clip(distance_weight * evidence_weight * prominence_weight, 0.0, 1.0)
        )
        evidence_scores[name] = float(evidence_z)
        prominence_scores[name] = prominence
        signed_prominence_scores[name] = signed_prominence
        template_displacements[name] = displacement

    return ElectrodeLocalization(
        positions_mri_mm=contacts,
        artifact_positions_mri_mm=artifacts,
        confidence=confidence,
        evidence_z=evidence_scores,
        prominence_mm=prominence_scores,
        signed_prominence_mm=signed_prominence_scores,
        template_displacement_mm=template_displacements,
        cap_rotation_matrix=cap_rotation,
        scalp_surface_mri_mm=surface_points,
        scalp_center_mri_mm=center,
        scalp_radii_mm=radii,
    )


def _validate_localization(
    result: ElectrodeLocalization,
    parameters: LocalizationParameters,
) -> None:
    weak = sorted(
        name for name, score in result.evidence_z.items() if score < parameters.min_evidence_z
    )
    if weak:
        raise RuntimeError(
            "Electrode artifacts did not meet the minimum evidence threshold for: " f"{weak}."
        )

    insufficient_prominence = sorted(
        name
        for name, prominence in result.prominence_mm.items()
        if prominence < parameters.minimum_artifact_prominence_mm
    )
    if insufficient_prominence:
        raise RuntimeError(
            "Electrode artifacts did not meet the minimum prominence of "
            f"{parameters.minimum_artifact_prominence_mm:g} mm for: "
            f"{insufficient_prominence}."
        )

    topology_outliers = sorted(
        name
        for name, displacement in result.template_displacement_mm.items()
        if displacement > parameters.maximum_contact_displacement_mm
    )
    if topology_outliers:
        raise RuntimeError(
            "Electrode contacts exceed the maximum registered cap-topology displacement "
            f"of {parameters.maximum_contact_displacement_mm:g} mm for: {topology_outliers}."
        )

    _validate_unique_artifacts(
        result.artifact_positions_mri_mm,
        parameters.minimum_artifact_separation_mm,
    )


def localize_electrodes_from_array(
    data: np.ndarray,
    affine: np.ndarray,
    template_positions: Mapping[str, np.ndarray],
    parameters: LocalizationParameters | None = None,
) -> ElectrodeLocalization:
    """Detect, label, and strictly validate electrode artifacts from full-head T1."""
    options = parameters or LocalizationParameters()
    result = assess_electrodes_from_array(
        data,
        affine,
        template_positions,
        options,
    )
    _validate_localization(result, options)
    return result


__all__ = [
    "ElectrodeLocalization",
    "LocalizationParameters",
    "assess_electrodes_from_array",
    "localize_electrodes_from_array",
]
