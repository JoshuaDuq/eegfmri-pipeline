"""Infer a standard EEG montage on an individual full-head T1w scalp."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from studies.pain_study.scripts.t1_anatomical_reference import (
    AnatomicalReferenceParameters,
    estimate_anatomical_reference,
    fit_anatomically_constrained_scalp,
)
from studies.pain_study.scripts.t1_electrode_localization import (
    LocalizationParameters,
    _ellipsoid_contact,
    _segment_full_head,
    _surface_points_mri_mm,
    _surface_prominence_mm,
    _validate_full_head_extent,
)


@dataclass(frozen=True)
class TemplateProjectionParameters:
    """Assumptions controlling anatomical standard-montage projection."""

    threshold_scale: float = 0.40
    closing_iterations: int = 2
    surface_smoothing_neighbors: int = 512
    surface_footprint_neighbors: int = 64
    neighbor_query_chunk_size: int = 5_000
    minimum_head_spans_mm: tuple[float, float, float] = (105.0, 125.0, 125.0)
    orientation_uncertainty_degrees: float = 10.0
    surface_uncertainty_mm: float = 3.0
    minimum_electrode_separation_mm: float = 15.0
    maximum_smooth_scalp_gap_mm: float = 8.0
    anatomical_reference: AnatomicalReferenceParameters = field(
        default_factory=AnatomicalReferenceParameters
    )

    def __post_init__(self) -> None:
        if not 0.0 < self.threshold_scale < 1.0:
            raise ValueError("threshold_scale must lie between zero and one.")
        if self.closing_iterations < 0:
            raise ValueError("closing_iterations cannot be negative.")
        if self.surface_smoothing_neighbors < 8:
            raise ValueError("surface_smoothing_neighbors must be at least eight.")
        if not 8 <= self.surface_footprint_neighbors < self.surface_smoothing_neighbors:
            raise ValueError(
                "surface_footprint_neighbors must be at least eight and smaller than "
                "surface_smoothing_neighbors."
            )
        if self.neighbor_query_chunk_size < 1:
            raise ValueError("neighbor_query_chunk_size must be positive.")
        spans = np.asarray(self.minimum_head_spans_mm, dtype=float)
        if spans.shape != (3,) or not np.isfinite(spans).all() or np.any(spans <= 0.0):
            raise ValueError("minimum_head_spans_mm must contain three positive values.")
        if not 0.0 < self.orientation_uncertainty_degrees <= 30.0:
            raise ValueError(
                "orientation_uncertainty_degrees must lie between zero and 30 degrees."
            )
        if self.surface_uncertainty_mm <= 0.0:
            raise ValueError("surface_uncertainty_mm must be positive.")
        if self.minimum_electrode_separation_mm <= 0.0:
            raise ValueError("minimum_electrode_separation_mm must be positive.")
        if self.maximum_smooth_scalp_gap_mm <= 0.0:
            raise ValueError("maximum_smooth_scalp_gap_mm must be positive.")
        if not isinstance(self.anatomical_reference, AnatomicalReferenceParameters):
            raise TypeError("anatomical_reference must be AnatomicalReferenceParameters.")


@dataclass(frozen=True)
class TemplateMontageInference:
    """Anatomically projected positions and sensitivity-based uncertainty."""

    positions_mri_mm: dict[str, np.ndarray]
    positional_uncertainty_mm: dict[str, float]
    orientation_sensitivity_mm: dict[str, float]
    scalp_surface_mri_mm: np.ndarray
    scalp_center_mri_mm: np.ndarray
    scalp_radii_mm: np.ndarray
    anatomical_reference_mri_mm: np.ndarray
    anatomical_reference_candidate_volume_mm3: float
    anatomical_reference_candidate_spans_mm: np.ndarray
    scalp_fit_median_residual_mm: float
    scalp_fit_p95_residual_mm: float
    scalp_superior_radius_ratio: float
    minimum_electrode_separation_mm: float | None
    maximum_smooth_scalp_gap_mm: float
    orientation_matrix: np.ndarray
    orientation_uncertainty_degrees: float
    surface_uncertainty_mm: float
    anatomical_reference_uncertainty_mm: float
    head_center_offset_uncertainty_mm: float


@dataclass(frozen=True)
class TemplateMontageOutputPaths:
    """Files written for one anatomically projected montage."""

    electrodes_tsv: Path
    coordinate_system_json: Path
    diagnostics_json: Path
    qc_png: Path
    mne_head_montage_fif: Path
    scanner_ras_to_head_json: Path


def _validated_template_directions(
    template_positions: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    if not template_positions:
        raise ValueError("At least one template electrode position is required.")
    directions: dict[str, np.ndarray] = {}
    for name, position in template_positions.items():
        coordinates = np.asarray(position, dtype=float)
        if coordinates.shape != (3,) or not np.isfinite(coordinates).all():
            raise ValueError(f"Template position for {name!r} must contain three finite values.")
        norm = float(np.linalg.norm(coordinates))
        if norm == 0.0:
            raise ValueError(f"Template position for {name!r} cannot be the origin.")
        directions[name] = coordinates / norm
    return directions


def _project_direction_to_smooth_scalp(
    direction: np.ndarray,
    center: np.ndarray,
    radii: np.ndarray,
    surface_points: np.ndarray,
    surface_tree: cKDTree,
    local_baseline_residual_mm: np.ndarray,
) -> np.ndarray:
    unit_direction = direction / np.linalg.norm(direction)
    ellipsoid_position = _ellipsoid_contact(center, radii, unit_direction)
    _, nearest_index = surface_tree.query(ellipsoid_position)
    ellipsoid_radius = float(np.linalg.norm(ellipsoid_position - center))
    smooth_radius = ellipsoid_radius + local_baseline_residual_mm[int(nearest_index)]
    return center + smooth_radius * unit_direction


def _orientation_scenarios(maximum_degrees: float) -> tuple[np.ndarray, ...]:
    rotations = [np.eye(3)]
    for axis in "xyz":
        for angle in (-maximum_degrees, maximum_degrees):
            rotations.append(Rotation.from_euler(axis, angle, degrees=True).as_matrix())
    return tuple(rotations)


def infer_template_montage_from_array(
    data: np.ndarray,
    affine: np.ndarray,
    template_positions: Mapping[str, np.ndarray],
    parameters: TemplateProjectionParameters | None = None,
) -> TemplateMontageInference:
    """Project montage directions to a smooth T1-derived scalp with uncertainty."""
    options = parameters or TemplateProjectionParameters()
    affine_array = np.asarray(affine, dtype=float)
    if affine_array.shape != (4, 4) or not np.isfinite(affine_array).all():
        raise ValueError("T1 affine must be a finite 4-by-4 matrix.")
    template_directions = _validated_template_directions(template_positions)

    segmentation_options = LocalizationParameters(
        threshold_scale=options.threshold_scale,
        closing_iterations=options.closing_iterations,
        prominence_neighbor_count=options.surface_smoothing_neighbors,
        footprint_neighbor_count=options.surface_footprint_neighbors,
        neighbor_query_chunk_size=options.neighbor_query_chunk_size,
        minimum_head_spans_mm=options.minimum_head_spans_mm,
    )
    image = np.asarray(data, dtype=float)
    head_mask = _segment_full_head(image, segmentation_options)
    surface_points = _surface_points_mri_mm(head_mask, affine_array)
    _validate_full_head_extent(surface_points, options.minimum_head_spans_mm)
    anatomical_reference = estimate_anatomical_reference(
        image,
        affine_array,
        surface_points,
        options.anatomical_reference,
    )
    scalp_model = fit_anatomically_constrained_scalp(
        surface_points,
        anatomical_reference.position_mri_mm,
        options.anatomical_reference,
    )
    center = scalp_model.center_mri_mm
    radii = scalp_model.radii_mm
    surface_tree = cKDTree(surface_points)
    _, local_baseline = _surface_prominence_mm(
        surface_points,
        center,
        radii,
        surface_tree,
        options.surface_smoothing_neighbors,
        options.surface_footprint_neighbors,
        options.neighbor_query_chunk_size,
    )

    scenario_rotations = _orientation_scenarios(options.orientation_uncertainty_degrees)
    scenario_positions: dict[str, list[np.ndarray]] = {name: [] for name in template_directions}
    for rotation in scenario_rotations:
        for name, direction in template_directions.items():
            scenario_positions[name].append(
                _project_direction_to_smooth_scalp(
                    rotation @ direction,
                    center,
                    radii,
                    surface_points,
                    surface_tree,
                    local_baseline,
                )
            )

    positions = {name: values[0] for name, values in scenario_positions.items()}
    names = list(positions)
    position_array = np.stack([positions[name] for name in names])
    minimum_separation_mm: float | None = None
    if len(names) > 1:
        separation_matrix = np.linalg.norm(
            position_array[:, np.newaxis] - position_array[np.newaxis, :],
            axis=2,
        )
        np.fill_diagonal(separation_matrix, np.inf)
        first_index, second_index = np.unravel_index(
            np.argmin(separation_matrix),
            separation_matrix.shape,
        )
        minimum_separation_mm = float(separation_matrix[first_index, second_index])
        if minimum_separation_mm < options.minimum_electrode_separation_mm:
            raise RuntimeError(
                "Projected montage violates the minimum electrode separation of "
                f"{options.minimum_electrode_separation_mm:g} mm: "
                f"{names[first_index]}--{names[second_index]} = "
                f"{minimum_separation_mm:.2f} mm."
            )
    scalp_gaps_mm = surface_tree.query(position_array)[0]
    maximum_scalp_gap_mm = float(np.max(scalp_gaps_mm))
    if maximum_scalp_gap_mm > options.maximum_smooth_scalp_gap_mm:
        channel = names[int(np.argmax(scalp_gaps_mm))]
        raise RuntimeError(
            "Projected montage exceeds the maximum smooth-scalp gap of "
            f"{options.maximum_smooth_scalp_gap_mm:g} mm: {channel} = "
            f"{maximum_scalp_gap_mm:.2f} mm."
        )
    uncertainty: dict[str, float] = {}
    orientation_sensitivity: dict[str, float] = {}
    for name, values in scenario_positions.items():
        deviations = np.linalg.norm(np.stack(values[1:]) - values[0], axis=1)
        orientation_deviation = float(np.max(deviations))
        orientation_sensitivity[name] = orientation_deviation
        uncertainty_components = np.array(
            [
                orientation_deviation,
                options.surface_uncertainty_mm,
                options.anatomical_reference.reference_uncertainty_mm,
                options.anatomical_reference.head_center_offset_uncertainty_mm,
            ]
        )
        uncertainty[name] = float(np.linalg.norm(uncertainty_components))

    return TemplateMontageInference(
        positions_mri_mm=positions,
        positional_uncertainty_mm=uncertainty,
        orientation_sensitivity_mm=orientation_sensitivity,
        scalp_surface_mri_mm=surface_points,
        scalp_center_mri_mm=center,
        scalp_radii_mm=radii,
        anatomical_reference_mri_mm=anatomical_reference.position_mri_mm,
        anatomical_reference_candidate_volume_mm3=(anatomical_reference.candidate_volume_mm3),
        anatomical_reference_candidate_spans_mm=anatomical_reference.candidate_spans_mm,
        scalp_fit_median_residual_mm=scalp_model.median_residual_mm,
        scalp_fit_p95_residual_mm=scalp_model.p95_residual_mm,
        scalp_superior_radius_ratio=scalp_model.superior_radius_ratio,
        minimum_electrode_separation_mm=minimum_separation_mm,
        maximum_smooth_scalp_gap_mm=maximum_scalp_gap_mm,
        orientation_matrix=np.eye(3),
        orientation_uncertainty_degrees=options.orientation_uncertainty_degrees,
        surface_uncertainty_mm=options.surface_uncertainty_mm,
        anatomical_reference_uncertainty_mm=(options.anatomical_reference.reference_uncertainty_mm),
        head_center_offset_uncertainty_mm=(
            options.anatomical_reference.head_center_offset_uncertainty_mm
        ),
    )


def _project_fiducials(result: TemplateMontageInference) -> dict[str, np.ndarray]:
    inferior_angle = np.deg2rad(25.0)
    horizontal = np.cos(inferior_angle)
    inferior = -np.sin(inferior_angle)
    directions = {
        "nasion": np.array([0.0, horizontal, inferior]),
        "lpa": np.array([-horizontal, 0.0, inferior]),
        "rpa": np.array([horizontal, 0.0, inferior]),
    }
    surface_tree = cKDTree(result.scalp_surface_mri_mm)
    fiducials: dict[str, np.ndarray] = {}
    for name, direction in directions.items():
        predicted = _ellipsoid_contact(
            result.scalp_center_mri_mm,
            result.scalp_radii_mm,
            direction,
        )
        _, index = surface_tree.query(predicted)
        fiducials[name] = result.scalp_surface_mri_mm[int(index)]
    return fiducials


def _write_mne_montage(
    result: TemplateMontageInference,
    output_directory: Path,
    subject_id: str,
) -> tuple[Path, Path]:
    import mne

    fiducials_mri_m = {
        name: position / 1_000.0 for name, position in _project_fiducials(result).items()
    }
    montage = mne.channels.make_dig_montage(
        ch_pos={name: position / 1_000.0 for name, position in result.positions_mri_mm.items()},
        nasion=fiducials_mri_m["nasion"],
        lpa=fiducials_mri_m["lpa"],
        rpa=fiducials_mri_m["rpa"],
        coord_frame="mri",
    )
    scanner_ras_to_head = mne.channels.compute_native_head_t(montage)
    montage.apply_trans(scanner_ras_to_head)
    montage_path = output_directory / f"{subject_id}_desc-templateinferred-dig.fif"
    montage.save(montage_path, overwrite=True)

    matrix = np.asarray(scanner_ras_to_head["trans"], dtype=float)
    transform_path = output_directory / f"{subject_id}_from-scannerRAS_to-head_transform.json"
    transform = {
        "from": "native T1w scanner RAS+ (m)",
        "to": "MNE head (m)",
        "scanner_ras_to_head_matrix": matrix.tolist(),
        "head_to_scanner_ras_matrix": np.linalg.inv(matrix).tolist(),
        "fiducials_scanner_ras_m": {
            name: position.tolist() for name, position in fiducials_mri_m.items()
        },
        "warning": (
            "Fiducials and electrodes are anatomically inferred, not measured. Scanner RAS "
            "must be converted to FreeSurfer surface RAS before constructing an MNE MRI trans."
        ),
    }
    transform_path.write_text(json.dumps(transform, indent=2) + "\n", encoding="utf-8")
    return montage_path, transform_path


def _write_qc_render(
    result: TemplateMontageInference,
    output_directory: Path,
    subject_id: str,
) -> Path:
    import matplotlib.pyplot as plt

    surface = result.scalp_surface_mri_mm
    maximum_surface_points = 25_000
    if surface.shape[0] > maximum_surface_points:
        indices = np.linspace(0, surface.shape[0] - 1, maximum_surface_points, dtype=int)
        surface = surface[indices]
    names = list(result.positions_mri_mm)
    contacts = np.stack([result.positions_mri_mm[name] for name in names])
    uncertainty = np.asarray([result.positional_uncertainty_mm[name] for name in names])

    figure, axes = plt.subplots(1, 3, figsize=(16, 5.5), constrained_layout=True)
    views = (
        ("Superior", 0, 1, "R → (mm)", "A → (mm)"),
        ("Anterior", 0, 2, "R → (mm)", "S → (mm)"),
        ("Right", 1, 2, "A → (mm)", "S → (mm)"),
    )
    artist = None
    for axis, (title, horizontal_axis, vertical_axis, horizontal_label, vertical_label) in zip(
        axes, views, strict=True
    ):
        axis.scatter(
            surface[:, horizontal_axis],
            surface[:, vertical_axis],
            c="#a7adb4",
            s=0.45,
            alpha=0.16,
            rasterized=True,
        )
        artist = axis.scatter(
            contacts[:, horizontal_axis],
            contacts[:, vertical_axis],
            c=uncertainty,
            cmap="magma",
            s=30,
            edgecolors="black",
            linewidths=0.4,
            zorder=3,
        )
        for name, contact in zip(names, contacts, strict=True):
            axis.annotate(
                name,
                (contact[horizontal_axis], contact[vertical_axis]),
                xytext=(2, 2),
                textcoords="offset points",
                fontsize=5,
                zorder=4,
            )
        axis.set_title(title)
        axis.set_xlabel(horizontal_label)
        axis.set_ylabel(vertical_label)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(color="#d8dce1", linewidth=0.4, alpha=0.7)
    if artist is not None:
        colorbar = figure.colorbar(artist, ax=axes, shrink=0.75, pad=0.02)
        colorbar.set_label("Sensitivity envelope (mm)")
    figure.suptitle(
        f"{subject_id} anatomical template montage · inferred, not measured",
        fontsize=11,
    )
    qc_path = output_directory / f"{subject_id}_space-T1w_desc-templateinferred_qc.png"
    figure.savefig(qc_path, dpi=220)
    plt.close(figure)
    return qc_path


def write_template_montage_outputs(
    result: TemplateMontageInference,
    output_directory: str | Path,
    subject_id: str,
    t1w_path: str | Path,
) -> TemplateMontageOutputPaths:
    """Write coordinates, provenance, uncertainty, montage, transform, and QC."""
    if not subject_id.startswith("sub-"):
        raise ValueError("subject_id must use the BIDS 'sub-' prefix.")
    if not result.positions_mri_mm:
        raise ValueError("Cannot export an empty template montage inference.")
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    electrodes_path = output_path / f"{subject_id}_space-T1w_desc-templateinferred_electrodes.tsv"
    rows = ["name\tx\ty\tz\tpositional_uncertainty_mm\testimation_method"]
    for name, position in result.positions_mri_mm.items():
        rows.append(
            f"{name}\t{position[0]:.6f}\t{position[1]:.6f}\t{position[2]:.6f}\t"
            f"{result.positional_uncertainty_mm[name]:.6f}\tanatomical-template-projection"
        )
    electrodes_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    coordinate_system_path = output_path / (
        f"{subject_id}_space-T1w_desc-templateinferred_coordsystem.json"
    )
    coordinate_system = {
        "EEGCoordinateSystem": "Other",
        "EEGCoordinateSystemDescription": (
            "Native T1w NIfTI scanner RAS+ coordinates. Positions are an anatomical "
            "projection of a standard montage and are not measured electrode locations."
        ),
        "EEGCoordinateUnits": "mm",
    }
    coordinate_system_path.write_text(
        json.dumps(coordinate_system, indent=2) + "\n",
        encoding="utf-8",
    )

    diagnostics_path = output_path / (f"{subject_id}_space-T1w_desc-templateinferred_qc.json")
    diagnostics = {
        "subject_id": subject_id,
        "source_t1w": str(t1w_path),
        "method": "anatomical scalp projection of a standard montage",
        "measurement_status": "inferred, not measured",
        "orientation_assumption": "canonical T1w scanner RAS axes approximate head RAS axes",
        "orientation_uncertainty_degrees": result.orientation_uncertainty_degrees,
        "surface_uncertainty_mm": result.surface_uncertainty_mm,
        "anatomical_reference_mri_ras_mm": result.anatomical_reference_mri_mm.tolist(),
        "anatomical_reference_candidate_volume_mm3": (
            result.anatomical_reference_candidate_volume_mm3
        ),
        "anatomical_reference_candidate_spans_mm": (
            result.anatomical_reference_candidate_spans_mm.tolist()
        ),
        "uncertainty_components_mm": {
            "surface": result.surface_uncertainty_mm,
            "anatomical_reference": result.anatomical_reference_uncertainty_mm,
            "head_center_offset": result.head_center_offset_uncertainty_mm,
        },
        "uncertainty_definition": (
            "Maximum displacement over independent positive and negative pitch, roll, and "
            "yaw perturbations, combined in quadrature with scalp-surface, anatomical-"
            "reference, and head-center-offset uncertainty."
        ),
        "fitted_scalp_center_mri_ras_mm": result.scalp_center_mri_mm.tolist(),
        "head_center_offset_mri_mm": (
            result.scalp_center_mri_mm - result.anatomical_reference_mri_mm
        ).tolist(),
        "fitted_scalp_radii_mm": result.scalp_radii_mm.tolist(),
        "scalp_fit": {
            "median_residual_mm": result.scalp_fit_median_residual_mm,
            "p95_residual_mm": result.scalp_fit_p95_residual_mm,
            "superior_radius_ratio": result.scalp_superior_radius_ratio,
        },
        "geometry_qc": {
            "minimum_electrode_separation_mm": (result.minimum_electrode_separation_mm),
            "maximum_smooth_scalp_gap_mm": result.maximum_smooth_scalp_gap_mm,
        },
        "electrodes": {
            name: {
                "position_mri_ras_mm": position.tolist(),
                "positional_uncertainty_mm": result.positional_uncertainty_mm[name],
                "orientation_sensitivity_mm": result.orientation_sensitivity_mm[name],
            }
            for name, position in result.positions_mri_mm.items()
        },
    }
    diagnostics_path.write_text(
        json.dumps(diagnostics, indent=2) + "\n",
        encoding="utf-8",
    )
    montage_path, transform_path = _write_mne_montage(result, output_path, subject_id)
    qc_path = _write_qc_render(result, output_path, subject_id)
    return TemplateMontageOutputPaths(
        electrodes_tsv=electrodes_path,
        coordinate_system_json=coordinate_system_path,
        diagnostics_json=diagnostics_path,
        qc_png=qc_path,
        mne_head_montage_fif=montage_path,
        scanner_ras_to_head_json=transform_path,
    )


__all__ = [
    "TemplateMontageInference",
    "TemplateMontageOutputPaths",
    "TemplateProjectionParameters",
    "infer_template_montage_from_array",
    "write_template_montage_outputs",
]
