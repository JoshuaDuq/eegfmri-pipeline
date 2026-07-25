"""Coordinate and visual QC outputs for T1 electrode localization."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from studies.pain_study.scripts.t1_electrode_localization import ElectrodeLocalization


def _ellipsoid_point(
    center: np.ndarray,
    radii: np.ndarray,
    direction: np.ndarray,
) -> np.ndarray:
    unit_direction = direction / np.linalg.norm(direction)
    radius = 1.0 / np.sqrt(np.sum((unit_direction / radii) ** 2))
    return center + radius * unit_direction


def write_coordinate_files(
    result: ElectrodeLocalization,
    output_directory: str | Path,
    subject_id: str,
    t1w_path: str | Path,
) -> tuple[Path, Path, Path]:
    """Write BIDS-shaped coordinates and explicit T1 localization diagnostics."""
    if not subject_id.startswith("sub-"):
        raise ValueError("subject_id must use the BIDS 'sub-' prefix.")
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    electrode_names = list(result.positions_mri_mm)
    if not electrode_names:
        raise ValueError("Cannot export an empty electrode localization.")

    electrodes_path = output_path / f"{subject_id}_space-T1w_electrodes.tsv"
    lines = ["name\tx\ty\tz"]
    for name in electrode_names:
        coordinates = np.asarray(result.positions_mri_mm[name], dtype=float)
        if coordinates.shape != (3,) or not np.isfinite(coordinates).all():
            raise ValueError(f"Electrode {name!r} has invalid MRI coordinates.")
        lines.append(
            f"{name}\t{coordinates[0]:.6f}\t{coordinates[1]:.6f}\t" f"{coordinates[2]:.6f}"
        )
    electrodes_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    coordinate_system_path = output_path / f"{subject_id}_space-T1w_coordsystem.json"
    coordinate_system = {
        "EEGCoordinateSystem": "Other",
        "EEGCoordinateSystemDescription": (
            "Native T1w NIfTI scanner RAS+ coordinates. Electrode centers were inferred "
            "from full-head T1w scalp-surface artifacts and projected to the fitted "
            "local scalp surface; they are not optical digitizer measurements."
        ),
        "EEGCoordinateUnits": "mm",
    }
    coordinate_system_path.write_text(
        json.dumps(coordinate_system, indent=2) + "\n",
        encoding="utf-8",
    )

    diagnostics_path = output_path / f"{subject_id}_space-T1w_desc-localization_qc.json"
    diagnostics = {
        "subject_id": subject_id,
        "source_t1w": str(t1w_path),
        "method": "full-head T1w scalp-surface artifact localization",
        "cap_rotation_matrix": result.cap_rotation_matrix.tolist(),
        "fitted_scalp_center_mri_ras_mm": result.scalp_center_mri_mm.tolist(),
        "fitted_scalp_radii_mm": result.scalp_radii_mm.tolist(),
        "electrodes": {
            name: {
                "artifact_position_mri_ras_mm": result.artifact_positions_mri_mm[name].tolist(),
                "contact_position_mri_ras_mm": result.positions_mri_mm[name].tolist(),
                "confidence": result.confidence[name],
                "evidence_z": result.evidence_z[name],
                "prominence_mm": result.prominence_mm[name],
                "signed_prominence_mm": result.signed_prominence_mm[name],
                "artifact_morphology": (
                    "protrusion" if result.signed_prominence_mm[name] > 0.0 else "depression"
                ),
                "registered_template_displacement_mm": result.template_displacement_mm[name],
            }
            for name in electrode_names
        },
    }
    diagnostics_path.write_text(
        json.dumps(diagnostics, indent=2) + "\n",
        encoding="utf-8",
    )
    return electrodes_path, coordinate_system_path, diagnostics_path


def write_mne_head_montage(
    result: ElectrodeLocalization,
    output_directory: str | Path,
    subject_id: str,
) -> tuple[Path, Path]:
    """Write an inferred MNE head-frame montage and its scanner-RAS transform."""
    import mne

    rotation = np.asarray(result.cap_rotation_matrix, dtype=float)
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise ValueError("Cap rotation must be a finite three-by-three matrix.")
    center = np.asarray(result.scalp_center_mri_mm, dtype=float)
    radii = np.asarray(result.scalp_radii_mm, dtype=float)
    fiducial_directions = {
        "nasion": np.array([0.0, 1.0, 0.0]),
        "lpa": np.array([-1.0, 0.0, 0.0]),
        "rpa": np.array([1.0, 0.0, 0.0]),
    }
    fiducials_mri_m = {
        name: _ellipsoid_point(center, radii, rotation @ direction) / 1_000.0
        for name, direction in fiducial_directions.items()
    }
    channels_mri_m = {
        name: np.asarray(position, dtype=float) / 1_000.0
        for name, position in result.positions_mri_mm.items()
    }
    scanner_ras_montage = mne.channels.make_dig_montage(
        ch_pos=channels_mri_m,
        nasion=fiducials_mri_m["nasion"],
        lpa=fiducials_mri_m["lpa"],
        rpa=fiducials_mri_m["rpa"],
        coord_frame="mri",
    )
    scanner_ras_to_head = mne.channels.compute_native_head_t(scanner_ras_montage)
    head_montage = scanner_ras_montage.copy()
    head_montage.apply_trans(scanner_ras_to_head)

    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    montage_path = output_path / f"{subject_id}_desc-t1-inferred-dig.fif"
    head_montage.save(montage_path, overwrite=True)

    transform_path = output_path / f"{subject_id}_from-scannerRAS_to-head_transform.json"
    transform = {
        "from": "native T1w scanner RAS+ (m)",
        "to": "MNE head (m)",
        "matrix": np.asarray(scanner_ras_to_head["trans"], dtype=float).tolist(),
        "inferred_fiducials_scanner_ras_m": {
            name: position.tolist() for name, position in fiducials_mri_m.items()
        },
        "warning": (
            "Fiducials are inferred from the cap-registered scalp ellipsoid. This is "
            "not a scanner-RAS to FreeSurfer surface-RAS transform."
        ),
    }
    transform_path.write_text(
        json.dumps(transform, indent=2) + "\n",
        encoding="utf-8",
    )
    return montage_path, transform_path


def write_qc_render(
    result: ElectrodeLocalization,
    output_directory: str | Path,
    subject_id: str,
) -> Path:
    """Render exterior scalp views with inferred contacts and artifact locations."""
    import matplotlib.pyplot as plt

    surface = np.asarray(result.scalp_surface_mri_mm, dtype=float)
    if surface.ndim != 2 or surface.shape[1] != 3 or surface.shape[0] < 6:
        raise ValueError("QC rendering requires at least six three-dimensional scalp points.")
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    qc_path = output_path / f"{subject_id}_space-T1w_desc-electrode-localization_qc.png"

    maximum_surface_points = 25_000
    if surface.shape[0] > maximum_surface_points:
        indices = np.linspace(
            0,
            surface.shape[0] - 1,
            maximum_surface_points,
            dtype=int,
        )
        displayed_surface = surface[indices]
    else:
        displayed_surface = surface
    names = list(result.positions_mri_mm)
    contacts = np.stack([result.positions_mri_mm[name] for name in names])
    artifacts = np.stack([result.artifact_positions_mri_mm[name] for name in names])
    evidence = np.asarray([result.evidence_z[name] for name in names])

    figure, axes = plt.subplots(1, 3, figsize=(16, 5.5), constrained_layout=True)
    views = (
        ("Superior", 0, 1, "R → (mm)", "A → (mm)"),
        ("Anterior", 0, 2, "R → (mm)", "S → (mm)"),
        ("Right", 1, 2, "A → (mm)", "S → (mm)"),
    )
    contact_artist = None
    for axis, (title, horizontal, vertical, horizontal_label, vertical_label) in zip(
        axes, views, strict=True
    ):
        axis.scatter(
            displayed_surface[:, horizontal],
            displayed_surface[:, vertical],
            c="#a7adb4",
            s=0.45,
            alpha=0.16,
            rasterized=True,
        )
        contact_artist = axis.scatter(
            contacts[:, horizontal],
            contacts[:, vertical],
            c=evidence,
            cmap="viridis",
            s=28,
            edgecolors="black",
            linewidths=0.4,
            zorder=3,
        )
        axis.scatter(
            artifacts[:, horizontal],
            artifacts[:, vertical],
            c="#d62728",
            marker="x",
            s=20,
            linewidths=0.7,
            zorder=4,
        )
        for name, contact in zip(names, contacts, strict=True):
            axis.annotate(
                name,
                (contact[horizontal], contact[vertical]),
                xytext=(2, 2),
                textcoords="offset points",
                fontsize=5,
                color="black",
                zorder=5,
            )
        for contact, artifact in zip(contacts, artifacts, strict=True):
            axis.plot(
                [contact[horizontal], artifact[horizontal]],
                [contact[vertical], artifact[vertical]],
                color="#d62728",
                linewidth=0.45,
                zorder=2,
            )
        axis.set_title(title)
        axis.set_xlabel(horizontal_label)
        axis.set_ylabel(vertical_label)
        axis.set_aspect("equal", adjustable="box")
        axis.grid(color="#d8dce1", linewidth=0.4, alpha=0.7)
    if contact_artist is not None:
        colorbar = figure.colorbar(contact_artist, ax=axes, shrink=0.75, pad=0.02)
        colorbar.set_label("Artifact evidence (robust z)")
    figure.suptitle(
        f"{subject_id} T1w electrode localization · contacts colored by evidence z · "
        "red × = observed artifact",
        fontsize=11,
    )
    figure.savefig(qc_path, dpi=220)
    plt.close(figure)
    return qc_path


__all__ = ["write_coordinate_files", "write_mne_head_montage", "write_qc_render"]
