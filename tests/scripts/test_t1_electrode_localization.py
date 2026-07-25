from __future__ import annotations

import json
from collections.abc import Mapping

import nibabel as nib
import mne
import numpy as np
import pytest
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

import studies.pain_study.scripts.t1_electrode_localization as localization_module
import studies.pain_study.scripts.run_t1_electrode_localization as localization_runner
from studies.pain_study.scripts.t1_anatomical_reference import (
    AnatomicalReferenceParameters,
)
from studies.pain_study.scripts.run_t1_template_montage import (
    load_template_projection_configuration,
    run_template_participant,
)
from studies.pain_study.scripts.t1_electrode_localization import (
    ElectrodeLocalization,
    LocalizationParameters,
    assess_electrodes_from_array,
    localize_electrodes_from_array,
)
from studies.pain_study.scripts.t1_electrode_localization_inputs import (
    RunConfiguration,
    SubjectLocalizationInput,
    discover_eeg_channel_names,
    load_canonical_t1,
    load_run_configuration,
    make_template_positions,
)
from studies.pain_study.scripts.t1_electrode_localization_outputs import (
    write_coordinate_files,
    write_mne_head_montage,
    write_qc_render,
)
from studies.pain_study.scripts.t1_template_montage import (
    TemplateProjectionParameters,
    infer_template_montage_from_array,
    write_template_montage_outputs,
)
from studies.pain_study.scripts.run_t1_electrode_localization import (
    ParticipantBatchError,
    run_configuration,
    run_participant,
)


def _ellipsoid_radius(direction: np.ndarray, radii_mm: np.ndarray) -> float:
    unit_direction = direction / np.linalg.norm(direction)
    return float(1.0 / np.sqrt(np.sum((unit_direction / radii_mm) ** 2)))


def _synthetic_t1(
    template_positions: Mapping[str, np.ndarray],
    artifact_rotation: np.ndarray | None = None,
    artifact_names: set[str] | None = None,
    radii_mm: np.ndarray | None = None,
    artifact_direction_overrides: Mapping[str, np.ndarray] | None = None,
    broad_bulge_direction: np.ndarray | None = None,
    broad_bulge_height_mm: float = 0.0,
    artifact_modes: Mapping[str, str] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    shape = (144, 160, 160)
    center_mm = np.array([72.0, 76.0, 70.0])
    head_radii_mm = np.array([58.0, 66.0, 70.0]) if radii_mm is None else radii_mm
    coordinates = np.indices(shape, dtype=float)
    centered = np.moveaxis(coordinates, 0, -1) - center_mm
    voxel_radius = np.linalg.norm(centered, axis=-1)
    voxel_directions = np.zeros_like(centered)
    nonzero = voxel_radius > 0.0
    voxel_directions[nonzero] = centered[nonzero] / voxel_radius[nonzero, np.newaxis]
    surface_radius = np.zeros_like(voxel_radius)
    surface_radius[nonzero] = 1.0 / np.sqrt(
        np.sum((voxel_directions[nonzero] / head_radii_mm) ** 2, axis=-1)
    )
    bulge_direction = None
    deformation = np.zeros_like(voxel_radius)
    if broad_bulge_direction is not None:
        bulge_direction = broad_bulge_direction / np.linalg.norm(broad_bulge_direction)
        angular_distance = np.arccos(
            np.clip(np.sum(voxel_directions * bulge_direction, axis=-1), -1.0, 1.0)
        )
        deformation = broad_bulge_height_mm * np.exp(-0.5 * (angular_distance / 0.30) ** 2)

    random = np.random.default_rng(42)
    bias = 1.0 + 0.18 * (coordinates[1] / shape[1] - 0.5)
    data = random.normal(0.0, 1.0, size=shape)
    inside_head = voxel_radius <= surface_radius + deformation
    data[inside_head] += 110.0 * bias[inside_head]
    brain_center_mm = center_mm - np.array([0.0, 7.0, -5.0])
    brain_radii_mm = np.array([43.0, 50.0, 48.0])
    inside_brain = (
        np.sum(
            ((np.moveaxis(coordinates, 0, -1) - brain_center_mm) / brain_radii_mm) ** 2,
            axis=-1,
        )
        <= 1.0
    )
    data[inside_brain] += 70.0

    expected_contacts: dict[str, np.ndarray] = {}
    rotation = np.eye(3) if artifact_rotation is None else artifact_rotation
    for name, position in template_positions.items():
        direction = rotation @ (position / np.linalg.norm(position))
        if artifact_direction_overrides is not None and name in artifact_direction_overrides:
            override = artifact_direction_overrides[name]
            direction = override / np.linalg.norm(override)
        contact_radius = _ellipsoid_radius(direction, head_radii_mm)
        if bulge_direction is not None:
            angle = np.arccos(np.clip(np.dot(direction, bulge_direction), -1.0, 1.0))
            contact_radius += broad_bulge_height_mm * np.exp(-0.5 * (angle / 0.30) ** 2)
        contact = center_mm + contact_radius * direction
        expected_contacts[name] = contact

        if artifact_names is not None and name not in artifact_names:
            continue
        mode = "protrusion" if artifact_modes is None else artifact_modes.get(name, "protrusion")
        if mode == "protrusion":
            artifact_center = contact + 2.5 * direction
            distance = np.linalg.norm(centered - (artifact_center - center_mm), axis=-1)
            data[distance <= 4.5] += 110.0
        elif mode == "depression":
            artifact_center = contact - 2.5 * direction
            distance = np.linalg.norm(centered - (artifact_center - center_mm), axis=-1)
            data[distance <= 4.5] = 0.0
        else:
            raise ValueError(f"Unknown synthetic artifact mode: {mode}")

    data[5:7, 5:7, 5:7] = 150.0
    return data.astype(np.float32), np.eye(4), expected_contacts


def test_localization_recovers_synthetic_scalp_contacts() -> None:
    template_positions = {
        "F3": np.array([-0.45, 0.55, 0.70]),
        "F4": np.array([0.45, 0.55, 0.70]),
        "C3": np.array([-0.65, 0.00, 0.76]),
        "C4": np.array([0.65, 0.00, 0.76]),
        "P3": np.array([-0.45, -0.55, 0.70]),
        "P4": np.array([0.45, -0.55, 0.70]),
        "Cz": np.array([0.00, 0.00, 1.00]),
    }
    data, affine, expected_contacts = _synthetic_t1(template_positions)

    result = localize_electrodes_from_array(
        data,
        affine,
        template_positions,
        LocalizationParameters(
            candidate_radius_mm=16.0,
            min_evidence_z=1.0,
        ),
    )

    errors_mm = np.array(
        [
            np.linalg.norm(result.positions_mri_mm[name] - expected_contacts[name])
            for name in template_positions
        ]
    )
    assert np.median(errors_mm) < 3.0
    assert np.max(errors_mm) < 5.0
    assert set(result.positions_mri_mm) == set(template_positions)
    assert all(result.confidence[name] >= 0.0 for name in template_positions)


def test_localization_rejects_reusing_one_artifact_for_two_channels() -> None:
    template_positions = {
        "C1": np.array([-0.08, 0.00, 1.00]),
        "C2": np.array([0.08, 0.00, 1.00]),
    }
    data, affine, _ = _synthetic_t1(
        template_positions,
        artifact_names={"C1"},
    )

    with pytest.raises(RuntimeError, match="one-to-one"):
        localize_electrodes_from_array(
            data,
            affine,
            template_positions,
            LocalizationParameters(
                candidate_radius_mm=16.0,
                min_evidence_z=1.0,
            ),
        )


def test_localization_registers_a_rotated_electrode_cap() -> None:
    template_positions = {
        "Fp1": np.array([-0.30, 0.78, 0.55]),
        "Fp2": np.array([0.30, 0.78, 0.55]),
        "F3": np.array([-0.45, 0.55, 0.70]),
        "F4": np.array([0.45, 0.55, 0.70]),
        "C3": np.array([-0.65, 0.00, 0.76]),
        "C4": np.array([0.65, 0.00, 0.76]),
        "P3": np.array([-0.45, -0.55, 0.70]),
        "P4": np.array([0.45, -0.55, 0.70]),
        "O1": np.array([-0.28, -0.82, 0.50]),
        "O2": np.array([0.28, -0.82, 0.50]),
        "Cz": np.array([0.00, 0.00, 1.00]),
    }
    artifact_rotation = Rotation.from_euler("xyz", [8.0, -11.0, 7.0], degrees=True).as_matrix()
    data, affine, expected_contacts = _synthetic_t1(
        template_positions,
        artifact_rotation,
    )

    result = localize_electrodes_from_array(
        data,
        affine,
        template_positions,
        LocalizationParameters(
            candidate_radius_mm=18.0,
            min_evidence_z=1.0,
        ),
    )

    errors_mm = np.array(
        [
            np.linalg.norm(result.positions_mri_mm[name] - expected_contacts[name])
            for name in template_positions
        ]
    )
    assert np.median(errors_mm) < 3.0
    assert np.max(errors_mm) < 5.0


def test_localization_rejects_a_cap_rotation_beyond_the_configured_limit() -> None:
    template_positions = {"Cz": np.array([0.00, 0.00, 1.00])}
    rotation_axis = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
    artifact_rotation = Rotation.from_rotvec(np.deg2rad(22.0) * rotation_axis).as_matrix()
    data, affine, _ = _synthetic_t1(template_positions, artifact_rotation)

    with pytest.raises(RuntimeError, match="maximum cap rotation"):
        localize_electrodes_from_array(
            data,
            affine,
            template_positions,
            LocalizationParameters(
                candidate_radius_mm=35.0,
                maximum_contact_displacement_mm=30.0,
                max_rotation_degrees=20.0,
                min_evidence_z=1.0,
            ),
        )


def test_localization_rejects_a_brain_extracted_image() -> None:
    template_positions = {"Cz": np.array([0.00, 0.00, 1.00])}
    data, affine, _ = _synthetic_t1(
        template_positions,
        radii_mm=np.array([45.0, 52.0, 54.0]),
    )

    with pytest.raises(ValueError, match="full-head extent"):
        localize_electrodes_from_array(
            data,
            affine,
            template_positions,
            LocalizationParameters(min_evidence_z=1.0),
        )


def test_localization_rejects_a_channel_without_a_discrete_scalp_artifact() -> None:
    template_positions = {"Cz": np.array([0.00, 0.00, 1.00])}
    data, affine, _ = _synthetic_t1(template_positions, artifact_names=set())

    with pytest.raises(RuntimeError, match="minimum prominence"):
        localize_electrodes_from_array(
            data,
            affine,
            template_positions,
            LocalizationParameters(min_evidence_z=1.0),
        )


def test_assessment_returns_candidates_without_weak_evidence_acceptance() -> None:
    template_positions = {"Cz": np.array([0.00, 0.00, 1.00])}
    data, affine, _ = _synthetic_t1(template_positions, artifact_names=set())

    result = assess_electrodes_from_array(
        data,
        affine,
        template_positions,
        LocalizationParameters(min_evidence_z=10.0),
    )

    assert set(result.positions_mri_mm) == {"Cz"}


def test_localization_rejects_an_artifact_that_breaks_cap_topology() -> None:
    template_positions = {
        "F3": np.array([-0.45, 0.55, 0.70]),
        "F4": np.array([0.45, 0.55, 0.70]),
        "C3": np.array([-0.65, 0.00, 0.76]),
        "C4": np.array([0.65, 0.00, 0.76]),
        "P3": np.array([-0.45, -0.55, 0.70]),
        "P4": np.array([0.45, -0.55, 0.70]),
        "Cz": np.array([0.00, 0.00, 1.00]),
    }
    displaced_f3 = Rotation.from_euler("z", 16.0, degrees=True).apply(template_positions["F3"])
    data, affine, _ = _synthetic_t1(
        template_positions,
        artifact_direction_overrides={"F3": displaced_f3},
    )

    with pytest.raises(RuntimeError, match="cap-topology displacement"):
        localize_electrodes_from_array(
            data,
            affine,
            template_positions,
            LocalizationParameters(
                candidate_radius_mm=18.0,
                min_evidence_z=1.0,
                maximum_contact_displacement_mm=10.0,
            ),
        )


def test_contact_projection_follows_the_smooth_local_scalp_surface() -> None:
    template_positions = {
        "F3": np.array([-0.45, 0.55, 0.70]),
        "F4": np.array([0.45, 0.55, 0.70]),
        "C3": np.array([-0.65, 0.00, 0.76]),
        "C4": np.array([0.65, 0.00, 0.76]),
        "P3": np.array([-0.45, -0.55, 0.70]),
        "P4": np.array([0.45, -0.55, 0.70]),
        "Cz": np.array([0.00, 0.00, 1.00]),
    }
    data, affine, expected_contacts = _synthetic_t1(
        template_positions,
        broad_bulge_direction=template_positions["F3"],
        broad_bulge_height_mm=6.0,
    )

    result = localize_electrodes_from_array(
        data,
        affine,
        template_positions,
        LocalizationParameters(
            candidate_radius_mm=18.0,
            min_evidence_z=1.0,
        ),
    )

    error_mm = np.linalg.norm(result.positions_mri_mm["F3"] - expected_contacts["F3"])
    assert error_mm < 3.0


def test_localization_recovers_an_inward_electrode_depression() -> None:
    template_positions = {
        "F3": np.array([-0.45, 0.55, 0.70]),
        "F4": np.array([0.45, 0.55, 0.70]),
        "C3": np.array([-0.65, 0.00, 0.76]),
        "C4": np.array([0.65, 0.00, 0.76]),
        "P3": np.array([-0.45, -0.55, 0.70]),
        "P4": np.array([0.45, -0.55, 0.70]),
        "Cz": np.array([0.00, 0.00, 1.00]),
    }
    data, affine, expected_contacts = _synthetic_t1(
        template_positions,
        artifact_modes={"Cz": "depression"},
    )

    result = localize_electrodes_from_array(
        data,
        affine,
        template_positions,
        LocalizationParameters(candidate_radius_mm=16.0, min_evidence_z=1.0),
    )

    error_mm = np.linalg.norm(result.positions_mri_mm["Cz"] - expected_contacts["Cz"])
    center = np.array([72.0, 76.0, 70.0])
    assert error_mm < 3.0
    assert result.signed_prominence_mm["Cz"] < 0.0
    assert np.linalg.norm(result.artifact_positions_mri_mm["Cz"] - center) < np.linalg.norm(
        result.positions_mri_mm["Cz"] - center
    )


def test_coordinate_export_declares_native_t1_mri_ras_space(tmp_path) -> None:
    result = ElectrodeLocalization(
        positions_mri_mm={"Cz": np.array([1.25, -2.5, 93.0])},
        artifact_positions_mri_mm={"Cz": np.array([1.5, -2.0, 96.0])},
        confidence={"Cz": 0.88},
        evidence_z={"Cz": 5.2},
        prominence_mm={"Cz": 3.1},
        signed_prominence_mm={"Cz": 3.1},
        template_displacement_mm={"Cz": 1.4},
        cap_rotation_matrix=np.eye(3),
        scalp_surface_mri_mm=np.array(
            [
                [-60.0, 0.0, 0.0],
                [60.0, 0.0, 0.0],
                [0.0, -70.0, 0.0],
                [0.0, 70.0, 0.0],
                [0.0, 0.0, -75.0],
                [0.0, 0.0, 95.0],
            ]
        ),
        scalp_center_mri_mm=np.array([0.0, 0.0, 10.0]),
        scalp_radii_mm=np.array([60.0, 70.0, 85.0]),
    )

    electrodes_path, coordinate_system_path, diagnostics_path = write_coordinate_files(
        result=result,
        output_directory=tmp_path,
        subject_id="sub-0014",
        t1w_path="/data/sub-0014_desc-preproc_T1w.nii.gz",
    )

    rows = electrodes_path.read_text(encoding="utf-8").splitlines()
    assert rows[0] == "name\tx\ty\tz"
    assert rows[1] == "Cz\t1.250000\t-2.500000\t93.000000"
    coordinate_system = json.loads(coordinate_system_path.read_text(encoding="utf-8"))
    assert coordinate_system["EEGCoordinateSystem"] == "Other"
    assert coordinate_system["EEGCoordinateUnits"] == "mm"
    assert "scanner RAS+" in coordinate_system["EEGCoordinateSystemDescription"]
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    assert diagnostics["source_t1w"] == "/data/sub-0014_desc-preproc_T1w.nii.gz"
    assert diagnostics["electrodes"]["Cz"]["evidence_z"] == 5.2
    assert diagnostics["electrodes"]["Cz"]["artifact_morphology"] == "protrusion"

    montage_path, transform_path = write_mne_head_montage(
        result,
        tmp_path,
        "sub-0014",
    )
    montage = mne.channels.read_dig_fif(montage_path)
    assert montage.get_positions()["coord_frame"] == "head"
    assert set(montage.get_positions()["ch_pos"]) == {"Cz"}
    transform = json.loads(transform_path.read_text(encoding="utf-8"))
    assert transform["from"] == "native T1w scanner RAS+ (m)"
    assert transform["to"] == "MNE head (m)"

    qc_path = write_qc_render(result, tmp_path, "sub-0014")
    assert qc_path.name == "sub-0014_space-T1w_desc-electrode-localization_qc.png"
    assert qc_path.stat().st_size > 1_000


def test_channel_discovery_uses_only_consistent_eeg_channels(tmp_path) -> None:
    first = tmp_path / "ses-01" / "eeg" / "sub-0014_ses-01_task-pain_channels.tsv"
    second = tmp_path / "ses-02" / "eeg" / "sub-0014_ses-02_task-pain_channels.tsv"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    contents = "name\ttype\nFp1\tEEG\nFp2\tEEG\nECG\tECG\n"
    first.write_text(contents, encoding="utf-8")
    second.write_text(contents, encoding="utf-8")
    (first.parent / f"._{first.name}").write_bytes(b"\x00\x05\x16\x07\x00\x02\x00\x00Mac OS X\xb0")

    assert discover_eeg_channel_names(tmp_path) == ("Fp1", "Fp2")

    second.write_text("name\ttype\nFp1\tEEG\nF3\tEEG\n", encoding="utf-8")
    with pytest.raises(ValueError, match="identical EEG channel sets"):
        discover_eeg_channel_names(tmp_path)


def test_template_positions_match_the_recorded_easycap_channels() -> None:
    positions = make_template_positions(("Fp1", "Cz", "O2"), "easycap-M1")

    assert tuple(positions) == ("Fp1", "Cz", "O2")
    assert all(position.shape == (3,) for position in positions.values())
    assert all(np.isfinite(position).all() for position in positions.values())


def test_t1_loader_reorients_to_canonical_scanner_ras(tmp_path) -> None:
    data = np.ones((120, 140, 150), dtype=np.float32)
    affine = np.array(
        [
            [-1.0, 0.0, 0.0, 60.0],
            [0.0, 1.0, 0.0, -70.0],
            [0.0, 0.0, 1.0, -75.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    t1w_path = tmp_path / "sub-0014_T1w.nii.gz"
    nib.save(nib.Nifti1Image(data, affine), t1w_path)

    canonical_data, canonical_affine = load_canonical_t1(t1w_path)

    assert canonical_data.shape == data.shape
    assert nib.aff2axcodes(canonical_affine) == ("R", "A", "S")
    assert np.allclose(nib.affines.voxel_sizes(canonical_affine), 1.0)


def test_run_configuration_resolves_paths_and_parameters(tmp_path) -> None:
    config_path = tmp_path / "localization.yaml"
    config_path.write_text(
        "\n".join(
            [
                "output_root: results",
                "montage: easycap-M1",
                "parameters:",
                "  min_evidence_z: 3.0",
                "participants:",
                "  sub-0014:",
                "    t1w: inputs/sub-0014_T1w.nii.gz",
                "    eeg_bids_subject_directory: inputs/sub-0014",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    configuration = load_run_configuration(config_path)

    assert configuration.output_root == tmp_path / "results"
    assert configuration.montage_name == "easycap-M1"
    assert configuration.parameters.min_evidence_z == 3.0
    assert configuration.participants[0].subject_id == "sub-0014"
    assert configuration.participants[0].t1w_path == (tmp_path / "inputs" / "sub-0014_T1w.nii.gz")


def test_participant_runner_writes_coordinates_diagnostics_and_render(tmp_path) -> None:
    channel_names = ("F3", "F4", "C3", "C4", "P3", "P4", "Cz")
    template_positions = make_template_positions(channel_names, "easycap-M1")
    data, affine, _ = _synthetic_t1(template_positions)
    t1w_path = tmp_path / "input" / "sub-0014_T1w.nii.gz"
    eeg_directory = tmp_path / "input" / "sub-0014" / "eeg"
    t1w_path.parent.mkdir(parents=True)
    eeg_directory.mkdir(parents=True)
    nib.save(nib.Nifti1Image(data, affine), t1w_path)
    channel_lines = ["name\ttype", *[f"{name}\tEEG" for name in channel_names]]
    (eeg_directory / "sub-0014_task-pain_channels.tsv").write_text(
        "\n".join(channel_lines) + "\n",
        encoding="utf-8",
    )

    paths = run_participant(
        SubjectLocalizationInput(
            subject_id="sub-0014",
            t1w_path=t1w_path,
            eeg_bids_subject_directory=eeg_directory.parent,
        ),
        output_root=tmp_path / "outputs",
        montage_name="easycap-M1",
        parameters=LocalizationParameters(
            candidate_radius_mm=16.0,
            min_evidence_z=1.0,
        ),
    )

    assert paths.electrodes_tsv.is_file()
    assert paths.coordinate_system_json.is_file()
    assert paths.diagnostics_json.is_file()
    assert paths.qc_png.is_file()
    assert paths.mne_head_montage_fif.is_file()
    assert paths.scanner_ras_to_head_json.is_file()


def test_run_configuration_processes_each_participant(monkeypatch, tmp_path) -> None:
    configuration = RunConfiguration(
        output_root=tmp_path / "outputs",
        montage_name="easycap-M1",
        parameters=LocalizationParameters(),
        participants=(
            SubjectLocalizationInput("sub-0014", tmp_path / "14.nii.gz", tmp_path / "14"),
            SubjectLocalizationInput("sub-0015", tmp_path / "15.nii.gz", tmp_path / "15"),
        ),
    )
    processed: list[str] = []

    def record_participant(participant, output_root, montage_name, parameters):
        processed.append(participant.subject_id)
        return participant.subject_id

    monkeypatch.setattr(localization_runner, "run_participant", record_participant)

    outputs = run_configuration(configuration)

    assert processed == ["sub-0014", "sub-0015"]
    assert outputs == {"sub-0014": "sub-0014", "sub-0015": "sub-0015"}


def test_run_configuration_reports_failures_after_attempting_every_participant(
    monkeypatch,
    tmp_path,
) -> None:
    configuration = RunConfiguration(
        output_root=tmp_path / "outputs",
        montage_name="easycap-M1",
        parameters=LocalizationParameters(),
        participants=(
            SubjectLocalizationInput("sub-0014", tmp_path / "14.nii.gz", tmp_path / "14"),
            SubjectLocalizationInput("sub-0015", tmp_path / "15.nii.gz", tmp_path / "15"),
        ),
    )
    processed: list[str] = []

    def fail_first_participant(participant, output_root, montage_name, parameters):
        processed.append(participant.subject_id)
        if participant.subject_id == "sub-0014":
            raise RuntimeError("insufficient MRI evidence")
        return participant.subject_id

    monkeypatch.setattr(
        localization_runner,
        "run_participant",
        fail_first_participant,
    )

    with pytest.raises(ParticipantBatchError, match="sub-0014") as error:
        run_configuration(configuration)

    assert processed == ["sub-0014", "sub-0015"]
    assert error.value.outputs == {"sub-0015": "sub-0015"}
    assert str(error.value.failures["sub-0014"]) == "insufficient MRI evidence"


def test_chunked_neighbor_median_matches_an_unchunked_query() -> None:
    random = np.random.default_rng(91)
    points = random.normal(size=(101, 3))
    values = random.normal(size=101)
    tree = cKDTree(points)
    _, indices = tree.query(points, k=17)
    expected = np.median(values[indices], axis=1)

    observed = localization_module._neighbor_median(
        values,
        points,
        tree,
        neighbor_count=17,
        chunk_size=13,
    )

    assert np.allclose(observed, expected)


def test_template_projection_follows_smooth_scalp_without_artifact_evidence() -> None:
    template_positions = {
        "Fp1": np.array([-0.30, 0.78, 0.55]),
        "Fp2": np.array([0.30, 0.78, 0.55]),
        "F3": np.array([-0.45, 0.55, 0.70]),
        "F4": np.array([0.45, 0.55, 0.70]),
        "C3": np.array([-0.65, 0.00, 0.76]),
        "C4": np.array([0.65, 0.00, 0.76]),
        "P3": np.array([-0.45, -0.55, 0.70]),
        "P4": np.array([0.45, -0.55, 0.70]),
        "O1": np.array([-0.28, -0.82, 0.50]),
        "O2": np.array([0.28, -0.82, 0.50]),
        "Cz": np.array([0.00, 0.00, 1.00]),
    }
    data, affine, expected_contacts = _synthetic_t1(
        template_positions,
        artifact_names=set(),
        broad_bulge_direction=template_positions["F3"],
        broad_bulge_height_mm=6.0,
    )

    result = infer_template_montage_from_array(
        data,
        affine,
        template_positions,
        TemplateProjectionParameters(
            orientation_uncertainty_degrees=8.0,
            anatomical_reference=AnatomicalReferenceParameters(maximum_p95_residual_mm=18.0),
        ),
    )

    errors_mm = np.array(
        [
            np.linalg.norm(result.positions_mri_mm[name] - expected_contacts[name])
            for name in template_positions
        ]
    )
    reported_uncertainty_mm = np.array(
        [result.positional_uncertainty_mm[name] for name in template_positions]
    )
    assert np.median(errors_mm) < 10.0
    assert np.max(errors_mm) < 12.0
    assert np.all(errors_mm <= reported_uncertainty_mm)
    assert set(result.positional_uncertainty_mm) == set(template_positions)
    assert all(value >= 3.0 for value in result.positional_uncertainty_mm.values())
    assert np.allclose(
        result.scalp_center_mri_mm - result.anatomical_reference_mri_mm,
        np.array([0.0, 7.0, -5.0]),
    )
    assert result.scalp_fit_median_residual_mm <= 6.0
    assert result.scalp_fit_p95_residual_mm <= 18.0


def test_template_projection_uncertainty_increases_with_orientation_range() -> None:
    template_positions = {
        "Fpz": np.array([0.00, 1.00, 0.00]),
        "Cz": np.array([0.00, 0.00, 1.00]),
    }
    data, affine, _ = _synthetic_t1(template_positions, artifact_names=set())

    narrow = infer_template_montage_from_array(
        data,
        affine,
        template_positions,
        TemplateProjectionParameters(
            orientation_uncertainty_degrees=4.0,
            anatomical_reference=AnatomicalReferenceParameters(maximum_p95_residual_mm=18.0),
        ),
    )
    wide = infer_template_montage_from_array(
        data,
        affine,
        template_positions,
        TemplateProjectionParameters(
            orientation_uncertainty_degrees=12.0,
            anatomical_reference=AnatomicalReferenceParameters(maximum_p95_residual_mm=18.0),
        ),
    )

    for name in template_positions:
        assert wide.positional_uncertainty_mm[name] > narrow.positional_uncertainty_mm[name]


def test_template_projection_rejects_colliding_channels() -> None:
    template_positions = {
        "C1": np.array([0.00, 0.00, 1.00]),
        "C2": np.array([0.00, 0.00, 1.00]),
    }
    data, affine, _ = _synthetic_t1(template_positions, artifact_names=set())

    with pytest.raises(RuntimeError, match="minimum electrode separation"):
        infer_template_montage_from_array(
            data,
            affine,
            template_positions,
            TemplateProjectionParameters(
                anatomical_reference=AnatomicalReferenceParameters(maximum_p95_residual_mm=18.0)
            ),
        )


def test_template_projection_outputs_are_explicitly_inferential(tmp_path) -> None:
    template_positions = {"Cz": np.array([0.00, 0.00, 1.00])}
    data, affine, _ = _synthetic_t1(template_positions, artifact_names=set())
    result = infer_template_montage_from_array(
        data,
        affine,
        template_positions,
        TemplateProjectionParameters(
            anatomical_reference=AnatomicalReferenceParameters(maximum_p95_residual_mm=18.0)
        ),
    )

    paths = write_template_montage_outputs(
        result,
        tmp_path,
        "sub-0014",
        "/data/sub-0014_T1w.nii.gz",
    )

    header = paths.electrodes_tsv.read_text(encoding="utf-8").splitlines()[0]
    assert header == "name\tx\ty\tz\tpositional_uncertainty_mm\testimation_method"
    assert "templateinferred" in paths.electrodes_tsv.name
    coordinate_system = json.loads(paths.coordinate_system_json.read_text(encoding="utf-8"))
    assert "not measured" in coordinate_system["EEGCoordinateSystemDescription"]
    diagnostics = json.loads(paths.diagnostics_json.read_text(encoding="utf-8"))
    assert diagnostics["method"] == "anatomical scalp projection of a standard montage"
    assert diagnostics["electrodes"]["Cz"]["positional_uncertainty_mm"] >= 3.0
    assert len(diagnostics["anatomical_reference_mri_ras_mm"]) == 3
    assert diagnostics["scalp_fit"]["p95_residual_mm"] <= 18.0
    assert diagnostics["uncertainty_components_mm"]["anatomical_reference"] == 10.0
    assert diagnostics["electrodes"]["Cz"]["orientation_sensitivity_mm"] > 0.0
    assert diagnostics["head_center_offset_mri_mm"] == [0.0, 7.0, -5.0]
    assert diagnostics["geometry_qc"]["minimum_electrode_separation_mm"] is None
    assert diagnostics["geometry_qc"]["maximum_smooth_scalp_gap_mm"] <= 8.0
    assert paths.qc_png.stat().st_size > 1_000
    montage = mne.channels.read_dig_fif(paths.mne_head_montage_fif)
    assert set(montage.get_positions()["ch_pos"]) == {"Cz"}


def test_template_projection_configuration_is_strict_and_resolves_paths(tmp_path) -> None:
    configuration_path = tmp_path / "template_projection.yaml"
    configuration_path.write_text(
        "\n".join(
            [
                "output_root: outputs",
                "montage: easycap-M1",
                "parameters:",
                "  orientation_uncertainty_degrees: 12.0",
                "  anatomical_reference:",
                "    reference_uncertainty_mm: 9.0",
                "participants:",
                "  sub-0014:",
                "    t1w: inputs/sub-0014_T1w.nii.gz",
                "    eeg_bids_subject_directory: inputs/sub-0014",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    configuration = load_template_projection_configuration(configuration_path)

    assert configuration.output_root == tmp_path / "outputs"
    assert configuration.parameters.orientation_uncertainty_degrees == 12.0
    assert configuration.parameters.anatomical_reference.reference_uncertainty_mm == 9.0
    assert configuration.participants[0].t1w_path == (tmp_path / "inputs" / "sub-0014_T1w.nii.gz")


def test_template_participant_runner_writes_inferential_outputs(tmp_path) -> None:
    channel_names = ("F3", "F4", "C3", "C4", "P3", "P4", "Cz")
    template_positions = make_template_positions(channel_names, "easycap-M1")
    data, affine, _ = _synthetic_t1(template_positions, artifact_names=set())
    t1w_path = tmp_path / "inputs" / "sub-0014_T1w.nii.gz"
    eeg_directory = tmp_path / "inputs" / "sub-0014" / "eeg"
    t1w_path.parent.mkdir(parents=True)
    eeg_directory.mkdir(parents=True)
    nib.save(nib.Nifti1Image(data, affine), t1w_path)
    channel_rows = ["name\ttype", *[f"{name}\tEEG" for name in channel_names]]
    (eeg_directory / "sub-0014_task-pain_channels.tsv").write_text(
        "\n".join(channel_rows) + "\n",
        encoding="utf-8",
    )

    paths = run_template_participant(
        SubjectLocalizationInput("sub-0014", t1w_path, eeg_directory.parent),
        tmp_path / "outputs",
        "easycap-M1",
        TemplateProjectionParameters(
            orientation_uncertainty_degrees=8.0,
            anatomical_reference=AnatomicalReferenceParameters(maximum_p95_residual_mm=18.0),
        ),
    )

    assert paths.electrodes_tsv.is_file()
    assert paths.diagnostics_json.is_file()
    assert paths.mne_head_montage_fif.is_file()
