from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from tests import REPO_ROOT

SCRIPT = REPO_ROOT / "local_workflows" / "alliance_canada" / "build_upload_manifest.py"
PRUNE_SCRIPT = REPO_ROOT / "local_workflows" / "alliance_canada" / "prune_upload_root.py"


def _write(path: Path, text: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _read_manifest(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def _write_subjects(path: Path) -> None:
    _write(path, "0001\n# skipped\n\n")


def _write_fmri_bids(root: Path) -> None:
    _write(root / "dataset_description.json", "{}")
    _write(root / "participants.tsv", "participant_id\nsub-0001\n")
    _write(root / "task-thermalactive_events.json", "{}")
    _write(root / "sub-0001" / "anat" / "sub-0001_T1w.nii.gz")
    _write(root / "sub-0001" / "anat" / "sub-0001_T1w.json", "{}")
    _write(root / "sub-0001" / "func" / "sub-0001_task-thermalactive_run-01_bold.nii.gz")
    _write(root / "sub-0001" / "func" / "sub-0001_task-thermalactive_run-01_bold.json")
    _write(root / "sub-0001" / "func" / "sub-0001_task-thermalactive_run-01_events.tsv")
    _write(root / "sub-0001" / "func" / "sub-0001_task-rest_run-01_bold.nii.gz")
    _write(root / "sub-0001" / "fmap" / "sub-0001_dir-AP_epi.nii.gz")
    _write(root / "sub-0002" / "anat" / "sub-0002_T1w.nii.gz")
    _write(root / "sub-0002" / "func" / "sub-0002_task-thermalactive_run-01_bold.nii.gz")


def _write_eeg_bids(root: Path) -> None:
    _write(root / "dataset_description.json", "{}")
    _write(root / "task-thermalactive_events.json", "{}")
    _write(root / "sub-0001" / "sub-0001_scans.tsv")
    eeg_root = root / "sub-0001" / "eeg"
    _write(eeg_root / "sub-0001_task-thermalactive_eeg.vhdr")
    _write(eeg_root / "sub-0001_task-thermalactive_eeg.vmrk")
    _write(eeg_root / "sub-0001_task-thermalactive_eeg.eeg")
    _write(eeg_root / "sub-0001_task-thermalactive_channels.tsv")
    _write(eeg_root / "sub-0001_task-rest_eeg.vhdr")
    _write(eeg_root / "sub-0001_space-CapTrak_electrodes.tsv")
    _write(eeg_root / "sub-0001_space-CapTrak_coordsystem.json")


def _write_derivatives(root: Path) -> None:
    _write(root / "dataset_description.json", "{}")
    eeg_root = root / "preprocessed" / "eeg" / "sub-0001" / "eeg"
    _write(eeg_root / "sub-0001_task-thermalactive_proc-clean_epo.fif")
    _write(eeg_root / "sub-0001_task-thermalactive_proc-clean_events.tsv")
    _write(eeg_root / "sub-0001_task-rest_proc-clean_epo.fif")

    study1 = root / "group" / "multimodal" / "study1"
    _write(study1 / "reports" / "study1_report.tsv")
    _write(study1 / "targets" / "primary_targets.parquet")
    _write(
        study1
        / "features_trial_ml_safe"
        / "sub-0001"
        / "eeg"
        / "features"
        / "power"
        / "features_power.parquet"
    )
    _write(
        study1
        / "features_trial_ml_safe"
        / "sub-0001"
        / "eeg"
        / "features"
        / "power"
        / "metadata"
        / "extraction_config.json",
        "{}",
    )
    _write(
        study1
        / "feature_benchmark"
        / "primary"
        / "NPS"
        / "alpha_beta_gamma"
        / "model_comparison"
        / "model_comparison.tsv"
    )

    study2 = root / "group" / "multimodal" / "study2"
    _write(study2 / "source_stage" / "source_stage_input.tsv")
    _write(study2 / "old_output_that_should_not_upload.tsv")


def test_fmriprep_manifest_uses_subject_task_anatomy_and_root_metadata(tmp_path: Path) -> None:
    subjects = tmp_path / "subjects.txt"
    fmri_root = tmp_path / "bids" / "fmri"
    output = tmp_path / "fmriprep_files.txt"
    _write_subjects(subjects)
    _write_fmri_bids(fmri_root)

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "fmriprep",
            "--subjects-file",
            str(subjects),
            "--local-fmri-root",
            str(fmri_root),
            "--task",
            "thermalactive",
            "--output",
            str(output),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    files = _read_manifest(output)
    assert "dataset_description.json" in files
    assert "participants.tsv" in files
    assert "task-thermalactive_events.json" in files
    assert "sub-0001/anat/sub-0001_T1w.nii.gz" in files
    assert "sub-0001/anat/sub-0001_T1w.json" in files
    assert "sub-0001/func/sub-0001_task-thermalactive_run-01_bold.nii.gz" in files
    assert "sub-0001/func/sub-0001_task-thermalactive_run-01_events.tsv" in files
    assert "sub-0001/fmap/sub-0001_dir-AP_epi.nii.gz" in files
    assert "sub-0001/func/sub-0001_task-rest_run-01_bold.nii.gz" not in files
    assert "sub-0002/anat/sub-0002_T1w.nii.gz" not in files


def test_study2_manifest_uploads_required_handoff_files_only(tmp_path: Path) -> None:
    subjects = tmp_path / "subjects.txt"
    fmri_root = tmp_path / "bids" / "fmri"
    eeg_root = tmp_path / "bids" / "eeg"
    deriv_root = tmp_path / "derivatives"
    manifest_dir = tmp_path / "manifests"
    _write_subjects(subjects)
    _write_fmri_bids(fmri_root)
    _write_eeg_bids(eeg_root)
    _write_derivatives(deriv_root)

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "study2",
            "--subjects-file",
            str(subjects),
            "--local-fmri-root",
            str(fmri_root),
            "--local-eeg-root",
            str(eeg_root),
            "--local-deriv-root",
            str(deriv_root),
            "--task",
            "thermalactive",
            "--study1-root-name",
            "study1",
            "--study2-root-name",
            "study2",
            "--output-dir",
            str(manifest_dir),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    fmri_files = _read_manifest(manifest_dir / "fmri_bids_files.txt")
    eeg_files = _read_manifest(manifest_dir / "eeg_bids_files.txt")
    deriv_files = _read_manifest(manifest_dir / "derivative_files.txt")

    assert fmri_files == [
        "dataset_description.json",
        "participants.tsv",
        "sub-0001/anat/sub-0001_T1w.json",
        "sub-0001/anat/sub-0001_T1w.nii.gz",
        "task-thermalactive_events.json",
    ]
    assert "sub-0001/sub-0001_scans.tsv" in eeg_files
    assert "task-thermalactive_events.json" in eeg_files
    assert "sub-0001/eeg/sub-0001_task-thermalactive_eeg.vhdr" in eeg_files
    assert "sub-0001/eeg/sub-0001_task-rest_eeg.vhdr" not in eeg_files
    assert "sub-0001/eeg/sub-0001_space-CapTrak_electrodes.tsv" in eeg_files
    assert (
        "preprocessed/eeg/sub-0001/eeg/sub-0001_task-thermalactive_proc-clean_epo.fif"
        in deriv_files
    )
    assert (
        "preprocessed/eeg/sub-0001/eeg/sub-0001_task-thermalactive_proc-clean_events.tsv"
        in deriv_files
    )
    assert "preprocessed/eeg/sub-0001/eeg/sub-0001_task-rest_proc-clean_epo.fif" not in deriv_files
    assert "dataset_description.json" in deriv_files
    assert "group/multimodal/study1/reports/study1_report.tsv" in deriv_files
    assert "group/multimodal/study1/targets/primary_targets.parquet" in deriv_files
    assert (
        "group/multimodal/study1/feature_benchmark/primary/NPS/"
        "alpha_beta_gamma/model_comparison/model_comparison.tsv"
    ) in deriv_files
    assert "group/multimodal/study2/source_stage/source_stage_input.tsv" in deriv_files
    assert "group/multimodal/study2/old_output_that_should_not_upload.tsv" not in deriv_files


def test_study2_manifest_fails_before_rsync_when_clean_events_are_missing(
    tmp_path: Path,
) -> None:
    subjects = tmp_path / "subjects.txt"
    fmri_root = tmp_path / "bids" / "fmri"
    eeg_root = tmp_path / "bids" / "eeg"
    deriv_root = tmp_path / "derivatives"
    manifest_dir = tmp_path / "manifests"
    _write_subjects(subjects)
    _write_fmri_bids(fmri_root)
    _write_eeg_bids(eeg_root)
    _write_derivatives(deriv_root)
    (
        deriv_root
        / "preprocessed"
        / "eeg"
        / "sub-0001"
        / "eeg"
        / "sub-0001_task-thermalactive_proc-clean_events.tsv"
    ).unlink()

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "study2",
            "--subjects-file",
            str(subjects),
            "--local-fmri-root",
            str(fmri_root),
            "--local-eeg-root",
            str(eeg_root),
            "--local-deriv-root",
            str(deriv_root),
            "--task",
            "thermalactive",
            "--study1-root-name",
            "study1",
            "--study2-root-name",
            "study2",
            "--output-dir",
            str(manifest_dir),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "Missing required clean EEG events for sub-0001" in result.stderr


def test_study2_manifest_can_preserve_missing_exact_handoff_paths_for_cleanup(
    tmp_path: Path,
) -> None:
    subjects = tmp_path / "subjects.txt"
    fmri_root = tmp_path / "bids" / "fmri"
    eeg_root = tmp_path / "bids" / "eeg"
    deriv_root = tmp_path / "derivatives"
    manifest_dir = tmp_path / "manifests"
    _write_subjects(subjects)
    _write_fmri_bids(fmri_root)
    _write_eeg_bids(eeg_root)
    _write_derivatives(deriv_root)
    (
        deriv_root / "group" / "multimodal" / "study2" / "source_stage" / "source_stage_input.tsv"
    ).unlink()

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "study2",
            "--subjects-file",
            str(subjects),
            "--local-fmri-root",
            str(fmri_root),
            "--local-eeg-root",
            str(eeg_root),
            "--local-deriv-root",
            str(deriv_root),
            "--task",
            "thermalactive",
            "--study1-root-name",
            "study1",
            "--study2-root-name",
            "study2",
            "--output-dir",
            str(manifest_dir),
            "--allow-missing-exact-paths",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Missing required Study 2 source-stage input" in result.stderr
    assert "group/multimodal/study2/source_stage/source_stage_input.tsv" in _read_manifest(
        manifest_dir / "derivative_files.txt"
    )


def test_prune_upload_root_dry_run_reports_unlisted_files_without_deleting(
    tmp_path: Path,
) -> None:
    root = tmp_path / "remote_root"
    manifest = tmp_path / "allowed.txt"
    _write(root / "sub-0001" / "anat" / "sub-0001_T1w.nii.gz")
    _write(root / "sub-0002" / "anat" / "sub-0002_T1w.nii.gz")
    _write(root / ".DS_Store")
    _write(manifest, "sub-0001/anat/sub-0001_T1w.nii.gz\n")

    result = subprocess.run(
        [
            sys.executable,
            str(PRUNE_SCRIPT),
            "--root",
            str(root),
            "--manifest",
            str(manifest),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "[dry-run]" in result.stdout
    assert "sub-0002/anat/sub-0002_T1w.nii.gz" in result.stdout
    assert ".DS_Store" in result.stdout
    assert (root / "sub-0002" / "anat" / "sub-0002_T1w.nii.gz").exists()
    assert (root / ".DS_Store").exists()


def test_prune_upload_root_apply_deletes_only_unlisted_files(tmp_path: Path) -> None:
    root = tmp_path / "remote_root"
    manifest = tmp_path / "allowed.txt"
    _write(root / "sub-0001" / "anat" / "sub-0001_T1w.nii.gz")
    _write(root / "sub-0002" / "anat" / "sub-0002_T1w.nii.gz")
    _write(root / ".DS_Store")
    _write(manifest, "sub-0001/anat/sub-0001_T1w.nii.gz\n")

    result = subprocess.run(
        [
            sys.executable,
            str(PRUNE_SCRIPT),
            "--root",
            str(root),
            "--manifest",
            str(manifest),
            "--apply",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "[apply]" in result.stdout
    assert (root / "sub-0001" / "anat" / "sub-0001_T1w.nii.gz").exists()
    assert not (root / "sub-0002" / "anat" / "sub-0002_T1w.nii.gz").exists()
    assert not (root / ".DS_Store").exists()
