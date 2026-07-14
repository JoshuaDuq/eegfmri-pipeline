from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from tests import REPO_ROOT


WORKFLOW_ROOT = REPO_ROOT / "local_workflows" / "alliance_canada"
MANIFEST_SCRIPT = WORKFLOW_ROOT / "build_upload_manifest.py"


def _write(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x", encoding="utf-8")


def _write_study1_inputs(root: Path) -> tuple[Path, Path, Path, Path, Path]:
    subjects = root / "subjects.txt"
    fmri = root / "bids" / "fmri"
    eeg = root / "bids" / "eeg"
    derivatives = root / "derivatives"
    external = root / "external"
    _write(subjects)
    subjects.write_text("0001\n", encoding="utf-8")

    _write(fmri / "dataset_description.json")
    _write(fmri / "sub-0001" / "anat" / "sub-0001_T1w.nii.gz")
    bold_name = "sub-0001_task-thermalactive_run-01_bold.nii.gz"
    _write(fmri / "sub-0001" / "func" / bold_name)
    _write(fmri / "sub-0001" / "func" / bold_name.replace(".nii.gz", ".json"))
    _write(fmri / "sub-0001" / "func" / bold_name.replace("_bold.nii.gz", "_events.tsv"))

    _write(eeg / "dataset_description.json")
    eeg_subject = eeg / "sub-0001" / "eeg"
    _write(eeg_subject / "sub-0001_task-thermalactive_eeg.vhdr")
    _write(eeg_subject / "sub-0001_task-thermalactive_eeg.vmrk")
    _write(eeg_subject / "sub-0001_task-thermalactive_eeg.eeg")
    _write(eeg_subject / "sub-0001_space-CapTrak_electrodes.tsv")
    _write(eeg_subject / "sub-0001_space-CapTrak_coordsystem.json")

    clean = derivatives / "preprocessed" / "eeg" / "sub-0001" / "eeg"
    _write(clean / "sub-0001_task-thermalactive_proc-clean_epo.fif")
    _write(clean / "sub-0001_task-thermalactive_proc-clean_events.tsv")

    func = derivatives / "preprocessed" / "fmri" / "fmriprep" / "sub-0001" / "func"
    prefix = "sub-0001_task-thermalactive_run-01"
    _write(func / f"{prefix}_desc-confounds_timeseries.tsv")
    _write(func / f"{prefix}_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz")
    _write(func / f"{prefix}_space-MNI152NLin2009cAsym_res-2_desc-brain_mask.nii.gz")

    _write(external / "NPS" / "weights_NSF_grouppred_cvpcr.nii.gz")
    _write(external / "SIIPS1" / "nonnoc_v11_4_137subjmap_weighted_mean.nii.gz")
    _write(external / "signature_manifest.yaml")
    _write(external / "tpl-MNI152NLin2009cAsym_res-02_desc-brain_mask.nii.gz")
    return subjects, fmri, eeg, derivatives, external


def _run_manifest(tmp_path: Path) -> subprocess.CompletedProcess[str]:
    subjects, fmri, eeg, derivatives, external = _write_study1_inputs(tmp_path)
    return subprocess.run(
        [
            sys.executable,
            str(MANIFEST_SCRIPT),
            "study1",
            "--subjects-file",
            str(subjects),
            "--local-fmri-root",
            str(fmri),
            "--local-eeg-root",
            str(eeg),
            "--local-deriv-root",
            str(derivatives),
            "--local-external-root",
            str(external),
            "--task",
            "thermalactive",
            "--output-dir",
            str(tmp_path / "manifests"),
        ],
        text=True,
        capture_output=True,
        check=False,
    )


def test_study1_subject_manifest_contains_complete_cohort_without_0006() -> None:
    subjects = (WORKFLOW_ROOT / "study1_subjects.txt").read_text().splitlines()

    assert subjects == [
        "0000", "0001", "0003", "0004", "0005", "0007", "0008",
        "0009", "0010", "0011", "0012", "0013", "0014",
    ]


def test_study1_manifest_includes_all_input_domains(tmp_path: Path) -> None:
    result = _run_manifest(tmp_path)

    assert result.returncode == 0, result.stderr
    manifest_names = {path.name for path in (tmp_path / "manifests").iterdir()}
    assert manifest_names == {
        "eeg_bids_files.txt",
        "fmri_bids_files.txt",
        "eeg_derivative_files.txt",
        "fmri_derivative_files.txt",
        "external_files.txt",
    }


def test_study1_manifest_fails_when_fmriprep_output_is_missing(tmp_path: Path) -> None:
    subjects, fmri, eeg, derivatives, external = _write_study1_inputs(tmp_path)
    missing = next((derivatives / "preprocessed" / "fmri").rglob("*desc-preproc_bold.nii.gz"))
    missing.unlink()

    result = subprocess.run(
        [
            sys.executable,
            str(MANIFEST_SCRIPT),
            "study1",
            "--subjects-file", str(subjects),
            "--local-fmri-root", str(fmri),
            "--local-eeg-root", str(eeg),
            "--local-deriv-root", str(derivatives),
            "--local-external-root", str(external),
            "--task", "thermalactive",
            "--output-dir", str(tmp_path / "manifests"),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "Missing required fMRIPrep preprocessed BOLD" in result.stderr
    assert not (tmp_path / "manifests").exists()
