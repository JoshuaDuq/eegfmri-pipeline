from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from tests import REPO_ROOT


def _write_executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


def _write_workflow_config(workflow_dir: Path, tmp_path: Path) -> tuple[Path, Path, Path]:
    remote_derivatives = tmp_path / "remote" / "derivatives"
    local_derivatives = tmp_path / "local" / "derivatives"
    local_bids_root = tmp_path / "bids" / "fmri"

    (remote_derivatives / "preprocessed" / "fmri").mkdir(parents=True)
    (local_derivatives / "preprocessed" / "fmri").mkdir(parents=True)
    (local_bids_root / "sub-0001" / "func").mkdir(parents=True)

    (workflow_dir / "alliance_env.sh").write_text(
        "\n".join(
            [
                f'export FMRIPREP_DERIV_ROOT="{remote_derivatives}"',
                'export FMRIPREP_OUTPUT_SPACES="MNI152NLin2009cAsym T1w"',
                'export FMRIPREP_TASK_ID="thermalactive"',
            ]
        ),
        encoding="utf-8",
    )
    (workflow_dir / "local_env.sh").write_text(
        "\n".join(
            [
                'export ALLIANCE_CLUSTER="rorqual"',
                'export ALLIANCE_HOST="fake-rorqual"',
                f'export ALLIANCE_SSH_CONTROL_PATH="{tmp_path / "ssh-control"}"',
                f'export LOCAL_BIDS_FMRI_ROOT="{local_bids_root}"',
                (
                    'export LOCAL_FMRIPREP_OUTPUT_ROOT="'
                    f'{local_derivatives / "preprocessed" / "fmri"}"'
                ),
            ]
        ),
        encoding="utf-8",
    )
    (workflow_dir / "subjects.txt").write_text("0001\n", encoding="utf-8")

    return remote_derivatives, local_derivatives, local_bids_root


def _write_fake_commands(bin_dir: Path) -> None:
    bin_dir.mkdir()
    _write_executable(
        bin_dir / "ssh",
        """#!/usr/bin/env bash
set -euo pipefail
if [[ " $* " == *" -O check "* ]]; then
    exit 0
fi
host_seen=0
cmd=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -o|-S|-O)
            shift 2
            ;;
        -*)
            shift
            ;;
        *)
            if [[ "$host_seen" -eq 0 ]]; then
                host_seen=1
                shift
            else
                cmd+=("$1")
                shift
            fi
            ;;
    esac
done
bash -lc "${cmd[*]}"
""",
    )
    _write_executable(
        bin_dir / "rsync",
        """#!/usr/bin/env bash
set -euo pipefail
args=("$@")
src="${args[$((${#args[@]} - 2))]}"
dest="${args[$((${#args[@]} - 1))]}"
src="${src#*:}"
cp "$src" "$dest"
""",
    )


def _write_minimal_fmriprep_output(remote_derivatives: Path, local_bids_root: Path) -> None:
    raw_func = local_bids_root / "sub-0001" / "func"
    for task in ("thermalactive", "rest"):
        raw_bold = raw_func / f"sub-0001_task-{task}_run-01_bold.nii.gz"
        raw_bold.write_text("raw", encoding="utf-8")

    remote_root = remote_derivatives / "preprocessed" / "fmri"
    subject_func = remote_root / "sub-0001" / "func"
    subject_func.mkdir(parents=True)
    for space in ["MNI152NLin2009cAsym", "T1w"]:
        (subject_func / f"sub-0001_task-thermalactive_run-01_space-{space}_desc-preproc_bold.nii.gz").write_text(
            "bold",
            encoding="utf-8",
        )
        (subject_func / f"sub-0001_task-thermalactive_run-01_space-{space}_desc-brain_mask.nii.gz").write_text(
            "mask",
            encoding="utf-8",
        )
    (subject_func / "sub-0001_task-thermalactive_run-01_desc-confounds_timeseries.tsv").write_text(
        "confound\n",
        encoding="utf-8",
    )
    (remote_root / "sub-0001.html").write_text("<html></html>", encoding="utf-8")
    (remote_root / "sourcedata" / "freesurfer" / "sub-0001").mkdir(parents=True)
    (remote_root / "._sub-0001.html").write_text("metadata", encoding="utf-8")


def test_fetch_fmriprep_outputs_uses_tar_archive_and_verifies_before_cleanup(tmp_path: Path) -> None:
    workflow_dir = tmp_path / "workflow"
    shutil.copytree(REPO_ROOT / "local_workflows" / "alliance_canada", workflow_dir)
    remote_derivatives, local_derivatives, local_bids_root = _write_workflow_config(
        workflow_dir,
        tmp_path,
    )
    _write_minimal_fmriprep_output(remote_derivatives, local_bids_root)

    stale_dir = local_derivatives / "preprocessed" / "fmri" / "fmriprep"
    stale_dir.mkdir(parents=True)
    (stale_dir / "stale.txt").write_text("old", encoding="utf-8")

    bin_dir = tmp_path / "bin"
    _write_fake_commands(bin_dir)
    env = {"PATH": f"{bin_dir}:{REPO_ROOT / '.venv' / 'bin'}:/usr/bin:/bin"}

    result = subprocess.run(
        ["bash", str(workflow_dir / "fetch_fmriprep_outputs.sh")],
        cwd=workflow_dir,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr + result.stdout
    output_root = local_derivatives / "preprocessed" / "fmri"
    assert (output_root / "fmriprep" / "sub-0001" / "func").is_dir()
    assert not (output_root / "fmriprep" / "stale.txt").exists()
    assert not (output_root / "fmriprep" / "._sub-0001.html").exists()
    assert not (local_derivatives / "fmriprep_preprocessed_fmri.tar").exists()
    assert not (remote_derivatives.parent / "fmriprep_preprocessed_fmri.tar").exists()
