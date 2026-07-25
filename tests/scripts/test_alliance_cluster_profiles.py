from __future__ import annotations

import os
import subprocess

from tests import REPO_ROOT

WORKFLOW_DIR = REPO_ROOT / "local_workflows" / "alliance_canada"
EXPORTED_FIELDS = (
    "ALLIANCE_CLUSTER",
    "ALLIANCE_HOST",
    "ALLIANCE_SSH_CONTROL_PATH",
    "ALLIANCE_ACCOUNT",
    "ALLIANCE_PROJECT_ROOT",
    "ALLIANCE_SCRATCH_ROOT",
    "FMRIPREP_SLURM_MEMORY",
    "FMRIPREP_MEM_MB",
    "STUDY1_PREPARE_SLURM_MEMORY",
)


def _source_profile(cluster: str) -> subprocess.CompletedProcess[str]:
    fields = " ".join(f'"${{{field}}}"' for field in EXPORTED_FIELDS)
    command = "set -e; " f'source "{WORKFLOW_DIR / "alliance_env.sh"}"; ' f"printf '%s\\n' {fields}"
    env = os.environ.copy()
    env["ALLIANCE_CLUSTER"] = cluster
    return subprocess.run(
        ["bash", "-c", command],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def _values(result: subprocess.CompletedProcess[str]) -> dict[str, str]:
    return dict(zip(EXPORTED_FIELDS, result.stdout.splitlines(), strict=True))


def test_rorqual_profile_exports_generic_values() -> None:
    result = _source_profile("rorqual")

    assert result.returncode == 0, result.stderr
    values = _values(result)
    assert values["ALLIANCE_HOST"] == "joshduq@rorqual.alliancecan.ca"
    assert values["ALLIANCE_SSH_CONTROL_PATH"] == "/tmp/rorqual-joshduq-ssh-control"
    assert values["ALLIANCE_ACCOUNT"] == "def-mpcoll"
    assert values["ALLIANCE_PROJECT_ROOT"] == "/project/def-mpcoll/joshduq"
    assert values["ALLIANCE_SCRATCH_ROOT"] == "/scratch/joshduq"
    assert values["FMRIPREP_SLURM_MEMORY"] == "700G"
    assert values["FMRIPREP_MEM_MB"] == "680000"
    assert values["STUDY1_PREPARE_SLURM_MEMORY"] == "64G"


def test_trillium_profile_omits_slurm_memory() -> None:
    result = _source_profile("trillium")

    assert result.returncode == 0, result.stderr
    values = _values(result)
    assert values["ALLIANCE_HOST"] == "joshduq@trillium.alliancecan.ca"
    assert values["ALLIANCE_SSH_CONTROL_PATH"] == "/tmp/trillium-joshduq-ssh-control"
    assert values["FMRIPREP_SLURM_MEMORY"] == ""
    assert values["FMRIPREP_MEM_MB"] == "700000"
    assert values["STUDY1_PREPARE_SLURM_MEMORY"] == ""


def test_unknown_cluster_profile_fails() -> None:
    result = _source_profile("cedar")

    assert result.returncode != 0
    assert "Unsupported ALLIANCE_CLUSTER: cedar" in result.stderr


def test_alliance_workflows_have_generic_entrypoints() -> None:
    expected = (
        "setup_alliance_fmriprep.sh",
        "setup_alliance_runtime.sh",
        "setup_alliance_study1.sh",
        "setup_alliance_study2.sh",
        "setup_alliance_study2_runtime.sh",
        "submit_fmriprep_alliance.sh",
        "submit_study1_alliance.sh",
        "submit_study2_alliance.sh",
    )

    for filename in expected:
        assert (WORKFLOW_DIR / filename).is_file(), filename


def test_tracked_workflows_do_not_use_rorqual_specific_variables() -> None:
    result = subprocess.run(
        [
            "git",
            "ls-files",
            "local_workflows/alliance_canada/*.sh",
            "local_workflows/alliance_canada/lib/*.sh",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    workflow_files = [REPO_ROOT / relative_path for relative_path in result.stdout.splitlines()]
    contents = "\n".join(path.read_text() for path in workflow_files)

    assert "RORQUAL_HOST" not in contents
    assert "RORQUAL_SSH_CONTROL_PATH" not in contents
