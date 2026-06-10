from pathlib import Path


WORKFLOW_DIR = Path("local_workflows/alliance_canada")


def test_trillium_fmriprep_submission_does_not_request_slurm_memory() -> None:
    submit_script = (WORKFLOW_DIR / "submit_fmriprep_array.sh").read_text()

    assert "ALLIANCE_MEM" not in submit_script
    assert '--mem="' not in submit_script


def test_trillium_fmriprep_requests_six_hours() -> None:
    env_file = (WORKFLOW_DIR / "alliance_env.sh").read_text()

    assert 'export ALLIANCE_TIME="06:00:00"' in env_file


def test_trillium_fmriprep_uses_explicit_fmriprep_memory_mb() -> None:
    env_file = (WORKFLOW_DIR / "alliance_env.sh").read_text()
    job_script = (WORKFLOW_DIR / "fmriprep_array.sbatch").read_text()

    assert 'export FMRIPREP_MEM_MB="700000"' in env_file
    assert "FMRIPREP_MEM_MB" in job_script
    assert "SLURM_MEM_PER_NODE" not in job_script
    assert '--mem-mb "${FMRIPREP_MEM_MB}"' in job_script


def test_trillium_fmriprep_exports_subjects_file_to_array_job() -> None:
    submit_script = (WORKFLOW_DIR / "submit_fmriprep_array.sh").read_text()

    assert '--export="ALL,SUBJECTS_FILE=${SUBJECTS_FILE}"' in submit_script


def test_trillium_fmriprep_sets_numba_cache_on_scratch() -> None:
    job_script = (WORKFLOW_DIR / "fmriprep_array.sbatch").read_text()

    assert 'export NUMBA_CACHE_DIR="${FMRIPREP_WORK_ROOT}/numba_cache"' in job_script
    assert 'mkdir -p "${NUMBA_CACHE_DIR}"' in job_script


def test_trillium_fmriprep_writes_derivatives_to_scratch() -> None:
    env_file = (WORKFLOW_DIR / "alliance_env.sh").read_text()
    job_script = (WORKFLOW_DIR / "fmriprep_array.sbatch").read_text()

    assert 'export FMRIPREP_DERIV_ROOT="/scratch/joshduq/derivatives"' in env_file
    assert "FMRIPREP_DERIV_ROOT" in job_script
    assert '--deriv-root "${FMRIPREP_DERIV_ROOT}"' in job_script


def test_trillium_fmriprep_uses_scratch_templateflow_cache() -> None:
    env_file = (WORKFLOW_DIR / "alliance_env.sh").read_text()

    assert 'export TEMPLATEFLOW_HOME="/scratch/joshduq/templateflow"' in env_file


def test_trillium_runtime_setup_creates_container_directory() -> None:
    setup_script = (WORKFLOW_DIR / "setup_python_and_container.sh").read_text()

    assert 'mkdir -p "$(dirname "${FMRIPREP_IMAGE}")"' in setup_script


def test_trillium_fmriprep_skips_validation_for_narrow_task_upload() -> None:
    job_script = (WORKFLOW_DIR / "fmriprep_array.sbatch").read_text()

    assert "--skip-bids-validation" in job_script
