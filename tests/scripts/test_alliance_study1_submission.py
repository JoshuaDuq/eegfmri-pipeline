from tests import REPO_ROOT


WORKFLOW_ROOT = REPO_ROOT / "local_workflows" / "alliance_canada"
STUDY1_ROOT = WORKFLOW_ROOT / "study1"


def test_study1_workflow_uses_tracked_scripts() -> None:
    expected = {
        "common_args.sh",
        "prepare.sh",
        "dispatch.sh",
        "cell_array.sh",
        "report.sh",
        "submit.sh",
    }

    assert {path.name for path in STUDY1_ROOT.glob("*.sh")} == expected


def test_study1_common_args_use_complete_cohort_and_confirmatory_settings() -> None:
    common = (STUDY1_ROOT / "common_args.sh").read_text()
    submit = (STUDY1_ROOT / "submit.sh").read_text()

    assert 'study1.cohort.min_subjects=13' in common
    assert 'study1.feature_benchmark.n_perm=${STUDY1_N_PERM}' in common
    assert 'STUDY1_N_PERM="5000"' in submit
    assert 'study1_subjects.txt' in common


def test_study1_submission_defines_prepare_dispatch_cell_report_chain() -> None:
    submit = (STUDY1_ROOT / "submit.sh").read_text()
    dispatch = (STUDY1_ROOT / "dispatch.sh").read_text()

    assert '--dependency="afterok:${prepare_job}"' in submit
    assert 'append_optional_memory_arg prepare_args "${STUDY1_PREPARE_SLURM_MEMORY}"' in submit
    assert 'append_optional_memory_arg dispatch_args "${STUDY1_DISPATCH_SLURM_MEMORY}"' in submit
    assert 'append_optional_memory_arg cell_args "${STUDY1_CELL_SLURM_MEMORY}"' in dispatch
    assert '--dependency="afterok:${cell_job}"' in dispatch
    assert 'append_optional_memory_arg report_args "${STUDY1_REPORT_SLURM_MEMORY}"' in dispatch


def test_study1_cell_array_validates_manifest_rows() -> None:
    cell_array = (STUDY1_ROOT / "cell_array.sh").read_text()

    assert "Malformed Study 1 cell manifest row" in cell_array
    assert 'case "${partition}:${target}" in' in cell_array


def test_local_study1_submitter_uses_selected_alliance_connection() -> None:
    submit = (WORKFLOW_ROOT / "submit_study1_alliance.sh").read_text()

    assert "require_alliance_connection" in submit
    assert '"${ALLIANCE_HOST}"' in submit
    assert 'ALLIANCE_CLUSTER=${ALLIANCE_CLUSTER}' in submit
