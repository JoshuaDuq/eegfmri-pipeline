# Alliance Cluster Profiles Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `ALLIANCE_CLUSTER=rorqual|trillium` select and validate all Alliance infrastructure behavior, then stage and submit the 13-subject Study 1 workflow on Rorqual.

**Architecture:** A strict Bash profile loader resolves one tracked cluster profile into generic Alliance variables. Existing fMRIPrep and Study 2 workflows consume that interface, while new repository-owned Study 1 scripts stage a validated cohort from KINGSTON and submit a dependency-linked Slurm workflow. Cluster-local storage is never assumed to be shared; a dedicated migration command retrieves missing fMRIPrep outputs from Trillium before Rorqual upload.

**Tech Stack:** Bash, Slurm, SSH control sockets, rsync, Python 3.11, pytest, Ruff

---

### Task 1: Cluster Profile Contract

**Files:**
- Create: `local_workflows/alliance_canada/clusters/rorqual.sh`
- Create: `local_workflows/alliance_canada/clusters/trillium.sh`
- Create: `local_workflows/alliance_canada/load_cluster_profile.sh`
- Modify: `local_workflows/alliance_canada/local_env.sh`
- Modify: `local_workflows/alliance_canada/alliance_env.sh`
- Create: `tests/scripts/test_alliance_cluster_profiles.py`

- [ ] **Step 1: Write failing profile-selection tests**

Test subprocess sourcing with `ALLIANCE_CLUSTER=rorqual`, `trillium`, and an invalid name. Assert the generic host, socket, roots, resource values, and non-zero invalid-selection exit:

```python
def test_rorqual_profile_exports_generic_values() -> None:
    values = source_profile("rorqual")
    assert values["ALLIANCE_HOST"] == "joshduq@rorqual.alliancecan.ca"
    assert values["ALLIANCE_SSH_CONTROL_PATH"] == "/tmp/rorqual-joshduq-ssh-control"
    assert values["FMRIPREP_SLURM_MEMORY"] == "700G"


def test_trillium_profile_omits_slurm_memory() -> None:
    values = source_profile("trillium")
    assert values["ALLIANCE_HOST"] == "joshduq@trillium.alliancecan.ca"
    assert values["FMRIPREP_SLURM_MEMORY"] == ""


def test_unknown_profile_fails() -> None:
    result = run_profile("cedar")
    assert result.returncode != 0
    assert "Unsupported ALLIANCE_CLUSTER" in result.stderr
```

- [ ] **Step 2: Run tests and confirm they fail because the loader does not exist**

Run: `.venv/bin/python -m pytest tests/scripts/test_alliance_cluster_profiles.py -q`

Expected: FAIL on missing `load_cluster_profile.sh`.

- [ ] **Step 3: Implement strict profiles and loader**

Profiles define host, socket, account, project/scratch roots, module list, and workflow memory. The loader accepts only `rorqual` and `trillium`, sources one file, and validates every required variable with no hostname fallback.

```bash
case "${ALLIANCE_CLUSTER:-}" in
    rorqual|trillium) ;;
    *) echo "Unsupported ALLIANCE_CLUSTER: ${ALLIANCE_CLUSTER:-<empty>}" >&2; return 1 ;;
esac
source "${SCRIPT_DIR}/clusters/${ALLIANCE_CLUSTER}.sh"
```

`local_env.sh` defaults explicitly to Rorqual. `alliance_env.sh` derives all remote paths from profile roots and exports generic variables only.

- [ ] **Step 4: Run profile tests**

Run: `.venv/bin/python -m pytest tests/scripts/test_alliance_cluster_profiles.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add local_workflows/alliance_canada/clusters local_workflows/alliance_canada/load_cluster_profile.sh local_workflows/alliance_canada/local_env.sh local_workflows/alliance_canada/alliance_env.sh tests/scripts/test_alliance_cluster_profiles.py
git commit -m "feat: add Alliance cluster profiles"
```

### Task 2: Generic Alliance Connection Interface

**Files:**
- Create: `local_workflows/alliance_canada/start_alliance_connection.sh`
- Create: `local_workflows/alliance_canada/stop_alliance_connection.sh`
- Create: `local_workflows/alliance_canada/lib/alliance_connection.sh`
- Delete: `local_workflows/alliance_canada/start_rorqual_connection.sh`
- Delete: `local_workflows/alliance_canada/stop_rorqual_connection.sh`
- Modify: every script under `local_workflows/alliance_canada/` that references `RORQUAL_HOST` or `RORQUAL_SSH_CONTROL_PATH`
- Modify: `tests/scripts/test_alliance_fetch_fmriprep_outputs.py`
- Create: `tests/scripts/test_alliance_connection.py`

- [ ] **Step 1: Write failing generic connection tests**

Use a fake `ssh` executable to assert that start/check/stop use `ALLIANCE_HOST` and the profile socket. Assert that missing profile values fail before SSH.

- [ ] **Step 2: Run tests and confirm old Rorqual variables fail expectations**

Run: `.venv/bin/python -m pytest tests/scripts/test_alliance_connection.py tests/scripts/test_alliance_fetch_fmriprep_outputs.py -q`

Expected: FAIL because generic scripts and variables are absent.

- [ ] **Step 3: Implement connection helper and migrate consumers**

The helper validates an active control connection:

```bash
require_alliance_connection() {
    ssh -S "${ALLIANCE_SSH_CONTROL_PATH}" -O check "${ALLIANCE_HOST}" >/dev/null 2>&1 || {
        echo "No active ${ALLIANCE_CLUSTER} SSH control connection." >&2
        return 1
    }
}
```

Rename generic entrypoints and update setup, run, fetch, cleanup, and Study 2 scripts. Do not leave aliases using `RORQUAL_*`.

- [ ] **Step 4: Run connection and existing transfer tests**

Run: `.venv/bin/python -m pytest tests/scripts/test_alliance_connection.py tests/scripts/test_alliance_fetch_fmriprep_outputs.py tests/scripts/test_alliance_upload_manifest.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A local_workflows/alliance_canada tests/scripts
git commit -m "refactor: use generic Alliance connection settings"
```

### Task 3: Profile-Aware Slurm Resources for fMRIPrep and Study 2

**Files:**
- Create: `local_workflows/alliance_canada/lib/slurm_args.sh`
- Modify: `local_workflows/alliance_canada/submit_fmriprep_array.sh`
- Modify: `local_workflows/alliance_canada/fmriprep_array.sbatch`
- Rename: `local_workflows/alliance_canada/setup_rorqual.sh` to `setup_alliance_fmriprep.sh`
- Rename: `local_workflows/alliance_canada/setup_rorqual_runtime.sh` to `setup_alliance_runtime.sh`
- Rename: `local_workflows/alliance_canada/setup_rorqual_study2.sh` to `setup_alliance_study2.sh`
- Rename: `local_workflows/alliance_canada/setup_rorqual_study2_runtime.sh` to `setup_alliance_study2_runtime.sh`
- Modify: `local_workflows/alliance_canada/setup_and_run_fmriprep.sh`
- Modify: `local_workflows/alliance_canada/setup_and_run_study2.sh`
- Modify: `studies/pain_study/study2/alliance/run_full_study2_alliance.sh`
- Replace: `tests/scripts/test_trillium_fmriprep_submission.py` with `tests/scripts/test_alliance_fmriprep_submission.py`
- Modify: `tests/scripts/test_study2_alliance_workflow.py`

- [ ] **Step 1: Write failing resource-policy tests**

Assert Rorqual submission contains `--mem=700G`, Trillium contains no `--mem`, fMRIPrep internal memory is below the Slurm request, and Study 2 receives profile resource arguments.

- [ ] **Step 2: Run focused tests and confirm failure**

Run: `.venv/bin/python -m pytest tests/scripts/test_alliance_fmriprep_submission.py tests/scripts/test_study2_alliance_workflow.py -q`

Expected: FAIL because resource arguments are currently Trillium-specific.

- [ ] **Step 3: Implement Slurm argument construction**

Use one helper that appends memory only when configured:

```bash
append_optional_memory_arg() {
    local -n target=$1
    local memory=$2
    if [[ -n "${memory}" ]]; then
        target+=(--mem="${memory}")
    fi
}
```

Build `sbatch` commands as arrays. Validate numeric fMRIPrep memory and profile identity before submission. Update setup output to name the selected cluster.

- [ ] **Step 4: Run fMRIPrep and Study 2 tests**

Run: `.venv/bin/python -m pytest tests/scripts/test_alliance_fmriprep_submission.py tests/scripts/test_study2_alliance_workflow.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A local_workflows/alliance_canada studies/pain_study/study2/alliance tests/scripts
git commit -m "feat: apply cluster resource profiles to Alliance workflows"
```

### Task 4: Study 1 Input Manifest and Trillium Migration

**Files:**
- Create: `local_workflows/alliance_canada/study1_subjects.txt`
- Extend: `local_workflows/alliance_canada/build_upload_manifest.py`
- Create: `local_workflows/alliance_canada/fetch_study1_fmriprep_from_trillium.sh`
- Create: `local_workflows/alliance_canada/setup_alliance_study1.sh`
- Create: `tests/scripts/test_alliance_study1_staging.py`

- [ ] **Step 1: Write failing manifest/staging tests**

Create temporary KINGSTON-style trees. Assert the 13-subject list excludes `0006`, includes BIDS EEG/fMRI, clean EEG derivatives, mapped fMRIPrep outputs, and signature assets. Assert a missing subject derivative fails before fake `rsync` is called.

- [ ] **Step 2: Run tests and confirm missing workflow failure**

Run: `.venv/bin/python -m pytest tests/scripts/test_alliance_study1_staging.py -q`

Expected: FAIL because Study 1 staging does not exist.

- [ ] **Step 3: Implement narrow manifests and migration**

The migration script accepts explicit Trillium source host/socket variables and retrieves only missing configured subjects into:

```text
${LOCAL_DERIV_ROOT}/preprocessed/fmri/fmriprep/sub-<id>
```

Study 1 setup validates and uploads:

```text
BIDS EEG -> ${BIDS_EEG_ROOT}
BIDS fMRI -> ${BIDS_FMRI_ROOT}
clean EEG derivatives -> ${FMRIPREP_DERIV_ROOT}/preprocessed/eeg
fMRIPrep subjects -> ${FMRIPREP_DERIV_ROOT}/preprocessed/fmri
external signatures -> ${SIGNATURE_ROOT}
```

It validates remote subject counts and required files after transfer.

- [ ] **Step 4: Run Study 1 staging tests**

Run: `.venv/bin/python -m pytest tests/scripts/test_alliance_study1_staging.py tests/scripts/test_alliance_upload_manifest.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add local_workflows/alliance_canada/study1_subjects.txt local_workflows/alliance_canada/build_upload_manifest.py local_workflows/alliance_canada/fetch_study1_fmriprep_from_trillium.sh local_workflows/alliance_canada/setup_alliance_study1.sh tests/scripts/test_alliance_study1_staging.py
git commit -m "feat: stage Study 1 inputs for Alliance clusters"
```

### Task 5: Repository-Owned Study 1 Slurm Workflow

**Files:**
- Create: `local_workflows/alliance_canada/study1/common_args.sh`
- Create: `local_workflows/alliance_canada/study1/prepare.sh`
- Create: `local_workflows/alliance_canada/study1/dispatch.sh`
- Create: `local_workflows/alliance_canada/study1/cell_array.sh`
- Create: `local_workflows/alliance_canada/study1/report.sh`
- Create: `local_workflows/alliance_canada/submit_study1_alliance.sh`
- Create: `tests/scripts/test_alliance_study1_submission.py`

- [ ] **Step 1: Write failing dependency-chain tests**

Use fake `sbatch` output IDs and assert:

```text
prepare -> dispatcher -> cell array -> report
```

Assert profile account/memory values, tracked script paths, run-specific cell manifest, cohort arguments, `min_subjects=13`, `n_perm=5000`, and complete job-record fields. Assert malformed cell rows fail.

- [ ] **Step 2: Run tests and confirm scripts are missing**

Run: `.venv/bin/python -m pytest tests/scripts/test_alliance_study1_submission.py -q`

Expected: FAIL on missing `submit_study1_alliance.sh`.

- [ ] **Step 3: Implement Study 1 scripts**

`common_args.sh` builds subject arguments from `study1_subjects.txt` and uses profile paths for BIDS, derivatives, and signatures. Submission creates a run ID, logs, and job record. Every `sbatch` command is an array and uses optional profile memory.

The dispatcher writes a run-specific cell manifest. If it is empty, it records `CELL_COUNT=0` and runs the report directly rather than failing.

- [ ] **Step 4: Run Study 1 submission tests**

Run: `.venv/bin/python -m pytest tests/scripts/test_alliance_study1_submission.py tests/pipelines/test_study1_targets.py tests/pipelines/test_study1_feature_benchmark_config.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add local_workflows/alliance_canada/study1 local_workflows/alliance_canada/submit_study1_alliance.sh tests/scripts/test_alliance_study1_submission.py
git commit -m "feat: add reproducible Alliance Study 1 workflow"
```

### Task 6: Documentation and Full Local Verification

**Files:**
- Modify: `local_workflows/alliance_canada/README.md`
- Modify: `studies/pain_study/study1/RUN_GUIDE.md`
- Modify: `studies/pain_study/study2/alliance/README.md`
- Modify: any structure tests that enumerate workflow entrypoints

- [ ] **Step 1: Update documentation**

Document the one-line selector:

```bash
export ALLIANCE_CLUSTER=rorqual
```

Document connection, migration, setup, submission, monitoring, fetching, and switching back to Trillium. Remove instructions using deleted Rorqual-specific generic entrypoints.

- [ ] **Step 2: Run shell syntax checks**

Run: `bash -n local_workflows/alliance_canada/*.sh local_workflows/alliance_canada/clusters/*.sh local_workflows/alliance_canada/lib/*.sh local_workflows/alliance_canada/study1/*.sh`

Expected: exit 0.

- [ ] **Step 3: Run focused and structural tests**

Run: `.venv/bin/python -m pytest tests/scripts tests/pipelines/test_study1_targets.py tests/pipelines/test_study1_feature_benchmark_config.py tests/utils/test_repo_hygiene_guards.py -q`

Expected: PASS.

Run: `.venv/bin/ruff check local_workflows tests/scripts`

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add local_workflows/alliance_canada/README.md studies/pain_study/study1/RUN_GUIDE.md studies/pain_study/study2/alliance/README.md tests
git commit -m "docs: document Alliance cluster switching"
```

### Task 7: Rorqual Migration, Validation, and Study 1 Submission

**Files:**
- Runtime state only: KINGSTON, Trillium storage, and Rorqual project/scratch

- [ ] **Step 1: Keep Trillium Study 1 cancelled**

Verify jobs `1925912` and `1925913` remain cancelled with `sacct`.

- [ ] **Step 2: Retrieve missing fMRIPrep subjects to KINGSTON**

Run the migration for `0009`, `0012`, `0013`, and `0014`. Verify each subject has preprocessed BOLD, confounds, and brain masks for every configured thermalactive run.

- [ ] **Step 3: Sync and validate Rorqual**

Run:

```bash
ALLIANCE_CLUSTER=rorqual bash local_workflows/alliance_canada/start_alliance_connection.sh
ALLIANCE_CLUSTER=rorqual bash local_workflows/alliance_canada/setup_alliance_study1.sh
ALLIANCE_CLUSTER=rorqual bash local_workflows/alliance_canada/setup_alliance_runtime.sh
```

Verify modules, venv imports, signature checksums, subject counts, and required derivative files.

- [ ] **Step 4: Validate scheduler arguments**

Run `sbatch --test-only` for the prepare and cell job shapes. Confirm `def-mpcoll`, explicit Rorqual memory, time limits, and dependencies are accepted.

- [ ] **Step 5: Submit Study 1**

Run:

```bash
ALLIANCE_CLUSTER=rorqual bash local_workflows/alliance_canada/submit_study1_alliance.sh
```

Expected: a run ID, prepare job ID, dispatcher job ID, and job-record path.

- [ ] **Step 6: Report monitoring commands**

Provide exact `squeue`, `sacct`, and log-tail commands using the submitted IDs. Leave jobs running unless the user explicitly requests cancellation.
