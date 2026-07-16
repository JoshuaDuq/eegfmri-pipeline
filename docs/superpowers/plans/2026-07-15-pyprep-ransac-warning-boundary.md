# PyPREP RANSAC Warning Boundary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Silence only verified spurious PyPREP matrix warnings while failing on genuinely non-finite RANSAC output, then restart the cohort with compatible bad-channel metadata.

**Architecture:** A private helper in the existing PyPREP preprocessing module owns the warning boundary and output validation. The existing detection loop calls the helper without changing bad-channel selection behavior. Runtime configuration selects subject-union synchronization for the shared ICA workflow.

**Tech Stack:** Python 3.14, NumPy, MNE-Python, PyPREP, pytest.

---

### Task 1: Test the Warning Boundary Contract

**Files:**
- Modify: `tests/preprocessing/test_bads_detection_fail_fast.py`

- [ ] **Step 1: Add a failing test for exact warning suppression and finite output**

Create a fake `NoisyChannels` object whose RANSAC method emits the three exact matrix
RuntimeWarnings and stores a finite two-dimensional `ransac_correlations` array. Assert that
`_find_bad_channels_by_ransac()` emits none of those warnings.

- [ ] **Step 2: Add failing tests for unrelated and non-finite results**

Assert that an unrelated RuntimeWarning remains visible. Store a correlation matrix containing
`NaN` and assert that the helper raises `FloatingPointError` containing
`non-finite RANSAC correlations`.

- [ ] **Step 3: Verify RED**

Run:

```bash
/Users/joduq24/Desktop/EEG_fMRI_Pipeline/.venv/bin/python -m pytest \
  tests/preprocessing/test_bads_detection_fail_fast.py -q
```

Expected: collection or attribute failures because `_find_bad_channels_by_ransac` does not
exist.

### Task 2: Implement the Narrow Warning Boundary

**Files:**
- Modify: `eeg_pipeline/preprocessing/pipeline/preprocess.py`
- Test: `tests/preprocessing/test_bads_detection_fail_fast.py`

- [ ] **Step 1: Implement exact warning filters**

Add `_find_bad_channels_by_ransac(noisy_channels)` using `warnings.catch_warnings()` and
`warnings.filterwarnings()` for the three exact `matmul` RuntimeWarning messages in only
`scipy.linalg._basic`, `mne.channels.interpolation`, and `pyprep.ransac`.

- [ ] **Step 2: Validate the PyPREP correlation postcondition**

Require `noisy_channels._extra_info["bad_by_ransac"]["ransac_correlations"]` to be a non-empty
finite two-dimensional NumPy array. Raise `RuntimeError` for a malformed PyPREP result and
`FloatingPointError` for non-finite values.

- [ ] **Step 3: Replace the direct RANSAC call**

Replace `nc.find_bad_by_ransac()` with `_find_bad_channels_by_ransac(nc)` without changing any
other detector ordering or bad-channel aggregation.

- [ ] **Step 4: Verify GREEN and lint**

Run:

```bash
/Users/joduq24/Desktop/EEG_fMRI_Pipeline/.venv/bin/python -m pytest \
  tests/preprocessing/test_bads_detection_fail_fast.py \
  tests/pipelines/test_pipeline_preprocessing.py -q
/Users/joduq24/Desktop/EEG_fMRI_Pipeline/.venv/bin/python -m ruff check \
  eeg_pipeline/preprocessing/pipeline/preprocess.py \
  tests/preprocessing/test_bads_detection_fail_fast.py
```

Expected: all tests and lint checks pass.

- [ ] **Step 5: Commit**

```bash
git add eeg_pipeline/preprocessing/pipeline/preprocess.py \
  tests/preprocessing/test_bads_detection_fail_fast.py
git commit -m "fix: validate and quiet PyPREP RANSAC numerics"
```

### Task 3: Restart Full Cohort Preprocessing

**Files:**
- Write runtime log under `native_mne_preprocessing-v3/logs/`.

- [ ] **Step 1: Launch independently**

Start the full preprocessing command in a detached `screen` session with `--n-jobs 10` and
`--bad-channel-sync-policy subject_union`, using the feature worktree through `PYTHONPATH`.

- [ ] **Step 2: Verify process and clean log startup**

Confirm the detached screen and Python process exist. Inspect the log to verify subject-union
configuration, absence of the targeted warning flood, and normal PyPREP progress.
