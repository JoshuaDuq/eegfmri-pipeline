# Narrow Study Upload Manifests Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upload only the Study 1 and Study 2 files required by the local Alliance workflow.

**Architecture:** Add a standard-library Python manifest builder under `local_workflows/alliance_canada/` and have the setup shell scripts feed those manifests to `rsync --files-from`. The builder validates required files before any transfer starts, preserving fail-fast behavior and avoiding broad fallback syncs.

**Tech Stack:** Bash, Python standard library, pytest.

---

### Task 1: Manifest Builder Tests

**Files:**
- Test: `tests/scripts/test_alliance_upload_manifest.py`

- [ ] Write tests that create minimal fMRI BIDS, EEG BIDS, derivatives, and subject-list fixtures.
- [ ] Verify fMRIPrep manifests include only selected subjects, configured task files, root metadata, anatomy, and fieldmaps.
- [ ] Verify Study 2 manifests include selected subject anatomy, selected task EEG, clean epochs/events, Study 1 handoff artifacts, and Study 2 source-stage input.
- [ ] Verify missing clean EEG events fails with a concrete error.

### Task 2: Manifest Builder

**Files:**
- Create: `local_workflows/alliance_canada/build_upload_manifest.py`

- [ ] Implement subject normalization and newline-safe manifest writing.
- [ ] Implement fMRIPrep mode.
- [ ] Implement Study 2 mode.
- [ ] Keep required and optional path logic separate so errors stay explicit.

### Task 3: Shell Integration

**Files:**
- Modify: `local_workflows/alliance_canada/setup_rorqual.sh`
- Modify: `local_workflows/alliance_canada/setup_rorqual_study2.sh`
- Modify: `local_workflows/alliance_canada/alliance_env.sh`
- Modify: `local_workflows/alliance_canada/README.md`

- [ ] Replace broad BIDS/derivatives rsync calls with manifest-driven `rsync --files-from`.
- [ ] Require non-empty task/root-name env settings used to build manifests.
- [ ] Update local workflow docs to describe narrow uploads.

### Task 4: Verification

**Files:**
- Test: `tests/scripts/test_alliance_upload_manifest.py`
- Test: `tests/scripts/test_alliance_fetch_fmriprep_outputs.py`

- [ ] Run the focused upload-manifest tests.
- [ ] Run the existing Alliance fetch test.
- [ ] Run shell syntax checks for edited workflow scripts.
