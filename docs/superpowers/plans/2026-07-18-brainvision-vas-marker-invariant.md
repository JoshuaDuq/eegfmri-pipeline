# BrainVision VAS Marker Invariant Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate ambiguous `Vas_on/V  1` markers from every supported BrainVision workflow without modifying source EEG data.

**Architecture:** Generalize the metadata-only staging command around explicit recording paths and source-relative output paths. Enforce one shared annotation invariant at analysis boundaries, point consumers at the canonical v2 derivative, and independently audit the staged cohort.

**Tech Stack:** Python 3.11, MNE-Python, NumPy, pytest, Ruff, BrainVision header/marker files

---

### Task 1: Generalize metadata-only staging

**Files:**
- Modify: `studies/pain_study/scripts/sanitize_brainvision_vas_markers.py`
- Test: `tests/scripts/test_sanitize_brainvision_vas_markers.py`

- [ ] Add a failing test that stages a 1 kHz Analyzer recording under its source-relative
  `sub-*/eeg/brainvision_processed_1khz` path and asserts that the staged header references the
  original signal file.
- [ ] Run
  `.venv/bin/python -m pytest tests/scripts/test_sanitize_brainvision_vas_markers.py -q` and
  confirm the new assertion fails because staging currently requires 5 kHz and flattens layout.
- [ ] Replace the 5 kHz-only validation with frequency-agnostic structural validation, pass the
  source-data root into staging, and construct the output directory from
  `source_vhdr.relative_to(source_data_root).parent`.
- [ ] Add manifest fields for relative source path and sampling frequency, then run the focused
  test file and confirm all tests pass.

### Task 2: Enforce the annotation invariant

**Files:**
- Modify: `eeg_pipeline/preprocessing/eeg_fmri/mne_io.py`
- Modify: `studies/pain_study/scripts/export_brainvision_matlab.py`
- Test: `tests/preprocessing/test_eeg_fmri_mne.py`
- Test: `tests/scripts/test_export_brainvision_matlab.py`

- [ ] Add failing tests showing the shared validator and MATLAB exporter reject a raw recording
  containing `Vas_on/V  1`.
- [ ] Run the two named tests and confirm failures are caused by the missing invariant check.
- [ ] Add one focused `validate_unambiguous_vas_markers(raw)` function and call it from volume
  extraction and immediately after each MATLAB BrainVision read.
- [ ] Point the exporter default at the v2 processed 1 kHz derivative and rerun both test files.

### Task 3: Publish all thermal recording layouts

**Files:**
- Modify: `studies/pain_study/scripts/sanitize_brainvision_vas_markers.py`
- Modify: `studies/pain_study/scripts/run_native_eeg_fmri_artifact_correction.py`
- Test: `tests/scripts/test_sanitize_brainvision_vas_markers.py`

- [ ] Add a failing discovery test containing matching original 5 kHz and processed 1 kHz
  triplets and assert that both are returned exactly once under explicit subject selection.
- [ ] Generalize discovery to both supported thermal layouts, preserve explicit recording
  exclusions, and remove sampling-frequency assumptions from cohort publication.
- [ ] Change canonical defaults from `brainvision_marker_sanitized-v1` to v2 without fallback.
- [ ] Run the sanitation and native-correction tests.

### Task 4: Verify and materialize the derivative

**Files:**
- Generated outside Git: `/Volumes/KINGSTON/EEG_fMRI_data/derivatives/brainvision_marker_sanitized-v2`
- Generated in workspace: `outputs/matlab_exports/sub-0015/brainvision_processed_1khz`

- [ ] Run the focused pytest files and Ruff over every modified Python file.
- [ ] Run the sanitation command for all cohort subjects into the new v2 derivative; it must
  refuse pre-existing output and publish atomically.
- [ ] Audit every staged marker file: zero `Vas_on,V  1`, expected `Vas_on,VAS_ON`, unchanged
  `Volume,V  1` counts, and no copied EEG bytes.
- [ ] Run the sub-0015 MATLAB exporter using its v2 default and execute its independent reload
  verification.
- [ ] Run the broader preprocessing and scripts test groups and inspect `git diff --check`.
