# Dynamic EEG-fMRI Cohort Discovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the native EEG-fMRI workflow discover all available recordings without fixed participant lists or run counts.

**Architecture:** Original 5 kHz BrainVision headers are the authoritative inventory. Each stage validates the discovered set for emptiness, filename identity, uniqueness, companion files, and required downstream metadata; manifests propagate the exact dynamic inventory. Explicit subject selection remains optional for targeted runs.

**Tech Stack:** Python, pathlib, argparse, MNE-Python, pytest

---

### Task 1: Discover the source cohort dynamically

**Files:**
- Modify: `studies/pain_study/scripts/sanitize_brainvision_vas_markers.py`
- Test: `tests/scripts/test_sanitize_brainvision_vas_markers.py`

- [ ] Replace the fixed subject tuple and expected run count with discovery from `sub-*/eeg/original_5khz/*.vhdr`.
- [ ] Validate non-empty discovery, directory/filename subject agreement, unique subject/run identities, and complete BrainVision companions.
- [ ] Add optional repeated `--subject` selection without changing the all-subject default.
- [ ] Run the focused sanitizer tests.

### Task 2: Consume the dynamic manifest

**Files:**
- Modify: `studies/pain_study/scripts/run_native_eeg_fmri_artifact_correction.py`
- Test: `tests/scripts/test_run_native_eeg_fmri_artifact_correction.py`

- [ ] Remove expected-count arguments and constants.
- [ ] Reject an empty manifest and preserve duplicate, path, provenance, and matching BOLD validation.
- [ ] Run the focused native-correction tests.

### Task 3: Remove the fixed organizer cohort

**Files:**
- Modify: `studies/pain_study/scripts/organize_source_eeg.py`
- Test: `tests/scripts/test_organize_source_eeg.py`

- [ ] Discover participants from matching source-data subject directories and KINGSTON raw directories.
- [ ] Reject missing, ambiguous, or empty inventories and allow optional explicit subject selection.
- [ ] Run the focused organizer tests.

### Task 4: Verify and run

**Files:**
- Modify: `studies/pain_study/scripts/README.md`

- [ ] Remove documentation that describes a fixed cohort boundary.
- [ ] Run all affected focused tests and search the workflow for remaining cohort-size constants.
- [ ] Recreate the marker-sanitized derivative and resume native correction, EEG-BIDS conversion, PsychoPy merge, and MNE preprocessing for the dynamically discovered cohort.
