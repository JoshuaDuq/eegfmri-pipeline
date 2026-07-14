# Native EEG-fMRI NeuXus QRS Integration Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents
> available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`)
> syntax for tracking.

**Goal:** Finish a GPL-licensed, BrainVision-independent offline EEG-fMRI correction pipeline that
uses NeuXus LSTM R-peak detection, MNE PCA-OBS, and explicit scanner and cardiac QC for all 83 runs.

**Architecture:** Keep the existing 5 kHz synchronized gradient AAS unchanged. Adapt the small
published NeuXus QRS detector into a deterministic array-in/peaks-out module with safe NumPy model
assets, classify detection quality separately, and pass permitted peaks to MNE PCA-OBS. Persist
provenance and warning-level QC without introducing detector fallbacks or mandatory manual review.

**Tech Stack:** Python 3.11+, MNE-Python 1.12, NumPy, SciPy, Numba, Pytest, BrainVision/FIF, YAML,
GPL-3.0-only.

---

### Task 1: Relicense and record NeuXus provenance

**Files:**
- Modify: `LICENSE`
- Modify: `README.md`
- Modify: `pyproject.toml`
- Create: `THIRD_PARTY_NOTICES.md`
- Create: `tests/architecture/test_project_license.py`

- [ ] **Step 1: Write failing license consistency tests**

  Test that the license file starts with GNU GPL v3, project metadata uses `GPL-3.0-only`, the GPL
  classifier is present, the README badge and license section are GPL, NeuXus attribution exists,
  and the MIT project classifier is absent.

- [ ] **Step 2: Run the license test and verify RED**

  Run: `pytest tests/architecture/test_project_license.py -v`

  Expected: failures showing the existing MIT metadata and missing attribution.

- [ ] **Step 3: Apply the GPL changes**

  Replace the root license with GNU GPL version 3, update package metadata and README, add the
  NeuXus paper/repository/tag attribution and modification notice, add `numba>=0.62,<1.0`, and add
  the detector asset path to setuptools package data.

- [ ] **Step 4: Run the license test and verify GREEN**

  Run: `pytest tests/architecture/test_project_license.py -v`

  Expected: all license tests pass.

- [ ] **Step 5: Commit licensing and provenance**

  Commit: `refactor: relicense project under GPLv3`

### Task 2: Add safe model assets and deterministic LSTM inference

**Files:**
- Create: `eeg_pipeline/preprocessing/eeg_fmri/neuxus_qrs.py`
- Create: `eeg_pipeline/preprocessing/eeg_fmri/assets/neuxus_qrs_weights.npz`
- Create: `eeg_pipeline/preprocessing/eeg_fmri/assets/neuxus_qrs_weights.sha256`
- Create: `scripts/development/convert_neuxus_qrs_weights.py`
- Create: `tests/preprocessing/test_neuxus_qrs.py`

- [ ] **Step 1: Write failing model-loader tests**

  Require an immutable `NeuXusQrsModel` loader to reject missing keys, wrong shapes, non-float32
  arrays, and checksum mismatches. Require `np.load(..., allow_pickle=False)` semantics.

- [ ] **Step 2: Run the loader tests and verify RED**

  Run: `pytest tests/preprocessing/test_neuxus_qrs.py -k model -v`

  Expected: import failure because `neuxus_qrs` does not exist.

- [ ] **Step 3: Implement the loader and conversion utility**

  Define the exact NeuXus parameter names and shapes, load the packaged archive through
  `importlib.resources`, verify its SHA-256 digest before loading, and expose model window length
  and hidden size. The development-only converter accepts the pinned upstream pickle, validates
  every array, writes float32 arrays plus scalar metadata to NPZ, and never runs at pipeline runtime.

- [ ] **Step 4: Generate the immutable model asset**

  Run the converter against NeuXus tag `v0.0.4`'s `weights-input-500.pkl`, record the upstream file
  digest in `THIRD_PARTY_NOTICES.md`, and generate the packaged NPZ checksum.

- [ ] **Step 5: Run loader tests and verify GREEN**

  Run: `pytest tests/preprocessing/test_neuxus_qrs.py -k model -v`

- [ ] **Step 6: Write failing inference tests**

  Require finite probabilities in `[0, 1]`, exact output length 500, deterministic repeated output,
  rejection of non-finite/wrong-shaped windows, and agreement with a hard-coded golden output digest
  generated once from the pinned upstream predictor on a deterministic input vector.

- [ ] **Step 7: Run inference tests and verify RED**

  Run: `pytest tests/preprocessing/test_neuxus_qrs.py -k inference -v`

- [ ] **Step 8: Implement minimal Numba inference**

  Adapt the two bidirectional LSTM layers and dense output from NeuXus into focused GPL-derived
  functions. Use float32 throughout, explicit parameter access, and no alternate pure-Python path.

- [ ] **Step 9: Run inference tests and verify GREEN**

  Run: `pytest tests/preprocessing/test_neuxus_qrs.py -k inference -v`

- [ ] **Step 10: Commit the model and inference boundary**

  Commit: `feat: add NeuXus QRS model inference`

### Task 3: Implement offline peak detection and graded quality

**Files:**
- Modify: `eeg_pipeline/preprocessing/eeg_fmri/neuxus_qrs.py`
- Modify: `eeg_pipeline/preprocessing/eeg_fmri/cardiac.py`
- Modify: `eeg_pipeline/preprocessing/eeg_fmri/config.py`
- Modify: `studies/pain_study/scripts/config/native_eeg_fmri_artifact_correction.yaml`
- Modify: `tests/preprocessing/test_neuxus_qrs.py`
- Modify: `tests/preprocessing/test_eeg_fmri_config.py`
- Modify: `tests/preprocessing/test_eeg_fmri_mne.py`

- [ ] **Step 1: Write failing offline detector tests**

  Cover whole-signal zero-phase 0.5–30 Hz filtering, 250 Hz resampling, overlapping 500-sample
  windows with 50-sample stride, probability averaging, threshold-region support of at least five
  samples, local maximum snapping, a 0.4-second refractory period, and exact time conversion.

- [ ] **Step 2: Run detector tests and verify RED**

  Run: `pytest tests/preprocessing/test_neuxus_qrs.py -k detection -v`

- [ ] **Step 3: Implement the offline detector**

  Add immutable detector parameters and result objects. Normalize each window to `[-1, 1]`, reject
  flat/non-finite input, aggregate probabilities with explicit support counts, consolidate peaks
  deterministically, and return peak times plus probability diagnostics. Do not call NeuXus's
  streaming framework or WFDB.

- [ ] **Step 4: Run detector tests and verify GREEN**

  Run: `pytest tests/preprocessing/test_neuxus_qrs.py -k detection -v`

- [ ] **Step 5: Write failing quality-classification tests**

  Require individual short/long RR intervals to produce warnings while permitting correction.
  Require hard failure for insufficient, non-finite, unordered, out-of-bounds, implausible-median,
  or less-than-90%-spanning detections. Verify all summary metrics.

- [ ] **Step 6: Run quality tests and verify RED**

  Run: `pytest tests/preprocessing/test_eeg_fmri_mne.py -k qrs -v`

- [ ] **Step 7: Replace generic MNE detection with NeuXus detection and quality classification**

  Define `QrsQuality` and extend `QrsDetection`. Remove `mne.preprocessing.find_ecg_events` and its
  frequency parameters. Increment the strict schema to version 2; version 1 is rejected rather than
  migrated. Add only scientifically necessary YAML values: detection rate, band-pass,
  stride, threshold, support, refractory interval, RR warning limits, median-rate hard limits,
  minimum count, and minimum temporal coverage.

- [ ] **Step 8: Run detector, config, and cardiac tests and verify GREEN**

  Run: `pytest tests/preprocessing/test_neuxus_qrs.py tests/preprocessing/test_eeg_fmri_config.py tests/preprocessing/test_eeg_fmri_mne.py -v`

- [ ] **Step 9: Commit automatic QRS detection**

  Commit: `feat: detect EEG-fMRI R peaks with NeuXus`

### Task 4: Add cardiac-locked QC around PCA-OBS

**Files:**
- Modify: `eeg_pipeline/preprocessing/eeg_fmri/qc.py`
- Modify: `eeg_pipeline/preprocessing/eeg_fmri/pipeline.py`
- Modify: `eeg_pipeline/preprocessing/eeg_fmri/cardiac.py`
- Modify: `eeg_pipeline/preprocessing/eeg_fmri/__init__.py`
- Create: `tests/preprocessing/test_eeg_fmri_qc.py`
- Modify: `tests/preprocessing/test_eeg_fmri_pipeline.py`

- [ ] **Step 1: Write failing cardiac-locked QC tests**

  Build synthetic heartbeat-locked EEG and require median evoked RMS, peak-to-peak amplitude, and
  pre/post attenuation in dB. Reject insufficient valid epochs and mismatched channels/timelines.

- [ ] **Step 2: Run QC tests and verify RED**

  Run: `pytest tests/preprocessing/test_eeg_fmri_qc.py -v`

- [ ] **Step 3: Implement focused cardiac QC**

  Add one function to summarize heartbeat-locked EEG over a fixed peri-R window and one function to
  compare pre/post summaries. Keep scanner-harmonic and cardiac QC data structures separate.

- [ ] **Step 4: Run QC tests and verify GREEN**

  Run: `pytest tests/preprocessing/test_eeg_fmri_qc.py -v`

- [ ] **Step 5: Write failing orchestration tests**

  Require QRS detection once, cardiac QC immediately before and after OBS, unchanged ECG/auxiliary
  channels, four OBS components, warning preservation, and final result fields.

- [ ] **Step 6: Run orchestration tests and verify RED**

  Run: `pytest tests/preprocessing/test_eeg_fmri_pipeline.py tests/preprocessing/test_eeg_fmri_mne.py -v`

- [ ] **Step 7: Implement orchestration**

  Separate detection from OBS application so the same accepted peaks drive pre/post QC and
  subtraction. Construct one detector in the cohort runner and inject it into each run so model
  validation and Numba compilation occur once without hidden global state. Extend
  `NativeCorrectionResult` with QRS quality, model identity, and cardiac QC.

- [ ] **Step 8: Run orchestration tests and verify GREEN**

  Run: `pytest tests/preprocessing/test_eeg_fmri_pipeline.py tests/preprocessing/test_eeg_fmri_mne.py tests/preprocessing/test_eeg_fmri_qc.py -v`

- [ ] **Step 9: Commit PCA-OBS and cardiac QC integration**

  Commit: `feat: quantify cardiac artifact correction`

### Task 5: Persist complete provenance and output state

**Files:**
- Modify: `studies/pain_study/scripts/run_native_eeg_fmri_artifact_correction.py`
- Modify: `tests/scripts/test_run_native_eeg_fmri_artifact_correction.py`

- [ ] **Step 1: Write failing runner tests**

  Require run QC JSON and cohort TSV to contain detector/model/checksum, QRS warnings and interval
  metrics, cardiac attenuation, exact method names, and all 83 verified inputs. Require an atomic
  R-peak TSV and ECG/probability diagnostic PNG for each run. Require failed runs to remain only
  under the incomplete root.

- [ ] **Step 2: Run runner tests and verify RED**

  Run: `pytest tests/scripts/test_run_native_eeg_fmri_artifact_correction.py -v`

- [ ] **Step 3: Implement provenance serialization**

  Serialize immutable QRS and cardiac QC structures without lossy stringification. Save peak times
  and warning status in TSV, render the fixed ECG/probability diagnostic, and update the BIDS
  derivative description to name synchronized volume AAS, NeuXus LSTM detection, and MNE PCA-OBS.
  Publish to a new `native_eeg_fmri_correction-v2` root; do not overwrite or delete the incomplete
  version-1 qualification output.

- [ ] **Step 4: Run runner tests and verify GREEN**

  Run: `pytest tests/scripts/test_run_native_eeg_fmri_artifact_correction.py -v`

- [ ] **Step 5: Commit derivative provenance outputs**

  Commit: `feat: persist native MRI correction provenance`

### Task 6: Documentation and repository verification

**Files:**
- Modify: `docs/native_eeg_fmri_artifact_correction.md`
- Modify: `studies/pain_study/scripts/README.md`
- Modify: `README.md`
- Modify: `docs/superpowers/plans/2026-07-14-native-eeg-fmri-neuxus-qrs.md`

- [ ] **Step 1: Update documentation**

  Document automatic NeuXus detection, warning versus hard-failure semantics, GPL attribution,
  cardiac QC, residual harmonic exclusions, and the exact 83-run cohort boundary. Remove the old
  generic-MNE QRS counts and any unqualified replacement claim.

- [ ] **Step 2: Run focused tests**

  Run: `pytest tests/preprocessing/test_neuxus_qrs.py tests/preprocessing/test_eeg_fmri_config.py tests/preprocessing/test_eeg_fmri_mne.py tests/preprocessing/test_eeg_fmri_qc.py tests/preprocessing/test_eeg_fmri_pipeline.py tests/scripts/test_run_native_eeg_fmri_artifact_correction.py tests/architecture/test_project_license.py -v`

  Expected: all focused tests pass without warnings attributable to the implementation.

- [ ] **Step 3: Run repository gates**

  Run: `ruff check eeg_pipeline fmri_pipeline tests scripts studies/pain_study/scripts`

  Run: `black --check .`

  Run: `make verify-structure`

  Run: `make verify-architecture`

  Run: `make verify-maintainability`

  Run: `pytest`

- [ ] **Step 4: Run real-data qualification**

  First run the fixed detector on the four audited recordings and compare its QRS metrics with the
  prior generic detector. Then execute the atomically staged 83-run cohort. Regenerate scanner-line,
  cardiac-locked, and neural-preservation summaries. Do not publish the final derivative if any run
  raises a hard failure or any preservation gate fails.

- [ ] **Step 5: Review the complete diff and commit documentation**

  Confirm no unrelated user changes are included, run `git diff --check`, and commit the completed
  implementation with an imperative behavior-focused subject.

  Commit: `docs: document native EEG-fMRI correction`
