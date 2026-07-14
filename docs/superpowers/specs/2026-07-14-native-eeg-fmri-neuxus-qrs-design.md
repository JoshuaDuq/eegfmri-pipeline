# Native EEG-fMRI NeuXus QRS Integration Design

**Status:** Approved

**Date:** 2026-07-14

## Objective

Complete the native offline EEG-fMRI artifact-correction pipeline without BrainVision
Analyzer. Preserve the existing synchronized gradient-artifact subtraction, replace the
generic ECG detector with an offline adaptation of NeuXus's EEG-fMRI-specific LSTM, and
apply MNE PCA-OBS automatically using the detected R-peaks.

The pipeline must remain conservative about neuronal signal removal, expose correction
quality for every run, and fail only when cardiac detection is unusable. Individual
abnormal RR intervals are quality warnings, not a manual-review requirement.

## Scientific Basis

The synchronized 21-volume average-artifact subtraction already matches the established
offline BrainVision and FMRIB gradient-correction boundary. Qualification data show about
25 dB of scanner-line attenuation, comparable to the existing BrainVision workflow.
Additional full-band residual PCA, motion-selected templates, and adaptive cancellation
did not materially reduce the residual scanner comb and risked removing injected neural
signals. They are excluded from this design.

NeuXus provides the strongest automatic EEG-fMRI R-peak candidates observed during the
cohort audit. Its LSTM was trained on ECG acquired during simultaneous EEG-fMRI and was
validated against manual R-peak annotations. The full NeuXus pipeline is designed for
real-time operation: it omits offline volume alignment and uses cardiac average-artifact
subtraction. This project will therefore adapt only the published QRS detector and retain
the stronger offline gradient stage and MNE's maintained PCA-OBS implementation.

The conventional 0.21-second R-to-pulse-artifact delay does not require millisecond-perfect
R-peak localization. Occasional missed or duplicated detections reduce correction quality
but do not invalidate the complete run. Catastrophic failures still need to surface before
PCA-OBS is applied.

## Licensing

The repository will be relicensed from MIT to GPL-3.0-only. Git history identifies one
contributor, so no additional contributor permission is required.

The change includes:

- replacing the root `LICENSE` with the GPL version 3 license text;
- changing package metadata and the README license badge;
- adding third-party attribution for NeuXus, its authors, the publication, the upstream
  repository, the source revision, and the adapted files;
- identifying locally modified NeuXus-derived source and model assets;
- distributing the adapted detector and weights under GPL-3.0-only.

The runtime will not load NeuXus's pickle model. The weights will be converted once to a
non-executable NumPy archive, stored with a checksum, and loaded with pickle disabled.

## Architecture

### Gradient Correction

The native gradient stage remains unchanged:

1. Read the original 5 kHz BrainVision recording and exact scanner-volume markers.
2. Validate synchronization and volume boundaries.
3. Construct centered, leave-one-out 21-volume templates.
4. Align volumes at sub-sample resolution.
5. Fit the template and temporal derivative per volume and channel.
6. Subtract the fitted artifact.
7. Apply the 100 Hz anti-alias low-pass and resample to 1 kHz.

Residual gradient PCA/OBS, motion-selected template banks, adaptive cancellation, and
scanner-harmonic notch banks are outside this stage. The observed harmonic exclusions
remain part of feature extraction.

### Automatic QRS Detection

A focused `neuxus_qrs` module will own NeuXus model inference and peak consolidation. It
will have no dependency on NeuXus's streaming graph, Lab Streaming Layer, pandas nodes, or
BrainVision Recorder interfaces.

The detector will:

1. Receive the gradient-corrected ECG channel at 1 kHz.
2. Create a detection-only copy filtered from 0.5 to 30 Hz with a zero-phase filter.
3. Resample that copy to the model's validated 250 Hz input rate.
4. Normalize each two-second window to the model's expected range.
5. Run deterministic bidirectional LSTM inference using the published weights.
6. Average probabilities from overlapping windows using a 50-sample stride.
7. Consolidate threshold crossings, snap them to local ECG maxima, and enforce the
   published 0.4-second refractory interval.
8. Convert peak locations to seconds on the original MNE timeline.

The implementation uses one detector and one set of parameters. There is no algorithmic
fallback. Missing or malformed model assets raise immediately.

### QRS Quality Classification

Cardiac quality is separated from detection. A `QrsQuality` result records:

- detected beat count;
- median heart rate;
- minimum, maximum, and median RR interval;
- abnormal RR count and fraction;
- maximum inter-beat gap;
- first-to-last-peak temporal coverage;
- warning messages;
- whether correction is permitted.

Automatic PCA-OBS is permitted when the detector returns enough strictly increasing peaks,
the median rate is physiologically plausible, and peaks cover the recording. Individual RR
intervals outside the nominal range and isolated long gaps produce warnings. They do not
block correction.

Detection is unusable, and the run fails before PCA-OBS, when any of these are true:

- fewer than the configured minimum number of peaks are present;
- peak times are non-finite, unordered, or outside the data interval;
- the median heart rate is outside the configured physiological range;
- the detected peaks do not span at least 90% of the correction interval.

These entry-point assumptions remain explicit in YAML and are validated when loaded.

### Pulse-Artifact Correction

MNE `apply_pca_obs` will process EEG channels only, using four components and the automatic
NeuXus R-peak times. ECG and auxiliary channels remain unchanged. The implementation will
not introduce a separate fixed 0.21-second shift because MNE fits its basis over complete
cardiac windows centered on the supplied QRS events.

Downstream ICA and ICLabel remain responsible for residual cardiac and motion components.
PCA-OBS does not replace general EEG preprocessing.

### Output States

Output names must represent completed processing boundaries:

- `gradientclean`: synchronized gradient correction, filtering, and resampling completed;
- `mriartifactclean`: gradient correction and automatic PCA-OBS completed;
- QC metadata: detection metrics, warnings, correction parameters, software versions,
  model checksum, input identity, and output identity.

A run with warning-level QRS irregularities may produce `mriartifactclean` output. A run
with unusable QRS detection does not. Cohort publication remains atomic: an incomplete
cohort stays in the staging directory and is never presented as final.

## Quality Control

Every run will include:

- pre/post gradient harmonic power and local prominence;
- QRS probability and peak diagnostics;
- RR distribution and detection-coverage metrics;
- ECG-locked EEG root-mean-square and peak-to-peak artifact before and after PCA-OBS;
- cardiac-locked attenuation in decibels;
- warnings and hard-failure reasons;
- an MNE-compatible diagnostic plot for visual investigation.

`mne.preprocessing.annotate_amplitude` is not part of MRI artifact modeling. Existing
PyPREP, ICA/ICLabel, and autoreject stages continue to handle bad channels, generic
high-amplitude intervals, and trial-level artifacts.

## Components

The implementation will use focused modules:

- `neuxus_qrs.py`: model loading, inference, and offline peak consolidation;
- `cardiac.py`: ECG validation, QRS quality classification, and PCA-OBS orchestration;
- `qc.py`: scanner-harmonic and cardiac-locked metrics;
- `pipeline.py`: ordered stage composition and output-state enforcement;
- package data: immutable NumPy model weights and checksum;
- YAML: the small set of scientifically meaningful correction parameters.

No module will import the complete NeuXus framework. The detector accepts arrays and returns
an immutable result, making it independently testable.

## Verification

Unit tests will establish:

- model assets load without pickle and match the expected parameter names and shapes;
- inference is deterministic and produces finite probabilities of the expected length;
- window overlap and boundary consolidation return stable peak times;
- the refractory period prevents duplicate detections;
- QRS warnings do not block correction;
- catastrophic QRS failures do block correction;
- only EEG channels change during PCA-OBS;
- cardiac and scanner QC metrics have known behavior on synthetic signals;
- licensing metadata consistently reports GPL-3.0-only.

Integration tests will establish:

- the NeuXus-adapted detector improves the audited benchmark recordings relative to the
  current generic MNE detector;
- participant 15 receives automatic correction without a manual gate;
- early-participant and high-motion runs retain plausible median heart rates while exposing
  irregular detections as warnings;
- injected neural signals remain within the predefined preservation tolerance;
- incomplete cohort output is not published.

The qualification run will cover all 83 verified recordings and regenerate the cohort
scanner-harmonic and cardiac-locked summaries before the native pipeline is described as a
replacement for BrainVision Analyzer.

## Acceptance Criteria

The implementation is acceptable when:

1. the focused and full repository test suites pass;
2. repository structure, architecture, maintainability, formatting, and lint gates pass;
3. GPL and NeuXus attribution are complete and internally consistent;
4. all cohort runs either produce corrected outputs with explicit QC or fail with a precise
   unusable-detection error;
5. automatic correction requires no BrainVision-generated pulse markers;
6. cohort QC demonstrates gradient and cardiac attenuation without violating the neural
   preservation gates;
7. documentation distinguishes residual scanner harmonics from incomplete correction and
   retains the required harmonic-aware feature exclusions.
