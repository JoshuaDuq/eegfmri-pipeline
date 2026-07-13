# BrainVision Residual Gradient OBS Design

## Purpose

Add an outcome-blind native-Python benchmark for residual scanner-gradient structure that remains
after the established BrainVision Analyzer preprocessing. The benchmark determines whether a
small residual optimal-basis projection can reduce the narrow scanner-locked spectral comb without
materially changing adjacent EEG.

This is not a replacement for the existing BrainVision scanner or pulse correction. It is not a
second implementation of full FASTR. Promotion into routine preprocessing requires the acceptance
criteria in this specification to be met on the prespecified pilot data.

## Verified input boundary

The source history file `/Volumes/KINGSTON/mri_preprocess_aug2025.ehtp` establishes the exact
processing boundary. BrainVision Analyzer 2.3 performs, in order:

1. continuous scanner-artifact template subtraction aligned to `V  1` volume markers with
   `TR = 900 ms`, zero offset, full-interval baseline correction, and a 21-volume sliding average;
2. correction of all 64 channels, including ECG;
3. a 100 Hz, 24 dB/octave IIR low-pass and downsampling from 5,000 Hz to 1,000 Hz within the
   scanner-correction stage;
4. ECG R-peak detection;
5. pulse-artifact correction of the EEG channels using 21 cardiac intervals; and
6. BrainVision export as `*_scannerpulse_corrected.{vhdr,vmrk,eeg}`.

An inspected export has a 1,000 Hz sampling frequency and exposes volume annotations to MNE as
`Volume/V  1`. These files are already gradient-AAS corrected, BCG corrected, filtered, and
downsampled. The residual benchmark must treat them as such.

## Scientific decision

The benchmark applies only residual optimal-basis modeling. It must not:

- repeat average artifact subtraction;
- estimate or shift scanner triggers as if the data were unsynchronized;
- synthesize missing volume markers;
- upsample the 1,000 Hz data under the claim that discarded 5,000 Hz information is recovered;
- repeat pulse-artifact correction;
- notch the measured scanner peaks; or
- overwrite BrainVision sources, BIDS sources, or existing MNE derivatives.

The implementation is named residual OBS rather than FASTR because the input has already passed
the template-subtraction, filtering, downsampling, and BCG stages that form the surrounding FASTR
workflow. This distinction must be retained in code, logs, reports, and methods text.

## Processing architecture

The correction algorithm belongs in a reusable core module. Study-specific discovery, configuration,
benchmarking, and report generation remain in the pain-study layer.

```text
BrainVision *_scannerpulse_corrected.vhdr (1,000 Hz)
    -> strict input and marker validation
    -> immutable MNE Raw load
    -> complete 900-sample volume-epoch extraction
    -> deterministic cross-fitted residual basis estimation
    -> candidate outputs for 0, 1, 2, 3, and 4 components
    -> scanner-harmonic and signal-preservation audits
    -> pilot decision report
```

The initial implementation is a standalone benchmark command. It does not become an implicit step
inside the existing MNE preprocessing pipeline. This separation prevents unvalidated corrected data
from silently replacing the current canonical derivatives.

Proposed ownership:

- `eeg_pipeline/preprocessing/residual_gradient.py`: validation, volume epoching, basis estimation,
  cross-fitted projection, and immutable correction results;
- `studies/pain_study/scripts/benchmark_residual_gradient.py`: study file discovery, pilot
  execution, QC comparison, and derivative writing;
- `studies/pain_study/scripts/config/residual_gradient_benchmark.yaml`: the finite benchmark grid
  and fixed acquisition contract;
- `tests/preprocessing/test_residual_gradient.py`: numerical and failure-contract tests; and
- `tests/scripts/test_benchmark_residual_gradient.py`: discovery, routing, and audit tests.

## Input contract

Every run must satisfy all of the following before numerical correction begins:

- the input is a `.vhdr` file with its referenced `.vmrk` and `.eeg` files present;
- the filename ends in `_scannerpulse_corrected.vhdr`;
- the sampling frequency is exactly 1,000 Hz;
- `Volume/V  1` annotations are present;
- volume onsets map monotonically to samples;
- complete volume epochs contain exactly 900 samples;
- at least 50 complete volume epochs are available;
- at least one EEG channel is present; and
- ECG and every other non-EEG channel are excluded from OBS fitting and subtraction.

A gap longer than one TR defines a boundary between contiguous acquisition blocks. Only complete
900-sample epochs inside a block are eligible. A short, overlapping, duplicated, or non-monotonic
volume interval is an error. Samples before the first complete volume, after the final complete
volume, and inside declared long gaps remain byte-for-byte numerically unchanged.

The implementation must surface violations with the source path and failed invariant. It must not
infer alternate marker labels, resample unexpected inputs, repair markers, or switch algorithms.

## Residual basis estimation

Residual OBS is performed independently for each EEG channel and run. Let the eligible data form a
matrix with one 900-sample row per complete volume epoch.

1. Remove the temporal mean from each epoch. The removed scalar is restored implicitly because the
   correction contains no constant basis vector.
2. Assign volume epochs deterministically to five interleaved folds by acquisition index.
3. For each held-out fold, estimate the temporal principal-component basis from the other four
   folds only. This prevents the held-out epoch's idiosyncratic EEG from contributing to its own
   basis.
4. Fit the requested number of temporal basis vectors to each held-out epoch by ordinary least
   squares and subtract only the fitted projection.
5. Reassemble folds in acquisition order and replace only samples belonging to eligible epochs.

The implementation evaluates exactly 0, 1, 2, 3, and 4 components. Zero components is the unchanged
BrainVision reference. Component signs are normalized deterministically for reproducible audit
outputs. Randomized SVD is prohibited.

This full residual-band model is intentionally more conservative in scope than full FASTR but is
not identical to the original FMRIB high-pass residual PCA. The original high-pass formulation is
primarily aimed at high-frequency residuals; this study must evaluate peaks at approximately 20,
41, 61, and 82 Hz. The deviation must be stated explicitly, and the signal-preservation gates below
are therefore mandatory.

## Configuration

The YAML file exposes only values that define the fixed acquisition or the finite pilot comparison:

- expected sampling frequency: 1,000 Hz;
- volume marker: `Volume/V  1`;
- TR: 0.9 s;
- minimum complete epochs: 50;
- cross-fitting folds: 5;
- candidate component counts: `[0, 1, 2, 3, 4]`;
- pilot subject: `sub-0006`;
- harmonic windows: 18–23, 38–43, 56–67, and 77–85 Hz; and
- deterministic PSD and injected-signal evaluation settings.

There is no automatic component-order mode and no per-run component-order adaptation. After the
pilot decision, a separate production configuration will contain one frozen component count or the
method will be rejected. The benchmark configuration is never interpreted as production approval.

## Outputs and provenance

All outputs are written beneath a dedicated benchmark derivative root. Candidate continuous files
use FIF to preserve MNE channel types and annotations without modifying BrainVision sources. Each
candidate name contains the component count. Existing files are an error unless explicit command
overwrite behavior is requested by the user at invocation time.

Each run produces:

- one unchanged-reference audit row and one row per nonzero component count;
- candidate FIF files for the nonzero component counts;
- a component-variance table by channel and fold;
- a before/after harmonic summary using the repository's established Welch estimator;
- a signal-preservation summary; and
- a machine-readable provenance JSON containing input path, input fingerprint, configuration,
  software versions, eligible blocks, excluded samples, and output paths.

The cohort report weights runs within the pilot subject equally. It never reads pain ratings, fMRI
targets, behavioral measures, or downstream model results.

## Evaluation and acceptance criteria

Artifact suppression and EEG preservation are co-primary. A component count is ineligible if it
fails any preservation criterion, regardless of harmonic attenuation.

### Scanner-residual metrics

For every candidate, compute the existing absolute PSD and local-prominence summaries in the four
prespecified harmonic windows. Also report volume-locked residual RMS before and after correction.
The selected order must:

- reduce pilot median local prominence in every harmonic window relative to zero components;
- reduce pilot median absolute peak power in every harmonic window;
- reduce volume-locked residual RMS; and
- avoid increasing any run's harmonic prominence by more than 1 dB.

No single aggregate score may hide failure in one harmonic window.

### Signal-preservation metrics

Deterministic synthetic tests add known non-scanner-locked sinusoids and transient waveforms to
copies of real pilot data before correction. Frequencies span the retained beta and gamma
regions: 13–18, 23–30, 30.1–38, 43–56, and 67–77 Hz. Events are deliberately not locked to
the 0.9 s volume cycle.

An eligible component count must retain at least 95% of injected sinusoid amplitude, keep injected
phase error below 5 degrees, retain at least 95% of transient peak amplitude, and change median PSD
outside the harmonic windows by no more than 0.5 dB. The audit also reports the maximum channel-wise
change so that a stable cohort median cannot conceal localized distortion.

### Selection rule

Select the smallest component count satisfying every artifact-suppression and signal-preservation
criterion across all six `sub-0006` runs. If no nonzero count passes, residual OBS is rejected and
the existing BrainVision preprocessing remains canonical. The code must not choose a weaker
criterion, change frequency windows, or transition to a raw-data method automatically.

## Testing

Unit tests use synthetic MNE `RawArray` objects and temporary BrainVision fixtures where integration
behavior is required. They cover:

- exact 1,000 Hz, marker, channel-type, and filename validation;
- missing companion files;
- complete-epoch extraction and long-gap boundaries;
- rejection of short, overlapping, duplicate, and non-monotonic intervals;
- exclusion and numerical preservation of ECG and other non-EEG channels;
- unchanged samples outside eligible epochs;
- exact identity for zero components;
- deterministic five-fold cross-fitting;
- known low-rank residual recovery;
- preservation of injected non-volume-locked signals;
- annotation, sample-count, channel-order, and measurement-date preservation;
- finite output values and bounded removed variance;
- deterministic report schemas and provenance; and
- refusal to overwrite existing benchmark derivatives by default.

The full pilot run is a data validation step, not a repository unit test. Completion requires
reviewing its TSV/JSON audits and before/after spectra before designing production integration.

## Failure policy

Errors surface immediately. The benchmark does not contain format fallbacks, alternate marker
searches, automatic interpolation, automatic component selection, implicit overwrites, or silent
run exclusion. A failed run makes the benchmark command fail after writing no partial corrected
file for that run.

## Literature basis

The design follows the residual optimal-basis concept of Niazy et al. (2005), while respecting the
FMRIB warning that component order depends on the dataset and that additional components can remove
EEG. It retains synchronized AAS as the established first correction, consistent with Allen et al.
(2000) and Mandelkow et al. (2006). Its paired artifact-reduction and signal-preservation evaluation
follows Ritter et al. (2007) and the frequency-dependent recovery concerns demonstrated by Ryali et
al. (2009). The benchmark does not claim that post-BCG residual OBS is canonical FASTR.
