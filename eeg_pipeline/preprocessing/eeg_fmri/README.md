# Native EEG-fMRI artifact correction

This pipeline replaces the BrainVision Analyzer scanner-gradient and pulse-artifact stages. It
reads the original marker-sanitized 5 kHz BrainVision recordings and writes a separate 1 kHz FIF
derivative. It does not use the previously Analyzer-corrected 1 kHz files.

## Inputs

Each run requires:

- a 5,000 Hz, 64-channel BrainVision recording containing an `ECG` channel;
- exact scanner-volume annotations named `Volume/V  1`;
- the matching BIDS BOLD JSON containing `RepetitionTime`, `SliceTiming`, and
  `MultibandAccelerationFactor`.

Marker sanitation is performed before this pipeline. Task markers must not retain the scanner code
`V  1`.

The default paths are:

| Purpose | Path |
| --- | --- |
| Marker-sanitized BrainVision metadata | `/Volumes/KINGSTON/EEG_fMRI_data/derivatives/brainvision_marker_sanitized-v2` |
| Referenced original signal | `/Volumes/KINGSTON/EEG_fMRI_data/source_data/sub-*/eeg/original_5khz` |
| Run-matched BOLD metadata | `/Volumes/KINGSTON/EEG_fMRI_data/bids_output/fmri` |
| Corrected 1 kHz output | `/Volumes/KINGSTON/EEG_fMRI_data/source_data/native_eeg_fmri_processed_1khz` |

The marker-sanitized headers preserve references to the immutable `.eeg` files in
`original_5khz`; the pipeline does not duplicate or modify those source signals.

## Run the cohort

From the repository root, with the project environment activated:

```bash
python -m studies.pain_study.scripts.run_native_eeg_fmri_artifact_correction
```

The default output is:

```text
/Volumes/KINGSTON/EEG_fMRI_data/source_data/native_eeg_fmri_processed_1khz
```

Paths can be specified explicitly:

```bash
python -m studies.pain_study.scripts.run_native_eeg_fmri_artifact_correction \
  --input-root /path/to/marker-sanitized-brainvision \
  --bold-root /path/to/bids/fmri \
  --output-root /path/to/native_eeg_fmri_processed_1khz \
  --config studies/pain_study/scripts/config/native_eeg_fmri_artifact_correction.yaml
```

The cohort runner expects the fixed 83-run manifest. It writes into a temporary sibling directory
and publishes the derivative atomically only after every run succeeds. Existing output and
incomplete staging directories are never overwritten.

## Correction sequence

### 1. Acquisition and timing validation

The pipeline requires a 0.9-second repetition time, equal to 4,500 samples at 5 kHz. It selects only
exact `Volume/V  1` annotations, detects scanner blocks separated by acquisition restarts, and
canonicalizes marker deviations of at most one sample. A final incomplete scanner interval may be
cropped only when it contains no task annotation.

The run-matched BOLD metadata are validated against the EEG repetition time. The study acquisition
contains 54 slices with multiband factor 3, yielding 18 simultaneous acquisition groups per volume.
Group timing is retained for alignment diagnostics; gradient subtraction operates on the complete
volume waveform.

### 2. Scanner-gradient artifact subtraction at 5 kHz

Scanner artifact is corrected with synchronized, phase-aligned average artifact subtraction (AAS):

1. EEG channels provide a robust shared alignment reference; ECG is excluded from this reference.
2. A fractional timing shift is estimated for every volume using fourfold interpolation, with a
   maximum shift of two native samples.
3. Each volume receives a centered, leave-one-out template derived from 21 neighboring volumes.
4. The template and its temporal derivative are fitted by least squares to accommodate small
   amplitude and timing differences.
5. The fitted artifact is subtracted independently from every channel. ECG receives gradient
   subtraction but does not influence alignment.

Whole-volume templates preserve continuity between multiband acquisition groups. Residual
scanner-locked PCA/OBS is implemented but disabled (`residual_obs_components: 0`) because the fixed
study-data qualification did not improve the scanner lines and showed non-harmonic signal
distortion. No notch filter is used for scanner-gradient correction.

### 3. Low-pass filtering and resampling

After gradient subtraction, all channels receive a zero-phase 100 Hz Hamming-window FIR low-pass.
The recording is then resampled from 5,000 Hz to 1,000 Hz using polyphase resampling.

### 4. R-peak detection

R peaks are detected from a separate ECG detection copy:

1. band-pass filter from 0.5 to 30 Hz;
2. resample to 250 Hz;
3. apply the NeuXus v0.0.4 bidirectional-LSTM detector using overlapping two-second windows;
4. if the NeuXus result fails validation, try MNE automatic QRS detection and then a
   Pan-Tompkins-style detector.

Accepted peaks must be finite, ordered, and in range, with at least ten peaks, 40–160 bpm median
heart rate, and at least 90% temporal coverage. RR intervals outside 0.375–1.5 seconds are retained
and reported as warnings rather than replaced with a fixed delay.

### 5. Pulse-artifact correction

Cardiac-locked EEG is measured from -0.2 to 0.6 seconds around the accepted R peaks. MNE-Python
`apply_pca_obs` then removes four optimal-basis components from EEG channels only. ECG is preserved.
The same cardiac-locked measurement is repeated after correction to quantify attenuation.

### 6. Quality control and output

Scanner spectra are computed before gradient correction, after gradient correction, and after pulse
correction. Each run receives a five-panel 15–90 Hz scanner-comb figure with enlarged windows at
18–23, 38–43, 56–67, and 77–85 Hz. A separate physiological figure reports ECG detection, the RR
timeline, and cardiac-locked EEG before and after pulse correction.

Each run writes:

- `*_desc-mriartifactclean_raw.fif`: corrected 1 kHz recording;
- `*_qrs.tsv`: accepted R peaks and detection information;
- `*_qc.json`: parameters, timing diagnostics, software versions, hashes, and numerical QC;
- `*_physiology_qc.png`: R-peak and pulse-correction QC;
- `*_scanner_spectrum_qc.png`: scanner-harmonic spectral QC.

After all runs succeed, the root derivative also contains the cohort manifest,
`cohort_mriartifact_qc.png`, `cohort_scanner_spectrum_qc.png`, and the numerical cohort spectrum TSV.
Cohort spectra aggregate runs within participants before estimating the equally weighted cohort
median and deterministic 95% participant-bootstrap intervals.

## Scientific basis

- Synchronized AAS: Allen et al. (2000), [doi:10.1006/nimg.2000.0599](https://doi.org/10.1006/nimg.2000.0599).
- Scanner/EEG clock synchronization: Mandelkow et al. (2006), [doi:10.1016/j.neuroimage.2006.04.231](https://doi.org/10.1016/j.neuroimage.2006.04.231).
- NeuXus EEG-fMRI processing: Caetano et al. (2023), [doi:10.1016/j.neuroimage.2023.120353](https://doi.org/10.1016/j.neuroimage.2023.120353).
- Optimal-basis pulse correction: Niazy et al. (2005), [doi:10.1016/j.neuroimage.2005.06.067](https://doi.org/10.1016/j.neuroimage.2005.06.067).

The fixed configuration is
[`studies/pain_study/scripts/config/native_eeg_fmri_artifact_correction.yaml`](../../../studies/pain_study/scripts/config/native_eeg_fmri_artifact_correction.yaml).
