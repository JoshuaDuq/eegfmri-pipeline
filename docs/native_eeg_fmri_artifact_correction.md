# Native EEG-fMRI MRI-artifact correction

This is the native candidate replacement for the BrainVision Analyzer MRI-artifact stages. It reads
the immutable original 5 kHz BrainVision signal from `source_data/sub-*/eeg/original_5khz`
through the marker-sanitized metadata derivative and writes a separate 1 kHz FIF derivative. It
never reads `source_data/sub-*/eeg/brainvision_processed_1khz`, which contains the existing
BrainVision Analyzer-corrected 1 kHz recordings.
The candidate is not considered a validated replacement until the fixed 83-run qualification
finishes successfully.

## Fixed processing sequence

1. Require a 5 kHz, 64-channel recording with one channel named `ECG`.
2. Select only exact `Volume/V  1` annotations. Any remaining marker type ending in `/V  1` is an
   error.
3. Require a 0.9-second scanner grid (4,500 samples). Marker quantization may differ from that grid by
   at most one sample. The one affected cohort run is canonicalized to the grid because the artifact
   correlation is 0.9998 on-grid and 0.9542 at the delayed marker.
4. Crop a final triggered scanner interval when the EEG recording ends before that interval is
   complete. Cropping is rejected if the interval contains a task annotation.
5. Correct the gradient artifact at 5 kHz with phase-aligned average artifact subtraction:
   a centered, leave-one-out 21-volume template; fourfold interpolation for fractional phase
   estimation; and least-squares template plus temporal-derivative fitting. The ECG receives gradient
   subtraction but is excluded from the alignment reference.
6. Apply a zero-phase 100 Hz FIR low-pass and polyphase resampling to 1 kHz.
7. Filter a detection-only ECG copy from 0.5 to 30 Hz, resample it to 250 Hz, and detect R peaks with
   the NeuXus v0.0.4 bidirectional LSTM using overlapping two-second windows.
8. Require finite, ordered, in-range peaks, a median heart rate of 40–160 bpm, at least ten peaks,
   and at least 90% temporal coverage. Individual RR intervals outside 0.375–1.5 seconds are retained
   as warnings rather than silently repaired.
9. Quantify cardiac-locked EEG from -0.2 to 0.6 seconds, apply MNE-Python `apply_pca_obs` with four
   components to EEG channels only, and repeat the same cardiac-locked measurement.
10. Save single-precision FIF, QRS TSV, separate physiological and scanner-spectrum QC PNGs,
    SHA-256 provenance/QC JSON, and a cohort manifest in an atomically published versioned
    derivative root.

No notch filter is used for gradient correction. A notch centered at 60 Hz cannot remove the observed
61.10 Hz scanner line and would instead remove neighboring physiological signal. MNE
`annotate_amplitude` is likewise not a gradient correction: it is suitable for detecting sustained
flat or extreme consecutive-sample changes after MRI correction.

## Scientific basis

- Synchronized average artifact subtraction follows Allen et al. (2000),
  [doi:10.1006/nimg.2000.0599](https://doi.org/10.1006/nimg.2000.0599). Scanner/EEG clock
  synchronization is specifically supported by Mandelkow et al. (2006),
  [doi:10.1016/j.neuroimage.2006.04.231](https://doi.org/10.1016/j.neuroimage.2006.04.231).
- Fractional alignment and per-volume template/derivative fitting address small sampling-phase and
  amplitude changes without adding a full-band residual PCA stage. These are conservative
  study-specific refinements of synchronized AAS and are tested for neural-signal preservation.
- NeuXus was developed for EEG-fMRI and evaluated its LSTM R-peak detector against manual labels
  (Caetano et al., 2023),
  [doi:10.1016/j.neuroimage.2023.120353](https://doi.org/10.1016/j.neuroimage.2023.120353). Its model
  is related to the bidirectional LSTM method of Laitala et al. (2020),
  [doi:10.1145/3341105.3373945](https://doi.org/10.1145/3341105.3373945). Because the NeuXus training
  cohort was limited, this pipeline preserves probability, RR, coverage, and cardiac-attenuation QC
  rather than treating every prediction as equally trustworthy.
- Pulse artifact removal uses [MNE-Python's maintained `apply_pca_obs`
  implementation](https://mne.tools/stable/generated/mne.preprocessing.apply_pca_obs.html) of the
  optimal-basis method from Niazy et al. (2005),
  [doi:10.1016/j.neuroimage.2005.06.067](https://doi.org/10.1016/j.neuroimage.2005.06.067), rather
  than a study-specific OBS reimplementation.

The fMRI metadata contain 54 slices with multiband factor 3, hence 18 acquisition groups per volume.
Those group waveforms must not be pooled into a generic slice template: in the qualification run,
adjacent-group median waveform correlation was 0.07, whereas the same group in successive volumes was
0.99997. The pipeline therefore retains the complete volume waveform.

A custom residual PCA across fixed-TR epochs, a motion-selected template bank, adaptive cancellation,
and harmonic notch banks were rejected after the study-data audit: they did not materially improve
the residual comb and residual PCA removed injected neural signal. This does not imply that
motion-informed templates are ineffective in general; they require reliable external motion or fMRI
realignment regressors, which this EEG-only boundary does not have. PCA-OBS remains appropriate for
heartbeat epochs because cardiac timing is not phase-locked to the scanner grid.

## Qualification and QC

Every run reports the raw, gradient-corrected, and final spectra using equal-duration 16.384-second
MNE Welch windows. The raw scanner-line frequency in each fixed window is carried forward so
attenuation is measured at the same frequency and with the same frequency resolution at 5 kHz and
1 kHz. Local prominence is the line power relative to neighboring bins 0.35–2 Hz away. The run-level
scanner-spectrum figure plots the exact channel-median spectra used for these metrics. A full
15–90 Hz panel shows the complete comb, and four enlarged panels cover 18–23, 38–43, 56–67, and
77–85 Hz. Raw, gradient-corrected, and final curves share one scale across the enlarged panels. Each
panel reports AAS attenuation and final local prominence at the matched raw reference frequency.
This retains peak shape, sidebands, and the neighboring spectral floor rather than reducing each
line to a bar. A separate physiological figure gives the ECG/QRS probability window, complete RR
timeline, and cardiac-locked EEG before and after OBS enough space for independent inspection.

The cohort figure reports run-level distributions of raw-to-final harmonic attenuation, final local
prominence, and cardiac-locked attenuation. Its QRS panel jointly displays median heart rate and the
fraction of warning-level RR intervals, so physiological plausibility and interval burden cannot be
mistaken for independent endpoints.

Each run also records the pinned NeuXus model checksum, peak probabilities and window support, all RR
summary fields and warnings, and the RMS and peak-to-peak cardiac-locked EEG before and after OBS.
Hard QRS failures stop that run; warning-level RR intervals do not. The complete cohort is published
only if every one of the 83 verified marker-sanitized recordings finishes.

Until the v2 cohort QC demonstrates otherwise, downstream spectral features retain the established
scanner-aware frequency boundaries:

- beta: 13–18 Hz and 23–30 Hz;
- gamma: 30.1–38 Hz, 43–56 Hz, and 67–77 Hz.

These exclusions are not evidence that correction failed. They prevent narrow residual scanner lines
from dominating band summaries while absolute attenuation and local prominence are evaluated
separately.

Run the fixed cohort with:

```bash
python -m studies.pain_study.scripts.run_native_eeg_fmri_artifact_correction
```

The default output is
`/Volumes/KINGSTON/EEG_fMRI_data/derivatives/native_eeg_fmri_correction-v2`. An existing v2 output or
incomplete v2 staging tree is never reused.

`mne.preprocessing.annotate_amplitude` remains a downstream generic bad-segment detector. It can
identify sustained flat or extreme consecutive-sample changes after MRI correction, but it cannot
model a synchronized gradient waveform or a cardiac-locked pulse artifact.
