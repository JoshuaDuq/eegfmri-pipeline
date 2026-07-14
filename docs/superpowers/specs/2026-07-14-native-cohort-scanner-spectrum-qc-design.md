# Native Cohort Scanner-Spectrum QC Design

## Purpose

Extend the native EEG-fMRI correction run so that its final cohort artifact includes a true
cohort analogue of every run-level scanner-gradient spectral QC figure. The existing
`cohort_mriartifact_qc.png` remains a distribution summary; the new output shows the complete
raw, gradient-corrected, and final power spectra.

## Scientific aggregation

The unit of cohort inference is the participant, not the run. Each run already contains a Welch
spectrum formed by taking the pointwise median linear PSD across the 12 prespecified EEG channels
and converting it to dB V²/Hz. The cohort calculation will:

1. retain only the common 15–90 Hz frequency grid from each completed run;
2. take the pointwise median across runs within each participant;
3. take the equally weighted pointwise median across participants;
4. calculate deterministic 95% paired participant-bootstrap intervals;
5. derive the cohort scanner references from the raw cohort spectrum and use those same
   frequencies for gradient-corrected and final attenuation and prominence annotations.

All stages must have identical frequency bins after the explicit 15–90 Hz selection. Missing
stages, inconsistent bins, duplicate participant/run identities, non-finite values, or invalid
bootstrap settings are errors.

## Figure and audit output

The pipeline will automatically write two new root-level artifacts before atomic publication:

- `cohort_scanner_spectrum_qc.png`, a 300-dpi five-panel figure matching the run-level layout;
- `cohort_scanner_spectrum_qc.tsv`, containing stage, frequency, cohort median, confidence bounds,
  participant count, and run count for every plotted point.

Panel A covers 15–90 Hz. Panels B–E enlarge 18–23, 38–43, 56–67, and 77–85 Hz. Every panel
shows raw, gradient-corrected, and final cohort medians with restrained participant-bootstrap
bands. Local panels retain the raw-reference frequency, AAS attenuation, and final local
prominence annotations used by the run figures. The title states the participant-first
aggregation, number of participants, number of runs, and channel count.

## Pipeline integration

`process_recording` will return a small immutable completed-run object containing its manifest row
and only the cropped scanner spectra needed for cohort aggregation. It will not retain the MNE Raw
object or full 5 kHz frequency vectors. `run_cohort` will collect these small records, write the
existing manifest and distribution QC, aggregate the cohort spectra, write the TSV and spectral
figure, and then atomically publish the derivative root.

The versioned default derivative becomes `native_eeg_fmri_correction-v3`. Corrected FIF behavior is
unchanged; the version increment reflects the stronger output and audit contract and prevents
overwriting the completed v2 derivative.

## Reproducibility and validation

Bootstrap iterations, confidence level, and random seed live in the native correction YAML and
are validated strictly. Tests will establish participant-first weighting, deterministic confidence
intervals, exact frequency validation, five-panel plot semantics, TSV content, atomic cohort
publication, and the versioned default output. Focused tests, Ruff, architecture checks, and a full
83-run Kingston execution will be completed. The resulting cohort figure will be inspected at
original resolution before acceptance.
