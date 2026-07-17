# Scanner-Harmonic QC: Native Correction and Final MNE Data

## Decision

The new native EEG-fMRI pipeline is functioning correctly, but it does not completely remove the
scanner-gradient comb. Whole-volume adaptive average artifact subtraction (AAS) removes
approximately 26–29 dB at the four dominant raw scanner lines. Final MNE preprocessing suppresses
most of those dominant lines further, but several narrower scanner-linked residuals remain.

- With respect to scanner-gradient contamination, the corrected data are suitable for conventional
  low-frequency analyses.
- Retain beta at 13–18 Hz and 23–30 Hz. The strong approximately 20 Hz scanner cluster remains
  excluded.
- Do not treat the current combined gamma intervals (30.1–38, 43–56, and 67–77 Hz) as a clean
  confirmatory gamma estimand. The 43–56 Hz interval contains reproducible residuals at 51.57 and
  52.80 Hz after final MNE preprocessing.
- Keep gamma exploratory unless scanner-linked bins are explicitly masked and the result is shown
  to be robust to reasonable mask widths.
- Do not replace AAS with ordinary notch filters. Keep residual gradient OBS disabled: both
  cross-fitted native-boundary formulations failed the cohort qualification.

The `Vas_on/V  1` marker defect was not the cause of the scanner harmonics. The gradient correction
uses the `Volume/V  1` markers, not VAS events. Sanitizing the VAS label was necessary for event
parsing and EEG-BIDS conversion, but it cannot explain the spectral correction result.

## Data and estimators

This review covers the complete current numbered cohort: 90 thermal-task runs from 15 participants,
with six runs per participant (`sub-0000`, `sub-0001`, and `sub-0003` through `sub-0015`).

The native QC compares:

1. original 5 kHz EEG;
2. 5 kHz phase-aligned whole-volume AAS output; and
3. final 1 kHz output after low-pass filtering, resampling, and cardiac OBS.

Native spectra use 16.384-second Welch segments (0.0610 Hz bins), 50% overlap, and the median across
12 prespecified EEG channels. Runs are aggregated within participant before the cohort median and
participant bootstrap confidence interval are calculated.

The standard end-of-MNE plot uses 4-second segments (0.25 Hz bins). Because this grid under-resolves
the narrow residual lines, the final clean epochs were also audited with 16.384-second segments,
0.0610 Hz bins, and the same 12 channels. Local prominence is the line power minus the median power
0.35–2.0 Hz on either side. The 3 dB counts below are descriptive prevalence summaries, not a
preregistered pass/fail threshold.

## Timing integrity

All 90 runs use the same verified acquisition parameters: TR = 0.9 seconds, 54 slices, multiband
factor 3, 5 kHz input, and 1 kHz native output. The volume-marker grid is internally consistent:

- 89 runs had no marker offset;
- `sub-0009` run 5 had one marker offset of one 5 kHz sample (0.2 ms), within tolerance;
- the largest estimated AAS alignment shift was 0.556 input samples (0.111 ms); and
- only the incomplete terminal fraction after the final complete 0.9-second volume was discarded.

These results rule out gross volume-marker or synchronization failure.

## Native correction

| Raw reference | Raw PSD | Native final PSD | Total attenuation | Native final prominence | Runs above 1 dB |
|---:|---:|---:|---:|---:|---:|
| 20.02 Hz | -82.90 dB | -110.19 dB | 27.29 dB | 8.18 dB | 89/90 |
| 41.14 Hz | -72.93 dB | -99.05 dB | 26.12 dB | 16.49 dB | 90/90 |
| 61.10 Hz | -68.64 dB | -95.28 dB | 26.64 dB | 20.93 dB | 90/90 |
| 82.21 Hz | -70.20 dB | -99.10 dB | 28.90 dB | 17.98 dB | 90/90 |

AAS accounts for most of the attenuation: 25.23–26.32 dB at these four references. The remaining
native operations add only approximately 0.6–2.6 dB. The correction is therefore substantial and
stable, but the residual is still periodic and locally prominent, especially near 41, 61, and
82 Hz.

Run-level median attenuation was 26.38–27.26 dB across the four windows. Every run retained more
than 1 dB prominence at the 41, 61, and 82 Hz references; all 90 runs retained more than 6 dB at
61 and 82 Hz. This is a cohort-wide residual, not an isolated-participant failure.

## Final MNE boundary

At matched 0.0610 Hz resolution, final MNE preprocessing largely suppresses the original dominant
20.02, 41.14, and 82.21 Hz references. The 61.10 Hz reference remains visible at 4.01 dB cohort
prominence. More importantly, the final spectrum retains other peaks that coincide with lines in
the original scanner comb:

| Scanner-linked frequency | Final cohort prominence | Participants above 3 dB | Current interval |
|---:|---:|---:|---|
| 37.17 Hz | 3.82 dB | 7/15 | retained gamma |
| 38.39 Hz | 4.36 dB | 12/15 | excluded gamma |
| 51.57 Hz | 8.85 dB | 15/15 | retained gamma |
| 52.80 Hz | 5.30 dB | 12/15 | retained gamma |
| 57.19 Hz | 12.85 dB | 13/15 | excluded gamma |
| 61.10 Hz | 4.01 dB | 8/15 | excluded gamma |
| 83.98 Hz | 4.71 dB | 12/15 | excluded gamma |

The 0.25 Hz standard MNE QC plot visually attenuates these narrow lines through frequency-bin
averaging. It remains useful as a broadband preprocessing comparison, but it must not be used alone
to declare the scanner comb absent.

## Feature implications

The final matched-resolution audit found no cohort scanner-linked peak above 3 dB in 13–18 or
23–30 Hz. The existing beta restriction is therefore supported.

The gamma decision is different:

- 30.1–38 Hz contains the 37.17 Hz residual;
- 43–56 Hz contains the reproducible 51.57 and 52.80 Hz residuals; and
- 67–77 Hz has no cohort scanner-linked peak above 3 dB, although participant-specific residuals
  remain.

Consequently, joining all three intervals into one confirmatory gamma feature is not justified by
the new data. A gamma analysis should either remain exploratory or use an explicitly documented
frequency mask with participant-level sensitivity analyses. The 60 Hz notch is not a solution:
scanner lines occur beside 60 Hz and throughout the comb, while a conventional notch removes only
the nominal line-frequency neighborhood.

## Residual OBS

Residual OBS was tested at the correct native boundary: 5 kHz data immediately after native AAS
and before low-pass filtering, resampling, or cardiac OBS. The benchmark used run 1 from all 15
participants, the 12 prespecified QC channels, exact BOLD timing, fivefold cross-fitting, and nested
component orders 1–4. Candidate selection required at least 1 dB reduction in both power and local
prominence at every fixed residual line, at least 1% reduction in volume-locked RMS, and preservation
of injected non-volume-locked sinusoids, transients, and non-line PSD.

Two formulations were tested independently:

| Model | Order | 51.57 Hz power reduction | 57.19 Hz power reduction | 61.10 Hz power reduction | Minimum injected-sinusoid ratio | Maximum channel non-line PSD change | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| Slice-group OBS | 1 | -0.02 dB | -0.13 dB | 0.00 dB | 0.440 | 2.38 dB | Reject |
| Slice-group OBS | 4 | -0.40 dB | 0.57 dB | 0.00 dB | 0.087 | 4.34 dB | Reject |
| Whole-volume OBS | 1 | -0.00 dB | -0.00 dB | 0.00 dB | 0.985 | 0.13 dB | Reject |
| Whole-volume OBS | 4 | -0.04 dB | -0.05 dB | 0.00 dB | 0.961 | 0.27 dB | Reject |

Positive values denote attenuation; negative values denote increased power. Slice-group OBS removed
9.34–10.91 µV RMS from the data but did not reduce the dominant 61.10 Hz line and materially altered
valid EEG. Whole-volume OBS preserved the injected signals and non-line spectrum, but it produced
essentially no scanner-line or volume-locked RMS improvement. No component order passed.

This is strong evidence against enabling top-variance residual PCA/OBS in production. The failure is
not merely a poor component-count choice: one formulation was unsafe, while the safer formulation
was ineffective across every tested order. Production therefore correctly remains at zero residual
gradient-OBS components. If additional correction is pursued, it should test a different mechanism—
for example, faster adaptive or motion-stratified AAS templates—under the same signal-preservation
gates, not add more unconstrained PCA components.

## Conclusion

The acquisition is usable and the new pipeline performs a valid, substantial scanner-gradient
correction. The remaining problem is incomplete suppression of a narrow, reproducible residual
comb—not failed synchronization and not the VAS marker label. Cross-fitted residual OBS does not fix
that residual and must remain disabled. Beta is adequately protected by the current exclusion around
20 Hz. Gamma is improved but is not yet clean enough for an unrestricted confirmatory interpretation.

## Evidence files

- Native cohort spectrum:
  `/Volumes/KINGSTON/EEG_fMRI_data/source_data/native_eeg_fmri_processed_1khz/cohort_scanner_spectrum_qc.tsv`
- Native run manifest:
  `/Volumes/KINGSTON/EEG_fMRI_data/source_data/native_eeg_fmri_processed_1khz/native_correction_manifest.tsv`
- MNE cohort spectrum:
  `/Volumes/KINGSTON/EEG_fMRI_data/derivatives/native_mne_preprocessing/preprocessed/eeg/qc/task-thermalactive_desc-scannerharmoniccomb_qc.tsv`
- Final clean epochs:
  `/Volumes/KINGSTON/EEG_fMRI_data/derivatives/native_mne_preprocessing/preprocessed/eeg/sub-*/eeg/*_proc-clean_epo.fif`
- Slice-group native residual-OBS benchmark:
  `outputs/native_residual_obs_benchmark/`
- Whole-volume native residual-OBS benchmark:
  `outputs/native_residual_obs_whole_volume_benchmark/`

## References

Allen, P. J., Josephs, O., & Turner, R. (2000). A method for removing imaging artifact from
continuous EEG recorded during functional MRI. *NeuroImage, 12*, 230–239.
https://doi.org/10.1006/nimg.2000.0599

Mullinger, K. J., Yan, W. X., & Bowtell, R. (2011). Reducing the gradient artefact in simultaneous
EEG-fMRI by adjusting the subject's axial position. *NeuroImage, 54*, 1942–1950.
https://doi.org/10.1016/j.neuroimage.2010.09.079

Niazy, R. K., Beckmann, C. F., Iannetti, G. D., Brady, J. M., & Smith, S. M. (2005). Removal of
fMRI environment artifacts from EEG data using optimal basis sets. *NeuroImage, 28*, 720–737.
https://doi.org/10.1016/j.neuroimage.2005.06.067
