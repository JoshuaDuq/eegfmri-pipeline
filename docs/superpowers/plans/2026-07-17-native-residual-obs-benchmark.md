# Native Residual OBS Qualification Plan

**Goal:** Determine whether cross-fitted, slice-group-locked OBS should be enabled after native 5 kHz gradient AAS without compromising non-scanner-locked EEG.

**Method:** Discover all marker-sanitized recordings dynamically, select the prespecified first run from every participant, and reproduce the native AAS boundary using each run's exact BOLD slice schedule. Evaluate OBS orders 0–4 on the 12 established scanner-QC channels. Qualify the smallest nonzero order only if it improves every prespecified residual line and volume-locked RMS across participants while passing deterministic sinusoid, transient, and outside-line PSD preservation gates.

**Implementation:**

1. Add fixed-frequency spectral metrics and explicit cohort selection gates in `eeg_pipeline/analysis/qc/native_residual_obs.py`.
2. Add `studies/pain_study/scripts/benchmark_native_residual_obs.py` and a strict YAML configuration with no expected cohort count.
3. Cover numerical metrics, configuration, discovery, and decision logic with focused tests.
4. Run the benchmark against the original marker-sanitized 5 kHz data and exact BOLD metadata.
5. Write TSV/JSON audit artifacts and revise `studies/pain_study/SCANNER_HARMONICS_QC_README.md` from the observed results.

**Scientific gates:** fixed residual-line power and local prominence, no run-level prominence increase above tolerance, decreased volume-locked RMS, sinusoid amplitude/phase preservation, transient preservation, and bounded PSD change outside narrow scanner-line exclusions.
