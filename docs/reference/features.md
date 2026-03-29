# Feature Reference

This is the technical reference for trialwise EEG feature extraction.
Each accepted epoch can contribute one row to a feature matrix aligned by `trial_id`.

## Main Feature Families

- power and spectral summaries
- aperiodic summaries
- ERP and ERDS measurements
- phase metrics including ITPC and PAC
- connectivity graphs and directed metrics
- complexity, bursts, and microstates
- source-space summaries
- quality-control metrics

## Notation

| Symbol | Meaning |
| --- | --- |
| $e$ | trial index |
| $c$ | channel index |
| $f$ | frequency |
| $t$ | time |
| $P_{e,c}(f,t)$ | time-frequency power |
| $\mathrm{PSD}_{e,c}(f)$ | power spectral density |

## Windows, Bands, And IAF

Feature extraction is window-aware and band-aware.
When IAF is enabled, band definitions are adjusted from baseline spectral content and
kept isolated to the relevant training data in cross-validation-safe contexts.

## Shared Intermediates

The pipeline precomputes reusable spectral and windowed intermediates so multiple
feature families can share the same validated inputs instead of recomputing them.

## Normalization And CV Hygiene

Public ML-facing feature paths require train-only statistics for normalization and
fold-specific handling of any IAF-dependent operations. The repo does not allow
test-fold information to shape the learned feature space.

## Outputs

Expected outputs:

- parquet feature tables
- metadata describing selected families, windows, and bands
- QC fields for missingness, finite fractions, and transform status

## Related Pages

- public workflow: [feature extraction](../workflows/feature-extraction.md)
- source-space details: [source localization](../eeg/source-localization.md)
