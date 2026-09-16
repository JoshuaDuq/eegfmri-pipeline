# Study 1 code and scientific review — 2026-09-15

## Corrections applied

1. **SIIPS1 deep-regression nuisance input was dropped.** The training function
   correctly requests NPS adjustment, but dataset assembly excluded NPS from its
   metadata. The normal SIIPS1 path could therefore fail before fitting. Assembly
   now preserves NPS for SIIPS1; a loader-level regression test verifies its values
   and participant order.
2. **CPU early stopping did not preserve the best model.** Detached CPU tensors
   still shared storage with the live model. Subsequent updates overwrote the
   saved checkpoint. Checkpoints now clone their tensors. A controlled training
   test makes epoch one optimal and verifies restoration after two worse epochs.
3. **The deep-regression temporal audit could accept mismatched trials.** It only
   checked overlapping time keys and compared target values rather than trial
   identities. Entirely shifted times, swapped times with equal target values,
   and missing event times could pass. Every matched trial now must have the same
   rounded onset/duration key as its prepared target row, independently of target
   values. Missing or duplicate target identifiers also fail explicitly.
4. **Exploratory feature results lacked the documented multiplicity correction.**
   The report now applies Holm correction jointly to the available exploratory
   feature cells. The primary gate, secondary confirmatory family, and temporal
   comparison family retain their separate corrections. Regenerate reports after
   all intended exploratory cells have completed; an incomplete report cannot
   account for analyses whose outputs have not been supplied.
5. **Invalid HRF support was silently replaced with zero nuisance values.**
   Target preparation now rejects non-finite timing, nonpositive durations, and
   absent or non-finite HRF support with a contextual error. Synthetic BOLD inputs
   reproduce the original zero-substitution defect without needing cohort data.
6. **Scientific descriptions overstated or misidentified implemented operations.**
   The protocol now distinguishes Hilbert amplitude envelopes from power,
   describes R² as unbounded below, explains the actual HRF-weighted nuisance
   summaries and their preparation before cross-validation, and describes
   permutation error handling accurately. It explicitly conditions confirmatory
   permutation interpretation on empirical null calibration.

## Continued review: additional corrections

7. **Whole-brain validity GLMs ignored the slice-timing reference.** These models
   used Nilearn's default `slice_time_ref=0`, unlike the signature-target models.
   A synthetic corrected-data case reproduced a half-TR design shift. The validity
   model now reads the BOLD sidecar reference, records it in the design audit,
   and rejects inconsistent references within a multi-run participant fit.
8. **Deep EEG standardization depended on voltage units.** A fixed standard
   deviation cutoff of `1e-6` bypassed scaling of varying sub-microvolt inputs,
   although MNE returns EEG in volts. Standardization now divides every varying
   band/channel by its training standard deviation. Constant features still
   center to zero. A unit-rescaling test reproduces and prevents the defect.
9. **Frontal nuisance rasterization ignored actual frame timestamps.** Separately
   rounding onset and duration to indices both ignored the slice reference and
   changed event boundaries. The proxy now occupies frames satisfying
   `onset <= frame_time < onset + duration`. A synthetic half-TR example changed
   the resulting weighted proxy from 1.700677 to the timestamp-based 1.190932;
   these numbers demonstrate the defect, not its size in the cohort.
10. **Fractional deep-model identifiers were silently rounded.** Run 1.2 could
    become run 1 and trial 1.2 could become trial 1. Both must now be integer-valued
    before constructing trial keys.
11. **Prediction metadata could invent replacement trial numbers.** Alignment
    accepted the existing `epoch` identifier, but output assembly replaced it with
    sequential rows. Both paths now use the same identifier resolver and preserve
    gaps left by censoring. A loader test preserves trial IDs 2 and 5.
12. **Partially recorded analysis windows were silently clipped.** Deep feature
    construction only required overlap with the requested interval; MNE cropping
    could then shorten it. Full coverage is now required within half a sample,
    preventing a shorter window from being labeled as the requested analysis.

## Further review: time axes and report integrity

13. **Equal tensor shapes could conceal different EEG time axes.** Participant
    tensors were concatenated without checking the time represented by each
    sample. Synthetic equal-length epochs with differing sampling frequencies or
    onset offsets both passed the old loader. Band construction now returns its
    actual cropped MNE time axis, and assembly requires agreement across subjects
    to one nanosecond. Different epoch padding remains valid if the retained
    sample times agree; a separate regression verifies this case.
14. **Primary report validation accepted invalid numeric results.** Infinity
    passed the old missing-value check, and integer casts silently truncated
    fractional permutation counts. Required statistics must now be finite;
    counts must be nonnegative integers; the primary incremental p-value and
    exclusion fraction must lie in [0, 1].
15. **Holm correction silently reduced incomplete families and accepted invalid
    probabilities.** Missing or nonnumeric entries were dropped, potentially
    understating the correction, while out-of-range values reached statsmodels.
    A populated family now requires a finite probability in [0, 1] for every
    reported cell. Optional statistics absent for the whole family remain absent.
    This does not detect result directories that were never supplied: reports
    still require all intended exploratory analyses to have completed.
16. **Folder location could confer confirmatory status on unprespecified models.**
    Any non-gate result under `primary` was labeled secondary confirmatory.
    Classification now rejects targets, feature presets, or estimators outside
    the actual prespecified grid. Tests cover an ROI target, broadband features,
    and a Random Forest model placed in that partition.

## Scientific limits requiring empirical evidence

- The complete circular-shift group does not prove exchangeability. Assess false
  positives with realistic null simulations of the complete nested procedure,
  including censoring, run boundaries, nuisance estimation, and actual timing.
- Signature-manifest validation checks declared provenance and image properties;
  this review did not independently validate the anatomical transforms of the
  actual NPS/SIIPS1 assets.
- Scanner-clean passbands and artifact covariates do not establish neural origin
  of high-frequency EEG effects. Actual residual spectra and physiological
  sensitivity analyses remain necessary.
- Condition-cell split-half reproducibility does not establish residual
  single-trial reliability. LSS effect recovery, HRF/smoothing sensitivity, and
  painful-trial sensitivity for SIIPS1 require data-based evaluation.
- No cohort-level analysis was rerun here. The primary pooled prediction endpoint
  still must not be interpreted as demonstrating prediction of fluctuations
  within a fixed stimulus condition.

## Rerun implications

- Refit affected exploratory deep models to obtain genuine best-epoch CPU
  predictions and exercise the corrected SIIPS1 input and temporal audit.
- Regenerate Study 1 reports to populate exploratory Holm-adjusted p-values.
- Rebuild reports with the stricter numeric and family checks. Invalid summaries
  require correction or regeneration at their source; incomplete inference
  families must be completed. Deep inputs with different retained time axes
  require harmonized preprocessing before refitting; this review adds no implicit
  resampling or interpolation.
- The initial HRF-support fix adds validation. The continued-review rasterization
  fix also changes frontal nuisance values when old rounding selected different
  frames. Regenerate target tables and refit their dependent primary, secondary,
  temporal and exploratory models before reporting updated results. Inputs that
  previously produced fabricated zero nuisance values require timing correction.
- Refit whole-brain construct-validity GLMs and regenerate their group maps for
  slice-time-corrected inputs. Refit deep models with corrected feature scaling.
- These edits do not change the primary ElasticNet/Ridge fitting implementation.

## Verification

Further-review validation:

- **577 passed, 7 warnings:** the full Study 1 selection under `studies/tests`.
- **23 passed:** root Study 1 target/report/benchmark tests and repository layout,
  hygiene, and architecture checks.
- All 20 new adversarial cases failed before correction: two mismatched EEG time
  axes, ten invalid primary summary values, five invalid or incomplete Holm
  families, and three unprespecified primary result cells. These now pass.
- Two additional positive regressions preserve valid common cropped time axes,
  wholly absent optional inference, and probability boundaries.
- The first focused run exposed an existing supported case with no optional
  absolute-R² p-values. That statistic remains optional when absent throughout
  its family; the full suite confirms this behavior is preserved.
- Ruff and `git diff --check` passed. No cohort analyses were rerun.

Continued-review validation:

- **555 passed, 7 warnings:** the full Study 1 selection under `studies/tests`.
- **23 passed:** root Study 1 target/report/benchmark tests and repository layout,
  hygiene, and architecture checks.
- **2 passed:** explicit zero- and half-TR synthetic GLM checks after adding the
  final slice-reference audit assertion.
- Ruff passed for the Python files changed in this pass; `git diff --check` passed.
- Synthetic regressions reproduced the slice-reference shift, unit-dependent
  scaling, proxy raster error, rounded identifiers, replaced epoch IDs, and clipped
  windows before their corresponding corrections. No cohort models were rerun.

Initial-review validation:

- **546 passed, 7 warnings:** all Study 1 tests under `studies/tests/pipelines`,
  `studies/tests/fmri`, and `studies/tests/config`.
- **41 passed:** shared staged-nesting, model-comparison, target-residualization,
  complete-permutation-fold and permutation-validity tests; root Study 1 target,
  reporting and benchmark tests; repository layout, hygiene and architecture checks.
- **25 passed:** final nuisance-helper run after the last HRF validation edit.
- Ruff passed for all changed Python files; `git diff --check` passed.
- The new checkpoint, SIIPS1 input, timing-audit, exploratory Holm, and missing-HRF
  regressions were observed failing before their fixes and passing afterward.
- Tests used the repository `.venv`, `MPLBACKEND=Agg`, and temporary MNE/Matplotlib
  settings directories. Initial attempts encountered a protected MNE settings
  lock and the macOS GUI plotting backend; neither required production-code changes.

## References checked

- [Nilearn: first-level slice reference and multi-run modeling](https://nilearn.github.io/stable/modules/generated/nilearn.glm.first_level.FirstLevelModel.html).
- [MNE: default EEG data units](https://mne.tools/stable/generated/mne.Epochs.html#mne.Epochs.get_data).
- [MNE: epoch-relative sample times and cropping](https://mne.tools/stable/generated/mne.Epochs.html#mne.Epochs.crop).
- [scikit-learn: training-based standardization and constant features](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html).
- [PyTorch: saving the best model requires a copy of its state](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html).
- [MNE: Hilbert envelopes are the absolute analytic signal](https://mne.tools/stable/generated/mne.Epochs.html#mne.Epochs.apply_hilbert).
- [Nilearn: HRF regressor construction](https://nilearn.github.io/stable/modules/generated/nilearn.glm.first_level.compute_regressor.html).
- [Nilearn: LSS/LSA beta-series example](https://nilearn.github.io/stable/auto_examples/07_advanced/plot_beta_series.html).
- [scikit-learn: leakage and fold-contained preprocessing](https://scikit-learn.org/stable/common_pitfalls.html).
- [statsmodels: Holm multiple-test correction](https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html).
