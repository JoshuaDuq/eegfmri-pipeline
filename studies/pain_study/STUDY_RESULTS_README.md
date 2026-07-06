# Pain Study Subject Results Log

This file tracks subject-level Study 1 and Study 2 result values and QC metrics.
It does not provide automatic interpretation of those values. Keep acquisition
issues and manual cohort decisions in `STUDY_ISSUES_README.md`.

The current values come from these Trillium runs:

- Study 1:
  `/scratch/joshduq/derivatives/group/multimodal/study1_plateau_allavailable_perm5000_20260611_confirmatory`
- Study 2:
  `/scratch/joshduq/derivatives/group/multimodal/study2_plateau_allavailable_perm1000_20260611_confirmatory`

## Glossary

Terms below match the definitions used by `study_subject_qc_summary.py`,
`study1_timing_audit.py`, and the Study 1/2 analysis code that writes the
underlying TSV/parquet files. Group-level inference columns (p-values,
confidence intervals, cluster statistics) live in the generated study reports,
not here.

### Study 1 targets and behavioral variables

All target-table values come from `targets/primary_targets.parquet`, which
contains only retained plateau-phase trials that met Study 1 target
construction and censoring.

| Term | Definition |
| --- | --- |
| Trials | Row count for the subject in `targets/primary_targets.parquet`. |
| Runs | Distinct `run` values in retained target rows. Each run is an 11-trial thermal sequence (`study1.targets.trials_per_run`). |
| Stimulus temperatures | Distinct `stimulus_temp` values in retained trials. |
| Selected surfaces | Distinct `selected_surface` values in retained trials. |
| Onset range | Earliest-to-latest `onset` value, in seconds from run start, across retained trials. |
| VAS / Mean VAS | Trial-wise `vas_final_coded_rating` (BIDS final coded rating) and its mean across retained trials. Non-pain trials use 0–99; pain trials use 100–200. |
| VAS-temperature *r* | Pearson correlation between `stimulus_temp` and `vas_final_coded_rating` across retained trials. |
| NPS | Trial-wise Neurologic Pain Signature expression: weighted sum of the trial LSS beta map with the fixed a priori NPS mask (`sum_v beta(v) * mask(v)`). Stored in column `NPS`. |
| Mean NPS | Mean of retained-trial `NPS` values for the subject. |
| NPS-temperature *r* | Pearson correlation between `stimulus_temp` and `NPS` across retained trials (`study_subject_qc_summary.py` uses pandas `.corr`, minimum two trials). |
| SIIPS1 | Trial-wise SIIPS1 expression computed the same way as NPS with the fixed SIIPS1 mask. Stored in column `SIIPS1`. |
| Mean SIIPS1 | Mean of retained-trial `SIIPS1` values for the subject. |
| SIIPS1-temperature *r* | Pearson correlation between `stimulus_temp` and `SIIPS1` across retained trials. |
| Mask voxels | `{signature}_fmri_n_voxels` in the target table: voxel count of the fixed scoring mask used for that signature. |
| Mask SHA256 | `{signature}_fmri_scoring_mask_sha256` in the target table. One unique hash per signature is required across retained rows. |

### Study 1 nuisance and motion summaries

These are simple means of trial-wise nuisance columns already stored in the
target table. They use the same HRF-weighted Study 1 Level 2 artifact
covariates that enter the prespecified nuisance regression.

| Term | Definition |
| --- | --- |
| Mean FD | Mean of `hrf_weighted_framewise_displacement` (mm) across retained trials. |
| Mean stdDVARS | Mean of `hrf_weighted_std_dvars` across retained trials. |
| Mean residual ECG coupling | Mean of `residual_ecg_coupling` across retained trials. |

### Study 1 prediction metrics

Values are read from per-fold `model_comparison.tsv` rows where
`test_subject` equals the held-out subject. Under leave-one-subject-out (LOSO)
cross-validation, each subject contributes one outer fold. Metrics are computed
by `model_comparison_cv_predictions()` using staged residual learning: the
nuisance-only model is fit on training subjects with the prespecified nuisance
design (continuous covariates `run`, `onset`, `within_run_trial`,
`hrf_weighted_framewise_displacement`, `hrf_weighted_std_dvars`,
`hrf_weighted_fp1_fp2_high_frequency_power`, `residual_ecg_coupling`, plus
categorical dummies for `stimulus_temp` and `selected_surface`); the EEG model
fits a Yeo-Johnson-transformed nuisance residual and predictions are
inverse-transformed with nuisance prediction added back before scoring on raw
NPS.

For each held-out fold, R² = 1 − SS_res/SS_tot, where SS_tot uses the mean of
the training subjects' raw NPS values in that fold as the zero-skill baseline.

| Term | Definition |
| --- | --- |
| Nuisance-only R² | `r2_nuisance` in `model_comparison.tsv`: out-of-sample R² of the nuisance-only model on raw NPS for the held-out subject. |
| EEG model R² | `r2` in `model_comparison.tsv`: out-of-sample R² of the nuisance-plus-EEG model on raw NPS after inverse transformation and nuisance add-back. |
| Delta R² | `delta_r2` = EEG model R² − nuisance-only R² for the held-out subject. |
| Nuisance MAE | `mae_nuisance`: mean absolute error of nuisance-only predictions on raw held-out NPS. |
| EEG model MAE | `mae`: mean absolute error of nuisance-plus-EEG predictions on raw held-out NPS. |
| Negative R² | R² < 0: predictions are worse than predicting the training-fold mean, even when delta R² is positive. |
| Primary check | NPS predicted from alpha+beta+gamma individual-channel log-power features with elastic net. Source: `feature_benchmark/primary/NPS/alpha_beta_gamma/model_comparison/model_comparison.tsv`, `model == elasticnet`. |
| Secondary check | NPS predicted from gamma-only features with elastic net and ridge. Source: `feature_benchmark/primary/NPS/gamma/model_comparison/model_comparison.tsv`. |
| Gamma elastic-net delta R² | `delta_r2` for `model == elasticnet` in the gamma benchmark table. |
| Gamma ridge delta R² | `delta_r2` for `model == ridge` in the gamma benchmark table. |

### Study 1 timing and temporal metrics

Timing values come from `study1_timing_audit.py` and are summarized in
`qc/timing_audit/study1_timing_audit_summary.tsv`. Alignment uses
`{run}|{trial_index}` keys shared by targets, clean EEG events,
fMRI BIDS plateau events, LSS plateau trials, and temporal-feature rows.

| Term | Definition |
| --- | --- |
| Timing audit | For each retained target row, verify a matching clean EEG event, fMRI BIDS stimulation plateau event, LSS plateau trial, and temporal-feature row; verify plateau onset is target onset + 3.0 s and plateau duration is 7.5 s within 20 ms. |
| fMRI plateau events | Count of fMRI BIDS rows with `trial_type == stimulation` and `stim_phase == plateau` for the subject (includes plateau rows not retained in Study 1 targets). |
| LSS plateau trials | Count of LSS rows with `events_stim_phase == plateau` for the subject. |
| Missing target-linked rows | Sum of unmatched targets, missing/invalid fMRI plateau matches, missing/invalid LSS plateau matches, and missing temporal-feature rows (`timing_alignment_missing_count()` in `study_subject_qc_summary.py`). |
| Max plateau-start delta | max(|fMRI plateau-start delta|, |LSS plateau-start delta|) × 1000 ms, where each delta is observed plateau onset minus (target onset + 3.0 s). |
| Temporal delta R² | Held-out NPS delta R² from the same staged nuisance-plus-EEG pipeline, but with EEG features extracted only from the named temporal window (`study1.temporal_negative_controls` in `study1_config.yaml`, transform `raw_log_power`, no baseline window). Source: `feature_benchmark/temporal_control/NPS/temporal_<window>/model_comparison/model_comparison.tsv`. |
| `prestimulus_wide` | Negative-control window [−5.0, −0.01] s relative to stimulus onset. |
| `immediate_prestimulus` | Negative-control window [−0.2, −0.01] s relative to stimulus onset. |
| `ramp_up` | Wrong-lag control window [0.0, 3.0] s relative to stimulus onset. |
| `early_plateau` | Plateau-sensitivity window [3.0, 5.5] s relative to stimulus onset. |
| `mid_plateau` | Plateau-sensitivity window [5.5, 8.0] s relative to stimulus onset. |
| `late_plateau` | Plateau-sensitivity window [8.0, 10.5] s relative to stimulus onset. |

### Study 2 source-stage variables

Values come from `source_stage/qc_{alpha,beta,gamma}.tsv`,
`source_stage/source_stage_input.tsv`, and `sub-*/eeg/source/source_power_*.npy`.
Per-subject QC fields are identical across bands for the primary combined-score source-stage run.

| Term | Definition |
| --- | --- |
| Source-stage criteria met | `source_stage_criteria_met` value from the source-stage QC table. |
| Retained trials | Row count after `_permutation_valid_source_runs()` filtering (Study 1 circular-shift QC criteria). |
| Valid runs | Distinct `run` values in that retained trial set. Requires ≥ 3 valid runs and ≥ 25 retained trials for the source-stage criteria. |
| Design rank | Rank of the source-stage nuisance design matrix (continuous nuisance covariates plus categorical dummies for `run`, `stimulus_temp`, `selected_surface`). Must be full rank. |
| Residual df | `retained_trials − design_rank`. Must be ≥ 15 for the source-stage criteria. |
| Condition number | Condition number of the contribution-stability design (nuisance design augmented with the band contribution target), evaluated on mean-centered, column-normalized columns. Must be ≤ 100 (`study2.source_stage.max_condition_number`). |
| Source-stage input rows | Row count for the subject in `source_stage/source_stage_input.tsv`. |
| Source-power rows | First dimension of `source_power_{alpha,beta,gamma}.npy` for the subject. |
| Source vertices | Second dimension of `source_power_{alpha,beta,gamma}.npy` (cortical source locations). |
| Anatomy files found | Whether `{subject}-trans.fif` and `{subject}-5120-5120-5120-bem-sol.fif` exist under the FreeSurfer subjects directory (`trans+BEM`, `trans_only`, `BEM_only`, or `missing`). |
| BEM surface issue | Manual issue entry when watershed BEM surfaces are anatomically invalid. |
| Rank deficient | Source-stage QC `unmet_criteria` includes `source_stage_design_rank` or `contribution_design_rank` (`source_stage_criteria_met == false`). |

## Run Provenance

| Field | Current value |
| --- | --- |
| Trillium repo path | `/project/def-mpcoll/joshduq/EEG_fMRI_Pipeline` |
| Pipeline commit on Trillium | `dc9d137` |
| Study 1 target table | `targets/primary_targets.parquet` |
| Study 1 report manifests | `reports/article_tables/article_table_manifest.json`; `reports/full_picture/full_picture_manifest.json` |
| Study 1 timing-alignment audit | `qc/timing_audit/study1_timing_audit_summary.tsv`; `qc/timing_audit/study1_timing_audit_trials.tsv` |
| Study 2 source-stage input | `source_stage/source_stage_input.tsv` |
| Study 2 source QC tables | `source_stage/qc_alpha.tsv`; `source_stage/qc_beta.tsv`; `source_stage/qc_gamma.tsv` |
| Generated subject QC package | `reports/subject_qc/subject_qc_summary.md`; `reports/subject_qc/subject_qc_summary.tsv`; `reports/subject_qc/subject_temporal_qc.tsv`; `reports/subject_qc/subject_timing_alignment_qc.tsv`; `reports/subject_qc/signature_mask_qc.tsv`; `reports/subject_qc/qc_completeness.tsv` |

## Study 1 Subject Metrics

These checks are per-subject descriptive QC values. They document retained
behavioral, nuisance, and signature-target data. They do not test a group-level
hypothesis.

| Subject | Trials | Blocks | Stimulus temperatures | Selected surfaces | Mean FD | Mean stdDVARS | Mean residual ECG coupling | Mean VAS | VAS-temperature r | Mean NPS | NPS-temperature r | Mean SIIPS1 | SIIPS1-temperature r |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `sub-0000` | 66 | 6 | 6 | 5 | 0.049 | 1.030 | 0.073 | 78.3 | 0.904 | 3.37 | 0.624 | 1276.4 | 0.396 |
| `sub-0001` | 65 | 6 | 6 | 5 | 0.068 | 1.006 | 0.112 | 93.6 | 0.848 | 1.60 | 0.449 | 436.6 | 0.583 |
| `sub-0003` | 55 | 5 | 6 | 5 | 0.063 | 1.037 | 0.140 | 73.0 | 0.926 | -0.15 | 0.462 | 1449.9 | 0.455 |
| `sub-0004` | 58 | 6 | 6 | 5 | 0.091 | 1.000 | 0.190 | 102.4 | 0.818 | 3.70 | 0.501 | -128.9 | 0.032 |
| `sub-0005` | 61 | 6 | 6 | 5 | 0.070 | 1.002 | 0.063 | 93.2 | 0.920 | 1.17 | 0.550 | 661.9 | 0.214 |

## Study 1 Retained Trial Checks

These values come from `targets/primary_targets.parquet`. They log retained
target rows by subject and run. Run-level reasons
and exclusions are tracked in `STUDY_ISSUES_README.md`.

| Subject | Retained trials | Runs | Trials by run | Onset range, s |
| --- | ---: | ---: | --- | --- |
| `sub-0000` | 66 | 6 | 1:11, 2:11, 3:11, 4:11, 5:11, 6:11 | 18.6-478.8 |
| `sub-0001` | 65 | 6 | 1:11, 2:11, 3:11, 4:11, 5:10, 6:11 | 20.2-464.1 |
| `sub-0003` | 55 | 5 | 1:11, 2:11, 4:11, 5:11, 6:11 | 19.1-444.7 |
| `sub-0004` | 58 | 6 | 1:11, 2:10, 3:9, 4:11, 5:7, 6:10 | 20.2-460.3 |
| `sub-0005` | 61 | 6 | 1:11, 2:11, 3:10, 4:8, 5:11, 6:10 | 19.1-466.5 |

## QC Scope

| QC category | Current scope | Source |
| --- | --- | --- |
| Study 1 retained trial counts | Recorded by subject and run | `targets/primary_targets.parquet` |
| Study 1 acquisition/run issues | Tracked as manual subject/run issue entries | `STUDY_ISSUES_README.md` |
| EEG-fMRI timing alignment | Checks clean EEG events, fMRI BIDS plateau events, LSS plateau trials, and temporal-feature rows by subject/run/trial key | `qc/timing_audit/*`; `reports/subject_qc/subject_timing_alignment_qc.tsv` |
| EEG-fMRI timing specificity | Evaluated through temporal control windows and wrong-lag controls | `subject_temporal_qc.tsv`; `feature_benchmark/temporal_control/*` |
| Study 2 source-stage QC | Recorded by alpha/beta/gamma source-stage QC tables | `source_stage/qc_*.tsv` |

## Study 1 Timing-Alignment Metrics

This audit checks whether retained Study 1 target rows align to clean EEG trigger
events, fMRI BIDS plateau-phase events, LSS plateau trials, and temporal EEG
feature rows using subject/run/trial identifiers.

The raw audit is written under `qc/timing_audit/`. The generated subject QC
package now also records the subject-level version at
`reports/subject_qc/subject_timing_alignment_qc.tsv` and includes it in
`reports/subject_qc/subject_qc_summary.md`.

| Subject | Target trials | fMRI plateau events | LSS plateau trials | Missing target-linked rows | Max plateau-start delta, ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| `sub-0000` | 66 | 66 | 66 | 0 | 3.245 |
| `sub-0001` | 65 | 66 | 66 | 0 | 7.218 |
| `sub-0003` | 55 | 55 | 55 | 0 | 2.300 |
| `sub-0004` | 58 | 66 | 63 | 0 | 2.528 |
| `sub-0005` | 61 | 66 | 66 | 0 | 1.930 |

## Study 1 Signature Target Metrics

Fixed scoring-mask checks:

| Signature | Mask voxels | Mask SHA256 count | Mask SHA256 |
| --- | ---: | ---: | --- |
| NPS | 69892 | 1 | `153d07463bec77415f40ba76e41f7e5ded559dfd11d9dad546bd453ce8685188` |
| SIIPS1 | 230939 | 1 | `b2fed1185fbf3beb284758d03f7152d06237604ec57050c993b3f0cde86471b1` |

## Study 1 Per-Subject Prediction Metrics

These are held-out subject diagnostics. They are not group inference.

Primary check: NPS predicted from alpha + beta + gamma EEG features with elastic net.

| Subject | Nuisance-only R2 | EEG model R2 | Delta R2 | Nuisance MAE | EEG model MAE |
| --- | ---: | ---: | ---: | ---: | ---: |
| `sub-0000` | -0.731 | -0.484 | 0.248 | 3.488 | 3.200 |
| `sub-0001` | -0.649 | -0.666 | -0.017 | 1.800 | 1.810 |
| `sub-0003` | -0.261 | -0.187 | 0.074 | 3.255 | 3.226 |
| `sub-0004` | -0.450 | -0.167 | 0.283 | 4.456 | 3.972 |
| `sub-0005` | 0.105 | 0.182 | 0.077 | 3.181 | 3.081 |

Secondary check: NPS prediction from gamma EEG features.

| Subject | Gamma elastic-net delta R2 | Gamma ridge delta R2 |
| --- | ---: | ---: |
| `sub-0000` | 0.278 | 0.412 |
| `sub-0001` | 0.207 | 0.089 |
| `sub-0003` | 0.053 | 0.135 |
| `sub-0004` | 0.371 | 0.486 |
| `sub-0005` | 0.117 | 0.122 |

Delta R2 greater than zero means the EEG model improved over the nuisance-only model for
that held-out subject. Negative R2 means the absolute prediction was worse than predicting
the held-out mean, even when delta R2 is positive.

## Study 1 Per-Subject Temporal Metrics

These values are held-out NPS delta R2 values by subject and temporal window.

Configured windows:

| Window | Time from stimulus onset | Type | Metric purpose |
| --- | ---: | --- | --- |
| `prestimulus_wide` | -5.0 to -0.01 s | Negative control | Should not carry pain-signature prediction if alignment and confound control are valid. |
| `immediate_prestimulus` | -0.2 to -0.01 s | Negative control | Checks leakage immediately before stimulation onset. |
| `ramp_up` | 0.0 to 3.0 s | Wrong-lag control | Checks whether the model is driven by ramp-up instead of plateau response. |
| `early_plateau` | 3.0 to 5.5 s | Plateau sensitivity | Early plateau response, after ramp-up and before mid-plateau. |
| `mid_plateau` | 5.5 to 8.0 s | Plateau sensitivity | Mid-plateau response, centered in the sustained heat interval. |
| `late_plateau` | 8.0 to 10.5 s | Plateau sensitivity | Late plateau response, ending before ramp-down. |

Temporal benchmark tables are written under:

`feature_benchmark/temporal_control/<signature>/temporal_<window>/model_comparison/model_comparison.tsv`

The values below are held-out NPS delta R2 values by subject. Delta R2 greater than zero
means the temporal-window EEG features improved over the nuisance-only model for that
held-out subject.

Elastic-net temporal delta R2:

| Subject | Pre-stimulus wide | Immediate pre-stimulus | Ramp-up | Early plateau | Mid plateau | Late plateau |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `sub-0000` | 0.296 | 0.138 | 0.202 | 0.894 | 0.509 | -0.017 |
| `sub-0001` | -0.425 | 0.081 | -0.151 | -0.317 | 0.187 | 0.014 |
| `sub-0003` | 0.193 | 0.367 | -0.363 | 0.626 | 0.371 | 0.337 |
| `sub-0004` | 0.253 | 0.465 | 0.167 | 0.127 | 0.294 | 0.187 |
| `sub-0005` | -0.404 | -0.127 | -1.195 | -1.411 | -0.600 | -0.191 |

Ridge temporal delta R2:

| Subject | Pre-stimulus wide | Immediate pre-stimulus | Ramp-up | Early plateau | Mid plateau | Late plateau |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `sub-0000` | 1.016 | 0.238 | 1.018 | 0.989 | 0.796 | 0.166 |
| `sub-0001` | -0.367 | -0.171 | 0.056 | -0.192 | 0.084 | 0.227 |
| `sub-0003` | -0.068 | 0.466 | -0.035 | 0.610 | 0.810 | 0.716 |
| `sub-0004` | 0.217 | 0.354 | 0.001 | -0.038 | 0.137 | -0.312 |
| `sub-0005` | -0.513 | -0.036 | -0.777 | -1.106 | -0.351 | -0.130 |

## Study 2 Source Metrics

These values come from source-stage QC and source-array outputs.

| Subject | Source-stage criteria met | Retained trials | Valid runs | Design rank | Residual df | Condition number |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `sub-0000` | -- | -- | -- | -- | -- | -- |
| `sub-0001` | false | 65 | 6 | 21 | 44 | inf |
| `sub-0003` | true | 55 | 5 | 20 | 35 | 151.8 |
| `sub-0004` | -- | -- | -- | -- | -- | -- |
| `sub-0005` | true | 53 | 5 | 20 | 33 | 112.1 |

Study 2 source-stage input and source-power file checks:

| Subject | Source-stage input rows | Input runs | Source-power rows | Source vertices | Anatomy files found |
| --- | ---: | ---: | ---: | ---: | --- |
| `sub-0001` | 65 | 6 | 65 | 8196 | trans + BEM |
| `sub-0003` | 55 | 5 | 55 | 8196 | trans + BEM |
| `sub-0005` | 53 | 5 | 61 | 8196 | trans + BEM |

## Update Checklist

When a new full run is completed:

1. Update the run paths at the top of this file.
2. Update the Trillium commit and provenance table.
3. Regenerate `reports/subject_qc/*` with `study_subject_qc_summary.py`.
4. Update the generated QC summary table.
5. Update the cohort-use table.
6. Update Study 1 retained trial counts by run.
7. Check that each signature still has one fixed scoring-mask hash.
8. Update per-subject prediction and temporal metric tables.
9. Add new Study 2 source-stage QC, source-input, and source-array values.
10. Keep acquisition notes, exclusions, and troubleshooting details in
   `STUDY_ISSUES_README.md`.
