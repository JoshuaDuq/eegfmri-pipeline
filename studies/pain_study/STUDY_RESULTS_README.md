# Pain Study Subject Validity Log

This file tracks subject-level validity and QC checks for Study 1 and Study 2. It should
not be used to report group-level results, model significance, or scientific conclusions.
Keep group inference and manuscript-style interpretation in the generated study reports.

The current values come from these Trillium runs:

- Study 1:
  `/scratch/joshduq/derivatives/group/multimodal/study1_plateau_allavailable_perm5000_20260611_confirmatory`
- Study 2:
  `/scratch/joshduq/derivatives/group/multimodal/study2_plateau_allavailable_perm1000_20260611_confirmatory`

## Run Provenance

| Field | Current value |
| --- | --- |
| Trillium repo path | `/project/def-mpcoll/joshduq/EEG_fMRI_Pipeline` |
| Pipeline commit on Trillium | `dc9d137` |
| Study 1 target table | `targets/primary_targets.parquet` |
| Study 1 report manifests | `reports/article_tables/article_table_manifest.json`; `reports/full_picture/full_picture_manifest.json` |
| Study 2 source-stage input | `source_stage/source_stage_input.tsv` |
| Study 2 source QC tables | `source_stage/qc_alpha.tsv`; `source_stage/qc_beta.tsv`; `source_stage/qc_gamma.tsv` |
| Generated subject QC package | `reports/subject_qc/subject_qc_summary.md`; `reports/subject_qc/subject_qc_summary.tsv`; `reports/subject_qc/subject_temporal_qc.tsv`; `reports/subject_qc/signature_mask_qc.tsv`; `reports/subject_qc/qc_completeness.tsv` |

## Cohort Status

| Subject | Study 1 status | Study 2 status | Validity note |
| --- | --- | --- | --- |
| `sub-0000` | Included | Excluded | Study 1 target checks passed. Study 2 excluded because watershed BEM surfaces are anatomically invalid. |
| `sub-0001` | Included | Excluded from source inference | Study 1 target checks passed. Study 2 source-stage contribution design is rank deficient. |
| `sub-0002` | Excluded | Excluded | Participant wanted out of the MRI; incomplete experiment. |
| `sub-0003` | Included | Included | Study 1 target checks passed. Study 2 source-stage QC passed. |
| `sub-0004` | Included | Excluded | Study 1 target checks passed. Study 2 excluded because watershed BEM surfaces are anatomically invalid. |
| `sub-0005` | Included | Included | Study 1 target checks passed. Study 2 source-stage QC passed. |

## Study 1 Subject Validity Checks

These checks are per-subject descriptive QC values. They document whether each participant
has usable behavioral, nuisance, and signature-target data. They do not test a group-level
hypothesis.

| Subject | Trials | Blocks | Stimulus temperatures | Mean FD | Mean stdDVARS | Mean residual ECG coupling | Mean VAS | VAS-temperature r | Mean NPS | NPS-temperature r | Mean SIIPS1 | SIIPS1-temperature r | Validity summary |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `sub-0000` | 66 | 6 | 6 | 0.049 | 1.030 | 0.073 | 78.3 | 0.904 | 3.37 | 0.624 | 1276.4 | 0.396 | Valid Study 1 subject. Strong behavioral temperature response and positive NPS-temperature coupling. |
| `sub-0001` | 65 | 6 | 6 | 0.068 | 1.006 | 0.112 | 93.6 | 0.848 | 1.60 | 0.449 | 436.6 | 0.583 | Valid Study 1 subject. Strong behavioral temperature response and positive signature-temperature coupling. |
| `sub-0003` | 55 | 5 | 6 | 0.063 | 1.037 | 0.140 | 73.0 | 0.926 | -0.15 | 0.462 | 1449.9 | 0.455 | Valid Study 1 subject. One fewer valid block, but behavioral and signature checks are usable. |
| `sub-0004` | 58 | 6 | 6 | 0.091 | 1.000 | 0.190 | 102.4 | 0.818 | 3.70 | 0.501 | -128.9 | 0.032 | Valid Study 1 subject. Behavioral and NPS checks are usable; SIIPS1-temperature coupling is weak. |
| `sub-0005` | 61 | 6 | 6 | 0.070 | 1.002 | 0.063 | 93.2 | 0.920 | 1.17 | 0.550 | 661.9 | 0.214 | Valid Study 1 subject. Strong behavioral temperature response and positive NPS-temperature coupling. |

## Study 1 Retained Trial Checks

These values come from `targets/primary_targets.parquet`. They log retained target rows by
subject, acquisition run, and block. They do not describe dropped-trial causes; acquisition
or exclusion reasons belong in `STUDY_ISSUES_README.md`.

| Subject | Retained trials | Acquisition runs | Blocks | Trials by acquisition run | Trials by block | Onset range, s |
| --- | ---: | ---: | ---: | --- | --- | --- |
| `sub-0000` | 66 | 6 | 6 | 1:11, 2:11, 3:11, 4:11, 5:11, 6:11 | 1:11, 2:11, 3:11, 4:11, 5:11, 6:11 | 18.6-478.8 |
| `sub-0001` | 65 | 6 | 6 | 1:11, 2:11, 3:11, 4:11, 5:10, 6:11 | 1:11, 2:11, 3:11, 4:11, 5:10, 6:11 | 20.2-464.1 |
| `sub-0003` | 55 | 5 | 5 | 1:11, 2:11, 4:11, 5:11, 6:11 | 1:11, 2:11, 4:11, 5:11, 6:11 | 19.1-444.7 |
| `sub-0004` | 58 | 6 | 6 | 1:11, 2:10, 3:9, 4:11, 5:7, 6:10 | 1:11, 2:10, 3:9, 4:11, 5:7, 6:10 | 20.2-460.3 |
| `sub-0005` | 61 | 6 | 6 | 1:11, 2:11, 3:10, 4:8, 5:11, 6:10 | 1:11, 2:11, 3:10, 4:8, 5:11, 6:10 | 19.1-466.5 |

Explicit EEG-fMRI timing error columns were not present in the current Study 1 target
table, and no separate timing/alignment QC file was found under the searched derivative
tree. If a future run writes alignment residuals, add them here by subject.

## Study 1 Signature Validity Notes

Use these notes only as subject-level target-validity context:

- NPS-temperature coupling is positive for every included Study 1 subject.
- SIIPS1-temperature coupling is weaker and should be checked carefully in each new subject.
- `sub-0004` has the weakest SIIPS1-temperature coupling among the current included subjects.
- Motion and DVARS summaries are retained here so later reruns can identify subjects with
  unusually high nuisance burden.

Fixed scoring-mask checks:

| Signature | Mask voxels | Mask SHA256 count | Mask SHA256 |
| --- | ---: | ---: | --- |
| NPS | 69892 | 1 | `153d07463bec77415f40ba76e41f7e5ded559dfd11d9dad546bd453ce8685188` |
| SIIPS1 | 230939 | 1 | `b2fed1185fbf3beb284758d03f7152d06237604ec57050c993b3f0cde86471b1` |

Each signature used one fixed scoring mask across retained Study 1 rows. Any future run
with more than one hash for the same signature should be treated as a provenance problem.

## Study 1 Per-Subject Prediction Checks

These are held-out subject diagnostics. They are useful for checking whether a participant
contributes usable prediction information, but they are not group inference.

Primary check: NPS predicted from alpha + beta + gamma EEG features with elastic net.

| Subject | Nuisance-only R2 | EEG model R2 | Delta R2 | Nuisance MAE | EEG model MAE | Validity note |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `sub-0000` | -0.731 | -0.484 | 0.248 | 3.488 | 3.200 | EEG improves over nuisance, but absolute prediction remains poor. |
| `sub-0001` | -0.649 | -0.666 | -0.017 | 1.800 | 1.810 | EEG does not improve the primary elastic-net prediction for this subject. |
| `sub-0003` | -0.261 | -0.187 | 0.074 | 3.255 | 3.226 | Small EEG improvement over nuisance. |
| `sub-0004` | -0.450 | -0.167 | 0.283 | 4.456 | 3.972 | Largest primary elastic-net improvement among current subjects. |
| `sub-0005` | 0.105 | 0.182 | 0.077 | 3.181 | 3.081 | Positive absolute prediction and positive EEG improvement. |

Secondary check: NPS prediction from gamma EEG features.

| Subject | Gamma elastic-net delta R2 | Gamma ridge delta R2 | Validity note |
| --- | ---: | ---: | --- |
| `sub-0000` | 0.278 | 0.412 | Gamma features improve over nuisance. |
| `sub-0001` | 0.207 | 0.089 | Gamma features improve over nuisance despite weak primary elastic-net result. |
| `sub-0003` | 0.053 | 0.135 | Small positive gamma contribution. |
| `sub-0004` | 0.371 | 0.486 | Strongest gamma contribution among current subjects. |
| `sub-0005` | 0.117 | 0.122 | Consistent positive gamma contribution. |

Delta R2 greater than zero means the EEG model improved over the nuisance-only model for
that held-out subject. Negative R2 means the absolute prediction was worse than predicting
the held-out mean, even when delta R2 is positive.

## Study 1 Per-Subject Temporal Validity Checks

These temporal checks ask whether subject-level prediction information is concentrated in
the physiologically intended stimulation interval instead of appearing equally in invalid
or wrong-lag windows. They are validity diagnostics, not group-level temporal inference.

Configured windows:

| Window | Time from stimulus onset | Type | Validity purpose |
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

| Subject | Pre-stimulus wide | Immediate pre-stimulus | Ramp-up | Early plateau | Mid plateau | Late plateau | Temporal validity note |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `sub-0000` | 0.296 | 0.138 | 0.202 | 0.894 | 0.509 | -0.017 | Strong early/mid plateau signal, but negative controls are also positive. |
| `sub-0001` | -0.425 | 0.081 | -0.151 | -0.317 | 0.187 | 0.014 | Weak temporal pattern; mid plateau is the clearest positive window. |
| `sub-0003` | 0.193 | 0.367 | -0.363 | 0.626 | 0.371 | 0.337 | Plateau windows are positive, but pre-stimulus controls are also positive. |
| `sub-0004` | 0.253 | 0.465 | 0.167 | 0.127 | 0.294 | 0.187 | Timing specificity is weak because immediate pre-stimulus is high. |
| `sub-0005` | -0.404 | -0.127 | -1.195 | -1.411 | -0.600 | -0.191 | No positive temporal-window evidence in this model. |

Ridge temporal delta R2:

| Subject | Pre-stimulus wide | Immediate pre-stimulus | Ramp-up | Early plateau | Mid plateau | Late plateau | Temporal validity note |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `sub-0000` | 1.016 | 0.238 | 1.018 | 0.989 | 0.796 | 0.166 | Large positive values include pre-stimulus and ramp-up controls; specificity is poor. |
| `sub-0001` | -0.367 | -0.171 | 0.056 | -0.192 | 0.084 | 0.227 | Weak positive ramp-up/mid/late values; no clear plateau-specific pattern. |
| `sub-0003` | -0.068 | 0.466 | -0.035 | 0.610 | 0.810 | 0.716 | Strong plateau values, but immediate pre-stimulus is also positive. |
| `sub-0004` | 0.217 | 0.354 | 0.001 | -0.038 | 0.137 | -0.312 | Weak plateau evidence and positive pre-stimulus controls. |
| `sub-0005` | -0.513 | -0.036 | -0.777 | -1.106 | -0.351 | -0.130 | No positive temporal-window evidence in this model. |

Subject-level temporal validity is strongest when pre-stimulus and ramp-up delta R2 values
are near zero or negative, while one or more plateau windows are positive. A strong ramp-up
or pre-stimulus value should be treated as a warning about timing specificity, leakage, or
non-pain confounding for that subject.

## Study 2 Source Validity Checks

These checks document whether each subject has usable source-stage data. They do not report
source-family significance or group-level source inference.

| Subject | Eligible for source inference | Retained trials | Valid blocks | Design rank | Residual df | Condition number | Validity summary |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `sub-0000` | No | Not run | Not run | Not run | Not run | Not run | Excluded before source inference because watershed BEM surfaces are anatomically invalid. |
| `sub-0001` | No | 65 | 6 | 21 | 44 | inf | Excluded from source inference because the source-stage contribution design is rank deficient. |
| `sub-0003` | Yes | 55 | 5 | 20 | 35 | 151.8 | Valid source-stage subject. |
| `sub-0004` | No | Not run | Not run | Not run | Not run | Not run | Excluded before source inference because watershed BEM surfaces are anatomically invalid. |
| `sub-0005` | Yes | 53 | 5 | 20 | 33 | 112.1 | Valid source-stage subject. |

Study 2 source-stage input and source-power file checks:

| Subject | Source-stage input rows | Input blocks | Input runs | Source-power rows | Source vertices | Anatomy files found | Validity note |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `sub-0001` | 65 | 6 | 6 | 65 | 8196 | trans + BEM | Source arrays exist, but source-stage design is rank deficient. |
| `sub-0003` | 55 | 5 | 5 | 55 | 8196 | trans + BEM | Source input and source arrays match. |
| `sub-0005` | 53 | 5 | 5 | 61 | 8196 | trans + BEM | Source arrays include more rows than retained source-stage input; QC uses the retained 53-row input. |

The source anatomy files were found under `/scratch/joshduq/study2_freesurfer_subjects`
for `sub-0001`, `sub-0003`, and `sub-0005`. `sub-0000` and `sub-0004` are not listed in
the source-stage input because their watershed BEM surfaces were invalid.

## Update Checklist

When a new full run is completed:

1. Update the run paths at the top of this file.
2. Update the Trillium commit and provenance table.
3. Update the cohort status table.
4. Add new included Study 1 subjects to the subject-validity table.
5. Update retained trial counts by run and block.
6. Check that each signature still has one fixed scoring-mask hash.
7. Update per-subject prediction and temporal-validity checks.
8. Add new Study 2 source-stage QC, source-input, and source-array values.
9. Keep acquisition notes, exclusions, and troubleshooting details in
   `STUDY_ISSUES_README.md`.
10. Do not add group-level model results, p values, corrected q values, clusters, or broad
   scientific interpretation to this file.
