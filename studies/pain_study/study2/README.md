# Study 2: Methods for Trial-Wise Cortical EEG-BOLD Coupling

This document describes Study 2 as an analysis methods section rather than as a generic runbook. The description is implementation-grounded: every method summarized here reflects the current behavior of the Study 2 coupling stack in `studies/pain_study/study2/`.

Study 2 quantifies trial-wise coupling between source-localized EEG power and subject-surface BOLD responses within a frozen set of cortical regions of interest (ROIs). The workflow is executed through the shared CLI command:

```bash
python -m eeg_pipeline.cli.main coupling compute ...
```

The production Study 2 configuration is:

```text
studies/pain_study/study2/config/eeg_bold_coupling_study2.yaml
```

## 1. Analytical Aim

The primary analysis asks whether within-trial fluctuations in cortical EEG power predict trial-wise BOLD responses measured in homologous or functionally matched cortical ROIs. The implemented workflow is trial-based, subject-specific, and multimodal:

- EEG is source-localized on each subject's cortical source space.
- fMRI responses are estimated with one least-squares-separate (LSS) model per selected trial.
- Trial-wise EEG and BOLD values are aligned by run and trial number.
- Confirmatory within-ROI models are summarized at the subject level and then aggregated with a mixed-effects model at the group level.

The main pipeline does not depend on generic precomputed EEG feature extraction, generic behavioral summaries, or precomputed first-level fMRI contrast maps. It computes its own trial-wise EEG and fMRI summaries inside `coupling compute`.

## 2. Data Requirements

### 2.1 Raw Inputs

If the study is started from raw data, the required sources are:

- BrainVision EEG source files
- raw fMRI DICOM data
- PsychoPy trial summary CSV files

These raw files are converted to BIDS through the study-specific scripts described in:

```text
studies/pain_study/scripts/README.md
```

The raw-conversion order is:

1. `eeg-raw-to-bids`
2. `fmri-raw-to-bids`
3. `merge-psychopy`

Those scripts are study-specific preparatory utilities. They are not part of the core `coupling` command.

### 2.2 Required Derived Inputs

The coupling workflow requires the following derived assets before it can run:

| Asset | Requirement |
| --- | --- |
| EEG BIDS root | `paths.bids_root` must resolve and exist |
| fMRI BIDS root | `paths.bids_fmri_root` must resolve and exist |
| derivatives root | `paths.deriv_root` must resolve and exist |
| clean EEG epochs | subject/task clean epochs must be loadable with `load_epochs_for_analysis(..., align="strict")` |
| clean EEG events TSV | must exist beside the clean epochs file |
| fMRIPrep outputs | subject-space (`T1w`) preprocessed BOLD and confounds must be available |
| FreeSurfer subject anatomy | required for subject surfaces and ROI morphing |
| EEG↔MRI coregistration transform | `sub-<id>-trans.fif` |
| BEM solution | at least one `sub-<id>-*-bem-sol.fif` |

### 2.3 Clean EEG Event Contract

The clean event table is a hard dependency. It must contain:

- `trial_id`
- `onset`
- `duration`
- a usable run column (`block`, `run_id`, `run`, or `session`)
- a usable trial-number column (`trial_number`, `events_trial_number`, or `events_trial_index`)

For the production Study 2 configuration, the clean event table must also contain the two upstream artifact-QC columns used by the EEG nuisance model:

- `residual_ecg_coupling`
- `peripheral_low_gamma_power`

These columns are already produced by the repository's default preprocessing configuration:

- `preprocessing.clean_events_qc.ecg_coupling.output_column = "residual_ecg_coupling"`
- `preprocessing.clean_events_qc.peripheral_low_gamma.output_column = "peripheral_low_gamma_power"`

### 2.4 Subject Surface and Forward-Model Contract

The workflow assumes a valid FreeSurfer subject directory and fails fast if the following subject files are missing:

```text
<SUBJECTS_DIR>/sub-<ID>/surf/lh.pial
<SUBJECTS_DIR>/sub-<ID>/surf/rh.pial
<SUBJECTS_DIR>/sub-<ID>/surf/lh.white
<SUBJECTS_DIR>/sub-<ID>/surf/rh.white
<SUBJECTS_DIR>/sub-<ID>/bem/sub-<ID>-trans.fif
<SUBJECTS_DIR>/sub-<ID>/bem/sub-<ID>-*-bem-sol.fif
```

Study 2 validates those files but does not generate them internally. BEM and coregistration preparation are documented in:

```text
docs/eeg/source-localization.md
fmri_pipeline/README.md
```

## 3. Fixed Study 2 Specification

Study 2 is a fixed specialization of the general pain-study coupling framework. The frozen scientific contract is defined by:

```text
studies/pain_study/study2/config/eeg_bold_coupling_study2.yaml
```

### 3.1 Frozen ROIs

Study 2 does not build ROIs dynamically at runtime. The ROI builder is disabled, and the analysis uses frozen label files defined in:

```text
studies/pain_study/study2/config/roi_library/study2/fsaverage/study2_roi_manifest.json
```

The two production ROIs are:

| ROI | Hemisphere | Runtime source |
| --- | --- | --- |
| `right_operculo_periinsular` | right | `rh.study2_right_operculo_parietal.label` |
| `midcingulate_pre_sma` | bilateral | `lh.study2_cingulo_pre_sma.label`, `rh.study2_cingulo_pre_sma.label` |

At runtime, each ROI is:

1. loaded on `fsaverage`,
2. combined if multiple label files belong to the same ROI,
3. morphed to the target subject,
4. intersected with that subject's source space,
5. converted to an area-weighted source-row representation.

### 3.2 EEG Specification

The production EEG settings are:

| Parameter | Value |
| --- | --- |
| bands | `["alpha", "beta"]` |
| active window | `[3.0, 10.5]` s |
| baseline window | `[-5.0, -0.01]` s |
| source method | `lcmv` |
| spacing | `oct6` |
| `mindist_mm` | `5.0` |
| `reg` | `0.05` |
| `snr` | `3.0` |
| `loose` | `0.2` |
| `depth` | `0.8` |
| feature batch size | `256` |

For each trial, band-limited source power is summarized within each ROI and expressed as:

```text
10 * log10(active_power / baseline_power)
```

The confirmatory EEG predictor columns are therefore:

- `eeg_right_operculo_periinsular_alpha`
- `eeg_right_operculo_periinsular_beta`
- `eeg_midcingulate_pre_sma_alpha`
- `eeg_midcingulate_pre_sma_beta`

### 3.3 fMRI Specification

The production fMRI settings are:

| Parameter | Value |
| --- | --- |
| contrast name | `plateau_trials` |
| input source | `fmriprep` |
| fMRIPrep space | `T1w` |
| require fMRIPrep | `true` |
| extraction method | `surface_glm` |
| selection column | `stim_phase` |
| selection values | `["plateau"]` |
| HRF model | `spm` |
| drift model | `cosine` |
| high-pass | `0.008` Hz |
| smoothing FWHM | `0.0` |
| confounds strategy | `auto` |
| LSS other regressors | `all` |

This configuration restricts the main analysis to plateau trials and requires subject-space fMRIPrep outputs. The primary workflow performs one LSS model per selected trial, then samples the resulting effect and variance estimates to the subject cortical surface for ROI summarization.

### 3.4 Alignment Specification

Study 2 enforces:

```text
eeg_bold_coupling.alignment.key_mode = "run_trial_number"
```

Thus the canonical multimodal trial key is:

```text
<run_num>|trial-<trial_number>
```

No alternative primary alignment mode is implemented for this study.

### 3.5 Covariate and Nuisance Specification

The production confirmatory model terms are:

```text
temperature
temperature_sq
fd
eeg_artifact
exp_site
exp_global
block_start
```

The production nuisance configuration enables:

| Metric | Status | Threshold |
| --- | --- | --- |
| FD | enabled | censor `> 0.5` |
| DVARS | enabled | censor `> 2.5` |
| EEG artifact component z | enabled | censor `> 5.0` |
| EEG artifact composite | enabled | censor `> 3.0` |
| require finite model terms | enabled | yes |

The EEG artifact composite is built from:

- `event_residual_ecg_coupling`
- `event_peripheral_low_gamma_power`

Both are robust-z-scored within run and combined as a positive-burden metric.

### 3.6 Statistics Specification

The group model uses the `nlme::lme` AR(1) backend through the Study 2 R wrapper. The production parameters are:

| Parameter | Value |
| --- | --- |
| backend | `nlme_lme_ar1` |
| fit method | `reml` |
| minimum trials per subject per cell | `20` |
| minimum runs per subject per cell | `2` |
| include run fixed effect | `true` |
| use outcome variance | `true` |
| alpha | `0.05` |
| R executable | `Rscript` |
| max iterations | `200` |
| EM iterations | `50` |
| singular tolerance | `1e-8` |

Only interpretable confirmatory group results participate in Holm correction.

## 4. Upstream Processing Before Coupling

### 4.1 Paradigm-Specific BIDS Preparation

Raw EEG and fMRI data are first converted into BIDS format, and PsychoPy trial-level metadata are merged into the EEG events files. The study ships recommended paradigm-specific override templates in:

```text
studies/pain_study/scripts/config/thermal_pain_eeg_overrides.yaml
studies/pain_study/scripts/config/thermal_pain_fmri_overrides.yaml
```

These templates define the intended task-level assumptions for the thermal pain paradigm, including:

- `project.task: thermalactive`
- event-column aliases for temperature, ratings, and binary pain labels
- `alignment.fmri_onset_reference: first_iti_start`
- thermal-pain fMRI event-scoping defaults

### 4.2 EEG Preprocessing

Before the coupling stage, EEG preprocessing must produce:

- clean subject/task epochs,
- an aligned clean events TSV,
- the required event-level artifact-QC columns.

The coupling workflow then loads those outputs with strict alignment enabled.

### 4.3 fMRIPrep, FreeSurfer, and Forward-Model Preparation

The coupling stage requires subject-space fMRIPrep outputs, FreeSurfer cortical surfaces, a valid EEG↔MRI transform, and a valid BEM solution. In other words, the fMRI and source-model preparation steps are methodological prerequisites, not optional conveniences.

## 5. Subject-Level Coupling Workflow

The implemented subject-level workflow proceeds through a fixed sequence of stages.

### 5.1 Preflight Validation

The workflow begins with a preflight step that fails immediately if required roots, subject surfaces, BEM files, ROI labels, or optional signature maps are missing. This step is intended to surface structural problems before any trial-level modeling is attempted.

### 5.2 EEG Loading and Source Modeling

Clean EEG epochs and aligned clean events are loaded with strict alignment. The subject-specific forward model is then resolved from the FreeSurfer subject, BEM solution, and transform. The frozen Study 2 ROIs are morphed from `fsaverage` to the subject and mapped onto the subject source space.

### 5.3 Trial-Wise EEG Feature Extraction

For each trial, band-limited source estimates are summarized inside each ROI for the active and baseline windows. ROI values are area-weighted across the source rows represented by the morphed ROI. The resulting trial-wise EEG predictor table contains:

- `trial_id`
- `run_num`
- `onset`
- `duration`
- `trial_number`
- `trial_key`
- `events_*` metadata copied from clean events
- one EEG predictor column per ROI/band

If EEG artifact modeling is enabled, the trial table is further extended with raw, z-scored, and composite artifact metrics.

### 5.4 Trial-Wise LSS fMRI Modeling

Study 2 invokes the shared trial-signature extraction machinery in LSS mode with trial beta and trial variance writing enabled. The analysis operates on trials selected by:

```text
stim_phase == "plateau"
```

The LSS output directory is written under the subject coupling directory as `lss/`. Key outputs include:

- `trials.tsv`
- `provenance.json`
- `trial_betas/run-XX/..._trial-YYY_beta.nii.gz`
- `trial_betas/run-XX/..._trial-YYY_var.nii.gz`

### 5.5 Full-Session Trial History Recovery

The LSS `trials.tsv` file must contain `events_file` and `events_row`, because Study 2 reconstructs the full selected-trial history from the original source events. This is how the workflow derives trial-history covariates rather than relying only on the already selected LSS subset.

The derived covariates include:

- `temperature`
- `temperature_sq`
- `delta_temperature`
- `exp_site`
- `exp_global`
- `block_start`
- `trial_position`

### 5.6 Surface-Based Trial-Wise BOLD Extraction

For each run, the subject-space BOLD image is sampled to the subject cortical surfaces. A subject-surface LSS GLM is fit for each trial, and the target beta and variance images are summarized within each ROI as area-weighted cortical means. The resulting BOLD table contains:

- `trial_key`
- `bold_<roi>`
- `boldvar_<roi>`

If local signature-expression analysis is enabled, additional `bold_localexpr_*` columns are also created. In the production Study 2 configuration, local signature expression is disabled.

### 5.7 fMRI Quality Control

The workflow computes two fMRI-QC families:

1. design QC, based on each trial-level LSS design matrix
2. ROI QC, based on BOLD outlier burden within each ROI

The production thresholds are:

- maximum absolute target-to-other-regressor correlation: `0.5`
- minimum target efficiency: `0.01`
- maximum target variance: `100.0`
- maximum ROI outlier proportion: `0.1`
- ROI outlier MAD threshold: `5.0`

If any design-level criterion fails, the subject receives a global fMRI QC failure. If only the ROI outlier criterion fails, the failure is ROI-specific rather than global.

### 5.8 Multimodal Merge and Trial Censoring

The EEG table, augmented LSS trial table, and ROI BOLD table are merged on `trial_key`. The merged table must contain every configured model term; otherwise the workflow stops with an error.

The censoring step writes explicit exclusion columns:

- `exclude_fd_threshold`
- `exclude_dvars_threshold`
- `exclude_eeg_artifact_component`
- `exclude_eeg_artifact_composite`
- `exclude_non_finite_model_term`
- `keep_trial`
- `exclude_reason`

The kept analysis table consists only of rows for which all exclusion criteria are false.

### 5.9 Subject-Level Confirmatory Cells

The confirmatory analysis family consists of one within-ROI EEG→BOLD cell per ROI and frequency band. Under the production configuration this yields four confirmatory cells:

- `confirmatory__right_operculo_periinsular__alpha`
- `confirmatory__right_operculo_periinsular__beta`
- `confirmatory__midcingulate_pre_sma__alpha`
- `confirmatory__midcingulate_pre_sma__beta`

At the subject level, each cell is evaluated for eligibility and assigned a status such as:

- `ok`
- `insufficient_trials`
- `insufficient_runs`
- `insufficient_variance`
- `fmri_qc_failed`

These subject-level status labels determine which subjects are allowed to contribute to each group-level confirmatory fit.

## 6. Group-Level Statistical Analysis

When at least two subjects are processed, the pipeline automatically performs group aggregation.

For each unique confirmatory cell:

1. only subjects with subject-level status `ok` for that cell are retained,
2. all kept trials from those subjects are pooled,
3. predictor, outcome, and continuous non-binary covariates are standardized within subject,
4. the AR(1) mixed-effects model is fit using the configured R backend,
5. Holm correction is applied across interpretable confirmatory p-values.

`trial_position` is required by the model because it defines within-run temporal spacing for the AR(1) structure.

## 7. Sensitivity Analyses

The production Study 2 configuration enables five sensitivity branches.

### 7.1 Residualized Correlation

`residualized_correlation` computes per-subject residualized EEG/BOLD effects and aggregates them at group level using:

- `5000` bootstrap iterations
- `5000` permutation iterations

### 7.2 Primary Permutation

`primary_permutation` computes per-subject confirmatory-family permutation effects and aggregates them at group level using:

- `5000` bootstrap iterations
- `5000` permutation iterations

### 7.3 Source-Method Sensitivity

The enabled source-method item is:

- `eloreta`

This branch recomputes the EEG predictors with `method="eloreta"` while retaining the same production ROIs and the same alpha/beta band set.

### 7.4 Anatomical Specificity

The enabled anatomical-specificity item is:

- `control_rois`

This branch replaces the confirmatory ROIs with two control ROIs:

- `left_operculo_periinsular_control`
- `posterior_midcingulate_control`

EEG and BOLD values are recomputed for those control ROIs before confirmatory-style models are rerun.

### 7.5 Artifact-Model Sensitivity

The enabled artifact-model item is:

- `expanded_artifact`

This branch augments the production EEG artifact model by adding `global_amplitude` as a third component and then reruns the censoring and confirmatory-style analyses.

### 7.6 Within-Between Decomposition

`within_between` decomposes each confirmatory EEG predictor into:

- `<predictor>_within_subject`
- `<predictor>_subject_mean`

The within-subject term is used as the main predictor, and the subject-mean term is added as a nonstandardized covariate.

### 7.7 Implemented but Disabled Branches

The following branches exist in code but are disabled in the frozen production configuration:

- `painful_only`
- `alternative_fmri`
- `delta_temperature`
- `temperature_categorical`

## 8. Negative Control

The production negative control is:

- `trial_shuffle`

This branch shuffles the BOLD table within run, merges the shuffled BOLD values back onto the real EEG predictors, reapplies censoring, and reruns confirmatory-style models with the family label `negative_control_trial_shuffle`.

At group level, the adjudication logic treats the negative control as passing when it is either non-interpretable or not significant at the configured alpha level.

## 9. Robustness Adjudication

The group pipeline writes leave-one-out refits for confirmatory cells:

- leave-one-subject-out
- leave-one-run-out

These are summarized in:

- `leave_one_out_refits.tsv`
- `leave_one_out_summary.tsv`

The final robustness table is:

```text
robustness/adjudication_summary.tsv
```

For each confirmatory cell, that table records whether the result passes:

- Holm-significant confirmatory inference
- all enabled source-method sign checks
- all enabled artifact-model sign checks
- the within-between sign check
- leave-one-subject sign stability
- leave-one-run sign stability
- the negative-control criterion

The final boolean summary is `overall_robust_pass`.

## 10. Outputs and Reproducibility

### 10.1 Subject Output Root

If `eeg_bold_coupling.output_dir` is null, Study 2 writes subject outputs to:

```text
<paths.deriv_root>/sub-<ID>/multimodal/eeg_bold_coupling/task-<task>/contrast-plateau_trials/
```

### 10.2 Group Output Root

If `eeg_bold_coupling.output_dir` is null, group outputs are written to:

```text
<paths.deriv_root>/group/multimodal/eeg_bold_coupling/task-<task>/contrast-plateau_trials/
```

### 10.3 Core Subject Outputs

The main subject directory contains:

- `runtime_profile.json`
- `coupling_config.json`
- `qc_summary.json`
- `fmri_design_qc.tsv`
- `fmri_roi_qc.tsv`
- `fmri_qc_summary.json`
- `roi_manifest.tsv`
- `analysis_cells.tsv`
- `trialwise_eeg.tsv`
- `trialwise_trials.tsv`
- `trialwise_bold.tsv`
- `trialwise_merged.tsv`
- `trialwise_qc.tsv`
- `lss/`
- `sensitivities/`
- `negative_controls/`
- `secondary/` when enabled

`trialwise_merged.tsv` and `trialwise_qc.tsv` both contain the post-censoring QC table; the second filename is a more explicit duplicate of the first.

### 10.4 Core Group Outputs

The main group directory contains:

- `analysis_cells_all.tsv`
- `group_results.tsv`
- `sensitivities/`
- `negative_controls/`
- `secondary/`
- `robustness/`

### 10.5 Batch and Run Metadata

Independent of the subject/group analysis trees, the pipeline also writes:

```text
<paths.deriv_root>/logs/eeg_bold_coupling_batch_ledger.tsv
<paths.deriv_root>/logs/run_metadata/eeg_bold_coupling/run_<timestamp>_<id>.json
```

These files capture batch success/failure state and run-level reproducibility metadata.

## 11. Execution

### 11.1 Production Run

The frozen Study 2 configuration is not the CLI default. A production Study 2 run should therefore pass the Study 2 YAML explicitly.

Example:

```bash
python -m eeg_pipeline.cli.main coupling compute \
  --subject 0001 \
  --subject 0002 \
  --task thermalactive \
  --bids-root /path/to/bids_output/eeg \
  --bids-fmri-root /path/to/bids_output/fmri \
  --deriv-root /path/to/derivatives \
  --source-subjects-dir /path/to/freesurfer_subjects_dir \
  --coupling-config studies/pain_study/study2/config/eeg_bold_coupling_study2.yaml
```

The console-script equivalent is:

```bash
eeg-pipeline coupling compute \
  --subject 0001 \
  --subject 0002 \
  --task thermalactive \
  --bids-root /path/to/bids_output/eeg \
  --bids-fmri-root /path/to/bids_output/fmri \
  --deriv-root /path/to/derivatives \
  --source-subjects-dir /path/to/freesurfer_subjects_dir \
  --coupling-config studies/pain_study/study2/config/eeg_bold_coupling_study2.yaml
```

### 11.2 Smoke Configuration

The repository also ships a smoke configuration:

```text
studies/pain_study/study2/config/eeg_bold_coupling_smoketest.yaml
```

This file is useful for fast validation, but it is not the same scientific specification as the frozen Study 2 production config.

### 11.3 Optional Signature Maps

The production Study 2 YAML intentionally sets:

```yaml
paths:
  signature_maps: []
```

Therefore:

- the main Study 2 analysis does not require signature maps,
- local signature-expression outputs are disabled by default,
- if local signature expression is enabled later, valid `paths.signature_maps` must also be supplied.

## 12. Failure Conditions That Commonly Stop the Workflow

Common hard failures include:

- missing `paths.bids_root`
- missing `paths.bids_fmri_root`
- missing clean epochs or clean events
- missing usable run or trial-number columns in EEG events
- missing `events_file` or `events_row` in LSS trials
- no selected source events matching `stim_phase == "plateau"`
- missing subject surfaces, transform, or BEM solution
- missing fMRIPrep confounds when confound selection is enabled
- missing configured model terms after multimodal merge
- unavailable `Rscript` executable for group modeling

All of these conditions surface as explicit errors rather than hidden fallbacks.
