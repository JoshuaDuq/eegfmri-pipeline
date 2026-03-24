# Study 2: Manuscript-Style Methods for Trial-Wise Cortical EEG-BOLD Coupling

This document is the publication-style Methods companion for Study 2. It is
implementation-grounded: every methodological statement below reflects the current
behavior of the Study 2 coupling workflow in `studies/pain_study/study2/`.

The frozen production specification is:

```text
studies/pain_study/study2/config/eeg_bold_coupling_study2.yaml
```

The repository's pain-paradigm examples and override templates use the BIDS task label
`thermalactive`.

Study 2 asks whether trial-wise fluctuations in source-localized cortical EEG power
predict trial-wise cortical BOLD responses estimated from subject-surface fMRI in a
small set of a priori cortical regions of interest (ROIs). EEG and fMRI are summarized
independently at the single-trial level, aligned by run and trial number, filtered by
explicit nuisance and quality-control rules, and entered into confirmatory mixed-effects
models with prespecified sensitivity analyses and negative controls.

## 1. Confirmatory Objective and Inferential Family

The primary Study 2 objective is to test whether trial-wise cortical EEG power covaries
with trial-wise cortical BOLD response within anatomically matched ROIs during the
thermal pain task. The implemented analysis is not a whole-brain screen and not a
generic multimodal discovery workflow. It is a fixed confirmatory ROI analysis
restricted to:

- plateau-phase trials,
- source-localized cortical EEG,
- subject-surface BOLD extraction in `T1w` space,
- a frozen ROI library,
- a fixed confirmatory family of four EEG-to-BOLD cells.

The confirmatory family is the Cartesian product of two ROIs and two EEG bands:

| Analysis ID | EEG predictor | BOLD outcome |
| --- | --- | --- |
| `confirmatory__right_operculo_periinsular__alpha` | `eeg_right_operculo_periinsular_alpha` | `bold_right_operculo_periinsular` |
| `confirmatory__right_operculo_periinsular__beta` | `eeg_right_operculo_periinsular_beta` | `bold_right_operculo_periinsular` |
| `confirmatory__midcingulate_pre_sma__alpha` | `eeg_midcingulate_pre_sma_alpha` | `bold_midcingulate_pre_sma` |
| `confirmatory__midcingulate_pre_sma__beta` | `eeg_midcingulate_pre_sma_beta` | `bold_midcingulate_pre_sma` |

The null hypothesis for each confirmatory cell is that the trial-wise EEG fixed effect
equals zero after adjustment for prespecified nuisance terms. Multiplicity correction is
applied only across interpretable confirmatory tests.

Throughout this document:

- `s` indexes subject,
- `r` indexes run,
- `i` indexes trial,
- `q` indexes ROI,
- `b` indexes frequency band,
- `j` indexes source-space rows,
- `v` indexes cortical surface vertices.

## 2. Data Requirements and Upstream Preconditions

### 2.1 Raw Study Material

If Study 2 is started from raw data, the upstream sources are:

- BrainVision EEG source files,
- raw fMRI DICOM data,
- PsychoPy trial-summary CSV files.

These are converted to BIDS with the study-specific utilities described in:

```text
studies/pain_study/scripts/README.md
```

The expected preparation order is:

1. `eeg-raw-to-bids`
2. `fmri-raw-to-bids`
3. `merge-psychopy`

Study 2 itself does not perform those conversions.

When the repository's pain-paradigm conversion utilities are used, the upstream
repository contract is:

- EEG raw-to-BIDS expects BrainVision input, defaults to the `easycap-M1` montage,
  defaults to `60.0` Hz line frequency, filters annotations on the thermode trigger
  prefix `Trig_therm/T  1`, and can trim the EEG recording to the first fMRI volume
  trigger for multimodal alignment.
- fMRI raw-to-BIDS expects DICOM input, writes BIDS events from PsychoPy timing,
  defaults to phase-level event granularity, and uses `first_iti_start` as the default
  onset reference in the paradigm CLI.
- PsychoPy merging expects `TrialSummary.csv`, matches behavioral rows to trigger events
  using the same thermode prefix, and by default fails rather than silently trimming row
  mismatches.

These repository defaults define the supported upstream data contract. They are not a
substitute for reporting the actual acquisition and conversion settings used for the
cohort in the manuscript.

### 2.2 Required Derived Inputs

Before coupling can run, the following derived assets must already exist:

| Asset | Requirement |
| --- | --- |
| EEG BIDS root | `paths.bids_root` must resolve and exist |
| fMRI BIDS root | `paths.bids_fmri_root` must resolve and exist |
| derivatives root | `paths.deriv_root` must resolve and exist |
| clean EEG epochs | loadable with strict event alignment |
| clean EEG events TSV | must exist beside the clean epochs file |
| fMRIPrep outputs | subject-space (`T1w`) preprocessed BOLD and confounds must exist |
| FreeSurfer anatomy | subject cortical surfaces required for source modeling and surface sampling |
| EEG-to-MRI transform | `sub-<id>-trans.fif` |
| BEM solution | at least one `sub-<id>-*-bem-sol.fif` |

The coupling workflow validates these inputs but does not generate them internally.
Source-model and anatomy preparation are documented in:

```text
docs/eeg/source-localization.md
fmri_pipeline/README.md
```

### 2.3 Clean EEG Event Contract

The clean event table is a hard dependency. It must contain:

- `trial_id`,
- `onset`,
- `duration`,
- one usable run column from `{block, run_id, run, session}`,
- one usable trial-number column from
  `{trial_number, events_trial_number, events_trial_index}`.

For the production Study 2 nuisance model, the same table must also contain the
event-level artifact columns exported by upstream preprocessing:

- `residual_ecg_coupling`,
- `peripheral_low_gamma_power`.

These are carried forward into the coupling table as `events_*` metadata and reused in
the EEG artifact composite.

### 2.4 Task Event Semantics and Behavioral Fields

Study 2 depends on the pain-paradigm event schema because plateau selection,
temperature/site covariates, and full-trial history reconstruction all reference these
upstream fields. When the repository's phase-level event generator is used, the BIDS
events contract is:

| Event field | Implemented meaning |
| --- | --- |
| `trial_type == "fixation_rest"` | pre-stimulation fixation interval at `35°C` baseline |
| `trial_type == "fixation_poststim"` | fixation interval between stimulation end and pain-question onset; the generator documents a random `4.5-8.5` s interval |
| `trial_type == "stimulation"` | thermal stimulation event of total duration `12.5` s |
| `stim_phase == "ramp_up"` | first `3.0` s of the stimulation row when phase-level events are enabled |
| `stim_phase == "plateau"` | middle `7.5` s of the stimulation row when phase-level events are enabled |
| `stim_phase == "ramp_down"` | final `2.0` s of the stimulation row when phase-level events are enabled |
| `trial_type == "pain_question"` | binary pain question window, maximum `4.0` s, can terminate early on response |
| `trial_type == "vas_rating"` | visual-analogue-scale rating window, maximum `7.0` s, can terminate early on response |
| `onset` | seconds from BOLD run start after the configured onset-reference transformation |
| `run_id` | one-based run index from PsychoPy |
| `trial_number` | one-based within-run trial index |
| `stimulus_temp` | thermode target temperature in `°C` |
| `selected_surface` | experiment-defined stimulation-site index |
| `pain_binary_coded` | pain yes/no response coded `1` or `0` |
| `vas_final_coded_rating` | final coded rating; non-pain trials use `0-99`, pain trials use `100-200` |
| `vas_scale_min`, `vas_scale_max` | trial-specific rating-scale bounds written by the event generator |

The production Study 2 coupling analysis selects plateau-phase stimulation rows through
`stim_phase == "plateau"`, but the broader task history is preserved because the
underlying events tables also include the non-selected fixation, question, and rating
rows.

### 2.5 Surface and Forward-Model Contract

The subject-specific surface and BEM prerequisites are strict. The workflow fails fast
if any of the following are missing:

```text
<SUBJECTS_DIR>/sub-<ID>/surf/lh.pial
<SUBJECTS_DIR>/sub-<ID>/surf/rh.pial
<SUBJECTS_DIR>/sub-<ID>/surf/lh.white
<SUBJECTS_DIR>/sub-<ID>/surf/rh.white
<SUBJECTS_DIR>/sub-<ID>/bem/sub-<ID>-trans.fif
<SUBJECTS_DIR>/sub-<ID>/bem/sub-<ID>-*-bem-sol.fif
```

No fallback or substitute path is used when these files are absent.

## 3. Fixed Anatomical Specification

Study 2 disables dynamic ROI building at runtime and instead uses a frozen ROI library
declared in:

```text
studies/pain_study/study2/config/roi_library/study2/fsaverage/study2_roi_manifest.json
```

The production ROIs are:

| ROI | Hemisphere | Anatomical construction | Runtime label files |
| --- | --- | --- | --- |
| `right_operculo_periinsular` | right | union of right long insular/subcentral and circular insular labels in `aparc.a2009s` | `rh.study2_right_operculo_parietal.label` |
| `midcingulate_pre_sma` | bilateral | bilateral mid-anterior cingulate plus a custom medial pre-SMA extension constrained from `BA6_exvivo.thresh.label` | `lh.study2_cingulo_pre_sma.label`, `rh.study2_cingulo_pre_sma.label` |

In the manifest, the right operculo-periinsular ROI is defined as the union of:

- `G_Ins_lg_and_S_cent_ins-rh`,
- `G_and_S_subcentral-rh`,
- `S_circular_insula_inf-rh`,
- `S_circular_insula_sup-rh`.

The bilateral midcingulate/pre-SMA ROI is defined from:

- `G_and_S_cingul-Mid-Ant-lh`,
- `G_and_S_cingul-Mid-Ant-rh`,
- a medial pre-SMA mask constrained to superior frontal cortex with
  `|x| <= 20 mm` and `y >= 0 mm`.

At runtime each ROI is:

1. loaded on `fsaverage`,
2. combined across label components when needed,
3. morphed to the target subject,
4. intersected with the subject source space,
5. converted into an area-weighted set of source rows for EEG extraction,
6. converted into an area-weighted set of cortical vertices for BOLD extraction.

If a morphed ROI has no overlap with the subject source space or no valid cortical
surface vertices, the workflow stops with an explicit error.

## 4. EEG Signal Model and Trial-Wise Feature Construction

### 4.1 Production EEG Specification

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

The production configuration fixes the band names to `alpha` and `beta`. Their numeric
cutoffs are resolved at runtime from the shared `frequency_bands` mapping in the active
repository configuration.

### 4.2 Source Modeling

Clean EEG epochs and aligned clean events are loaded with strict alignment. A
subject-specific cortical forward model is then resolved from the subject's FreeSurfer
surfaces, BEM solution, and EEG-to-MRI transform.

In the production analysis, source estimation uses a band-specific LCMV beamformer.
For each band:

- sensor data are band-pass filtered with a fourth-order zero-phase Butterworth filter,
- a full-epoch data covariance is estimated with OAS shrinkage,
- a baseline-window noise covariance is estimated with OAS shrinkage,
- both covariances are validated as positive semidefinite and sufficiently conditioned,
- LCMV filters are built with `pick_ori="normal"` and
  `weight_norm="unit-noise-gain"`.

Alternative source methods are not part of the confirmatory model but can be rerun in
the sensitivity branch.

### 4.3 Trial-Wise Source-Power Estimation

For each trial, each ROI, and each band, the source time series are converted to
analytic power by Hilbert transformation and squared magnitude. Let
`P_{sriqjb}(t)` denote the band-limited analytic power at source row `j` in ROI `q`.
Let `T_active` and `T_base` denote the active and baseline sample sets.

The area-weighted ROI power summaries are:

$$
\bar{P}^{active}_{sriqb}
=
\frac{\sum_{j \in q} w_{qj}
\left(
\frac{1}{|T_{active}|}
\sum_{t \in T_{active}} P_{sriqjb}(t)
\right)}
{\sum_{j \in q} w_{qj}}
$$

$$
\bar{P}^{base}_{sriqb}
=
\frac{\sum_{j \in q} w_{qj}
\left(
\frac{1}{|T_{base}|}
\sum_{t \in T_{base}} P_{sriqjb}(t)
\right)}
{\sum_{j \in q} w_{qj}}
$$

where `w_{qj}` is the cortical-area weight assigned to source row `j` after ROI
morphing and source-space intersection.

The final EEG predictor is the active-to-baseline power ratio expressed in decibels:

$$
X_{sriqb}
=
10 \log_{10}
\left(
\frac{\bar{P}^{active}_{sriqb}}
{\bar{P}^{base}_{sriqb}}
\right)
$$

The production EEG predictor columns are:

- `eeg_right_operculo_periinsular_alpha`,
- `eeg_right_operculo_periinsular_beta`,
- `eeg_midcingulate_pre_sma_alpha`,
- `eeg_midcingulate_pre_sma_beta`.

The trial-wise EEG table also carries:

- `trial_id`,
- `run_num`,
- `onset`,
- `duration`,
- `trial_number`,
- `trial_key`,
- all clean-event metadata with the `events_` prefix.

## 5. fMRI Trial-Wise Response Estimation

### 5.1 Production fMRI Specification

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

The main analysis is therefore restricted to plateau trials:

```text
stim_phase == "plateau"
```

In the repository's pain-paradigm event generator, `stim_phase` is attached only to
`trial_type == "stimulation"` rows. The associated pain-paradigm fMRI override template
models `stimulation`, `pain_question`, and `vas_rating`, while applying phase scoping
only to stimulation rows.

### 5.2 Trial-Wise Surface GLM

Study 2 invokes the shared trial-signature extraction machinery in least-squares
separate (LSS) mode with trial beta and trial variance writing enabled. One LSS model is
fit per selected trial. The selected-trial table written by this stage must contain
`events_file` and `events_row`, because Study 2 reconstructs trial history from the
original source events rather than from the reduced LSS subset alone.

For cortical extraction, each preprocessed BOLD run is sampled from volume to the
subject's pial/white surfaces using depth sampling. A subject-surface first-level GLM is
then fit for each target trial, and the target contrast is exported as:

- an effect-size surface,
- an effect-variance surface when variance-weighted modeling is enabled.

### 5.3 ROI-Level BOLD Summaries

Let `beta_{sriqv}` denote the trial-wise LSS effect estimate at cortical vertex `v` and
let `a_{qv}` denote the positive cortical area weight for that vertex. The ROI mean BOLD
response is:

$$
Y_{sriq}
=
\frac{\sum_{v \in q} a_{qv} \beta_{sriqv}}
{\sum_{v \in q} a_{qv}}
$$

When trial-wise effect variances are available, Study 2 also computes the variance of
the area-weighted mean:

$$
Var(Y_{sriq})
=
\sum_{v \in q}
\left(
\frac{a_{qv}}{\sum_{u \in q} a_{qu}}
\right)^2
\sigma^2_{sriqv}
$$

The production BOLD columns are:

- `bold_right_operculo_periinsular`,
- `bold_midcingulate_pre_sma`,
- `boldvar_right_operculo_periinsular`,
- `boldvar_midcingulate_pre_sma`.

An optional local signature-expression branch exists in code, but it is disabled in the
frozen production configuration and the production YAML sets `paths.signature_maps: []`.

### 5.4 fMRI Quality Control

The production workflow computes two fMRI quality-control families:

1. design QC from each trial-specific LSS design matrix,
2. ROI QC from ROI-level BOLD outlier burden.

The production thresholds are:

| QC metric | Threshold |
| --- | --- |
| maximum absolute target-to-other-regressor correlation | `0.5` |
| minimum target efficiency | `0.01` |
| maximum target variance | `100.0` |
| ROI outlier MAD threshold | `5.0` |
| maximum ROI outlier proportion | `0.1` |

Design-level failures mark the subject as globally non-interpretable for fMRI. ROI
outlier failures are ROI-specific and only invalidate the affected ROI cells.

## 6. Multimodal Alignment, Trial History, and Nuisance Modeling

### 6.1 Trial Alignment

Study 2 enforces:

```text
eeg_bold_coupling.alignment.key_mode = "run_trial_number"
```

The canonical multimodal trial key is therefore:

```text
<run_num>|trial-<trial_number>
```

Both the EEG table and the LSS trial table independently build this key from run number
and trial number. The multimodal table is created by exact one-to-one merges on
`trial_key`. If timing or metadata disagree across sources, the workflow stops with an
explicit merge error.

### 6.2 Full-Session Trial History Covariates

Because the LSS `trials.tsv` retains `events_file` and `events_row`, Study 2 rebuilds
the complete selected-trial history from the original source events and derives
trial-history covariates before the multimodal merge.

The derived covariates are:

- `temperature`,
- `temperature_sq`,
- `delta_temperature`,
- `exp_site`,
- `exp_global`,
- `block_start`,
- `trial_position`.

Their implemented definitions are:

- `temperature`: resolved numeric temperature column from the selected source events;
  in production this is fixed to `events_stimulus_temp`,
- `temperature_sq`: squared temperature, when enabled,
- `delta_temperature`: within-run difference from the immediately preceding selected
  trial,
- `exp_global`: zero-based cumulative exposure count over all selected trials after
  sorting by run, onset, and duration,
- `exp_site`: cumulative count within stimulation site; in production the site identity
  is taken from `events_selected_surface`,
- `block_start`: indicator that the current selected trial is the first selected trial in
  its run,
- `trial_position`: one-based within-run index after sorting selected trials by onset.

### 6.3 Trial-Wise Motion and Signal Nuisance Terms

The confirmatory production model uses:

```text
temperature
temperature_sq
fd
eeg_artifact
exp_site
exp_global
block_start
```

`fd` and `dvars` are derived from run-level confound time series by convolving a
canonical trial regressor with the confound vector and taking the weighted mean over scan
times. For a confound series `m(t)` and trial-specific HRF regressor `h_i(t)`, the
trial-level summary is:

$$
M_i
=
\frac{\sum_t h_i(t) m(t)}
{\sum_t h_i(t)}
$$

The production nuisance thresholds are:

| Metric | Status | Threshold |
| --- | --- | --- |
| FD | enabled | censor `> 0.5` |
| DVARS | enabled | censor `> 2.5` |
| EEG artifact component z | enabled | censor `> 5.0` |
| EEG artifact composite | enabled | censor `> 3.0` |
| finite model terms | enabled | all configured terms must be finite |

### 6.4 EEG Artifact Composite

The production EEG artifact composite is built from two event-level quantities imported
from the clean EEG events table:

- `event_residual_ecg_coupling`,
- `event_peripheral_low_gamma_power`.

Each component is robust-z-scored within run:

$$
z_{ik}
=
\frac{x_{ik} - median_r(x_{\cdot k})}
{1.4826 \; MAD_r(x_{\cdot k})}
$$

If the within-run MAD is zero or non-finite, the implemented robust z-score defaults to
`0.0` for that run. The production composite then keeps only positive burden and averages
across required components:

$$
A_i
=
\frac{1}{K}
\sum_{k=1}^{K} \max(z_{ik}, 0)
$$

This yields the trial-wise `eeg_artifact` predictor used both for censoring and as a
covariate in the confirmatory model.

### 6.5 Trial Censoring

The post-merge QC table includes explicit exclusion flags:

- `exclude_fd_threshold`,
- `exclude_dvars_threshold`,
- `exclude_eeg_artifact_component`,
- `exclude_eeg_artifact_composite`,
- `exclude_non_finite_model_term`,
- `keep_trial`,
- `exclude_reason`.

Only rows with `keep_trial == true` enter confirmatory or sensitivity modeling.

## 7. Confirmatory Statistical Model

### 7.1 Subject-Level Eligibility

Each confirmatory cell is first evaluated within subject for basic interpretability. The
production subject-level requirements are:

| Criterion | Threshold |
| --- | --- |
| minimum retained trials per cell | `20` |
| minimum retained runs per cell | `2` |
| predictor variance | must be non-zero |
| outcome variance | must be non-zero |
| fMRI QC | cell must not be globally or ROI-specifically failed |

Subject-level status labels include:

- `ok`,
- `insufficient_trials`,
- `insufficient_runs`,
- `insufficient_variance`,
- `fmri_qc_failed`.

Only subjects with status `ok` contribute to the corresponding group model.

### 7.2 Group Model

For each confirmatory cell, all retained trials from eligible subjects are pooled and fit
with the `nlme::lme` backend through the Study 2 R wrapper. The production model uses:

- restricted maximum likelihood (`REML`),
- a random intercept and random EEG slope by subject,
- a continuous-time AR(1) correlation structure within subject and run,
- an optional fixed run effect, enabled in production,
- trial-wise BOLD variance weighting, enabled in production.

A group fit is only attempted when at least two eligible subjects remain for that cell.

The fixed-effect model for cell `(q, b)` can be written as:

$$
Y^{(q)}_{sri}
=
\beta_0
+ \beta_1 X^{(q,b)}_{sri}
+ \sum_{m=1}^{M} \gamma_m C_{sri,m}
+ \delta_{run(r)}
+ u_{0s}
+ u_{1s} X^{(q,b)}_{sri}
+ \varepsilon_{sri}
$$

where `C_{sri,m}` denotes the prespecified covariates
`{temperature, temperature_sq, fd, eeg_artifact, exp_site, exp_global, block_start}`.

Within subject, the production pipeline standardizes:

- the EEG predictor,
- the BOLD outcome,
- every non-binary continuous covariate not explicitly marked as nonstandardized.

Binary columns remain unscaled. When `boldvar_<roi>` is present, the ROI outcome variance
is rescaled into the same standardized units and passed into `nlme::lme` through
`varFixed`, so trials with larger estimated BOLD uncertainty contribute less information.

The continuous-time dependence structure is indexed by trial onset in seconds:

$$
Cor(\varepsilon_{sri}, \varepsilon_{sr'i'})
=
\rho^{|\tau_{sri} - \tau_{sr'i'}|}
$$

for observations in the same subject/run series, where `tau` is trial onset.

### 7.3 Inference

The production statistical settings are:

| Parameter | Value |
| --- | --- |
| backend | `nlme_lme_ar1` |
| fit method | `reml` |
| alpha | `0.05` |
| include run fixed effect | `true` |
| use outcome variance | `true` |
| max iterations | `200` |
| EM iterations | `50` |
| singular tolerance | `1e-8` |

For each fitted cell, the group output includes:

- the standardized EEG slope (`beta`),
- its standard error,
- a Wald-style `z_value`,
- the model `p_value`,
- a `95%` confidence interval,
- the estimated AR(1) parameter `rho`,
- convergence and singularity status.

Holm correction is applied only across interpretable confirmatory p-values. Non-fitted or
non-interpretable cells are retained in the results table but do not enter multiplicity
correction.

## 8. Sensitivity Analyses, Negative Control, and Robustness Adjudication

### 8.1 Enabled Sensitivity Branches

The frozen production configuration enables the following sensitivity branches:

| Branch | Implemented change |
| --- | --- |
| `residualized_correlation` | rank-transform predictor, outcome, and numeric nuisance terms; residualize both on the nuisance design; compute per-subject residual Pearson correlations; aggregate with `5000` bootstrap and `5000` permutation iterations |
| `primary_permutation` | compute subject-level standardized EEG slopes after nuisance adjustment; aggregate with `5000` bootstrap and `5000` permutation iterations |
| `source_methods` | recompute EEG predictors with `method="eloreta"` while retaining the same ROI and band definitions |
| `anatomical_specificity` | replace the confirmatory ROIs with control ROIs and rerun confirmatory-style models |
| `artifact_models` | augment the EEG artifact model with `global_amplitude` and rerun censoring and confirmatory-style models |
| `within_between` | replace the primary predictor with a centered predictor column and add an additional mean term as a nonstandardized covariate |

The enabled control ROIs in the anatomical-specificity branch are:

- `left_operculo_periinsular_control`,
- `posterior_midcingulate_control`.

The enabled artifact-model item is:

- `expanded_artifact`.

### 8.2 Implemented but Disabled Branches

The following branches exist in code but are disabled in the frozen production
configuration:

- `painful_only`,
- `alternative_fmri`,
- `delta_temperature`,
- `temperature_categorical`.

### 8.3 Negative Control

The production negative control is `trial_shuffle`. Within each run, the ROI BOLD table
is permuted across trial keys, merged back onto the unchanged EEG predictors and
covariates, and rerun through the same censoring and confirmatory-style modeling steps.
At the group level, the negative-control branch is considered acceptable when it is
either non-interpretable or non-significant at the configured alpha level.

### 8.4 Robustness Adjudication

The group pipeline also writes leave-one-out refits for each confirmatory cell:

- leave-one-subject-out,
- leave-one-run-out.

These refits are summarized in:

```text
robustness/leave_one_out_refits.tsv
robustness/leave_one_out_summary.tsv
robustness/adjudication_summary.tsv
```

The final adjudication table records whether each confirmatory cell passes:

- Holm-significant confirmatory inference,
- all enabled source-method sign checks,
- all enabled artifact-model sign checks,
- the within-between sign check,
- leave-one-subject sign stability,
- leave-one-run sign stability,
- the negative-control criterion.

The final boolean summary column is `overall_robust_pass`.

## 9. Reproducibility, Execution, and Outputs

### 9.1 Default Output Roots

If `eeg_bold_coupling.output_dir` is null, subject outputs are written to:

```text
<paths.deriv_root>/sub-<ID>/multimodal/eeg_bold_coupling/task-<task>/contrast-plateau_trials/
```

If `eeg_bold_coupling.output_dir` is null, group outputs are written to:

```text
<paths.deriv_root>/group/multimodal/eeg_bold_coupling/task-<task>/contrast-plateau_trials/
```

### 9.2 Core Subject Outputs

The main subject directory contains:

- `runtime_profile.json`,
- `coupling_config.json`,
- `qc_summary.json`,
- `fmri_design_qc.tsv`,
- `fmri_roi_qc.tsv`,
- `fmri_qc_summary.json`,
- `roi_manifest.tsv`,
- `analysis_cells.tsv`,
- `trialwise_eeg.tsv`,
- `trialwise_trials.tsv`,
- `trialwise_bold.tsv`,
- `trialwise_merged.tsv`,
- `trialwise_qc.tsv`,
- `lss/`,
- `sensitivities/`,
- `negative_controls/`,
- `secondary/` when secondary analyses are enabled.

`trialwise_merged.tsv` and `trialwise_qc.tsv` both contain the post-censoring QC table;
the second filename is a more explicit duplicate.

### 9.3 Core Group Outputs

The main group directory contains:

- `analysis_cells_all.tsv`,
- `group_results.tsv`,
- `sensitivities/`,
- `negative_controls/`,
- `secondary/`,
- `robustness/`.

### 9.4 Batch Metadata

Independent of the subject/group analysis trees, the workflow also writes:

```text
<paths.deriv_root>/logs/eeg_bold_coupling_batch_ledger.tsv
<paths.deriv_root>/logs/run_metadata/eeg_bold_coupling/run_<timestamp>_<id>.json
```

These files capture batch success/failure state and run-level reproducibility metadata.

### 9.5 Production Execution

The production Study 2 configuration is not the CLI default. A production run should
therefore pass the Study 2 YAML explicitly.

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

The repository also ships a smoke configuration for fast validation:

```text
studies/pain_study/study2/config/eeg_bold_coupling_smoketest.yaml
```

This smoke file is useful for pipeline validation but it is not the frozen scientific
specification used for Study 2.

The production YAML intentionally sets `paths.signature_maps: []`. Consequently, the
main Study 2 analysis does not require signature maps, local signature-expression
outputs are disabled by default, and any future activation of that branch requires valid
signature-map inputs.

### 9.6 Common Hard Failures

Common explicit stop conditions include:

- missing `paths.bids_root`,
- missing `paths.bids_fmri_root`,
- missing clean epochs or clean events,
- missing usable run or trial-number columns in EEG events,
- missing `events_file` or `events_row` in LSS trials,
- no selected source events matching `stim_phase == "plateau"`,
- missing subject surfaces, transform, or BEM solution,
- missing fMRIPrep confounds when confound selection is enabled,
- missing configured model terms after multimodal merge,
- unavailable `Rscript` executable for group modeling.

Study 2 deliberately surfaces these conditions as explicit errors rather than attempting
fallback behavior or backward-compatibility shims.

## 10. Article-Writing Completion Inventory

Sections 1-9 define the full repository-grounded computational methods contract for
Study 2. A manuscript writer still needs a second class of information: facts that are
required in a journal Methods section but cannot be inferred from code or configuration
alone. To make article drafting operationally complete, those required non-code inputs
are enumerated here rather than left implicit.

### 10.1 What This Document Already Fixes

The present file already supplies the implementation-grounded material needed for an
article Methods section:

- the confirmatory objective and inferential family,
- the frozen ROI definitions and source/surface extraction rules,
- the EEG, fMRI, alignment, censoring, and nuisance specifications,
- the exact mixed-effects model and multiplicity rule,
- the enabled sensitivity, negative-control, and robustness branches,
- the execution contract, output locations, and explicit hard-failure conditions.

### 10.2 Study-Record Information That Must Be Added Manually

The following manuscript-critical facts are not discoverable from the repository and
must be transcribed from the study record before a full article can be written:

| Category | Required facts | Typical source outside the repository |
| --- | --- | --- |
| ethics and governance | IRB/REB approval body, approval identifier, consent procedure, preregistration or SAP provenance if any | ethics submission, protocol, preregistration record |
| participant flow | recruited `n`, excluded `n`, analyzed `n`, and exact exclusion reasons by stage | screening log, QC log, final cohort spreadsheet |
| cohort descriptors | age summary, sex/gender reporting, handedness, inclusion/exclusion criteria, clinical descriptors if relevant | demographics sheet, protocol |
| EEG acquisition | amplifier/manufacturer, cap layout, sampling rate, online reference, impedance policy, recording environment, synchronization method | acquisition SOP, amplifier export, lab protocol |
| MRI acquisition | scanner/vendor, field strength, head coil, sequence names, voxel size, TR, TE, flip angle, multiband factor, fieldmap strategy | scan protocol, DICOM header summary |
| thermal stimulation paradigm | thermode hardware, stimulation site definitions, temperature calibration procedure, trial count per run, run count, question wording, rating instructions | experiment script, protocol, lab notebook |
| timing details | actual inter-trial timing, duration tolerances, whether early responses truncated question/rating windows in practice, any run-to-run deviations | PsychoPy task code, behavioral exports |
| preprocessing attrition | subjects/runs/trials removed at each preprocessing and QC stage, including plateau-trial retention after censoring | pipeline QC outputs, manual adjudication log |
| analysis justification | scientific rationale for selected ROIs, EEG bands, time windows, nuisance thresholds, and robustness branches | analysis plan, manuscript discussion notes |
| software provenance | repository commit hash, package versions, container/environment details if reported | git metadata, environment export |
| result-linked counts | exact eligible-subject count and retained-trial count for each confirmatory cell and sensitivity branch | `group_results.tsv`, sensitivity summaries, QC tables |
| supplement material | QC/attrition figure, ROI visualization figure, design-matrix diagnostics if reported | manuscript figures, exported QC summaries |
| limitations | explicit statement of source leakage risk, ROI dependence, residual autocorrelation assumptions, trial-selection dependence, and non-causal interpretation | manuscript discussion draft |

### 10.3 Minimal Fill-In Blocks for the Final Methods Section

When converting this repository specification into article prose, the following
manuscript blocks must be completed with study-record facts:

- `Participants and ethics`: insert recruitment source, final analyzed sample, exclusion
  flow, ethics approval, and consent language.
- `EEG acquisition`: insert hardware, sampling rate, reference, impedance policy, and
  synchronization details.
- `MRI acquisition`: insert scanner and sequence parameters.
- `Thermal pain paradigm`: insert thermode hardware, calibration procedure, number of
  runs and trials, stimulation sites, and rating instructions.
- `Quality control and attrition`: insert counts removed by subject, run, and trial,
  plus confirmatory-cell sample sizes.
- `Limitations`: insert the study-specific interpretation limits that cannot be read from
  code alone.

Until those external facts are inserted, this file should be treated as a complete
computational methods specification and an article-writing scaffold, not as a fully
submission-ready journal Methods section.
