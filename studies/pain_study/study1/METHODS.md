# Study 1: Manuscript-Style Methods for Trial-Wise EEG Prediction of fMRI Pain-Signature Expression

This document is the publication-style Methods companion for Study 1. It is
implementation-grounded: every methodological statement below reflects the current
behavior of the Study 1 workflow in `studies/pain_study/study1/`.

The frozen production specification is:

```text
studies/pain_study/study1/config/study1_config.yaml
```

Study 1 asks whether trial-wise EEG predicts trial-wise expression of two established
pain-related fMRI signatures, `NPS` and `SIIPS1`. The implemented study is multimodal
and trial-resolved: it first constructs a shared table of trial-wise fMRI signature
targets aligned to clean EEG trials, then evaluates two predictive lanes under a
subject-level leave-one-subject-out design:

- a feature-based machine-learning lane using a Study 1-owned `trial_ml_safe` EEG
  feature store,
- a deep-regression lane that learns directly from band-limited EEG trial tensors.

## 1. Analytical Objective and Fixed Scope

The primary Study 1 objective is to predict continuous trial-wise fMRI signature
expression from EEG. The study is not a generic signature screen and not a flexible
multitask framework. The frozen production analysis is restricted to:

- the two targets `NPS` and `SIIPS1`,
- trial-wise `lss` fMRI signature extraction,
- the contrast `pain_vs_nonpain`,
- `trial_type == "stimulation"`,
- `stim_phase == "plateau"`,
- MNI-space fMRI inputs,
- a fixed confirmatory EEG feature family (`power`),
- a fixed set of exploratory EEG feature families,
- a fixed set of deep-regression band presets.

Throughout this document:

- `s` indexes subject,
- `i` indexes retained trial,
- `v` indexes voxel,
- `b` indexes frequency band,
- `c` indexes EEG channel,
- `t` indexes time sample.

## 2. Data Requirements and Upstream Preconditions

### 2.1 Required roots

Study 1 depends on three configured roots:

| Root | Requirement |
| --- | --- |
| EEG BIDS root | `paths.bids_root` must resolve when EEG access is needed |
| fMRI BIDS root | `paths.bids_fmri_root` must resolve for target preparation |
| derivatives root | `paths.deriv_root` must resolve and serves as the Study 1 output base |

### 2.2 Required derived inputs

Before Study 1 can run, the following upstream products must already exist:

- clean EEG events for each subject and task,
- clean EEG epochs loadable with strict event alignment,
- fMRI inputs compatible with trial-wise signature extraction,
- a valid signature-map configuration containing both `NPS` and `SIIPS1`.

Study 1 does not perform generic EEG preprocessing internally. Missing clean epochs,
missing clean events, or missing signature maps surface as explicit errors.

### 2.3 Clean EEG event contract

The clean EEG event table is a hard dependency for target preparation and deep-regression
alignment. It must contain:

- `onset`,
- `duration`,
- one usable run or block column from `{block, run_id, run, session}`.

The preferred trial-index columns are:

- `trial_number`,
- `trial_index`.

If explicit trial numbering is absent, parts of the workflow can still align by run plus
onset and duration, but the intended contract is a clean, trial-resolved event table.

### 2.4 Clean EEG epoch contract

`prepare-features` and `deep-regression` both require clean EEG epochs loadable with
strict alignment to the clean event table. Deep regression further requires a non-empty
intersection of EEG channels across all included subjects.

### 2.5 fMRI target contract

The production Study 1 target specification is:

| Parameter | Value |
| --- | --- |
| input source | `fmriprep` |
| fMRIPrep space | `MNI152NLin2009cAsym` |
| extraction method | `lss` |
| metric | `dot` |
| normalization | `none` |
| contrast name | `pain_vs_nonpain` |
| HRF model | `spm` |
| drift model | `cosine` |
| high-pass | `0.008` Hz |
| confounds strategy | `auto` |
| LSS other regressors | `all` |

Study 1 requires MNI-space fMRI inputs. If the configured fMRI space does not contain
`MNI`, target preparation fails.

### 2.6 Subject-count rules

Study 1 uses different subject-count rules for different stages:

- `prepare-targets` can run on a single subject,
- `prepare-features`, `feature-benchmark`, and `deep-regression` require at least
  `study1.cohort.min_subjects` subjects after cohort resolution from the primary target
  table.

The production default is:

```yaml
study1:
  cohort:
    min_subjects: 2
```

This minimum exists because both predictive lanes use subject-level leave-one-subject-out
validation.

## 3. Trial-Wise fMRI Target Construction

### 3.1 Fixed target set

The main target set is fixed and ordered:

```text
["NPS", "SIIPS1"]
```

Study 1 does not accept arbitrary signature sets as equivalent replacements for the main
analysis.

### 3.2 Subject-level trial-signature extraction

For each requested subject, Study 1 builds a `TrialSignatureExtractionConfig` from
`study1.targets.*` and runs the shared trial-signature extraction routine. The current
production mode is least-squares separate (`lss`) restricted to:

- `pain_binary_coded == "1"` versus `pain_binary_coded == "0"`,
- `trial_type == "stimulation"`,
- `stim_phase == "plateau"`.

Let `beta_{s,i}(v)` denote the LSS-derived effect estimate for subject `s`, trial `i`,
and voxel `v`. Let `M_k(v)` denote the signature-map weight for target `k`, where
`k in {NPS, SIIPS1}`. Because Study 1 uses the `dot` metric with `normalization = none`,
the trial-wise target is:

$$
y_{s,i}^{(k)} = \sum_{v \in V} \beta_{s,i}(v) \, M_k(v).
$$

### 3.3 EEG-fMRI alignment

After trial-signature extraction, Study 1 aligns `NPS` and `SIIPS1` back onto EEG trials.
Two keying strategies are considered:

1. run plus trial number or trial index,
2. run plus onset and duration rounded to `study1.targets.round_decimals`.

If `r_i` is the run index, `q_i` the trial index, `o_i` the onset, and `d_i` the
duration, the candidate keys are:

$$
\kappa_i^{trial} = (r_i, q_i)
$$

and

$$
\kappa_i^{time} = \left(r_i, \mathrm{round}(o_i, 3), \mathrm{round}(d_i, 3)\right).
$$

The alignment mode with the larger number of successful matches is used. If neither
strategy yields any successful matches, the stage fails.

### 3.4 Primary target table

The output of `prepare-targets` is a wide table with one row per retained trial and the
required columns:

- `subject_id`,
- `task`,
- `block`,
- `trial_index`,
- `onset`,
- `duration`,
- `NPS`,
- `SIIPS1`.

Finite values for both `NPS` and `SIIPS1` are required for every retained trial. The
primary target table is then reused as the canonical Study 1 cohort contract for all
downstream stages.

## 4. Study 1 EEG Feature Preparation

### 4.1 Feature-family contract

The feature-based lane uses a Study 1-owned `trial_ml_safe` feature store. The feature
families are split into one confirmatory family and several exploratory families:

| Role | Families |
| --- | --- |
| confirmatory | `power` |
| exploratory | `spectral`, `aperiodic`, `erds`, `ratios`, `asymmetry`, `complexity`, `bursts` |

Two exploratory families are explicitly windowed:

- `erds`,
- `bursts`.

These use `time_frequency_analysis.baseline_window` and
`time_frequency_analysis.active_window`, both of which must be finite two-element ranges
with `start < end`.

### 4.2 Pinned provenance contract

Study 1 does not accept arbitrary shared feature outputs as equivalent. During feature
preparation it pins the following provenance settings:

| Setting | Required value |
| --- | --- |
| `feature_engineering.analysis_mode` | `trial_ml_safe` |
| `feature_engineering.power.subtract_evoked` | `false` |
| `feature_engineering.precomputed.subtract_evoked` | `false` |
| `feature_engineering.aperiodic.subtract_evoked` | `false` |
| `feature_engineering.bands.use_iaf` | `false` |
| `feature_engineering.bursts.threshold_reference` | `trial` |

After extraction, metadata written beside each prepared feature family are validated
against this contract. If the metadata do not prove that the outputs were generated under
these settings, the stage fails.

### 4.3 Study-owned output location

Prepared features are written under the Study 1 namespace:

```text
<study1_root>/features_trial_ml_safe/sub-<id>/eeg/features/<family>/
```

This isolation is deliberate. The feature benchmark reads only from the Study 1-owned
feature root and does not fall back to generic shared feature directories.

## 5. Feature-Based Model Comparison

The feature benchmark is a nested cross-validated model-comparison analysis that predicts
trial-wise fMRI signature expression from Study 1-owned features.

### 5.1 Confirmatory and exploratory partitions

The benchmark is divided into two partitions:

- `primary`, which evaluates the fixed `power` family under the preset bands,
- `exploratory`, which evaluates each configured exploratory family without a preset
  band restriction.

The fixed presets are:

| Preset | Bands |
| --- | --- |
| `alpha` | `["alpha"]` |
| `beta` | `["beta"]` |
| `gamma` | `["gamma"]` |
| `alpha_beta_gamma` | `["alpha", "beta", "gamma"]` |

### 5.2 Compared models and validation

Each benchmark run compares three regressors under the same outer subject-level folds:

- `elasticnet`,
- `ridge`,
- `rf`.

All models share the same leave-one-subject-out outer validation structure. When the
outer training split contains at least two unique subjects, inner hyperparameter tuning
is also group-aware.

The foldwise evaluation metrics are:

$$
R_f^2 = 1 - \frac{\sum_{i \in f}(y_i - \hat{y}_i)^2}{\sum_{i \in f}(y_i - \bar{y}_f)^2}
$$

and

$$
\mathrm{MAE}_f = \frac{1}{|f|}\sum_{i \in f}|y_i - \hat{y}_i|.
$$

The production benchmark defaults are:

| Parameter | Value |
| --- | --- |
| permutations | `0` |
| inner CV splits | `5` |
| outer jobs | `1` |
| feature harmonization | `intersection` |

If permutations are enabled, the shared model-comparison backend also writes paired
sign-flip inference and Holm-corrected pairwise model-comparison statistics.

## 6. Deep Regression from Band-Limited EEG Tensors

The `deep-regression` stage implements the second predictive lane.

### 6.1 Tensor construction

For each target and preset, the stage:

1. resolves the Study 1 cohort from the primary target table,
2. loads clean epochs and aligned clean events for each subject,
3. intersects EEG channels across subjects and retains only the common channel set,
4. filters the EEG separately into the requested frequency bands,
5. stacks the resulting data into tensors of shape:

$$
X \in \mathbb{R}^{n_{trials} \times n_{bands} \times n_{channels} \times n_{times}}.
$$

Unknown band names raise immediately.

### 6.2 Target alignment and LOSO training

Target alignment in the deep-regression lane is performed subject by subject against the
Study 1 primary target table using the same two candidate key strategies described above.
If neither key produces matches, the stage fails. All aligned targets must be finite.

Training uses `LeaveOneGroupOut`, where the held-out unit is subject. Within each outer
fold:

- inputs are standardized using training data only,
- targets are standardized using training data only and inverted after prediction,
- a group-based validation split is carved out of the training subjects when possible,
- early stopping is applied when validation loss stops improving.

If `X_{n,b,c,t}` denotes the training tensor, the foldwise input normalization is:

$$
\tilde{X}_{n,b,c,t} =
\frac{X_{n,b,c,t} - \mu_{b,c}}{\max(\sigma_{b,c}, 10^{-6})},
$$

where `mu_{b,c}` and `sigma_{b,c}` are computed from the training fold only over the
trial and time axes.

The model is a compact band-temporal regressor with:

- a temporal convolution,
- a spatial convolution across channels,
- batch normalization,
- ELU nonlinearities,
- dropout,
- adaptive average pooling,
- a small regression head.

The training objective is mean squared error on standardized targets:

$$
\mathcal{L}(\theta) = \frac{1}{m}\sum_{n=1}^{m}(\hat{\tilde{y}}_n - \tilde{y}_n)^2.
$$

The production training defaults are:

| Parameter | Value |
| --- | --- |
| epochs | `25` |
| batch size | `32` |
| learning rate | `0.001` |
| weight decay | `0.0001` |
| validation fraction | `0.2` |
| patience | `5` |
| temporal kernel size | `15` |
| temporal filters | `8` |
| dropout | `0.25` |
| CUDA | `false` |

Deep regression requires PyTorch. If PyTorch is unavailable, the stage raises an
explicit import error.

### 6.3 Outputs

Deep-regression outputs are written under:

```text
<study1_root>/deep_regression/<target>/<preset>/
```

Each run writes:

- `predictions.tsv`,
- `predictions.parquet`,
- `fold_metrics.tsv`,
- `summary.json`.

The prediction files contain trial metadata plus `y_true` and `y_pred`. The summary file
records the preset, target, band set, and average foldwise performance.

## 7. Report Aggregation

The `report` stage aggregates existing Study 1 outputs into one compact study-level
report.

At the Study 1 runner layer, the report stage does not use the passed subject list.
However, the top-level `signature-prediction` CLI is still registered as a
subject-requiring command, so at least one subject must still be provided when invoking
`report` from the CLI.

The aggregator scans the Study 1 output root for:

- feature-benchmark summaries at
  `feature_benchmark/*/*/*/model_comparison/metrics/model_comparison_summary.json`,
- deep-regression summaries at `deep_regression/*/*/summary.json`.

It can aggregate:

- feature-benchmark outputs only,
- deep-regression outputs only,
- both analysis lanes together.

It fails only when neither lane has produced any summary files.

The report stage writes:

- `study1_report.tsv`,
- `study1_report.parquet`,
- `study1_report.json`.

## 8. Outputs and Failure Semantics

With the default configuration, Study 1 writes under:

```text
<paths.deriv_root>/group/multimodal/study1/
```

The core output layout is:

```text
study1/
  targets/
    primary_targets.parquet
    primary_targets.tsv
  features_trial_ml_safe/
  feature_benchmark/
  deep_regression/
  reports/
    study1_report.tsv
    study1_report.parquet
    study1_report.json
```

Common hard failures include:

- missing or empty clean EEG events,
- missing clean EEG epochs,
- missing or invalid `paths.signature_maps`,
- missing `NPS` or `SIIPS1` values,
- non-MNI fMRI space during target preparation,
- no successful EEG-to-fMRI trial alignment,
- missing primary target table for downstream stages,
- requested subjects absent from the primary target table,
- fewer than the required minimum number of eligible subjects,
- missing prepared Study 1 feature families,
- invalid prepared-feature metadata showing the wrong provenance settings,
- missing common EEG channels for deep regression,
- non-finite deep-regression targets,
- unavailable PyTorch for deep regression,
- no feature-benchmark or deep-regression summaries for report aggregation.

All of these conditions surface as explicit errors rather than hidden fallbacks.
