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

The document is meant to be sufficient for writing the computational Methods section of
an article about Study 1. When a required paper field is dataset-specific rather than
code-fixed, the final section states exactly where it must be recovered from the study
outputs or upstream acquisition documentation.

## 1. Study Design, Objective, and Fixed Analytical Scope

The primary Study 1 objective is to predict continuous trial-wise fMRI signature
expression from EEG. The workflow is not a generic signature screen and not a flexible
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

The production workflow has two primary analysis lanes that share the same trial-wise
target definition:

1. a tabular benchmark comparing `elasticnet`, `ridge`, and `rf` on identical
   held-out-subject folds,
2. a deep-regression benchmark using a single compact band-temporal regressor on the
   same held-out-subject principle.

Throughout this document:

- `s` indexes subject,
- `r` indexes run or block,
- `i` indexes retained trial,
- `v` indexes voxel,
- `p` indexes EEG feature column,
- `b` indexes frequency band,
- `c` indexes EEG channel,
- `t` indexes time sample,
- `f` indexes held-out LOSO fold.

## 2. Data Sources, Upstream Processing, and Preconditions

### 2.1 Raw-study boundary

Study 1 itself does not start from raw acquisition files. If the pain study is being
reconstructed from source data, the upstream raw materials are documented in
`studies/pain_study/scripts/README.md` and the expected preparation order is:

1. BrainVision EEG to BIDS EEG,
2. fMRI DICOM to BIDS fMRI,
3. PsychoPy trial-summary merge into BIDS `events.tsv`,
4. downstream EEG preprocessing, feature extraction, and fMRI analysis.

The raw-conversion utilities describe the shared pain-study source types:

- BrainVision EEG source files,
- raw fMRI DICOM series,
- PsychoPy `TrialSummary.csv` behavior logs.

Study 1 consumes the resulting BIDS and derivative products. It does not perform raw
conversion internally.

### 2.2 Required roots and derived inputs

Study 1 depends on three configured roots:

| Root | Requirement |
| --- | --- |
| EEG BIDS root | `paths.bids_root` must resolve when EEG access is needed |
| fMRI BIDS root | `paths.bids_fmri_root` must resolve for target preparation |
| derivatives root | `paths.deriv_root` must resolve and serves as the Study 1 output base |

Before Study 1 can run, the following derived assets must already exist:

| Asset | Requirement |
| --- | --- |
| clean EEG epochs | loadable with strict event alignment |
| clean EEG events TSV | non-empty and trial-resolved |
| fMRIPrep-derived fMRI inputs | available in `MNI152NLin2009cAsym` space for trial-wise signature extraction |
| signature maps | must include both `NPS` and `SIIPS1` |
| Study 1 feature store | required before `feature-benchmark` |
| PyTorch | required only for `deep-regression` |

Study 1 validates these assets but does not create them all internally. Upstream EEG
cleaning and epoching are part of the shared EEG preprocessing stack; the definitions of
those upstream steps belong to `eeg_pipeline/preprocessing/README.md`. Shared
feature-family definitions belong to `eeg_pipeline/analysis/features/README.md`. Shared
fMRI first-level and trial-signature extraction behavior belongs to `fmri_pipeline/README.md`.

### 2.3 Clean EEG event contract

The clean EEG event table is a hard dependency for target preparation, feature loading,
and deep-regression target alignment. It must contain:

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
strict alignment to the clean event table. Deep regression additionally requires a
non-empty intersection of valid EEG channels across all included subjects. Channels are
restricted to true EEG channels and exclude marked bad channels.

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
| condition A column | `pain_binary_coded` |
| condition A value | `"1"` |
| condition B column | `pain_binary_coded` |
| condition B value | `"0"` |
| trial-type scope column | `trial_type` |
| trial-type scope | `["stimulation"]` |
| phase scope column | `stim_phase` |
| phase scope | `["plateau"]` |
| HRF model | `spm` |
| drift model | `cosine` |
| high-pass | `0.008` Hz |
| low-pass | `null` |
| smoothing FWHM | `null` |
| confounds strategy | `auto` |
| LSS other regressors | `all` |
| alignment rounding | `3` decimals |

Study 1 requires MNI-space fMRI inputs. If the configured fMRI space does not contain
`MNI`, target preparation fails.

### 2.6 Study namespace and stage dependency

With the default configuration, Study 1 writes to:

```text
<paths.deriv_root>/group/multimodal/study1/
```

The dependency graph is:

```text
prepare-targets -> prepare-features -> feature-benchmark
prepare-targets -> deep-regression
feature-benchmark and/or deep-regression -> report
```

The primary target table is therefore both a saved artifact and the cohort contract for
all downstream Study 1 stages.

### 2.7 Subject-count rules and inferential unit

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

The inferential and cross-validation unit is the subject. In both predictive lanes, the
outer split is leave-one-subject-out.

## 3. Trial-Wise fMRI Signature Target Construction

### 3.1 Fixed target set

The main target set is fixed and ordered:

```text
["NPS", "SIIPS1"]
```

Study 1 does not accept arbitrary signature sets as interchangeable replacements for the
main analysis.

### 3.2 Subject-level trial-signature extraction

For each requested subject, Study 1 builds a `TrialSignatureExtractionConfig` from
`study1.targets.*` and runs the shared trial-signature extraction routine. The current
production mode is least-squares separate (`lss`) restricted to:

- `pain_binary_coded == "1"` versus `pain_binary_coded == "0"`,
- `trial_type == "stimulation"`,
- `stim_phase == "plateau"`.

Let `beta_{s,i}(v)` denote the LSS-derived effect estimate for subject `s`, trial `i`,
and voxel `v`. Let `M_k(v)` denote the signature-map weight for target `k`, where
`k in {NPS, SIIPS1}`. Because Study 1 uses the `dot` metric with
`normalization = none`, the trial-wise target is:

$$
y_{s,i}^{(k)} = \sum_{v \in V} \beta_{s,i}(v) \, M_k(v).
$$

No within-run or within-subject normalization is applied in the frozen production
configuration.

### 3.3 EEG-fMRI alignment

After trial-signature extraction, Study 1 aligns `NPS` and `SIIPS1` back onto EEG trials
through the shared fMRI-signature target loader. Two keying strategies are considered:

1. run plus trial number or trial index,
2. run plus onset and duration rounded to `study1.targets.round_decimals`.

If `r_i` is the run index, `q_i` the trial index, `o_i` the onset, and `d_i` the
duration, the candidate keys are:

$$
\kappa_i^{trial} = (r_i, q_i)
$$

and

$$
\kappa_i^{time} =
\left(
r_i,
\mathrm{round}(o_i, 3),
\mathrm{round}(d_i, 3)
\right).
$$

The implementation constructs both key families for the clean EEG events table and for
the subject-level fMRI signature table, counts successful matches under each mode, and
uses the mode with the larger number of matches. When duplicate signature rows map to
the same key, the corresponding target value is aggregated by arithmetic mean before
back-projection onto EEG trials. If neither strategy yields any successful matches, the
stage fails.

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

### 3.5 Downstream cohort resolution

Downstream Study 1 stages resolve their working cohort from `primary_targets.parquet`.
For a requested subject set `S_req`, task `tau`, and available subject set `S_tau`
present in the primary table for that task, the resolved cohort is:

$$
S_{study1} =
\begin{cases}
S_{req}, & \text{if } S_{req} \neq \varnothing \\
S_{\tau}, & \text{otherwise.}
\end{cases}
$$

The stage fails if:

- the primary target table is missing,
- the table has no rows for the requested task,
- a requested subject is absent from the primary table,
- `|S_{study1}| < study1.cohort.min_subjects` for LOSO stages.

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

Study 1 itself fixes which feature families are admissible, but the exact mathematical
definitions of each family are inherited unchanged from the shared feature pipeline
documented in `eeg_pipeline/analysis/features/README.md`.

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
against this contract. If the metadata do not prove that the outputs were generated
under these settings, the stage fails.

### 4.3 Deterministic subject-scoped regeneration

For the resolved Study 1 cohort, the `prepare-features` stage:

1. creates the Study 1 feature root if needed,
2. deletes any existing Study 1-owned feature tree for those resolved subjects,
3. reruns the shared `FeaturePipeline` with:
   - `fail_fast = true`,
   - `analysis_mode = "trial_ml_safe"`,
   - `save_canonical_trial_table = false`,
   - the Study 1-pinned provenance settings above.

Standard families and windowed families are run in separate batches. For the windowed
families (`erds`, `bursts`), the stage keeps the canonical Study 1 outputs and removes
redundant window-specific side products afterwards. It also removes macOS AppleDouble
sidecars (`._*`) from the Study 1 feature tree.

### 4.4 Study-owned output location

Prepared features are written under the Study 1 namespace:

```text
<study1_root>/features_trial_ml_safe/sub-<id>/eeg/features/<family>/
```

The feature benchmark reads only from this Study 1-owned feature root and does not fall
back to generic shared feature directories.

## 5. Feature-Based Model Comparison

The feature benchmark is a nested cross-validated model-comparison analysis that
predicts trial-wise fMRI signature expression from Study 1-owned features.

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

The primary partition therefore evaluates two targets under four prespecified band
presets using the same confirmatory feature family (`power`). The exploratory partition
evaluates two targets across the configured exploratory families.

### 5.2 Actual target-loading path in the benchmark lane

The feature benchmark uses the primary target table only to resolve which subjects are
eligible for Study 1. The machine-learning backend then reloads the per-trial fMRI
signature targets subject by subject against the clean EEG events table through the
shared fMRI-signature target loader.

This means the tabular benchmark uses the same target family and alignment rules as
`prepare-targets`, but the targets are re-materialized at matrix-assembly time rather
than read directly from `primary_targets.parquet`. Trials with non-finite reloaded
targets are dropped before matrix assembly.

### 5.3 Design matrix assembly and feature harmonization

Let

$$
X \in \mathbb{R}^{N \times P}
$$

denote the combined trial-by-feature matrix after concatenating subject-specific feature
tables over all included subjects. Study 1 requests
`feature_harmonization = "intersection"`. In the shared ML backend, this means:

1. the load-time matrix preserves the union of available feature columns across subjects,
2. within each outer training fold, a train-only subject intersection mask is computed
   and applied foldwise before fitting.

For an outer training fold with subject labels `g`, the retained EEG feature set is the
set of columns that contain at least one finite value in every training subject. If the
strict per-subject intersection is empty, the shared backend retains any-finite training
columns rather than aborting.

### 5.4 Foldwise preprocessing pipeline

For each model family, preprocessing is fit on the outer training data only. If
`X_{train}` denotes the fold-specific training matrix, the shared preprocessing stack is:

$$
X^{(1)} = \mathrm{ReplaceInfWithNaN}(X_{train})
$$

$$
X^{(2)} = \mathrm{DropAllNaNColumns}(X^{(1)})
$$

$$
X^{(3)} = \mathrm{Impute}_{median}(X^{(2)})
$$

$$
X^{(4)} = \mathrm{VarianceThreshold}_{\tau}(X^{(3)}),
$$

where `tau` is tuned by inner cross-validation. For `elasticnet` and `ridge`, an
additional `StandardScaler` is fitted after the variance-threshold stage. PCA and
deconfounding are present in the shared backend but disabled under the frozen production
configuration. Optional spatial ROI filtering and percentile-based feature selection also
exist in the shared backend, but they are inactive under the frozen Study 1 defaults.

The regression target is wrapped in a `TransformedTargetRegressor` using a Yeo-Johnson
power transform with standardization:

$$
\tilde{y} = \mathrm{YJ}(y).
$$

This target transform is learned on the training data of each estimator fit and inverted
after prediction.

### 5.5 Compared models and hyperparameter grids

Each benchmark run compares three regressors under identical outer folds:

- `elasticnet`,
- `ridge`,
- `rf`.

The frozen shared hyperparameter defaults inherited from the ML stack are:

| Model | Production grid |
| --- | --- |
| ElasticNet | `alpha in [0.001, 0.01, 0.1, 1, 10]`, `l1_ratio in [0.2, 0.5, 0.8]`, variance-threshold grid `[0.0, 0.01, 0.1]` |
| Ridge | `alpha in [0.01, 0.1, 1.0, 10.0, 100.0]`, variance-threshold grid `[0.0, 0.01, 0.1]` |
| Random forest | `n_estimators = 500`, `bootstrap = true`, `max_depth in [5, 10, 20, null]`, `min_samples_split in [2, 5, 10]`, `min_samples_leaf in [1, 2, 4]`, variance-threshold grid `[0.0, 0.01, 0.1]` |

For the linear models, the optimized objectives are:

ElasticNet:

$$
\min_{\beta_0,\beta}
\frac{1}{2n}
\left\|
\tilde{y} - \beta_0 - X\beta
\right\|_2^2
+
\alpha \rho \|\beta\|_1
+
\frac{\alpha(1-\rho)}{2}\|\beta\|_2^2
$$

Ridge:

$$
\min_{\beta_0,\beta}
\frac{1}{2n}
\left\|
\tilde{y} - \beta_0 - X\beta
\right\|_2^2
+
\frac{\alpha}{2}\|\beta\|_2^2.
$$

`rf` is implemented as a bootstrap random-forest regressor under the parameter grid
above.

### 5.6 Nested LOSO validation and summary statistics

All compared models share the same outer subject-level folds:

$$
\mathcal{F}_{outer} = \mathrm{LeaveOneGroupOut}(subject).
$$

When the outer training split contains at least two unique subjects, inner
hyperparameter tuning uses:

$$
\mathcal{F}_{inner} = \mathrm{GroupKFold}(n_{splits} = \min(5, n_{train\_subjects})).
$$

The inner grid search uses a scoring dictionary containing Pearson correlation and
negative mean squared error, and refits the best estimator by maximizing inner-CV
Pearson `r`. If the outer training split contains fewer than two unique subjects, no
grid search is run and the estimator is fit at its default parameterization.

For each held-out fold `f`, the recorded metrics are:

$$
R_f^2 =
1 - \frac{\sum_{i \in f}(y_i - \hat{y}_i)^2}
{\sum_{i \in f}(y_i - \bar{y}_f)^2}
$$

and

$$
\mathrm{MAE}_f = \frac{1}{|f|}\sum_{i \in f}|y_i - \hat{y}_i|.
$$

The summary JSON reports the arithmetic mean and standard deviation of these foldwise
subject-level metrics, plus bootstrap confidence intervals over held-out-subject folds.
The shared default number of bootstrap iterations is:

```text
machine_learning.evaluation.bootstrap_iterations = 1000
```

Under the frozen Study 1 production defaults:

| Parameter | Value |
| --- | --- |
| permutations | `0` |
| inner CV splits | `5` |
| outer jobs | `1` |
| feature harmonization | `intersection` |
| random seed | `project.random_state = 42` |

Because `n_perm = 0` in the frozen production configuration, the default output is
descriptive: it reports held-out-subject performance and bootstrap intervals but does
not compute permutation-derived p-values. If permutations are enabled, the shared backend
also computes subject-paired sign-flip inference for pairwise model differences in
`delta R^2` and `delta MAE`, with Holm correction across the pairwise tests.

### 5.7 Benchmark outputs

Feature-benchmark outputs are written under:

```text
<study1_root>/feature_benchmark/
```

with paths of the form:

```text
primary/<target>/<preset>/
exploratory/<target>/<family>/
```

Each run writes a `model_comparison` directory containing, at minimum:

- `model_comparison.tsv`,
- `metrics/model_comparison_summary.json`,
- included and excluded subject reports,
- reproducibility metadata.

## 6. Deep Regression from Band-Limited EEG Tensors

The `deep-regression` stage implements the second predictive lane.

### 6.1 Tensor construction

For each target and preset, the stage:

1. resolves the Study 1 cohort from the primary target table,
2. loads clean epochs and aligned clean events for each subject,
3. computes the intersection of valid EEG channels across subjects,
4. band-pass filters the epochs separately for each requested band using the shared
   `frequency_bands` mapping,
5. stacks the result into a band tensor.

If `n_trials`, `n_bands`, `n_channels`, and `n_times` denote the pooled dimensions, the
input tensor is:

$$
X \in \mathbb{R}^{n_{trials} \times n_{bands} \times n_{channels} \times n_{times}}.
$$

Unknown band names raise immediately. No deep-regression run is attempted when the
common channel intersection is empty.

The frozen preset mapping is:

| Preset | Bands |
| --- | --- |
| `alpha` | `["alpha"]` |
| `beta` | `["beta"]` |
| `gamma` | `["gamma"]` |
| `alpha_beta_gamma` | `["alpha", "beta", "gamma"]` |

### 6.2 Target alignment in the tensor lane

Target alignment in the deep-regression lane is performed subject by subject against the
Study 1 primary target table using the same two candidate key strategies as above:

- run plus trial index,
- run plus rounded onset and duration.

As in the target-preparation path, duplicate target keys are aggregated by arithmetic
mean. The key family with the larger number of successful matches is retained. If
neither strategy yields matches, or if any aligned target remains non-finite, the stage
fails.

### 6.3 Outer LOSO structure and foldwise normalization

Training uses subject-level `LeaveOneGroupOut`, where the held-out unit is the subject.
For each outer fold:

1. the pooled training trials define `X_train`,
2. the held-out subject defines `X_test`,
3. inputs are standardized using training-fold statistics only,
4. a validation subject subset is carved out of the outer training fold when possible,
5. the target is standardized on the fitting subset,
6. the network is optimized by mean squared error on standardized targets.

If `X_{n,b,c,t}` denotes the outer-fold tensor, the input normalization is:

$$
\tilde{X}_{n,b,c,t} =
\frac{X_{n,b,c,t} - \mu_{b,c}}
{\sigma_{b,c}^{*}},
$$

with

$$
\mu_{b,c} = \mathrm{mean}_{n,t}(X_{train,n,b,c,t}),
\qquad
\sigma_{b,c}^{*} = \max\left(\mathrm{sd}_{n,t}(X_{train,n,b,c,t}), 10^{-6}\right).
$$

The validation split is group-based. If the outer training split contains at least two
unique subjects, the training subjects are shuffled with a fold-specific seed and a
fraction

```text
study1.deep_regression.validation_fraction = 0.2
```

is reserved for validation, with at least one subject and at least one remaining
training subject. If fewer than two unique training subjects are available, no
validation split is created and early stopping is effectively disabled for that fold.
In that case, the model simply runs for the full epoch budget and keeps the final state.

Let `y_fit` denote the target values in the fitting subset after the validation holdout.
The target normalization used for optimization is:

$$
\tilde{y}_n =
\frac{y_n - \mu_y}{\sigma_y^{*}},
$$

where

$$
\mu_y = \mathrm{mean}(y_{fit}),
\qquad
\sigma_y^{*} = \max(\mathrm{sd}(y_{fit}), 10^{-6}).
$$

Predictions are inverted back to the original target scale after inference.

### 6.4 Band-temporal regressor architecture

The implemented model is a compact PyTorch regressor. For a per-trial input tensor

$$
\tilde{X} \in \mathbb{R}^{B \times C \times T},
$$

the network applies:

1. a temporal convolution over time with kernel `(1, k)`,
2. batch normalization and ELU,
3. a spatial convolution with kernel `(n_channels, 1)` collapsing the channel axis,
4. batch normalization and ELU,
5. dropout,
6. adaptive average pooling to `8` temporal bins,
7. a two-layer regression head.

With `F = study1.deep_regression.temporal_filters = 8` and `k` the odd temporal kernel
length constrained by the available trial duration, the feature extractor is:

$$
H^{(1)}_{n,f,c,t}
=
\mathrm{ELU}
\left(
\mathrm{BN}
\left(
\sum_{b,u}
W^{temp}_{f,b,u}\tilde{X}_{n,b,c,t+u}
\right)
\right)
$$

$$
H^{(2)}_{n,f,t}
=
\mathrm{ELU}
\left(
\mathrm{BN}
\left(
\sum_{c}
W^{spat}_{f,c} H^{(1)}_{n,f,c,t}
\right)
\right)
$$

$$
z_n = \mathrm{Pool}_{8}\left(\mathrm{Dropout}(H^{(2)}_n)\right)
$$

and the regression head is:

$$
\hat{y}_n =
W_2
\,
\mathrm{Dropout}
\left(
\mathrm{ELU}
\left(
W_1 \, \mathrm{vec}(z_n) + b_1
\right)
\right)
+
b_2.
$$

The frozen architectural defaults are:

| Parameter | Value |
| --- | --- |
| temporal filters | `8` |
| temporal kernel size | `15` |
| pooled temporal bins | `8` |
| dropout | `0.25` |

### 6.5 Optimization and evaluation

The training objective is mean squared error on standardized targets:

$$
\mathcal{L}(\theta)
=
\frac{1}{m}
\sum_{n=1}^{m}
\left(
\hat{\tilde{y}}_n - \tilde{y}_n
\right)^2.
$$

The frozen optimization defaults are:

| Parameter | Value |
| --- | --- |
| optimizer | `AdamW` |
| epochs | `25` |
| batch size | `32` |
| learning rate | `0.001` |
| weight decay | `0.0001` |
| patience | `5` |
| CUDA | `false` |

The random seed for fold `f` is `project.random_state + f`, applied to NumPy and
PyTorch. GPU execution is used only when `study1.deep_regression.use_cuda = true` and
CUDA is available.

For each held-out subject fold, the stage records:

$$
r_f = \mathrm{corr}(y_f, \hat{y}_f),
$$

$$
R_f^2 =
1 - \frac{\sum_{i \in f}(y_i - \hat{y}_i)^2}
{\sum_{i \in f}(y_i - \bar{y}_f)^2},
$$

and

$$
\mathrm{MAE}_f = \frac{1}{|f|}\sum_{i \in f}|y_i - \hat{y}_i|.
$$

The output summary reports the arithmetic mean of `r_f`, `R_f^2`, and `MAE_f` across
held-out-subject folds. Deep regression requires PyTorch; if PyTorch is unavailable, the
stage raises an explicit import error.

### 6.6 Deep-regression outputs

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

The tabular report contains one row per summary record with:

- `lane`,
- `analysis_partition`,
- `target`,
- `feature_spec`,
- `model`,
- `mean_r2`,
- `mean_mae`,
- `n_folds`,
- `summary_path`.

## 8. Reproducibility, Outputs, and Failure Semantics

### 8.1 Configuration and seed hierarchy

The frozen Study 1 defaults are loaded from:

```text
studies/pain_study/study1/config/study1_config.yaml
```

At runtime the Study 1 config precedence is:

1. explicit `--study1-config`,
2. `PAIN_STUDY_STUDY1_CONFIG`,
3. the packaged Study 1 YAML above.

The shared project random seed defaults to:

```text
project.random_state = 42
```

and is reused by both the tabular benchmark and deep-regression lane.

### 8.2 Output layout

With the default configuration, the core Study 1 layout is:

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

### 8.3 Common hard failures

Common explicit failure conditions include:

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
- no common EEG channels for deep regression,
- non-finite deep-regression targets,
- unavailable PyTorch for deep regression,
- no feature-benchmark or deep-regression summaries for report aggregation.

Study 1 is intended to fail fast on invalid inputs rather than silently fabricate
replacement behavior.

## 9. Manuscript-Complete Reporting Requirements

Sections 1-8 fully specify the computational workflow implemented in this repository.
For a paper Methods section, the remaining required fields are the dataset-resolved items
that are not frozen by code. They should be filled from the pain-study acquisition
records or from the Study 1 outputs listed below.

### 9.1 Items that must come from acquisition or upstream preprocessing records

These are not encoded by `study1/` itself and must be taken from the study acquisition
documentation:

- participant counts, inclusion and exclusion criteria, demographics, and ethics,
- EEG acquisition hardware, reference, channel count, sampling rate, and recording
  environment,
- MRI acquisition parameters and scanner details,
- exact clean-EEG preprocessing steps if the paper intends to describe them in detail,
- exact fMRIPrep and first-level fMRI preprocessing settings beyond the Study 1 target
  contract summarized here,
- exact versions and provenance of the `NPS` and `SIIPS1` map files used in the study.

### 9.2 Items that must be reported from Study 1 outputs

The following paper-ready quantities are dataset-specific but recoverable from the Study 1
artifacts:

| Paper field | Source |
| --- | --- |
| final included Study 1 subjects per task | `targets/primary_targets.parquet` |
| final retained trial counts per subject | `targets/primary_targets.parquet` |
| confirmatory and exploratory benchmark fold counts | `feature_benchmark/*/*/*/model_comparison/metrics/model_comparison_summary.json` |
| held-out subject metrics for each compared model | `feature_benchmark/*/*/*/model_comparison/model_comparison.tsv` |
| included and excluded subjects for each benchmark cell | benchmark subject-selection reports in each `model_comparison` directory |
| deep-regression fold counts and total trial counts | `deep_regression/*/*/summary.json` |
| deep-regression foldwise held-out metrics | `deep_regression/*/*/fold_metrics.tsv` |
| cross-lane summary table for manuscript results assembly | `reports/study1_report.tsv` |

### 9.3 Minimal manuscript checklist for Study 1

When turning this document into a journal article, the final Methods or Supplement should
state at least:

1. how many subjects were requested, how many entered `prepare-targets`, and how many
   remained in each LOSO analysis lane,
2. how many trials per subject survived the final Study 1 target table,
3. which target (`NPS` or `SIIPS1`), feature family, preset, and model family correspond
   to each reported result,
4. whether the report concerns the feature-benchmark lane, the deep-regression lane, or
   both,
5. whether any permutation-based model-difference inference was enabled beyond the frozen
   descriptive defaults,
6. the exact signature-map files and software versions used for the final run.

Without those dataset-resolved fields, the repository still defines the computational
method, but not the full study-specific reporting layer expected in a paper.
