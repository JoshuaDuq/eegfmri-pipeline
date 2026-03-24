# Study 1: Methods for Trial-Wise EEG Prediction of fMRI Pain-Signature Expression

This document describes Study 1 as an analysis methods section rather than as a generic runbook.
The description is implementation-grounded: every method summarized here reflects the current
behavior of the Study 1 stack in `studies/pain_study/study1/`.

Study 1 predicts trial-wise fMRI pain-signature expression from EEG. The implemented workflow has
one shared target-preparation stage and two downstream analysis lanes:

- a feature-based machine-learning lane that uses a Study 1-owned `trial_ml_safe` EEG feature
  store,
- a deep-regression lane that learns directly from band-limited EEG trial tensors.

Both lanes use the same shared trial-wise target table and both are summarized by a final report
aggregation step.

The production Study 1 configuration is loaded automatically from:

```text
studies/pain_study/study1/config/study1_config.yaml
```

unless a replacement is supplied through `--study1-config` or the
`PAIN_STUDY_STUDY1_CONFIG` environment variable.

## 1. Analytical Aim

The primary aim of Study 1 is to predict trial-wise expression of two established pain-related
fMRI signatures from EEG:

- `NPS`
- `SIIPS1`

The implementation treats these as continuous trial-level targets. The workflow first constructs a
shared multimodal target table with one row per retained trial, then evaluates two predictive
approaches:

1. feature-based regression using Study 1-owned tabular EEG features,
2. deep regression using band-limited EEG trial tensors.

The tabular lane is designed to test whether curated EEG feature families contain predictive signal
for trial-wise pain-signature expression. The deep-regression lane is designed to test whether a
less engineered band-temporal representation carries additional predictive structure.

## 2. Data Requirements

### 2.1 Required Roots and Derived Inputs

The workflow depends on three configured roots:

| Root | Requirement |
| --- | --- |
| EEG BIDS root | `paths.bids_root` must resolve when EEG BIDS access is needed |
| fMRI BIDS root | `paths.bids_fmri_root` must resolve for target preparation |
| derivatives root | `paths.deriv_root` must resolve and is the base for all Study 1 outputs |

Before Study 1 can run successfully, the following upstream products must already exist:

- clean EEG events for each subject/task,
- clean EEG epochs loadable through the shared EEG utilities,
- fMRI inputs compatible with trial-wise signature extraction,
- valid signature-map configuration containing both required primary signatures.

### 2.2 Clean EEG Event Contract

The clean EEG event table is a hard dependency for target preparation and deep-regression target
alignment. It must contain:

- `onset`
- `duration`
- a usable run or block column:
  - `block`
  - `run_id`
  - `run`
  - `session`

The preferred trial-indexing columns are:

- `trial_number`
- `trial_index`

If those are absent, some downstream logic can still align by run plus onset/duration, but the
event table is expected to be a clean, trial-resolved table rather than a loosely structured raw
log.

### 2.3 Clean EEG Epoch Contract

`prepare-features` and `deep-regression` both rely on clean EEG epochs that can be loaded with
strict alignment to the clean event table. If clean epochs or the aligned clean event table are
missing, those stages stop with explicit errors.

### 2.4 fMRI Signature Inputs

Study 1 target preparation requires:

- fMRI inputs in MNI space,
- valid `paths.signature_maps`,
- both `NPS` and `SIIPS1` available in the configured signature-map set.

The current Study 1 target specification uses:

| Parameter | Value |
| --- | --- |
| input source | `fmriprep` |
| fMRIPrep space | `MNI152NLin2009cAsym` |
| extraction method | `lss` |
| metric | `dot` |
| normalization | `none` |
| contrast name | `pain_vs_nonpain` |

### 2.5 Subject-Count Contract

Study 1 uses different subject-count rules for different stages:

- `prepare-targets` can run on a single subject.
- `prepare-features`, `feature-benchmark`, and `deep-regression` resolve their working cohort
  from the prepared Study 1 target table and require at least
  `study1.cohort.min_subjects` subjects.

The default is:

```yaml
study1:
  cohort:
    min_subjects: 2
```

This minimum exists because the downstream analyses are subject-level leave-one-subject-out
analyses.

## 3. Fixed Study 1 Specification

### 3.1 Study 1 Output Namespace

Study 1 writes its group-level products into a study-owned namespace under the EEG derivatives
root. With the default configuration:

```text
<paths.deriv_root>/group/multimodal/study1/
```

This namespace contains:

- the shared primary target table,
- the Study 1-owned prepared feature store,
- feature-benchmark outputs,
- deep-regression outputs,
- the aggregated report.

The namespace is controlled by:

```yaml
study1:
  outputs:
    root_name: study1
```

### 3.2 Primary Target Specification

The target definition is fixed by `study1.targets`. The production defaults are:

| Parameter | Value |
| --- | --- |
| signature names | `["NPS", "SIIPS1"]` |
| method | `lss` |
| metric | `dot` |
| normalization | `none` |
| round decimals | `3` |
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
| confounds strategy | `auto` |
| LSS other regressors | `all` |

These settings define the current production target contract. Study 1 does not accept arbitrary
signature sets for the main analysis: the required primary signatures are exactly `NPS` and
`SIIPS1`, in that order.

### 3.3 EEG Feature Specification

The feature-based lane uses a Study 1-owned `trial_ml_safe` feature store. The feature contract is
split into one confirmatory family and several exploratory families.

The confirmatory family is:

- `power`

The supported exploratory families are:

- `spectral`
- `aperiodic`
- `erds`
- `ratios`
- `asymmetry`
- `complexity`
- `bursts`

The default exploratory set is all supported exploratory families.

Two families are treated as explicit windowed analyses:

- `erds`
- `bursts`

Those families use:

- `time_frequency_analysis.baseline_window`
- `time_frequency_analysis.active_window`

and both windows must be finite two-element ranges with `start < end`.

### 3.4 Pinned Study 1 Feature Provenance

Study 1 does not accept generic feature outputs as equivalent. During feature preparation it pins a
specific provenance contract:

| Setting | Required value |
| --- | --- |
| `feature_engineering.analysis_mode` | `trial_ml_safe` |
| `feature_engineering.power.subtract_evoked` | `false` |
| `feature_engineering.precomputed.subtract_evoked` | `false` |
| `feature_engineering.aperiodic.subtract_evoked` | `false` |
| `feature_engineering.bands.use_iaf` | `false` |
| `feature_engineering.bursts.threshold_reference` | `trial` |

The metadata written beside each prepared feature family are validated after extraction. If the
metadata do not prove that the outputs were generated under this contract, the stage fails.

### 3.5 Feature-Benchmark Specification

The feature benchmark is a nested cross-validated model-comparison analysis that predicts
trial-wise fMRI signature expression from Study 1-owned features.

The confirmatory band presets are fixed as:

| Preset | Bands |
| --- | --- |
| `alpha` | `["alpha"]` |
| `beta` | `["beta"]` |
| `gamma` | `["gamma"]` |
| `alpha_beta_gamma` | `["alpha", "beta", "gamma"]` |

The confirmatory benchmark therefore evaluates:

- both targets (`NPS`, `SIIPS1`),
- the `power` family,
- each of the four fixed band presets.

The exploratory benchmark evaluates:

- both targets,
- each configured exploratory family,
- no preset-specific band restriction.

Each benchmark run compares three models under the same outer subject-level folds:

- `elasticnet`
- `ridge`
- `rf`

The production benchmark defaults are:

| Parameter | Value |
| --- | --- |
| permutations | `0` |
| inner CV splits | `5` |
| outer jobs | `1` |
| feature harmonization | `intersection` |

### 3.6 Deep-Regression Specification

The deep-regression lane uses a subject-level LOSO design over band-limited EEG trial tensors.

The default preset mapping is:

| Preset | Bands |
| --- | --- |
| `alpha` | `["alpha"]` |
| `beta` | `["beta"]` |
| `gamma` | `["gamma"]` |
| `alpha_beta_gamma` | `["alpha", "beta", "gamma"]` |

The current architecture is a compact band-temporal regressor with:

- a temporal convolution,
- a spatial convolution across channels,
- batch normalization,
- ELU nonlinearities,
- dropout,
- adaptive average pooling,
- a small regression head.

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

## 4. Upstream Processing Before Study 1

### 4.1 EEG Preprocessing

Before Study 1, EEG preprocessing must already have produced:

- clean subject/task epochs,
- an aligned clean event table,
- a retained-trial structure suitable for `trial_ml_safe` analysis.

The Study 1 stack does not internally perform generic EEG preprocessing. It assumes that the clean
EEG derivatives already exist and can be loaded.

### 4.2 fMRI Preparation

The target-preparation stage invokes shared trial-wise signature extraction machinery, but it still
assumes that the necessary fMRI inputs and signature maps are already available. MNI-space support
is a hard requirement for Study 1 target preparation.

### 4.3 Stage Dependency Structure

The implemented dependency structure is:

```text
prepare-targets -> prepare-features -> feature-benchmark
prepare-targets -> deep-regression
feature-benchmark and/or deep-regression -> report
```

This matters scientifically and operationally:

- the primary target table is the shared cohort contract for both analysis lanes,
- the feature-based lane requires a Study 1-specific prepared feature store,
- the deep-regression lane bypasses that feature store and works directly from clean EEG epochs.

## 5. Shared Trial-Target Construction

The `prepare-targets` stage constructs the primary multimodal target table.

### 5.1 Preflight Validation

Before writing outputs, the stage validates that:

- `study1.targets.names` is exactly `["NPS", "SIIPS1"]`,
- the configured signature-map set includes both required signatures,
- the configured fMRIPrep space contains `MNI`,
- clean EEG events exist and are non-empty.

### 5.2 Subject-Level Trial-Signature Extraction

For each requested subject, Study 1 builds a `TrialSignatureExtractionConfig` from
`study1.targets.*` and runs the shared trial-signature extraction routine. The current production
workflow therefore uses an LSS-based trial-signature extraction strategy restricted by the Study 1
contrast and trial-scope settings.

### 5.3 Alignment Between EEG Trials and fMRI Targets

After trial-signature extraction, Study 1 loads the clean EEG event table and aligns `NPS` and
`SIIPS1` back onto EEG trials. The target loader attempts two keying strategies:

1. run plus trial number or trial index,
2. run plus onset and duration, rounded to `study1.targets.round_decimals`.

The alignment mode with the larger number of matches is used. If neither key yields any successful
matches, the stage fails.

### 5.4 Primary Target Table

The output is a wide table with one row per retained trial and the following required columns:

- `subject_id`
- `task`
- `block`
- `trial_index`
- `onset`
- `duration`
- `NPS`
- `SIIPS1`

Finite values for both `NPS` and `SIIPS1` are required for every retained trial. Non-finite values
raise immediately.

### 5.5 Downstream Cohort Resolution

The primary target table is not only an output artifact. It is also the cohort contract for the
remaining Study 1 stages. `prepare-features`, `feature-benchmark`, and `deep-regression` all
resolve their subject sets from this table and reject:

- tasks with no matching rows,
- requested subjects absent from the table,
- resolved cohorts smaller than `study1.cohort.min_subjects`.

## 6. Study 1 EEG Feature Preparation

The `prepare-features` stage regenerates the Study 1-owned EEG feature store under the study
namespace rather than under the shared subject feature namespace.

### 6.1 Study-Owned Output Location

Prepared features are written to:

```text
<study1_root>/features_trial_ml_safe/sub-<id>/eeg/features/<family>/
```

This isolation is deliberate. The feature-based lane reads only from the Study 1-owned feature
root and does not fall back to generic shared feature directories.

### 6.2 Deterministic Subject-Scoped Regeneration

For the resolved Study 1 cohort, the stage:

1. creates the Study 1 feature root if needed,
2. removes any existing Study 1-owned feature outputs for those resolved subjects,
3. reruns the shared `FeaturePipeline` under the fixed Study 1 provenance contract.

The deletion is limited to the Study 1 feature tree for the resolved subjects. The stage does not
touch unrelated shared derivatives.

### 6.3 Standard and Windowed Feature Families

Feature preparation is split into two batches:

- standard families, which are run without explicit baseline/active ranges,
- windowed families (`erds`, `bursts`), which are run with named baseline and active ranges.

For the windowed families, the canonical merged Study 1 outputs are retained and redundant
window-specific side products are removed after extraction.

### 6.4 Metadata Validation and Cleanup

After extraction, Study 1 verifies that every configured family has:

- a prepared feature table,
- a metadata file,
- metadata consistent with the Study 1 provenance contract.

The stage also removes:

- redundant window-specific artifacts for windowed families,
- macOS AppleDouble sidecars (`._*`) from the Study 1 feature tree.

If any configured family is missing or recorded under the wrong provenance settings, the stage
fails rather than silently accepting partial or incompatible inputs.

## 7. Feature-Based Model Comparison

The `feature-benchmark` stage runs the tabular predictive analysis.

### 7.1 Confirmatory and Exploratory Partitions

The feature benchmark is divided into two analysis partitions:

- `primary`
- `exploratory`

The `primary` partition evaluates the fixed `power` confirmatory family under the four production
band presets. The `exploratory` partition evaluates each configured exploratory family as a
family-level benchmark.

### 7.2 Target Configuration

For each target (`NPS`, `SIIPS1`), Study 1 builds a target-specific ML configuration with:

- regression target set to `fmri_signature`,
- `trial_ml_safe` required,
- `trial_ml_safe` analysis mode declared,
- Study 1 target method, contrast, metric, normalization, and rounding propagated into the ML
  configuration.

### 7.3 Model-Comparison Procedure

Each benchmark run uses the shared model-comparison machinery. All models share the same outer
leave-one-subject-out folds. When the training fold contains at least two unique subjects, the
inner hyperparameter search is also group-aware.

The compared models are:

- `elasticnet`
- `ridge`
- `rf`

If permutations are enabled, the summary also includes paired sign-flip inference and Holm
correction across pairwise model-comparison tests.

### 7.4 Feature Harmonization

The production feature-harmonization mode is `intersection`. This means the benchmark uses the
shared ML harmonization logic configured for Study 1 rather than ad hoc family-specific behavior.

### 7.5 Output Structure

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

- fold-wise comparison tables,
- `model_comparison_summary.json`,
- included and excluded subject reports,
- reproducibility metadata.

## 8. Deep Regression from Band-Limited EEG Tensors

The `deep-regression` stage implements the second predictive lane.

### 8.1 Tensor Construction

For each target and preset, the stage:

1. resolves the Study 1 cohort from the primary target table,
2. loads clean epochs and aligned clean events for each subject,
3. intersects EEG channels across subjects and retains only the common channel set,
4. filters the EEG into the requested frequency bands,
5. stacks the resulting data into tensors of shape:
   - `(n_trials, n_bands, n_channels, n_times)`

Band names are resolved through the shared EEG band definitions. Unknown bands raise immediately.

### 8.2 Target Alignment

Target alignment in the deep-regression lane is performed subject by subject against the Study 1
primary target table. The implemented alignment again compares:

1. run plus trial index,
2. run plus onset and duration.

If neither alignment path yields matches, the stage fails. All aligned targets must be finite.

### 8.3 Subject-Level LOSO Training

The training loop uses `LeaveOneGroupOut`, where the held-out unit is subject. Within each outer
fold:

- inputs are standardized using training data only,
- targets are standardized using training data only and inverted after prediction,
- a group-based validation split is carved out of the training subjects,
- early stopping is applied when validation loss stops improving.

### 8.4 Device and Dependency Contract

Deep regression requires PyTorch. If PyTorch is unavailable, the stage raises an explicit import
error. GPU execution is optional and only used when:

- `study1.deep_regression.use_cuda = true`,
- CUDA is actually available.

### 8.5 Output Structure

Deep-regression outputs are written under:

```text
<study1_root>/deep_regression/<target>/<preset>/
```

Each run writes:

- `predictions.tsv`
- `predictions.parquet`
- `fold_metrics.tsv`
- `summary.json`

The prediction files contain trial metadata plus `y_true` and `y_pred`. The summary file records
the preset, target, band set, and average fold-wise performance.

## 9. Report Aggregation

The `report` stage aggregates existing Study 1 outputs into one compact study-level report.

### 9.1 Inputs

The aggregator scans the Study 1 output root for:

- feature-benchmark summaries located at
  `feature_benchmark/*/*/*/model_comparison/metrics/model_comparison_summary.json`,
- deep-regression summaries located at
  `deep_regression/*/*/summary.json`.

### 9.2 Aggregation Logic

The stage can aggregate:

- feature-benchmark outputs only,
- deep-regression outputs only,
- both analysis lanes together.

It fails only when neither lane has produced any summary files.

### 9.3 Subject-Selection Caveat

The top-level `signature-prediction` command still requires subject selection at the CLI layer, but
the `report` stage itself does not subset already-written outputs by the passed subject list. It
summarizes whatever valid Study 1 results are already present under the Study 1 output root.

### 9.4 Report Outputs

The report stage writes:

- `study1_report.tsv`
- `study1_report.parquet`
- `study1_report.json`

The tabular report contains one row per summary record with:

- `lane`
- `analysis_partition`
- `target`
- `feature_spec`
- `model`
- `mean_r2`
- `mean_mae`
- `n_folds`
- `summary_path`

The JSON summary records:

- `task`
- `n_records`
- `lanes`
- `targets`

## 10. Outputs and Reproducibility

### 10.1 Study 1 Output Root

With the default configuration, the Study 1 tree is:

```text
<paths.deriv_root>/group/multimodal/study1/
```

### 10.2 Core Output Layout

The expected top-level layout is:

```text
study1/
  targets/
    primary_targets.parquet
    primary_targets.tsv
  features_trial_ml_safe/
    sub-<id>/
      eeg/
        features/
          power/
          spectral/
          aperiodic/
          erds/
          ratios/
          asymmetry/
          complexity/
          bursts/
  feature_benchmark/
  deep_regression/
  reports/
    study1_report.tsv
    study1_report.parquet
    study1_report.json
```

### 10.3 Reproducibility-Relevant Artifacts

The feature-benchmark lane writes standard ML reproducibility metadata into each comparison
directory, including:

- included and excluded subject reports,
- a reproducibility-information JSON file,
- the benchmark summary JSON.

The report aggregation stage then creates a compact cross-lane summary for the study as a whole.

## 11. Execution Appendix

This appendix keeps the operational surface area compact while the main body remains methods-first.

### 11.1 Public CLI

Study 1 is exposed through:

```bash
python -m eeg_pipeline.cli.main signature-prediction <mode> [options]
```

Supported modes are:

- `prepare-targets`
- `prepare-features`
- `feature-benchmark`
- `deep-regression`
- `report`

### 11.2 Production Example

```bash
cd /Users/joduq24/Desktop/EEG_fMRI_Pipeline
source .venv/bin/activate

EEG_BIDS_ROOT=/Volumes/KINGSTON/EEG_fMRI_data/bids_output/eeg
FMRI_BIDS_ROOT=/Volumes/KINGSTON/EEG_fMRI_data/bids_output/fmri
DERIV_ROOT=/Volumes/KINGSTON/EEG_fMRI_data/derivatives
NPS_MAP=/Volumes/KINGSTON/EEG_fMRI_data/external/NPS/weights_NSF_grouppred_cvpcr.nii.gz
SIIPS1_MAP=/Volumes/KINGSTON/EEG_fMRI_data/external/SIIPS1/nonnoc_v11_4_137subjmap_weighted_mean.nii.gz
SIGNATURE_MAPS='[{"name":"NPS","path":"'"$NPS_MAP"'"},{"name":"SIIPS1","path":"'"$SIIPS1_MAP"'"}]'

python -m eeg_pipeline.cli.main signature-prediction prepare-targets \
  --subject 0000 \
  --subject 0001 \
  --task thermalactive \
  --bids-root "$EEG_BIDS_ROOT" \
  --bids-fmri-root "$FMRI_BIDS_ROOT" \
  --deriv-root "$DERIV_ROOT" \
  --set "paths.signature_maps=$SIGNATURE_MAPS"

python -m eeg_pipeline.cli.main signature-prediction prepare-features \
  --subject 0000 \
  --subject 0001 \
  --task thermalactive \
  --bids-root "$EEG_BIDS_ROOT" \
  --bids-fmri-root "$FMRI_BIDS_ROOT" \
  --deriv-root "$DERIV_ROOT"

python -m eeg_pipeline.cli.main signature-prediction feature-benchmark \
  --subject 0000 \
  --subject 0001 \
  --task thermalactive \
  --bids-root "$EEG_BIDS_ROOT" \
  --bids-fmri-root "$FMRI_BIDS_ROOT" \
  --deriv-root "$DERIV_ROOT"

python -m eeg_pipeline.cli.main signature-prediction deep-regression \
  --subject 0000 \
  --subject 0001 \
  --task thermalactive \
  --bids-root "$EEG_BIDS_ROOT" \
  --bids-fmri-root "$FMRI_BIDS_ROOT" \
  --deriv-root "$DERIV_ROOT"

python -m eeg_pipeline.cli.main signature-prediction report \
  --subject 0000 \
  --subject 0001 \
  --task thermalactive \
  --bids-root "$EEG_BIDS_ROOT" \
  --bids-fmri-root "$FMRI_BIDS_ROOT" \
  --deriv-root "$DERIV_ROOT"
```

### 11.3 Smoke Configuration

The repository also ships:

```text
studies/pain_study/study1/config/study1_smoketest.yaml
```

This is useful for fast validation but is not the same scientific specification as the production
Study 1 configuration.

### 11.4 Runtime Overrides

Study 1 supports:

- `--study1-config /path/to/file.yaml`
- `PAIN_STUDY_STUDY1_CONFIG=/path/to/file.yaml`
- repeatable `--set KEY=VALUE`

The standard shared path overrides are also available:

- `--bids-root`
- `--bids-fmri-root`
- `--bids-rest-root`
- `--deriv-root`
- `--deriv-rest-root`

## 12. Failure Conditions That Commonly Stop the Workflow

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
