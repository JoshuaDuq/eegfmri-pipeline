# Study 1 Feature Preparation Design

> **For agentic workers:** REQUIRED: Use superpowers:executing-plans to implement this design. Steps should stay bounded, fail fast, and avoid backward-compatibility shims.

**Goal:** Add an explicit `prepare-features` stage to Study 1 that always regenerates Study 1-owned `trial_ml_safe` primary EEG power features into an isolated study namespace, then make `feature-benchmark` consume only those study-owned artifacts.

**Architecture:** Study 1 becomes a five-stage flow: `prepare-targets`, `prepare-features`, `feature-benchmark`, `deep-regression`, `report`. `prepare-features` loads clean EEG epochs from the normal preprocessing derivatives, but writes primary `trial_ml_safe` feature artifacts into a Study 1-owned root under `group/multimodal/study1/`. `feature-benchmark` reads only from that Study 1-owned feature store and fails immediately if those artifacts are missing or invalid.

**Tech Stack:** Python, existing `FeaturePipeline`, Study 1 CLI wiring, existing ML orchestration, pytest.

---

## Context

Study 1 target preparation now works on the KINGSTON dataset and writes a valid shared target table under:

- `group/multimodal/study1/targets/primary_targets.{parquet,tsv}`

The current feature benchmark cannot run on the real dataset because the shared subject feature tables were previously extracted with:

- `analysis_mode=group_stats`

Those artifacts live in the shared subject feature namespace:

- `sub-<id>/eeg/features/...`

and are correctly rejected by the ML loader as leakage-prone for cross-validated ML.

Study 1 must not overwrite those shared artifacts. It needs its own isolated `trial_ml_safe` feature store and an explicit preparation step that makes the confirmatory feature lane reproducible and thesis-clean.

## Scope

### In scope

- Add a new Study 1 CLI mode: `prepare-features`
- Add a Study 1-owned primary feature root under:
  - `<deriv_root>/group/multimodal/study1/features_trial_ml_safe/sub-<id>/...`
- Make `prepare-features` always recompute Study 1 primary `trial_ml_safe` power features
- Keep prepared Study 1 features isolated from the shared subject feature namespace
- Make `feature-benchmark` read only from the Study 1-owned feature store
- Add tests for path isolation, CLI wiring, missing prepared features, and provenance checks

### Out of scope

- Automatic exploratory feature-family preparation
- Replacing the existing shared feature pipeline
- Changing the deep-regression lane storage model
- Preserving or mixing old shared `group_stats` feature artifacts into Study 1

## Approved Design

### 1. Add an explicit `prepare-features` stage

Study 1 will expose a new CLI mode:

- `prepare-features`

This stage will be required before running `feature-benchmark`.

Study 1 stage order becomes:

1. `prepare-targets`
2. `prepare-features`
3. `feature-benchmark`
4. `deep-regression`
5. `report`

`feature-benchmark` will no longer be responsible for recomputing features internally.

### 2. Use a Study 1-owned feature namespace

Prepared Study 1 features will live under the Study 1 output root, not under shared subject derivatives.

Target layout:

```text
<deriv_root>/group/multimodal/study1/
  targets/
  features_trial_ml_safe/
    sub-0000/
      eeg/
        features/
          power/
          metadata/
    sub-0001/
      ...
  feature_benchmark/
  deep_regression/
  reports/
```

This preserves a familiar `sub-<id>/eeg/features/...` internal layout while isolating Study 1 ownership under one study-specific root.

### 3. `prepare-features` is primary-only

`prepare-features` will generate only the confirmatory Study 1 feature set:

- feature family: `power`
- provenance mode: `trial_ml_safe`
- enough content to support:
  - `alpha`
  - `beta`
  - `gamma`
  - `alpha+beta+gamma`

Exploratory feature families will not be bundled into this stage. If exploratory preparation is needed later, it should be introduced as a separate explicit stage rather than hidden inside the primary workflow.

### 4. Inputs and outputs stay separated

`prepare-features` must load EEG inputs from the standard preprocessing derivatives:

- clean epochs
- clean events

but write its feature outputs into the Study 1-owned root.

This requires a narrow output-root override in the feature pipeline. The feature pipeline must not reinterpret the Study 1 feature root as its input preprocessing root.

Likewise, `feature-benchmark` needs a narrow feature-input-root override when loading prepared feature tables, while preserving the normal derivatives root for:

- target alignment
- fMRI signature loading
- any non-feature dependencies

### 5. No fallback to shared subject features

Study 1 must not read from:

- `sub-<id>/eeg/features/...`

when the Study 1-owned feature store is missing.

If prepared Study 1 features do not exist, `feature-benchmark` must fail immediately with an explicit error telling the user to run `prepare-features`.

This is intentional. Silent fallback would mix incompatible provenance and violate the study’s isolation requirement.

## Components

### Study 1 modules

Add or update:

- `studies/pain_study/cli/signature_prediction.py`
  - add `prepare-features` parser mode
- `studies/pain_study/study1/runner.py`
  - dispatch `prepare-features`
- `studies/pain_study/study1/prepare_features.py`
  - new stage orchestration for Study 1-owned feature generation
- `studies/pain_study/study1/cohort.py`
  - add Study 1 feature-root helpers
- `studies/pain_study/study1/README.md`
  - document the new stage and feature layout

### Generic infrastructure

Make only the narrow generic changes needed:

- `eeg_pipeline/pipelines/features.py`
  - add a feature-output-root override for writing feature artifacts outside the standard subject derivative tree
- `eeg_pipeline/utils/data/machine_learning.py`
  - add a feature-input-root override so ML can load feature tables from the Study 1-owned store

These overrides must be explicit and opt-in. Default non-Study-1 behavior must remain unchanged.

## Data Flow

### `prepare-targets`

- unchanged
- writes the shared retained-trial target table

### `prepare-features`

- resolves the Study 1 eligible subjects
- loads clean epochs/events from normal preprocessing derivatives
- runs `FeaturePipeline` in `trial_ml_safe`
- writes primary power features into the Study 1-owned feature store
- overwrites prior Study 1-owned prepared features for those subjects deterministically

### `feature-benchmark`

- validates that the primary target table exists
- validates that Study 1-owned features exist
- validates `trial_ml_safe` provenance from the Study 1-owned feature metadata
- runs the ML comparison using only the Study 1-owned prepared features

### `deep-regression`

- unchanged in data source
- continues to read clean epochs directly from preprocessing derivatives

### `report`

- unchanged in role
- aggregates feature and deep results under the Study 1 root

## Error Handling

Fail fast in these cases:

- `prepare-features` is called before clean epochs/events exist
- `feature-benchmark` is called before `prepare-targets`
- `feature-benchmark` is called before `prepare-features`
- prepared Study 1 feature tables are missing
- prepared Study 1 feature metadata does not show `trial_ml_safe`
- Study 1 feature loader is pointed at the wrong root

Do not:

- silently fall back to shared subject features
- silently skip subjects with invalid Study 1 feature artifacts
- auto-generate exploratory features during the primary path

## Testing Plan

Add coverage for:

- CLI registration of `prepare-features`
- Study 1 runner dispatch for `prepare-features`
- Study 1 feature-root path helpers
- feature pipeline output-root override behavior
- ML feature-input-root override behavior
- `feature-benchmark` hard failure when Study 1-owned prepared features are missing
- `feature-benchmark` rejection of non-`trial_ml_safe` Study 1 feature artifacts
- `prepare-features` writing into the Study 1 root without touching shared subject features
- regression tests proving default non-Study-1 paths still behave unchanged

## Acceptance Criteria

- Study 1 exposes a dedicated `prepare-features` stage
- `prepare-features` always regenerates primary Study 1-owned `trial_ml_safe` power features
- Study 1-owned prepared features are written under:
  - `<deriv_root>/group/multimodal/study1/features_trial_ml_safe/...`
- shared subject feature folders remain untouched
- `feature-benchmark` consumes only Study 1-owned prepared features
- missing or invalid prepared Study 1 features cause explicit errors
- exploratory feature preparation is not included in the primary Study 1 path
