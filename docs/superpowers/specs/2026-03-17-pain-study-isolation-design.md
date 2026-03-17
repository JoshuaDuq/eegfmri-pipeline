# Pain Study Isolation Design

## Goal

Move all pain-study-specific code, configuration, and standalone scripts out of
`eeg_pipeline/` and `fmri_pipeline/` into a dedicated top-level package at
`studies/pain_study/`, without changing runtime behavior, outputs, or side
effects.

## Current State

Pain-study ownership is currently split across multiple locations:

- `eeg_pipeline/analysis/paradigms/pain/`
- `eeg_pipeline/pipelines/paradigms/pain/`
- `eeg_pipeline/utils/config/paradigms/pain/`
- `eeg_pipeline/cli/commands/coupling.py`
- `paradigm-specific-scripts/`

This mixes study-specific logic into core EEG pipeline namespaces and leaves
study assets distributed across unrelated package roots.

## Target State

All pain-study-specific assets live under one isolated package root:

```text
studies/pain_study/
  __init__.py
  analysis/
  cli/
  config/
  pipelines/
  scripts/
```

Core packages remain responsible only for generic infrastructure:

- `eeg_pipeline/` keeps shared CLI infrastructure, shared pipeline base
  classes, generic analysis code, and generic utilities.
- `fmri_pipeline/` remains paradigm-agnostic.
- `studies/pain_study/` owns all pain-study-specific code and assets.

## Design Decisions

### Single Study Root

Use a single top-level root, `studies/pain_study/`, instead of splitting the
study into multiple top-level directories.

Reasoning:

- The pain-study workflow is multimodal and tightly coupled.
- A single root makes ownership obvious and searchable.
- It avoids scattering study-specific code across multiple unrelated roots.

### No Compatibility Shims

Old pain-study import paths will be removed instead of preserved as wrappers or
aliases.

Reasoning:

- Repository instructions explicitly forbid fallbacks and backward
  compatibility.
- Hidden import shims would preserve the same structural problem under a new
  directory layout.
- Broken references should surface immediately and be updated explicitly.

### Keep Generic CLI Registration In Core

The core CLI may continue to register a `coupling` command, but the
implementation for that command moves into `studies.pain_study.cli`.

Reasoning:

- Command registration is generic infrastructure.
- Command behavior and defaults for the pain study are paradigm-specific and
  belong to the study package.

## File Ownership Changes

### Move Analysis Modules

Move all files from:

- `eeg_pipeline/analysis/paradigms/pain/`

To:

- `studies/pain_study/analysis/`

This includes:

- `eeg_bold_coupling.py`
- `eeg_bold_fmri_qc.py`
- `eeg_bold_nuisance.py`
- `eeg_bold_permutation.py`
- `eeg_bold_residualized.py`
- `eeg_bold_roi_builder.py`
- `eeg_bold_sensitivity.py`
- `eeg_bold_signatures.py`
- `eeg_bold_source.py`
- `eeg_bold_statistics.py`
- `eeg_bold_nlme_backend.R`

Internal imports will be rewritten from:

- `eeg_pipeline.analysis.paradigms.pain...`

To:

- `studies.pain_study.analysis...`

### Move Pipeline Wrappers

Move all files from:

- `eeg_pipeline/pipelines/paradigms/pain/`

To:

- `studies/pain_study/pipelines/`

The coupling pipeline wrapper remains thin and continues to depend on generic
`PipelineBase` from `eeg_pipeline.pipelines.base`.

### Move Config Loaders And Study Assets

Move all files from:

- `eeg_pipeline/utils/config/paradigms/pain/`

To:

- `studies/pain_study/config/`

This includes:

- YAML defaults
- smoke-test YAMLs
- the runtime config loader
- the ROI library assets

Imports will be rewritten from:

- `eeg_pipeline.utils.config.paradigms.pain...`

To:

- `studies.pain_study.config...`

### Move Script Entrypoints

Move all files from:

- `paradigm-specific-scripts/`

To:

- `studies/pain_study/scripts/`

This includes:

- `run_paradigm_specific.py`
- `eeg_raw_to_bids.py`
- `fmri_raw_to_bids.py`
- `merge_psychopy.py`
- `fix_fmri_bids_outputs.py`
- README and config override YAMLs

User-facing examples and repository guards will be updated to the new script
path.

### Move Pain-Specific CLI Logic

Move:

- `eeg_pipeline/cli/commands/coupling.py`

To:

- `studies/pain_study/cli/coupling.py`

Then update:

- `eeg_pipeline/cli/commands/__init__.py`

So the central registry imports `setup_coupling` and `run_coupling` from
`studies.pain_study.cli.coupling`.

This keeps the command available through the existing CLI while transferring
ownership of all pain-study behavior out of `eeg_pipeline`.

## Runtime Behavior

Behavior is preserved by keeping these invariants unchanged:

- the CLI command name remains `coupling`
- the parser arguments remain unchanged
- config keys remain unchanged
- pipeline execution order remains unchanged
- subject-level and group-level outputs remain unchanged
- standalone pain-study scripts keep the same CLI flags and semantics

Only import paths and filesystem locations change.

## Deletions

After all imports and tests are updated, remove these old study-specific
locations entirely:

- `eeg_pipeline/analysis/paradigms/pain/`
- `eeg_pipeline/pipelines/paradigms/pain/`
- `eeg_pipeline/utils/config/paradigms/pain/`
- `paradigm-specific-scripts/`
- `eeg_pipeline/cli/commands/coupling.py`

If the now-empty parent `paradigms` packages under analysis or pipelines serve
no remaining purpose, remove them as well.

## Required Reference Updates

The following references must be updated as part of the move:

- CLI registry imports in `eeg_pipeline/cli/commands/__init__.py`
- test imports and patch targets in
  `tests/pipelines/test_pipeline_coupling_robustness.py`
- repository hygiene guidance in
  `tests/utils/test_repo_hygiene_guards.py`
- all README usage examples pointing at `paradigm-specific-scripts/...`
- any remaining internal imports referencing old pain-study package paths

## Error Handling

No fallback path resolution or import aliasing will be added.

If any caller still references old locations after the move, that failure should
surface immediately and be fixed directly.

## Verification Plan

Verification should be narrow and explicit:

1. Run the coupling robustness tests.
2. Run the repository hygiene tests.
3. Run targeted import-level verification for the coupling CLI command.
4. Search the repository for remaining references to:
   - `eeg_pipeline.analysis.paradigms.pain`
   - `eeg_pipeline.pipelines.paradigms.pain`
   - `eeg_pipeline.utils.config.paradigms.pain`
   - `paradigm-specific-scripts`

The migration is complete only when those legacy references are gone from
tracked source files.
