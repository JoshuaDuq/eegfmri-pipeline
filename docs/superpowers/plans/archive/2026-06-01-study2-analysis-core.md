# Study 2 Analysis Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the Study 2 statistical analysis-core modules that are currently marked unimplemented, without claiming end-to-end MNE/sLORETA execution is complete.

**Architecture:** Add focused modules under `studies/pain_study/study2/` for source inference, directional consistency, artifact gates, spatial comparison, behavioral convergence, and reporting intervals. Each module accepts arrays or DataFrames, validates strictly, and returns dataclasses or tidy tables.

**Tech Stack:** Python 3.11, NumPy, pandas, SciPy, pytest, existing Study 2 YAML config helpers.

---

### Task 1: Source Group Inference

**Files:**
- Create: `studies/pain_study/study2/source_inference.py`
- Test: `studies/tests/pipelines/test_study2_source_inference.py`

- [ ] Write failing tests for one-sample t maps, sign-preserving cluster extraction from explicit adjacency, plus-one empirical p-values from supplied null maps, and invalid input failures.
- [ ] Run `python -m pytest studies/tests/pipelines/test_study2_source_inference.py -q` and verify failure from the missing module.
- [ ] Implement `compute_group_source_inference`.
- [ ] Re-run the source inference tests and verify pass.

### Task 2: Directional Consistency

**Files:**
- Create: `studies/pain_study/study2/directional_consistency.py`
- Test: `studies/tests/pipelines/test_study2_directional_consistency.py`

- [ ] Write failing tests for spatial correlation, same-sign fraction inside a mask, threshold pass/fail, and invalid mask failures.
- [ ] Run the directional consistency tests and verify failure from the missing module.
- [ ] Implement `evaluate_directional_consistency`.
- [ ] Re-run the directional consistency tests and verify pass.

### Task 3: Artifact And Robustness Gates

**Files:**
- Create: `studies/pain_study/study2/artifact_controls.py`
- Test: `studies/tests/pipelines/test_study2_artifact_controls.py`

- [ ] Write failing tests for template-threshold contamination, Holm-corrected expression p-values, gamma relabeling, and robustness threshold summaries.
- [ ] Run the artifact tests and verify failure from the missing module.
- [ ] Implement `evaluate_artifact_controls` and `evaluate_robustness_summary`.
- [ ] Re-run the artifact tests and verify pass.

### Task 4: Spatial Comparison

**Files:**
- Create: `studies/pain_study/study2/spatial_comparison.py`
- Test: `studies/tests/pipelines/test_study2_spatial_comparison.py`

- [ ] Write failing tests for EEG-fMRI spatial correlation, surrogate p-values, meaningful-effect flag, and shape validation.
- [ ] Run the spatial comparison tests and verify failure from the missing module.
- [ ] Implement `compute_spatial_correspondence`.
- [ ] Re-run the spatial comparison tests and verify pass.

### Task 5: Behavioral Convergence

**Files:**
- Create: `studies/pain_study/study2/behavioral_convergence.py`
- Test: `studies/tests/pipelines/test_study2_behavioral_convergence.py`

- [ ] Write failing tests for within-subject nuisance-adjusted standardized slopes, circular-shift permutation p-values, rated-trial/block QC, and zero-variance rating failures.
- [ ] Run the behavioral convergence tests and verify failure from the missing module.
- [ ] Implement `compute_behavioral_convergence`.
- [ ] Re-run the behavioral convergence tests and verify pass.

### Task 6: Reporting Intervals

**Files:**
- Create: `studies/pain_study/study2/reporting.py`
- Test: `studies/tests/pipelines/test_study2_reporting.py`

- [ ] Write failing tests for deterministic bootstrap mean intervals and invalid resample counts.
- [ ] Run the reporting tests and verify failure from the missing module.
- [ ] Implement `bootstrap_mean_interval`.
- [ ] Re-run the reporting tests and verify pass.

### Task 7: Public API And Implementation Status

**Files:**
- Modify: `studies/pain_study/study2/__init__.py`
- Modify: `studies/pain_study/study2/implementation_status.py`
- Modify: `studies/tests/pipelines/test_study2_implementation_status.py`

- [ ] Write failing tests that completed analysis-core components are no longer listed as unimplemented while MNE/source-extraction pieces remain listed.
- [ ] Export the new public functions and dataclasses from `studies.pain_study.study2`.
- [ ] Update implementation status for only the completed analysis-core pieces.
- [ ] Run all Study 2 tests and Ruff on touched Study 2 modules.
