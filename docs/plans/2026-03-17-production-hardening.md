# Production Hardening Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove silent fallbacks and runtime degradations that can hide invalid scientific outputs or incomplete derivatives.

**Architecture:** Tighten fail-fast behavior at the exact decision points that currently mutate requested work, downgrade statistical methods, or suppress enabled reporting failures. Add regression tests beside the existing validity-guard suites so the new behavior is enforced by the repository itself.

**Tech Stack:** Python 3.11+, unittest, project-specific validity guard tests

---

### Task 1: Lock in fail-fast feature-context behavior

**Files:**
- Modify: `tests/features/test_feature_scientific_validity_guards.py`
- Modify: `eeg_pipeline/context/features.py`

**Step 1: Write the failing test**
- Assert that `FeatureContext` raises when `analysis_mode='trial_ml_safe'` requests cross-trial features without `train_mask`.

**Step 2: Run test to verify it fails**
- Run the new unittest target for the feature-context validity guard.

**Step 3: Write minimal implementation**
- Replace warning-and-mutate behavior with a `ValueError`.

**Step 4: Run test to verify it passes**
- Re-run the same unittest target.

### Task 2: Remove silent degradation in behavior/statistics helpers

**Files:**
- Modify: `tests/behavior/test_behavior_validity_fixes.py`
- Modify: `eeg_pipeline/analysis/behavior/feature_inference.py`
- Modify: `eeg_pipeline/utils/parallel.py`

**Step 1: Write the failing tests**
- Assert feature inference surfaces registry failures.
- Assert paired t-tests reject unequal-length inputs instead of downgrading to unpaired tests.

**Step 2: Run tests to verify they fail**
- Run the narrow unittest targets for the new behavior tests.

**Step 3: Write minimal implementation**
- Remove broad exception fallback from feature inference.
- Raise `ValueError` on paired length mismatch.

**Step 4: Run tests to verify they pass**
- Re-run the same unittest targets.

### Task 3: Remove silent ML feature-family fallback

**Files:**
- Modify: `tests/machine_learning/test_machine_learning_validity_fixes.py`
- Modify: `eeg_pipeline/utils/data/machine_learning.py`

**Step 1: Write the failing test**
- Assert combined-feature resolution surfaces import/catalog failures rather than collapsing to `["power"]`.

**Step 2: Run test to verify it fails**
- Run the narrow unittest target for the ML validity guard.

**Step 3: Write minimal implementation**
- Remove the conservative fallback and let the import/config error surface.

**Step 4: Run test to verify it passes**
- Re-run the same unittest target.

### Task 4: Make enabled fMRI reporting and sidecar writes strict

**Files:**
- Modify: `tests/pipelines/test_pipeline_fmri.py`
- Modify: `fmri_pipeline/pipelines/fmri_analysis.py`

**Step 1: Write the failing tests**
- Assert enabled plotting/report generation failures propagate.
- Assert sidecar write failures propagate.

**Step 2: Run tests to verify they fail**
- Run the narrow unittest targets for the fMRI pipeline tests.

**Step 3: Write minimal implementation**
- Remove broad suppression around enabled plotting/reporting and sidecar writes.

**Step 4: Run tests to verify they pass**
- Re-run the same unittest targets.

### Task 5: Strengthen repository quality gates

**Files:**
- Modify: `tests/utils/test_repo_hygiene_guards.py`
- Modify: `pyproject.toml`

**Step 1: Write the failing test**
- Assert Ruff is not configured as an `F821`-only gate.

**Step 2: Run test to verify it fails**
- Run the narrow unittest target for repo hygiene.

**Step 3: Write minimal implementation**
- Expand Ruff selection beyond `F821` to a broader Pyflakes gate and forbid bare `except`.

**Step 4: Run test to verify it passes**
- Re-run the same unittest target.
