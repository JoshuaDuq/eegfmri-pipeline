# Import And Metadata Hardening Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove import-time fragility and remaining silent analysis degradations in package exports, stats config parsing, ROI filtering, signature discovery, and run-metadata persistence.

**Architecture:** Replace eager package re-export imports with explicit lazy exports via `__getattr__`, and make the remaining runtime helpers fail fast when configuration or metadata prerequisites are invalid. The tests combine structural hygiene checks with narrow behavior regressions.

**Tech Stack:** Python 3.11+, `unittest`, `pytest`, AST-based repo hygiene tests

---

### Task 1: Lock package exports to lazy loading

**Files:**
- Modify: `tests/utils/test_package_init_hygiene.py`
- Modify: `eeg_pipeline/analysis/__init__.py`
- Modify: `eeg_pipeline/utils/data/__init__.py`
- Modify: `eeg_pipeline/pipelines/__init__.py`

**Step 1: Write the failing tests**
- Assert these package `__init__` files do not eagerly import internal submodules.
- Assert they expose a module-level `__getattr__` for lazy exports.

**Step 2: Run test to verify it fails**
- Run the package-hygiene unittest target.

**Step 3: Write minimal implementation**
- Replace eager re-export imports with lazy `import_module` dispatch.

**Step 4: Run test to verify it passes**
- Re-run the package-hygiene unittest target.

### Task 2: Make predictor-control parsing strict

**Files:**
- Modify: `tests/behavior/test_behavior_validity_fixes.py`
- Modify: `eeg_pipeline/utils/analysis/stats/permutation.py`
- Modify: `eeg_pipeline/utils/analysis/stats/partial.py`

**Step 1: Write the failing tests**
- Assert invalid predictor-control config values raise.
- Assert explicit `"none"` does not silently downgrade to spline control.

**Step 2: Run test to verify it fails**
- Run the narrow behavior validity tests.

**Step 3: Write minimal implementation**
- Validate the config value strictly and support `"none"` explicitly.

**Step 4: Run test to verify it passes**
- Re-run the narrow behavior validity tests.

### Task 3: Make ROI feature selection strict

**Files:**
- Modify: `tests/machine_learning/test_machine_learning_validity_fixes.py`
- Modify: `eeg_pipeline/analysis/machine_learning/preprocessing.py`

**Step 1: Write the failing test**
- Assert `SpatialFeatureSelector` raises when region filtering is requested but feature names are unavailable.

**Step 2: Run test to verify it fails**
- Run the narrow ML validity test.

**Step 3: Write minimal implementation**
- Replace warn-and-keep-all behavior with a clear `ValueError`.

**Step 4: Run test to verify it passes**
- Re-run the narrow ML validity test.

### Task 4: Make signature discovery strict

**Files:**
- Modify: `tests/fmri/test_fmri_signature_paths.py`
- Modify: `fmri_pipeline/utils/signature_paths.py`

**Step 1: Write the failing tests**
- Assert config getter failures surface.
- Assert invalid derivative-root inputs surface.

**Step 2: Run test to verify it fails**
- Run the narrow fMRI signature-path tests.

**Step 3: Write minimal implementation**
- Remove broad exception fallbacks from config/path resolution.

**Step 4: Run test to verify it passes**
- Re-run the narrow fMRI signature-path tests.

### Task 5: Make run-metadata persistence strict

**Files:**
- Modify: `tests/pipelines/test_pipeline_base.py`
- Modify: `eeg_pipeline/pipelines/base.py`

**Step 1: Write the failing test**
- Assert run-metadata write failures surface instead of being logged and ignored.

**Step 2: Run test to verify it fails**
- Run the narrow pipeline-base test.

**Step 3: Write minimal implementation**
- Raise on missing derivatives root or metadata write/serialization failure.

**Step 4: Run test to verify it passes**
- Re-run the narrow pipeline-base test.
