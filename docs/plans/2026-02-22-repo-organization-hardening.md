# Repository Organization Hardening Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reduce organizational complexity and drift risk by splitting monolithic modules, clarifying boundaries, and enforcing architectural guardrails.

**Architecture:** Introduce bounded subpackages for ML orchestration, connectivity extraction, and plotting CLI configuration. Keep existing public APIs stable during migration via thin compatibility imports, then remove transitional shims after parity tests pass.

**Tech Stack:** Python 3.11, pytest, argparse, Go (TUI untouched except boundary checks), ruff/compileall, Make.

---

### Task 1: Add Architecture Guardrails and ADR

**Files:**
- Create: `docs/architecture/README.md`
- Create: `docs/architecture/adr-0001-module-boundaries.md`
- Create: `tests/utils/test_architecture_import_boundaries.py`
- Modify: `Makefile`

**Step 1: Write the failing test**

```python
# tests/utils/test_architecture_import_boundaries.py
# Assert forbidden imports such as:
# - eeg_pipeline.analysis.* must not import eeg_pipeline.cli.*
# - eeg_pipeline.utils.* must not import eeg_pipeline.cli.*
# - eeg_pipeline.analysis.features.* must not import eeg_pipeline.analysis.machine_learning.*
```

**Step 2: Run test to verify it fails**

Run: `pytest -q tests/utils/test_architecture_import_boundaries.py`
Expected: FAIL because test file does not exist yet.

**Step 3: Write minimal implementation**

- Implement AST-based import scanning in `tests/utils/test_architecture_import_boundaries.py`.
- Add `make verify-architecture` target in `Makefile`.
- Document approved boundaries in ADR and architecture README.

**Step 4: Run tests to verify they pass**

Run: `make verify-architecture`
Expected: PASS.

**Step 5: Commit**

```bash
git add docs/architecture/README.md docs/architecture/adr-0001-module-boundaries.md tests/utils/test_architecture_import_boundaries.py Makefile
git commit -m "chore: add architecture boundary tests and adr"
```

### Task 2: Refactor Plotting CLI Command into Focused Modules

**Files:**
- Create: `eeg_pipeline/cli/commands/plotting/catalog.py`
- Create: `eeg_pipeline/cli/commands/plotting/overrides.py`
- Create: `eeg_pipeline/cli/commands/plotting/parser.py`
- Create: `eeg_pipeline/cli/commands/plotting/runner.py`
- Create: `eeg_pipeline/cli/commands/plotting/__init__.py`
- Modify: `eeg_pipeline/cli/commands/plotting.py`
- Test: `tests/cli/test_cli_plotting_help.py`

**Step 1: Write the failing test**

- Add/extend a test in `tests/cli/test_cli_plotting_help.py` asserting core options still appear in `--help` output.

**Step 2: Run test to verify it fails**

Run: `pytest -q tests/cli/test_cli_plotting_help.py`
Expected: FAIL after temporary wiring move.

**Step 3: Write minimal implementation**

- Move parser construction to `parser.py`.
- Move config override functions to `overrides.py`.
- Move plot-definition loading/mapping to `catalog.py`.
- Keep `plotting.py` as a compatibility façade that delegates to package modules.

**Step 4: Run tests to verify they pass**

Run: `pytest -q tests/cli/test_cli_plotting_help.py tests/pipelines/test_pipeline_features.py`
Expected: PASS.

**Step 5: Commit**

```bash
git add eeg_pipeline/cli/commands/plotting.py eeg_pipeline/cli/commands/plotting tests/cli/test_cli_plotting_help.py
git commit -m "refactor: split plotting command into focused modules"
```

### Task 3: Decompose ML Orchestration into Submodules

**Files:**
- Create: `eeg_pipeline/analysis/machine_learning/orchestration/classification.py`
- Create: `eeg_pipeline/analysis/machine_learning/orchestration/regression.py`
- Create: `eeg_pipeline/analysis/machine_learning/orchestration/permutation.py`
- Create: `eeg_pipeline/analysis/machine_learning/orchestration/reporting.py`
- Create: `eeg_pipeline/analysis/machine_learning/orchestration/shared.py`
- Create: `eeg_pipeline/analysis/machine_learning/orchestration/__init__.py`
- Modify: `eeg_pipeline/analysis/machine_learning/orchestration.py`
- Test: `tests/machine_learning/test_machine_learning_validity_fixes.py`
- Test: `tests/machine_learning/test_machine_learning_plotting_outputs.py`

**Step 1: Write the failing test**

- Add regression/classification smoke assertions that import paths stay stable from `eeg_pipeline.analysis.machine_learning.orchestration`.

**Step 2: Run test to verify it fails**

Run: `pytest -q tests/machine_learning/test_machine_learning_validity_fixes.py -k orchestration_import`
Expected: FAIL before compatibility exports are added.

**Step 3: Write minimal implementation**

- Move grouped functions to submodules by concern.
- In `orchestration.py`, re-export public functions with thin imports.
- Add docstrings on migration intent and deprecation horizon.

**Step 4: Run tests to verify they pass**

Run: `pytest -q tests/machine_learning/test_machine_learning_validity_fixes.py tests/machine_learning/test_machine_learning_plotting_outputs.py`
Expected: PASS.

**Step 5: Commit**

```bash
git add eeg_pipeline/analysis/machine_learning/orchestration.py eeg_pipeline/analysis/machine_learning/orchestration tests/machine_learning/test_machine_learning_validity_fixes.py
git commit -m "refactor: split machine-learning orchestration by concern"
```

### Task 4: Decompose Connectivity Feature Extraction Module

**Files:**
- Create: `eeg_pipeline/analysis/features/connectivity/config.py`
- Create: `eeg_pipeline/analysis/features/connectivity/dynamic.py`
- Create: `eeg_pipeline/analysis/features/connectivity/directed.py`
- Create: `eeg_pipeline/analysis/features/connectivity/graph.py`
- Create: `eeg_pipeline/analysis/features/connectivity/api.py`
- Create: `eeg_pipeline/analysis/features/connectivity/__init__.py`
- Modify: `eeg_pipeline/analysis/features/connectivity.py`
- Test: `tests/features/test_feature_connectivity_dynamic.py`
- Test: `tests/features/test_feature_connectivity_validity_guards.py`
- Test: `tests/features/test_feature_source_connectivity_validity.py`

**Step 1: Write the failing test**

- Add a compatibility import test ensuring `extract_connectivity_features` and directed variants remain importable from legacy path.

**Step 2: Run test to verify it fails**

Run: `pytest -q tests/features/test_feature_connectivity_dynamic.py -k legacy_import`
Expected: FAIL before shim/re-export.

**Step 3: Write minimal implementation**

- Move helpers into focused files: dynamic windows, directed metrics, graph metrics, API glue.
- Keep `connectivity.py` as temporary façade re-exporting stable APIs.

**Step 4: Run tests to verify they pass**

Run: `pytest -q tests/features/test_feature_connectivity_dynamic.py tests/features/test_feature_connectivity_validity_guards.py tests/features/test_feature_source_connectivity_validity.py`
Expected: PASS.

**Step 5: Commit**

```bash
git add eeg_pipeline/analysis/features/connectivity.py eeg_pipeline/analysis/features/connectivity tests/features/test_feature_connectivity_dynamic.py
git commit -m "refactor: split connectivity feature module into subpackage"
```

### Task 5: Consolidate Entry Points and Deprecate Duplicate Utility Scripts

**Files:**
- Modify: `scripts/eeg_raw_to_bids.py`
- Modify: `scripts/merge_psychopy.py`
- Create: `docs/migration/cli-entrypoint-deprecations.md`
- Test: `tests/scripts/test_fix_restart_trial_triggers_script.py` (reference style)
- Create: `tests/scripts/test_deprecated_entrypoints.py`

**Step 1: Write the failing test**

- Add tests asserting deprecated scripts print deprecation message and still execute CLI equivalent behavior.

**Step 2: Run test to verify it fails**

Run: `pytest -q tests/scripts/test_deprecated_entrypoints.py`
Expected: FAIL before deprecation behavior is added.

**Step 3: Write minimal implementation**

- Convert both scripts to strict wrappers that call `eeg-pipeline utilities ...` internals.
- Emit explicit deprecation warning with target removal version/date.
- Document migration in `docs/migration/cli-entrypoint-deprecations.md`.

**Step 4: Run tests to verify they pass**

Run: `pytest -q tests/scripts/test_deprecated_entrypoints.py tests/pipelines/test_pipeline_utilities_merge.py`
Expected: PASS.

**Step 5: Commit**

```bash
git add scripts/eeg_raw_to_bids.py scripts/merge_psychopy.py docs/migration/cli-entrypoint-deprecations.md tests/scripts/test_deprecated_entrypoints.py
git commit -m "chore: deprecate duplicate utility script entrypoints"
```

### Task 6: Documentation Consolidation and Discoverability

**Files:**
- Create: `docs/index.md`
- Create: `docs/fmri/index.md`
- Create: `docs/eeg/index.md`
- Modify: `README.md`
- Modify: `fmri_pipeline/README.md`
- Move: `README/FMRI_RAW_TO_BIDS.md` -> `docs/fmri/raw-to-bids.md`
- Move: `README/SOURCE_LOCALIZATION_TUTORIAL.md` -> `docs/eeg/source-localization.md`

**Step 1: Write the failing test**

- Add path/reference checks in a new docs-link test:
  `tests/utils/test_docs_entrypoints.py` to verify README links target existing docs files.

**Step 2: Run test to verify it fails**

Run: `pytest -q tests/utils/test_docs_entrypoints.py`
Expected: FAIL before links and moved files are aligned.

**Step 3: Write minimal implementation**

- Create docs indexes and migrate tutorial files.
- Keep lightweight compatibility references in root README sections.
- Ensure fMRI README points to unified docs index.

**Step 4: Run tests to verify they pass**

Run: `pytest -q tests/utils/test_docs_entrypoints.py`
Expected: PASS.

**Step 5: Commit**

```bash
git add docs/index.md docs/fmri/index.md docs/eeg/index.md README.md fmri_pipeline/README.md docs/fmri/raw-to-bids.md docs/eeg/source-localization.md tests/utils/test_docs_entrypoints.py
git commit -m "docs: consolidate documentation structure and entrypoints"
```

### Task 7: Final Verification and Shim Retirement Decision

**Files:**
- Modify: `tests/utils/test_phase2_organization_shims.py`
- Modify: `tests/utils/test_repo_hygiene_guards.py`
- Modify: `Makefile`

**Step 1: Write the failing test**

- Add explicit tests that track allowed shims and fail when temporary facades exceed agreed timeline.

**Step 2: Run test to verify it fails**

Run: `pytest -q tests/utils/test_phase2_organization_shims.py`
Expected: FAIL until shim policy is updated.

**Step 3: Write minimal implementation**

- Update shim tests to reflect current transitional set.
- Add `make verify-maintainability` target combining structure, architecture, and shim checks.

**Step 4: Run full verification**

Run: `make verify-structure && make verify-architecture && make verify-maintainability && pytest -q`
Expected: PASS.

**Step 5: Commit**

```bash
git add tests/utils/test_phase2_organization_shims.py tests/utils/test_repo_hygiene_guards.py Makefile
git commit -m "test: enforce maintainability policy and shim lifecycle"
```

## Rollout Notes

- Execute tasks in order; each task is independently reviewable.
- Keep APIs backward compatible until Task 7 completes.
- Do not edit Go TUI behavior in this plan; only Python organization and docs.
- After Task 7, open one follow-up plan for optional TUI subpackage cleanup.
