# fMRI Analysis Plotting + HTML Report Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a TUI-configurable plotting/reporting section to the `fmri-analysis` pipeline that generates Nature-quality per-subject figures (PNG/SVG) and an HTML report, with default threshold `|z| > 2.3`, and optional `native`, `MNI`, or `both` spaces.

**Architecture:** Extend the `fmri-analysis` CLI to accept plotting/report options; persist and surface them in the Go TUI wizard. After producing the contrast map(s), run a best-effort plotting/report step that uses nilearn when available, degrades gracefully when not, and writes outputs under each subject’s contrast directory.

**Tech Stack:** Python (nilearn, nibabel, matplotlib), Go TUI (Bubble Tea), existing pipeline + config loader.

---

## Outputs (per subject, per contrast)

- `.../contrast-<name>/plots/native/` (PNG/SVG) when space includes `native`
- `.../contrast-<name>/plots/mni/` (PNG/SVG) when space includes `mni`
- `.../contrast-<name>/report.html` (single self-contained HTML referencing local images)

Default plots (configurable):
- Stat map mosaic slices (unthresholded + thresholded)
- Glass brain (unthresholded + thresholded)
- Z-stat histogram (masked, best-effort)
- Cluster/peak table (best-effort via nilearn reporting)
- Design matrix QC: link/embed any existing `qc/*design_matrix.(png|tsv)` files

## User-Facing Configuration (TUI + CLI)

New fMRI analysis “Plotting” group:
- Enable plotting (on/off)
- Enable HTML report (on/off)
- Formats: PNG, SVG (multi-select)
- Space: native | mni | both (default both)
- Threshold z for thresholded panels (default 2.3)
- Include unthresholded panels (default on)
- Plot types toggles: slices, glass, histogram, clusters table

## Behavior Notes

- “both spaces” runs the GLM twice (once per fMRIPrep space). If MNI BOLD is unavailable, the MNI section is skipped and the report still builds.
- If `--resample-to-freesurfer` is enabled, the pipeline still writes the FreeSurfer-space NIfTI; plotting uses the *pre-resample* map(s).
- Plotting is best-effort and must never fail the GLM/contrast computation.

---

# Implementation Tasks (TDD-oriented)

### Task 1: Add plotting config model + unit tests

**Files:**
- Create: `fmri_pipeline/analysis/plotting_config.py`
- Create: `tests/test_fmri_plotting_config.py`

**Step 1: Write the failing test**

- Parse/normalize CLI-like inputs:
  - default threshold = 2.3
  - default space = both
  - formats must include at least one of png/svg when plotting enabled

**Step 2: Run test to verify it fails**

Run: `pytest -q`
Expected: FAIL (missing module / config not implemented).

**Step 3: Write minimal implementation**

- Implement a small dataclass `FmriPlottingConfig` with:
  - `enabled`, `html_report`, `formats`, `space`, `z_threshold`, `include_unthresholded`, `plot_types`
  - `validate()` and `normalized()` helpers

**Step 4: Run test to verify it passes**

Run: `pytest -q`
Expected: PASS.

---

### Task 2: Extend `fmri-analysis` CLI with plotting/report args

**Files:**
- Modify: `fmri_pipeline/cli/commands/fmri_analysis.py`
- Create: `tests/test_fmri_analysis_cli_plotting_args.py`

**Step 1: Write failing test**

- Ensure `--plotting` args are accepted and mapped into an `FmriPlottingConfig`.

**Step 2: Run failing test**

Run: `pytest -q`
Expected: FAIL.

**Step 3: Implement minimal CLI plumbing**

- Add `Plotting / Report` arg group with defaults:
  - `--plots/--no-plots`
  - `--plot-formats png svg`
  - `--plot-space both`
  - `--plot-z-threshold 2.3`
  - `--plot-include-unthresholded/--no-plot-include-unthresholded`
  - `--plot-types slices glass hist clusters`
  - `--plot-html-report/--no-plot-html-report`

**Step 4: Run test**

Run: `pytest -q`
Expected: PASS.

---

### Task 3: Implement plotting + HTML report generator (best-effort)

**Files:**
- Create: `fmri_pipeline/analysis/reporting.py`
- Create: `tests/test_fmri_reporting_html.py`

**Step 1: Write failing test**

- Given a synthetic “assets list” (images + titles), render HTML:
  - includes threshold text
  - includes sections for native/mni when present
  - includes links to QC design matrix files

**Step 2: Run failing test**

Run: `pytest -q`
Expected: FAIL.

**Step 3: Minimal implementation**

- Provide:
  - `build_report_html(...) -> str`
  - Small helpers to format metadata safely (no Jinja2 dependency).

**Step 4: Run test**

Run: `pytest -q`
Expected: PASS.

---

### Task 4: Integrate plotting/reporting into `FmriAnalysisPipeline`

**Files:**
- Modify: `fmri_pipeline/pipelines/fmri_analysis.py`

**Step 1: Write failing test (smoke)**

- A unit test that calls a small “postprocess” function with `plotting_enabled=False` and asserts no side effects.
- (If full pipeline testing is too heavy, keep the test at the helper level.)

**Step 2: Run failing test**

Run: `pytest -q`
Expected: FAIL.

**Step 3: Implement**

- After contrast save:
  - If plotting enabled, generate figures into `plots/<space>/`
  - Generate `report.html` when enabled
- For MNI:
  - Re-run `build_contrast_from_runs` with `fmriprep_space` set to MNI space string
  - Skip with warning if BOLD not found
- Never throw if plotting fails.

**Step 4: Run test**

Run: `pytest -q`
Expected: PASS.

---

### Task 5: Add “Plotting” group to the TUI fMRI analysis advanced config

**Files (Go):**
- Modify: `eeg_pipeline/cli/tui/views/wizard/model.go`
- Modify: `eeg_pipeline/cli/tui/views/wizard/render_steps.go`
- Modify: `eeg_pipeline/cli/tui/views/wizard/handlers.go`
- Modify: `eeg_pipeline/cli/tui/views/wizard/config_persist.go`
- Modify: `eeg_pipeline/cli/tui/views/wizard/commands.go`

**Step 1: Add new model fields + defaults**

- Group expansion bool and per-option values matching CLI defaults.

**Step 2: Render new group in fMRI analysis advanced view**

- Collapsible section similar to Input/Contrast/GLM/Confounds/Output.

**Step 3: Wire handlers**

- Space toggles/cycles for enums.
- Boolean toggles for checkboxes.
- Numeric editing for `z-threshold`.

**Step 4: Add CLI arg emission**

- Only emit args when non-default to keep command lines clean.

**Step 5: Persist values**

- Save/load the new fields via `config_persist.go`.

**Step 6: Verify**

Run: `go test ./...`
Expected: PASS (may require `go mod download` first).

---

### Task 6: Verification (before claiming completion)

- Python syntax: `python3 -m compileall fmri_pipeline eeg_pipeline`
- Pytest (if deps available): `pytest -q`
- Go: `cd eeg_pipeline/cli/tui && go test ./...`
- Optional: run TUI and verify the new “Plotting” group affects emitted `fmri-analysis` command args.

