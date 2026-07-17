# Remove TUI-Listed EEG Plotting Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the dedicated TUI Plotting workflow, its Python CLI command/catalog, and EEG-pipeline behavior plots while preserving feature visualization and every study-owned or explicitly protected plot.

**Architecture:** Trace `eeg_pipeline/plotting/plot_catalog.json` through the dedicated TUI and CLI, deleting nodes with no preserved consumer, plus the explicitly removed EEG behavior-plot tree. Retain feature visualization and protected preprocessing, scanner, component-TFR, ML, fMRI, and study plotting as independent roots. Do not leave compatibility aliases, empty screens, or ignored stale calls.

**Tech Stack:** Python 3.11+, argparse, pytest, Go, Bubble Tea, Ruff, repository architecture gates.

---

### Task 1: Add the removal contract tests

**Files:**
- Create: `tests/architecture/test_tui_plotting_removal.py`
- Modify: `eeg_pipeline/cli/tui/types/types_test.go`
- Modify: `eeg_pipeline/cli/tui/views/mainmenu/model_test.go`

- [ ] **Step 1: Write a Python structural test that defines removed and protected roots**

Assert that the command registry has no `plotting`, dedicated command/catalog paths are
absent, and these protected roots remain:

```python
PROTECTED_PLOTS = (
    "eeg_pipeline/preprocessing/eeg_fmri/plotting.py",
    "eeg_pipeline/plotting/scanner_harmonic_comb.py",
)
REMOVED_ENTRY_POINTS = (
    "eeg_pipeline/plotting/plot_catalog.json",
    "eeg_pipeline/cli/commands/plotting.py",
    "eeg_pipeline/cli/commands/plotting_parser.py",
    "eeg_pipeline/cli/commands/plotting_orchestrator.py",
)
```

Also assert `cli/commands/__init__.py` has no `setup_plotting`/`run_plotting` and
`cli/main.py` has no plotting usage example. Parse `behavior visualize --help` and assert
argparse rejects it. Assert `eeg_pipeline/plotting/behavioral/` and
`eeg_pipeline/plotting/orchestration/behavior.py` are absent.

- [ ] **Step 2: Run the Python contract test and verify RED**

Run: `/Users/joduq24/Desktop/EEG_fMRI_Pipeline/.venv/bin/python -m pytest tests/architecture/test_tui_plotting_removal.py -q`

Expected: FAIL because the plotting command and catalog exist.

- [ ] **Step 3: Replace positive TUI Plotting tests with absence tests**

Assert no pipeline name/command equals `Plotting`/`plotting` and no main-menu utility runs
`eeg-pipeline plotting`.

- [ ] **Step 4: Run `go test ./types ./views/mainmenu` and verify RED**

Expected: FAIL because `PipelinePlotting` and its menu item exist.

- [ ] **Step 5: Commit**

```bash
git add tests/architecture/test_tui_plotting_removal.py eeg_pipeline/cli/tui/types/types_test.go eeg_pipeline/cli/tui/views/mainmenu/model_test.go
git commit -m "test: define TUI plotting removal contract"
```

### Task 2: Remove the dedicated Python plotting command and discovery mode

**Files:**
- Delete: `eeg_pipeline/cli/commands/plotting.py`
- Delete: `eeg_pipeline/cli/commands/plotting_catalog.py`
- Delete: `eeg_pipeline/cli/commands/plotting_config_overrides.py`
- Delete: `eeg_pipeline/cli/commands/plotting_definition_helpers.py`
- Delete: `eeg_pipeline/cli/commands/plotting_item_overrides.py`
- Delete: `eeg_pipeline/cli/commands/plotting_orchestrator.py`
- Delete: `eeg_pipeline/cli/commands/plotting_parser.py`
- Delete: `eeg_pipeline/cli/commands/plotting_runner_helpers.py`
- Delete: `eeg_pipeline/cli/commands/plotting_selection.py`
- Delete: `eeg_pipeline/cli/commands/plotting_tfr_mode.py`
- Modify: `eeg_pipeline/cli/commands/__init__.py`
- Modify: `eeg_pipeline/cli/commands/info_helpers.py`
- Modify: `eeg_pipeline/cli/commands/info_orchestrator.py`
- Modify: `eeg_pipeline/cli/commands/info_parser.py`
- Modify: `eeg_pipeline/cli/commands/behavior_parser.py`
- Modify: `eeg_pipeline/cli/commands/behavior_orchestrator.py`
- Modify: `eeg_pipeline/cli/main.py`
- Delete: `tests/cli/test_cli_plotting_connectivity_overrides.py`
- Delete: `tests/cli/test_cli_plotting_definition_helpers.py`
- Delete: `tests/cli/test_cli_plotting_help.py`
- Delete: `tests/cli/test_cli_plotting_item_overrides_helpers.py`
- Delete: `tests/cli/test_cli_plotting_selection_helpers.py`

- [ ] **Step 1: Remove plotting registration and the usage example**

Delete plotting imports and its `CommandSpec` from `commands/__init__.py`; remove the
example from `cli/main.py`.

- [ ] **Step 2: Remove plotter discovery**

Delete `MODE_PLOTTERS`, `_handle_plotters_mode`, the parser choice, and orchestrator branch.

- [ ] **Step 3: Remove behavior visualization and preserve feature visualization**

Restrict the behavior parser to `compute`, remove its visualize-only flags, and simplify
the orchestrator to computation with explicit invalid-mode failure. Keep `features
visualize` and every plotting module it reaches unchanged.

- [ ] **Step 4: Delete the dedicated modules and tests listed above**

- [ ] **Step 5: Run focused tests**

```bash
/Users/joduq24/Desktop/EEG_fMRI_Pipeline/.venv/bin/python -m pytest tests/architecture/test_tui_plotting_removal.py tests/cli/test_cli_features_help.py tests/cli/test_cli_behavior_help.py tests/cli/test_cli_info_help.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add -A eeg_pipeline/cli tests/cli tests/architecture/test_tui_plotting_removal.py
git commit -m "refactor: remove EEG plotting CLI surface"
```

### Task 3: Trace and preserve shared plot implementations

**Files:**
- Delete: `eeg_pipeline/plotting/plot_catalog.json`
- Delete: `eeg_pipeline/plotting/behavioral/`
- Delete: `eeg_pipeline/plotting/orchestration/behavior.py`
- Delete: `eeg_pipeline/plotting/io/collections.py`
- Modify: `eeg_pipeline/plotting/__init__.py`
- Modify: `eeg_pipeline/plotting/io/__init__.py`
- Modify: `eeg_pipeline/plotting/io/figures.py`
- Modify: `eeg_pipeline/plotting/config.py`
- Modify: `eeg_pipeline/pipelines/constants.py`
- Modify: `eeg_pipeline/cli/commands/base.py`
- Delete: `tests/behavior/test_plotting_behavior_dose_response.py`
- Delete: `tests/behavior/test_dose_response_trial_table_loading.py`
- Modify: `tests/behavior/test_behavior_validity_fixes.py`
- Preserve: `eeg_pipeline/plotting/core/`
- Preserve: `eeg_pipeline/plotting/erp/`
- Preserve: `eeg_pipeline/plotting/features/`
- Preserve: `eeg_pipeline/plotting/orchestration/`
- Preserve: `eeg_pipeline/plotting/tfr/`
- Preserve: `eeg_pipeline/plotting/config.py`
- Preserve: `eeg_pipeline/plotting/io/`
- Preserve: `eeg_pipeline/plotting/scanner_harmonic_comb.py`
- Preserve: `eeg_pipeline/plotting/style.py`
- Preserve: `tests/plotting/`
- Preserve: all non-plotting behavior analysis tests
- Preserve: `studies/`

- [ ] **Step 1: Inventory every catalog entry against non-catalog consumers**

Trace registrations and repository imports. Record that feature plotters remain reachable
from `features visualize` and shared core/config/I/O/style code remains reachable from that
path. Confirm no file under `studies/` imports the EEG behavior plotting tree. The catalog
file itself and dedicated CLI mapping helpers are removed in Task 2.

- [ ] **Step 2: Verify generic TFR and ERP ownership**

Confirm their concrete plot functions remain registered by
`eeg_pipeline/plotting/features/registrations.py`. Do not confuse generic catalog TFR with
the separately developed component-TFR implementation.

- [ ] **Step 3: Delete EEG behavior plotting only**

Delete the behavioral implementation and orchestration tree plus its exclusive collection
helper. Remove lazy exports from plotting package initializers, behavior-only footer helpers
from `io/figures.py`, `bins_behavioral`/behavior plot accessors from `plotting/config.py`,
and `BEHAVIOR_VISUALIZE_CATEGORIES` from pipeline/CLI constants. Delete dedicated plotting
tests, but in `test_behavior_validity_fixes.py` remove only the test importing
`behavioral.temporal.topomaps`; retain all behavior-analysis validity coverage. Do not edit
or delete any path under `studies/`.

- [ ] **Step 4: Verify protected independent roots**

Confirm native preprocessing, scanner-harmonic, and main-checkout component-TFR imports do
not depend on a deleted dedicated CLI/catalog module.

- [ ] **Step 5: Run shared and protected plotting tests**

```bash
/Users/joduq24/Desktop/EEG_fMRI_Pipeline/.venv/bin/python -m pytest tests/plotting/test_scanner_harmonic_comb_plot.py tests/scripts/test_run_native_eeg_fmri_artifact_correction.py tests/pipelines/test_pipeline_base_metadata.py tests/pipelines/test_pipeline_features.py -q
```

Expected: PASS.

- [ ] **Step 6: Search for accidental plotting-tree deletions**

Run `git diff --name-status 36cad2f -- eeg_pipeline/plotting tests/plotting tests/behavior`.
Expected: catalog and EEG behavior-plot paths are deleted; shared feature implementations,
non-plotting behavior tests, and all study paths remain.

- [ ] **Step 7: Commit the plotting and behavior-plot removal**

```bash
git add -A eeg_pipeline/plotting eeg_pipeline/pipelines/constants.py eeg_pipeline/cli/commands/base.py tests/behavior
git commit -m "refactor: remove EEG behavior plots"
```

### Task 4: Remove the complete TUI Plotting workflow

**Files:**
- Delete: `eeg_pipeline/cli/tui/views/wizard/model_options_plotting.go`
- Delete: `eeg_pipeline/cli/tui/views/wizard/model_options_plotting_field_data.go`
- Delete: `eeg_pipeline/cli/tui/views/wizard/model_options_plotting_fields.go`
- Delete: `eeg_pipeline/cli/tui/views/wizard/model_options_plotting_types.go`
- Delete: `eeg_pipeline/cli/tui/views/wizard/model_plotting_helpers.go`
- Delete: `eeg_pipeline/cli/tui/views/wizard/model_text_editing_plot_items.go`
- Delete: `eeg_pipeline/cli/tui/views/wizard/plotting_advanced.go`
- Modify: `eeg_pipeline/cli/tui/views/wizard/render_steps_plot.go`
- Modify: `eeg_pipeline/cli/tui/views/wizard/render_steps_plot_test.go`
- Delete: `eeg_pipeline/cli/tui/views/wizard/commands_test.go.1873381661183868199`
- Modify: `eeg_pipeline/cli/tui/types/types.go`
- Modify: `eeg_pipeline/cli/tui/messages/messages.go`
- Modify: `eeg_pipeline/cli/tui/executor/subprocess.go`
- Modify: `eeg_pipeline/cli/tui/executor/subprocess_loaders_test.go`
- Modify: `eeg_pipeline/cli/tui/app/model.go`
- Modify: `eeg_pipeline/cli/tui/app/model_messages.go`
- Modify: `eeg_pipeline/cli/tui/app/model_stateflow.go`
- Modify: `eeg_pipeline/cli/tui/app/model_helpers_test.go`
- Modify: `eeg_pipeline/cli/tui/app/model_message_routing_test.go`
- Modify: `eeg_pipeline/cli/tui/views/mainmenu/model.go`
- Modify: `eeg_pipeline/cli/tui/views/mainmenu/model_test.go`
- Modify: mixed files returned by `rg -l 'PipelinePlotting|plottingScope|plotItem|featurePlotter' eeg_pipeline/cli/tui/views/wizard`

- [ ] **Step 1: Remove pipeline and utility identifiers**

Delete `PipelinePlotting`, its mappings, `UtilityPlotting`, and its main-menu item. Re-index
enum values naturally; do not preserve numeric compatibility.

- [ ] **Step 2: Remove plotter discovery and messages**

Delete `PlottersResponse`, `LoadPlotters`, `PlottersLoadedMsg`, handlers, and load branches.

- [ ] **Step 3: Delete dedicated wizard files and split mixed render code**

Delete dedicated plotting files. In `render_steps_plot.go`, remove only dedicated Plotting
renderers and retain `renderTimeRange`, which serves Features compute mode. Retain its
non-Plotting tests in `render_steps_plot_test.go` while removing Plotting-only tests.

- [ ] **Step 4: Clean mixed wizard files**

Remove fields, initialization, hydration, command building, rendering, scrolling, mouse,
validation, and review logic reachable only from `PipelinePlotting`. Keep ML/fMRI plotting.

- [ ] **Step 5: Format and run all TUI tests**

Run `go fmt ./...` and then `go test ./...` from `eeg_pipeline/cli/tui`.

Expected: PASS with no dedicated plotting references.

- [ ] **Step 6: Commit**

```bash
git add -A eeg_pipeline/cli/tui
git commit -m "refactor: remove TUI plotting workflow"
```

### Task 5: Remove obsolete configuration, packaging, and documentation

**Files:**
- Modify: `eeg_pipeline/utils/config/eeg_config.yaml`
- Modify: `pyproject.toml`
- Delete: `docs/user_guide/cli/plotting.rst`
- Modify: `docs/index.rst`
- Modify: `docs/user_guide/cli/index.rst`
- Modify: `docs/user_guide/cli/behavior.rst`
- Modify: `docs/methods/eeg/behavior.rst`
- Modify: `docs/user_guide/configuration.rst`
- Modify: `docs/user_guide/output_formats.rst`
- Modify: `docs/user_guide/quickstart.rst`

- [ ] **Step 1: Remove catalog packaging and utility configuration**

Delete the catalog package-data entry. Remove configuration when a repository-wide
key-usage search proves the key served the removed dedicated command or behavior plotting,
rather than preserved feature visualization. Preserve shared feature plotting, ML, fMRI,
preprocessing, and component-TFR configuration.

- [ ] **Step 2: Remove command documentation**

Delete the dedicated plotting-command page and its indexes. Remove behavior visualization
documentation while preserving behavior computation and all feature visualization docs.

- [ ] **Step 3: Search for stale public references**

Run `rg -n 'eeg-pipeline plotting|behavior visualize|compute \\| visualize|plot_catalog|PipelinePlotting|UtilityPlotting|PlottersLoadedMsg|BEHAVIOR_VISUALIZE_CATEGORIES|get_behavioral_config|bins_behavioral' README.md docs eeg_pipeline tests scripts pyproject.toml`.

Expected: only the approved spec and plan match.

- [ ] **Step 4: Run config/hygiene tests**

```bash
/Users/joduq24/Desktop/EEG_fMRI_Pipeline/.venv/bin/python -m pytest tests/config tests/utils/test_repo_hygiene_guards.py tests/utils/test_package_init_hygiene.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A docs eeg_pipeline/utils/config/eeg_config.yaml pyproject.toml
git commit -m "docs: remove EEG plotting utility references"
```

### Task 6: Full verification and protected-path audit

**Files:**
- Modify only when a verification failure identifies a root-cause defect.

- [ ] **Step 1: Run `ruff check eeg_pipeline tests scripts`**

Expected: PASS.

- [ ] **Step 2: Run `make verify-structure` and `make verify-architecture`**

Expected: PASS.

- [ ] **Step 3: Run `go test ./...` from `eeg_pipeline/cli/tui`**

Expected: PASS.

- [ ] **Step 4: Run the full Python suite**

Run: `/Users/joduq24/Desktop/EEG_fMRI_Pipeline/.venv/bin/python -m pytest`

Expected: PASS.

- [ ] **Step 5: Audit the diff and protected roots**

Inspect `git diff --stat 36cad2f..HEAD` and `git diff --name-status 36cad2f..HEAD`.
Confirm native preprocessing and scanner-harmonic plotting were not deleted and the
external component-TFR file can coexist in the retained namespace.
Run `git diff --exit-code 36cad2f..HEAD -- studies/` and confirm no study-owned file changed.

- [ ] **Step 6: Commit verification-only corrections if needed**

```bash
git add -A
git commit -m "test: verify EEG plotting removal"
```
