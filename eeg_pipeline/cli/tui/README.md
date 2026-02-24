# TUI (Text User Interface)

Terminal-based interactive interface for the EEG/fMRI pipeline, built with
Go 1.21 and [Bubble Tea](https://github.com/charmbracelet/bubbletea).
Wraps the Python CLI (`eeg-pipeline`) with guided wizards, live execution
monitoring, and persistent configuration — all in an alternate-screen
terminal application with mouse support.

![Main menu](../../../docs/screenshots/tui_main_menu.png)

## Prerequisites

- **Go 1.21+** — verify with `go version`
- **Python environment** with the `eeg_pipeline` package installed
  (the TUI auto-discovers virtual environments at `.venv311`, `.venv`, or `venv`)

## Building and Running

```bash
# One-liner from a fresh terminal
cd eeg_pipeline/cli/tui && go mod download && go build -o eeg-tui . && ./eeg-tui

# Or run directly without a build step
cd eeg_pipeline/cli/tui && go run main.go
```

The TUI searches upward for the `eeg_pipeline` directory to locate the
repository root. All Python commands are executed from that root.

---

## Architecture

```
main.go                  Entry point, terminal reset, panic recovery
app/
  model.go               Root Model (state machine, nav stack, persistence)
  model_stateflow.go     Per-view update delegation and navigation flow
  model_messages.go      Subject/config message handlers
  model_persistence.go   Repo-root discovery and JSON state file
views/
  mainmenu/              Pipeline selector (three sections)
  wizard/                Multi-step configuration wizard
  execution/             Live subprocess runner with log viewport
  globalsetup/           Project paths and settings editor
  dashboard/             Read-only project statistics
  history/               Execution history browser
  quickactions/          Command-palette overlay
  pipelinesmoke/         Pipeline smoke-test selector (CLI parser/runtime checks)
animation/               Cursor blink and progress pulse loops
components/              Toast, HelpOverlay, Spinner, ScrollIndicator, InfoPanel
executor/                Python subprocess launcher, clipboard, file browser
messages/                Shared Bubble Tea message types
styles/                  Colors, constants, layout helpers
types/                   Pipeline and WizardStep enums
```

Single Bubble Tea program using an alternate screen with mouse cell-motion.
A navigation stack (`navStack`) tracks view history; `Esc` always returns to
the previous screen.

---

## Views

### Main Menu

Three sections, navigated as a single vertical list with wrap-around:

| Section | Items |
|---|---|
| **Preprocessing** | EEG Preprocessing, fMRI Preprocessing |
| **Analysis** | Features, Behavior, Machine Learning, Plotting, fMRI Analysis |
| **Utilities** | Global Setup, Pipeline Smoke Test |

Selecting a pipeline or utility opens the corresponding wizard or settings view.

### Pipeline Wizard

A multi-step guided configuration flow. Steps vary by pipeline; common steps
include:

| Step | Description |
|---|---|
| **Select subjects** | Auto-discovered from BIDS/derivatives with status badges |
| **Select mode** | Compute, visualize, or pipeline-specific modes |
| **Select computations** | Toggle individual analyses (behavior pipeline) |
| **Select feature files** | Choose which feature parquet files to use |
| **Select bands** | Frequency bands (delta through gamma), editable |
| **Select ROIs** | Regions of interest with channel lists, editable |
| **Select spatial** | ROI / All Channels / Global aggregation |
| **Time range** | Named time windows with tmin/tmax |
| **Advanced config** | Pipeline-specific parameters (filtering, ICA, epochs, ML models, plot styling, fMRI options) |
| **Select plots** | Plot catalog with per-plot advanced overrides |

On confirmation the wizard builds a CLI command string and hands it to the
execution view.

### Execution View

Runs the Python subprocess and streams output in real time:

- **Progress bar** with subject-level and step-level tracking
- **Per-subject status** (pending / running / done / failed)
- **ETA estimation** based on completed subject durations
- **Resource monitor** — CPU and memory usage, per-core breakdown
- **Scrollable log viewport** with mouse wheel support
- **Copy mode** (`M`) disables mouse capture for native text selection
- **Clipboard copy** (`C`) copies the full log
- **Open results** (`O`) opens the output folder in the system file browser

### Global Setup

Edit project-level configuration persisted to `.tui_overrides.json`:

| Section | Fields |
|---|---|
| **Project** | Task name, random state, subject list |
| **Paths** | BIDS root, BIDS fMRI root, derivatives root, source data, FreeSurfer dir |

Supports inline text editing and native folder-picker dialogs (`B` key).

### Dashboard

Read-only overview of project progress:

- **EEG section** — subject counts for Total, BIDS, EEG Prep, Epochs, Features
- **fMRI section** — subject counts for Total, BIDS, fMRI Prep, First Level, Beta Series, LSS
- **Feature categories** — per-category completion bars
- Refreshable with `R`

### History

Browsable list of past pipeline executions (persisted in `.cache/history.json`):

- Pipeline name, mode, duration, relative timestamp, success/failure icon
- Delete individual records (`D`) or clear all (`C`)
- Up to 50 entries retained

### Pipeline Smoke Test

A selectable checklist of pipeline commands to run quick parser and runtime checks:

- Covers all pipeline commands: `preprocessing`, `features`, `behavior`, `ml`, `plotting`, `fmri`, `fmri-analysis`, `validate`, `info`, `stats`, and an end-to-end `runtime_version` dispatch check
- Toggle individual checks with `Space` or all with `A`
- Runs `scripts/tui_pipeline_smoke.py` with the selected pipeline IDs

### Quick Actions

A command-palette overlay activated with `Ctrl+K` from the main menu or wizard:

| Shortcut | Action |
|---|---|
| `S` | Project Stats (opens Dashboard) |
| `H` | History |
| `V` | Validate data integrity |
| `X` | Export features to CSV |
| `C` | View configuration (opens Global Setup) |
| `R` | Refresh subject data |

---

## Keyboard Shortcuts

### Global

| Key | Action |
|---|---|
| `Ctrl+C` / `Q` | Quit (blocked during active execution) |
| `Esc` | Go back / pop navigation stack |
| `D` | Open Dashboard (from main menu) |
| `H` | Open History (from main menu) |
| `Ctrl+K` | Quick Actions overlay |

### Main Menu

| Key | Action |
|---|---|
| `↑` / `↓` / `J` / `K` | Move cursor (wraps between sections) |
| `Enter` / `Space` | Select pipeline or utility |
| `?` | Toggle help overlay |

### Wizard

| Key | Action |
|---|---|
| `↑` / `↓` / `J` / `K` | Move cursor |
| `Space` | Toggle selection |
| `Enter` | Confirm step / proceed |
| `A` | Select all |
| `N` | Select none |
| `E` | Edit selected item (bands, ROIs) |
| `+` | Add new item |
| `D` | Delete selected item |
| `Esc` | Go back one step |

### Execution

| Key | Action |
|---|---|
| `↑` / `↓` | Scroll log viewport |
| `G` / `Shift+G` | Jump to top / bottom of log |
| `M` | Toggle copy mode (disables mouse capture) |
| `C` | Copy log to clipboard |
| `O` | Open results folder (after success) |
| `R` | Re-run the same command (after completion) |
| `Enter` | Return to main menu (after completion) |
| `Ctrl+C` | Cancel running process / copy log if done |

### Global Setup

| Key | Action |
|---|---|
| `←` / `→` / `H` / `L` | Switch section (Project / Paths) |
| `↑` / `↓` / `J` / `K` | Move cursor |
| `Enter` / `Space` | Edit field |
| `B` | Browse for folder (path fields) |
| `R` | Reset overrides to defaults |
| `Esc` | Save and return |

### Dashboard

| Key | Action |
|---|---|
| `R` | Refresh statistics |
| `Esc` | Back |

### History

| Key | Action |
|---|---|
| `↑` / `↓` / `J` / `K` | Navigate records |
| `D` | Delete selected record |
| `C` | Clear all history |
| `Esc` | Back |

---

## Pipelines

| Pipeline | CLI subcommand | Data source | Description |
|---|---|---|---|
| EEG Preprocessing | `preprocessing` | `bids` | Bad channel detection, filtering, ICA, epoching |
| fMRI Preprocessing | `fmri` | `bids_fmri` | fMRIPrep-style preprocessing |
| Features | `features` | `epochs` | Power, connectivity, aperiodic, ITPC, PAC, complexity, ratios, asymmetry, microstates, ERDS, spectral |
| Behavior | `behavior` | `epochs` | Trial tables, correlations, regression, condition comparison, mediation, mixed effects |
| Machine Learning | `ml` | `features` | LOSO regression (ElasticNet/Ridge/RF), classification (SVM/LR/RF/CNN), time generalization |
| Plotting | `plotting` | `all` | 40+ plot types across power, connectivity, TFR, ERP, behavior, and more |
| fMRI Analysis | `fmri-analysis` | `bids_fmri` | First-level contrasts, trial-wise signatures |
| Pipeline Smoke Test | `scripts/tui_pipeline_smoke.py` | — | Quick CLI parser and runtime checks across all pipeline commands (excludes paradigm-specific scripts) |

---

## Persistence

The TUI persists state across sessions in two locations:

- **`eeg_pipeline/data/derivatives/.tui_state.json`** — last selected pipeline, time ranges,
  band/ROI/spatial selections, per-pipeline advanced configuration
- **`eeg_pipeline/data/derivatives/.tui_overrides.json`** — global setup overrides (task, paths)
- **`.cache/history.json`** — execution history (up to 50 records)

Subject discovery results are cached in memory per session (keyed by
`task|data_source`) and can be force-refreshed from the wizard.

---

## Python Environment Discovery

The TUI searches for a Python interpreter in this order:

1. `eeg_pipeline/.venv311/bin/python`
2. `.venv311/bin/python`
3. `.venv/bin/python`
4. `venv/bin/python`
5. System `python3` (or `python` on Windows)

---

## Design Principles

- **Minimal and scientific** — static cursors, sentence-case labels, normal
  borders, single-color progress bars
- **No external runtime dependencies** — compiles to a single static binary
- **Cross-platform** — macOS, Linux, Windows (clipboard, file browser, folder
  picker adapt per OS)
- **Crash-safe** — deferred panic handler resets terminal attributes, disables
  mouse tracking, and exits the alternate screen
