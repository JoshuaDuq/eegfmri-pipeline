Interactive TUI
===============

Terminal-based interactive interface for the EEG/fMRI pipeline, built with
Go 1.21 and `Bubble Tea <https://github.com/charmbracelet/bubbletea>`_.
Wraps the Python CLI (``eeg-pipeline``) with guided wizards, live execution
monitoring, and persistent configuration — all in an alternate-screen terminal
application with mouse support.

.. figure:: ../screenshots/tui_main_menu.png
   :width: 800px
   :align: center
   :alt: TUI main menu

   The main menu gives access to all pipeline stages.

Prerequisites and Build
-----------------------

- **Go 1.21+** — verify with ``go version``
- **Python environment** with the ``eeg_pipeline`` package installed

.. code-block:: bash

   # One-liner build and launch
   cd eeg_pipeline/cli/tui && go mod download && go build -o eeg-tui . && ./eeg-tui

   # Or run directly without a build step
   cd eeg_pipeline/cli/tui && go run main.go

The TUI searches upward for the ``eeg_pipeline`` directory to locate the
repository root. All Python commands execute from that root.

Pipelines
---------

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Pipeline
     - CLI subcommand
     - Description
   * - EEG Preprocessing
     - ``preprocessing``
     - Bad channel detection, filtering, ICA, epoching
   * - fMRI Preprocessing
     - ``fmri``
     - fMRIPrep-style preprocessing
   * - Features
     - ``features``
     - Power, connectivity, aperiodic, ITPC, PAC, complexity, ERP, ERDS, and more
   * - Behavior
     - ``behavior``
     - Trial tables, correlations, regression, condition comparison
   * - Machine Learning
     - ``ml``
     - LOSO regression and classification, time generalization, SHAP, permutation tests
   * - Plotting
     - ``plotting``
     - 40+ plot types across power, connectivity, TFR, ERP, and behavior
   * - fMRI Analysis
     - ``fmri-analysis``
     - First-level contrasts, group inference, trial-wise signatures, resting-state connectivity

Views
-----

Main Menu
~~~~~~~~~

Three sections navigated as a single vertical list with wrap-around:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Section
     - Items
   * - **Preprocessing**
     - EEG Preprocessing, fMRI Preprocessing
   * - **Analysis**
     - Features, Behavior, Machine Learning, Plotting, fMRI Analysis
   * - **Utilities**
     - Global Setup, Pipeline Smoke Test

Pipeline Wizard
~~~~~~~~~~~~~~~~

A multi-step guided configuration flow. Steps vary by pipeline:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Step
     - Description
   * - Select subjects
     - Auto-discovered from BIDS/derivatives with status badges
   * - Select mode
     - Compute, visualize, or pipeline-specific modes
   * - Select computations
     - Toggle individual analyses (behavior pipeline)
   * - Select feature files
     - Choose which feature parquet files to use
   * - Select bands
     - Frequency bands (delta through gamma), editable
   * - Select ROIs
     - Regions of interest with channel lists, editable
   * - Select spatial
     - ROI / All Channels / Global aggregation
   * - Time range
     - Named time windows with tmin/tmax
   * - Advanced config
     - Pipeline-specific parameters (filtering, ICA, epochs, ML models, plot styling, fMRI options)
   * - Select plots
     - Plot catalog with per-plot advanced overrides

On confirmation, the wizard builds a CLI command string and hands it to the
execution view.

Execution View
~~~~~~~~~~~~~~

Runs the Python subprocess and streams output in real time:

- **Progress bar** with subject-level and step-level tracking
- **Per-subject status** (pending / running / done / failed)
- **Resource monitor** — CPU and memory usage, per-core breakdown
- **Scrollable log viewport** with mouse wheel support
- **Copy mode** (``M``) disables mouse capture for native text selection
- **Clipboard copy** (``C``) copies the full log
- **Open results** (``O``) opens the output folder in the system file browser

Global Setup
~~~~~~~~~~~~~

Edit project-level configuration persisted to ``data/derivatives/.tui_overrides.json``:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Section
     - Fields
   * - **Project**
     - Task name, random state, subject list
   * - **Paths**
     - BIDS root, BIDS rest root, BIDS fMRI root, derivatives root, derivatives rest root, source data, FreeSurfer dir, FreeSurfer license

Supports inline text editing and native folder-picker dialogs (``B`` key).

Dashboard
~~~~~~~~~

Read-only overview of project progress:

- **EEG section** — subject counts for Total, BIDS, EEG Prep, Epochs, Features
- **fMRI section** — subject counts for Total, BIDS, fMRI Prep, First Level, Beta Series, LSS
- **Feature categories** — per-category completion bars
- Refreshable with ``R``

History
~~~~~~~

Browsable list of past pipeline executions (persisted in ``eeg_pipeline/cli/tui/.cache/history.json``):

- Pipeline name, mode, duration, relative timestamp, success/failure icon
- Delete individual records (``D``) or clear all (``C``)
- Up to 50 entries retained

Pipeline Smoke Test
~~~~~~~~~~~~~~~~~~~~

A selectable checklist of pipeline commands to run quick parser and runtime checks.
Covers all commands: ``preprocessing``, ``features``, ``behavior``, ``ml``,
``plotting``, ``fmri``, ``fmri-analysis``, ``validate``, ``info``, ``stats``.

Quick Actions
~~~~~~~~~~~~~

Command-palette overlay activated with ``Ctrl+K`` from the main menu or wizard:

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Key
     - Action
   * - ``S``
     - Project Stats (opens Dashboard)
   * - ``H``
     - History
   * - ``V``
     - Validate data integrity
   * - ``X``
     - Export features to CSV
   * - ``C``
     - View configuration (opens Global Setup)
   * - ``R``
     - Refresh subject data

Keyboard Shortcuts
------------------

**Global**

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Key
     - Action
   * - ``Ctrl+C`` / ``Q``
     - Quit (blocked during active execution)
   * - ``Esc``
     - Go back / pop navigation stack
   * - ``D``
     - Open Dashboard (from main menu)
   * - ``H``
     - Open History (from main menu)
   * - ``Ctrl+K``
     - Quick Actions overlay

**Wizard**

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Key
     - Action
   * - ``↑`` / ``↓`` / ``J`` / ``K``
     - Move cursor
   * - ``Space``
     - Toggle selection
   * - ``Enter``
     - Confirm step / proceed
   * - ``A``
     - Select all
   * - ``N``
     - Select none
   * - ``E``
     - Edit selected item (bands, ROIs)
   * - ``+``
     - Add new item
   * - ``D``
     - Delete selected item

**Execution**

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Key
     - Action
   * - ``↑`` / ``↓``
     - Scroll log viewport
   * - ``G`` / ``Shift+G``
     - Jump to top / bottom of log
   * - ``M``
     - Toggle copy mode (disables mouse capture)
   * - ``C``
     - Copy log to clipboard
   * - ``O``
     - Open results folder (after success)
   * - ``R``
     - Re-run the same command (after completion)

**Global Setup**

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Key
     - Action
   * - ``←`` / ``→`` / ``H`` / ``L``
     - Switch section (Project / Paths)
   * - ``↑`` / ``↓`` / ``J`` / ``K``
     - Move cursor
   * - ``Enter`` / ``Space``
     - Edit field
   * - ``B``
     - Browse for folder (path fields)
   * - ``R``
     - Reset overrides to defaults

Persistence
-----------

The TUI persists state across sessions in three locations:

- **``eeg_pipeline/data/derivatives/.tui_state.json``** — last selected pipeline,
  time ranges, band/ROI/spatial selections, per-pipeline advanced configuration
- **``data/derivatives/.tui_overrides.json``** — global setup overrides (task, paths)
- **``eeg_pipeline/cli/tui/.cache/history.json``** — execution history (up to 50 records)

Python Environment Discovery
-----------------------------

The TUI searches for a Python interpreter in this order:

1. ``eeg_pipeline/.venv311/bin/python``
2. ``.venv311/bin/python``
3. ``.venv/bin/python``
4. ``venv/bin/python``
5. System ``python3`` (or ``python`` on Windows)

Design Principles
-----------------

- **Minimal and scientific** — static cursors, sentence-case labels, normal borders, single-color progress bars
- **No external runtime dependencies** — compiles to a single static binary
- **Cross-platform** — macOS, Linux, Windows (clipboard, file browser, folder picker adapt per OS)
- **Crash-safe** — deferred panic handler resets terminal attributes, disables mouse tracking, and exits the alternate screen
