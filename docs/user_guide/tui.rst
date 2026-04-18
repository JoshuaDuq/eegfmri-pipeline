Interactive TUI
===============

.. raw:: html

   <p class="hero-intro">
     Terminal UI for running the full pipeline without memorizing commands.
     Guided wizards handle configuration, subject selection, feature families,
     and execution. Built with Go 1.21 +
     <a href="https://github.com/charmbracelet/bubbletea">Bubble Tea</a>;
     compiles to a single static binary.
   </p>

.. figure:: ../screenshots/tui_main_menu.png
   :width: 800px
   :align: center
   :alt: TUI main menu

   The main menu — all pipeline stages accessible through guided wizards.

Build and Run
-------------

.. grid:: 2
   :gutter: 2

   .. grid-item-card:: Requirements

      Go 1.21+ (``go version``) · Python env with ``eeg_pipeline`` installed

   .. grid-item-card:: Quick start

      .. code-block:: bash

         cd eeg_pipeline/cli/tui
         go mod download
         go build -o eeg-tui .
         ./eeg-tui

      .. code-block:: powershell

         cd eeg_pipeline/cli/tui
         go mod download
         go build -o eeg-tui.exe .
         .\eeg-tui.exe

.. note::

   Windows and macOS/Linux do not use the same setup commands.
   On Windows, use PowerShell, ``.\eeg-tui.exe``, and virtual environments
   under ``Scripts\python.exe``. The most reliable Windows workflow is to call
   ``.\.venv\Scripts\python.exe`` and ``.\.venv\Scripts\eeg-pipeline.exe``
   directly rather than relying on shell activation. On macOS/Linux, use
   ``./eeg-tui`` and ``bin/python``.

.. note::

   The TUI searches upward for the ``eeg_pipeline/`` directory and runs all
   Python commands from that repository root. Run ``go run main.go`` to skip
   the build step.

First-Run Setup
~~~~~~~~~~~~~~~

Before running any pipeline, open **Global Setup** (main menu → *Utilities →
Global Setup*, or press ``C`` from the main menu) and set:

- **Task name** — must match the ``task-<name>`` label in your BIDS filenames
- **BIDS root / BIDS rest root** — paths to your EEG (and optionally resting-state) BIDS datasets
- **Derivatives root** — where processed outputs are written
- **fMRI BIDS root** — required only for fMRI workflows

If the TUI shows subjects but marks epochs as missing, first check that
**Task name** matches the ``task-<name>`` segment in your epoch filenames as
well as your raw BIDS files. Existing files such as
``sub-0001_task-thermalactive_proc-clean_epo.fif`` will not be discovered if
the configured task is still ``task``.

These overrides are saved to ``data/derivatives/.tui_overrides.json`` and
persist across sessions. You do not need to edit the YAML config files directly.

Pipelines
---------

Each TUI pipeline maps directly to a CLI command family.

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

Core Views
----------

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
~~~~~~~~~~~~~~~

Multi-step configuration flow. Steps vary by pipeline:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Step
     - Description
   * - Select subjects
     - Auto-discovered from BIDS/derivatives with status badges
   * - Select mode
     - Compute, visualize, or pipeline-specific modes (e.g. ``group_stats`` vs ``trial_ml_safe`` for feature extraction)
   * - Select computations
     - Toggle individual analyses (behavior pipeline)
   * - Select feature families
     - All 16 families listed with checkboxes; select any subset
   * - Select feature files
     - Choose which feature Parquet files to use (behavior / ML pipelines)
   * - Select bands
     - Frequency bands (delta through gamma), editable in-place
   * - Select ROIs
     - Regions of interest with channel lists, editable in-place
   * - Select spatial
     - ROI / All Channels / Global aggregation
   * - Time range
     - Named time windows with tmin/tmax
   * - Advanced config
     - Pipeline-specific parameters (filtering, ICA, epochs, ML models, plot styling, fMRI options, ``--set`` overrides)
   * - Select plots
     - Plot catalog with per-plot advanced overrides

On confirmation, the wizard assembles the full CLI command string (shown in the
execution view header) and launches it as a subprocess.

Execution View
~~~~~~~~~~~~~~

Runs the Python subprocess and streams output in real time:

- Progress bar with subject-level and step-level tracking
- Per-subject status (pending / running / done / failed)
- Resource monitor (CPU and memory usage)
- Scrollable log viewport with mouse wheel support
- Copy mode (``M``) disables mouse capture for native text selection
- Clipboard copy (``C``) copies the full log
- Open results (``O``) opens the output folder in the system file browser

Global Setup
~~~~~~~~~~~~

Edits project-level configuration persisted to ``data/derivatives/.tui_overrides.json``.
Supports inline text editing and native folder-picker dialogs (``B`` key).

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Section
     - Fields
   * - **Project**
     - Task name, random state, subject list
   * - **Paths**
     - BIDS root, BIDS rest root, BIDS fMRI root, derivatives root, derivatives rest root, source data, FreeSurfer dir, FreeSurfer license

Utilities
---------

.. dropdown:: Dashboard
   :icon: graph

   Read-only overview of project progress:

   - EEG section: subject counts for Total, BIDS, EEG Prep, Epochs, Features
   - fMRI section: subject counts for Total, BIDS, fMRI Prep, First Level, Beta Series, LSS
   - Feature categories: per-category completion bars
   - Refreshable with ``R``

.. dropdown:: History
   :icon: history

   Browsable list of past pipeline executions (persisted in
   ``eeg_pipeline/cli/tui/.cache/history.json``):

   - Pipeline name, mode, duration, relative timestamp, success/failure icon
   - Delete individual records (``D``) or clear all (``C``)
   - Up to 50 entries retained

.. dropdown:: Pipeline Smoke Test
   :icon: checklist

   A selectable checklist of pipeline commands to run quick parser and runtime checks.
   Covers: ``preprocessing``, ``features``, ``behavior``, ``ml``, ``plotting``,
   ``fmri``, ``fmri-analysis``, ``validate``, ``info``, ``stats``.

.. dropdown:: Quick Actions
   :icon: terminal

   Command-palette overlay activated with ``Ctrl+K`` from the wizard:

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

.. tab-set::

   .. tab-item:: Global

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
   .. tab-item:: Wizard

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
         * - ``Ctrl+K``
           - Quick Actions overlay

   .. tab-item:: Execution

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

   .. tab-item:: Global Setup

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

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Path
     - Purpose
   * - ``eeg_pipeline/data/derivatives/.tui_state.json``
     - Last selected pipeline, time ranges, band/ROI/spatial selections, per-pipeline advanced configuration
   * - ``data/derivatives/.tui_overrides.json``
     - Global setup overrides (task, paths)
   * - ``eeg_pipeline/cli/tui/.cache/history.json``
     - Execution history (up to 50 records)

Python Environment Discovery
----------------------------

The TUI searches for a Python interpreter in this order:

1. ``eeg_pipeline/.venv311/bin/python`` or ``eeg_pipeline\.venv311\Scripts\python.exe``
2. ``.venv311/bin/python`` or ``.venv311\Scripts\python.exe``
3. ``.venv/bin/python`` or ``.venv\Scripts\python.exe``
4. ``venv/bin/python`` or ``venv\Scripts\python.exe``
5. System ``python3`` on macOS/Linux, or ``python`` then ``py -3`` on Windows

Native Windows support covers the TUI, CLI bootstrap, validation, and smoke
checks. Container-backed fMRI preprocessing and Docker-based BEM/source-localization
helpers should be run from WSL2 or a Linux/macOS host.

Design Principles
-----------------

- Minimal and scientific: restrained UI and single-color progress bars
- No external runtime dependencies: compiles to a single static binary
- Cross-platform: clipboard, file browser, and folder picker adapt per OS
- Crash-safe: panic handler resets terminal attributes and exits the alternate screen
