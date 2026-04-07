Quick Start
===========

.. raw:: html

   <p class="hero-intro">
     Operational walkthrough from raw BIDS data to derivatives. Each section
     states the purpose, files read/written, and commands. Use the tab panels
     in <a href="#run-the-pipeline">step 5</a> for the full command surface.
   </p>

.. admonition:: New to the pipeline?

   Start with the :doc:`TUI <tui>`. It exposes the same workflow interactively:
   configuration, subject selection, pipeline parameters, command assembly, and
   execution — no terminal commands required.

.. _qs-overview:

Pipeline Overview
-----------------

Four derivative-producing stages; ``trial_id`` in ``proc-clean_events.tsv`` is
the canonical join key aligning EEG features, fMRI betas, and behavioral targets.

.. grid:: 2
   :gutter: 2

   .. grid-item-card:: 01 — EEG Preprocessing

      **In:** BIDS EEG (``.vhdr`` · ``.edf`` · ``.fif``) + ``events.tsv``

      PyPREP bad-channel detection (3 iterations, optional RANSAC) ·
      extended Infomax ICA (99% variance, 1 Hz HP) · ICLabel (p > 0.8) ·
      epoching ``[−7, 15] s``, baseline ``[−0.2, 0] s``, autoreject.

      **Out:** ``proc-clean_epo.fif`` · ``proc-clean_events.tsv`` · ICA logs

   .. grid-item-card:: 02 — Feature Extraction

      **In:** ``proc-clean_epo.fif`` · ``proc-clean_events.tsv``

      16 families — ``power`` · ``spectral`` · ``aperiodic`` · ``erp`` ·
      ``erds`` · ``ratios`` · ``asymmetry`` · ``microstates`` ·
      ``connectivity`` · ``directedconnectivity`` · ``itpc`` · ``pac`` ·
      ``sourcelocalization`` · ``complexity`` · ``bursts`` · ``quality``

      Granularity: trial / condition / subject · scopes: ROI, channels, global

      **Out:** ``features/<family>/features_*.parquet`` + ``metadata/*.json``

   .. grid-item-card:: 03a — Behavioral Statistics

      Partial Spearman correlations · predictor residualization (spline OLS,
      AIC-selected) · trial-wise OLS (HC3) · ICC(3,1) reliability · condition
      contrasts (Welch t / Cohen's d / Hedges' g) · temporal cluster tests ·
      Benjamini–Hochberg + hierarchical Simes FDR.

      **Out:** ``stats/`` (TSV / Parquet per stage)

   .. grid-item-card:: 03b — Machine Learning

      Nested LOSO CV (outer LOGO · inner GroupKFold) · ElasticNet / Ridge / RF
      regression · SVM / LR / RF / EEGNet classification · temporal
      generalization · conformal intervals · SHAP importance (fold-aggregated)
      · permutation test on shuffled labels.

      **Out:** ``ml/`` (summaries, predictions, SHAP, figures)

.. grid:: 1
   :gutter: 0

   .. grid-item-card:: 04 — fMRI Pipeline *(optional)*

      **In:** BIDS fMRI (fMRIPrep derivatives by default;
      ``--input-source bids_raw`` for raw BOLD)

      Containerized fMRIPrep (``nipreps/fmriprep:25.2.4``) · Nilearn
      ``FirstLevelModel`` (HRF ``spm``, drift ``cosine``, HP 0.008 Hz) ·
      trial-wise betas (LSA / LSS) · group one-sample GLM + max-T permutation
      (5 000 iterations) · resting-state ROI connectivity · EEG–fMRI fusion.

      **Out:** ``sub-*/fmri/first_level/`` · ``sub-*/fmri/beta_series/`` ·
      ``group/fmri/second_level/``

      .. note::

         The fMRI pipeline is **still under active development**. Command-line
         options, defaults, and file layouts may change between releases.

.. _qs-install:

1. Install
----------

.. grid:: 1
   :gutter: 0

   .. grid-item-card::

      **Step 1 of 8** — Create an isolated environment and install the package.

Create an isolated Python environment and install the package:

macOS / Linux:

.. code-block:: bash

   git clone https://github.com/JoshuaDuq/eegfmri-pipeline.git
   cd eegfmri-pipeline
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev,ml]"

Windows PowerShell:

.. code-block:: powershell

   git clone https://github.com/JoshuaDuq/eegfmri-pipeline.git
   cd eegfmri-pipeline
   py -m venv .venv
   .venv\Scripts\Activate.ps1
   pip install -e ".[dev,ml]"

.. note::

   Windows setup is different from macOS/Linux:

   - use PowerShell or ``cmd`` instead of ``source``
   - create the environment with any ``Python 3.11+`` interpreter
   - activate from ``.venv\Scripts\Activate.ps1``
   - use ``Scripts\python.exe`` rather than ``bin/python``
   - if multiple Python versions are installed, select one explicitly, for
     example ``python3.12 -m venv .venv`` or ``py -3.12 -m venv .venv``

.. note::

   The ``[ml]`` extra installs PyTorch, which is only required for the CNN
   classifier (``ml classify --classification-model cnn``). For all other
   workflows, ``pip install -e ".[dev]"`` is sufficient.

See :doc:`../install` for environment variables, the optional TUI build, and
the FreeSurfer/MNE Docker image for source localization. On native Windows,
use WSL2 or a Linux/macOS host for container-backed fMRI preprocessing and
Docker-based BEM/source-localization helpers.

.. _qs-prepare:

2. Prepare Data
---------------

.. grid:: 1
   :gutter: 0

   .. grid-item-card::

      **Step 2 of 8** — Organize BIDS-formatted EEG data on disk.

Place BIDS-formatted EEG data under ``paths.bids_root``
(default ``data/bids_output/eeg/``):

.. code-block:: text

   data/bids_output/eeg/
   ├── dataset_description.json
   ├── participants.tsv
   └── sub-0001/
       └── eeg/
           ├── sub-0001_task-task_run-01_eeg.vhdr
           ├── sub-0001_task-task_run-01_eeg.vmrk
           ├── sub-0001_task-task_run-01_eeg.eeg
           ├── sub-0001_task-task_run-01_events.tsv
           ├── sub-0001_task-task_run-01_channels.tsv
           └── sub-0001_task-task_run-01_electrodes.tsv

**Required events columns:** ``onset``, ``duration``, ``trial_type``.
Add any study-specific predictor or outcome columns alongside these.
All downstream stages (behavior, ML, fMRI fusion) read from these files.

Full data layout and fMRI BIDS requirements: :doc:`data_layout`.

.. _qs-validate:

3. Validate and Inspect
-----------------------

.. grid:: 1
   :gutter: 0

   .. grid-item-card::

      **Step 3 of 8** — Run read-only checks before processing any subjects.

Run read-only checks before touching any data:

.. code-block:: bash

   eeg-pipeline validate quick        # BIDS structure + config consistency
   eeg-pipeline info subjects         # List discovered subjects and run counts
   eeg-pipeline info config           # Print resolved active configuration
   eeg-pipeline info ml-feature-space # Show feature matrix dimensions (post-extraction)

``validate quick`` confirms that the BIDS layout is parseable, required files
exist, and the ``eeg_config.yaml`` values are internally consistent.
``info subjects`` lists every subject the pipeline will process when
``--all-subjects`` is used. Run this before any batch job.

For a deeper sweep:

.. code-block:: bash

   eeg-pipeline validate all           # Full validation including epochs and features
   eeg-pipeline info features 0001     # Inspect extracted feature tables for sub-0001
   eeg-pipeline info discover          # Auto-discover available data and report gaps
   eeg-pipeline stats summary          # Dataset-level statistics across all subjects

.. _qs-configure:

4. Configure
------------

.. grid:: 1
   :gutter: 0

   .. grid-item-card::

      **Step 4 of 8** — Set task name, paths, and pipeline parameters.

All pipeline behavior is controlled by ``eeg_pipeline/utils/config/eeg_config.yaml``
(and ``behavior_config.yaml`` / ``fmri_config.yaml`` for their respective domains).

**Preferred path — TUI Global Setup:**
Launch the TUI, navigate to *Global Setup* (or press ``C`` from the main menu),
and set your task name and all path roots through the interactive editor.
Settings are saved to ``data/derivatives/.tui_overrides.json`` and take effect
immediately. No YAML editing required.

**Alternative — edit YAML directly or inspect at runtime:**

.. code-block:: bash

   eeg-pipeline info config   # print the resolved active configuration

Key entries to verify before feature extraction:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Key
     - What to check
   * - ``project.task``
     - Matches the ``task-<name>`` label in your BIDS files
   * - ``project.subject_list``
     - Set to a list of IDs to restrict processing; ``null`` = all discovered subjects
   * - ``feature_engineering.feature_categories``
     - The active feature families to extract
   * - ``feature_engineering.analysis_mode``
     - ``trial_ml_safe`` for ML inputs; ``group_stats`` for descriptive analyses
   * - ``frequency_bands``
     - Band edges in Hz (default: delta 1–4, theta 4–8, alpha 8–13, beta 13–30, gamma 30–80)
   * - ``time_windows.active``
     - Active-epoch window in seconds (default ``[3.0, 10.5]``)
   * - ``time_windows.baseline_tfr``
     - TFR baseline window in seconds (default ``[-5.0, -0.01]``)

Override any key at runtime without editing the YAML:

.. code-block:: bash

   eeg-pipeline features compute --subject 0001 \
     --set feature_engineering.analysis_mode=trial_ml_safe \
     --set feature_engineering.parallel.n_jobs_bands=-1 \
     --set feature_engineering.parallel.n_jobs_connectivity=-1

See :doc:`configuration` for the complete key reference.

Treat configuration as part of the analysis specification. Before processing a
full cohort, confirm that dataset roots, task labels, time windows, feature
families, and analysis mode match the protocol you intend to run.

.. _qs-run:

5. Run the Pipeline
-------------------

.. grid:: 1
   :gutter: 0

   .. grid-item-card::

      **Step 5 of 8** — Execute preprocessing, feature extraction, and analysis.

Use the tabs below for the full command matrix and focused examples.

.. tab-set::

   .. tab-item:: EEG Preprocessing

      Operates on BIDS EEG data and writes clean epochs and ICA logs
      to ``derivatives/preprocessed/eeg/``.

      **Output:** ``*_proc-clean_epo.fif``, ``*_proc-clean_events.tsv``,
      ``icalabel_task_*_log.csv``, ``pyprep_task_*_log.csv``.

      Modes:

      .. list-table::
         :header-rows: 1
         :widths: 22 78

         * - Mode
           - What it does
         * - ``full``
           - Runs bad-channel detection, ICA fitting, ICA labeling, and epoch
             creation in sequence. Use this for a fresh subject.
         * - ``bad-channels``
           - PyPREP bad-channel detection only (deviation + correlation, optional
             RANSAC). Updates ``channels.tsv`` and synchronizes bads across runs.
         * - ``ica``
           - Fits ICA (extended Infomax, 99% variance, 1 Hz high-pass) via
             MNE-BIDS-Pipeline, then labels components with ICLabel
             (threshold ``p > 0.8``; keeps ``brain`` and ``other``).
         * - ``epochs``
           - Creates epochs from clean ICA-applied data. Default window:
             ``tmin = -7.0 s``, ``tmax = 15.0 s``, baseline ``[-0.2, 0.0] s``,
             rejection via ``autoreject_local``.

      Examples:

      .. code-block:: bash

         eeg-pipeline preprocessing full --subject 0001
         eeg-pipeline preprocessing full --all-subjects
         eeg-pipeline preprocessing bad-channels --subject 0001 --ransac
         eeg-pipeline preprocessing ica --subject 0001
         eeg-pipeline preprocessing epochs --subject 0001 --tmin -7.0 --tmax 15.0

      See :doc:`../methods/eeg/preprocessing` for algorithm details and
      default parameter values.

   .. tab-item:: Feature Extraction

      Reads ``proc-clean`` epochs and writes one or more Parquet tables per family to
      ``derivatives/sub-<id>/eeg/features/<family>/``.

      **Output:** ``features_<family>.parquet`` + ``metadata/features_<family>.json``
      for each requested family.

      Modes:

      .. list-table::
         :header-rows: 1
         :widths: 18 82

         * - Mode
           - Purpose
         * - ``compute``
           - Extract features and write derivative tables.
         * - ``visualize``
           - Render summary plots from already-computed feature tables.

      Feature families:

      .. list-table::
         :header-rows: 1
         :widths: 26 74

         * - Family
           - What it computes
         * - ``power``
           - Morlet TFR band power, baseline-normalized (dB / percent / log-ratio)
         * - ``spectral``
           - PSD-based descriptors: peak frequency, bandwidth, spectral entropy
         * - ``aperiodic``
           - 1/f slope and offset (FOOOF/specparam); oscillatory peak rejection
         * - ``erp``
           - ERP component windows: N1/N2/P2 peak amplitude, latency, AUC, PTP
         * - ``erds``
           - Event-related desynchronization/synchronization vs. baseline
         * - ``ratios``
           - Band-power ratios (e.g. theta/alpha, delta/beta)
         * - ``asymmetry``
           - Hemispheric asymmetry indices on configured electrode pairs
         * - ``microstates``
           - GFP-peak K-means microstate templates: coverage, duration, occurrence rate
         * - ``connectivity``
           - wPLI, PLI, imCoh, PLV, AEC/AEC-orth; optional graph metrics and dynamic states
         * - ``directedconnectivity``
           - Phase Slope Index (PSI), DTF, PDC from MVAR model
         * - ``itpc``
           - Inter-trial phase clustering; CV-safe ``fold_global`` mode by default
         * - ``pac``
           - Phase–amplitude coupling (mean vector length); surrogate z-score
         * - ``sourcelocalization``
           - LCMV beamformer or eLORETA source estimates → ROI power / connectivity
         * - ``complexity``
           - LZC, permutation entropy, sample entropy, multiscale entropy (MSE)
         * - ``bursts``
           - Threshold-based transient oscillation detection: count, rate, duration, amplitude
         * - ``quality``
           - Per-trial QC: variance, SNR, muscle artifact index, PTP

      Feature extraction is config-driven. The families, bands, time windows,
      and spatial defaults are read from ``eeg_config.yaml`` unless overridden
      on the command line. The ``analysis_mode`` governs CV safety:

      .. list-table::
         :header-rows: 1
         :widths: 25 75

         * - Mode
           - When to use
         * - ``group_stats`` *(default)*
           - Descriptive statistics, group-level behavioral analyses
         * - ``trial_ml_safe``
           - Feature inputs to cross-validated ML; leakage-prone paths disabled

      .. code-block:: bash

         eeg-pipeline features compute --subject 0001
         eeg-pipeline features compute --subject 0001 --categories power connectivity aperiodic
         eeg-pipeline features compute --subject 0001 --analysis-mode trial_ml_safe
         eeg-pipeline features compute --subject 0001 --iaf-enabled
         eeg-pipeline features visualize --subject 0001

      See :doc:`../methods/eeg/features` for formulas, spatial modes, and
      CV hygiene details.

   .. tab-item:: Behavioral Analysis

      Reads ``proc-clean_events.tsv`` (behavioral targets/predictors) and the
      feature Parquet tables, and writes results to
      ``derivatives/sub-<id>/eeg/stats/``.

      **Output:** Per-analysis TSV/Parquet tables for correlations, regression
      coefficients, ICC values, condition contrasts, temporal statistics, and
      FDR summaries.

      Modes:

      .. list-table::
         :header-rows: 1
         :widths: 18 82

         * - Mode
           - Purpose
         * - ``compute``
           - Run the behavioral analysis DAG and write numerical outputs.
         * - ``visualize``
           - Render standardized plots from computed results.

      Analysis stages (``--computations`` selects a subset):

      .. list-table::
         :header-rows: 1
         :widths: 28 72

         * - Computation
           - What it produces
         * - ``trial_table``
           - Merges events and features on ``trial_id``; the alignment contract
         * - ``correlations``
           - Partial Spearman/Pearson :math:`r` between each feature and the
             behavioral outcome; permutation p-values; FDR correction
         * - ``predictor_residual``
           - Outcome residualized on predictor (spline OLS, AIC-selected)
             to isolate pain beyond intensity or analogous
         * - ``regression``
           - Trial-wise OLS feature ~ outcome with predictor controls,
             incremental :math:`R^2`, HC3 standard errors
         * - ``icc``
           - ICC(3,1) run-level reliability of each EEG feature
         * - ``condition``
           - Between-condition Welch t-test; Cohen's d / Hedges' g
         * - ``temporal``
           - Time-resolved correlations and cluster permutation tests

      .. code-block:: bash

         eeg-pipeline behavior compute --subject 0001
         eeg-pipeline behavior compute --all-subjects
         eeg-pipeline behavior compute --subject 0001 \
           --computations correlations condition temporal
         eeg-pipeline behavior visualize --subject 0001

      See :doc:`../methods/eeg/behavior` for the full DAG, partial correlation
      formulas, and permutation scheme.

   .. tab-item:: Machine Learning

      Reads feature Parquet tables and behavioral targets, runs nested
      cross-validation, and writes predictions and metrics to
      ``derivatives/ml/``.

      **Output:** ``results_summary.tsv``, ``predictions_all_subjects.tsv``,
      permutation p-values, SHAP importance tables, and figures.

      Outer CV is Leave-One-Subject-Out (LOSO); inner CV is GroupKFold
      (5 splits). All preprocessing statistics are estimated on the training
      fold only. The primary regression metric is subject-level Fisher-z
      aggregated Pearson correlation :math:`\bar{r}`.

      Modes:

      .. list-table::
         :header-rows: 1
         :widths: 28 72

         * - Mode
           - What it does
         * - ``regression``
           - Continuous outcome prediction with ElasticNet, Ridge, and Random
             Forest; Yeo-Johnson target transform
         * - ``classify``
           - Binary classification with SVM (RBF), logistic regression,
             Random Forest, or EEGNet CNN
         * - ``timegen``
           - Temporal generalization: train at one window, test at all others
         * - ``model_comparison``
           - Compare model families under identical CV with bootstrap CIs
         * - ``incremental_validity``
           - Quantify the added predictive value of EEG features over a baseline
         * - ``uncertainty``
           - Conformal prediction intervals with coverage guarantees
         * - ``shap``
           - SHAP-based feature importance (TreeExplainer / LinearExplainer)
         * - ``permutation``
           - Permutation test: re-run full nested CV on shuffled labels

      .. code-block:: bash

         eeg-pipeline ml regression --subject 0001 --subject 0002 --subject 0003
         eeg-pipeline ml regression --all-subjects
         eeg-pipeline ml classify --subject 0001 --subject 0002 \
           --classification-model svm
         eeg-pipeline ml timegen --subject 0001 --subject 0002
         eeg-pipeline ml model_comparison --subject 0001 --subject 0002
         eeg-pipeline ml incremental_validity --subject 0001 --subject 0002
         eeg-pipeline ml uncertainty --subject 0001 --subject 0002
         eeg-pipeline ml shap --subject 0001 --subject 0002
         eeg-pipeline ml permutation --subject 0001 --subject 0002

      See :doc:`../methods/eeg/machine_learning` for model formulas, CV
      scheme details, and the EEG–fMRI fusion regression target.

   .. tab-item:: fMRI

      .. note::

         The fMRI pipeline is **still under active development**. Expect
         evolving APIs and outputs; verify behavior against your dataset.

      Containerized fMRI preprocessing via fMRIPrep (Docker or Apptainer),
      followed by Nilearn GLM analyses and resting-state connectivity.

      **Output:** Contrast maps (NIfTI), trial-wise beta volumes, group
      inference maps, ROI connectivity matrices, provenance JSON.

      .. list-table::
         :header-rows: 1
         :widths: 26 18 56

         * - Command
           - Mode
           - What it does
         * - ``fmri``
           - ``preprocess``
           - Run fMRIPrep (image ``nipreps/fmriprep:25.2.4`` by default);
             outputs under ``derivatives/preprocessed/fmri/``
         * - ``fmri-analysis``
           - ``first-level``
           - Nilearn ``FirstLevelModel`` GLM; HRF ``spm``, drift ``cosine``,
             high-pass 0.008 Hz; writes z-score/t-stat/cope maps
         * - ``fmri-analysis``
           - ``second-level``
           - Group one-sample GLM from MNI cope/effect-size maps; optional max-T
             permutation inference (default 5000 permutations)
         * - ``fmri-analysis``
           - ``beta-series``
           - Least-squares-all (LSA) trial-wise beta estimates
         * - ``fmri-analysis``
           - ``lss``
           - Least-squares-separate (LSS) trial-wise beta estimates
         * - ``fmri-analysis``
           - ``rest``
           - Resting-state ROI timeseries extraction and connectivity matrix;
             requires atlas label image

      .. code-block:: bash

         eeg-pipeline fmri preprocess --subject 0001
         eeg-pipeline fmri preprocess --subject 0001 --engine apptainer

         eeg-pipeline fmri-analysis first-level --subject 0001 \
           --cond-a-value stimulation --cond-b-value fixation_rest
         eeg-pipeline fmri-analysis second-level \
           --subject 0001 --subject 0002 \
           --group-contrast-names stimulation_vs_rest
         eeg-pipeline fmri-analysis beta-series --subject 0001 \
           --cond-a-value stimulation --cond-b-value fixation_rest
         eeg-pipeline fmri-analysis lss --subject 0001 \
           --cond-a-value stimulation --cond-b-value fixation_rest
         eeg-pipeline fmri-analysis rest --subject 0001 \
           --atlas-labels-img /path/to/atlas.nii.gz \
           --atlas-labels-tsv /path/to/atlas.tsv

      .. note::

         ``fmri_contrast.enabled`` and ``fmri_group_level.enabled`` default to
         ``false`` in config, but CLI ``first-level`` / ``second-level`` modes
         run when explicitly invoked.

      See :doc:`../methods/fmri/pipeline` for the GLM specification,
      confound strategy, and signature readout methods.

   .. tab-item:: Plotting

      .. note::

         The plotting pipeline is **still under active development**. Plot
         catalog entries, CLI flags, and default figure styles may change.

      Renders visualization suites from computed features and statistical results.

      **Output:** PNG by default; add ``--formats svg pdf`` for additional formats.

      Modes:

      .. list-table::
         :header-rows: 1
         :widths: 18 82

         * - Mode
           - Purpose
         * - ``visualize``
           - Render the configured plot suites for a subject or at group level.
         * - ``tfr``
           - Time–frequency representations for a single subject.

      .. code-block:: bash

         eeg-pipeline plotting visualize --subject 0001 --all-plots
         eeg-pipeline plotting visualize --all-subjects --analysis-scope group
         eeg-pipeline plotting visualize --subject 0001 --formats png svg pdf
         eeg-pipeline plotting tfr --subject 0001

.. _qs-full-workflow:

6. Typical Full Workflow
------------------------

.. grid:: 1
   :gutter: 0

   .. grid-item-card::

      **Step 6 of 8** — End-to-end script from raw BIDS to ML results.

A complete subject-level run from raw BIDS data to ML-ready features:

.. code-block:: bash

   # 1. Validate inputs and confirm subject discovery
   eeg-pipeline validate quick
   eeg-pipeline info subjects

   # 2. Preprocess all subjects (bad channels → ICA → epochs)
   eeg-pipeline preprocessing full --all-subjects

   # 3. Extract features in ML-safe mode
   eeg-pipeline features compute --all-subjects \
     --analysis-mode trial_ml_safe \
     --categories power connectivity aperiodic erp erds itpc pac

   # 4. Run behavioral statistics (correlations + temporal)
   eeg-pipeline behavior compute --all-subjects \
     --computations correlations temporal

   # 5. Run cross-subject regression and feature importance
   eeg-pipeline ml regression --all-subjects
   eeg-pipeline ml shap --all-subjects

   # 6. Inspect results
   eeg-pipeline info ml-feature-space
   eeg-pipeline stats summary

For fMRI integration, run preprocessing and first-level GLM after step 1:

.. code-block:: bash

   eeg-pipeline fmri preprocess --all-subjects
   eeg-pipeline fmri-analysis first-level --all-subjects \
     --cond-a-value stimulation --cond-b-value fixation_rest
   eeg-pipeline fmri-analysis beta-series --all-subjects \
     --cond-a-value stimulation --cond-b-value fixation_rest

Then predict fMRI signature expression from EEG features:

.. code-block:: bash

   eeg-pipeline ml regression --all-subjects \
     --target fmri_signature \
     --fmri-signature-name SIGNATURE_A \
     --fmri-signature-method beta-series \
     --fmri-signature-metric dot

.. _qs-tui:

7. TUI Reference
----------------

.. grid:: 1
   :gutter: 0

   .. grid-item-card::

      **Step 7 of 8** — Use the interactive terminal UI for guided execution.

Build and launch the TUI from the repository root:

.. code-block:: bash

   cd eeg_pipeline/cli/tui && go build -o eeg-tui . && cd -
   ./eeg_pipeline/cli/tui/eeg-tui

The TUI covers the full workflow above — configuration, subject selection,
feature family and band selection, mode selection, and execution — all
through guided wizards. It is the recommended entry point for interactive use.

See :doc:`tui` for the complete reference, keyboard shortcuts, and persistence details.

.. _qs-docs:

8. Documentation Build
----------------------

.. grid:: 1
   :gutter: 0

   .. grid-item-card::

      **Step 8 of 8** — Build and validate these docs locally.

Build and validate the docs locally:

.. code-block:: bash

   python -m pip install -e ".[docs]"
   make docs
   make docs-check
