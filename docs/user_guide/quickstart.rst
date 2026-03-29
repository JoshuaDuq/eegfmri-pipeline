Quick Start
===========

This page gives a clean start path from install to outputs, then groups the
full CLI surface by analysis family. Use the tabs when you need the exact
command for a specific stage.

.. note::

   Prefer a guided interface? :doc:`tui` wraps the same CLI in an interactive
   wizard.

1. Install
----------

Create an isolated Python environment and install the package:

.. code-block:: bash

   git clone https://github.com/JoshuaDuq/eegfmri-pipeline.git
   cd eegfmri-pipeline
   python3.11 -m venv .venv311
   source .venv311/bin/activate
   pip install -e ".[dev,ml]"

2. Prepare Data
---------------

Place BIDS-formatted EEG data under ``data/bids_output/eeg/``. The expected
layout is documented in :doc:`data_layout`.

.. code-block:: text

   data/bids_output/eeg/
   ├── dataset_description.json
   └── sub-0001/eeg/
       ├── sub-0001_task-task_run-01_eeg.vhdr
       ├── sub-0001_task-task_run-01_events.tsv
       └── sub-0001_task-task_run-01_channels.tsv

3. Validate and Inspect
-----------------------

Start with read-only checks:

.. code-block:: bash

   eeg-pipeline validate quick
   eeg-pipeline info subjects
   eeg-pipeline info config
   eeg-pipeline info ml-feature-space

If you need a broader sweep, use the same family with explicit modes:

.. code-block:: bash

   eeg-pipeline validate all
   eeg-pipeline info features 0001
   eeg-pipeline info discover
   eeg-pipeline stats summary

4. Run the Pipeline
-------------------

Use the tabs below for the full command matrix and focused examples.

.. tab-set::

   .. tab-item:: EEG Preprocessing

      Modes:

      .. list-table::
         :header-rows: 1
         :widths: 22 78

         * - Mode
           - Purpose
         * - ``full``
           - Run bad-channel detection, ICA, and epoch creation in sequence.
         * - ``bad-channels``
           - Detect and interpolate bad channels only.
         * - ``ica``
           - Fit and apply ICA only.
         * - ``epochs``
           - Create epochs only.

      Examples:

      .. code-block:: bash

         eeg-pipeline preprocessing full --subject 0001
         eeg-pipeline preprocessing full --all-subjects
         eeg-pipeline preprocessing bad-channels --subject 0001 --ransac
         eeg-pipeline preprocessing ica --subject 0001
         eeg-pipeline preprocessing epochs --subject 0001 --tmin -7.0 --tmax 15.0

      See :doc:`../methods/eeg/preprocessing` for the full preprocessing
      contract.

   .. tab-item:: Feature Extraction

      Modes:

      .. list-table::
         :header-rows: 1
         :widths: 18 82

         * - Mode
           - Purpose
         * - ``compute``
           - Extract features and write derivatives.
         * - ``visualize``
           - Inspect already-computed feature tables.

      Feature categories:

      .. list-table::
         :header-rows: 1
         :widths: 24 76

         * - Category
           - Scope
         * - ``power``
           - Band-limited oscillatory power.
         * - ``spectral``
           - Spectral summary measures.
         * - ``ratios``
           - Band-power ratios.
         * - ``aperiodic``
           - 1/f background structure.
         * - ``connectivity``
           - Functional connectivity.
         * - ``directedconnectivity``
           - Directed connectivity.
         * - ``microstates``
           - Microstate sequence statistics.
         * - ``pac``
           - Phase-amplitude coupling.
         * - ``itpc``
           - Inter-trial phase coherence.
         * - ``erp``
           - ERP amplitudes.
         * - ``bursts``
           - Transient oscillatory bursts.
         * - ``complexity``
           - Signal complexity measures.
         * - ``asymmetry``
           - Hemispheric asymmetry indices.
         * - ``erds``
           - Event-related desynchronization/synchronization.
         * - ``quality``
           - Data quality indicators.
         * - ``sourcelocalization``
           - Source-space features.

      Examples:

      .. code-block:: bash

         eeg-pipeline features compute --subject 0001
         eeg-pipeline features compute --all-subjects
         eeg-pipeline features compute --all-subjects --categories power connectivity aperiodic
         eeg-pipeline features visualize --subject 0001

      See :doc:`../methods/eeg/features` for formulas, spatial modes, and
      configuration details.

   .. tab-item:: Behavioral Analysis

      Modes:

      .. list-table::
         :header-rows: 1
         :widths: 18 82

         * - Mode
           - Purpose
         * - ``compute``
           - Run behavioral analyses and write numerical outputs.
         * - ``visualize``
           - Render standardized plots from computed results.

      Examples:

      .. code-block:: bash

         eeg-pipeline behavior compute --subject 0001
         eeg-pipeline behavior compute --all-subjects
         eeg-pipeline behavior compute --subject 0001 --computations correlations condition temporal
         eeg-pipeline behavior visualize --subject 0001

      Behavioral analyses always operate on the trial table and use explicit
      trial-wise joins.

   .. tab-item:: Machine Learning

      Modes:

      .. list-table::
         :header-rows: 1
         :widths: 24 76

         * - Mode
           - Purpose
         * - ``regression``
           - Continuous outcome prediction with LOSO or within-subject CV.
         * - ``classify``
           - Binary classification.
         * - ``timegen``
           - Temporal generalization across windows.
         * - ``model_comparison``
           - Compare model families under the same CV scheme.
         * - ``incremental_validity``
           - Quantify the added value of EEG features over a baseline.
         * - ``uncertainty``
           - Conformal prediction intervals.
         * - ``shap``
           - SHAP-based feature importance.
         * - ``permutation``
           - Permutation-based feature importance.

      Examples:

      .. code-block:: bash

         eeg-pipeline ml regression --subject 0001 --subject 0002 --subject 0003
         eeg-pipeline ml classify --subject 0001 --subject 0002 --classification-model svm
         eeg-pipeline ml timegen --subject 0001 --subject 0002
         eeg-pipeline ml model_comparison --subject 0001 --subject 0002
         eeg-pipeline ml incremental_validity --subject 0001 --subject 0002
         eeg-pipeline ml uncertainty --subject 0001 --subject 0002
         eeg-pipeline ml shap --subject 0001 --subject 0002
         eeg-pipeline ml permutation --subject 0001 --subject 0002

   .. tab-item:: fMRI

      Commands:

      .. list-table::
         :header-rows: 1
         :widths: 22 78

         * - Command
           - Modes
         * - ``fmri``
           - ``preprocess``
         * - ``fmri-analysis``
           - ``first-level`` · ``second-level`` · ``beta-series`` · ``lss`` · ``rest``

      Examples:

      .. code-block:: bash

         eeg-pipeline fmri preprocess --subject 0001
         eeg-pipeline fmri preprocess --subject 0001 --engine apptainer
         eeg-pipeline fmri-analysis first-level --subject 0001 --cond-a-value stimulation --cond-b-value fixation_rest
         eeg-pipeline fmri-analysis second-level --subject 0001 --subject 0002
         eeg-pipeline fmri-analysis beta-series --subject 0001 --cond-a-value stimulation --cond-b-value fixation_rest
         eeg-pipeline fmri-analysis lss --subject 0001 --cond-a-value stimulation --cond-b-value fixation_rest
         eeg-pipeline fmri-analysis rest --subject 0001 --atlas-labels-img /path/to/atlas.nii.gz --atlas-labels-tsv /path/to/atlas.tsv

      See :doc:`../methods/fmri/pipeline` for the fMRI methods reference.

   .. tab-item:: Plotting

      Modes:

      .. list-table::
         :header-rows: 1
         :widths: 18 82

         * - Mode
           - Purpose
         * - ``visualize``
           - Render selected plot suites.
         * - ``tfr``
           - Time-frequency plots.

      Examples:

      .. code-block:: bash

         eeg-pipeline plotting visualize --subject 0001 --all-plots
         eeg-pipeline plotting visualize --all-subjects --analysis-scope group
         eeg-pipeline plotting tfr --subject 0001

5. Optional TUI
---------------

The Go TUI wraps the same CLI from the repository root.

.. code-block:: bash

   cd eeg_pipeline/cli/tui
   go build -o eeg-tui .
   ./eeg-tui

6. Documentation Build
----------------------

Build and validate the docs locally:

.. code-block:: bash

   python -m pip install -e ".[docs]"
   make docs
   make docs-check
