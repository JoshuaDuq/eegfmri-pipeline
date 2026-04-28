CLI Reference
=============

.. raw:: html

   <p class="hero-lede">
     Full command surface for scripting, headless batch jobs, and option
     overrides. For interactive use, the
     <a href="../tui.html">TUI</a> assembles these commands through guided
     wizards — no flags to memorize.
   </p>

All commands follow the same pattern:

.. code-block:: bash

   eeg-pipeline <command> [mode] [--subject XXXX | --all-subjects] [options]

Append ``--help`` to any command for full option details.

Shared Options
--------------

Most analysis commands accept these subject-selection and runtime flags:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Option
     - Description
   * - ``--subject XXXX`` / ``-s XXXX``
     - Single subject; repeat the flag for multiple subjects
   * - ``--all-subjects``
     - Process every discovered subject
   * - ``--group all`` or ``--group A,B,C``
     - Select a named group or comma-separated subject list
   * - ``--task`` / ``-t``
     - Override the task label from config
   * - ``--dry-run``
     - Preview work without executing
   * - ``--json``
     - Emit JSON output for scripting or the TUI
   * - ``--progress-json``
     - Emit progress events as JSON lines
   * - ``--set KEY=VALUE``
     - Override config values at runtime (see :doc:`../configuration`)
   * - ``--bids-root``
     - Override ``paths.bids_root`` at runtime
   * - ``--bids-fmri-root``
     - Override ``paths.bids_fmri_root`` at runtime
   * - ``--bids-rest-root``
     - Override ``paths.bids_rest_root`` at runtime (resting-state dataset root; EEG and fMRI rest modes)
   * - ``--deriv-root``
     - Override ``paths.deriv_root`` at runtime
   * - ``--deriv-rest-root``
     - Override ``paths.deriv_rest_root`` at runtime (resting-state derivatives root; EEG and fMRI rest modes)

.. note::

   ``validate`` uses ``--subjects``; ``info features`` takes a positional subject ID.
   ``info``, ``stats``, and ``validate`` are reduced read-only/validation
   interfaces and do not accept ``--set`` or path override flags.


.. note::

   The **fMRI** commands (``fmri``, ``fmri-analysis``) and the **plotting**
   command are **still under active development**. Expect evolving options and
   outputs; pin a release or check the changelog when upgrading.

.. toctree::
   :maxdepth: 1
   :hidden:

   preprocessing
   features
   behavior
   ml
   fmri_preprocessing
   fmri_analysis
   plotting
   validation
   stats_info

Command Matrix
--------------

.. list-table::
   :header-rows: 1
   :widths: 22 28 50

   * - Command family
     - Modes
     - Purpose
   * - ``preprocessing``
     - ``full`` · ``bad-channels`` · ``ica`` · ``epochs``
     - EEG preprocessing stages from bad-channel detection to epoch creation.
   * - ``features``
     - ``compute`` · ``visualize``
     - Extract 16 EEG feature families or visualize existing tables.
   * - ``behavior``
     - ``compute`` · ``visualize``
     - Behavioral statistics and plots derived from trial tables.
   * - ``ml``
     - ``regression`` · ``classify`` · ``timegen`` · ``model_comparison`` · ``incremental_validity`` · ``uncertainty`` · ``shap`` · ``permutation``
     - Trial-level predictive modeling and feature attribution.
   * - ``fmri``
     - ``preprocess``
     - Containerized fMRIPrep-style preprocessing.
   * - ``fmri-analysis``
     - ``first-level`` · ``second-level`` · ``beta-series`` · ``lss`` · ``rest``
     - Event-related GLM, group inference, trial-wise betas, and resting-state connectivity.
   * - ``plotting``
     - ``visualize`` · ``tfr``
     - Render the curated plot catalog and TFR outputs.
   * - ``validate``
     - ``quick`` · ``all`` · ``epochs`` · ``features`` · ``behavior`` · ``bids``
     - Validate data integrity and schema consistency.
   * - ``info``
     - ``subjects`` · ``features`` · ``config`` · ``version`` · ``plotters`` · ``discover`` · ``rois`` · ``fmri-conditions`` · ``fmri-columns`` · ``multigroup-stats`` · ``ml-feature-space``
     - Read-only discovery for data availability, configuration, and feature-space structure.
   * - ``stats``
     - ``summary`` · ``subjects`` · ``features`` · ``storage`` · ``timeline``
     - Project-wide summaries and storage/timeline inspection.

.. grid:: 2
   :gutter: 3
   :class-container: nav-cards

   .. grid-item-card:: Preprocessing
      :link: preprocessing
      :link-type: doc

      ``full`` · ``bad-channels`` · ``ica`` · ``epochs``

      Filtering, resampling, resting-state mode, PyPREP and ICLabel options.

   .. grid-item-card:: Feature Extraction
      :link: features
      :link-type: doc

      ``compute`` · ``visualize``

      Category selection, spatial transforms, IAF mode, analysis mode.

   .. grid-item-card:: Behavioral Statistics
      :link: behavior
      :link-type: doc

      ``compute`` · ``visualize``

      Correlation, regression, ICC, and condition comparison flags.

   .. grid-item-card:: Machine Learning
      :link: ml
      :link-type: doc

      ``regression`` · ``classify`` · ``timegen`` · ``shap`` · ``permutation`` · …

      Full LOSO modeling suite, feature filtering, and harmonization.

   .. grid-item-card:: fMRI Preprocessing
      :link: fmri_preprocessing
      :link-type: doc

      ``preprocess``

      fMRIPrep container flags, output spaces, memory and thread controls.

   .. grid-item-card:: fMRI Analysis
      :link: fmri_analysis
      :link-type: doc

      ``first-level`` · ``second-level`` · ``beta-series`` · ``lss`` · ``rest``

      GLM specification, confound strategy, beta estimation, connectivity.

   .. grid-item-card:: Plotting
      :link: plotting
      :link-type: doc

      ``visualize`` · ``tfr``

      40+ plot types, format flags, group vs. subject mode, style overrides.

   .. grid-item-card:: Validation
      :link: validation
      :link-type: doc

      ``bids`` · ``epochs`` · ``features`` · ``behavior`` · ``quick`` · ``all``

      Data integrity checks at every pipeline stage.

   .. grid-item-card:: Stats & Info
      :link: stats_info
      :link-type: doc

      ``stats`` and ``info`` modes for coverage summaries and read-only state inspection.
