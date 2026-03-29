CLI Reference
=============

All commands follow the same pattern:

.. code-block:: bash

   eeg-pipeline <command> [mode] [--subject XXXX | --all-subjects] [options]

Append ``--help`` to any command for full option details.
Shared subject-selection flags (``--subject``, ``--all-subjects``, ``--task``,
``--dry-run``, ``--set``) are documented in :doc:`../subject_selection`.

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

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: :octicon:`pulse` Preprocessing
      :link: preprocessing
      :link-type: doc

      ``full`` · ``bad-channels`` · ``ica`` · ``epochs``

      Filtering, resampling, resting-state mode, PyPREP and ICLabel options.

   .. grid-item-card:: :octicon:`graph` Feature Extraction
      :link: features
      :link-type: doc

      ``compute`` · ``visualize``

      Category selection, spatial transforms, IAF mode, analysis mode.

   .. grid-item-card:: :octicon:`table` Behavioural Statistics
      :link: behavior
      :link-type: doc

      ``compute`` · ``visualize``

      Correlation, regression, ICC, and condition comparison flags.

   .. grid-item-card:: :octicon:`dependabot` Machine Learning
      :link: ml
      :link-type: doc

      ``regression`` · ``classify`` · ``timegen`` · ``shap`` · ``permutation`` · …

      Full LOSO modeling suite, feature filtering, and harmonization.

   .. grid-item-card:: :octicon:`container` fMRI Preprocessing
      :link: fmri_preprocessing
      :link-type: doc

      ``preprocess``

      fMRIPrep container flags, output spaces, memory and thread controls.

   .. grid-item-card:: :octicon:`workflow` fMRI Analysis
      :link: fmri_analysis
      :link-type: doc

      ``first-level`` · ``second-level`` · ``beta-series`` · ``lss`` · ``rest``

      GLM specification, confound strategy, beta estimation, connectivity.

   .. grid-item-card:: :octicon:`image` Plotting
      :link: plotting
      :link-type: doc

      ``visualize`` · ``tfr``

      40+ plot types, format flags, group vs. subject mode, style overrides.

   .. grid-item-card:: :octicon:`shield-check` Validation
      :link: validation
      :link-type: doc

      ``bids`` · ``derivatives`` · ``events`` · ``features``

      Data integrity checks at every pipeline stage.

   .. grid-item-card:: :octicon:`info` Stats & Info
      :link: stats_info
      :link-type: doc

      ``stats`` · ``info``

      Pipeline-wide coverage summaries and read-only state inspection.
