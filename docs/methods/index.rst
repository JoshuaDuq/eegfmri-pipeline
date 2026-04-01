Methods Reference
=================

.. container:: page-intro

   Scientific and algorithmic reference for each pipeline stage. These pages
   define the method-level contract for the software: notation, model choices,
   configuration keys, and output schemas.

.. toctree::
   :maxdepth: 2
   :hidden:

   eeg/index
   fmri/index

.. rst-class:: section-kicker

Use This Section For

.. rst-class:: summary-grid

.. grid:: 3
   :gutter: 2

   .. grid-item-card:: Checking method details
      :link: eeg/index
      :link-type: doc

      Use these pages when you need algorithm choices, defaults, and derivations.

   .. grid-item-card:: Validating outputs
      :link: ../user_guide/output_formats
      :link-type: doc

      Pair methods pages with output schemas when confirming downstream analyses.

   .. grid-item-card:: Mapping commands to methods
      :link: ../user_guide/cli/index
      :link-type: doc

      Use the CLI reference when you need the exact runtime interface for a method.

EEG
---

.. rst-class:: dashboard-grid

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Preprocessing
      :link: eeg/preprocessing
      :link-type: doc

      Bad-channel detection (PyPREP), ICA fitting, ICLabel
      classification, epoch creation, and rejection.
      Covers both task-based and resting-state modes.

   .. grid-item-card:: Feature Extraction
      :link: eeg/features
      :link-type: doc

      Sixteen feature families with formulas, configuration keys,
      and derivative schemas: power, connectivity, aperiodic,
      complexity, microstates, source ROI, and more.

   .. grid-item-card:: Behavioral Statistics
      :link: eeg/behavior
      :link-type: doc

      Trial-level correlations, regression, ICC, condition
      comparisons, and time-resolved inference with FDR control.

   .. grid-item-card:: Machine Learning
      :link: eeg/machine_learning
      :link-type: doc

      LOSO cross-validation, permutation inference, SHAP feature
      importance, temporal generalization, and performance reporting.

   .. grid-item-card:: Source Localization
      :link: eeg/source_localization
      :link-type: doc

      LCMV and eLORETA inverse solutions. Template (fsaverage)
      and subject-specific fMRI-constrained paths.

fMRI
----

.. rst-class:: dashboard-grid

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Raw-to-BIDS Contract
      :link: fmri/raw_to_bids
      :link-type: doc

      BIDS input contract, ``events.tsv`` requirements,
      and validation expectations for raw fMRI datasets.

   .. grid-item-card:: fMRI Analysis Pipeline
      :link: fmri/pipeline
      :link-type: doc

      fMRIPrep preprocessing, first-level GLM, group inference,
      trial-wise beta estimation, resting-state connectivity,
      and multivariate signature readouts.
