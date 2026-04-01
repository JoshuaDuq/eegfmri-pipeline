EEG Methods
===========

.. container:: page-intro

   Scientific reference for every EEG analysis stage. Each page covers
   notation, methods, configuration keys, and output schemas for the EEG
   portions of the pipeline.

.. toctree::
   :maxdepth: 1
   :hidden:

   preprocessing
   features
   behavior
   machine_learning
   source_localization

.. rst-class:: section-kicker

At A Glance

.. rst-class:: summary-grid

.. grid:: 3
   :gutter: 2

   .. grid-item-card:: Inputs

      Primarily BIDS EEG plus derivative epochs and trial tables produced by preprocessing.

   .. grid-item-card:: Outputs

      FIF derivatives, Parquet feature tables, statistics tables, model artifacts, and ROI summaries.

   .. grid-item-card:: Use these pages when

      You need method assumptions, parameter meanings, or precise derivative expectations.

.. rst-class:: dashboard-grid

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Preprocessing
      :link: preprocessing
      :link-type: doc

      Bad channel detection (PyPREP), ICA fitting and ICLabel
      classification. Modes: ``bad-channels``, ``ica``, ``epochs``, ``full``.

   .. grid-item-card:: Feature Extraction
      :link: features
      :link-type: doc

      16 trial-level feature families with full mathematical derivations —
      power, connectivity, aperiodic, ITPC, PAC, complexity, ERP, ERDS,
      microstates, spectral, and source ROI.

   .. grid-item-card:: Behavioral Statistics
      :link: behavior
      :link-type: doc

      Trial-level correlations, regression, reliability (ICC), condition
      comparisons, and time-resolved analyses with FDR correction.

   .. grid-item-card:: Machine Learning
      :link: machine_learning
      :link-type: doc

      LOSO cross-validation for regression and classification. Feature
      selection, permutation testing, SHAP importance, and temporal
      generalization.

   .. grid-item-card:: Source Localization
      :link: source_localization
      :link-type: doc

      LCMV and eLORETA inverse solutions. Fast template path (fsaverage)
      and subject-specific fMRI-constrained path.
