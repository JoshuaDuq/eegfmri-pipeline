EEG Methods
===========

.. raw:: html

   <p class="hero-lede">
     Scientific reference for every EEG analysis stage. Each page covers
     <strong>notation</strong>, <strong>methods</strong>, <strong>configuration
     keys</strong>, and <strong>output schemas</strong>.
   </p>

.. toctree::
   :maxdepth: 1
   :hidden:

   preprocessing
   features
   behavior
   machine_learning
   source_localization

.. grid:: 2
   :gutter: 3
   :class-container: nav-cards

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
