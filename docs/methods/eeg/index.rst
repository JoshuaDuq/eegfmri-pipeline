EEG Methods
===========

Scientific reference for every EEG analysis stage. Each page covers notation,
methods, configuration keys, and output schemas.

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

   .. grid-item-card:: :octicon:`pulse` Preprocessing
      :link: preprocessing
      :link-type: doc

      Bad channel detection (PyPREP), ICA fitting and ICLabel
      classification. Modes: ``bad-channels``, ``ica``, ``epochs``, ``full``.

   .. grid-item-card:: :octicon:`graph` Feature Extraction
      :link: features
      :link-type: doc

      16 trial-level feature families with full mathematical derivations —
      power, connectivity, aperiodic, ITPC, PAC, complexity, ERP, ERDS,
      microstates, spectral, and source ROI.

   .. grid-item-card:: :octicon:`table` Behavioural Statistics
      :link: behavior
      :link-type: doc

      Trial-level correlations, regression, reliability (ICC), condition
      comparisons, and time-resolved analyses with FDR correction.

   .. grid-item-card:: :octicon:`dependabot` Machine Learning
      :link: machine_learning
      :link-type: doc

      LOSO cross-validation for regression and classification. Feature
      selection, permutation testing, SHAP importance, and temporal
      generalisation.

   .. grid-item-card:: :octicon:`location` Source Localisation
      :link: source_localization
      :link-type: doc

      LCMV and eLORETA inverse solutions. Fast template path (fsaverage)
      and subject-specific fMRI-constrained path.
