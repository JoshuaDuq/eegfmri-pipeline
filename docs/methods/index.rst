Methods Reference
=================

Scientific documentation for every pipeline stage. Each page provides
notation, method details, configuration keys, and output schemas.

.. toctree::
   :maxdepth: 2
   :hidden:

   eeg/index
   fmri/index

EEG
---

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: :octicon:`pulse` Preprocessing
      :link: eeg/preprocessing
      :link-type: doc

      Bad channel detection (PyPREP), ICA fitting and ICLabel
      classification, epoch creation and rejection.
      Supports task-based and resting-state modes.

   .. grid-item-card:: :octicon:`graph` Feature Extraction
      :link: eeg/features
      :link-type: doc

      16 feature families with full mathematical derivations —
      power, connectivity, aperiodic, complexity, microstates,
      source ROI, and more. Supports task-based and resting-state
      paradigms.

   .. grid-item-card:: :octicon:`table` Behavioral Statistics
      :link: eeg/behavior
      :link-type: doc

      Trial-level correlations, regression, ICC, condition
      comparisons, and time-resolved analyses with FDR correction.

   .. grid-item-card:: :octicon:`dependabot` Machine Learning
      :link: eeg/machine_learning
      :link-type: doc

      LOSO cross-validation, permutation inference, SHAP feature
      importance, and temporal generalization.

   .. grid-item-card:: :octicon:`location` Source Localization
      :link: eeg/source_localization
      :link-type: doc

      LCMV and eLORETA inverse solutions. Template (fsaverage)
      and subject-specific fMRI-constrained paths.

fMRI
----

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: :octicon:`file-code` Raw-to-BIDS Contract
      :link: fmri/raw_to_bids
      :link-type: doc

      BIDS input contract, ``events.tsv`` requirements,
      and BIDS validation steps.

   .. grid-item-card:: :octicon:`workflow` fMRI Analysis Pipeline
      :link: fmri/pipeline
      :link-type: doc

      fMRIPrep preprocessing, first-level GLM, group inference,
      trial-wise beta estimation, resting-state connectivity,
      and multivariate signature readouts.
