Methods Reference
=================

.. raw:: html

   <p class="hero-lede">
     Scientific and algorithmic reference for each pipeline stage. Each page
     defines the method contract: <strong>notation</strong>, <strong>model
     choices</strong>, <strong>configuration keys</strong>, and <strong>output
     schemas</strong>.
   </p>

.. toctree::
   :maxdepth: 2
   :hidden:

   eeg/index
   fmri/index

EEG
---

.. grid:: 3
   :gutter: 3
   :class-container: nav-cards

   .. grid-item-card:: Preprocessing
      :link: eeg/preprocessing
      :link-type: doc

      PyPREP bad-channel detection, ICA fitting, ICLabel classification,
      epoch creation and autoreject. Task-based and resting-state modes.

   .. grid-item-card:: Feature Extraction
      :link: eeg/features
      :link-type: doc

      16 families with formulas, config keys, and derivative schemas:
      power, connectivity, aperiodic, complexity, microstates, source ROI.

   .. grid-item-card:: Behavioral Statistics
      :link: eeg/behavior
      :link-type: doc

      Trial-level correlations, OLS regression (HC3), ICC, condition
      contrasts, temporal cluster tests, FDR control.

   .. grid-item-card:: Machine Learning
      :link: eeg/machine_learning
      :link-type: doc

      Nested LOSO CV, permutation inference, SHAP importance, temporal
      generalization, conformal intervals.

   .. grid-item-card:: Source Localization
      :link: eeg/source_localization
      :link-type: doc

      LCMV beamformer and eLORETA. fsaverage template path (no MRI
      required) or subject-specific fMRI-constrained path.

fMRI
----

.. grid:: 2
   :gutter: 3
   :class-container: nav-cards

   .. grid-item-card:: Raw-to-BIDS Contract
      :link: fmri/raw_to_bids
      :link-type: doc

      BIDS input contract, ``events.tsv`` requirements, validation
      expectations for raw fMRI datasets.

   .. grid-item-card:: fMRI Analysis Pipeline
      :link: fmri/pipeline
      :link-type: doc

      fMRIPrep preprocessing, first-level GLM, group inference,
      trial-wise betas (LSA / LSS), resting-state connectivity,
      multivariate EEG–fMRI signature readouts.
