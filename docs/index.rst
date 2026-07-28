EEG–fMRI Analysis Pipeline
===========================

.. raw:: html

   <p class="hero-lede">
     Reproducible <strong>EEG</strong>, <strong>fMRI</strong>, and
     multimodal pipelines on <strong>BIDS</strong>-organized data.
     Six explicit stages — preprocessing, features, behavior, machine
     learning, fMRI, and source localization — accessible through a
     single CLI and an interactive terminal UI.
   </p>

.. grid:: 3
   :gutter: 3
   :class-container: nav-cards

   .. grid-item-card:: Install
      :link: install
      :link-type: doc

      Environment, FreeSurfer + MNE container, required env vars.

   .. grid-item-card:: Quick Start
      :link: user_guide/quickstart
      :link-type: doc

      Raw BIDS to derivatives in one end-to-end walkthrough.

   .. grid-item-card:: Methods
      :link: methods/index
      :link-type: doc

      Algorithms, configuration keys, and output schemas.

.. image:: screenshots/tui_main_menu.png
   :width: 640px
   :align: center
   :alt: Interactive TUI main menu — all pipeline stages reachable via guided wizards
   :class: hero-screenshot

----

Pipeline Stages
---------------

.. grid:: 3
   :gutter: 3
   :class-container: stage-grid

   .. grid-item-card:: 01 — Preprocessing
      :link: methods/eeg/preprocessing
      :link-type: doc

      PyPREP + RANSAC bad-channel detection, extended Infomax ICA,
      ICLabel pruning, autoreject epochs.

      ``proc-clean_epo.fif``

   .. grid-item-card:: 02 — Feature Extraction
      :link: methods/eeg/features
      :link-type: doc

      16 trial-level families: power, connectivity, aperiodic, ITPC,
      PAC, ERP, ERDS, complexity, microstates, source.

      ``features/<family>/*.parquet``

   .. grid-item-card:: 03 — Behavioral Statistics
      :link: methods/eeg/behavior
      :link-type: doc

      Partial correlations, OLS (HC3), ICC(3,1), condition contrasts,
      cluster tests, hierarchical Simes FDR.

      ``stats/*.tsv``

   .. grid-item-card:: 04 — Machine Learning
      :link: methods/eeg/machine_learning
      :link-type: doc

      Nested LOSO CV. ElasticNet / Ridge / RF regression. SVM / LR /
      RF / EEGNet classification. SHAP, conformal, permutation.

      ``ml/``

   .. grid-item-card:: 05 — fMRI Pipeline
      :link: methods/fmri/pipeline
      :link-type: doc

      Containerized fMRIPrep, Nilearn GLM, group inference, trial-wise
      betas (LSA / LSS), resting state, EEG–fMRI fusion.

      ``sub-*/fmri/``

   .. grid-item-card:: 06 — Source Localization
      :link: methods/eeg/source_localization
      :link-type: doc

      LCMV beamformer and eLORETA inverse solutions. fsaverage template
      or subject-specific fMRI-constrained.

      ``sourcelocalization/*.parquet``

----

Explore the Documentation
-------------------------

.. grid:: 4
   :gutter: 2
   :class-container: docs-nav

   .. grid-item-card:: User Guide
      :link: user_guide/index
      :link-type: doc

      Configuration, data layout, CLI, TUI.

   .. grid-item-card:: FAQ
      :link: faq
      :link-type: doc

      Common issues and resolutions.

   .. grid-item-card:: Glossary
      :link: glossary
      :link-type: doc

      Domain terminology and acronyms.

   .. grid-item-card:: Contributing
      :link: contributing
      :link-type: doc

      Development setup and standards.

.. raw:: html

   <p class="dev-status">
     <strong>Status</strong> — the fMRI pipeline remains under active
     development. Validate critical workflows
     after upgrades and confirm derivative paths before downstream use.
   </p>

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Getting Started

   install
   user_guide/quickstart
   faq
   glossary

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: User Guide

   user_guide/index
   native_eeg_fmri_artifact_correction
   pulse_artifact_correction_recovery

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Methods Reference

   methods/index

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Development

   contributing
