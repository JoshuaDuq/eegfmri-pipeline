EEG–fMRI Analysis Pipeline
===========================

.. raw:: html

   <p class="hero-intro">
     An integrated research software stack for EEG, fMRI, and multimodal EEG–fMRI
     studies. Operates on <strong>BIDS</strong>-organized data and produces documented
     derivatives across six explicit pipeline stages — all accessible through a single
     CLI and an interactive terminal UI.
   </p>

.. admonition:: Development status

   The **plotting pipeline** and parts of the **fMRI analysis** command surface remain
   in flux. Interfaces or derivative layouts may change between releases. Feedback from
   real datasets is welcome — open an issue if a workflow is unclear or produces
   unexpected outputs.

.. figure:: screenshots/tui_main_menu.png
   :width: 720px
   :align: center
   :alt: Interactive TUI main menu

   The TUI main menu — all pipeline stages accessible through guided wizards.

----

.. grid:: 3
   :gutter: 2

   .. grid-item-card:: Installation
      :link: install
      :link-type: doc

      Environment setup, optional TUI build, FreeSurfer + MNE Docker image,
      and required environment variables.

   .. grid-item-card:: Quick Start
      :link: user_guide/quickstart
      :link-type: doc

      End-to-end walkthrough from raw BIDS data to derivatives: validation,
      preprocessing, feature extraction, ML, and fMRI.

   .. grid-item-card:: User Guide
      :link: user_guide/index
      :link-type: doc

      Input layout, YAML configuration, command usage, derivative locations,
      and the interactive TUI.

   .. grid-item-card:: Methods Reference
      :link: methods/index
      :link-type: doc

      Algorithms, notation, configuration keys, and output schemas for each
      analysis stage.

   .. grid-item-card:: FAQ
      :link: faq
      :link-type: doc

      Subject discovery, ICA failures, epoch rejection, aperiodic fits,
      NaN metrics, fMRIPrep paths, and TUI usage.

   .. grid-item-card:: Glossary
      :link: glossary
      :link-type: doc

      Definitions for :term:`BIDS`, :term:`ICA`, :term:`wPLI`, :term:`PAC`,
      :term:`LOSO`, :term:`HRF`, and other domain terms.

----

Pipeline Stages
---------------

Six derivative-producing stages with explicit input/output contracts.

.. grid:: 3
   :gutter: 2

   .. grid-item-card:: 01 — Preprocessing
      :link: methods/eeg/preprocessing
      :link-type: doc

      PyPREP + RANSAC bad-channel detection · extended Infomax ICA ·
      ICLabel (p > 0.8) · autoreject epochs.
      → ``proc-clean_epo.fif`` + ``proc-clean_events.tsv``

   .. grid-item-card:: 02 — Feature Extraction
      :link: methods/eeg/features
      :link-type: doc

      16 trial-level families: power, connectivity, aperiodic, ITPC, PAC,
      ERP, ERDS, complexity, microstates, source localization, and more.
      → ``features/<family>/*.parquet``

   .. grid-item-card:: 03 — Behavioral Statistics
      :link: methods/eeg/behavior
      :link-type: doc

      Partial correlations · OLS regression (HC3) · ICC(3,1) reliability ·
      condition contrasts · temporal cluster tests · hierarchical Simes FDR.
      → ``stats/*.tsv``

   .. grid-item-card:: 04 — Machine Learning
      :link: methods/eeg/machine_learning
      :link-type: doc

      Nested LOSO CV · ElasticNet / Ridge / RF regression · SVM / LR / RF /
      EEGNet classification · SHAP importance · conformal intervals ·
      permutation inference. → ``ml/``

   .. grid-item-card:: 05 — fMRI Pipeline
      :link: methods/fmri/pipeline
      :link-type: doc

      Containerized fMRIPrep · Nilearn GLM · group inference · trial-wise
      betas (LSA / LSS) · resting-state connectivity · EEG–fMRI fusion.
      → ``sub-*/fmri/`` · ``group/fmri/``

      *Under active development.*

   .. grid-item-card:: 06 — Source Localization
      :link: methods/eeg/source_localization
      :link-type: doc

      LCMV beamformer and eLORETA inverse solutions. Fast fsaverage template
      (no MRI required) or subject-specific fMRI-constrained path.
      → ``sourcelocalization/*.parquet``

.. note::

   The **fMRI pipeline** and the **plotting** command are **still under active
   development**. Validate critical workflows after upgrades and confirm derivative
   paths before use in downstream analyses or manuscripts.

----

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

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Methods Reference

   methods/index

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: API Reference

   api/index

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Development

   contributing
