EEG–fMRI Analysis Pipeline
===========================

The EEG–fMRI Analysis Pipeline is an integrated research software stack for
EEG, fMRI, and multimodal EEG–fMRI studies. It operates on :term:`BIDS`-organized
data and produces documented derivatives for preprocessing, feature extraction,
behavioral statistics, machine learning, fMRI :term:`GLM`, and EEG source
localization workflows. All stages are accessible through a single CLI and an
interactive terminal UI (TUI).

The project began as an attempt to replace ad hoc research scripting with a
single, documented workflow that makes analysis steps easier to inspect,
reproduce, and rerun. The current emphasis is on explicit data contracts,
method-specific documentation, and practical day-to-day usability for research
work.

.. admonition:: Development status

   The project is under active development. The **plotting pipeline** and parts
   of the **fMRI analysis** command surface remain in flux, and some interfaces
   or derivative layouts may change between releases.

   Feedback from real datasets is useful. If a workflow is unclear, incomplete,
   or produces unexpected outputs, open an issue or discussion so the behavior
   can be tightened and documented.

.. figure:: screenshots/tui_main_menu.png
   :width: 760px
   :align: center
   :alt: Interactive TUI main menu

   The TUI main menu: all pipeline stages accessible through guided wizards,
   no terminal commands required.

----

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Installation
      :link: install
      :link-type: doc

      Environment setup, optional TUI build, FreeSurfer + MNE Docker image,
      and required environment variables.

   .. grid-item-card:: Quick Start
      :link: user_guide/quickstart
      :link-type: doc

      End-to-end operational walkthrough from installation to derivatives:
      validation, preprocessing, feature extraction, ML, and fMRI.

   .. grid-item-card:: User Guide
      :link: user_guide/index
      :link-type: doc

      Operational reference for input layout, YAML configuration, command
      usage, derivative locations, and the interactive TUI.

   .. grid-item-card:: Methods Reference
      :link: methods/index
      :link-type: doc

      Algorithmic reference: methods, notation, configuration keys, and output
      schemas for each analysis stage.

   .. grid-item-card:: FAQ
      :link: faq
      :link-type: doc

      Common operational issues: subject discovery, ICA failures, epoch
      rejection, aperiodic fits, NaN metrics, fMRIPrep paths, and TUI usage.

   .. grid-item-card:: Glossary
      :link: glossary
      :link-type: doc

      Definitions for :term:`BIDS`, :term:`ICA`, :term:`wPLI`, :term:`PAC`,
      :term:`LOSO`, :term:`HRF`, and other domain-specific terms used
      throughout the documentation.

----

Pipeline Stages
---------------

The pipeline is organized as explicit derivative-producing stages. Each stage
documents its expected inputs, method family, and primary outputs so downstream
analyses can be validated against concrete files rather than implicit workflow
state.

.. grid:: 3
   :gutter: 2

   .. grid-item-card:: 1 · Preprocessing
      :link: methods/eeg/preprocessing
      :link-type: doc

      Bad channel detection (PyPREP + RANSAC), ICA fitting (extended
      Infomax), ICLabel classification (p > 0.8), epoch creation, and
      autoreject. Writes ``proc-clean_epo.fif`` + ``proc-clean_events.tsv``.

   .. grid-item-card:: 2 · Feature Extraction
      :link: methods/eeg/features
      :link-type: doc

      16 trial-level families: power, connectivity, aperiodic, ITPC, PAC,
      ERP, ERDS, complexity, microstates, source localization, and more.
      One row per trial; Parquet output with full metadata.

   .. grid-item-card:: 3 · Behavioral Statistics
      :link: methods/eeg/behavior
      :link-type: doc

      Trial-level partial correlations, predictor residualization, OLS
      regression (HC3), ICC(3,1) reliability, condition contrasts, temporal
      cluster tests, and hierarchical Simes FDR.

   .. grid-item-card:: 4 · Machine Learning
      :link: methods/eeg/machine_learning
      :link-type: doc

      Nested LOSO CV; ElasticNet / Ridge / RF regression; SVM / LR / RF /
      EEGNet classification; SHAP importance; conformal intervals;
      permutation inference. Primary metric: subject-level Fisher-z :math:`r`.

   .. grid-item-card:: 5 · fMRI Pipeline
      :link: methods/fmri/pipeline
      :link-type: doc

      Containerized fMRIPrep, Nilearn first-level GLM, group-level inference,
      trial-wise beta estimation (LSA / LSS), resting-state connectivity, and
      multivariate EEG–fMRI signature fusion. **Still under active development.**

   .. grid-item-card:: 6 · Source Localization
      :link: methods/eeg/source_localization
      :link-type: doc

      LCMV beamformer and eLORETA inverse solutions. Fast fsaverage template
      path (no MRI required) and subject-specific fMRI-constrained path.

.. note::

   The **fMRI pipeline** and the **plotting** command (``eeg-pipeline plotting``)
   are **still under active development**. Validate critical workflows after
   upgrades and confirm derivative paths before using them in downstream
   analyses or manuscripts.

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
