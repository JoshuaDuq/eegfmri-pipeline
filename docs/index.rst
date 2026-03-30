EEG–fMRI Analysis Pipeline
===========================

A modular analysis suite for multimodal neuroimaging research. The pipeline
takes :term:`BIDS`-formatted EEG and fMRI recordings through six stages —
automated preprocessing, 16 EEG feature families, behavioral statistics,
machine learning, fMRI :term:`GLM`, and source localization — from a single
CLI or interactive TUI, with reproducible outputs and explicit CV safeguards
throughout.

.. figure:: screenshots/tui_main_menu.png
   :width: 760px
   :align: center
   :alt: Interactive TUI — pipeline stages at a glance

----

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: :octicon:`download` Installation
      :link: install
      :link-type: doc

      Python environment, TUI binary, Docker image for FreeSurfer + MNE,
      and environment variables.

   .. grid-item-card:: :octicon:`rocket` Quick Start
      :link: user_guide/quickstart
      :link-type: doc

      Full walkthrough from install to results — pipeline overview, validation,
      preprocessing, features, ML, fMRI, and a complete chained workflow.

   .. grid-item-card:: :octicon:`book` User Guide
      :link: user_guide/index
      :link-type: doc

      Data layout, YAML configuration, subject selection, CLI command
      reference, output formats, and the interactive TUI.

   .. grid-item-card:: :octicon:`beaker` Methods Reference
      :link: methods/index
      :link-type: doc

      Full scientific documentation — notation, formulas, configuration keys,
      and output schemas for every pipeline stage.

   .. grid-item-card:: :octicon:`comment-discussion` FAQ
      :link: faq
      :link-type: doc

      Common issues: subject discovery, ICA failures, epoch rejection,
      aperiodic fits, NaN metrics, fMRIPrep paths, and the TUI.

   .. grid-item-card:: :octicon:`list-unordered` Glossary
      :link: glossary
      :link-type: doc

      Definitions for :term:`BIDS`, :term:`ICA`, :term:`wPLI`, :term:`PAC`,
      :term:`LOSO`, :term:`HRF`, and all other domain terms.

----

Pipeline Stages
---------------

.. grid:: 3
   :gutter: 2

   .. grid-item-card:: :octicon:`cpu` 1 · Preprocessing
      :link: methods/eeg/preprocessing
      :link-type: doc

      Bad channel detection (PyPREP + RANSAC), ICA fitting (extended
      Infomax), ICLabel classification (p > 0.8), epoch creation, and
      autoreject. Writes ``proc-clean_epo.fif`` + ``proc-clean_events.tsv``.

   .. grid-item-card:: :octicon:`graph` 2 · Feature Extraction
      :link: methods/eeg/features
      :link-type: doc

      16 trial-level families: power, connectivity, aperiodic, ITPC, PAC,
      ERP, ERDS, complexity, microstates, source localization, and more.
      One row per trial; Parquet output with full metadata.

   .. grid-item-card:: :octicon:`table` 3 · Behavioral Statistics
      :link: methods/eeg/behavior
      :link-type: doc

      Trial-level partial correlations, predictor residualization, OLS
      regression (HC3), ICC(3,1) reliability, condition contrasts, temporal
      cluster tests, and hierarchical Simes FDR.

   .. grid-item-card:: :octicon:`dependabot` 4 · Machine Learning
      :link: methods/eeg/machine_learning
      :link-type: doc

      Nested LOSO CV; ElasticNet / Ridge / RF regression; SVM / LR / RF /
      EEGNet classification; SHAP importance; conformal intervals;
      permutation inference. Primary metric: subject-level Fisher-z :math:`r`.

   .. grid-item-card:: :octicon:`telescope` 5 · fMRI Pipeline
      :link: methods/fmri/pipeline
      :link-type: doc

      Containerized fMRIPrep, Nilearn first-level GLM, group-level inference,
      trial-wise beta estimation (LSA / LSS), resting-state connectivity, and
      multivariate EEG–fMRI signature fusion. **Still under active development.**

   .. grid-item-card:: :octicon:`location` 6 · Source Localization
      :link: methods/eeg/source_localization
      :link-type: doc

      LCMV beamformer and eLORETA inverse solutions. Fast fsaverage template
      path (no MRI required) and subject-specific fMRI-constrained path.

.. note::

   The **fMRI pipeline** and the **plotting** command (``eeg-pipeline plotting``)
   are **still under active development**. Interfaces and outputs may change;
   validate critical workflows after upgrading.

----

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Getting Started

   install
   contributing
   glossary
   faq

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
