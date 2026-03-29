EEG–fMRI Analysis Pipeline
===========================

A modular, reproducible analysis suite for multimodal neuroimaging research.
From raw :term:`BIDS` data to EEG features, behavioral statistics, machine
learning models, and full fMRI :term:`GLM` — driven by a single CLI or
interactive TUI.

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
      and all environment variables.

   .. grid-item-card:: :octicon:`rocket` Quick Start
      :link: user_guide/quickstart
      :link-type: doc

      Clean walkthrough from ``git clone`` to plotted results.
      Covers validation, preprocessing, features, behavior, ML, fMRI, and plotting.

   .. grid-item-card:: :octicon:`book` User Guide
      :link: user_guide/index
      :link-type: doc

      Data layout, YAML configuration, subject selection, CLI reference,
      output formats, and the interactive TUI.

   .. grid-item-card:: :octicon:`beaker` Methods Reference
      :link: methods/index
      :link-type: doc

      Full scientific documentation — notation, formulas, configuration
      keys, and output schemas for every pipeline stage.

----

**Pipeline stages**

.. grid:: 3
   :gutter: 2

   .. grid-item-card:: 1 · Preprocessing
      :link: methods/eeg/preprocessing
      :link-type: doc

      Bad channel detection (PyPREP), ICA fitting, ICLabel classification,
      epoch creation and rejection.

   .. grid-item-card:: 2 · Feature Extraction
      :link: methods/eeg/features
      :link-type: doc

      16 trial-level feature families: power, connectivity, aperiodic,
      ITPC, PAC, ERP, ERDS, complexity, microstates, and source ROI.

   .. grid-item-card:: 3 · Behavioral Statistics
      :link: methods/eeg/behavior
      :link-type: doc

      Trial-level correlations, regression, ICC, condition comparisons,
      and time-resolved analyses with FDR correction.

   .. grid-item-card:: 4 · Machine Learning
      :link: methods/eeg/machine_learning
      :link-type: doc

      LOSO regression and classification, permutation tests,
      SHAP feature importance, and temporal generalization.

   .. grid-item-card:: 5 · fMRI Pipeline
      :link: methods/fmri/pipeline
      :link-type: doc

      fMRIPrep preprocessing, first-level GLM, group inference,
      trial-wise beta estimation, and resting-state connectivity.

   .. grid-item-card:: 6 · Source Localization
      :link: methods/eeg/source_localization
      :link-type: doc

      LCMV and eLORETA inverse solutions. Template (fsaverage) and
      subject-specific fMRI-constrained paths.

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
