User Guide
==========

.. container:: page-intro

   Operational reference for dataset layout, configuration, outputs, and
   running the pipeline in practice. New users should start with
   :doc:`/user_guide/quickstart`; this section is for day-to-day execution,
   inspection, and troubleshooting.

.. toctree::
   :maxdepth: 1
   :hidden:

   data_layout
   configuration
   output_formats
   tui
   cli/index

.. rst-class:: section-kicker

Use This Section For

.. rst-class:: summary-grid

.. grid:: 3
   :gutter: 2

   .. grid-item-card:: Preparing datasets
      :link: data_layout
      :link-type: doc

      Confirm BIDS inputs, required columns, and derivative locations before running pipelines.

   .. grid-item-card:: Freezing configuration
      :link: configuration
      :link-type: doc

      Verify YAML defaults and runtime overrides before processing a cohort.

   .. grid-item-card:: Operating the pipeline
      :link: cli/index
      :link-type: doc

      Look up exact command surfaces for scripted or headless execution.

Data and Configuration
-----------------------

.. rst-class:: dashboard-grid

.. grid:: 3
   :gutter: 3

   .. grid-item-card:: Data Layout
      :link: data_layout
      :link-type: doc

      What the pipeline expects on disk:
      BIDS inputs, required TSV columns, and derivatives structure.

   .. grid-item-card:: Configuration
      :link: configuration
      :link-type: doc

      YAML entry points and defaults.
      Includes runtime overrides via ``--set``.

   .. grid-item-card:: Outputs
      :link: output_formats
      :link-type: doc

      Where results are written (Parquet, figures, and derivatives),
      plus workflow notes (resting-state, IAF, EEG–fMRI).

Operate the CLI and TUI
-----------------------

.. rst-class:: dashboard-grid

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: CLI Reference
      :link: cli/index
      :link-type: doc

      Command families, modes, and flags.
      Use this when you need the exact option surface.

   .. grid-item-card:: Interactive TUI *(recommended)*
      :link: tui
      :link-type: doc

      **Start here for interactive use.** Guided wizards handle
      configuration, subject selection, feature families, bands,
      and analysis modes — no flags to memorize.
