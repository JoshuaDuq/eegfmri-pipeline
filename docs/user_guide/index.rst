User Guide
==========

Operational reference for data layout, configuration, and running the
pipeline. New users: start with :doc:`/user_guide/quickstart` in the
*Getting Started* section of the sidebar.

.. toctree::
   :maxdepth: 1
   :hidden:

   data_layout
   configuration
   output_formats
   tui
   cli/index

Data and Configuration
-----------------------

.. grid:: 3
   :gutter: 3

   .. grid-item-card:: :octicon:`database` Data Layout
      :link: data_layout
      :link-type: doc

      What the pipeline expects on disk:
      BIDS inputs, required TSV columns, and derivatives structure.

   .. grid-item-card:: :octicon:`gear` Configuration
      :link: configuration
      :link-type: doc

      YAML entry points and defaults.
      Includes runtime overrides via ``--set``.

   .. grid-item-card:: :octicon:`file-directory` Outputs
      :link: output_formats
      :link-type: doc

      Where results are written (Parquet, figures, and derivatives),
      plus workflow notes (resting-state, IAF, EEG–fMRI).

Operate the CLI and TUI
-----------------------

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: :octicon:`code` CLI Reference
      :link: cli/index
      :link-type: doc

      Command families, modes, and flags.
      Use this when you need the exact option surface.

   .. grid-item-card:: :octicon:`terminal` Interactive TUI *(recommended)*
      :link: tui
      :link-type: doc

      **Start here for interactive use.** Guided wizards handle
      configuration, subject selection, feature families, bands,
      and analysis modes — no flags to memorize.
