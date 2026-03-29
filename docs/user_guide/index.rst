User Guide
==========

Start here for the operational docs: install, validate inputs, run the
pipelines, and interpret outputs.

.. toctree::
   :maxdepth: 1
   :hidden:

   quickstart
   data_layout
   configuration
   subject_selection
   output_formats
   tui
   cli/index

Start Here
----------

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: :octicon:`rocket` Quick Start
      :link: quickstart
      :link-type: doc

      Clean walkthrough from install to plotted results.
      Organized by pipeline family with copy-paste commands.

   .. grid-item-card:: :octicon:`database` Data Layout
      :link: data_layout
      :link-type: doc

      What the pipeline expects on disk:
      BIDS inputs, required TSV columns, and derivatives structure.

Configure and Run
-----------------

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: :octicon:`gear` Configuration
      :link: configuration
      :link-type: doc

      YAML entry points and defaults.
      Includes runtime overrides via ``--set``.

   .. grid-item-card:: :octicon:`person` Subject Selection
      :link: subject_selection
      :link-type: doc

      Shared flags and runtime controls used across commands.

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

   .. grid-item-card:: :octicon:`terminal` Interactive TUI
      :link: tui
      :link-type: doc

      The same pipeline via a guided terminal UI:
      steps, shortcuts, and environment discovery.
