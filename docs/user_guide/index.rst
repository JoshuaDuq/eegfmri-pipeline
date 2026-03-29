User Guide
==========

Task-oriented guides covering every aspect of running the pipeline —
from loading your first dataset to advanced multimodal workflows.

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

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: :octicon:`rocket` Quick Start
      :link: quickstart
      :link-type: doc

      9-step walkthrough from ``git clone`` to plotted results.

   .. grid-item-card:: :octicon:`database` Data Layout
      :link: data_layout
      :link-type: doc

      BIDS layout, ``events.tsv`` columns, fMRI sidecar fields,
      and the full derivatives directory tree.

   .. grid-item-card:: :octicon:`gear` Configuration
      :link: configuration
      :link-type: doc

      Key / default / description tables for every YAML section.
      Covers runtime ``--set`` overrides.

   .. grid-item-card:: :octicon:`person` Subject Selection
      :link: subject_selection
      :link-type: doc

      Shared flags: ``--subject``, ``--all-subjects``, ``--task``,
      ``--dry-run``, path overrides.

   .. grid-item-card:: :octicon:`file-directory` Outputs & Workflows
      :link: output_formats
      :link-type: doc

      Feature Parquet layout, plots, resting-state mode, IAF bands,
      EEG–fMRI fusion, and analysis modes.

   .. grid-item-card:: :octicon:`terminal` Interactive TUI
      :link: tui
      :link-type: doc

      Views, wizard steps, keyboard shortcuts, and environment
      discovery for the Go-based terminal interface.

   .. grid-item-card:: :octicon:`code` CLI Reference
      :link: cli/index
      :link-type: doc

      Command-by-command reference for every subcommand, mode,
      and flag.
