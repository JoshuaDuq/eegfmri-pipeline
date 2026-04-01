User Guide
==========

.. raw:: html

   <p class="hero-intro">
     Operational reference for data layout, configuration, and running the
     pipeline. New users: start with the
     <a href="quickstart.html">Quick Start</a> or the
     <a href="tui.html">Interactive TUI</a>.
   </p>

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
   :gutter: 2

   .. grid-item-card:: Data Layout
      :link: data_layout
      :link-type: doc

      BIDS inputs · required TSV columns · derivatives structure.

   .. grid-item-card:: Configuration
      :link: configuration
      :link-type: doc

      YAML entry points and defaults · runtime overrides via ``--set``.

   .. grid-item-card:: Outputs
      :link: output_formats
      :link-type: doc

      Parquet tables · figures · derivatives · resting-state and
      IAF workflow notes.

Operate the CLI and TUI
-----------------------

.. grid:: 2
   :gutter: 2

   .. grid-item-card:: Interactive TUI *(recommended)*
      :link: tui
      :link-type: doc

      **Start here for interactive use.** Guided wizards — configuration,
      subject selection, feature families, bands, modes. No flags required.

   .. grid-item-card:: CLI Reference
      :link: cli/index
      :link-type: doc

      Command families, modes, and flags.
      Use when you need the exact option surface.
