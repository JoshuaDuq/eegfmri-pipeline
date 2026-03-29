Plotting
========

Curated visualization suites driven by a JSON plot catalog.

.. code-block:: bash

   eeg-pipeline plotting [mode] [options]

Modes
-----

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Mode
     - Description
   * - ``visualize``
     - Render selected plot suites
   * - ``tfr``
     - Time-frequency representation plots

Available Plot Groups
---------------------

``power``, ``connectivity``, ``aperiodic``, ``phase``, ``erds``,
``complexity``, ``spectral``, ``ratios``, ``asymmetry``, ``microstates``,
``bursts``, ``erp``, ``tfr``, ``behavior``

Examples
--------

.. code-block:: bash

   # All available plots
   eeg-pipeline plotting visualize --subject 0001 --all-plots

   # Specific plot groups
   eeg-pipeline plotting visualize --subject 0001 --groups power behavior

   # TFR
   eeg-pipeline plotting tfr --subject 0001

   # Export as SVG and PDF
   eeg-pipeline plotting visualize --subject 0001 --all-plots --formats svg pdf

   # Group-level aggregate plots
   eeg-pipeline plotting visualize --subject 0001 --subject 0002 --analysis-scope group
