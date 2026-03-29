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

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Group
     - Typical content
   * - ``power``
     - Band power summaries and topographies
   * - ``connectivity``
     - Connectivity matrices and derived summaries
   * - ``aperiodic``
     - 1/f slope/offset distributions and QC
   * - ``phase``
     - Phase-based measures (ITPC/PAC summaries)
   * - ``erds``
     - ERDS time-course and summary figures
   * - ``complexity``
     - Complexity feature summaries
   * - ``spectral``
     - Spectral edge and peak summaries
   * - ``ratios``
     - Band ratio summaries
   * - ``asymmetry``
     - Asymmetry indices and comparisons
   * - ``microstates``
     - Microstate statistics and transitions
   * - ``bursts``
     - Burst rate/duration summaries
   * - ``erp``
     - ERP component summaries
   * - ``tfr``
     - Time-frequency plots
   * - ``behavior``
     - Behavioral and model-summary plots

Examples
--------

.. tab-set::

   .. tab-item:: Visualize

      .. code-block:: bash

         # All available plots for one subject
         eeg-pipeline plotting visualize --subject 0001 --all-plots

         # Specific plot groups
         eeg-pipeline plotting visualize --subject 0001 --groups power behavior

         # Export as SVG and PDF
         eeg-pipeline plotting visualize --subject 0001 --all-plots --formats svg pdf

         # Group-level aggregate plots
         eeg-pipeline plotting visualize --subject 0001 --subject 0002 \
           --analysis-scope group

   .. tab-item:: TFR

      .. code-block:: bash

         eeg-pipeline plotting tfr --subject 0001
