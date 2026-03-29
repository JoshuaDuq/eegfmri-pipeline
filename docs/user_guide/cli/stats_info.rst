Stats and Info
==============

Inspect pipeline state, subject availability, and current configuration.

Info Command
------------

Read-only inspection — never modifies state.

.. code-block:: bash

   eeg-pipeline info [mode] [options]

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Mode
     - Reports
   * - ``subjects``
     - Discovered subjects across BIDS, epochs, and feature derivatives
   * - ``features``
     - Feature availability per subject (which families/tables exist)
   * - ``config``
     - Current effective configuration (including runtime overrides)
   * - ``version``
     - Installed pipeline version and dependency snapshot
   * - ``plotters``
     - Available plot definitions and groups
   * - ``discover``
     - Discover available columns and values from events, trial tables, and condition-effects data
   * - ``rois``
     - Configured ROI definitions (channel groupings)
   * - ``fmri-conditions``
     - fMRI event conditions available for GLM specification
   * - ``fmri-columns``
     - Columns present in ``events.tsv`` files for fMRI analyses
   * - ``multigroup-stats``
     - Cross-subject summary statistics for selected features
   * - ``ml-feature-space``
     - Dimensions and structure of the ML feature matrix

Stats Command
-------------

Pipeline-wide dashboard. Modes: ``summary``, ``subjects``, ``features``,
``storage``, ``timeline``.

.. code-block:: bash

   eeg-pipeline stats [mode]

Examples
--------

.. code-block:: bash

   eeg-pipeline info subjects
   eeg-pipeline info features 0001
   eeg-pipeline info config
   eeg-pipeline info ml-feature-space
   eeg-pipeline stats
   eeg-pipeline stats storage
