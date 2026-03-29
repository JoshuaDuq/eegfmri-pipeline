Stats and Info
==============

Inspect pipeline state, subject availability, and current configuration.

.. tab-set::

   .. tab-item:: info

      Read-only discovery (never modifies state).

      .. code-block:: bash

         eeg-pipeline info [mode] [options]

      Modes:

      .. dropdown:: Available info modes
         :icon: info

         .. list-table::
            :header-rows: 1
            :widths: 30 70

            * - Mode
              - Reports
            * - ``subjects``
              - Subjects discovered across BIDS, epochs, and feature derivatives
            * - ``features``
              - Feature availability per subject (which tables exist)
            * - ``config``
              - Effective configuration (including runtime overrides)
            * - ``version``
              - Installed pipeline version and dependency snapshot
            * - ``plotters``
              - Available plot definitions and groups
            * - ``discover``
              - Column/value discovery from events, trial tables, and condition-effects
            * - ``rois``
              - ROI definitions (channel groupings)
            * - ``fmri-conditions``
              - fMRI event conditions available for GLM specification
            * - ``fmri-columns``
              - Columns present in fMRI ``events.tsv``
            * - ``multigroup-stats``
              - Cross-subject summary statistics for selected features
            * - ``ml-feature-space``
              - ML feature-space dimensions and structure

   .. tab-item:: stats

      Pipeline-wide dashboard (read-only).

      .. code-block:: bash

         eeg-pipeline stats [mode]

      Modes: ``summary`` (default), ``subjects``, ``features``, ``storage``, ``timeline``.

Examples
--------

.. code-block:: bash

   eeg-pipeline info subjects
   eeg-pipeline info features 0001
   eeg-pipeline info config
   eeg-pipeline info ml-feature-space
   eeg-pipeline stats
   eeg-pipeline stats storage

See also:
:doc:`../subject_selection` (shared runtime flags) and
:doc:`../../methods/index` (methods reference for outputs and contracts).
