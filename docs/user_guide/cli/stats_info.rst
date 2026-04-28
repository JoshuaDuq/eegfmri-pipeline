Stats and Info
==============

Read-only inspection commands. Neither ``info`` nor ``stats`` modifies
pipeline state.

.. tab-set::

   .. tab-item:: info

      Discover subjects, features, configuration, and data availability.

      .. code-block:: bash

         eeg-pipeline info [mode] [options]

      .. list-table::
         :header-rows: 1
         :widths: 30 70

         * - Mode
           - What it reports
         * - ``subjects``
           - Subjects discovered across BIDS, epochs, and feature derivatives;
             run counts and availability summary
         * - ``features``
           - Feature table availability per subject — which families exist,
             row counts, and any missing tables
         * - ``config``
           - Fully resolved active configuration, including all runtime overrides
             and path resolutions
         * - ``version``
           - Installed pipeline version and key dependency versions
         * - ``plotters``
           - Available plot definitions, groups, and their configuration
         * - ``discover``
           - Auto-discover ``trial_type`` values, condition columns, and
             event columns from ``events.tsv`` and trial tables
         * - ``rois``
           - Configured ROI definitions (channel groupings) with member channels
         * - ``fmri-conditions``
           - Available condition values from fMRI ``events.tsv`` for GLM spec
         * - ``fmri-columns``
           - All columns present in fMRI ``events.tsv``
         * - ``multigroup-stats``
           - Cross-subject summary statistics for selected feature families
         * - ``ml-feature-space``
           - ML feature matrix dimensions: subjects × features after harmonization

   .. tab-item:: stats

      Project-wide dashboard and storage inspection.

      .. code-block:: bash

         eeg-pipeline stats [mode]

      .. list-table::
         :header-rows: 1
         :widths: 22 78

         * - Mode
           - What it reports
         * - ``summary`` *(default)*
           - High-level overview: subjects, preprocessing status, extracted
             families, ML results availability
         * - ``subjects``
           - Per-subject processing status across all pipeline stages
         * - ``features``
           - Feature table coverage and row counts across subjects and families
         * - ``storage``
           - Derivatives directory size breakdown by subdirectory
         * - ``timeline``
           - Chronological log of pipeline runs (timestamps from file metadata)

Examples
--------

.. code-block:: bash

   # Discover all subjects and their data availability
   eeg-pipeline info subjects

   # Inspect feature tables for a specific subject
   eeg-pipeline info features 0001

   # Show the fully resolved configuration (useful for debugging)
   eeg-pipeline info config

   # Show ML feature matrix dimensions before running ML
   eeg-pipeline info ml-feature-space

   # Discover condition values in the events file
   eeg-pipeline info discover

   # Project-wide status dashboard
   eeg-pipeline stats

   # Storage breakdown
   eeg-pipeline stats storage

   # JSON output for scripting
   eeg-pipeline info subjects --json
   eeg-pipeline validate all --json

.. seealso::

   :doc:`index`
      Command matrix and shared flags used by the analysis command families.

   :doc:`validation`
      Data integrity checks that complement these read-only inspection commands.
