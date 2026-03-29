Validation
==========

Data integrity and schema checks across the pipeline.

.. code-block:: bash

   eeg-pipeline validate [mode] [options]

Modes
-----

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Mode
     - Scope
   * - ``quick``
     - Fast, high-level check across key data products (default)
   * - ``all``
     - Exhaustive validation (all available checks)
   * - ``epochs``
     - Cleaned epochs (``.fif``) and metadata
   * - ``features``
     - Feature table schema, completeness, and basic ranges
   * - ``behavior``
     - Behavioral data tables and required columns
   * - ``bids``
     - BIDS layout and metadata for EEG/fMRI inputs

Examples
--------

.. code-block:: bash

   # Quick validation (default)
   eeg-pipeline validate

   # Full validation for specific subjects
   eeg-pipeline validate all --subjects 0001 0002

   # JSON output (for CI/scripting)
   eeg-pipeline validate all --json
