Validation
==========

Data integrity and schema checks across the pipeline. Run ``validate quick``
before any batch job to confirm BIDS layout, config consistency, and subject
discovery. The default mode is ``quick``.

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

.. note::

   ``validate`` uses ``--subjects`` (plural), not ``--subject``. This differs
   from all other commands. Omitting it validates all discovered subjects.

Recommended Pre-Run Sequence
-----------------------------

Run these before any analysis stage to catch problems early:

.. code-block:: bash

   eeg-pipeline validate quick        # BIDS layout + config consistency
   eeg-pipeline info subjects         # confirm subject discovery and run counts
   eeg-pipeline info config           # review the resolved active configuration

Examples
--------

.. code-block:: bash

   # Quick validation (default)
   eeg-pipeline validate

   # Full validation for specific subjects
   eeg-pipeline validate all --subjects 0001 0002

   # Validate only epochs (after preprocessing)
   eeg-pipeline validate epochs

   # Validate feature tables (after feature extraction)
   eeg-pipeline validate features

   # JSON output for CI/scripting
   eeg-pipeline validate all --json

.. seealso::

   :doc:`stats_info`
      Read-only ``info`` and ``stats`` commands for subject discovery and
      feature coverage inspection.

   :doc:`index`
      Shared runtime flags (``--task``, ``--set``, ``--bids-root``, etc.).
