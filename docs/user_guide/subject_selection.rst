Subject Selection & Runtime Options
=====================================

Most commands accept these shared subject and runtime options:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Option
     - Description
   * - ``--subject XXXX`` / ``-s XXXX``
     - Single subject; repeat the flag for multiple subjects
   * - ``--all-subjects``
     - Process every discovered subject
   * - ``--group all`` or ``--group A,B,C``
     - Select a named group or comma-separated subject list
   * - ``--task`` / ``-t``
     - Override the task label from config
   * - ``--dry-run``
     - Preview work without executing
   * - ``--json``
     - Emit JSON output for scripting or the TUI
   * - ``--progress-json``
     - Emit progress events as JSON lines
   * - ``--set KEY=VALUE``
     - Override config values at runtime (see :doc:`configuration`)
   * - ``--bids-root``
     - Override ``paths.bids_root`` at runtime
   * - ``--bids-fmri-root``
     - Override ``paths.bids_fmri_root`` at runtime
   * - ``--bids-rest-root``
     - Override ``paths.bids_rest_root`` at runtime (resting-state EEG)
   * - ``--deriv-root``
     - Override ``paths.deriv_root`` at runtime
   * - ``--deriv-rest-root``
     - Override ``paths.deriv_rest_root`` at runtime (resting-state EEG)

Examples
--------

.. code-block:: bash

   # Single subject
   eeg-pipeline features compute --subject 0001

   # Multiple subjects
   eeg-pipeline features compute --subject 0001 --subject 0002 --subject 0003

   # All subjects
   eeg-pipeline features compute --all-subjects

   # Dry run (preview without executing)
   eeg-pipeline ml regression --all-subjects --dry-run

.. note::

   ``validate`` uses ``--subjects``; ``info features`` takes a positional subject ID.
