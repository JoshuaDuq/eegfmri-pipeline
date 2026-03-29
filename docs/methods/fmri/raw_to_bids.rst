fMRI Raw-to-BIDS Contract
=========================

This pipeline starts from BIDS-formatted fMRI inputs. Raw DICOM-to-BIDS
conversion and event-log harmonization are expected to run in an external,
dataset-specific step before ``eeg-pipeline fmri preprocess`` /
``eeg-pipeline fmri-analysis ...``.

.. list-table::
   :header-rows: 1
   :widths: 20 80
   :stub-columns: 1

   * - Inputs
     - BIDS fMRI layout: ``*_bold.nii.gz``, ``*_bold.json``, ``*_events.tsv``
   * - Outputs
     - Validated BIDS structure ready for ``fmri preprocess``
   * - CLI
     - ``eeg-pipeline validate bids``
   * - Requires
     - ``dcm2niix`` on ``PATH`` (for DICOM conversion); BIDS Validator (optional)

Expected Input Contract
-----------------------

Provide a valid BIDS fMRI layout with at least:

.. code-block:: text

   bids_root/
   ├── dataset_description.json
   └── sub-XXXX/
       └── func/
           ├── sub-XXXX_task-<task>_run-01_bold.nii.gz
           ├── sub-XXXX_task-<task>_run-01_bold.json
           └── sub-XXXX_task-<task>_run-01_events.tsv

Recommended (for robust preprocessing):

- JSON BOLD sidecars (``RepetitionTime``, slice timing, etc.)
- Fieldmap images and ``IntendedFor`` pointers
- ``dataset_description.json`` at the BIDS root

``events.tsv`` Requirements
----------------------------

For GLM workflows, ``events.tsv`` must include:

- **Required BIDS columns:** ``onset``, ``duration``, ``trial_type``
- **Additional columns** referenced by condition selectors
  (``condition_a.column``, ``condition_b.column``) and optional filters

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Column
     - Purpose
   * - ``onset``
     - Trial onset time in seconds from BOLD run start
   * - ``duration``
     - Trial duration in seconds
   * - ``trial_type``
     - Condition label for GLM regressor assignment
   * - Any additional column
     - Used by ``condition_a.column``, ``condition_b.column``, or phase filters

Tooling Note
------------

``dcm2niix`` is a standard choice for DICOM conversion, but this repository does
not provide a built-in raw-to-BIDS CLI wrapper for all paradigms. Study-specific
conversion scripts live under ``studies/``.

Validation
----------

Before running the fMRI pipeline, validate your BIDS layout:

.. code-block:: bash

   eeg-pipeline validate bids
   # or with the BIDS Validator directly:
   bids-validator /path/to/bids_root

.. seealso::

   :doc:`pipeline`
      Full fMRI analysis pipeline starting from the BIDS layout created here.

   :doc:`../../user_guide/data_layout`
      EEG and fMRI BIDS layout requirements, events.tsv columns, and
      bold.json sidecar field reference.

   :doc:`../../user_guide/cli/validation`
      CLI commands for BIDS and derivatives validation.
