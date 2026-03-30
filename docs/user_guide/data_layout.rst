Data Requirements & Layout
==========================

All paths default to ``data/`` at the repository root and are fully configurable
via ``paths`` in ``eeg_config.yaml``. See :doc:`configuration` for all path keys.

EEG BIDS Layout
---------------

Place BIDS-formatted EEG under ``paths.bids_root`` (default ``data/bids_output/eeg/``):

.. code-block:: text

   data/bids_output/eeg/
   ├── dataset_description.json
   ├── participants.tsv
   └── sub-XXXX/
       └── eeg/
           ├── sub-XXXX_task-YYY_run-01_eeg.vhdr   (or .set, .edf, .fif)
           ├── sub-XXXX_task-YYY_run-01_eeg.vmrk
           ├── sub-XXXX_task-YYY_run-01_eeg.eeg
           ├── sub-XXXX_task-YYY_run-01_events.tsv
           ├── sub-XXXX_task-YYY_run-01_channels.tsv   # recommended
           └── sub-XXXX_task-YYY_run-01_electrodes.tsv # recommended

.. note::

   Starting from raw BrainVision / DICOM data? Perform BIDS conversion and
   event-log merging externally before running this pipeline.
   See :doc:`../methods/fmri/raw_to_bids` for the fMRI BIDS contract.

Events Data
-----------

Behavior and ML workflows read trial-level predictors from ``*_events.tsv``.

**Required BIDS columns:**

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Column
     - Purpose
   * - ``onset``
     - Event onset in seconds from the start of the recording
   * - ``duration``
     - Event duration in seconds
   * - ``trial_type``
     - Condition label; used by the GLM and behavioral condition contrasts

Add any study-specific predictor or outcome columns alongside these.
Column name aliases (e.g. ``intensity``, ``rating``) are resolved via
``event_columns`` in ``eeg_config.yaml`` (see :doc:`configuration`).

**The** ``trial_id`` **alignment contract**

After preprocessing, the pipeline writes ``*_proc-clean_events.tsv`` to
``derivatives/preprocessed/eeg/``. This file contains only the rows
corresponding to kept epochs, with a canonical integer ``trial_id`` column
added by the pipeline.

``trial_id`` is the **only** accepted join key between:

- EEG feature tables (``features_<family>.parquet``)
- fMRI trial-wise beta volumes (``beta-series`` / ``lss``)
- Behavioral predictor and outcome columns

Row-order alignment across files is not a valid join strategy and will cause
silent misalignment. Any code that builds the feature–target matrix must join
on ``trial_id`` explicitly.

fMRI Data (Optional)
----------------------

Place fMRI BIDS data under ``paths.bids_fmri_root`` (default ``data/bids_output/fmri/``)
and anatomical T1w images under ``data/fMRI_data/sub-XXXX/anat/`` for FreeSurfer
reconstruction and source localization.

Full BIDS-fMRI layout for ``fmri preprocess`` + ``fmri-analysis``:

.. code-block:: text

   data/bids_output/fmri/
   ├── dataset_description.json
   ├── participants.tsv
   └── sub-XXXX/
       ├── anat/
       │   ├── sub-XXXX_T1w.nii.gz
       │   └── sub-XXXX_T1w.json
       └── func/
           ├── sub-XXXX_task-task_run-01_bold.nii.gz
           ├── sub-XXXX_task-task_run-01_bold.json    # TR, SliceTiming, etc.
           └── sub-XXXX_task-task_run-01_events.tsv

**Required sidecar fields** in ``*_bold.json``:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Field
     - Purpose
   * - ``RepetitionTime``
     - TR in seconds; required by fMRIPrep and nilearn GLM
   * - ``TaskName``
     - Must match the ``task-<name>`` BIDS entity
   * - ``SliceTiming``
     - Required when ``slice_time_ref`` correction is enabled

For fMRI-only analysis (no fMRIPrep), only the NIfTI + events files are
required. Set ``fmri_contrast.input_source: "bids_raw"`` to bypass fMRIPrep
output discovery.

Default Directory Layout
--------------------------

.. code-block:: text

   data/
   ├── source_data/                # Raw recordings (EEG .vhdr, fMRI DICOMs)
   │   └── sub-XXXX/
   │       ├── eeg/                # BrainVision triplets (.vhdr/.vmrk/.eeg)
   │       └── fmri/               # DICOM series folders
   ├── bids_output/
   │   ├── eeg/                    # BIDS-formatted EEG
   │   └── fmri/                   # BIDS-formatted fMRI
   ├── fMRI_data/                  # Subject anatomicals (T1w)
   │   └── sub-XXXX/anat/
   └── derivatives/                # All pipeline outputs
       ├── preprocessed/
       │   ├── eeg/                # ICA components, bad channel logs
       │   └── fmri/               # fMRIPrep outputs
       ├── freesurfer/             # FreeSurfer reconstructions
       ├── group/
       │   └── fmri/
       │       └── second_level/   # Group GLM inference maps
       └── sub-XXXX/
           ├── eeg/
           │   ├── sub-XXXX_task-*_proc-clean_epo.fif
           │   └── features/       # Extracted feature tables (.parquet)
           │       ├── power/
           │       ├── connectivity/
           │       ├── aperiodic/
           │       └── ...
           └── fmri/
               ├── first_level/    # GLM contrast maps
               ├── beta_series/    # Trial-wise beta estimates
               └── lss/            # Least-squares-separate betas

All paths are configurable via the ``paths`` section in ``eeg_config.yaml``:

.. code-block:: yaml

   paths:
     bids_root:       "../../../data/bids_output/eeg"
     bids_fmri_root:  "../../../data/bids_output/fmri"
     deriv_root:      "../../../data/derivatives"
     source_data:     "../../../data/source_data"
     freesurfer_dir:  "../../../data/derivatives/freesurfer"
