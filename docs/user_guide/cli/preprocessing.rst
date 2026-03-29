Preprocessing
=============

Automated EEG preprocessing: bad-channel detection, artifact removal, and epoching.

.. code-block:: bash

   eeg-pipeline preprocessing [mode] [options]

Modes
-----

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Mode
     - Description
   * - ``full``
     - Run bad-channels, ICA, and epochs in sequence
   * - ``bad-channels``
     - Detect and interpolate bad channels only (PyPREP)
   * - ``ica``
     - Fit and apply ICA only
   * - ``epochs``
     - Create epochs only (including rejection)

Examples
--------

.. code-block:: bash

   # End-to-end preprocessing
   eeg-pipeline preprocessing full --subject 0001

   # Bad channel detection (PyPREP + RANSAC)
   eeg-pipeline preprocessing bad-channels --subject 0001 --ransac

   # Epoching with an explicit window and Autoreject
   eeg-pipeline preprocessing epochs --subject 0001 \
     --tmin -7.0 --tmax 15.0 --reject-method autoreject_local

   # Disable ICLabel (use heuristic ICA labeling)
   eeg-pipeline preprocessing full --subject 0001 --no-icalabel

   # SSP instead of ICA for artifact removal
   eeg-pipeline preprocessing full --subject 0001 --spatial-filter ssp

   # Resting-state mode (fixed-length epochs; no event conditions required)
   eeg-pipeline preprocessing full --subject 0001 --task-is-rest

   # Write clean events.tsv aligned to kept epochs
   eeg-pipeline preprocessing full --subject 0001 --write-clean-events

   # EEG–fMRI simultaneous acquisition: trim EEG to first fMRI volume
   eeg-pipeline preprocessing epochs --subject 0001 --trim-to-first-volume

See also:
:doc:`../subject_selection` (shared subject/task flags) and
:doc:`../../methods/eeg/preprocessing` (methods + configuration).

For full pipeline steps, CLI options, and configuration details, see
:doc:`../../methods/eeg/preprocessing`.
