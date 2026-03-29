Preprocessing
=============

Automated EEG preprocessing: bad channel detection, ICA artifact removal, and epoching.

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
     - Run all preprocessing steps sequentially
   * - ``bad-channels``
     - Detect and interpolate bad channels only
   * - ``ica``
     - Fit and apply ICA only
   * - ``epochs``
     - Create epochs only

Examples
--------

.. code-block:: bash

   # Full preprocessing pipeline
   eeg-pipeline preprocessing full --subject 0001

   # Bad channel detection (PyPREP + RANSAC)
   eeg-pipeline preprocessing bad-channels --subject 0001 --ransac

   # Custom epoch window with autoreject
   eeg-pipeline preprocessing epochs --subject 0001 \
     --tmin -7.0 --tmax 15.0 --reject-method autoreject_local

   # Without ICALabel (fall back to MNE-BIDS pipeline detection)
   eeg-pipeline preprocessing full --subject 0001 --no-icalabel

   # SSP instead of ICA for artifact removal
   eeg-pipeline preprocessing full --subject 0001 --spatial-filter ssp

   # Resting-state mode (fixed-length epochs; no event conditions required)
   eeg-pipeline preprocessing full --subject 0001 --task-is-rest

   # Write clean events.tsv aligned to kept epochs (for downstream alignment)
   eeg-pipeline preprocessing full --subject 0001 --write-clean-events

   # EEG–fMRI simultaneous acquisition: trim EEG to first fMRI volume
   eeg-pipeline preprocessing epochs --subject 0001 --trim-to-first-volume

   # Disable automatic break detection
   eeg-pipeline preprocessing full --subject 0001 --no-find-breaks

For full pipeline steps, CLI options, and configuration details, see
:doc:`../../methods/eeg/preprocessing`.
