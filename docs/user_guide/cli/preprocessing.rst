Preprocessing
=============

Automated EEG preprocessing: bad-channel detection, ICA, and epoching.

.. code-block:: bash

   eeg-pipeline preprocessing [mode] [options]

Modes
-----

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Mode
     - What it runs
   * - ``full``
     - Bad-channel detection → ICA fitting → ICA labeling → epoch creation,
       in sequence. Use for a fresh subject.
   * - ``bad-channels``
     - PyPREP bad-channel detection only (deviation + correlation, optional RANSAC).
      Updates ``channels.tsv``. Cross-run synchronization is optional via
      ``pyprep.bad_channel_sync_policy=subject_union``.
   * - ``ica``
     - Fits ICA (extended Infomax, 99% variance, 1 Hz high-pass) via
       MNE-BIDS-Pipeline, then labels components with ICLabel
       (threshold ``p > 0.8``; keeps ``brain`` and ``other`` labels).
   * - ``epochs``
     - Creates epochs from the ICA-cleaned signal. Default window:
       ``tmin = −7 s``, ``tmax = 15 s``, baseline ``[−0.2, 0] s``,
       rejection via ``autoreject_local``.

Key Options
-----------

.. list-table::
   :header-rows: 1
   :widths: 35 45 20

   * - Option
     - Description
     - Default
   * - ``--ransac`` / ``--no-ransac``
     - Force-enable or force-disable RANSAC during bad-channel detection;
       ``--ransac`` is more sensitive but substantially slower
     - from config (default: enabled)
   * - ``--tmin``
     - Epoch start time in seconds
     - ``-7.0``
   * - ``--tmax``
     - Epoch end time in seconds
     - ``15.0``
   * - ``--reject-method``
     - Epoch rejection strategy: ``none``, ``autoreject_local``, or ``autoreject_global``
     - ``autoreject_local``
   * - ``--no-icalabel``
     - Skip ICLabel; fall back to MNE-BIDS heuristic component labeling
     - disabled
   * - ``--no-pyprep``
     - Skip PyPREP bad-channel detection entirely
     - disabled
   * - ``--ica-method``
     - ICA algorithm: ``extended_infomax``, ``fastica``, ``infomax``, ``picard``
     - from config (default: ``extended_infomax``)
   * - ``--ica-components``
     - Number of ICA components (int) or explained-variance fraction (float < 1)
     - from config (default: ``0.99``)
   * - ``--ica-l-freq``
     - High-pass filter applied before ICA fitting (Hz)
     - from config (default: ``1.0``)
   * - ``--prob-threshold``
     - ICLabel probability threshold for artifact exclusion
     - from config (default: ``0.8``)
   * - ``--ica-labels-to-keep``
     - ICLabel component classes to retain (space-separated)
     - from config (default: ``brain other``)
   * - ``--spatial-filter``
     - Artifact removal method: ``ica`` or ``ssp``
     - ``ica``
   * - ``--task-is-rest``
     - Resting-state mode: creates fixed-length epochs of duration
       ``preprocessing.rest_epochs_duration``; no event conditions required
     - disabled
   * - ``--write-clean-events``
     - Write ``proc-clean_events.tsv`` aligned to kept epochs for event-related
       workflows (used by trial/event alignment in downstream analyses)
     - from config (default: enabled)
   * - ``--trim-to-first-volume``
     - Trim EEG onset to the first fMRI volume timestamp (simultaneous EEG–fMRI)
     - config-dependent (enabled in the default config)
   * - ``--l-freq``
     - High-pass filter cutoff (Hz)
     - from config (default: ``0.1``)
   * - ``--h-freq``
     - Low-pass filter cutoff (Hz)
     - from config (default: ``100``)
   * - ``--notch``
     - Notch filter frequency (Hz)
     - from config (default: ``60``)
   * - ``--n-jobs``
     - Number of parallel jobs for bad-channel detection and autoreject (``-1`` = all cores)
     - from config (default: ``1``)

Examples
--------

.. code-block:: bash

   # End-to-end preprocessing for a single subject
   eeg-pipeline preprocessing full --subject 0001

   # Full preprocessing across all subjects (in parallel at the subject level)
   eeg-pipeline preprocessing full --all-subjects

   # Bad-channel detection with RANSAC (slower but more sensitive)
   eeg-pipeline preprocessing bad-channels --subject 0001 --ransac

   # ICA only (assumes bad-channels was already run)
   eeg-pipeline preprocessing ica --subject 0001

   # Custom epoch window
   eeg-pipeline preprocessing epochs --subject 0001 \
     --tmin -7.0 --tmax 15.0 --reject-method autoreject_local

   # Skip ICLabel (use heuristic ICA classification)
   eeg-pipeline preprocessing full --subject 0001 --no-icalabel

   # SSP instead of ICA for artifact removal
   eeg-pipeline preprocessing full --subject 0001 --spatial-filter ssp

   # Resting-state mode (fixed-length epochs; no events.tsv conditions required)
   eeg-pipeline preprocessing full --subject 0001 --task-is-rest

   # Simultaneous EEG–fMRI: align EEG onset to first fMRI volume
   eeg-pipeline preprocessing epochs --subject 0001 --trim-to-first-volume

.. note::

   The ``full`` mode runs all stages in order. If a stage fails mid-way,
   re-run only the failed mode (e.g., ``ica`` or ``epochs``) after fixing
   the issue. Each mode is idempotent and will overwrite its own outputs.

.. seealso::

   :doc:`../../methods/eeg/preprocessing`
      Algorithm details, default parameter values, and ICA labeling rules.

   :doc:`index`
      Shared ``--subject``, ``--all-subjects``, ``--task``, and ``--set`` flags.

   :doc:`../configuration`
      ``pyprep``, ``ica``, and ``epochs`` sections of ``eeg_config.yaml``.
