Preprocessing
=============

Automated EEG preprocessing: bad-channel detection, ICA review, and epoching.

.. code-block:: bash

   eeg-pipeline preprocessing [mode] [options]

Modes
-----

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Mode
     - What it runs
   * - ``bad-channels``
     - PyPREP bad-channel detection only (deviation + correlation, optional RANSAC).
       Updates ``channels.tsv``. Cross-run synchronization is optional via
       ``pyprep.bad_channel_sync_policy=subject_union``.
   * - ``ica``
     - Fits a near-rank ICA decomposition via MNE-BIDS-Pipeline and labels
       components with its native ICLabel integration. Review the generated
       component table before running ``epochs``.
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
   * - ``--no-pyprep``
     - Skip PyPREP bad-channel detection entirely
     - disabled
   * - ``--ica-method``
     - ICA algorithm: ``extended_infomax``, ``fastica``, ``infomax``, ``picard``
     - from config (default: ``extended_infomax``)
   * - ``--ica-components``
     - Number of ICA components (int) or explained-variance fraction (float < 1)
     - from config (default: near data rank)
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

   # Bad-channel detection with RANSAC (slower but more sensitive)
   eeg-pipeline preprocessing bad-channels --subject 0001 --ransac

   # Fit and label ICA (assumes bad-channels was already run)
   eeg-pipeline preprocessing ica --subject 0001

   # Review the MNE-BIDS component table, then acknowledge review and create epochs
   eeg-pipeline preprocessing epochs --subject 0001 \
     --set ica.manual_review_complete=true

   # Custom epoch window
   eeg-pipeline preprocessing epochs --subject 0001 \
     --tmin -7.0 --tmax 15.0 --reject-method autoreject_local \
     --set ica.manual_review_complete=true

   # Resting-state mode (fixed-length epochs; no events.tsv conditions required)
   eeg-pipeline preprocessing ica --subject 0001 --task-is-rest
   # Review components before continuing.
   eeg-pipeline preprocessing epochs --subject 0001 --task-is-rest \
     --set ica.manual_review_complete=true

   # Simultaneous EEG–fMRI: align EEG onset to first fMRI volume
   eeg-pipeline preprocessing epochs --subject 0001 --trim-to-first-volume \
     --set ica.manual_review_complete=true

.. note::

   ``ica`` and ``epochs`` are intentionally separate so artifact-component
   exclusions can be reviewed before ICA is applied. When
   ``ica.require_manual_review`` is enabled, ``epochs`` fails until
   ``ica.manual_review_complete`` is explicitly set to ``true``. Each mode is
   idempotent and overwrites its own outputs.

BrainVision Analyzer inputs
---------------------------

Set ``preprocessing.brainvision_analyzer.enabled: true`` when scanner-gradient
and pulse artifacts were corrected in BrainVision Analyzer before BIDS import.
This enables strict validation of the preserved ``Pulse Artifact/R`` markers,
marker-locked CTPS diagnostics, cardiac attenuation QC, and scanner-harmonic QC.

Set it to ``false`` for ordinary EEG or for data that enters the native MNE
EEG-fMRI correction workflow without prior Analyzer processing. The additional
Analyzer-specific validation and QC steps are then omitted.

.. seealso::

   :doc:`../../methods/eeg/preprocessing`
      Algorithm details, default parameter values, and ICA labeling rules.

   :doc:`index`
      Shared ``--subject``, ``--all-subjects``, ``--task``, and ``--set`` flags.

   :doc:`../configuration`
      ``pyprep``, ``ica``, and ``epochs`` sections of ``eeg_config.yaml``.
