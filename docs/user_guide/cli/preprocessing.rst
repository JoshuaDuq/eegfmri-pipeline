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

   # Also append exploratory band-specific ICA diagnostics to the MNE report
   eeg-pipeline preprocessing ica --subject 0001 \
     --set ica.band_specific_report.enabled=true

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

Band-specific ICA report
------------------------

Set ``ica.band_specific_report.enabled: true`` to append six exploratory
sections to each subject's existing MNE HTML report:

* delta + theta (1–8 Hz);
* alpha (8–13 Hz);
* beta (13–30 Hz);
* gamma (30–100 Hz);
* broadband 1–30 Hz;
* broadband 30–100 Hz.

Each section comes from an independent extended-infomax ICA fitted to epochs
filtered to that range. Every component has a topomap, MNE Welch power spectrum,
FieldTrip-style DPSS multitaper time-frequency representation, and an
exploratory ICLabel class and probability. TFRs use baseline-relative dB,
``turbo`` colors, symmetric limits, 1 Hz frequency spacing, and 100 ms time
spacing. Windows and smoothing are 3 seconds / ±1 Hz for delta-theta, 2 seconds
/ ±1.5 Hz for alpha, 2 seconds / ±2.5 Hz for beta, and 1 second / ±5 Hz for
gamma. The fitted ICA and component table are written beneath the subject EEG
derivative's ``band-specific-ica/`` directory.

Every figure title records the active band, condition or comparison, metadata
column and values, retained-trial count, DPSS window and smoothing, time/frequency
grid, and baseline. Color limits use symmetric relative scaling independently
for each displayed result: ``[-max(abs(dB)), +max(abs(dB))]``.

Metadata comparisons
~~~~~~~~~~~~~~~~~~~~

When comparisons are configured, ``ica`` creates pre-ICA task epochs and appends
provisional condition TFRs before manual component review. These use every task
epoch and are labeled ``Provisional — all task epochs``. After review, ``epochs``
recomputes the same sections from the final retained-trial mask and labels them
``Finalized — retained epochs``. Each entry names one metadata column and the
values assigned to group A and group B:

.. code-block:: yaml

   ica:
     band_specific_report:
       enabled: true
       comparisons:
         - name: high_vs_low_temperature
           column: stimulus_temp
           group_a:
             label: High temperature
             values: [48.3, 49.3]
           group_b:
             label: Low temperature
             values: [44.3, 45.3]
         - name: painful_vs_nonpainful
           column: pain_binary_coded
           group_a:
             label: Painful
             values: [1]
           group_b:
             label: Non-painful
             values: [0]

The report shows group A, group B, and group A minus group B for every component
and band. A missing column, missing configured value, empty group, or overlapping
group definition stops the run instead of silently omitting the comparison.

The standard 1–100 Hz ICA remains authoritative for artifact removal. Band-
specific component numbers do not correspond across sections, their ICLabel
results are outside ICLabel's validated broadband use, and they never update
``*_proc-ica_components.tsv`` or control exclusions. This option is disabled by
default because fitting six additional decompositions is computationally
expensive.

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
