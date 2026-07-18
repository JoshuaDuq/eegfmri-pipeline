Feature Extraction
==================

Extract EEG features from cleaned epochs and write Parquet tables.
Supports both task-based (trial-level) and resting-state paradigms
(set ``preprocessing.task_is_rest: true`` in config or pass ``--task-is-rest``).

.. code-block:: bash

   eeg-pipeline features [mode] [options]

Modes
-----

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Mode
     - Description
   * - ``compute``
     - Extract features and write derivatives. Parquet by default;
       add ``--also-save-csv`` for a plain-text copy.

Feature Families
----------------

All 16 families are available. Select a subset with ``--categories``.

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - Family
     - What it computes
   * - ``power``
     - Morlet TFR band power (delta · theta · alpha · beta · gamma),
       baseline-normalized (dB / percent / log-ratio)
   * - ``spectral``
     - PSD descriptors: peak frequency, center frequency, bandwidth,
       spectral entropy (multitaper or Welch)
   * - ``aperiodic``
     - 1/f slope and offset (fixed or knee model via specparam/FOOOF);
       ``powcorr`` band power corrected for aperiodic background
   * - ``erp``
     - ERP component windows (N1 / N2 / P2): peak amplitude, latency,
       AUC, peak-to-peak
   * - ``erds``
     - Event-related desynchronization / synchronization vs. baseline;
       onset and rebound latencies
   * - ``ratios``
     - Band-power ratios (e.g., theta/alpha, delta/beta) from PSD
   * - ``asymmetry``
     - Hemispheric asymmetry indices and log-difference on configured
       electrode pairs
   * - ``microstates``
     - GFP-peak K-means microstate templates: coverage, duration,
       occurrence rate, transition probabilities
   * - ``connectivity``
     - wPLI, PLI, imCoh, PLV, AEC/AEC-orth; optional graph metrics
       (global efficiency, small-world index) and dynamic connectivity states
   * - ``directedconnectivity``
     - Phase Slope Index (PSI), DTF, PDC from MVAR model
   * - ``itpc``
     - Inter-trial phase clustering; CV-safe ``fold_global`` mode by default
   * - ``pac``
     - Phase–amplitude coupling (mean vector length); surrogate z-score
       from trial-shuffle or circular-shift surrogates
   * - ``sourcelocalization``
     - LCMV beamformer or eLORETA inverse → ROI band power and
       AEC/wPLI/PLV in source space; optional fMRI constraint mask
   * - ``complexity``
     - LZC, permutation entropy, sample entropy, multiscale entropy (MSE)
   * - ``bursts``
     - Threshold-based transient oscillation detection on band envelopes:
       count, rate, mean duration, mean amplitude, occupancy fraction
   * - ``quality``
     - Per-trial QC: variance, SNR, muscle artifact index, PTP, finite fraction

Key Options
-----------

.. list-table::
   :header-rows: 1
   :widths: 35 45 20

   * - Option
     - Description
     - Default
   * - ``--categories``
     - Space-separated list of feature families to extract
     - all enabled families from config
   * - ``--analysis-mode``
     - ``group_stats`` (descriptive) or ``trial_ml_safe`` (CV-safe;
       disables cross-trial leakage paths)
     - ``group_stats``
   * - ``--spatial-transform``
     - Global spatial transform override (``csd``, ``laplacian``, or ``none``)
       applied to all families for this run; per-family defaults remain configurable
     - per-family config default
   * - ``--spatial``
     - Spatial aggregation scopes: ``roi``, ``channels``, ``global``
     - all scopes
   * - ``--frequency-bands``
     - Custom band definitions as ``name:f_min:f_max`` (space-separated);
       replaces config bands for this run
     - config ``frequency_bands``
   * - ``--rois``
     - Custom ROI definitions as ``name:ch1,ch2,...``
     - config ROIs
   * - ``--iaf-enabled``
     - Estimate each subject's Individual Alpha Frequency from baseline PSD
       and shift the alpha band accordingly
     - disabled
   * - ``--also-save-csv``
     - Write a ``.csv`` copy alongside the Parquet output
     - disabled
   * - ``--compute-change-scores``
     - Append change-score columns (``difference``, ``percent``,
       ``log_ratio``; ``ratio`` accepted as an alias of ``percent``) for baseline/active pairs
     - config-dependent (enabled in the default config)
   * - ``--change-scores-transform``
     - Change-score transform for derived columns
     - ``difference`` (``ratio`` accepted as alias of ``percent``)
   * - ``--change-scores-window-pairs``
     - Explicit window pairs for change scores (e.g., ``baseline:active``)
     - ``baseline:active``
   * - ``--n-jobs-bands``, ``--n-jobs-connectivity``, ``--n-jobs-aperiodic``
     - Parallel jobs for band/connectivity/aperiodic loops
     - config ``feature_engineering.parallel``

Resting-State Compatibility
---------------------------

When ``task_is_rest: true``, the following event-locked families are
scientifically invalid and will raise an error if requested (they require
trial onset markers):

- ``erp`` — requires event-onset-aligned epochs
- ``erds`` — requires baseline window relative to event onset
- ``itpc`` — inter-trial phase clustering requires repeated trials

All other families (``power``, ``spectral``, ``aperiodic``, ``connectivity``,
``directedconnectivity``, ``asymmetry``, ``ratios``, ``microstates``,
``complexity``, ``bursts``, ``quality``, ``pac``, and ``sourcelocalization``)
are accepted by the feature-category validator for rest data.

Resting-state extraction also requires ``group_stats`` mode; requesting
``--analysis-mode trial_ml_safe`` with ``--task-is-rest`` raises an error.

Analysis Modes
--------------

The ``--analysis-mode`` flag is critical when features feed ML models.

.. list-table::
   :header-rows: 1
   :widths: 22 38 40

   * - Mode
     - When to use
     - What changes
   * - ``group_stats``
     - Descriptive analyses, behavioral correlations, group-level summaries
     - All cross-trial computations permitted
   * - ``trial_ml_safe``
     - Feature inputs to cross-validated ML pipelines
     - Any computation that uses test-trial data (evoked subtraction,
       ITPC global average, microstate clustering, etc.) either
       restricts to training trials or raises an error

Run feature extraction twice — once in each mode — to feed both behavioral
analyses and ML models without recomputing from scratch.

Examples
--------

.. code-block:: bash

   # Default feature extraction (all configured categories)
   eeg-pipeline features compute --subject 0001

   # Subset of families, ML-safe
   eeg-pipeline features compute --subject 0001 \
     --categories power connectivity aperiodic erp erds itpc pac \
     --analysis-mode trial_ml_safe

   # All subjects, parallel jobs, also write CSV
   eeg-pipeline features compute --all-subjects \
     --categories power spectral aperiodic \
     --n-jobs-bands -1 --n-jobs-aperiodic -1 --also-save-csv

   # Custom frequency bands and ROIs
   eeg-pipeline features compute --subject 0001 \
     --frequency-bands "mu:8.0:13.0" "high_beta:20.0:30.0" \
     --rois "Motor:C3,C4,Cz" "Occipital:O1,O2,Oz"

   # IAF-adaptive alpha band
   eeg-pipeline features compute --subject 0001 --iaf-enabled

   # CSD spatial transform for phase-based families
   eeg-pipeline features compute --subject 0001 --spatial-transform csd

.. seealso::

   :doc:`../../methods/eeg/features`
      Formulas, spatial modes, CV hygiene table, and normalization schemes.

   :doc:`../output_formats`
      Parquet layout, metadata JSON, and directory structure.

   :doc:`index`
      Shared ``--subject``, ``--all-subjects``, ``--task``, and ``--set`` flags.
