Feature Extraction
==================

Extract trial-level EEG features from cleaned epochs. Each feature family
produces one row per trial with clearly documented columns.

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
     - Extract features and write to derivatives (Parquet; optionally TSV/CSV via ``--also-save-csv``) plus provenance metadata
   * - ``visualize``
     - Generate descriptive plots from already-computed tables

Feature Families
----------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Category
     - What is quantified
   * - ``power``
     - Band-limited oscillatory power (delta, theta, alpha, beta, gamma), baseline-normalized
   * - ``spectral``
     - Spectral summary measures (spectral edge, peak frequency, bandwidth)
   * - ``ratios``
     - Band power ratios (theta/beta, theta/alpha, alpha/beta, delta/alpha, delta/theta)
   * - ``aperiodic``
     - 1/f background (slope, offset) via iterative specparam-style fits
   * - ``connectivity``
     - Functional connectivity (wPLI, imcoh, AEC, PLV, PLI) with per-family spatial transforms
   * - ``directedconnectivity``
     - Directed connectivity (PSI, DTF, PDC) from MVAR models
   * - ``microstates``
     - Microstate sequence statistics (coverage, duration, occurrence, transitions)
   * - ``pac``
     - Phase–amplitude coupling (theta–gamma, alpha–gamma) with surrogate-based nulls
   * - ``itpc``
     - Inter-trial phase coherence with CV-safe aggregation modes
   * - ``erp``
     - ERP amplitudes within configurable component windows (N1, N2, P2, …)
   * - ``bursts``
     - Transient oscillatory bursts identified by envelope thresholding
   * - ``complexity``
     - Signal complexity (permutation entropy, sample entropy, MSE, LZC)
   * - ``asymmetry``
     - Hemispheric asymmetry indices for canonical electrode pairs
   * - ``erds``
     - Event-related desynchronization/synchronization relative to baseline
   * - ``quality``
     - Data quality indicators (SNR, muscle artifact burden, line noise)
   * - ``sourcelocalization``
     - Source-space features from LCMV beamformer or eLORETA solutions

All feature computations are configurable through the TUI or YAML. For exact
formulas and configuration details, see :doc:`../../methods/eeg/features`.

Examples
--------

.. code-block:: bash

   # All feature categories
   eeg-pipeline features compute --subject 0001

   # Specific categories with spatial modes
   eeg-pipeline features compute --subject 0001 \
     --categories power connectivity aperiodic \
     --spatial roi global

   # Custom frequency bands and ROIs
   eeg-pipeline features compute --subject 0001 \
     --frequency-bands "mu:8.0:13.0" "high_beta:20.0:30.0" \
     --rois "Motor:C3,C4,Cz" "Occipital:O1,O2,Oz"

   # ML-safe mode (prevents cross-trial leakage)
   eeg-pipeline features compute --subject 0001 --analysis-mode trial_ml_safe

   # CSD spatial transform for phase-based features
   eeg-pipeline features compute --subject 0001 --spatial-transform csd

   # Visualize extracted features
   eeg-pipeline features visualize --subject 0001
