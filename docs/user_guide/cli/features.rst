Feature Extraction
==================

Extract trial-level EEG features from cleaned epochs.

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
     - Extract features and write derivatives (Parquet; optional TSV/CSV via ``--also-save-csv``)
   * - ``visualize``
     - Plot from already-computed tables

.. dropdown:: Feature families (categories)
   :icon: list-unordered

   .. list-table::
      :header-rows: 1
      :widths: 25 75

      * - Category
        - What is quantified
      * - ``power``
        - Band-limited oscillatory power (delta, theta, alpha, beta, gamma)
      * - ``spectral``
        - Spectral summary measures (spectral edge, peak frequency, bandwidth)
      * - ``ratios``
        - Band-power ratios (theta/beta, theta/alpha, alpha/beta, delta/alpha, delta/theta)
      * - ``aperiodic``
        - 1/f background (slope, offset) via iterative fits
      * - ``connectivity``
        - Functional connectivity (wPLI, imcoh, AEC, PLV, PLI)
      * - ``directedconnectivity``
        - Directed connectivity (PSI, DTF, PDC) from MVAR models
      * - ``microstates``
        - Microstate sequence statistics (coverage, duration, occurrence, transitions)
      * - ``pac``
        - Phase-amplitude coupling (theta-gamma, alpha-gamma) with surrogates
      * - ``itpc``
        - Inter-trial phase coherence
      * - ``erp``
        - ERP amplitudes in configurable component windows
      * - ``bursts``
        - Transient oscillatory bursts from envelope thresholding
      * - ``complexity``
        - Signal complexity (permutation entropy, sample entropy, MSE, LZC)
      * - ``asymmetry``
        - Hemispheric asymmetry indices for canonical electrode pairs
      * - ``erds``
        - Event-related desynchronization/synchronization
      * - ``quality``
        - Data quality indicators (SNR, muscle, line noise)
      * - ``sourcelocalization``
        - Source-space features from LCMV or eLORETA solutions

For formulas and configuration details, see :doc:`../../methods/eeg/features`.

Examples
--------

.. code-block:: bash

   # Default categories
   eeg-pipeline features compute --subject 0001

   # Selected categories with explicit spatial aggregation
   eeg-pipeline features compute --subject 0001 \
     --categories power connectivity aperiodic \
     --spatial roi global

   # Custom frequency bands and ROIs
   eeg-pipeline features compute --subject 0001 \
     --frequency-bands "mu:8.0:13.0" "high_beta:20.0:30.0" \
     --rois "Motor:C3,C4,Cz" "Occipital:O1,O2,Oz"

   # ML-safe mode (avoid CV leakage from cross-trial computations)
   eeg-pipeline features compute --subject 0001 --analysis-mode trial_ml_safe

   # CSD spatial transform (useful for phase-based families)
   eeg-pipeline features compute --subject 0001 --spatial-transform csd

   # Visualize from existing tables
   eeg-pipeline features visualize --subject 0001

See also:
:doc:`../subject_selection` (shared subject/task flags) and
:doc:`../../methods/eeg/features` (methods + configuration).
