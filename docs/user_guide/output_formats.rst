Outputs & Advanced Workflows
============================

Feature Tables
--------------

Feature tables are saved as **Parquet** by default (recommended) with optional
TSV/CSV export:

.. code-block:: bash

   eeg-pipeline features compute --subject 0001 --also-save-csv

Per-subject feature output structure:

.. code-block:: text

   derivatives/sub-XXXX/eeg/features/
   ├── power/
   │   ├── features_power.parquet
   │   └── metadata/
   │       └── features_power.json        # Extraction config + column descriptions
   ├── connectivity/
   │   ├── features_connectivity.parquet
   │   └── metadata/
   │       └── features_connectivity.json
   ├── aperiodic/
   │   ├── features_aperiodic.parquet
   │   ├── aperiodic_qc.tsv              # Per-segment/channel aperiodic fit QC
   │   └── metadata/
   │       └── features_aperiodic.json
   └── ...

Source localization outputs live under
``sourcelocalization/<method>/source_estimates/``. The final mode-specific
subdirectory is ``eeg_only/`` or ``fmri_informed/`` depending on
``feature_engineering.sourcelocalization.mode``.

Plots
-----

Plots are saved as PNG by default. Use ``--formats`` to add more:

.. code-block:: bash

   eeg-pipeline plotting visualize --subject 0001 --formats png svg pdf

Source Localization
-------------------

Supports a template-based path (fsaverage, no MRI needed) and a
subject-specific fMRI-constrained path (requires FreeSurfer + Docker).
The TUI includes a wizard for BEM model generation and coregistration.

See :doc:`../methods/eeg/source_localization` for the full reference.

fMRI Raw-to-BIDS
----------------

DICOM-to-BIDS conversion with optional event generation from behavioral logs.
Requires ``dcm2niix`` on ``PATH``.

See :doc:`../methods/fmri/raw_to_bids` for the BIDS contract and validation steps.

EEG–fMRI Fusion
---------------

Predict trial-wise fMRI signature expression from EEG features:

.. code-block:: bash

   eeg-pipeline ml regression --subject 0001 --subject 0002 \
     --target fmri_signature \
     --fmri-signature-name SIGNATURE_A \
     --fmri-signature-method beta-series \
     --fmri-signature-metric dot

Methods: ``beta-series``, ``lss``. Metrics: ``dot``, ``cosine``, ``pearson_r``.
Signature names come from ``paths.signature_maps`` in the config.

Spatial Transforms
------------------

Phase-based families (connectivity, ITPC, PAC) have CSD applied by default to
reduce volume conduction. Override globally or per-family:

.. code-block:: bash

   eeg-pipeline features compute --subject 0001 --spatial-transform csd

Individual Alpha Frequency (IAF)
--------------------------------

Adaptive frequency bands derived from each subject's baseline PSD:

.. code-block:: bash

   eeg-pipeline features compute --subject 0001 --iaf-enabled

Analysis Modes
--------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Mode
     - Intended use
   * - ``group_stats`` *(default)*
     - Cross-trial estimators permitted; one row per subject or condition.
   * - ``trial_ml_safe``
     - Leakage-prone paths disabled. Use when outputs feed cross-validated models.

.. code-block:: bash

   eeg-pipeline features compute --subject 0001 --analysis-mode trial_ml_safe

Resting-State Workflows
-----------------------

Enable resting-state mode in ``eeg_config.yaml``:

.. code-block:: yaml

   preprocessing:
     task_is_rest: true
     rest_epochs_duration: 10.0
     rest_epochs_overlap: 0.0

When ``task_is_rest: true``, preprocessing creates fixed-length epochs,
no ``events.tsv`` conditions are required, and event-locked feature categories
(``erp``, ``erds``, ``itpc``, ``phase``) are disabled.

.. code-block:: bash

   eeg-pipeline preprocessing full --subject 0001 --task-is-rest

   eeg-pipeline features compute --subject 0001 \
     --categories power connectivity aperiodic spectral

   eeg-pipeline fmri-analysis rest --subject 0001 \
     --atlas-labels-img /path/to/atlas_parc.nii.gz \
     --atlas-labels-tsv /path/to/atlas_labels.tsv
