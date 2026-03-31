Outputs & Advanced Workflows
============================

This page summarizes where the main derived artifacts are written and which
workflow flags change the output shape. Use the sections below as a compact
map, then jump to the methods reference for the scientific details.

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Feature Tables

      Parquet by default, optional TSV/CSV, with one table per feature family.

   .. grid-item-card:: Plot Exports

      PNG by default, with SVG and PDF available through ``--formats``.

   .. grid-item-card:: Fusion Workflows

      Trial-wise EEG→fMRI signature prediction and shared analysis modes.

   .. grid-item-card:: Source / Resting-State Outputs

      Source estimates, resting-state fMRI analyses, and EEG resting-state epochs.

.. tab-set::

   .. tab-item:: Artifacts

      .. rubric:: Feature Tables

      Feature tables are saved as **Parquet** by default. TSV/CSV export remains
      available when you need a plain-text copy:

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

      Source-localization outputs live under
      ``sourcelocalization/<method>/source_estimates/``. The final mode-specific
      subdirectory is ``eeg_only/`` or ``fmri_informed/`` depending on
      ``feature_engineering.sourcelocalization.mode``.

      .. rubric:: Plot Exports

      Plots are saved as PNG by default. Use ``--formats`` to add more:

      .. code-block:: bash

         eeg-pipeline plotting visualize --subject 0001 --formats png svg pdf

      .. rubric:: Source Localization

      Supports a template-based path (fsaverage, no MRI needed) and a
      subject-specific fMRI-constrained path (requires FreeSurfer + Docker).
      The TUI includes a wizard for BEM model generation and coregistration.

      See :doc:`../methods/eeg/source_localization` for the full reference.

   .. tab-item:: Workflows

      .. rubric:: fMRI Raw-to-BIDS

      DICOM-to-BIDS conversion with optional event generation from behavioral logs.
      Requires ``dcm2niix`` on ``PATH``.

      See :doc:`../methods/fmri/raw_to_bids` for the BIDS contract and validation steps.

      .. rubric:: EEG–fMRI Fusion

      Predict trial-wise fMRI signature expression from EEG features:

      .. code-block:: bash

         eeg-pipeline ml regression --subject 0001 --subject 0002 \
           --target fmri_signature \
           --fmri-signature-name SIGNATURE_A \
           --fmri-signature-method beta-series \
           --fmri-signature-metric dot

      Methods: ``beta-series``, ``lss``. Metrics: ``dot``, ``cosine``, ``pearson_r``.
      Signature names come from ``paths.signature_maps`` in the config.

      .. rubric:: Spatial Transforms

      Phase-based families (connectivity, ITPC, PAC) have CSD applied by default to
      reduce volume conduction. Override globally or per-family:

      .. code-block:: bash

         eeg-pipeline features compute --subject 0001 --spatial-transform csd

      .. rubric:: Individual Alpha Frequency (IAF)

      Adaptive frequency bands derived from each subject's baseline PSD:

      .. code-block:: bash

         eeg-pipeline features compute --subject 0001 --iaf-enabled

      .. rubric:: Analysis Modes

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

      .. rubric:: Resting-State Workflows

      Enable resting-state mode in ``eeg_config.yaml``:

      .. code-block:: yaml

         preprocessing:
           task_is_rest: true
           rest_epochs_duration: 10.0
           rest_epochs_overlap: 0.0

      When ``task_is_rest: true``, preprocessing creates fixed-length segments
      and no ``events.tsv`` is required. Event-locked families that depend on
      trial onset markers are automatically skipped:

      - **Skipped:** ``erp``, ``erds``, ``itpc``, ``pac``
      - **Compatible:** ``power``, ``spectral``, ``aperiodic``, ``connectivity``,
        ``directedconnectivity``, ``asymmetry``, ``ratios``, ``microstates``,
        ``complexity``, ``bursts``, ``quality``

      .. code-block:: bash

         eeg-pipeline preprocessing full --subject 0001 --task-is-rest

         eeg-pipeline features compute --subject 0001 \
           --categories power connectivity aperiodic spectral

         eeg-pipeline fmri-analysis rest --subject 0001 \
           --atlas-labels-img /path/to/atlas_parc.nii.gz \
           --atlas-labels-tsv /path/to/atlas_labels.tsv
