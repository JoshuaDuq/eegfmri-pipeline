Outputs & Advanced Workflows
============================

.. raw:: html

   <p class="hero-lede">
     Where derived artifacts are written and which workflow flags change the
     output shape. Use these as a compact map, then follow the links to the
     methods reference for scientific details.
   </p>

.. grid:: 2
   :gutter: 2

   .. grid-item-card:: Feature Tables

      Parquet by default · optional CSV export · canonical family tables plus
      optional auxiliary tables (for example PAC trial/time tables), each with
      metadata sidecars.

   .. grid-item-card:: EEG–fMRI Fusion

      Trial-wise EEG → fMRI signature prediction using beta-series or LSS.

   .. grid-item-card:: Source / Resting-State

      Source estimates (LCMV / eLORETA) · resting-state fMRI connectivity ·
      EEG resting-state fixed-length epochs.

.. tab-set::

   .. tab-item:: Artifacts

      .. rubric:: Feature Tables

      Feature tables are saved as **Parquet** by default. CSV export remains
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

      Phase-based families (connectivity, directed connectivity, ITPC, PAC) have
      CSD applied by default to reduce volume conduction. Override globally or
      per-family:

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
           - Cross-trial estimators permitted; output granularity is family-dependent (often trial-level rows with optional condition/global summaries).
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
      trial onset markers are rejected as invalid (fail fast):

      - **Rejected:** ``erp``, ``erds``, ``itpc``
      - **Compatible:** ``power``, ``spectral``, ``aperiodic``, ``connectivity``,
        ``directedconnectivity``, ``asymmetry``, ``ratios``, ``microstates``,
        ``complexity``, ``bursts``, ``quality``, ``pac``, ``sourcelocalization``

      Resting-state feature extraction must run in ``group_stats`` mode.

      .. code-block:: bash

         eeg-pipeline preprocessing full --subject 0001 --task-is-rest

         eeg-pipeline features compute --subject 0001 \
           --categories power connectivity aperiodic spectral

         eeg-pipeline fmri-analysis rest --subject 0001 \
           --atlas-labels-img /path/to/atlas_parc.nii.gz \
           --atlas-labels-tsv /path/to/atlas_labels.tsv
