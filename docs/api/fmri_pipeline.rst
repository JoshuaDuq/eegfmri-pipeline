fMRI Pipeline API
=================

.. note::

   Full auto-generated API docs require the complete project environment
   (``pip install -e ".[dev]"``). The tables below list the primary public
   symbols; docstrings are rendered when the package is importable.

Pipelines
---------

.. list-table::
   :header-rows: 1
   :widths: 55 45

   * - Symbol
     - Description
   * - ``fmri_pipeline.pipelines.fmri_preprocessing.FmriPreprocessingPipeline``
     - fMRIPrep container orchestrator
   * - ``fmri_pipeline.pipelines.fmri_analysis.FmriAnalysisPipeline``
     - First-level GLM pipeline
   * - ``fmri_pipeline.pipelines.fmri_second_level.FmriSecondLevelPipeline``
     - Group-level second-level inference pipeline
   * - ``fmri_pipeline.pipelines.fmri_trial_signatures.FmriTrialSignaturesPipeline``
     - Trial-wise beta and signature pipeline
   * - ``fmri_pipeline.pipelines.fmri_resting_state.FmriRestingStatePipeline``
     - Resting-state ROI connectivity pipeline

Analysis
--------

.. list-table::
   :header-rows: 1
   :widths: 55 45

   * - Symbol
     - Description
   * - ``fmri_pipeline.analysis.contrast_builder``
     - First-level GLM fitting and contrast computation
   * - ``fmri_pipeline.analysis.second_level``
     - Group-level design assembly and second-level inference
   * - ``fmri_pipeline.analysis.trial_signatures``
     - Trial-wise beta-series (LSA) and LSS extraction
   * - ``fmri_pipeline.analysis.multivariate_signatures.compute_signature_expression``
     - Multivariate signature dot-product / cosine / Pearson readout
   * - ``fmri_pipeline.analysis.resting_state.run_resting_state_connectivity``
     - Fisher-z aggregated ROI connectivity analysis
   * - ``fmri_pipeline.analysis.confounds_selection.select_confounds``
     - fMRIPrep confound regressor selection
   * - ``fmri_pipeline.analysis.bem_generation``
     - Docker-based BEM model, solution, and coregistration transform
   * - ``fmri_pipeline.analysis.reporting.generate_html_report``
     - Self-contained HTML QC report generation

Utilities
---------

.. list-table::
   :header-rows: 1
   :widths: 55 45

   * - Symbol
     - Description
   * - ``fmri_pipeline.utils.bold_discovery.discover_fmriprep_preproc_bold``
     - Discover a preprocessed BOLD file for one run
   * - ``fmri_pipeline.utils.bold_discovery.select_consistent_run_source``
     - Select consistent BOLD source across runs
   * - ``fmri_pipeline.utils.signature_paths.resolve_signature_paths``
     - Resolve configured signature weight map paths
