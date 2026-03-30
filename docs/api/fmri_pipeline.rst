fMRI Pipeline API
=================

.. note::

   Full auto-generated API docs require the complete project environment
   (``pip install -e ".[dev]"``). Run ``make docs`` with the activated
   virtual environment to render all docstrings. The tables below are
   available without a full installation.

Pipelines
---------

Top-level pipeline classes for the fMRI workflow.

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
   * - ``fmri_pipeline.pipelines.fmri_trial_signatures.FmriTrialSignaturePipeline``
     - Trial-wise beta and signature pipeline
   * - ``fmri_pipeline.pipelines.fmri_resting_state.FmriRestingStatePipeline``
     - Resting-state ROI connectivity pipeline

.. autoclass:: fmri_pipeline.pipelines.fmri_preprocessing.FmriPreprocessingPipeline
   :members:
   :show-inheritance:

.. autoclass:: fmri_pipeline.pipelines.fmri_analysis.FmriAnalysisPipeline
   :members:
   :show-inheritance:

.. autoclass:: fmri_pipeline.pipelines.fmri_second_level.FmriSecondLevelPipeline
   :members:
   :show-inheritance:

.. autoclass:: fmri_pipeline.pipelines.fmri_trial_signatures.FmriTrialSignaturePipeline
   :members:
   :show-inheritance:

.. autoclass:: fmri_pipeline.pipelines.fmri_resting_state.FmriRestingStatePipeline
   :members:
   :show-inheritance:

----

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
   * - ``fmri_pipeline.analysis.resting_state.run_resting_state_analysis_for_subject``
     - Fisher-z aggregated ROI connectivity analysis for one subject
   * - ``fmri_pipeline.analysis.confounds_selection.select_fmriprep_confounds_columns``
     - fMRIPrep confound regressor selection
   * - ``fmri_pipeline.analysis.events_selection``
     - Event filtering and condition remapping for GLM design
   * - ``fmri_pipeline.analysis.bem_generation``
     - Docker-based BEM model, solution, and coregistration transform
   * - ``fmri_pipeline.analysis.reporting.build_fmri_report_html``
     - Self-contained HTML QC report generation
   * - ``fmri_pipeline.analysis.smoothing``
     - Optional spatial smoothing before GLM fitting
   * - ``fmri_pipeline.analysis.constraint_masking``
     - fMRI activation mask construction for EEG source priors

.. automodule:: fmri_pipeline.analysis.contrast_builder
   :members:
   :undoc-members: False
   :show-inheritance:

.. automodule:: fmri_pipeline.analysis.second_level
   :members:
   :undoc-members: False
   :show-inheritance:

.. automodule:: fmri_pipeline.analysis.trial_signatures
   :members:
   :undoc-members: False
   :show-inheritance:

.. autofunction:: fmri_pipeline.analysis.multivariate_signatures.compute_signature_expression

.. autofunction:: fmri_pipeline.analysis.resting_state.run_resting_state_analysis_for_subject

.. autofunction:: fmri_pipeline.analysis.confounds_selection.select_fmriprep_confounds_columns

.. autofunction:: fmri_pipeline.analysis.reporting.build_fmri_report_html

.. automodule:: fmri_pipeline.analysis.events_selection
   :members:
   :undoc-members: False

.. automodule:: fmri_pipeline.analysis.constraint_masking
   :members:
   :undoc-members: False

----

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
   * - ``fmri_pipeline.utils.signature_paths.get_signature_specs``
     - Extract configured signature weight map specs from config
   * - ``fmri_pipeline.utils.signature_paths.discover_signature_root_and_specs``
     - Resolve signature root path and all signature specs

.. autofunction:: fmri_pipeline.utils.bold_discovery.discover_fmriprep_preproc_bold

.. autofunction:: fmri_pipeline.utils.bold_discovery.select_consistent_run_source

.. autofunction:: fmri_pipeline.utils.signature_paths.get_signature_specs

.. autofunction:: fmri_pipeline.utils.signature_paths.discover_signature_root_and_specs
