EEG Pipeline API
================

.. note::

   Full auto-generated API docs require the complete project environment
   (``pip install -e ".[dev]"``). Run ``make docs`` with the activated
   virtual environment to render all docstrings. The tables below are
   available without a full installation.

Pipelines
---------

Top-level pipeline classes. Each wraps a full analysis stage and is the
recommended entry point for batch processing.

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Symbol
     - Description
   * - ``eeg_pipeline.pipelines.preprocessing.PreprocessingPipeline``
     - Orchestrates bad-channel detection, ICA, and epoch creation
   * - ``eeg_pipeline.pipelines.behavior.BehaviorPipeline``
     - DAG-based behavioral statistics pipeline
   * - ``eeg_pipeline.pipelines.features.FeaturesPipeline``
     - Trial-level EEG feature extraction pipeline
   * - ``eeg_pipeline.pipelines.machine_learning.MachineLearningPipeline``
     - LOSO regression and classification pipeline

.. autoclass:: eeg_pipeline.pipelines.preprocessing.PreprocessingPipeline
   :members:
   :show-inheritance:

.. autoclass:: eeg_pipeline.pipelines.behavior.BehaviorPipeline
   :members:
   :show-inheritance:

----

Analysis — Features
-------------------

.. list-table::
   :header-rows: 1
   :widths: 55 45

   * - Symbol
     - Description
   * - ``eeg_pipeline.analysis.features.api.extract_all_features``
     - TFR-based feature extraction entry point
   * - ``eeg_pipeline.analysis.features.api.extract_precomputed_features``
     - Precomputed-envelope feature extraction entry point

.. autofunction:: eeg_pipeline.analysis.features.api.extract_all_features

.. autofunction:: eeg_pipeline.analysis.features.api.extract_precomputed_features

----

Analysis — Machine Learning
----------------------------

.. list-table::
   :header-rows: 1
   :widths: 55 45

   * - Symbol
     - Description
   * - ``eeg_pipeline.analysis.machine_learning.orchestration.run_regression_ml``
     - LOSO regression pipeline runner
   * - ``eeg_pipeline.analysis.machine_learning.orchestration.run_classification_ml``
     - LOSO classification pipeline runner
   * - ``eeg_pipeline.analysis.machine_learning.orchestration.run_time_generalization``
     - Temporal generalization runner
   * - ``eeg_pipeline.analysis.machine_learning.pipelines.create_elasticnet_pipeline``
     - ElasticNet sklearn pipeline factory
   * - ``eeg_pipeline.analysis.machine_learning.pipelines.create_ridge_pipeline``
     - Ridge sklearn pipeline factory
   * - ``eeg_pipeline.analysis.machine_learning.pipelines.create_rf_pipeline``
     - Random Forest sklearn pipeline factory
   * - ``eeg_pipeline.analysis.machine_learning.uncertainty.compute_prediction_intervals``
     - Conformal prediction interval computation
   * - ``eeg_pipeline.analysis.machine_learning.shap_importance.compute_shap_importance``
     - Per-fold SHAP feature importance aggregation
   * - ``eeg_pipeline.analysis.machine_learning.time_generalization.time_generalization_regression``
     - Train-at-one-window, evaluate-at-all-windows temporal generalization

.. autofunction:: eeg_pipeline.analysis.machine_learning.orchestration.run_regression_ml

.. autofunction:: eeg_pipeline.analysis.machine_learning.orchestration.run_classification_ml

.. autofunction:: eeg_pipeline.analysis.machine_learning.pipelines.create_elasticnet_pipeline

.. autofunction:: eeg_pipeline.analysis.machine_learning.pipelines.create_ridge_pipeline

.. autofunction:: eeg_pipeline.analysis.machine_learning.pipelines.create_rf_pipeline

.. autofunction:: eeg_pipeline.analysis.machine_learning.uncertainty.compute_prediction_intervals

.. autofunction:: eeg_pipeline.analysis.machine_learning.shap_importance.compute_shap_importance

.. autofunction:: eeg_pipeline.analysis.machine_learning.time_generalization.time_generalization_regression

----

Preprocessing Utilities
------------------------

Low-level preprocessing helpers invoked by :class:`~eeg_pipeline.pipelines.preprocessing.PreprocessingPipeline`.

.. list-table::
   :header-rows: 1
   :widths: 55 45

   * - Symbol
     - Description
   * - ``eeg_pipeline.preprocessing.pipeline.preprocess.run_bads_detection``
     - PyPREP-based bad channel detection across files
   * - ``eeg_pipeline.preprocessing.pipeline.preprocess.synchronize_bad_channels_across_runs``
     - Propagate per-run bads to all runs of a subject
   * - ``eeg_pipeline.preprocessing.pipeline.ica.run_ica_label``
     - ICA fitting and ICLabel automatic labeling
   * - ``eeg_pipeline.preprocessing.pipeline.stats.collect_preprocessing_stats``
     - Per-subject preprocessing statistics collection
   * - ``eeg_pipeline.preprocessing.pipeline.tfr.custom_tfr``
     - Morlet wavelet TFR computation on clean epochs

.. autofunction:: eeg_pipeline.preprocessing.pipeline.preprocess.run_bads_detection

.. autofunction:: eeg_pipeline.preprocessing.pipeline.preprocess.synchronize_bad_channels_across_runs

.. autofunction:: eeg_pipeline.preprocessing.pipeline.ica.run_ica_label

.. autofunction:: eeg_pipeline.preprocessing.pipeline.stats.collect_preprocessing_stats

.. autofunction:: eeg_pipeline.preprocessing.pipeline.tfr.custom_tfr

----

Types and Data Structures
--------------------------

Core dataclasses, protocols, and type aliases used throughout the pipeline.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Symbol
     - Description
   * - ``eeg_pipeline.types.ConfigLike``
     - Protocol for configuration objects with dict-like access
   * - ``eeg_pipeline.types.CorrelationResult``
     - Result of a single correlation analysis
   * - ``eeg_pipeline.types.PrecomputedData``
     - Container for precomputed band envelopes and TFR power

.. automodule:: eeg_pipeline.types
   :members:
   :undoc-members: False
   :show-inheritance:

----

Context
-------

Immutable per-subject context objects passed through analysis stages.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Symbol
     - Description
   * - ``eeg_pipeline.context.features.FeatureContext``
     - Holds epochs, config, and subject metadata for feature extraction
   * - ``eeg_pipeline.context.behavior.BehaviorContext``
     - Holds behavioral data and feature tables for behavior pipeline

.. autoclass:: eeg_pipeline.context.features.FeatureContext
   :members:
   :show-inheritance:

.. autoclass:: eeg_pipeline.context.behavior.BehaviorContext
   :members:
   :show-inheritance:

----

Plotting
--------

Visualization configuration and global style management.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Symbol
     - Description
   * - ``eeg_pipeline.plotting.config``
     - Plot catalog definitions and group configuration
   * - ``eeg_pipeline.plotting.style``
     - Global matplotlib style and color palette management

.. automodule:: eeg_pipeline.plotting.config
   :members:
   :undoc-members: False

.. automodule:: eeg_pipeline.plotting.style
   :members:
   :undoc-members: False
