EEG Pipeline API
================

.. note::

   Full auto-generated API docs require the complete project environment
   (``pip install -e ".[dev]"``). The tables below list the primary public
   symbols; docstrings are rendered when the package is importable.

Pipelines
---------

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Symbol
     - Description
   * - ``eeg_pipeline.pipelines.preprocessing.PreprocessingPipeline``
     - Orchestrates bad-channel detection, ICA, and epoch creation

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
   * - ``eeg_pipeline.analysis.features.preparation.precompute_data``
     - Band filtering, envelope and TFR precomputation
   * - ``eeg_pipeline.analysis.features.normalization.normalize_features``
     - Group-wise feature normalization
   * - ``eeg_pipeline.analysis.features.normalization.normalize_train_test``
     - Train/test-separated normalization
   * - ``eeg_pipeline.analysis.features.normalization.FeatureNormalizer``
     - Stateful normalizer (fit/transform)
   * - ``eeg_pipeline.analysis.features.cv_hygiene.compute_iaf_for_fold``
     - CV-safe IAF estimation from training trials only

Analysis — Behavior
--------------------

.. list-table::
   :header-rows: 1
   :widths: 55 45

   * - Symbol
     - Description
   * - ``eeg_pipeline.analysis.behavior.orchestration.BehaviorPipeline``
     - DAG-based behavioral statistics pipeline

Analysis — Machine Learning
----------------------------

.. list-table::
   :header-rows: 1
   :widths: 55 45

   * - Symbol
     - Description
   * - ``eeg_pipeline.analysis.machine_learning.orchestration.run_regression_pipeline``
     - LOSO regression pipeline runner
   * - ``eeg_pipeline.analysis.machine_learning.orchestration.run_classification_pipeline``
     - LOSO classification pipeline runner
   * - ``eeg_pipeline.analysis.machine_learning.cv.run_permutation_test``
     - Full nested-CV permutation test
   * - ``eeg_pipeline.analysis.machine_learning.pipelines.create_elasticnet_pipeline``
     - ElasticNet sklearn pipeline factory
   * - ``eeg_pipeline.analysis.machine_learning.pipelines.create_ridge_pipeline``
     - Ridge sklearn pipeline factory
   * - ``eeg_pipeline.analysis.machine_learning.pipelines.create_rf_pipeline``
     - Random Forest sklearn pipeline factory
   * - ``eeg_pipeline.analysis.machine_learning.uncertainty.compute_conformal_intervals``
     - Conformal prediction interval computation

Preprocessing Utilities
------------------------

.. list-table::
   :header-rows: 1
   :widths: 55 45

   * - Symbol
     - Description
   * - ``eeg_pipeline.pipeline.preprocess.run_bads_detection``
     - PyPREP-based bad channel detection
   * - ``eeg_pipeline.pipeline.ica.run_ica_label``
     - ICA fitting and ICLabel labeling
   * - ``eeg_pipeline.pipeline.stats.collect_preprocessing_stats``
     - Per-subject preprocessing statistics collection
   * - ``eeg_pipeline.pipeline.tfr.custom_tfr``
     - Morlet wavelet TFR computation
