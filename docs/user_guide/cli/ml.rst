Machine Learning
================

Trial-level predictive modeling with leave-one-subject-out (LOSO)
cross-validation.

.. code-block:: bash

   eeg-pipeline ml [mode] [options]

Modes
-----

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Mode
     - Description
   * - ``regression``
     - LOSO or within-subject regression for continuous outcomes (e.g. pain ratings)
   * - ``classify``
     - Binary classification (SVM, logistic regression, random forest, CNN)
   * - ``timegen``
     - Temporal generalization: train at one time window, evaluate across all windows
   * - ``model_comparison``
     - Compare ElasticNet vs Ridge vs RandomForest under a shared CV scheme
   * - ``incremental_validity``
     - Quantify ΔR² when adding EEG features over a baseline predictor
   * - ``uncertainty``
     - Conformal prediction intervals for calibrated uncertainty estimates
   * - ``shap``
     - SHAP-based feature importance
   * - ``permutation``
     - Permutation-based feature importance

For the full architecture, CV schemes, evaluation metrics, and configuration
details, see :doc:`../../methods/eeg/machine_learning`.

Examples
--------

.. code-block:: bash

   # LOSO regression (requires ≥2 subjects)
   eeg-pipeline ml regression --subject 0001 --subject 0002 --subject 0003

   # Model family selection: elasticnet (default), ridge, or rf
   eeg-pipeline ml regression --subject 0001 --subject 0002 --model ridge

   # SVM classification with explicit binary threshold
   eeg-pipeline ml classify --subject 0001 --subject 0002 \
     --classification-model svm --binary-threshold 30

   # SHAP importance
   eeg-pipeline ml shap --subject 0001 --subject 0002

   # Within-subject CV
   eeg-pipeline ml regression --subject 0001 --cv-scope subject

   # Predict fMRI signature expression from EEG
   eeg-pipeline ml regression --subject 0001 --subject 0002 \
     --target fmri_signature --fmri-signature-name SIGNATURE_A

   # Model comparison with custom hyperparameters
   eeg-pipeline ml model_comparison --subject 0001 --subject 0002 \
     --elasticnet-alpha-grid 0.01 0.1 1 10 \
     --rf-n-estimators 500

   # Restrict to specific feature families and bands
   eeg-pipeline ml regression --subject 0001 --subject 0002 \
     --feature-families power connectivity --feature-bands alpha beta

   # Fine-grained feature filtering: scope, segment, stat
   eeg-pipeline ml regression --subject 0001 --subject 0002 \
     --feature-scopes roi global --feature-segments active \
     --feature-stats wpli aec

   # Feature harmonization across subjects (default: intersection)
   eeg-pipeline ml regression --all-subjects \
     --feature-harmonization union_impute

   # Append meta covariates to the feature matrix
   eeg-pipeline ml regression --subject 0001 --subject 0002 \
     --covariates predictor trial_index

   # Incremental validity: EEG over temperature baseline
   eeg-pipeline ml incremental_validity --subject 0001 --subject 0002 \
     --baseline-predictors predictor

   # Enforce ML-safe mode (prevents CV leakage from cross-trial features)
   eeg-pipeline ml regression --subject 0001 --subject 0002 \
     --require-trial-ml-safe

   # Pipeline preprocessing overrides
   eeg-pipeline ml regression --subject 0001 --subject 0002 \
     --imputer mean --pca-enabled --pca-n-components 0.95 \
     --feature-selection-percentile 50
