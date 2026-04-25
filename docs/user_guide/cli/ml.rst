Machine Learning
================

Trial-level predictive modeling with leave-one-subject-out (LOSO) or
within-subject cross-validation, plus model comparison and feature attribution.

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
     - Continuous outcomes with LOSO or within-subject CV (e.g., pain ratings)
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
     - SHAP-based feature importance (fold-aggregated TreeExplainer / LinearExplainer)
   * - ``permutation``
    - Permutation-based feature importance (drop-in-score under shuffled feature columns)

For CV schemes, metrics, and the full configuration surface, see
:doc:`../../methods/eeg/machine_learning`.

.. note::

   Features must be extracted with ``--analysis-mode trial_ml_safe`` before
   running any ML mode. Use ``eeg-pipeline info ml-feature-space`` to inspect
   the resulting design matrix dimensions and check for missing families.

.. seealso::

   :doc:`../../methods/eeg/machine_learning`
      CV scheme, model formulas, SHAP aggregation, and conformal prediction.

   :doc:`features`
      Feature extraction; ``trial_ml_safe`` mode and ``--categories`` selection.

   :doc:`index`
      Shared ``--subject``, ``--all-subjects``, ``--task``, and ``--set`` flags.

   :doc:`../output_formats`
      EEG–fMRI fusion workflow (``--target fmri_signature``).

Examples
--------

.. tab-set::

   .. tab-item:: Regression

      .. code-block:: bash

         # LOSO regression (requires >= 2 subjects)
         eeg-pipeline ml regression --subject 0001 --subject 0002 --subject 0003

         # Within-subject CV (single-subject)
         eeg-pipeline ml regression --subject 0001 --cv-scope subject

         # Predict trial-wise fMRI signature expression from EEG
         eeg-pipeline ml regression --subject 0001 --subject 0002 \
           --target fmri_signature --fmri-signature-name SIGNATURE_A

   .. tab-item:: Classification

      .. code-block:: bash

         # SVM classification with an explicit threshold to binarize the target
         eeg-pipeline ml classify --subject 0001 --subject 0002 \
           --classification-model svm --binary-threshold 30

   .. tab-item:: Feature Space

      .. code-block:: bash

         # Restrict to specific feature families and bands
         eeg-pipeline ml regression --subject 0001 --subject 0002 \
           --feature-families power connectivity --feature-bands alpha beta

         # Fine-grained feature filtering: scope, segment, stat
         eeg-pipeline ml regression --subject 0001 --subject 0002 \
           --feature-scopes roi global --feature-segments active \
           --feature-stats wpli aec

         # Feature harmonization across subjects
         eeg-pipeline ml regression --all-subjects \
           --feature-harmonization union_impute

         # Append covariates to the feature matrix
         eeg-pipeline ml regression --subject 0001 --subject 0002 \
           --covariates predictor trial_index

         # Enforce ML-safe mode (prevents CV leakage from cross-trial features)
         eeg-pipeline ml regression --subject 0001 --subject 0002 \
           --require-trial-ml-safe

   .. tab-item:: Time Generalization

      .. code-block:: bash

         eeg-pipeline ml timegen --subject 0001 --subject 0002

   .. tab-item:: Model Comparison

      .. code-block:: bash

         # Compare model families under a shared CV scheme
         eeg-pipeline ml model_comparison --subject 0001 --subject 0002

         # Custom grids / hyperparameters
         eeg-pipeline ml model_comparison --subject 0001 --subject 0002 \
           --elasticnet-alpha-grid 0.01 0.1 1 10 \
           --rf-n-estimators 500

   .. tab-item:: Incremental Validity

      .. code-block:: bash

         eeg-pipeline ml incremental_validity --subject 0001 --subject 0002 \
           --baseline-predictors predictor

   .. tab-item:: Uncertainty

      .. code-block:: bash

         eeg-pipeline ml uncertainty --subject 0001 --subject 0002

   .. tab-item:: SHAP

      .. code-block:: bash

         eeg-pipeline ml shap --subject 0001 --subject 0002

   .. tab-item:: Permutation

      .. code-block:: bash

         eeg-pipeline ml permutation --subject 0001 --subject 0002
