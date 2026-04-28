Machine Learning
================

.. raw:: html

   <p class="hero-lede">
     Trial-level predictive modeling under nested LOSO cross-validation.
     Supports continuous regression and binary classification. The subject
     is the statistical unit — all primary metrics are subject-level
     aggregates.
   </p>

.. grid:: 2
   :gutter: 2
   :class-container: meta-cards

   .. grid-item-card:: Inputs

      Feature Parquet tables · behavioral target column from
      ``*_proc-clean_events.tsv``

   .. grid-item-card:: Outputs

      Per-subject metrics · permutation p-values · SHAP values ·
      model weights

   .. grid-item-card:: CLI

      ``eeg-pipeline ml [regression | classify | timegen |
      model_comparison | incremental_validity | uncertainty |
      shap | permutation]``

   .. grid-item-card:: Config

      ``machine_learning`` section of ``eeg_config.yaml``

.. seealso::

   :doc:`features`
      Produces the Parquet feature tables used as the ML design matrix.

   :doc:`behavior`
      Behavioral statistics complement to ML — correlations, regression, ICC.

   :doc:`../../user_guide/output_formats`
      Feature Parquet layout and analysis mode reference.

   :doc:`../../user_guide/cli/ml`
      CLI flags for all ML modes.

Notation
--------

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Symbol
     - Meaning
   * - :math:`N`
     - Number of trials in the design matrix
   * - :math:`P`
     - Number of EEG feature columns
   * - :math:`S`
     - Number of subjects
   * - :math:`K`
     - Number of outer CV folds
   * - :math:`X \in \mathbb{R}^{N \times P}`
     - EEG feature design matrix
   * - :math:`y \in \mathbb{R}^N`
     - Continuous regression target
   * - :math:`\alpha`
     - Regularization strength
   * - :math:`\rho`
     - L1 mixing ratio (ElasticNet ``l1_ratio``)
   * - :math:`r_i`
     - Pearson correlation for subject :math:`i`
   * - :math:`z_i = \mathrm{arctanh}(r_i)`
     - Fisher :math:`z`-transformed correlation
   * - :math:`\bar{r}`
     - Subject-level Fisher-:math:`z`-aggregated Pearson correlation (primary metric)

Modeling Map
------------

The modeling methods are organized by where leakage could occur: design-matrix
preprocessing, estimator fitting, cross-validation, inference, and interpretation.

.. list-table::
   :header-rows: 1
   :widths: 24 36 40

   * - Section
     - Role
     - Key quantities
   * - Preprocessing pipeline
     - Build a fold-safe design matrix from heterogeneous feature tables
     - Imputation, variance thresholding, scaling, PCA, deconfounding
   * - Regression models
     - Predict continuous outcomes under regularized or nonlinear estimators
     - Yeo-Johnson transform, ElasticNet, Ridge, Random Forest
   * - Classification models
     - Predict binary labels under class imbalance
     - SVM objective, logistic probability, EEGNet logits
   * - Cross-validation schemes
     - Estimate generalization with subject/block isolation
     - Nested LOSO, block-aware within-subject CV
   * - Evaluation metrics
     - Aggregate trial predictions at the scientific unit
     - Subject-level :math:`r`, :math:`R^2`, AUC, calibration metrics
   * - Permutation testing
     - Build the empirical null with full-pipeline refits
     - Grouped label permutations and two-tailed :math:`p`
   * - Interpretation and uncertainty
     - Summarize feature importance and predictive intervals
     - SHAP values, temporal generalization, conformal intervals

Preprocessing Pipeline
----------------------

All model pipelines share this chain. Every transformer is CV-safe: fitted on
the training fold only, applied to both train and test.

.. list-table::
   :header-rows: 1
   :widths: 5 25 70

   * - Step
     - Transformer
     - Behavior
   * - 1
     - ``ReplaceInfWithNaN``
     - Replaces :math:`\pm\infty` with ``NaN`` before imputation
   * - 2
     - ``DropAllNaNColumns``
     - Removes columns with no finite value in training fold
   * - 3
     - ``SpatialFeatureSelector``
     - Retains features whose inferred ROI appears in ``spatial_regions_allowed``
   * - 4
     - ``SimpleImputer``
     - Replaces remaining ``NaN`` with training-fold statistic (default: ``median``)
   * - 5
     - ``VarianceThreshold``
     - Drops features with variance :math:`\sigma^2 < \theta` (default :math:`\theta = 0`)
   * - 6
     - ``SelectPercentile`` *(optional)*
     - Retains top-:math:`k\%` features by univariate score
   * - 7
     - ``StandardScaler``
     - Zero-mean, unit-variance standardization (linear models and PCA)
   * - 8
     - ``PCA`` *(optional)*
     - Dimensionality reduction; ``n_components`` as variance fraction or integer

Deconfounding
~~~~~~~~~~~~~

When ``machine_learning.preprocessing.deconfound = true``, residualize EEG features on covariate
matrix :math:`Z` using training-fold regression coefficients:

.. math::

   \hat{B} = (Z_\text{train}^\top Z_\text{train})^{-1} Z_\text{train}^\top X_\text{EEG,train}, \qquad
   \tilde{X}_\text{EEG} = X_\text{EEG} - Z\hat{B}.

Feature Harmonization Across Subjects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Mode
     - Behavior
   * - ``intersection`` *(default)*
     - Retain features for which every training subject has at least one finite value
   * - ``union_impute``
     - Retain all features; imputation handles remaining missing values

Regression Models
-----------------

All regression pipelines wrap the estimator in ``TransformedTargetRegressor``,
applying a **Yeo-Johnson power transform** to the target :math:`y` before fitting
(MLE-estimated on training fold) and back-transforming predictions:

.. math::

   \psi_\lambda(y) = \begin{cases}
   \dfrac{(y+1)^\lambda - 1}{\lambda} & y \ge 0,\; \lambda \ne 0 \\[4pt]
   \ln(y+1) & y \ge 0,\; \lambda = 0 \\[4pt]
   -\dfrac{(1-y)^{2-\lambda}-1}{2-\lambda} & y < 0,\; \lambda \ne 2 \\[4pt]
   -\ln(1-y) & y < 0,\; \lambda = 2
   \end{cases}

ElasticNet
~~~~~~~~~~

Combined L1 + L2 penalized regression:

.. math::

   \hat{\beta} = \underset{\beta}{\arg\min}\;
   \frac{1}{2n}\|y - X\beta\|_2^2
   + \alpha\!\left[\rho\|\beta\|_1 + \frac{1-\rho}{2}\|\beta\|_2^2\right].

Default hyperparameter grid: :math:`\alpha \in \{0.01, 0.1, 1, 10\}`,
:math:`\rho \in \{0.1, 0.5, 0.9\}`, :math:`\theta \in \{0, 0.01, 0.1\}`.

Ridge
~~~~~

L2-regularized regression:

.. math::

   \hat{\beta} = \underset{\beta}{\arg\min}\; \|y - X\beta\|_2^2 + \alpha\|\beta\|_2^2.

Default grid: :math:`\alpha \in \{0.01, 0.1, 1, 10, 100\}`.

Random Forest Regressor
~~~~~~~~~~~~~~~~~~~~~~~~

Bagged ensemble of decision trees; invariant to monotonic feature transforms.
No feature scaling applied.

.. math::

   \hat{f}(x) = \frac{1}{B}\sum_{b=1}^{B} T_b(x).

Default grid: ``max_depth`` ∈ {5, 10, 20, null}, ``min_samples_split`` ∈ {2, 5, 10}.

Classification Models
---------------------

All classifiers use ``class_weight="balanced"``. Optional class resampling:
``none`` (default), ``undersample`` (RandomUnderSampler), or ``smote`` (SMOTE).

SVM (RBF)
~~~~~~~~~

.. math::

   \min_{w,b,\xi}\;\frac{1}{2}\|w\|^2 + C\sum_i \xi_i, \qquad
   K(x,x') = \exp(-\gamma\|x-x'\|^2).

Probability calibration via Platt scaling.

Logistic Regression
~~~~~~~~~~~~~~~~~~~~

.. math::

   \hat{p}(y=1 \mid x) = \sigma(x^\top\beta + b) = \frac{1}{1+e^{-x^\top\beta - b}}.

Supports L2 (default), L1, and ElasticNet penalties.

EEGNet CNN
~~~~~~~~~~

EEGNet-style CNN operating on raw epoch tensors :math:`(N, C, T)`.

.. code-block:: text

   Input: (N, 1, C, T)
   Block 1 — Temporal + Depthwise Spatial:
     Conv2d(1, F₁, (1, K₁)) → BatchNorm2d → Conv2d depthwise spatial
     → BatchNorm2d → ELU → AvgPool2d(1,4) → Dropout(p)
   Block 2 — Separable Temporal:
     Conv2d depthwise temporal → Conv2d pointwise
     → BatchNorm2d → ELU → AvgPool2d(1,8) → Dropout(p)
   Head: AdaptiveAvgPool2d → Flatten → Linear(F₂, 1) → σ(·)

Default: :math:`F_1=8`, :math:`D=2`, :math:`F_2=16`, :math:`K_1=64`, :math:`p=0.5`.
Training: AdamW, BCEWithLogitsLoss with ``pos_weight``, early stopping (patience 10),
gradient clipping (max norm 1.0).

Cross-Validation Schemes
------------------------

Nested LOSO (Primary)
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   Outer loop: LeaveOneGroupOut
     └─ Per fold: feature harmonization on training set only
        Inner loop: GroupKFold (k ≤ n_train_subjects)
          Scoring: Pearson r + neg_MSE
          Refit: neg_MSE (regression) | average_precision (classification)

Within-Subject CV (Block-Aware)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   For each subject:
     Outer loop: GroupKFold on block labels
       Inner loop: GroupKFold on remaining blocks

**Ordered block mode** (``machine_learning.cv.within_subject_ordered_blocks = true``): folds respect
temporal ordering — all preceding blocks train, the next block is test.

CV Hygiene
~~~~~~~~~~

All preprocessing statistics (imputation, variance, scaling, PCA) are estimated
exclusively on the training fold. IAF estimation and global features (ITPC, etc.)
are restricted to ``train_mask`` only. Controlled by ``machine_learning.cv.hygiene_enabled`` (default ``true``).

Evaluation Metrics
------------------

Regression
~~~~~~~~~~

Primary Metric
^^^^^^^^^^^^^^

Subject-level Fisher-:math:`z`-aggregated Pearson correlation:

.. math::

   r_i =
   \frac{\sum_t (\hat{y}_{it} - \bar{\hat{y}}_i)(y_{it} - \bar{y}_i)}
        {\sqrt{\sum_t (\hat{y}_{it} - \bar{\hat{y}}_i)^2 \cdot \sum_t (y_{it} - \bar{y}_i)^2}}.

Aggregation across :math:`S` subjects: clip → arctanh → weighted mean → tanh back-transform.

Weighting modes: ``equal`` (default, :math:`w_i=1`) or ``trial_count``
(:math:`w_i = \max(n_i - 3, 1)`).

Confidence intervals for :math:`\bar{r}`: bootstrap (resample subjects, :math:`B` times)
or fixed-effects (:math:`\mathrm{SE} = 1/\sqrt{\sum_i w_i}`).

Secondary: subject-level mean :math:`R^2`, MAE, RMSE.

Classification
~~~~~~~~~~~~~~

Primary metric: subject-level mean AUC (unweighted mean of per-subject ROC-AUC).

Additional metrics: accuracy, balanced accuracy, average precision, F1, sensitivity,
specificity, Brier score, and ECE (Expected Calibration Error over 10 uniform bins).

Permutation Testing
-------------------

Constructs an empirical null distribution by re-running the full nested CV pipeline
with permuted labels. All hyperparameter tuning is repeated on each permuted dataset.

Permutation Schemes
~~~~~~~~~~~~~~~~~~~

- ``within_subject`` *(default)*: permute labels within each subject.
- ``within_subject_within_block``: permute labels within each subject × block;
  requested block labels must be available and valid.

P-Value
~~~~~~~

Two-tailed regression statistic:

.. math::

   p = \frac{\#\bigl\{|s(y^\pi_j)| \ge |s(y)|\bigr\} + 1}{n_\text{perm,valid} + 1}.

Time Generalization
-------------------

.. container:: module-ref

   Module: ``time_generalization.py``

Trains a regression model at one time window and evaluates it at all other windows.
Output: a time × time generalization matrix of Pearson correlations.

SHAP Feature Importance
-----------------------

.. container:: module-ref

   Module: ``shap_importance.py``

Per-fold SHAP values aggregated across folds:

- ``TreeExplainer`` for Random Forest.
- ``LinearExplainer`` for ElasticNet and Ridge.

Outputs: mean absolute SHAP per feature, grouped by EEG family and band,
and ranked importance tables.

Uncertainty Quantification
--------------------------

.. container:: module-ref

   Module: ``uncertainty.py``

Conformal prediction intervals providing coverage guarantees:

.. math::

   \hat{C}_{1-\alpha}(x) = \hat{f}(x) \pm q_{1-\alpha}\bigl(\{|y_i - \hat{f}(x_i)|\}_{i \in \mathcal{C}}\bigr),

where :math:`\mathcal{C}` is the calibration set (held-out subject in LOSO). Target
coverage is configurable (default 90%). Produces per-trial interval widths and
subject-level coverage summaries.

Output Structure
----------------

.. code-block:: text

   derivatives/machine_learning/
   ├── regression/
   │   ├── results_summary.tsv          # Subject-level r, R², MAE, RMSE
   │   ├── predictions_all_subjects.tsv # Trial-level predictions and targets
   │   ├── permutation_pvalues.tsv      # Permutation test p-values
   │   ├── shap_importance.tsv          # Per-feature SHAP values
   │   ├── model_comparison.tsv         # ElasticNet vs Ridge vs RF
   │   └── figures/
   │       ├── scatter_predicted_vs_actual.png
   │       ├── feature_importance_shap.png
   │       └── permutation_distribution.png
   └── classification/
       ├── results_summary.tsv
       ├── predictions_all_subjects.tsv
       └── figures/
           ├── roc_curve.png
           └── confusion_matrix.png

.. seealso::

   :doc:`features`
      EEG feature tables used as the design matrix.

   :doc:`behavior`
      Behavioral statistics pipeline for feature–behavior correlations.

   :doc:`../../user_guide/output_formats`
      EEG–fMRI fusion workflow for predicting fMRI signatures from EEG.

   :doc:`../../user_guide/cli/ml`
      CLI flags for all modeling modes (regression, classify, timegen, SHAP).
