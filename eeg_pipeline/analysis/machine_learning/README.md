# Machine Learning Pipeline (under development due to lack of source data)

Trial-level predictive modeling for pain research, supporting both continuous regression (pain intensity) and binary classification (pain vs. no-pain). All evaluation uses strict cross-validation with subject as the statistical unit.

## Architecture

The pipeline is organized into modular layers:

```
orchestration.py          ← High-level runners (entry points)
├── cv.py                 ← Cross-validation fold creation, metrics, permutation tests
├── pipelines.py          ← Regression pipeline factories (ElasticNet, Ridge, RF)
├── classification.py     ← Classification pipeline factories (SVM, LR, RF, Ensemble)
├── cnn.py                ← EEGNet-style CNN classifier (PyTorch)
├── preprocessing.py      ← CV-safe transformers (variance, NaN, inf handling)
├── config.py             ← Unified configuration loader with defaults
├── shap_importance.py    ← SHAP feature importance (per-fold aggregated)
├── uncertainty.py        ← Conformal prediction intervals
├── time_generalization.py← Temporal generalization analysis
└── feature_metadata.py   ← Feature name parsing and grouped importance
```

**Data flow:** Per-trial feature tables (from the feature extraction pipeline) are loaded via `load_active_matrix`, merged with behavioral targets from `events.tsv`, and passed through nested CV.

---

## Preprocessing

**Module:** `preprocessing.py`, `pipelines.py`

All pipelines share a common preprocessing chain that is fitted on the training fold only (CV-safe):

| Step | Transformer | Method |
|------|-------------|--------|
| 1. **Inf → NaN** | `ReplaceInfWithNaN` | Replaces `±inf` with `NaN` so downstream imputers can handle them. |
| 2. **Drop all-NaN columns** | `DropAllNaNColumns` | Removes columns where every value is `NaN`/`inf` in the training fold. Requires at least `min_finite=1` finite value per column. |
| 3. **Imputation** | `SimpleImputer` | Fills remaining `NaN` values using the training-fold statistic. Default strategy: `median`. |
| 4. **Variance filter** | `VarianceThreshold` | Removes features with variance below threshold (default `0.0`). Threshold is tuned via grid search (`variance_threshold_grid: [0.0, 0.01, 0.1]`). Raises an actionable error when all features are removed. |
| 5. **Scaling** | `StandardScaler` | Zero-mean, unit-variance standardization. Applied for linear models (ElasticNet, Ridge, SVM, LR); skipped for Random Forest. |
| 6. **PCA** (optional) | `PCA` | Dimensionality reduction retaining `n_components` fraction of variance (default 95%). Disabled by default (`pca.enabled: false`). |

### Feature Harmonization

When subjects have different feature sets (e.g., different channel counts), fold-level harmonization ensures consistent dimensionality:

| Mode | Behavior |
|------|----------|
| `intersection` | Keep only features present (finite) in every training subject. Default for LOSO. |
| `union_impute` | Keep all features; missing values are imputed. |

---

## Regression Models

### ElasticNet

**Module:** `pipelines.py` → `create_elasticnet_pipeline`

Linear regression with combined L1 + L2 regularization and Yeo-Johnson target transformation.

**Pipeline:**

```
Preprocessing → StandardScaler → [PCA] → TransformedTargetRegressor(
    regressor = ElasticNet(α, l1_ratio),
    transformer = PowerTransformer(method="yeo-johnson", standardize=True)
)
```

**Target transformation:** The Yeo-Johnson power transform normalizes the target distribution before fitting, improving performance on skewed pain ratings. Predictions are back-transformed to the original scale.

**Hyperparameter grid:**

| Parameter | Search space |
|-----------|-------------|
| `alpha` | `[0.01, 0.1, 1.0, 10.0]` |
| `l1_ratio` | `[0.1, 0.5, 0.9]` |
| `variance_threshold` | `[0.0, 0.01, 0.1]` |

### Ridge

**Module:** `pipelines.py` → `create_ridge_pipeline`

L2-regularized linear regression with Yeo-Johnson target transformation. Same pipeline structure as ElasticNet but without L1 sparsity.

**Hyperparameter grid:**

| Parameter | Search space |
|-----------|-------------|
| `alpha` | `[0.01, 0.1, 1.0, 10.0, 100.0]` |

### Random Forest Regressor

**Module:** `pipelines.py` → `create_rf_pipeline`

Ensemble of decision trees. No scaling or target transformation (tree-based models are invariant to monotonic transforms).

**Pipeline:**

```
Preprocessing (no scaling) → RandomForestRegressor(n_estimators, bootstrap=True)
```

**Hyperparameter grid:**

| Parameter | Search space |
|-----------|-------------|
| `max_depth` | `[5, 10, 20, None]` |
| `min_samples_split` | `[2, 5, 10]` |
| `min_samples_leaf` | `[1, 2, 4]` |

---

## Classification Models

**Module:** `classification.py`

Binary classification for pain vs. no-pain prediction. All classifiers use `class_weight="balanced"` to handle class imbalance.

### SVM

**Pipeline:**

```
Preprocessing → StandardScaler → [PCA] → SVC(kernel="rbf", probability=True, class_weight="balanced")
```

**Hyperparameter grid:**

| Parameter | Search space |
|-----------|-------------|
| `C` | `[0.1, 1.0, 10.0]` |
| `gamma` | `["scale", "auto"]` |
| `variance_threshold` | `[0.0, 0.01, 0.1]` |

### Logistic Regression

**Pipeline:**

```
Preprocessing → StandardScaler → [PCA] → LogisticRegression(penalty, solver, class_weight="balanced")
```

Solver is automatically selected: `saga` for L1/ElasticNet penalties, `lbfgs` for L2.

**Hyperparameter grid:**

| Parameter | Search space |
|-----------|-------------|
| `C` | `[0.01, 0.1, 1.0, 10.0]` |
| `variance_threshold` | `[0.0, 0.01, 0.1]` |

### Random Forest Classifier

**Pipeline:**

```
Preprocessing (no scaling) → RandomForestClassifier(n_estimators=100, class_weight="balanced")
```

**Hyperparameter grid:**

| Parameter | Search space |
|-----------|-------------|
| `max_depth` | `[5, 10, 20, None]` |
| `min_samples_leaf` | `[1, 3, 5]` |
| `variance_threshold` | `[0.0, 0.01, 0.1]` |

### Soft Voting Ensemble

**Module:** `classification.py` → `create_ensemble_pipeline`

Combines SVM, Logistic Regression, and Random Forest via `VotingClassifier(voting="soft")`. Each base classifier produces class probabilities; the ensemble averages them for the final prediction.

### EEGNet CNN

**Module:** `cnn.py` → `fit_predict_cnn_binary_classifier`

EEGNet-style convolutional neural network operating on raw epoch tensors `(n_trials, n_channels, n_timepoints)` rather than extracted features.

**Architecture:**

```
Block 1:
  Conv2d(1, F₁, kernel=(1, K₁))          ← Temporal convolution (K₁=64)
  BatchNorm2d(F₁)
  Conv2d(F₁, F₁·D, kernel=(C, 1), groups=F₁)  ← Depthwise spatial convolution
  BatchNorm2d(F₁·D)
  ELU → AvgPool2d(1, 4) → Dropout(p)

Block 2:
  Conv2d(F₁·D, F₁·D, kernel=(1, K₂), groups=F₁·D)  ← Separable temporal (K₂=16)
  Conv2d(F₁·D, F₂, kernel=(1, 1))                     ← Pointwise
  BatchNorm2d(F₂)
  ELU → AvgPool2d(1, 8) → Dropout(p)

Head:
  AdaptiveAvgPool2d(1, 1) → Linear(F₂, 1)
```

| Parameter | Default |
|-----------|---------|
| `F₁` (temporal filters) | 8 |
| `D` (depth multiplier) | 2 |
| `F₂` (pointwise filters) | 16 |
| `K₁` (temporal kernel) | 64 (forced odd) |
| `K₂` (separable kernel) | 16 (forced odd) |
| `dropout` | 0.5 |

**Training:**

- **Optimizer:** AdamW (lr=1e-3, weight_decay=1e-3)
- **Loss:** BCEWithLogitsLoss with automatic positive-class weighting (`pos_weight = n_neg / n_pos`)
- **Early stopping:** Patience of 10 epochs on validation loss
- **Gradient clipping:** Max norm 1.0
- **Channelwise standardization:** Per-channel zero-mean, unit-variance computed on training data only
- **Validation split:** 20% of training data, group-aware (`GroupShuffleSplit`) when multiple subjects are present

---

## Cross-Validation Schemes

**Module:** `cv.py`

### Leave-One-Subject-Out (LOSO)

The primary evaluation scheme. Each subject is held out once as the test set; all remaining subjects form the training set.

**Nested CV structure:**

```
Outer loop: LOSO (one subject held out per fold)
  Inner loop: GroupKFold on training subjects (hyperparameter tuning)
    Scoring: Pearson r (primary, refit metric) + neg_MSE (secondary)
```

- Inner CV splits are capped at the number of unique training subjects.
- If fewer than 2 training subjects remain, the model is fitted without hyperparameter tuning.

### Within-Subject CV

Block-aware cross-validation within each subject. Uses `GroupKFold` with run/block labels as groups to prevent temporal leakage.

**Structure:**

```
For each subject:
  Outer loop: GroupKFold on blocks/runs
    Inner loop: Block-aware GroupKFold (hyperparameter tuning)
```

- Requires `block`/`run_id` labels in `events.tsv`.
- Trials with missing block labels are dropped.

### CV Hygiene

**Module:** `cv.py` → `apply_fold_specific_hygiene`

Prevents data leakage from test trials into fold-specific computations:

- **IAF (Individual Alpha Frequency):** Band definitions computed on training trials only.
- **Global/broadcast features:** ITPC and similar cross-trial features use `train_mask` to exclude test trials.
- **Feature scaling:** All parameters learned on training fold only.

Controlled by `machine_learning.cv.hygiene_enabled` (default `true`).

---

## Evaluation Metrics

### Regression

**Primary metric:** Subject-level Fisher-z–averaged Pearson correlation.

**Method:**

1. Compute per-subject Pearson r between predicted and true values.
2. Clip r values to `[clip_min, clip_max]` (configurable, prevents arctanh explosion).
3. Fisher z-transform: `z = arctanh(r)`.
4. Weighted average: `z̄ = Σ(wᵢ × zᵢ) / Σ(wᵢ)`, where `wᵢ = max(nᵢ − 3, 1)`.
5. Back-transform: `r̄ = tanh(z̄)`.

**Confidence intervals** (configurable via `machine_learning.evaluation.ci_method`):

| Method | Computation |
|--------|-------------|
| `fixed_effects` | SE = `√(1 / Σwᵢ)`, CI = `tanh(z̄ ± 1.96 × SE)`. Standard meta-analytic fixed-effects variance. |
| `bootstrap` | Subject-level bootstrap (1000 iterations). Resample subjects with replacement, recompute weighted z̄, take 2.5th/97.5th percentiles. |

**Subject-level error metrics:** Unweighted mean MAE and RMSE across subjects, with optional bootstrap CI.

**Pooled (secondary):** Trial-level Pearson r, R², MAE, RMSE reported for reference but not used as the primary inferential statistic.

### Classification

**Primary metric:** Subject-level mean AUC (averaged across per-subject AUC values from LOSO).

| Metric | Computation |
|--------|-------------|
| `accuracy` | `correct / total` |
| `balanced_accuracy` | Mean of per-class recall: `(sensitivity + specificity) / 2` |
| `auc` | Area under ROC curve (requires probability predictions) |
| `average_precision` | Area under precision-recall curve |
| `f1` | `2 × (precision × recall) / (precision + recall)` |
| `sensitivity` (recall) | `TP / (TP + FN)` |
| `specificity` | `TN / (TN + FP)` |
| `brier_score` | `mean((y_prob − y_true)²)` — calibration quality |
| `ECE` | Expected Calibration Error: weighted mean of `|fraction_positive − mean_predicted|` across 10 uniform probability bins |

**AUC inference gating:** Subject-level AUC is only reported when at least `min_subjects_with_auc_for_inference` (default 2) subjects have evaluable AUC (requires both classes present per subject).

---

## Permutation Testing

**Module:** `cv.py` → `run_permutation_test`, `orchestration.py`

Constructs a null distribution by repeating the full nested CV with permuted labels.

**Method:**

1. Permute target labels within the dependence structure (preserving subject × block grouping).
2. Re-run the complete nested LOSO pipeline (including inner CV hyperparameter tuning) on permuted labels.
3. Compute the subject-level metric (Fisher-z r for regression, subject-mean AUC for classification) for each permutation.
4. Two-tailed p-value: `p = (n_extreme + 1) / (n_perm + 1)`, where `n_extreme = |{null_metric : |null| ≥ |observed|}|`.

**Permutation schemes** (configurable via `machine_learning.cv.permutation_scheme`):

| Scheme | Behavior |
|--------|----------|
| `within_subject_within_block` | Permute labels within each subject × block combination. **Default.** Preserves both subject-level and block-level dependence. |
| `within_subject` | Permute labels within each subject (ignoring block structure). Fallback when block labels are unavailable. |

**Completion threshold:** At least 50% of permutations must produce finite metrics; otherwise a `RuntimeError` is raised.

---

## Time Generalization Analysis

**Module:** `time_generalization.py` → `time_generalization_regression`

Trains a model in one time window and tests in all other windows, producing a time × time generalization matrix of prediction performance.

**Method:**

1. Load epoch tensors and behavioral targets across subjects.
2. Define sliding time windows within the active period (configurable `window_len`, `step`, `active_window`).
3. For each window, compute mean channel activity as features: `X[trial, window, channel] = mean(data[trial, channel, t_start:t_end])`.
4. LOSO outer loop:
   - For each training window `i`, fit a Ridge regression model (`StandardScaler → Ridge`).
   - Optionally tune Ridge α via `GridSearchCV` with inner `GroupKFold` (`use_ridgecv=true`).
   - For each test window `j`, predict on held-out subject's data and compute Pearson r and R².
5. Aggregate across folds using Fisher-z averaging (same method as regression metrics).

**Significance testing** (when `n_perm > 0`):

Three correction methods are applied simultaneously:

| Method | Description |
|--------|-------------|
| **FDR** | Benjamini-Hochberg correction across all tested cells. |
| **Max-stat (FWER)** | 95th percentile of the maximum absolute null r across all cells per permutation. |
| **Cluster (FWER)** | Cluster-based permutation test: contiguous cells exceeding a forming threshold are grouped; cluster sizes are compared to the null distribution of maximum cluster sizes. |

**Cell-level p-value:** `p = (|{null_r : |null| ≥ |observed|}| + 1) / (n_perm_valid + 1)`.

**Quality gates:**

| Gate | Default |
|------|---------|
| `min_subjects_per_cell` | 2 |
| `min_count_per_cell` | 15 |
| `min_samples_per_window` | 15 |
| `min_samples_for_corr` | 10 |
| `min_valid_permutation_fraction` | 0.8 |

---

## SHAP Feature Importance

**Module:** `shap_importance.py`

Computes SHAP (SHapley Additive exPlanations) values for model interpretation, aggregated across CV folds.

**Method:**

1. For each LOSO fold, fit the model on training data (with inner CV hyperparameter tuning).
2. Select the appropriate SHAP explainer based on model type:

| Model type | Explainer | Detection |
|------------|-----------|-----------|
| Tree-based (RF) | `shap.TreeExplainer` | `hasattr(estimator, "feature_importances_")` |
| Linear (ElasticNet, Ridge, LR) | `shap.LinearExplainer` | `hasattr(estimator, "coef_")` |
| Other | `shap.KernelExplainer` | Fallback; uses 100 background samples |

3. Compute SHAP values on the test fold.
4. Per-feature importance: `mean(|SHAP_values|)` across test samples.
5. Aggregate across folds: mean and std of per-fold importance.

**Pipeline awareness:** For sklearn `Pipeline` objects, preprocessing steps are applied to transform X before SHAP computation. Feature names are tracked through selectors (`get_support`) and transformers (`get_feature_names_out`). PCA components are labeled `PC1`, `PC2`, etc.

**Grouped summaries** (when `machine_learning.interpretability.grouped_outputs=true`):

Feature names are parsed via `NamingSchema` into structured metadata (group, band, ROI, scope, stat). Importance is aggregated by:
- Group × band
- Group × band × ROI

Each group's `importance_share = sum / total` provides a normalized contribution fraction.

---

## Permutation Importance

**Module:** `orchestration.py` → `_run_permutation_importance_stage`

Sklearn's `permutation_importance` applied per LOSO fold and averaged.

**Method:**

1. For each fold, fit the tuned model on training data.
2. Compute `permutation_importance(model, X_test, y_test, n_repeats, scoring="r2")`.
3. Map fold-level importances back to the full feature set (features excluded by harmonization receive `NaN`).
4. Average across folds: `mean` and `std` of per-fold importance.

Same grouped summary outputs as SHAP (by group × band, group × band × ROI).

---

## Uncertainty Quantification

**Module:** `uncertainty.py`

Distribution-free prediction intervals via conformal prediction.

### Split Conformal

**Method:**

1. Split training data into proper training (80%) and calibration (20%, min 2 calibration samples; requires at least 5 total training samples).
2. Fit model on proper training set.
3. Compute calibration residuals: `rᵢ = |yᵢ − ŷᵢ|` on calibration set.
4. Conformal quantile: `q̂ = quantile(residuals, ⌈(n+1)(1−α)⌉/n)`.
5. Prediction intervals: `[ŷ − q̂, ŷ + q̂]`.

### CV+ (Jackknife+)

**Method:**

1. Cross-validation on training data (group-aware via `GroupKFold` or `LeaveOneGroupOut`).
2. Collect out-of-fold residuals: `rᵢ = |yᵢ − ŷᵢ|` for each training sample.
3. Fit full model on all training data.
4. Conformal quantile from LOO residuals.
5. Prediction intervals: `[ŷ − q̂, ŷ + q̂]`.

### Conformalized Quantile Regression (CQR)

Adaptive intervals for heteroscedastic data where prediction uncertainty varies across the feature space.

**Method:**

1. Cross-validation on training data to collect out-of-fold quantile predictions.
2. For each fold, fit two `GradientBoostingRegressor` models:
   - Lower quantile: `α/2`
   - Upper quantile: `1 − α/2`
3. Compute conformity scores: `E = max(q̂_low − y, y − q̂_high)`.
4. Conformal quantile `Q̂` from conformity scores.
5. Fit full quantile models on all training data.
6. Prediction intervals: `[q̂_low(x) − Q̂, q̂_high(x) + Q̂]`.

**Coverage guarantee:** For exchangeable data, empirical coverage ≥ `1 − α` (default α=0.1 → 90% intervals).

---

## Pipeline Runners

**Module:** `orchestration.py`

### `run_regression_ml`

LOSO regression on per-trial feature tables.

**Steps:**

1. Load feature matrix via `load_active_matrix` (handles multi-subject assembly, feature filtering, target extraction).
2. Validate target is continuous (warns if binary-like; blocks if `strict_regression_target_continuous=true`).
3. Build model pipeline and hyperparameter grid (ElasticNet, Ridge, or RF).
4. Run `nested_loso_predictions_matrix` (nested LOSO with inner GroupKFold tuning).
5. Export predictions, per-subject correlations, per-subject errors.
6. Compute subject-level Fisher-z r with CI.
7. Optionally run permutation test for p-value.
8. Compute and export baseline (mean predictor) for sanity check.
9. Write `pooled_metrics.json` with subject-level (primary) and pooled-trial (secondary) metrics.

### `run_within_subject_regression_ml`

Block-aware within-subject regression.

**Steps:**

1. Load feature matrix; require block/run labels.
2. Create within-subject folds via `create_within_subject_folds` (GroupKFold on blocks within each subject).
3. For each fold, fit with block-aware inner CV hyperparameter tuning.
4. Aggregate predictions; compute subject-level Fisher-z r.
5. Optionally run block-aware permutation test (permute within subject × block).

### `run_classification_ml`

LOSO classification for pain vs. no-pain.

**Steps:**

1. Load feature matrix (or epoch tensor for CNN) with binary target.
2. Run `nested_loso_classification` (outer LOSO, inner `StratifiedGroupKFold` tuned on `roc_auc`).
3. Compute per-subject metrics (AUC, balanced accuracy, accuracy).
4. Compute calibration metrics (Brier score, ECE, calibration curve).
5. Optionally run permutation test (subject-mean AUC null distribution).
6. Validate fold failure fraction against `max_failed_fold_fraction` (default 0.25).

### `run_within_subject_classification_ml`

Block-aware within-subject classification.

Same structure as LOSO classification but with `GroupKFold` on blocks within each subject. Inner CV uses `StratifiedGroupKFold` on block labels.

### `run_model_comparison_ml`

Compares ElasticNet, Ridge, and Random Forest on identical outer folds.

**Method:**

1. Create shared LOSO outer folds.
2. For each model, run nested CV with inner `GroupKFold` tuning (scoring: R²).
3. Record per-fold R² and MAE for each model.
4. Export `model_comparison.tsv` (per-fold) and `model_comparison_summary.json` (aggregated mean ± std).

### `run_incremental_validity_ml`

Quantifies ΔR² when adding EEG features over baseline predictors.

**Method:**

1. Define baseline predictor matrix (default: `temperature` from `events.tsv`).
2. Define full predictor matrix: `[baseline_predictors | EEG_features]`.
3. For each LOSO fold:
   - Fit baseline model (Ridge with tuned α) on baseline predictors only.
   - Fit full model (ElasticNet) on all predictors.
   - Compute per-fold `Δ R² = R²_full − R²_baseline`.
4. Export per-fold results and summary statistics.

### `run_time_generalization`

Delegates to `time_generalization_regression` (see Time Generalization Analysis above).

---

## Baseline Model

**Module:** `orchestration.py` → `compute_baseline_predictions`

A null (intercept-only) model for sanity checking.

| Task | Baseline |
|------|----------|
| Regression | LOSO mean predictor: predict the training-set mean for each held-out subject. |
| Classification | LOSO majority class: predict the most frequent class in the training set. |

Baseline predictions and metrics (`baseline_r2`, `baseline_mae` or `baseline_accuracy`, `baseline_balanced_accuracy`) are exported alongside model results.

---

## Reproducibility

**Module:** `orchestration.py` → `write_reproducibility_info`

Each ML run writes `reproducibility_info.json` containing:

- `sklearn`, `numpy`, `pandas` versions
- RNG seed
- Subject list and data signature (SHA-256 hash)
- Configuration snapshot (`machine_learning` section)

---

## Configuration

All settings live under `machine_learning` in `eeg_config.yaml`:

### Constants and Preprocessing

| Key | Description | Default |
|-----|-------------|---------|
| `constants.variance_threshold` | Base variance threshold | `0.0` |
| `preprocessing.imputer_strategy` | Imputation method | `median` |
| `preprocessing.power_transformer_method` | Target transform | `yeo-johnson` |
| `preprocessing.power_transformer_standardize` | Standardize after transform | `true` |
| `preprocessing.variance_threshold_grid` | Grid search values | `[0.0, 0.01, 0.1]` |
| `preprocessing.pca.enabled` | Enable PCA | `false` |
| `preprocessing.pca.n_components` | Variance fraction to retain | `0.95` |
| `preprocessing.pca.whiten` | Whiten components | `false` |

### Models

| Key | Description | Default |
|-----|-------------|---------|
| `models.elasticnet.alpha_grid` | ElasticNet α search | `[0.01, 0.1, 1.0, 10.0]` |
| `models.elasticnet.l1_ratio_grid` | L1/L2 mixing search | `[0.1, 0.5, 0.9]` |
| `models.elasticnet.max_iter` | Max solver iterations | `10000` |
| `models.ridge.alpha_grid` | Ridge α search | `[0.01, 0.1, 1.0, 10.0, 100.0]` |
| `models.random_forest.n_estimators` | Number of trees | `100` |
| `models.random_forest.max_depth_grid` | Depth search | `[5, 10, 20, None]` |
| `models.svm.kernel` | SVM kernel | `rbf` |
| `models.svm.C_grid` | Regularization search | `[0.1, 1.0, 10.0]` |
| `models.svm.gamma_grid` | Kernel coefficient search | `["scale", "auto"]` |
| `models.logistic_regression.penalty` | Regularization type | `l2` |
| `models.logistic_regression.C_grid` | Regularization search | `[0.01, 0.1, 1.0, 10.0]` |

### CNN

| Key | Description | Default |
|-----|-------------|---------|
| `models.cnn.temporal_filters` | F₁ | `8` |
| `models.cnn.depth_multiplier` | D | `2` |
| `models.cnn.pointwise_filters` | F₂ | `16` |
| `models.cnn.kernel_length` | K₁ | `64` |
| `models.cnn.separable_kernel_length` | K₂ | `16` |
| `models.cnn.dropout` | Dropout rate | `0.5` |
| `models.cnn.batch_size` | Training batch size | `64` |
| `models.cnn.max_epochs` | Maximum training epochs | `75` |
| `models.cnn.patience` | Early stopping patience | `10` |
| `models.cnn.learning_rate` | AdamW learning rate | `1e-3` |
| `models.cnn.weight_decay` | AdamW weight decay | `1e-3` |
| `models.cnn.gradient_clip_norm` | Max gradient norm | `1.0` |
| `models.cnn.val_fraction` | Validation split fraction | `0.2` |
| `models.cnn.use_cuda` | Enable GPU | `false` |

### Cross-Validation

| Key | Description | Default |
|-----|-------------|---------|
| `cv.default_n_splits` | Default inner CV splits | `5` |
| `cv.hygiene_enabled` | Enable CV hygiene | `true` |
| `cv.permutation_scheme` | Permutation structure | `within_subject_within_block` |
| `cv.min_valid_permutation_fraction` | Min valid permutation rate | `0.5` |

### Evaluation

| Key | Description | Default |
|-----|-------------|---------|
| `evaluation.ci_method` | CI computation method | `bootstrap` |
| `evaluation.bootstrap_iterations` | Bootstrap resamples | `1000` |

### Classification

| Key | Description | Default |
|-----|-------------|---------|
| `classification.model` | Default classifier | `svm` |
| `classification.max_failed_fold_fraction` | Max allowed fold failures | `0.25` |
| `classification.min_subjects_with_auc_for_inference` | Min subjects for AUC reporting | `2` |
| `targets.classification` | Classification target column | `binary_outcome` |
| `targets.regression` | Regression target column | (configurable) |
| `targets.binary_threshold` | Threshold for binarization | `None` |

### Time Generalization

| Key | Description | Default |
|-----|-------------|---------|
| `analysis.time_generalization.active_window` | Time range (seconds) | `[3.0, 10.5]` |
| `analysis.time_generalization.window_len` | Window duration (seconds) | `0.75` |
| `analysis.time_generalization.step` | Window step (seconds) | `0.25` |
| `analysis.time_generalization.default_alpha` | Ridge α (no tuning) | `1.0` |
| `analysis.time_generalization.use_ridgecv` | Enable inner CV for Ridge α | `false` |
| `analysis.time_generalization.cluster_threshold` | Cluster forming threshold | `0.05` |

### Data

| Key | Description | Default |
|-----|-------------|---------|
| `data.feature_harmonization` | Harmonization mode | `intersection` |
| `data.max_excluded_subject_fraction` | Max subject exclusion rate | `1.0` |

---

## Dependencies

- **scikit-learn** — Pipelines, CV, GridSearchCV, metrics, imputation, scaling, PCA, Ridge, ElasticNet, RF, SVM, LR
- **NumPy / SciPy** — Numerical computation, Pearson correlation, Fisher z-transform
- **pandas** — Feature table I/O, metadata handling
- **statsmodels** — Multiple testing correction (FDR)
- **joblib** — Parallel fold execution
- **SHAP** — Feature importance (optional)
- **PyTorch** — CNN classifier (optional)
- **MNE-Python** — Epoch loading for time generalization
