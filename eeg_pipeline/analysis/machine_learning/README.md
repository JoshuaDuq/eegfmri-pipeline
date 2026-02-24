# Machine Learning Pipeline

Trial-level predictive modeling from EEG features. Supports continuous regression (pain intensity) and binary classification (pain vs. no-pain). The **subject is the statistical unit** throughout: all primary metrics are subject-level aggregates computed under leave-one-subject-out (LOSO) cross-validation.

---

## Table of Contents

1. [Architecture](#1-architecture)
2. [Preprocessing](#2-preprocessing)
3. [Regression Models](#3-regression-models)
4. [Classification Models](#4-classification-models)
5. [Cross-Validation Schemes](#5-cross-validation-schemes)
6. [Evaluation Metrics](#6-evaluation-metrics)
7. [Permutation Testing](#7-permutation-testing)
8. [Time Generalization Analysis](#8-time-generalization-analysis)
9. [SHAP Feature Importance](#9-shap-feature-importance)
10. [Permutation Feature Importance](#10-permutation-feature-importance)
11. [Uncertainty Quantification](#11-uncertainty-quantification)
12. [Pipeline Runners](#12-pipeline-runners)
13. [Output Structure](#13-output-structure)
14. [Reproducibility](#14-reproducibility)
15. [Configuration Reference](#15-configuration-reference)
16. [Dependencies](#16-dependencies)

---

## 1. Architecture

```
orchestration.py           ← High-level pipeline runners (entry points)
├── cv.py                  ← CV fold creation, aggregation, permutation tests
├── pipelines.py           ← Regression pipeline factories (ElasticNet, Ridge, RF)
├── classification.py      ← Classification pipeline factories (SVM, LR, RF, Ensemble)
├── cnn.py                 ← EEGNet-style CNN classifier (PyTorch)
├── preprocessing.py       ← CV-safe transformers (inf/NaN, variance, deconfound)
├── config.py              ← Unified configuration loader with defaults
├── shap_importance.py     ← SHAP feature importance (per-fold aggregated)
├── uncertainty.py         ← Conformal prediction intervals
├── time_generalization.py ← Temporal generalization regression
├── feature_metadata.py    ← Feature name parsing and grouped importance
└── plotting.py            ← Auto-generated publication-style figures
```

**Data flow:**  Per-trial feature tables (written by the feature extraction pipeline to `derivatives/*/eeg/features/`) are assembled into a design matrix $ X \in \mathbb{R}^{N \times P} $ via `load_active_matrix`, merged with behavioral targets from `events.tsv`, and passed through nested CV. All preprocessing parameters are estimated exclusively on the training fold of each split.

---

## 2. Preprocessing

**Modules:** `preprocessing.py`, `pipelines.py`

### 2.1 Standard Preprocessing Steps

All model pipelines share a common chain. Every transformer is **CV-safe**: fitted on the training fold only, applied to train and test.

| Step | Transformer | Behavior |
|------|-------------|----------|
| 1 | `ReplaceInfWithNaN` | Replaces $\pm\infty$ with `NaN` so downstream imputers can operate. |
| 2 | `DropAllNaNColumns` | Removes columns with no finite value in the training fold. Raises if all columns are removed. |
| 3 | `SpatialFeatureSelector` | Retains only features whose inferred ROI appears in `spatial_regions_allowed`. Skipped when list is empty; passes all features if names are unavailable. |
| 4 | `SimpleImputer` | Replaces remaining `NaN` with the training-fold statistic (default: `median`). |
| 5 | `VarianceThreshold` | Drops features with variance $\sigma^2 < \theta$. Default $\theta = 0.0$. Jointly tuned via grid search. Raises a descriptive error if all features are removed. |
| 6 | `SelectPercentile` *(optional)* | Retains the top-$k\%$ features ranked by univariate score (`f_regression` for regression, `f_classif` for classification). Activated only when `feature_selection_percentile < 100`. |
| 7 | `StandardScaler` | Zero-mean, unit-variance standardization. Applied for all linear models (ElasticNet, Ridge, SVM, LR) and whenever PCA is enabled. Skipped for Random Forest. |
| 8 | `PCA` *(optional)* | Dimensionality reduction. `n_components` is either a float (fraction of explained variance, default 0.95) or an integer count. Disabled by default. |

### 2.2 Covariate Handling

When `covariates` are provided, they are appended as the last $C$ columns of $X$ and the pipeline splits into two parallel branches via `ColumnTransformer`:

```
ColumnTransformer
├── "eeg" branch : columns [0 … P−C−1]  → full feature pipeline (steps 1–8)
└── "cov" branch : columns [P−C … P−1]  → ReplaceInfWithNaN → SimpleImputer(most_frequent) → [StandardScaler]
```

Covariate columns are never dropped by variance filtering, spatial selection, or feature harmonization.

### 2.3 Deconfounding

**Transformer:** `Deconfounder` — applied after `ColumnTransformer` when `preprocessing.deconfound: true`.

Given the design matrix partitioned as $X = [X_\text{EEG} \mid Z]$ where $Z \in \mathbb{R}^{N \times C}$ are covariates, the transformer fits a linear model on the training fold:

$$

\hat{B} = (Z_\text{train}^\top Z_\text{train})^{-1} Z_\text{train}^\top X_{\text{EEG,train}}

$$

and returns residuals for both train and test:

$$

\tilde{X}_\text{EEG} = X_\text{EEG} - Z\hat{B}

$$

The output matrix contains only the deconfounded EEG features — covariate columns are discarded.

### 2.4 Feature Harmonization

Fold-level harmonization ensures consistent dimensionality across subjects with heterogeneous feature coverage:

| Mode | Behavior |
|------|----------|
| `intersection` | Retain only features for which **every** training subject has at least one finite value. Falls back to any-finite-in-train if strict intersection is empty. **Default.** |
| `union_impute` | Retain all features; missing values handled by imputation downstream. |

Covariate columns are always protected and excluded from harmonization logic.

---

## 3. Regression Models

**Module:** `pipelines.py`

All regression pipelines wrap the estimator in `TransformedTargetRegressor`, applying a Yeo-Johnson power transform to the target $y$ before fitting and back-transforming predictions to the original scale. This normalizes skewed pain-rating distributions.

**Target transformation:** The Yeo-Johnson transform $\psi_\lambda(y)$ is defined piecewise for $\lambda \neq 0, 2$:

$$

\psi_\lambda(y) = \begin{cases}
\tfrac{(y+1)^\lambda - 1}{\lambda} & y \geq 0,\; \lambda \neq 0 \\
\ln(y+1) & y \geq 0,\; \lambda = 0 \\
-\tfrac{(1-y)^{2-\lambda}-1}{2-\lambda} & y < 0,\; \lambda \neq 2 \\
-\ln(1-y) & y < 0,\; \lambda = 2
\end{cases}

$$

$\lambda$ is estimated by maximum likelihood on the **training fold only**.

---

### 3.1 ElasticNet

**Factory:** `create_elasticnet_pipeline`

Combined L1 + L2 penalized linear regression. Encourages both sparsity (L1) and stability for correlated features (L2):

$$

\hat{\beta} = \underset{\beta}{\arg\min} \;\frac{1}{2n}\bigl\|y - X\beta\bigr\|_2^2 + \alpha\left[\rho\|\beta\|_1 + \frac{1-\rho}{2}\|\beta\|_2^2\right]

$$

where $\alpha > 0$ controls regularization strength and $\rho \in [0,1]$ is the L1 mixing ratio.

**Pipeline:**
```
Preprocessing → StandardScaler → [SelectPercentile] → [PCA]
  → TransformedTargetRegressor(ElasticNet(α, ρ), PowerTransformer("yeo-johnson"))
```

**Hyperparameter grid (inner CV):**

| Parameter | Default search space |
|-----------|----------------------|
| $\alpha$ | `[0.01, 0.1, 1.0, 10.0]` |
| $\rho$ (`l1_ratio`) | `[0.1, 0.5, 0.9]` |
| $\theta$ (`variance_threshold`) | `[0.0, 0.01, 0.1]` |

---

### 3.2 Ridge

**Factory:** `create_ridge_pipeline`

L2-regularized linear regression (Tikhonov regularization). Numerically stable for correlated feature sets:

$$

\hat{\beta} = \underset{\beta}{\arg\min} \;\bigl\|y - X\beta\bigr\|_2^2 + \alpha\|\beta\|_2^2

$$

**Pipeline:** Identical to ElasticNet; the L1 term is absent.

**Hyperparameter grid:**

| Parameter | Default search space |
|-----------|----------------------|
| $\alpha$ | `[0.01, 0.1, 1.0, 10.0, 100.0]` |
| $\theta$ (`variance_threshold`) | `[0.0, 0.01, 0.1]` |

---

### 3.3 Random Forest Regressor

**Factory:** `create_rf_pipeline`

Bagged ensemble of decision trees. Invariant to monotonic feature transforms; no scaling is applied to features.

$$

\hat{f}(x) = \frac{1}{B}\sum_{b=1}^{B} T_b(x), \quad T_b \text{ trained on bootstrap resample } \mathcal{D}_b

$$

**Pipeline:**
```
Preprocessing (no scaling)
  → TransformedTargetRegressor(RandomForestRegressor(B, bootstrap=True), PowerTransformer("yeo-johnson"))
```

**Hyperparameter grid:**

| Parameter | Default search space |
|-----------|----------------------|
| `max_depth` | `[5, 10, 20, None]` |
| `min_samples_split` | `[2, 5, 10]` |
| `min_samples_leaf` | `[1, 2, 4]` |
| $\theta$ (`variance_threshold`) | `[0.0, 0.01, 0.1]` |

---

## 4. Classification Models

**Module:** `classification.py`

Binary classification (pain vs. no-pain). All classifiers use `class_weight="balanced"` and are wrapped in `imblearn.pipeline.Pipeline` to support optional class resampling.

### 4.1 Class Imbalance Handling

Configured via `classification.resampler`:

| Value | Strategy |
|-------|---------|
| `none` | No resampling (default). |
| `undersample` | `RandomUnderSampler`: randomly remove majority-class samples to balance class counts. |
| `smote` | `SMOTE`: synthetically generate minority-class samples via k-NN interpolation. |

---

### 4.2 Support Vector Machine (SVM)

**Factory:** `create_svm_pipeline`

SVM with RBF kernel. Soft-margin formulation:

$$

\min_{w,b,\xi}\;\frac{1}{2}\|w\|^2 + C\sum_i \xi_i
\quad\text{s.t.}\quad y_i(w^\top\phi(x_i)+b)\geq 1-\xi_i,\;\xi_i\geq 0

$$

with $K(x,x') = \exp(-\gamma\|x-x'\|^2)$. Probability calibration via Platt scaling (`probability=True`).

**Pipeline:**
```
Preprocessing → StandardScaler → [PCA] → [Resampler] → SVC(RBF, probability=True, class_weight="balanced")
```

**Hyperparameter grid:**

| Parameter | Default search space |
|-----------|----------------------|
| `C` | `[0.1, 1.0, 10.0]` |
| $\gamma$ | `["scale", "auto"]` |
| $\theta$ | `[0.0, 0.01, 0.1]` |

---

### 4.3 Logistic Regression

**Factory:** `create_logistic_pipeline`

Regularized logistic regression with sigmoid output:

$$

\hat{p}(y=1 \mid x) = \sigma(x^\top\beta + b) = \frac{1}{1+e^{-x^\top\beta - b}}

$$

Supports L2 (default), L1, and ElasticNet penalties. Solver is auto-selected: `saga` for L1/ElasticNet, `lbfgs` for L2 (with scikit-learn ≥ 1.8.0 compatibility).

**Hyperparameter grid:**

| Parameter | Default search space |
|-----------|----------------------|
| $C = 1/\alpha$ | `[0.01, 0.1, 1.0, 10.0]` |
| `l1_ratio` (ElasticNet only) | `[0.1, 0.5, 0.9]` |
| $\theta$ | `[0.0, 0.01, 0.1]` |

---

### 4.4 Random Forest Classifier

**Factory:** `create_rf_classification_pipeline`

Bagged ensemble of decision trees for binary classification. No feature scaling applied.

```
Preprocessing (no scaling) → [Resampler] → RandomForestClassifier(B=100, class_weight="balanced")
```

**Hyperparameter grid:** Same structure as the regressor (max_depth, min_samples_split, min_samples_leaf, variance_threshold).

---

### 4.5 Soft Voting Ensemble

**Factory:** `create_ensemble_pipeline`

Combines SVM, Logistic Regression, and Random Forest via probability averaging:

$$

\hat{p}_\text{ens}(y=1\mid x) = \frac{1}{3}\left[\hat{p}_\text{SVM}(x) + \hat{p}_\text{LR}(x) + \hat{p}_\text{RF}(x)\right]

$$

When `classification.calibrate_ensemble: true`, SVM and RF are individually wrapped in `CalibratedClassifierCV(method="sigmoid", cv=2)` before voting to correct probability calibration independently.

---

### 4.6 EEGNet CNN

**Module:** `cnn.py`

EEGNet-style convolutional neural network operating on raw epoch tensors $(N, C, T)$ — not extracted features. Loaded via `load_epoch_tensor_matrix` (distinct from `load_active_matrix`).

**Architecture:**

```
Input: (N, 1, C, T)

Block 1 — Temporal + Depthwise Spatial:
  Conv2d(1, F₁, (1, K₁), pad=(0, K₁//2))      ← temporal filters
  BatchNorm2d(F₁)
  Conv2d(F₁, F₁·D, (C, 1), groups=F₁)          ← depthwise spatial
  BatchNorm2d(F₁·D) → ELU → AvgPool2d(1,4) → Dropout(p)

Block 2 — Separable Temporal:
  Conv2d(F₁·D, F₁·D, (1, K₂), groups=F₁·D)    ← depthwise temporal
  Conv2d(F₁·D, F₂, (1,1))                        ← pointwise
  BatchNorm2d(F₂) → ELU → AvgPool2d(1,8) → Dropout(p)

Head:
  AdaptiveAvgPool2d(1,1) → Flatten → Linear(F₂, 1) → σ(·) → probability
```

**Default parameters:**

| Symbol | Config key | Default |
|--------|-----------|---------|
| `F_1` | `temporal_filters` | 8 |
| `D` | `depth_multiplier` | 2 |
| `F_2` | `pointwise_filters` | 16 |
| `K_1` | `kernel_length` | 64 (forced odd) |
| `K_2` | `separable_kernel_length` | 16 (forced odd) |
| `p` | `dropout` | 0.5 |

**Training:**

- **Loss:** `BCEWithLogitsLoss` with $\mathrm{pos\_weight} = n_{\text{neg}} / n_{\text{pos}}$ for automatic class balancing
- **Optimizer:** AdamW (lr = 10⁻³, weight_decay = 10⁻³)
- **Early stopping:** Patience 10 epochs on held-out validation loss
- **Gradient clipping:** Max norm 1.0
- **Normalization:** Channelwise standardization — mean and std computed on the training partition only
- **Validation split:** 20% of training data; group-aware (`GroupShuffleSplit`) when multiple subjects present, preserving independence during early stopping
- **LOSO:** Inner loop provides early stopping only — no hyperparameter grid search

---

## 5. Cross-Validation Schemes

**Module:** `cv.py`

### 5.1 Nested LOSO (Primary)

Leave-one-subject-out outer loop with group-aware inner CV for hyperparameter selection. Guarantees unbiased generalization estimates across subjects.

```
Outer loop: LeaveOneGroupOut
  └─ Per fold: feature harmonization on training set only
     Inner loop: GroupKFold (k ≤ n_train_subjects)
       Scoring: Pearson r (logged) + neg_MSE
       Refit criterion: neg_MSE (regression) | average_precision (classification, configurable)
```

**Inner CV fallback:** When fewer than 2 training subjects remain, the model is fitted directly without hyperparameter search.

---

### 5.2 Within-Subject CV (Block-Aware)

Block-aware CV within each subject. Run/block labels serve as groups to prevent temporal leakage across sessions.

```
For each subject:
  Outer loop: GroupKFold on block labels (k = min(n_blocks, outer_cv_splits))
    Inner loop: GroupKFold on remaining blocks (refit: neg_MSE for regression; roc_auc for classification)
```

**Ordered block mode** (`cv.within_subject_ordered_blocks: true`): Folds respect temporal ordering — all preceding blocks form the training set, the next block is the test set. Appropriate when temporal autocorrelation is present.

Block labels are read from `block`, `run_id`, `run`, `session`, or `run_num` columns of `events.tsv`. Subjects with missing or constant block labels are skipped.

---

### 5.3 CV Hygiene

**Function:** `apply_fold_specific_hygiene`

Prevents data leakage from test trials into fold-specific computations:

- **Individual Alpha Frequency (IAF):** Band boundary estimation computed on training trials only.
- **Global features (e.g., ITPC):** Cross-trial computations use `train_mask` to exclude test data.
- **All preprocessing statistics:** Imputation, variance thresholds, scaling, PCA — all estimated on training fold only.

Controlled by `machine_learning.cv.hygiene_enabled` (default `true`).

---

## 6. Evaluation Metrics

### 6.1 Regression

**Primary metric — Subject-level Fisher-z–aggregated Pearson correlation**

For each held-out subject $i$ with $n_i$ test trials, compute the Pearson correlation between predicted and true ratings:

$$

r_i = \frac{\sum_t (\hat{y}_{it} - \bar{\hat{y}}_i)(y_{it} - \bar{y}_i)}{\sqrt{\sum_t (\hat{y}_{it} - \bar{\hat{y}}_i)^2 \cdot \sum_t (y_{it} - \bar{y}_i)^2}}

$$

Aggregate across $S$ subjects using the Fisher z-transformation to handle the bounded support of $r$:

1. **Clip** to prevent arctanh explosion near $\pm 1$: $r_i^* = \text{clip}(r_i, r_\text{min}, r_\text{max})$
2. **Transform:** $z_i = \mathrm{arctanh}(r_i^*)$
3. **Weighted average:** $\bar{z} = \sum_i w_i z_i / \sum_i w_i$
4. **Back-transform:** $\bar{r} = \tanh(\bar{z})$

Weighting modes (configurable via `evaluation.subject_weighting`):

| Mode | Weight `w_i` | Notes |
|------|----------------|-------|
| `equal` (default) | `1` | Treats each subject as independent; robust to variable trial counts |
| `trial_count` | $\max(n_i - 3, 1)$ | Fisher information weighting; standard meta-analytic fixed-effects scheme |

**Secondary metric — Subject-level mean R²:**

$$

\overline{R^2} = \frac{1}{S}\sum_{i=1}^{S} R^2_i, \quad R^2_i = 1 - \frac{\sum_t(y_{it}-\hat{y}_{it})^2}{\sum_t(y_{it}-\bar{y}_i)^2}

$$

Both $\bar{r}$ and $\overline{R^2}$ receive permutation p-values when `n_perm > 0`.

**Confidence intervals** for $\bar{r}$ (`evaluation.ci_method`):

| Method | Computation |
|--------|-------------|
| `bootstrap` (default) | Resample subjects $B$ times with replacement; recompute weighted $\bar{z}$; $\text{CI} = [\mathrm{perc}_{2.5}, \mathrm{perc}_{97.5}]$ of $\tanh(\bar{z}^{(b)})$ |
| `fixed_effects` | $\mathrm{SE} = 1/\sqrt{\sum_i w_i}$; $\text{CI} = \tanh\bigl(\bar{z} \pm 1.96\mathrm{SE}\bigr)$ |

**Subject-level error metrics:**

$$

\overline{\text{MAE}} = \frac{1}{S}\sum_i\mathrm{MAE}_i, \qquad \overline{\text{RMSE}} = \frac{1}{S}\sum_i\mathrm{RMSE}_i

$$

with optional bootstrap CIs (subjects resampled). Pooled trial-level Pearson r, MAE, and RMSE are logged as **secondary diagnostics only** (subject is the inferential unit for LOSO).

---

### 6.2 Classification

**Primary metric:** Subject-level mean AUC — the unweighted mean of per-subject ROC-AUC values. Reported as valid only when at least `min_subjects_with_auc_for_inference` (default 2) subjects have both classes present.

**Bootstrap CIs** (subject-level bootstrap, $B = 1000$ by default) are computed for all metrics listed below.

| Metric | Formula |
|--------|---------|
| Accuracy | $(\text{TP}+\text{TN})/N$ |
| Balanced accuracy | $(\text{sensitivity} + \text{specificity})/2$ |
| AUC (ROC) | Area under the receiver operating characteristic curve |
| Average precision | Area under the precision–recall curve |
| F1 | $2\text{precision}\cdot\text{recall}/(\text{precision}+\text{recall})$ |
| Sensitivity (recall) | $\text{TP}/(\text{TP}+\text{FN})$ |
| Specificity | $\text{TN}/(\text{TN}+\text{FP})$ |
| Brier score | $\frac{1}{N}\sum_i(\hat{p}_i - y_i)^2$ — probability calibration |
| ECE | $\sum_{k=1}^{10}\frac{|B_k|}{N}\bigl|\bar{p}_k - \bar{y}_k\bigr|$ — Expected Calibration Error over 10 uniform probability bins |

Pooled trial-level metrics are reported alongside subject-level metrics for diagnostics. Subject-level means with bootstrap CIs are the primary inferential statistics.

---

## 7. Permutation Testing

**Functions:** `cv.py → run_permutation_test`; `orchestration.py → _run_classification_permutations`

Constructs an empirical null distribution by re-running the full nested CV pipeline with permuted labels. All hyperparameter tuning is repeated on each permuted dataset.

### 7.1 Permutation Schemes

| Scheme | Behavior |
|--------|----------|
| `within_subject_within_block` | Permute labels within each subject × block combination. Preserves both subject-level and block-level dependence structure. **Default.** |
| `within_subject` | Permute labels within each subject. Automatic fallback when block labels are unavailable or when block-level permutation is ineffective. |

A permutation is **effective** only if it changes at least `min_label_shuffle_fraction` (default 1%) of labels. When the primary scheme fails, a fallback to `within_subject` is attempted per permutation before discarding.

### 7.2 P-value Computation

**Regression** (two-tailed):

$$

p = \frac{\#\bigl\{|s(y^\pi_j)| \geq |s(y)|\bigr\} + 1}{n_\text{perm,valid} + 1}

$$

where $s(\cdot)$ denotes the subject-level Fisher-z–aggregated Pearson r. Computed independently for both $\bar{r}$ and $\overline{R^2}$.

**Classification** (one-tailed; AUC $\geq$ observed):

$$

p = \frac{\#\bigl\{\text{AUC}(y^\pi_j) \geq \text{AUC}(y)\bigr\} + 1}{n_\text{perm,valid} + 1}

$$

### 7.3 Quality Gate

At least `min_valid_permutation_fraction` (default 50%) of permutations must produce finite metrics; a `RuntimeError` is raised otherwise.

> **Recommendation:** Use ≥ 1 000 permutations for a single test. For multiple comparisons across frequency bands or feature groups, use ≥ 5 000–10 000.

---

## 8. Time Generalization Analysis

**Module:** `time_generalization.py → time_generalization_regression`

Trains a decoding model in each time window and evaluates it in all other windows. The resulting $W \times W$ matrix of prediction performance reveals whether the neural representation is **temporally stable** (broad off-diagonal generalization) or **dynamic** (diagonal-only).

### 8.1 Method

1. **Epoch loading:** Load raw epoch tensors across subjects; align to the common channel set within each LOSO fold.

2. **Sliding windows:** Partition the active period $[t_\text{min}, t_\text{max}]$ (configurable) into overlapping windows of length $\Delta t$ with step $\delta t$. Window centers form both axes of the generalization matrix.

3. **Window feature extraction:** For window $w$, compute mean channel activity per trial:

$$

x_{w,\text{trial},c} = \frac{1}{|W_w|}\sum_{t \in W_w} \text{data}[\text{trial}, c, t]

$$

yielding an $N_\text{trials} \times C$ feature matrix per window.

4. **LOSO outer loop:** For each training window $i$, fit a Ridge regression model:
```
SimpleImputer → StandardScaler → TransformedTargetRegressor(Ridge(α), PowerTransformer)
```
Optionally tune $\alpha$ via inner `GroupKFold` (`use_ridgecv: true`). For each test window $j$, predict on the held-out subject and record Pearson r and R².

5. **Aggregation:** Stack per-fold matrices; for cell $(i,j)$, aggregate over $S_{ij}$ valid subjects using Fisher-z averaging (equal subject weights):

$$

r_{ij}^\text{agg} = \tanh\left(\frac{1}{S_{ij}}\sum_{s=1}^{S_{ij}} \mathrm{arctanh}\bigl(\text{clip}(r_{ij}^{(s)})\bigr)\right)

$$

Cells with $S_{ij} < $  `min_subjects_per_cell` or total trial count $ < $  `min_count_per_cell` are excluded (set to `NaN`).

### 8.2 Significance Testing (`n_perm > 0`)

Three corrections are applied simultaneously to all tested cells:

| Method | Description |
|--------|-------------|
| **FDR-BH** | Benjamini-Hochberg procedure applied across all finite cells. Controls false discovery rate. |
| **Max-stat (FWER)** | Threshold at the 95th percentile of the permutation null distribution of $\max_{(i,j)} |r_{ij}^\pi|$. Controls familywise error rate. |
| **Cluster-FWER** | Contiguous cells exceeding a forming threshold are grouped; cluster sizes compared to the null distribution of maximum cluster sizes. Controls FWER at cluster level. |

**Cell-level p-value:**

$$

p_{ij} = \frac{\#\bigl\{|r_{ij}^\pi| \geq |r_{ij}^\text{obs}|\bigr\} + 1}{n_\text{perm,valid} + 1}

$$

**Quality gates:**

| Gate | Config key | Default |
|------|-----------|---------|
| Min subjects per cell | `min_subjects_per_cell` | 2 |
| Min trials per cell | `min_count_per_cell` | 15 |
| Min samples per window | `min_samples_per_window` | 15 |
| Min samples for correlation | `min_samples_for_corr` | 10 |
| Min valid fold fraction | `min_valid_fold_fraction` | 0.8 |
| Min valid permutation fraction | `min_valid_permutation_fraction` | 0.8 |

---

## 9. SHAP Feature Importance

**Module:** `shap_importance.py`

SHAP (SHapley Additive exPlanations) values decompose each prediction into per-feature additive contributions satisfying efficiency, symmetry, and the dummy axioms. Per-feature importance is the mean absolute SHAP value:

$$

\phi_k = \frac{1}{N}\sum_{i=1}^{N} |\phi_k(x_i)|

$$

### 9.1 Explainer Selection

Automatically determined from the fitted estimator type (after pipeline preprocessing steps have been applied):

| Model type | Explainer | Detection criterion |
|------------|-----------|---------------------|
| Tree-based (RF) | `shap.TreeExplainer` | `hasattr(estimator, "feature_importances_")` |
| Linear (ElasticNet, Ridge, LR) | `shap.LinearExplainer` | `hasattr(estimator, "coef_")` |
| Other (SVM, ensemble) | `shap.KernelExplainer` | Fallback; 100 background samples |

### 9.2 Pipeline Awareness

Pipeline preprocessing is applied to transform $X$ before SHAP computation. Feature names are propagated through selectors (`get_support`) and transformers (`get_feature_names_out`). After PCA, components are relabeled `PC1`, `PC2`, ….

### 9.3 Cross-Fold Aggregation

For each LOSO fold: fit model on training data (with inner CV tuning), compute SHAP values on the test fold. Aggregate by feature name across folds, reporting mean and inter-fold standard deviation:

$$

\bar{\phi}_k = \frac{1}{K}\sum_{f=1}^{K} \phi_k^{(f)}, \qquad \sigma_k = \mathrm{std}_f\left(\phi_k^{(f)}\right)

$$

**Completion gate:** At least `analysis.shap.min_valid_fold_fraction` (default 0.8) of folds must succeed.

### 9.4 Grouped Summaries

When `interpretability.grouped_outputs: true`, feature names are parsed via `NamingSchema` into structured metadata (group, frequency band, ROI, scope, statistic). Importance is aggregated at two levels:

$$

\phi_\mathcal{G} = \sum_{k \in \mathcal{G}} \phi_k, \qquad \text{share}_\mathcal{G} = \frac{\phi_\mathcal{G}}{\sum_k \phi_k}

$$

Outputs: `shap_importance_by_group_band.tsv`, `shap_importance_by_group_band_roi.tsv`.

---

## 10. Permutation Feature Importance

**Function:** `orchestration.py → _run_permutation_importance_stage`

Model-agnostic importance via column permutation (Breiman 2001). For each feature $k$:

$$

\text{imp}_k = \mathrm{score}(y, \hat{y}) - \mathbb{E}_\pi\left[\mathrm{score}\left(y, \hat{y}^{(\pi_k)}\right)\right]

$$

where $\hat{y}^{(\pi_k)}$ is the prediction after randomly permuting column $k$ in the test set. Scoring metric: $R^2$ (`n_repeats` permutations per feature, default 5).

**Aggregation:** Features excluded by fold harmonization receive `NaN`. Mean and std are reported across folds. Same grouped summaries as SHAP (by group × band and group × band × ROI).

---

## 11. Uncertainty Quantification

**Module:** `uncertainty.py` — integrated into pipeline runners via `_run_uncertainty_stage`

Distribution-free prediction intervals via conformal prediction. For exchangeable data:

$$

\mathbb{P}\left(y \in \hat{C}(x)\right) \geq 1 - \alpha

$$

The stage runs within LOSO: models are tuned via inner CV, then conformal intervals are computed on held-out subjects using the **CV+ method** by default. Empirical coverage and mean interval width are reported per subject.

---

### 11.1 Split Conformal

1. Split training set into proper training ($\approx 80\%$) and calibration ($\approx 20\%$, ≥ 2 samples). Group-aware via `GroupShuffleSplit` when subject labels are available.
2. Fit model; compute calibration residuals $s_i = |y_i - \hat{y}_i|$, $i \in \mathcal{D}_\text{cal}$.
3. Conformal quantile:

$$

\hat{q} = \mathrm{Quantile}\left(\{s_i\},\; \frac{\lceil(|\mathcal{D}_\text{cal}|+1)(1-\alpha)\rceil}{|\mathcal{D}_\text{cal}|}\right)

$$

4. Prediction interval: $\hat{C}(x) = [\hat{y}(x) - \hat{q},\; \hat{y}(x) + \hat{q}]$.

---

### 11.2 CV+ (Jackknife+) — Default in Orchestration Stage

More data-efficient than split conformal; uses cross-validation residuals:

1. $k$-fold CV on training data (group-aware). For each fold $f$ and test point $x$:
   - Train on folds $\neq f$; compute residuals $\{s_i^{(f)}\}_{i \in \text{fold}_f}$.
   - Predict $\hat{y}_f(x)$.
2. Prediction interval at $x$ using conservative order-statistic quantiles over the full ensemble of $\{(\hat{y}_f(x), s^{(f)})\}$:

$$

\hat{C}(x) = \left[\hat{q}_\alpha\left(\{\hat{y}_f(x) - s_j^{(f)}\}_{f,j}\right),\; \hat{q}_{1-\alpha}\left(\{\hat{y}_f(x) + s_j^{(f)}\}_{f,j}\right)\right]

$$

A full model trained on all training data provides the point prediction.

---

### 11.3 Conformalized Quantile Regression (CQR)

Adaptive intervals for heteroscedastic data — interval width adapts to local uncertainty.

1. CV to collect out-of-fold quantile predictions at levels $\alpha/2$ and $1-\alpha/2$ via `GradientBoostingRegressor(loss="quantile")`.
2. Conformity scores: $E_i = \max\bigl(\hat{q}_{\alpha/2}(x_i) - y_i,\; y_i - \hat{q}_{1-\alpha/2}(x_i)\bigr)$.
3. Conformal quantile $\hat{Q}$ from $\{E_i\}$.
4. Prediction interval: $\hat{C}(x) = \bigl[\hat{q}_{\alpha/2}(x) - \hat{Q},\; \hat{q}_{1-\alpha/2}(x) + \hat{Q}\bigr]$.

---

## 12. Pipeline Runners

**Module:** `orchestration.py`

All runners accept feature subsetting arguments (`feature_families`, `feature_bands`, `feature_segments`, `feature_scopes`, `feature_stats`) and write structured output directories (see §13).

---

### 12.1 `run_regression_ml` — LOSO Regression

1. Assemble design matrix via `load_active_matrix` (multi-subject, optional covariate appending, target validation).
2. Warn (or raise if `strict_regression_target_continuous: true`) when target appears binary-like.
3. Build model pipeline and hyperparameter grid (ElasticNet, Ridge, or RF).
4. Run `nested_loso_predictions_matrix`: outer LOSO with inner `GroupKFold`; refit criterion `neg_MSE`.
5. Compute subject-level $\bar{r}$ (primary) and $\overline{R^2}$ (secondary), both with bootstrap CI.
6. Optionally run permutation test → two-tailed p-values for $\bar{r}$ and $\overline{R^2}$.
7. Compute and export LOSO mean-predictor baseline.
8. Run SHAP and permutation importance stages when enabled.
9. Generate diagnostic plots.

---

### 12.2 `run_within_subject_regression_ml` — Within-Subject Regression

Block-aware regression within each subject. Requires block/run labels in `events.tsv`.

1. Load and validate design matrix; drop trials with missing block labels.
2. Create within-subject folds via `create_within_subject_folds` (GroupKFold or temporal ordering per subject).
3. For each fold, fit with block-aware inner CV (refit: `neg_MSE`).
4. Aggregate; compute subject-level $\bar{r}$ with CI.
5. Block-aware permutation test (within-subject × block shuffle).
6. Baseline: per-fold training-set mean prediction (within-subject mean predictor).

---

### 12.3 `run_classification_ml` — LOSO Classification

1. Load feature matrix via `load_active_matrix` (standard models) or epoch tensor via `load_epoch_tensor_matrix` (CNN). Binary target; optional `binary_threshold` to binarize a continuous column.
2. Run `nested_loso_classification` (SVM/LR/RF/Ensemble) or `nested_loso_cnn_classification` (CNN). Inner loop: `StratifiedGroupKFold` with configurable scoring (default `average_precision`).
3. Compute per-subject metrics (AUC, balanced accuracy, F1, etc.) with bootstrap CIs.
4. Compute calibration metrics (Brier score, ECE, calibration curve from 10 uniform bins).
5. Validate fold failure fraction against `max_failed_fold_fraction` (default 0.25).
6. Optionally run permutation test → one-tailed AUC p-value.

---

### 12.4 `run_within_subject_classification_ml` — Within-Subject Classification

Same structure as LOSO classification but uses `GroupKFold` on block labels within each subject. Inner loop: `StratifiedGroupKFold` with `scoring="roc_auc"`. Same block-aware permutation test.

---

### 12.5 `run_model_comparison_ml` — Model Comparison

Compares ElasticNet, Ridge, and Random Forest on **identical outer LOSO folds**.

1. Create shared LOSO outer folds.
2. For each model, run nested CV with inner `GroupKFold` (refit: Pearson r). Record per-fold R² and MAE.
3. Compute summary statistics (mean ± std and bootstrap CI per model).
4. **Pairwise inference:** Paired sign-flip permutation test on per-fold $\Delta R^2$ and $\Delta\text{MAE}$:

$$

p\bigl(\Delta R^2_{AB}\bigr) = \frac{\#\bigl\{\bigl|\overline{s \cdot \Delta R^2_{AB}}\bigr| \geq \bigl|\overline{\Delta R^2_{AB}}\bigr|\bigr\} + 1}{n_\text{perm} + 1}, \quad s_j \in \{-1,+1\}

$$

5. Holm–Bonferroni multiple-comparison correction across all pairwise $\Delta R^2$ and $\Delta\text{MAE}$ tests.
6. Export `model_comparison.tsv` (per fold) and `model_comparison_summary.json` (aggregated with pairwise inference).

---

### 12.6 `run_incremental_validity_ml` — Incremental Validity

Quantifies the out-of-fold gain in R² when adding EEG features over a baseline predictor (e.g., stimulus temperature). Uses **identical model family** (ElasticNet with covariate protection) for both conditions to isolate information gain rather than algorithmic differences.

1. Define baseline predictor matrix $X_\text{base}$ from `events.tsv` (default: `temperature`). Guard against target-variable leakage.
2. Full matrix: $X_\text{full} = [X_\text{EEG} \mid X_\text{base}]$ — baseline columns appended as protected covariates.
3. For each LOSO fold:
   - Fit tuned ElasticNet on $X_\text{base,train}$ → predict $\hat{y}_\text{base}$
   - Fit tuned ElasticNet on $X_\text{full,train}$ → predict $\hat{y}_\text{full}$
   - Compute: $\Delta R^2_\text{fold} = R^2\left(y_\text{test},\hat{y}_\text{full}\right) - R^2\left(y_\text{test},\hat{y}_\text{base}\right)$

4. **Primary estimate:**

$$

\overline{\Delta R^2} = \frac{1}{K}\sum_{k=1}^{K} \Delta R^2_k

$$

5. Bootstrap CI on $\overline{\Delta R^2}$ (resample folds). Paired sign-flip permutation p-value when `n_perm > 0`.

---

### 12.7 `run_time_generalization` — Time Generalization

Delegates to `time_generalization_regression` (see §8). Saves outputs to `results_root/time_generalization/`.

---

## 13. Output Structure

Each runner creates a self-contained results directory:

```
results_root/{mode}/
├── data/
│   ├── loso_predictions.tsv      ← y_true, y_pred, subject_id, fold_id
│   ├── loso_predictions.parquet
│   ├── loso_indices.tsv
│   └── baseline_predictions.tsv
├── metrics/
│   ├── pooled_metrics.json       ← subject-level (primary) + pooled-trial (secondary)
│   ├── per_subject_correlations.tsv
│   ├── per_subject_errors.tsv
│   └── per_subject_metrics.tsv  (classification)
├── models/
│   └── best_params_{model}.jsonl
├── null/
│   └── loso_null_{model}.npz    ← null_r, null_r2, n_completed
├── importance/
│   ├── shap_importance.tsv
│   ├── shap_importance_by_group_band.tsv
│   ├── shap_importance_by_group_band_roi.tsv
│   ├── permutation_importance.tsv
│   └── permutation_importance_by_group_band.tsv
├── plots/
│   └── *.{png,pdf,svg}           ← publication-style diagnostic figures
└── reports/
    ├── reproducibility_info.json
    ├── included_subjects.tsv
    └── excluded_subjects.tsv
```

---

## 14. Reproducibility

**Function:** `write_reproducibility_info`

Each run writes `reports/reproducibility_info.json` containing:

- `sklearn`, `numpy`, `pandas` library versions
- RNG seed used throughout
- Subject list and SHA-256 data signature (hash of sorted subject IDs)
- Full `machine_learning` configuration snapshot

---

## 15. Configuration Reference

All settings reside under `machine_learning` in `eeg_config.yaml`.

### 15.1 Preprocessing

| Key | Description | Default |
|-----|-------------|---------|
| `constants.variance_threshold` | Base variance filter threshold $\theta$ | `0.0` |
| `preprocessing.imputer_strategy` | Imputation statistic (`median`, `mean`) | `"median"` |
| `preprocessing.power_transformer_method` | Target power transform | `"yeo-johnson"` |
| `preprocessing.power_transformer_standardize` | Standardize after transform | `true` |
| `preprocessing.variance_threshold_grid` | Grid search values for $\theta$ | `[0.0, 0.01, 0.1]` |
| `preprocessing.feature_selection_percentile` | Univariate top-$k\%$ selection (100 = disabled) | `100.0` |
| `preprocessing.deconfound` | Regress out covariates from EEG features | `false` |
| `preprocessing.spatial_regions_allowed` | ROI whitelist for `SpatialFeatureSelector` | `[]` |
| `preprocessing.pca.enabled` | Enable PCA | `false` |
| `preprocessing.pca.n_components` | Variance fraction or integer component count | `0.95` |
| `preprocessing.pca.whiten` | Whiten principal components | `false` |
| `preprocessing.pca.svd_solver` | SVD solver (`auto`, `full`, `randomized`) | `"auto"` |

### 15.2 Regression Models

| Key | Description | Default |
|-----|-------------|---------|
| `models.elasticnet.alpha_grid` | ElasticNet $\alpha$ search | `[0.01, 0.1, 1.0, 10.0]` |
| `models.elasticnet.l1_ratio_grid` | ElasticNet $\rho$ search | `[0.1, 0.5, 0.9]` |
| `models.elasticnet.max_iter` | Solver max iterations | `10000` |
| `models.elasticnet.tol` | Solver convergence tolerance | `1e-4` |
| `models.ridge.alpha_grid` | Ridge $\alpha$ search | `[0.01, 0.1, 1.0, 10.0, 100.0]` |
| `models.random_forest.n_estimators` | Number of trees `B` | `100` |
| `models.random_forest.bootstrap` | Bootstrap resampling | `true` |
| `models.random_forest.max_depth_grid` | Depth search | `[5, 10, 20, null]` |
| `models.random_forest.min_samples_split_grid` | Split threshold search | `[2, 5, 10]` |
| `models.random_forest.min_samples_leaf_grid` | Leaf size search | `[1, 2, 4]` |

### 15.3 Classification Models

| Key | Description | Default |
|-----|-------------|---------|
| `models.svm.kernel` | SVM kernel type | `"rbf"` |
| `models.svm.C_grid` | SVM regularization `C` search | `[0.1, 1.0, 10.0]` |
| `models.svm.gamma_grid` | RBF kernel $\gamma$ search | `["scale", "auto"]` |
| `models.svm.class_weight` | Class weighting | `"balanced"` |
| `models.logistic_regression.penalty` | Regularization type | `"l2"` |
| `models.logistic_regression.C_grid` | Inverse regularization `C` search | `[0.01, 0.1, 1.0, 10.0]` |
| `models.logistic_regression.l1_ratio_grid` | ElasticNet mixing search | `[0.1, 0.5, 0.9]` |
| `models.logistic_regression.class_weight` | Class weighting | `"balanced"` |
| `classification.resampler` | Class imbalance strategy (`none`, `undersample`, `smote`) | `"none"` |
| `classification.calibrate_ensemble` | Calibrate SVM/RF in ensemble via Platt scaling | `false` |
| `classification.scoring` | Inner CV scoring metric | `"average_precision"` |
| `classification.model` | Default classifier | `"svm"` |
| `classification.max_failed_fold_fraction` | Maximum allowed fold failure rate | `0.25` |
| `classification.min_subjects_with_auc_for_inference` | Minimum subjects for AUC inference | `2` |

### 15.4 CNN

| Key | Description | Default |
|-----|-------------|---------|
| `models.cnn.temporal_filters` | `F_1` | `8` |
| `models.cnn.depth_multiplier` | `D` | `2` |
| `models.cnn.pointwise_filters` | `F_2` | `16` |
| `models.cnn.kernel_length` | `K_1` | `64` |
| `models.cnn.separable_kernel_length` | `K_2` | `16` |
| `models.cnn.dropout` | Dropout rate `p` | `0.5` |
| `models.cnn.batch_size` | Mini-batch size | `64` |
| `models.cnn.max_epochs` | Maximum training epochs | `75` |
| `models.cnn.patience` | Early stopping patience (epochs) | `10` |
| `models.cnn.learning_rate` | AdamW lr | `1e-3` |
| `models.cnn.weight_decay` | AdamW weight decay | `1e-3` |
| `models.cnn.gradient_clip_norm` | Max gradient norm | `1.0` |
| `models.cnn.val_fraction` | Validation fraction for early stopping | `0.2` |
| `models.cnn.use_cuda` | Enable GPU | `false` |

### 15.5 Cross-Validation

| Key | Description | Default |
|-----|-------------|---------|
| `cv.default_n_splits` | Default inner CV splits | `5` |
| `cv.inner_splits` | Inner CV splits (used by importance/SHAP stages) | `5` |
| `cv.outer_splits` | Outer CV splits for within-subject mode | `5` |
| `cv.hygiene_enabled` | Enable fold-specific CV hygiene | `true` |
| `cv.within_subject_ordered_blocks` | Temporal block ordering within subjects | `false` |
| `cv.permutation_scheme` | Label permutation structure | `"within_subject_within_block"` |
| `cv.min_label_shuffle_fraction` | Min fraction of labels changed per permutation | `0.01` |
| `cv.min_valid_permutation_fraction` | Min fraction of permutations yielding finite metrics | `0.5` |
| `cv.min_valid_permutation_fold_fraction` | Min fold completion rate per permutation (within-subject) | `1.0` |

### 15.6 Evaluation

| Key | Description | Default |
|-----|-------------|---------|
| `evaluation.ci_method` | CI method for regression r (`bootstrap`, `fixed_effects`) | `"bootstrap"` |
| `evaluation.bootstrap_iterations` | Bootstrap resamples | `1000` |
| `evaluation.subject_weighting` | Fisher-z weighting (`equal`, `trial_count`) | `"equal"` |

### 15.7 Targets

| Key | Description | Default |
|-----|-------------|---------|
| `targets.regression` | Continuous target column in `events.tsv` | *(configurable)* |
| `targets.classification` | Binary target column | `"binary_outcome"` |
| `targets.binary_threshold` | Threshold to binarize a continuous target | `null` |
| `targets.strict_regression_target_continuous` | Raise error for binary-like regression targets | `false` |

### 15.8 Time Generalization

| Key | Description | Default |
|-----|-------------|---------|
| `analysis.time_generalization.active_window` | Time range `[t_min, t_max]` (seconds) | `[3.0, 10.5]` |
| `analysis.time_generalization.window_len` | Window duration (seconds) | `0.75` |
| `analysis.time_generalization.step` | Window step (seconds) | `0.25` |
| `analysis.time_generalization.default_alpha` | Ridge $\alpha$ when not tuning | `1.0` |
| `analysis.time_generalization.use_ridgecv` | Tune Ridge $\alpha$ via inner GroupKFold | `false` |
| `analysis.time_generalization.alpha_grid` | $\alpha$ grid for inner CV | `[0.01, 0.1, 1.0, 10.0, 100.0]` |
| `analysis.time_generalization.cluster_threshold` | Cluster-forming threshold $\alpha_\text{cluster}$ | `0.05` |
| `analysis.time_generalization.min_subjects_per_cell` | Min subjects for a valid cell | `2` |
| `analysis.time_generalization.min_count_per_cell` | Min trials for a valid cell | `15` |
| `analysis.time_generalization.min_valid_fold_fraction` | Min fold completion rate | `0.8` |
| `analysis.time_generalization.min_valid_permutation_fraction` | Min permutation completion rate | `0.8` |

### 15.9 Feature Importance

| Key | Description | Default |
|-----|-------------|---------|
| `analysis.shap.enabled` | Run SHAP importance | `true` |
| `analysis.shap.min_valid_fold_fraction` | Min fold completion for SHAP | `0.8` |
| `analysis.permutation_importance.enabled` | Run permutation importance | `false` |
| `analysis.permutation_importance.n_repeats` | Shuffles per feature per fold | `5` |
| `analysis.permutation_importance.min_valid_fold_fraction` | Min fold completion | `0.8` |
| `interpretability.grouped_outputs` | Export importance grouped by band/ROI | `true` |

### 15.10 Incremental Validity

| Key | Description | Default |
|-----|-------------|---------|
| `incremental_validity.baseline_predictors` | Baseline predictor columns from `events.tsv` | `["temperature"]` |
| `incremental_validity.require_baseline_predictors` | Raise if baseline columns are missing | `true` |

### 15.11 Uncertainty Quantification

| Key | Description | Default |
|-----|-------------|---------|
| `analysis.uncertainty.min_valid_fold_fraction` | Min fold completion for conformal intervals | `0.8` |

### 15.12 Data

| Key | Description | Default |
|-----|-------------|---------|
| `data.feature_harmonization` | Default harmonization mode | `"intersection"` |
| `data.max_excluded_subject_fraction` | Max fraction of requested subjects that may be excluded | `1.0` |

### 15.13 Plotting

| Key | Description | Default |
|-----|-------------|---------|
| `plotting.enabled` | Auto-generate diagnostic figures | `true` |
| `plotting.formats` | Output formats (`["png"]`, `["png","svg"]`, …) | `["png"]` |
| `plotting.dpi` | Figure resolution | `300` |
| `plotting.top_n_features` | Top N features in importance figures | `20` |
| `plotting.include_diagnostics` | Include prediction scatter / calibration plots | `true` |

---

## 16. Dependencies

| Library | Role |
|---------|------|
| **scikit-learn** | Pipelines, CV, `GridSearchCV`, metrics, imputation, scaling, PCA, Ridge, ElasticNet, RF, SVM, LR |
| **imbalanced-learn** | `ImbPipeline`, `SMOTE`, `RandomUnderSampler` |
| **NumPy / SciPy** | Numerical computation, Pearson correlation, Fisher z-transform |
| **pandas** | Feature table I/O, metadata handling |
| **statsmodels** | FDR (Benjamini-Hochberg), Holm correction |
| **joblib** | Parallel outer-fold execution |
| **SHAP** | Feature importance *(optional — `pip install shap`)* |
| **PyTorch** | EEGNet CNN classifier *(optional — `pip install torch`)* |
| **MNE-Python** | Epoch loading for time generalization |
| **matplotlib** | Diagnostic figures |
