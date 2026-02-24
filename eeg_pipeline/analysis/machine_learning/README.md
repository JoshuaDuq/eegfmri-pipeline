# Machine Learning Pipeline

**Module:** `eeg_pipeline.analysis.machine_learning`

This document is the methods reference for the trial-level predictive modeling pipeline.
The pipeline supports continuous regression (pain intensity) and binary classification
(pain vs. no-pain).
The **subject is the statistical unit** throughout: all primary metrics are subject-level
aggregates computed under leave-one-subject-out (LOSO) cross-validation.

---

## Table of Contents

1. [Notation](#1-notation)
2. [Module Structure](#2-module-structure)
3. [Data Flow](#3-data-flow)
4. [Preprocessing](#4-preprocessing)
5. [Regression Models](#5-regression-models)
6. [Classification Models](#6-classification-models)
7. [Cross-Validation Schemes](#7-cross-validation-schemes)
8. [Evaluation Metrics](#8-evaluation-metrics)
9. [Permutation Testing](#9-permutation-testing)
10. [Time Generalization Analysis](#10-time-generalization-analysis)
11. [SHAP Feature Importance](#11-shap-feature-importance)
12. [Permutation Feature Importance](#12-permutation-feature-importance)
13. [Uncertainty Quantification](#13-uncertainty-quantification)
14. [Pipeline Runners](#14-pipeline-runners)
15. [Output Structure](#15-output-structure)
16. [Reproducibility](#16-reproducibility)
17. [Configuration Reference](#17-configuration-reference)
18. [Dependencies](#18-dependencies)

---

## 1. Notation

| Symbol | Meaning |
|--------|---------|
| $N$ | Number of trials in the design matrix |
| $P$ | Number of EEG feature columns |
| $C$ | Number of covariate columns |
| $S$ | Number of subjects |
| $K$ | Number of outer CV folds |
| $B$ | Number of trees (Random Forest) or bootstrap resamples |
| $X \in \mathbb{R}^{N \times P}$ | EEG feature design matrix |
| $Z \in \mathbb{R}^{N \times C}$ | Covariate matrix |
| $y \in \mathbb{R}^N$ | Continuous regression target |
| $\alpha$ | Regularization strength (ElasticNet, Ridge) or significance level (context-dependent) |
| $\rho$ | L1 mixing ratio (ElasticNet `l1_ratio`) |
| $\theta$ | Variance threshold for feature filtering |
| $r_i$ | Pearson correlation for subject $i$ |
| $z_i = \mathrm{arctanh}(r_i)$ | Fisher $z$-transformed correlation |
| $\bar{r}$ | Subject-level Fisher-$z$-aggregated Pearson correlation (primary metric) |
| $n_\text{perm}$ | Number of permutations |

---

## 2. Module Structure

| File | Responsibility |
|------|---------------|
| `orchestration.py` | High-level pipeline runners (entry points) |
| `cv.py` | CV fold creation, aggregation, permutation tests |
| `pipelines.py` | Regression pipeline factories (ElasticNet, Ridge, RF) |
| `classification.py` | Classification pipeline factories (SVM, LR, RF, Ensemble) |
| `cnn.py` | EEGNet-style CNN classifier (PyTorch) |
| `preprocessing.py` | CV-safe transformers (inf/NaN, variance, deconfounding) |
| `config.py` | Unified configuration loader with defaults |
| `shap_importance.py` | SHAP feature importance (per-fold aggregated) |
| `uncertainty.py` | Conformal prediction intervals |
| `time_generalization.py` | Temporal generalization regression |
| `feature_metadata.py` | Feature name parsing and grouped importance |
| `plotting.py` | Auto-generated publication-style figures |

---

## 3. Data Flow

Per-trial feature tables written by the feature extraction pipeline to
`derivatives/*/eeg/features/` are assembled into a design matrix $X \in \mathbb{R}^{N \times P}$
via `load_active_matrix`, merged with behavioral targets from `events.tsv`, and passed
through nested CV.

All preprocessing parameters are estimated exclusively on the training fold of each split.
No statistics from the test fold influence any fitted transformer.

---

## 4. Preprocessing

**Modules:** `preprocessing.py`, `pipelines.py`

### 4.1 Standard Preprocessing Chain

All model pipelines share the following chain. Every transformer is CV-safe: fitted on
the training fold only, applied to both train and test.

| Step | Transformer | Behavior |
|------|-------------|---------|
| 1 | `ReplaceInfWithNaN` | Replaces $\pm\infty$ with `NaN` before imputation. |
| 2 | `DropAllNaNColumns` | Removes columns with no finite value in the training fold. Raises a `ValueError` if all columns are removed. |
| 3 | `SpatialFeatureSelector` | Retains features whose inferred ROI appears in `spatial_regions_allowed`. Skipped when the list is empty. |
| 4 | `SimpleImputer` | Replaces remaining `NaN` with the training-fold statistic (default: `median`). |
| 5 | `VarianceThreshold` | Drops features with variance $\sigma^2 < \theta$ (default $\theta = 0.0$). Raises a descriptive error if all features are removed. |
| 6 | `SelectPercentile` *(optional)* | Retains top-$k\%$ features by univariate score (`f_regression` or `f_classif`). Active only when `feature_selection_percentile < 100`. |
| 7 | `StandardScaler` | Zero-mean, unit-variance standardization. Applied for all linear models and whenever PCA is enabled. Skipped for Random Forest. |
| 8 | `PCA` *(optional)* | Dimensionality reduction. `n_components` is a variance fraction (default 0.95) or integer count. Disabled by default. |

### 4.2 Covariate Handling

When `covariates` are provided, they are appended as the last $C$ columns of $X$
and the pipeline splits into two parallel branches via `ColumnTransformer`:

```
ColumnTransformer
├── "eeg" branch : columns [0 … P−C−1]  → full pipeline (steps 1–8)
└── "cov" branch : columns [P−C … P−1]  → ReplaceInfWithNaN → SimpleImputer(most_frequent) → [StandardScaler]
```

Covariate columns are never dropped by variance filtering, spatial selection,
or feature harmonization.

### 4.3 Deconfounding

**Transformer:** `Deconfounder` — applied after `ColumnTransformer` when
`preprocessing.deconfound = true`.

Given the partitioned design matrix $X = [X_\text{EEG} \mid Z]$, the transformer
fits a linear model on the training fold:

```math
\hat{B} = (Z_\text{train}^\top Z_\text{train})^{-1} Z_\text{train}^\top X_\text{EEG,train},
```

and returns residuals:

```math
\tilde{X}_\text{EEG} = X_\text{EEG} - Z\hat{B}.
```

Covariate columns are discarded from the output matrix.

### 4.4 Feature Harmonization

Ensures consistent dimensionality across subjects with heterogeneous feature coverage.

| Mode | Behavior |
|------|---------|
| `intersection` *(default)* | Retain features for which every training subject has at least one finite value. Raises a `ValueError` if no features satisfy this criterion. |
| `union_impute` | Retain all features; imputation handles remaining missing values downstream. |

Covariate columns are excluded from harmonization.

---

## 5. Regression Models

**Module:** `pipelines.py`

All regression pipelines wrap the estimator in `TransformedTargetRegressor`, applying a
Yeo-Johnson power transform to the target $y$ before fitting and back-transforming
predictions to the original scale. This normalizes skewed pain-rating distributions.

**Target transform** — Yeo-Johnson $\psi_\lambda(y)$ (estimated by MLE on the training fold):

```math
\psi_\lambda(y) = \begin{cases}
\dfrac{(y+1)^\lambda - 1}{\lambda} & y \ge 0,\; \lambda \ne 0 \\[4pt]
\ln(y+1) & y \ge 0,\; \lambda = 0 \\[4pt]
-\dfrac{(1-y)^{2-\lambda}-1}{2-\lambda} & y < 0,\; \lambda \ne 2 \\[4pt]
-\ln(1-y) & y < 0,\; \lambda = 2
\end{cases}
```

### 5.1 ElasticNet

**Factory:** `create_elasticnet_pipeline`

Combined L1 + L2 penalized regression. Encourages sparsity (L1) while stabilizing
correlated features (L2):

```math
\hat{\beta} = \underset{\beta}{\arg\min}\;
\frac{1}{2n}\|y - X\beta\|_2^2
+ \alpha\!\left[\rho\|\beta\|_1 + \frac{1-\rho}{2}\|\beta\|_2^2\right].
```

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

### 5.2 Ridge

**Factory:** `create_ridge_pipeline`

L2-regularized regression (Tikhonov regularization):

```math
\hat{\beta} = \underset{\beta}{\arg\min}\; \|y - X\beta\|_2^2 + \alpha\|\beta\|_2^2.
```

**Hyperparameter grid:**

| Parameter | Default search space |
|-----------|----------------------|
| $\alpha$ | `[0.01, 0.1, 1.0, 10.0, 100.0]` |
| $\theta$ | `[0.0, 0.01, 0.1]` |

### 5.3 Random Forest Regressor

**Factory:** `create_rf_pipeline`

Bagged ensemble of decision trees; invariant to monotonic feature transforms.
No feature scaling applied.

```math
\hat{f}(x) = \frac{1}{B}\sum_{b=1}^{B} T_b(x), \quad T_b \text{ trained on bootstrap resample } \mathcal{D}_b.
```

**Pipeline:**

```
Preprocessing (no scaling)
  → TransformedTargetRegressor(RandomForestRegressor(B, bootstrap=True), PowerTransformer("yeo-johnson"))
```

**Hyperparameter grid:**

| Parameter | Default search space |
|-----------|----------------------|
| `max_depth` | `[5, 10, 20, null]` |
| `min_samples_split` | `[2, 5, 10]` |
| `min_samples_leaf` | `[1, 2, 4]` |
| $\theta$ | `[0.0, 0.01, 0.1]` |

---

## 6. Classification Models

**Module:** `classification.py`

Binary classification (pain vs. no-pain). All classifiers use `class_weight="balanced"`
and are wrapped in `imblearn.pipeline.Pipeline` to support optional class resampling.

### 6.1 Class Imbalance Handling

Configured via `classification.resampler`:

| Value | Strategy |
|-------|---------|
| `none` *(default)* | No resampling. |
| `undersample` | `RandomUnderSampler`: remove majority-class samples to match class counts. |
| `smote` | `SMOTE`: synthesize minority-class samples via k-NN interpolation. |

### 6.2 Support Vector Machine (SVM)

**Factory:** `create_svm_pipeline`

Soft-margin SVM with RBF kernel:

```math
\min_{w,b,\xi}\;\frac{1}{2}\|w\|^2 + C\sum_i \xi_i
\quad\text{s.t.}\quad
y_i(w^\top\phi(x_i)+b) \ge 1-\xi_i,\; \xi_i \ge 0,
```

with $K(x,x') = \exp(-\gamma\|x-x'\|^2)$.
Probability calibration via Platt scaling (`probability=True`).

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

### 6.3 Logistic Regression

**Factory:** `create_logistic_pipeline`

Regularized logistic regression:

```math
\hat{p}(y=1 \mid x) = \sigma(x^\top\beta + b) = \frac{1}{1+e^{-x^\top\beta - b}}.
```

Supports L2 (default), L1, and ElasticNet penalties.
Solver: `saga` for L1/ElasticNet; `lbfgs` for L2.

**Hyperparameter grid:**

| Parameter | Default search space |
|-----------|----------------------|
| $C = 1/\alpha$ | `[0.01, 0.1, 1.0, 10.0]` |
| `l1_ratio` (ElasticNet only) | `[0.1, 0.5, 0.9]` |
| $\theta$ | `[0.0, 0.01, 0.1]` |

### 6.4 Random Forest Classifier

**Factory:** `create_rf_classification_pipeline`

Bagged ensemble for binary classification. No feature scaling applied.

```
Preprocessing (no scaling) → [Resampler] → RandomForestClassifier(B=100, class_weight="balanced")
```

Hyperparameter grid: same structure as the regressor (§5.3).

### 6.5 Soft Voting Ensemble

**Factory:** `create_ensemble_pipeline`

Combines SVM, Logistic Regression, and Random Forest via probability averaging:

```math
\hat{p}_\text{ens}(y=1\mid x) =
\frac{1}{3}\bigl[\hat{p}_\text{SVM}(x) + \hat{p}_\text{LR}(x) + \hat{p}_\text{RF}(x)\bigr].
```

When `classification.calibrate_ensemble = true`, SVM and RF are individually wrapped in
`CalibratedClassifierCV(method="sigmoid", cv=2)` before voting.

### 6.6 EEGNet CNN

**Module:** `cnn.py`

EEGNet-style convolutional neural network operating on raw epoch tensors $(N, C, T)$,
loaded via `load_epoch_tensor_matrix` (distinct from `load_active_matrix`).

**Architecture:**

```
Input: (N, 1, C, T)

Block 1 — Temporal + Depthwise Spatial:
  Conv2d(1, F₁, (1, K₁), pad=(0, K₁//2))       ← temporal filters
  BatchNorm2d(F₁)
  Conv2d(F₁, F₁·D, (C, 1), groups=F₁)           ← depthwise spatial
  BatchNorm2d(F₁·D) → ELU → AvgPool2d(1,4) → Dropout(p)

Block 2 — Separable Temporal:
  Conv2d(F₁·D, F₁·D, (1, K₂), groups=F₁·D)     ← depthwise temporal
  Conv2d(F₁·D, F₂, (1,1))                         ← pointwise
  BatchNorm2d(F₂) → ELU → AvgPool2d(1,8) → Dropout(p)

Head:
  AdaptiveAvgPool2d(1,1) → Flatten → Linear(F₂, 1) → σ(·) → probability
```

**Default parameters:**

| Symbol | Config key | Default |
|--------|-----------|---------|
| $F_1$ | `temporal_filters` | 8 |
| $D$ | `depth_multiplier` | 2 |
| $F_2$ | `pointwise_filters` | 16 |
| $K_1$ | `kernel_length` | 64 (forced odd) |
| $K_2$ | `separable_kernel_length` | 16 (forced odd) |
| $p$ | `dropout` | 0.5 |

**Training:**

| Setting | Value |
|---------|-------|
| Loss | `BCEWithLogitsLoss` with `pos_weight` $= n_\text{neg} / n_\text{pos}$ |
| Optimizer | AdamW (lr $= 10^{-3}$, weight_decay $= 10^{-3}$) |
| Early stopping | Patience 10 epochs on held-out validation loss |
| Gradient clipping | Max norm 1.0 |
| Normalization | Channel-wise mean and std from training partition only |
| Validation split | 20% of training data; `GroupShuffleSplit` when multiple subjects are present |
| LOSO inner loop | Early stopping only — no hyperparameter grid search |

---

## 7. Cross-Validation Schemes

**Module:** `cv.py`

### 7.1 Nested LOSO (Primary)

Leave-one-subject-out outer loop with group-aware inner CV for hyperparameter selection.
Guarantees subject-independent generalization estimates.

```
Outer loop: LeaveOneGroupOut
  └─ Per fold: feature harmonization on training set only
     Inner loop: GroupKFold (k ≤ n_train_subjects)
       Scoring: Pearson r (logged) + neg_MSE
       Refit criterion: neg_MSE (regression) | average_precision (classification)
```

When fewer than 2 training subjects are available in a fold, the model is fitted with the
default hyperparameters (no grid search). This condition is recorded in the reproducibility
report.

### 7.2 Within-Subject CV (Block-Aware)

Block-aware CV within each subject. Run/block labels serve as groups to prevent
temporal leakage across sessions.

```
For each subject:
  Outer loop: GroupKFold on block labels (k = min(n_blocks, outer_cv_splits))
    Inner loop: GroupKFold on remaining blocks (refit: neg_MSE / roc_auc)
```

**Ordered block mode** (`cv.within_subject_ordered_blocks = true`): Folds respect
temporal ordering — all preceding blocks form the training set, the next block is the
test set. Use when temporal autocorrelation is present.

Block labels are read from `block`, `run_id`, `run`, `session`, or `run_num` columns
of `events.tsv`. Subjects with missing or constant block labels are excluded and
logged.

### 7.3 CV Hygiene

**Function:** `apply_fold_specific_hygiene`

Prevents data leakage from test trials into fold-specific computations:

| Component | Leakage risk | Safeguard |
|-----------|-------------|-----------|
| IAF estimation | Test-trial spectra influence band definitions | IAF computed from `train_mask` only |
| Global features (ITPC, etc.) | Cross-trial aggregation includes test data | `train_mask` passed to feature extractor |
| All preprocessing statistics | Imputation, variance, scaling, PCA | Fit on training fold exclusively |

Controlled by `cv.hygiene_enabled` (default `true`).

---

## 8. Evaluation Metrics

### 8.1 Regression

**Primary metric — Subject-level Fisher-$z$-aggregated Pearson correlation**

For each held-out subject $i$ with $n_i$ test trials:

```math
r_i =
\frac{\sum_t (\hat{y}_{it} - \bar{\hat{y}}_i)(y_{it} - \bar{y}_i)}
     {\sqrt{\sum_t (\hat{y}_{it} - \bar{\hat{y}}_i)^2 \cdot \sum_t (y_{it} - \bar{y}_i)^2}}.
```

Aggregation across $S$ subjects:

1. Clip: $r_i^* = \mathrm{clip}(r_i,\, r_\text{min},\, r_\text{max})$.
2. Transform: $z_i = \mathrm{arctanh}(r_i^*)$.
3. Weighted average: $\bar{z} = \sum_i w_i z_i \,/\, \sum_i w_i$.
4. Back-transform: $\bar{r} = \tanh(\bar{z})$.

| Weighting mode | $w_i$ | Notes |
|----------------|--------|-------|
| `equal` *(default)* | $1$ | Treats each subject independently; robust to variable trial counts. |
| `trial_count` | $\max(n_i - 3,\, 1)$ | Fisher information weighting; standard meta-analytic fixed-effects scheme. |

**Secondary metric — Subject-level mean $R^2$:**

```math
\overline{R^2} = \frac{1}{S}\sum_{i=1}^{S} R^2_i, \qquad
R^2_i = 1 - \frac{\sum_t (y_{it} - \hat{y}_{it})^2}{\sum_t (y_{it} - \bar{y}_i)^2}.
```

Both $\bar{r}$ and $\overline{R^2}$ receive permutation p-values when `n_perm > 0`.

**Confidence intervals for $\bar{r}$:**

| Method | Computation |
|--------|-------------|
| `bootstrap` *(default)* | Resample subjects $B$ times; recompute weighted $\bar{z}$; $\mathrm{CI} = [\mathrm{perc}_{2.5},\, \mathrm{perc}_{97.5}]$ of $\tanh(\bar{z}^{(b)})$. |
| `fixed_effects` | $\mathrm{SE} = 1/\sqrt{\sum_i w_i}$; $\mathrm{CI} = \tanh(\bar{z} \pm 1.96\,\mathrm{SE})$. |

**Subject-level error metrics:**

```math
\overline{\text{MAE}} = \frac{1}{S}\sum_i \mathrm{MAE}_i, \qquad
\overline{\text{RMSE}} = \frac{1}{S}\sum_i \mathrm{RMSE}_i.
```

Pooled trial-level Pearson $r$, MAE, and RMSE are logged as secondary diagnostics only;
the subject is the inferential unit for LOSO.

### 8.2 Classification

**Primary metric:** Subject-level mean AUC — unweighted mean of per-subject ROC-AUC.
Valid only when at least `min_subjects_with_auc_for_inference` (default 2) subjects
have both classes present in their test fold.

Bootstrap CIs (subject-level, $B = 1000$) are computed for all metrics below.

| Metric | Formula |
|--------|---------|
| Accuracy | $(\text{TP}+\text{TN})/N$ |
| Balanced accuracy | $(\text{sensitivity} + \text{specificity})/2$ |
| AUC (ROC) | Area under the receiver operating characteristic curve |
| Average precision | Area under the precision–recall curve |
| F1 | $2\,\text{precision}\cdot\text{recall}\,/\,(\text{precision}+\text{recall})$ |
| Sensitivity | $\text{TP}\,/\,(\text{TP}+\text{FN})$ |
| Specificity | $\text{TN}\,/\,(\text{TN}+\text{FP})$ |
| Brier score | $\frac{1}{N}\sum_i (\hat{p}_i - y_i)^2$ |
| ECE | $\sum_{k=1}^{10}\frac{|B_k|}{N}\bigl|\bar{p}_k - \bar{y}_k\bigr|$ — Expected Calibration Error over 10 uniform bins |

---

## 9. Permutation Testing

**Functions:** `cv.py → run_permutation_test`; `orchestration.py → _run_classification_permutations`

Constructs an empirical null distribution by re-running the full nested CV pipeline
with permuted labels. All hyperparameter tuning is repeated on each permuted dataset.

### 9.1 Permutation Schemes

| Scheme | Behavior |
|--------|----------|
| `within_subject_within_block` *(default)* | Permute labels within each subject × block. Preserves both subject-level and block-level dependence structure. |
| `within_subject` | Permute labels within each subject. Used when block labels are unavailable or block-level permutation is degenerate (changes fewer than `min_label_shuffle_fraction` = 1% of labels). |

A permutation that changes fewer than `min_label_shuffle_fraction` labels is discarded;
if all attempts fail for a given permutation index, a `RuntimeError` is raised.

### 9.2 P-value Computation

**Regression** (two-tailed):

```math
p = \frac{\#\bigl\{|s(y^\pi_j)| \ge |s(y)|\bigr\} + 1}{n_\text{perm,valid} + 1},
```

where $s(\cdot)$ denotes the Fisher-$z$-aggregated $\bar{r}$. Computed for both
$\bar{r}$ and $\overline{R^2}$.

**Classification** (one-tailed; AUC $\ge$ observed):

```math
p = \frac{\#\bigl\{\mathrm{AUC}(y^\pi_j) \ge \mathrm{AUC}(y)\bigr\} + 1}{n_\text{perm,valid} + 1}.
```

### 9.3 Quality Gate

At least `min_valid_permutation_fraction` (default 50%) of permutations must produce
finite metrics. A `RuntimeError` is raised if this threshold is not met.

For multiple comparisons across frequency bands or feature groups, at least 5 000–10 000
permutations are recommended to ensure reliable tail estimates.

---

## 10. Time Generalization Analysis

**Module:** `time_generalization.py → time_generalization_regression`

Trains a decoding model in each time window and evaluates it in all other windows.
The resulting $W \times W$ matrix reveals whether the neural representation is
temporally stable (broad off-diagonal generalization) or dynamic (diagonal-only).

### 10.1 Method

**Step 1 — Epoch loading:** Load raw epoch tensors across subjects; align to the common
channel set within each LOSO fold.

**Step 2 — Sliding windows:** Partition the active period $[t_\text{min}, t_\text{max}]$
into overlapping windows of length $\Delta t$ with step $\delta t$.
Window centers form both axes of the generalization matrix.

**Step 3 — Window features:** For window $w$, compute the mean channel activity per trial:

```math
x_{w,\text{trial},c} = \frac{1}{|W_w|}\sum_{t \in W_w} \text{data}[\text{trial},\, c,\, t],
```

yielding an $N_\text{trials} \times C$ feature matrix per window.

**Step 4 — LOSO outer loop:** For each training window $i$, fit a Ridge model:

```
SimpleImputer → StandardScaler → TransformedTargetRegressor(Ridge(α), PowerTransformer)
```

Optionally tune $\alpha$ via inner `GroupKFold` (`use_ridgecv = true`).
For each test window $j$, predict on the held-out subject and record $r$ and $R^2$.

**Step 5 — Aggregation:** For cell $(i,j)$, aggregate over $S_{ij}$ valid subjects
using Fisher-$z$ averaging (equal weights):

```math
r_{ij}^\text{agg} =
\tanh\!\left(
  \frac{1}{S_{ij}} \sum_{s=1}^{S_{ij}}
  \mathrm{arctanh}\!\bigl(\mathrm{clip}(r_{ij}^{(s)})\bigr)
\right).
```

Cells with $S_{ij} < \texttt{min\_subjects\_per\_cell}$ or total trial count
$< \texttt{min\_count\_per\_cell}$ are excluded (set to `NaN`).

### 10.2 Significance Testing

Three corrections are applied simultaneously to all tested cells:

| Method | Controls | Description |
|--------|---------|-------------|
| FDR-BH | FDR | Benjamini–Hochberg across all finite cells. |
| Max-stat | FWER | Threshold at the 95th percentile of $\max_{(i,j)} |r_{ij}^\pi|$ under the null. |
| Cluster-FWER | FWER | Contiguous cells exceeding a forming threshold are compared to the null distribution of maximum cluster sizes. |

**Cell-level p-value:**

```math
p_{ij} =
\frac{\#\bigl\{|r_{ij}^\pi| \ge |r_{ij}^\text{obs}|\bigr\} + 1}{n_\text{perm,valid} + 1}.
```

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

## 11. SHAP Feature Importance

**Module:** `shap_importance.py`

SHAP values decompose each prediction into per-feature additive contributions satisfying
efficiency, symmetry, and the dummy axioms. Per-feature importance is the mean absolute
SHAP value:

```math
\phi_k = \frac{1}{N}\sum_{i=1}^{N} |\phi_k(x_i)|.
```

### 11.1 Explainer Selection

Determined automatically from the fitted estimator type after preprocessing:

| Model type | Explainer | Detection criterion |
|------------|-----------|---------------------|
| Tree-based (RF) | `shap.TreeExplainer` | `hasattr(estimator, "feature_importances_")` |
| Linear (ElasticNet, Ridge, LR) | `shap.LinearExplainer` | `hasattr(estimator, "coef_")` |
| Other (SVM, Ensemble) | `shap.KernelExplainer` | All other cases; 100 background samples |

### 11.2 Pipeline Awareness

Preprocessing is applied to transform $X$ before SHAP computation.
Feature names are propagated through selectors (`get_support`) and transformers
(`get_feature_names_out`). After PCA, components are relabeled `PC1`, `PC2`, ….

### 11.3 Cross-Fold Aggregation

For each LOSO fold: fit on training data (with inner CV tuning), compute SHAP values on
the test fold. Aggregate by feature name across folds:

```math
\bar\phi_k = \frac{1}{K}\sum_{f=1}^{K} \phi_k^{(f)}, \qquad
\sigma_k = \mathrm{std}_f\!\bigl(\phi_k^{(f)}\bigr).
```

A `RuntimeError` is raised if fewer than `analysis.shap.min_valid_fold_fraction`
(default 0.8) of folds succeed.

### 11.4 Grouped Summaries

When `interpretability.grouped_outputs = true`, feature names are parsed via
`NamingSchema` into structured metadata (group, frequency band, ROI, scope, statistic).
Importance is aggregated at two levels:

```math
\phi_\mathcal{G} = \sum_{k \in \mathcal{G}} \phi_k, \qquad
\mathrm{share}_\mathcal{G} = \frac{\phi_\mathcal{G}}{\sum_k \phi_k}.
```

Outputs: `shap_importance_by_group_band.tsv`, `shap_importance_by_group_band_roi.tsv`.

---

## 12. Permutation Feature Importance

**Function:** `orchestration.py → _run_permutation_importance_stage`

Model-agnostic importance via column permutation (Breiman 2001). For each feature $k$:

```math
\mathrm{imp}_k =
\mathrm{score}(y, \hat{y}) -
\mathbb{E}_\pi\!\left[\mathrm{score}(y, \hat{y}^{(\pi_k)})\right],
```

where $\hat{y}^{(\pi_k)}$ is the prediction after randomly permuting column $k$ in the
test set. Scoring metric: $R^2$; `n_repeats` permutations per feature (default 5).

Features excluded by fold harmonization receive `NaN`. Mean and standard deviation are
reported across folds. Same grouped summaries as SHAP (§11.4).

---

## 13. Uncertainty Quantification

**Module:** `uncertainty.py`

Distribution-free prediction intervals via conformal prediction. For exchangeable data
the coverage guarantee is exact:

```math
\mathbb{P}\!\left(y \in \hat{C}(x)\right) \ge 1 - \alpha.
```

The stage runs within the LOSO loop: the model is tuned via inner CV, then conformal
intervals are constructed on the held-out subject. Empirical coverage and mean interval
width are reported per subject.

### 13.1 Split Conformal

1. Split the training set into a proper training set ($\approx 80\%$) and a calibration
   set ($\approx 20\%$, minimum 2 samples), group-aware via `GroupShuffleSplit`.
2. Fit model; compute calibration residuals $s_i = |y_i - \hat{y}_i|$.
3. Conformal quantile:

```math
\hat{q} =
\mathrm{Quantile}\!\left(
  \{s_i\},\;
  \frac{\lceil(|\mathcal{D}_\text{cal}|+1)(1-\alpha)\rceil}{|\mathcal{D}_\text{cal}|}
\right).
```

4. Prediction interval: $\hat{C}(x) = [\hat{y}(x) - \hat{q},\; \hat{y}(x) + \hat{q}]$.

### 13.2 CV+ (Jackknife+) — Default

More data-efficient than split conformal; uses cross-validation residuals.

1. $k$-fold CV on training data (group-aware). For each fold $f$ and test point $x$:
   train on folds $\ne f$; compute residuals $\{s_i^{(f)}\}_{i \in \text{fold}_f}$;
   predict $\hat{y}_f(x)$.
2. Prediction interval using conservative order-statistic quantiles over the full
   ensemble $\{(\hat{y}_f(x),\, s^{(f)})\}$:

```math
\hat{C}(x) = \left[
  \hat{q}_\alpha\!\left(\{\hat{y}_f(x) - s_j^{(f)}\}_{f,j}\right),\;
  \hat{q}_{1-\alpha}\!\left(\{\hat{y}_f(x) + s_j^{(f)}\}_{f,j}\right)
\right].
```

A full model trained on all training data provides the point prediction.

### 13.3 Conformalized Quantile Regression (CQR)

Adaptive intervals for heteroscedastic data.

1. CV to collect out-of-fold quantile predictions at levels $\alpha/2$ and $1-\alpha/2$
   via `GradientBoostingRegressor(loss="quantile")`.
2. Conformity scores: $E_i = \max(\hat{q}_{\alpha/2}(x_i) - y_i,\; y_i - \hat{q}_{1-\alpha/2}(x_i))$.
3. Conformal quantile $\hat{Q}$ from $\{E_i\}$.
4. Prediction interval: $\hat{C}(x) = [\hat{q}_{\alpha/2}(x) - \hat{Q},\; \hat{q}_{1-\alpha/2}(x) + \hat{Q}]$.

---

## 14. Pipeline Runners

**Module:** `orchestration.py`

All runners accept feature subsetting arguments (`feature_families`, `feature_bands`,
`feature_segments`, `feature_scopes`, `feature_stats`) and write structured output
directories (see §15).

### 14.1 `run_regression_ml` — LOSO Regression

1. Assemble design matrix via `load_active_matrix`.
2. Raise a `ValueError` (or warning if `strict_regression_target_continuous = false`)
   when the target appears binary-like.
3. Build model pipeline and hyperparameter grid (ElasticNet, Ridge, or RF).
4. Run `nested_loso_predictions_matrix`: outer LOSO + inner `GroupKFold`; refit criterion `neg_MSE`.
5. Compute subject-level $\bar{r}$ (primary) and $\overline{R^2}$ (secondary), both with bootstrap CI.
6. Optionally run permutation test → two-tailed p-values for $\bar{r}$ and $\overline{R^2}$.
7. Compute and export LOSO mean-predictor baseline.
8. Run SHAP and permutation importance stages when enabled.
9. Generate diagnostic plots.

### 14.2 `run_within_subject_regression_ml` — Within-Subject Regression

Block-aware regression within each subject. Requires block/run labels in `events.tsv`.

1. Load and validate design matrix; exclude trials with missing block labels.
2. Create within-subject folds via `create_within_subject_folds` (GroupKFold or temporal ordering).
3. Fit with block-aware inner CV (refit: `neg_MSE`) per fold.
4. Aggregate; compute subject-level $\bar{r}$ with CI.
5. Block-aware permutation test.
6. Baseline: per-fold training-set mean prediction.

### 14.3 `run_classification_ml` — LOSO Classification

1. Load feature matrix via `load_active_matrix` (standard models) or epoch tensor via
   `load_epoch_tensor_matrix` (CNN).
2. Run `nested_loso_classification` (SVM/LR/RF/Ensemble) or `nested_loso_cnn_classification` (CNN).
   Inner loop: `StratifiedGroupKFold` (default scoring: `average_precision`).
3. Compute per-subject metrics (AUC, balanced accuracy, F1, etc.) with bootstrap CIs.
4. Compute calibration metrics (Brier score, ECE, calibration curve).
5. Raise a `RuntimeError` if fold failure fraction exceeds `max_failed_fold_fraction` (default 0.25).
6. Optionally run permutation test → one-tailed AUC p-value.

### 14.4 `run_within_subject_classification_ml` — Within-Subject Classification

Same structure as LOSO classification (§14.3) using `GroupKFold` on block labels within
each subject. Inner loop: `StratifiedGroupKFold` with `scoring="roc_auc"`.
Same block-aware permutation test.

### 14.5 `run_model_comparison_ml` — Model Comparison

Compares ElasticNet, Ridge, and Random Forest on identical outer LOSO folds.

1. Create shared LOSO outer folds.
2. For each model, run nested CV with inner `GroupKFold` (refit: Pearson $r$).
   Record per-fold $R^2$ and MAE.
3. Compute mean ± std and bootstrap CI per model.
4. **Pairwise inference** — paired sign-flip permutation test on per-fold $\Delta R^2$
   and $\Delta\text{MAE}$:

```math
p\bigl(\Delta R^2_{AB}\bigr) =
\frac{
  \#\bigl\{\bigl|\overline{s \cdot \Delta R^2_{AB}}\bigr|
  \ge \bigl|\overline{\Delta R^2_{AB}}\bigr|\bigr\} + 1
}{n_\text{perm} + 1},
\quad s_j \in \{-1,+1\}.
```

5. Holm–Bonferroni correction across all pairwise $\Delta R^2$ and $\Delta\text{MAE}$ tests.
6. Export `model_comparison.tsv` (per fold) and `model_comparison_summary.json`.

### 14.6 `run_incremental_validity_ml` — Incremental Validity

Quantifies the out-of-fold $\Delta R^2$ when adding EEG features over a baseline
predictor (e.g. stimulus temperature). Uses identical model family (ElasticNet with
covariate protection) for both conditions.

1. Define baseline predictor matrix $X_\text{base}$ from `events.tsv` (default: `temperature`).
   Guard against target-variable leakage raises a `ValueError` if the baseline column matches the target.
2. Full matrix: $X_\text{full} = [X_\text{EEG} \mid X_\text{base}]$ — baseline appended
   as protected covariates.
3. Per LOSO fold:
   - Fit tuned ElasticNet on $X_\text{base,train}$ → $\hat{y}_\text{base}$.
   - Fit tuned ElasticNet on $X_\text{full,train}$ → $\hat{y}_\text{full}$.
   - $\Delta R^2_\text{fold} = R^2(y_\text{test}, \hat{y}_\text{full}) - R^2(y_\text{test}, \hat{y}_\text{base})$.
4. Primary estimate:

```math
\overline{\Delta R^2} = \frac{1}{K}\sum_{k=1}^{K} \Delta R^2_k.
```

5. Bootstrap CI on $\overline{\Delta R^2}$ (resample folds).
   Paired sign-flip permutation p-value when `n_perm > 0`.

### 14.7 `run_time_generalization` — Time Generalization

Delegates to `time_generalization_regression` (§10). Saves outputs to
`results_root/time_generalization/`.

---

## 15. Output Structure

```
results_root/{mode}/
├── data/
│   ├── loso_predictions.tsv          ← y_true, y_pred, subject_id, fold_id
│   ├── loso_predictions.parquet
│   ├── loso_indices.tsv
│   └── baseline_predictions.tsv
├── metrics/
│   ├── pooled_metrics.json           ← subject-level (primary) + pooled-trial (secondary)
│   ├── per_subject_correlations.tsv
│   ├── per_subject_errors.tsv
│   └── per_subject_metrics.tsv       (classification)
├── models/
│   └── best_params_{model}.jsonl
├── null/
│   └── loso_null_{model}.npz         ← null_r, null_r2, n_completed
├── importance/
│   ├── shap_importance.tsv
│   ├── shap_importance_by_group_band.tsv
│   ├── shap_importance_by_group_band_roi.tsv
│   ├── permutation_importance.tsv
│   └── permutation_importance_by_group_band.tsv
├── plots/
│   └── *.{png,pdf,svg}
└── reports/
    ├── reproducibility_info.json
    ├── included_subjects.tsv
    └── excluded_subjects.tsv
```

---

## 16. Reproducibility

**Function:** `write_reproducibility_info`

Each run writes `reports/reproducibility_info.json` containing:

- `sklearn`, `numpy`, `pandas` library versions.
- RNG seed used throughout.
- Subject list and SHA-256 data signature (hash of sorted subject IDs).
- Full `machine_learning` configuration snapshot.

---

## 17. Configuration Reference

All settings reside under `machine_learning` in `eeg_config.yaml`.

### 17.1 Preprocessing

| Key | Description | Default |
|-----|-------------|---------|
| `constants.variance_threshold` | Base variance filter threshold $\theta$ | `0.0` |
| `preprocessing.imputer_strategy` | Imputation statistic | `"median"` |
| `preprocessing.power_transformer_method` | Target power transform | `"yeo-johnson"` |
| `preprocessing.power_transformer_standardize` | Standardize after transform | `true` |
| `preprocessing.variance_threshold_grid` | Grid search values for $\theta$ | `[0.0, 0.01, 0.1]` |
| `preprocessing.feature_selection_percentile` | Univariate top-$k\%$ selection (100 = disabled) | `100.0` |
| `preprocessing.deconfound` | Regress covariates out of EEG features | `false` |
| `preprocessing.spatial_regions_allowed` | ROI allowlist for `SpatialFeatureSelector` | `[]` |
| `preprocessing.pca.enabled` | Enable PCA | `false` |
| `preprocessing.pca.n_components` | Variance fraction or integer count | `0.95` |
| `preprocessing.pca.whiten` | Whiten principal components | `false` |
| `preprocessing.pca.svd_solver` | SVD solver | `"auto"` |

### 17.2 Regression Models

| Key | Description | Default |
|-----|-------------|---------|
| `models.elasticnet.alpha_grid` | $\alpha$ search | `[0.01, 0.1, 1.0, 10.0]` |
| `models.elasticnet.l1_ratio_grid` | $\rho$ search | `[0.1, 0.5, 0.9]` |
| `models.elasticnet.max_iter` | Solver max iterations | `10000` |
| `models.elasticnet.tol` | Convergence tolerance | `1e-4` |
| `models.ridge.alpha_grid` | $\alpha$ search | `[0.01, 0.1, 1.0, 10.0, 100.0]` |
| `models.random_forest.n_estimators` | Number of trees $B$ | `100` |
| `models.random_forest.bootstrap` | Bootstrap resampling | `true` |
| `models.random_forest.max_depth_grid` | Depth search | `[5, 10, 20, null]` |
| `models.random_forest.min_samples_split_grid` | Split threshold search | `[2, 5, 10]` |
| `models.random_forest.min_samples_leaf_grid` | Leaf size search | `[1, 2, 4]` |

### 17.3 Classification Models

| Key | Description | Default |
|-----|-------------|---------|
| `models.svm.kernel` | SVM kernel | `"rbf"` |
| `models.svm.C_grid` | $C$ search | `[0.1, 1.0, 10.0]` |
| `models.svm.gamma_grid` | $\gamma$ search | `["scale", "auto"]` |
| `models.svm.class_weight` | Class weighting | `"balanced"` |
| `models.logistic_regression.penalty` | Regularization type | `"l2"` |
| `models.logistic_regression.C_grid` | $C$ search | `[0.01, 0.1, 1.0, 10.0]` |
| `models.logistic_regression.l1_ratio_grid` | ElasticNet mixing search | `[0.1, 0.5, 0.9]` |
| `models.logistic_regression.class_weight` | Class weighting | `"balanced"` |
| `classification.resampler` | Imbalance strategy | `"none"` |
| `classification.calibrate_ensemble` | Platt-scale SVM/RF in ensemble | `false` |
| `classification.scoring` | Inner CV scoring metric | `"average_precision"` |
| `classification.model` | Default classifier | `"svm"` |
| `classification.max_failed_fold_fraction` | Maximum allowed fold failure rate | `0.25` |
| `classification.min_subjects_with_auc_for_inference` | Minimum subjects for AUC validity | `2` |

### 17.4 CNN

| Key | Description | Default |
|-----|-------------|---------|
| `models.cnn.temporal_filters` | $F_1$ | `8` |
| `models.cnn.depth_multiplier` | $D$ | `2` |
| `models.cnn.pointwise_filters` | $F_2$ | `16` |
| `models.cnn.kernel_length` | $K_1$ | `64` |
| `models.cnn.separable_kernel_length` | $K_2$ | `16` |
| `models.cnn.dropout` | Dropout rate $p$ | `0.5` |
| `models.cnn.batch_size` | Mini-batch size | `64` |
| `models.cnn.max_epochs` | Maximum training epochs | `75` |
| `models.cnn.patience` | Early stopping patience (epochs) | `10` |
| `models.cnn.learning_rate` | AdamW lr | `1e-3` |
| `models.cnn.weight_decay` | AdamW weight decay | `1e-3` |
| `models.cnn.gradient_clip_norm` | Max gradient norm | `1.0` |
| `models.cnn.val_fraction` | Validation fraction | `0.2` |
| `models.cnn.use_cuda` | Enable GPU | `false` |

### 17.5 Cross-Validation

| Key | Description | Default |
|-----|-------------|---------|
| `cv.default_n_splits` | Default inner CV splits | `5` |
| `cv.inner_splits` | Inner splits for importance stages | `5` |
| `cv.outer_splits` | Outer splits for within-subject mode | `5` |
| `cv.hygiene_enabled` | Enable fold-specific CV hygiene | `true` |
| `cv.within_subject_ordered_blocks` | Temporal block ordering | `false` |
| `cv.permutation_scheme` | Label permutation structure | `"within_subject_within_block"` |
| `cv.min_label_shuffle_fraction` | Min fraction of labels changed per permutation | `0.01` |
| `cv.min_valid_permutation_fraction` | Min fraction of permutations with finite metrics | `0.5` |
| `cv.min_valid_permutation_fold_fraction` | Min fold completion rate per permutation | `1.0` |

### 17.6 Evaluation

| Key | Description | Default |
|-----|-------------|---------|
| `evaluation.ci_method` | CI method for $\bar{r}$ | `"bootstrap"` |
| `evaluation.bootstrap_iterations` | Bootstrap resamples | `1000` |
| `evaluation.subject_weighting` | Fisher-$z$ weighting scheme | `"equal"` |

### 17.7 Targets

| Key | Description | Default |
|-----|-------------|---------|
| `targets.regression` | Continuous target column in `events.tsv` | *(required)* |
| `targets.classification` | Binary target column | `"binary_outcome"` |
| `targets.binary_threshold` | Threshold to binarize a continuous target | `null` |
| `targets.strict_regression_target_continuous` | Raise error for binary-like regression targets | `false` |

### 17.8 Time Generalization

| Key | Description | Default |
|-----|-------------|---------|
| `analysis.time_generalization.active_window` | Time range $[t_\text{min}, t_\text{max}]$ (s) | `[3.0, 10.5]` |
| `analysis.time_generalization.window_len` | Window duration (s) | `0.75` |
| `analysis.time_generalization.step` | Window step (s) | `0.25` |
| `analysis.time_generalization.default_alpha` | Ridge $\alpha$ when not tuning | `1.0` |
| `analysis.time_generalization.use_ridgecv` | Tune $\alpha$ via inner GroupKFold | `false` |
| `analysis.time_generalization.alpha_grid` | $\alpha$ grid for inner CV | `[0.01, 0.1, 1.0, 10.0, 100.0]` |
| `analysis.time_generalization.cluster_threshold` | Cluster-forming threshold | `0.05` |
| `analysis.time_generalization.min_subjects_per_cell` | Min subjects per cell | `2` |
| `analysis.time_generalization.min_count_per_cell` | Min trials per cell | `15` |
| `analysis.time_generalization.min_valid_fold_fraction` | Min fold completion rate | `0.8` |
| `analysis.time_generalization.min_valid_permutation_fraction` | Min permutation completion rate | `0.8` |

### 17.9 Feature Importance

| Key | Description | Default |
|-----|-------------|---------|
| `analysis.shap.enabled` | Run SHAP importance | `true` |
| `analysis.shap.min_valid_fold_fraction` | Min fold completion for SHAP | `0.8` |
| `analysis.permutation_importance.enabled` | Run permutation importance | `false` |
| `analysis.permutation_importance.n_repeats` | Shuffles per feature per fold | `5` |
| `analysis.permutation_importance.min_valid_fold_fraction` | Min fold completion | `0.8` |
| `interpretability.grouped_outputs` | Export importance grouped by band/ROI | `true` |

### 17.10 Incremental Validity

| Key | Description | Default |
|-----|-------------|---------|
| `incremental_validity.baseline_predictors` | Baseline predictor columns from `events.tsv` | `["temperature"]` |
| `incremental_validity.require_baseline_predictors` | Raise if baseline columns are missing | `true` |

### 17.11 Uncertainty Quantification

| Key | Description | Default |
|-----|-------------|---------|
| `analysis.uncertainty.min_valid_fold_fraction` | Min fold completion for conformal intervals | `0.8` |

### 17.12 Data

| Key | Description | Default |
|-----|-------------|---------|
| `data.feature_harmonization` | Harmonization mode | `"intersection"` |
| `data.max_excluded_subject_fraction` | Max fraction of requested subjects that may be excluded | `1.0` |

### 17.13 Plotting

| Key | Description | Default |
|-----|-------------|---------|
| `plotting.enabled` | Auto-generate diagnostic figures | `true` |
| `plotting.formats` | Output formats | `["png"]` |
| `plotting.dpi` | Figure resolution | `300` |
| `plotting.top_n_features` | Top N features in importance figures | `20` |
| `plotting.include_diagnostics` | Include prediction scatter / calibration plots | `true` |

---

## 18. Dependencies

| Library | Role |
|---------|------|
| **scikit-learn** | Pipelines, CV, `GridSearchCV`, metrics, imputation, scaling, PCA, Ridge, ElasticNet, RF, SVM, LR |
| **imbalanced-learn** | `ImbPipeline`, `SMOTE`, `RandomUnderSampler` |
| **NumPy / SciPy** | Numerical computation, Pearson correlation, Fisher $z$-transform |
| **pandas** | Feature table I/O, metadata handling |
| **statsmodels** | FDR (Benjamini–Hochberg), Holm correction |
| **joblib** | Parallel outer-fold execution |
| **SHAP** | Feature importance *(optional — `pip install shap`)* |
| **PyTorch** | EEGNet CNN classifier *(optional — `pip install torch`)* |
| **MNE-Python** | Epoch loading for time generalization |
| **matplotlib** | Diagnostic figures |
