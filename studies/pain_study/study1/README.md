# Study 1: Trial-Wise EEG Prediction of fMRI Pain-Signature Expression

This document serves as the exhaustive computational and methodological reference for **Study 1**. It is designed for direct integration into scientific manuscripts, strictly defining the research objectives, data contracts, mathematical formulations, and rigorous cross-validation paradigms.

---

## 1. Research Objectives and Scope

### 1.1 Primary Objective
**To determine whether continuous, trial-wise expression of established fMRI pain signatures can be directly predicted from concurrent band-limited EEG dynamics.**
The study targets the **Neurologic Pain Signature (NPS)** and the **Stimulus Intensity Independent Pain Signature-1 (SIIPS1)** using a subject-level, trial-resolved multimodal design.

### 1.2 Secondary Sensitivity Objective
**To test whether EEG-derived predictive accuracy remains robust after controlling for known behavioral and temporal confounds.**
The model's validity is evaluated against fMRI signatures that are strictly residualized against subjective pain ratings, stimulus temperatures, trial timing, and block effects using fold-contained nuisance regression.

### 1.3 Fixed Analytical Scope
The analysis is restricted to:
- **Conditions:** `trial_type == "stimulation"` and `stim_phase == "plateau"`
- **Contrasts:** `pain_vs_nonpain`
- **Spaces:** `MNI152NLin2009cAsym` fMRI inputs
- **Predictive Lanes:**
  1. Feature-Based Machine Learning (Standard Regressors)
  2. Deep Regression (Band-Temporal PyTorch Neural Network)

---

## 2. Preconditions and Data Contracts

### 2.1 EEG Epoch Contract
The pipeline relies on clean task epochs with the following shared baselines:
- `tmin = -7.0 s`, `tmax = 15.0 s`
- `baseline = [-0.2, 0.0] s`

Time-frequency analyses utilize independent, family-level baselining:
- `baseline_mode = "logratio"`
- `baseline_window = [-5.0, -0.01] s`
- `active_window = [3.0, 10.5] s`

### 2.2 Frequency Band Definitions
The exact numeric filter ranges applied across all analyses:
| Band | Range (Hz) |
| --- | --- |
| `alpha` | `[8.0, 12.9]` |
| `beta` | `[13.0, 30.0]` |
| `gamma` | `[30.1, 80.0]` |
| `alpha_beta_gamma` | (Concatenated combinations of the above) |

---

## 3. fMRI Signature Target Construction

### 3.1 Trial-Signature Extraction Parameters
Targets are extracted trial-by-trial using the following fixed configuration:
| Parameter | Value |
| --- | --- |
| Extraction Method | Least-Squares Separate (`lss`) |
| Metric | Dot Product (`dot`) |
| Normalization | None |
| HRF Model | `spm` |
| Drift Model | `cosine` |
| High-pass Filter | `0.008 Hz` |
| Smoothing FWHM | `null` |
| Alignment Rounding | `3` decimals |

### 3.2 Target Mathematical Formulation
Let $\beta_{s,i}(v)$ denote the LSS-derived BOLD effect estimate for subject $s$, trial $i$, and voxel $v$. Let $M_k(v)$ denote the signature-map weight for target $k \in \{\text{NPS}, \text{SIIPS1}\}$. The primary signature-expression metric is:

```math
y_{s,i}^{(k)} = \sum_{v \in V} \beta_{s,i}(v)M_k(v)
```

### 3.3 Trial Alignment
fMRI targets $y_{s,i}^{(k)}$ are back-projected to clean EEG trials using exact trial indices or rounded onset/duration keys. Duplicate keys are aggregated by arithmetic mean.

---

## 4. Feature-Based Machine Learning Lane

### 4.1 Feature Families
- **Confirmatory Family:** `power` (subjected to strict Holm correction).
- **Exploratory Families:** `spectral`, `aperiodic`, `erds`, `ratios`, `asymmetry`, `complexity`, `bursts`.

### 4.2 Strict Foldwise Preprocessing Sequence
To prevent cross-subject leakage and outlier injection during cross-validation, processing occurs inside each outer LOSO fold. Let $X \in \mathbb{R}^{N \times P}$ denote the pooled trial-by-feature matrix.

1. **Subject-Wise Standardization:** Resolves scale-variant baseline differences (e.g., skull thickness) prior to imputation.
   ```math
   X_{s} \leftarrow \frac{X_s - \mu_s}{\sigma_s}
   ```
2. **Missing Feature Imputation:** Missing trials are imputed via the cohort `median`.
3. **Global Scaling:** Features are standardized to unit variance across the entire training fold.
4. **Variance Thresholding:** To prevent absolute-scale frequency bias (e.g., heavily penalizing Gamma bands), `VarianceThreshold` is applied *after* scaling, strictly locked to `[0.0]` to remove only zero-variance features.

### 4.3 Models and Optimization Objectives
Targets are transformed using a Yeo-Johnson transform: $\tilde{y} = \mathrm{YJ}(y)$. 

**ElasticNet:**
```math
\min_{\beta_0,\beta} \frac{1}{2n} \left\| \tilde{y} - \beta_0 - X\beta \right\|_2^2 + \alpha \rho \|\beta\|_1 + \frac{\alpha(1-\rho)}{2}\|\beta\|_2^2
```
*Grid:* `alpha \in [0.001, 0.01, 0.1, 1, 10]`, `l1_ratio \in [0.2, 0.5, 0.8]`

**Ridge Regression:**
```math
\min_{\beta_0,\beta} \frac{1}{2n} \left\| \tilde{y} - \beta_0 - X\beta \right\|_2^2 + \frac{\alpha}{2}\|\beta\|_2^2
```
*Grid:* `alpha \in [0.01, 0.1, 1.0, 10.0, 100.0]`

**Random Forest:** Bootstrap forest utilizing $500$ estimators.
*Grid:* `max_depth \in [5, 10, 20, null]`, `min_samples_split \in [2, 5, 10]`, `min_samples_leaf \in [1, 2, 4]`

Hyperparameters are tuned via 5-fold inner group-k-fold cross-validation.

---

## 5. Deep Regression Lane

### 5.1 Band-Limited Analytic Envelopes
To capture oscillatory power free from edge artifacts and arbitrary phase dynamics, tensors are explicitly processed as follows:
1. Continuous epochs are band-pass filtered into target ranges.
2. The **analytic signal envelope** is extracted via the Hilbert transform.
3. Tensors are cropped to the active stimulation window (`[3.0, 10.5] s`).

Input tensor shape: $X \in \mathbb{R}^{n_{trials} \times n_{bands} \times n_{channels} \times n_{times}}$

### 5.2 Foldwise Tensor Normalization
Inputs are standardized via training cohort statistics:
```math
\tilde{X}_{n,b,c,t} = \frac{X_{n,b,c,t} - \mu_{b,c}}{\sigma_{b,c}^{*}}
```
where $\mu_{b,c} = \mathrm{mean}_{n,t}(X_{train})$ and $\sigma_{b,c}^{*} = \max(\mathrm{sd}_{n,t}(X_{train}), 10^{-6})$. Targets are similarly scaled: $\tilde{y}_n = \frac{y_n - \mu_y}{\sigma_y^{*}}$.

### 5.3 Network Architecture & Equations
The model extracts spatial topographies independently for each band prior to temporal integration.

1. **Depthwise Spatial Convolution** (Extracts spatial maps):
   ```math
   H^{(1)}_{n,b,f,t} = \mathrm{ELU}\left(\mathrm{BN}\left(\sum_{c} W^{spat}_{b,f,c}\tilde{X}_{n,b,c,t}\right)\right)
   ```
   *Parameters:* Kernel `(n_channels, 1)`, Groups `= n_bands`

2. **Temporal Convolution** (Integrates power over time):
   ```math
   H^{(2)}_{n,f,t} = \mathrm{ELU}\left(\mathrm{BN}\left(\sum_{b,u} W^{temp}_{f,b,u} H^{(1)}_{n,b,f,t+u}\right)\right)
   ```
   *Parameters:* Kernel `(1, 15)`, Temporal Filters `= 8`

3. **Pooling and Head**:
   ```math
   z_n = \mathrm{Pool}_{8}\left(\mathrm{Dropout}_{0.25}(H^{(2)}_n)\right)
   ```
   ```math
   \hat{y}_n = W_2 \, \mathrm{Dropout}_{0.25}\left(\mathrm{ELU}\left(W_1 \, \mathrm{vec}(z_n) + b_1\right)\right) + b_2
   ```

**Optimization Defaults:** `AdamW` (lr=`0.001`, wd=`0.0001`), `batch_size=32`, `epochs=25`, `patience=5`, `validation_fraction=0.2`.

---

## 6. Evaluation Metrics and Inference

### 6.1 Strict Zero-Skill $R^2$
To prevent mean-variance data leakage, the coefficient of determination calculates its zero-skill baseline using the *training fold's* target mean ($\bar{y}_{train,f}$):
```math
R_f^2 = 1 - \frac{\sum_{i \in f}(y_i - \hat{y}_i)^2}{\sum_{i \in f}(y_i - \bar{y}_{train,f})^2}
```
Mean Absolute Error is calculated simply as: $\mathrm{MAE}_f = \frac{1}{|f|}\sum_{i \in f}|y_i - \hat{y}_i|$.

### 6.2 Permutation Testing
Statistical inference utilizes **5,000 permutations**. Within the nested CV pipeline, targets $y$ are permuted *prior* to the outer cross-validation splits to ensure unbiased hyperparameter null-tuning. Permutations are performed **within-subject** to preserve block dependencies while dismantling trial-wise predictability. Confirmatory model p-values are corrected using the **Holm** step-down procedure.

---

## 7. Manuscript Reporting Checklist

When transitioning this computational reference into a manuscript, ensure the following dynamically generated values are sourced from the output artifacts:
1. **Target Yield:** Final trial counts surviving artifact rejection and target alignment (`primary_targets.parquet`).
2. **Subject Inclusion:** Total requested vs. final included subjects per analytical lane (`model_comparison_summary.json`).
3. **Cross-Validation Yield:** Total fold counts and exact hyperparameter parameters selected (`model_comparison.tsv`).
4. **Signature Details:** Exact version and mapping file provenance of the `NPS` and `SIIPS1` maps.
5. **Random State:** Explicitly declare that all initializations and CV folds utilized the deterministic `project.random_state = 42`.
