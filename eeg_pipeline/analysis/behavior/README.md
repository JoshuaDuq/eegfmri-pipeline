# Behavior Computation Pipeline

This document describes the behavioral statistics pipeline. Stages run as a dependency-resolved DAG over EEG/behavior data (pain ratings, temperature, experimental conditions), with explicit non-i.i.d. safeguards and per-stage outputs.

**Scope:** trial-level associations between behavioral variables (ratings, temperature, conditions, covariates) and EEG-derived features, plus group-level aggregation and validation.

The README is organized like a methods section:

- **1. Overview and notation**
- **2. Architecture (stage DAG)**
- **3. Statistical safeguards**
- **4. Stage definitions (trial-wise, temporal, advanced, diagnostics)**
- **5. Group-level utilities**
- **6. FDR reference**

## 1. Overview and Notation

We work at the **trial level**. Let:

- $i \in \{1,\dots, N_\text{trials}\}$ index trials within a subject,
- $g$ index grouping units (runs/blocks) used for non-i.i.d. safeguards,
- $s$ index subjects (for group-level aggregation),
- $f \in \{1,\dots, F\}$ index EEG-derived features.

For each trial $i$ we define:

- Behavioral **outcome** (e.g. pain rating) $y_i$,
- Thermal **predictor** (e.g. temperature) $T_i$,
- Feature vector $x_i \in \mathbb{R}^{F}$ with components $x_{i,f}$,
- Optional covariate vector $z_i$ (age, sex, questionnaire scores, etc.),
- Group label $G_i \in \{1,\dots, N_\text{groups}\}$ (run/block/cluster).

The **trial table** stores one row per trial $i$ with columns:

- trial metadata (subject, run/block, condition, indices),
- behavioral variables ($y_i$, $T_i$, covariates),
- EEG feature columns ($x_{i,f}$) joined from feature tables.

Trials are **not i.i.d.**: they are clustered within runs/blocks and subjects. All permutation-based inference respects this structure via grouped labels, and group-level analyses aggregate subject-level statistics rather than pooling trials.

## 2. Architecture

The pipeline runs as dependency-resolved stages:

```text
load
 → trial_table
 → lag_features
 → predictor_residual
 → temperature_models
 → feature_qc
 → correlate_design
 → correlate_effect_sizes
 → correlate_pvalues
 → correlate_primary_selection
 → correlate_fdr
 → predictor_sensitivity
 → regression
 → models
 → stability
 → icc
 → condition_column
 → condition_window
 → temporal_tfr
 → temporal_stats
 → cluster
 → mediation
 → moderation
 → mixed_effects (subject-level stage skips)
 → consistency
 → influence
 → hierarchical_fdr_summary
 → report
 → export
```

Group-level computations run outside the subject DAG via `BehaviorPipeline.run_group_level(...)`.

## Statistical Safeguards

**Non-i.i.d. trial data:**

- Correlations, predictor sensitivity, regression, condition column, mediation, and moderation enforce grouped permutation in trial mode unless `allow_iid_trials=true`.
- Models, condition window, and condition multigroup require an explicit i.i.d. override for trial-level inference.
- Temporal trial-level inference requires cluster correction with valid grouped labels when `allow_iid_trials=false`.

**Temperature control:**

- Partial/control and residualization paths are implemented. Temperature control is skipped when the target is temperature itself.

**Multiple-comparison correction:**

- Unified hierarchical/global FDR is applied in wrapped stages.
- Regression and model-family helpers compute local BH (`p_fdr`).
- Temporal correction mode is configurable: `cluster`, `fdr`, `bonferroni`, or `none`.

---

## Stage Categories

### 0. Load and Metadata

**Modules:** `stages/metadata.py`, `utils/data/behavior.py`

- **Load stage (`load`)**: reads aligned behavioral events, thermal predictor series, covariates, and trial-wise EEG feature tables into the `BehaviorContext`. It produces the core behavioral series (`rating`, `temperature`) and a mapping of named feature tables used downstream.
- **Metadata/QC stage (`metadata`)**: builds a JSON summary of behavioral data quality (trial counts, missingness, basic rating/temperature distributions, pain vs. non-pain contrasts, covariate coverage, feature counts) and records key analysis configuration fields (method labels, permutation counts, FDR level). Outputs live alongside stats files and support reproducibility/diagnostics.

---

### 1. Trial Table

**Module:** `stages/trial_table.py` → `utils/data/trial_table.py`

Builds the canonical trial-level table by joining aligned events and all feature tables.

---

### 2. Lag Features

**Module:** `stages/trial_table.py` → `utils/data/trial_table.py::add_lag_and_delta_features`

Computed within run/block grouping:

$$

\text{prev\_temperature}_i = T_{i-1}, \quad \Delta\text{temperature}_i = T_i - T_{i-1}

$$

$$

\text{prev\_rating}_i = y_{i-1}, \quad \Delta\text{rating}_i = y_i - y_{i-1}, \quad \text{trial\_index}_i = i

$$

---

### 3. Predictor Residual

**Module:** `stages/trial_table.py` → `utils/analysis/stats/predictor_residual.py`

Core quantity: $\text{residual}_i = y_i - \hat{y}_i$, where $\hat{y}_i = f(T_i)$.

Model path:

- Spline OLS candidates: `rating ~ bs(temp, df=d, degree=3)` (pick lowest AIC)
- Fallback polynomial: $\hat{y} = \sum_j a_j T^j$
- Optional cross-fit residual (`predictor_residual_cv`) via GroupKFold.

---

### 4. Temperature Models

**Module:** `stages/models.py` → `utils/analysis/stats/temperature_models.py`

Model comparison:

- Linear: $y = \beta_0 + \beta_1 T + \varepsilon$
- Polynomial: $y = \beta_0 + \beta_1 T + \cdots + \beta_d T^d + \varepsilon$
- Spline: $y = \beta_0 + \sum_j \beta_j B_j(T) + \varepsilon$

Metrics:

$$

\text{RMSE} = \sqrt{\operatorname{mean}((y - \hat{y})^2)}, \qquad \Delta\text{AIC} = \text{AIC}_\text{model} - \text{AIC}_\text{best}

$$

Breakpoint (hinge) model:

$$

y = \beta_0 + \beta_1 T + \beta_2 \max(0,\, T - c) + \varepsilon

$$

Linear-vs-hinge F-test:

$$

F = \frac{(RSS_\text{lin} - RSS_\text{hinge})/df_\text{num}}{RSS_\text{hinge}/df_\text{den}}

$$

---

### 5. Feature QC

**Module:** `stages/feature_qc.py`

Applies feature-quality screening and emits QC metadata.

---

### 6. Correlations Sub-Pipeline

**Modules:** `stages/correlate.py`, `stats/correlation.py`, `stats/partial.py`, `stats/permutation.py`

#### 6.1 Correlate Design

Defines targets, covariates, and permutation groups.

#### 6.2 Correlate Effect Sizes

Raw correlations:

- Pearson: $r = \operatorname{corr}(x, y)$
- Spearman: $r = \operatorname{corr}(\operatorname{rank}(x),\, \operatorname{rank}(y))$

Partial correlation (covariate control): residualize $x$ and $y$ on $Z$ and correlate residuals. Test statistic:

$$

t = r\sqrt{\frac{n-k-2}{1-r^2}}, \qquad p = 2\,P\!\left(|T_{n-k-2}| \geq |t|\right)

$$

Run-mean mode computes correlations on run-aggregated means.

#### 6.3 Correlate P-values

Permutation p-value:

$$

p_\text{perm} = \frac{\#\text{extreme} + 1}{n_\text{perm} + 1}

$$

Grouped schemes: shuffle or circular-shift (`permute_within_groups`).
Partial permutation uses Freedman-Lane residual permutation.

#### 6.4 Primary Selection

Selects primary effect/p-value according to control path, unit (trial vs run), and strict non-i.i.d mode.

#### 6.5 FDR

BH adjustment core:

$$

q_{(i)} = \min_{j \geq i} \frac{m}{j}\,p_{(j)}

$$

Wrapped stages use unified hierarchical/global FDR helpers.

---

### 7. Predictor Sensitivity

**Module:** `stages/correlate.py` → `stats/correlation.py::run_predictor_sensitivity_correlations`

Pain sensitivity index: fit $y = \beta_0 + \beta_1 T + \varepsilon$, compute residual $\psi = y - (\beta_0 + \beta_1 T)$, then correlate each feature with $\psi$.

Permutation p-values: $(\#\text{extreme} + 1)/(n_\text{perm} + 1)$.

---

### 8. Regression

**Module:** `stages/models.py` → `stats/trialwise_regression.py`

Per-feature model:

$$

y = Z\gamma + \beta_f x_f + \beta_\text{int}(x_f \cdot T) + \varepsilon \quad \text{(full)}, \qquad y = Z\gamma + \varepsilon \quad \text{(reduced)}

$$

Outputs include:

- $\Delta R^2 = R^2_\text{full} - R^2_\text{reduced}$
- HC3 heteroskedasticity-consistent standard errors:

$$

h_i = x_i^\top (X^\top X)^{-1} x_i, \qquad w_i = \frac{e_i^2}{(1-h_i)^2}

$$

$$

\widehat{\text{Cov}}_\text{HC3} = (X^\top X)^{-1} X^\top \operatorname{diag}(w)\, X\, (X^\top X)^{-1}

$$

- $t = \hat{\beta} / \text{SE}$, two-sided t p-values

Permutation feature-term inference uses reduced-model residual permutation (Freedman-Lane style).

---

### 9. Model Families

**Module:** `stages/models.py` → `stats/feature_models.py`

Families:

- `ols_hc3`
- `robust_rlm` (Huber M-estimation)
- `quantile_50` (median regression)
- `logit`

Logistic form:

$$

\operatorname{logit}(P(Y=1)) = X\beta, \qquad \text{OR} = \exp(\beta_\text{feature})

$$

Additional diagnostics include McFadden pseudo-$R^2$ and AUC/delta-AUC.

---

### 10. Stability

**Module:** `stages/diagnostics.py` → `stats/stability.py`

Per run/block group $g$: $r_g = \operatorname{corr}(x_g, y_g)$ (optionally partial on temperature).

Summary metrics: $\bar{r}_g$, $s_{r_g}$, $\min(r_g)$, $\max(r_g)$; sign consistency against overall effect; fraction with $p_g < \alpha$.

---

### 11. ICC Reliability

**Module:** `stages/diagnostics.py` → `stats/reliability.py`

Subject stage uses ICC(3,1):

$$

\text{ICC}(3,1) = \frac{MS_\text{rows} - MS_\text{error}}{MS_\text{rows} + (k-1)\,MS_\text{error}}

$$

---

### 12. Condition Comparisons

**Modules:** `stages/condition.py`, `stats/effect_size.py`, `utils/parallel.py`

#### 12.1 Condition Column (two-group)

- Unpaired default: Welch t-test

$$

t = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{s_1^2/n_1 + s_2^2/n_2}}

$$

- Paired path (run-level): paired tests/effects
- Effect sizes:

$$

d = \frac{\bar{x}_1 - \bar{x}_2}{s_\text{pooled}}, \qquad g = d\!\left(1 - \frac{3}{4\,df - 1}\right), \qquad d_z = \frac{\bar{d}}{s_d} \text{ (paired)}

$$

Permutation p-values (unpaired mean-difference and paired sign-flip): $(\#\text{extreme} + 1)/(n_\text{perm} + 1)$.

#### 12.2 Condition Multigroup

Pairwise tests only (no stage-level omnibus):

- unpaired: Mann-Whitney U
- paired run-level: Wilcoxon signed-rank

#### 12.3 Condition Window

Paired window comparison uses Wilcoxon and difference-score effects:

$$

d_z = \bar{d}/s_d, \qquad g_z = d_z\!\left(1 - \frac{3}{4n-1}\right), \quad \text{where } d_i = v_{2i} - v_{1i}

$$

---

### 13. Temporal Statistics

**Modules:** `stages/temporal.py`, `stats/temporal.py`

Power-bin statistic:

$$

b_{i,f,w} = \operatorname{mean}_{t \in w} P_{i,f,t}

$$

Correlate $b$ with behavior (simple or partial). Correlation-to-$t$ transform for cluster forming:

$$

t = r\sqrt{\frac{\text{dof}}{1-r^2}}

$$

Temporal correction modes: `fdr` (BH), `bonferroni`, `cluster`, `none`.

ITPC trial metric:

$$

\text{ITPC}_\text{trial} = \operatorname{mean}\!\left(\cos(\phi_\text{trial} - \bar{\phi})\right)

$$

ERDS trial metrics:

$$

\text{ERDS\%} = 100 \cdot \frac{P_\text{active} - P_\text{base}}{P_\text{base}}, \qquad \text{ERDS}_z = \frac{P_\text{active} - P_\text{base}}{\sigma_\text{base}}

$$

---

### 14. Cluster Tests

**Modules:** `stages/temporal.py::stage_cluster_impl`, `stats/cluster.py`

Key formulas:

$$

d = \frac{\mu_A - \mu_B}{s_\text{pooled}}, \qquad M_c = \sum_{i \in c} |t_i|, \qquad p_c = \frac{\#\{M_\text{max}^\text{perm} \geq M_c\} + 1}{n_\text{perm} + 1}

$$

---

### 15. Mediation

**Module:** `stages/advanced.py` → `stats/mediation.py`

Path models:

$$

M = \alpha_0 + aX + \varepsilon_M, \qquad Y = c_0 + cX + \varepsilon_Y, \qquad Y = c_0' + c'X + bM + \varepsilon_Y'

$$

Derived quantities:

$$

\text{indirect} = ab, \qquad \text{SE}_{ab} = \sqrt{a^2\,\text{SE}_b^2 + b^2\,\text{SE}_a^2}, \qquad z = ab/\text{SE}_{ab}, \qquad \text{prop.\ mediated} = ab/c

$$

Bootstrap CI is percentile-based. Optional permutation p-values: $(\#\text{extreme}+1)/(n_\text{perm}+1)$.

---

### 16. Moderation

**Module:** `stages/advanced.py` → `stats/moderation.py`

Model:

$$

Y = \beta_0 + \beta_1 X + \beta_2 W + \beta_3 (X \cdot W) + \varepsilon

$$

Key computations:

$$

\Delta R^2 = R^2_\text{full} - R^2_\text{reduced}, \qquad F = \frac{(R^2_\text{full} - R^2_\text{reduced})/1}{(1 - R^2_\text{full})/(n-4)}

$$

Simple slope at moderator level $W$:

$$

\text{slope}(W) = \beta_1 + \beta_3 W, \qquad \text{Var}[\text{slope}(W)] = \text{Var}(\beta_1) + W^2\text{Var}(\beta_3) + 2W\,\text{Cov}(\beta_1,\beta_3)

$$

Johnson-Neyman interval solved from the t-critical boundary equation.
Optional permutation test: permute $Y$ and evaluate $|\hat{\beta}_3^\text{perm}| \geq |\hat{\beta}_3^\text{obs}|$.

---

### 17. Mixed Effects

**Module:** `stages/advanced.py::stage_mixed_effects_impl`

The subject-level stage is a no-op by design. Mixed-effects models require multiple subjects and are provided at the group level (see [Group-Level Utilities](#group-level-utilities)).

---

### 18. Consistency

**Module:** `stages/diagnostics.py` → `stats/consistency.py`

Computes sign-consistency/sign-flip diagnostics:

$$

\text{flip} = \mathbf{1}[\operatorname{sign}(a) \cdot \operatorname{sign}(b) < 0] \quad \text{(finite/nonzero values only)}

$$

---

### 19. Influence

**Module:** `stages/diagnostics.py` → `stats/influence.py`

Linear-model influence metrics:

$$

h_i = x_i^\top (X^\top X)^{-1} x_i, \qquad D_i = \frac{e_i^2}{p \cdot \text{MSE}} \cdot \frac{h_i}{(1-h_i)^2}

$$

Default thresholds: $D_i > 4/n$ and $h_i > 2p/n$.

---

### 20. Hierarchical FDR Summary

**Module:** `stages/fdr.py`

Aggregates cached unified-FDR metadata from prior stages.

---

### 21. Report and Export

**Modules:** `stages/report.py`, `stages/export.py`

Serializes and normalizes outputs for downstream use.

---

## Group-Level Utilities

### 1. Group-Level Mixed Effects

**Module:** `analysis/behavior/group_level.py::run_group_level_mixed_effects_impl`

Fits feature-wise mixed models across subjects (`MixedLM`), then applies hierarchical FDR.

### 2. Group-Level Multilevel Correlations

**Module:** `analysis/behavior/group_level.py::run_group_level_correlations_impl`

Per-subject associations are aggregated with Fisher averaging:

$$

r_\text{group} = \tanh\!\left(\operatorname{mean}_s\left[\operatorname{atanh}(r_s)\right]\right)

$$

Permutation primary p-value: $p_\text{perm} = (\#\text{extreme} + 1)/(n_\text{perm} + 1)$.

Optional fallback (config): one-sample t-test of subject Fisher-z values vs 0.

---

## FDR Formula Reference

BH q-values:

$$

q_{(i)} = \min_{j \geq i} \frac{m}{j}\,p_{(j)}

$$

Hierarchical FDR family gate (Simes):

$$

p_\text{Simes} = \min_i \frac{m_f}{i}\,p_{(i,\text{family})}

$$

Within-family rejections are retained only when the family gate rejects.
