# Behavioral Statistics Pipeline

**Module:** `eeg_pipeline.analysis.behavior`

This document is the methods reference for the behavioral statistics pipeline.
Analyses run as a dependency-resolved DAG over aligned EEG and behavioral data
(outcome variable, predictor, experimental conditions).
The pipeline targets trial-level associations between behavioral variables and
EEG-derived features, with explicit non-i.i.d. safeguards and per-stage outputs.

The pipeline is **paradigm-agnostic**: it supports any combination of continuous,
binary, or categorical predictors and any scalar outcome measure. See §4 for how
`predictor_type` gates analyses that require a continuous predictor.

---

## Table of Contents

1. [Notation](#1-notation)
2. [Module Structure](#2-module-structure)
3. [Pipeline DAG](#3-pipeline-dag)
4. [Statistical Safeguards](#4-statistical-safeguards)
5. [Stage Definitions](#5-stage-definitions)
   - 5.1 [Load and Metadata](#51-load-and-metadata)
   - 5.2 [Trial Table](#52-trial-table)
   - 5.3 [Predictor Residual](#53-predictor-residual)
   - 5.4 [Correlations](#54-correlations)
   - 5.5 [Regression](#55-regression)
   - 5.6 [ICC Reliability](#56-icc-reliability)
   - 5.7 [Condition Comparisons](#57-condition-comparisons)
   - 5.8 [Temporal Statistics](#58-temporal-statistics)
   - 5.9 [Cluster Tests](#59-cluster-tests)
   - 5.10 [Hierarchical FDR Summary](#510-hierarchical-fdr-summary)
   - 5.11 [Report and Export](#511-report-and-export)
6. [Group-Level Analysis](#6-group-level-analysis)
7. [Multiple Comparison Correction](#7-multiple-comparison-correction)

---

## 1. Notation

| Symbol | Meaning |
|--------|---------|
| $i \in \{1,\dots,N_\text{trials}\}$ | Trial index within a subject |
| $g$ | Grouping unit (run / block) for non-i.i.d. safeguards |
| $s$ | Subject index (group-level aggregation) |
| $f \in \{1,\dots,F\}$ | EEG-derived feature index |
| $y_i$ | Behavioral outcome for trial $i$ (e.g. subjective rating, detection score) |
| $P_i$ | Predictor value for trial $i$ (e.g. stimulus intensity, dose, condition label) |
| $x_{i,f}$ | EEG feature $f$ for trial $i$ |
| $z_i$ | Covariate vector (age, sex, questionnaire scores, …) |
| $G_i \in \{1,\dots,N_\text{groups}\}$ | Group label (run / block / cluster) |
| $n_\text{perm}$ | Number of permutations |
| $\varepsilon$ | Model residual |

The **trial table** stores one row per trial $i$ with columns for trial metadata
(subject, run/block, condition), behavioral variables ($y_i$, $P_i$, covariates),
and EEG feature columns ($x_{i,f}$) joined from feature tables.

Trials are **not i.i.d.**: they are clustered within runs/blocks and subjects.
All permutation-based inference respects this structure via grouped label shuffling,
and group-level analyses aggregate subject-level statistics rather than pooling trials.

---

## 2. Module Structure

```
behavior/
├── api.py                   # Statistical entry points (correlations, conditions, cluster, temporal)
├── orchestration.py         # DAG execution engine
├── stage_registry.py        # Stage registration and dependency resolution
├── stage_catalog.py         # Complete stage catalog with metadata
├── stage_runners.py         # Per-stage runner dispatch
├── stage_execution.py       # Execution context and error handling
├── config_resolver.py       # Configuration resolution per stage
├── group_level.py           # Group-level mixed effects and correlations
├── feature_correlator.py    # Feature-wise correlation utilities
├── feature_inference.py     # Feature-level inference helpers
├── feature_filters.py       # Feature selection and QC filtering
├── trial_table_helpers.py   # Trial table construction utilities
├── change_scores.py         # Change-score computation
├── result_types.py          # Result dataclasses
├── result_cache.py          # Stage output caching
├── common_helpers.py        # Shared statistical helpers
└── stages/
    ├── metadata.py          # load, metadata stages
    ├── trial_table.py       # trial_table, predictor_residual
    ├── models.py            # regression
    ├── correlate.py         # correlate_* stages
    ├── condition.py         # condition_column, multigroup (when 3+ groups)
    ├── temporal.py          # temporal_tfr, temporal_stats, cluster
    ├── diagnostics.py       # icc
    ├── fdr.py               # hierarchical_fdr_summary
    ├── report.py            # report stage
    └── export.py            # export stage
```

---

## 3. Pipeline DAG

Stages run in dependency order. Arrows denote data dependencies.

```
load
└── trial_table
    ├── predictor_residual          [continuous predictor only]
    ├── correlate_design
    │   ├── correlate_effect_sizes
    │   │   ├── correlate_pvalues
    │   │   │   └── correlate_primary_selection
    │   │   │       └── correlate_fdr
    ├── regression
    ├── icc
    ├── condition_column
    ├── temporal_tfr
    │   └── temporal_stats
    │       └── cluster
    └── hierarchical_fdr_summary
        ├── report
        └── export
```

Group-level computations run outside the subject DAG via `BehaviorPipeline.run_group_level(...)`.

---

## 4. Statistical Safeguards

### 4.1 Non-i.i.d. Trial Structure

Trials within a subject are clustered within runs/blocks and are not exchangeable.
The following stages enforce grouped permutation unless `allow_iid_trials = true`
is explicitly set in configuration:

- `correlate_*`, `regression`, `condition_column`

The following require an explicit i.i.d. override for trial-level inference:

- multigroup condition comparison (when 3+ groups)

Temporal trial-level inference (`temporal_stats`) requires cluster correction with
valid group labels when `allow_iid_trials = false`.

### 4.2 Predictor Control

Partial correlation and residualization paths are both available for controlling the
effect of the predictor. The available control strategies depend on `predictor_type`:

| Strategy | Description | Requires |
|----------|-------------|----------|
| `linear` | Add predictor as a linear covariate | Any predictor type |
| `outcome_hat` | Use $\hat{y} = f(P)$ (fitted outcome) as covariate | `continuous` only |
| `spline` | Restricted cubic spline of predictor as covariate | `continuous` only |

Predictor control is skipped when the analysis target is the predictor itself.

### 4.3 Predictor Type Validation

The `behavior_analysis.predictor_type` key declares the nature of the predictor:

- **`continuous`** — ordered numeric scale with ≥ 5 distinct levels (e.g. stimulus intensity, dose). Enables `predictor_residual` and spline/outcome_hat control.
- **`binary`** — two-level factor (0/1 or condition A/B). Disables curve-fitting analyses.
- **`categorical`** — unordered multi-level factor (≥ 3 levels). Same restrictions as binary.

Attempting to run `predictor_residual` when `predictor_type ≠ continuous` raises a `ValueError`
at the analysis entry point via `assert_continuous_predictor()` / `assert_predictor_type_continuous()`
(in `eeg_pipeline.utils.analysis.stats.validation`).

### 4.4 Multiple Comparison Correction

- **Unified hierarchical FDR** is applied across wrapped stages (see §7).
- **Local BH correction** (`p_fdr`) is computed within the regression stage.
- **Temporal correction** mode is configurable per analysis:
  `cluster`, `fdr`, `bonferroni`, or `none`.

---

## 5. Stage Definitions

### 5.1 Load and Metadata

**Module:** `stages/metadata.py`

**`load`** — Reads aligned behavioral events, predictor series, covariates,
and trial-wise EEG feature tables into `BehaviorContext`.
Produces the core behavioral series ($y_i$, $P_i$) and a named mapping
of feature tables used by all downstream stages.

The pipeline writes a JSON QC summary (`analysis_metadata.json`) after stage
execution (not as a separate DAG stage), via `stages/metadata.py`:

- Trial counts, missingness fractions, outcome and predictor distributions.
- Condition contrasts, covariate coverage, feature counts.
- Analysis configuration fields (method labels, permutation counts, FDR level).

---

### 5.2 Trial Table

**Module:** `stages/trial_table.py` → `trial_table_helpers.py`

Constructs the canonical trial-level DataFrame by joining aligned behavioral
events with all named feature tables on trial identifiers.
One row per trial; one column per behavioral or EEG-feature variable.

---

### 5.3 Predictor Residual

**Module:** `stages/trial_table.py`

**Requires:** `predictor_type = continuous` (≥ 5 unique predictor values).

Residualizes the outcome on the predictor to isolate variance not explained
by stimulus intensity:

```math
\text{predictor\_residual}_i = y_i - \hat{y}_i, \qquad \hat{y}_i = f(P_i).
```

Model selection procedure:

1. Fit spline OLS candidates `outcome ~ bs(predictor, df=d, degree=3)` for increasing $d$; retain the model with lowest AIC.
2. If no spline model converges, fit a polynomial: $\hat{y} = \sum_j a_j P^j$.
3. Optionally compute cross-fit residuals (`predictor_residual_cv`) via `GroupKFold` to avoid in-sample bias.

---

### 5.4 Correlations

**Modules:** `stages/correlate.py`, `feature_correlator.py`

#### 5.4.1 Design (`correlate_design`)

Defines analysis targets, covariate sets, and permutation group assignments.

#### 5.4.2 Effect Sizes (`correlate_effect_sizes`)

**Pearson:** $r = \mathrm{corr}(x, y)$.

**Spearman:** $r = \mathrm{corr}(\mathrm{rank}(x), \mathrm{rank}(y))$.

**Partial correlation** (covariate-controlled): residualize $x$ and $y$ on $Z$,
then correlate residuals. Test statistic with $k$ covariates:

```math
t = r\sqrt{\frac{n - k - 2}{1 - r^2}}, \qquad
p = 2\, P\!\left(|T_{n-k-2}| \ge |t|\right).
```

Run-mean mode computes correlations on run-aggregated means rather than trial-level values.

#### 5.4.3 P-values (`correlate_pvalues`)

Permutation p-value (one-sided extreme, Phipson–Smyth):

```math
p_\text{perm} = \frac{N_{\text{extreme}} + 1}{n_\text{perm} + 1}.
```

Grouped permutation schemes: shuffle or circular-shift within groups (`permute_within_groups`).
Partial permutation uses Freedman–Lane residual permutation.

#### 5.4.4 Primary Selection (`correlate_primary_selection`)

Selects the primary effect size and p-value according to control path (partial vs. simple),
analysis unit (trial vs. run), and non-i.i.d. enforcement status.

#### 5.4.5 FDR Correction (`correlate_fdr`)

Benjamini–Hochberg q-values applied across features (see §7).

---

### 5.5 Regression

**Module:** `stages/models.py`

Per-feature OLS model with optional predictor interaction:

```math
y = Z\gamma + \beta_f x_f + \beta_\text{int}(x_f \cdot P) + \varepsilon \quad \text{(full)}, \qquad
y = Z\gamma + \varepsilon \quad \text{(reduced)}.
```

**Incremental explained variance:**

```math
\Delta R^2 = R^2_\text{full} - R^2_\text{reduced}.
```

**HC3 heteroskedasticity-consistent standard errors:**

```math
h_i = x_i^\top (X^\top X)^{-1} x_i, \qquad
w_i = \frac{e_i^2}{(1 - h_i)^2},
```

```math
\widehat{\mathrm{Cov}}_\text{HC3} =
(X^\top X)^{-1} X^\top \mathrm{diag}(w)\, X (X^\top X)^{-1}.
```

**Inference:** $t = \hat\beta / \mathrm{SE}$, two-sided $t$ p-values.
Feature-term permutation uses reduced-model residual permutation (Freedman–Lane).

---

### 5.6 ICC Reliability

**Module:** `stages/diagnostics.py`

Intra-class correlation ICC(3,1) for assessing test–retest reliability
at the subject level:

```math
\mathrm{ICC}(3,1) =
\frac{MS_\text{rows} - MS_\text{error}}
     {MS_\text{rows} + (k-1)\, MS_\text{error}}.
```

---

### 5.7 Condition Comparisons

**Module:** `stages/condition.py`

#### 5.7.1 Two-Group (`condition_column`)

**Unpaired (default) — Welch t-test:**

```math
t = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{s_1^2/n_1 + s_2^2/n_2}}.
```

**Effect sizes:**

```math
d = \frac{\bar{x}_1 - \bar{x}_2}{s_\text{pooled}}, \qquad
g = d\!\left(1 - \frac{3}{4\,df - 1}\right), \qquad
d_z = \frac{\bar{d}}{s_d} \text{ (paired)}.
```

Permutation p-values: unpaired mean-difference and paired sign-flip,
both using $(N_{\text{extreme}} + 1) / (n_\text{perm} + 1)$.

#### 5.7.2 Multi-Group (condition flow, 3+ groups)

When the condition column has 3+ levels, the pipeline runs a multigroup comparison
(via the same condition stage; not a separate DAG node). Pairwise tests only (no omnibus):

- Unpaired: Mann–Whitney U.
- Paired run-level: Wilcoxon signed-rank.

---

### 5.8 Temporal Statistics

**Module:** `stages/temporal.py`

Power-bin statistic averaged within time window $w$:

```math
b_{i,f,w} = \mathrm{mean}_{t \in w}\, P_{i,f,t}.
```

Bin values $b$ are correlated with the behavioral outcome (simple or partial).
Correlation-to-$t$ transform for cluster forming:

```math
t = r\sqrt{\frac{\mathrm{dof}}{1 - r^2}}.
```

Temporal multiple-comparison correction: `fdr`, `bonferroni`, `cluster`, or `none`.

**ITPC trial metric:**

```math
\mathrm{ITPC}_\text{trial} = \mathrm{mean}\!\left(\cos(\phi_\text{trial} - \bar\phi)\right).
```

**ERDS trial metrics:**

```math
\mathrm{ERDS\%} =
100 \cdot \frac{P_\text{active} - P_\text{base}}{P_\text{base}}, \qquad
\mathrm{ERDS}_z =
\frac{P_\text{active} - P_\text{base}}{\sigma_\text{base}}.
```

---

### 5.9 Cluster Tests

**Module:** `stages/temporal.py`

Cluster-mass permutation test over time–frequency maps:

```math
d = \frac{\mu_A - \mu_B}{s_\text{pooled}}, \qquad
M_c = \sum_{i \in c} |t_i|, \qquad
p_c = \frac{\#\{M_\text{max}^\text{perm} \ge M_c\} + 1}{n_\text{perm} + 1}.
```

---

### 5.10 Hierarchical FDR Summary

**Module:** `stages/fdr.py`

Aggregates unified-FDR metadata cached by prior stages and emits a cross-stage
multiple-comparison summary (see §7).

---

### 5.11 Report and Export

**Modules:** `stages/report.py`, `stages/export.py`

Serialize and normalize per-stage outputs into standardized files for downstream
use (fMRI integration, ML pipeline, group-level aggregation).

---

## 6. Group-Level Analysis

Group-level computations operate across subjects and run outside the per-subject DAG
via `BehaviorPipeline.run_group_level(...)`.

### 6.1 Mixed Effects Models

**Module:** `group_level.py` → `run_group_level_mixed_effects_impl`

Fits feature-wise linear mixed models (`MixedLM`) across subjects with subject as
a random effect, then applies hierarchical FDR across features.

### 6.2 Multilevel Correlations

**Module:** `group_level.py` → `run_group_level_correlations_impl`

Per-subject correlation estimates $r_s$ are aggregated using Fisher $z$-averaging:

```math
r_\text{group} = \tanh\!\left(\mathrm{mean}_s\bigl[\mathrm{atanh}(r_s)\bigr]\right).
```

Group-level permutation p-value:

```math
p_\text{perm} = \frac{N_{\text{extreme}} + 1}{n_\text{perm} + 1}.
```

---

## 7. Multiple Comparison Correction

### 7.1 Benjamini–Hochberg (BH) Correction

Given $m$ ordered p-values $p_{(1)} \le \cdots \le p_{(m)}$, BH q-values are:

```math
q_{(i)} = \min_{j \ge i} \frac{m}{j}\, p_{(j)}.
```

### 7.2 Hierarchical FDR (Family-Gated)

Families of hypotheses (e.g. per feature domain) are gated by a Simes family-level test
before within-family corrections are applied:

```math
p_\text{Simes} = \min_i \frac{m_f}{i}\, p_{(i,\text{family})}.
```

Within-family rejections are retained only when the family gate rejects at the
configured FDR level $\alpha$.
