# Behavior Computation Pipeline (THIS IS STILL UNDER DEVELOPMENT AND IS SUBJECT TO CHANGES)

DAG-based statistical analysis pipeline correlating EEG features with behavioral measures (pain ratings, temperature, experimental conditions). Executes as a dependency-resolved sequence of stages, each with single responsibility.

## Architecture

The pipeline is orchestrated by a **stage registry** with explicit dependency resolution. Each stage declares its required inputs and produced outputs. The DAG executor resolves prerequisites automatically, so requesting `correlations` will auto-enable `load → trial_table → correlate_design → ...`.

```
load → trial_table → lag_features
                   → pain_residual
                   → temperature_models
                   → correlate_design → correlate_effect_sizes → correlate_pvalues → correlate_primary_selection → correlate_fdr
                   → pain_sensitivity
                   → regression
                   → models
                   → stability
                   → condition_column / condition_window
                   → temporal_tfr / temporal_stats
                   → cluster
                   → mediation
                   → moderation
                   → mixed_effects (group-level only)
                   → consistency (requires correlations)
                   → influence (requires correlations)
                   → hierarchical_fdr_summary
                   → report → export
```

## Statistical Safeguards

### Non-i.i.d. Inference

Trial-level analyses (correlations, regression, condition comparisons) enforce valid inference for non-independent observations:

- **Default:** Permutation testing with block/run-restricted permutations is required.
- **Alternative:** Run-level aggregation (`primary_unit=run_mean`) averages features per run before testing, eliminating pseudo-replication.
- **Override:** `allow_iid_trials=true` bypasses the check (not recommended).

### Temperature Control

Pain ratings covary with stimulus temperature. The pipeline separates temperature-driven from neural-driven effects:

1. **Partial correlations** — Regress out temperature (and trial order) before computing feature–rating correlations.
2. **Pain residual** — `rating − f(temperature)` removes the stimulus-intensity component, leaving "pain beyond what temperature predicts."
3. **Temperature spline** — Flexible nonlinear temperature→rating curve (natural cubic spline with configurable knots).

When the target *is* temperature, temperature control is automatically disabled to avoid circular partialling.

### Multiple Comparison Correction

All stages use a unified FDR framework:

1. **Within-family FDR** — Benjamini-Hochberg correction within each feature type (power, connectivity, etc.).
2. **Hierarchical FDR** — Two-level correction: family-level significance gates, then within-family q-values.
3. **Global FDR** — Cross-family correction for the final combined results table.

P-value priority: `permutation > partial_cov_temp > partial_temp > partial_cov > raw`.

---

## Stages

### 1. Trial Table

**Module:** `orchestration.py` → `stage_trial_table`

Builds a canonical trial-level DataFrame merging all feature tables with behavioral event metadata.

**Method:**

1. Load all feature DataFrames from the feature extraction pipeline (power, connectivity, aperiodic, ERP, etc.).
2. Validate index alignment across all feature tables.
3. Prefix each feature column with its category name (e.g., `power_`, `connectivity_`).
4. Merge with `aligned_events` (from BIDS `events.tsv`) containing `rating`, `temperature`, `pain_binary`, `trial_index`, `run_id`.
5. Optionally compute **change scores** (active − baseline) for features that have both segment variants.

**Output:** `trials.parquet` — One row per trial, all features + behavioral variables.

---

### 2. Lag Features

**Module:** `orchestration.py` → `stage_lag_features`

Adds temporal dynamics variables for habituation and sensitization analysis.

**Computed variables:**

| Variable | Computation |
|----------|-------------|
| `prev_temperature` | Temperature of the preceding trial (within run). |
| `delta_temperature` | `temperature − prev_temperature` |
| `prev_rating` | Rating of the preceding trial (within run). |
| `delta_rating` | `rating − prev_rating` |
| `trial_index_within_group` | 0-indexed trial position within each run/block. |

Lag computation respects run boundaries — no carry-over across runs.

---

### 3. Pain Residual

**Module:** `orchestration.py` → `stage_pain_residual`

Computes `pain_residual = rating − f(temperature)`, representing pain perception beyond what stimulus intensity predicts.

**Method:**

1. Fit a flexible temperature→rating curve (natural cubic spline or polynomial, configurable).
2. Optionally use **cross-validated (out-of-run) prediction** to avoid overfitting the temperature–rating relationship.
3. Compute residuals: `pain_residual = rating − predicted_rating`.

**Output:** `pain_residual` column added to the trial table. Used as the preferred correlation target when `prefer_pain_residual=true` (default).

---

### 4. Temperature Models

**Module:** `orchestration.py` → `stage_temperature_models`

Characterizes the temperature→rating psychophysical function.

**Sub-computations:**

- **Model comparison** — Fits linear, polynomial (degree 2–4), and natural spline models to temperature→rating. Compares via AIC/BIC to select the best-fitting curve shape.
- **Breakpoint detection** — Tests for threshold temperatures where pain sensitivity changes (piecewise regression with breakpoint search).

**Output:** Model comparison table + breakpoint candidates with test statistics.

---

### 5. Correlations (5-stage sub-pipeline)

The correlation pipeline is decomposed into five atomic stages:

#### 5a. Correlate Design

**Module:** `orchestration.py` → `stage_correlate_design`

Assembles the design matrix: identifies target columns, builds covariate DataFrame (trial order, run dummies), resolves temperature series, and determines permutation groups.

**Targets** (configurable, default order): `pain_residual`, `rating`, `temperature`.

**Covariates:**
- `trial_index_within_group` — Controls for habituation/sensitization.
- Run dummy variables — Controls for between-run variance (when `run_adjustment.enabled=true`).

#### 5b. Correlate Effect Sizes

**Module:** `orchestration.py` → `stage_correlate_effect_sizes`

Computes correlation coefficients for every feature × target pair. Parallelized via `joblib` for large feature sets (≥100 pairs).

**Correlation types** (configurable, computed selectively):

| Type | Method |
|------|--------|
| `raw` | Bivariate Spearman/Pearson correlation. |
| `partial_cov` | Partial correlation controlling for trial order (+ run dummies). |
| `partial_temp` | Partial correlation controlling for temperature. |
| `partial_cov_temp` | Partial correlation controlling for both covariates and temperature. **Default.** |
| `run_mean` | Run-level aggregated correlation (mean feature/target per run). |

**Robust methods** (optional): `percentage_bend`, `winsorized`, `shepherd` — when enabled, partial correlations are disabled (robust methods don't support partialling).

#### 5c. Correlate P-values

**Module:** `orchestration.py` → `stage_correlate_pvalues`

Adds permutation-based p-values to each effect size record.

**Method:**
1. For each feature × target pair, permute the target within blocks/runs (`permute_within_groups`).
2. Recompute the correlation on each permuted dataset.
3. Two-tailed p-value: `p = (n_extreme + 1) / (n_perm + 1)`.
4. Permutation p-values are computed for all correlation types (raw, partial_cov, partial_temp, partial_cov_temp).

**Permutation schemes:** `shuffle` (within-group exchangeability), `circular_shift` (within-group circular shift; preserves within-run autocorrelation).

#### 5d. Correlate Primary Selection

**Module:** `orchestration.py` → `stage_correlate_primary_selection`

Selects the primary p-value and effect size for each record based on the control settings:

| Control Settings | Primary Selected |
|-----------------|------------------|
| `control_temperature + control_trial_order` | `r_partial_cov_temp`, `p_partial_cov_temp` |
| `control_temperature` only | `r_partial_temp`, `p_partial_temp` |
| `control_trial_order` only | `r_partial_cov`, `p_partial_cov` |
| Neither | `r_raw`, `p_raw` |

If permutation p-values are available, the corresponding permutation p-value replaces the asymptotic one.

#### 5e. Correlate FDR

**Module:** `orchestration.py` → `stage_correlate_fdr`

Applies Benjamini-Hochberg FDR correction to the primary p-values. Saves the final correlation results table.

**Output columns:** `r_raw`, `p_raw`, `r_partial_cov_temp`, `p_partial_cov_temp`, `p_perm_*`, `p_primary`, `p_fdr`, `q_within_family`, `q_global`.

---

### 6. Pain Sensitivity

**Module:** `orchestration.py` → `stage_pain_sensitivity`

Correlates EEG features with a temperature-adjusted pain index derived from the temperature→rating relationship.

**Method:**

1. Fit a simple temperature→rating model and compute residuals (rating minus predicted-from-temperature).
2. Correlate each EEG feature with this residual index.
3. Optional permutation testing with within-group permutations (e.g., run/block-restricted).
4. FDR correction within the pain_sensitivity analysis family.

**Unit of analysis:** Configurable as `trial` (with permutation requirement) or `run_mean` (aggregated per run).

---

### 7. Regression

**Module:** `orchestration.py` → `stage_regression`

Trialwise multiple regression for each feature:

```
rating ~ temperature + trial_order + feature (+ temperature × feature interaction)
```

**Method:**

1. For each feature, fit an OLS model with HC3 robust standard errors.
2. Extract the feature coefficient (`b_feature`), its p-value, and the interaction term.
3. Optional permutation testing for the feature coefficient.

**Unit of analysis:** `trial` (with permutation) or `run_mean` (aggregated per run to avoid pseudo-replication).

**Output:** Per-feature regression coefficients, standard errors, p-values, and interaction effects.

---

### 8. Models

**Module:** `orchestration.py` → `stage_models`

Fits multiple model families per feature to assess robustness of effects across estimation methods.

**Model families:**

| Family | Method |
|--------|--------|
| **OLS-HC3** | Ordinary least squares with heteroscedasticity-consistent (HC3) standard errors. |
| **Robust** | M-estimator regression (Huber or bisquare weights). |
| **Quantile** | Quantile regression at the median (robust to outliers and skewness). |
| **Logistic** | Binary logistic regression predicting `pain_binary` from the feature. |

**Outcomes:** `rating`, `pain_residual`, and `pain_binary` (configurable).

---

### 9. Stability

**Module:** `orchestration.py` → `stage_stability`

Assesses within-subject cross-run stability of feature→outcome associations.

**Method:**

1. For each feature, compute the feature–outcome correlation separately within each run/block.
2. Summarize: mean correlation across runs, standard deviation, and a stability index.
3. Features with inconsistent effects across runs are flagged (non-gating — does not exclude features).

**Output:** Per-feature stability metrics (`r_mean`, `r_std`, `stability_index`).

---

### 10. Consistency

**Module:** `orchestration.py` → `stage_consistency`

Merges results from correlations, regression, and models stages to flag **effect-direction contradictions**.

**Method:**

1. For each feature, extract the sign of the effect from each analysis method (correlation r, regression β, model coefficients).
2. Flag features where the effect direction disagrees across methods.
3. Non-gating: flags are informational, not exclusionary.

---

### 11. Influence

**Module:** `orchestration.py` → `stage_influence`

Detects influential observations that disproportionately drive statistical results.

**Diagnostics:**

| Metric | Description |
|--------|-------------|
| **Cook's D** | Influence of each trial on the regression fit. Trials with `D > 4/n` are flagged. |
| **Leverage** | Hat-matrix diagonal values. High-leverage points have unusual predictor values. |
| **DFFITS** | Standardized difference in fitted values when each observation is removed. |

Computed for top-ranked features from the correlation/regression results.

---

### 12. Condition Comparison

**Module:** `orchestration.py` → `stage_condition_column`, `stage_condition_window`

Compares EEG features between experimental conditions.

#### Column-based contrast

Splits trials by a categorical column (e.g., `pain_binary`: pain vs. non-pain).

**Effect size metrics:**

| Metric | Computation |
|--------|-------------|
| `cohens_d` | `(mean_group2 − mean_group1) / SD_pooled` (paired: Cohen's dz from difference scores). |
| `hedges_g` | Bias-corrected Cohen's d: `d × (1 − 3/(4n−1))`. |
| `mean_diff` | Raw mean difference between conditions. |

**Inference:** Permutation-tested (block-restricted) or run-level aggregated. FDR-corrected.

**Multigroup:** When `compare_values` has 3+ levels, delegates to Kruskal-Wallis / one-way ANOVA with post-hoc pairwise comparisons.

#### Window-based contrast

Compares features across time windows (e.g., baseline vs. active) using paired effect sizes (Cohen's dz).

---

### 13. Temporal Analysis

**Module:** `orchestration.py` → `stage_temporal_tfr`, `stage_temporal_stats`

Time-resolved feature–behavior correlations.

#### Time-Frequency Correlations (`temporal_tfr`)

Correlates TFR power at each (time, frequency) bin with pain ratings. Produces a time-frequency correlation map.

#### Temporal Statistics (`temporal_stats`)

Computes time-resolved correlations for power, ITPC, and ERDS features across trial phases.

**Multiple comparison correction** (configurable):

| Method | Description |
|--------|-------------|
| `fdr` | Benjamini-Hochberg across all time × frequency cells. **Default.** |
| `cluster` | Cluster-based permutation test (preferred for dense grids). |
| `bonferroni` | Conservative Bonferroni correction. |
| `none` | No correction (use with caution). |

---

### 14. Cluster Permutation Tests

**Module:** `orchestration.py` → `stage_cluster`

Non-parametric cluster-based permutation tests on epoch data (MNE `spatio_temporal_cluster_test`).

**Method:**

1. For each frequency band, form clusters of contiguous time × channel points exceeding a forming threshold.
2. Compute cluster-level statistics (sum of t-values within each cluster).
3. Build a null distribution by permuting condition labels and recomputing cluster statistics.
4. Cluster p-value: proportion of permutation cluster statistics exceeding the observed one.

---

### 15. Mediation

**Module:** `orchestration.py` → `stage_mediation`

Tests whether EEG features mediate the temperature→rating causal pathway.

**Model:**

```
Path a: temperature → feature (mediator)
Path b: feature → rating (controlling for temperature)
Path c': temperature → rating (direct, controlling for feature)
Indirect effect: ab = a × b
```

**Inference:**

| Method | Description |
|--------|-------------|
| **Sobel test** | Asymptotic test of `ab ≠ 0`. |
| **Bootstrap CI** | Percentile bootstrap confidence interval for `ab` (default 1000 resamples). |
| **Permutation** | Block-restricted permutation p-value for the indirect effect. |

**Feature selection:** Optionally limited to top `max_mediators` features by variance.

---

### 16. Moderation

**Module:** `orchestration.py` → `stage_moderation`

Tests whether EEG features moderate the temperature→rating relationship (interaction effects).

**Model:**

```
rating = b0 + b1 × temperature + b2 × feature + b3 × (temperature × feature) + ε
```

If `b3` is significant, the feature moderates how temperature affects pain perception.

**Output per feature:**

| Field | Description |
|-------|-------------|
| `b3_interaction` | Interaction coefficient. |
| `p_interaction` | Asymptotic p-value for the interaction. |
| `p_interaction_perm` | Permutation p-value (block-restricted). |
| `slope_low_w` / `slope_high_w` | Simple slopes at ±1 SD of the feature (Johnson-Neyman probing). |
| `jn_low` / `jn_high` | Johnson-Neyman significance transition points. |
| `r_squared_change` | ΔR² from adding the interaction term. |

---

### 17. Mixed Effects (Group-Level)

**Module:** `orchestration.py` → `run_group_level_mixed_effects`

Multi-subject mixed-effects models with subject as random effect. **Only runs at group level** (≥2 subjects).

**Model:**

```
rating ~ feature_value + (1 | subject_id)          # random intercept (default)
rating ~ feature_value + (feature_value | subject_id)  # random slope
```

**Method:**

1. Concatenate trial tables across subjects.
2. For each feature, fit `statsmodels.MixedLM` with REML estimation.
3. Extract fixed-effect coefficient, z-statistic, and p-value.
4. Apply hierarchical FDR correction by feature family.

**Output:** Fixed-effect coefficients, AIC/BIC, convergence status, hierarchical q-values.

---

### 18. Multilevel Correlations (Group-Level)

**Module:** `orchestration.py` → `run_group_level_correlations`

Cross-subject correlations with block-aware permutation testing.

**Method:**

1. Concatenate trial tables across subjects.
2. Compute feature–rating correlations on the combined data.
3. Permute within blocks (subject × run) to preserve the dependence structure.
4. Hierarchical FDR correction by feature family.

---

### 19. Feature QC Screen

**Module:** `orchestration.py` → `stage_feature_qc_screen`

Pre-inference quality control filtering.

**Filters:**

| Criterion | Threshold |
|-----------|-----------|
| **High missingness** | `> max_missing_pct` (default 50%) of values are NaN. |
| **Near-zero variance** | Variance `< min_variance` (default 1e-10). |
| **Constant within run** | No variation within any run (all run-level variances ≈ 0). |

Features failing QC are excluded from downstream inference. Non-gating: the QC table is saved for inspection.

---

### 20. Report

**Module:** `orchestration.py` → `stage_report`

Generates a self-diagnosing Markdown report summarizing all analysis results for a single subject.

---

### 21. Export

**Module:** `orchestration.py` → `stage_export`

Normalizes and exports all analysis results to disk. Produces:
- Combined correlation tables (rating, temperature, pain_sensitivity).
- Condition effect tables.
- Regression, mediation, moderation results.
- Outputs manifest JSON for downstream consumption.

---

## Feature-Behavior Correlator

**Module:** `feature_correlator.py` → `FeatureBehaviorCorrelator`

An alternative entry point that loads all feature files from the registry and runs a complete correlation analysis in a single call. Used by `run_unified_feature_correlations` from the `BehaviorContext`.

**Capabilities:**
- Loads all feature types from the feature registry.
- Computes raw, partial, and permutation-tested correlations.
- Bootstrap confidence intervals.
- Bayes factors (optional).
- LOSO (leave-one-session-out) stability.
- Split-half reliability.
- ROI-level correlations for power features.

---

## Configuration

All behavior analysis settings live under `behavior_analysis` in `eeg_config.yaml`:

| Key | Description | Default |
|-----|-------------|---------|
| `statistics.correlation_method` | `spearman` or `pearson` | `spearman` |
| `statistics.fdr_alpha` | FDR correction threshold | `0.05` |
| `statistics.n_permutations` | Permutation iterations | `1000` |
| `statistics.allow_iid_trials` | Bypass non-i.i.d. check | `false` |
| `control_temperature` | Partial out temperature | `true` |
| `control_trial_order` | Partial out trial order | `true` |
| `correlations.types` | Which correlation types to compute | `["partial_cov_temp"]` |
| `correlations.prefer_pain_residual` | Use pain_residual as primary target | `true` |
| `correlations.primary_unit` | `trial` or `run_mean` | `trial` |
| `pain_sensitivity.primary_unit` | `trial` or `run_mean` | `trial` |
| `regression.primary_unit` | `trial` or `run_mean` | `trial` |
| `condition.compare_column` | Column for condition split | `pain` |
| `condition.compare_values` | Values defining conditions | `[1, 0]` |
| `mediation.n_bootstrap` | Bootstrap resamples for mediation | `1000` |
| `mediation.max_mediators` | Max features to test as mediators | `20` |
| `moderation.max_features` | Max features for moderation | `50` |
| `robust_correlation` | Robust method (`percentage_bend`, `winsorized`, `shepherd`) | `none` |
| `run_adjustment.enabled` | Include run dummies as covariates | `false` |
| `permutation.scheme` | `shuffle` or `circular_shift` | `circular_shift` |

## Dependencies

- **statsmodels** — Mixed-effects models, robust regression, quantile regression
- **scipy.stats** — Spearman/Pearson correlations, permutation testing
- **joblib** — Parallel computation for large feature sets
- **pandas** — Trial table construction, data alignment
- **NumPy** — Numerical computation
- **MNE-Python** — Cluster permutation tests, epoch handling
