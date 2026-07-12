# Study 1 and Study 2 Data-Validity Repairs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct confirmed Study 1/2 code defects that can alter data alignment, null distributions, prediction-derived scores, source estimates, or inferential summaries without adding hard confirmatory runtime gates.

**Architecture:** Shared permutation helpers will define one index mapping per null draw; model predictions will expose full, nuisance, and EEG-residual components explicitly; Study 2 will consume the residual component and configured band groups. Source and spatial stages will use explicit conventions and provenance, while sample-size and artifact criteria remain advisory output fields.

**Tech Stack:** Python 3.11, NumPy, pandas, SciPy, scikit-learn, MNE-Python, Nilearn, pytest, Ruff, BrainSMASH.

---

### Task 1: Reconcile the Study 1 run-identifier contract

**Files:**
- Modify: `studies/tests/fmri/test_study1_targets.py`
- Modify: `studies/pain_study/study1/RUN_GUIDE.md`
- Verify: `studies/pain_study/study1/targets.py`

- [ ] **Step 1: Change one stale fixture to `run_id` and verify its former expectation fails**

```python
def _events_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "run_id": [1, 1],
            "trial_number": [1, 2],
            "pain_binary_coded": [1, 0],
            "onset": [22.150, 65.084],
            "duration": [0.001, 0.001],
        }
    )
```

Run: `pytest studies/tests/fmri/test_study1_targets.py -q`
Expected: the stale `uses_run_column` test still fails because it constructs `run` explicitly.

- [ ] **Step 2: Replace event-boundary `run` fixtures with `run_id` and preserve output assertions on `run`**

```python
events = pd.DataFrame(
    {
        "run_id": [1, 1, 2, 2],
        "trial_number": [1, 2, 12, 13],
        "pain_binary_coded": [1, 0, 1, 0],
        "onset": [10.0, 20.0, 30.0, 40.0],
        "duration": [0.001, 0.001, 0.001, 0.001],
    }
)
assert output["run"].tolist() == [1, 1, 2, 2]
```

- [ ] **Step 3: Run the target tests**

Run: `pytest studies/tests/fmri/test_study1_targets.py tests/pipelines/test_study1_targets.py -q`
Expected: all target tests pass.

### Task 2: Correct circular distance and define coherent permutation mappings

**Files:**
- Modify: `eeg_pipeline/analysis/machine_learning/circular_shift.py`
- Modify: `eeg_pipeline/analysis/machine_learning/orchestration.py`
- Modify: `studies/pain_study/study2/target_retrained_null.py`
- Modify: `studies/tests/pipelines/test_study1_feature_benchmark.py`
- Modify: `studies/tests/pipelines/test_study2_target_retrained_null.py`

- [ ] **Step 1: Write failing shortest-distance tests**

```python
def test_full_run_excludes_bidirectionally_short_shifts() -> None:
    shifts = admissible_circular_shifts(np.arange(1, 12))
    assert shifts == (5, 6)
```

Run the single test and confirm it reports `(5, 6, 7, 8, 9, 10)`.

- [ ] **Step 2: Implement shortest circular distance**

```python
forward = (retained - source_trials) % original_block_length
distance = np.minimum(forward, original_block_length - forward)
if np.all(distance >= min_original_distance):
    shifts.append(shift)
```

- [ ] **Step 3: Write failing tests for one mapping reused across folds**

```python
assert reconstruct.call_count == 1
np.testing.assert_array_equal(first_fold_mapping, second_fold_mapping)
```

- [ ] **Step 4: Add mapping helpers and apply them to fold-specific residuals**

```python
def sample_permutation_indices(
    groups: np.ndarray,
    *,
    runs: np.ndarray | None,
    trial_indices: np.ndarray | None,
    rng: np.random.Generator,
    scheme: str,
) -> np.ndarray:
    indices = np.arange(len(groups))
    # Choose one shift/permutation per subject-run and return source-row indices.
    return indices

def apply_permutation_indices(values: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return np.asarray(values)[np.asarray(indices, dtype=int)]
```

Sample once per draw in Study 1 and once in `stacked_permuted_targets`; reuse the mapping for every
fold while retaining fold-specific nuisance predictions and residuals.

- [ ] **Step 5: Run permutation tests**

Run: `pytest studies/tests/pipelines/test_study1_feature_benchmark.py studies/tests/pipelines/test_study2_target_retrained_null.py studies/tests/pipelines/test_study2_target_permutations.py -q`
Expected: all pass.

### Task 3: Expose and use held-out EEG residual predictions

**Files:**
- Modify: `eeg_pipeline/analysis/machine_learning/orchestration.py`
- Modify: `studies/pain_study/study2/target_retrained_null.py`
- Modify: `studies/pain_study/scripts/study2_prepare_source_stage_input.py`
- Modify: `tests/machine_learning/test_model_comparison_protocol.py`
- Modify: `studies/tests/pipelines/test_study2_target_retrained_null.py`

- [ ] **Step 1: Write a failing component-decomposition test**

```python
result = model_comparison_cv_predictions(
    model_name="elasticnet",
    pipe=pipe,
    param_grid=param_grid,
    X=X,
    y=y,
    groups=groups,
    meta=meta,
    outer_folds=outer_folds,
    inner_splits=2,
    outer_jobs=1,
    config=config,
    harmonization_mode="intersection",
    covariates=None,
    target_residualization_columns=("stimulus_temp",),
    collect_records=True,
)
np.testing.assert_allclose(result.full_prediction, result.nuisance_prediction + result.residual_prediction)
assert not np.array_equal(result.full_prediction, result.residual_prediction)
```

- [ ] **Step 2: Return a dataclass with explicit arrays**

```python
@dataclass(frozen=True)
class ModelComparisonPredictions:
    evaluation_target: np.ndarray
    full_prediction: np.ndarray
    nuisance_prediction: np.ndarray
    residual_prediction: np.ndarray
    records: tuple[dict[str, Any], ...]  # one mapping per outer fold
```

Populate the nuisance and residual arrays for every test row and update all callers.

- [ ] **Step 3: Make Study 2 assemble residual scores**

```python
fold_result = model_comparison_cv_predictions(
    model_name="study2_target_retrained_null",
    pipe=context.pipe,
    param_grid=dict(context.param_grid),
    X=context.X,
    y=stacked_targets[fold_index],
    groups=context.groups,
    meta=context.meta,
    outer_folds=[(train_idx, test_idx)],
    inner_splits=context.inner_splits,
    outer_jobs=1,
    config=context.config,
    harmonization_mode=context.harmonization_mode,
    covariates=context.covariates,
    target_residualization_columns=context.target_residualization_columns,
    collect_records=False,
    fixed_params=dict(context.fixed_params_by_fold[fold_index]),
)
scores[test_idx] = fold_result.residual_prediction[test_idx]
```

- [ ] **Step 4: Run model and Study 2 null tests**

Run: `pytest tests/machine_learning/test_model_comparison_protocol.py studies/tests/pipelines/test_study2_target_retrained_null.py -q`
Expected: all pass.

### Task 4: Produce combined and band-specific Study 2 scores

**Files:**
- Modify: `studies/pain_study/study2/contributions.py`
- Modify: `studies/pain_study/study2/sensor_patterns.py`
- Modify: `studies/pain_study/scripts/study2_prepare_source_stage_input.py`
- Modify: `studies/tests/pipelines/test_study2_contributions.py`
- Modify: `studies/tests/pipelines/test_study2_stages.py`

- [ ] **Step 1: Write a failing clean-gamma aggregation test**

```python
scores = compute_band_contribution_scores(
    X=X,
    feature_names=[
        NamingSchema.build("power", "active", "alpha", "ch", "logratio_mean", channel="Cz"),
        NamingSchema.build(
            "power", "active", "gamma_low_clean", "ch", "logratio_mean", channel="Cz"
        ),
        NamingSchema.build(
            "power", "active", "gamma_high_clean", "ch", "logratio_mean", channel="Cz"
        ),
    ],
    coefficients=np.ones(3),
    band_members={"alpha": ("alpha",), "gamma": ("gamma_low_clean", "gamma_mid_clean", "gamma_high_clean")},
)
np.testing.assert_allclose(scores["eta_gamma"], X[:, 1] + X[:, 2])
```

- [ ] **Step 2: Replace exact matching with explicit configured membership**

```python
feature_mask = np.asarray([feature_band in band_members[band] for feature_band in feature_bands])
```

- [ ] **Step 3: Reuse frozen-fold preprocessing to calculate held-out contribution arrays**

Extract the fold refit currently duplicated in `sensor_patterns.py` into a helper that returns the
fitted estimator, transformed test matrix, retained names, and inverse-transformed residual
prediction. Calculate contributions in transformed target space, sum configured feature groups, and
standardize each score within held-out subject.

- [ ] **Step 4: Write all expected source-stage columns**

```python
for column in ("eta_combined", "eta_alpha", "eta_beta", "eta_gamma"):
    frame[f"{column}_z"] = standardized[column]
```

- [ ] **Step 5: Run contribution and stage tests**

Run: `pytest studies/tests/pipelines/test_study2_contributions.py studies/tests/pipelines/test_study2_stages.py -q`
Expected: all pass.

### Task 5: Correct point-spread orientation and metadata

**Files:**
- Modify: `studies/pain_study/study2/point_spread.py`
- Modify: `studies/pain_study/study2/stages.py`
- Modify: `studies/tests/pipelines/test_study2_point_spread.py`

- [ ] **Step 1: Write an asymmetric failing test that distinguishes rows from columns**

```python
resolution = np.array([[1.0, 0.6, 0.0], [0.1, 1.0, 0.6], [0.0, 0.1, 1.0]])
report = compute_point_spread_fwhm(resolution_matrix=resolution, distances_mm=distances)
np.testing.assert_allclose(report.vertex_fwhm_mm, expected_from_columns)
```

- [ ] **Step 2: Iterate over columns and rename the helper argument**

```python
vertex_fwhm = np.asarray([_single_vertex_fwhm(resolution[:, index], distances) for index in range(resolution.shape[1])])
```

- [ ] **Step 3: Record `resolution_matrix_axis: columns_are_psfs` in the summary**

- [ ] **Step 4: Run point-spread tests**

Run: `pytest studies/tests/pipelines/test_study2_point_spread.py -q`
Expected: all pass.

### Task 6: Repair spatial-surrogate inference and multiplicity

**Files:**
- Modify: `pyproject.toml`
- Create: `studies/pain_study/study2/spatial_surrogates.py`
- Modify: `studies/pain_study/study2/spatial_comparison.py`
- Modify: `studies/pain_study/study2/stages.py`
- Modify: `studies/pain_study/study2/runner.py`
- Create: `studies/tests/pipelines/test_study2_spatial_surrogates.py`
- Modify: `studies/tests/pipelines/test_study2_spatial_comparison.py`
- Modify: `studies/tests/pipelines/test_study2_stages.py`

- [ ] **Step 1: Write failing validation and Holm tests**

```python
with pytest.raises(ValueError, match="configured 5000"):
    compute_spatial_correspondence(
        eeg_map=np.arange(n_vertices, dtype=float),
        fmri_map=np.arange(n_vertices, dtype=float),
        surrogate_maps=np.zeros((999, n_vertices), dtype=float),
        mask=np.ones(n_vertices, dtype=bool),
        config=load_study2_config(),
    )

summary = adjust_spatial_family(pd.DataFrame({"band": ["alpha", "beta", "gamma"], "p_value": [.01, .03, .2]}))
np.testing.assert_allclose(summary["p_value_holm"], [.03, .06, .2])
```

- [ ] **Step 2: Add deterministic BrainSMASH generation**

```python
def generate_brainsmash_surrogates(surface_values, coordinates, *, n_surrogates, seed):
    from brainsmash.mapgen.base import Base
    return np.asarray(Base(x=surface_values, D=distance_matrix, seed=seed)(n=n_surrogates))
```

Add `brainsmash` to project dependencies and a stage that writes arrays plus JSON metadata containing
seed, hemisphere, vertex checksum, generator, and count.

- [ ] **Step 3: Require exact configured count and matching provenance**

- [ ] **Step 4: Holm-adjust the three band p-values in `run_spatial_correspondence`**

- [ ] **Step 5: Run spatial tests**

Run: `pytest studies/tests/pipelines/test_study2_spatial_surrogates.py studies/tests/pipelines/test_study2_spatial_comparison.py studies/tests/pipelines/test_study2_stages.py -q`
Expected: all pass.

### Task 7: Add non-blocking data-validity diagnostics

**Files:**
- Modify: `eeg_pipeline/analysis/machine_learning/orchestration.py`
- Modify: `studies/pain_study/study1/reporting.py`
- Modify: `studies/pain_study/study2/source_family.py`
- Modify: `studies/pain_study/study2/stages.py`
- Modify: `studies/tests/pipelines/test_study1_reporting_extended.py`
- Modify: `studies/tests/pipelines/test_study2_source_family.py`

- [ ] **Step 1: Write failing tests for centered gain and source analysis tier**

```python
assert metrics["within_subject_centered_delta_r2"] == pytest.approx(expected)
assert summary["analysis_tier"].unique().tolist() == ["feasibility"]
```

- [ ] **Step 2: Compute participant-centered full and nuisance R² gains from held-out arrays**

Record the gain and number of estimable participants; do not abort because a participant has zero
within-subject target variance.

- [ ] **Step 3: Add primary-minus-temporal-control gain columns to reporting**

Match by target, model, and control kind; retain missing values when the control was not run.

- [ ] **Step 4: Add non-blocking source tier labels**

```python
def source_analysis_tier(n_subjects, *, confirmatory_min, feasibility_min):
    if n_subjects >= confirmatory_min: return "confirmatory"
    if n_subjects >= feasibility_min: return "feasibility"
    return "below_feasibility"
```

- [ ] **Step 5: Run reporting and source-family tests**

### Task 8: Correct secondary permutation and artifact summaries

**Files:**
- Modify: `studies/pain_study/study2/behavioral_convergence.py`
- Modify: `studies/pain_study/study2/artifact_controls.py`
- Modify: `studies/pain_study/study2/stages.py`
- Modify: `studies/tests/pipelines/test_study2_behavioral_convergence.py`
- Modify: `studies/tests/pipelines/test_study2_artifact_controls.py`

- [ ] **Step 1: Write failing behavioral ordering and eligible-run tests**

Use deliberately shuffled rows and one-trial runs; require ordered circular shifts and exclude runs
that do not satisfy `admissible_circular_shifts`.

- [ ] **Step 2: Reuse shared circular-shift eligibility and order by trial index**

- [ ] **Step 3: Write a failing incomplete-artifact-family test**

```python
assert qc.artifact_control_criteria_met is False
assert "missing_metric:scanner_residual_power" in qc.unmet_criteria
```

- [ ] **Step 4: Add configured required metric names and return advisory unmet criteria**

Do not raise or block other Study 2 stages.

- [ ] **Step 5: Run behavioral and artifact tests**

### Task 9: Add a scanner-harmonic-safe beta sensitivity preset

**Files:**
- Modify: `studies/pain_study/study1/feature_benchmark.py`
- Modify: `studies/pain_study/study1/config/study1_config.yaml`
- Modify: `studies/pain_study/study1/config/study1_smoketest.yaml`
- Modify: `studies/pain_study/study2/config/study2_config.yaml`
- Modify: `studies/pain_study/study2/config/study2_smoketest.yaml`
- Modify: relevant configuration and feature-selection tests

- [ ] **Step 1: Write failing tests for the sensitivity preset**

Require `beta_clean` to combine configured lower and upper beta intervals while broad `beta` remains
available and no new hard eligibility rule is introduced.

- [ ] **Step 2: Add `beta_low_clean` and `beta_high_clean` intervals around the measured 20.02 Hz peak**

Use explicit YAML boundaries and reuse them in Study 1 features and Study 2 source power.

- [ ] **Step 3: Add `beta_clean` and `alpha_beta_clean_gamma` sensitivity presets**

- [ ] **Step 4: Run feature/config tests**

### Task 10: Documentation, complete verification, and integration

**Files:**
- Modify: `studies/pain_study/study1/README.md`
- Modify: `studies/pain_study/study1/RUN_GUIDE.md`
- Modify: `studies/pain_study/study2/README.md`
- Modify: `studies/pain_study/SCANNER_HARMONICS_QC_README.md`

- [ ] **Step 1: Document corrected permutation, score, point-spread, beta sensitivity, and advisory-tier behavior**

- [ ] **Step 2: Reconcile scanner-audit counts and `run_id` terminology**

- [ ] **Step 3: Run focused Study 1/2 tests**

Run: `pytest studies/tests tests/machine_learning/test_model_comparison_protocol.py tests/pipelines/test_study1_targets.py tests/pipelines/test_study1_feature_benchmark_config.py -q`
Expected: zero failures.

- [ ] **Step 4: Run repository validation**

Run: `make verify-architecture`
Expected: exit 0.

Run: `make verify-maintainability`
Expected: exit 0.

Run: `ruff check` on every modified Python module and test.
Expected: exit 0.

- [ ] **Step 5: Verify no production paths changed**

Run: `git status --short` and inspect the diff. Confirm all modifications are inside the isolated
worktree and no `/Volumes/KINGSTON` outputs were written.
