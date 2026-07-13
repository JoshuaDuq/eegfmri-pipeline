# Study 1 Condition-Summary Support Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build condition and grouped fMRI summaries from their contributing runs, construct A−B
from matched within-run descriptive differences, and validate signature-weight coverage before
fixed-mask scoring.

**Architecture:** Replace parallel effect/variance dictionaries with run-associated summary entries.
Keep the fixed signature scoring mask separate from a new coverage mask: coverage controls validity
and QC, while the fixed mask controls the scored vector and its hash. Trial-level signature scoring
does not pass a coverage mask and remains unchanged.

**Tech Stack:** Python 3.11, NumPy, NiBabel, Nilearn, pytest, Ruff.

---

## File Structure

- Modify `fmri_pipeline/analysis/multivariate_signatures.py`: add coverage-specific support metrics
  and an optional coverage mask without changing fixed scoring extent.
- Modify `fmri_pipeline/analysis/trial_signatures.py`: retain run provenance, build condition/group
  coverage, and construct matched-run A−B summaries.
- Modify `tests/fmri/test_fmri_signature_paths.py`: test coverage denominators, thresholds, zero-fill,
  and fixed scoring identity.
- Modify `tests/fmri/test_fmri_analysis_validity_guards.py`: reproduce the six-run `sub-0012`
  condition-support pattern and test run-matched differences.
- Modify `studies/pain_study/study1/README.md` and `studies/pain_study/study1/RUN_GUIDE.md`: document
  contributing-run coverage and threshold-guarded background fill.

### Task 1: Separate signature coverage QC from fixed scoring

**Files:**
- Modify: `fmri_pipeline/analysis/multivariate_signatures.py`
- Test: `tests/fmri/test_fmri_signature_paths.py`

- [ ] **Step 1: Write failing coverage-metric tests**

Create a four-voxel signature with weights `[1.0, -2.0, 3.0, 4.0]`. Make the fixed
mask cover two positive voxels and the negative voxel while excluding the third positive voxel.
Make the coverage mask retain only one fixed-mask positive voxel. Pass an effect image whose two
unsupported fixed-mask voxels are `NaN`.

Assert that:

```python
result.n_voxels == 3
result.coverage_nonzero_support_fraction == pytest.approx(1.0 / 3.0)
result.coverage_positive_support_fraction == pytest.approx(0.5)
result.coverage_negative_support_fraction == pytest.approx(0.0)
result.coverage_positive_weight_mass_loss_fraction == pytest.approx(0.75)
result.coverage_negative_weight_mass_loss_fraction == pytest.approx(1.0)
result.dot == pytest.approx(effect_on_positive_voxel * positive_weight)
```

The expected nonzero coverage fraction is `1 / 3`, proving the denominator is fixed-mask support
rather than all four original signature weights. Compare the coverage call against a second call
using a manually zero-filled finite effect and no coverage mask. Assert identical dot, cosine,
non-`None` Pearson, `n_voxels`, and `scoring_mask_sha256`; coverage changes QC only, not the scored
vector. The positive mass loss is `1 - 1 / (1 + 3) = 0.75`; the excluded fourth weight is outside
the fixed mask and therefore never enters that denominator.

Add a second test requiring `min_support_fraction=0.90` and
`max_weight_mass_change_fraction=0.10` to raise a coverage-support error.

- [ ] **Step 2: Run the new tests and verify RED**

Run:

```bash
/Users/joduq24/Desktop/EEG_fMRI_Pipeline/.venv/bin/python -m pytest \
  tests/fmri/test_fmri_signature_paths.py \
  -k 'coverage_support' -q
```

Expected: FAIL because `coverage_mask_img` and coverage metrics do not exist.

- [ ] **Step 3: Add explicit coverage metrics**

Add optional fields to `SignatureResult`:

```python
coverage_nonzero_support_fraction: Optional[float] = None
coverage_positive_support_fraction: Optional[float] = None
coverage_negative_support_fraction: Optional[float] = None
coverage_positive_weight_mass_loss_fraction: Optional[float] = None
coverage_negative_weight_mass_loss_fraction: Optional[float] = None
```

Implement a focused helper whose denominator is signature support/mass inside the fixed scoring
mask and whose numerator is additionally intersected with coverage. Define mass loss exactly as:

```python
loss = 1.0 - retained_mass / fixed_mask_mass
```

Return `None` only when the fixed-mask denominator is zero.

- [ ] **Step 4: Add `coverage_mask_img` to signature expression**

Add an optional keyword argument to `compute_signature_expression`. Resample the fixed scoring mask
and coverage mask independently with nearest-neighbor interpolation. Use coverage (or the existing
scoring mask when coverage is absent) to validate/fill non-finite image background before continuous
resampling.

Keep `_signature_support_summary`, `_scoring_mask`, `n_voxels`, and `scoring_mask_sha256` based on
the fixed scoring mask. When coverage is present, calculate separate coverage metrics and apply the
configured support/mass thresholds to them before scoring the zero-filled image on the fixed mask.

- [ ] **Step 5: Run Task 1 tests and verify GREEN**

Run:

```bash
/Users/joduq24/Desktop/EEG_fMRI_Pipeline/.venv/bin/python -m pytest \
  tests/fmri/test_fmri_signature_paths.py -q
```

Expected: all signature-path tests pass; the existing resampling warning may remain.

- [ ] **Step 6: Commit Task 1**

```bash
git add fmri_pipeline/analysis/multivariate_signatures.py \
  tests/fmri/test_fmri_signature_paths.py
git commit -m "fix: separate signature coverage from scoring extent"
```

### Task 2: Retain contributing-run provenance in condition summaries

**Files:**
- Modify: `fmri_pipeline/analysis/trial_signatures.py`
- Test: `tests/fmri/test_fmri_analysis_validity_guards.py`

- [ ] **Step 1: Write a failing six-run condition-support regression**

Build a beta-series test fixture with six runs:

- A occurs in runs 1–6;
- B occurs in runs 1, 2, 5, and 6;
- one voxel is covered only in runs 3 and 4;
- the fixed scoring mask covers every voxel.

Capture condition signature calls and assert their passed coverage masks:

```python
assert coverage_by_map["cond_a"][affected_voxel]
assert not coverage_by_map["cond_b"][affected_voxel]
assert not coverage_by_map["cond_a_minus_b"][affected_voxel]
assert np.isfinite(scoring_image_by_map["cond_b"][affected_voxel])
assert np.isfinite(scoring_image_by_map["cond_a_minus_b"][affected_voxel])
```

Make A values in runs 3/4 extreme and assert the A−B result is unchanged, proving unmatched A-only
runs cannot enter the difference.

- [ ] **Step 2: Run the six-run test and verify RED**

Run the exact new test with pytest. Expected: FAIL because all maps currently receive the all-run
coverage union and the difference includes A-only runs.

- [ ] **Step 3: Introduce run-associated summary entries**

Add a private immutable entry:

```python
@dataclass(frozen=True)
class _RunSummaryEffect:
    run_num: int
    effect_img: Any
    variance_img: Optional[Any]
```

Replace parallel condition/group effect and variance containers with
`Dict[str, List[_RunSummaryEffect]]`. Store masks as `Dict[int, Any]` and fail on duplicate run
numbers. Derive contributing run numbers from entries; resolve every requested run through the mask
mapping and fail on missing or empty coverage.

- [ ] **Step 4: Build condition/group coverage from contributing entries**

For A, B, across-run groups, and per-run groups, pass only masks belonging to the selected entries
to `_prepare_summary_signature_inputs`. Extend its return value with the resulting coverage mask and
pass that mask to `compute_signature_expression(coverage_mask_img=...)` while preserving the fixed
`signature_mask_img` as `mask_img`.

Add all new coverage fields to `_signature_support_fields` so TSV outputs record the QC values.

- [ ] **Step 5: Build the descriptive A−B map from matched runs**

Intersect A and B run-number sets. Raise if both conditions exist but the intersection is empty.
For each matched run:

1. combine that run's A entries with configured weighting;
2. combine that run's B entries with configured weighting;
3. subtract B from A;
4. resample that run's brain mask to the difference grid;
5. reject any non-finite difference inside the run mask;
6. set voxels outside the run mask to `NaN`.

Combine run-wise differences with `method="mean"`, which uses `np.nanmean` voxelwise. Use the union
of matched-run masks as A−B coverage. Do not synthesize a difference variance. Write A−B signature
rows with `map_inference="descriptive_trial_summary"` even when beta-series A/B summaries retain
their existing `run_level_fixed_effects` label.

- [ ] **Step 6: Add an in-coverage invalid-value regression**

Change one B-contributing matched run so its combined B effect remains non-finite at a voxel with
finite, positive variance and in-mask coverage while another matched run is finite there. Assert
the per-run difference validation raises before `np.nanmean`, proving another run cannot hide the
invalid value.

- [ ] **Step 7: Add grouped-summary and provenance boundary regressions**

Add one across-run grouped fixture where a group occurs in only a subset of runs and assert only
those masks form coverage. Add one per-run grouped fixture with differing masks and assert each
summary receives only its own run mask.

Add focused failures for duplicate discovered run numbers, a summary entry referencing an unknown
run mask, and an empty contributing-run request. These must fail at the provenance boundary rather
than fall back to all-run coverage.

- [ ] **Step 8: Run Task 2 tests and verify GREEN**

Run:

```bash
/Users/joduq24/Desktop/EEG_fMRI_Pipeline/.venv/bin/python -m pytest \
  tests/fmri/test_fmri_analysis_validity_guards.py \
  tests/fmri/test_fmri_signature_paths.py -q
```

Expected: all tests pass.

- [ ] **Step 9: Commit Task 2**

```bash
git add fmri_pipeline/analysis/trial_signatures.py \
  tests/fmri/test_fmri_analysis_validity_guards.py
git commit -m "fix: scope condition summaries to contributing runs"
```

### Task 3: Document and verify the scientific boundary

**Files:**
- Modify: `studies/pain_study/study1/README.md`
- Modify: `studies/pain_study/study1/RUN_GUIDE.md`

- [ ] **Step 1: Update Study 1 documentation**

Document condition-specific contributing-run unions, equal-weight matched-run descriptive A−B,
coverage-specific signature support/mass QC, and the distinction between unsupported background
and invalid in-coverage values. State that trial-level targets are unchanged.

- [ ] **Step 2: Run final verification**

Run:

```bash
/Users/joduq24/Desktop/EEG_fMRI_Pipeline/.venv/bin/python -m pytest \
  tests/fmri/test_fmri_analysis_validity_guards.py \
  tests/fmri/test_fmri_signature_paths.py \
  tests/pipelines/test_study1_targets.py \
  studies/tests/fmri/test_study1_targets.py -q
/Users/joduq24/Desktop/EEG_fMRI_Pipeline/.venv/bin/ruff check \
  fmri_pipeline/analysis/multivariate_signatures.py \
  fmri_pipeline/analysis/trial_signatures.py \
  tests/fmri/test_fmri_signature_paths.py \
  tests/fmri/test_fmri_analysis_validity_guards.py
make PYTHON=/Users/joduq24/Desktop/EEG_fMRI_Pipeline/.venv/bin/python verify-architecture
git diff --check
```

Expected: all commands pass, with only the pre-existing resampling warning permitted.

- [ ] **Step 3: Commit documentation**

```bash
git add studies/pain_study/study1/README.md \
  studies/pain_study/study1/RUN_GUIDE.md
git commit -m "docs: define condition-summary spatial support"
```
