# Study 1 Fixed-Effects Missing-Support Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent zero-weight non-finite LSS effects from contaminating estimable Study 1 condition summaries.

**Architecture:** Keep the existing inverse-variance estimator and validity boundaries. Change only
the weighted-numerator calculation so multiplication occurs for nonzero weights; supported
non-finite effects continue to propagate and unsupported voxels remain `NaN` when total weight is
zero.

**Tech Stack:** Python 3.11, NumPy, NiBabel, pytest, Ruff.

---

## File Structure

- Modify `tests/fmri/test_fmri_analysis_validity_guards.py`: reproduce mixed valid and unsupported
  voxel support in the existing fixed-effects regression test.
- Modify `fmri_pipeline/analysis/trial_signatures.py`: calculate weighted effects without evaluating
  zero-weight products.
- Preserve the existing uncommitted edits in both files; they implement the run-coverage validity
  boundary that exposed this arithmetic defect.

### Task 1: Reproduce and repair zero-weight contamination

**Files:**
- Modify: `tests/fmri/test_fmri_analysis_validity_guards.py:41`
- Modify: `tests/fmri/test_fmri_analysis_validity_guards.py:709`
- Modify: `fmri_pipeline/analysis/trial_signatures.py:1038`

- [ ] **Step 1: Write the failing mixed-support regression test**

Change the second effect in
`test_combine_effect_images_preserves_missing_support_as_nan` to an unsupported value while keeping
its infinite variance:

```python
effect_b = nib.Nifti1Image(np.array([[[np.nan]]], dtype=np.float32), np.eye(4))
```

Keep the expected combined value at `1.0`, supplied by the first finite effect with variance `1.0`.

- [ ] **Step 2: Add a supported non-finite characterization test**

Import `_prepare_summary_signature_inputs` from `trial_signatures` and add a test that passes a
`NaN` effect with finite variance through the fixed-effects combiner before checking run coverage:

```python
def test_summary_signature_inputs_reject_nonfinite_weighted_effect_inside_coverage() -> None:
    nib = pytest.importorskip("nibabel")
    effect = nib.Nifti1Image(np.array([[[np.nan]]], dtype=np.float32), np.eye(4))
    variance = nib.Nifti1Image(np.array([[[1.0]]], dtype=np.float32), np.eye(4))
    coverage = nib.Nifti1Image(np.ones((1, 1, 1), dtype=np.uint8), np.eye(4))
    combined = _combine_effect_images(
        effects=[effect],
        variances=[variance],
        method="variance",
    )

    with pytest.raises(ValueError, match="inside the analysis mask"):
        _prepare_summary_signature_inputs(
            summary_img=combined,
            signature_mask_img=coverage,
            run_brain_masks=[coverage],
            summary_name="Condition-level",
        )
```

This characterization must pass before and after the repair. It prevents the multiplication mask
from being broadened to `np.isfinite(eff)`, which would silently discard an invalid supported
effect.

- [ ] **Step 3: Run the focused tests and verify RED**

Run:

```bash
.venv/bin/python -m pytest \
  tests/fmri/test_fmri_analysis_validity_guards.py::test_combine_effect_images_preserves_missing_support_as_nan \
  tests/fmri/test_fmri_analysis_validity_guards.py::test_summary_signature_inputs_reject_nonfinite_weighted_effect_inside_coverage \
  -q
```

Expected: one failure and one pass. The mixed-support test fails because its combined value is
`NaN`; the supported non-finite characterization passes through the existing validity error. The
failure must demonstrate `0 * NaN` contamination rather than a fixture or import error.

- [ ] **Step 4: Implement the minimal weighted-product repair**

In `_combine_effect_images()`, preserve weight sanitization and denominator calculation. Replace
the unconditional product with a zero-initialized contribution array and conditional NumPy
multiplication:

```python
weighted_effects = np.zeros_like(eff, dtype=float)
np.multiply(w, eff, out=weighted_effects, where=w != 0.0)
num = np.sum(weighted_effects, axis=0)
```

Do not add finite-value filtering to the multiplication condition. A non-finite effect with a
nonzero weight must remain non-finite and reach the existing validity guard.

- [ ] **Step 5: Run the focused fixed-effects and condition-summary tests and verify GREEN**

Run:

```bash
.venv/bin/python -m pytest \
  tests/fmri/test_fmri_analysis_validity_guards.py::test_combine_effect_images_preserves_missing_support_as_nan \
  tests/fmri/test_fmri_analysis_validity_guards.py::test_summary_signature_inputs_reject_nonfinite_weighted_effect_inside_coverage \
  tests/fmri/test_fmri_analysis_validity_guards.py::test_trial_signature_condition_signatures_zero_background_outside_run_coverage \
  tests/fmri/test_fmri_signature_paths.py::test_compute_signature_expression_rejects_nonfinite_values_inside_fixed_mask \
  -q
```

Expected: 4 passed. This confirms supported mixed voxels combine, a weighted non-finite effect
reaches the run-coverage error, non-finite background outside run coverage is prepared for
resampling, and true non-finite values inside the scoring mask still fail.

### Task 2: Verify the repair in repository context

**Files:**
- Verify: `fmri_pipeline/analysis/trial_signatures.py`
- Verify: `tests/fmri/test_fmri_analysis_validity_guards.py`
- Verify: `tests/fmri/test_fmri_signature_paths.py`

- [ ] **Step 1: Run the complete fMRI validity-guard modules**

Run:

```bash
.venv/bin/python -m pytest \
  tests/fmri/test_fmri_analysis_validity_guards.py \
  tests/fmri/test_fmri_signature_paths.py \
  -q
```

Expected: all tests pass without warnings attributable to the repair.

- [ ] **Step 2: Run Ruff on the modified Python files**

Run:

```bash
.venv/bin/ruff check \
  fmri_pipeline/analysis/trial_signatures.py \
  tests/fmri/test_fmri_analysis_validity_guards.py
```

Expected: success with no diagnostics.

- [ ] **Step 3: Run the architecture gate**

Run:

```bash
make verify-architecture
```

Expected: success.

- [ ] **Step 4: Review the final diff**

Run:

```bash
git diff --check
git diff -- fmri_pipeline/analysis/trial_signatures.py \
  tests/fmri/test_fmri_analysis_validity_guards.py
```

Expected: no whitespace errors; the final diff preserves the pre-existing run-coverage edits and
adds only the regression fixture and conditional weighted multiplication required by this repair.

Do not stage or commit the overlapping code/test files: they contained user-owned uncommitted work
before this repair began.
