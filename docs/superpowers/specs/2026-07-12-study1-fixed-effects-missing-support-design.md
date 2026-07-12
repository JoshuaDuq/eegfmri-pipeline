# Study 1 Fixed-Effects Missing-Support Repair

## Objective

Prevent unsupported LSS effect maps from contaminating an otherwise estimable Study 1 condition
summary while preserving hard failures for non-finite values that carry statistical weight.

## Root Cause

`_combine_effect_images()` converts non-finite inverse-variance weights to zero, then evaluates the
weighted numerator as `weight * effect`. IEEE arithmetic defines `0 * NaN` as `NaN`, so an
unsupported effect map can contaminate the numerator even when its weight is exactly zero. The
resulting non-finite summary lies inside the union of run coverage whenever another run supports
that voxel, causing the downstream validity guard to reject a summary that should be estimable.

## Design

Keep inverse-variance fixed-effects estimation and its existing denominator unchanged. Build the
weighted contributions in a zero-initialized array and multiply only where the sanitized weight is
nonzero. A zero-weight effect then contributes exactly zero to both numerator and denominator.

For voxel `v`, the estimator remains:

```text
beta(v) = sum_i(w_i(v) * beta_i(v)) / sum_i(w_i(v))
```

where non-finite inverse-variance weights are already sanitized to zero. The implementation must
not replace non-finite effects carrying a nonzero weight: those values must continue to propagate
to the condition summary and be rejected by the existing run-coverage validity guard. Voxels with
no positive total weight remain `NaN`, preserving the existing missing-support representation.

The fixed a-priori signature scoring mask and the existing preparation of non-finite background
outside the run-coverage union remain unchanged. The repair does not shrink masks, impute supported
data, or add fallback behavior.

## Testing

Add a regression case with two aligned inputs at one voxel:

- one finite effect with finite variance and therefore nonzero weight;
- one `NaN` effect with infinite variance and therefore zero weight.

The combined value must equal the supported effect. The test must fail against the current
implementation because `0 * NaN` contaminates the numerator.

Retain the existing assertion that a voxel with no weighted support remains `NaN`. Add or preserve
coverage showing that a non-finite effect with nonzero weight is not silently discarded and reaches
the existing validity error path.

## Verification

Run the focused fixed-effects and condition-summary validity tests, the complete fMRI validity-guard
module, Ruff on the modified Python files, and the repository architecture check. Existing
uncommitted changes in `trial_signatures.py` and its validity-guard tests must be preserved.

## Success Criteria

- Mixed run support produces the mathematically defined inverse-variance estimate.
- Zero-weight non-finite effects cannot contaminate supported voxels.
- True non-finite supported values still fail fast.
- Voxels without any weighted support remain non-finite background.
- No scoring-mask, target-definition, or output-schema behavior changes.
