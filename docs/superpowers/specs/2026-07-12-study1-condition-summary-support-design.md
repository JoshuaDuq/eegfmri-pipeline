# Study 1 Condition-Summary Support Repair

## Objective

Make condition-level and grouped fMRI signature summaries use the spatial coverage of the runs
that actually contributed effects. Preserve fixed a-priori signature scoring while distinguishing
unsupported voxels from invalid values inside true contributing coverage.

## Scope

This repair changes descriptive condition/group summary maps and their signature-expression QC.
It does not change trial selection, trial LSS models, trial beta maps, or the trial-level signature
targets used by the primary Study 1 analysis.

## Run Provenance and Coverage

Store each run brain mask by run number. Existing effect containers already retain condition/group
effects by run; derive contributing run sets from those containers rather than from all discovered
runs.

- Condition A coverage is the union of masks for runs contributing A effects.
- Condition B coverage is the union of masks for runs contributing B effects.
- Across-run group coverage is the union of masks for runs contributing that group.
- Per-run group coverage is that run's mask.

Missing run masks, empty contributing-run sets, and references to unknown runs raise explicit
errors. Coverage must never be inferred from finite summary-map voxels.

## Descriptive A−B Map

The difference map uses only runs containing both A and B. Within each matched run, combine that
run's A trial effects and B trial effects using the configured condition-summary weighting, then
subtract B from A. Average the resulting run-wise differences equally across matched runs.

Equal run weighting is intentional. LSS condition estimates come from separate trial models, so
their covariance is unavailable and an inverse-variance estimate for A−B would claim unsupported
precision. The output remains labeled `descriptive_trial_summary`; it is not an inferential
contrast. If A and B exist but no run contains both, fail because the within-run condition
difference is undefined.

The A−B spatial coverage is the union of the matched-run masks. Each run-wise difference is valid
only on its own run grid. Before averaging, set every voxel outside that run's brain mask to `NaN`;
the across-run descriptive mean is then computed voxelwise from only the matched runs that actually
support that voxel. Zero/background values outside a run mask must never enter the mean.

## Fixed-Mask Signature Scoring and Coverage QC

Extend signature expression with an optional coverage mask distinct from the fixed scoring mask:

- the coverage mask defines where non-finite summary values are invalid;
- non-finite values outside coverage are converted to zero only to permit continuous resampling and
  fixed-mask dot-product scoring;
- the fixed a-priori scoring mask continues to define the scored vector and scoring-mask hash;
- existing signature support metrics retain their current meaning relative to the original
  signature and fixed scoring mask;
- separate coverage support fractions use signature support inside the fixed scoring mask as the
  denominator and support inside fixed-mask ∩ condition/group coverage as the numerator;
- separate positive/negative coverage weight-mass loss uses absolute signature weight mass inside
  the fixed scoring mask as the denominator and retained mass inside fixed-mask ∩ coverage as the
  numerator;
- existing configured support and weight-mass thresholds fail fast when missing coverage is too
  influential.

This makes zero filling an explicit, threshold-guarded representation of unsupported background,
not an assertion that the unobserved effect equals zero. The fixed scored vector still includes
zero-filled unsupported voxels for dot, cosine, and Pearson outputs; `n_voxels` and
`scoring_mask_sha256` therefore remain independent of subject/condition coverage. Coverage metrics
are recorded separately in signature result tables.

## Testing

Use red-green-refactor with a six-run regression matching `sub-0012`:

- A effects occur in all six runs;
- B effects occur only in runs 1, 2, 5, and 6;
- one voxel is covered only by runs 3 and 4;
- condition A retains that voxel;
- condition B and A−B treat it as unsupported background rather than an invalid in-coverage value;
- a non-finite value inside a B-contributing run remains an error.

Add focused tests for matched-run A−B construction, group-specific/per-run coverage, and
coverage-aware signature support denominators and thresholds. Retain the fixed scoring-mask hash,
fixed scored voxel count, and exact float32 fixed-effects regressions.

## Success Criteria

- The `sub-0012` support pattern no longer fails merely because A-only runs cover extra voxels.
- Runs without a condition never expand that condition's coverage or enter its summary.
- A−B contains no between-run condition mismatch.
- Missing signature-weight support is quantified and thresholded before zero-filled scoring.
- Genuine non-finite values inside contributing coverage still fail immediately.
- Trial-level Study 1 targets and fixed scoring-mask identity are unchanged.
