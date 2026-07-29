# fMRI subject report: scientific robustness

Date: 2026-07-28

## Problem

Three classes of defect, all verified against a real derivatives tree
(`outputs/fmri_report_verification/`) and the fitting path.

### 1. Rigorous figures that never render

The manifest a real fit writes records `effect_map: null`, `variance_map: null`,
`design_matrices: []`, `contrast_vector: null`. Consequently the dual-coded panel, the
standard-error panel, and the entire design section (matrix, contrast strip, regressor
correlation, VIF, condition number, contrast efficiency) are unreachable in production.

The effect and variance maps are already computed at
`pipelines/fmri_analysis.py:466` and discarded. The design-matrix TSVs are already
written to `contrast-*/qc/` and never recorded.

### 2. Statements that are false

- The manifest's `mask` is a *discovered* single-run fMRIPrep brain mask, while the GLM
  fits inside `_build_intersection_brain_mask` (`intersect_masks(threshold=1.0)`,
  `contrast_builder.py:1404`). The coverage panel therefore prints "intersection across
  N run(s)" over run-01's mask, and every colour limit is computed in the wrong voxel
  set.
- `clipped_fraction` on a thresholded panel is taken over all mask voxels, describing a
  population the panel does not draw.
- `_masked_values` reports "no mask supplied" when a mask *was* supplied but mismatched
  in shape.
- The z histogram is fed `data[np.isfinite(data)]` — the whole volume. On the real map
  62.9% of that is exact background zeros, so the panel is a spike at zero with its
  N(0,1) curve scaled to 136,416 voxels when only 50,626 are brain.

### 3. Missing rigor

- No anatomical background on any volume panel, though `PlotAssets.background` is
  discovered. Clusters cannot be judged against grey matter or ventricle.
- `estimate_fwhm` is implemented, validated and tested, and never called, so
  `cluster_min_voxels` is reported as a bare count.
- Nothing states what the applied height threshold is worth. Measured on the real map:

  | quantity | value |
  |---|---|
  | in-mask voxels | 50,626 |
  | empirical null | mu = -0.609, sigma = 1.508 |
  | applied \|z\| > 2.30 | 8,463 survive (1,086 expected under N(0,1)) |
  | FDR q = 0.05 | \|z\| > 2.85, 4,469 survive |
  | Bonferroni 0.05 | \|z\| > 4.89, 368 survive |

- `threshold_mode: "fdr"` is a supported, validated config that yields a contrast
  section containing no panels at all.

## Design

### A. Fitting path records what was fit

`pipelines/fmri_analysis.py` saves, beside each contrast:

- the effect and variance maps in the analysis space, from the already-fitted model;
- `glm_result.mask_img`, the intersection mask the GLM actually used, resampled
  alongside the contrast when `resample_to_freesurfer` is on.

`write_report_manifest` additionally records `design_matrices` (from
`run_meta["design_matrix_tsv_paths"]`) and the numeric contrast vector and columns,
expanded from the contrast expression with `nilearn.glm.contrasts.
expression_to_contrast_vector`.

Effect and variance computation moves out from under `if plotting.enabled` and is gated
on `include_effect_size` / `include_standard_error`. The report is a separate step and
cannot ask for these retroactively.

### B. Render-path corrections

- Thread `PlotAssets.background` into every volume panel.
- Compute the clipped fraction over the voxels a panel draws.
- Say which mask was used, and say so truthfully when one was supplied and rejected.
- Resolve a height threshold for every `threshold_mode`, so `fdr` and `none` produce a
  populated section.

### C. Inference calibration

New `analysis/report/inference.py`, computing from a saved map inside the analysis mask
only — no model, so the report package's no-fitting invariant holds:

- `empirical_null` — median and IQR/1.349 (Efron central matching, robust form);
- `fdr_threshold` — Benjamini-Hochberg over in-mask p-values;
- `bonferroni_threshold`;
- `expected_false_positives`.

`null_calibration_figure` replaces `z_histogram`: the in-mask z distribution against
both the theoretical N(0,1) and the fitted empirical null, with the applied, FDR and
Bonferroni thresholds drawn as labelled reference lines carrying their surviving-voxel
counts. Both null curves are scaled assuming every voxel is null; the figure states
this, because the diagnostic is the *width* of the null, not its height.

Promoted out of collapsed diagnostics into the contrast section: it is no longer a
diagnostic but the panel that says what the threshold beside it is worth.

`estimate_fwhm` is called on the stat map inside the analysis mask, and
`cluster_min_voxels` is reported in mm^3 and resels alongside the raw count.

### D. Removals

- `z_histogram`, replaced by `null_calibration_figure`.
- Coverage's "N of M voxels in the field of view": M counts air, so the ratio is
  meaningless. Replaced by mask volume in mm^3, drawn over anatomy, with smoothness.

Not in scope, flagged separately: `analysis/reporting.py` (1,523 lines) is dead in
production — `tests/fmri/report/test_report_entry_point.py:79` asserts the pipeline no
longer calls it. It is a second, less rigorous rendering of the same figures that can
drift from the real one.

## No verdicts

Every number added here is a measurement. The report states the applied threshold beside
the corrected ones and the survivor counts, and leaves the judgement to the reader. No
cutoff invented by this pipeline appears anywhere.
