# Study 2 Primary Cortical Source Associations: Design Specification

Date: 2026-07-11

## Purpose

Create the primary Study 2 cortical figure showing where source-resolved EEG power is
associated with the frozen, held-out Study 1 NPS prediction score. The figure must answer
one prespecified spatial question across the three primary frequency bands: alpha, beta,
and scanner-clean gamma.

This is a construct-localization figure, not a new predictive analysis. It must display the
unthresholded cohort effect while using corrected cluster contours only as an inferential
overlay. It must not introduce post-hoc regions of interest, threshold-only maps, or
uncorrected significance markers.

## Scientific Claim and Interpretation Boundary

The supported claim is:

> Across participants, cortical source power covaries with the held-out EEG prediction of
> NPS response in spatially organized patterns, with family-corrected evidence identified
> by target-retrained cluster inference.

The figure does not establish a unique neural generator, causal mediation, or anatomical
specificity beyond the spatial resolution and assumptions of the configured inverse
solution. The caption must identify the cortical maps as source estimates and state that
deep generators are not evaluated.

## Prespecified Analysis

### Bands and participant cohort

The figure contains exactly the configured primary bands:

1. alpha;
2. beta;
3. scanner-clean gamma.

The valid participant identifiers and their order must be recovered from the source-stage
QC tables. All bands must contain the same valid participant cohort in the same order.
Different band-specific cohorts are an error because they would make the visual comparison
and family correction ambiguous. The number of maps must equal the number of valid QC rows
and must meet `source_stage.min_source_valid_subjects`.

### Vertex-level effect

For participant \(i\), band \(b\), and vertex \(v\), the existing source-stage estimate is
the within-participant partial correlation

\[
r_{ibv} = \operatorname{cor}(P_{ibv}, \widehat{NPS}_{i} \mid C_i),
\]

where source power and the frozen held-out Study 1 prediction score are residualized using
the prespecified nuisance covariates \(C_i\). No association is recomputed in the figure
code.

The displayed cohort effect is the equal-participant Fisher mean back-transformed to the
correlation scale:

\[
\bar r_{bv} = \tanh\left(\frac{1}{n}\sum_i \operatorname{arctanh}(r_{ibv})\right).
\]

This quantity is computed from the saved Fisher-z maps and cross-checked against the saved
partial-r maps. The display is not the arithmetic mean of participant correlations.

### Corrected inference

Inference must use the existing Study 2 source-family implementation without a competing
statistical path:

- group statistics are computed from participant Fisher-z maps;
- the cluster-forming threshold is two-sided and determined by the configured
  `source_inference.primary_cluster_forming_p`;
- spatial family-wise error is controlled by the target-retrained maximum-cluster null;
- the minimum corrected cluster probability in each band is Holm corrected across the
  three primary bands at `source_inference.family_alpha`.

The figure reader must deterministically recompute `compute_source_family_inference` from
the saved Fisher-z maps, target-retrained null maps, and adjacency matrix, then verify its
band-level results against the saved source-family summary. A mismatch is an error.

A cluster receives a contour only when both conditions hold:

1. its maximum-cluster probability is at or below the configured family alpha; and
2. its band's Holm-adjusted minimum-cluster probability is at or below family alpha.

The cortical fill always shows the unthresholded effect. Non-significant vertices remain
visible and are never set to zero or made transparent.

## Required Artifacts and Validation

The reader requires, for every primary band:

- `source_stage/partial_r_<band>.npy`;
- `source_stage/fisher_z_<band>.npy`;
- `source_stage/qc_<band>.tsv`;
- `inference/null_<band>.npy`;
- `inference/source_family_summary.tsv`;
- `inference/adjacency.npy`.

The saved NumPy maps must be finite, two-dimensional participant-by-vertex arrays. Partial-r
values must lie strictly within the Fisher transform domain. Partial-r and Fisher-z shapes
must match, and `arctanh(partial_r)` must agree numerically with the saved Fisher-z values.
Target-retrained null arrays and the adjacency matrix must satisfy the existing inference
contracts and match the observed vertex count.

### Stable cortical vertex identity

The current array files contain vertex values but do not preserve the corresponding
fsaverage vertex numbers. The source stage must therefore write one explicit common-space
vertex manifest:

- `source_model/common_source_vertices.npz`, containing one-dimensional integer arrays
  `lh_vertices` and `rh_vertices`;
- `source_model/common_source_vertices.json`, containing the common subject, source-space
  spacing, hemisphere counts, total count, and provenance.

The manifest is created from the actual morphed source estimate or common source space used
by the source stage. It must never be reconstructed by splitting an array in half or by
assuming contiguous vertex numbers. Subsequent participants must match the first manifest
exactly. The figure must fail if the manifest is absent, inconsistent with configured
`fsaverage`/`oct6`, or inconsistent with the map and adjacency dimensions.

## Figure Design

### Layout

The deliverable is one exact-size vector figure, 183 mm wide, organized as three columns:
alpha, beta, and scanner-clean gamma. Each column contains four aligned cortical views:

- left lateral;
- left medial;
- right medial;
- right lateral.

The views use the configured fsaverage inflated surface with sulcal curvature providing a
quiet anatomical background. Camera, surface, lighting, and crop are identical across
bands. Medial-wall vertices are visually distinct from valid mapped cortex.

### Encoding

All bands share one symmetric diverging color scale centered exactly at zero. The limits
are the maximum absolute displayed cohort effect across all three maps, so no effect value
is clipped. The palette must be perceptually balanced and color-vision-accessible, with
cool colors for negative and warm colors for positive associations.

One shared horizontal color bar is labeled:

> Fisher mean partial correlation, r

Family-corrected clusters are outlined using a thin charcoal contour that remains legible
on both sides of the diverging scale. Contours encode corrected support only; contour color
or width does not encode probability or cluster mass.

Each band header reports the valid participant count and one compact inferential status:

- `Holm q = ...` when at least one family-corrected cluster is present; or
- `no family-corrected cluster` otherwise.

Cluster-level probabilities, signs, masses, sizes, and vertex identities belong in the
machine-readable audit rather than crowded onto the cortex. There are no stars, opaque
legends over data, ROI callouts, or decorative brain icons.

### Caption contract

The generated caption text must define:

- the displayed Fisher-mean partial correlation;
- the within-participant residualization and frozen held-out prediction target;
- the two-sided cluster-forming threshold;
- target-retrained maximum-cluster correction;
- Holm correction across the three primary bands;
- the meaning of the dark contours;
- the cortical-source interpretation boundary.

If the participant count is below the configured publication threshold, writing the figure
is an error rather than producing a preliminary publication artifact.

## Output Contract

One writer script produces the complete figure family:

- `figures/primary_source_associations.svg`;
- `figures/primary_source_associations.png` for visual QA;
- `figures/primary_source_associations_vertices.tsv` with band, hemisphere, source-space
  vertex number, displayed effect, group statistic, cluster identifier, and corrected
  contour membership;
- `figures/primary_source_associations_clusters.tsv` with band, cluster sign, size, mass,
  maximum-cluster probability, band Holm q, and contour status;
- `figures/primary_source_associations_summary.tsv` with participant count, vertex count,
  display range, inference settings, and band-level corrected result;
- `figures/primary_source_associations_caption.txt`;
- `figures/primary_source_associations_manifest.json` with input paths, hashes, configuration,
  dimensions, software versions, and output paths.

Tabular outputs use one row per declared observational unit and stable column order. Exact
probabilities are retained in the audits; display formatting must not alter stored values.

## Code Organization

The implementation follows the existing Study 2 figure architecture:

- `primary_source_associations.py`: immutable data structures, artifact reading, validation,
  effect computation, inference reconciliation, and audit construction;
- `primary_source_associations_plot.py`: deterministic cortical rendering only;
- `plot_primary_source_associations.py`: configuration loading, orchestration, and atomic
  output writing;
- `source_vertex_manifest.py`: creation and validation of the shared source-space vertex
  identity artifact;
- explicit path helpers in `study2/paths.py` for all new inputs and outputs.

Scientific computations stay independent of Matplotlib and surface rendering. The writer
accepts no alternate estimand, band list, correction method, or threshold flag.

## Verification

Tests must cover:

- exact Fisher-mean back-transformation;
- partial-r/Fisher-z consistency checks;
- identical valid-participant cohorts across bands;
- vertex-manifest identity and dimension validation;
- reconciliation with the saved source-family summary;
- the joint cluster and Holm contour rule;
- shared symmetric unclipped color limits;
- exact band order, four-view layout, labels, and figure dimensions;
- complete audit and manifest schemas;
- fail-fast behavior for missing, stale, non-finite, or inconsistent artifacts;
- deterministic SVG and tabular output structure.

A synthetic end-to-end fixture must exercise both a corrected cluster and a band without a
corrected cluster. Final verification includes focused tests, the full Study 2 test suite,
repository lint and architecture gates, and visual inspection of the SVG and PNG. A real
data preflight must stop before writing any output when required upstream artifacts are not
yet available.
