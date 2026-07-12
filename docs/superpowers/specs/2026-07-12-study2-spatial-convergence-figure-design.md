# Study 2 Multimodal Spatial-Convergence Figure

**Date:** 2026-07-12

## Goal

Create a publication-ready Study 2 figure that closes the multimodal narrative between the
Study 1 fMRI target and the Study 2 EEG source maps. The figure must show the spatial maps and the
spatial-autocorrelation-preserving null inference without recomputing a different scientific
estimand or accepting incomplete artifacts.

## Scientific Scope

The figure consumes the existing `spatial` stage outputs:

- the common cortical analysis mask;
- the alpha, beta, and scanner-clean gamma EEG association maps;
- the resolution-matched NPS-Level-2 fMRI forward covariance map;
- 5,000 BrainSMASH surrogate maps per band;
- the saved spatial-correspondence summary; and
- the common-source vertex manifest and fsaverage surface used by the source analysis.

The spatial-surrogate stage metadata binds the arrays to that source space by recording the exact
vertex-manifest SHA256, configured common subject and spacing, analysis-mask SHA256, and per-band
fMRI-map SHA256. Equal array length alone is not accepted as evidence of vertex identity.

The three band-specific fMRI artifacts must be numerically identical because the prespecified
target is one fMRI reference map smoothed to the cohort-median EEG point-spread. A mismatch is an
upstream scientific inconsistency and must fail before rendering.

Motion/physiological covariance maps and a smoothing-kernel sensitivity sweep are not currently
produced by a validated pipeline stage. They are therefore outside this implementation rather than
being fabricated, treated as optional, or silently omitted after a failed load.

## Figure Design

The exact 183 x 112 mm figure has two rows.

### Panel a: Resolution-matched cortical patterns

Four columns show the fMRI NPS-Level-2 reference followed by alpha, beta, and scanner-clean gamma
EEG maps. Each column contains left- and right-lateral cortical views on the exact common source
surface. The fMRI map has its own symmetric blue-neutral-orange scale because its units differ from
the EEG partial-correlation maps. The three EEG maps share one symmetric scale.

All masked values are displayed without thresholding. Vertices outside the prespecified cortical
analysis mask are neutral and are not used to determine display limits.

### Panel b: Spatial-null inference

One row per EEG band shows the complete distribution of BrainSMASH surrogate correlations as a
light density histogram. A prominent band-colored diamond marks the observed Pearson spatial
correlation. The row reports the observed `r`, the plus-one two-sided permutation p-value, and the
Holm-adjusted p-value. A dark outline marks only rows that survive the configured Holm family alpha.

The null distribution, rather than a conventional vertexwise regression interval, carries the
inference because cortical vertices are spatially autocorrelated. A concise in-figure note and the
caption state this explicitly.

## Data Contract

`SpatialConvergenceSummary` owns validated arrays and a tidy audit table. Loading must:

1. require exactly the configured alpha, beta, and gamma bands;
2. load the common-source vertex manifest and verify configured space identity;
3. require one-dimensional finite EEG/fMRI maps with the manifest vertex count;
4. require one finite surrogate matrix per band with the configured draw count;
5. require a stored mask with exact NumPy boolean dtype, a non-empty inclusion set, and at least
   two included vertices before any conversion;
6. verify the spatial metadata's subject, spacing, manifest hash, mask hash, and map hashes;
7. recompute each spatial correlation through the existing
   `compute_spatial_correspondence()` function;
8. verify the saved summary schema, band order, observed r, raw p, meaningful flag,
   Holm-adjusted p, and Holm-significant flag against the recomputed results; and
9. require the three fMRI reference maps to be identical with `rtol=0` and `atol=1e-12`.

No legacy column aliases, inferred vertex layouts, optional artifacts, or compatibility paths are
permitted.

## Output Family

The writer produces atomically where applicable:

- `spatial_convergence.svg` with editable text and 600-dpi embedded image layers;
- `spatial_convergence.png` at configured publication resolution;
- `spatial_convergence_summary.tsv` with one validated row per band;
- `spatial_convergence_caption.txt`; and
- `spatial_convergence_manifest.json` with figure dimensions, analysis parameters, software
  versions, and SHA256 hashes of all source and output artifacts.

The writer validates every scientific input and builds the figure before creating the output
directory.

## Verification

Tests use deterministic synthetic surfaces and artifacts to verify:

- strict artifact validation and saved-summary reconciliation;
- exact band and vertex identity;
- shared EEG scale and separate fMRI scale;
- unthresholded surface layers and analysis-mask handling;
- complete null distributions, observed markers, and adjusted-p annotations;
- exact physical dimensions and editable SVG text;
- the complete output family and checksum manifest;
- failure before output on missing or inconsistent artifacts; and
- deterministic rendering.

After automated tests, the synthetic SVG is rasterized and inspected at final aspect ratio for
clipping, hierarchy, color balance, and legibility.
