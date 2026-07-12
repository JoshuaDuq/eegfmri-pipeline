# Study 1 and Study 2 Figure Publication Hardening

**Date:** 2026-07-12

## Goal

Harden the existing Study 1 and Study 2 figure suite for high-impact-journal artwork
requirements without changing any scientific estimand, cohort, model, threshold, confidence
interval, or inferential result.

The work applies to the current standalone supplementary validity figures, the Study 1
double-column prediction and construct-validity figures, and the Study 2 sensor- and
source-pattern figures.

## Audit Findings

The current suite already has a strong scientific grammar:

- participant-level observations are shown rather than hidden behind summaries;
- cohort estimates and uncertainty intervals are explicit;
- repeated-measures structure is retained where relevant;
- comparable panels share scales, while effects with different units use separate scales;
- colors are target-consistent and avoid red-green contrasts;
- physical widths are fixed at 89 mm or 183 mm with editable Arial text;
- preliminary cohorts are labeled rather than presented as article-ready; and
- inferential contours are separated from unthresholded effect maps.

The audit identified four publication-facing weaknesses.

1. Matplotlib SVG output embeds the Study 1 fMRI maps, Study 2 scalp topographies, and
   continuous color bars at 100 dpi. The surrounding axes and text are vector objects, but
   the scientific image layers are below common final-artwork requirements.
2. The standalone Study 1 dose-response and behavioral-validity legends sit inside the data
   region and can cover observations in a full cohort.
3. The Study 2 Haufe figure uses a different diverging palette from the cortical figure,
   applies unexplained bold emphasis to only the first two bands, and does not define its
   fold-pair points and median diamonds in the figure.
4. The Study 2 source figure calls Holm-adjusted p-values `q` values. That terminology is
   easily confused with false-discovery-rate q-values. It also reports adjusted probability
   only for significant bands, reducing inferential transparency.

The repository does not contain real Study 1 or Study 2 article outputs. Visual assessment
therefore uses the deterministic synthetic figure fixtures. Real-data regeneration remains a
required final check on the compute environment that owns the study derivatives.

## Considered Approaches

### 1. Minimal cosmetic polish

Move crowded legends and adjust labels while preserving the current export implementation.
This has the smallest diff but leaves embedded scientific layers at screen resolution.

### 2. Publication hardening

Preserve the current compositions and scientific encodings while correcting export quality,
legend placement, palette consistency, symbol definitions, panel navigation, and statistical
terminology. This is the selected approach because it addresses objective submission risks
without redesigning the scientific story.

### 3. Full manuscript-led recomposition

Rebuild the figures into a new main-text sequence and combine or remove panels based on a final
paper narrative. This should wait for the real final cohort and a target journal because it could
otherwise optimize synthetic fixtures and force premature manuscript decisions.

## Design

### Export quality

Both Study 1 and Study 2 publication SVG writers will render embedded raster artists at 600 dpi
while retaining editable SVG text, paths, markers, and axes. The physical dimensions and RGB
colors remain unchanged. Pure vector figures are unaffected by the DPI parameter.

Regression tests will decode embedded PNG payloads from saved SVG files and calculate their
effective resolution from SVG display dimensions. Any embedded scientific image below 600 dpi
will fail the tests.

### Study 1 standalone figures

The three dose-response plots and two behavioral-validity coefficient plots remain separate
89 x 70 mm SVGs. Their legends move outside the axes into layout-managed upper space. No data
layer, scale, order, color, label, or uncertainty interval changes. This prevents a legend from
covering participant observations while keeping every standalone figure self-explanatory.

### Study 1 fMRI construct-validity figure

The existing fixed surfaces, axial slices, effect scales, and max-T contours remain unchanged.
Only the embedded surface, volume, and color-bar resolution changes. The saved figure must retain
its exact 183 x 112 mm dimensions and editable text.

### Study 2 Haufe forward-pattern figure

The five scalp maps will use the same color-vision-accessible blue-neutral-orange diverging map as
the Study 2 cortical figure. Every band title will use equal visual weight; inferential importance
will not be implied by unexplained typography. The fold-stability panel will define gray fold-pair
correlations and the median diamond in a compact legend outside the data region. The shared
symmetric scale, full correlation range, sensor markers, descriptive status, and interpretation
boundary remain unchanged.

### Study 2 primary cortical figure

The three band columns receive lowercase panel labels for unambiguous caption references. Every
band header reports its Holm-adjusted p-value, regardless of significance. The color-bar label
will state that displayed correlations are Fisher-z averaged and back-transformed to r.

The statistical helper, source-family artifact, figure summary, cluster audit, dataclasses, and
tests will consistently use `holm_adjusted_p_value` rather than `holm_q_value`. Existing artifacts
with the old column name will fail validation and must be regenerated; no compatibility alias or
schema fallback will be added.

The effect maps, target-retrained maximum-cluster inference, joint contour rule, family alpha,
shared display limit, and caption interpretation boundary remain unchanged.

## Code Boundaries

- Study-specific style modules own export DPI and shared color-map construction.
- Plot modules own layout, symbol keys, panel labels, and display wording.
- Study 2 statistical modules own adjusted-p terminology and artifact schemas.
- Figure readers continue to validate artifacts before rendering.
- Tests inspect both semantic figure structure and serialized output properties.

No plotting module will load alternate data, infer legacy schemas, or downgrade output when an
input is invalid.

## Verification

Focused tests will be written before implementation and will cover:

- embedded image resolution in Study 1 and Study 2 SVGs;
- legends outside Study 1 standalone data axes;
- a shared Study 2 diverging palette and equal band-title weight;
- an explicit fold-pair/median key;
- lowercase panel labels in the cortical figure;
- adjusted p-values shown for all cortical bands;
- corrected adjusted-p names in source-family and figure audit schemas; and
- unchanged physical dimensions, axes, scales, contours, and output families.

After focused tests pass, the complete Study 1/2 figure test set will be rerun into a retained
temporary directory. Every SVG will be rasterized at its intended aspect ratio and inspected for
clipping, overlap, legibility, scale consistency, and color balance. Ruff, architecture,
maintainability, and relevant broader tests will run before completion.

## Acceptance Criteria

- No embedded scientific image in a publication SVG is below 600 dpi at final size.
- Standalone Study 1 legends do not overlap the data region.
- Study 2 maps use one accessible diverging visual language without unexplained emphasis.
- Study 2 stability symbols are defined in the figure.
- Holm-adjusted p-values are named accurately and reported for every primary source band.
- Panel references, dimensions, scales, uncertainty, and inferential overlays remain correct.
- All focused and repository verification gates pass without touching unrelated user changes.
