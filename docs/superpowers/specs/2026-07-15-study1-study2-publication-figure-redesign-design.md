# Study 1 and Study 2 Publication Figure Redesign

**Date:** 2026-07-15

## Goal

Turn the complete current Study 1 and Study 2 figure suite into one coherent,
submission-quality visual system without changing scientific data, estimands,
statistics, uncertainty intervals, thresholds, cohort membership, or output
families.

## Current-State Audit

The repository already fixes physical figure widths at 89 mm and 183 mm, retains
editable SVG text, embeds raster layers at 600 dpi, exposes individual observations,
uses color-vision-accessible target colors, and publishes audit tables beside the
main analysis figures. The deterministic fixture suite also confirms shared scales,
zero references, and output dimensions.

Rendering every current figure at final size exposed problems not covered by those
structural checks:

- the standalone Study 1 dose-response and coefficient legends still occupy the
  data region;
- Study 1 and Study 2 duplicate most typography and export styling, allowing the
  two visual systems to drift;
- MNE's local Haufe interpolation extends colored polygons beyond the head outline;
- the Haufe stability points and median diamonds are not defined in the figure;
- Haufe band headings use unequal unexplained font weight;
- the Study 2 primary source figure still calls a Holm-adjusted p-value `q` and
  hides adjusted values for nonsignificant bands;
- the Study 2 spatial-null headings compress three statistics into one long line;
- whole-brain fMRI plotting explicitly disables the default zero threshold, so
  exact-zero voxels can obscure the anatomical background; and
- titles, muted annotations, tick marks, line caps, and font embedding are not
  governed by one repository-wide publication contract.

The deterministic cortical fixtures use intentionally tiny triangle meshes and are
not anatomical mock data. Their geometry is useful for testing layout, masks, and
contours, but it is not used to judge the appearance of a real cortical surface.

## Publication Constraints

The design targets the existing Nature-sized artwork contract because the repository
already uses its 89 mm single-column and 183 mm double-column widths. The current
Nature figure guide asks for editable standard sans-serif text at 5–7 pt, efficient
panel arrangement, accessible colors, and minimal excess white space. Nature also
requests exact sample sizes and exact significant and nonsignificant probability
values where relevant. These constraints are documented at:

- https://research-figure-guide.nature.com/figures/building-and-exporting-figure-panels/
- https://www.nature.com/nature/for-authors/initial-submission

The existing SVG plus 600 dpi embedded-raster strategy exceeds the raster minimums
while preserving vector axes, labels, contours, and markers.

## Considered Approaches

### 1. Style-only cleanup

Change only global font and line parameters. This would make the suite more
consistent, but it would leave legends over data, scalp colors outside the head,
ambiguous statistical terminology, and dense null-panel headings.

### 2. Shared publication system with targeted recomposition

Centralize the common visual contract and correct the layouts that failed visual
inspection. This is the selected approach. It produces a meaningful improvement
without changing the scientific story or rebuilding plots that are already clear.

### 3. Manuscript-led figure consolidation

Combine plots into a smaller main-text sequence and move remaining panels to an
extended-data hierarchy. That requires a final manuscript narrative and target
journal decisions that are not present in the repository, so it would risk changing
the communication scope rather than improving the current figures.

## Design

### Shared visual contract

A study-wide style module will own final-size conversion and common Matplotlib
parameters. Study-specific style modules will continue to own configuration lookup,
deterministic hash salts, atomic output, and closing semantics.

The shared contract will require Arial, keep all text within 5–7 pt, preserve white
backgrounds and editable SVG text, embed TrueType fonts for PDF/PS, use dark-neutral
text rather than pure black, standardize tick length and weight, use rounded line
caps, and remove legend frames. It will expose one helper for a figure-level legend
above a single data axis. Missing fonts, labels, and invalid dimensions will continue
to fail immediately.

Every current Study 1 and Study 2 publication renderer already enters one of the two
study-specific style contexts, so this change reaches the entire suite without
duplicating edits across every plot module.

### Standalone Study 1 figures

Dose-response and behavioral-validity legends will become figure legends placed in
layout-managed space above the axes. Their physical 89 × 70 mm size, data layers,
axis scales, target colors, participant traces, cohort estimates, confidence
intervals, reference lines, and labels will remain unchanged.

### Study 2 Haufe forward patterns

Scalp maps will use the same blue–neutral–orange diverging map as the cortical source
figures and one symmetric scale. Head extrapolation will terminate interpolation at
the anatomical head outline instead of exposing a sensor-convex-hull polygon beyond
the head. All five band headings will use equal weight.

The stability panel will define gray fold-pair observations and the median diamond in
an outside legend. The figure will state that scalp colors are interpolated within
the head outline and will retain the boundary that these are sensor-level forward
patterns, not cortical localization.

### Study 2 source and convergence figures

The primary cortical figure will show a lowercase panel letter for each band and the
Holm-adjusted p-value for every band. The term `q` will not appear. Significance will
still be communicated only by the existing corrected contour rule, not by color,
font weight, or omission of nonsignificant values.

Spatial-null panels will split their annotation into a concise observed correlation
line and a probability line. The exact plus-one two-sided p-value and Holm-adjusted
p-value will remain visible for every band. Existing histogram data, observed marker,
significance outline, common limits, and BrainSMASH statement remain unchanged.

### Study 1 whole-brain fMRI figure

Axial overlays will mask exact-zero effect voxels so the anatomical template remains
visible outside the modeled effect support. Nonzero effects remain unthresholded,
and the max-T contour is unchanged. Surface projections, robust symmetric limits,
fixed slices, units, and color bars remain unchanged.

### Titles and supporting text

The shared style will regularize title and annotation colors, sizes, and weights.
Figure-level titles remain because the current outputs are designed to be understood
outside a manuscript assembly step, but inferential qualifications stay in muted
supporting text rather than competing with the scientific panels.

## Code Boundaries

- `studies/pain_study/figure_style.py` owns only cross-study visual constants,
  dimension conversion, rc parameters, font validation, and outside-legend layout.
- Study 1 and Study 2 style modules own their existing public context managers and
  writers while delegating the common contract.
- Plot modules own composition and wording, never statistical calculation.
- Existing summary and artifact modules remain unchanged.
- Tests inspect semantic artists and serialized output; visual inspection uses only
  deterministic fixtures and does not substitute for real-data regeneration.

No compatibility aliases, fallback fonts, automatic layout downgrades, or alternate
data paths will be introduced.

## Test Strategy

Test-driven changes will cover:

- exact shared rc parameters and font failure;
- figure-level legends fully above standalone Study 1 axes;
- one accessible Haufe palette, equal title weight, head-bounded interpolation, and
  an explicit stability-symbol legend;
- panel labels and Holm-adjusted p-values for all primary cortical bands;
- readable two-line exact statistics in every spatial-null panel;
- exact-zero masking in axial fMRI plotting; and
- unchanged dimensions, shared scales, contours, axes, audit artifacts, and 600 dpi
  embedded layers.

Every affected figure will be rendered to PNG at final aspect ratio and inspected for
clipping, collisions, hierarchy, interpolation boundaries, anatomical readability,
and color consistency. Focused tests, figure-family tests, Ruff, architecture,
maintainability, and whitespace checks will run before completion.

## Acceptance Criteria

- Every current Study 1 and Study 2 renderer inherits one visual contract.
- No legend overlaps a standalone Study 1 data axis.
- No Haufe interpolation is visible beyond the head outline.
- Haufe bands use one palette and equal visual hierarchy; stability symbols are
  defined.
- Study 2 uses accurate adjusted-p terminology and reports exact adjusted values for
  every displayed band.
- Spatial-null statistics are legible without removing exact values.
- Exact-zero fMRI voxels do not hide the anatomical background.
- Scientific values, output dimensions, uncertainty, inference, contours, and audit
  families remain unchanged.
- Focused and repository verification gates pass with no unintended files.
