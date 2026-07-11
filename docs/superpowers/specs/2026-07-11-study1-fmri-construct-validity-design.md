# Study 1 Whole-Brain fMRI Construct-Validity Figure Design

## Purpose

Create one whole-brain fMRI validity figure showing where BOLD activity tracks delivered thermal
intensity and where it tracks subjective intensity beyond delivered temperature. This fills the
remaining spatial-neurobiology gap in Study 1. It does not repeat the existing NPS/SIIPS1
dose-response, EEG prediction, spectral-specificity, temporal-specificity, or sensor-pattern
figures.

The figure supports two claims only:

1. the thermal manipulation produces a spatially plausible whole-brain BOLD response; and
2. within-temperature variation in subjective intensity has a measurable whole-brain BOLD
   correlate.

It does not claim source localization of EEG, mechanistic mediation, or anatomical specificity
from post-hoc regions of interest.

## Inputs and cohort

- Use the retained Study 1 target cohort so the fMRI validity figure describes the same
  participants as the EEG-to-fMRI analyses.
- Read preprocessed MNI152NLin2009cAsym BOLD images, brain masks, fMRIPrep confounds, and current
  clean events directly. Do not depend on optional stored LSS trial-beta maps.
- Use plateau stimulation events only for the two estimands. Model every other task event as a
  nuisance event rather than dropping it from the first-level design.
- Use the validated within-scale 0–100 intensity score. Never use the discontinuous 0–200 display
  code.
- Require the current protocol fields, all six temperature levels, finite behavioral values,
  unique subject/run/trial identifiers, one thermode surface per run, valid MNI geometry, and a
  common participant cohort across both estimands.

## First-level estimands

Fit two separate multi-run first-level GLMs per participant. Separate models prevent subjective
rating, which may mediate the temperature response, from changing the temperature estimand.
Both models use the Study 1 HRF, drift, high-pass, 6 mm smoothing, and motion24 confound settings.
Non-steady-state and motion-outlier volumes are handled through the existing explicit sample-mask
contract. Each run has its own intercept and nuisance structure; this absorbs run-level thermode
surface differences after validating that surface is constant within run.

### Temperature response

For each plateau trial, use delivered temperature centered within run and scaled in 1 °C units.
Include centered within-run trial order as an event-level nuisance term. The contrast is the BOLD
effect per 1 °C increase. Rating is not adjusted because it is downstream of the randomized
thermal manipulation. Construct the design explicitly by giving the temperature and trial-order
modulators distinct regressors at the same plateau onset and duration; retain non-plateau events
as unit-amplitude nuisance regressors.

### Rating response beyond temperature

Represent the six delivered temperatures as categorical event regressors. Define the subjective
regressor as the 0–100 intensity score centered within participant and temperature, then scale it
to 10-point units. Include centered within-run trial order as an event-level nuisance term. The
contrast is the BOLD effect per 10-point increase in subjective intensity among trials delivered
at the same temperature. Give the rating, trial-order, and six temperature-condition regressors
their own rows at the same plateau onset and duration so their meanings remain explicit in the
saved design matrices.

Reject rank-deficient designs, excessive configured condition numbers, absent residual rating
variation, missing runs, inconsistent masks or affines, and non-finite contrast maps. Do not
silently omit participants or nuisance terms.

## Group inference

- Write one participant effect-size map for each estimand.
- Fit an intercept-only second-level model to the participant effect-size maps.
- Report the group mean effect-size map, not a thresholded statistic map, as the visual substrate.
- Use 10,000 deterministic sign-flipping permutations, a two-sided voxelwise max-T family-wise
  error correction, and alpha = 0.05.
- Store the negative-log10 FWE p map and derive the significance mask at
  `-log10(0.05)`. The inferential unit is the participant.
- Require at least the configured Study 1 article cohort size for an article-ready label; smaller
  synthetic or smoke cohorts must be marked preliminary.

## Figure design

Create one 183 × 112 mm editable SVG with two aligned columns:

- **Panel a — Delivered temperature:** group mean BOLD change per 1 °C.
- **Panel b — Subjective intensity beyond temperature:** group mean BOLD change per 10 rating
  points at fixed temperature.

Each panel contains four fixed fsaverage5 cortical views (left and right lateral, left and right
medial) followed by six fixed axial MNI slices at z = -12, 0, 12, 24, 36, and 48 mm. These
prespecified coordinates prevent slice selection by appearance and retain subcortical and
brainstem visibility.

Show the unthresholded group mean effect with a perceptually uniform, color-vision-safe diverging
map centered at zero. Use a separate symmetric robust display range for each estimand because the
units differ. Overlay the max-T FWE significance boundary on volume and surface views. Include one
horizontal color bar per panel with exact units, the participant count, correction method, and
article-readiness status. Do not use opaque thresholding, significance stars, cluster-size
filtering, decorative brain icons, or post-hoc ROI summaries.

Peak coordinates and cluster summaries belong in paired audit tables, not on the brain image.

## Architecture

- `fmri_construct_data.py`: cohort resolution, event/confound loading, current-protocol
  validation, and first-level design construction;
- `fmri_construct_models.py`: participant GLMs, contrast-map writing, second-level effect and
  permutation inference, and peak/cluster audit creation;
- `fmri_construct_validity.py`: immutable summary objects and end-to-end orchestration;
- `fmri_construct_validity_plot.py`: deterministic surface/volume rendering only;
- `plot_fmri_construct_validity.py`: the only public writer and CLI;
- `study1_figure_config.yaml`: figure dimensions, fixed slices, inference seed/count/alpha,
  display mesh, and article-readiness threshold.

Reuse existing fMRIPrep discovery, confound selection, Study 1 cohort, clean-event validation,
publication style, NIfTI writers, and second-level conventions. Do not add fallback schemas or
duplicate a generic fMRI pipeline inside the renderer.

## Outputs

- `fmri_construct_validity.svg`;
- participant temperature and rating effect-size NIfTI maps;
- group mean temperature and rating effect-size NIfTI maps;
- group temperature and rating max-T FWE negative-log10 p NIfTI maps;
- `fmri_construct_validity_subjects.tsv` and `.parquet`;
- `fmri_construct_validity_design_audit.tsv` and `.parquet`;
- `fmri_construct_validity_peaks.tsv` and `.parquet`;
- a JSON provenance sidecar containing input checksums, model settings, software versions,
  permutation seed, and output checksums.

The writer validates and computes all required analysis products before writing the final SVG.
Unexpected errors surface. Partial or stale derivative sets never produce an article figure.

## Verification

- Unit-test temperature centering/scaling, within-temperature rating centering, event duplication,
  nuisance construction, design rank/condition validation, contrast units, and exact cohort
  alignment.
- Use small synthetic NIfTI series with known temperature and rating signals to verify first-level
  effect direction and recovery without weakening production inference settings.
- Test deterministic max-T inference with a dedicated small-test inference object rather than a
  production flag.
- Test fixed physical dimensions, view order, slice coordinates, symmetric color limits,
  significance overlays, editable SVG text, external labels, and clipping.
- Write exact output-contract tests and ensure repeat runs are byte-stable where the underlying
  libraries permit deterministic serialization.
- Inspect a synthetic rendered PNG before attempting the real-data figure.
- Run the focused tests, full Study 1 suite, Ruff, structure, architecture, and maintainability
  gates.
- Run a real-data preflight. If current derivatives are stale or incomplete, report the first
  exact incompatibility and confirm that no final SVG was written.
