# Study 1 Supplementary Validity Plots Design

**Goal:** Add three clean, publication-quality Study 1 validity plots for supplementary
materials. Each scientific question has one executable plotting script and exactly one SVG
image output.

**Classification:** Supplementary figures. This work does not create or label any main-text
figure.

**Target style:** A conservative Nature/Science-compatible figure system: editable SVG text,
single-column 89 mm width, 5–7 pt Arial lettering, accessible colour, visible axes and tick marks,
minimal whitespace, and no background gridlines or decorative elements.

## Scientific Scope

The initial validity figure set contains exactly three standalone dose-response plots:

1. `behavioral_dose_response.svg`
   - Displayed behavioral rating from 0 to 200 versus stimulus temperature.
   - A subtle horizontal reference at 100 identifies the protocol's pain-category threshold.
2. `nps_dose_response.svg`
   - Trial-wise NPS signature expression summarized by participant and temperature.
3. `siips1_dose_response.svg`
   - Trial-wise SIIPS1 signature expression summarized by participant and temperature.

Each plot answers one question: whether the behavioral response or fMRI signature changes across
the prespecified thermal stimulus levels. NPS and SIIPS1 remain in separate figures because their
native dot-product scales are not comparable.

The following report quantities remain tables in this first implementation:

- sample and retained-trial counts;
- isolated validity and nuisance correlations;
- residual target variance fractions; and
- target support and LSS design metrics.

Plotting these small collections of values would not improve their interpretation. The
1,000-partition split-half reliability distribution may become a later standalone supplementary
plot, but it is outside the approved initial scope.

## Statistical Display

### Observed participant layer

For every participant and temperature, the plotting data contain the arithmetic mean of all
retained matched trials. Small markers show these participant-by-temperature means. Thin,
low-opacity gray lines connect adjacent observed temperatures within each participant. Missing
participant-temperature cells remain missing and create gaps; they are never imputed.

Participant-temperature means are the displayed observational unit. Individual trials are not
jittered behind the trajectories because the final cohort may contain thousands of trials and the
resulting overplotting would obscure the repeated-measures structure.

### Cohort estimate layer

At each temperature, the cohort estimate is the unweighted mean of the available participant-level
means. This gives participants equal weight and matches the Study 1 subject-weighted inferential
emphasis.

The uncertainty interval is a percentile 95% confidence interval from 10,000 participant-cluster
bootstrap resamples. Each resample draws participants with replacement and uses the same sampled
participant set across every temperature, preserving the repeated-measures structure. The random
seed is fixed in configuration. A bootstrap replicate computes a temperature mean from sampled
participants who have a retained cell at that temperature.

A draw is invalid when it selects no participant with an observed cell at any configured
temperature. Invalid draws are resampled until 10,000 valid draws are obtained, subject to the
configured maximum invalid-draw fraction. The figure computation raises when it exceeds that
budget. This preserves one sampled participant set across the temperature curve without imputing
missing cells or silently changing the estimand.

The cohort estimate is drawn as a stronger target-specific line with circular markers and vertical
confidence-interval whiskers. No hypothesis-test stars, categorical validity verdicts, or fitted
regression curves are added.

## Data Contract

All plots use only retained rows from the Study 1 primary target table. The behavioral plot obtains
ratings by joining those target rows to clean EEG events on the existing canonical trial key:

- `subject_id`;
- `run`; and
- `within_run_trial` / clean-event `trial_number`.

The join must be one-to-one. All target rows must have matching clean events. Extra clean-event rows
are allowed because target preparation may exclude trials before the validity display. The joined
table therefore represents the same retained participant and trial cohort for all three figures.

The shared loader validates:

- required columns;
- numeric and finite temperature, rating, NPS, and SIIPS1 values;
- integer-valued run and trial keys;
- unique target and clean-event trial keys;
- exact agreement between observed and configured stimulus-temperature levels;
- at least two represented participants at every temperature; and
- at least one retained trial in every displayed participant-temperature cell.

The implementation supports only the current Study 1 schema. It does not translate older output
schemas or search alternate artifact locations.

## Code Organization

The figures live in a Study 1-specific package:

```text
studies/pain_study/study1/figures/
├── __init__.py
├── validity_data.py
├── validity_style.py
├── dose_response.py
├── plot_behavioral_dose_response.py
├── plot_nps_dose_response.py
└── plot_siips1_dose_response.py
```

Responsibilities are separated as follows:

- `validity_data.py`
  - loads and validates retained trial data;
  - performs the canonical target/event merge;
  - creates participant-temperature summaries; and
  - computes deterministic participant-cluster bootstrap estimates.
- `validity_style.py`
  - validates the required Arial font;
  - applies the approved publication dimensions and Matplotlib parameters; and
  - saves editable SVG output with no raster companion.
- `dose_response.py`
  - draws the shared scientific grammar from a validated summary and an explicit outcome
    specification;
  - does not load data or select output paths.
- Each `plot_*.py` module
  - owns one outcome specification, axis labels, colour, and output filename;
  - exposes a callable writer for report integration;
  - exposes a small `main()` entry point for direct execution; and
  - writes exactly one SVG image.

Existing report data-loading and merging logic will be moved to the shared validity-data layer and
reused by `reporting.py`. This is a behavior-preserving refactor that prevents two implementations
of the canonical trial join.

## Configuration

Figure settings live in a dedicated YAML file under the Study 1 configuration package. Only values
that affect the publication contract or statistical estimator are configurable:

- expected stimulus temperatures;
- figure width and height in millimeters;
- font family and font sizes;
- line widths, marker sizes, and opacity;
- behavioral, NPS, SIIPS1, and participant colours;
- bootstrap iterations, confidence level, seed, and maximum invalid-draw fraction; and
- SVG output directory relative to the Study 1 report root.

The initial defaults are:

- 89 mm single-column width;
- Arial, with all plot text between 5 and 7 pt;
- Okabe–Ito blue for NPS and vermillion for SIIPS1;
- dark neutral ink for the behavioral cohort estimate;
- low-opacity neutral gray for participants;
- 10,000 bootstrap iterations;
- 95% confidence intervals; and
- one fixed bootstrap seed; and
- at most 20% invalid bootstrap draws.

Axes use outward ticks, visible left and bottom spines, no top or right spines, and no grid. Figure
titles stay in the manuscript caption rather than inside the SVG. The behavioral y-axis spans the
protocol range 0–200. Signature axes are determined from the observed participant summaries and
confidence intervals with a small deterministic margin; they are never forced to share limits.

## Output Contract and Report Integration

The three output paths are:

```text
<study1-root>/reports/figures/supplementary/validity/behavioral_dose_response.svg
<study1-root>/reports/figures/supplementary/validity/nps_dose_response.svg
<study1-root>/reports/figures/supplementary/validity/siips1_dose_response.svg
```

The `report` stage invokes all three plot writers after it has validated the Study 1 primary
outputs. It adds their paths under a `supplementary_figures` field in the existing full-picture
manifest. No new figure numbering is hard-coded because manuscript numbering may change during
submission.

Each plotting script writes through a temporary SVG in the destination directory and atomically
replaces its final output only after Matplotlib finishes successfully. It does not create PNG, PDF,
EPS, thumbnail, legend-only, or data-sidecar files.

## Failure Behavior

The plotting code fails immediately and explicitly when:

- the primary target table is absent;
- a required clean-event table is absent;
- the target/event merge is incomplete or non-unique;
- required values are missing, non-numeric, or non-finite;
- configured and observed temperature levels disagree;
- a temperature has fewer than two represented participants;
- the valid bootstrap count cannot be reached within the configured invalid-draw budget;
- accepted bootstrap estimates are non-finite;
- Arial is unavailable; or
- SVG writing fails.

The code does not skip malformed participants, substitute another font, select an alternate rating
column, infer legacy trial keys, write a reduced plot, or fall back to another image format.

## Testing and Visual Verification

Tests will be written before implementation and will cover:

- strict target/event join validation;
- participant-temperature aggregation with unequal retained-trial counts;
- gaps for missing participant-temperature cells without imputation;
- equal participant weighting of cohort estimates;
- participant-cluster resampling across the full temperature trajectory;
- invalid-draw resampling and strict enforcement of its configured budget;
- deterministic confidence intervals under the configured seed;
- rejection of invalid temperatures, values, keys, fonts, and bootstrap outputs;
- one SVG output from each plotting module;
- absence of PNG, PDF, EPS, and extra SVG artifacts;
- exact output directory classification as `supplementary/validity`;
- expected physical SVG dimensions, axis labels, and behavioral threshold annotation;
- distinct NPS and SIIPS1 scales; and
- unchanged Study 1 article and full-picture tables after the reporting refactor.

The three SVGs will then be rendered to raster previews solely for visual QA. Verification occurs at
the intended 89 mm publication width and checks legibility, clipping, overlapping labels, line and
marker balance, confidence-interval visibility, grayscale distinction, and colour-vision
accessibility. Raster previews are temporary verification artifacts and are not report outputs.

## Acceptance Criteria

The implementation is complete when:

- the Study 1 report deterministically writes the three approved SVG files;
- each executable plotting module creates only its assigned SVG;
- all figures use the same retained target-trial cohort;
- observed participant trajectories, cohort estimates, and 95% participant-bootstrap intervals are
  visible and correctly computed;
- behavioral, NPS, and SIIPS1 plots remain separate and scientifically labelled;
- outputs are explicitly classified as supplementary;
- invalid inputs surface errors instead of producing partial or downgraded figures;
- reporting behavior and existing table contents remain unchanged; and
- automated and publication-size visual verification pass.
