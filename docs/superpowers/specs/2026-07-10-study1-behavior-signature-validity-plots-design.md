# Study 1 Behavior–Signature Validity Plots Design

**Goal:** Add the next article-essential Study 1 supplementary validity figures: two
publication-quality coefficient plots testing whether NPS and SIIPS1 track reported pain beyond
delivered temperature.

**Classification:** Supplementary validity/QC figures. These plots do not display EEG prediction
results and do not receive fixed manuscript figure numbers.

**Target style:** The existing Study 1 Nature/Science-compatible figure system: editable and
byte-reproducible SVG, 89 mm single-column width, Arial text between 5 and 7 pt, accessible target
colours, visible axes, minimal whitespace, and no decorative elements.

## Scientific Rationale

The existing behavioral, NPS, and SIIPS1 dose-response figures establish sensitivity to delivered
temperature. Temperature sensitivity alone does not establish that a signature tracks perceived
pain rather than stimulus intensity. The next validity figures therefore quantify participant-level
associations between each fMRI signature and two complementary behavioral terms while controlling
the temperature manipulation:

- the categorical transition between non-painful and painful reports; and
- intensity variation within the reported heat or pain category.

NPS is evaluated beyond categorical temperature. SIIPS1 is evaluated beyond categorical
temperature and NPS because Study 1 interprets SIIPS1 as pain-related expression not explained by
stimulus intensity or NPS.

## Figure Set

The implementation writes exactly two standalone plots:

1. `nps_behavioral_validity.svg`
   - outcome: trial-wise NPS expression;
   - focal terms: binary pain report and within-scale intensity; and
   - adjustment: categorical stimulus temperature.
2. `siips1_behavioral_validity.svg`
   - outcome: trial-wise SIIPS1 expression;
   - focal terms: binary pain report and within-scale intensity; and
   - adjustment: categorical stimulus temperature and trial-wise NPS expression.

The targets remain separate because their dot-product scales and adjustment models differ. Each
plot has one executable script and produces only its assigned SVG.

## Behavioral Data Contract

The figures use the retained, one-to-one target/event cohort already returned by
`load_validity_trial_data`. They do not read alternate event files or select substitute columns.

The protocol-defined within-scale intensity is constructed once at the validity-data boundary:

\[
I =
\begin{cases}
R, & P = 0 \\
R - 100, & P = 1,
\end{cases}
\]

where \(R\) is `vas_final_coded_rating` and \(P\) is `pain_binary_coded`. The loader requires:

- `pain_binary_coded` to contain only finite integer values 0 and 1;
- non-painful ratings to be in `[0, 99]`;
- painful ratings to be in `[100, 200]`; and
- the derived within-scale value to be finite and in `[0, 100]`.

Any inconsistent pain/rating pair raises an error. The plotting estimator never uses the raw
discontinuous 0–200 displayed rating as a continuous pain-intensity variable.

## Participant-Level Estimands

One joint ordinary least-squares model is fit separately for every retained participant and target.
All continuous focal variables and the target are standardized within participant using their
sample mean and sample standard deviation. The binary pain report is also standardized within
participant so both displayed focal coefficients are standardized partial coefficients. Stimulus
temperature enters as treatment-coded categorical indicators and is not treated as linear.

For NPS, the participant model is:

\[
z(\mathrm{NPS}) = \beta_0 + C(\mathrm{temperature})
+ \beta_{pain} z(P) + \beta_{intensity} z(I) + \epsilon.
\]

For SIIPS1, the participant model is:

\[
z(\mathrm{SIIPS1}) = \beta_0 + C(\mathrm{temperature}) + \beta_{NPS} z(\mathrm{NPS})
+ \beta_{pain} z(P) + \beta_{intensity} z(I) + \epsilon.
\]

The two displayed quantities are \(\beta_{pain}\) and \(\beta_{intensity}\). They are estimated
jointly, so the pain-report coefficient captures the category transition conditional on
within-category intensity and the intensity coefficient captures variation within the heat/pain
scale conditional on the category transition.

A participant model is estimable only when:

- the target, pain report, within-scale intensity, and SIIPS1-adjustment NPS term where applicable
  each have non-zero finite variance;
- every configured temperature is represented;
- the design matrix is full column rank under the repository's numerical rank tolerance; and
- residual degrees of freedom are positive.

Non-estimable participants are explicitly recorded with a reason and do not contribute a partial
coefficient. The figure requires at least `study1.cohort.min_subjects` estimable participants for
the target; otherwise it raises. Both focal coefficients therefore use the same participant set
and sample size.

## Cohort Estimation and Uncertainty

The cohort point estimate is the arithmetic mean of the participant coefficients. Participants,
not trials, are the observational units and receive equal weight.

Each 95% confidence interval is a percentile interval from the existing configured number of
participant-bootstrap resamples. Each draw samples complete two-coefficient participant rows with
replacement, preserving their covariance. The configured seed makes the result deterministic.
Accepted draws must contain finite means for both coefficients. Failure to obtain the required
number of valid draws within the configured invalid-draw budget raises an error.

These are descriptive construct-validity estimates. The SVG contains confidence intervals but no
p-values, significance stars, categorical pass/fail labels, or multiplicity claims.

## Visual Encoding

Each plot is a compact horizontal coefficient plot with two rows, ordered:

1. `Painful report`;
2. `Within-scale intensity`.

For each row:

- small, low-opacity gray circles show participant coefficients;
- deterministic vertical jitter prevents exact overlap without implying another variable;
- a target-coloured diamond shows the equally weighted cohort mean;
- a horizontal target-coloured whisker shows the participant-bootstrap 95% confidence interval;
  and
- a right-aligned annotation reports the common estimable participant count.

A thin dashed vertical line marks zero. The x-axis is labelled
`Standardized partial coefficient (β)`. Limits are symmetric around zero and are derived from the
participant estimates and confidence bounds with a deterministic margin. The plot has no title,
background grid, density shape, trial-level scatter, or significance annotation.

The NPS plot uses the existing Okabe–Ito blue (`#0072B2`); SIIPS1 uses the existing vermillion
(`#D55E00`). Participant gray, line weights, marker sizes, fonts, spines, and SVG saving behavior
reuse the current validity configuration and style helpers. Both SVGs retain the existing
89 × 70 mm physical size.

## Code Organization

The additions extend the existing Study 1 figure package:

```text
studies/pain_study/study1/figures/
├── validity_data.py
├── validity_style.py
├── behavioral_validity.py
├── coefficient_plot.py
├── plot_nps_behavioral_validity.py
└── plot_siips1_behavioral_validity.py
```

Responsibilities are separated as follows:

- `validity_data.py` validates the behavioral protocol coding and exposes the canonical
  within-scale intensity column on enriched retained trials.
- `behavioral_validity.py` validates participant model inputs, builds explicit design matrices,
  estimates participant coefficients, records non-estimability reasons, and computes the paired
  participant bootstrap summary.
- `coefficient_plot.py` renders a validated coefficient summary without loading data or selecting
  output paths.
- Each `plot_*.py` module owns one target specification, output filename, writer, and CLI entry
  point. Each writes exactly one SVG.

Shared estimation and drawing code is not duplicated between target scripts. No compatibility
aliases or alternate estimators are introduced.

## Report Outputs

The report stage writes both SVGs under:

```text
<study1-root>/reports/figures/supplementary/validity/nps_behavioral_validity.svg
<study1-root>/reports/figures/supplementary/validity/siips1_behavioral_validity.svg
```

The full-picture manifest registers both paths under `supplementary_figures`. The report also
writes reusable full-picture tables for auditability:

- participant coefficients and explicit non-estimability status; and
- cohort means, confidence bounds, and estimable-participant counts.

The diagnostic reporting schema is corrected to use the protocol-defined within-scale intensity
for continuous pain-intensity associations. It does not retain a misleading legacy alias that
describes the discontinuous 0–200 displayed code as a continuous VAS intensity measure.

## Failure Behavior

The new code fails immediately when:

- behavioral pain/rating coding violates the protocol;
- a required target or adjustment column is absent, non-numeric, or non-finite;
- configured and observed temperature levels disagree;
- fewer than the configured minimum participants have estimable models;
- an accepted participant coefficient or bootstrap estimate is non-finite;
- the bootstrap valid-draw budget is exceeded;
- the required font is unavailable; or
- SVG writing fails.

The code does not silently drop malformed trials, change adjustment sets, pool targets, use the raw
0–200 rating as intensity, switch to a mixed model, substitute a different font, or write a
downgraded image format.

## Testing and Visual Verification

Tests are written before implementation and cover:

- conditional validation of pain-report and rating pairs;
- deterministic construction of the within-scale intensity score;
- exact NPS and SIIPS1 design matrices;
- within-participant standardization and categorical temperature adjustment;
- recovery of known partial coefficients from synthetic data;
- rejection and explicit reporting of zero variance, rank deficiency, missing temperatures, and
  insufficient residual degrees of freedom;
- enforcement of `study1.cohort.min_subjects`;
- equal participant weighting and paired participant bootstrap resampling;
- deterministic confidence intervals under the configured seed;
- physical SVG dimensions, editable text, axis labels, zero reference, participant layer, cohort
  markers, confidence whiskers, colours, and sample-size annotations;
- exactly one SVG from each executable plot module;
- report manifest and full-picture table integration; and
- the corrected within-scale diagnostic schema.

Both SVGs are rendered to temporary raster previews for visual QA at the intended 89 mm width.
Verification checks clipping, label overlap, coefficient and CI visibility, participant/cohort
hierarchy, symmetric limits, grayscale readability, and colour-vision accessibility. Raster
previews are verification artifacts, not report outputs.

## Acceptance Criteria

Implementation is complete when:

- the report deterministically writes the two approved standalone SVGs;
- each target has one executable plotting script that creates only its assigned SVG;
- the figures use the retained target/event cohort and the protocol-defined within-scale score;
- participant partial coefficients, equal-weight cohort means, and participant-bootstrap 95% CIs
  are correctly computed and displayed;
- NPS and SIIPS1 use their prespecified, distinct adjustment models;
- non-estimability is explicit and the configured minimum cohort requirement is enforced;
- manifest and tabular audit outputs are complete;
- invalid inputs surface errors instead of producing partial or altered figures; and
- focused tests and publication-size visual verification pass.
