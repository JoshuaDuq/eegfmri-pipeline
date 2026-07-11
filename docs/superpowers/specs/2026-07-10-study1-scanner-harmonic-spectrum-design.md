# Study 1 Scanner-Harmonic Spectrum Design

## Purpose and necessity

Create one supplementary QC figure that validates the frequency intervals used for Study 1 EEG
spectral predictors. This figure earns article space because the location, width, and
participant-consistency of narrow spectral peaks cannot be judged from a table of peak centres or
described adequately in prose. Split-half target reliability and cohort attrition remain tables;
the later temporal-specificity analysis belongs with the predictive-results figure.

The figure is outcome-blind. It reads final-clean thermal-task EEG and never uses pain reports,
NPS, SIIPS1, feature-model performance, or inferential results.

## Analytic sample and unit of analysis

The script discovers numbered `sub-*` participants with final-clean thermal-task FIF files under
the configured EEG derivative root. Pilot-style identifiers are excluded by the exact subject
pattern. The prespecified pilot `sub-0006` is excluded explicitly. Optional subject selection is
allowed at the command line for reproducible reruns.

Participants, not runs, are the cohort unit:

1. For each run, estimate Welch PSD from 15 to 90 Hz over EEG channels with 500 Hz sampling,
   `n_fft = n_per_seg = 8192`, and `n_overlap = 4096`.
2. At each frequency, take the median linear PSD across EEG channels and convert it to dB.
3. At each frequency, take the median across runs within participant.
4. Subtract each participant spectrum's median across 15 to 90 Hz. This is a constant vertical
   translation that preserves spectral shape and peak amplitudes while removing arbitrary
   between-participant offsets.
5. Summarize participants with the cohort median and a 95% percentile interval from 10,000 paired
   participant-bootstrap resamples using seed 42.

The run audit must record source file, subject, run, channel count, frequency resolution, and peak
frequency/prominence. The participant audit must record run count and median peak frequency and
prominence for each window.

## Scanner-locking test

Detect run-level peaks on the across-channel median dB spectrum with minimum prominence 1 dB and
minimum distance four frequency bins. Select the most prominent peak within each fixed window:
18–23, 38–43, 56–67, and 77–85 Hz.

For a 0.9 s volume repetition time, compare each participant's median peak centre with the
prespecified harmonics 18, 37, 55, and 74 of the volume frequency. Plot the observed-minus-predicted
offset. A reference band of plus or minus one Welch bin (500/8192 Hz) makes acquisition-frequency
agreement directly interpretable.

## Figure design

The final SVG is an exact 183 × 82 mm two-column figure using the existing Arial 5.5–7 pt Study 1
publication style.

### Panel a: participant and cohort spectra

- Thin, low-opacity gray lines: one median spectrum per participant.
- Thick black line: equally weighted cohort median.
- Light neutral band: 95% paired participant-bootstrap interval.
- Pale vermillion vertical spans: the four scanner-harmonic windows.
- Blue lower-axis segments: retained scanner-clean gamma intervals 30.1–38, 43–56, and 67–77 Hz.
- Annotation: participant and run counts.
- Axes: `Frequency (Hz)` and `PSD relative to participant median (dB)`.

### Panel b: agreement with MRI timing

- Gray dots: one median peak offset per participant and harmonic window.
- Hollow black diamonds with vertical 95% bootstrap intervals: cohort medians.
- Dashed zero line: exact agreement with the predicted TR harmonic.
- Pale blue horizontal band: plus or minus one Welch bin.
- Category labels state harmonic order and predicted frequency.

There is no title inside the figure, no significance stars, no bar chart, and no run-level
pseudoreplication in cohort summaries. The color palette is color-vision-accessible and the SVG
retains editable text.

The five-item legend is a single figure-level row above both panels, outside both plotting axes.
The layout reserves explicit top margin for the legend; neither panel may own an in-axis legend or
allow legend artists to cover data.

## Code organization

- `scanner_harmonic_spectrum.py`: discovery, PSD estimation, aggregation, audit tables, bootstrap,
  and rendering as small single-purpose functions.
- `plot_scanner_harmonic_spectrum.py`: the only CLI script for this plot.
- `study1_figure_config.yaml`: figure dimensions and fixed acquisition/QC parameters.
- `validity_style.py`: reusable atomic publication-SVG writer with explicit dimensions.

The plot is standalone rather than part of `write_study1_report`, because reading 77 continuous
EEG runs is a deliberately expensive raw-data QC stage and must not be repeated whenever the model
report is regenerated.

## Validation

Tests use synthetic PSD arrays and lightweight fake raw readers; repository tests never depend on
the external EEG volume. They verify:

- numeric-subject discovery and pilot exclusion;
- exact participant weighting and run-to-participant aggregation;
- peak selection and TR-harmonic offsets;
- deterministic paired bootstrap output;
- required audit-table columns;
- panel semantics, exact physical SVG dimensions, editable text, and deterministic SVG bytes;
- one-file CLI output routing.

The final data run must cover 13 participants and 77 runs, reproduce narrow peaks near 20, 41, 61,
and 82 Hz, and receive visual inspection at publication size before completion.
