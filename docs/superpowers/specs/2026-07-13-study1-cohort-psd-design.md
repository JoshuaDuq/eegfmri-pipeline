# Study 1 Cohort Power Spectral Density Figure Design

## Purpose

Create a reproducible 1–90 Hz cohort power spectral density (PSD) artifact for
Study 1 data-quality reporting. The artifact summarizes full final-clean
continuous EEG runs and remains scientifically and operationally separate from
the existing scanner-harmonic validation figure.

## Scope

The new artifact will:

- analyze final-clean continuous Study 1 EEG runs;
- estimate participant-first and cohort-level PSDs;
- show individual participants, a cohort median, and a 95% participant-bootstrap
  confidence interval;
- annotate Study 1 neural frequency bands and prespecified scanner-harmonic
  windows;
- write an SVG and exact TSV/Parquet audit tables through a standalone CLI.

The existing scanner-harmonic figure and its outputs will not change. The normal
Study 1 `report` command will not recompute continuous-data PSDs.

## Analysis Definition

### Inputs and validation

Discover numbered-participant files matching the final-clean Study 1 run naming
contract for the requested task. Apply the configured Study 1 subject exclusions
and any explicit CLI participant filters.

Each included run must:

- contain EEG channels;
- have the configured 500 Hz sampling frequency;
- contain enough samples for the configured Welch segment length;
- produce a one-dimensional 1–90 Hz frequency axis shared by every included run;
- produce finite, strictly positive PSD values.

Missing eligible files, malformed filenames, invalid arrays, unexpected sampling
rates, and inconsistent frequency axes must raise explicit errors. The analysis
must not interpolate spectra, silently skip invalid runs, or substitute fallback
settings.

### Run-level PSD

Estimate each run with MNE-Python's Welch implementation using:

- frequency range: 1–90 Hz;
- FFT and segment length: 8,192 samples;
- overlap: 4,096 samples;
- EEG channels only;
- the library's standard exclusion of samples marked by bad annotations.

These settings give a nominal frequency resolution of 500 / 8,192, approximately
0.061 Hz. Compute the channel median at every frequency in linear V²/Hz.

### Participant-first aggregation

For each participant, take the pointwise median across that participant's run
spectra in linear units. This gives every participant one spectrum regardless of
the number of retained runs. Convert the participant spectrum to dB µV²/Hz as:

`10 * log10(PSD_V2_per_Hz * 1e12)`.

### Cohort summary

At every frequency, calculate the median across participant dB spectra. Calculate
a paired percentile-bootstrap confidence interval by resampling complete
participants, preserving the frequency vector within every resample. Use the
Study 1 validity bootstrap iteration count, confidence level, and random seed.

The plotted confidence interval is descriptive uncertainty in the cohort median;
it is not a simultaneous confidence band or an inferential frequency-by-frequency
test.

## Figure Design

Use one 183 × 92 mm publication panel with a linear 1–90 Hz x-axis. The main data
layers, from back to front, are:

1. a restrained gray 95% participant-bootstrap confidence band;
2. thin, low-alpha gray participant spectra;
3. a solid black cohort-median spectrum.

Label the axes `Frequency (Hz)` and `PSD (dB µV²/Hz)`. Add a compact annotation
with the included participant and run counts. Keep the legend outside the data
region and use the existing Study 1 publication style, fonts, axis weights, and
deterministic SVG writer.

Mark the four prespecified scanner-harmonic windows with pale orange vertical
shading. Add a narrow frequency-band strip separate from the plotted data. The
strip will label:

- delta: 1.0–3.9 Hz;
- theta: 4.0–7.9 Hz;
- alpha: 8.0–12.9 Hz;
- beta: 13.0–30.0 Hz;
- low scanner-clean gamma: 30.1–38.0 Hz;
- mid scanner-clean gamma: 43.0–56.0 Hz;
- high scanner-clean gamma: 67.0–77.0 Hz.

Separating the band strip from the axes prevents overlapping neural-band and
scanner-window colors from obscuring the spectra.

## Components and Responsibilities

### Continuous PSD analysis

A shared continuous-spectrum component will own final-clean run discovery,
filename parsing, EEG-channel Welch estimation, and run-spectrum validation. The
new cohort analysis and the existing scanner-harmonic analysis will both consume
that component. The scanner-harmonic module will continue to own peak detection,
relative-power normalization, harmonic offsets, and its figure-specific summary.
The refactor must preserve the existing scanner-harmonic API and output bytes and
must not introduce compatibility wrappers or fallback behavior.

A focused cohort component will own participant-first aggregation, dB µV²/Hz
conversion, and cohort bootstrap summarization.

### Plot rendering

A pure figure builder will accept the validated cohort summary and Study 1
configuration. It will render the scientific layers and return a Matplotlib
figure without reading files or writing outputs.

### Artifact writer and CLI

A standalone writer will orchestrate analysis, rendering, deterministic SVG
serialization, and audit-table output. Its CLI will accept:

- the pipeline configuration path;
- an optional Study 1 configuration path;
- the task name;
- an optional EEG derivative root;
- repeatable optional participant filters;
- an optional SVG output path.

Without an explicit output path, the writer will use the configured Study 1
supplementary validity figure directory.

## Outputs

The artifact family consists exactly of:

- `cohort_power_spectral_density.svg`;
- `cohort_power_spectral_density_by_run.tsv`;
- `cohort_power_spectral_density_by_run.parquet`;
- `cohort_power_spectral_density_by_subject.tsv`;
- `cohort_power_spectral_density_by_subject.parquet`;
- `cohort_power_spectral_density_summary.tsv`;
- `cohort_power_spectral_density_summary.parquet`.

The run audit records source path, participant and run identifiers, channel count,
sampling frequency, sample count, recording duration, clipped union duration of
`BAD` annotations, effective analyzed duration, Welch settings, and frequency
resolution. Effective analyzed duration is recording duration minus the union of
overlapping `BAD` annotation intervals clipped to the recording bounds. The
participant table records the exact participant PSD value at each frequency. The
summary table records frequency, cohort median, confidence limits, and participant
count.

## Configuration

Add a dedicated `study1.figures.cohort_power_spectral_density` mapping to the
Study 1 figure YAML. It will contain only values that define the artifact:

- dimensions;
- frequency range;
- Welch FFT length and overlap;
- expected sampling frequency;
- scanner-harmonic and neural-band display colors.

Bootstrap settings remain shared with the existing Study 1 validity figures.
Frequency-band boundaries and scanner-harmonic windows must reuse the existing
Study 1 scientific definitions rather than duplicate competing constants.

## Verification

Automated tests will verify:

- strict configuration and input validation;
- channel-median Welch estimation in linear units;
- participant-first run aggregation;
- correct V²/Hz to dB µV²/Hz conversion;
- paired, deterministic participant bootstrapping;
- exact subject and summary schemas;
- required figure layers, labels, annotations, and legend placement;
- exact configured SVG dimensions and deterministic SVG bytes;
- exact artifact filenames and TSV/Parquet parity;
- CLI help and argument behavior without runtime warnings;
- unchanged scanner-harmonic artifact behavior.

The implementation will also run the focused Study 1 tests, Ruff on changed
Python files, and the repository's architecture and maintainability checks that
cover the touched modules.
