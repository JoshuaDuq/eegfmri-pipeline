# MNE Scanner-Harmonic Cohort Comb Design

## Objective

Automatically generate one publication-quality scanner-harmonic comb plot whenever task-based
MNE preprocessing produces final cleaned epochs. The figure compares the MRI-corrected BIDS
input with the final MNE-cleaned epochs and weights every participant equally.

## Scope

The QC step runs at the end of `full` and `epochs` preprocessing modes. It does not run for
`bad-channels` or `ica`, because those modes do not produce final cleaned epochs. The feature
produces one task-level cohort figure and one numerical spectrum table for the subjects selected
in the current preprocessing invocation.

## Spectral Estimation

Both stages use Welch spectra from 15 to 90 Hz on a fixed 0.25 Hz frequency grid. The identical
grid permits pointwise comparison even when the BIDS input and cleaned epochs have different
sampling frequencies.

- **MRI-corrected input:** compute Welch power from each run's BIDS raw EEG.
- **Final MNE:** compute Welch power independently within each retained cleaned epoch; never
  concatenate epochs across discontinuous boundaries.
- Use EEG channels only. Exclude channels marked bad at the relevant stage.
- Convert power to dB only after validating that every analyzed value is finite and positive.

Aggregation is hierarchical:

1. summarize Welch segments and channels within each run or epoch;
2. summarize runs or epochs within each participant;
3. summarize participant spectra with an equal-weight cohort median;
4. compute deterministic 95% participant-bootstrap confidence intervals with 10,000 resamples
   and the configured project random seed.

Participants must have both input and final spectra. Missing stage data, an empty retained-epoch
set, incompatible frequency grids, non-finite power, or an empty EEG channel selection is an
error. The implementation does not silently omit failed participants.

## Harmonic Definition

Reuse the established scanner-harmonic windows from
`eeg_pipeline.analysis.qc.scanner_harmonics.DEFAULT_HARMONIC_WINDOWS`. Within each window, the
displayed comb reference is the strongest cohort input peak selected by the established peak
selection rule. This keeps the MNE QC definition aligned with the native EEG-fMRI correction QC.

## Figure

The output is a single, wide comb plot:

- x-axis: frequency in Hz, 15–90 Hz;
- y-axis: PSD in dB V²/Hz;
- MRI-corrected BIDS input: neutral gray line and confidence band;
- final MNE-cleaned epochs: blue line and confidence band;
- four scanner-harmonic windows: subtle neutral shading;
- input-derived comb frequencies: thin dotted vertical reference lines;
- direct legend labels and participant count;
- restrained scientific styling consistent with the native correction spectrum QC;
- 300-DPI PNG output.

The plot contains no secondary panels, metric cards, or unrelated preprocessing diagnostics.

## Numerical Output

Write an adjacent TSV with one row per frequency and these columns:

- `frequency_hz`;
- `input_median_db`, `input_ci_low_db`, `input_ci_high_db`;
- `final_median_db`, `final_ci_low_db`, `final_ci_high_db`;
- `n_participants`.

The TSV and figure describe the same selected subjects and frequency bins.

## Pipeline Integration

Introduce a dedicated preprocessing step named `scanner-harmonic-qc` after statistics collection
for `full` and `epochs` modes. The step receives the selected subjects, task, BIDS root,
derivative root, and validated QC configuration. Spectral computation, cohort aggregation,
plotting, and orchestration remain separate focused functions.

Outputs are written under:

`<deriv-root>/preprocessed/eeg/qc/`

with task-specific names:

- `task-<task>_desc-scannerharmoniccomb_qc.png`;
- `task-<task>_desc-scannerharmoniccomb_qc.tsv`.

Successful output paths are included in preprocessing run metadata. QC errors fail the pipeline
and surface the exact participant, stage, and violated assumption.

## Configuration

Keep configuration minimal in `eeg_pipeline/utils/config/eeg_config.yaml`:

- frequency range: 15–90 Hz;
- Welch duration: 4 seconds;
- frequency resolution: 0.25 Hz;
- bootstrap resamples: 10,000;
- confidence level: 0.95.

The scanner windows and project random seed remain single-source values from their established
locations rather than being duplicated.

## Validation

Tests cover:

- identical frequency grids across different sampling frequencies;
- epoch-wise estimation without boundary concatenation;
- participant-equal aggregation when run counts differ;
- deterministic paired participant bootstrap intervals;
- strict failures for missing stages, empty epochs, bad channels, and invalid spectra;
- correct harmonic shading and reference markers;
- task-specific PNG and TSV paths;
- automatic execution in `full` and `epochs`, but not incomplete preprocessing modes.

