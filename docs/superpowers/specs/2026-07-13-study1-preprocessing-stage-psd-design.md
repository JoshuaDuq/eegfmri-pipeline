# Study 1 Preprocessing-Stage Power Spectral Density Design

## Purpose

Extend the reproducible Study 1 cohort power spectral density reporting artifact
to the three EEG checkpoints stored on the Kingston volume:

1. original BrainVision recordings at 5,000 Hz;
2. BrainVision-processed recordings at 1,000 Hz;
3. final MNE-preprocessed recordings at 500 Hz.

Each checkpoint will produce an independent cohort report. The implementation
will not claim that a separate 5,000-Hz post-gradient-correction checkpoint
exists because no such files are stored on the volume.

## Source Contracts

### Original BrainVision recordings

Discover participant `raw.zip` archives below the configured Kingston source
root. Within each archive, select complete BrainVision triplets whose names match
the Study 1 thermal-task run contract. Read one triplet at a time through a
temporary extraction directory and record the archive and member path in the
audit table.

Most participants store the same triplets in an uncompressed `raw` directory;
the discoverer supports these two explicit source representations. Participant
directories must follow the canonical fMRI-session naming contract. Acquisition
folders suffixed `_EXCL` and follow-up folders suffixed `_eeg_only` are not part
of this cohort.

Every selected header must declare a sampling frequency of 5,000 Hz. Duplicate
participant/run identifiers, missing triplet members, malformed filenames, and
unexpected sampling frequencies are errors.

### BrainVision-processed recordings

Discover complete `*_scannerpulse_corrected` BrainVision triplets below the
configured Kingston source root. Select only the Study 1 thermal-task runs and
require every selected header to declare 1,000 Hz.

Duplicate participant/run identifiers, missing triplet members, malformed
filenames, and unexpected sampling frequencies are errors. Files inside system
trash and macOS metadata files are outside the discovery scope.

### Documented source corrections

Two exact `sub-0003` acquisition issues are declared in Study 1 YAML rather than
handled by permissive filename parsing:

- the full recording at `11h30.23.962` reused the run-1 task sequence and is
  identified as run 3;
- the earlier `11h10.39.899` recording is an aborted 8.76-second run-1 start and
  is excluded from the full-run cohort.

The corresponding processed run is stored under a parenthetical run-3 filename
while its BrainVision header and marker still reference run-1 filenames. The
manifest requires the exact observed header, data, marker, and internal-reference
names. The loader materializes a corrected temporary triplet for MNE without
modifying Kingston. Any mismatch or missing manifest entry raises an error. The
run audit records the correction reason for every corrected source.

### Final MNE-preprocessed recordings

Reuse the existing final-clean FIF discovery contract below the configured EEG
derivative root. Every selected file must declare 500 Hz.

All three stages apply the configured Study 1 participant exclusions and any
explicit CLI participant filters. Each stage is summarized independently from
all eligible runs available at that checkpoint; the implementation will not
silently intersect or discard runs to manufacture a paired cohort.

## Spectral Analysis

All checkpoints use the same 16.384-second Welch segment duration and 50%
overlap. The implementation derives integer sample counts from the validated
sampling frequency:

| Checkpoint | Sampling frequency | Segment samples | Overlap samples |
| --- | ---: | ---: | ---: |
| Original BrainVision | 5,000 Hz | 81,920 | 40,960 |
| BrainVision processed | 1,000 Hz | 16,384 | 8,192 |
| Final MNE processed | 500 Hz | 8,192 | 4,096 |

This preserves the existing final-clean frequency grid and gives every report a
nominal resolution of approximately 0.061 Hz from 1–90 Hz. The estimator will
not resample, interpolate, filter, or otherwise alter source recordings.

Before selecting EEG channels from BrainVision files, apply the pipeline's
established channel typing so the dedicated ECG channel is excluded. For every
run, calculate Welch PSDs for EEG channels, reject samples marked by `BAD`
annotations when present, and take the pointwise channel median in linear V²/Hz.

Aggregate runs and participants exactly as in the existing cohort PSD artifact:

1. pointwise run median within each participant in linear units;
2. conversion to dB µV²/Hz;
3. pointwise cohort median across participant spectra;
4. deterministic paired participant bootstrap confidence interval.

## Software Design

A source-stage model will define the stage identifier, display label, expected
sampling frequency, and source kind. Separate discoverers will own BrainVision
archive discovery, BrainVision directory discovery, and final-clean FIF
discovery. They will return one common immutable run-source structure.

The continuous-spectrum estimator will accept a validated raw recording plus
run identity, keeping file-format handling separate from spectral analysis. A
small loader boundary will read each source kind and close or remove temporary
resources immediately after the run spectrum is calculated.

A dedicated stage artifact writer will orchestrate discovery, estimation,
aggregation, plotting, and table serialization. The existing final-clean cohort
PSD command and its output filenames will remain unchanged.

The new CLI will require one or more explicit stage selections from:

- `raw`;
- `processed`;
- `mne`.

It will accept the Kingston source root, EEG derivative root, task, configuration
paths, repeatable participant filters, and output directory. All requested
stages will be discovered and validated before any report is written, preventing
partial output when a selected input contract fails.

## Outputs

Each selected stage writes one SVG and three TSV/Parquet audit-table pairs using
the stage identifier in every filename:

- `cohort_power_spectral_density_<stage>.svg`;
- `cohort_power_spectral_density_<stage>_by_run.tsv` and `.parquet`;
- `cohort_power_spectral_density_<stage>_by_subject.tsv` and `.parquet`;
- `cohort_power_spectral_density_<stage>_summary.tsv` and `.parquet`.

Every table contains the stage identifier. Run audits additionally record the
source representation, source path, configured source correction, channel count,
sampling frequency, sample count, durations, Welch segment duration, FFT and
overlap sample counts, and frequency resolution. The three figures share dimensions, axes, scientific
annotations, colors, and visual hierarchy with the existing cohort PSD report.
Each figure states its checkpoint, sampling frequency, participant count, and run
count.

## Configuration

The Study 1 figure YAML will contain only the necessary stage-report settings:

- stage display labels and expected sampling frequencies;
- common segment duration and overlap proportion;
- default output directory and figure dimensions.

Frequency range, exclusions, bootstrap settings, neural bands, scanner windows,
and publication colors remain shared with the existing Study 1 definitions.
Source roots remain explicit CLI inputs rather than machine-specific YAML paths.

## Verification

Automated tests will cover:

- strict discovery and filename parsing for all source kinds;
- archive triplet completeness and duplicate detection;
- exact sampling-frequency validation;
- ECG exclusion from BrainVision EEG picks;
- equal segment duration and frequency axes across sampling rates;
- unchanged participant-first aggregation and bootstrap behavior;
- stage columns and exact output filenames;
- all-input validation before writes;
- deterministic SVG and TSV/Parquet parity;
- unchanged existing final-clean cohort PSD behavior.

The implementation will run focused Study 1 tests, Ruff on changed Python files,
and relevant architecture and maintainability checks. It will then generate and
visually inspect the raw, processed, and MNE reports from the Kingston data.
