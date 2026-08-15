# Decomb Spectral Availability Integration Design

**Date:** 2026-08-15
**Status:** Approved for implementation planning

## Purpose

Integrate Decomb's recording-specific unavailable-frequency intervals into the EEG
pipeline without changing behavior for datasets that do not explicitly opt in. The
pipeline must distinguish attenuation caused by filtering from evidence of weak neural
activity. It must retain usable broad-band information while preventing inference from
frequencies that a notch and its transitions have changed.

## Context

Decomb writes a BrainVision BIDS derivative and a root-level
`line_notch_manifest.tsv`. Each finite `unavailable_low_hz` to
`unavailable_high_hz` interval includes a stopband and its FIR transitions. The manifest
is recording-specific and can contain:

- repeated interval geometry on channel-evidence rows;
- multiple adaptive removal rounds;
- overlapping intervals across rounds;
- one terminal-null row per recording with empty interval fields.

The example derivative at
`/Volumes/KINGSTON/EEG_fMRI_data/bids_output/eeg_decombed_auto` contains 90 recordings:
15 participants with six runs each. Its manifest has 90 recording identifiers and one
terminal-null row for every recording. The downstream MNE output at
`/Volumes/KINGSTON/EEG_fMRI_data/derivatives/preprocessed/eeg` preserves the six runs as
`run_id` values in each participant's consolidated clean-events table, with 11 retained
epochs per run. This gives the analysis pipeline an exact recording-to-epoch join.

The example downstream run also used `preprocessing.notch_freq: 60`. Decomb had already
made 60 Hz unavailable in 68 recordings and retained it in 22 recordings. The fixed
downstream notch therefore changed data beyond the Decomb manifest. Future derivatives
using this integration must disable that second notch and be regenerated.

## Goals

1. Preserve the current EEG pipeline exactly when no Decomb manifest is configured.
2. Load and validate Decomb exclusions only through an explicit configuration path.
3. Resolve exclusions independently for every BIDS recording and every aligned epoch.
4. Exclude unavailable bins from broad-band PSD integration and normalize by retained
   bandwidth.
5. Mask unavailable frequencies in frequency-resolved and spectral-model analyses.
6. Reject band-filtered analyses whose requested band intersects an unavailable interval.
7. Produce a machine-readable availability audit without changing existing feature-table
   schemas.
8. Fail before expensive processing when provenance, geometry, or recording alignment is
   invalid.

## Non-goals

- Reconstructing or imputing neural activity removed by a notch.
- Correcting power by dividing by a retained-bandwidth percentage.
- Encoding frequency exclusions as MNE temporal annotations or custom `Raw.info` fields.
- Applying a cohort-wide union of recording-specific notches.
- Supporting arbitrary manifest formats or a plugin registry in the first implementation.
- Reworking unrelated preprocessing, feature extraction, or report architecture.
- Using the mounted 90-recording dataset as an automated test dependency.

## Configuration and activation

The integration is activated only by an explicit path:

```yaml
paths:
  decomb_manifest: "/path/to/line_notch_manifest.tsv"

preprocessing:
  notch_freq: null
```

When `paths.decomb_manifest` is absent or null, no Decomb module is imported, no spectral
availability validation runs, and all existing code paths and outputs remain unchanged.

When the path is configured:

- the file must exist;
- its sibling `dataset_description.json` must describe a BIDS derivative with exactly one
  `GeneratedBy` entry named `decomb`;
- `preprocessing.notch_freq` must be null;
- the manifest must cover every recording contributing epochs to the requested analysis;
- every aligned epoch must identify a run and match exactly one manifest recording.

The explicit path is authoritative. The pipeline does not search parent directories or
guess a manifest from the BIDS root.

## Chosen approach and alternatives

The chosen design is a generic spectral-availability contract with one isolated Decomb
adapter. This keeps analysis semantics independent of the producer while limiting
Decomb-specific parsing and provenance checks to one boundary.

Two alternatives were rejected:

- feature-specific Decomb handling would duplicate interval logic across PSD, TFR,
  spectral-model, Hilbert, burst, and connectivity paths and make omissions likely;
- encoding exclusions in MNE annotations or `Raw.info` would overload temporal metadata
  with frequency-domain meaning for which MNE has no standard representation.

## Architecture

### Generic spectral-availability model

A small generic module owns immutable data structures for:

- a BIDS recording key: subject, task, run, and optional session;
- one finite closed frequency interval;
- the sorted, merged exclusion intervals for one recording;
- epoch-aligned recording keys and exclusions.

The model has no dependency on Decomb, pandas, MNE, or feature extractors. Construction
validates finite interval edges, strictly positive widths, sorted non-overlapping merged
geometry, unique recording keys, and one exclusion set per recording.

The model exposes focused operations:

- build a valid-frequency mask for supplied estimator centres and their spectral
  half-support widths;
- calculate interval overlap with a requested band;
- calculate retained bandwidth from frequency-bin integration weights;
- determine whether a contiguous band is eligible for band-filtered analysis.

An estimate is valid only when its spectral support does not overlap an unavailable
interval. Testing the centre frequency alone is insufficient because Welch windows,
multitapers, and wavelets smooth across neighbouring frequencies.

### Decomb adapter

One adapter reads `dataset_description.json` and `line_notch_manifest.tsv`. It validates
the manifest columns needed by the integration:

- `recording`;
- `unavailable_low_hz`;
- `unavailable_high_hz`;
- `outcome`;
- `removal_round`.

It parses BIDS entities from `recording`, drops no information by row order, and accepts
empty interval fields only on the single terminal row whose `outcome` is
`no_line_detected`. It takes every finite interval from every removal round, removes
exact duplicates, merges overlapping or touching intervals, and constructs one generic
exclusion set per recording. Evidence columns and the precomputed band-share columns are
not used to reconstruct geometry.

The adapter also computes the manifest SHA-256 checksum for provenance. The generic
analysis layer receives only validated data structures and the checksum; it never imports
or interprets Decomb.

### Pipeline integration boundary

The feature pipeline loads the configured manifest once, after loading clean epochs and
aligned events but before computing shared spectral intermediates. For each participant:

1. subject and task come from the feature context;
2. run comes from the clean-events `run_id` column;
3. session comes from aligned events when the dataset has sessions;
4. numeric identifiers such as `1.0` are canonicalized to the BIDS value `1` only when
   they are finite integer-valued numbers;
5. each epoch key must match exactly one manifest key.

Missing `run_id`, non-integral run identifiers, absent recordings, duplicate manifest
keys, or ambiguity caused by missing session identity raise errors. The pipeline never
falls back to participant-level exclusions, a cohort union, or event order.

The resulting epoch-aligned availability is attached to the feature context and shared
precomputed-data container. Existing consumers that do not perform spectral analysis do
not receive or inspect it.

## Scientific behavior

### PSD band power

PSD is estimated from the filtered data as it is today. For each epoch independently:

1. derive frequency-bin integration weights from the returned frequency grid;
2. select bins inside the requested band;
3. remove estimates whose spectral support intersects that epoch's unavailable
   intervals;
4. integrate only retained `PSD * bin_width` values;
5. when bandwidth normalization is requested, divide by the sum of retained bin widths.

The denominator is never the nominal band width when bins have been removed. No scaling
attempts to reconstruct the missing neural power. An epoch with no retained bandwidth is
unavailable for that band. If no eligible epoch remains for a requested output, the
analysis raises an error.

The support width comes from the estimator actually used:

- multitaper PSD uses MNE's configured full `bandwidth`, whose documented smoothing
  interval is `centre ± bandwidth / 2`;
- Welch PSD measures the half-power main-lobe support of the exact configured window and
  `n_per_seg` geometry;
- an estimator with missing or invalid resolution metadata raises an error when
  exclusions are active.

These rules are deterministic and shared by masking and audit generation. They do not
pretend that a displayed Fourier coordinate is an infinitesimal observation.

### Frequency-resolved PSD and TFR

Unavailable frequency cells are represented as `NaN` together with an explicit validity
mask. A cell is unavailable when the estimator's spectral support overlaps an exclusion:
multitaper TFR uses its configured full bandwidth, and Morlet TFR measures the half-power
support of the exact wavelet defined by frequency, `n_cycles`, and sampling rate. Averages
use only eligible epochs at each frequency and carry the contributing epoch count. A
frequency with no eligible epoch remains unavailable rather than being filled or
interpolated.

Plots may show gaps or shaded unavailable regions, but plotting changes are limited to
consuming the same mask. Plotting does not define scientific validity.

### Peaks, IAF, and aperiodic fits

Peak searches, individual-alpha-frequency estimation, and aperiodic fitting receive only
retained frequency bins. A peak cannot be selected inside an unavailable interval. Each
algorithm must retain its existing minimum-support requirements after masking. If masking
leaves insufficient or disconnected support for the requested fit, that estimate is
unavailable and the reason is recorded.

### Hilbert, phase, burst, and connectivity features

These analyses require a contiguous requested passband. A notch creates disconnected
spectral support, and phase across disconnected sub-bands has no single interpretation.
Therefore, an epoch or run is ineligible for these feature families whenever its
unavailable intervals overlap the requested band.

Trialwise outputs mark epochs from ineligible runs unavailable. Cross-trial estimates use
only eligible runs and report both contributing run and epoch counts. They must continue
to satisfy their existing minimum-data requirements. If no eligible data remain, the
analysis raises an error instead of returning a numerical suppression as neural activity.

### Non-spectral features

ERP and other non-spectral computations continue to consume the filtered time series.
The integration applies no additional transform and no automatic frequency masking to
them. Their provenance records that the source samples were Decomb-filtered. This design
does not claim that notch filtering leaves time-domain estimates unaffected, and it does
not redefine time-domain inference.

## Preprocessing conflict validation

When `paths.decomb_manifest` is configured, preprocessing preflight requires
`preprocessing.notch_freq: null`. The purpose is not merely to avoid redundant
computation: another notch changes the unavailable-frequency geometry beyond the
authoritative manifest.

The first implementation does not derive geometry for an additional MNE notch or merge
multiple filter-provenance sources. A user who intentionally needs another notch must
produce a new authoritative availability manifest before downstream spectral inference.

The existing example derivatives under `derivatives/preprocessed/eeg` were generated with
an additional fixed 60 Hz notch. They remain useful as read-only structural evidence but
must be regenerated from `bids_output/eeg_decombed_auto` with the fixed notch disabled for
the new availability contract to be exact.

## Availability audit

Each subject-level feature run writes a companion
`sub-<subject>_task-<task>_desc-spectralavailability.tsv` beside its feature outputs. This
avoids concurrent cohort-file mutation and does not add columns to existing feature
tables. Each row contains:

- subject, task, run, and optional session;
- analysis band or frequency-grid identifier;
- unavailable intervals intersecting the requested range;
- nominal bandwidth in Hz;
- retained bandwidth in Hz;
- retained share;
- estimator type and spectral-support rule;
- eligibility for PSD integration;
- eligibility for contiguous-band analyses;
- numbers of aligned, eligible, and ineligible epochs;
- manifest SHA-256 checksum.

The audit is written atomically. Repeated rows for the same recording and analysis target
are an error.

## Error handling

The integration fails early and specifically for:

- a configured manifest path that does not exist;
- missing or invalid Decomb provenance;
- missing required manifest columns;
- non-finite, reversed, or zero-width interval geometry;
- an empty interval on a non-terminal row;
- multiple terminal-null rows or no terminal-null row for a recording;
- duplicate recording keys with conflicting identity;
- epochs without a usable run or required session identifier;
- aligned epochs that cannot be matched to one manifest recording;
- an enabled downstream notch;
- no retained frequency support for a requested output;
- insufficient retained support for an existing estimator's requirements.

Unexpected exceptions are not caught and converted into fallback behavior.

## Testing strategy

### Compatibility tests

- Run representative existing spectral tests with no configured manifest and require
  unchanged values, shapes, and metadata.
- Assert that manifest loading is not called when the path is absent or null.
- Retain current configuration behavior for non-Decomb datasets.

### Unit tests

- Validate immutable interval and recording-key construction.
- Merge duplicate, overlapping, touching, and multi-round intervals.
- Accept one terminal-null row and reject malformed null evidence.
- Parse names such as `sub-0000_task-thermalactive_run-1_eeg`.
- Canonicalize aligned `run_id` values such as `1.0` through `6.0`.
- Reject fractional, missing, infinite, and ambiguous identifiers.
- Exercise masks and retained-bandwidth calculations on regular and irregular grids.
- Verify estimator-support expansion for Welch, multitaper, and Morlet calculations.
- Verify per-epoch PSD normalization against hand-calculated values.
- Verify complete-band exclusion and zero-retained-bandwidth failures.

### Consumer tests

- PSD band power excludes and renormalizes unavailable bins per epoch.
- Frequency-resolved PSD and TFR preserve shapes and mask expected cells.
- Frequency averages report the correct eligible-epoch counts.
- Peak, IAF, and aperiodic estimators cannot consume masked frequencies.
- Hilbert, phase, burst, and connectivity consumers reject overlapping bands while
  retaining non-overlapping bands.
- Cross-trial consumers enforce existing minimum-data requirements after eligibility
  filtering.
- Preprocessing preflight rejects a configured Decomb manifest together with a non-null
  notch.

### Integration test

Use a synthetic two-run participant with:

- one small Decomb-format manifest;
- duplicate evidence rows and multiple removal rounds;
- one terminal-null row per run;
- clean events containing float-valued `run_id` values;
- distinct exclusions in each run;
- one broad-band PSD output, one masked TFR output, and one rejected contiguous-band
  feature.

The test verifies exact run-specific behavior and the availability audit. The mounted
90-recording dataset is never required by the test suite.

## Implementation boundaries

The implementation may add small focused modules for the generic model, Decomb adapter,
and analysis helpers, then thread the optional object through existing contexts and
spectral entry points. It must not refactor unrelated feature code or modify the active
study-specific line-comb implementation.

The EEG repository currently contains unrelated uncommitted work. Implementation and
commits must preserve it and stage only files belonging to this integration.

## Acceptance criteria

The implementation is accepted when:

1. all existing tests pass without a configured manifest;
2. no-manifest spectral outputs are unchanged;
3. the synthetic two-run integration test demonstrates recording-specific behavior;
4. broad-band PSD uses retained-bandwidth normalization;
5. frequency-resolved outputs expose gaps rather than suppressions interpreted as data;
6. contiguous-band analyses reject notch overlap;
7. double notching fails preflight;
8. each subject-level spectral-availability TSV fully audits every analyzed recording and
   target;
9. the real 90-recording structure can be validated read-only with exact manifest-to-run
   coverage;
10. regenerated MNE derivatives use `preprocessing.notch_freq: null`.
