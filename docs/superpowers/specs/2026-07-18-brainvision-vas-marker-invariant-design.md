# BrainVision VAS Marker Invariant Design

## Problem

BrainVision encodes event type and description separately. The study recordings contain
`Vas_on,V  1`, while scanner pulses contain `Volume,V  1`. Applications that key on the
description alone therefore confuse VAS events with scanner pulses. The existing sanitation
derivative covers selected original 5 kHz recordings, but Analyzer-processed 1 kHz recordings
and other repository entry points can still receive the ambiguous marker.

## Required invariant

Every BrainVision recording accepted by EEG-fMRI code must use `Vas_on,VAS_ON` for VAS onset
and `Volume,V  1` for scanner volumes. An encountered `Vas_on,V  1` is an error at analysis
entry points. Sanitation is explicit and non-destructive; it is never performed silently while
loading analysis data.

## Sanitation derivative

The sanitation command will accept explicit BrainVision header paths, stage a header and marker
file for each recording, and reference the existing `.eeg` signal file. It will support both
original 5 kHz and Analyzer-processed 1 kHz recordings without imposing a sampling-frequency
value. Output paths mirror each recording's path relative to the source-data root, preventing
filename collisions and retaining provenance.

For each marker file, the command will change only the description of records whose type is
exactly `Vas_on` and whose description is exactly `V  1`. It will reject unexpected uses of
`V  1`, already-sanitized inputs, malformed records, invalid coordinates, missing volume
markers, or missing VAS markers. It will publish atomically and refuse to overwrite output.

Verification reopens both source and staged recordings with MNE. Sampling frequency, sample
count, measurement date, channel order, annotation onset, annotation duration, and sampled EEG
values must match. Annotation descriptions must differ only by the declared VAS replacement.
The manifest records source and staged paths, sampling frequency, marker counts, hashes, and
signal-file metadata.

## Analysis entry points

The shared EEG-fMRI marker validator will reject any remaining `Vas_on/V  1` annotation before
volume extraction. The sub-0015 MATLAB exporter will invoke the same validation immediately
after reading each run, so it cannot create an export from ambiguous source metadata. Its
default source will be the processed 1 kHz sanitation derivative, not the mutable source-data
directory.

## Cohort publication

The canonical derivative is `brainvision_marker_sanitized-v2`. It contains both
`original_untrimmed_5khz` and `brainvision_processed_1khz` layouts for all selected thermal-task
recordings. Source marker and signal files remain unchanged. Consumers are updated to v2 in one
change; there is no fallback to v1 or unsanitized source data.

## Tests

Regression tests cover staging a 1 kHz recording, preserving relative layout, accepting both
sampling frequencies, exact marker replacement, rejection by analysis entry points, and the
MATLAB exporter's rejection of ambiguous input. Focused tests must demonstrate red before the
implementation and green afterward. Full preprocessing and script test groups, Ruff, and an
inventory audit of the published derivative complete verification.
