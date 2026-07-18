# BrainVision Analyzer FieldTrip ICA/TFR Runner Design

## Objective

Run the existing MATLAB/FieldTrip ICA and component-space TFR workflow for `sub-0015`
using the pre-ICA filtered FIF files produced from the BrainVision Analyzer-corrected
1 kHz recordings. Keep every export and MATLAB result separate from the native
EEG-fMRI correction analysis.

## Inputs and outputs

The workflow reads:

- BIDS events from `/Volumes/KINGSTON/EEG_fMRI_data/bids_output/eeg`;
- pre-ICA FIF files from
  `/Volumes/KINGSTON/EEG_fMRI_data/derivatives/brainvision_analyzer_mne_preprocessing_sub-0015/preprocessed/eeg`;
- the installed FieldTrip tree at
  `/Users/joduq24/Documents/MATLAB/fieldtrip-master`.

All generated exports, ICA files, provenance, and TFR results are written under:

`/Volumes/KINGSTON/EEG_fMRI_data/derivatives/fieldtrip_brainvision_analyzer_sub-0015`

## Components

Add one comparison-specific YAML configuration beside the existing FieldTrip config.
It retains the established study, epoch, ICA, and TFR parameters while replacing only
the MNE derivative and output roots.

Add one MATLAB runner dedicated to this comparison. The runner invokes the existing
Python FieldTrip exporter for `sub-0015`, receives the generated runtime JSON path,
then calls `prepareIcaBatch` and `computeTfrBatch` with an explicit `Subjects="sub-0015"`
selection.

The existing exporter remains the only FIF-to-FieldTrip conversion boundary. The
MATLAB runner does not directly parse FIF files or reconstruct event metadata.

## Data flow

1. Validate six pre-ICA filtered FIF runs and six matching BIDS event tables.
2. Build 66 aligned broadband epochs and a separate 1 Hz high-pass ICA-fit copy.
3. Export both FieldTrip structures and trial metadata to one subject package.
4. Fit deterministic pooled-run extended `runica` using the configured seed.
5. Apply the unmixing matrix to the broadband epochs.
6. Compute the existing condition and temperature component-space TFR outputs.
7. Save export, ICA, provenance, and TFR artifacts only in the comparison output root.

## Error handling

Missing inputs, inconsistent channels or bad-channel sets, trial-count mismatches,
non-finite samples, missing FieldTrip functions, and existing outputs all raise errors.
There are no fallback readers, inferred paths, compatibility shims, or silent
overwrites. The MATLAB runner exposes one `overwrite` variable, defaulting to `false`,
and passes that choice consistently to all three stages.

The runner checks the Python exporter exit status and stops before MATLAB ICA when the
export fails.

## Verification

Before delivery:

- run the exporter in validation-only mode for `sub-0015` with the new config;
- check the MATLAB runner and config for syntax and path consistency;
- if MATLAB is callable in the environment, execute the runner with overwrite disabled
  against a fresh output root;
- otherwise, report that MATLAB execution remains a user-side verification step and
  provide the exact runner path.

