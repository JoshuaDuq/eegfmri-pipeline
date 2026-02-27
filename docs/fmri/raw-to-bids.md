# fMRI Raw (DICOM) -> BIDS

This repository's main pipeline now starts from BIDS-formatted fMRI inputs.
Raw DICOM-to-BIDS conversion and event-log harmonization are expected to run in an
external, dataset-specific step before `eeg-pipeline fmri preprocess` /
`eeg-pipeline fmri-analysis ...`.

## Expected output contract

Provide a valid BIDS fMRI layout with at least:

- `sub-*/func/*_bold.nii.gz`
- matching `sub-*/func/*_events.tsv`
- recommended sidecars and metadata (`*.json`, fieldmaps, `dataset_description.json`)

For GLM workflows, `events.tsv` should include:

- required BIDS columns: `onset`, `duration`, `trial_type`
- any additional columns referenced by condition selectors
  (`condition_a.column`, `condition_b.column`) and optional filters.

## Tooling note

`dcm2niix` is still a standard choice for DICOM conversion, but this repository does
not provide a single built-in raw-to-BIDS CLI wrapper for all paradigms.
