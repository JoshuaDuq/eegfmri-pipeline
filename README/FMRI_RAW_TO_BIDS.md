# fMRI Raw (DICOM) → BIDS Utility

This project includes a **fMRI raw-to-BIDS** converter for the simultaneous EEG–fMRI thermal pain study.

## What it does

- Converts per-series DICOM folders under `data/source_data/sub-*/fmri/` into a BIDS fMRI dataset (NIfTI + JSON).
- Writes task `*_events.tsv` from PsychoPy `*TrialSummary.csv` for each pain run.
- Optionally links or copies the original DICOMs into `<bids_fmri_root>/sourcedata/` for provenance.

## Prerequisite: `dcm2niix`

Conversion requires the external binary `dcm2niix` available on `PATH`, or passed explicitly via `--dcm2niix-path`.

## CLI usage

```
eeg-pipeline utilities fmri-raw-to-bids --all-subjects --task thermalactive
```

Common options:

- `--source-root <path>`: where `sub-*/fmri/` lives (default from config: `paths.source_data`)
- `--bids-fmri-root <path>`: output BIDS fMRI root (default from config: `paths.bids_fmri_root`)
- `--session 01`: optional BIDS session label
- `--rest-task rest`: task label to use for resting-state series
- `--dicom-mode symlink|copy|skip`: how to store original DICOMs under `sourcedata/`
- `--no-rest`, `--no-fieldmaps`: skip those series types
- `--no-events`: do not generate `*_events.tsv`
- `--event-granularity phases|trial`: stimulation modeling
- `--onset-reference as_is|first_iti_start|first_stim_start` and `--onset-offset-s <sec>`

## TUI usage

In the TUI:

- Main Menu → Utilities → `fMRI Raw to BIDS`
- Configure advanced options (session/rest-task/events/dcm2niix), select subjects, then execute.

## Study paradigm → `events.tsv`

Events are generated per run from PsychoPy `TrialSummary.csv` timing columns. The default (recommended) is:

- `fixation_rest`: ITI fixation interval (15–20 s; thermode at 35°C)
- `stimulation`: split into `stim_phase = ramp_up (3 s), plateau (7.5 s), ramp_down (~2 s)` when `--event-granularity phases`
- `fixation_poststim`: fixation between stimulation end and pain question onset (random 4.5–8.5 s)
- `pain_question`: “Was it painful?” window (can end early on response)
- `vas_rating`: VAS window (can end early on response)

A dataset-level column dictionary is written to `task-<task>_events.json`.

