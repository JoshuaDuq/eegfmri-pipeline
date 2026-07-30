# Conversion: source recordings into BIDS

**Entrypoint:** `python studies/pain_study/scripts/run_paradigm_specific.py <command>`

## Why this folder exists

Everything else in the repository assumes BIDS. These scripts are what turns a session as
it comes off the scanner — BrainVision triplets, DICOM series, PsychoPy CSVs — into that.
They run **before** any `eeg-pipeline` command and are deliberately outside the main CLI:
they encode conventions specific to this pain paradigm, and a paradigm-agnostic pipeline
should not carry them.

They are the only scripts here that must run in a fixed order, because each one's output is
the next one's input.

## The files

| File | What it contributes |
|---|---|
| `organize_source_eeg.py` | Stages newly acquired recordings into the `sub-*/eeg/original_untrimmed_5khz/` layout the rest of the conversion expects. Run first, on fresh acquisitions only. |
| `eeg_raw_to_bids.py` | BrainVision `.vhdr` → BIDS EEG via `mne-bids`. Applies the `easycap-M1` montage and the `Trig_therm/T  1` thermode trigger convention, and optionally trims to the volume-trigger bounds so EEG and fMRI share a clock. |
| `fmri_raw_to_bids.py` | DICOM → BIDS fMRI via `dcm2niix`, including rest runs and fieldmaps, and writing phase-level `events.tsv`. |
| `merge_psychopy.py` | Merges PsychoPy `TrialSummary.csv` behavioural columns into the BIDS `events.tsv` written by `eeg_raw_to_bids`. Must run after it — it matches row-for-row and refuses a misaligned merge unless told otherwise. |
| `fix_fmri_bids_outputs.py` | Repairs BIDS fMRI outputs after the fact when a conversion needs patching rather than redoing. |
| `sanitize_brainvision_vas_markers.py` | Repairs VAS rating markers in BrainVision files whose marker stream violates the invariant downstream code relies on. |
| `export_brainvision_matlab.py` | Exports recordings in the form the MATLAB/FieldTrip side of the project reads. |

## Order

```
1. organize_source_eeg     (fresh acquisitions only)
2. eeg-raw-to-bids         → BIDS EEG
3. fmri-raw-to-bids        → BIDS fMRI
4. merge-psychopy          → behaviour into events.tsv
```

Steps 2–4 must complete before any `eeg-pipeline` command. Full flag documentation is in
the [scripts README](../README.md).

## Adapting to another paradigm

These encode this paradigm's conventions: BrainVision format and `easycap-M1`; the
`Trig_therm/T  1` trigger prefix; DICOM input with phase-level events; PsychoPy
`TrialSummary.csv` with `run_id`, `stim_start_time`, `stimulus_temp` and `condition`.
Change them here. Do not change `eeg_pipeline/` or `fmri_pipeline/` to suit one paradigm.
