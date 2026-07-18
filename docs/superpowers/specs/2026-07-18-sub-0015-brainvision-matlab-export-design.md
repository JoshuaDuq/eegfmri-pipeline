# sub-0015 BrainVision MATLAB Export Design

## Objective

Create a shareable MATLAB package for `sub-0015` from the subject's BrainVision
Analyzer-processed 1 kHz thermal-task recordings. The package will contain trial-locked EEG
epochs and a separate, exactly aligned trial-information file.

## Source Data

The exporter will read only the six thermal runs under:

`/Volumes/KINGSTON/EEG_fMRI_data/source_data/sub-0015/eeg/brainvision_processed_1khz`

Each run must contain a complete `*_scannerpulse_corrected.vhdr`, `.vmrk`, and `.eeg`
BrainVision triplet. The baseline recording and the `original_5khz` directory are outside the
export scope.

Trial metadata will come from the six current run-level BIDS event tables under:

`/Volumes/KINGSTON/EEG_fMRI_data/bids_output/eeg/sub-0015/eeg`

Only rows with `trial_type == "Trig_therm/T  1"` and a populated `stimulus_temp` are thermal
trials. Each run must provide exactly 11 such trials, ordered by `trial_number` 1 through 11.

## Epoch Definition

Each trial will be locked to the matching `Trig_therm/T  1` BrainVision annotation. Time zero
is the thermode trigger. Each epoch spans -7.0 through +15.0 seconds, inclusive, at the source
sampling frequency of 1,000 Hz. No baseline correction, filtering, resampling, rereferencing,
artifact rejection, or channel interpolation will be applied during export.

The package will contain 66 epochs ordered first by `run_id` 1 through 6 and then by
`trial_number` 1 through 11. All 64 recorded channels will be preserved in their source order,
including `ECG`. BrainVision samples will be exported in volts, which is the physical unit
returned by MNE after applying the BrainVision channel-unit scaling. The data array will use
single precision; time and metadata values will use double precision where applicable.

## MATLAB Files

The export directory will contain two files:

1. `sub-0015_task-thermalactive_desc-brainvisionprocessed_epochs.mat`
2. `sub-0015_task-thermalactive_trial_info.mat`

The EEG file will expose one top-level FieldTrip-compatible variable named `data` with:

- `data.label`: 64 channel labels;
- `data.trial`: 66 cells, each containing a `channels x samples` single-precision matrix;
- `data.time`: 66 cells containing the common -7.0-to-15.0-second time vector;
- `data.fsample`: `1000`;
- `data.sampleinfo`: epoch sample ranges in the exported representation;
- `data.trialinfo`: a 66-row numeric matrix with one row per trial;
- `data.trialinfo_labels`: labels for the columns in `data.trialinfo`.

The separate trial-information file will expose one top-level structure named `trial_info`.
It will retain the 66 selected BIDS rows in export order, including event onset and sample,
run and trial identifiers, stimulus temperature, selected surface, pain classification, VAS
rating, and all available behavioral timing fields. It will also contain the source event-file
path and source BrainVision header path for each row. Numeric missing values will remain `NaN`;
missing text will remain empty rather than being imputed.

The numeric `data.trialinfo` matrix will contain these columns in this order:

1. `run_id`
2. `trial_number`
3. `stimulus_temp`
4. `selected_surface`
5. `pain_binary_coded`
6. `vas_final_coded_rating`

## Alignment and Validation

The export will fail immediately if any required source triplet or event table is missing, a
BrainVision header does not report 1,000 Hz, channel labels or ordering differ across runs, a
run does not contain exactly 11 thermal trials, trial identifiers are duplicated or unordered,
or an epoch extends outside its recording.

Each BIDS trial onset must match one unused BrainVision thermode annotation within 2 ms. The
export will also fail on dropped epochs, non-finite EEG samples, unexpected data shapes, or any
disagreement between the 66 EEG epochs and 66 trial-information rows. No fallback matching,
metadata inference, or silent trial removal is permitted.

## Verification

After writing both files, an independent verification step will reload them and confirm:

- both expected top-level MATLAB variables exist;
- 66 epochs and 66 trial-information rows are present;
- every epoch has 64 channels and the expected 22,001 samples;
- `fsample` is 1,000 Hz and the time vector runs from -7.0 to +15.0 seconds;
- run/trial identifiers are unique and ordered;
- `data.trialinfo` equals the corresponding fields in `trial_info`;
- all EEG samples are finite;
- source paths and an export timestamp are recorded for provenance.

The completed files will be written to
`outputs/matlab_exports/sub-0015/brainvision_processed_1khz` inside the workspace so they can be
reviewed and copied to the colleague without modifying source data on the KINGSTON volume.
