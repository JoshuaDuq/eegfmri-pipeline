# Sub-0015 Band-Specific FieldTrip ICA Design

## Objective

Run three independent FieldTrip ICA decompositions for `sub-0015` using the six pre-ICA
BrainVision Analyzer/MNE `proc-filt_raw.fif` runs, then compute the existing pain-study TFR
outputs separately for each decomposition.

## Input

Use only:

```text
/Volumes/KINGSTON/EEG_fMRI_data/derivatives/
brainvision_analyzer_mne_preprocessing_sub-0015/preprocessed/eeg/sub-0015/eeg/
sub-0015_task-thermalactive_run-<1-6>_proc-filt_raw.fif
```

Do not use the MNE ICA, `proc-ica`, or `proc-clean` derivatives in that directory.

## Decompositions

Fit independent `runica` models to the same 66 concatenated thermal trials:

- Alpha: 6–14 Hz
- Beta: 14–30 Hz
- Gamma: 30–100 Hz

Determine the component count from the numerical rank of each band-filtered dataset. Use the
existing deterministic seed, extended runica setting, and 2,000-iteration limit. Apply each
band-specific unmixing matrix to the unfiltered broadband concatenated epochs, matching the
structure of the colleague's original script.

## Outputs

Keep the existing broadband ICA/TFR files untouched. Write band-specific outputs under the
configured `fieldtrip_brainvision_analyzer_sub-0015` derivative root:

```text
ica/<band>/sub-0015/eeg/*_desc-fieldtripica_components.mat
pow/<band>/sub-0015_ICA_pow_EEG.mat
```

Each TFR file contains the existing temperature, painful/non-painful, high/low, slope, and
average `pow*` variables plus `comp`. TFR settings remain 1–100 Hz, -5 to 14.5 seconds, DPSS,
and 32-second padding so the three decompositions can be compared over the same axes.

## Execution and Errors

Provide one MATLAB runner for export, the three ICA fits, and the three TFR batches. Fail on
missing runs, missing metadata, existing outputs unless overwrite is explicitly enabled, rank
failure, ICA failure, or incompatible TFR structures. Do not add browser UI, component-review
prompts, fallbacks, or automatic component rejection.
