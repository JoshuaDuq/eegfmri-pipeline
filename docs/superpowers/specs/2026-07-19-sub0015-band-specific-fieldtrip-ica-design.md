# Sub-0015 Band-Specific FieldTrip ICA Design

## Objective

Run three independent FieldTrip ICA decompositions for `sub-0015` using pre-ICA signals
restricted to the final post-rejection trial set, then compute the existing pain-study TFR
outputs separately for each decomposition.

## Input

Use signal values only from:

```text
/Volumes/KINGSTON/EEG_fMRI_data/derivatives/
brainvision_analyzer_mne_preprocessing_sub-0015/preprocessed/eeg/sub-0015/eeg/
sub-0015_task-thermalactive_epo.fif
```

Use the saved MNE epoch-selection indices from
`sub-0015_task-thermalactive_proc-clean_epo.fif` only as the final trial-rejection mask. This
retains 59 of the 66 pre-ICA epochs while preserving the correct run, temperature, and pain
labels. Do not use signal values from either `proc-ica_epo.fif` or `proc-clean_epo.fif`.

## Decompositions

Fit independent `runica` models to the same 59 retained thermal trials:

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
