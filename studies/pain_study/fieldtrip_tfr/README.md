# MNE ICA component review in MATLAB

This workflow visualizes the exact ICA decomposition saved by the MNE preprocessing
pipeline. It does not refit ICA in FieldTrip. Component numbers therefore match the MNE
rejection decisions exactly.

The exporter constructs thermal-event epochs directly from the pre-ICA filtered runs,
projects them through the saved MNE ICA, and exports all component time series and
topographies. This runs immediately after ICA labeling, before ICA application or trial
rejection. Components automatically proposed as bad by MNE-ICALabel remain present and are
marked in red in MATLAB; the plots do not treat that proposal as a manual decision.

## Configure

Edit `config/mne_ica_review.yaml`:

- `paths.mne_derivatives`: one or more MNE `preprocessed/eeg` roots;
- `study.participants`: `"all"` or a list such as `["sub-0015"]`;
- `conditions`: any combination of the supported condition names;
- `plots.frequency_view`: `alpha`, `beta`, `gamma`, or `full`;
- `execution.overwrite_exports` and `execution.overwrite_tfr`: output replacement controls.

Supported conditions are:

```text
high_vs_low
high_temperature
low_temperature
painful_vs_nonpainful
painful
nonpainful
individual_temperatures
temperature_slope
grand_average
```

Condition plots are baseline-normalized in decibels. Contrast plots are differences of
baseline-normalized decibel power.

## Run

In MATLAB:

```matlab
run('/Users/joduq24/Desktop/EEG_fMRI_Pipeline/studies/pain_study/fieldtrip_tfr/run_MNE_ICA_review.m')
```

Then open the figures:

```matlab
run('/Users/joduq24/Desktop/EEG_fMRI_Pipeline/studies/pain_study/fieldtrip_tfr/plot_MNE_ICA_review.m')
```

The plotting script creates topography pages and TFR pages containing 16 components per
figure. Red component titles and red axes identify components automatically proposed as
bad by MNE-ICALabel.
