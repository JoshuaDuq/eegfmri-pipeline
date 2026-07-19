# FieldTrip ICA TFR

This adapts `do_ICA_TFR.m` to the pain study and runs it for every available participant.
The Python export converts the pre-ICA FIF derivatives to FieldTrip data, MATLAB computes
ICA, and `computeTfrBatch` computes the component-space TFR variables and saves `comp` in
one `<subject>_ICA_pow_EEG.mat` file per participant.

No blinded component selection, review manifest, artifact-category prompt, or browser UI is
part of this analysis.

```matlab
addpath('/Users/joduq24/Desktop/EEG_fMRI_Pipeline/studies/pain_study/fieldtrip_tfr/matlab');
runtimeConfig = '/Volumes/KINGSTON/EEG_fMRI_data/derivatives/fieldtrip_manual_ica_tfr/fieldtrip_tfr_runtime.json';
prepareIcaBatch(runtimeConfig);
computeTfrBatch(runtimeConfig);
```

The TFR spans 1--100 Hz. Outputs are written under:

```text
/Volumes/KINGSTON/EEG_fMRI_data/derivatives/fieldtrip_manual_ica_tfr/pow
```
