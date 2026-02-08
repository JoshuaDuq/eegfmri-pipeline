# SIIPS1 (Stimulus Intensity Independent Pain Signature 1)

SIIPS1 is public (Woo et al. 2017, *Nature Communications*). This folder can contain the weight map so the pipeline finds it without extra configuration.

**Expected file:** `nonnoc_v11_4_137subjmap_weighted_mean.nii.gz`

Place that file here (or add it to the repo) so fMRI signature extraction and EEG–fMRI ML regression can use SIIPS1 when the signature root points at `eeg_pipeline/data/external` or is set via `--pain-signature-weights`.
