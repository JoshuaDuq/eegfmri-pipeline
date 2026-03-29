# Configuration Reference

The public repo keeps most runtime configuration in YAML rather than in code constants.

## Public Config Files

| File | Purpose |
| --- | --- |
| `eeg_pipeline/utils/config/eeg_config.yaml` | Shared EEG, behavior, machine-learning, plotting, and fMRI analysis defaults |
| `fmri_pipeline/utils/config/fmri_config.yaml` | fMRI-only reference defaults and path contracts |

## Key Public Sections

| Section | Used by |
| --- | --- |
| `paths` | BIDS roots, derivatives, anatomy, FreeSurfer resources |
| `preprocessing` | filtering, epoching, rejection, and statistics behavior |
| `ica` | decomposition method, components, thresholds, reproducibility |
| `feature_engineering` | feature families, bands, windows, transforms, normalization |
| `behavior_analysis` | stage selection, correction methods, grouped permutation safeguards |
| `machine_learning` and `classification` | model families, CV behavior, resampling, importance outputs |
| `fmri_preprocessing` | fMRIPrep container options, spaces, resources |
| `fmri_contrast`, `fmri_group_level`, `fmri_resting_state` | first-level, group-level, and resting-state analysis |

## Usage

Public commands load YAML first and then apply explicit CLI overrides where supported.
The repo intentionally fails fast on invalid combinations instead of silently repairing them.
