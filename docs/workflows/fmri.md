# fMRI Workflow

Use this workflow for public fMRI preprocessing, first-level modeling, group analysis,
resting-state connectivity, and reporting.

## Typical Run Order

1. Prepare or validate the BIDS fMRI dataset.
2. Run `fmri` for fMRIPrep-style preprocessing.
3. Run `fmri-analysis` for first-level contrasts, second-level models, trialwise betas, or resting-state analysis.
4. Review QC reports, maps, and derivative summaries.

## Main Command Surface

```bash
eeg-pipeline fmri --help
eeg-pipeline fmri-analysis --help
```

## Inputs

- BOLD runs and events in BIDS format
- subject anatomy and FreeSurfer resources when required
- confounds and masks from fMRIPrep derivatives

## Outputs

- preprocessed BOLD derivatives
- first-level contrast maps
- second-level group outputs
- trialwise beta and signature-expression tables
- resting-state connectivity summaries

## Reference

Full methods, GLM assumptions, resting-state details, and output layout:
[fMRI reference](../reference/fmri.md)
