# Data Layout

The public pipelines assume repository-root `data/` directories with configurable paths
defined in [eeg config](../../eeg_pipeline/utils/config/eeg_config.yaml) and
[fMRI config](../../fmri_pipeline/utils/config/fmri_config.yaml).

## Minimum Inputs By Workflow

| Workflow | Minimum inputs |
| --- | --- |
| EEG preprocessing | `data/bids_output/eeg/sub-XXXX/eeg/*_eeg.<format>` plus BIDS sidecars |
| Feature extraction | cleaned epochs in `data/derivatives/sub-XXXX/eeg/` |
| Behavioral analysis | aligned `*_events.tsv` plus trialwise EEG feature tables |
| Machine learning | trialwise feature tables and target columns derived from events |
| fMRI preprocessing | BIDS fMRI dataset or raw DICOMs prepared for `fmri` |
| fMRI analysis | BOLD images, events, masks, and confounds discoverable from derivatives |

## BIDS EEG Layout

```text
data/bids_output/eeg/
├── dataset_description.json
├── participants.tsv
└── sub-XXXX/
    └── eeg/
        ├── sub-XXXX_task-YYY_run-01_eeg.vhdr
        ├── sub-XXXX_task-YYY_run-01_eeg.vmrk
        ├── sub-XXXX_task-YYY_run-01_eeg.eeg
        ├── sub-XXXX_task-YYY_run-01_events.tsv
        ├── sub-XXXX_task-YYY_run-01_channels.tsv
        └── sub-XXXX_task-YYY_run-01_electrodes.tsv
```

## BIDS fMRI Layout

```text
data/bids_output/fmri/
└── sub-XXXX/
    └── func/
        ├── sub-XXXX_task-task_run-01_bold.nii.gz
        ├── sub-XXXX_task-task_run-01_events.tsv
        └── sub-XXXX_task-task_run-01_bold.json
```

## Default Repository Layout

```text
data/
├── source_data/
├── bids_output/
├── fMRI_data/
└── derivatives/
```

## Events And Trial Alignment

Public behavior and machine-learning workflows use `trial_id` as the alignment contract
between cleaned events and feature tables. Required BIDS event columns are:

- `onset`
- `duration`
- `trial_type`

Additional predictor, condition, and target columns must already exist in the events files
you plan to analyze.

## Outputs

Expected derivative families:

- cleaned EEG epochs and QC summaries
- trialwise EEG feature tables
- behavior analysis tables and reports
- machine-learning metrics, importances, and plots
- fMRIPrep outputs and fMRI statistical maps
