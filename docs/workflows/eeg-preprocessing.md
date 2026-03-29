# EEG Preprocessing Workflow

Use this workflow when starting from BIDS EEG and aiming to produce clean epochs and
aligned clean events for downstream analysis.

## Typical Run Order

1. Validate the BIDS EEG layout and channel metadata.
2. Run bad-channel detection and synchronization across runs.
3. Fit ICA and label artifact components.
4. Create epochs, reject bad segments, and export clean events.
5. Inspect preprocessing statistics before extracting features.

## Main Command Surface

```bash
eeg-pipeline preprocessing --help
```

## Inputs

- BIDS EEG recordings
- `channels.tsv` alongside each run
- `events.tsv` for event-based designs
- a montage or electrode positions resolvable by MNE

## Outputs

- cleaned epochs in FIF format
- epoch-aligned clean events tables
- ICA diagnostics and rejected component records
- per-subject preprocessing statistics

## Reference

Full methods, notation, configuration keys, and output layout:
[preprocessing reference](../reference/preprocessing.md)
