# EEG Preprocessing Reference

This is the technical reference for the public EEG preprocessing pipeline.
It turns BIDS EEG recordings into cleaned epochs, aligned clean events, and QC outputs.

## Core Stages

1. bad-channel detection with PyPREP
2. cross-run bad-channel synchronization
3. ICA fitting through MNE-BIDS-Pipeline
4. ICA label assignment with MNE-ICAlabel
5. epoch creation and artifact rejection
6. clean-events export
7. preprocessing statistics export
8. optional time-frequency computation

## Inputs

Required files per run:

- EEG recording file such as `.vhdr`, `.edf`, or `.fif`
- `channels.tsv`
- `events.tsv` for event-based designs

The pipeline expects channel typing to be available from BIDS sidecars and applies a
standard montage when digitized positions are absent.

## Key Methods

### Bad Channels

Bad channels are detected on continuous raw data before ICA or epoching.
The public pipeline supports repeated detection passes and writes the final bad-channel
state back to `channels.tsv`.

### ICA

ICA fitting is delegated to MNE-BIDS-Pipeline with configuration-driven filtering,
reproducible random state handling, and explicit artifact labeling. The public docs
assume ICLabel-compatible decomposition settings when automated component labels are used.

### Epochs And Clean Events

Epoch creation applies rejection logic and exports a clean events table aligned to the
accepted epochs only. Public downstream workflows rely on this output and its `trial_id`
contract.

## Important Config Areas

- `preprocessing`
- `pyprep`
- `ica`
- `eeg`

See [configuration reference](configuration/index.md) for the public YAML entrypoints.

## Outputs

Expected derivative families:

- cleaned epochs
- clean events tables
- channel and ICA exclusion summaries
- preprocessing statistics
- optional time-frequency outputs
