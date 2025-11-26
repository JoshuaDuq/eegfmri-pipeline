# EEG Pipeline

## Installation

### Python environment (recommended)

Use a dedicated virtual environment with Python 3.11 for this project:

```bash
cd /Users/joduq24/Desktop/EEG_fMRI_Pipeline

# create the environment (only once)
/opt/homebrew/bin/python3.11 -m venv .venv311

# activate the environment (every new shell)
source .venv311/bin/activate

# upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

Make sure you have `python3.11` installed via Homebrew:

```bash
brew install python@3.11
```

```bash
pip install -r requirements.txt
```

## Commands

### Preprocessing

**Convert BrainVision to BIDS:**
```bash
python eeg_pipeline/preprocessing/raw_to_bids.py \
    [--source_root PATH] \
    [--bids_root PATH] \
    [--task TASK] \
    [--subjects SUB1 SUB2 ...] \
    [--montage MONTAGE_NAME] \
    [--line_freq FREQ] \
    [--overwrite] \
    [--zero_base_onsets] \
    [--trim_to_first_volume] \
    [--event_prefix PREFIX] \
    [--keep_all_annotations]
```

**Merge behavioral data into events:**
```bash
python eeg_pipeline/preprocessing/merge_behavior_to_events.py \
    [--bids_root PATH] \
    [--source_root PATH] \
    [--task TASK] \
    [--event_prefix PREFIX] \
    [--event_type TYPE] \
    [--dry_run]
```

### Main Pipeline

**Behavior Analysis:**
```bash
python eeg_pipeline/scripts/run_pipeline.py behavior {compute|visualize|aggregate} \
    [--subject SUB] [--all-subjects] [--group all|SUB1,SUB2] \
    [--task TASK] \
    [--correlation-method {spearman|pearson}] \
    [--bootstrap N] \
    [--n-perm N] \
    [--rng-seed N] \
    [--plots PLOT1 PLOT2 ...] \
    [--all-plots] \
    [--skip-scatter] \
    [--do-group]
```

**Features Extraction:**
```bash
python eeg_pipeline/scripts/run_pipeline.py features {compute|visualize|aggregate} \
    [--subject SUB] [--all-subjects] [--group all|SUB1,SUB2] \
    [--task TASK] \
    [--fixed-templates PATH] \
    [--feature-categories CAT1 CAT2 ...]
```

Available feature categories: `power`, `connectivity`, `microstates`, `aperiodic`, `itpc`, `pac`

**ERP Analysis:**
```bash
python eeg_pipeline/scripts/run_pipeline.py erp {compute|visualize} \
    [--subject SUB] [--all-subjects] [--group all|SUB1,SUB2] \
    [--task TASK] \
    [--crop-tmin T] \
    [--crop-tmax T]
```

**TFR Visualization:**
```bash
python eeg_pipeline/scripts/run_pipeline.py tfr visualize \
    [--subject SUB] [--all-subjects] [--group all|SUB1,SUB2] \
    [--task TASK] \
    [--do-group] \
    [--tfr-roi] \
    [--tfr-topomaps-only]
```

**Decoding:**
```bash
python eeg_pipeline/scripts/run_pipeline.py decoding \
    [--subject SUB1] [--subject SUB2] ... \
    [--task TASK] \
    [--n-perm N] \
    [--inner-splits N] \
    [--outer-jobs N] \
    [--rng-seed N] \
    [--skip-time-gen]
```

## Help

For detailed options:
```bash
python eeg_pipeline/scripts/run_pipeline.py <command> --help
```
