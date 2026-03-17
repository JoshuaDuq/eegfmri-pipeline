# Paradigm-Specific Scripts (CLI Only)

Scripts specific to the simultaneous EEG–fMRI pain paradigm. **Not integrated into the TUI or `eeg-pipeline` CLI** — must be run manually before any downstream analysis.

## Required Workflow Order

```
1. eeg-raw-to-bids     → Convert BrainVision EEG source files → BIDS EEG
2. fmri-raw-to-bids    → Convert fMRI DICOMs → BIDS fMRI
3. merge-psychopy      → Merge PsychoPy TrialSummary.csv → BIDS events.tsv
4. eeg-pipeline ...    → Run preprocessing, features, behavior, ml, fmri, fmri-analysis
```

Steps 1–3 must complete successfully before running any `eeg-pipeline` command.

## FreeSurfer License (fMRIPrep/BEM)

Default license location is `~/license.txt`. Place your FreeSurfer `license.txt` there, or override with `paths.freesurfer_license`, `EEG_PIPELINE_FREESURFER_LICENSE`, or `--fs-license-file`.

---

## Entrypoint

All commands are dispatched through a single entrypoint:

```bash
python studies/pain_study/scripts/run_paradigm_specific.py <command> [options]
```

---

## Commands

### `eeg-raw-to-bids`

Converts BrainVision (`.vhdr`) source files to BIDS EEG format using `mne-bids`.

**Source layout expected:**
```
<source-root>/
  sub-<ID>/
    eeg/
      sub-<ID>_task-<task>_run-<N>.vhdr
      sub-<ID>_task-<task>_run-<N>.vmrk
      sub-<ID>_task-<task>_run-<N>.eeg
```

**Usage:**
```bash
python studies/pain_study/scripts/run_paradigm_specific.py eeg-raw-to-bids \
  --source-root data/source_data \
  --bids-root data/bids_output/eeg \
  --task task \
  --subject 0001 \
  --subject 0002
```

**All options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--source-root` | *(required)* | Root directory containing raw `sub-*/eeg/*.vhdr` files |
| `--bids-root` | *(required)* | Output BIDS root for EEG data |
| `--task` | *(required)* | BIDS task label (e.g. `task`) |
| `--subject` | all found | Subject ID(s) to process (repeat flag for multiple) |
| `--montage` | `easycap-M1` | MNE montage name for electrode positions |
| `--line-freq` | `60.0` | Power line frequency in Hz (60 for North America, 50 for Europe) |
| `--overwrite` | `False` | Overwrite existing BIDS files |
| `--trim-to-first-volume` | `False` | Trim EEG recording to start at first fMRI volume trigger (recommended for EEG-fMRI alignment) |
| `--event-prefix` | `Trig_therm/T  1` | Keep only annotations matching this prefix (repeat for multiple). Defaults to the thermode trigger prefix. |
| `--keep-all-annotations` | `False` | Keep all annotations regardless of prefix filtering |

**Example — all subjects, trim to fMRI volume, custom event prefix:**
```bash
python studies/pain_study/scripts/run_paradigm_specific.py eeg-raw-to-bids \
  --source-root data/source_data \
  --bids-root data/bids_output/eeg \
  --task task \
  --montage easycap-M1 \
  --line-freq 60 \
  --trim-to-first-volume \
  --event-prefix "Trig_therm/T  1" \
  --overwrite
```

---

### `fmri-raw-to-bids`

Converts fMRI DICOM data to BIDS fMRI format using `dcm2niix`.

**Source layout expected:**
```
<source-root>/
  sub-<ID>/
    DICOM/
      <series folders containing .dcm files>
    PsychoPy_Data/
      sub-<ID>_run-<N>_TrialSummary.csv
```

**Usage:**
```bash
python studies/pain_study/scripts/run_paradigm_specific.py fmri-raw-to-bids \
  --source-root data/source_data \
  --bids-fmri-root data/bids_output/fmri \
  --task task \
  --subject 0001 \
  --subject 0002
```

**All options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--source-root` | *(required)* | Root directory containing raw `sub-*/DICOM/` folders |
| `--bids-fmri-root` | *(required)* | Output BIDS root for fMRI data |
| `--task` | *(required)* | BIDS task label for the main task runs |
| `--subject` | all found | Subject ID(s) to process (repeat flag for multiple) |
| `--session` | `None` | BIDS session label (omit if single-session) |
| `--rest-task` | `rest` | BIDS task label for resting-state runs |
| `--no-rest` | `False` | Skip resting-state run conversion |
| `--no-fieldmaps` | `False` | Skip fieldmap conversion |
| `--dicom-mode` | `symlink` | How to stage DICOMs: `symlink`, `copy`, or `skip` |
| `--overwrite` | `False` | Overwrite existing BIDS files |
| `--no-events` | `False` | Skip events.tsv generation |
| `--event-granularity` | `phases` | Event granularity: `trial` (one row per trial) or `phases` (one row per stimulus phase) |
| `--onset-reference` | `first_iti_start` | Onset reference: `as_is`, `first_iti_start`, or `first_stim_start` |
| `--onset-offset-s` | `0.0` | Constant offset (seconds) added to all event onsets |
| `--dcm2niix-path` | system PATH | Path to `dcm2niix` executable |
| `--dcm2niix-arg` | *(none)* | Extra arguments forwarded to `dcm2niix` (repeat for multiple) |

**Example — skip rest, use phase-level events:**
```bash
python studies/pain_study/scripts/run_paradigm_specific.py fmri-raw-to-bids \
  --source-root data/source_data \
  --bids-fmri-root data/bids_output/fmri \
  --task task \
  --no-rest \
  --event-granularity phases \
  --onset-reference first_iti_start \
  --overwrite
```

---

### `merge-psychopy`

Merges PsychoPy `TrialSummary.csv` behavioral columns into the BIDS `*_events.tsv` files produced by `eeg-raw-to-bids`. Must be run **after** `eeg-raw-to-bids`.

**Behavioral file layout expected:**
```
<source-root>/
  sub-<ID>/
    PsychoPy_Data/
      sub-<ID>_run-<N>_TrialSummary.csv
```

**Usage:**
```bash
python studies/pain_study/scripts/run_paradigm_specific.py merge-psychopy \
  --source-root data/source_data \
  --bids-root data/bids_output/eeg \
  --task task \
  --subject 0001 \
  --subject 0002
```

**All options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--source-root` | *(required)* | Root directory containing `sub-*/PsychoPy_Data/*.csv` |
| `--bids-root` | *(required)* | BIDS EEG root (same as `--bids-root` used for `eeg-raw-to-bids`) |
| `--task` | *(required)* | BIDS task label |
| `--subject` | all found | Subject ID(s) to process (repeat flag for multiple) |
| `--event-prefix` | `Trig_therm/T  1` | Filter events by prefix before merging (repeat for multiple). Must match what was used in `eeg-raw-to-bids`. |
| `--event-type` | *(none)* | Filter events by exact type (repeat for multiple) |
| `--dry-run` | `False` | Preview merge without writing files |
| `--allow-misaligned-trim` | `False` | Allow and silently trim when PsychoPy row count does not match event count. Use only for debugging. |

**Example — dry run first, then apply:**
```bash
# Preview
python studies/pain_study/scripts/run_paradigm_specific.py merge-psychopy \
  --source-root data/source_data \
  --bids-root data/bids_output/eeg \
  --task task \
  --dry-run

# Apply
python studies/pain_study/scripts/run_paradigm_specific.py merge-psychopy \
  --source-root data/source_data \
  --bids-root data/bids_output/eeg \
  --task task
```

---

## Global Options

Available on all commands:

| Flag | Default | Description |
|------|---------|-------------|
| `--log-level` | `INFO` | Logging verbosity: `DEBUG`, `INFO`, `WARNING`, `ERROR` |

---

## Full Example: End-to-End Conversion

```bash
# 1. Convert EEG to BIDS
python studies/pain_study/scripts/run_paradigm_specific.py eeg-raw-to-bids \
  --source-root data/source_data \
  --bids-root data/bids_output/eeg \
  --task task \
  --trim-to-first-volume \
  --overwrite

# 2. Convert fMRI to BIDS
python studies/pain_study/scripts/run_paradigm_specific.py fmri-raw-to-bids \
  --source-root data/source_data \
  --bids-fmri-root data/bids_output/fmri \
  --task task \
  --event-granularity phases \
  --overwrite

# 3. Merge PsychoPy behavioral data into EEG events
python studies/pain_study/scripts/run_paradigm_specific.py merge-psychopy \
  --source-root data/source_data \
  --bids-root data/bids_output/eeg \
  --task task

# 4. Run downstream analysis (TUI or CLI)
eeg-pipeline preprocessing --subjects 0001 0002 --task task
eeg-pipeline features      --subjects 0001 0002 --task task
eeg-pipeline behavior      --subjects 0001 0002 --task task
eeg-pipeline fmri          --task task
eeg-pipeline fmri-analysis --task task
```

---

## Adapting for a Different Paradigm

These scripts encode conventions specific to this pain paradigm:

- **EEG**: BrainVision format, `easycap-M1` montage, thermode trigger prefix `Trig_therm/T  1`, optional volume-trigger trimming for EEG-fMRI alignment.
- **fMRI**: DICOM input via `dcm2niix`, phase-level event granularity, rest + fieldmap runs.
- **Behavior**: PsychoPy `TrialSummary.csv` with `run_id`, `stim_start_time`, `stimulus_temp`, `condition` columns.

To adapt for a different paradigm, modify the scripts in this folder. **Do not modify** `eeg_pipeline/` or `fmri_pipeline/` core code.

Paradigm-specific configuration templates are also isolated here:

- `studies/pain_study/scripts/config/thermal_pain_eeg_overrides.yaml`
- `studies/pain_study/scripts/config/thermal_pain_fmri_overrides.yaml`

Use these as optional overrides when running the thermal pain paradigm, while keeping core pipeline defaults paradigm-agnostic.
