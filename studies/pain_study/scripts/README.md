# Paradigm-Specific Scripts

Everything specific to the simultaneous EEG–fMRI pain paradigm. Each folder is one job, and
each has a README explaining what it is for and why it works the way it does.

| Folder | What it is for | How it runs |
|---|---|---|
| [`conversion/`](conversion/) | Source recordings → BIDS. Runs before anything else. | `run_paradigm_specific.py <command>` |
| [`line_comb/`](line_comb/) | Diagnose and remove the scanner room's narrowband line comb. | `eeg-pipeline line-comb <mode>` |
| [`cardiac_gaps/`](cardiac_gaps/) | Recover the heartbeats Analyzer never marked, and correct only there. | `eeg-pipeline cardiac-gaps <mode>` |
| [`t1/`](t1/) | Electrode localization in a participant's own T1, and the template fallback. | `python -m …scripts.t1.run_t1_*` |
| [`study_support/`](study_support/) | Batch machinery for running study1 and study2 at cohort scale. | see its README |
| [`config/`](config/) | Paradigm override templates and per-script configs. | — |

The measurement and algorithm halves of the two artifact workflows live in
[`../analysis/`](../analysis/); these folders decide which files to read and where results
go. The EEG coupling workflow lives under [`../eeg_coupling/`](../eeg_coupling/) and runs as
`eeg-pipeline coupling compute`.

## Order of operations

```
1. conversion/          → BIDS EEG, BIDS fMRI, behaviour merged into events.tsv
2. cardiac_gaps/        → recover unmarked beats, correct the gaps       (optional)
3. line_comb/           → diagnose, benchmark, apply, verify             (optional)
4. eeg-pipeline ...     → preprocessing, features, behavior, ml, fmri, fmri-analysis
```

Step 1 must complete before any `eeg-pipeline` command. Steps 2 and 3 are artifact
remediation: each writes its own output tree rather than editing data in place, so the
pipeline reads them only once you point `paths.bids_root` at what they wrote.

## Configuration

Paths — `bids_root`, `deriv_root`, `source_data` — are answered once, by the core
`eeg_pipeline/utils/config/eeg_config.yaml`. Workflow folders inherit them and override only
when they genuinely need something else. Settings that belong to a single workflow live in
that folder's own `config.yaml`, next to the code that reads them. See
[`workflow_config.py`](workflow_config.py) for the resolution order.

## FreeSurfer License (fMRIPrep/BEM)

Default license location is `~/license.txt`. Place your FreeSurfer `license.txt` there, or override with `paths.freesurfer_license`, `EEG_PIPELINE_FREESURFER_LICENSE`, or `--fs-license-file`.

---

## Conversion commands

Staging, conversion and event merging are dispatched through a single entrypoint:

```bash
python studies/pain_study/scripts/run_paradigm_specific.py <command> [options]
```

Newly acquired EEG source recordings are staged first, outside that entrypoint:

```bash
python -m studies.pain_study.scripts.conversion.organize_source_eeg \
  --kingston-root /Volumes/KINGSTON \
  --source-data-root /Volumes/KINGSTON/EEG_fMRI_data/source_data
```

This copies original 5 kHz triplets into `sub-*/eeg/original_untrimmed_5khz/`. If legacy
BrainVision-processed 1 kHz files are present, it also moves them into
`sub-*/eeg/brainvision_processed_1khz/`; they are not required by the native pipeline. By default
the organizer discovers every unorganized `sub-*` EEG directory. Repeat `--subject <ID>` to
restrict a run.

---

## Commands

### `eeg-raw-to-bids`

Converts BrainVision (`.vhdr`) source files to BIDS EEG format using `mne-bids`.

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

# 4. Artifact remediation (optional; each writes its own output tree)
eeg-pipeline cardiac-gaps report      # measure Analyzer's pulse-marker gaps
eeg-pipeline line-comb diagnose       # measure the room's line comb
eeg-pipeline line-comb benchmark      # check preservation before removing anything
eeg-pipeline line-comb apply          # write the cleaned BIDS copy
# Then point paths.bids_root at what you want the pipeline to read.

# 5. Fit ICA, review component exclusions, then create epochs
# Core's packaged defaults are not this study's data roots. Name the study config
# explicitly from here on, or these commands run against whatever eeg_config.yaml
# currently points at.
eeg-pipeline --config studies/pain_study/config/pain_study.yaml preprocessing ica --subject 0001 --subject 0002 --task task
# Review the generated MNE-BIDS component tables before continuing.
eeg-pipeline --config studies/pain_study/config/pain_study.yaml preprocessing epochs --subject 0001 --subject 0002 --task task \
  --set ica.manual_review_complete=true
eeg-pipeline --config studies/pain_study/config/pain_study.yaml features compute --subject 0001 --subject 0002 --task task
eeg-pipeline --config studies/pain_study/config/pain_study.yaml behavior compute --subject 0001 --subject 0002 --task task
eeg-pipeline --config studies/pain_study/config/pain_study.yaml fmri preprocess --subject 0001 --subject 0002 --task task
eeg-pipeline --config studies/pain_study/config/pain_study.yaml fmri-analysis first-level --subject 0001 --subject 0002 --task task \
  --cond-a-value stimulation --cond-b-value fixation_rest

# 6. Run EEG–BOLD coupling (integrated CLI)
eeg-pipeline --config studies/pain_study/config/pain_study.yaml coupling compute --subject 0001 --subject 0002 --task task
```

---

## Adapting for a Different Paradigm

These scripts encode conventions specific to this pain paradigm:

- **EEG**: BrainVision format, `easycap-M1` montage, thermode trigger prefix `Trig_therm/T  1`, optional volume-trigger trimming for EEG-fMRI alignment.
- **fMRI**: DICOM input via `dcm2niix`, phase-level event granularity, rest + fieldmap runs.
- **Behavior**: PsychoPy `TrialSummary.csv` with `run_id`, `stim_start_time`, `stimulus_temp`, `condition` columns.

The artifact workflows encode more than conventions — they encode findings. `line_comb/`
holds the measured frequencies of one scanner room and means nothing at another site;
`cardiac_gaps/` exists to repair one vendor's failure mode on one set of exports.

To adapt for a different paradigm, modify the scripts in this folder. **Do not modify** `eeg_pipeline/` or `fmri_pipeline/` core code.

Paradigm-specific configuration is isolated here too — see [`config/README.md`](config/README.md)
for which files are loaded automatically and which are templates you apply yourself.
