# EEG Preprocessing Pipeline

**Module:** `eeg_pipeline.preprocessing`

Automated, reproducible EEG preprocessing built on MNE-Python, MNE-BIDS-Pipeline,
PyPREP, and MNE-ICAlabel. Operates on BIDS-formatted EEG data and produces clean,
epoched datasets ready for feature extraction and statistical analysis.

---

## Table of Contents

1. [Notation](#1-notation)
2. [Module Structure](#2-module-structure)
3. [Pipeline Overview](#3-pipeline-overview)
4. [Input Data Requirements](#4-input-data-requirements)
5. [Step 1 — Bad Channel Detection](#5-step-1--bad-channel-detection)
6. [Step 2 — Bad Channel Synchronization](#6-step-2--bad-channel-synchronization)
7. [Step 3 — ICA Fitting](#7-step-3--ica-fitting)
8. [Step 4 — ICA Component Labeling](#8-step-4--ica-component-labeling)
9. [Step 5 — Epoch Creation and Artifact Rejection](#9-step-5--epoch-creation-and-artifact-rejection)
10. [Step 6 — Clean Events Export](#10-step-6--clean-events-export)
11. [Step 7 — Preprocessing Statistics](#11-step-7--preprocessing-statistics)
12. [Step 8 — Time-Frequency Representation (Optional)](#12-step-8--time-frequency-representation-optional)
13. [Execution Modes](#13-execution-modes)
14. [Configuration Reference](#14-configuration-reference)
15. [Output Structure](#15-output-structure)
16. [Dependencies](#16-dependencies)

---

## 1. Notation

| Symbol | Meaning |
|--------|---------|
| $f$ | Frequency (Hz) |
| $n_\text{cycles}(f)$ | Number of Morlet wavelet cycles at frequency $f$ |
| $p_k$ | ICLabel predicted probability for component class $k$ |
| `l_freq`, `h_freq` | High-pass and low-pass filter cutoffs (Hz) |
| ICA | Independent Component Analysis |
| ITC | Inter-Trial Coherence |
| PTP | Peak-to-peak amplitude |
| TFR | Time-Frequency Representation |

---

## 2. Module Structure

| File | Responsibility |
|------|---------------|
| `pipeline/preprocess.py` | Bad channel detection (`run_bads_detection`) and cross-run synchronization (`synchronize_bad_channels_across_runs`) |
| `pipeline/ica.py` | ICA component labeling via MNE-ICAlabel (`run_ica_label`) |
| `pipeline/stats.py` | Preprocessing statistics collection (`collect_preprocessing_stats`) |
| `pipeline/tfr.py` | Morlet wavelet TFR computation (`custom_tfr`) |
| `pipeline/utils.py` | BIDS file discovery, path manipulation, condition name sanitization |
| `pipeline/io.py` | MNE object I/O; channels/components TSV read/write |
| `pipelines/preprocessing.py` | `PreprocessingPipeline` orchestrator: batch execution, subprocess management, logging |

---

## 3. Pipeline Overview

```
BIDS EEG Data (.vhdr/.edf)
        │
        ▼
┌───────────────────────────────────┐
│  1. Bad Channel Detection         │  PyPREP (NoisyChannels)
│     Deviation + correlation       │  Repeated N times; union of bads
│     Optional: RANSAC              │
└───────────┬───────────────────────┘
            │
            ▼
┌───────────────────────────────────┐
│  2. Bad Channel Synchronization   │  Union of bads across runs
│     per subject                   │  Written back to channels.tsv
└───────────┬───────────────────────┘
            │
            ▼
┌───────────────────────────────────┐
│  3. ICA Fitting                   │  MNE-BIDS-Pipeline subprocess
│     Bandpass filter               │  Steps: init → _01 → _04 → _05 → _06a1
│     Artifact regression           │
│     ICA decomposition             │
└───────────┬───────────────────────┘
            │
            ▼
┌───────────────────────────────────┐
│  4. ICA Component Labeling        │  MNE-ICAlabel (ICLabel classifier)
│     Probabilistic classification  │  Threshold: p > 0.8
│     Exclude non-brain/other       │  Labels kept: ["brain", "other"]
└───────────┬───────────────────────┘
            │
            ▼
┌───────────────────────────────────┐
│  5. Epoch Creation & Rejection    │  MNE-BIDS-Pipeline subprocess
│     Epoch segmentation            │  Steps: _07 → _08a → _09
│     ICA component removal         │
│     PTP / autoreject              │
└───────────┬───────────────────────┘
            │
            ▼
┌───────────────────────────────────┐
│  6. Clean Events Export           │  Epoch-aligned events.tsv
│     Rejected epochs excluded      │  Written to derivatives
└───────────┬───────────────────────┘
            │
            ▼
┌───────────────────────────────────┐
│  7. Preprocessing Statistics      │  Per-subject summary TSV
│     Bad channels, ICA exclusions  │  + descriptive statistics
│     Epoch rejection counts        │
└───────────┬───────────────────────┘
            │
            ▼
┌───────────────────────────────────┐
│  8. TFR Computation (optional)    │  Morlet wavelets
│     Power / ITC per condition     │  Configurable freq range & decimation
└───────────────────────────────────┘
```

---

## 4. Input Data Requirements

The pipeline expects BIDS-formatted EEG data:

```
bids_root/
├── dataset_description.json
└── sub-XXXX/
    └── eeg/
        ├── sub-XXXX_task-<task>_eeg.vhdr
        ├── sub-XXXX_task-<task>_eeg.vmrk
        ├── sub-XXXX_task-<task>_eeg.eeg
        ├── sub-XXXX_task-<task>_channels.tsv
        └── sub-XXXX_task-<task>_events.tsv
```

| File | Requirement |
|------|-------------|
| EEG recording | BrainVision (`.vhdr`) by default; configurable via `pyprep.file_extension` |
| `channels.tsv` | Must exist alongside each EEG file. Used to identify channel types (`EEG`, `EOG`, `ECG`, `EMG`, `MISC`) and to read/write bad channel status |
| `events.tsv` | Required for event-related paradigms. Must contain a `trial_type` column. Multi-run designs use per-run files (e.g., `run-01_events.tsv`) combined automatically |
| Montage | Standard electrode montage (default: `easycap-M1`) applied when the recording lacks digitized positions |

---

## 5. Step 1 — Bad Channel Detection

**Module:** `pipeline/preprocess.py` → `run_bads_detection()`

Automated detection of noisy channels using PyPREP's `NoisyChannels` class,
operating on continuous raw data before ICA or epoching.

### 5.1 Method

1. Load raw data via `mne_bids.read_raw_bids()`.
2. Apply standard montage (`easycap-M1` by default) when digitized positions are absent.
3. Optional low-pass filter at `h_freq` Hz (default 100 Hz) on EEG channels only.
4. Optional notch filter at `notch_freq` Hz (default 60 Hz) on EEG channels.
5. Optional average re-reference before detection (disabled by default).
6. Iterative bad channel detection (`repeats` iterations, default 3):
   - `find_bad_by_deviation()` — channels whose robust z-scored amplitude deviates
     from the cross-channel median.
   - `find_bad_by_correlation()` — channels with low Pearson correlation to
     neighboring channels.
   - `find_bad_by_ransac()` — channels that cannot be predicted from neighbors
     via RANSAC interpolation (optional; disabled by default).
   - After each iteration, detected bads are accumulated as a union and marked in
     `raw.info["bads"]`, allowing subsequent iterations to find channels that were
     masked by prior bad channels.
7. Inject custom bad channels from `custom_bad_dict` (per-task, per-subject dict).
8. Write results to `channels.tsv` (sets `status = "bad"` for detected channels).
9. Write per-file CSV log (`pyprep_task_<task>_log.csv`) with detected bads,
   parameters, and any errors.

Supports parallel execution across files via `joblib.Parallel` (`n_jobs`).

### 5.2 Configuration

| Key | Default | Description |
|-----|---------|-------------|
| `pyprep.ransac` | `false` | Enable RANSAC-based detection |
| `pyprep.repeats` | `3` | Detection iterations (union across iterations) |
| `pyprep.average_reref` | `false` | Average re-reference before detection |
| `pyprep.file_extension` | `".vhdr"` | Raw data file extension |
| `pyprep.consider_previous_bads` | `false` | Retain previously marked bads from `channels.tsv` |
| `pyprep.overwrite_chans_tsv` | `true` | Overwrite original `channels.tsv` |
| `pyprep.delete_breaks` | `false` | Annotate recording breaks (for logging; not cropped) |
| `pyprep.breaks_min_length` | `20` | Minimum break duration to annotate (s) |
| `pyprep.t_start_after_previous` | `2` | Seconds after last event to start break search |
| `pyprep.t_stop_before_next` | `2` | Seconds before next event to end break search |
| `pyprep.custom_bad_dict` | `null` | Manual bad channels: `{task: {subject: [channels]}}` |
| `preprocessing.h_freq` | `100` | Low-pass cutoff applied before detection (Hz) |
| `preprocessing.notch_freq` | `60` | Notch filter frequency (Hz) |
| `eeg.montage` | `"easycap-M1"` | Standard montage name |

---

## 6. Step 2 — Bad Channel Synchronization

**Module:** `pipeline/preprocess.py` → `synchronize_bad_channels_across_runs()`

For multi-run paradigms, bad channels detected in any run are propagated to all runs
of the same subject. This ensures a consistent channel set before ICA fitting.

### 6.1 Method

1. For each subject, glob all `channels.tsv` files matching the task.
2. Compute the union of all channels marked `status == "bad"` across runs.
3. Write the unified bad channel set to every run's `channels.tsv`.

MNE-BIDS-Pipeline reads `channels.tsv` to determine which channels to exclude from
ICA fitting and interpolation. Inconsistent bad sets across runs produce incompatible
ICA decompositions.

---

## 7. Step 3 — ICA Fitting

**Module:** `pipelines/preprocessing.py` → `_run_ica_fitting()`

ICA fitting is delegated to MNE-BIDS-Pipeline via subprocess.
A temporary Python config file is generated from the YAML configuration and passed
to the pipeline runner.

### 7.1 MNE-BIDS-Pipeline Steps

| Step | Description |
|------|-------------|
| `init` | Initialize pipeline; validate BIDS dataset |
| `preprocessing/_01_data_quality` | Data quality assessment; Maxwell filtering (if applicable) |
| `preprocessing/_04_frequency_filter` | Bandpass filter: `l_freq`–`h_freq` (default 0.1–100 Hz) |
| `preprocessing/_05_regress_artifact` | Regress out artifact signals (EOG/ECG regression) |
| `preprocessing/_06a1_fit_ica` | Fit ICA decomposition |
| `preprocessing/_06a2_find_ica_artifacts` | MNE built-in EOG/ECG artifact detection (only when `use_icalabel = false`) |

### 7.2 ICA Configuration

| Key | Default | Description |
|-----|---------|-------------|
| `ica.algorithm` | `"extended_infomax"` | ICA algorithm. Options: `"extended_infomax"`, `"picard"`, `"fastica"` |
| `ica.n_components` | `0.99` | Component count: float = variance explained; int = exact count |
| `ica.l_freq` | `1.0` | High-pass cutoff for ICA fitting epochs (Hz); removes slow drifts |
| `ica.reject` | `null` | PTP rejection for ICA fitting epochs (e.g., `{"eeg": 300e-6}`) |
| `preprocessing.l_freq` | `0.1` | Main bandpass high-pass cutoff (Hz) |
| `preprocessing.h_freq` | `100` | Main bandpass low-pass cutoff (Hz) |
| `preprocessing.notch_freq` | `60` | Notch filter frequency (Hz); `null` to skip |
| `preprocessing.resample_freq` | `500` | Resampling frequency (Hz); `null` to skip |
| `preprocessing.find_breaks` | `true` | Annotate recording breaks |
| `preprocessing.random_state` | `42` | Random seed for reproducibility |
| `eeg.reference` | `"average"` | EEG reference (`"average"` or specific channel) |
| `eeg.eog_channels` | `null` | EOG channel names (e.g., `["Fp1", "Fp2"]` for virtual EOG) |

`"extended_infomax"` is the required algorithm when using ICLabel (Step 4), because
the ICLabel classifier was trained on extended infomax decompositions. Using a different
algorithm degrades classification accuracy.

---

## 8. Step 4 — ICA Component Labeling

**Module:** `pipeline/ica.py` → `run_ica_label()`

Automated classification of ICA components using MNE-ICAlabel, which wraps the
ICLabel deep learning classifier.

### 8.1 Method

1. Load the fitted ICA object (`*_proc-icafit_ica.fif`) and its epochs
   (`*_proc-icafit_epo.fif`).
2. Apply average reference to the epochs (required by ICLabel).
3. Classify each component via `label_components(epochs, ica, method="iclabel")`.
   ICLabel assigns each component a probability $p_k$ over 7 classes.
4. Exclude component $i$ when:
   - $\max_k p_k > \texttt{probability\_threshold}$ (default 0.8), **and**
   - $\arg\max_k p_k \notin \texttt{labels\_to\_keep}$ (default `["brain", "other"]`).
5. Write component status to `*_proc-ica_components.tsv`:
   - `status`: `"good"` or `"bad"`
   - `status_description`: Reason for exclusion
   - `mne_icalabel_labels`: Predicted class label
   - `mne_icalabel_proba`: Predicted class probability
6. Save the updated ICA object with `ica.exclude` set.
7. Write per-file log (`icalabel_task_<task>_log.csv`).

### 8.2 ICLabel Classes

| Class | Description | Default action |
|-------|-------------|----------------|
| Brain | Neural activity | Retained |
| Muscle | EMG artifact | Excluded if $p > 0.8$ |
| Eye | Ocular artifact (blinks, saccades) | Excluded if $p > 0.8$ |
| Heart | Cardiac artifact | Excluded if $p > 0.8$ |
| Line Noise | Power line interference | Excluded if $p > 0.8$ |
| Channel Noise | Electrode/hardware noise | Excluded if $p > 0.8$ |
| Other | Unclassifiable / mixed | Retained |

### 8.3 Configuration

| Key | Default | Description |
|-----|---------|-------------|
| `ica.probability_threshold` | `0.8` | Minimum probability to classify a component as artifact |
| `ica.labels_to_keep` | `["brain", "other"]` | ICLabel classes to retain |
| `icalabel.keep_mnebids_bads` | `false` | Preserve previously flagged bad components from `components.tsv` |

---

## 9. Step 5 — Epoch Creation and Artifact Rejection

**Module:** `pipelines/preprocessing.py` → `_run_epoch_creation()`

Epoch segmentation and final artifact rejection are delegated to MNE-BIDS-Pipeline.

### 9.1 MNE-BIDS-Pipeline Steps

| Step | Description |
|------|-------------|
| `preprocessing/_07_make_epochs` | Segment continuous data into epochs around events |
| `preprocessing/_08a_apply_ica` | Subtract excluded ICA components from epoched data |
| `preprocessing/_09_ptp_reject` | Reject remaining bad epochs via PTP threshold or autoreject |

### 9.2 Epoch Configuration

| Key | Default | Description |
|-----|---------|-------------|
| `epochs.tmin` | `-7.0` | Epoch start time relative to event onset (s) |
| `epochs.tmax` | `15.0` | Epoch end time relative to event onset (s) |
| `epochs.baseline` | `[-0.2, 0.0]` | Baseline correction window (s); `null` to skip |
| `epochs.conditions` | *(auto-detected)* | `trial_type` values to epoch |
| `epochs.reject` | `"autoreject_local"` | Rejection method (see below) |
| `epochs.autoreject_n_interpolate` | `[4, 8, 16]` | Channels to interpolate per trial (autoreject cross-validation) |
| `epochs.reject_tmin` | `null` | Optional start of PTP rejection window (s) |
| `epochs.reject_tmax` | `null` | Optional end of PTP rejection window (s) |

### 9.3 Rejection Methods

| Method | Description |
|--------|-------------|
| `"autoreject_local"` | Per-channel, per-trial adaptive thresholding. Interpolates a configurable number of bad channels per trial before rejecting epochs that remain too noisy. Recommended for long epochs. |
| `"autoreject_global"` | Global PTP threshold estimated by autoreject across all channels. |
| `{"eeg": <value>}` | Fixed PTP threshold in volts (e.g., `{"eeg": 150e-6}`). |
| `null` / `"none"` | No epoch rejection. |

### 9.4 Condition Auto-Detection

When `epochs.conditions` is not set, the pipeline reads the first available BIDS
`events.tsv` and extracts unique `trial_type` values.
Heuristic filtering removes scanner/housekeeping markers (`Volume`, `Pulse`,
`SyncStatus`, `New Segment`, `Bad`, `EDGE`, `Response`) and prefers task-relevant
trigger prefixes defined in `preprocessing.condition_preferred_prefixes`
(default: `Trig_`).

### 9.5 Resting-State Data

Set `preprocessing.task_is_rest = true`. MNE-BIDS-Pipeline will segment the
continuous recording into fixed-duration epochs (`rest_epochs_duration`) with
optional overlap (`rest_epochs_overlap`) instead of event-locked epochs.

---

## 10. Step 6 — Clean Events Export

**Module:** `utils/data/preprocessing.py` → `write_clean_events_tsv_for_epochs()`

After epoch rejection, a clean events table is written to derivatives containing
only events for kept (non-rejected) epochs. This enables downstream analyses to
align behavioral data with surviving trials.

### 10.1 Method

1. Load the original BIDS `events.tsv` (combined across runs for multi-run data).
2. Filter to rows matching the epoching conditions.
3. Use `epochs.selection` (MNE's record of kept epoch indices) to select surviving
   event rows.
4. Write `*_proc-clean_events.tsv` alongside the clean epochs file with columns:
   - `trial_id` — Canonical 0-based trial identifier for the kept clean-event rows.
   - `epoch_index` — 0-based index into the clean epochs object.
   - `event_index` — Original index within the condition-filtered event set.
   - All original event columns (onset, duration, trial_type, behavioral columns, …).

`trial_id` is the canonical alignment contract for downstream trialwise artifacts.
Behavior analysis and feature-table loading require it; they do not infer joins
from paradigm-specific columns such as `trial_number`, `onset`, or `duration`.

### 10.2 Configuration

| Key | Default | Description |
|-----|---------|-------------|
| `preprocessing.write_clean_events` | `true` | Enable clean events export |
| `preprocessing.clean_events_overwrite` | `true` | Overwrite existing clean events files |
| `preprocessing.clean_events_strict` | `true` | Raise error if clean events cannot be written |

---

## 11. Step 7 — Preprocessing Statistics

**Module:** `pipeline/stats.py` → `collect_preprocessing_stats()`

Collects per-subject summary statistics across all processed subjects for QC.

### 11.1 Metrics

| Metric | Description |
|--------|-------------|
| `n_bad_channels` | Bad channels detected (from `*_bads.tsv`) |
| `n_bad_ica` | Excluded ICA components (from `*_components.tsv`) |
| `total_clean_epochs` | Epochs surviving rejection |
| `n_removed_epochs` | Epochs rejected |
| `boundary_n_removed_epochs` | Epochs rejected due to `BAD boundary` annotations |
| `<condition>_total_clean_epochs` | Per-condition epoch counts |

### 11.2 Output Files

| File | Content |
|------|---------|
| `task_<task>_preprocessing_stats.tsv` | Per-subject statistics table |
| `task_<task>_preprocessing_stats_desc.tsv` | Descriptive statistics (mean, std, min, max) across subjects |

---

## 12. Step 8 — Time-Frequency Representation (Optional)

**Module:** `pipeline/tfr.py` → `custom_tfr()`

Optional Morlet wavelet time-frequency decomposition on clean epochs.

### 12.1 Method

1. Load clean epochs (`*_proc-clean_epo.fif`).
2. Optionally interpolate bad channels (default: enabled).
3. Compute TFR via `mne.time_frequency.tfr_morlet()`:
   - Frequencies: configurable range (default 1–99 Hz in 1 Hz steps).
   - Cycles: adaptive $n_\text{cycles}(f) = f / 3$ by default (higher frequency → more cycles → better frequency resolution).
   - Decimation: configurable factor to reduce time resolution and memory.
4. Optionally compute ITC.
5. Save per-condition averaged and/or single-trial TFR objects as `.h5` files.

### 12.2 Configuration

| Key | Default | Description |
|-----|---------|-------------|
| `custom_tfr_freqs` | `np.arange(1, 100, 1)` | Frequency vector (Hz) |
| `custom_tfr_n_cycles` | `freqs / 3.0` | Cycles per frequency (adaptive) |
| `custom_tfr_decim` | `2` | Decimation factor |
| `custom_tfr_return_itc` | `false` | Compute inter-trial coherence |
| `custom_tfr_interpolate_bads` | `true` | Interpolate bad channels before TFR |
| `custom_tfr_average` | `false` | Average across epochs (vs. single-trial) |
| `custom_tfr_return_average` | `true` | Also save per-condition averages |
| `custom_tfr_crop` | `null` | Crop TFR to time window `[tmin, tmax]` |

---

## 13. Execution Modes

| Mode | Steps executed |
|------|---------------|
| `full` | Bad channels → ICA fit → ICA label → Epochs → Statistics |
| `bad-channels` | Bad channel detection only |
| `ica` | ICA fitting + ICA labeling only |
| `epochs` | Epoch creation + statistics (requires ICA already fitted) |

### 13.1 Python API

```python
from eeg_pipeline.pipelines.preprocessing import PreprocessingPipeline

pipeline = PreprocessingPipeline(config=config)

# Full preprocessing for all subjects
pipeline.run_batch(subjects=["001", "002", "003"], task="task", mode="full")

# Re-run epoch creation after adjusting epoch parameters
pipeline.run_batch(subjects=["001"], task="task", mode="epochs")

# Single subject
pipeline.process_subject("001", task="task", mode="full")
```

### 13.2 Unified CLI Interface

```bash
python -m eeg_pipeline.cli.main preprocessing full --all-subjects --task <task>
```

Use the unified CLI (`eeg_pipeline.cli.main`) for preprocessing runs. Runtime
overrides are supported via repeatable `--set KEY=VALUE`.

---

## 14. Configuration Reference

All preprocessing parameters are defined under the following sections in
`eeg_config.yaml`. The configuration within each step section (§5–§12) lists the
per-step keys; the template below shows the complete structure.

```yaml
eeg:
  montage: "easycap-M1"
  ch_types: "eeg"
  reference: "average"
  eog_channels: null
  ecg_channels: ["ECG"]

preprocessing:
  resample_freq: 500
  l_freq: 0.1
  h_freq: 100
  notch_freq: 60
  n_jobs: 1
  find_breaks: true
  random_state: 42
  write_clean_events: true
  clean_events_overwrite: true
  clean_events_strict: true

pyprep:
  ransac: false
  repeats: 3
  average_reref: false
  file_extension: ".vhdr"
  consider_previous_bads: false
  overwrite_chans_tsv: true
  delete_breaks: false
  breaks_min_length: 20
  t_start_after_previous: 2
  t_stop_before_next: 2

ica:
  algorithm: "extended_infomax"
  n_components: 0.99
  l_freq: 1.0
  probability_threshold: 0.8
  labels_to_keep: ["brain", "other"]

epochs:
  tmin: -7.0
  tmax: 15.0
  baseline: [-0.2, 0.0]
  reject: "autoreject_local"
  autoreject_n_interpolate: [4, 8, 16]
```

---

## 15. Output Structure

```
derivatives/preprocessed/eeg/
├── sub-XXXX/
│   └── eeg/
│       ├── sub-XXXX_task-<task>_proc-icafit_ica.fif       # Fitted ICA object
│       ├── sub-XXXX_task-<task>_proc-icafit_epo.fif       # Epochs used for ICA fitting
│       ├── sub-XXXX_task-<task>_proc-ica_ica.fif          # ICA with exclusions set
│       ├── sub-XXXX_task-<task>_proc-ica_components.tsv   # Component labels and probabilities
│       ├── sub-XXXX_task-<task>_proc-clean_epo.fif        # Final clean epochs
│       ├── sub-XXXX_task-<task>_proc-clean_events.tsv     # Events for kept epochs
│       ├── sub-XXXX_task-<task>_bads.tsv                  # Bad channel record
│       ├── sub-XXXX_task-<task>_power_epo-tfr.h5          # TFR power (optional)
│       └── sub-XXXX_task-<task>_itc_epo-tfr.h5            # TFR ITC (optional)
├── pyprep_task_<task>_log.csv                             # PyPREP detection log
├── icalabel_task_<task>_log.csv                           # ICLabel classification log
├── task_<task>_preprocessing_stats.tsv                    # Per-subject statistics
└── task_<task>_preprocessing_stats_desc.tsv               # Descriptive statistics
```

---

## 16. Dependencies

| Package | Minimum version | Role |
|---------|----------------|------|
| **mne** | ≥ 1.0.0 | Core EEG processing |
| **mne-bids** | ≥ 0.4.0 | BIDS I/O and path management |
| **mne-bids-pipeline** | ≥ 0.1.0 | Automated preprocessing steps (ICA, epochs) |
| **pyprep** | ≥ 0.4.0 | Bad channel detection (NoisyChannels) |
| **mne-icalabel** | ≥ 0.3.0 | ICA component classification (ICLabel) |
| **pandas** | ≥ 1.3.0 | Tabular data handling |
| **numpy** | ≥ 1.20.0 | Numerical operations |
| **scipy** | ≥ 1.7.0 | Signal processing |
| **joblib** | ≥ 1.0.0 | Parallel execution |
| **matplotlib** | ≥ 3.4.0 | Plotting backend |
