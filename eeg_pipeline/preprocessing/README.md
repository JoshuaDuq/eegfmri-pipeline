# EEG Preprocessing Pipeline (forked from the coll_lab_eeg_pipeline)

Automated, reproducible EEG preprocessing built on [MNE-Python](https://mne.tools/), [MNE-BIDS-Pipeline](https://mne.tools/mne-bids-pipeline/), [PyPREP](https://pyprep.readthedocs.io/), and [MNE-ICAlabel](https://mne.tools/mne-icalabel/). The pipeline operates on BIDS-formatted EEG data and produces clean, epoched datasets ready for feature extraction and statistical analysis.

---

## Table of Contents

1. [Pipeline Overview](#pipeline-overview)
2. [Prerequisites](#prerequisites)
3. [Input Data Requirements](#input-data-requirements)
4. [Pipeline Steps](#pipeline-steps)
   - [Step 1: Bad Channel Detection (PyPREP)](#step-1-bad-channel-detection-pyprep)
   - [Step 2: Bad Channel Synchronization Across Runs](#step-2-bad-channel-synchronization-across-runs)
   - [Step 3: ICA Fitting (MNE-BIDS-Pipeline)](#step-3-ica-fitting-mne-bids-pipeline)
   - [Step 4: ICA Component Labeling (MNE-ICAlabel)](#step-4-ica-component-labeling-mne-icalabel)
   - [Step 5: Epoch Creation and Artifact Rejection](#step-5-epoch-creation-and-artifact-rejection)
   - [Step 6: Clean Events Export](#step-6-clean-events-export)
   - [Step 7: Preprocessing Statistics Collection](#step-7-preprocessing-statistics-collection)
   - [Step 8: Time-Frequency Representation (Optional)](#step-8-time-frequency-representation-optional)
5. [Execution Modes](#execution-modes)
6. [Configuration Reference](#configuration-reference)
7. [Output Structure](#output-structure)
8. [Module Reference](#module-reference)

---

## Pipeline Overview

```
BIDS EEG Data (.vhdr/.edf)
        │
        ▼
┌───────────────────────────────────┐
│  1. Bad Channel Detection         │  PyPREP (NoisyChannels)
│     - Deviation criterion         │  Repeated N times, union of bads
│     - Correlation criterion       │  Optional: RANSAC
│     - Optional: RANSAC            │
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
│     - Data quality check          │  Steps: init → _01 → _04 → _05 → _06a1
│     - Bandpass filtering          │
│     - Artifact regression         │
│     - ICA decomposition           │
└───────────┬───────────────────────┘
            │
            ▼
┌───────────────────────────────────┐
│  4. ICA Component Labeling        │  MNE-ICAlabel (ICLabel classifier)
│     - Probabilistic classification│  Threshold: prob > 0.8
│     - Exclude non-brain/other     │  Labels kept: ["brain", "other"]
└───────────┬───────────────────────┘
            │
            ▼
┌───────────────────────────────────┐
│  5. Epoch Creation & Rejection    │  MNE-BIDS-Pipeline subprocess
│     - Epoch segmentation          │  Steps: _07 → _08a → _09
│     - ICA component removal       │
│     - Peak-to-peak / autoreject   │
└───────────┬───────────────────────┘
            │
            ▼
┌───────────────────────────────────┐
│  6. Clean Events Export           │  Epoch-aligned events.tsv
│     (rejected epochs excluded)    │  Written to derivatives
└───────────┬───────────────────────┘
            │
            ▼
┌───────────────────────────────────┐
│  7. Preprocessing Statistics      │  Per-subject summary TSV
│     - Bad channels count          │  + descriptive statistics
│     - Bad ICA components count    │
│     - Epoch rejection counts      │
└───────────┬───────────────────────┘
            │
            ▼
┌───────────────────────────────────┐
│  8. TFR Computation (optional)    │  Morlet wavelets
│     - Power / ITC per condition   │  Configurable freq range & decimation
└───────────────────────────────────┘
```

---

## Prerequisites

### Python Dependencies

| Package              | Minimum Version | Role                                        |
|----------------------|-----------------|---------------------------------------------|
| `mne`                | ≥ 1.0.0         | Core EEG processing                         |
| `mne-bids`           | ≥ 0.4.0         | BIDS I/O and path management                |
| `mne-bids-pipeline`  | ≥ 0.1.0         | Automated preprocessing steps (ICA, epochs) |
| `pyprep`             | ≥ 0.4.0         | Bad channel detection (NoisyChannels)       |
| `mne-icalabel`       | ≥ 0.3.0         | ICA component classification (ICLabel)      |
| `pandas`             | ≥ 1.3.0         | Tabular data handling                       |
| `numpy`              | ≥ 1.20.0        | Numerical operations                        |
| `scipy`              | ≥ 1.7.0         | Signal processing                           |
| `joblib`             | ≥ 1.0.0         | Parallel execution                          |
| `matplotlib`         | ≥ 3.4.0         | Plotting backend                            |

Install all dependencies:

```bash
pip install -r eeg_pipeline/preprocessing/requirements.txt
```

---

## Input Data Requirements

The pipeline expects **BIDS-formatted** EEG data:

```
bids_root/
├── dataset_description.json
├── sub-001/
│   └── eeg/
│       ├── sub-001_task-<task>_eeg.vhdr
│       ├── sub-001_task-<task>_eeg.vmrk
│       ├── sub-001_task-<task>_eeg.eeg
│       ├── sub-001_task-<task>_channels.tsv
│       └── sub-001_task-<task>_events.tsv
├── sub-002/
│   └── ...
```

- **File format**: BrainVision (`.vhdr`) by default; configurable via `pyprep.file_extension`.
- **`channels.tsv`**: Must exist alongside each EEG file. Used to identify channel types (`EEG`, `EOG`, `ECG`, `EMG`, `MISC`) and to read/write bad channel status.
- **`events.tsv`**: Required for event-related (non-resting-state) paradigms. Must contain a `trial_type` column. Multi-run designs use per-run files (`run-01_events.tsv`, etc.) that are combined automatically.
- **Montage**: A standard electrode montage (default: `easycap-M1`) is applied if the raw data lacks digitized positions.

---

## Pipeline Steps

### Step 1: Bad Channel Detection (PyPREP)

**Module**: `preprocessing/pipeline/preprocess.py` → `run_bads_detection()`

Automated detection of noisy channels using [PyPREP's `NoisyChannels`](https://pyprep.readthedocs.io/) class. This step operates on continuous (raw) data before any ICA or epoching.

#### Method

1. **Load raw data** via `mne_bids.read_raw_bids()` (preferred) or `mne.io.read_raw()`.
2. **Set montage** (`easycap-M1` by default) if the recording lacks digitized electrode positions.
3. **Optional low-pass filter** at `l_pass` Hz (default: 100 Hz) on EEG channels only, to reduce high-frequency noise before detection.
4. **Optional notch filter** at `notch` Hz (e.g., 60 Hz) on EEG channels to suppress line noise.
5. **Optional average re-reference** before detection (disabled by default).
6. **Iterative bad channel detection** (repeated `repeats` times, default: 3):
   - `NoisyChannels.find_bad_by_deviation()` — flags channels whose robust z-scored amplitude deviates from the median across channels.
   - `NoisyChannels.find_bad_by_correlation()` — flags channels with low Pearson correlation to neighboring channels.
   - `NoisyChannels.find_bad_by_ransac()` — (optional, disabled by default) flags channels that cannot be predicted from their neighbors via RANSAC interpolation.
   - After each iteration, detected bads are accumulated (union) and marked in `raw.info["bads"]`, allowing subsequent iterations to detect additional channels that were masked by the initial bad channels.
7. **Custom bad channels** can be injected via `custom_bad_dict` (per-task, per-subject dictionary).
8. **Write results** back to `channels.tsv` (sets `status` column to `"bad"` for detected channels).
9. **Log** a per-file CSV report (`pyprep_task_<task>_log.csv`) recording detected bads, parameters used, and any errors.

#### Configuration Parameters

| Parameter                  | Default        | Description                                                                 |
|----------------------------|----------------|-----------------------------------------------------------------------------|
| `pyprep.ransac`            | `false`        | Enable RANSAC-based detection (computationally expensive)                   |
| `pyprep.repeats`           | `3`            | Number of detection iterations (union of bads across iterations)            |
| `pyprep.average_reref`     | `false`        | Average re-reference before detection                                       |
| `pyprep.file_extension`    | `".vhdr"`      | Raw data file extension                                                     |
| `pyprep.consider_previous_bads` | `false`   | Retain previously marked bads in `channels.tsv`                             |
| `pyprep.overwrite_chans_tsv`    | `true`    | Overwrite original `channels.tsv` (vs. writing a separate file)             |
| `pyprep.delete_breaks`     | `false`        | Annotate recording breaks (not cropped; for logging only)                   |
| `pyprep.breaks_min_length` | `20`           | Minimum break duration (seconds) to annotate                                |
| `pyprep.t_start_after_previous` | `2`       | Seconds after last event to start break search                              |
| `pyprep.t_stop_before_next`     | `2`       | Seconds before next event to end break search                               |
| `pyprep.custom_bad_dict`   | `null`         | Manual bad channels: `{task: {subject: [channels]}}`                        |
| `preprocessing.h_freq`     | `100`          | Low-pass filter cutoff applied before PyPREP detection                      |
| `preprocessing.notch_freq` | `60`           | Notch filter frequency applied before PyPREP detection                      |
| `eeg.montage`              | `"easycap-M1"` | Standard montage name                                                       |

#### Parallelization

Bad channel detection supports parallel execution across files via `n_jobs` (uses `joblib.Parallel`). Each file is processed independently.

---

### Step 2: Bad Channel Synchronization Across Runs

**Module**: `preprocessing/pipeline/preprocess.py` → `synchronize_bad_channels_across_runs()`

For multi-run paradigms, bad channels detected in **any** run are propagated to **all** runs of the same subject. This ensures consistent channel sets across runs before ICA fitting.

#### Method

1. For each subject, glob all `channels.tsv` files matching the task.
2. Collect the **union** of all channels marked `status == "bad"` across runs.
3. Write the unified bad channel set back to every run's `channels.tsv`.

This step is critical because MNE-BIDS-Pipeline reads `channels.tsv` to determine which channels to exclude from ICA fitting and interpolation. Inconsistent bads across runs would produce incompatible ICA decompositions.

---

### Step 3: ICA Fitting (MNE-BIDS-Pipeline)

**Module**: `pipelines/preprocessing.py` → `_run_ica_fitting()`

ICA fitting is delegated to [MNE-BIDS-Pipeline](https://mne.tools/mne-bids-pipeline/) via a subprocess call. A temporary Python config file is generated from the YAML configuration and passed to the pipeline.

#### MNE-BIDS-Pipeline Steps Executed

| Step                                  | Description                                                        |
|---------------------------------------|--------------------------------------------------------------------|
| `init`                                | Initialize pipeline, validate BIDS dataset                         |
| `preprocessing/_01_data_quality`      | Data quality assessment and Maxwell filtering (if applicable)      |
| `preprocessing/_04_frequency_filter`  | Bandpass filter: `l_freq` – `h_freq` (default: 0.1–100 Hz)        |
| `preprocessing/_05_regress_artifact`  | Regress out artifact signals (e.g., EOG/ECG regression)            |
| `preprocessing/_06a1_fit_ica`         | Fit ICA decomposition                                              |
| `preprocessing/_06a2_find_ica_artifacts` | *(Only when `use_icalabel=false`)* MNE's built-in EOG/ECG correlation-based artifact detection |

#### ICA Configuration

| Parameter              | Default              | Description                                                                 |
|------------------------|----------------------|-----------------------------------------------------------------------------|
| `ica.algorithm`        | `"extended_infomax"` | ICA algorithm (`"extended_infomax"`, `"picard"`, `"fastica"`)               |
| `ica.n_components`     | `0.99`               | Number of components (float = variance explained; int = exact count)        |
| `ica.l_freq`           | `1.0`                | High-pass filter cutoff for ICA fitting (Hz); removes slow drifts           |
| `ica.reject`           | `null`               | Peak-to-peak rejection for ICA fitting epochs (e.g., `{"eeg": 300e-6}`)    |
| `preprocessing.l_freq` | `0.1`                | Main bandpass high-pass cutoff (Hz)                                         |
| `preprocessing.h_freq` | `100`                | Main bandpass low-pass cutoff (Hz)                                          |
| `preprocessing.notch_freq` | `60`             | Notch filter frequency (Hz); `null` to skip                                |
| `preprocessing.resample_freq` | `500`         | Resampling frequency (Hz); `null` to skip                                  |
| `preprocessing.find_breaks` | `true`           | Annotate recording breaks                                                   |
| `preprocessing.random_state` | `42`            | Random seed for reproducibility                                             |
| `eeg.reference`        | `"average"`          | EEG reference (`"average"` or specific channel)                             |
| `eeg.ch_types`         | `"eeg"`              | Channel types to process                                                    |
| `eeg.eog_channels`     | `null`               | EOG channel names (e.g., `["Fp1", "Fp2"]` for virtual EOG)                 |

#### Why Extended Infomax?

The default algorithm is `extended_infomax` because it is the recommended algorithm for [MNE-ICAlabel](https://mne.tools/mne-icalabel/), which uses the ICLabel classifier trained on extended infomax decompositions. Using a different ICA algorithm may degrade ICLabel classification accuracy.

---

### Step 4: ICA Component Labeling (MNE-ICAlabel)

**Module**: `preprocessing/pipeline/ica.py` → `run_ica_label()`

Automated classification of ICA components using [MNE-ICAlabel](https://mne.tools/mne-icalabel/), which wraps the [ICLabel](https://labeling.ucsd.edu/tutorial/about) deep learning classifier.

#### Method

1. **Load** the fitted ICA object (`*_proc-icafit_ica.fif`) and its corresponding epochs (`*_proc-icafit_epo.fif`).
2. **Apply average reference** to the epochs (required by ICLabel).
3. **Classify** each component using `label_components(epochs, ica, method="iclabel")`.
   - ICLabel assigns each component a probability distribution over 7 classes: **Brain**, **Muscle**, **Eye**, **Heart**, **Line Noise**, **Channel Noise**, **Other**.
4. **Exclude components** where:
   - The predicted class probability exceeds `prob_threshold` (default: 0.8), **and**
   - The predicted label is **not** in `labels_to_keep` (default: `["brain", "other"]`).
5. **Write** the component status to a BIDS-compliant `*_proc-ica_components.tsv` file, including:
   - `status`: `"good"` or `"bad"`
   - `status_description`: Reason for exclusion
   - `mne_icalabel_labels`: Predicted class label
   - `mne_icalabel_proba`: Predicted class probability
6. **Save** the updated ICA object with `ica.exclude` set to the bad component indices.
7. **Log** a per-file TSV report (`icalabel_task_<task>_log.csv`).

#### Configuration Parameters

| Parameter                     | Default              | Description                                                    |
|-------------------------------|----------------------|----------------------------------------------------------------|
| `ica.probability_threshold`   | `0.8`                | Minimum probability to classify a component as artifact        |
| `ica.labels_to_keep`          | `["brain", "other"]` | ICLabel classes to retain (all others are excluded if above threshold) |
| `icalabel.keep_mnebids_bads`  | `false`              | Preserve previously flagged bad components from `components.tsv` |

#### ICLabel Classes

| Class          | Description                                    | Default Action       |
|----------------|------------------------------------------------|----------------------|
| Brain          | Neural activity                                | **Kept**             |
| Muscle         | EMG artifact                                   | Excluded if p > 0.8  |
| Eye            | Ocular artifact (blinks, saccades)             | Excluded if p > 0.8  |
| Heart          | Cardiac artifact                               | Excluded if p > 0.8  |
| Line Noise     | Power line interference                        | Excluded if p > 0.8  |
| Channel Noise  | Electrode/hardware noise                       | Excluded if p > 0.8  |
| Other          | Unclassifiable / mixed                         | **Kept**             |

---

### Step 5: Epoch Creation and Artifact Rejection

**Module**: `pipelines/preprocessing.py` → `_run_epoch_creation()`

Epoch segmentation and final artifact rejection are delegated to MNE-BIDS-Pipeline.

#### MNE-BIDS-Pipeline Steps Executed

| Step                              | Description                                                              |
|-----------------------------------|--------------------------------------------------------------------------|
| `preprocessing/_07_make_epochs`   | Segment continuous data into epochs around events                        |
| `preprocessing/_08a_apply_ica`    | Subtract excluded ICA components from the epoched data                   |
| `preprocessing/_09_ptp_reject`    | Reject remaining bad epochs via peak-to-peak threshold or autoreject     |

#### Epoch Configuration

| Parameter                        | Default              | Description                                                                 |
|----------------------------------|----------------------|-----------------------------------------------------------------------------|
| `epochs.tmin`                    | `-7.0`               | Epoch start time relative to event onset (seconds)                          |
| `epochs.tmax`                    | `15.0`               | Epoch end time relative to event onset (seconds)                            |
| `epochs.baseline`                | `[-0.2, 0.0]`        | Baseline correction window (seconds); `null` to skip                        |
| `epochs.conditions`              | *(auto-detected)*    | Event `trial_type` values to epoch; auto-detected from BIDS `events.tsv`    |
| `epochs.reject`                  | `"autoreject_local"` | Rejection method (see below)                                                |
| `epochs.autoreject_n_interpolate`| `[4, 8, 16]`        | Number of channels to interpolate per trial (autoreject cross-validation)   |
| `epochs.reject_tmin`             | `null`               | Optional: restrict PTP rejection to this time window start                  |
| `epochs.reject_tmax`             | `null`               | Optional: restrict PTP rejection to this time window end                    |

#### Rejection Methods

| Method               | Description                                                                                     |
|----------------------|-------------------------------------------------------------------------------------------------|
| `"autoreject_local"` | Per-channel, per-trial adaptive thresholding via [autoreject](https://autoreject.github.io/). Interpolates a configurable number of bad channels per trial before rejecting trials that remain too noisy. **Recommended for long epochs.** |
| `"autoreject_global"` | Global peak-to-peak threshold estimated by autoreject across all channels.                     |
| `{"eeg": <value>}`  | Fixed peak-to-peak threshold in volts (e.g., `{"eeg": 150e-6}` for 150 µV).                    |
| `null` / `"none"`   | No epoch rejection.                                                                             |

#### Condition Auto-Detection

When `epochs.conditions` is not set, the pipeline reads the first available BIDS `events.tsv` and extracts unique `trial_type` values. Heuristic filtering excludes scanner/housekeeping markers (`Volume`, `Pulse`, `SyncStatus`, `New Segment`, `Bad`, `EDGE`, `Response`) and prefers task-relevant triggers (e.g., `Trig_therm*` prefixes for thermal pain paradigms).

#### Resting-State Data

Set `preprocessing.task_is_rest: true` in the configuration. MNE-BIDS-Pipeline will segment the continuous recording into fixed-duration epochs (`rest_epochs_duration`) with optional overlap (`rest_epochs_overlap`) instead of event-locked epochs.

---

### Step 6: Clean Events Export

**Module**: `utils/data/preprocessing.py` → `write_clean_events_tsv_for_epochs()`

After epoch rejection, a **clean events table** is written to derivatives. This table contains only the events corresponding to **kept** (non-rejected) epochs, enabling downstream analyses to align behavioral data with the surviving trials.

#### Method

1. Load the original BIDS `events.tsv` (combined across runs if multi-run).
2. Filter to rows matching the epoching conditions.
3. Use `epochs.selection` (MNE's record of kept epoch indices) to select the surviving event rows.
4. Write `*_proc-clean_events.tsv` alongside the clean epochs file, with columns:
   - `epoch_index`: 0-based index into the clean epochs object
   - `event_index`: Original index within the condition-filtered event set
   - All original event columns (onset, duration, trial_type, behavioral columns, etc.)

#### Configuration

| Parameter                              | Default | Description                                           |
|----------------------------------------|---------|-------------------------------------------------------|
| `preprocessing.write_clean_events`     | `true`  | Enable clean events export                            |
| `preprocessing.clean_events_overwrite` | `true`  | Overwrite existing clean events files                 |
| `preprocessing.clean_events_strict`    | `true`  | Raise error if clean events cannot be written         |

---

### Step 7: Preprocessing Statistics Collection

**Module**: `preprocessing/pipeline/stats.py` → `collect_preprocessing_stats()`

Collects summary statistics across all processed subjects for quality control.

#### Metrics Collected

| Metric                           | Description                                                     |
|----------------------------------|-----------------------------------------------------------------|
| `n_bad_channels`                 | Number of bad channels detected (from `*_bads.tsv`)             |
| `n_bad_ica`                      | Number of excluded ICA components (from `*_components.tsv`)     |
| `total_clean_epochs`             | Number of epochs surviving rejection                            |
| `n_removed_epochs`               | Number of epochs rejected                                       |
| `boundary_n_removed_epochs`      | Epochs rejected due to `BAD boundary` annotations               |
| `<condition>_total_clean_epochs` | Per-condition epoch counts                                      |

#### Output Files

- `task_<task>_preprocessing_stats.tsv` — Per-subject statistics table.
- `task_<task>_preprocessing_stats_desc.tsv` — Descriptive statistics (mean, std, min, max, etc.) across subjects.

---

### Step 8: Time-Frequency Representation (Optional)

**Module**: `preprocessing/pipeline/tfr.py` → `custom_tfr()`

Computes Morlet wavelet time-frequency decompositions on clean epochs. This step is optional and typically configured separately from the core preprocessing.

#### Method

1. Load clean epochs (`*_proc-clean_epo.fif`).
2. Optionally **interpolate bad channels** (default: enabled).
3. Compute TFR via `mne.time_frequency.tfr_morlet()`:
   - **Frequencies**: Configurable range (default: 1–99 Hz in 1 Hz steps).
   - **Cycles**: Adaptive `n_cycles = freqs / 3.0` by default (higher frequency → more cycles → better frequency resolution).
   - **Decimation**: Configurable factor to reduce time resolution and memory.
4. Optionally compute **inter-trial coherence (ITC)**.
5. Save per-condition averaged and/or single-trial TFR objects as `.h5` files.

#### Configuration Parameters

| Parameter                      | Default                | Description                                              |
|--------------------------------|------------------------|----------------------------------------------------------|
| `custom_tfr_freqs`             | `np.arange(1, 100, 1)` | Frequency vector (Hz)                                   |
| `custom_tfr_n_cycles`          | `freqs / 3.0`          | Cycles per frequency (adaptive)                         |
| `custom_tfr_decim`             | `2`                    | Decimation factor                                        |
| `custom_tfr_return_itc`        | `false`                | Compute inter-trial coherence                            |
| `custom_tfr_interpolate_bads`  | `true`                 | Interpolate bad channels before TFR                      |
| `custom_tfr_average`           | `false`                | Average across epochs (vs. single-trial)                 |
| `custom_tfr_return_average`    | `true`                 | Also save per-condition averages                         |
| `custom_tfr_crop`              | `null`                 | Time window to crop TFR `[tmin, tmax]`                   |

---

## Execution Modes

The pipeline supports selective execution via the `mode` parameter:

| Mode            | Steps Executed                                                          |
|-----------------|-------------------------------------------------------------------------|
| `full`          | Bad channels → ICA fit → ICA label → Epochs → Statistics               |
| `bad-channels`  | Bad channel detection only                                              |
| `ica`           | ICA fitting + ICA labeling only                                         |
| `epochs`        | Epoch creation + statistics only (assumes ICA already fitted)           |

### Usage

```python
from eeg_pipeline.pipelines.preprocessing import PreprocessingPipeline

pipeline = PreprocessingPipeline(config=config)

# Full preprocessing for all subjects
pipeline.run_batch(subjects=["001", "002", "003"], task="task", mode="full")

# Re-run only epoch creation after adjusting epoch parameters
pipeline.run_batch(subjects=["001"], task="task", mode="epochs")

# Single subject
pipeline.process_subject("001", task="task", mode="full")
```

### Legacy Script Interface

The pipeline can also be run via the standalone script:

```bash
python eeg_pipeline/preprocessing/scripts/run_pipeline.py <config_file.py>
```

This script reads a Python-format config file (generated via `create_config.py`) and executes all steps sequentially for each task listed in `tasks_to_process`.

---

## Configuration Reference

All preprocessing parameters are defined in the project YAML configuration file (`eeg_config.yaml`). The relevant sections are:

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

pyprep:
  ransac: true
  repeats: 3
  average_reref: false
  file_extension: ".vhdr"
  consider_previous_bads: true
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

## Output Structure

All outputs are written to the derivatives directory under `preprocessed/eeg/`:

```
derivatives/preprocessed/eeg/
├── sub-001/
│   └── eeg/
│       ├── sub-001_task-<task>_proc-icafit_ica.fif       # Fitted ICA object
│       ├── sub-001_task-<task>_proc-icafit_epo.fif       # Epochs used for ICA fitting
│       ├── sub-001_task-<task>_proc-ica_ica.fif          # ICA with exclusions set
│       ├── sub-001_task-<task>_proc-ica_components.tsv   # Component labels & probabilities
│       ├── sub-001_task-<task>_proc-clean_epo.fif        # Final clean epochs
│       ├── sub-001_task-<task>_proc-clean_events.tsv     # Events for kept epochs
│       ├── sub-001_task-<task>_bads.tsv                  # Bad channel record
│       ├── sub-001_task-<task>_power_epo-tfr.h5          # TFR power (optional)
│       └── sub-001_task-<task>_itc_epo-tfr.h5            # TFR ITC (optional)
├── pyprep_task_<task>_log.csv                            # PyPREP detection log
├── icalabel_task_<task>_log.csv                          # ICAlabel classification log
├── task_<task>_preprocessing_stats.tsv                   # Per-subject statistics
└── task_<task>_preprocessing_stats_desc.tsv              # Descriptive statistics
```

---

## Module Reference

### `preprocessing/pipeline/`

| Module          | Purpose                                                                 |
|-----------------|-------------------------------------------------------------------------|
| `preprocess.py` | Bad channel detection (`run_bads_detection`) and cross-run synchronization (`synchronize_bad_channels_across_runs`) |
| `ica.py`        | ICA component labeling via MNE-ICAlabel (`run_ica_label`)               |
| `stats.py`      | Preprocessing statistics collection (`collect_preprocessing_stats`)     |
| `tfr.py`        | Morlet wavelet TFR computation (`custom_tfr`)                           |
| `config.py`     | Configuration file read/write utilities                                 |
| `utils.py`      | BIDS file discovery, path manipulation, condition name sanitization      |
| `io.py`         | MNE object I/O, channels/components TSV read/write                      |

### `preprocessing/scripts/`

| Script              | Purpose                                                              |
|---------------------|----------------------------------------------------------------------|
| `run_pipeline.py`   | Standalone pipeline runner (reads Python config, executes all steps) |
| `create_config.py`  | Generates a template configuration file                              |

### `pipelines/preprocessing.py`

The `PreprocessingPipeline` class orchestrates all steps, generates temporary MNE-BIDS-Pipeline configs, and manages subprocess execution. It inherits from `PipelineBase` which provides batch processing, logging, error handling, and ledger tracking.
