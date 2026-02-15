# fMRI Analysis Pipeline

Subject-level fMRI analysis pipeline for thermal pain experiments. Converts raw DICOM data to BIDS, runs fMRIPrep preprocessing, fits first-level GLMs, computes statistical contrasts, extracts trial-wise beta maps, and quantifies multivariate pain signature expression (NPS, SIIPS1).

Designed for integration with EEG source localization: the resulting fMRI statistical maps constrain EEG inverse solutions.

---

## Table of Contents

1. [Pipeline Overview](#pipeline-overview)
2. [Directory Structure](#directory-structure)
3. [Stage 1 — Raw-to-BIDS Conversion](#stage-1--raw-to-bids-conversion)
4. [Stage 2 — fMRIPrep Preprocessing](#stage-2--fmriprep-preprocessing)
5. [Stage 3 — First-Level GLM Contrast Analysis](#stage-3--first-level-glm-contrast-analysis)
6. [Stage 4 — Trial-Wise Signature Extraction](#stage-4--trial-wise-signature-extraction)
7. [Stage 5 — Reporting and Quality Control](#stage-5--reporting-and-quality-control)
8. [BEM and Coregistration Generation](#bem-and-coregistration-generation)
9. [Multivariate Pain Signatures](#multivariate-pain-signatures)
10. [Configuration Reference](#configuration-reference)
11. [Dependencies](#dependencies)

---

## Pipeline Overview

```
DICOM ──► BIDS ──► fMRIPrep ──► First-Level GLM ──► Contrast Maps ──► EEG Source Localization
                                      │
                                      └──► Trial-Wise Betas ──► Signature Expression (NPS/SIIPS1)
```

The pipeline consists of five stages, each implemented as an independent module:

| Stage | Module | Purpose |
|-------|--------|---------|
| 1 | `analysis/raw_to_bids.py` | DICOM → BIDS conversion via `dcm2niix` |
| 2 | `pipelines/fmri_preprocessing.py` | fMRIPrep containerized preprocessing |
| 3 | `pipelines/fmri_analysis.py` + `analysis/contrast_builder.py` | Multi-run first-level GLM and contrast computation |
| 4 | `pipelines/fmri_trial_signatures.py` + `analysis/trial_signatures.py` | Trial-wise beta estimation and signature readout |
| 5 | `analysis/reporting.py` | HTML report generation with QC visualizations |

Supporting modules:

| Module | Purpose |
|--------|---------|
| `analysis/confounds_selection.py` | fMRIPrep confound regressor selection |
| `analysis/events_selection.py` | Trial-type filtering and normalization |
| `analysis/smoothing.py` | Spatial smoothing FWHM normalization |
| `analysis/pain_signatures.py` | NPS/SIIPS1 dot-product and correlation computation |
| `analysis/plotting_config.py` | Plot configuration and validation |
| `analysis/bem_generation.py` | Docker-based BEM model/solution/transform generation |

---

## Directory Structure

### Source Layout

```
fmri_pipeline/
├── __init__.py
├── analysis/
│   ├── bem_generation.py          # BEM model, solution, and coregistration transform
│   ├── confounds_selection.py     # fMRIPrep confound column selection
│   ├── contrast_builder.py        # First-level GLM fitting and contrast computation
│   ├── events_selection.py        # Trial-type normalization and filtering
│   ├── pain_signatures.py         # NPS/SIIPS1 multivariate signature computation
│   ├── plotting_config.py         # Plot configuration dataclass and validation
│   ├── raw_to_bids.py             # DICOM-to-BIDS conversion
│   ├── reporting.py               # HTML report and QC image generation
│   ├── smoothing.py               # Smoothing FWHM normalization
│   └── trial_signatures.py        # Trial-wise beta-series / LSS extraction
├── cli/
│   └── commands/
│       ├── fmri.py                # CLI for raw-to-BIDS and preprocessing
│       └── fmri_analysis.py       # CLI for first-level analysis and trial signatures
├── pipelines/
│   ├── fmri_analysis.py           # First-level analysis pipeline orchestrator
│   ├── fmri_preprocessing.py      # fMRIPrep pipeline orchestrator
│   └── fmri_trial_signatures.py   # Trial signature pipeline orchestrator
└── utils/
    └── config/
        └── fmri_config.yaml       # Reference fMRI-only config (runtime defaults come from eeg_config.yaml)
```

### Output Layout (per subject)

```
<deriv_root>/
├── preprocessed/fmri/
│   └── sub-XXXX/
│       ├── anat/                  # fMRIPrep anatomical outputs
│       └── func/                  # fMRIPrep functional outputs
│           ├── *_desc-preproc_bold.nii.gz
│           ├── *_desc-brain_mask.nii.gz
│           └── *_desc-confounds_timeseries.tsv
├── sub-XXXX/
│   └── fmri/
│       ├── first_level/
│       │   └── task-<task>/
│       │       └── contrast-<name>/
│       │           ├── *_stat-z_score_<hash>.nii.gz
│       │           ├── *_stat-z_score_<hash>.json    # Provenance sidecar
│       │           ├── *_space-MNI152NLin2009cAsym_*.nii.gz
│       │           ├── qc/                           # Design matrix TSVs and PNGs
│       │           ├── plots/
│       │           │   ├── native/                   # Subject-space visualizations
│       │           │   ├── mni/                      # MNI-space visualizations
│       │           │   └── qc/                       # Carpet, tSNR, motion plots
│       │           └── report.html                   # Self-contained HTML report
│       ├── beta_series/
│       │   └── task-<task>/contrast-<name>/
│       │       ├── trials.tsv
│       │       ├── trial_betas/                      # Per-trial NIfTI beta maps
│       │       ├── condition_betas/                  # Condition-averaged beta maps
│       │       ├── signatures/
│       │       │   ├── trial_signature_expression.tsv
│       │       │   ├── condition_signature_expression.tsv
│       │       │   └── group_signature_expression.tsv
│       │       └── provenance.json
│       └── lss/                                      # Same structure as beta_series/
│           └── task-<task>/contrast-<name>/
└── sub-XXXX/fmri_contrasts/                          # Legacy contrast output path
```

---

## Stage 1 — Raw-to-BIDS Conversion

**Module:** `analysis/raw_to_bids.py`

Converts per-series DICOM directories into a BIDS-compliant fMRI dataset.

### Method

1. **Series discovery** — Scans `<source_root>/sub-*/fmri/` for DICOM directories. Classifies each by name heuristics:
   - `mprage` / `t1w` / `t1` → anatomical T1w
   - `painrN` → task BOLD (run number extracted from suffix)
   - `rs_` / `rest` → resting-state BOLD
   - `field` + `map` → fieldmap

2. **DICOM-to-NIfTI** — Runs `dcm2niix` per series with flags `-b y` (BIDS sidecars), `-z y` (gzip), `-f %p_%s` (predictable naming). Output is written to a temporary directory, then moved to the BIDS tree.

3. **Fieldmap classification** — For fieldmap series, outputs are classified as `phasediff`, `magnitude1`, `magnitude2` based on:
   - Presence of `EchoTime1`/`EchoTime2` in the JSON sidecar
   - `ImageType` containing `PHASE`
   - Filename containing `phase`
   - Fallback: smallest file treated as phase, largest as magnitude

4. **Events file generation** — For each BOLD run, a BIDS `*_events.tsv` is generated from PsychoPy TrialSummary CSVs. Event types modeled:

   | `trial_type` | Description | Duration |
   |--------------|-------------|----------|
   | `fixation_rest` | Pre-stimulus rest interval (35°C baseline) | ITI duration |
   | `fixation_poststim` | Post-stimulus fixation before pain question | Variable (4.5–8.5 s) |
   | `stimulation` | Thermal stimulation epoch | 12.5 s total |
   | `pain_question` | Binary pain yes/no question | ≤ 4 s |
   | `vas_rating` | Visual analogue scale rating | ≤ 7 s |

   When `event_granularity=phases`, the stimulation epoch is split into three sub-phases:
   - **ramp_up**: 3 s
   - **plateau**: 7.5 s
   - **ramp_down**: 2 s

   Each event row carries per-trial metadata: `stimulus_temp`, `selected_surface`, `pain_binary_coded`, `vas_final_coded_rating`, `vas_scale_min`, `vas_scale_max`.

5. **Onset alignment** — Configurable via `onset_reference`:
   - `as_is` — Raw PsychoPy timestamps (default)
   - `first_iti_start` — Zero-referenced to the first ITI onset
   - `first_stim_start` — Zero-referenced to the first stimulation onset

   An optional `onset_offset_s` is added to all onsets after reference subtraction.

6. **Validation** — Events are validated against the BOLD run duration (from NIfTI header or JSON sidecar). If any event falls outside the run bounds (with a tolerance of `max(2×TR, 1s)`), conversion fails with an actionable error message.

7. **BIDS metadata** — `dataset_description.json`, `participants.tsv`, `task-<task>_events.json` (column descriptions), and per-run JSON sidecars with `TaskName` and `IntendedFor` (fieldmaps) are written automatically.

---

## Stage 2 — fMRIPrep Preprocessing

**Module:** `pipelines/fmri_preprocessing.py`

Runs [fMRIPrep](https://fmriprep.org/) in a Docker or Apptainer container for each subject.

### Method

1. **Container execution** — Constructs a `docker run` or `apptainer run` command with bind mounts for BIDS data, output, work directory, and FreeSurfer license.

2. **Default output spaces**: `MNI152NLin2009cAsym` and `T1w`.

3. **Key outputs consumed downstream**:
   - `*_space-T1w_desc-preproc_bold.nii.gz` — Preprocessed BOLD in subject space
   - `*_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz` — Preprocessed BOLD in MNI space
   - `*_desc-brain_mask.nii.gz` — Brain mask per run
   - `*_desc-confounds_timeseries.tsv` — Nuisance regressors

4. **Configurable parameters** (see [Configuration Reference](#configuration-reference)):
   - Thread/memory limits (`nthreads`, `omp_nthreads`, `mem_mb`)
   - Anatomical options (`fs_no_reconall`, `longitudinal`, `skull_strip_template`)
   - BOLD processing (`bold2t1w_init`, `bold2t1w_dof`, `slice_time_ref`, `dummy_scans`)
   - QC thresholds (`fd_spike_threshold`, `dvars_spike_threshold`)
   - Denoising (`use_aroma`)
   - Arbitrary extra arguments via `extra_args`

---

## Stage 3 — First-Level GLM Contrast Analysis

**Module:** `pipelines/fmri_analysis.py` → `analysis/contrast_builder.py`

Computes subject-level (first-level) statistical contrasts between experimental conditions using nilearn's `FirstLevelModel`.

### 3.1 BIDS Discovery

Automatic discovery of BOLD runs and associated files:

1. **Events files** — Glob `sub-*_task-*_run-*_events.tsv` in the subject's `func/` directory. Falls back to legacy `*_bold_events.tsv` naming.

2. **BOLD images** — Preference order:
   - fMRIPrep preprocessed: searches `derivatives/preprocessed/fmri/`, `derivatives/fmriprep/` for `*_space-{space}_desc-preproc_bold.nii.gz`
   - Raw BIDS: `*_bold.nii.gz` in the subject's `func/` directory

3. **Confounds** — Searches the same derivative directories for `*_desc-confounds_timeseries.tsv` or `*_desc-confounds_regressors.tsv`.

4. **Brain masks** — Discovers `*_desc-brain_mask.nii.gz` alongside each preprocessed BOLD file.

5. **TR extraction** — From the BOLD JSON sidecar (`RepetitionTime`) or NIfTI header (4th zoom dimension).

### 3.2 Confound Regression

**Module:** `analysis/confounds_selection.py`

Selects nuisance regressors from fMRIPrep confounds TSV. Strategies (in order of complexity):

| Strategy | Regressors |
|----------|------------|
| `none` | No confounds |
| `motion6` | 6 rigid-body motion parameters (3 translation + 3 rotation) |
| `motion12` | motion6 + temporal derivatives |
| `motion24` | motion12 + quadratic terms + derivative quadratics |
| `motion24+wmcsf` | motion24 + white matter + CSF mean signals |
| `motion24+wmcsf+fd` | motion24+wmcsf + framewise displacement |
| `auto` (default) | Adaptive: starts with motion24 if available, adds WM, CSF, FD, and up to 5 aCompCor components. Falls back gracefully for non-fMRIPrep data. |

Additional regressors automatically included:
- `motion_outlier*` columns (spike regressors)
- `non_steady_state_outlier*` columns
- `outlier*` columns

CompCor component selection preference: `a_comp_cor` (anatomical) > `t_comp_cor` (temporal) > `c_comp_cor` > `w_comp_cor`. Up to `auto_compcor_n` (default: 5) components from the highest-priority available prefix.

NaN values in confounds (e.g., first-frame FD) are filled with 0.

### 3.3 Event Filtering and Condition Remapping

Events undergo several filtering stages before GLM fitting:

1. **`events_to_model`** — Restricts which `trial_type` rows enter the GLM. For multi-phase pain tasks, a practical choice is `["stimulation", "pain_question", "vas_rating"]`.

2. **`stim_phases_to_model`** — When events include a `stim_phase` column, restricts stimulation events to specific sub-phases (e.g., `["plateau"]`) without dropping non-stimulation rows. Use `["all"]` to disable.

3. **Condition remapping** — When `condition_a.column` / `condition_a.value` are specified, events are remapped:
   - Rows matching condition A → `cond_a_<contrast_name>`
   - Rows matching condition B → `cond_b_<contrast_name>`
   - `condition_scope_trial_types` restricts which `trial_type` rows are eligible for remapping (prevents cross-phase contamination)
   - Value coercion handles type mismatches (e.g., CLI string `"1"` matched against integer column)

### 3.4 GLM Specification

The first-level model is configured as follows:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hrf_model` | `spm` | Hemodynamic response function. Options: `spm`, `flobs`, `fir` |
| `drift_model` | `cosine` | Slow drift removal. Options: `cosine`, `polynomial`, `none` |
| `high_pass` | 0.008 Hz | High-pass filter cutoff (128 s period) |
| `low_pass` | `null` | Optional low-pass filter (added only if nilearn supports it) |
| `noise_model` | `ar1` | Temporal autocorrelation model |
| `standardize` | `true` | Standardize BOLD signal |
| `signal_scaling` | `0` | Scale signal to percent signal change |
| `smoothing_fwhm` | `null` | Optional spatial smoothing at first level (mm FWHM) |
| `mask_img` | auto | fMRIPrep brain mask (intersection across runs for multi-run) |

### 3.5 Multi-Run GLM Fitting

Runs are combined using nilearn's native multi-run `FirstLevelModel` support (fixed-effects across runs). This is the statistically valid approach — averaging per-run z/t maps is explicitly avoided.

**Run exclusion logic:**
- Runs where condition A or B values are absent are excluded with a warning
- The GLM fails only if **no** runs contain both conditions
- Skipped runs are logged with reasons and recorded in metadata

**Brain masking:** When fMRIPrep brain masks exist for all included runs, their intersection (threshold=1.0) is used as the GLM mask.

### 3.6 Contrast Computation

Contrasts are computed from the fitted GLM:

| Contrast Type | Definition |
|---------------|------------|
| Two-condition | `cond_a_<name> - cond_b_<name>` |
| Single-condition | `cond_a_<name>` |
| Custom formula | User-provided string (e.g., `"stimulation - fixation_rest"`) |

Output types: `z_score` (default), `stat` (t-statistic), `effect_size` (COPE/beta).

For multi-run models, the contrast definition is replicated per run to avoid nilearn warnings.

### 3.7 Cluster-Extent Thresholding

When `cluster_correction=true` and the output is a z-score map, a cluster-extent thresholded binary mask is generated:

1. Voxelwise z-threshold derived from `cluster_p_threshold` (default: 0.001) via the inverse normal CDF
2. For `tail=abs`, the threshold is two-sided (p/2)
3. Connected components identified using 6-connectivity (face-adjacent voxels)
4. Clusters smaller than `cluster_min_voxels` (or `cluster_min_volume_mm3` when voxel dimensions are available) are removed

This mask is saved alongside the contrast map for downstream EEG source localization constraints.

### 3.8 Resampling to FreeSurfer Space

When `resample_to_freesurfer=true`, the contrast map is resampled to the FreeSurfer subject's anatomical grid using `nilearn.image.resample_to_img` with continuous interpolation. Target image priority: `T1.mgz` > `orig.mgz` > `brain.mgz`.

### 3.9 Caching and Reproducibility

- Contrast maps are named with an MD5 hash of key configuration parameters, enabling cache-friendly re-runs
- A JSON sidecar is written alongside each NIfTI with full provenance: subject, task, contrast definition, run inputs, confound columns, skipped runs, and event counts
- Design matrices can be written to disk (`write_design_matrix=true`) as TSV and PNG for QC

---

## Stage 4 — Trial-Wise Signature Extraction

**Module:** `pipelines/fmri_trial_signatures.py` → `analysis/trial_signatures.py`

Computes per-trial beta maps and quantifies multivariate pain signature expression for each trial.

### 4.1 Methods

Two estimation methods are supported:

#### Beta-Series (LSA)

One GLM per run with one regressor per selected trial. All selected trials receive unique regressors (`trial_<run>_<idx>_<condition>`). Non-selected events are modeled as nuisance regressors grouped by their original `trial_type`.

- Single model fit per run (computationally efficient)
- Beta for each trial extracted via a unit contrast vector targeting the trial's column in the design matrix

#### Least Squares Separate (LSS)

One GLM per trial within each run. For each target trial:
- The target trial gets a `"target"` regressor
- Other selected trials are grouped as:
  - `per_condition` mode: `"other_cond_a"` and `"other_cond_b"` (default)
  - `all` mode: single `"other_trials"` regressor
- Non-selected events modeled as nuisance regressors

LSS is more computationally expensive (N models per run) but produces less biased estimates when trials are temporally close.

### 4.2 Trial Selection

Trials are selected via column/value pairs:
- `condition_a_column` / `condition_a_value` — Selects condition A trials
- `condition_b_column` / `condition_b_value` — Selects condition B trials

Scoping controls:
- `condition_scope_trial_types` — Restricts selection to specific `trial_type` values (e.g., `["stimulation"]`)
- `condition_scope_stim_phases` — Restricts selection to specific `stim_phase` values (e.g., `["plateau"]`)
- `max_trials_per_run` — Caps the number of trials per run

#### Group-Based Selection

When `signature_group_column` and `signature_group_values` are specified, trials are selected by group membership (e.g., temperature levels) instead of A/B conditions. Group averaging can be scoped:
- `across_runs` — Pool trials across all runs before averaging
- `per_run` — Average within each run separately

### 4.3 Condition-Level Averaging

Per-trial betas are combined into condition-level maps using fixed-effects weighting:

| Method | Formula |
|--------|---------|
| `variance` (default) | Inverse-variance weighted mean: $\hat{\beta} = \frac{\sum w_i \beta_i}{\sum w_i}$ where $w_i = 1/\sigma^2_i$ |
| `mean` | Simple arithmetic mean |

Condition maps produced:
- `cond-a_beta.nii.gz` — Condition A average
- `cond-b_beta.nii.gz` — Condition B average
- `cond-a_minus_b_beta.nii.gz` — Difference map

### 4.4 Signature Readout

For each trial beta map and each condition-averaged map, multivariate pain signature expression is computed (see [Multivariate Pain Signatures](#multivariate-pain-signatures)).

**Scientific constraint:** Trial-wise signature extraction requires MNI-space images. The pipeline raises an error if `fmriprep_space` does not contain `"mni"`, since the bundled NPS/SIIPS1 weights are defined in MNI space.

### 4.5 Outputs

| File | Content |
|------|---------|
| `trials.tsv` | Per-trial metadata: run, trial index, condition, regressor name, onset, duration, original trial type, source events file/row, and all extra event columns |
| `signatures/trial_signature_expression.tsv` | Per-trial × per-signature: dot product, cosine similarity, Pearson r, voxel count |
| `signatures/condition_signature_expression.tsv` | Per-condition-average × per-signature |
| `signatures/group_signature_expression.tsv` | Per-group × per-signature (when group mode is active) |
| `condition_betas/*.nii.gz` | Condition-averaged NIfTI beta maps |
| `trial_betas/<run>/*.nii.gz` | Per-trial NIfTI beta maps (optional) |
| `provenance.json` | Full configuration, signature root, run count |

---

## Stage 5 — Reporting and Quality Control

**Module:** `analysis/reporting.py`

Generates a self-contained HTML report with statistical visualizations and QC diagnostics.

### 5.1 Statistical Visualizations

Generated per space (native and/or MNI):

| Plot Type | Description |
|-----------|-------------|
| `slices` | Mosaic stat map overlay (thresholded and unthresholded) via `nilearn.plotting.plot_stat_map` |
| `glass` | Glass brain projection (thresholded and unthresholded) via `nilearn.plotting.plot_glass_brain` |
| `hist` | Z-statistic voxel distribution histogram with threshold lines |
| `clusters` | Cluster/peak table via `nilearn.reporting.get_clusters_table` |

Additional panels:
- **Effect size** — Unthresholded COPE/beta map (slices + glass brain, `cold_hot` colormap)
- **Standard error** — Derived from variance map ($\text{SE} = \sqrt{\text{Var}}$), `viridis` colormap

### 5.2 Thresholding Modes

| Mode | Method |
|------|--------|
| `z` | Hard z-threshold (default: \|z\| > 2.3) |
| `fdr` | FDR-corrected threshold via `nilearn.glm.threshold_stats_img` (default q = 0.05) |
| `none` | No thresholding |

Cluster-extent filtering (`cluster_min_voxels`) is applied after voxelwise thresholding using 26-connectivity connected components.

### 5.3 Color Scaling

| Mode | Behavior |
|------|----------|
| `per_space_robust` (default) | 99th percentile of \|z\| within each space independently |
| `shared_robust` | Maximum of the per-space robust values (consistent scaling across native/MNI) |
| `manual` | User-specified `vmax_manual` value |

### 5.4 QC Diagnostics

| Diagnostic | Description |
|------------|-------------|
| **Motion QC** | Framewise displacement (FD) and DVARS time series concatenated across runs, with run boundaries marked |
| **Carpet plot** | Standardized voxel × time matrix (z-scored per voxel, clipped to ±3), subsampled to 6000 voxels for readability |
| **tSNR map** | Mean temporal signal-to-noise ratio across runs (sagittal/coronal/axial montage + histogram). Also saved as `tsnr_mean.nii.gz` |
| **Design matrix** | Per-run design matrix images (PNG) and TSVs from the GLM |
| **Design QC summary** | Per-run table with: number of regressors, max absolute correlation, top correlated pair, condition number (SVD), max variance inflation factor (VIF) |

### 5.5 Pain Signature Expression Table

When signature weight maps are available, the report includes a table of NPS/SIIPS1 expression computed on the unthresholded MNI effect-size map:
- Dot product (raw pattern expression)
- Cosine similarity (scale-invariant)
- Pearson correlation (scale-invariant)
- Voxel count

### 5.6 HTML Report

The report is a self-contained HTML file with:
- Embedded base64 images (configurable via `embed_images`)
- Responsive CSS layout
- Collapsible methods/provenance section (full JSON)
- Downloadable TSV links for cluster tables and design matrices
- Library version tracking (nilearn, nibabel, numpy)

---

## BEM and Coregistration Generation

**Module:** `analysis/bem_generation.py`

Generates boundary element model (BEM) surfaces, solutions, and EEG↔MRI coregistration transforms using Docker with FreeSurfer and MNE-Python.

### BEM Model

1. **Watershed BEM** — Runs `mne watershed_bem` inside Docker to create inner skull, outer skull, and outer skin surfaces from the FreeSurfer `recon-all` output.

2. **BEM surfaces** — `mne.make_bem_model()` with configurable:
   - `ico` — ICO downsampling level (default: 4 → 5120 triangles per surface)
   - `conductivity` — Three-layer conductivity values [scalp, skull, brain] (default: [0.3, 0.006, 0.3] S/m)

3. **BEM solution** — `mne.make_bem_solution()` computes the forward model matrix from the BEM surfaces.

### Coregistration Transform

- Uses MNE's fiducial-based coregistration
- Falls back to fsaverage fiducials scaled to the subject when subject-specific fiducials are unavailable
- **Safety guard:** Auto-generation of an identity transform is refused by default (`allow_identity_trans=false`) because it is scientifically invalid without proper digitization/coregistration. Set to `true` only for debugging.

### Docker Image

Default image: `freesurfer-mne:7.4.1` (custom image with FreeSurfer 7.4.1 + MNE-Python). The Dockerfile is provided at `eeg_pipeline/docker_setup/Dockerfile.freesurfer-mne`.

---

## Multivariate Pain Signatures

**Module:** `analysis/pain_signatures.py`

Computes expression of established multivariate pain biomarkers on statistical or beta maps.

### Supported Signatures

| Signature | Weight Map | Reference |
|-----------|-----------|-----------|
| **NPS** (Neurologic Pain Signature) | `NPS/weights_NSF_grouppred_cvpcr.nii.gz` | Wager et al., 2013, NEJM |
| **SIIPS1** (Stimulus-Independent Pain Signature) | `SIIPS1/nonnoc_v11_4_137subjmap_weighted_mean.nii.gz` | Woo et al., 2017, Nature Communications |

### Computation

For each signature weight map $\mathbf{w}$ and input image $\mathbf{x}$ (after resampling to a common grid):

1. **Resampling** — Two strategies:
   - `image_to_weights` (default): resample the input image to the signature's voxel grid (preserves signature resolution)
   - `weights_to_image`: resample signature weights to the input image grid

2. **Masking** — Intersection of finite voxels in both images. An optional analysis mask (e.g., brain mask) is resampled and applied.

3. **Metrics computed:**

   | Metric | Formula | Interpretation |
   |--------|---------|----------------|
   | Dot product | $\mathbf{x} \cdot \mathbf{w} = \sum_i x_i w_i$ | Raw pattern expression (scale-dependent) |
   | Cosine similarity | $\frac{\mathbf{x} \cdot \mathbf{w}}{\|\mathbf{x}\| \|\mathbf{w}\|}$ | Scale-invariant similarity |
   | Pearson r | $r(\mathbf{x}, \mathbf{w})$ | Scale- and mean-invariant correlation |

4. **Voxel count** — Number of finite, non-masked voxels used in the computation (for QC).

---

## Configuration Reference

Runtime settings are loaded from `eeg_pipeline/utils/config/eeg_config.yaml` (including `fmri_preprocessing`, `fmri_contrast`, and `fmri_constraint`).  
`fmri_pipeline/utils/config/fmri_config.yaml` is kept as a reference/template for fMRI-only workflows.

### Paths

| Key | Description |
|-----|-------------|
| `paths.bids_fmri_root` | BIDS root directory containing raw fMRI data |
| `paths.freesurfer_dir` | FreeSurfer `SUBJECTS_DIR` |
| `paths.freesurfer_license` | Path to FreeSurfer `license.txt` (or use `EEG_PIPELINE_FREESURFER_LICENSE`) |
| `paths.signature_dir` | Root directory containing NPS/SIIPS1 weight maps |

### fMRI Preprocessing (`fmri_preprocessing.fmriprep`)

| Key | Default | Description |
|-----|---------|-------------|
| `engine` | `docker` | Container engine (`docker` or `apptainer`) |
| `image` | `nipreps/fmriprep:25.2.4` | fMRIPrep container image |
| `output_spaces` | `[MNI152NLin2009cAsym, T1w]` | Output coordinate spaces |
| `fs_no_reconall` | `false` | Skip FreeSurfer surface reconstruction |
| `use_aroma` | `false` | Enable ICA-AROMA denoising |
| `nthreads` | `0` (auto) | Maximum threads |
| `mem_mb` | `0` (auto) | Memory limit |
| `fd_spike_threshold` | `0.5` | FD spike threshold (mm) |
| `dvars_spike_threshold` | `1.5` | Standardized DVARS threshold |
| `random_seed` | `0` | Reproducibility seed (0 = non-deterministic) |
| `extra_args` | `null` | Additional fMRIPrep CLI arguments |

### First-Level Contrast (`fmri_contrast`)

| Key | Default | Description |
|-----|---------|-------------|
| `enabled` | `false` | Enable contrast computation |
| `input_source` | `fmriprep` | BOLD source (`fmriprep` or `bids_raw`) |
| `fmriprep_space` | `T1w` | fMRIPrep output space for BOLD |
| `require_fmriprep` | `true` | Fail if preprocessed BOLD is missing |
| `type` | `t-test` | Contrast type |
| `condition_a.column` | `pain_binary_coded` | Events column for condition A |
| `condition_a.value` | `1` | Value to select condition A trials |
| `condition_b.column` | `pain_binary_coded` | Events column for condition B |
| `condition_b.value` | `0` | Value to select condition B trials |
| `condition_scope_trial_types` | `null` | Restrict condition selection to specific trial types |
| `events_to_model` | `null` | Restrict which event types enter the GLM |
| `stim_phases_to_model` | `null` | Restrict stimulation sub-phases |
| `formula` | `null` | Custom contrast formula (overrides conditions) |
| `name` | `pain_vs_nonpain` | Contrast output name |
| `runs` | `null` (auto) | Specific run numbers to include |
| `hrf_model` | `spm` | HRF model |
| `drift_model` | `cosine` | Drift model |
| `high_pass_hz` | `0.008` | High-pass cutoff (Hz) |
| `low_pass_hz` | `null` | Low-pass cutoff (Hz) |
| `confounds_strategy` | `auto` | Confound selection strategy |
| `smoothing_fwhm` | `null` | Spatial smoothing FWHM (mm) |
| `cluster_correction` | `true` | Generate cluster-extent mask |
| `cluster_p_threshold` | `0.001` | Cluster-forming p-threshold |
| `output_type` | `z-score` | Output map type |
| `resample_to_freesurfer` | `true` | Resample to FreeSurfer space |
| `write_design_matrix` | `false` | Write design matrices for QC |

### fMRI Constraint (for EEG source localization)

| Key | Default | Description |
|-----|---------|-------------|
| `fmri_constraint.enabled` | `false` | Enable fMRI-constrained source localization |
| `fmri_constraint.stats_map_path` | `null` | Pre-computed stats map (overrides contrast builder) |
| `fmri_constraint.threshold` | `2.0` | Z-threshold for constraint mask |
| `fmri_constraint.tail` | `pos` | Tail for thresholding (`pos` or `abs`) |
| `fmri_constraint.thresholding.mode` | `z` | Thresholding mode (`z` or `fdr`) |
| `fmri_constraint.cluster_min_voxels` | `10` | Minimum cluster size |
| `fmri_constraint.cluster_min_volume_mm3` | `null` | Minimum cluster volume (overrides voxel count) |
| `fmri_constraint.max_clusters` | `20` | Keep top-N clusters |

### BEM Generation

| Key | Default | Description |
|-----|---------|-------------|
| `bem_generation.create_trans` | `false` | Auto-create coregistration transform |
| `bem_generation.allow_identity_trans` | `false` | Allow identity transform (debug only) |
| `bem_generation.create_model` | `true` | Auto-create BEM model |
| `bem_generation.create_solution` | `true` | Auto-create BEM solution |
| `bem_generation.docker_image` | `freesurfer-mne:7.4.1` | Docker image |
| `bem_generation.ico` | `4` | ICO downsampling level |
| `bem_generation.conductivity` | `[0.3, 0.006, 0.3]` | Conductivity [scalp, skull, brain] (S/m) |

---

## Dependencies

### Python Packages

| Package | Purpose |
|---------|---------|
| **nilearn** | GLM fitting (`FirstLevelModel`), contrast computation, plotting, cluster tables |
| **nibabel** | NIfTI/MGZ I/O, image resampling |
| **numpy** | Array operations, statistics |
| **scipy** | Connected components (cluster filtering), normal distribution (z-thresholds) |
| **pandas** | Events and confounds TSV I/O |
| **matplotlib** | QC plots (carpet, tSNR, motion, histograms) |

### External Tools

| Tool | Purpose |
|------|---------|
| **dcm2niix** | DICOM-to-NIfTI conversion (Stage 1) |
| **Docker** or **Apptainer** | Container runtime for fMRIPrep and BEM generation |
| **fMRIPrep** (container) | BOLD preprocessing (Stage 2) |
| **FreeSurfer** (via Docker) | BEM watershed surfaces, coregistration |
| **MNE-Python** (via Docker) | BEM model/solution computation |
