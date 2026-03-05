# fMRI Analysis Pipeline

**Module:** `fmri_pipeline`

This document is the methods reference for the subject-level fMRI analysis pipeline.
The pipeline starts from BIDS-formatted inputs and covers fMRIPrep preprocessing,
first-level GLM contrast analysis, trial-wise beta estimation, and multivariate
signature readout (NPS, SIIPS1).

The resulting fMRI statistical maps are used to constrain EEG inverse solutions in the
source localization stage of the EEG pipeline.

---

## Table of Contents

1. [Notation](#1-notation)
2. [Module Structure](#2-module-structure)
3. [Pipeline Overview](#3-pipeline-overview)
4. [Stage 1 — BIDS Inputs and Event Assumptions](#4-stage-1--bids-inputs-and-event-assumptions)
5. [Stage 2 — fMRIPrep Preprocessing](#5-stage-2--fmriprep-preprocessing)
6. [Stage 3 — First-Level GLM Contrast Analysis](#6-stage-3--first-level-glm-contrast-analysis)
7. [Stage 4 — Trial-Wise Beta Estimation](#7-stage-4--trial-wise-beta-estimation)
8. [Stage 5 — Reporting and Quality Control](#8-stage-5--reporting-and-quality-control)
9. [BEM and Coregistration](#9-bem-and-coregistration)
10. [Multivariate Pain Signatures](#10-multivariate-pain-signatures)
11. [Output Layout](#11-output-layout)
12. [Configuration Reference](#12-configuration-reference)
13. [Dependencies](#13-dependencies)

---

## 1. Notation

| Symbol | Meaning |
|--------|---------|
| $\mathbf{x} \in \mathbb{R}^V$ | Vectorized voxel image (after masking) |
| $\mathbf{w} \in \mathbb{R}^V$ | Signature weight map (NPS or SIIPS1) |
| $V$ | Number of finite, unmasked voxels |
| $\beta_i$ | Beta estimate for trial $i$ |
| $\sigma^2_i$ | Variance of beta estimate for trial $i$ |
| $w_i = 1/\sigma^2_i$ | Inverse-variance weight for fixed-effects averaging |
| TR | Repetition time (s) |
| FD | Framewise displacement (mm) |
| HRF | Hemodynamic response function |

---

## 2. Module Structure

### Source Layout

| File | Responsibility |
|------|---------------|
| `pipelines/fmri_preprocessing.py` | fMRIPrep pipeline orchestrator |
| `pipelines/fmri_analysis.py` | First-level GLM pipeline orchestrator |
| `pipelines/fmri_trial_signatures.py` | Trial-wise beta and signature pipeline orchestrator |
| `analysis/multivariate_signatures.py` | NPS/SIIPS1 multivariate signature computation |
| `analysis/contrast_builder.py` | First-level GLM fitting and contrast computation |
| `analysis/trial_signatures.py` | Trial-wise beta-series (LSA) and LSS extraction |
| `analysis/confounds_selection.py` | fMRIPrep confound regressor selection |
| `analysis/events_selection.py` | Trial-type normalization and filtering |
| `analysis/bem_generation.py` | Docker-based BEM model, solution, and coregistration transform |
| `analysis/reporting.py` | HTML report generation with QC visualizations |
| `analysis/smoothing.py` | Spatial smoothing FWHM normalization |
| `analysis/plotting_config.py` | Plot configuration dataclass and validation |
| `cli/commands/fmri.py` | CLI entry point for fMRIPrep preprocessing |
| `cli/commands/fmri_analysis.py` | CLI entry point for first-level analysis and trial signatures |
| `utils/bold_discovery.py` | BOLD run and confound file discovery |
| `utils/signature_paths.py` | Signature weight map path resolution |
| `utils/config/fmri_config.yaml` | Reference fMRI-only config template |

---

## 3. Pipeline Overview

```
BIDS ──► fMRIPrep ──► First-Level GLM ──► Contrast Maps ──► EEG Source Localization
                           │
                           └──► Trial-Wise Betas ──► Signature Expression (NPS/SIIPS1)
```

| Stage | Module | Purpose |
|-------|--------|---------|
| 1 | Input BIDS dataset + `events.tsv` | Required input contract for downstream modeling |
| 2 | `pipelines/fmri_preprocessing.py` | fMRIPrep containerized preprocessing |
| 3 | `pipelines/fmri_analysis.py` + `analysis/contrast_builder.py` | Multi-run first-level GLM and contrast computation |
| 4 | `pipelines/fmri_trial_signatures.py` + `analysis/trial_signatures.py` | Trial-wise beta estimation and signature readout |
| 5 | `analysis/reporting.py` | HTML report generation with QC diagnostics |

Runtime configuration is loaded from `eeg_pipeline/utils/config/eeg_config.yaml`
(sections `fmri_preprocessing`, `fmri_contrast`, `fmri_constraint`).

---

## 4. Stage 1 — BIDS Inputs and Event Assumptions

The fMRI pipeline consumes a BIDS dataset and does not perform DICOM conversion.
Provide at minimum:

- `sub-*/func/*_bold.nii.gz`
- matching `sub-*/func/*_events.tsv`
- recommended sidecars (`*.json`, fieldmaps, and metadata) for robust preprocessing

Events can be study-specific. The GLM layer expects standard BIDS `events.tsv` columns
(`onset`, `duration`, `trial_type`) plus any additional columns used by
`condition_a.column`, `condition_b.column`, and optional scoping filters.

Validation includes:

1. Presence of BOLD/events pairs for selected subjects/runs.
2. Condition values resolvable against available event columns.
3. Event timing consistency with run duration tolerances during model construction.

---

## 5. Stage 2 — fMRIPrep Preprocessing

**Module:** `pipelines/fmri_preprocessing.py`

Runs [fMRIPrep](https://fmriprep.org/) in a Docker or Apptainer container per subject.

### 5.1 Container Execution

Constructs a `docker run` or `apptainer run` command with bind mounts for BIDS data,
output directory, work directory, and FreeSurfer license.

**Default output spaces:** `MNI152NLin2009cAsym` and `T1w`.

**Key outputs consumed downstream:**

| File pattern | Description |
|--------------|-------------|
| `*_space-T1w_desc-preproc_bold.nii.gz` | Preprocessed BOLD in subject space |
| `*_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz` | Preprocessed BOLD in MNI space |
| `*_desc-brain_mask.nii.gz` | Brain mask per run |
| `*_desc-confounds_timeseries.tsv` | Nuisance regressors |

### 5.2 Configurable Parameters

Thread/memory limits, anatomical options (`fs_no_reconall`, `longitudinal`,
`skull_strip_template`), BOLD processing (`bold2t1w_init`, `bold2t1w_dof`,
`slice_time_ref`, `dummy_scans`), QC thresholds (`fd_spike_threshold`,
`dvars_spike_threshold`), denoising (`use_aroma`), and arbitrary extra arguments
via `extra_args`. See §12 for all keys.

---

## 6. Stage 3 — First-Level GLM Contrast Analysis

**Module:** `pipelines/fmri_analysis.py` → `analysis/contrast_builder.py`

Subject-level statistical contrasts between experimental conditions via nilearn's
`FirstLevelModel`.

### 6.1 BIDS Discovery

Automatic discovery of BOLD runs and associated files per subject:

1. **Events** — Glob `sub-*_task-*_run-*_events.tsv` in `func/`; secondary pattern
   `*_bold_events.tsv` for legacy datasets.
2. **BOLD images** — Preference order:
   - fMRIPrep preprocessed: `derivatives/preprocessed/fmri/` then `derivatives/fmriprep/`
     for `*_space-{space}_desc-preproc_bold.nii.gz`.
   - Raw BIDS: `*_bold.nii.gz` in `func/`. Used only when `input_source = bids_raw`.
3. **Confounds** — `*_desc-confounds_timeseries.tsv` or `*_desc-confounds_regressors.tsv`
   from the same derivative directory as the BOLD file.
4. **Brain masks** — `*_desc-brain_mask.nii.gz` alongside each preprocessed BOLD file.
5. **TR** — From the BOLD JSON sidecar (`RepetitionTime`) or NIfTI header (4th zoom dimension).

### 6.2 Confound Regression

**Module:** `analysis/confounds_selection.py`

| Strategy | Regressors |
|----------|------------|
| `none` | No confounds |
| `motion6` | 6 rigid-body parameters (3 translation + 3 rotation) |
| `motion12` | motion6 + temporal derivatives |
| `motion24` | motion12 + quadratic terms + derivative quadratics |
| `motion24+wmcsf` | motion24 + white matter and CSF mean signals |
| `motion24+wmcsf+fd` | motion24+wmcsf + framewise displacement |
| `auto` *(default)* | Adaptive: motion24 if available, plus WM, CSF, FD, and up to `auto_compcor_n` (default 5) aCompCor components. When fMRIPrep confounds are absent, only the available columns are used. |

Additional regressors always included when present: `motion_outlier*`,
`non_steady_state_outlier*`, `outlier*` (spike regressors).

CompCor component selection priority: `a_comp_cor` > `t_comp_cor` > `c_comp_cor` > `w_comp_cor`.
NaN values in confounds (e.g., first-frame FD) are replaced with 0.

### 6.3 Event Filtering and Condition Remapping

**Module:** `analysis/events_selection.py`

Events undergo three filtering stages before GLM fitting:

1. **`events_to_model`** — Restricts which `trial_type` rows enter the GLM
   (e.g., `["stimulation", "pain_question", "vas_rating"]`).

2. **`stim_phases_to_model`** — When a `stim_phase` column is present, restricts
   stimulation events to specified sub-phases (e.g., `["plateau"]`).
   Non-stimulation rows are unaffected. Use `["all"]` to disable.

3. **Condition remapping** — When `condition_a.column` / `condition_a.value` are
   specified:
   - Rows matching condition A → `cond_a_<contrast_name>`
   - Rows matching condition B → `cond_b_<contrast_name>`
   - `condition_scope_trial_types` restricts which `trial_type` rows are eligible,
     preventing cross-phase contamination.
   - Value coercion handles type mismatches (e.g., CLI string `"1"` matched against
     an integer column).

### 6.4 GLM Specification

| Parameter | Default | Description |
|-----------|---------|-------------|
| `hrf_model` | `spm` | HRF. Options: `spm`, `flobs`, `fir` |
| `drift_model` | `cosine` | Slow-drift removal. Options: `cosine`, `polynomial`, `none` |
| `high_pass` | 0.008 Hz | High-pass filter cutoff (128 s period) |
| `low_pass` | `null` | Optional low-pass filter |
| `noise_model` | `ar1` | Temporal autocorrelation model |
| `standardize` | `true` | Standardize BOLD signal |
| `signal_scaling` | `0` | Scale signal to percent signal change |
| `smoothing_fwhm` | `null` | Optional spatial smoothing (mm FWHM) |
| `mask_img` | auto | fMRIPrep brain mask; intersection across runs for multi-run models |

### 6.5 Multi-Run GLM Fitting

Runs are combined using nilearn's native multi-run `FirstLevelModel`
(fixed-effects across runs). Averaging per-run z/t maps is explicitly avoided.

**Run exclusion:**
- Runs where condition A or B are absent are excluded with a logged warning.
- The GLM raises an error only if no runs contain both conditions.
- Skipped runs and their reasons are recorded in the provenance sidecar.

**Brain masking:** When fMRIPrep brain masks exist for all included runs, their
intersection (threshold = 1.0) is used as the GLM mask.

### 6.6 Contrast Computation

| Contrast type | Definition |
|---------------|------------|
| Two-condition | `cond_a_<name> - cond_b_<name>` |
| Single-condition | `cond_a_<name>` |
| Custom formula | User-provided string (e.g., `"stimulation - fixation_rest"`) |

Output types: `z_score` (default), `stat` (t-statistic), `effect_size` (COPE/beta).
For multi-run models, the contrast definition is replicated per run.

### 6.7 Cluster-Extent Thresholding

When `cluster_correction = true` on a z-score output:

1. Voxelwise z-threshold derived from `cluster_p_threshold` (default 0.001) via the
   inverse normal CDF. For `tail = abs`, the threshold is two-sided ($p/2$).
2. Connected components identified using 6-connectivity (face-adjacent voxels).
3. Components smaller than `cluster_min_voxels` (or `cluster_min_volume_mm3` when
   voxel dimensions are available) are removed.

The resulting binary mask is saved alongside the contrast map and consumed by the
EEG source localization stage.

### 6.8 Resampling to FreeSurfer Space

When `resample_to_freesurfer = true`, the contrast map is resampled to the
FreeSurfer subject anatomy using `nilearn.image.resample_to_img` with continuous
interpolation. Target image priority: `T1.mgz` > `orig.mgz` > `brain.mgz`.

### 6.9 Caching and Reproducibility

- Contrast maps are named with an MD5 hash of key configuration parameters,
  enabling cache-friendly re-runs without overwriting prior results.
- A JSON sidecar is written alongside each NIfTI with full provenance: subject,
  task, contrast definition, run inputs, confound columns, skipped runs, and
  event counts.
- Design matrices are written to disk as TSV and PNG when
  `write_design_matrix = true`.

---

## 7. Stage 4 — Trial-Wise Beta Estimation

**Module:** `pipelines/fmri_trial_signatures.py` → `analysis/trial_signatures.py`

Per-trial beta maps and multivariate pain signature expression.

### 7.1 Estimation Methods

#### Beta-Series (LSA)

One GLM per run; one unique regressor per selected trial
(`trial_<run>_<idx>_<condition>`). Non-selected events are modeled as nuisance
regressors grouped by `trial_type`. The beta for each trial is extracted via a
unit contrast vector targeting that trial's column.

- Single model fit per run (computationally efficient).
- Susceptible to correlated regressors when trials are temporally close.

#### Least Squares Separate (LSS)

One GLM per trial within each run. For each target trial:

- Target trial → `"target"` regressor.
- Other selected trials →
  - `per_condition` mode *(default)*: `"other_cond_a"` / `"other_cond_b"`.
  - `all` mode: single `"other_trials"` regressor.
- Non-selected events → nuisance regressors by `trial_type`.

LSS requires $N$ model fits per run but produces less biased beta estimates when
trials are temporally close.

### 7.2 Trial Selection

Trials are selected via column/value pairs:

| Parameter | Description |
|-----------|-------------|
| `condition_a_column` / `condition_a_value` | Selects condition A trials |
| `condition_b_column` / `condition_b_value` | Selects condition B trials |
| `condition_scope_trial_types` | Restricts selection to specified `trial_type` values |
| `condition_scope_stim_phases` | Restricts selection to specified `stim_phase` values |
| `max_trials_per_run` | Maximum trials per run |

**Group-based selection:** When `signature_group_column` and
`signature_group_values` are set, trials are selected by group membership
(e.g., temperature levels). Group averaging scope:

- `across_runs` — Pool trials across all runs before averaging.
- `per_run` — Average within each run separately.

### 7.3 Condition-Level Averaging

Per-trial betas are combined into condition-level maps:

| Method | Formula |
|--------|---------|
| `variance` *(default)* | Inverse-variance weighted mean: $\hat\beta = \sum_i w_i \beta_i \,/\, \sum_i w_i$, where $w_i = 1/\sigma^2_i$ |
| `mean` | Simple arithmetic mean |

Condition maps produced:

| File | Content |
|------|---------|
| `cond-a_beta.nii.gz` | Condition A average |
| `cond-b_beta.nii.gz` | Condition B average |
| `cond-a_minus_b_beta.nii.gz` | Difference map |

### 7.4 Signature Readout

For each trial beta map and condition-averaged map, multivariate pain signature
expression is computed (see §10).

**Spatial constraint:** Trial-wise signature extraction requires MNI-space images.
The pipeline raises a `ValueError` if `fmriprep_space` does not include `"mni"`,
since NPS and SIIPS1 weight maps are defined in MNI space.

### 7.5 Outputs

| File | Content |
|------|---------|
| `trials.tsv` | Per-trial metadata: run, trial index, condition, regressor name, onset, duration, original `trial_type`, source events file/row, all extra event columns |
| `signatures/trial_signature_expression.tsv` | Per-trial × per-signature: dot product, cosine similarity, Pearson $r$, voxel count |
| `signatures/condition_signature_expression.tsv` | Per-condition-average × per-signature |
| `signatures/group_signature_expression.tsv` | Per-group × per-signature (group mode only) |
| `condition_betas/*.nii.gz` | Condition-averaged NIfTI beta maps |
| `trial_betas/<run>/*.nii.gz` | Per-trial NIfTI beta maps (optional) |
| `provenance.json` | Full configuration, signature root, run count |

---

## 8. Stage 5 — Reporting and Quality Control

**Module:** `analysis/reporting.py`

Self-contained HTML report with statistical visualizations and QC diagnostics.

### 8.1 Statistical Visualizations

Generated per space (native and/or MNI):

| Plot | Description |
|------|-------------|
| `slices` | Mosaic stat-map overlay (thresholded and unthresholded) via `nilearn.plotting.plot_stat_map` |
| `glass` | Glass-brain projection (thresholded and unthresholded) via `nilearn.plotting.plot_glass_brain` |
| `hist` | Z-statistic voxel-distribution histogram with threshold lines |
| `clusters` | Cluster/peak table via `nilearn.reporting.get_clusters_table` |

Additional panels:

- **Effect size** — Unthresholded COPE/beta map (slices + glass brain, `cold_hot` colormap).
- **Standard error** — Derived as $\text{SE} = \sqrt{\text{Var}}$ (`viridis` colormap).

### 8.2 Thresholding Modes

| Mode | Method |
|------|--------|
| `z` *(default)* | Hard z-threshold (\|z\| > 2.3) |
| `fdr` | FDR-corrected threshold via `nilearn.glm.threshold_stats_img` (default $q = 0.05$) |
| `none` | No thresholding |

Cluster-extent filtering (`cluster_min_voxels`) is applied after voxelwise thresholding
using 26-connectivity connected components.

### 8.3 Color Scaling

| Mode | Behavior |
|------|---------|
| `per_space_robust` *(default)* | 99th percentile of \|z\| within each space independently |
| `shared_robust` | Maximum of the per-space robust values (consistent scaling across native/MNI) |
| `manual` | User-specified `vmax_manual` |

### 8.4 QC Diagnostics

| Diagnostic | Description |
|------------|-------------|
| Motion QC | FD and DVARS time series concatenated across runs, with run boundaries marked |
| Carpet plot | Standardized voxel × time matrix (z-scored per voxel, clipped to ±3), subsampled to 6 000 voxels |
| tSNR map | Mean temporal SNR across runs (sagittal/coronal/axial montage + histogram); also saved as `tsnr_mean.nii.gz` |
| Design matrix | Per-run design matrix images (PNG) and TSVs |
| Design QC summary | Per-run table: regressor count, max absolute correlation, top correlated pair, condition number (SVD), max VIF |

### 8.5 Pain Signature Expression

When signature weight maps are available, the report includes per-signature expression
computed on the unthresholded MNI effect-size map: dot product, cosine similarity,
Pearson $r$, and voxel count.

### 8.6 HTML Report Format

- Embedded base64 images (configurable via `embed_images`).
- Responsive CSS layout.
- Collapsible provenance section (full JSON).
- Downloadable TSV links for cluster tables and design matrices.
- Library version tracking (nilearn, nibabel, numpy).

---

## 9. BEM and Coregistration

**Module:** `analysis/bem_generation.py`

BEM surfaces, solutions, and EEG↔MRI coregistration transforms via Docker with
FreeSurfer and MNE-Python.

### 9.1 BEM Model

1. **Watershed BEM** — Runs `mne watershed_bem` inside Docker to create inner skull,
   outer skull, and outer skin surfaces from the FreeSurfer `recon-all` output.

2. **BEM surfaces** — `mne.make_bem_model()` with:
   - `ico` — ICO downsampling level (default 4 → 5 120 triangles per surface).
   - `conductivity` — Three-layer values [scalp, skull, brain]
     (default [0.3, 0.006, 0.3] S/m).

3. **BEM solution** — `mne.make_bem_solution()` computes the forward model matrix
   from the BEM surfaces.

### 9.2 Coregistration Transform

MNE fiducial-based coregistration is used. When subject-specific digitized fiducials
are unavailable, fsaverage fiducials are scaled to the subject's head size.

**Identity transform guard:** Auto-generation of an identity transform is refused by
default (`allow_identity_trans = false`) because it is scientifically invalid without
proper digitization. Set to `true` only for debugging purposes.

### 9.3 Docker Image

Default: `freesurfer-mne:7.4.1` (FreeSurfer 7.4.1 + MNE-Python).
Dockerfile: `eeg_pipeline/docker_setup/Dockerfile.freesurfer-mne`.

---

## 10. Multivariate Pain Signatures

**Module:** `analysis/multivariate_signatures.py`

Expression of established multivariate pain biomarkers on statistical or beta maps.

### 10.1 Supported Signatures

| Signature | Weight map | Reference |
|-----------|-----------|-----------|
| **NPS** (Neurologic Pain Signature) | `NPS/weights_NSF_grouppred_cvpcr.nii.gz` | Wager et al., 2013, *NEJM* |
| **SIIPS1** (Stimulus-Independent Pain Signature) | `SIIPS1/nonnoc_v11_4_137subjmap_weighted_mean.nii.gz` | Woo et al., 2017, *Nature Communications* |

### 10.2 Computation

For signature weight map $\mathbf{w}$ and input image $\mathbf{x}$:

**Step 1 — Resampling** (two strategies):

| Strategy | Behavior |
|----------|---------|
| `image_to_weights` *(default)* | Resample input image to the signature's voxel grid (preserves signature resolution). |
| `weights_to_image` | Resample signature weights to the input image grid. |

**Step 2 — Masking:** Intersection of finite voxels in both images. An optional
analysis mask (e.g., brain mask) is resampled and applied.

**Step 3 — Metrics:**

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| Dot product | $\mathbf{x} \cdot \mathbf{w} = \sum_i x_i w_i$ | Raw pattern expression (scale-dependent) |
| Cosine similarity | $\dfrac{\mathbf{x} \cdot \mathbf{w}}{\|\mathbf{x}\|\,\|\mathbf{w}\|}$ | Scale-invariant similarity |
| Pearson $r$ | $r(\mathbf{x}, \mathbf{w})$ | Scale- and mean-invariant correlation |

Voxel count $V$ (finite, unmasked voxels used in computation) is recorded for QC.

---

## 11. Output Layout

```
<deriv_root>/
├── preprocessed/fmri/
│   └── sub-XXXX/
│       ├── anat/                        # fMRIPrep anatomical outputs
│       └── func/                        # fMRIPrep functional outputs
│           ├── *_desc-preproc_bold.nii.gz
│           ├── *_desc-brain_mask.nii.gz
│           └── *_desc-confounds_timeseries.tsv
└── sub-XXXX/fmri/
    ├── first_level/
    │   └── task-<task>/contrast-<name>/
    │       ├── *_stat-z_score_<hash>.nii.gz
    │       ├── *_stat-z_score_<hash>.json    # Provenance sidecar
    │       ├── *_space-MNI152NLin2009cAsym_*.nii.gz
    │       ├── qc/                           # Design matrix TSVs and PNGs
    │       ├── plots/
    │       │   ├── native/                   # Subject-space visualizations
    │       │   ├── mni/                      # MNI-space visualizations
    │       │   └── qc/                       # Carpet, tSNR, motion plots
    │       └── report.html                   # Self-contained HTML report
    └── beta_series/                          # (or lss/ for LSS method)
        └── task-<task>/contrast-<name>/
            ├── trials.tsv
            ├── trial_betas/                  # Per-trial NIfTI beta maps
            ├── condition_betas/              # Condition-averaged beta maps
            ├── signatures/
            │   ├── trial_signature_expression.tsv
            │   ├── condition_signature_expression.tsv
            │   └── group_signature_expression.tsv
            └── provenance.json
```

---

## 12. Configuration Reference

Settings are loaded from `eeg_pipeline/utils/config/eeg_config.yaml`.
A standalone reference template is at `fmri_pipeline/utils/config/fmri_config.yaml`.

### 12.1 Paths

| Key | Description |
|-----|-------------|
| `paths.bids_fmri_root` | BIDS root directory containing raw fMRI data |
| `paths.freesurfer_dir` | FreeSurfer `SUBJECTS_DIR` |
| `paths.freesurfer_license` | Path to FreeSurfer `license.txt` (if unset: `EEG_PIPELINE_FREESURFER_LICENSE`, then `~/license.txt`) |
| `paths.signature_dir` | Root directory containing NPS/SIIPS1 weight maps |

### 12.2 fMRI Preprocessing (`fmri_preprocessing.fmriprep`)

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

### 12.3 First-Level Contrast (`fmri_contrast`)

| Key | Default | Description |
|-----|---------|-------------|
| `enabled` | `false` | Enable contrast computation |
| `input_source` | `fmriprep` | BOLD source (`fmriprep` or `bids_raw`) |
| `fmriprep_space` | `T1w` | fMRIPrep output space for BOLD |
| `require_fmriprep` | `true` | Raise error if preprocessed BOLD is missing |
| `type` | `t-test` | Contrast type |
| `condition_a.column` | `binary_outcome_coded` | Events column for condition A (study-specific; override as needed) |
| `condition_a.value` | `1` | Value selecting condition A trials |
| `condition_b.column` | `binary_outcome_coded` | Events column for condition B (study-specific; override as needed) |
| `condition_b.value` | `0` | Value selecting condition B trials |
| `condition_scope_trial_types` | `null` | Restrict condition selection to specific trial types |
| `events_to_model` | `null` | Restrict which event types enter the GLM |
| `stim_phases_to_model` | `null` | Restrict stimulation sub-phases |
| `formula` | `null` | Custom contrast formula (overrides conditions) |
| `name` | `contrast` | Contrast output name |
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
| `resample_to_freesurfer` | `true` | Resample contrast map to FreeSurfer space |
| `write_design_matrix` | `false` | Write design matrices to disk for QC |

### 12.4 fMRI Constraint (EEG Source Localization)

| Key | Default | Description |
|-----|---------|-------------|
| `fmri_constraint.enabled` | `false` | Enable fMRI-constrained source localization |
| `fmri_constraint.stats_map_path` | `null` | Pre-computed stats map (overrides contrast builder) |
| `fmri_constraint.threshold` | `2.0` | Z-threshold for constraint mask |
| `fmri_constraint.tail` | `pos` | Tail for thresholding (`pos` or `abs`) |
| `fmri_constraint.thresholding.mode` | `z` | Thresholding mode (`z` or `fdr`) |
| `fmri_constraint.cluster_min_voxels` | `10` | Minimum cluster size (voxels) |
| `fmri_constraint.cluster_min_volume_mm3` | `null` | Minimum cluster volume (overrides voxel count) |
| `fmri_constraint.max_clusters` | `20` | Maximum number of clusters to retain |

### 12.5 BEM Generation

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

## 13. Dependencies

### Python Packages

| Package | Role |
|---------|------|
| **nilearn** | GLM fitting (`FirstLevelModel`), contrast computation, plotting, cluster tables |
| **nibabel** | NIfTI/MGZ I/O, image resampling |
| **NumPy** | Array operations, statistics |
| **SciPy** | Connected components (cluster filtering), normal distribution (z-thresholds) |
| **pandas** | Events and confounds TSV I/O |
| **matplotlib** | QC plots (carpet, tSNR, motion, histograms) |

### External Tools

| Tool | Role |
|------|------|
| **dcm2niix** | DICOM-to-NIfTI conversion (Stage 1) |
| **Docker** or **Apptainer** | Container runtime for fMRIPrep and BEM generation |
| **fMRIPrep** (container) | BOLD preprocessing (Stage 2) |
| **FreeSurfer** (via Docker) | Watershed BEM surfaces, `recon-all` output |
| **MNE-Python** (via Docker) | BEM model and solution computation |
