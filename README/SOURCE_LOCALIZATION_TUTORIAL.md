# EEG Source Localization Tutorial

This tutorial covers two ways to run EEG source localization in this pipeline:

1. **EEG-only (template-based)** — uses `fsaverage` template head model; no fMRI, no subject MRI required
2. **fMRI-constrained** — uses subject-specific MRI + fMRI statistical map to constrain source space

---

## Quick Start: Which path should I use?

| Path | Requires | Use case |
|------|----------|----------|
| **EEG-only** | Cleaned epochs, standard montage | Quick validation, template-level analysis |
| **fMRI-constrained** | FreeSurfer subject, BEM, trans, fMRI stats map | Research-grade, subject-specific, fMRI-guided |

---

## Prerequisites (both paths)

### 1. Activate environment
```bash
source eeg_pipeline/.venv311/bin/activate
```

### 2. Verify cleaned epochs exist
You must have cleaned epochs for your subject/task. Example location:
```
eeg_pipeline/data/derivatives/preprocessed/sub-0000/eeg/sub-0000_task-thermalactive_proc-clean_epo.fif
```

## Path 1: EEG-Only Source Localization (Template-Based)

This uses the `fsaverage` template head model. No subject MRI or fMRI required.

### What you need

| File | Location | Purpose |
|------|----------|---------|
| Cleaned epochs | `derivatives/preprocessed/sub-XXXX/eeg/*_proc-clean_epo.fif` | EEG data with sensor geometry |
| Config | `eeg_pipeline/utils/config/eeg_config.yaml` | Feature extraction settings |

### Step 1: Enable source localization in feature extraction

Run feature extraction with source localization enabled:

```bash
python -m eeg_pipeline.cli.main features compute \
  --subject 0000 \
  --categories sourcelocalization \
  --spatial roi global
```

### Step 2: (Optional) Configure source localization method

Edit `eeg_pipeline/utils/config/eeg_config.yaml` or use CLI flags:

```bash
python -m eeg_pipeline.cli.main features compute \
  --subject 0000 \
  --categories sourcelocalization \
  --source-method lcmv \
  --source-spacing oct6 \
  --source-reg 0.05
```

Available methods:
- `lcmv` — LCMV beamformer (default)
- `eloreta` — eLORETA inverse solution

### Step 3: Verify outputs

Check for source localization features:
```bash
ls derivatives/sub-0000/eeg/features/sourcelocalization/
```

Expected files:
- `features_sourcelocalization.tsv` — source-space power/features per epoch

Feature columns will be named like:
- `src_full_lcmv_alpha_*_power`
- `src_full_lcmv_alpha_global_power`

### Step 4: Visualize (optional)

```bash
python -m eeg_pipeline.cli.main features visualize --subject 0000
```

---

## Path 2: fMRI-Constrained Source Localization

This uses subject-specific MRI + fMRI statistical map to constrain the source space. Requires FreeSurfer, BEM, coregistration transform, and an fMRI stats map.

### What you need

| File | Location | Purpose |
|------|----------|---------|
| Cleaned epochs | `derivatives/preprocessed/sub-XXXX/eeg/*_proc-clean_epo.fif` | EEG data with sensor geometry |
| T1w anatomical | `eeg_pipeline/data/fMRI_data/sub-XXXX/anat/*_T1w.nii.gz` | Subject MRI for FreeSurfer |
| FreeSurfer subject | `eeg_pipeline/data/derivatives/freesurfer/sub-XXXX/` | Reconstructed MRI surfaces |
| BEM solution | `eeg_pipeline/data/derivatives/freesurfer/sub-XXXX/bem/*-bem-sol.fif` | Boundary element model |
| Coregistration transform | `*trans.fif` | EEG ↔ MRI alignment |
| fMRI stats map | `*_stats.nii.gz` (e.g., t-map, z-map) | fMRI contrast to constrain sources |
| Config | `eeg_pipeline/utils/config/eeg_config.yaml` | Feature extraction settings |

---

### Option A: Automated TUI Workflow (Recommended)

The pipeline can **automatically generate BEM model, BEM solution, and coregistration transform** via Docker when you enable the corresponding options in the TUI wizard.

> **Note:** Requires Docker to be installed. See https://docs.docker.com/get-docker/

#### Prerequisites

1. **Docker installed and running**
2. **FreeSurfer license file** (free from https://surfer.nmr.mgh.harvard.edu/registration.html)
3. **FreeSurfer recon-all completed** for your subject (see Step 4 in manual workflow below)
4. **fMRI stats map** in FreeSurfer subject space

#### Using the TUI Wizard

1. Launch the TUI:
   ```bash
   ./eeg_pipeline/cli/tui/tui
   ```

2. Navigate to **Features → Source Localization**

3. Set **Mode** to `fMRI-informed`

4. Enable the auto-generation options:
   - **Create Trans**: `on` — auto-generates coregistration transform
   - **Create BEM Model**: `on` — auto-generates BEM surfaces (watershed)
   - **Create BEM Solution**: `on` — auto-generates BEM solution matrix

5. Set **FS License** to your FreeSurfer license path (default: `eeg_pipeline/licenses/license_freesurfer.txt`)

6. Set **FS Subject** to your FreeSurfer subject name (e.g., `sub-0000`)

7. Provide your **fMRI Stats Map** path

8. Run the pipeline — BEM and trans files will be generated automatically before source localization

#### CLI Equivalent

```bash
python -m eeg_pipeline.cli.main features compute \
  --subject 0000 \
  --categories sourcelocalization \
  --source-fmri \
  --source-fmri-stats-map /path/to/sub-0000_pain_vs_baseline_zmap.nii.gz \
  --source-subject sub-0000 \
  --source-subjects-dir /path/to/freesurfer \
  --source-create-trans \
  --source-create-bem-model \
  --source-create-bem-solution \
  --source-fs-license eeg_pipeline/licenses/license_freesurfer.txt
```

---

### Option B: Manual Docker Workflow (Step-by-step)

#### Step 1: Get FreeSurfer license

1. Go to https://surfer.nmr.mgh.harvard.edu/registration.html
2. Fill out the form (free academic license)
3. Download the license file
4. Save it as `license_freesurfer.txt` in `eeg_pipeline/licenses/`

#### Step 2: Set up directories

Run these commands in your terminal:

```bash
# Set SUBJECTS_DIR
export SUBJECTS_DIR=/Users/joduq24/Desktop/Pain_fMRI_EEG/eeg_pipeline/data/derivatives/freesurfer
mkdir -p $SUBJECTS_DIR

# Set path to your FreeSurfer license
export FS_LICENSE=/Users/joduq24/Desktop/Pain_fMRI_EEG/eeg_pipeline/licenses/license_freesurfer.txt

# Verify the license file exists
cat $FS_LICENSE
```

#### Step 3: Build FreeSurfer + MNE Docker image

First, build a Docker image that includes both FreeSurfer and MNE-Python:

```bash
docker build --platform linux/amd64 -t freesurfer-mne:7.4.1 -f Dockerfile.freesurfer-mne .
```

This takes 5-10 minutes (downloads ~2GB of packages). You only need to do this once.

#### Step 4: Run FreeSurfer recon-all (1-3 hours)

```bash
docker run --rm \
  -v /Users/joduq24/Desktop/Pain_fMRI_EEG:/data \
  -v $SUBJECTS_DIR:/subjects \
  -v $FS_LICENSE:/usr/local/freesurfer/.license \
  --platform linux/amd64 \
  freesurfer-mne:7.4.1 \
  recon-all \
    -subjid sub-0000 \
    -i /data/eeg_pipeline/data/fMRI_data/sub-0000/anat/sub-0000_acq-mprageipat2_T1w.nii.gz \
    -all \
    -sd /subjects
```

Wait for completion. This takes 1-3 hours depending on your hardware.

#### Step 5: Verify FreeSurfer output

```bash
ls -la $SUBJECTS_DIR/sub-0000/mri/
```

You should see files like `orig.mgz`, `T1.mgz`, `brain.mgz`, `wmparc.mgz`.

#### Step 6: Generate BEM model and solution

```bash
docker run --rm \
  -v /Users/joduq24/Desktop/Pain_fMRI_EEG:/data \
  -v $SUBJECTS_DIR:/subjects \
  -v $FS_LICENSE:/usr/local/freesurfer/.license \
  --platform linux/amd64 \
  freesurfer-mne:7.4.1 \
  bash -lc "
    set -e
    set +u
    source \$FREESURFER_HOME/SetUpFreeSurfer.sh
    set -u

    mne watershed_bem --subject sub-0000 --overwrite

    python -c 'import mne; bem_model = mne.make_bem_model(subject=\"sub-0000\", ico=4, subjects_dir=\"/subjects\", conductivity=[0.3, 0.006, 0.3]); mne.write_bem_surfaces(\"/subjects/sub-0000/bem/sub-0000-5120-5120-5120-bem.fif\", bem_model, overwrite=True); bem_solution = mne.make_bem_solution(\"/subjects/sub-0000/bem/sub-0000-5120-5120-5120-bem.fif\"); mne.write_bem_solution(\"/subjects/sub-0000/bem/sub-0000-5120-5120-5120-bem-sol.fif\", bem_solution, overwrite=True)'
  "
```

#### Step 7: Verify BEM solution

```bash
ls -la $SUBJECTS_DIR/sub-0000/bem/
```

You should see `sub-0000-5120-5120-5120-bem-sol.fif`.

#### Step 8: Create coregistration transform

Create a Python script `create_trans.py`:

```python
import mne
from mne_bids import BIDSPath, read_raw_bids

bids_root = "eeg_pipeline/data/bids_output"
subject = "0000"
task = "thermalactive"

bids_path = BIDSPath(
    subject=subject,
    task=task,
    run=1,
    datatype="eeg",
    suffix="eeg",
    root=bids_root,
)

raw = read_raw_bids(bids_path, verbose=False)

# Load CapTrak electrodes
montage = mne.channels.read_dig_bids(bids_path.copy().update(suffix="electrodes"))
raw.set_montage(montage)

# Coregister (requires MRI surfaces)
mne.gui.coregistration(
    subject="sub-0000",
    subjects_dir="/Users/joduq24/Desktop/Pain_fMRI_EEG/eeg_pipeline/data/derivatives/freesurfer",
    inst=raw,
)
```

Run the script:

```bash
python create_trans.py
```

A GUI will open. Manually align the electrodes to the MRI, then save the transform as `sub-0000-trans.fif` in the current directory.

#### Step 9: Move the transform file

```bash
mv sub-0000-trans.fif /Users/joduq24/Desktop/Pain_fMRI_EEG/eeg_pipeline/data/derivatives/freesurfer/sub-0000/bem/
```

#### Step 10: Generate fMRI statistical map

You have **two options** for generating the fMRI statistical map:

##### Option A: Automated Contrast Builder (Recommended)

The pipeline can automatically build the fMRI contrast from your BIDS-formatted BOLD data. This requires:
- BOLD NIfTI files in BIDS format (`sub-XXXX/func/*_bold.nii.gz`)
- Events TSV files with `onset`, `duration`, `trial_type` columns

Enable the contrast builder in the TUI advanced configuration or via CLI:

```bash
python -m eeg_pipeline.cli.main features compute \
  --subject 0000 \
  --categories sourcelocalization \
  --source-fmri \
  --source-fmri-contrast-enabled \
  --source-fmri-contrast-type t-test \
  --source-fmri-contrast-cond1 pain_high \
  --source-fmri-contrast-cond2 baseline \
  --source-fmri-contrast-name pain_vs_baseline \
  --source-fmri-hrf-model spm \
  --source-fmri-resample-to-fs
```

The contrast builder will:
1. Discover BOLD runs from your BIDS fMRI directory
2. Load events from corresponding TSV files
3. Fit a first-level GLM using nilearn
4. Compute the specified contrast
5. Resample the result to FreeSurfer subject space
6. Save the output to `derivatives/sub-XXXX/fmri_contrasts/`

##### Option B: Pre-computed Stats Map

If you already have an fMRI stats map or prefer to use external tools (SPM, FSL, AFNI), provide the path directly:

```bash
--source-fmri-stats-map /path/to/sub-0000_pain_vs_baseline_zmap.nii.gz
```

The stats map must be a 3D NIfTI file resampled to the FreeSurfer subject space (aligned with `T1.mgz` or `orig.mgz`).

Example manual creation with nilearn:

```python
from nilearn.glm.first_level import FirstLevelModel
from nilearn.image import resample_to_img

# Fit GLM and generate contrast
flm = FirstLevelModel(t_r=2.0, drift_model="cosine", hrf_model="spm")
flm.fit(fmri_img, events=events_df)
z_map = flm.compute_contrast("pain_high - baseline", output_type="z_score")

# Resample to FreeSurfer subject space
fs_t1 = "/path/to/freesurfer/sub-0000/mri/T1.mgz"
z_map_resampled = resample_to_img(z_map, fs_t1, interpolation="continuous")
z_map_resampled.to_filename("sub-0000_pain_vs_baseline_zmap.nii.gz")
```

#### Step 11: Run fMRI-constrained source localization

```bash
python -m eeg_pipeline.cli.main features compute \
  --subject 0000 \
  --categories sourcelocalization \
  --source-fmri \
  --source-fmri-stats-map /path/to/sub-0000_pain_vs_baseline_zmap.nii.gz \
  --source-fmri-threshold 3.1 \
  --source-fmri-tail pos \
  --source-subject sub-0000 \
  --source-subjects-dir /Users/joduq24/Desktop/Pain_fMRI_EEG/eeg_pipeline/data/derivatives/freesurfer \
  --source-trans /Users/joduq24/Desktop/Pain_fMRI_EEG/eeg_pipeline/data/derivatives/freesurfer/sub-0000/bem/sub-0000-trans.fif \
  --source-bem /Users/joduq24/Desktop/Pain_fMRI_EEG/eeg_pipeline/data/derivatives/freesurfer/sub-0000/bem/sub-0000-5120-5120-5120-bem-sol.fif \
  --source-method lcmv \
  --spatial global
```

#### Step 12: Verify outputs

```bash
ls -la derivatives/sub-0000/eeg/features/sourcelocalization/
```

You should see `features_sourcelocalization.tsv` with fMRI-constrained source features.

---

### Alternative: Local FreeSurfer installation

If you prefer to install FreeSurfer locally instead of using Docker:

1. Download FreeSurfer: https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall
2. Install it (requires ~10GB and license)
3. Set up environment:

```bash
export FREESURFER_HOME=/path/to/freesurfer
source $FREESURFER_HOME/SetUpFreeSurfer.sh
export SUBJECTS_DIR=/Users/joduq24/Desktop/Pain_fMRI_EEG/eeg_pipeline/data/derivatives/freesurfer

# Run recon-all
recon-all \
  -subjid sub-0000 \
  -i /Users/joduq24/Desktop/Pain_fMRI_EEG/eeg_pipeline/data/fMRI_data/sub-0000/anat/sub-0000_acq-mprageipat2_T1w.nii.gz \
  -all \
  -sd $SUBJECTS_DIR

# Generate BEM
cd $SUBJECTS_DIR/sub-0000
mne watershed_bem --subject sub-0000 --overwrite
mne make_bem_solution --subject sub-0000 --bem-model watershed --bem-sol-name sub-0000-5120-5120-5120-bem-sol
```

Then continue from Step 7 above (create coregistration transform).

---

## CLI Flags Reference

### Source localization flags

| Flag | Description | Default |
|------|-------------|---------|
| `--source-method` | Inverse method: `lcmv` or `eloreta` | `lcmv` |
| `--source-spacing` | Source space spacing: `oct5`, `oct6`, `ico4`, `ico5` | `oct6` |
| `--source-reg` | LCMV regularization parameter | `0.05` |
| `--source-snr` | eLORETA assumed SNR for regularization | `3.0` |
| `--source-loose` | eLORETA loose orientation constraint 0-1 | `0.2` |
| `--source-depth` | eLORETA depth weighting 0-1 | `0.8` |
| `--source-parc` | Parcellation for ROI extraction: `aparc`, `aparc.a2009s`, `HCPMMP1` | `aparc` |
| `--source-connectivity-method` | Connectivity method for source-space analysis: `aec`, `wpli`, `plv` | `aec` |
| `--source-subject` | FreeSurfer subject name (e.g., `sub-0001`). If unset, defaults to `sub-{subject}` | `sub-{subject}` |
| `--source-subjects-dir` | FreeSurfer SUBJECTS_DIR path for subject-specific source localization | (none) |
| `--source-trans` | EEG ↔ MRI coregistration transform .fif | (none) |
| `--source-bem` | BEM solution .fif (e.g., `*-bem-sol.fif`) | (none) |
| `--source-mindist-mm` | Minimum distance from sources to inner skull (mm) | `5.0` |

### fMRI constraint flags

| Flag | Description | Default |
|------|-------------|---------|
| `--source-fmri` | Enable fMRI-constrained source localization | `False` |
| `--no-source-fmri` | Disable fMRI-constrained source localization (overrides config) | - |
| `--source-fmri-stats-map` | Path to fMRI statistical map NIfTI in the same MRI space as the FreeSurfer subject | (none) |
| `--source-fmri-threshold` | Threshold applied to fMRI stats map | `3.1` |
| `--source-fmri-tail` | Threshold tail: `pos` (positive only) or `abs` (absolute value) | `pos` |
| `--source-fmri-cluster-min-voxels` | Minimum cluster size in voxels after thresholding | `50` |
| `--source-fmri-cluster-min-mm3` | Minimum cluster volume (mm³) after thresholding (preferred; stable across voxel sizes; overrides `--source-fmri-cluster-min-voxels`) | `400` |
| `--source-fmri-max-clusters` | Maximum number of clusters kept from fMRI map | `20` |
| `--source-fmri-max-voxels-per-cluster` | Maximum voxels sampled per cluster (set 0 for no limit) | `2000` |
| `--source-fmri-max-total-voxels` | Maximum total voxels across all clusters (set 0 for no limit) | `20000` |
| `--source-fmri-random-seed` | Random seed for voxel subsampling (0 = nondeterministic) | `0` |

### fMRI Contrast Builder flags

| Flag | Description | Default |
|------|-------------|---------|
| `--source-fmri-contrast-enabled` | Enable automatic contrast building from BOLD data | `False` |
| `--source-fmri-contrast-type` | Contrast type: `t-test`, `paired-t-test`, `f-test`, `custom` | `t-test` |
| `--source-fmri-contrast-cond1` | First condition name (e.g., `pain_high`) | (none) |
| `--source-fmri-contrast-cond2` | Second condition name (e.g., `baseline`) | (none) |
| `--source-fmri-contrast-formula` | Custom contrast formula (e.g., `pain_high - pain_low`) | (none) |
| `--source-fmri-contrast-name` | Output contrast name | `pain_vs_baseline` |
| `--source-fmri-runs` | Comma-separated run numbers to include (e.g., `1,2,3`) | (auto-detect) |
| `--source-fmri-hrf-model` | HRF model: `spm`, `flobs`, `fir` | `spm` |
| `--source-fmri-drift-model` | Drift model: `none`, `cosine`, `polynomial` | `cosine` |
| `--source-fmri-high-pass` | High-pass filter cutoff in Hz | `0.008` |
| `--source-fmri-low-pass` | Optional low-pass filter cutoff in Hz (generally avoid for task GLMs unless you know you need it) | (disabled) |
| `--source-fmri-cluster-correction` | Enable cluster-extent filtering (heuristic; **not** cluster-level FWE correction) | `True` |
| `--source-fmri-cluster-p-threshold` | Cluster-forming p-threshold | `0.001` |
| `--source-fmri-output-type` | Output type: `z-score`, `t-stat`, `cope`, `beta` | `z-score` |
| `--source-fmri-resample-to-fs` | Auto-resample stats map to FreeSurfer subject space | `True` |

---

## Troubleshooting

### Error: “No cleaned epochs found for sub-XXXX, task-YYYY”

- Verify epochs exist: `find derivatives/preprocessed -name "*proc-clean_epo.fif"`
- Check task name matches config: `project.task` in `eeg_config.yaml`

### Error: “fMRI constraint enabled but no stats map path provided”

- Add `--source-fmri-stats-map /path/to/stats.nii.gz`
- Or disable fMRI constraint: remove `--source-fmri`

### Error: “fMRI-constrained source localization requires feature_engineering.sourcelocalization.subjects_dir”

- Add `--source-subjects-dir $SUBJECTS_DIR`
- Ensure FreeSurfer subject exists: `ls $SUBJECTS_DIR/sub-XXXX`

### Error: “fMRI stats map threshold produced empty mask”

- Lower `--source-fmri-threshold` (e.g., `2.5`)
- Check stats map contains positive values
- Verify stats map is in correct MRI space

### Error: “Forward model: 0 sources”

- Check BEM solution exists and is valid
- Verify transform file is not corrupted
- Ensure electrode positions are in `epochs.info["dig"]`

---

## Summary Checklist

### EEG-only (template-based)
- [ ] Cleaned epochs exist
- [ ] `has dig: True` in epochs
- [ ] Run `features compute --categories sourcelocalization`
- [ ] Verify `features_sourcelocalization.tsv` exists

### fMRI-constrained
- [ ] Cleaned epochs exist with digitization
- [ ] FreeSurfer subject reconstructed
- [ ] BEM solution generated
- [ ] Coregistration transform created
- [ ] fMRI stats map in subject MRI space
- [ ] Run `features compute --source-fmri --source-fmri-stats-map ...`
- [ ] Verify fMRI-constrained features exist

---

## References

- MNE-Python source localization: https://mne.tools/stable/auto_tutorials/source-modeling/30_source_localization.html
- MNE-BIDS electrodes sidecars: https://mne.tools/mne-bids/stable/use/definitions.html#electrodes
- FreeSurfer recon-all: https://surfer.nmr.mgh.harvard.edu/fswiki/recon-all
