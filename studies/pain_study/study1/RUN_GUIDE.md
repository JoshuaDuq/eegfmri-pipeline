# Study 1 - Run Guide

## Alliance Canada

The repository-owned Alliance workflow selects infrastructure with
`ALLIANCE_CLUSTER=rorqual|trillium`. From the repository root on the local Mac:

```bash
export ALLIANCE_CLUSTER=rorqual
bash local_workflows/alliance_canada/start_alliance_connection.sh
bash local_workflows/alliance_canada/setup_alliance_study1.sh
bash local_workflows/alliance_canada/setup_alliance_runtime.sh
bash local_workflows/alliance_canada/submit_study1_alliance.sh
```

The tracked subject manifest includes `0000`, `0001`, `0003`, `0004`, `0005`,
`0007`, `0008`, `0009`, `0010`, `0011`, `0012`, `0013`, and `0014`; `0006` is
excluded. The workflow requires all 13 participants, uses 5,000 confirmatory
permutations, and fails before transfer when required BIDS, cleaned EEG,
fMRIPrep, or signature inputs are absent.

Before migrating a run from Trillium to Rorqual, retrieve any Trillium-only
fMRIPrep subjects with:

```bash
bash local_workflows/alliance_canada/fetch_study1_fmriprep_from_trillium.sh
```

Job records and logs are written under `/scratch/$USER/study1_logs` on the
selected cluster. Monitor the recorded IDs with `squeue` and `sacct`.

This guide runs the Study 1 signature-prediction workflow from prepared EEG/fMRI derivatives.
Study 1 does not run fMRIPrep itself. It expects the task fMRI BIDS data, cleaned EEG/event
derivatives, external signature maps, a fixed a-priori scoring mask, and fMRIPrep preprocessed BOLD
outputs to already be present in the configured roots.

Run the workflow in this order:

1. Install the private `studies` package so `eeg-pipeline` exposes `signature-prediction`.
2. Upload or verify external signature assets: `NPS`, `SIIPS1`, `signature_manifest.yaml`, and the
   fixed MNI scoring mask.

   `signature_manifest.yaml` is now generated from a declared provenance file rather than from a
   default space label. Copy
   `studies/pain_study/study1/config/signature_provenance.example.yaml`, fill in each signature's
   `space`, `source_space`, `spatial_reference` and — whenever the delivered space differs from the
   source space — a `transform` record, then regenerate:

   ```bash
   python -m studies.pain_study.study1.signature_manifest \
     "$SIGNATURE_DIR" "$SIGNATURE_DIR/signature_manifest.yaml" \
     --provenance studies/pain_study/study1/config/signature_provenance.yaml
   ```

   Manifests written before this change carry only `space` and are rejected by
   `prepare-targets`. The published SIIPS1 weights were trained in SPM MNI152 space, so declaring
   the unmodified file as MNI152NLin2009cAsym is rejected outright: resampling onto the fMRIPrep
   grid changes sampling, not anatomical correspondence. Until a validated transform exists,
   `prepare-targets` will not run against those assets, which is intended.
3. Run fMRIPrep for the Study 1 fMRI subjects if preprocessed BOLD files are missing.
4. Run the Study 1 stages: `prepare-targets`, `prepare-features`, `feature-benchmark`, then
   `report`. The two standalone sensor-topography families can be generated immediately after
   `prepare-features`; they do not require `feature-benchmark` or `report` outputs.

The command list must include `signature-prediction`. If it does not, install the private studies
package that exposes `studies.pain_study.cli.command_registry:signature_prediction_command`.

Confirm local registration before running Study 1 on Kingston:

```bash
cd /Users/joduq24/Desktop/EEG_fMRI_Pipeline
EEG_PIPELINE=".venv/bin/eeg-pipeline"
PYTHON=".venv/bin/python"

"$EEG_PIPELINE" --help
```

## Trillium End-to-End Order

Run Study 1 on Trillium only after the narrow upload has placed the required fMRI BIDS, EEG BIDS,
and cleaned derivatives under `/project/def-mpcoll/joshduq`. The Trillium commands use:

```text
/project/def-mpcoll/joshduq/bids/eeg
/project/def-mpcoll/joshduq/bids/fmri
/project/def-mpcoll/joshduq/derivatives
/project/def-mpcoll/joshduq/external
```

From a local Mac terminal, upload the external signature assets if they are missing on Trillium:

```bash
ssh joshduq@trillium.alliancecan.ca \
  'mkdir -p /project/def-mpcoll/joshduq/external/NPS /project/def-mpcoll/joshduq/external/SIIPS1'

rsync -avh \
  /Volumes/KINGSTON/EEG_fMRI_data/external/signature_manifest.yaml \
  joshduq@trillium.alliancecan.ca:/project/def-mpcoll/joshduq/external/

rsync -avh \
  /Volumes/KINGSTON/EEG_fMRI_data/external/tpl-MNI152NLin2009cAsym_res-02_desc-brain_mask.nii.gz \
  joshduq@trillium.alliancecan.ca:/project/def-mpcoll/joshduq/external/

rsync -avh \
  /Volumes/KINGSTON/EEG_fMRI_data/external/NPS/weights_NSF_grouppred_cvpcr.nii.gz \
  joshduq@trillium.alliancecan.ca:/project/def-mpcoll/joshduq/external/NPS/

rsync -avh \
  /Volumes/KINGSTON/EEG_fMRI_data/external/SIIPS1/nonnoc_v11_4_137subjmap_weighted_mean.nii.gz \
  joshduq@trillium.alliancecan.ca:/project/def-mpcoll/joshduq/external/SIIPS1/
```

fMRIPrep also requires the FreeSurfer license. Upload it from the Mac before submitting fMRIPrep:

```bash
ssh joshduq@trillium.alliancecan.ca 'mkdir -p /project/def-mpcoll/joshduq/licenses'

rsync -avh \
  /Users/joduq24/license.txt \
  joshduq@trillium.alliancecan.ca:/project/def-mpcoll/joshduq/licenses/license.txt
```

On Trillium, prepare the Python environment and private Study 1 command:

```bash
cd /project/def-mpcoll/joshduq/EEG_fMRI_Pipeline

source local_workflows/alliance_canada/alliance_env.sh

module load StdEnv/2023
module load python/3.11
module load gcc
module load arrow

if [[ ! -d "$EEG_PIPELINE_VENV" ]]; then
  python -m venv "$EEG_PIPELINE_VENV"
fi

source "$EEG_PIPELINE_VENV/bin/activate"

python -m pip install --upgrade pip
python -m pip install -e ".[dev,ml]"
python -m pip install -e ./studies

eeg-pipeline --help
```

The help output must include `signature-prediction`.

Before running Study 1, verify fMRIPrep outputs exist:

```bash
find /project/def-mpcoll/joshduq/derivatives/preprocessed/fmri/fmriprep \
  -type f \
  -name '*task-thermalactive*desc-preproc_bold.nii.gz' \
  | head
```

If that command prints no files, submit fMRIPrep first:

```bash
cd /project/def-mpcoll/joshduq/EEG_fMRI_Pipeline

bash local_workflows/alliance_canada/setup_python_and_container.sh

cd /project/def-mpcoll/joshduq/EEG_fMRI_Pipeline/local_workflows/alliance_canada

JOB_ID="$(bash submit_fmriprep_array.sh | tail -n 1)"
if [[ -z "$JOB_ID" ]]; then
  echo "fMRIPrep submission did not return a Slurm job id." >&2
  exit 1
fi

echo "Submitted fMRIPrep job: $JOB_ID"

squeue -j "$JOB_ID"
```

The submit wrapper is Trillium-specific: it writes Slurm logs to
`/scratch/joshduq/fmriprep_logs`, writes fMRIPrep outputs under
`/scratch/joshduq/derivatives`, uses `/scratch/joshduq/templateflow` for TemplateFlow, requests
less than 24 hours, does not pass `--mem`, and passes `--skip-bids-validation` because the narrow
task upload omits non-Study-1 rest BOLD files referenced by some fieldmap `IntendedFor` metadata.
Do not submit fMRIPrep with logs under the project repository or with an explicit memory request on
Trillium.

When the job leaves `squeue`, require `COMPLETED` before continuing:

```bash
sacct -j "$JOB_ID" --format=JobID,JobName%30,State,Elapsed,MaxRSS
```

After all array tasks are `COMPLETED`, copy the fMRIPrep outputs from scratch back to the project
derivatives root from the Trillium login node. The wrapper writes subject folders directly under
the scratch fMRI output root; downstream Study 1 discovery expects them under the project-side
`fmriprep` directory:

```bash
mkdir -p /project/def-mpcoll/joshduq/derivatives/preprocessed/fmri/fmriprep

rsync -avh --delete \
  /scratch/joshduq/derivatives/preprocessed/fmri/ \
  /project/def-mpcoll/joshduq/derivatives/preprocessed/fmri/fmriprep/
```

Then run the Study 1 smoke test on Trillium through Slurm. Compute nodes can read the project
inputs but must write Study 1 outputs on scratch, so expose the EEG inputs in the scratch
derivatives root and submit the script from scratch. Use the full uploaded EEG Study 1 cohort:
`sub-0000`, `sub-0001`, `sub-0003`, and `sub-0005`. This requires valid fMRIPrep
`desc-preproc_bold.nii.gz` outputs for those same four subjects.

```bash
mkdir -p /scratch/joshduq/derivatives/preprocessed
ln -sfn \
  /project/def-mpcoll/joshduq/derivatives/preprocessed/eeg \
  /scratch/joshduq/derivatives/preprocessed/eeg

cat > /scratch/joshduq/run_study1_smoke.sh <<'EOF'
set -euo pipefail

cd /project/def-mpcoll/joshduq/EEG_fMRI_Pipeline

source local_workflows/alliance_canada/alliance_env.sh

module load StdEnv/2023
module load python/3.11
module load gcc
module load arrow

export NUMBA_CACHE_DIR="/scratch/joshduq/study1_numba_cache"
mkdir -p "$NUMBA_CACHE_DIR"

source "$EEG_PIPELINE_VENV/bin/activate"

EEG_PIPELINE="$EEG_PIPELINE_VENV/bin/eeg-pipeline"

STUDY1_CONFIG="studies/pain_study/study1/config/study1_smoketest.yaml"
STUDY1_RUN_ID="study1_smoke_$(date +%Y%m%d_%H%M%S)"

TASK="thermalactive"
SIGNATURE_DIR="/project/def-mpcoll/joshduq/external"
NPS_MAP="NPS/weights_NSF_grouppred_cvpcr.nii.gz"
SIIPS1_MAP="SIIPS1/nonnoc_v11_4_137subjmap_weighted_mean.nii.gz"
SIGNATURE_MANIFEST="signature_manifest.yaml"
SIGNATURE_MASK="tpl-MNI152NLin2009cAsym_res-02_desc-brain_mask.nii.gz"
SIGNATURE_MAPS_JSON='[{"name":"NPS","path":"NPS/weights_NSF_grouppred_cvpcr.nii.gz"},{"name":"SIIPS1","path":"SIIPS1/nonnoc_v11_4_137subjmap_weighted_mean.nii.gz"}]'

SUBJECT_ARGS=(--subject 0000 --subject 0001 --subject 0003 --subject 0005)

COMMON_ARGS=(
  --task "$TASK"
  --study1-config "$STUDY1_CONFIG"
  --bids-root "$BIDS_EEG_ROOT"
  --bids-fmri-root "$BIDS_FMRI_ROOT"
  --deriv-root "$FMRIPREP_DERIV_ROOT"
  --set "paths.signature_dir=$SIGNATURE_DIR"
  --set "paths.signature_maps=$SIGNATURE_MAPS_JSON"
  --set "study1.targets.signature_manifest_path=$SIGNATURE_MANIFEST"
  --set "study1.targets.signature_provenance.NPS.path=$NPS_MAP"
  --set "study1.targets.signature_provenance.SIIPS1.path=$SIIPS1_MAP"
  --set "study1.targets.signature_scoring_mask_path=$SIGNATURE_MASK"
  --set "study1.cohort.min_subjects=4"
  --set "study1.outputs.root_name=$STUDY1_RUN_ID"
)

"$EEG_PIPELINE" signature-prediction prepare-targets "${SUBJECT_ARGS[@]}" "${COMMON_ARGS[@]}"
"$EEG_PIPELINE" signature-prediction prepare-features "${SUBJECT_ARGS[@]}" "${COMMON_ARGS[@]}"
"$EEG_PIPELINE" signature-prediction feature-benchmark "${SUBJECT_ARGS[@]}" "${COMMON_ARGS[@]}"
"$EEG_PIPELINE" signature-prediction report "${SUBJECT_ARGS[@]}" "${COMMON_ARGS[@]}"

echo "$FMRIPREP_DERIV_ROOT/group/multimodal/$STUDY1_RUN_ID"
EOF

chmod +x /scratch/joshduq/run_study1_smoke.sh
mkdir -p /scratch/joshduq/study1_logs

JOB_ID="$(
  sbatch --parsable \
    --account=def-mpcoll \
    --time=06:00:00 \
    --cpus-per-task=16 \
    --output=/scratch/joshduq/study1_logs/study1_smoke_%j.out \
    --error=/scratch/joshduq/study1_logs/study1_smoke_%j.err \
    --wrap="bash /scratch/joshduq/run_study1_smoke.sh"
)"

echo "Submitted Study 1 smoke job: $JOB_ID"
squeue -j "$JOB_ID"
```

For a formal Trillium rerun, edit only the two config lines in that script:

```bash
STUDY1_CONFIG="studies/pain_study/study1/config/study1_config.yaml"
STUDY1_RUN_ID="study1_$(date +%Y%m%d_%H%M%S)"
```

## Copy-Paste Run

Use this block for a normal rerun on the Kingston Study 1 layout. Change only `STUDY1_RUN_ID` when
you want a new output directory, and change `SUBJECT_ARGS` only when you need to audit a specific
cohort instead of all curated subjects.

```bash
cd /Users/joduq24/Desktop/EEG_fMRI_Pipeline

EEG_PIPELINE=".venv/bin/eeg-pipeline"
PYTHON=".venv/bin/python"
STUDY1_CONFIG="studies/pain_study/study1/config/study1_config.yaml"
STUDY1_RUN_ID="study1_$(date +%Y%m%d)"

EEG_BIDS_ROOT="/Volumes/KINGSTON/EEG_fMRI_data/bids_output/eeg"
FMRI_BIDS_ROOT="/Volumes/KINGSTON/EEG_fMRI_data/bids_output/fmri"
SOURCE_DATA_ROOT="/Volumes/KINGSTON/EEG_fMRI_data/source_data"
DERIV_ROOT="/Volumes/KINGSTON/EEG_fMRI_data/derivatives"
SIGNATURE_DIR="/Volumes/KINGSTON/EEG_fMRI_data/external"
TASK="thermalactive"

NPS_MAP="NPS/weights_NSF_grouppred_cvpcr.nii.gz"
SIIPS1_MAP="SIIPS1/nonnoc_v11_4_137subjmap_weighted_mean.nii.gz"
SIGNATURE_MANIFEST="signature_manifest.yaml"
SIGNATURE_MASK="tpl-MNI152NLin2009cAsym_res-02_desc-brain_mask.nii.gz"
SIGNATURE_MAPS_JSON='[
  {"name":"NPS","path":"NPS/weights_NSF_grouppred_cvpcr.nii.gz"},
  {"name":"SIIPS1","path":"SIIPS1/nonnoc_v11_4_137subjmap_weighted_mean.nii.gz"}
]'

SUBJECT_ARGS=(--all-subjects)
COMMON_ARGS=(
  --task "$TASK"
  --study1-config "$STUDY1_CONFIG"
  --bids-root "$EEG_BIDS_ROOT"
  --bids-fmri-root "$FMRI_BIDS_ROOT"
  --deriv-root "$DERIV_ROOT"
  --set "paths.signature_dir=$SIGNATURE_DIR"
  --set "paths.signature_maps=$SIGNATURE_MAPS_JSON"
  --set "study1.targets.signature_manifest_path=$SIGNATURE_MANIFEST"
  --set "study1.targets.signature_provenance.NPS.path=$NPS_MAP"
  --set "study1.targets.signature_provenance.SIIPS1.path=$SIIPS1_MAP"
  --set "study1.targets.signature_scoring_mask_path=$SIGNATURE_MASK"
  --set "study1.outputs.root_name=$STUDY1_RUN_ID"
)

"$EEG_PIPELINE" signature-prediction prepare-targets \
  "${SUBJECT_ARGS[@]}" \
  "${COMMON_ARGS[@]}"

"$EEG_PIPELINE" signature-prediction prepare-features \
  "${SUBJECT_ARGS[@]}" \
  "${COMMON_ARGS[@]}"

"$PYTHON" -m studies.pain_study.study1.figures.plot_sensor_power_topographies \
  --config eeg_pipeline/utils/config/eeg_config.yaml \
  --study1-config "$STUDY1_CONFIG" \
  --deriv-root "$DERIV_ROOT" \
  --task "$TASK" \
  --output "$DERIV_ROOT/group/multimodal/$STUDY1_RUN_ID/reports/figures/supplementary/validity/sensor_power_topographies.svg"

"$PYTHON" -m studies.pain_study.study1.figures.plot_signature_power_topographies \
  --config eeg_pipeline/utils/config/eeg_config.yaml \
  --study1-config "$STUDY1_CONFIG" \
  --deriv-root "$DERIV_ROOT" \
  --task "$TASK" \
  --output "$DERIV_ROOT/group/multimodal/$STUDY1_RUN_ID/reports/figures/supplementary/validity/signature_power_topographies.svg"

"$EEG_PIPELINE" signature-prediction feature-benchmark \
  "${SUBJECT_ARGS[@]}" \
  "${COMMON_ARGS[@]}"

"$EEG_PIPELINE" signature-prediction report \
  "${SUBJECT_ARGS[@]}" \
  "${COMMON_ARGS[@]}"

"$PYTHON" -m studies.pain_study.study1.figures.plot_primary_prediction \
  --config eeg_pipeline/utils/config/eeg_config.yaml \
  --study1-config "$STUDY1_CONFIG" \
  --report "$DERIV_ROOT/group/multimodal/$STUDY1_RUN_ID/reports/study1_report.tsv" \
  --output "$DERIV_ROOT/group/multimodal/$STUDY1_RUN_ID/reports/figures/supplementary/validity/primary_prediction_estimation.svg"

"$PYTHON" -m studies.pain_study.study1.figures.plot_spectral_specificity \
  --config eeg_pipeline/utils/config/eeg_config.yaml \
  --study1-config "$STUDY1_CONFIG" \
  --report "$DERIV_ROOT/group/multimodal/$STUDY1_RUN_ID/reports/study1_report.tsv" \
  --output "$DERIV_ROOT/group/multimodal/$STUDY1_RUN_ID/reports/figures/supplementary/validity/spectral_specificity.svg"

"$PYTHON" -m studies.pain_study.study1.figures.plot_power_construct_validity \
  --config eeg_pipeline/utils/config/eeg_config.yaml \
  --study1-config "$STUDY1_CONFIG" \
  --deriv-root "$DERIV_ROOT" \
  --task "$TASK" \
  --output "$DERIV_ROOT/group/multimodal/$STUDY1_RUN_ID/reports/figures/supplementary/validity/power_construct_validity.svg"

"$PYTHON" -m studies.pain_study.study1.figures.plot_band_power_epoch_evolution \
  --config eeg_pipeline/utils/config/eeg_config.yaml \
  --study1-config "$STUDY1_CONFIG" \
  --deriv-root "$DERIV_ROOT" \
  --task "$TASK" \
  --output "$DERIV_ROOT/group/multimodal/$STUDY1_RUN_ID/reports/figures/supplementary/validity/band_power_epoch_evolution.svg"

"$PYTHON" -m studies.pain_study.study1.figures.plot_band_time_frequency \
  --config eeg_pipeline/utils/config/eeg_config.yaml \
  --study1-config "$STUDY1_CONFIG" \
  --derivative-root "$DERIV_ROOT" \
  --task "$TASK" \
  --output-dir "$DERIV_ROOT/group/multimodal/$STUDY1_RUN_ID/reports/figures/supplementary/validity/band_time_frequency"

"$PYTHON" -m studies.pain_study.study1.figures.plot_fmri_construct_validity \
  --config eeg_pipeline/utils/config/eeg_config.yaml \
  --study1-config "$STUDY1_CONFIG" \
  --deriv-root "$DERIV_ROOT" \
  --task "$TASK" \
  --output "$DERIV_ROOT/group/multimodal/$STUDY1_RUN_ID/reports/figures/supplementary/validity/fmri_construct_validity.svg"

"$PYTHON" -m studies.pain_study.study1.figures.plot_temporal_specificity \
  --config eeg_pipeline/utils/config/eeg_config.yaml \
  --study1-config "$STUDY1_CONFIG" \
  --report "$DERIV_ROOT/group/multimodal/$STUDY1_RUN_ID/reports/study1_report.tsv" \
  --output "$DERIV_ROOT/group/multimodal/$STUDY1_RUN_ID/reports/figures/supplementary/validity/temporal_specificity.svg"

"$PYTHON" -m studies.pain_study.study1.figures.plot_scanner_harmonic_spectrum \
  --config eeg_pipeline/utils/config/eeg_config.yaml \
  --study1-config "$STUDY1_CONFIG" \
  --task "$TASK" \
  --derivative-root "$DERIV_ROOT/preprocessed/eeg" \
  --output "$DERIV_ROOT/group/multimodal/$STUDY1_RUN_ID/reports/figures/supplementary/validity/scanner_harmonic_spectrum.svg"

"$PYTHON" -m studies.pain_study.study1.figures.plot_cohort_power_spectral_density \
  --config eeg_pipeline/utils/config/eeg_config.yaml \
  --study1-config "$STUDY1_CONFIG" \
  --task "$TASK" \
  --derivative-root "$DERIV_ROOT/preprocessed/eeg" \
  --output "$DERIV_ROOT/group/multimodal/$STUDY1_RUN_ID/reports/figures/supplementary/validity/cohort_power_spectral_density.svg"

"$PYTHON" -m studies.pain_study.study1.figures.plot_preprocessing_stage_power_spectral_density \
  --config eeg_pipeline/utils/config/eeg_config.yaml \
  --study1-config "$STUDY1_CONFIG" \
  --task "$TASK" \
  --source-data-root "$SOURCE_DATA_ROOT" \
  --derivative-root "$DERIV_ROOT/preprocessed/eeg" \
  --stage raw \
  --stage processed \
  --stage mne \
  --output-dir "$DERIV_ROOT/group/multimodal/$STUDY1_RUN_ID/reports/figures/supplementary/validity"
```

The main outputs are:

```text
$DERIV_ROOT/group/multimodal/$STUDY1_RUN_ID/targets/primary_targets.parquet
$DERIV_ROOT/group/multimodal/$STUDY1_RUN_ID/reports/study1_report.tsv
$DERIV_ROOT/group/multimodal/$STUDY1_RUN_ID/reports/article_tables/
$DERIV_ROOT/group/multimodal/$STUDY1_RUN_ID/reports/full_picture/
$DERIV_ROOT/group/multimodal/$STUDY1_RUN_ID/reports/figures/supplementary/validity/band_power_epoch_evolution.svg
$DERIV_ROOT/group/multimodal/$STUDY1_RUN_ID/reports/figures/supplementary/validity/band_power_epoch_evolution_by_subject.tsv
$DERIV_ROOT/group/multimodal/$STUDY1_RUN_ID/reports/figures/supplementary/validity/band_power_epoch_evolution_summary.tsv
$DERIV_ROOT/group/multimodal/$STUDY1_RUN_ID/reports/figures/supplementary/validity/band_time_frequency/
$DERIV_ROOT/group/multimodal/$STUDY1_RUN_ID/reports/figures/supplementary/validity/behavioral_dose_response.svg
$DERIV_ROOT/group/multimodal/$STUDY1_RUN_ID/reports/figures/supplementary/validity/cohort_power_spectral_density.svg
$DERIV_ROOT/group/multimodal/$STUDY1_RUN_ID/reports/figures/supplementary/validity/cohort_power_spectral_density_by_run.tsv
$DERIV_ROOT/group/multimodal/$STUDY1_RUN_ID/reports/figures/supplementary/validity/cohort_power_spectral_density_by_subject.tsv
$DERIV_ROOT/group/multimodal/$STUDY1_RUN_ID/reports/figures/supplementary/validity/cohort_power_spectral_density_summary.tsv
$DERIV_ROOT/group/multimodal/$STUDY1_RUN_ID/reports/figures/supplementary/validity/cohort_power_spectral_density_raw.svg
$DERIV_ROOT/group/multimodal/$STUDY1_RUN_ID/reports/figures/supplementary/validity/cohort_power_spectral_density_processed.svg
$DERIV_ROOT/group/multimodal/$STUDY1_RUN_ID/reports/figures/supplementary/validity/cohort_power_spectral_density_mne.svg
$DERIV_ROOT/group/multimodal/$STUDY1_RUN_ID/reports/figures/supplementary/validity/fmri_construct_validity.svg
$DERIV_ROOT/group/multimodal/$STUDY1_RUN_ID/reports/figures/supplementary/validity/fmri_construct_validity_provenance.json
$DERIV_ROOT/group/multimodal/$STUDY1_RUN_ID/reports/figures/supplementary/validity/nps_behavioral_validity.svg
$DERIV_ROOT/group/multimodal/$STUDY1_RUN_ID/reports/figures/supplementary/validity/nps_dose_response.svg
$DERIV_ROOT/group/multimodal/$STUDY1_RUN_ID/reports/figures/supplementary/validity/primary_prediction_estimation.svg
$DERIV_ROOT/group/multimodal/$STUDY1_RUN_ID/reports/figures/supplementary/validity/primary_prediction_by_subject.tsv
$DERIV_ROOT/group/multimodal/$STUDY1_RUN_ID/reports/figures/supplementary/validity/primary_prediction_summary.tsv
$DERIV_ROOT/group/multimodal/$STUDY1_RUN_ID/reports/figures/supplementary/validity/power_construct_validity.svg
$DERIV_ROOT/group/multimodal/$STUDY1_RUN_ID/reports/figures/supplementary/validity/scanner_harmonic_spectrum.svg
$DERIV_ROOT/group/multimodal/$STUDY1_RUN_ID/reports/figures/supplementary/validity/scanner_harmonic_spectrum_by_run.tsv
$DERIV_ROOT/group/multimodal/$STUDY1_RUN_ID/reports/figures/supplementary/validity/scanner_harmonic_spectrum_by_subject.tsv
$DERIV_ROOT/group/multimodal/$STUDY1_RUN_ID/reports/figures/supplementary/validity/sensor_power_topographies.svg
$DERIV_ROOT/group/multimodal/$STUDY1_RUN_ID/reports/figures/supplementary/validity/signature_power_topographies.svg
$DERIV_ROOT/group/multimodal/$STUDY1_RUN_ID/reports/figures/supplementary/validity/siips1_behavioral_validity.svg
$DERIV_ROOT/group/multimodal/$STUDY1_RUN_ID/reports/figures/supplementary/validity/siips1_dose_response.svg
$DERIV_ROOT/group/multimodal/$STUDY1_RUN_ID/reports/figures/supplementary/validity/spectral_specificity.svg
$DERIV_ROOT/group/multimodal/$STUDY1_RUN_ID/reports/figures/supplementary/validity/spectral_specificity_by_subject.tsv
$DERIV_ROOT/group/multimodal/$STUDY1_RUN_ID/reports/figures/supplementary/validity/spectral_specificity_summary.tsv
$DERIV_ROOT/group/multimodal/$STUDY1_RUN_ID/reports/figures/supplementary/validity/temporal_specificity.svg
$DERIV_ROOT/group/multimodal/$STUDY1_RUN_ID/reports/figures/supplementary/validity/temporal_specificity_by_subject.tsv
$DERIV_ROOT/group/multimodal/$STUDY1_RUN_ID/reports/figures/supplementary/validity/temporal_specificity_summary.tsv
$DERIV_ROOT/group/multimodal/$STUDY1_RUN_ID/reports/full_picture/behavior_signature_validity_by_subject.tsv
$DERIV_ROOT/group/multimodal/$STUDY1_RUN_ID/reports/full_picture/behavior_signature_validity_summary.tsv
```

Each sensor-topography prefix (`sensor_power_topographies` and
`signature_power_topographies`) publishes this exact core family in the same directory:

```text
<prefix>.svg
<prefix>.png
<prefix>_by_subject.tsv
<prefix>_by_subject.parquet
<prefix>_sensors.tsv
<prefix>_sensors.parquet
<prefix>_clusters.tsv
<prefix>_family.tsv
<prefix>_caption.txt
<prefix>_manifest.json
```

The construct prefix `sensor_power_topographies` additionally publishes the complementary
Fp1/Fp2 channel-scope audit:

```text
sensor_power_topographies_sensitivity_by_subject.tsv
sensor_power_topographies_sensitivity_by_subject.parquet
sensor_power_topographies_sensitivity_summary.tsv
sensor_power_topographies_sensitivity_summary.parquet
```

The SVG and 600-dpi PNG are the two figure formats. The by-subject files contain the participant
estimands used for inference; the sensor and cluster files contain unthresholded cohort values,
one-sample t statistics, cluster membership, and corrected p-values. The family table records the
joint sign-flip procedure, and the manifest records source and output checksums. Publication is
atomic: an invalid or incomplete family is not promoted.

Run `report` again with sensitivity roots when you want one article-ready comparison table across
previous runs:

```bash
SENSITIVITY_OUTPUT_ROOTS='[
  {"label":"reference_minus2s","root_name":"study1_reference_minus2s_20260524"},
  {"label":"reference_minus0p2s","root_name":"study1_reference_minus0p2s_20260524"},
  {
    "label":"raw_active_reference_covariate",
    "root_name":"study1_raw_active_reference_covariate_20260524"
  }
]'

COMMON_ARGS+=(
  --set "study1.reporting.sensitivity_output_roots=$SENSITIVITY_OUTPUT_ROOTS"
)

"$EEG_PIPELINE" signature-prediction report \
  "${SUBJECT_ARGS[@]}" \
  "${COMMON_ARGS[@]}"
```

That writes:

```text
$DERIV_ROOT/group/multimodal/$STUDY1_RUN_ID/reports/full_picture/configured_sensitivity_model_summary.tsv
$DERIV_ROOT/group/multimodal/$STUDY1_RUN_ID/reports/full_picture/primary_sensitivity_comparison.tsv
```

### Configs

Use the production config for formal analysis:

```bash
STUDY1_CONFIG="studies/pain_study/study1/config/study1_config.yaml"
```

Use the smoke-test config only to verify data plumbing on a small local cohort:

```bash
STUDY1_CONFIG="studies/pain_study/study1/config/study1_smoketest.yaml"
```

The smoke config lowers the subject minimum, permutations, and deep-regression epochs. It is not
eligible for scientific interpretation.

### Paths

Set the study roots explicitly. The Kingston layout used during local testing is:

```bash
EEG_BIDS_ROOT="/Volumes/KINGSTON/EEG_fMRI_data/bids_output/eeg"
FMRI_BIDS_ROOT="/Volumes/KINGSTON/EEG_fMRI_data/bids_output/fmri"
SOURCE_DATA_ROOT="/Volumes/KINGSTON/EEG_fMRI_data/source_data"
DERIV_ROOT="/Volumes/KINGSTON/EEG_fMRI_data/derivatives"
SIGNATURE_DIR="/Volumes/KINGSTON/EEG_fMRI_data/external"
TASK="thermalactive"
NPS_MAP="NPS/weights_NSF_grouppred_cvpcr.nii.gz"
SIIPS1_MAP="SIIPS1/nonnoc_v11_4_137subjmap_weighted_mean.nii.gz"
SIGNATURE_MANIFEST="signature_manifest.yaml"
SIGNATURE_MASK="tpl-MNI152NLin2009cAsym_res-02_desc-brain_mask.nii.gz"
SIGNATURE_MAPS_JSON='[
  {"name":"NPS","path":"NPS/weights_NSF_grouppred_cvpcr.nii.gz"},
  {"name":"SIIPS1","path":"SIIPS1/nonnoc_v11_4_137subjmap_weighted_mean.nii.gz"}
]'
```

Keep the signature provenance paths relative to `SIGNATURE_DIR`. The Study 1 config loader resolves
YAML path-looking strings relative to the config file, and fMRI signature discovery requires
`paths.signature_maps`, so these overrides are required when running from an external signature
directory:

```bash
COMMON_ARGS=(
  --task "$TASK"
  --study1-config "$STUDY1_CONFIG"
  --bids-root "$EEG_BIDS_ROOT"
  --bids-fmri-root "$FMRI_BIDS_ROOT"
  --deriv-root "$DERIV_ROOT"
  --set "paths.signature_dir=$SIGNATURE_DIR"
  --set "paths.signature_maps=$SIGNATURE_MAPS_JSON"
  --set "study1.targets.signature_manifest_path=$SIGNATURE_MANIFEST"
  --set "study1.targets.signature_provenance.NPS.path=$NPS_MAP"
  --set "study1.targets.signature_provenance.SIIPS1.path=$SIIPS1_MAP"
  --set "study1.targets.signature_scoring_mask_path=$SIGNATURE_MASK"
)
```

If the frozen signature manifest is missing or the signature files changed, regenerate it before
target preparation:

```bash
"$PYTHON" -m studies.pain_study.study1.signature_manifest \
  "$SIGNATURE_DIR" \
  "$SIGNATURE_DIR/$SIGNATURE_MANIFEST"
```

### Preprocessing Prerequisite

Study 1 target alignment requires an explicit task `run` column plus original `trial_number` or
`trial_index`. Production nuisance regression requires
HRF-weighted artifact columns
`hrf_weighted_framewise_displacement`, `hrf_weighted_std_dvars`, and
`hrf_weighted_fp1_fp2_high_frequency_power`, plus `residual_ecg_coupling`. The raw Fp1/Fp2
artifact proxy remains `fp1_fp2_high_frequency_power`; it is an upstream input to the HRF-weighted
covariate and artifact-censoring audits, not the Level 2 nuisance column.

The smoke-test config uses the same Level 2 nuisance columns required by the report article tables:
`run`, `onset`, `within_run_trial`, `hrf_weighted_framewise_displacement`,
`hrf_weighted_std_dvars`, `hrf_weighted_fp1_fp2_high_frequency_power`,
`residual_ecg_coupling`, `stimulus_temp`, and `selected_surface`. This keeps the smoke run
scientifically aligned with the production estimand while reducing permutation counts and training
duration.

For a formal rerun, prefer a new output root rather than mixing outputs from different configs or
dates:

```bash
STUDY1_RUN_ID="study1_$(date +%Y%m%d)"
COMMON_ARGS+=(
  --set "study1.outputs.root_name=$STUDY1_RUN_ID"
)
```

Outputs are written to:

```text
$DERIV_ROOT/group/multimodal/$STUDY1_RUN_ID/
```

The examples below refer to that directory as:

```bash
STUDY1_ROOT="$DERIV_ROOT/group/multimodal/$STUDY1_RUN_ID"
```

### Subject Selection

Run production analyses only after at least 30 analyzable subjects are available. Use explicit
subject arguments when auditing a known cohort:

```bash
SUBJECT_ARGS=(
  --subject 0000
  --subject 0001
)
```

Use all discoverable subjects only when the BIDS and derivative roots are already curated:

```bash
SUBJECT_ARGS=(--all-subjects)
```

For `prepare-targets`, Study 1 filters `--all-subjects` to subjects with task fMRI `func`
directories and logs EEG-only exclusions. Later stages intentionally resolve their analysis cohort
from the prepared primary target table, so rerun `prepare-targets` after changing the intended
multimodal subject set.

### Audit Checkpoints

Treat the production run as a sequence of decision points, not just a command chain. After target
preparation, inspect signature provenance, scoring-mask support, LSS estimability, retained
subjects, and retained plateau-trial counts before investing in feature extraction or permutation
testing. Missing provenance, invalid signature support, unstable LSS designs, or inadequate retained
trials should be fixed at the derivative or event-log level rather than compensated for downstream.

The final report interprets results in this order: analysis validity, the prespecified `NPS` /
`alpha_beta_gamma` / `elasticnet` primary cell, interpretation diagnostics, and
secondary/exploratory analyses. Secondary benchmark cells and exploratory models are reported after
the primary cell.

### Pipeline Stages

Run stages in this order.

1. Prepare trial-wise fMRI signature targets:

   ```bash
   "$EEG_PIPELINE" signature-prediction prepare-targets \
     "${SUBJECT_ARGS[@]}" \
     "${COMMON_ARGS[@]}"
   ```

   Required output:

   ```text
   $STUDY1_ROOT/targets/primary_targets.parquet
   ```

   Check this output before continuing. A target table with too few retained subjects, sparse
   plateau trials, invalid signature provenance, or unresolved LSS design failures leaves the
   primary estimand unevaluable even if the feature benchmark runs.

2. Prepare Study 1 trial-safe EEG feature tables:

   ```bash
   "$EEG_PIPELINE" signature-prediction prepare-features \
     "${SUBJECT_ARGS[@]}" \
     "${COMMON_ARGS[@]}"
   ```

   Required per-subject output:

   ```text
   $STUDY1_ROOT/features_trial_ml_safe/sub-*/eeg/features/power/features_power.parquet
   $STUDY1_ROOT/features_temporal_controls/sub-*/eeg/features/power/features_power.parquet
   ```

   The temporal-control table contains unbaselined alpha, beta, and gamma log-power features for
   the configured pre-stimulus negative-control windows, the pre-plateau ramp-up wrong-lag window,
   and early/mid/late plateau-sensitivity windows. Plateau-sensitivity windows are constrained to
   `3.0` to `10.5` s and do not include ramp-down. Exploratory families produce additional
   per-family tables only when explicitly enabled.

   The primary reference window is `-5.0` to `-0.01` s. Reference-window sensitivity runs should
   rerun `prepare-features` and `feature-benchmark` under separate `study1.outputs.root_name`
   values after setting `time_frequency_analysis.baseline_window` to `[-2.0, -0.01]` or
   `[-0.2, -0.01]`. The raw-active-power sensitivity is a separate sibling run with active
   log-power features left unnormalized and `-5.0` to `-0.01` s reference power included as a
   covariate.

   Once both required outputs from stages 1 and 2 exist, generate the two standalone sensor-space
   families. No benchmark or report output is read by these commands:

   ```bash
   "$PYTHON" -m studies.pain_study.study1.figures.plot_sensor_power_topographies \
     --config eeg_pipeline/utils/config/eeg_config.yaml \
     --study1-config "$STUDY1_CONFIG" \
     --deriv-root "$DERIV_ROOT" \
     --task "$TASK" \
     --output "$STUDY1_ROOT/reports/figures/supplementary/validity/sensor_power_topographies.svg"

   "$PYTHON" -m studies.pain_study.study1.figures.plot_signature_power_topographies \
     --config eeg_pipeline/utils/config/eeg_config.yaml \
     --study1-config "$STUDY1_CONFIG" \
     --deriv-root "$DERIV_ROOT" \
     --task "$TASK" \
     --output "$STUDY1_ROOT/reports/figures/supplementary/validity/signature_power_topographies.svg"
   ```

   Both commands reconstruct channel-level active-versus-baseline dB power from the trial-ML-safe
   power features and align it exactly to retained clean events and the prepared target table.
   `sensor_power_topographies.svg` contains temperature-slope and subjective-intensity rows;
   `signature_power_topographies.svg` contains NPS and SIIPS1 partial-correlation rows. Columns are
   alpha, beta, low gamma, mid gamma, and high gamma. Unthresholded maps remain visible, and dark
   rings identify sensors in clusters surviving the separate joint 10-map family correction for
   that figure.

3. Run the feature benchmark:

   ```bash
   "$EEG_PIPELINE" signature-prediction feature-benchmark \
     "${SUBJECT_ARGS[@]}" \
     "${COMMON_ARGS[@]}"
   ```

   Each required primary-partition benchmark cell writes `metrics/model_comparison_summary.json`
   under:

   ```text
   $STUDY1_ROOT/feature_benchmark/primary/<target>/<preset>/model_comparison/
   ```

   Temporal-control cells write the same summary under:

   ```text
   $STUDY1_ROOT/feature_benchmark/temporal_control/<target>/temporal_<window>/model_comparison/
   ```

   Required benchmark targets are `NPS` and `SIIPS1`. The report validates the full primary
   benchmark preset set: `alpha`, `beta`, `gamma`, `alpha_beta`, and `alpha_beta_gamma`. The
   prespecified inferential primary cell is `NPS` / `alpha_beta_gamma` / `elasticnet`; the other
   required cells provide the alpha/beta/gamma frequency audit. Delta, theta, delta+theta, and
   all-band models are written under the exploratory partition. The benchmark filters predictors
   to active-window, individual-channel, log-ratio power columns and excludes Fp1/Fp2. The
   temporal-control benchmark uses individual-channel alpha, beta, and gamma raw log-power columns
   from the temporal-control feature root. Only the pre-stimulus and pre-plateau wrong-lag rows feed
   the `temporal_negative_controls_passed` verdict; plateau rows are response-period sensitivity
   analyses.

The default Study 1 config enables the theoretically prioritized exploratory feature families:

```text
[
  "aperiodic",
  "erds",
  "spectral",
  "bursts"
]
```

4. Run exploratory deep regression only when the thesis report should include that lane:

   ```bash
   "$EEG_PIPELINE" signature-prediction deep-regression \
     "${SUBJECT_ARGS[@]}" \
     "${COMMON_ARGS[@]}"
   ```

5. Write the Study 1 report:

   ```bash
   "$EEG_PIPELINE" signature-prediction report \
     "${SUBJECT_ARGS[@]}" \
     "${COMMON_ARGS[@]}"
   ```

   The report stage validates that all prespecified feature-benchmark outputs exist and contain the
   protocol audit fields: confidence intervals, valid permutation counts, invalid permutation
   attempts, subject-selection counts, fold-level best hyperparameters, analysis-validity status,
   primary prediction status, interpretation diagnostics, and interpretation flags.
   Temporal-control feature-benchmark summaries are included with separate Holm-corrected p-values.
   Deep-regression and exploratory feature-benchmark summaries are included when present, but they
   are not required for the primary Study 1 report.
   The report also writes `reports/full_picture/`, which contains model leaderboards,
   the target-validity gate, target-by-temperature summaries, subject-by-temperature summaries,
   behavioral construct-validity coefficients, and a manifest. It writes the behavioral, NPS, and
   SIIPS1 dose-response figures plus the NPS and SIIPS1 behavioral-validity coefficient plots as
   separate editable SVGs under `reports/figures/supplementary/validity/`; the manifest records
   their paths.
   Generate the primary held-out prediction estimation figure from the completed report:

   ```bash
   "$PYTHON" -m studies.pain_study.study1.figures.plot_primary_prediction \
     --config eeg_pipeline/utils/config/eeg_config.yaml \
     --study1-config "$STUDY1_CONFIG" \
     --report "$DERIV_ROOT/group/multimodal/$STUDY1_RUN_ID/reports/study1_report.tsv" \
     --output "$DERIV_ROOT/group/multimodal/$STUDY1_RUN_ID/reports/figures/supplementary/validity/primary_prediction_estimation.svg"
   ```

   This command plots participants as the statistical unit, verifies fold-level nuisance-only,
   nuisance+EEG, and incremental $R^2$ against the report, and writes subject-level and
   cohort-level TSV/parquet audits beside the editable SVG. Negative held-out $R^2$ values are
   expected when prediction performs worse than the held-out-target mean benchmark and are not
   clipped.

   Generate the prespecified spectral-specificity estimation figure from the same report:

   ```bash
   "$PYTHON" -m studies.pain_study.study1.figures.plot_spectral_specificity \
     --config eeg_pipeline/utils/config/eeg_config.yaml \
     --study1-config "$STUDY1_CONFIG" \
     --report "$DERIV_ROOT/group/multimodal/$STUDY1_RUN_ID/reports/study1_report.tsv" \
     --output "$DERIV_ROOT/group/multimodal/$STUDY1_RUN_ID/reports/figures/supplementary/validity/spectral_specificity.svg"
   ```

   This command includes only ElasticNet alpha, beta, gamma, alpha+beta, and
   alpha+beta+gamma cells. It verifies every participant effect and cohort mean
   against the fold tables and writes subject-level and cohort-level TSV/parquet audits. Ridge and
   exploratory low-frequency/all-band models remain tabular.

   Generate the EEG power construct-validity figure from the retained target cohort and
   trial-ML-safe power tables:

   ```bash
   "$PYTHON" -m studies.pain_study.study1.figures.plot_power_construct_validity \
     --config eeg_pipeline/utils/config/eeg_config.yaml \
     --study1-config "$STUDY1_CONFIG" \
     --deriv-root "$DERIV_ROOT" \
     --task "$TASK" \
     --output "$DERIV_ROOT/group/multimodal/$STUDY1_RUN_ID/reports/figures/supplementary/validity/power_construct_validity.svg"
   ```

   This command requires alpha, beta, and all three gamma features. It writes the
   editable SVG and seven TSV/parquet audit pairs covering trial power, participant and cohort
   temperature estimates, participant and cohort rating estimates, and the complementary Fp1/Fp2
   sensitivity. The primary figure includes Fp1/Fp2 when
   `study1.figures.power_construct_validity.channels.include_fp1_fp2` is true. The alternative
   channel scope is computed without selecting results by appearance. Its five aligned temperature
   rows use a shared scale and a simultaneous 95% participant-bootstrap band across the 30
   prespecified band-temperature cells. The adjacent adjusted-intensity forest plot uses pointwise
   95% participant-bootstrap intervals and reports its estimable participant count for each band.

   Generate the time-resolved global band-power figure from the retained target cohort and clean
   epochs:

   ```bash
   "$PYTHON" -m studies.pain_study.study1.figures.plot_band_power_epoch_evolution \
     --config eeg_pipeline/utils/config/eeg_config.yaml \
     --study1-config "$STUDY1_CONFIG" \
     --deriv-root "$DERIV_ROOT" \
     --task "$TASK" \
     --output "$DERIV_ROOT/group/multimodal/$STUDY1_RUN_ID/reports/figures/supplementary/validity/band_power_epoch_evolution.svg"
   ```

   The five vertically aligned panels show alpha, beta, and the three gamma sub-bands
   from −5 to 14.5 s. The final 0.5 s is omitted to limit right-edge Morlet convolution effects. Each
   thin line is one participant's retained-trial mean; the colored curve is the equally weighted
   cohort mean with a pointwise 95% percentile interval obtained by resampling participants as
   complete trajectories. Linear wavelet power is frequency-weighted and averaged across
   non-Fp1/Fp2 EEG channels before conversion to dB relative to the complete −5.0 to −0.01 s
   baseline. A direct protocol bar marks the five epoch intervals, and the header reports cohort,
   retained-trial, and channel counts. Display downsampling occurs only after normalization. The
   writer adds participant- and cohort-level TSV/parquet audits beside the editable SVG.

   Generate the frequency-resolved Hanning TFR family from every participant's newest final-clean
   epoch file:

   ```bash
   "$PYTHON" -m studies.pain_study.study1.figures.plot_band_time_frequency \
     --config eeg_pipeline/utils/config/eeg_config.yaml \
     --study1-config "$STUDY1_CONFIG" \
     --derivative-root "$DERIV_ROOT" \
     --task "$TASK" \
     --output-dir "$DERIV_ROOT/group/multimodal/$STUDY1_RUN_ID/reports/figures/supplementary/validity/band_time_frequency"
   ```

   The writer produces one participant SVG per band and one equally weighted cohort SVG per band.
   It follows the FieldTrip `mtmconvol` convention with one symmetric Hanning taper,
   frequency-dependent 7-cycle windows, 0.05 s steps, trial-wide DC removal, and 1 Hz resolution.
   Trial/channel/frequency power is converted to dB using complete Hanning windows inside the −5.0
   to −0.01 s baseline, after which all EEG channels are averaged. Participant maps show the
   delivered-temperature OLS slope in dB/°C adjusted for run, thermode surface, and within-run trial
   order; cohort maps average participant slopes equally. Trials lacking required model metadata are
   excluded explicitly and reconciled in the source audit. Participant maps share a robust per-band
   family scale, while cohort maps use a cohort-specific robust scale. TSV/parquet audits retain the
   exact slopes, matched source/event files, timestamps, trial reconciliation, design diagnostics,
   sampling frequencies, and channel lists.

   Generate the whole-brain fMRI construct-validity figure directly from the retained cohort,
   current clean events, and MNI-space fMRIPrep runs:

   ```bash
   "$PYTHON" -m studies.pain_study.study1.figures.plot_fmri_construct_validity \
     --config eeg_pipeline/utils/config/eeg_config.yaml \
     --study1-config "$STUDY1_CONFIG" \
     --deriv-root "$DERIV_ROOT" \
     --task "$TASK" \
     --output "$DERIV_ROOT/group/multimodal/$STUDY1_RUN_ID/reports/figures/supplementary/validity/fmri_construct_validity.svg"
   ```

   The command fits separate temperature and rating first-level GLMs so rating does not alter the
   total temperature estimand. The rating regressor is centered within participant and delivered
   temperature; temperature is represented categorically in that model. Group inference uses the
   participant effect maps, 10,000 deterministic two-sided sign-flipping permutations, and
   voxelwise max-T FWE correction. The SVG displays unthresholded mean effects with corrected
   significance outlines and fixed surface/axial views. Participant and group NIfTI maps, design,
   subject and peak TSV/parquet audits, plus a checksum provenance JSON are written beside it.

   Generate the temporal-specificity figure separately from the completed current-protocol report:

   ```bash
   "$PYTHON" -m studies.pain_study.study1.figures.plot_temporal_specificity \
     --config eeg_pipeline/utils/config/eeg_config.yaml \
     --study1-config "$STUDY1_CONFIG" \
     --report "$DERIV_ROOT/group/multimodal/$STUDY1_RUN_ID/reports/study1_report.tsv" \
     --output "$DERIV_ROOT/group/multimodal/$STUDY1_RUN_ID/reports/figures/supplementary/validity/temporal_specificity.svg"
   ```

   This command requires all six configured temporal-control windows exactly once for NPS and
   SIIPS1, verifies every cohort mean against its held-out-subject fold table, and writes
   subject-level and cohort-level TSV/parquet audits beside the editable SVG. Legacy temporal
   window names fail validation and produce no figure.

   Generate the cohort PSD from final-clean continuous EEG:

   ```bash
   "$PYTHON" -m studies.pain_study.study1.figures.plot_cohort_power_spectral_density \
     --config eeg_pipeline/utils/config/eeg_config.yaml \
     --study1-config "$STUDY1_CONFIG" \
     --task "$TASK" \
     --derivative-root "$DERIV_ROOT/preprocessed/eeg" \
     --output "$DERIV_ROOT/group/multimodal/$STUDY1_RUN_ID/reports/figures/supplementary/validity/cohort_power_spectral_density.svg"
   ```

   Run spectra are aggregated within participants as medians in linear power before conversion to
   dB. The cohort curve is the participant median, with a pointwise 95% participant-bootstrap
   interval. The command writes run-, participant-, and cohort-level TSV/parquet audits beside the
   SVG and fails on incomplete frequency grids, non-finite spectra, invalid intervals, or
   inconsistent cohort counts.

   Generate descriptive PSD checks at the raw BrainVision, BrainVision-processed, and final MNE
   checkpoints:

   ```bash
   "$PYTHON" -m studies.pain_study.study1.figures.plot_preprocessing_stage_power_spectral_density \
     --config eeg_pipeline/utils/config/eeg_config.yaml \
     --study1-config "$STUDY1_CONFIG" \
     --task "$TASK" \
     --source-data-root "$SOURCE_DATA_ROOT" \
     --derivative-root "$DERIV_ROOT/preprocessed/eeg" \
     --stage raw \
     --stage processed \
     --stage mne \
     --output-dir "$DERIV_ROOT/group/multimodal/$STUDY1_RUN_ID/reports/figures/supplementary/validity"
   ```

   Each stage is written as a separate SVG with run-, participant-, and cohort-level TSV/parquet
   audits. Headers state the exact checkpoint, source sampling frequency, and common 16.384 s
   Welch duration with 50% overlap. These are descriptive QC views; compare stages inferentially
   only after matching their participant-run inputs.

   Generate the scanner-harmonic spectrum separately after the report because it reads continuous
   final-clean EEG rather than report tables:

   ```bash
   "$PYTHON" -m studies.pain_study.study1.figures.plot_scanner_harmonic_spectrum \
     --config eeg_pipeline/utils/config/eeg_config.yaml \
     --study1-config "$STUDY1_CONFIG" \
     --deriv-root "$DERIV_ROOT" \
     --task "$TASK" \
     --derivative-root "$DERIV_ROOT/preprocessed/eeg" \
     --output "$DERIV_ROOT/group/multimodal/$STUDY1_RUN_ID/reports/figures/supplementary/validity/scanner_harmonic_spectrum.svg"
   ```

   This standalone command writes the editable SVG plus run- and participant-level TSV/parquet
   audits in the same directory. Numbered final-clean participants are included; pilot-style
   identifiers and the prespecified `sub-0006` pilot are excluded before any outcome is read.
   To compare sibling sensitivity runs in the same bundle, set
   `study1.reporting.sensitivity_output_roots` to a list of `{label, root_name}` entries before
   running `report`.

### Smoke Test on Kingston

The following smoke test checks the Kingston data path with two subjects and reduced permutations:

```bash
cd /Users/joduq24/Desktop/EEG_fMRI_Pipeline

EEG_PIPELINE=".venv/bin/eeg-pipeline"
PYTHON=".venv/bin/python"
STUDY1_CONFIG="studies/pain_study/study1/config/study1_smoketest.yaml"
STUDY1_RUN_ID="study1_smoke_$(date +%Y%m%d)"
EEG_BIDS_ROOT="/Volumes/KINGSTON/EEG_fMRI_data/bids_output/eeg"
FMRI_BIDS_ROOT="/Volumes/KINGSTON/EEG_fMRI_data/bids_output/fmri"
DERIV_ROOT="/Volumes/KINGSTON/EEG_fMRI_data/derivatives"
SIGNATURE_DIR="/Volumes/KINGSTON/EEG_fMRI_data/external"
TASK="thermalactive"
NPS_MAP="NPS/weights_NSF_grouppred_cvpcr.nii.gz"
SIIPS1_MAP="SIIPS1/nonnoc_v11_4_137subjmap_weighted_mean.nii.gz"
SIGNATURE_MANIFEST="signature_manifest.yaml"
SIGNATURE_MASK="tpl-MNI152NLin2009cAsym_res-02_desc-brain_mask.nii.gz"
SIGNATURE_MAPS_JSON='[
  {"name":"NPS","path":"NPS/weights_NSF_grouppred_cvpcr.nii.gz"},
  {"name":"SIIPS1","path":"SIIPS1/nonnoc_v11_4_137subjmap_weighted_mean.nii.gz"}
]'
SUBJECT_ARGS=(--subject 0000 --subject 0001)
SMOKE_EXTRA_ARGS=()
COMMON_ARGS=(
  --task "$TASK"
  --study1-config "$STUDY1_CONFIG"
  --bids-root "$EEG_BIDS_ROOT"
  --bids-fmri-root "$FMRI_BIDS_ROOT"
  --deriv-root "$DERIV_ROOT"
  --set "paths.signature_dir=$SIGNATURE_DIR"
  --set "paths.signature_maps=$SIGNATURE_MAPS_JSON"
  --set "study1.targets.signature_manifest_path=$SIGNATURE_MANIFEST"
  --set "study1.targets.signature_provenance.NPS.path=$NPS_MAP"
  --set "study1.targets.signature_provenance.SIIPS1.path=$SIIPS1_MAP"
  --set "study1.targets.signature_scoring_mask_path=$SIGNATURE_MASK"
  "${SMOKE_EXTRA_ARGS[@]}"
  --set "study1.outputs.root_name=$STUDY1_RUN_ID"
)

"$EEG_PIPELINE" signature-prediction prepare-targets \
  "${SUBJECT_ARGS[@]}" \
  "${COMMON_ARGS[@]}"

"$EEG_PIPELINE" signature-prediction prepare-features \
  "${SUBJECT_ARGS[@]}" \
  "${COMMON_ARGS[@]}"

"$EEG_PIPELINE" signature-prediction feature-benchmark \
  "${SUBJECT_ARGS[@]}" \
  "${COMMON_ARGS[@]}"
```

Interpret smoke-test output only as a pipeline integrity check. With two subjects, each LOSO
training fold contains one subject, so subject-grouped inner cross-validation cannot tune
hyperparameters.

### Grouped Inner-CV Smoke Test

To exercise the 3-fold inner `GroupKFold` used by the smoke config, run at least four analyzable
subjects. Each LOSO outer fold then trains on three subject groups. In the smoke-test block above,
replace `SUBJECT_ARGS`, `STUDY1_RUN_ID`, and `SMOKE_EXTRA_ARGS` before constructing `COMMON_ARGS`:

```bash
SUBJECT_ARGS=(
  --subject 0000
  --subject 0001
  --subject pilot001
  --subject pilot002
)
STUDY1_RUN_ID="study1_groupcv_smoke_$(date +%Y%m%d)"
SMOKE_EXTRA_ARGS=(
  --set "study1.cohort.min_subjects=4"
  --set "study1.feature_benchmark.n_perm=1"
)
```

After `feature-benchmark`, confirm grouped inner CV ran by checking that fold-level
`best_params` are not `{}`:

```bash
METRICS_PATH="$DERIV_ROOT/group/multimodal/$STUDY1_RUN_ID"
METRICS_PATH="$METRICS_PATH/feature_benchmark/primary/NPS/alpha"
METRICS_PATH="$METRICS_PATH/model_comparison/metrics/model_comparison.tsv"
export METRICS_PATH
"$PYTHON" - <<'PY'
import os
import pandas as pd

df = pd.read_csv(os.environ["METRICS_PATH"], sep="\t")
print(df[["model", "fold", "test_subject", "best_params"]])
PY
```

### Output Hygiene

Do not mix smoke-test and production outputs in the same `study1.outputs.root_name`. Before reading
or aggregating results, check for stale summaries:

```bash
find "$DERIV_ROOT/group/multimodal" \
  -path "*/feature_benchmark/*/*/*/model_comparison/metrics/model_comparison_summary.json" \
  -exec stat -f "%Sm %N" -t "%Y-%m-%d %H:%M:%S" {} +
```

Check for AppleDouble sidecars copied from external drives:

```bash
find "$DERIV_ROOT/group/multimodal/$STUDY1_RUN_ID" -name "._*" -print
```

Sidecars and stale outputs should be removed or isolated before formal reporting.

### Troubleshooting Target Preparation

`prepare-targets` fails fast when the fMRI inputs are not valid for trial-wise signature
extraction. Two common blockers are:

- selected fMRIPrep confounds contain non-finite values in retained volumes; fMRIPrep
  motion-outlier/non-steady-state columns and the undefined initial derivative or
  framewise-displacement row are converted to a Nilearn `sample_mask`, but any non-finite nuisance
  value that remains in a retained volume is an error
- trial beta images contain non-finite voxels inside the analysis mask; non-finite background
  outside the mask is filled only to permit continuous resampling, but in-mask non-finite values
  require repairing the image or mask

Do not zero-fill retained nuisance regressors or mask away non-finite voxels inside Study 1 as a
workaround.
