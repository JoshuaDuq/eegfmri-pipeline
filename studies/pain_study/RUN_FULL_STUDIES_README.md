# Run Study 1 and Study 2 on Trillium

This is the operational runbook for the pain-study analyses on Trillium. It is
the first file to use when rerunning the current Study 1 and Study 2 workflows.

Detailed references:

- Study 1 methods and edge cases: `studies/pain_study/study1/RUN_GUIDE.md`
- Study 2 Alliance workflow: `studies/pain_study/study2/alliance/README.md`
- Participant/workflow exclusions: `studies/pain_study/STUDY_ISSUES_README.md`

## Current Scope

Study 1 predicts trialwise fMRI pain-signature expression from EEG features.

Study 2 tests source-space EEG associations using the Study 1 prediction-derived
score.

Study 2 depends on Study 1. Do not start Study 2 until the Study 1 report
exists.

## Current Subjects

Study 1 all-available subjects:

```text
sub-0000
sub-0001
sub-0003
sub-0004
sub-0005
```

`sub-0002` is excluded because the MRI experiment was incomplete.

Study 2 source-level subject list:

```text
studies/pain_study/study2/alliance/subjects_study2_bem_valid.txt
```

Current contents:

```text
sub-0001
sub-0003
sub-0005
```

Do not add `sub-0000` or `sub-0004` to Study 2 source-level inference until the
BEM/trans issue is fixed and validated.

## Trillium Paths

These paths are assumed by the commands below:

```text
/project/def-mpcoll/joshduq/EEG_fMRI_Pipeline
/project/def-mpcoll/joshduq/bids/eeg
/project/def-mpcoll/joshduq/bids/fmri
/project/def-mpcoll/joshduq/external
/project/def-mpcoll/joshduq/licenses/license.txt
/scratch/joshduq/derivatives
```

Study 1 writes to:

```text
/scratch/joshduq/derivatives/group/multimodal/<STUDY1_RUN_ID>
```

Study 2 writes to:

```text
/scratch/joshduq/derivatives/group/multimodal/<STUDY2_RUN_ID>
```

## Login Setup

Run this at the start of a Trillium session:

```bash
cd /project/def-mpcoll/joshduq/EEG_fMRI_Pipeline

source local_workflows/alliance_canada/alliance_env.sh

module load StdEnv/2023
module load python/3.11
module load gcc
module load arrow

export NUMBA_CACHE_DIR="/scratch/joshduq/study_numba_cache"
mkdir -p "$NUMBA_CACHE_DIR"

source "$EEG_PIPELINE_VENV/bin/activate"
```

Check that the private study commands are installed:

```bash
eeg-pipeline --help
```

The command list must include:

```text
signature-prediction
source-interpretation
```

If either is missing:

```bash
python -m pip install -e ".[dev,ml]"
python -m pip install -e ./studies
```

## Verify Inputs

Check fMRIPrep task outputs:

```bash
find /scratch/joshduq/derivatives/preprocessed/fmri/fmriprep \
  -type f \
  -name '*task-thermalactive*desc-preproc_bold.nii.gz' \
  | head
```

If no files print, run fMRIPrep first:

```bash
cd /project/def-mpcoll/joshduq/EEG_fMRI_Pipeline/local_workflows/alliance_canada

JOB_ID="$(bash submit_fmriprep_array.sh | tail -n 1)"
echo "Submitted fMRIPrep job: $JOB_ID"
squeue -j "$JOB_ID"
```

Study 1 also needs the signature assets under:

```text
/project/def-mpcoll/joshduq/external
```

Required files:

```text
NPS/weights_NSF_grouppred_cvpcr.nii.gz
SIIPS1/nonnoc_v11_4_137subjmap_weighted_mean.nii.gz
signature_manifest.yaml
tpl-MNI152NLin2009cAsym_res-02_desc-brain_mask.nii.gz
```

## Study 1: Smoke Run

Use a smoke run to verify paths and dependencies. Do not interpret smoke
p-values.

```bash
cd /project/def-mpcoll/joshduq/EEG_fMRI_Pipeline
mkdir -p /scratch/joshduq/study1_logs
mkdir -p /scratch/joshduq/derivatives/preprocessed
ln -sfn \
  /project/def-mpcoll/joshduq/derivatives/preprocessed/eeg \
  /scratch/joshduq/derivatives/preprocessed/eeg

STUDY1_RUN_ID="study1_smoke_$(date +%Y%m%d_%H%M%S)"
STUDY1_N_PERM=10

JOB_ID="$(sbatch --parsable \
  --account=def-mpcoll \
  --time=02:00:00 \
  --cpus-per-task=16 \
  --output=/scratch/joshduq/study1_logs/study1_smoke_%j.out \
  --error=/scratch/joshduq/study1_logs/study1_smoke_%j.err \
  --export=ALL,STUDY1_RUN_ID="$STUDY1_RUN_ID",STUDY1_N_PERM="$STUDY1_N_PERM" \
  --wrap='bash studies/pain_study/scripts/trillium_run_study1_all_in_one.sh')"

echo "$JOB_ID"
```

For confirmatory `5000`-permutation runs, use the split-cell workflow below
instead of the all-in-one wrapper.

## Study 1: Confirmatory 5000-Permutation Run

A full `5000`-permutation Study 1 benchmark is too long for one sequential
Trillium job. Run it in three parts:

1. Prepare targets and features.
2. Run missing benchmark cells as a Slurm array.
3. Generate the report.

Set the run name:

```bash
cd /project/def-mpcoll/joshduq/EEG_fMRI_Pipeline

mkdir -p /scratch/joshduq/study1_logs
mkdir -p /scratch/joshduq/derivatives/preprocessed
ln -sfn \
  /project/def-mpcoll/joshduq/derivatives/preprocessed/eeg \
  /scratch/joshduq/derivatives/preprocessed/eeg

export STUDY1_RUN_ID="study1_perm5000_$(date +%Y%m%d_%H%M%S)"
export STUDY1_N_PERM=5000
export TASK="thermalactive"
export STUDY1_CONFIG="studies/pain_study/study1/config/study1_smoketest.yaml"
```

Create common Study 1 arguments:

```bash
cat > /scratch/joshduq/study1_common_args.sh <<'EOF'
SUBJECT_ARGS=(--subject 0000 --subject 0001 --subject 0003 --subject 0004 --subject 0005)

SIGNATURE_MAPS_JSON='[{"name":"NPS","path":"NPS/weights_NSF_grouppred_cvpcr.nii.gz"},{"name":"SIIPS1","path":"SIIPS1/nonnoc_v11_4_137subjmap_weighted_mean.nii.gz"}]'

COMMON_ARGS=(
  --task "$TASK"
  --study1-config "$STUDY1_CONFIG"
  --bids-root "$BIDS_EEG_ROOT"
  --bids-fmri-root "$BIDS_FMRI_ROOT"
  --deriv-root "$FMRIPREP_DERIV_ROOT"
  --set "paths.signature_dir=/project/def-mpcoll/joshduq/external"
  --set "paths.signature_maps=$SIGNATURE_MAPS_JSON"
  --set "study1.targets.signature_manifest_path=signature_manifest.yaml"
  --set "study1.targets.signature_provenance.NPS.path=NPS/weights_NSF_grouppred_cvpcr.nii.gz"
  --set "study1.targets.signature_provenance.SIIPS1.path=SIIPS1/nonnoc_v11_4_137subjmap_weighted_mean.nii.gz"
  --set "study1.targets.signature_scoring_mask_path=tpl-MNI152NLin2009cAsym_res-02_desc-brain_mask.nii.gz"
  --set "study1.cohort.min_subjects=5"
  --set "study1.feature_benchmark.n_perm=$STUDY1_N_PERM"
  --set "study1.outputs.root_name=$STUDY1_RUN_ID"
)
EOF
```

Prepare targets and features:

```bash
cat > /scratch/joshduq/run_study1_prepare.sh <<'EOF'
#!/usr/bin/env bash
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
source /scratch/joshduq/study1_common_args.sh

EEG_PIPELINE="$EEG_PIPELINE_VENV/bin/eeg-pipeline"
"$EEG_PIPELINE" signature-prediction prepare-targets "${SUBJECT_ARGS[@]}" "${COMMON_ARGS[@]}"
"$EEG_PIPELINE" signature-prediction prepare-features "${SUBJECT_ARGS[@]}" "${COMMON_ARGS[@]}"
EOF

chmod +x /scratch/joshduq/run_study1_prepare.sh

PREP_JOB="$(sbatch --parsable \
  --account=def-mpcoll \
  --time=04:00:00 \
  --cpus-per-task=16 \
  --output=/scratch/joshduq/study1_logs/study1_prepare_%j.out \
  --error=/scratch/joshduq/study1_logs/study1_prepare_%j.err \
  --export=ALL,STUDY1_RUN_ID="$STUDY1_RUN_ID",STUDY1_N_PERM="$STUDY1_N_PERM",TASK="$TASK",STUDY1_CONFIG="$STUDY1_CONFIG" \
  --wrap='bash /scratch/joshduq/run_study1_prepare.sh')"

echo "$PREP_JOB"
```

After the prepare job completes, list missing benchmark cells:

```bash
source local_workflows/alliance_canada/alliance_env.sh
source "$EEG_PIPELINE_VENV/bin/activate"
source /scratch/joshduq/study1_common_args.sh

python studies/pain_study/scripts/study1_missing_benchmark_cells.py \
  "${COMMON_ARGS[@]}" \
  > /scratch/joshduq/missing_study1_cells.txt

cat /scratch/joshduq/missing_study1_cells.txt
wc -l /scratch/joshduq/missing_study1_cells.txt
```

Run those cells as an array:

```bash
cat > /scratch/joshduq/run_study1_cell_array.sh <<'EOF'
#!/usr/bin/env bash
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
source /scratch/joshduq/study1_common_args.sh

CELL_LINE="$(sed -n "${SLURM_ARRAY_TASK_ID}p" /scratch/joshduq/missing_study1_cells.txt)"
read -r PARTITION TARGET SPEC <<< "$CELL_LINE"

python studies/pain_study/scripts/study1_benchmark_cell.py \
  --partition "$PARTITION" \
  --target "$TARGET" \
  --spec "$SPEC" \
  "${SUBJECT_ARGS[@]}" \
  "${COMMON_ARGS[@]}"
EOF

chmod +x /scratch/joshduq/run_study1_cell_array.sh

CELL_COUNT="$(wc -l < /scratch/joshduq/missing_study1_cells.txt | tr -d ' ')"
CELL_JOB="$(sbatch --parsable \
  --account=def-mpcoll \
  --time=23:00:00 \
  --cpus-per-task=16 \
  --array=1-"$CELL_COUNT"%8 \
  --output=/scratch/joshduq/study1_logs/study1_cell_%A_%a.out \
  --error=/scratch/joshduq/study1_logs/study1_cell_%A_%a.err \
  --export=ALL,STUDY1_RUN_ID="$STUDY1_RUN_ID",STUDY1_N_PERM="$STUDY1_N_PERM",TASK="$TASK",STUDY1_CONFIG="$STUDY1_CONFIG" \
  --wrap='bash /scratch/joshduq/run_study1_cell_array.sh')"

echo "$CELL_JOB"
```

Generate the Study 1 report after all cells complete:

```bash
cat > /scratch/joshduq/run_study1_report.sh <<'EOF'
#!/usr/bin/env bash
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
source /scratch/joshduq/study1_common_args.sh

EEG_PIPELINE="$EEG_PIPELINE_VENV/bin/eeg-pipeline"
"$EEG_PIPELINE" signature-prediction report "${SUBJECT_ARGS[@]}" "${COMMON_ARGS[@]}"
EOF

chmod +x /scratch/joshduq/run_study1_report.sh

REPORT_JOB="$(sbatch --parsable \
  --dependency=afterok:"$CELL_JOB" \
  --account=def-mpcoll \
  --time=02:00:00 \
  --cpus-per-task=16 \
  --output=/scratch/joshduq/study1_logs/study1_report_%j.out \
  --error=/scratch/joshduq/study1_logs/study1_report_%j.err \
  --export=ALL,STUDY1_RUN_ID="$STUDY1_RUN_ID",STUDY1_N_PERM="$STUDY1_N_PERM",TASK="$TASK",STUDY1_CONFIG="$STUDY1_CONFIG" \
  --wrap='bash /scratch/joshduq/run_study1_report.sh')"

echo "$REPORT_JOB"
```

Verify Study 1:

```bash
sacct -j "$PREP_JOB","$CELL_JOB","$REPORT_JOB" \
  --format=JobID,JobName%30,State,Elapsed,ExitCode,MaxRSS

STUDY1_ROOT="/scratch/joshduq/derivatives/group/multimodal/$STUDY1_RUN_ID"
test -f "$STUDY1_ROOT/targets/primary_targets.tsv"
test -f "$STUDY1_ROOT/reports/study1_report.tsv"
test -f "$STUDY1_ROOT/reports/article_tables/article_model_results.tsv"
```

## Study 2: Configure

Create or edit the Study 2 env file:

```bash
cd /project/def-mpcoll/joshduq/EEG_fMRI_Pipeline

cp studies/pain_study/study2/alliance/study2_alliance.env.example \
   studies/pain_study/study2/alliance/study2_alliance.env

nano studies/pain_study/study2/alliance/study2_alliance.env
```

For the current Trillium layout, use:

```text
ALLIANCE_ACCOUNT=def-mpcoll
STUDY2_MODULES="StdEnv/2023 python/3.11 scipy-stack freesurfer"
STUDY2_REPO_ROOT=/project/def-mpcoll/joshduq/EEG_fMRI_Pipeline
STUDY2_VENV=/project/def-mpcoll/joshduq/venvs/eeg_fmri_pipeline
STUDY2_BIDS_EEG_ROOT=/project/def-mpcoll/joshduq/bids/eeg
STUDY2_BIDS_MRI_ROOT=/project/def-mpcoll/joshduq/bids/fmri
STUDY2_DERIV_ROOT=/scratch/joshduq/derivatives
STUDY2_SUBJECTS_DIR=/scratch/joshduq/study2_freesurfer_subjects
STUDY2_LOG_ROOT=/scratch/joshduq/study2_logs
STUDY2_FS_LICENSE=/project/def-mpcoll/joshduq/licenses/license.txt
STUDY2_SUBJECTS_FILE=/project/def-mpcoll/joshduq/EEG_fMRI_Pipeline/studies/pain_study/study2/alliance/subjects_study2_bem_valid.txt
STUDY2_CONFIG=/project/def-mpcoll/joshduq/EEG_fMRI_Pipeline/studies/pain_study/study2/config/study2_smoketest.yaml
STUDY2_TASK=thermalactive
STUDY2_EEG_BIDS_RUN=
STUDY2_RECON_ARRAY_LIMIT=8
STUDY2_BEM_ARRAY_LIMIT=8
STUDY2_SOURCE_POWER_ARRAY_LIMIT=2
STUDY2_ADJACENCY_SUBJECT=sub-0001
STUDY2_EXPECTED_VERTICES=8196
```

Use `study2_config.yaml` only when the source-valid cohort is large enough for
production thresholds. With the current small source-valid cohort, use
`study2_smoketest.yaml` and interpret source results as exploratory.

Set the Study 1 and Study 2 roots:

```bash
STUDY2_ENV="studies/pain_study/study2/alliance/study2_alliance.env"
STUDY2_RUN_ID="study2_$(date +%Y%m%d_%H%M%S)"

grep -q '^STUDY1_OUTPUT_ROOT_NAME=' "$STUDY2_ENV" \
  && perl -0pi -e "s|^STUDY1_OUTPUT_ROOT_NAME=.*|STUDY1_OUTPUT_ROOT_NAME=${STUDY1_RUN_ID}|m" "$STUDY2_ENV" \
  || echo "STUDY1_OUTPUT_ROOT_NAME=$STUDY1_RUN_ID" >> "$STUDY2_ENV"

grep -q '^STUDY2_OUTPUT_ROOT_NAME=' "$STUDY2_ENV" \
  && perl -0pi -e "s|^STUDY2_OUTPUT_ROOT_NAME=.*|STUDY2_OUTPUT_ROOT_NAME=${STUDY2_RUN_ID}|m" "$STUDY2_ENV" \
  || echo "STUDY2_OUTPUT_ROOT_NAME=$STUDY2_RUN_ID" >> "$STUDY2_ENV"

grep -E '^(STUDY1_OUTPUT_ROOT_NAME|STUDY2_OUTPUT_ROOT_NAME)=' "$STUDY2_ENV"
```

## Study 2: Build Source-Stage Input

Build this after Study 1 report generation and before the Study 2 group stages:

```bash
source studies/pain_study/study2/alliance/lib/study2_alliance_common.sh
study2_load_env "$STUDY2_ENV"
study2_activate_python

python studies/pain_study/scripts/study2_prepare_source_stage_input.py \
  --subjects-file "$STUDY2_SUBJECTS_FILE" \
  --task "$STUDY2_TASK" \
  --study2-config "$STUDY2_CONFIG" \
  --deriv-root "$STUDY2_DERIV_ROOT" \
  --subjects-dir "$STUDY2_SUBJECTS_DIR" \
  --study1-root-name "$STUDY1_OUTPUT_ROOT_NAME" \
  --study2-root-name "$STUDY2_OUTPUT_ROOT_NAME" \
  --max-condition-number 200 \
  --target-retrained-valid-draws 1000
```

Verify:

```bash
test -f "/scratch/joshduq/derivatives/group/multimodal/$STUDY2_OUTPUT_ROOT_NAME/source_stage/source_stage_input.tsv"
```

## Study 2: Run

If anatomy, BEM/trans, adjacency, and source power are not already available,
run the full Alliance workflow:

```bash
bash studies/pain_study/study2/alliance/run_full_study2_alliance.sh "$STUDY2_ENV"
```

If source-power artifacts already exist and only the group stages need to be
rerun:

```bash
LOG_ROOT="/scratch/joshduq/study2_logs"

SOURCE_STAGE_JOB="$(sbatch --parsable \
  --account=def-mpcoll \
  --time=06:00:00 \
  --cpus-per-task=16 \
  --output="$LOG_ROOT/study2_source_%j.out" \
  --error="$LOG_ROOT/study2_source_%j.err" \
  --wrap="bash /project/def-mpcoll/joshduq/EEG_fMRI_Pipeline/studies/pain_study/study2/alliance/sbatch/05_study2_stage.sbatch $STUDY2_ENV source-stage")"

TARGET_PERM_JOB="$(sbatch --parsable \
  --dependency=afterok:"$SOURCE_STAGE_JOB" \
  --account=def-mpcoll \
  --time=23:00:00 \
  --cpus-per-task=16 \
  --output="$LOG_ROOT/study2_perm_%j.out" \
  --error="$LOG_ROOT/study2_perm_%j.err" \
  --wrap="bash /project/def-mpcoll/joshduq/EEG_fMRI_Pipeline/studies/pain_study/study2/alliance/sbatch/05_study2_stage.sbatch $STUDY2_ENV target-permutations")"

INFERENCE_JOB="$(sbatch --parsable \
  --dependency=afterok:"$TARGET_PERM_JOB" \
  --account=def-mpcoll \
  --time=02:00:00 \
  --cpus-per-task=16 \
  --output="$LOG_ROOT/study2_inference_%j.out" \
  --error="$LOG_ROOT/study2_inference_%j.err" \
  --wrap="bash /project/def-mpcoll/joshduq/EEG_fMRI_Pipeline/studies/pain_study/study2/alliance/sbatch/05_study2_stage.sbatch $STUDY2_ENV inference")"

echo "source-stage:        $SOURCE_STAGE_JOB"
echo "target-permutations: $TARGET_PERM_JOB"
echo "inference:           $INFERENCE_JOB"
```

Verify Study 2:

```bash
sacct -j "$SOURCE_STAGE_JOB","$TARGET_PERM_JOB","$INFERENCE_JOB" \
  --format=JobID,JobName%30,State,Elapsed,ExitCode,MaxRSS

STUDY2_ROOT="/scratch/joshduq/derivatives/group/multimodal/$STUDY2_OUTPUT_ROOT_NAME"
test -f "$STUDY2_ROOT/source_stage/qc_alpha.tsv"
test -f "$STUDY2_ROOT/source_stage/qc_beta.tsv"
test -f "$STUDY2_ROOT/source_stage/qc_gamma.tsv"
test -f "$STUDY2_ROOT/inference/source_family_summary.tsv"
cat "$STUDY2_ROOT/inference/source_family_summary.tsv"
```

## Generate Subject-Level QC

After Study 1 and Study 2 have finished, generate the subject-level QC package.
This writes machine-readable TSV files and a Markdown summary. It does not write
group-level p values, clusters, or automatic interpretation labels.

```bash
cd /project/def-mpcoll/joshduq/EEG_fMRI_Pipeline

module load StdEnv/2023
module load python/3.11
module load gcc
module load arrow

source /project/def-mpcoll/joshduq/venvs/eeg_fmri_pipeline/bin/activate

cat > /scratch/joshduq/pain_study_subjects_current.txt <<'EOF'
sub-0000
sub-0001
sub-0002
sub-0003
sub-0004
sub-0005
EOF

STUDY1_ROOT="/scratch/joshduq/derivatives/group/multimodal/<STUDY1_RUN_ID>"
STUDY2_ROOT="/scratch/joshduq/derivatives/group/multimodal/<STUDY2_RUN_ID>"
SUBJECTS_DIR="/scratch/joshduq/study2_freesurfer_subjects"
QC_OUT="$STUDY1_ROOT/reports/subject_qc"

python studies/pain_study/scripts/study_subject_qc_summary.py \
  --subjects-file /scratch/joshduq/pain_study_subjects_current.txt \
  --study1-root "$STUDY1_ROOT" \
  --study2-root "$STUDY2_ROOT" \
  --subjects-dir "$SUBJECTS_DIR" \
  --output-dir "$QC_OUT"

ls -lh "$QC_OUT"
```

Expected files:

- `subject_qc_summary.tsv`
- `subject_temporal_qc.tsv`
- `subject_timing_alignment_qc.tsv`
- `signature_mask_qc.tsv`
- `qc_completeness.tsv`
- `subject_qc_summary.md`

## How To Check A Recent Run

Queue:

```bash
squeue -u "$USER"
```

Accounting:

```bash
sacct -j <job_ids> --format=JobID,JobName%30,State,Elapsed,ExitCode,MaxRSS
```

Recent logs:

```bash
ls -lt /scratch/joshduq/study1_logs | head
ls -lt /scratch/joshduq/study2_logs | head
```

Study 1 output:

```bash
STUDY1_ROOT="/scratch/joshduq/derivatives/group/multimodal/<STUDY1_RUN_ID>"
ls "$STUDY1_ROOT/reports"
cat "$STUDY1_ROOT/reports/article_tables/article_model_results.tsv"
```

Study 2 output:

```bash
STUDY2_ROOT="/scratch/joshduq/derivatives/group/multimodal/<STUDY2_RUN_ID>"
cat "$STUDY2_ROOT/inference/source_family_summary.tsv"
cat "$STUDY2_ROOT/source_stage/qc_alpha.tsv"
```

## QC Checklist

Before reporting results:

1. Every Slurm job in the chain must be `COMPLETED` with `ExitCode 0:0`.
2. Study 1 report rows must show the intended `n_perm_completed`.
3. Study 2 `source_family_summary.tsv` must show the intended `n_permutations`.
4. Smoke runs are only path and dependency checks.
5. Study 2 `gate/gate_qc.json` must be checked for `confirmatory_criteria_met`
   and `unmet_criteria`.
6. If Study 2 has fewer than the intended source-valid subjects, report the
   source-stage QC table and do not overinterpret source localization.

## Common Failures

| Symptom | Meaning | Fix |
| --- | --- | --- |
| `signature-prediction` missing from `eeg-pipeline --help` | Private study package is not installed | `python -m pip install -e ./studies` |
| No fMRIPrep BOLD files | fMRIPrep is missing or under the wrong root | Run/verify fMRIPrep first |
| `signature_scoring_mask_path must be configured` | Fixed a-priori mask was not provided | Set `study1.targets.signature_scoring_mask_path` |
| Study 1 report fails on Numba/MNE import | `NUMBA_CACHE_DIR` is missing | Export `NUMBA_CACHE_DIR` before activating/running |
| Study 1 5000-permutation job times out | Full benchmark ran sequentially | Use the split-cell array workflow |
| `{subjects_dir}` override is parsed as JSON | Template path was not quoted | Use `study2_prepare_source_stage_input.py` or quote the value |
| `source_stage_input.tsv` missing | Study 2 input was not generated for this Study 1 root | Run `study2_prepare_source_stage_input.py` |
| BEM surfaces invalid | Subject anatomy cannot support source model | Exclude until BEM/trans is fixed and validated |
| Study 2 source-stage subject excluded | Rank deficient or over-conditioned design | Inspect `source_stage/qc_*.tsv` and report reduced source-valid N |
