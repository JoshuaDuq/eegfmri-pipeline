#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

cd "$REPO_ROOT"

source local_workflows/alliance_canada/alliance_env.sh

module load StdEnv/2023
module load python/3.11
module load gcc
module load arrow

export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-/scratch/joshduq/study1_numba_cache}"
mkdir -p "$NUMBA_CACHE_DIR"

source "$EEG_PIPELINE_VENV/bin/activate"

EEG_PIPELINE="$EEG_PIPELINE_VENV/bin/eeg-pipeline"

STUDY1_CONFIG="${STUDY1_CONFIG:-studies/pain_study/study1/config/study1_smoketest.yaml}"
STUDY1_RUN_ID="${STUDY1_RUN_ID:?Set STUDY1_RUN_ID before running this script.}"
STUDY1_N_PERM="${STUDY1_N_PERM:?Set STUDY1_N_PERM before running this script.}"
TASK="${TASK:-thermalactive}"

SIGNATURE_MAPS_JSON='[{"name":"NPS","path":"NPS/weights_NSF_grouppred_cvpcr.nii.gz"},{"name":"SIIPS1","path":"SIIPS1/nonnoc_v11_4_137subjmap_weighted_mean.nii.gz"}]'

SUBJECT_ARGS=(--subject 0000 --subject 0001 --subject 0003 --subject 0004 --subject 0005)
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

"$EEG_PIPELINE" signature-prediction prepare-targets "${SUBJECT_ARGS[@]}" "${COMMON_ARGS[@]}"
"$EEG_PIPELINE" signature-prediction prepare-features "${SUBJECT_ARGS[@]}" "${COMMON_ARGS[@]}"
"$EEG_PIPELINE" signature-prediction feature-benchmark "${SUBJECT_ARGS[@]}" "${COMMON_ARGS[@]}"
"$EEG_PIPELINE" signature-prediction report "${SUBJECT_ARGS[@]}" "${COMMON_ARGS[@]}"

echo "$FMRIPREP_DERIV_ROOT/group/multimodal/$STUDY1_RUN_ID"
