#!/usr/bin/env bash

STUDY1_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ALLIANCE_WORKFLOW_ROOT="$(cd "${STUDY1_SCRIPT_DIR}/.." && pwd)"
source "${ALLIANCE_WORKFLOW_ROOT}/alliance_env.sh"

study1_required_vars=(
    STUDY1_RUN_ID
    STUDY1_N_PERM
    TASK
    STUDY1_CONFIG
    BIDS_EEG_ROOT
    BIDS_FMRI_ROOT
    FMRIPREP_DERIV_ROOT
    SIGNATURE_ROOT
    EEG_PIPELINE_VENV
)
for variable_name in "${study1_required_vars[@]}"; do
    if [[ -z "${!variable_name:-}" ]]; then
        echo "Required Study 1 variable is empty: ${variable_name}" >&2
        exit 2
    fi
done
if [[ ! "${STUDY1_RUN_ID}" =~ ^[A-Za-z0-9_]+$ ]]; then
    echo "Invalid Study 1 run ID: ${STUDY1_RUN_ID}" >&2
    exit 2
fi

cd "${REPO_ROOT}"
# shellcheck disable=SC2086
module load ${ALLIANCE_MODULES}
source "${EEG_PIPELINE_VENV}/bin/activate"
export NUMBA_CACHE_DIR="${ALLIANCE_SCRATCH_ROOT}/study1_numba_cache"
mkdir -p "${NUMBA_CACHE_DIR}"

SUBJECT_ARGS=()
subject_count=0
while IFS= read -r subject_id; do
    [[ -z "${subject_id}" || "${subject_id}" == \#* ]] && continue
    subject_id="${subject_id#sub-}"
    if [[ "${subject_id}" == "0006" ]]; then
        echo "Excluded participant 0006 is present in Study 1 subject manifest." >&2
        exit 2
    fi
    SUBJECT_ARGS+=(--subject "${subject_id}")
    subject_count=$((subject_count + 1))
done < "${ALLIANCE_WORKFLOW_ROOT}/study1_subjects.txt"
if [[ "${subject_count}" -ne 13 ]]; then
    echo "Study 1 requires exactly 13 participants; found ${subject_count}." >&2
    exit 2
fi

SIGNATURE_MAPS_JSON='[{"name":"NPS","path":"NPS/weights_NSF_grouppred_cvpcr.nii.gz"},{"name":"SIIPS1","path":"SIIPS1/nonnoc_v11_4_137subjmap_weighted_mean.nii.gz"}]'
COMMON_ARGS=(
    --task "${TASK}"
    --study1-config "${STUDY1_CONFIG}"
    --bids-root "${BIDS_EEG_ROOT}"
    --bids-fmri-root "${BIDS_FMRI_ROOT}"
    --deriv-root "${FMRIPREP_DERIV_ROOT}"
    --set "paths.signature_dir=${SIGNATURE_ROOT}"
    --set "paths.signature_maps=${SIGNATURE_MAPS_JSON}"
    --set "study1.targets.signature_manifest_path=signature_manifest.yaml"
    --set "study1.targets.signature_provenance.NPS.path=NPS/weights_NSF_grouppred_cvpcr.nii.gz"
    --set "study1.targets.signature_provenance.SIIPS1.path=SIIPS1/nonnoc_v11_4_137subjmap_weighted_mean.nii.gz"
    --set "study1.targets.signature_scoring_mask_path=tpl-MNI152NLin2009cAsym_res-02_desc-brain_mask.nii.gz"
    --set "study1.cohort.min_subjects=13"
    --set "study1.feature_benchmark.n_perm=${STUDY1_N_PERM}"
    --set "study1.outputs.root_name=${STUDY1_RUN_ID}"
)
