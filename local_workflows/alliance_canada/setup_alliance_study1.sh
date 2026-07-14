#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_LOCAL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOCAL_ENV="${SCRIPT_DIR}/local_env.sh"

if [[ ! -f "${LOCAL_ENV}" ]]; then
    echo "Missing local environment file: ${LOCAL_ENV}" >&2
    exit 1
fi
source "${LOCAL_ENV}"
source "${SCRIPT_DIR}/alliance_env.sh"
source "${SCRIPT_DIR}/lib/alliance_connection.sh"

required_vars=(
    ALLIANCE_HOST
    ALLIANCE_SSH_CONTROL_PATH
    LOCAL_MANIFEST_PYTHON
    LOCAL_BIDS_FMRI_ROOT
    LOCAL_BIDS_EEG_ROOT
    LOCAL_DERIV_ROOT
    LOCAL_EXTERNAL_ROOT
    REPO_ROOT
    BIDS_FMRI_ROOT
    BIDS_EEG_ROOT
    FMRIPREP_DERIV_ROOT
    SIGNATURE_ROOT
)
for variable_name in "${required_vars[@]}"; do
    if [[ -z "${!variable_name:-}" ]]; then
        echo "Required variable is empty: ${variable_name}" >&2
        exit 1
    fi
done
if [[ ! -x "${LOCAL_MANIFEST_PYTHON}" ]]; then
    echo "Local manifest Python is not executable: ${LOCAL_MANIFEST_PYTHON}" >&2
    exit 1
fi

manifest_dir="$(mktemp -d)"
trap 'rm -rf "${manifest_dir}"' EXIT
"${LOCAL_MANIFEST_PYTHON}" "${SCRIPT_DIR}/build_upload_manifest.py" study1 \
    --subjects-file "${SCRIPT_DIR}/study1_subjects.txt" \
    --local-fmri-root "${LOCAL_BIDS_FMRI_ROOT}" \
    --local-eeg-root "${LOCAL_BIDS_EEG_ROOT}" \
    --local-deriv-root "${LOCAL_DERIV_ROOT}" \
    --local-external-root "${LOCAL_EXTERNAL_ROOT}" \
    --task "${FMRIPREP_TASK_ID}" \
    --output-dir "${manifest_dir}"

require_alliance_connection
ssh_options=(
    -o StrictHostKeyChecking=accept-new
    -o ControlPath="${ALLIANCE_SSH_CONTROL_PATH}"
    -o BatchMode=yes
)
rsync_ssh="ssh -o StrictHostKeyChecking=accept-new -o ControlPath=${ALLIANCE_SSH_CONTROL_PATH} -o BatchMode=yes"
remote_dirs=(
    "$(dirname "${REPO_ROOT}")"
    "${BIDS_FMRI_ROOT}"
    "${BIDS_EEG_ROOT}"
    "${FMRIPREP_DERIV_ROOT}/preprocessed/eeg"
    "${FMRIPREP_DERIV_ROOT}/preprocessed/fmri"
    "${SIGNATURE_ROOT}"
    "${STUDY1_LOG_ROOT}"
    "${STUDY1_RUNTIME_ROOT}"
)
printf -v mkdir_args '%q ' "${remote_dirs[@]}"
ssh "${ssh_options[@]}" "${ALLIANCE_HOST}" "mkdir -p ${mkdir_args}"

rsync -avh --delete \
    -e "${rsync_ssh}" \
    --exclude-from="${SCRIPT_DIR}/repo_rsync_excludes.txt" \
    "${REPO_LOCAL_ROOT}/" \
    "${ALLIANCE_HOST}:${REPO_ROOT}/"

rsync_manifest() {
    local source_root="$1"
    local manifest="$2"
    local destination_root="$3"
    rsync -avh \
        -e "${rsync_ssh}" \
        --files-from="${manifest}" \
        "${source_root}/" \
        "${ALLIANCE_HOST}:${destination_root}/"
}

rsync_manifest "${LOCAL_BIDS_FMRI_ROOT}" "${manifest_dir}/fmri_bids_files.txt" "${BIDS_FMRI_ROOT}"
rsync_manifest "${LOCAL_BIDS_EEG_ROOT}" "${manifest_dir}/eeg_bids_files.txt" "${BIDS_EEG_ROOT}"
rsync_manifest "${LOCAL_DERIV_ROOT}" "${manifest_dir}/eeg_derivative_files.txt" "${FMRIPREP_DERIV_ROOT}"
rsync_manifest \
    "${LOCAL_DERIV_ROOT}/preprocessed/fmri/fmriprep" \
    "${manifest_dir}/fmri_derivative_files.txt" \
    "${FMRIPREP_DERIV_ROOT}/preprocessed/fmri"
rsync_manifest "${LOCAL_EXTERNAL_ROOT}" "${manifest_dir}/external_files.txt" "${SIGNATURE_ROOT}"

while IFS= read -r subject_id; do
    [[ -z "${subject_id}" || "${subject_id}" == \#* ]] && continue
    subject="sub-${subject_id#sub-}"
    ssh "${ssh_options[@]}" "${ALLIANCE_HOST}" \
        "test -d '${BIDS_FMRI_ROOT}/${subject}/func' && \
         test -d '${BIDS_EEG_ROOT}/${subject}/eeg' && \
         test -d '${FMRIPREP_DERIV_ROOT}/preprocessed/eeg/${subject}' && \
         test -d '${FMRIPREP_DERIV_ROOT}/preprocessed/fmri/${subject}/func'"
done < "${SCRIPT_DIR}/study1_subjects.txt"

ssh "${ssh_options[@]}" "${ALLIANCE_HOST}" \
    "test -f '${SIGNATURE_ROOT}/NPS/weights_NSF_grouppred_cvpcr.nii.gz' && \
     test -f '${SIGNATURE_ROOT}/SIIPS1/nonnoc_v11_4_137subjmap_weighted_mean.nii.gz' && \
     test -f '${SIGNATURE_ROOT}/signature_manifest.yaml' && \
     test -f '${SIGNATURE_ROOT}/tpl-MNI152NLin2009cAsym_res-02_desc-brain_mask.nii.gz'"

echo "Study 1 inputs are staged and validated on ${ALLIANCE_CLUSTER}."
