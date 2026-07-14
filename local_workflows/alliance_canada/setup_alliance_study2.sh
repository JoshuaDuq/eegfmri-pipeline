#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_LOCAL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOCAL_ENV="${SCRIPT_DIR}/local_env.sh"
SUBJECTS_FILE="${REPO_LOCAL_ROOT}/studies/pain_study/study2/alliance/subjects_study2.txt"

if [[ ! -f "${LOCAL_ENV}" ]]; then
    echo "Missing local environment file: ${LOCAL_ENV}" >&2
    exit 1
fi
source "${LOCAL_ENV}"
source "${SCRIPT_DIR}/alliance_env.sh"
source "${SCRIPT_DIR}/lib/alliance_connection.sh"

manifest_dir="$(mktemp -d)"
trap 'rm -rf "${manifest_dir}"' EXIT
"${LOCAL_MANIFEST_PYTHON}" "${SCRIPT_DIR}/build_upload_manifest.py" study2 \
    --subjects-file "${SUBJECTS_FILE}" \
    --local-fmri-root "${LOCAL_BIDS_FMRI_ROOT}" \
    --local-eeg-root "${LOCAL_BIDS_EEG_ROOT}" \
    --local-deriv-root "${LOCAL_DERIV_ROOT}" \
    --task "${STUDY2_TASK}" \
    --study1-root-name "${STUDY1_OUTPUT_ROOT_NAME}" \
    --study2-root-name "${STUDY2_OUTPUT_ROOT_NAME}" \
    --output-dir "${manifest_dir}"

require_alliance_connection
rsync_ssh="ssh -o StrictHostKeyChecking=accept-new -o ControlPath=${ALLIANCE_SSH_CONTROL_PATH} -o BatchMode=yes"
ssh -o ControlPath="${ALLIANCE_SSH_CONTROL_PATH}" -o BatchMode=yes "${ALLIANCE_HOST}" \
    "mkdir -p '${BIDS_EEG_ROOT}' '${BIDS_FMRI_ROOT}' '${DERIV_ROOT}' \
        '${STUDY2_SUBJECTS_DIR}' '$(dirname "${FS_LICENSE_FILE}")'"
rsync -avh --delete -e "${rsync_ssh}" \
    --exclude-from="${SCRIPT_DIR}/repo_rsync_excludes.txt" \
    "${REPO_LOCAL_ROOT}/" "${ALLIANCE_HOST}:${REPO_ROOT}/"
rsync -avh -e "${rsync_ssh}" --files-from="${manifest_dir}/eeg_bids_files.txt" \
    "${LOCAL_BIDS_EEG_ROOT}/" "${ALLIANCE_HOST}:${BIDS_EEG_ROOT}/"
rsync -avh -e "${rsync_ssh}" --files-from="${manifest_dir}/fmri_bids_files.txt" \
    "${LOCAL_BIDS_FMRI_ROOT}/" "${ALLIANCE_HOST}:${BIDS_FMRI_ROOT}/"
rsync -avh -e "${rsync_ssh}" --files-from="${manifest_dir}/derivative_files.txt" \
    "${LOCAL_DERIV_ROOT}/" "${ALLIANCE_HOST}:${DERIV_ROOT}/"
rsync -avh -e "${rsync_ssh}" \
    "${LOCAL_FS_LICENSE_FILE}" "${ALLIANCE_HOST}:${FS_LICENSE_FILE}"
ssh -o ControlPath="${ALLIANCE_SSH_CONTROL_PATH}" -o BatchMode=yes "${ALLIANCE_HOST}" \
    "cd '${REPO_ROOT}' && ALLIANCE_CLUSTER=${ALLIANCE_CLUSTER} \
        bash local_workflows/alliance_canada/write_study2_env.sh"

echo "Study 2 inputs are staged on ${ALLIANCE_CLUSTER}."
