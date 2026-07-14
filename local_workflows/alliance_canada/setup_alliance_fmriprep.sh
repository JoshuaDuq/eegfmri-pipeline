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

for variable_name in LOCAL_MANIFEST_PYTHON LOCAL_BIDS_FMRI_ROOT LOCAL_FS_LICENSE_FILE; do
    if [[ -z "${!variable_name:-}" ]]; then
        echo "Required variable is empty: ${variable_name}" >&2
        exit 1
    fi
done
if [[ ! -x "${LOCAL_MANIFEST_PYTHON}" || ! -f "${LOCAL_FS_LICENSE_FILE}" ]]; then
    echo "Local Python or FreeSurfer license is unavailable." >&2
    exit 1
fi
if [[ ! -f "${SCRIPT_DIR}/subjects.txt" ]]; then
    echo "Missing fMRIPrep subject list: ${SCRIPT_DIR}/subjects.txt" >&2
    exit 1
fi

manifest_dir="$(mktemp -d)"
trap 'rm -rf "${manifest_dir}"' EXIT
manifest="${manifest_dir}/fmriprep_bids_files.txt"
"${LOCAL_MANIFEST_PYTHON}" "${SCRIPT_DIR}/build_upload_manifest.py" fmriprep \
    --subjects-file "${SCRIPT_DIR}/subjects.txt" \
    --local-fmri-root "${LOCAL_BIDS_FMRI_ROOT}" \
    --task "${FMRIPREP_TASK_ID}" \
    --output "${manifest}"

require_alliance_connection
rsync_ssh="ssh -o StrictHostKeyChecking=accept-new -o ControlPath=${ALLIANCE_SSH_CONTROL_PATH} -o BatchMode=yes"
ssh -o ControlPath="${ALLIANCE_SSH_CONTROL_PATH}" -o BatchMode=yes "${ALLIANCE_HOST}" \
    "mkdir -p '${BIDS_FMRI_ROOT}' '$(dirname "${FS_LICENSE_FILE}")' '$(dirname "${REPO_ROOT}")'"
rsync -avh --delete -e "${rsync_ssh}" \
    --exclude-from="${SCRIPT_DIR}/repo_rsync_excludes.txt" \
    "${REPO_LOCAL_ROOT}/" "${ALLIANCE_HOST}:${REPO_ROOT}/"
rsync -avh -e "${rsync_ssh}" --files-from="${manifest}" \
    "${LOCAL_BIDS_FMRI_ROOT}/" "${ALLIANCE_HOST}:${BIDS_FMRI_ROOT}/"
rsync -avh -e "${rsync_ssh}" \
    "${LOCAL_FS_LICENSE_FILE}" "${ALLIANCE_HOST}:${FS_LICENSE_FILE}"

echo "fMRIPrep inputs are staged on ${ALLIANCE_CLUSTER}."
