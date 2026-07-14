#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_ENV="${SCRIPT_DIR}/local_env.sh"
TRILLIUM_HOST="joshduq@trillium.alliancecan.ca"
TRILLIUM_SSH_CONTROL_PATH="/tmp/trillium-joshduq-ssh-control"
TRILLIUM_FMRIPREP_ROOT="/scratch/joshduq/derivatives/preprocessed/fmri"
MISSING_SUBJECTS=(0009 0012 0013 0014)

if [[ ! -f "${LOCAL_ENV}" ]]; then
    echo "Missing local environment file: ${LOCAL_ENV}" >&2
    exit 1
fi
source "${LOCAL_ENV}"

if [[ -z "${LOCAL_DERIV_ROOT:-}" ]]; then
    echo "Required variable is empty: LOCAL_DERIV_ROOT" >&2
    exit 1
fi
if ! ssh -S "${TRILLIUM_SSH_CONTROL_PATH}" -O check "${TRILLIUM_HOST}" >/dev/null 2>&1; then
    echo "No active Trillium SSH control connection." >&2
    exit 1
fi

destination_root="${LOCAL_DERIV_ROOT}/preprocessed/fmri/fmriprep"
mkdir -p "${destination_root}"

for subject_id in "${MISSING_SUBJECTS[@]}"; do
    subject="sub-${subject_id}"
    destination="${destination_root}/${subject}"
    if [[ -e "${destination}" ]]; then
        echo "Refusing to overwrite existing fMRIPrep subject: ${destination}" >&2
        exit 1
    fi

    ssh \
        -o ControlPath="${TRILLIUM_SSH_CONTROL_PATH}" \
        -o BatchMode=yes \
        "${TRILLIUM_HOST}" \
        "test -d '${TRILLIUM_FMRIPREP_ROOT}/${subject}/func'"
    rsync -avh --progress \
        -e "ssh -o ControlPath=${TRILLIUM_SSH_CONTROL_PATH} -o BatchMode=yes" \
        "${TRILLIUM_HOST}:${TRILLIUM_FMRIPREP_ROOT}/${subject}/" \
        "${destination}/"

    if ! find "${destination}/func" \
        -name "${subject}_task-thermalactive*_desc-preproc_bold.nii.gz" \
        -print -quit | grep -q .; then
        echo "Retrieved subject lacks thermalactive preprocessed BOLD: ${subject}" >&2
        exit 1
    fi
done

echo "Retrieved Study 1 fMRIPrep subjects from Trillium into ${destination_root}."
