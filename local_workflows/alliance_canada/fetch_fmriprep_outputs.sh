#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ALLIANCE_ENV="${SCRIPT_DIR}/alliance_env.sh"
LOCAL_ENV="${SCRIPT_DIR}/local_env.sh"

if [[ ! -f "${ALLIANCE_ENV}" ]]; then
    echo "Missing Alliance environment file: ${ALLIANCE_ENV}" >&2
    exit 1
fi

if [[ ! -f "${LOCAL_ENV}" ]]; then
    echo "Missing local environment file: ${LOCAL_ENV}" >&2
    exit 1
fi

source "${ALLIANCE_ENV}"
source "${LOCAL_ENV}"

required_vars=(
    RORQUAL_HOST
    RORQUAL_SSH_CONTROL_PATH
    DERIV_ROOT
    FMRIPREP_OUTPUT_SPACES
    LOCAL_BIDS_FMRI_ROOT
    LOCAL_FMRIPREP_OUTPUT_ROOT
)

for var_name in "${required_vars[@]}"; do
    if [[ -z "${!var_name:-}" ]]; then
        echo "Required variable is empty: ${var_name}" >&2
        exit 1
    fi
done

if ! ssh -S "${RORQUAL_SSH_CONTROL_PATH}" -O check "${RORQUAL_HOST}" 2>/dev/null; then
    echo "No active Rorqual SSH control connection." >&2
    echo "Run this in your local terminal first:" >&2
    echo "  bash ${SCRIPT_DIR}/start_rorqual_connection.sh" >&2
    exit 1
fi

remote_output_root="${DERIV_ROOT}/preprocessed/fmri"
remote_archive="$(dirname "${DERIV_ROOT}")/fmriprep_preprocessed_fmri.tar"
local_derivatives_root="$(dirname "$(dirname "${LOCAL_FMRIPREP_OUTPUT_ROOT}")")"
local_archive="${local_derivatives_root}/fmriprep_preprocessed_fmri.tar"
local_fmriprep_root="${LOCAL_FMRIPREP_OUTPUT_ROOT}/fmriprep"

read -r -a output_spaces <<< "${FMRIPREP_OUTPUT_SPACES}"
if [[ "${#output_spaces[@]}" -lt 1 ]]; then
    echo "FMRIPREP_OUTPUT_SPACES must contain at least one output space." >&2
    exit 1
fi

subjects=()
while IFS= read -r subject; do
    [[ -z "${subject}" || "${subject}" =~ ^[[:space:]]*# ]] && continue
    subjects+=("${subject}")
done < "${SCRIPT_DIR}/subjects.txt"
if [[ "${#subjects[@]}" -lt 1 ]]; then
    echo "No subjects found in ${SCRIPT_DIR}/subjects.txt" >&2
    exit 1
fi

ssh \
    -o ControlPath="${RORQUAL_SSH_CONTROL_PATH}" \
    -o BatchMode=yes \
    "${RORQUAL_HOST}" \
    "test -d '${remote_output_root}'"

echo "Creating remote fMRIPrep archive: ${remote_archive}"
ssh \
    -o ControlPath="${RORQUAL_SSH_CONTROL_PATH}" \
    -o BatchMode=yes \
    "${RORQUAL_HOST}" \
    "rm -f '${remote_archive}' && cd '${remote_output_root}' && tar -cf '${remote_archive}' . && test -s '${remote_archive}'"

mkdir -p "${local_derivatives_root}" "${LOCAL_FMRIPREP_OUTPUT_ROOT}"

rsync -avh --progress \
    -e "ssh -o ControlPath=${RORQUAL_SSH_CONTROL_PATH} -o BatchMode=yes" \
    "${RORQUAL_HOST}:${remote_archive}" \
    "${local_archive}"

rm -rf "${local_fmriprep_root}"
mkdir -p "${local_fmriprep_root}"

tar -xf "${local_archive}" -C "${local_fmriprep_root}"
find "${local_fmriprep_root}" \( -name '._*' -o -name '.DS_Store' \) -delete

require_path() {
    local path="$1"
    if [[ ! -e "${path}" ]]; then
        echo "Missing expected fMRIPrep output: ${path}" >&2
        exit 1
    fi
}

verify_subject_outputs() {
    local subject="$1"
    local subject_id="${subject#sub-}"
    local subject_label="sub-${subject_id}"
    local raw_func_dir="${LOCAL_BIDS_FMRI_ROOT}/${subject_label}/func"
    local output_subject_dir="${local_fmriprep_root}/${subject_label}"

    require_path "${output_subject_dir}"
    require_path "${local_fmriprep_root}/${subject_label}.html"
    require_path "${local_fmriprep_root}/sourcedata/freesurfer/${subject_label}"

    if [[ ! -d "${raw_func_dir}" ]]; then
        echo "Missing local raw BIDS func directory: ${raw_func_dir}" >&2
        exit 1
    fi

    shopt -s nullglob
    local raw_bold_files=("${raw_func_dir}"/*_bold.nii.gz)
    shopt -u nullglob
    if [[ "${#raw_bold_files[@]}" -lt 1 ]]; then
        echo "No raw BOLD files found for ${subject_label}: ${raw_func_dir}" >&2
        exit 1
    fi

    local raw_bold
    for raw_bold in "${raw_bold_files[@]}"; do
        local base_name
        base_name="$(basename "${raw_bold}")"
        base_name="${base_name%_bold.nii.gz}"

        require_path "${output_subject_dir}/func/${base_name}_desc-confounds_timeseries.tsv"

        local space
        for space in "${output_spaces[@]}"; do
            require_path "${output_subject_dir}/func/${base_name}_space-${space}_desc-preproc_bold.nii.gz"
            require_path "${output_subject_dir}/func/${base_name}_space-${space}_desc-brain_mask.nii.gz"
        done
    done
}

for subject in "${subjects[@]}"; do
    verify_subject_outputs "${subject}"
done

if find "${local_fmriprep_root}" \( -name '._*' -o -name '.DS_Store' \) | grep -q .; then
    echo "macOS metadata files remain in ${local_fmriprep_root}" >&2
    exit 1
fi

rm -f "${local_archive}"
ssh \
    -o ControlPath="${RORQUAL_SSH_CONTROL_PATH}" \
    -o BatchMode=yes \
    "${RORQUAL_HOST}" \
    "rm -f '${remote_archive}'"

echo "Verified fMRIPrep outputs in ${local_fmriprep_root}."
