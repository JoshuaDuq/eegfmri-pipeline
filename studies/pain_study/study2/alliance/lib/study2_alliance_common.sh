#!/usr/bin/env bash

set -euo pipefail

study2_script_dir() {
    local source_path="${BASH_SOURCE[0]}"
    while [[ -L "${source_path}" ]]; do
        source_path="$(readlink "${source_path}")"
    done
    cd "$(dirname "${source_path}")/.." && pwd
}

study2_require_var() {
    local name="$1"
    if [[ -z "${!name:-}" ]]; then
        echo "Missing required environment variable: ${name}" >&2
        exit 2
    fi
}

study2_load_env() {
    if [[ "$#" -ne 1 ]]; then
        echo "Usage: ${0##*/} /path/to/study2_alliance.env" >&2
        exit 2
    fi
    local env_file="$1"
    if [[ ! -f "${env_file}" ]]; then
        echo "Study 2 Alliance env file not found: ${env_file}" >&2
        exit 2
    fi

    set -a
    # shellcheck source=/dev/null
    source "${env_file}"
    set +a

    study2_require_var ALLIANCE_ACCOUNT
    study2_require_var STUDY2_REPO_ROOT
    study2_require_var STUDY2_VENV
    study2_require_var STUDY2_BIDS_EEG_ROOT
    study2_require_var STUDY2_BIDS_MRI_ROOT
    study2_require_var STUDY2_DERIV_ROOT
    study2_require_var STUDY2_SUBJECTS_DIR
    study2_require_var STUDY2_FS_LICENSE
    study2_require_var STUDY2_SUBJECTS_FILE
    study2_require_var STUDY2_CONFIG
    study2_require_var STUDY2_TASK
    study2_require_var STUDY2_LOG_ROOT
    study2_require_var STUDY1_OUTPUT_ROOT_NAME
    study2_require_var STUDY2_OUTPUT_ROOT_NAME
    study2_require_var STUDY2_RECON_ARRAY_LIMIT
    study2_require_var STUDY2_BEM_ARRAY_LIMIT
    study2_require_var STUDY2_SOURCE_POWER_ARRAY_LIMIT
}

study2_load_modules() {
    study2_require_var STUDY2_MODULES
    export FS_LICENSE="${STUDY2_FS_LICENSE}"
    # shellcheck disable=SC2086
    module load ${STUDY2_MODULES}
    if [[ -n "${FREESURFER_HOME:-}" ]]; then
        export PATH="${FREESURFER_HOME}/bin:${PATH}"
    fi
}

study2_activate_python() {
    study2_load_modules
    if [[ ! -d "${STUDY2_VENV}" ]]; then
        echo "Python environment not found: ${STUDY2_VENV}" >&2
        echo "Run bin/bootstrap_alliance_env.sh first." >&2
        exit 2
    fi
    # shellcheck source=/dev/null
    source "${STUDY2_VENV}/bin/activate"
    export PYTHONNOUSERSITE=1
    export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
    export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
    export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
    export NUMBA_CACHE_DIR="${STUDY2_LOG_ROOT}/numba_cache"
    mkdir -p "${NUMBA_CACHE_DIR}"
}

study2_subject_from_array() {
    study2_require_var SLURM_ARRAY_TASK_ID
    local subject
    subject="$(awk 'NF && $1 !~ /^#/ {print $1}' "${STUDY2_SUBJECTS_FILE}" \
        | sed -n "${SLURM_ARRAY_TASK_ID}p")"
    if [[ -z "${subject}" ]]; then
        echo "No subject at SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2
        exit 2
    fi
    echo "${subject}"
}

study2_count_subjects() {
    awk 'NF && $1 !~ /^#/ {count += 1} END {print count + 0}' "${STUDY2_SUBJECTS_FILE}"
}

study2_subject_cli_args() {
    study2_require_var STUDY2_SUBJECTS_FILE

    local subject
    while read -r subject; do
        [[ -z "${subject}" || "${subject}" == \#* ]] && continue
        printf '%s\n' "--subject"
        printf '%s\n' "${subject}"
    done < "${STUDY2_SUBJECTS_FILE}"
}

study2_runtime_overrides() {
    study2_require_var STUDY2_SUBJECTS_DIR
    study2_require_var STUDY1_OUTPUT_ROOT_NAME
    study2_require_var STUDY2_OUTPUT_ROOT_NAME

    printf '%s\n' "--set"
    printf '%s\n' "study2.source_modeling.anatomy.subjects_dir=${STUDY2_SUBJECTS_DIR}"
    printf '%s\n' "--set"
    printf '%s\n' 'study2.source_modeling.anatomy.trans_path_template="{subjects_dir}/{subject}/bem/{subject}-trans.fif"'
    printf '%s\n' "--set"
    printf '%s\n' 'study2.source_modeling.anatomy.bem_path_template="{subjects_dir}/{subject}/bem/{subject}-5120-5120-5120-bem-sol.fif"'
    printf '%s\n' "--set"
    printf '%s\n' "study2.inputs.study1_root_name=${STUDY1_OUTPUT_ROOT_NAME}"
    printf '%s\n' "--set"
    printf '%s\n' "study2.outputs.root_name=${STUDY2_OUTPUT_ROOT_NAME}"
}

study2_assert_path() {
    local path="$1"
    if [[ ! -e "${path}" ]]; then
        echo "Required path does not exist: ${path}" >&2
        exit 2
    fi
}
