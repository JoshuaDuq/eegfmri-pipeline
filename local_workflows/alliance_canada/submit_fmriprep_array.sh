#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/alliance_env.sh"
SUBJECTS_FILE="${SCRIPT_DIR}/subjects.txt"
JOB_SCRIPT="${SCRIPT_DIR}/fmriprep_array.sbatch"

if [[ ! -f "${ENV_FILE}" ]]; then
    echo "Missing environment file: ${ENV_FILE}" >&2
    exit 1
fi

source "${ENV_FILE}"
# shellcheck source=lib/slurm_args.sh
source "${SCRIPT_DIR}/lib/slurm_args.sh"

required_vars=(
    ALLIANCE_ACCOUNT
    ALLIANCE_TIME
    ALLIANCE_CPUS
    REPO_ROOT
    EEG_PIPELINE_VENV
    BIDS_FMRI_ROOT
    DERIV_ROOT
    FMRIPREP_DERIV_ROOT
    FMRIPREP_WORK_ROOT
    FMRIPREP_LOG_ROOT
    FS_LICENSE_FILE
    FMRIPREP_IMAGE
    FMRIPREP_OUTPUT_SPACES
)

for var_name in "${required_vars[@]}"; do
    if [[ -z "${!var_name:-}" ]]; then
        echo "Required variable is empty: ${var_name}" >&2
        exit 1
    fi
    if [[ "${!var_name}" == *"your-"* ]]; then
        echo "Replace placeholder value for ${var_name}: ${!var_name}" >&2
        exit 1
    fi
done

for path in "${REPO_ROOT}" "${EEG_PIPELINE_VENV}" "${BIDS_FMRI_ROOT}" "${FS_LICENSE_FILE}"; do
    if [[ ! -e "${path}" ]]; then
        echo "Required path does not exist: ${path}" >&2
        exit 1
    fi
done

module load StdEnv/2023
module load python/3.11
module load gcc
module load arrow

source "${EEG_PIPELINE_VENV}/bin/activate"
python - <<'PY'
import pyarrow
import eeg_pipeline

print(f"pyarrow import ok: {pyarrow.__version__}", file=__import__("sys").stderr)
PY

if [[ ! -f "${SUBJECTS_FILE}" ]]; then
    echo "Missing subject list: ${SUBJECTS_FILE}" >&2
    exit 1
fi

subject_count="$(grep -Ev '^[[:space:]]*($|#)' "${SUBJECTS_FILE}" | wc -l | tr -d '[:space:]')"
if [[ "${subject_count}" -lt 1 ]]; then
    echo "No subjects found in ${SUBJECTS_FILE}" >&2
    exit 1
fi

mkdir -p "${FMRIPREP_LOG_ROOT}" "${DERIV_ROOT}" "${FMRIPREP_DERIV_ROOT}" "${FMRIPREP_WORK_ROOT}"

export SUBJECTS_FILE

sbatch_args=(
    --parsable
    --account="${ALLIANCE_ACCOUNT}"
    --time="${ALLIANCE_TIME}"
    --cpus-per-task="${ALLIANCE_CPUS}"
    --array="1-${subject_count}"
    --export="ALL,SUBJECTS_FILE=${SUBJECTS_FILE}"
    --output="${FMRIPREP_LOG_ROOT}/fmriprep_%A_%a.out"
    --error="${FMRIPREP_LOG_ROOT}/fmriprep_%A_%a.err"
)
append_optional_memory_arg sbatch_args "${FMRIPREP_SLURM_MEMORY}"
sbatch_args+=("${JOB_SCRIPT}")

job_id="$(sbatch "${sbatch_args[@]}")"

echo "${job_id}"
