#!/usr/bin/env bash
#SBATCH --job-name=fmriprep
#SBATCH --nodes=1
#SBATCH --ntasks=1

set -euo pipefail

required_vars=(
    SUBJECTS_FILE
    ALLIANCE_MODULES
    REPO_ROOT
    EEG_PIPELINE_VENV
    BIDS_FMRI_ROOT
    FMRIPREP_DERIV_ROOT
    FMRIPREP_WORK_ROOT
    FMRIPREP_MEM_MB
    FS_LICENSE_FILE
    FMRIPREP_IMAGE
    FMRIPREP_OUTPUT_SPACES
    TEMPLATEFLOW_HOME
    SLURM_ARRAY_TASK_ID
    SLURM_CPUS_PER_TASK
)

for var_name in "${required_vars[@]}"; do
    if [[ -z "${!var_name:-}" ]]; then
        echo "Required variable is empty: ${var_name}" >&2
        exit 1
    fi
done

mapfile -t subjects < <(grep -Ev '^[[:space:]]*($|#)' "${SUBJECTS_FILE}")
subject="${subjects[$((SLURM_ARRAY_TASK_ID - 1))]}"
subject="${subject#sub-}"

if [[ -z "${subject}" ]]; then
    echo "Resolved empty subject for array task ${SLURM_ARRAY_TASK_ID}" >&2
    exit 1
fi

# shellcheck disable=SC2086
module load ${ALLIANCE_MODULES}

unset REQUESTS_CA_BUNDLE
unset SSL_CERT_FILE
unset CURL_CA_BUNDLE
export NUMBA_CACHE_DIR="${FMRIPREP_WORK_ROOT}/numba_cache"
export TEMPLATEFLOW_HOME
export APPTAINERENV_TEMPLATEFLOW_HOME="${TEMPLATEFLOW_HOME}"
export APPTAINER_BIND="${TEMPLATEFLOW_HOME}:${TEMPLATEFLOW_HOME}${APPTAINER_BIND:+,${APPTAINER_BIND}}"

source "${EEG_PIPELINE_VENV}/bin/activate"
cd "${REPO_ROOT}"

subject_work_dir="${FMRIPREP_WORK_ROOT}/sub-${subject}"
mkdir -p "${subject_work_dir}"
mkdir -p "${NUMBA_CACHE_DIR}"

read -r -a output_spaces <<< "${FMRIPREP_OUTPUT_SPACES}"

cmd=(
    python -m eeg_pipeline
    fmri preprocess
    --engine apptainer
    --subject "${subject}"
    --bids-fmri-root "${BIDS_FMRI_ROOT}"
    --deriv-root "${FMRIPREP_DERIV_ROOT}"
    --fmriprep-work-dir "${subject_work_dir}"
    --fs-license-file "${FS_LICENSE_FILE}"
    --fmriprep-image "${FMRIPREP_IMAGE}"
    --nthreads "${SLURM_CPUS_PER_TASK}"
    --mem-mb "${FMRIPREP_MEM_MB}"
    --output-spaces "${output_spaces[@]}"
    --skip-bids-validation
)

if [[ -n "${FMRIPREP_TASK_ID:-}" ]]; then
    cmd+=(--task-id "${FMRIPREP_TASK_ID}")
fi

srun "${cmd[@]}"
