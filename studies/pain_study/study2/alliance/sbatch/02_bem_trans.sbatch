#!/usr/bin/env bash
#SBATCH --job-name=s2_bem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$1")" && pwd)"
# shellcheck source=../lib/study2_alliance_common.sh
source "${SCRIPT_DIR}/lib/study2_alliance_common.sh"

study2_load_env "$@"
study2_activate_python

subject="$(study2_subject_from_array)"
exec > "${STUDY2_LOG_ROOT}/s2_bem_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" 2> "${STUDY2_LOG_ROOT}/s2_bem_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err"

run_args=()
if [[ -n "${STUDY2_EEG_BIDS_RUN:-}" ]]; then
    run_args=(--run "${STUDY2_EEG_BIDS_RUN}")
fi

python "${SCRIPT_DIR}/bin/generate_bem_trans.py" \
    --subject "${subject}" \
    --subjects-dir "${STUDY2_SUBJECTS_DIR}" \
    --bids-eeg-root "${STUDY2_BIDS_EEG_ROOT}" \
    --task "${STUDY2_TASK}" \
    --fs-license "${STUDY2_FS_LICENSE}" \
    --overwrite \
    "${run_args[@]}"
