#!/usr/bin/env bash
#SBATCH --job-name=s2_recon
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=23:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$1")" && pwd)"
# shellcheck source=../lib/study2_alliance_common.sh
source "${SCRIPT_DIR}/lib/study2_alliance_common.sh"

study2_load_env "$@"
study2_load_modules

subject="$(study2_subject_from_array)"
exec > "${STUDY2_LOG_ROOT}/s2_recon_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" 2> "${STUDY2_LOG_ROOT}/s2_recon_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err"

t1_path="${STUDY2_BIDS_MRI_ROOT}/${subject}/anat/${subject}_T1w.nii.gz"
study2_assert_path "${t1_path}"
study2_assert_path "${STUDY2_FS_LICENSE}"

mkdir -p "${STUDY2_SUBJECTS_DIR}"
export SUBJECTS_DIR="${STUDY2_SUBJECTS_DIR}"
export FS_LICENSE="${STUDY2_FS_LICENSE}"

recon-all \
    -subject "${subject}" \
    -i "${t1_path}" \
    -all \
    -sd "${STUDY2_SUBJECTS_DIR}" \
    -openmp "${SLURM_CPUS_PER_TASK:-1}"
