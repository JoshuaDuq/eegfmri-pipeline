#!/usr/bin/env bash
#SBATCH --job-name=s2_source_power
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
exec > "${STUDY2_LOG_ROOT}/s2_source_power_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out" 2> "${STUDY2_LOG_ROOT}/s2_source_power_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err"

subject_label="${subject}"
mapfile -t overrides < <(study2_runtime_overrides)

python -m eeg_pipeline.cli.main source-interpretation source-power \
    --subject "${subject_label}" \
    --task "${STUDY2_TASK}" \
    --deriv-root "${STUDY2_DERIV_ROOT}" \
    --study2-config "${STUDY2_CONFIG}" \
    "${overrides[@]}"
