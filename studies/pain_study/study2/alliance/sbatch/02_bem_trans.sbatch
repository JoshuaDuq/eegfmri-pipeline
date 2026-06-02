#!/usr/bin/env bash
#SBATCH --job-name=s2_bem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --output=studies/pain_study/study2/alliance/logs/slurm/%x_%A_%a.out

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=../lib/study2_alliance_common.sh
source "${SCRIPT_DIR}/lib/study2_alliance_common.sh"

study2_load_env "$@"
study2_activate_python

subject="$(study2_subject_from_array)"
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
    "${run_args[@]}"
