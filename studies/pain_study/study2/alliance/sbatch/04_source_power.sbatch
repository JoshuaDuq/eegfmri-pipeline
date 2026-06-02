#!/usr/bin/env bash
#SBATCH --job-name=s2_source_power
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=studies/pain_study/study2/alliance/logs/slurm/%x_%A_%a.out

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=../lib/study2_alliance_common.sh
source "${SCRIPT_DIR}/lib/study2_alliance_common.sh"

study2_load_env "$@"
study2_activate_python

subject="$(study2_subject_from_array)"
subject_label="${subject#sub-}"
mapfile -t overrides < <(study2_runtime_overrides)

python -m eeg_pipeline.cli.main source-interpretation source-power \
    --subject "${subject_label}" \
    --task "${STUDY2_TASK}" \
    --deriv-root "${STUDY2_DERIV_ROOT}" \
    --study2-config "${STUDY2_CONFIG}" \
    "${overrides[@]}"
