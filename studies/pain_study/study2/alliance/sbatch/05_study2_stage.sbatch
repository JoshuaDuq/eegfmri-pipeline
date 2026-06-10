#!/usr/bin/env bash
#SBATCH --job-name=s2_stage
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00

set -euo pipefail

if [[ "$#" -ne 2 ]]; then
    echo "Usage: 05_study2_stage.sbatch /path/to/study2_alliance.env stage" >&2
    exit 2
fi

env_file="$1"
stage="$2"
SCRIPT_DIR="$(cd "$(dirname "${env_file}")" && pwd)"
# shellcheck source=../lib/study2_alliance_common.sh
source "${SCRIPT_DIR}/lib/study2_alliance_common.sh"

study2_load_env "${env_file}"
study2_activate_python

mapfile -t subject_args < <(study2_subject_cli_args)
mapfile -t overrides < <(study2_runtime_overrides)

python -m eeg_pipeline.cli.main source-interpretation "${stage}" \
    "${subject_args[@]}" \
    --task "${STUDY2_TASK}" \
    --deriv-root "${STUDY2_DERIV_ROOT}" \
    --study2-config "${STUDY2_CONFIG}" \
    "${overrides[@]}"
