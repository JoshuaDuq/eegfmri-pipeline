#!/usr/bin/env bash
#SBATCH --job-name=s2_adjacency
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$1")" && pwd)"
# shellcheck source=../lib/study2_alliance_common.sh
source "${SCRIPT_DIR}/lib/study2_alliance_common.sh"

study2_load_env "$@"
study2_activate_python
study2_require_var STUDY2_ADJACENCY_SUBJECT
exec > "${STUDY2_LOG_ROOT}/s2_adjacency_${SLURM_JOB_ID}.out" 2> "${STUDY2_LOG_ROOT}/s2_adjacency_${SLURM_JOB_ID}.err"

adjacency_path="${STUDY2_DERIV_ROOT}/group/multimodal/${STUDY2_OUTPUT_ROOT_NAME}/inference/adjacency.npy"

expected_args=()
if [[ -n "${STUDY2_EXPECTED_VERTICES:-}" ]]; then
    expected_args=(--expected-vertices "${STUDY2_EXPECTED_VERTICES}")
fi

python "${SCRIPT_DIR}/bin/prepare_source_adjacency.py" \
    --subject "${STUDY2_ADJACENCY_SUBJECT}" \
    --subjects-dir "${STUDY2_SUBJECTS_DIR}" \
    --deriv-root "${STUDY2_DERIV_ROOT}" \
    --study2-config "${STUDY2_CONFIG}" \
    --output "${adjacency_path}" \
    "${expected_args[@]}"
