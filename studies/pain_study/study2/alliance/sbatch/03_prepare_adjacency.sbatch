#!/usr/bin/env bash
#SBATCH --job-name=s2_adjacency
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=studies/pain_study/study2/alliance/logs/slurm/%x_%j.out

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=../lib/study2_alliance_common.sh
source "${SCRIPT_DIR}/lib/study2_alliance_common.sh"

study2_load_env "$@"
study2_activate_python
study2_require_var STUDY2_ADJACENCY_SUBJECT

expected_args=()
if [[ -n "${STUDY2_EXPECTED_VERTICES:-}" ]]; then
    expected_args=(--expected-vertices "${STUDY2_EXPECTED_VERTICES}")
fi

python "${SCRIPT_DIR}/bin/prepare_source_adjacency.py" \
    --subject "${STUDY2_ADJACENCY_SUBJECT}" \
    --subjects-dir "${STUDY2_SUBJECTS_DIR}" \
    --deriv-root "${STUDY2_DERIV_ROOT}" \
    --study2-config "${STUDY2_CONFIG}" \
    "${expected_args[@]}"
