#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib/study2_alliance_common.sh
source "${SCRIPT_DIR}/lib/study2_alliance_common.sh"

study2_load_env "$@"

subject_count="$(study2_count_subjects)"
if [[ "${subject_count}" -lt 1 ]]; then
    echo "Subject list is empty: ${STUDY2_SUBJECTS_FILE}" >&2
    exit 2
fi

mkdir -p "${SCRIPT_DIR}/logs/slurm"

recon_job="$(sbatch \
    --parsable \
    --account "${ALLIANCE_ACCOUNT}" \
    --array "1-${subject_count}%${STUDY2_RECON_ARRAY_LIMIT}" \
    "${SCRIPT_DIR}/sbatch/01_recon_all.sbatch" \
    "$1")"

bem_job="$(sbatch \
    --parsable \
    --account "${ALLIANCE_ACCOUNT}" \
    --dependency "afterok:${recon_job}" \
    --array "1-${subject_count}%${STUDY2_BEM_ARRAY_LIMIT}" \
    "${SCRIPT_DIR}/sbatch/02_bem_trans.sbatch" \
    "$1")"

adjacency_job="$(sbatch \
    --parsable \
    --account "${ALLIANCE_ACCOUNT}" \
    --dependency "afterok:${bem_job}" \
    "${SCRIPT_DIR}/sbatch/03_prepare_adjacency.sbatch" \
    "$1")"

source_power_job="$(sbatch \
    --parsable \
    --account "${ALLIANCE_ACCOUNT}" \
    --dependency "afterok:${bem_job}" \
    --array "1-${subject_count}%${STUDY2_SOURCE_POWER_ARRAY_LIMIT}" \
    "${SCRIPT_DIR}/sbatch/04_source_power.sbatch" \
    "$1")"

gate_job="$(sbatch \
    --parsable \
    --account "${ALLIANCE_ACCOUNT}" \
    --time "01:00:00" \
    --mem "8G" \
    --dependency "afterok:${source_power_job}" \
    "${SCRIPT_DIR}/sbatch/05_study2_stage.sbatch" \
    "$1" \
    gate)"

source_stage_job="$(sbatch \
    --parsable \
    --account "${ALLIANCE_ACCOUNT}" \
    --time "12:00:00" \
    --mem "64G" \
    --dependency "afterok:${source_power_job}" \
    "${SCRIPT_DIR}/sbatch/05_study2_stage.sbatch" \
    "$1" \
    source-stage)"

target_permutations_job="$(sbatch \
    --parsable \
    --account "${ALLIANCE_ACCOUNT}" \
    --time "3-00:00:00" \
    --mem "96G" \
    --dependency "afterok:${source_stage_job}" \
    "${SCRIPT_DIR}/sbatch/05_study2_stage.sbatch" \
    "$1" \
    target-permutations)"

inference_job="$(sbatch \
    --parsable \
    --account "${ALLIANCE_ACCOUNT}" \
    --time "12:00:00" \
    --mem "64G" \
    --dependency "afterok:${target_permutations_job}:${adjacency_job}" \
    "${SCRIPT_DIR}/sbatch/05_study2_stage.sbatch" \
    "$1" \
    inference)"

cat <<EOF
Submitted Study 2 Alliance workflow:
  recon-all:             ${recon_job}
  BEM/trans:             ${bem_job}
  adjacency:             ${adjacency_job}
  source-power:          ${source_power_job}
  gate:                  ${gate_job}
  source-stage:          ${source_stage_job}
  target-permutations:   ${target_permutations_job}
  inference:             ${inference_job}

Monitor:
  squeue -u "$USER"
  sacct -j ${recon_job},${bem_job},${source_power_job},${inference_job} --format=JobID,JobName,State,Elapsed,MaxRSS
EOF
