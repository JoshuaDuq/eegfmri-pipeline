#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_args.sh"
source "${ALLIANCE_WORKFLOW_ROOT}/lib/slurm_args.sh"

if [[ -z "${JOB_RECORD:-}" ]]; then
    echo "JOB_RECORD is required." >&2
    exit 2
fi

run_root="${STUDY1_RUNTIME_ROOT}/${STUDY1_RUN_ID}"
cell_manifest="${run_root}/missing_benchmark_cells.txt"
mkdir -p "${run_root}" "${STUDY1_LOG_ROOT}"
python studies/pain_study/scripts/study1_missing_benchmark_cells.py \
    "${COMMON_ARGS[@]}" > "${cell_manifest}.tmp"

if ! awk '
    NF != 3 {exit 1}
    $1 !~ /^(primary|exploratory|temporal_control)$/ {exit 1}
    $2 !~ /^(NPS|SIIPS1)$/ {exit 1}
' "${cell_manifest}.tmp"; then
    echo "Generated malformed Study 1 cell manifest: ${cell_manifest}.tmp" >&2
    exit 2
fi
mv "${cell_manifest}.tmp" "${cell_manifest}"
cell_count="$(wc -l < "${cell_manifest}" | tr -d '[:space:]')"

job_exports="ALL,ALLIANCE_CLUSTER=${ALLIANCE_CLUSTER},STUDY1_RUN_ID=${STUDY1_RUN_ID},STUDY1_N_PERM=${STUDY1_N_PERM},TASK=${TASK},STUDY1_CONFIG=${STUDY1_CONFIG},JOB_RECORD=${JOB_RECORD}"
report_args=(
    --parsable
    --account="${ALLIANCE_ACCOUNT}"
    --time="04:00:00"
    --cpus-per-task=16
    --output="${STUDY1_LOG_ROOT}/${STUDY1_RUN_ID}_report_%j.out"
    --error="${STUDY1_LOG_ROOT}/${STUDY1_RUN_ID}_report_%j.err"
    --export="${job_exports}"
)
append_optional_memory_arg report_args "${STUDY1_REPORT_SLURM_MEMORY}"

if [[ "${cell_count}" -eq 0 ]]; then
    report_args+=("${SCRIPT_DIR}/report.sh")
    report_job="$(sbatch "${report_args[@]}")"
    {
        echo "CELL_COUNT=0"
        echo "REPORT_JOB=${report_job}"
    } >> "${JOB_RECORD}"
    exit 0
fi

cell_args=(
    --parsable
    --account="${ALLIANCE_ACCOUNT}"
    --time="23:00:00"
    --cpus-per-task=16
    --array="1-${cell_count}%8"
    --output="${STUDY1_LOG_ROOT}/${STUDY1_RUN_ID}_cell_%A_%a.out"
    --error="${STUDY1_LOG_ROOT}/${STUDY1_RUN_ID}_cell_%A_%a.err"
    --export="${job_exports},STUDY1_CELL_MANIFEST=${cell_manifest}"
)
append_optional_memory_arg cell_args "${STUDY1_CELL_SLURM_MEMORY}"
cell_args+=("${SCRIPT_DIR}/cell_array.sh")
cell_job="$(sbatch "${cell_args[@]}")"

report_args+=(--dependency="afterok:${cell_job}" "${SCRIPT_DIR}/report.sh")
report_job="$(sbatch "${report_args[@]}")"
{
    echo "CELL_COUNT=${cell_count}"
    echo "CELL_MANIFEST=${cell_manifest}"
    echo "CELL_JOB=${cell_job}"
    echo "REPORT_JOB=${report_job}"
} >> "${JOB_RECORD}"

echo "Submitted ${cell_count} Study 1 benchmark cells as job ${cell_job}."
echo "Submitted Study 1 report as job ${report_job} afterok:${cell_job}."
