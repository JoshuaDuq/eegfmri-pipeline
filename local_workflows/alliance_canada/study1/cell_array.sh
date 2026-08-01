#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_args.sh"

if [[ -z "${STUDY1_CELL_MANIFEST:-}" || ! -f "${STUDY1_CELL_MANIFEST}" ]]; then
    echo "Missing Study 1 cell manifest: ${STUDY1_CELL_MANIFEST:-<empty>}" >&2
    exit 2
fi
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "SLURM_ARRAY_TASK_ID is required." >&2
    exit 2
fi

cell_line="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${STUDY1_CELL_MANIFEST}")"
read -r partition target feature_spec extra <<< "${cell_line}"
if [[ -z "${partition}" || -z "${target}" || -z "${feature_spec}" || -n "${extra:-}" ]]; then
    echo "Malformed Study 1 cell manifest row ${SLURM_ARRAY_TASK_ID}: ${cell_line}" >&2
    exit 2
fi
case "${partition}:${target}" in
    primary:NPS|primary:SIIPS1|exploratory:NPS|exploratory:SIIPS1|\
        temporal_control:NPS|temporal_control:SIIPS1)
        ;;
    *)
        echo "Malformed Study 1 cell manifest row ${SLURM_ARRAY_TASK_ID}: ${cell_line}" >&2
        exit 2
        ;;
esac

python studies/pain_study/scripts/study_support/study1_benchmark_cell.py \
    --partition "${partition}" \
    --target "${target}" \
    --spec "${feature_spec}" \
    "${SUBJECT_ARGS[@]}" \
    "${COMMON_ARGS[@]}"
