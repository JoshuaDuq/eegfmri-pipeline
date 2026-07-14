#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ALLIANCE_WORKFLOW_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${ALLIANCE_WORKFLOW_ROOT}/alliance_env.sh"
source "${ALLIANCE_WORKFLOW_ROOT}/lib/slurm_args.sh"

STUDY1_RUN_ID="study1_$(date +%Y%m%d_%H%M%S)"
STUDY1_N_PERM="5000"
TASK="thermalactive"
STUDY1_CONFIG="${REPO_ROOT}/studies/pain_study/study1/config/study1_config.yaml"
run_root="${STUDY1_RUNTIME_ROOT}/${STUDY1_RUN_ID}"
JOB_RECORD="${STUDY1_LOG_ROOT}/${STUDY1_RUN_ID}_jobs.txt"
mkdir -p "${run_root}" "${STUDY1_LOG_ROOT}"

for required_path in \
    "${EEG_PIPELINE_VENV}/bin/eeg-pipeline" \
    "${STUDY1_CONFIG}" \
    "${ALLIANCE_WORKFLOW_ROOT}/study1_subjects.txt" \
    "${SCRIPT_DIR}/prepare.sh" \
    "${SCRIPT_DIR}/dispatch.sh"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "Missing required Study 1 path: ${required_path}" >&2
        exit 2
    fi
done

job_exports="ALL,ALLIANCE_CLUSTER=${ALLIANCE_CLUSTER},STUDY1_RUN_ID=${STUDY1_RUN_ID},STUDY1_N_PERM=${STUDY1_N_PERM},TASK=${TASK},STUDY1_CONFIG=${STUDY1_CONFIG},JOB_RECORD=${JOB_RECORD}"
prepare_args=(
    --parsable
    --account="${ALLIANCE_ACCOUNT}"
    --time="23:00:00"
    --cpus-per-task=16
    --output="${STUDY1_LOG_ROOT}/${STUDY1_RUN_ID}_prepare_%j.out"
    --error="${STUDY1_LOG_ROOT}/${STUDY1_RUN_ID}_prepare_%j.err"
    --export="${job_exports}"
)
append_optional_memory_arg prepare_args "${STUDY1_PREPARE_SLURM_MEMORY}"
prepare_args+=("${SCRIPT_DIR}/prepare.sh")
prepare_job="$(sbatch "${prepare_args[@]}")"

cat > "${JOB_RECORD}" <<EOF
RUN_ID=${STUDY1_RUN_ID}
CLUSTER=${ALLIANCE_CLUSTER}
ACCOUNT=${ALLIANCE_ACCOUNT}
N_PERM=${STUDY1_N_PERM}
PREPARE_JOB=${prepare_job}
JOB_RECORD=${JOB_RECORD}
EOF

dispatch_args=(
    --parsable
    --account="${ALLIANCE_ACCOUNT}"
    --time="01:00:00"
    --cpus-per-task=1
    --dependency="afterok:${prepare_job}"
    --output="${STUDY1_LOG_ROOT}/${STUDY1_RUN_ID}_dispatch_%j.out"
    --error="${STUDY1_LOG_ROOT}/${STUDY1_RUN_ID}_dispatch_%j.err"
    --export="${job_exports}"
)
append_optional_memory_arg dispatch_args "${STUDY1_DISPATCH_SLURM_MEMORY}"
dispatch_args+=("${SCRIPT_DIR}/dispatch.sh")
dispatch_job="$(sbatch "${dispatch_args[@]}")"
echo "DISPATCH_JOB=${dispatch_job}" >> "${JOB_RECORD}"

cat "${JOB_RECORD}"
