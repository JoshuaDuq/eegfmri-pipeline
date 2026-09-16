#!/usr/bin/env bash
# Sequential chain: fMRIPrep (all subjects) -> Study 1 fMRI signature targets.
#
# The targets job is submitted with afterok on the whole fMRIPrep array, so it
# starts only if every array task succeeds and never runs against a partial
# cohort. Nothing needs babysitting between the two stages.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/local_env.sh"
source "${SCRIPT_DIR}/alliance_env.sh"
source "${SCRIPT_DIR}/lib/alliance_connection.sh"
source "${SCRIPT_DIR}/lib/slurm_args.sh"

require_alliance_connection

# Always stage the working tree before submitting. Slurm reads job scripts and the
# editable install at run time, so a submission against a stale cluster checkout
# fails minutes or hours later for reasons that look nothing like staleness.
echo "=== staging working tree to ${ALLIANCE_HOST}:${REPO_ROOT} ==="
rsync -a --delete \
    -e "ssh -o ControlPath=${ALLIANCE_SSH_CONTROL_PATH} -o BatchMode=yes" \
    --exclude-from="${SCRIPT_DIR}/repo_rsync_excludes.txt" \
    "${SCRIPT_DIR}/../../" "${ALLIANCE_HOST}:${REPO_ROOT}/" 2>&1 | grep -v '^cannot delete' || true

local_head="$(git -C "${SCRIPT_DIR}/../.." rev-parse HEAD 2>/dev/null || echo unknown)"
remote_head="$(ssh -o ControlPath="${ALLIANCE_SSH_CONTROL_PATH}" -o BatchMode=yes \
    "${ALLIANCE_HOST}" "git -C '${REPO_ROOT}' rev-parse HEAD 2>/dev/null || echo unknown")"
if [[ "${local_head}" != "${remote_head}" ]]; then
    echo "  warning: git HEAD differs (local ${local_head:0:8}, remote ${remote_head:0:8});" >&2
    echo "  uncommitted work is staged by file content, so this is expected mid-development." >&2
fi
echo "  staged"

STUDY1_RUN_ID="${STUDY1_RUN_ID:-study1_fmri_$(date +%Y%m%d_%H%M%S)}"
TASK="${FMRIPREP_TASK_ID:-thermalactive}"
STUDY1_CONFIG="${REPO_ROOT}/studies/pain_study/study1/config/study1_config.yaml"

ssh_run() {
    ssh -o ControlPath="${ALLIANCE_SSH_CONTROL_PATH}" -o BatchMode=yes "${ALLIANCE_HOST}" "$@"
}

echo "=== stage 1/2: submitting fMRIPrep array ==="
fmriprep_job="$(ssh_run "cd '${REPO_ROOT}' && ALLIANCE_CLUSTER=${ALLIANCE_CLUSTER} \
    bash local_workflows/alliance_canada/submit_fmriprep_array.sh" | tail -1)"
if [[ ! "${fmriprep_job}" =~ ^[0-9]+$ ]]; then
    echo "Could not parse fMRIPrep job id from: ${fmriprep_job}" >&2
    exit 1
fi
echo "  fMRIPrep array job: ${fmriprep_job}"

echo "=== stage 2/2: submitting targets job, gated on afterok:${fmriprep_job} ==="
targets_job="$(ssh_run "cd '${REPO_ROOT}' && \
    sbatch --parsable \
      --account='${ALLIANCE_ACCOUNT}' \
      --time=03:00:00 \
      --cpus-per-task=8 \
      --mem=64G \
      --dependency=afterok:${fmriprep_job} \
      --job-name=study1_targets \
      --output='${STUDY1_LOG_ROOT}/${STUDY1_RUN_ID}_targets_%j.out' \
      --error='${STUDY1_LOG_ROOT}/${STUDY1_RUN_ID}_targets_%j.err' \
      --export=ALL,ALLIANCE_CLUSTER=${ALLIANCE_CLUSTER},ALLIANCE_WORKFLOW_ROOT='${REPO_ROOT}/local_workflows/alliance_canada',STUDY1_RUN_ID=${STUDY1_RUN_ID},TASK=${TASK},STUDY1_CONFIG='${STUDY1_CONFIG}' \
      local_workflows/alliance_canada/study1/prepare_targets_fmri.sh" | tail -1)"
if [[ ! "${targets_job}" =~ ^[0-9]+$ ]]; then
    echo "Could not parse targets job id from: ${targets_job}" >&2
    echo "fMRIPrep array ${fmriprep_job} is still queued; cancel it if you do not want it." >&2
    exit 1
fi

cat <<EOF

Chain submitted.
  fMRIPrep array : ${fmriprep_job}   (21 subjects, 3 output spaces)
  Study 1 targets: ${targets_job}   (afterok:${fmriprep_job}, 20 subjects)
  run id         : ${STUDY1_RUN_ID}
  targets output : ${FMRIPREP_DERIV_ROOT}/group/multimodal/${STUDY1_RUN_ID}/targets/primary_targets.parquet
  logs           : ${FMRIPREP_LOG_ROOT} and ${STUDY1_LOG_ROOT}
EOF
