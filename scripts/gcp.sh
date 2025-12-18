#!/usr/bin/env bash
set -euo pipefail

###################################################################
# Configuration
###################################################################

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

REMOTE_USER="${REMOTE_USER:-}"
REMOTE_HOST="${REMOTE_HOST:-104.197.215.15}"
REMOTE_BASE="${REMOTE_BASE:-/mnt/data/Thermal_Pain_EEG_Pipeline}"

if [[ -n "${REMOTE_USER}" ]]; then
  REMOTE_TARGET="${REMOTE_USER}@${REMOTE_HOST}"
else
  REMOTE_TARGET="${REMOTE_HOST}"
fi

LOCAL_CODE_DIR="${REPO_ROOT}"
LOCAL_DATA_DIR="${LOCAL_DATA_DIR:-${REPO_ROOT}/eeg_pipeline/data}"

PULL_EXCLUDE_PREPROCESSED="${PULL_EXCLUDE_PREPROCESSED:-1}"

REMOTE_PYTHON="${REMOTE_PYTHON:-python3}"
REMOTE_VENV_ACTIVATE="${REMOTE_VENV_ACTIVATE:-}"
REMOTE_N_JOBS="${REMOTE_N_JOBS:--1}"
PARALLEL_SUBJECTS="${PARALLEL_SUBJECTS:-1}"

###################################################################
# Rsync Options
###################################################################

RSYNC_BASE_ARGS=(
  -avz
  --delete
  --partial
  --progress
  --human-readable
)

CODE_EXCLUDES=(
  --exclude '.git'
  --exclude '__pycache__'
  --exclude '*.pyc'
  --exclude '.pytest_cache'
  --exclude '*.egg-info'
  --exclude '.venv'
  --exclude 'venv'
  --exclude '.mypy_cache'
  --exclude '.DS_Store'
  --exclude 'eeg_pipeline/data'
)

DATA_EXCLUDES=(
  --exclude '.DS_Store'
)

###################################################################
# Functions
###################################################################

usage() {
  cat <<EOF
Usage: $(basename "$0") [OPTIONS] [COMMAND]

Sync code/data to GCP instance and run pipeline remotely.

OPTIONS:
  --code-only     Sync only code (no data)
  --data-only     Sync only data (no code)
  --dry-run       Show what would be transferred without doing it
  --no-delete     Do not delete remote files missing locally (default: delete)
  -h, --help      Show this help

COMMANDS:
  sync            Sync code and data to remote (default)
  run [ARGS]      Sync then run pipeline with ARGS on remote
  ssh             Sync then open SSH session
  pull            Pull derivatives from remote to local
  batch [ARGS]    Sync, run pipeline, wait for completion, pull results
  batch-stop [ARGS]
                  Same as batch, then stop the GCE instance (requires gcloud)

EXAMPLES:
  $(basename "$0")                                      # Sync code + data
  $(basename "$0") --code-only                          # Sync code only
  $(basename "$0") run features compute --subject 0001  # Sync + run
  $(basename "$0") ssh                                  # Sync + SSH in
  $(basename "$0") pull                                 # Pull derivatives
  $(basename "$0") batch features compute --subject 0001

ENVIRONMENT:
  REMOTE_USER     Remote username (default: empty; rely on ~/.ssh/config)
  REMOTE_HOST     Remote IP or SSH alias (default: 104.197.215.15)
  REMOTE_BASE     Remote base path (default: /mnt/data/Thermal_Pain_EEG_Pipeline)
  LOCAL_DATA_DIR  Local data directory (default: \$REPO_ROOT/eeg_pipeline/data)

  GCP_PROJECT     GCP project id for stopping the VM (required for batch-stop)
  GCP_ZONE        GCP zone for stopping the VM (required for batch-stop)
  GCP_INSTANCE    GCE instance name to stop (required for batch-stop)
  SHUTDOWN_WHEN   When to stop the VM: always|success (default: always)
EOF
}

stop_gce_instance() {
  local when="${SHUTDOWN_WHEN:-always}"

  if [[ -z "${GCP_PROJECT:-}" || -z "${GCP_ZONE:-}" || -z "${GCP_INSTANCE:-}" ]]; then
    echo "==> Skipping shutdown: set GCP_PROJECT, GCP_ZONE, and GCP_INSTANCE to enable batch-stop." >&2
    return 0
  fi

  if ! command -v gcloud >/dev/null 2>&1; then
    echo "==> Skipping shutdown: gcloud not found on local machine." >&2
    return 0
  fi

  if [[ "${when}" != "always" && "${when}" != "success" ]]; then
    echo "==> Invalid SHUTDOWN_WHEN='${when}'. Use 'always' or 'success'." >&2
    return 1
  fi

  if [[ "${when}" == "success" && "${PIPELINE_EXIT_CODE}" != "0" ]]; then
    echo "==> Not stopping VM (SHUTDOWN_WHEN=success, pipeline exit=${PIPELINE_EXIT_CODE})." >&2
    return 0
  fi

  echo "==> Stopping GCE instance: ${GCP_INSTANCE} (${GCP_PROJECT}, ${GCP_ZONE})"
  gcloud compute instances stop "${GCP_INSTANCE}" --project "${GCP_PROJECT}" --zone "${GCP_ZONE}"
}

sync_code() {
  echo "==> Syncing code to ${REMOTE_TARGET}:${REMOTE_BASE}/"
  local args=("${RSYNC_BASE_ARGS[@]}" "${CODE_EXCLUDES[@]}")
  [[ "${DRY_RUN}" == "1" ]] && args+=(--dry-run --itemize-changes)
  [[ "${NO_DELETE}" == "1" ]] && args=(${args[@]/--delete/})
  
  rsync "${args[@]}" "${LOCAL_CODE_DIR}/" "${REMOTE_TARGET}:${REMOTE_BASE}/"
}

sync_data() {
  if [[ ! -d "${LOCAL_DATA_DIR}" ]]; then
    echo "Warning: LOCAL_DATA_DIR does not exist: ${LOCAL_DATA_DIR}" >&2
    echo "Skipping data sync." >&2
    return 0
  fi
  
  echo "==> Syncing data to ${REMOTE_TARGET}:${REMOTE_BASE}/eeg_pipeline/data/"
  local args=("${RSYNC_BASE_ARGS[@]}" "${DATA_EXCLUDES[@]}")
  [[ "${DRY_RUN}" == "1" ]] && args+=(--dry-run --itemize-changes)
  [[ "${NO_DELETE}" == "1" ]] && args=(${args[@]/--delete/})
  
  rsync "${args[@]}" "${LOCAL_DATA_DIR}/" "${REMOTE_TARGET}:${REMOTE_BASE}/eeg_pipeline/data/"
}

run_remote() {
  local prefix="cd ${REMOTE_BASE}"
  if [[ -n "${REMOTE_VENV_ACTIVATE}" ]]; then
    prefix="${prefix} && source ${REMOTE_VENV_ACTIVATE}"
  fi
  prefix="${prefix} && export EEG_PIPELINE_N_JOBS=${REMOTE_N_JOBS} && export MNE_N_JOBS=${REMOTE_N_JOBS}"
  local cmd="${prefix} && ${REMOTE_PYTHON} -m eeg_pipeline $*"
  echo "==> Running on remote: ${cmd}"
  ssh "${REMOTE_TARGET}" "${cmd}"
}

run_remote_blocking() {
  local prefix="cd ${REMOTE_BASE}"
  if [[ -n "${REMOTE_VENV_ACTIVATE}" ]]; then
    prefix="${prefix} && source ${REMOTE_VENV_ACTIVATE}"
  fi
  prefix="${prefix} && export EEG_PIPELINE_N_JOBS=${REMOTE_N_JOBS} && export MNE_N_JOBS=${REMOTE_N_JOBS}"
  local cmd="${prefix} && ${REMOTE_PYTHON} -m eeg_pipeline $*"
  echo "==> Running on remote (blocking): ${cmd}"
  echo "    Waiting for pipeline to complete..."
  ssh "${REMOTE_TARGET}" "${cmd}"
  local exit_code=$?
  if [[ ${exit_code} -ne 0 ]]; then
    echo "==> Pipeline finished with exit code ${exit_code}" >&2
  else
    echo "==> Pipeline completed successfully."
  fi
  return ${exit_code}
}

open_ssh() {
  echo "==> Opening SSH session..."
  ssh -t "${REMOTE_TARGET}" "cd ${REMOTE_BASE} && exec \$SHELL -l"
}

pull_derivatives() {
  local remote_derivatives="${REMOTE_TARGET}:${REMOTE_BASE}/eeg_pipeline/data/derivatives/"
  local local_derivatives="${LOCAL_DATA_DIR}/derivatives/"
  
  echo "==> Pulling derivatives from ${remote_derivatives}"
  echo "    to ${local_derivatives}"
  
  mkdir -p "${local_derivatives}"
  
  local args=("${RSYNC_BASE_ARGS[@]}" --exclude '.DS_Store')
  if [[ "${PULL_EXCLUDE_PREPROCESSED}" == "1" ]]; then
    args+=(--exclude 'preprocessed')
  fi
  [[ "${DRY_RUN}" == "1" ]] && args+=(--dry-run --itemize-changes)
  
  rsync "${args[@]}" "${remote_derivatives}" "${local_derivatives}"
}

###################################################################
# Main
###################################################################

SYNC_CODE=1
SYNC_DATA=1
DRY_RUN=0
NO_DELETE=0
COMMAND="sync"
COMMAND_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --code-only)
      SYNC_DATA=0
      shift
      ;;
    --data-only)
      SYNC_CODE=0
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --no-delete)
      NO_DELETE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    sync|run|ssh|pull|batch|batch-stop|batch-multi)
      COMMAND="$1"
      shift
      COMMAND_ARGS=("$@")
      break
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

case "${COMMAND}" in
  sync)
    [[ "${SYNC_CODE}" == "1" ]] && sync_code
    [[ "${SYNC_DATA}" == "1" ]] && sync_data
    echo "==> Sync complete."
    ;;
  run)
    [[ "${SYNC_CODE}" == "1" ]] && sync_code
    [[ "${SYNC_DATA}" == "1" ]] && sync_data
    run_remote "${COMMAND_ARGS[@]}"
    ;;
  ssh)
    [[ "${SYNC_CODE}" == "1" ]] && sync_code
    [[ "${SYNC_DATA}" == "1" ]] && sync_data
    open_ssh
    ;;
  pull)
    pull_derivatives
    echo "==> Pull complete."
    ;;
  batch)
    [[ "${SYNC_CODE}" == "1" ]] && sync_code
    [[ "${SYNC_DATA}" == "1" ]] && sync_data
    run_remote_blocking "${COMMAND_ARGS[@]}"
    pipeline_exit_code=$?
    pull_derivatives
    if [[ ${pipeline_exit_code} -ne 0 ]]; then
      echo "==> Batch complete: pipeline failed, results pulled (exit=${pipeline_exit_code})." >&2
      exit ${pipeline_exit_code}
    fi
    echo "==> Batch complete: sync → run → pull finished."
    ;;
  batch-stop)
    [[ "${SYNC_CODE}" == "1" ]] && sync_code
    [[ "${SYNC_DATA}" == "1" ]] && sync_data
    run_remote_blocking "${COMMAND_ARGS[@]}"
    PIPELINE_EXIT_CODE=$?
    pull_derivatives
    stop_gce_instance
    if [[ ${PIPELINE_EXIT_CODE} -ne 0 ]]; then
      echo "==> Batch-stop complete: pipeline failed, results pulled (exit=${PIPELINE_EXIT_CODE})." >&2
      exit ${PIPELINE_EXIT_CODE}
    fi
    echo "==> Batch-stop complete: sync → run → pull → stop finished."
    ;;
  batch-multi)
    if [[ ${#COMMAND_ARGS[@]} -lt 2 ]]; then
      echo "Usage: gcp.sh batch-multi <pipeline_command> <subject1> [subject2] ..." >&2
      echo "Example: gcp.sh batch-multi 'features compute' 0001 0002 0003 0004" >&2
      exit 1
    fi
    pipeline_cmd="${COMMAND_ARGS[0]}"
    subjects=("${COMMAND_ARGS[@]:1}")
    n_subjects=${#subjects[@]}
    
    [[ "${SYNC_CODE}" == "1" ]] && sync_code
    [[ "${SYNC_DATA}" == "1" ]] && sync_data
    
    echo "==> Running ${n_subjects} subjects in parallel (max ${PARALLEL_SUBJECTS} concurrent)..."
    
    prefix="cd ${REMOTE_BASE}"
    if [[ -n "${REMOTE_VENV_ACTIVATE}" ]]; then
      prefix="${prefix} && source ${REMOTE_VENV_ACTIVATE}"
    fi
    prefix="${prefix} && export EEG_PIPELINE_N_JOBS=${REMOTE_N_JOBS} && export MNE_N_JOBS=${REMOTE_N_JOBS} && export PYTHONPATH=${REMOTE_BASE}:\${PYTHONPATH:-}"

    subjects_str="${subjects[*]}"

    # Run multiple subjects concurrently on the remote using xargs -P.
    # We avoid embedding the pipeline command in a single-quoted SSH string to prevent local shell quoting issues.
    echo "==> Remote: running subjects: ${subjects_str}"
    ssh "${REMOTE_TARGET}" bash -lc \
      "${prefix} && printf '%s\\n' ${subjects_str} | xargs -P ${PARALLEL_SUBJECTS} -I {} ${REMOTE_PYTHON} -m eeg_pipeline ${pipeline_cmd} --subject {}"
    pipeline_exit_code=$?
    
    pull_derivatives
    echo "==> Batch-multi complete (exit=${pipeline_exit_code})."
    exit ${pipeline_exit_code}
    ;;
esac
