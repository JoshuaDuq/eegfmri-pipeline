#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_ENV="${SCRIPT_DIR}/local_env.sh"

if [[ ! -f "${LOCAL_ENV}" ]]; then
    echo "Missing local environment file: ${LOCAL_ENV}" >&2
    echo "Create it from ${SCRIPT_DIR}/local_env.example.sh" >&2
    exit 1
fi

source "${LOCAL_ENV}"
source "${SCRIPT_DIR}/alliance_env.sh"

if ssh -S "${ALLIANCE_SSH_CONTROL_PATH}" -O check "${ALLIANCE_HOST}" 2>/dev/null; then
    echo "${ALLIANCE_CLUSTER} SSH control connection is already active."
    exit 0
fi

ssh \
    -M \
    -S "${ALLIANCE_SSH_CONTROL_PATH}" \
    -o StrictHostKeyChecking=accept-new \
    -o ControlPersist=4h \
    -Nf \
    "${ALLIANCE_HOST}"

ssh -S "${ALLIANCE_SSH_CONTROL_PATH}" -O check "${ALLIANCE_HOST}"
echo "${ALLIANCE_CLUSTER} SSH control connection is active."
