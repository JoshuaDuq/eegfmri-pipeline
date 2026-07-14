#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_ENV="${SCRIPT_DIR}/local_env.sh"

if [[ ! -f "${LOCAL_ENV}" ]]; then
    echo "Missing local environment file: ${LOCAL_ENV}" >&2
    exit 1
fi

source "${LOCAL_ENV}"
source "${SCRIPT_DIR}/alliance_env.sh"

ssh -S "${ALLIANCE_SSH_CONTROL_PATH}" -O exit "${ALLIANCE_HOST}"
