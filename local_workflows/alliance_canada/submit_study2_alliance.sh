#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/local_env.sh"
source "${SCRIPT_DIR}/alliance_env.sh"
source "${SCRIPT_DIR}/lib/alliance_connection.sh"

require_alliance_connection
ssh -o ControlPath="${ALLIANCE_SSH_CONTROL_PATH}" -o BatchMode=yes "${ALLIANCE_HOST}" \
    "cd '${REPO_ROOT}' && bash studies/pain_study/study2/alliance/run_full_study2_alliance.sh '${STUDY2_ENV_FILE}'"
