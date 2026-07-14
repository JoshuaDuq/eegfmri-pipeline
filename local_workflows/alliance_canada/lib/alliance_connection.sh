#!/usr/bin/env bash

require_alliance_connection() {
    if ! ssh \
        -S "${ALLIANCE_SSH_CONTROL_PATH}" \
        -O check \
        "${ALLIANCE_HOST}" >/dev/null 2>&1; then
        echo "No active ${ALLIANCE_CLUSTER} SSH control connection." >&2
        echo "Run: bash ${SCRIPT_DIR}/start_alliance_connection.sh" >&2
        return 1
    fi
}
