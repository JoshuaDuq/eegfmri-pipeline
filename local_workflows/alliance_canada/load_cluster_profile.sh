#!/usr/bin/env bash

PROFILE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case "${ALLIANCE_CLUSTER:-}" in
    rorqual|trillium)
        ;;
    *)
        echo "Unsupported ALLIANCE_CLUSTER: ${ALLIANCE_CLUSTER:-<empty>}" >&2
        return 1 2>/dev/null || exit 1
        ;;
esac

source "${PROFILE_DIR}/clusters/${ALLIANCE_CLUSTER}.sh"

required_profile_vars=(
    ALLIANCE_HOST
    ALLIANCE_SSH_CONTROL_PATH
    ALLIANCE_ACCOUNT
    ALLIANCE_PROJECT_ROOT
    ALLIANCE_SCRATCH_ROOT
    ALLIANCE_MODULES
    FMRIPREP_MEM_MB
)

for var_name in "${required_profile_vars[@]}"; do
    if [[ -z "${!var_name:-}" ]]; then
        echo "Cluster profile variable is empty: ${var_name}" >&2
        return 1 2>/dev/null || exit 1
    fi
done

expected_host="joshduq@${ALLIANCE_CLUSTER}.alliancecan.ca"
if [[ "${ALLIANCE_HOST}" != "${expected_host}" ]]; then
    echo "Cluster profile host mismatch: ${ALLIANCE_HOST} != ${expected_host}" >&2
    return 1 2>/dev/null || exit 1
fi
