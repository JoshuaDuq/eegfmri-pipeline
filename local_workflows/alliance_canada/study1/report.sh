#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_args.sh"

"${EEG_PIPELINE_VENV}/bin/eeg-pipeline" signature-prediction report \
    "${SUBJECT_ARGS[@]}" "${COMMON_ARGS[@]}"
