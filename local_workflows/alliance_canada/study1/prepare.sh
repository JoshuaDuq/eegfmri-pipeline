#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common_args.sh"

EEG_PIPELINE="${EEG_PIPELINE_VENV}/bin/eeg-pipeline"
"${EEG_PIPELINE}" signature-prediction prepare-targets \
    "${SUBJECT_ARGS[@]}" "${COMMON_ARGS[@]}"
"${EEG_PIPELINE}" signature-prediction prepare-features \
    "${SUBJECT_ARGS[@]}" "${COMMON_ARGS[@]}"
