#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${SCRIPT_DIR}/alliance_env.sh"

if [[ ! -f "${ENV_FILE}" ]]; then
    echo "Missing environment file: ${ENV_FILE}" >&2
    exit 1
fi

source "${ENV_FILE}"

required_vars=(
    REPO_ROOT
    EEG_PIPELINE_VENV
    FMRIPREP_IMAGE
    FS_LICENSE_FILE
    TEMPLATEFLOW_HOME
)

for var_name in "${required_vars[@]}"; do
    if [[ -z "${!var_name:-}" ]]; then
        echo "Required variable is empty: ${var_name}" >&2
        exit 1
    fi
done

if [[ ! -d "${REPO_ROOT}" ]]; then
    echo "Repository root does not exist: ${REPO_ROOT}" >&2
    exit 1
fi

if [[ ! -f "${FS_LICENSE_FILE}" ]]; then
    echo "FreeSurfer license file is missing: ${FS_LICENSE_FILE}" >&2
    echo "Copy license.txt there before running fMRIPrep." >&2
    exit 1
fi

# shellcheck disable=SC2086
module load ${ALLIANCE_MODULES}

if [[ ! -d "${EEG_PIPELINE_VENV}" ]]; then
    python -m venv "${EEG_PIPELINE_VENV}"
fi

source "${EEG_PIPELINE_VENV}/bin/activate"
python -m pip install --upgrade pip
python -m pip install -e "${REPO_ROOT}[dev,ml]"

mkdir -p "$(dirname "${FMRIPREP_IMAGE}")"
if [[ ! -f "${FMRIPREP_IMAGE}" ]]; then
    apptainer pull "${FMRIPREP_IMAGE}" docker://nipreps/fmriprep:25.2.5
fi

python -m eeg_pipeline --help >/dev/null
env -u REQUESTS_CA_BUNDLE -u SSL_CERT_FILE -u CURL_CA_BUNDLE \
    apptainer exec --cleanenv "${FMRIPREP_IMAGE}" fmriprep --version

bash "${SCRIPT_DIR}/prefetch_templateflow.sh"
