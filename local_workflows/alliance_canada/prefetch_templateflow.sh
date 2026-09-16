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
    FMRIPREP_IMAGE
    TEMPLATEFLOW_HOME
)

for var_name in "${required_vars[@]}"; do
    if [[ -z "${!var_name:-}" ]]; then
        echo "Required variable is empty: ${var_name}" >&2
        exit 1
    fi
done

if [[ ! -f "${FMRIPREP_IMAGE}" ]]; then
    echo "fMRIPrep image does not exist: ${FMRIPREP_IMAGE}" >&2
    exit 1
fi

module load StdEnv/2023
module load apptainer

mkdir -p "${TEMPLATEFLOW_HOME}"

env -u REQUESTS_CA_BUNDLE -u SSL_CERT_FILE -u CURL_CA_BUNDLE \
    apptainer exec \
    --cleanenv \
    -B "${TEMPLATEFLOW_HOME}:${TEMPLATEFLOW_HOME}" \
    --env "TEMPLATEFLOW_HOME=${TEMPLATEFLOW_HOME}" \
    "${FMRIPREP_IMAGE}" \
    python - <<'PY'
import templateflow.api as tf

template_requests = {
    "MNI152NLin2009cAsym": [
        dict(resolution=1, suffix="T1w", extension="nii.gz"),
        dict(resolution=2, suffix="T1w", extension="nii.gz"),
        dict(resolution=1, suffix="T2w", extension="nii.gz"),
        dict(resolution=2, suffix="T2w", extension="nii.gz"),
        dict(resolution=1, desc="brain", suffix="mask", extension="nii.gz"),
        dict(resolution=2, desc="brain", suffix="mask", extension="nii.gz"),
        dict(resolution=1, desc="carpet", suffix="dseg", extension="nii.gz"),
        dict(resolution=2, desc="fMRIPrep", suffix="boldref", extension="nii.gz"),
        dict(resolution=1, label="brain", suffix="probseg", extension="nii.gz"),
        dict(resolution=1, suffix="probseg", label=["CSF", "GM", "WM"], extension="nii.gz"),
        dict(resolution=2, suffix="probseg", label=["CSF", "GM", "WM"], extension="nii.gz"),
    ],
    # Signature scoring space: the CANlab NPS/SIIPS1 weights are distributed on
    # SPM/FSL MNI152 grids, so the BOLD data is resampled here rather than the
    # published weight maps being warped. Compute nodes have no outbound network,
    # so every asset fMRIPrep needs for this space must be prefetched.
    "MNI152NLin6Asym": [
        dict(resolution=1, suffix="T1w", extension="nii.gz"),
        dict(resolution=2, suffix="T1w", extension="nii.gz"),
        dict(resolution=1, desc="brain", suffix="mask", extension="nii.gz"),
        dict(resolution=2, desc="brain", suffix="mask", extension="nii.gz"),
        dict(resolution=1, desc="carpet", suffix="dseg", extension="nii.gz"),
        dict(resolution=2, desc="fMRIPrep", suffix="boldref", extension="nii.gz"),
        dict(resolution=1, label="brain", suffix="probseg", extension="nii.gz"),
        dict(resolution=1, suffix="probseg", label=["CSF", "GM", "WM"], extension="nii.gz"),
        dict(resolution=2, suffix="probseg", label=["CSF", "GM", "WM"], extension="nii.gz"),
    ],
    "OASIS30ANTs": [
        dict(resolution=1, suffix="T1w", extension="nii.gz"),
        dict(resolution=1, label="brain", suffix="probseg", extension="nii.gz"),
        dict(
            resolution=1,
            desc="BrainCerebellumExtraction",
            suffix="mask",
            extension="nii.gz",
        ),
        dict(resolution=1, label="WM", suffix="probseg", extension="nii.gz"),
        dict(resolution=1, label="BS", suffix="probseg", extension="nii.gz"),
    ],
}

for template, requests in template_requests.items():
    for query in requests:
        print(f"Fetching TemplateFlow {template}: {query}", flush=True)
        tf.get(template, **query)

print("TemplateFlow prefetch complete.", flush=True)
PY
