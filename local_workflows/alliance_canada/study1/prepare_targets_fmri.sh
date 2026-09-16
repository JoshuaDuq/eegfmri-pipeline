#!/usr/bin/env bash
# fMRI-only Study 1 stage: trial-wise NPS/SIIPS1 signature targets.
#
# Deliberately NOT study1/prepare.sh, which also runs prepare-features. That stage
# needs clean EEG epochs, and sub-0016..0021 have none, so it would fail the job.
#
# Scoring happens in MNI152NLin6Asym: the CANlab NPS/SIIPS1 weights are distributed
# on SPM/FSL MNI152 grids, so the BOLD data is brought to the weights rather than
# the published weight maps being warped. The scoring mask must match that space.
set -euo pipefail

# Slurm copies the batch script into a spool directory, so BASH_SOURCE points at
# /localscratch/... and cannot locate the workflow tree. The submitter exports
# ALLIANCE_WORKFLOW_ROOT; fall back to BASH_SOURCE only when run directly.
if [[ -z "${ALLIANCE_WORKFLOW_ROOT:-}" ]]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    ALLIANCE_WORKFLOW_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
fi
if [[ ! -f "${ALLIANCE_WORKFLOW_ROOT}/alliance_env.sh" ]]; then
    echo "Cannot locate alliance_env.sh under ALLIANCE_WORKFLOW_ROOT=${ALLIANCE_WORKFLOW_ROOT}" >&2
    exit 2
fi
source "${ALLIANCE_WORKFLOW_ROOT}/alliance_env.sh"

for variable_name in STUDY1_RUN_ID TASK STUDY1_CONFIG BIDS_EEG_ROOT BIDS_FMRI_ROOT \
    FMRIPREP_DERIV_ROOT SIGNATURE_ROOT EEG_PIPELINE_VENV REPO_ROOT; do
    if [[ -z "${!variable_name:-}" ]]; then
        echo "Required variable is empty: ${variable_name}" >&2
        exit 2
    fi
done

cd "${REPO_ROOT}"
# shellcheck disable=SC2086
module load ${ALLIANCE_MODULES}
source "${EEG_PIPELINE_VENV}/bin/activate"
export NUMBA_CACHE_DIR="${ALLIANCE_SCRATCH_ROOT}/study1_numba_cache"
mkdir -p "${NUMBA_CACHE_DIR}"

# All 21 fMRIPrep subjects minus sub-0006 (pilot participant, excluded 2026-06-30).
# sub-0002 never existed in BIDS; pilots are outside the analytic sample.
# Trial events come from the fMRI BIDS events (events_source=fmri_bids), not the
# clean EEG table: EEG artifact rejection drops sound BOLD trials for a reason
# unrelated to BOLD quality, and it left 10 of 14 subjects with 53-65 of 66 trials.
SUBJECT_ARGS=()
for subject_id in 0000 0001 0003 0004 0005 0007 0008 0009 0010 0011 0012 0013 0014 \
                  0015 0016 0017 0018 0019 0020 0021; do
    if [[ "${subject_id}" == "0006" ]]; then
        echo "Excluded participant 0006 present in subject list." >&2
        exit 2
    fi
    SUBJECT_ARGS+=(--subject "${subject_id}")
done
if [[ "${#SUBJECT_ARGS[@]}" -ne 40 ]]; then
    echo "Expected 20 subjects (40 args); got ${#SUBJECT_ARGS[@]}." >&2
    exit 2
fi

SIGNATURE_MAPS_JSON='[{"name":"NPS","path":"NPS/weights_NSF_grouppred_cvpcr.nii.gz"},{"name":"SIIPS1","path":"SIIPS1/nonnoc_v11_4_137subjmap_weighted_mean.nii.gz"}]'
COMMON_ARGS=(
    --task "${TASK}"
    --study1-config "${STUDY1_CONFIG}"
    --bids-root "${BIDS_EEG_ROOT}"
    --bids-fmri-root "${BIDS_FMRI_ROOT}"
    --deriv-root "${FMRIPREP_DERIV_ROOT}"
    --set "paths.signature_dir=${SIGNATURE_ROOT}"
    --set "paths.signature_maps=${SIGNATURE_MAPS_JSON}"
    --set "study1.targets.signature_manifest_path=signature_manifest.yaml"
    --set "study1.targets.signature_provenance.NPS.path=NPS/weights_NSF_grouppred_cvpcr.nii.gz"
    --set "study1.targets.signature_provenance.SIIPS1.path=SIIPS1/nonnoc_v11_4_137subjmap_weighted_mean.nii.gz"
    --set "study1.targets.signature_scoring_mask_path=tpl-MNI152NLin6Asym_res-02_desc-brain_mask.nii.gz"
    --set "study1.targets.events_source=fmri_bids"
    --set "study1.outputs.root_name=${STUDY1_RUN_ID}"
)

# The generated bin/eeg-pipeline console script on the cluster predates the
# signature-prediction command; the editable install exposes it via -m.
python -m eeg_pipeline signature-prediction prepare-targets \
    "${SUBJECT_ARGS[@]}" "${COMMON_ARGS[@]}"

targets="${FMRIPREP_DERIV_ROOT}/group/multimodal/${STUDY1_RUN_ID}/targets/primary_targets.parquet"
if [[ ! -f "${targets}" ]]; then
    echo "prepare-targets finished but ${targets} is missing." >&2
    exit 3
fi
echo "Study 1 targets written: ${targets}"
