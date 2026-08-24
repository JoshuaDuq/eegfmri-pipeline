#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/load_cluster_profile.sh"

export ALLIANCE_TIME="06:00:00"
export ALLIANCE_CPUS="16"

# Repository and Python environment on the cluster.
export REPO_ROOT="${ALLIANCE_PROJECT_ROOT}/EEG_fMRI_Pipeline"
export EEG_PIPELINE_VENV="${ALLIANCE_PROJECT_ROOT}/venvs/eeg_fmri_pipeline"

# Study data paths on the cluster.
export BIDS_FMRI_ROOT="${ALLIANCE_PROJECT_ROOT}/bids/fmri"
export BIDS_EEG_ROOT="${ALLIANCE_PROJECT_ROOT}/bids/eeg"
export DERIV_ROOT="${ALLIANCE_PROJECT_ROOT}/derivatives"
export FMRIPREP_DERIV_ROOT="${ALLIANCE_SCRATCH_ROOT}/derivatives"
export FMRIPREP_WORK_ROOT="${ALLIANCE_SCRATCH_ROOT}/fmriprep_work"
export FMRIPREP_LOG_ROOT="${ALLIANCE_SCRATCH_ROOT}/fmriprep_logs"
export TEMPLATEFLOW_HOME="${ALLIANCE_SCRATCH_ROOT}/templateflow"
export STUDY1_LOG_ROOT="${ALLIANCE_SCRATCH_ROOT}/study1_logs"
export STUDY1_RUNTIME_ROOT="${ALLIANCE_SCRATCH_ROOT}/study1_runtime"
export STUDY2_SUBJECTS_DIR="${ALLIANCE_SCRATCH_ROOT}/study2_freesurfer_subjects"
export STUDY2_LOG_ROOT="${ALLIANCE_SCRATCH_ROOT}/study2_logs"
export STUDY2_ENV_FILE="${REPO_ROOT}/studies/pain_study/study2/alliance/study2_alliance.env"
export STUDY2_TASK="thermalactive"
export STUDY1_OUTPUT_ROOT_NAME="study1"
export STUDY2_OUTPUT_ROOT_NAME="study2"
export STUDY2_RECON_ARRAY_LIMIT="8"
export STUDY2_BEM_ARRAY_LIMIT="8"
export STUDY2_SOURCE_POWER_ARRAY_LIMIT="2"
export STUDY2_ADJACENCY_SUBJECT="sub-0001"
export STUDY2_EXPECTED_VERTICES="8196"

# Required by fMRIPrep.
export FS_LICENSE_FILE="${ALLIANCE_PROJECT_ROOT}/licenses/license.txt"

# Prefer a pre-pulled .sif path. A docker:// URI is acceptable only when the
# cluster permits Apptainer image pulls from compute jobs.
export FMRIPREP_IMAGE="${ALLIANCE_PROJECT_ROOT}/containers/fmriprep_25.2.5.sif"
export SIGNATURE_ROOT="${ALLIANCE_PROJECT_ROOT}/external"

# fMRIPrep task restriction used by the narrow upload manifest.
export FMRIPREP_TASK_ID="thermalactive"

# Optional output spaces. Keep these aligned with downstream analysis configs.
export FMRIPREP_OUTPUT_SPACES="MNI152NLin2009cAsym T1w"
