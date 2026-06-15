#!/usr/bin/env bash

# Required Slurm allocation. Ask your supervisor for the exact account string.
export ALLIANCE_ACCOUNT="def-mpcoll"

# Resource request. Trillium compute jobs must request less than 24 hours and
# must not pass --mem because compute nodes grant all available node memory.
export ALLIANCE_TIME="06:00:00"
export ALLIANCE_CPUS="16"

# Repository and Python environment on the cluster.
export REPO_ROOT="/project/def-mpcoll/joshduq/EEG_fMRI_Pipeline"
export EEG_PIPELINE_VENV="/project/def-mpcoll/joshduq/venvs/eeg_fmri_pipeline"

# Study data paths on the cluster.
export BIDS_FMRI_ROOT="/project/def-mpcoll/joshduq/bids/fmri"
export BIDS_EEG_ROOT="/project/def-mpcoll/joshduq/bids/eeg"
export DERIV_ROOT="/project/def-mpcoll/joshduq/derivatives"
export FMRIPREP_DERIV_ROOT="/scratch/joshduq/derivatives"
export FMRIPREP_WORK_ROOT="/scratch/joshduq/fmriprep_work"
export FMRIPREP_LOG_ROOT="/scratch/joshduq/fmriprep_logs"
export FMRIPREP_MEM_MB="700000"
export TEMPLATEFLOW_HOME="/scratch/joshduq/templateflow"
export STUDY2_SUBJECTS_DIR="/scratch/joshduq/study2_freesurfer_subjects"
export STUDY2_LOG_ROOT="/scratch/joshduq/study2_logs"
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
export FS_LICENSE_FILE="/project/def-mpcoll/joshduq/licenses/license.txt"

# Prefer a pre-pulled .sif path. A docker:// URI is acceptable only when the
# cluster permits Apptainer image pulls from compute jobs.
export FMRIPREP_IMAGE="/project/def-mpcoll/joshduq/containers/fmriprep_25.2.4.sif"

# fMRIPrep task restriction used by the narrow upload manifest.
export FMRIPREP_TASK_ID="thermalactive"

# Optional output spaces. Keep these aligned with downstream analysis configs.
export FMRIPREP_OUTPUT_SPACES="MNI152NLin2009cAsym T1w"
