#!/usr/bin/env bash

export ALLIANCE_HOST="joshduq@rorqual.alliancecan.ca"
export ALLIANCE_SSH_CONTROL_PATH="/tmp/rorqual-joshduq-ssh-control"
export ALLIANCE_ACCOUNT="def-mpcoll"
export ALLIANCE_PROJECT_ROOT="/project/def-mpcoll/joshduq"
export ALLIANCE_SCRATCH_ROOT="/scratch/joshduq"
export ALLIANCE_MODULES="StdEnv/2023 python/3.11 gcc arrow"

export FMRIPREP_SLURM_MEMORY="700G"
export FMRIPREP_MEM_MB="680000"
export STUDY1_PREPARE_SLURM_MEMORY="64G"
export STUDY1_DISPATCH_SLURM_MEMORY="4G"
export STUDY1_CELL_SLURM_MEMORY="64G"
export STUDY1_REPORT_SLURM_MEMORY="64G"

