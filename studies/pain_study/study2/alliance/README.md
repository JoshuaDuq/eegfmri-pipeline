# Study 2 on Alliance Canada

This directory contains the Alliance Canada/Slurm workflow for running Study 2
from anatomical reconstruction through source-family inference. It deliberately
does not use the local Docker BEM helper: Docker is not available on Alliance
clusters, so these jobs run with Alliance modules, a Python virtual environment,
and Slurm job dependencies.

The workflow follows Alliance conventions documented for modules, Python
wheelhouse installs, Slurm submission/accounting, and shared storage. See the
reference links at the end of this README.

## What This Runs

The workflow submits these stages:

1. FreeSurfer `recon-all` for every subject.
2. MNE watershed BEM, BEM solution, and EEG-to-MRI transform from BIDS EEG
   digitization points.
3. Dense source-space adjacency for the Study 2 cluster test.
4. Per-subject sLORETA source-power extraction for alpha, beta, and gamma.
5. Study 2 gate, source-stage association maps, target-retrained null maps, and
   source-family inference.

The scripts fail if required inputs are missing. They do not create identity
transforms or silently substitute anatomy.

## Required Inputs

Edit `study2_alliance.env` so these paths are real on the cluster:

- `STUDY2_BIDS_MRI_ROOT`: BIDS anatomical root with
  `sub-*/anat/sub-*_T1w.nii.gz`.
- `STUDY2_BIDS_EEG_ROOT`: BIDS EEG root with CapTrak/digitization fiducials.
- `STUDY2_DERIV_ROOT`: derivatives root containing cleaned EEG epochs and
  Study 1 outputs.
- `STUDY2_FS_LICENSE`: FreeSurfer license file.
- `STUDY2_SUBJECTS_FILE`: one `sub-*` identifier per line.
- `STUDY2_CONFIG`: Study 2 YAML config.

Study 2 group stages also require the Study 1 artifacts consumed by the runner:

- `<deriv_root>/group/multimodal/study1/reports/study1_report.tsv`
- the Study 1 primary model-comparison table used by
  `studies.pain_study.study2.study1_context.study1_model_comparison_path`
- `<deriv_root>/group/multimodal/study2/source_stage/source_stage_input.tsv`

The last file must contain the standardized Study 1 contribution score columns
and Level 2 nuisance columns declared in `study2_config.yaml`. The source-stage
job will fail with the missing path or missing columns if it is not present.

## First-Time Setup

Run these commands on an Alliance login node:

```bash
cd /project/def-yourpi/$USER/EEG_fMRI_Pipeline
cp studies/pain_study/study2/alliance/study2_alliance.env.example \
   studies/pain_study/study2/alliance/study2_alliance.env
nano studies/pain_study/study2/alliance/study2_alliance.env
```

Check module names on your cluster before bootstrapping:

```bash
module spider python
module spider scipy-stack
module spider freesurfer
```

Then create the Python environment:

```bash
bash studies/pain_study/study2/alliance/bin/bootstrap_alliance_env.sh \
  studies/pain_study/study2/alliance/study2_alliance.env
```

The bootstrap uses Alliance's wheelhouse with `pip --no-index`. If dependency
installation fails, fix the module stack or package availability before
submitting compute jobs.

## Run One Subject First

Before submitting the full cohort, run one array element for each heavy stage:

```bash
ENV=studies/pain_study/study2/alliance/study2_alliance.env
ACCOUNT=def-yourpi

sbatch --account "$ACCOUNT" --array=1-1 \
  studies/pain_study/study2/alliance/sbatch/01_recon_all.sbatch "$ENV"
```

After `recon-all` completes successfully, test BEM/trans and source-power:

```bash
sbatch --account "$ACCOUNT" --array=1-1 \
  studies/pain_study/study2/alliance/sbatch/02_bem_trans.sbatch "$ENV"

sbatch --account "$ACCOUNT" --array=1-1 \
  studies/pain_study/study2/alliance/sbatch/04_source_power.sbatch "$ENV"
```

Inspect logs in `studies/pain_study/study2/alliance/logs/slurm/`. A successful
source-power test writes:

```text
<deriv_root>/group/multimodal/study2/sub-XXXX/eeg/source/source_power_alpha.npy
<deriv_root>/group/multimodal/study2/sub-XXXX/eeg/source/source_power_beta.npy
<deriv_root>/group/multimodal/study2/sub-XXXX/eeg/source/source_power_gamma.npy
```

## Submit the Full Workflow

After the one-subject test passes:

```bash
bash studies/pain_study/study2/alliance/run_full_study2_alliance.sh \
  studies/pain_study/study2/alliance/study2_alliance.env
```

The wrapper submits job arrays and dependencies in the correct order. It passes
the Alliance account via `sbatch --account`, so `#SBATCH --account` is not
hard-coded inside the scripts.

`target-permutations` is submitted with a 72-hour walltime override because the
configured null uses 1,000 valid target-retrained draws. After the first cohort
run, use `sacct` `MaxRSS` and `Elapsed` to tune memory and walltime requests.

## Monitor and Audit

Use Slurm accounting to check status and memory:

```bash
squeue -u "$USER"
sacct -j <job_id> --format=JobID,JobName,State,Elapsed,MaxRSS
```

The source-power job currently requests `80G` and limits concurrent source-power
subjects with `STUDY2_SOURCE_POWER_ARRAY_LIMIT`. Increase or decrease that limit
only after checking `MaxRSS` on completed jobs.

## Expected Final Outputs

The final Study 2 products are:

```text
<deriv_root>/group/multimodal/study2/gate/gate_qc.json
<deriv_root>/group/multimodal/study2/source_stage/fisher_z_alpha.npy
<deriv_root>/group/multimodal/study2/source_stage/fisher_z_beta.npy
<deriv_root>/group/multimodal/study2/source_stage/fisher_z_gamma.npy
<deriv_root>/group/multimodal/study2/inference/adjacency.npy
<deriv_root>/group/multimodal/study2/inference/null_alpha.npy
<deriv_root>/group/multimodal/study2/inference/null_beta.npy
<deriv_root>/group/multimodal/study2/inference/null_gamma.npy
<deriv_root>/group/multimodal/study2/inference/source_family_summary.tsv
```

## Operational Notes

- Keep raw BIDS and long-lived derivatives under project storage.
- Use scratch for `STUDY2_SUBJECTS_DIR` if the cluster policy permits; copy the
  final FreeSurfer outputs back to project storage if they must be retained.
- Do not run `recon-all`, BEM generation, source-power, or permutation jobs on
  login nodes.
- If BIDS EEG has multiple runs for a task, set `STUDY2_EEG_BIDS_RUN` so
  transform generation selects exactly one recording.

## Alliance References

- Alliance available software and modules:
  <https://docs.alliancecan.ca/wiki/Available_software>
- Alliance job submission and Slurm monitoring:
  <https://docs.computecanada.ca/wiki/Running_jobs>
- Alliance filesystem guidance:
  <https://docs.alliancecan.ca/mediawiki/images/9/99/File_Systems.pdf>
