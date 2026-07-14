# Alliance Canada Workflows

All workflows use one required cluster selector:

```bash
export ALLIANCE_CLUSTER=rorqual  # or trillium
```

Copy `local_env.example.sh` to the ignored personal file `local_env.sh`, verify
the KINGSTON and FreeSurfer-license paths, and keep the selector there. Invalid
or missing cluster names fail immediately.

## Connection

```bash
bash local_workflows/alliance_canada/start_alliance_connection.sh
bash local_workflows/alliance_canada/stop_alliance_connection.sh
```

The profile controls the host, SSH socket, allocation, storage roots, modules,
and Slurm memory policy.

## fMRIPrep

Put one participant ID per line in the ignored personal `subjects.txt`, then:

```bash
bash local_workflows/alliance_canada/setup_alliance_fmriprep.sh
bash local_workflows/alliance_canada/setup_alliance_runtime.sh
bash local_workflows/alliance_canada/submit_fmriprep_alliance.sh
```

Retrieve completed output with `fetch_fmriprep_outputs.sh`.

## Study 1

The tracked cohort is `study1_subjects.txt`; it contains 13 participants and
excludes `0006`.

```bash
bash local_workflows/alliance_canada/fetch_study1_fmriprep_from_trillium.sh
bash local_workflows/alliance_canada/setup_alliance_study1.sh
bash local_workflows/alliance_canada/setup_alliance_runtime.sh
bash local_workflows/alliance_canada/submit_study1_alliance.sh
```

The workflow submits `prepare`, a dependent dispatcher, only the missing
benchmark cells, and a final dependent report. Run records and logs are under
`/scratch/$USER/study1_logs`.

## Study 2

```bash
bash local_workflows/alliance_canada/setup_alliance_study2.sh
bash local_workflows/alliance_canada/setup_alliance_study2_runtime.sh
bash local_workflows/alliance_canada/submit_study2_alliance.sh
```

`write_study2_env.sh` generates cluster paths from the selected profile.

## Monitoring

```bash
squeue -u "$USER"
sacct -j <job_ids> --format=JobID,JobName,State,Elapsed,ExitCode,MaxRSS
```
