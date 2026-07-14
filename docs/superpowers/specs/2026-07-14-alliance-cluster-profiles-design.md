# Alliance Cluster Profiles Design

## Objective

Provide one explicit `ALLIANCE_CLUSTER=rorqual|trillium` setting that selects the
Alliance Canada login host, SSH control socket, scheduler resource policy, remote
paths, setup behavior, submission behavior, monitoring, and output retrieval for
fMRIPrep, Study 1, and Study 2.

The immediate operational target is Rorqual while Trillium compute nodes are shut
down from July 13 through July 24, 2026. The design must remain usable after the
maintenance period without editing workflow code.

## Current State

The workflow currently exposes `ALLIANCE_LOGIN_HOST`, but the implementation is
not cluster-selectable:

- variables and entrypoints are named `RORQUAL_*` while defaults point to Trillium;
- Trillium's whole-node scheduling and no-`--mem` rule are hard-coded;
- project and scratch paths are treated as if their contents were shared;
- Study 1 submission exists only as scripts generated manually in Trillium scratch;
- Rorqual currently has the repository, virtual environment, container, license,
  project derivatives, and eight fMRI BIDS subjects, but lacks EEG BIDS, external
  signature assets, scratch derivatives, and Study 1 logs;
- KINGSTON has complete EEG BIDS and EEG derivatives for the 13-subject Study 1
  cohort, but fMRIPrep outputs for `sub-0009`, `sub-0012`, `sub-0013`, and
  `sub-0014` currently exist only on Trillium storage.

Rorqual access and the `def-mpcoll` allocation were verified. Its standard CPU
nodes expose 192 cores and approximately 768 GB RAM, but unlike Trillium they
support by-core allocation and therefore require explicit memory requests for
memory-heavy workflows.

## Considered Approaches

### Selected: Explicit shell profiles

Store one profile per supported cluster and load exactly one profile from the
single `ALLIANCE_CLUSTER` setting. Export generic variable names to all workflow
scripts.

This keeps SSH and bootstrap operations in Bash, validates configuration before
network or scheduler actions, and makes scheduler differences visible.

### Rejected: Hostname-only override

Changing only the SSH hostname leaves Trillium-specific sockets, memory rules,
scratch assumptions, and incomplete Rorqual storage in place. This can connect to
Rorqual but cannot run the workflows correctly.

### Rejected: YAML profile parsed by Python

A YAML source would be structured, but every local and remote shell bootstrap
would need a working Python environment before it could resolve the host or create
that environment. The additional bootstrap layer provides no benefit for two
small, static cluster profiles.

## Configuration Architecture

`local_workflows/alliance_canada/local_env.sh` contains local filesystem roots and
the explicit cluster selection:

```bash
export ALLIANCE_CLUSTER="${ALLIANCE_CLUSTER:-rorqual}"
```

`local_workflows/alliance_canada/load_cluster_profile.sh` validates the selection,
sources `clusters/${ALLIANCE_CLUSTER}.sh`, and validates every exported profile
field. Unsupported or empty selections fail immediately.

Profiles export generic variables only:

```text
ALLIANCE_CLUSTER
ALLIANCE_HOST
ALLIANCE_SSH_CONTROL_PATH
ALLIANCE_ACCOUNT
ALLIANCE_PROJECT_ROOT
ALLIANCE_SCRATCH_ROOT
ALLIANCE_MODULES
FMRIPREP_SLURM_MEMORY
FMRIPREP_MEM_MB
STUDY1_PREPARE_SLURM_MEMORY
STUDY1_CELL_SLURM_MEMORY
STUDY1_REPORT_SLURM_MEMORY
```

An empty Slurm memory value means the selected cluster grants whole-node memory
and the submission command must omit `--mem`. A non-empty value must be passed
explicitly. No script infers behavior from the hostname.

`alliance_env.sh` derives repository, BIDS, derivative, container, license, cache,
log, and output paths from the selected profile roots. It contains no
cluster-specific conditionals.

The old `RORQUAL_HOST` and `RORQUAL_SSH_CONTROL_PATH` variables are removed. The
generic connection and setup entrypoints are named for Alliance rather than one
cluster:

```text
start_alliance_connection.sh
stop_alliance_connection.sh
setup_alliance_fmriprep.sh
setup_alliance_runtime.sh
setup_alliance_study1.sh
setup_alliance_study2.sh
```

Old Rorqual-named generic entrypoints are removed rather than retained as aliases.

## Data Staging

KINGSTON is the canonical local source. Cluster setup never assumes that project
or scratch content is shared between Alliance systems.

Each workflow has a dedicated subject list and narrow staging contract:

- fMRIPrep uploads raw fMRI BIDS files required by its configured subjects/task;
- Study 1 uploads raw EEG/fMRI BIDS metadata and events, clean EEG derivatives,
  fMRIPrep derivatives, and external signature assets for its configured cohort;
- Study 2 uploads its existing EEG/fMRI BIDS and derivative inputs.

The Study 1 cohort file contains:

```text
0000 0001 0003 0004 0005 0007 0008 0009 0010 0011 0012 0013 0014
```

`sub-0006` is not included.

Before Rorqual setup, a migration command retrieves the four fMRIPrep subjects
missing from KINGSTON from Trillium's available login/storage service. Retrieval
uses a Trillium-specific source profile explicitly; it does not mutate the active
Rorqual selection. Retrieved files are verified in KINGSTON before upload. This
one-time migration makes subsequent Rorqual setup reproducible from the canonical
local store.

Local fMRIPrep paths under
`derivatives/preprocessed/fmri/fmriprep/sub-<id>` map to the remote pipeline layout
under `FMRIPREP_DERIV_ROOT/preprocessed/fmri/sub-<id>`. The mapping is explicit;
the nested local directory is not copied verbatim.

Every setup script validates all subject inputs and required signature files
before transferring anything. It validates the corresponding remote paths after
transfer. Partial or stale inputs cause a non-zero exit and no submission.

## Study 1 Workflow

Study 1 becomes a repository-owned Alliance workflow rather than a collection of
scratch scripts. It consists of:

1. a common argument builder sourced by every Study 1 job;
2. a prepare job that runs `prepare-targets` and `prepare-features`;
3. a dispatcher that materializes missing benchmark cells;
4. a benchmark cell array;
5. a dependent report job;
6. a job record containing the run ID and every Slurm job ID.

All paths, account names, memory requests, concurrency limits, task names,
permutation counts, and subject lists come from validated environment variables or
tracked workflow files. No generated script uses a fixed `_0009` or `_current`
filename.

The dependency chain is:

```text
prepare -> dispatcher -> benchmark array -> report
```

The dispatcher exits successfully without submitting an array when all benchmark
cells already exist, and records that there was no remaining work. Any malformed
cell row, empty subject list, missing derivative, failed dependency, or missing
report output surfaces as an error.

## fMRIPrep and Study 2

Existing fMRIPrep and Study 2 workflows retain their domain behavior. They change
only at the infrastructure boundary:

- source the selected profile;
- use generic host/socket names;
- use profile-derived remote paths;
- construct Slurm memory arguments from profile values;
- report the selected cluster in setup and submission output.

Trillium continues to omit `--mem`. Rorqual uses explicit workflow memory
requests. fMRIPrep's internal `--mem-mb` remains lower than its Slurm allocation.

## Error Handling

The implementation fails fast for:

- an unsupported or empty `ALLIANCE_CLUSTER`;
- missing profile fields;
- a profile hostname that does not match the selected cluster;
- a missing SSH control connection;
- missing local or remote subject inputs;
- fMRIPrep internal memory greater than or equal to its Slurm memory request;
- unavailable modules, virtual environments, containers, licenses, or signatures;
- malformed Slurm job IDs or dependency records;
- failed Slurm states.

There are no hostname fallbacks, compatibility aliases, or automatic switches to
another cluster.

## Testing

Shell behavior is tested through subprocess tests with fake `ssh`, `rsync`,
`sbatch`, `squeue`, and `sacct` commands. Tests cover:

- selecting Rorqual and Trillium;
- rejecting unknown clusters;
- profile-specific host, socket, paths, and memory arguments;
- omission of `--mem` on Trillium and inclusion on Rorqual;
- subject-scoped Study 1 staging and local-to-remote fMRIPrep path mapping;
- missing local inputs failing before transfer;
- the complete Study 1 dependency chain and job record;
- dispatcher behavior when no benchmark cells remain;
- existing fMRIPrep fetch verification and Study 2 submission behavior.

Repository structure, shell syntax, Ruff, and the focused pytest suites form the
final local verification gate. Rorqual validation then checks profile resolution,
remote paths, modules, Python imports, `sbatch --test-only`, and a small Study 1
setup dry run before the real submission.

## Completion Criteria

The work is complete when:

1. changing only `ALLIANCE_CLUSTER` selects either supported cluster;
2. all Alliance setup, submit, monitor, fetch, and connection scripts use the
   selected profile;
3. KINGSTON contains the full Study 1 input cohort;
4. Rorqual passes remote input/runtime validation;
5. the corrected 13-subject Study 1 chain is submitted on Rorqual;
6. the run ID and job IDs are reported with commands for monitoring.
