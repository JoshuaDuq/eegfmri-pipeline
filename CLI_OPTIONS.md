# CLI Options by Pipeline

Entry points:
- Python module: `python -m eeg_pipeline.cli.main <command> [options]`
- Installed console script: `eeg-pipeline <command> [options]`

Common flags (all commands):
- `--group all|A,B,C` | `--subject/-s ID` (repeatable) | `--all-subjects`
- `--task/-t TASK` (defaults to config `project.task`)

---

## preprocessing

Modes: `raw-to-bids`, `merge-behavior`

**Base options:**
- `--source-root PATH` (raw EEG source directory)
- `--bids-root PATH` (BIDS output directory)
- `--subjects ID ID ...` (only for raw-to-bids)

**raw-to-bids options:**
- `--montage NAME` (default `easycap-M1`)
- `--line-freq FLOAT` (default 60.0)
- `--overwrite`
- `--zero-base-onsets`
- `--trim-to-first-volume`
- `--event-prefix PREFIX` (repeatable)
- `--keep-all-annotations`

**merge-behavior options:**
- `--event-type TYPE` (repeatable)
- `--dry-run`

---

## behavior

Modes: `compute`, `visualize`

**Shared options:**
- `--categories ...` (select feature categories to process)
  - Choices: `psychometrics`, `power`, `dynamics`, `aperiodic`, `connectivity`, `itpc`, `temporal`, `dose_response`

**Compute options:**
- `--correlation-method spearman|pearson`
- `--bootstrap INT` (default 0)
- `--n-perm INT`
- `--rng-seed INT`
- `--computations ...` (select stages to run)
  - Choices: `correlations`, `pain_sensitivity`, `condition`, `temporal`, `cluster`, `mediation`, `mixed_effects`, `export`

**Visualize options:**
- `--plots ...` or `--all-plots` (mutually exclusive)
- `--skip-scatter`

---

## features

Modes: `compute`, `visualize`

**Shared options:**
- `--categories ...` (select feature categories to process)
  - Compute choices: `power`, `connectivity`, `microstates`, `aperiodic`, `itpc`, `pac`, `precomputed`, `cfc`, `dynamics_advanced`, `complexity`, `quality`
  - Visualize choices: `power`, `connectivity`, `microstates`, `aperiodic`, `itpc`, `pac`, `dynamics`, `burst`, `erds`, `complexity`
  - Note: behavior-only categories like `psychometrics` and `dose_response` are not valid for `features`

**Compute options:**
- `--fixed-templates PATH` (.npz file containing fixed microstate templates)
- `--precomputed-groups ...` (override config precomputed groups)
  - Choices: `erds`, `spectral`, `gfp`, `roi`, `temporal`, `ratios`, `complexity`, `asymmetry`, `aperiodic`, `connectivity`, `microstates`, `pac`, `cfc`, `dynamics_advanced`, `itpc`, `quality`

---

## erp

Modes: `compute`, `visualize`

**Options:**
- `--crop-tmin FLOAT`
- `--crop-tmax FLOAT`

---

## tfr

Modes: `visualize`

**Options:**
- `--do-group`
- `--tfr-roi`
- `--tfr-topomaps-only`
- `--n-jobs INT`

---

## decoding

No explicit mode argument (compute only).

**Options:**
- `--n-perm INT` (default 0)
- `--inner-splits INT` (default 3)
- `--outer-jobs INT` (default 1)
- `--rng-seed INT`
- `--skip-time-gen`

---

## Examples

### Compute and visualize specific features

```bash
# Compute only power features
python -m eeg_pipeline.cli.main features compute --subject 0001 --categories power

# Compute a broader set of feature families
python -m eeg_pipeline.cli.main features compute --subject 0001 --categories power connectivity microstates aperiodic itpc pac precomputed cfc dynamics_advanced complexity quality

# Visualize only power and connectivity plots
python -m eeg_pipeline.cli.main features visualize --subject 0001 --categories power connectivity

# Visualize only ITPC behavioral correlations
python -m eeg_pipeline.cli.main behavior visualize --subject 0001 --categories itpc

# Compute correlations and visualize power scatter plots
python -m eeg_pipeline.cli.main behavior compute --subject 0001
python -m eeg_pipeline.cli.main behavior visualize --subject 0001 --categories power
```

### Preprocessing

```bash
# Convert raw EEG to BIDS format
python -m eeg_pipeline.cli.main preprocessing raw-to-bids --source-root data/raw --bids-root data/bids

# Merge behavioral data
python -m eeg_pipeline.cli.main preprocessing merge-behavior --dry-run
```

### Full pipeline example

```bash
# Run features for multiple subjects
python -m eeg_pipeline.cli.main features compute --group all --categories power connectivity

# Run behavior correlations with bootstrap CI
python -m eeg_pipeline.cli.main behavior compute --all-subjects --bootstrap 1000

# Run decoding with permutation testing
python -m eeg_pipeline.cli.main decoding --subject 0001 --subject 0002 --n-perm 100
```





