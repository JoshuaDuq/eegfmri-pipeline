# CLI Options by Pipeline

Entry points:
- Python module: `python -m eeg_pipeline.cli.main <command> [options]`
- Installed console script: `eeg-pipeline <command> [options]`

Common flags (all commands):
- `--group all|A,B,C` | `--subject/-s ID` (repeatable) | `--all-subjects`
- `--task/-t TASK` (defaults to config `project.task`)

## preprocessing
Modes: `raw-to-bids`, `merge-behavior`

Base options:
- `--source-root PATH` (raw) | `--bids-root PATH`
- `--subjects ID ID ...` (only for raw-to-bids)

raw-to-bids:
- `--montage NAME` (default `easycap-M1`)
- `--line-freq FLOAT` (default 60.0)
- `--overwrite`
- `--zero-base-onsets`
- `--trim-to-first-volume`
- `--event-prefix PREFIX` (repeatable)
- `--keep-all-annotations`

merge-behavior:
- `--event-type TYPE` (repeatable)
- `--dry-run`

## behavior
Modes: `compute`, `visualize`

Compute options:
- `--correlation-method spearman|pearson`
- `--bootstrap INT`
- `--n-perm INT`
- `--rng-seed INT`
- `--computations ...` (select stages to run)
  - Canonical: `correlations`, `pain_sensitivity`, `condition`, `temporal`, `cluster`, `mediation`, `mixed_effects`, `export`
  - Legacy aliases accepted: `power_roi`, `connectivity_roi`, `connectivity_heatmaps`, `sliding_connectivity`, `time_frequency`, `temporal_correlations`, `cluster_test`, `precomputed_correlations`, `condition_correlations`, `exports`

Visualize options:
- `--plots ...` or `--all-plots`
- `--skip-scatter`

## features
Modes: `compute`, `visualize`

Compute options:
- `--fixed-templates PATH` (.npz)
- `--feature-categories ...` choices:
  - `power`, `connectivity`, `microstates`, `aperiodic`, `itpc`, `pac`, `precomputed`, `cfc`, `dynamics_advanced`, `complexity`, `quality`
- `--precomputed-groups ...` choices:
  - `erds`, `spectral`, `gfp`, `roi`, `temporal`, `complexity`, `ratios`, `asymmetry`, `aperiodic`, `connectivity`, `microstates`, `pac`, `cfc`, `dynamics_advanced`, `itpc`, `quality`

## erp
Modes: `compute`, `visualize`
- `--crop-tmin FLOAT`
- `--crop-tmax FLOAT`

## tfr
Modes: `visualize`
- `--do-group`
- `--tfr-roi`
- `--tfr-topomaps-only`
- `--n-jobs INT`

## decoding
Modes: compute only (no explicit mode argument)
- `--n-perm INT` (default 0)
- `--inner-splits INT` (default 3)
- `--outer-jobs INT` (default 1)
- `--rng-seed INT`
- `--skip-time-gen`
