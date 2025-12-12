# EEG/fMRI Pipeline — CLI Reference

Entrypoints
- Module: `python -m eeg_pipeline.cli.main <command> [options]`
- Console script (if installed): `eeg-pipeline <command> [options]`

Common arguments (all commands)
- `--group all|A,B,C` | `--subject/-s ID` (repeatable) | `--all-subjects`
- `--task/-t TASK` (defaults to config `project.task`)

## Commands and options

### preprocessing
Modes: `raw-to-bids`, `merge-behavior`
- Base: `--source-root PATH` (raw) | `--bids-root PATH`; `--subjects ID ID ...` (raw-only)
- raw-to-bids: `--montage NAME` (default `easycap-M1`), `--line-freq FLOAT` (default 60.0), `--overwrite`, `--zero-base-onsets`, `--trim-to-first-volume`, `--event-prefix PREFIX` (repeatable), `--keep-all-annotations`
- merge-behavior: `--event-type TYPE` (repeatable), `--dry-run`

### behavior
Modes: `compute`, `visualize`
- Compute: `--correlation-method spearman|pearson`, `--bootstrap INT`, `--n-perm INT`, `--rng-seed INT`, `--computations ...`
  - Canonical flags: `correlations`, `pain_sensitivity`, `condition`, `temporal`, `cluster`, `mediation`, `mixed_effects`, `export`
  - Legacy aliases still accepted: `power_roi`, `connectivity_roi`, `connectivity_heatmaps`, `sliding_connectivity`, `time_frequency`, `temporal_correlations`, `cluster_test`, `precomputed_correlations`, `condition_correlations`, `exports`
- Visualize: `--plots ...` or `--all-plots`, `--skip-scatter`

### features
Modes: `compute`, `visualize`
- Compute: `--fixed-templates PATH` (.npz), `--feature-categories ...`, `--precomputed-groups ...`
  - Feature categories: `power`, `connectivity`, `microstates`, `aperiodic`, `itpc`, `pac`, `precomputed`, `cfc`, `dynamics_advanced`, `complexity`, `quality`
  - Precomputed groups: `erds`, `spectral`, `gfp`, `roi`, `temporal`, `complexity`, `ratios`, `asymmetry`, `aperiodic`, `connectivity`, `microstates`, `pac`, `cfc`, `dynamics_advanced`, `itpc`, `quality`

### erp
Modes: `compute`, `visualize`
- `--crop-tmin FLOAT`, `--crop-tmax FLOAT`

### tfr
Modes: `visualize`
- `--do-group`, `--tfr-roi`, `--tfr-topomaps-only`, `--n-jobs INT`

### decoding
Modes: compute only
- `--n-perm INT` (default 0), `--inner-splits INT` (default 3), `--outer-jobs INT` (default 1), `--rng-seed INT`, `--skip-time-gen`

## Quick examples
- Features: `python -m eeg_pipeline.cli.main features compute --subject 0001 --feature-categories power connectivity --precomputed-groups spectral erds`
- Behavior: `python -m eeg_pipeline.cli.main behavior compute --subject 0001 --computations correlations condition`
- Decoding: `python -m eeg_pipeline.cli.main decoding --subject 0001 --n-perm 100`
