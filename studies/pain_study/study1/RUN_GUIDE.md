# Study 1 - Run Guide

This guide runs the Study 1 signature-prediction workflow from prepared EEG/fMRI derivatives.
Commands assume the private `studies` package has registered the `signature-prediction` command
through the `eeg_pipeline.cli_commands` entry-point group. Confirm that registration before running
Study 1:

```bash
eeg-pipeline --help
```

The command list must include `signature-prediction`. If it does not, install the private studies
package that exposes `studies.pain_study.cli.command_registry:signature_prediction_command`.

### Configs

Use the production config for formal analysis:

```bash
STUDY1_CONFIG="studies/pain_study/study1/config/study1_config.yaml"
```

Use the smoke-test config only to verify data plumbing on a small local cohort:

```bash
STUDY1_CONFIG="studies/pain_study/study1/config/study1_smoketest.yaml"
```

The smoke config lowers the subject minimum, permutations, and deep-regression epochs. It is not
eligible for scientific interpretation.

### Paths

Set the study roots explicitly. The Kingston layout used during local testing is:

```bash
EEG_BIDS_ROOT="/Volumes/KINGSTON/EEG_fMRI_data/bids_output/eeg"
FMRI_BIDS_ROOT="/Volumes/KINGSTON/EEG_fMRI_data/bids_output/fmri"
DERIV_ROOT="/Volumes/KINGSTON/EEG_fMRI_data/derivatives"
SIGNATURE_DIR="/Volumes/KINGSTON/EEG_fMRI_data/external"
TASK="thermalactive"
NPS_MAP="NPS/weights_NSF_grouppred_cvpcr.nii.gz"
SIIPS1_MAP="SIIPS1/nonnoc_v11_4_137subjmap_weighted_mean.nii.gz"
SIGNATURE_MAPS_JSON='[
  {"name":"NPS","path":"NPS/weights_NSF_grouppred_cvpcr.nii.gz"},
  {"name":"SIIPS1","path":"SIIPS1/nonnoc_v11_4_137subjmap_weighted_mean.nii.gz"}
]'
```

Keep the signature provenance paths relative to `SIGNATURE_DIR`. The Study 1 config loader resolves
YAML path-looking strings relative to the config file, and fMRI signature discovery requires
`paths.signature_maps`, so these overrides are required when running from an external signature
directory:

```bash
COMMON_ARGS=(
  --task "$TASK"
  --study1-config "$STUDY1_CONFIG"
  --bids-root "$EEG_BIDS_ROOT"
  --bids-fmri-root "$FMRI_BIDS_ROOT"
  --deriv-root "$DERIV_ROOT"
  --set "paths.signature_dir=$SIGNATURE_DIR"
  --set "paths.signature_maps=$SIGNATURE_MAPS_JSON"
  --set "study1.targets.signature_provenance.NPS.path=$NPS_MAP"
  --set "study1.targets.signature_provenance.SIIPS1.path=$SIIPS1_MAP"
)
```

For a formal rerun, prefer a new output root rather than mixing outputs from different configs or
dates:

```bash
COMMON_ARGS+=(
  --set "study1.outputs.root_name=study1_$(date +%Y%m%d)"
)
```

Outputs are written to:

```text
$DERIV_ROOT/group/multimodal/<study1.outputs.root_name>/
```

The examples below refer to that directory as:

```bash
STUDY1_ROOT="$DERIV_ROOT/group/multimodal/<study1.outputs.root_name>"
```

### Subject Selection

Run production analyses only after at least 30 analyzable subjects are available. Use explicit
subject arguments when auditing a known cohort:

```bash
SUBJECT_ARGS=(
  --subject 0000
  --subject 0001
)
```

Use all discoverable subjects only when the BIDS and derivative roots are already curated:

```bash
SUBJECT_ARGS=(--all-subjects)
```

### Pipeline Stages

Run stages in this order.

1. Prepare trial-wise fMRI signature targets:

   ```bash
   eeg-pipeline signature-prediction prepare-targets \
     "${SUBJECT_ARGS[@]}" \
     "${COMMON_ARGS[@]}"
   ```

   Required output:

   ```text
   $STUDY1_ROOT/targets/primary_targets.parquet
   ```

2. Prepare Study 1 trial-safe EEG feature tables:

   ```bash
   eeg-pipeline signature-prediction prepare-features \
     "${SUBJECT_ARGS[@]}" \
     "${COMMON_ARGS[@]}"
   ```

   Required per-subject outputs include:

   ```text
   $STUDY1_ROOT/features_trial_ml_safe/sub-*/eeg/features/power/features_power.parquet
   $STUDY1_ROOT/features_trial_ml_safe/sub-*/eeg/features/spectral/features_spectral.parquet
   ```

3. Run the feature benchmark:

   ```bash
   eeg-pipeline signature-prediction feature-benchmark \
     "${SUBJECT_ARGS[@]}" \
     "${COMMON_ARGS[@]}"
   ```

   Each required primary cell writes `metrics/model_comparison_summary.json` under:

   ```text
   $STUDY1_ROOT/feature_benchmark/primary/<target>/<preset>/model_comparison/
   ```

   Required primary targets are `NPS` and `SIIPS1`. Required primary presets are `alpha`, `beta`,
   and `alpha_beta`.

4. Run exploratory deep regression if the report stage will be used:

   ```bash
   eeg-pipeline signature-prediction deep-regression \
     "${SUBJECT_ARGS[@]}" \
     "${COMMON_ARGS[@]}"
   ```

5. Write the Study 1 report:

   ```bash
   eeg-pipeline signature-prediction report \
     "${SUBJECT_ARGS[@]}" \
     "${COMMON_ARGS[@]}"
   ```

   The report stage validates that all prespecified feature-benchmark outputs and configured
   deep-regression outputs exist. If deep regression is intentionally omitted, do not run the report
   stage.

### Smoke Test on Kingston

The following smoke test checks the Kingston data path with two subjects and reduced permutations:

```bash
STUDY1_CONFIG="studies/pain_study/study1/config/study1_smoketest.yaml"
EEG_BIDS_ROOT="/Volumes/KINGSTON/EEG_fMRI_data/bids_output/eeg"
FMRI_BIDS_ROOT="/Volumes/KINGSTON/EEG_fMRI_data/bids_output/fmri"
DERIV_ROOT="/Volumes/KINGSTON/EEG_fMRI_data/derivatives"
SIGNATURE_DIR="/Volumes/KINGSTON/EEG_fMRI_data/external"
TASK="thermalactive"
NPS_MAP="NPS/weights_NSF_grouppred_cvpcr.nii.gz"
SIIPS1_MAP="SIIPS1/nonnoc_v11_4_137subjmap_weighted_mean.nii.gz"
SIGNATURE_MAPS_JSON='[
  {"name":"NPS","path":"NPS/weights_NSF_grouppred_cvpcr.nii.gz"},
  {"name":"SIIPS1","path":"SIIPS1/nonnoc_v11_4_137subjmap_weighted_mean.nii.gz"}
]'
SUBJECT_ARGS=(--subject 0000 --subject 0001)
COMMON_ARGS=(
  --task "$TASK"
  --study1-config "$STUDY1_CONFIG"
  --bids-root "$EEG_BIDS_ROOT"
  --bids-fmri-root "$FMRI_BIDS_ROOT"
  --deriv-root "$DERIV_ROOT"
  --set "paths.signature_dir=$SIGNATURE_DIR"
  --set "paths.signature_maps=$SIGNATURE_MAPS_JSON"
  --set "study1.targets.signature_provenance.NPS.path=$NPS_MAP"
  --set "study1.targets.signature_provenance.SIIPS1.path=$SIIPS1_MAP"
  --set "study1.outputs.root_name=study1_smoke_$(date +%Y%m%d)"
)

eeg-pipeline signature-prediction prepare-targets \
  "${SUBJECT_ARGS[@]}" \
  "${COMMON_ARGS[@]}"

eeg-pipeline signature-prediction prepare-features \
  "${SUBJECT_ARGS[@]}" \
  "${COMMON_ARGS[@]}"

eeg-pipeline signature-prediction feature-benchmark \
  "${SUBJECT_ARGS[@]}" \
  "${COMMON_ARGS[@]}"
```

Interpret smoke-test output only as a pipeline integrity check. With two subjects, each LOSO
training fold contains one subject, so subject-grouped inner cross-validation cannot tune
hyperparameters.

### Grouped Inner-CV Smoke Test

To exercise the 3-fold inner `GroupKFold` used by the smoke config, run at least four analyzable
subjects. Each LOSO outer fold then trains on three subject groups:

```bash
SUBJECT_ARGS=(
  --subject 0000
  --subject 0001
  --subject pilot001
  --subject pilot002
)
COMMON_ARGS+=(
  --set "study1.cohort.min_subjects=4"
  --set "study1.features.exploratory_feature_families=[]"
  --set "study1.feature_benchmark.n_perm=1"
  --set "study1.outputs.root_name=study1_groupcv_smoke_$(date +%Y%m%d)"
)
```

After `feature-benchmark`, confirm grouped inner CV ran by checking that fold-level
`best_params` are not `{}`:

```bash
METRICS_PATH="$DERIV_ROOT/group/multimodal/study1_groupcv_smoke_$(date +%Y%m%d)"
METRICS_PATH="$METRICS_PATH/feature_benchmark/primary/NPS/alpha"
METRICS_PATH="$METRICS_PATH/model_comparison/metrics/model_comparison.tsv"
export METRICS_PATH
python - <<'PY'
import os
import pandas as pd

df = pd.read_csv(os.environ["METRICS_PATH"], sep="\t")
print(df[["model", "fold", "test_subject", "best_params"]])
PY
```

### Output Hygiene

Do not mix smoke-test and production outputs in the same `study1.outputs.root_name`. Before reading
or aggregating results, check for stale summaries:

```bash
find "$DERIV_ROOT/group/multimodal" \
  -path "*/feature_benchmark/*/*/*/model_comparison/metrics/model_comparison_summary.json" \
  -exec stat -f "%Sm %N" -t "%Y-%m-%d %H:%M:%S" {} +
```

Check for AppleDouble sidecars copied from external drives:

```bash
find "$DERIV_ROOT/group/multimodal/<study1.outputs.root_name>" -name "._*" -print
```

Sidecars and stale outputs should be removed or isolated before formal reporting.

### Troubleshooting Target Preparation

`prepare-targets` fails fast when the fMRI inputs are not valid for trial-wise signature
extraction. Two common blockers are:

- selected fMRIPrep confounds contain missing values, often in derivative or framewise-displacement
  columns; fix the confounds file or use an explicitly justified confound/censoring policy
- trial beta images contain non-finite voxels that require continuous resampling to the signature
  grid; align the image and signature grids or repair the image before scoring

Do not zero-fill nuisance regressors or mask away non-finite voxels inside Study 1 as a workaround.

