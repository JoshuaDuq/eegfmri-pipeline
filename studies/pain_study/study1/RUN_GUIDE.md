# Study 1 - Run Guide

This guide runs the Study 1 signature-prediction workflow from prepared EEG/fMRI derivatives.
Commands assume the private `studies` package has registered the `signature-prediction` command
through the `eeg_pipeline.cli_commands` entry-point group. Confirm that registration before running
Study 1:

```bash
cd /Users/joduq24/Desktop/EEG_fMRI_Pipeline
EEG_PIPELINE=".venv/bin/eeg-pipeline"
PYTHON=".venv/bin/python"

"$EEG_PIPELINE" --help
```

The command list must include `signature-prediction`. If it does not, install the private studies
package that exposes `studies.pain_study.cli.command_registry:signature_prediction_command`.

## Copy-Paste Run

Use this block for a normal rerun on the Kingston Study 1 layout. Change only `STUDY1_RUN_ID` when
you want a new output directory, and change `SUBJECT_ARGS` only when you need to audit a specific
cohort instead of all curated subjects.

```bash
cd /Users/joduq24/Desktop/EEG_fMRI_Pipeline

EEG_PIPELINE=".venv/bin/eeg-pipeline"
PYTHON=".venv/bin/python"
STUDY1_CONFIG="studies/pain_study/study1/config/study1_config.yaml"
STUDY1_RUN_ID="study1_$(date +%Y%m%d)"

EEG_BIDS_ROOT="/Volumes/KINGSTON/EEG_fMRI_data/bids_output/eeg"
FMRI_BIDS_ROOT="/Volumes/KINGSTON/EEG_fMRI_data/bids_output/fmri"
DERIV_ROOT="/Volumes/KINGSTON/EEG_fMRI_data/derivatives"
SIGNATURE_DIR="/Volumes/KINGSTON/EEG_fMRI_data/external"
TASK="thermalactive"

NPS_MAP="NPS/weights_NSF_grouppred_cvpcr.nii.gz"
SIIPS1_MAP="SIIPS1/nonnoc_v11_4_137subjmap_weighted_mean.nii.gz"
SIGNATURE_MANIFEST="signature_manifest.yaml"
SIGNATURE_MAPS_JSON='[
  {"name":"NPS","path":"NPS/weights_NSF_grouppred_cvpcr.nii.gz"},
  {"name":"SIIPS1","path":"SIIPS1/nonnoc_v11_4_137subjmap_weighted_mean.nii.gz"}
]'

SUBJECT_ARGS=(--all-subjects)
COMMON_ARGS=(
  --task "$TASK"
  --study1-config "$STUDY1_CONFIG"
  --bids-root "$EEG_BIDS_ROOT"
  --bids-fmri-root "$FMRI_BIDS_ROOT"
  --deriv-root "$DERIV_ROOT"
  --set "paths.signature_dir=$SIGNATURE_DIR"
  --set "paths.signature_maps=$SIGNATURE_MAPS_JSON"
  --set "study1.targets.signature_manifest_path=$SIGNATURE_MANIFEST"
  --set "study1.targets.signature_provenance.NPS.path=$NPS_MAP"
  --set "study1.targets.signature_provenance.SIIPS1.path=$SIIPS1_MAP"
  --set "study1.outputs.root_name=$STUDY1_RUN_ID"
)

"$EEG_PIPELINE" signature-prediction prepare-targets \
  "${SUBJECT_ARGS[@]}" \
  "${COMMON_ARGS[@]}"

"$EEG_PIPELINE" signature-prediction prepare-features \
  "${SUBJECT_ARGS[@]}" \
  "${COMMON_ARGS[@]}"

"$EEG_PIPELINE" signature-prediction feature-benchmark \
  "${SUBJECT_ARGS[@]}" \
  "${COMMON_ARGS[@]}"

"$EEG_PIPELINE" signature-prediction report \
  "${SUBJECT_ARGS[@]}" \
  "${COMMON_ARGS[@]}"
```

The main outputs are:

```text
$DERIV_ROOT/group/multimodal/$STUDY1_RUN_ID/targets/primary_targets.parquet
$DERIV_ROOT/group/multimodal/$STUDY1_RUN_ID/reports/study1_report.tsv
$DERIV_ROOT/group/multimodal/$STUDY1_RUN_ID/reports/article_tables/
$DERIV_ROOT/group/multimodal/$STUDY1_RUN_ID/reports/full_picture/
```

Run `report` again with sensitivity roots when you want one article-ready comparison table across
previous runs:

```bash
SENSITIVITY_OUTPUT_ROOTS='[
  {"label":"no_stimulus_temp_control","root_name":"study1_no_stimulus_intensity_20260524"},
  {"label":"raw_target","root_name":"study1_raw_targets_20260524"},
  {"label":"stimulus_temp_prediction","root_name":"study1_stimulus_temp_prediction_20260524"}
]'

COMMON_ARGS+=(
  --set "study1.reporting.sensitivity_output_roots=$SENSITIVITY_OUTPUT_ROOTS"
)

"$EEG_PIPELINE" signature-prediction report \
  "${SUBJECT_ARGS[@]}" \
  "${COMMON_ARGS[@]}"
```

That writes:

```text
$DERIV_ROOT/group/multimodal/$STUDY1_RUN_ID/reports/full_picture/configured_sensitivity_model_summary.tsv
$DERIV_ROOT/group/multimodal/$STUDY1_RUN_ID/reports/full_picture/primary_sensitivity_comparison.tsv
```

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
SIGNATURE_MANIFEST="signature_manifest.yaml"
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
  --set "study1.targets.signature_manifest_path=$SIGNATURE_MANIFEST"
  --set "study1.targets.signature_provenance.NPS.path=$NPS_MAP"
  --set "study1.targets.signature_provenance.SIIPS1.path=$SIIPS1_MAP"
)
```

If the frozen signature manifest is missing or the signature files changed, regenerate it before
target preparation:

```bash
"$PYTHON" -m studies.pain_study.study1.signature_manifest \
  "$SIGNATURE_DIR" \
  "$SIGNATURE_DIR/$SIGNATURE_MANIFEST"
```

### Preprocessing Prerequisite

Study 1 target alignment requires an explicit task `block` column or the protocol event column
`run_id`, plus original `trial_number` or `trial_index`. Production nuisance regression requires
HRF-weighted artifact columns
`hrf_weighted_framewise_displacement`, `hrf_weighted_std_dvars`, and
`hrf_weighted_fp1_fp2_high_frequency_power`, plus `residual_ecg_coupling`. The raw Fp1/Fp2
artifact proxy remains `fp1_fp2_high_frequency_power`; it is an upstream input to the HRF-weighted
covariate and artifact-censoring audits, not the Level 2 nuisance column.

The smoke-test config uses the nuisance columns available in the current cleaned Kingston events:
`block`, `onset`, `within_block_trial`, `residual_ecg_coupling`, `stimulus_temp`, and
`selected_surface`. This keeps the smoke report structurally identical to the production
incremental benchmark while avoiding production-only HRF-weighted artifact prerequisites.

For a formal rerun, prefer a new output root rather than mixing outputs from different configs or
dates:

```bash
STUDY1_RUN_ID="study1_$(date +%Y%m%d)"
COMMON_ARGS+=(
  --set "study1.outputs.root_name=$STUDY1_RUN_ID"
)
```

Outputs are written to:

```text
$DERIV_ROOT/group/multimodal/$STUDY1_RUN_ID/
```

The examples below refer to that directory as:

```bash
STUDY1_ROOT="$DERIV_ROOT/group/multimodal/$STUDY1_RUN_ID"
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

For `prepare-targets`, Study 1 filters `--all-subjects` to subjects with task fMRI `func`
directories and logs EEG-only exclusions. Later stages intentionally resolve their analysis cohort
from the prepared primary target table, so rerun `prepare-targets` after changing the intended
multimodal subject set.

### Audit Checkpoints

Treat the production run as a sequence of decision points, not just a command chain. After target
preparation, inspect signature provenance, scoring-mask support, LSS estimability, retained
subjects, and retained plateau-trial counts before investing in feature extraction or permutation
testing. Missing provenance, invalid signature support, unstable LSS designs, or inadequate retained
trials should be fixed at the derivative or event-log level rather than compensated for downstream.

The final report interprets results in this order: analysis validity, the prespecified `NPS` /
`alpha_beta` / `elasticnet` primary cell, interpretation diagnostics, and secondary/exploratory
analyses. Secondary benchmark cells and exploratory models are reported after the primary cell.

### Pipeline Stages

Run stages in this order.

1. Prepare trial-wise fMRI signature targets:

   ```bash
   "$EEG_PIPELINE" signature-prediction prepare-targets \
     "${SUBJECT_ARGS[@]}" \
     "${COMMON_ARGS[@]}"
   ```

   Required output:

   ```text
   $STUDY1_ROOT/targets/primary_targets.parquet
   ```

   Check this output before continuing. A target table with too few retained subjects, sparse
   plateau trials, invalid signature provenance, or unresolved LSS design failures leaves the
   primary estimand unevaluable even if the feature benchmark runs.

2. Prepare Study 1 trial-safe EEG feature tables:

   ```bash
   "$EEG_PIPELINE" signature-prediction prepare-features \
     "${SUBJECT_ARGS[@]}" \
     "${COMMON_ARGS[@]}"
   ```

   Required per-subject output:

   ```text
   $STUDY1_ROOT/features_trial_ml_safe/sub-*/eeg/features/power/features_power.parquet
   ```

   Exploratory families produce additional per-family tables only when explicitly enabled.

3. Run the feature benchmark:

   ```bash
   "$EEG_PIPELINE" signature-prediction feature-benchmark \
     "${SUBJECT_ARGS[@]}" \
     "${COMMON_ARGS[@]}"
   ```

   Each required primary-partition benchmark cell writes `metrics/model_comparison_summary.json`
   under:

   ```text
   $STUDY1_ROOT/feature_benchmark/primary/<target>/<preset>/model_comparison/
   ```

   Required benchmark targets are `NPS` and `SIIPS1`. The report validates the full primary
   benchmark preset set: `delta`, `theta`, `alpha`, `beta`, `gamma`, `delta_theta`,
   `alpha_beta`, `alpha_beta_gamma`, and `all_bands`. The prespecified inferential primary cell
   remains `NPS` / `alpha_beta` / `elasticnet`; the other required cells provide the broader
   full-picture frequency audit. The benchmark filters predictors to active-window,
   individual-channel, log-ratio power columns and excludes Fp1/Fp2.

Exploratory feature families are disabled in the default Study 1 configs. Enable them explicitly
when they are part of the planned run:

```bash
EXPLORATORY_FEATURE_FAMILIES='[
  "spectral",
  "aperiodic",
  "erds",
  "ratios",
  "asymmetry",
  "bursts"
]'

COMMON_ARGS+=(
  --set "study1.features.exploratory_feature_families=$EXPLORATORY_FEATURE_FAMILIES"
)
```

4. Run exploratory deep regression only when the thesis report should include that lane:

   ```bash
   "$EEG_PIPELINE" signature-prediction deep-regression \
     "${SUBJECT_ARGS[@]}" \
     "${COMMON_ARGS[@]}"
   ```

5. Write the Study 1 report:

   ```bash
   "$EEG_PIPELINE" signature-prediction report \
     "${SUBJECT_ARGS[@]}" \
     "${COMMON_ARGS[@]}"
   ```

   The report stage validates that all prespecified feature-benchmark outputs exist and contain the
   protocol audit fields: confidence intervals, valid permutation counts, invalid permutation
   attempts, subject-selection counts, fold-level best hyperparameters, analysis-validity status,
   primary prediction status, interpretation diagnostics, and interpretation flags.
   Deep-regression and exploratory feature-benchmark summaries are included when present, but they
   are not required for the primary Study 1 report.
   The report also writes `reports/full_picture/`, which contains model leaderboards,
   target-by-temperature summaries, subject-by-temperature summaries, and a manifest. To compare
   sibling sensitivity runs in the same bundle, set
   `study1.reporting.sensitivity_output_roots` to a list of `{label, root_name}` entries before
   running `report`.

### Smoke Test on Kingston

The following smoke test checks the Kingston data path with two subjects and reduced permutations:

```bash
cd /Users/joduq24/Desktop/EEG_fMRI_Pipeline

EEG_PIPELINE=".venv/bin/eeg-pipeline"
PYTHON=".venv/bin/python"
STUDY1_CONFIG="studies/pain_study/study1/config/study1_smoketest.yaml"
STUDY1_RUN_ID="study1_smoke_$(date +%Y%m%d)"
EEG_BIDS_ROOT="/Volumes/KINGSTON/EEG_fMRI_data/bids_output/eeg"
FMRI_BIDS_ROOT="/Volumes/KINGSTON/EEG_fMRI_data/bids_output/fmri"
DERIV_ROOT="/Volumes/KINGSTON/EEG_fMRI_data/derivatives"
SIGNATURE_DIR="/Volumes/KINGSTON/EEG_fMRI_data/external"
TASK="thermalactive"
NPS_MAP="NPS/weights_NSF_grouppred_cvpcr.nii.gz"
SIIPS1_MAP="SIIPS1/nonnoc_v11_4_137subjmap_weighted_mean.nii.gz"
SIGNATURE_MANIFEST="signature_manifest.yaml"
SIGNATURE_MAPS_JSON='[
  {"name":"NPS","path":"NPS/weights_NSF_grouppred_cvpcr.nii.gz"},
  {"name":"SIIPS1","path":"SIIPS1/nonnoc_v11_4_137subjmap_weighted_mean.nii.gz"}
]'
SUBJECT_ARGS=(--subject 0000 --subject 0001)
SMOKE_EXTRA_ARGS=()
COMMON_ARGS=(
  --task "$TASK"
  --study1-config "$STUDY1_CONFIG"
  --bids-root "$EEG_BIDS_ROOT"
  --bids-fmri-root "$FMRI_BIDS_ROOT"
  --deriv-root "$DERIV_ROOT"
  --set "paths.signature_dir=$SIGNATURE_DIR"
  --set "paths.signature_maps=$SIGNATURE_MAPS_JSON"
  --set "study1.targets.signature_manifest_path=$SIGNATURE_MANIFEST"
  --set "study1.targets.signature_provenance.NPS.path=$NPS_MAP"
  --set "study1.targets.signature_provenance.SIIPS1.path=$SIIPS1_MAP"
  "${SMOKE_EXTRA_ARGS[@]}"
  --set "study1.outputs.root_name=$STUDY1_RUN_ID"
)

"$EEG_PIPELINE" signature-prediction prepare-targets \
  "${SUBJECT_ARGS[@]}" \
  "${COMMON_ARGS[@]}"

"$EEG_PIPELINE" signature-prediction prepare-features \
  "${SUBJECT_ARGS[@]}" \
  "${COMMON_ARGS[@]}"

"$EEG_PIPELINE" signature-prediction feature-benchmark \
  "${SUBJECT_ARGS[@]}" \
  "${COMMON_ARGS[@]}"
```

Interpret smoke-test output only as a pipeline integrity check. With two subjects, each LOSO
training fold contains one subject, so subject-grouped inner cross-validation cannot tune
hyperparameters.

### Grouped Inner-CV Smoke Test

To exercise the 3-fold inner `GroupKFold` used by the smoke config, run at least four analyzable
subjects. Each LOSO outer fold then trains on three subject groups. In the smoke-test block above,
replace `SUBJECT_ARGS`, `STUDY1_RUN_ID`, and `SMOKE_EXTRA_ARGS` before constructing `COMMON_ARGS`:

```bash
SUBJECT_ARGS=(
  --subject 0000
  --subject 0001
  --subject pilot001
  --subject pilot002
)
STUDY1_RUN_ID="study1_groupcv_smoke_$(date +%Y%m%d)"
SMOKE_EXTRA_ARGS=(
  --set "study1.cohort.min_subjects=4"
  --set "study1.feature_benchmark.n_perm=1"
)
```

After `feature-benchmark`, confirm grouped inner CV ran by checking that fold-level
`best_params` are not `{}`:

```bash
METRICS_PATH="$DERIV_ROOT/group/multimodal/$STUDY1_RUN_ID"
METRICS_PATH="$METRICS_PATH/feature_benchmark/primary/NPS/alpha"
METRICS_PATH="$METRICS_PATH/model_comparison/metrics/model_comparison.tsv"
export METRICS_PATH
"$PYTHON" - <<'PY'
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
find "$DERIV_ROOT/group/multimodal/$STUDY1_RUN_ID" -name "._*" -print
```

Sidecars and stale outputs should be removed or isolated before formal reporting.

### Troubleshooting Target Preparation

`prepare-targets` fails fast when the fMRI inputs are not valid for trial-wise signature
extraction. Two common blockers are:

- selected fMRIPrep confounds contain non-finite values in retained volumes; fMRIPrep
  motion-outlier/non-steady-state columns and the undefined initial derivative or
  framewise-displacement row are converted to a Nilearn `sample_mask`, but any non-finite nuisance
  value that remains in a retained volume is an error
- trial beta images contain non-finite voxels inside the analysis mask; non-finite background
  outside the mask is filled only to permit continuous resampling, but in-mask non-finite values
  require repairing the image or mask

Do not zero-fill retained nuisance regressors or mask away non-finite voxels inside Study 1 as a
workaround.
