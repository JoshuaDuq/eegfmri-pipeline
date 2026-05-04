# Study 1: Trial-Wise EEG Prediction of fMRI Pain-Signature Expression

Study 1 is the pain-study EEG-to-fMRI-signature workflow for trial-wise prediction of
raw `NPS` and `SIIPS1` expression from clean EEG.

This `README.md` is the technical entry point. The publication-style narrative lives in:

- [METHODS.md](METHODS.md)

The frozen production configuration is:

```text
studies/pain_study/study1/config/study1_config.yaml
```

## What Study 1 Does

The production workflow:

1. loads clean EEG events and validates trial structure,
2. runs trial-wise LSS fMRI signature extraction in MNI space,
3. aligns `NPS` and `SIIPS1` back to EEG trials,
4. writes a shared Study 1 primary target table,
5. prepares a Study 1-owned `trial_ml_safe` EEG feature store,
6. runs a feature-based model-comparison benchmark for the raw-expression primary objective,
7. runs deep regression from band-limited EEG trial tensors for the same primary objective,
8. supports nuisance-adjusted sensitivity analyses when explicitly enabled,
9. validates and aggregates the prespecified primary summaries into one study-level report.

The main analysis is restricted to:

- the fixed targets `NPS` and `SIIPS1`,
- the contrast `pain_vs_nonpain`,
- `trial_type == "stimulation"`,
- `stim_phase == "plateau"`,
- fMRI inputs in `MNI152NLin2009cAsym`,
- raw Study 1 targets for the primary analysis,
- optional fold-contained nuisance residualization for sensitivity analyses,
- Study 1-owned `trial_ml_safe` EEG features,
- the fixed deep-regression presets `alpha`, `beta`, `gamma`, and `alpha_beta_gamma`.

## Key Files

| Purpose | Path |
| --- | --- |
| publication-style methods | `studies/pain_study/study1/METHODS.md` |
| technical entry point | `studies/pain_study/study1/README.md` |
| production config | `studies/pain_study/study1/config/study1_config.yaml` |
| smoke config | `studies/pain_study/study1/config/study1_smoketest.yaml` |
| runner | `studies/pain_study/study1/runner.py` |
| target preparation | `studies/pain_study/study1/targets.py` |
| feature preparation | `studies/pain_study/study1/prepare_features.py` |
| feature benchmark | `studies/pain_study/study1/feature_benchmark.py` |
| deep regression | `studies/pain_study/study1/deep_regression/` |
| report aggregation | `studies/pain_study/study1/reporting.py` |

## Execution

The production Study 1 configuration is loaded automatically, but for pinned runs you can
pass it explicitly:

```bash
python -m eeg_pipeline.cli.main signature-prediction prepare-targets \
  --subject 0001 \
  --subject 0002 \
  --task pain \
  --study1-config studies/pain_study/study1/config/study1_config.yaml

python -m eeg_pipeline.cli.main signature-prediction prepare-features \
  --subject 0001 \
  --subject 0002 \
  --task pain \
  --study1-config studies/pain_study/study1/config/study1_config.yaml

python -m eeg_pipeline.cli.main signature-prediction feature-benchmark \
  --subject 0001 \
  --subject 0002 \
  --task pain \
  --study1-config studies/pain_study/study1/config/study1_config.yaml

python -m eeg_pipeline.cli.main signature-prediction deep-regression \
  --subject 0001 \
  --subject 0002 \
  --task pain \
  --study1-config studies/pain_study/study1/config/study1_config.yaml

python -m eeg_pipeline.cli.main signature-prediction report \
  --subject 0001 \
  --task pain \
  --study1-config studies/pain_study/study1/config/study1_config.yaml
```

## Inputs and Outputs

Study 1 requires:

- EEG BIDS root,
- fMRI BIDS root,
- derivatives root,
- clean EEG epochs and clean events,
- MNI-space fMRI inputs compatible with trial-wise signature extraction,
- signature maps containing both `NPS` and `SIIPS1`,
- PyTorch if `deep-regression` will be run.

With the default output layout, Study 1 writes to:

```text
<paths.deriv_root>/group/multimodal/study1/
```

Key outputs are:

```text
targets/primary_targets.parquet
features_trial_ml_safe/
feature_benchmark/
deep_regression/
reports/study1_report.tsv
```

## Notes

- For the scientific rationale, formulas, workflow contracts, and detailed stage behavior,
  read [METHODS.md](METHODS.md).
- `prepare-targets` can run on a single subject, but `prepare-features`,
  `feature-benchmark`, and `deep-regression` require at least
  `study1.cohort.min_subjects` subjects because they are LOSO analyses.
- The `report` stage requires the prespecified primary feature-benchmark and
  deep-regression outputs to be present before writing the study-level table.
- Study 1 intentionally fails fast on missing or invalid inputs. It does not implement
  fallback behavior or backward-compatibility shims.
