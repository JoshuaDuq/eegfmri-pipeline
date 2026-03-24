# Study 2: Trial-Wise Cortical EEG-BOLD Coupling

Study 2 is the pain-study EEG-BOLD coupling workflow for trial-wise cortical coupling
between source-localized EEG power and subject-surface BOLD responses in a frozen set of
cortical ROIs.

This `README.md` is the technical entry point. The publication-style narrative lives in:

- [METHODS.md](METHODS.md)

The frozen production configuration is:

```text
studies/pain_study/study2/config/eeg_bold_coupling_study2.yaml
```

## What Study 2 Does

The production workflow:

1. loads clean EEG epochs and aligned clean events,
2. builds subject-specific cortical source models,
3. extracts trial-wise ROI EEG power in alpha and beta bands,
4. runs trial-wise LSS fMRI modeling on plateau trials,
5. summarizes subject-surface BOLD responses within frozen ROIs,
6. aligns EEG and fMRI trials by run and trial number,
7. applies prespecified nuisance and QC rules,
8. fits confirmatory mixed-effects models and enabled sensitivity analyses.

The main analysis is restricted to:

- `stim_phase == "plateau"`,
- source-localized cortical EEG,
- subject-surface fMRI in `T1w` space,
- the frozen ROIs `right_operculo_periinsular` and `midcingulate_pre_sma`,
- the EEG bands `alpha` and `beta`.

## Key Files

| Purpose | Path |
| --- | --- |
| publication-style methods | `studies/pain_study/study2/METHODS.md` |
| technical entry point | `studies/pain_study/study2/README.md` |
| production config | `studies/pain_study/study2/config/eeg_bold_coupling_study2.yaml` |
| smoke config | `studies/pain_study/study2/config/eeg_bold_coupling_smoketest.yaml` |
| coupling implementation | `studies/pain_study/study2/analysis/eeg_bold_coupling.py` |
| statistics backend | `studies/pain_study/study2/analysis/eeg_bold_statistics.py` |
| nuisance model | `studies/pain_study/study2/analysis/eeg_bold_nuisance.py` |

## Execution

The production Study 2 configuration is not the CLI default. Pass it explicitly:

```bash
python -m eeg_pipeline.cli.main coupling compute \
  --subject 0001 \
  --subject 0002 \
  --task thermalactive \
  --bids-root /path/to/bids_output/eeg \
  --bids-fmri-root /path/to/bids_output/fmri \
  --deriv-root /path/to/derivatives \
  --source-subjects-dir /path/to/freesurfer_subjects_dir \
  --coupling-config studies/pain_study/study2/config/eeg_bold_coupling_study2.yaml
```

## Inputs and Outputs

Study 2 requires:

- EEG BIDS root,
- fMRI BIDS root,
- derivatives root,
- clean EEG epochs and clean events,
- fMRIPrep subject-space outputs,
- FreeSurfer surfaces,
- EEG-to-MRI transform and BEM solution.

With the default output layout, subject results are written under:

```text
<paths.deriv_root>/sub-<ID>/multimodal/eeg_bold_coupling/task-<task>/contrast-plateau_trials/
```

and group results under:

```text
<paths.deriv_root>/group/multimodal/eeg_bold_coupling/task-<task>/contrast-plateau_trials/
```

## Notes

- For the scientific rationale, formulas, inferential hierarchy, and article-writing
  completion inventory, read [METHODS.md](METHODS.md).
- Study 2 intentionally fails fast on missing or invalid inputs. It does not implement
  fallback behavior or backward-compatibility shims.
