# Sub-0015 Band-Specific FieldTrip ICA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fit alpha, beta, and gamma FieldTrip ICA models to pre-ICA `sub-0015` signals restricted to the final 59-trial rejection set, then compute the existing pain-study TFR result set for each model.

**Architecture:** Load signal values from the 66-trial `_epo.fif`, apply only the retained-trial indices recorded by `proc-clean_epo.fif`, and map those indices to the original BIDS condition table. New MATLAB batch functions filter that 59-trial pre-ICA export for each band, fit runica, apply its weights to broadband epochs, and pass each decomposition through the existing condition/TFR calculations without changing the restored broadband functions.

**Tech Stack:** MATLAB R2026a, FieldTrip, runica, MNE-Python export bridge, YAML/JSON configuration.

---

### Task 1: Configure ICA bands

**Files:**
- Modify: `studies/pain_study/fieldtrip_tfr/config/fieldtrip_tfr_brainvision_analyzer_sub0015.yaml`
- Create: `studies/pain_study/fieldtrip_tfr/export_clean_epochs_fieldtrip.py`

- [ ] Add named bands under `ica`:

```yaml
  bands:
    alpha: [6.0, 14.0]
    beta: [14.0, 30.0]
    gamma: [30.0, 100.0]
```

- [ ] Add explicit pre-ICA and trial-rejection-mask inputs to the exporter. Confirm provenance identifies both FIF files and the 59 retained BIDS trial rows.

### Task 2: Fit band-specific ICA

**Files:**
- Create: `studies/pain_study/fieldtrip_tfr/matlab/prepareBandIcaBatch.m`

- [ ] Validate exactly the `alpha`, `beta`, and `gamma` configuration fields and their two finite increasing frequency bounds.
- [ ] For each band, use `ft_preprocessing` with `bpfilter="yes"`, select the existing ICA channel set, compute numerical rank, and fit extended runica with the configured seed and iteration limit.
- [ ] Apply the fitted unmixing matrix to `broadbandData` and save:

```text
ica/<band>/sub-0015/eeg/sub-0015_task-thermalactive_desc-fieldtripica_components.mat
```

- [ ] Save `componentFit`, `componentBroadband`, `broadbandData`, `exportId`, `dataRank`, `bandName`, and `bandFrequencyHz` plus a matching provenance JSON.

### Task 3: Compute TFRs per decomposition

**Files:**
- Create: `studies/pain_study/fieldtrip_tfr/matlab/computeBandTfrBatch.m`
- Reuse: `studies/pain_study/fieldtrip_tfr/matlab/computeTfrBatch.m`

- [ ] Keep the restored broadband function unchanged.
- [ ] For each configured band, load its `componentBroadband`, reproduce the existing temperature and pain masks, and compute the same 1–100 Hz `pow*` result set.
- [ ] Save `comp`, `ica_band`, `ica_band_frequency_hz`, all six temperature powers and dB versions, painful/non-painful powers and contrasts, high/low powers and contrast, temperature slope, and overall average to:

```text
pow/<band>/sub-0015_ICA_pow_EEG.mat
```

### Task 4: Add and run the sub-0015 entry point

**Files:**
- Create: `studies/pain_study/fieldtrip_tfr/run_BrainVision_Analyzer_band_ICA_TFR_sub0015.m`

- [ ] Run the Python export for `sub-0015` from the configured pre-ICA derivative root.
- [ ] Call `prepareBandIcaBatch` followed by `computeBandTfrBatch` with explicit overwrite control.
- [ ] Run MATLAB Code Analyzer on all new MATLAB files and require zero findings.
- [ ] Execute the runner and verify three ICA MAT files, three ICA provenance JSON files, and three TFR MAT files exist and contain their declared band identity.
