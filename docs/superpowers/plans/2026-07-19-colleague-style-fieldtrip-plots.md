# Colleague-Style FieldTrip Plots Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce colleague-style component and sensor TFR plots for every `sub-0015` condition and contrast.

**Architecture:** Extract the duplicated condition/TFR calculations into two shared MATLAB functions, then use them from the existing component batches and a new one-time sensor batch. Replace the sub-0015 inset viewer with separate ICA-topomap and component-grid plots, and add a sensor `ft_multiplotTFR` viewer.

**Tech Stack:** MATLAB R2026a, FieldTrip, runica outputs, MATLAB v7.3 MAT files.

---

### Task 1: Share condition and TFR calculations

**Files:**
- Create: `studies/pain_study/fieldtrip_tfr/matlab/buildTfrConditionMasks.m`
- Create: `studies/pain_study/fieldtrip_tfr/matlab/computeTfrResultSet.m`
- Modify: `studies/pain_study/fieldtrip_tfr/matlab/computeBandTfrBatch.m`
- Modify: `studies/pain_study/fieldtrip_tfr/matlab/computeTfrBatch.m`

- [ ] **Step 1: Extract metadata validation and masks**

Move the existing run/trial uniqueness checks, six temperature masks, and painful/non-painful
masks into:

```matlab
function masks = buildTfrConditionMasks(config, metadata)
```

Require all configured conditions to contain trials and preserve configured low/high
temperature definitions.

- [ ] **Step 2: Extract the result calculation**

Move the existing `mtmconvol`, dB baseline, log-ratio, weighted-average, and temperature-slope
logic into:

```matlab
function results = computeTfrResultSet(config, data, masks)
```

Return the same 22 `pow*` structures without changing numerical settings.

- [ ] **Step 3: Replace nested duplicates**

Update both component batches to call the shared functions and remove their local copies.

Expected: the public batch signatures and saved variables remain unchanged.

### Task 2: Compute one sensor TFR set

**Files:**
- Create: `studies/pain_study/fieldtrip_tfr/matlab/computeSensorTfrBatch.m`
- Modify: `studies/pain_study/fieldtrip_tfr/run_BrainVision_Analyzer_band_ICA_TFR_sub0015.m`

- [ ] **Step 1: Add the sensor batch**

Load each FieldTrip export, use `package.broadband_data`, build the shared masks, and calculate
the shared result set. Save `data` and all `pow*` fields to:

```text
pow/sensor/sub-0015_pow_EEG.mat
```

Fail if the output exists unless `Overwrite=true`.

- [ ] **Step 2: Add sensor computation to the full runner**

After `computeBandTfrBatch`, call:

```matlab
computeSensorTfrBatch(runtimeConfig, Subjects=subject, Overwrite=true);
```

Expected: future full runs produce component and sensor outputs together.

### Task 3: Replace the component viewer

**Files:**
- Create: `studies/pain_study/fieldtrip_tfr/matlab/tfrPlotVariables.m`
- Create: `studies/pain_study/fieldtrip_tfr/plot_ICA_TFR_sub0015.m`
- Delete: `studies/pain_study/fieldtrip_tfr/plot_all_component_TFRs_sub0015.m`
- Delete: `studies/pain_study/fieldtrip_tfr/plot_BrainVision_Analyzer_TFR_sub0015.m`
- Delete: `studies/pain_study/fieldtrip_tfr/plot_FieldTrip_TFR.m`

- [ ] **Step 1: Define the exact plot registry**

Return a table containing all 22 saved variable names and whether each is signed. Raw power
is nonnegative; dB, ratios, and temperature slope are signed.

- [ ] **Step 2: Plot four ICA-map figures**

Load the selected band and use `ft_topoplotIC` with groups `1:16`, `17:32`, `33:48`, and
`49:62`, the saved electrode geometry, and `turbo`.

- [ ] **Step 3: Plot every component TFR result**

For every registry row, open four 4-by-4 figures and use:

```matlab
imagesc(power.time, power.freq, ...
    squeeze(power.powspctrm(componentNumber, :, :)), colorLimits);
axis xy;
colormap turbo;
colorbar;
```

Use symmetric per-component limits for signed values and `[0, maximum]` for raw power.

### Task 4: Add the sensor viewer

**Files:**
- Create: `studies/pain_study/fieldtrip_tfr/plot_TFR_sub0015.m`

- [ ] **Step 1: Load and validate the sensor file**

Require `data.elec` and every registry variable, with 63 channel labels and finite power.

- [ ] **Step 2: Plot every sensor result**

Prepare one layout from `data.elec`. For every registry row, open a named figure and call
`ft_multiplotTFR` with `interactive="yes"`, `turbo`, full saved axes, and a result-wide
signed or nonnegative color range.

Expected: 22 interactive channel-layout figures.

### Task 5: Verify and produce the sensor output

**Files:**
- Verify all MATLAB files above.
- Create externally: `pow/sensor/sub-0015_pow_EEG.mat`

- [ ] **Step 1: Run MATLAB Code Analyzer**

Run `checkcode` on every new or modified MATLAB file and require zero findings.

- [ ] **Step 2: Execute only the sensor batch**

Call `computeSensorTfrBatch` with the existing runtime JSON, `sub-0015`, and overwrite enabled.
Do not rerun ICA or component TFRs.

- [ ] **Step 3: Verify the sensor output contract**

Load `data` and representative raw, dB, and contrast results. Assert 59 trials, 63 channels,
frequencies `1:100`, times `-5:0.1:14.4`, finite power, and all 22 registry fields.

- [ ] **Step 4: Render one invisible component grid and one sensor multiplot**

Export both to temporary PNG files and require MATLAB to exit successfully.

- [ ] **Step 5: Commit only the scoped files**

```bash
git add studies/pain_study/fieldtrip_tfr
git commit -m "feat: add colleague-style FieldTrip plots"
```
