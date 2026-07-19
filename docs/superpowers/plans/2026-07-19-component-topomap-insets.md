# Component Topomap Insets Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the matching band-specific ICA scalp map as an inset in every component TFR tile.

**Architecture:** The existing plotting script loads the selected band's TFR file and matching ICA decomposition. It validates their component identity once, prepares one electrode layout, and passes explicit MATLAB axes handles to FieldTrip so each TFR tile receives a fixed ICA-map inset without creating extra figures.

**Tech Stack:** MATLAB R2026a, FieldTrip `ft_singleplotTFR`, `ft_prepare_layout`, and `ft_topoplotIC`.

---

### Task 1: Load and validate the matching ICA maps

**Files:**
- Modify: `studies/pain_study/fieldtrip_tfr/plot_all_component_TFRs_sub0015.m`

- [ ] **Step 1: Load the selected band's ICA decomposition**

Build `icaFile` under `ica/<band>/sub-0015/eeg`, require it to exist, and load
`componentFit` only.

- [ ] **Step 2: Validate the plotting contract**

Require `componentFit.label`, `componentFit.topo`, `componentFit.topolabel`, and electrode
geometry. Before plotting each TFR result, require exact equality between
`string(power.label)` and `string(componentFit.label)` so a topomap cannot be attached to the
wrong component.

- [ ] **Step 3: Prepare the electrode layout once**

Use:

```matlab
layoutConfig = [];
layoutConfig.elec = componentFit.elec;
componentLayout = ft_prepare_layout(layoutConfig);
```

Expected: one reusable FieldTrip layout matching the 63-channel ICA topology.

### Task 2: Draw one topomap inset per TFR tile

**Files:**
- Modify: `studies/pain_study/fieldtrip_tfr/plot_all_component_TFRs_sub0015.m`

- [ ] **Step 1: Retain the main TFR axes handle**

Store the axes returned by `subplot(4, 4, subplotIndex)` and pass that handle through
`cfg.figure` to `ft_singleplotTFR`.

- [ ] **Step 2: Create a bounded inset axes**

Derive a small position inside the upper-right corner of the TFR axes. Create an axes handle
at that position without changing the four-figure, 16-component layout.

- [ ] **Step 3: Plot the fixed component map into the inset**

Call `ft_topoplotIC` with:

```matlab
topographyConfig = [];
topographyConfig.component = componentNumber;
topographyConfig.layout = componentLayout;
topographyConfig.figure = topographyAxes;
topographyConfig.zlim = "maxabs";
topographyConfig.marker = "off";
topographyConfig.comment = "no";
topographyConfig.title = "off";
topographyConfig.colorbar = "no";
ft_topoplotIC(topographyConfig, componentFit);
```

Expected: the selected band's fixed ICA scalp map appears in every matching component tile,
while the surrounding TFR varies by condition.

### Task 3: Verify the script

**Files:**
- Verify: `studies/pain_study/fieldtrip_tfr/plot_all_component_TFRs_sub0015.m`

- [ ] **Step 1: Run MATLAB Code Analyzer**

Run `checkcode` on the plotting script in batch mode.

Expected: no Code Analyzer findings.

- [ ] **Step 2: Validate saved alpha data without opening figures**

Load the alpha ICA and TFR MAT files and assert exact component-label equality, 62 component
maps, 63 topography channels, and finite topography values.

Expected: MATLAB exits successfully and prints the validated component and channel counts.

- [ ] **Step 3: Commit the implementation**

```bash
git add studies/pain_study/fieldtrip_tfr/plot_all_component_TFRs_sub0015.m
git commit -m "feat: add component maps to TFR figures"
```
