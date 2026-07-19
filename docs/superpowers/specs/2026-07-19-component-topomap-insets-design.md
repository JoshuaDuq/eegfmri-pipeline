# Component Topomap Insets Design

## Objective

Extend the `sub-0015` band-specific TFR viewer so every component TFR tile also shows the
component's ICA scalp topography. Preserve the existing four 4-by-4 figures per saved TFR
result and the explicit alpha, beta, or gamma band selection.

## Data Sources

Load condition-specific TFR values from:

```text
pow/<band>/sub-0015_ICA_pow_EEG.mat
```

Load fixed component scalp maps from the matching decomposition:

```text
ica/<band>/sub-0015/eeg/
sub-0015_task-thermalactive_desc-fieldtripica_components.mat
```

Require identical component labels and counts between `componentFit` and every plotted TFR.
Do not reinterpret ICA maps as condition-specific: the inset remains fixed within a band,
while the surrounding TFR changes with the selected condition or contrast.

## Figure Layout

For each of the 22 saved TFR results, create four full-screen figures containing at most 16
components each. Every 4-by-4 component tile contains:

- the component's 1-100 Hz TFR as the main axes;
- a stimulation-onset line at zero seconds;
- the component label;
- a small scalp-topography inset from the matching ICA model.

Use one TFR color scale across all components for a given result. Use one symmetric scalp-map
color scale per component because ICA topography polarity and amplitude are arbitrary. Keep
the topomap free of a separate colorbar to avoid obscuring the TFR.

## Validation and Errors

Fail on missing ICA or TFR files, missing variables, mismatched component labels/counts,
more than 64 components, constant TFR values, or missing electrode geometry. Use FieldTrip's
standard plotting functions and MATLAB figures; add no custom browser or review interface.
