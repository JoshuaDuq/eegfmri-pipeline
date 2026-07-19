# Colleague-Style FieldTrip Plots Design

## Objective

Replace the existing `sub-0015` component-TFR plotting scripts with adaptations of the
provided `plot_ICA_TFR.m` and `plot_TFR.m`. Plot every saved temperature, pain, grouped
temperature, contrast, slope, and average result using the colleague's separate topomap,
component-grid, and sensor-multiplot figure structure.

## Component Plotter

Create `plot_ICA_TFR_sub0015.m`. The user selects one of `alpha`, `beta`, or `gamma` at the
top of the script. Load the matching band-specific ICA and TFR files.

First create four ICA scalp-map figures with components 1-16, 17-32, 33-48, and 49-62 using
`ft_topoplotIC` and the `turbo` colormap. Then, for each of the 22 saved TFR result variables,
create four 4-by-4 figures containing the same component groups. Each tile uses `imagesc`,
`axis xy`, `turbo`, an individual colorbar, the component number, and the configured time and
frequency limits. The last TFR center is 14.4 s so the one-second window remains strictly
inside the -7 to 15 s epochs. Signed dB results and contrasts use symmetric limits around zero; strictly
nonnegative raw-power results use limits from zero to their component maximum.

Do not repeat topomaps inside condition figures. ICA topographies remain fixed within each
band and appear once in the initial four figures.

## Sensor TFR Computation

Compute one sensor-space TFR result set from the exported 59 retained pre-ICA epochs. Sensor
power does not depend on the alpha, beta, or gamma ICA decomposition, so save a single file:

```text
pow/sensor/sub-0015_pow_EEG.mat
```

Use the existing FieldTrip TFR parameters and exactly the same condition masks, baseline
normalization, contrasts, weighted averages, and temperature slope as the band-component
analysis. Store the original sensor data as `data` alongside the 22 `pow*` results. Extract
shared condition and TFR calculations into focused MATLAB functions so component and sensor
outputs cannot diverge.

## Sensor Plotter

Create `plot_TFR_sub0015.m`. Load the sensor TFR file and create one interactive
`ft_multiplotTFR` figure for each of its 22 results. Use `turbo`, the saved electrode geometry,
the configured time/frequency axes, and symmetric zero-centered limits for signed results.
Use zero-to-maximum limits for raw nonnegative power. Figure names identify the participant
and result variable.

## Replacement Scope

Remove the redundant sub-0015 plotting scripts:

- `plot_all_component_TFRs_sub0015.m`
- `plot_BrainVision_Analyzer_TFR_sub0015.m`
- `plot_FieldTrip_TFR.m`

Add the sensor computation to the existing sub-0015 runner after band-specific TFR computation
so future full runs generate both output families. Keep no obsolete plotting implementation.

## Validation and Errors

Fail on missing files or variables, invalid band names, incompatible component ordering,
missing electrode geometry, more than 64 components, non-finite power/topography values, or
constant plot values. Run MATLAB Code Analyzer on all new or modified MATLAB files. Verify the
sensor output contains 59 trials, 63 EEG channels, frequencies 1-100 Hz, times -5 to 14.4 s,
and all 22 declared results before plotting.
