package wizard

// Static lookup data used by plotting field-selection helpers.

var comparisonCapablePlotIDs = map[string]struct{}{
	// Aperiodic
	"aperiodic_topomaps":     {},
	"aperiodic_by_condition": {},
	// Connectivity
	"connectivity_by_condition":     {},
	"connectivity_circle_condition": {},
	"connectivity_network":          {},
	// Feature families
	"erds_by_condition":       {},
	"complexity_by_condition": {},
	"spectral_by_condition":   {},
	"ratios_by_condition":     {},
	"asymmetry_by_condition":  {},
	"bursts_by_condition":     {},
	"itpc_by_condition":       {},
	"pac_by_condition":        {},
	"power_by_condition":      {},
	"power_spectral_density":  {},
	// ERP
	"erp_butterfly": {},
	"erp_roi":       {},
	"erp_contrast":  {},
	// TFR contrast plots
	"tfr_scalpmean_contrast": {},
	"tfr_channels_contrast":  {},
	"tfr_rois_contrast":      {},
	"tfr_topomaps":           {},
	"tfr_band_evolution":     {},
}

var preComparisonFieldsByPlotID = map[string][]plotItemConfigField{
	"behavior_temporal_topomaps": {
		plotItemConfigFieldBehaviorTemporalStatsFeatureFolder,
	},
	"behavior_dose_response": {
		plotItemConfigFieldDoseResponseDoseColumn,
		plotItemConfigFieldDoseResponseResponseColumn,
		plotItemConfigFieldDoseResponseSegment,
		plotItemConfigFieldDoseResponseBands,
		plotItemConfigFieldDoseResponseROIs,
		plotItemConfigFieldDoseResponseScopes,
		plotItemConfigFieldDoseResponseStat,
	},
	"behavior_binary_outcome_probability": {
		plotItemConfigFieldDoseResponseDoseColumn,
		plotItemConfigFieldDoseResponseBinaryOutcomeColumn,
	},
	"band_power_topomaps": {
		plotItemConfigFieldTopomapWindow,
		plotItemConfigFieldCompareWindows,
		plotItemConfigFieldCompareColumns,
		plotItemConfigFieldComparisonWindows,
		plotItemConfigFieldComparisonColumn,
		plotItemConfigFieldComparisonValues,
		plotItemConfigFieldComparisonLabels,
	},
}

var comparisonFieldProfilesByPlotID = map[string][]plotItemConfigField{
	"connectivity_circle_condition": {
		plotItemConfigFieldConnectivityCircleTopFraction,
		plotItemConfigFieldConnectivityCircleMinLines,
		plotItemConfigFieldComparisonSegment,
		plotItemConfigFieldComparisonColumn,
		plotItemConfigFieldComparisonValues,
		plotItemConfigFieldComparisonLabels,
	},
	"connectivity_network": {
		plotItemConfigFieldConnectivityNetworkTopFraction,
		plotItemConfigFieldComparisonSegment,
		plotItemConfigFieldComparisonColumn,
		plotItemConfigFieldComparisonValues,
		plotItemConfigFieldComparisonLabels,
	},
	// These plots only use column comparisons.
	"erp_butterfly": {
		plotItemConfigFieldCompareColumns,
		plotItemConfigFieldComparisonColumn,
		plotItemConfigFieldComparisonValues,
		plotItemConfigFieldComparisonLabels,
	},
	"erp_roi": {
		plotItemConfigFieldCompareColumns,
		plotItemConfigFieldComparisonColumn,
		plotItemConfigFieldComparisonValues,
		plotItemConfigFieldComparisonLabels,
		plotItemConfigFieldComparisonROIs,
	},
	"erp_contrast": {
		plotItemConfigFieldCompareColumns,
		plotItemConfigFieldComparisonColumn,
		plotItemConfigFieldComparisonValues,
		plotItemConfigFieldComparisonLabels,
	},
	"aperiodic_topomaps": {
		plotItemConfigFieldCompareColumns,
		plotItemConfigFieldComparisonColumn,
		plotItemConfigFieldComparisonValues,
		plotItemConfigFieldComparisonLabels,
	},
	// CompareColumns toggle not needed; column comparison is always required.
	"power_spectral_density": {
		plotItemConfigFieldComparisonColumn,
		plotItemConfigFieldComparisonValues,
		plotItemConfigFieldComparisonLabels,
	},
}

var extraPlotFieldsByPlotID = map[string][]plotItemConfigField{
	"itpc_topomaps": {
		plotItemConfigFieldItpcSharedColorbar,
	},
	"behavior_scatter": {
		plotItemConfigFieldBehaviorScatterFeatures,
		plotItemConfigFieldBehaviorScatterColumns,
		plotItemConfigFieldBehaviorScatterAggregationModes,
		plotItemConfigFieldBehaviorScatterSegment,
	},
	"tfr_topomaps": {
		plotItemConfigFieldTfrTopomapActiveWindow,
		plotItemConfigFieldTfrTopomapWindowSizeMs,
		plotItemConfigFieldTfrTopomapWindowCount,
		plotItemConfigFieldTfrTopomapLabelXPosition,
		plotItemConfigFieldTfrTopomapLabelYPositionBottom,
		plotItemConfigFieldTfrTopomapLabelYPosition,
		plotItemConfigFieldTfrTopomapTitleY,
		plotItemConfigFieldTfrTopomapTitlePad,
		plotItemConfigFieldTfrTopomapSubplotsRight,
		plotItemConfigFieldTfrTopomapTemporalHspace,
		plotItemConfigFieldTfrTopomapTemporalWspace,
	},
	"source_localization_3d": {
		plotItemConfigFieldSourceHemi,
		plotItemConfigFieldSourceViews,
		plotItemConfigFieldSourceCortex,
		plotItemConfigFieldSourceSubjectsDir,
	},
}

var roiComparisonExcludedPlotIDs = map[string]struct{}{
	"tfr_rois":               {},
	"tfr_channels_contrast":  {},
	"tfr_scalpmean_contrast": {},
}
