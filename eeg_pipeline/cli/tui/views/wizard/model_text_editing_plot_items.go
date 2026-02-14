package wizard

import "strings"

// Plot-item specific text-field getters/setters and value validation.

func (m Model) getPlotItemTextFieldValue(plotID string, field plotItemConfigField) string {
	cfg, ok := m.plotItemConfigs[plotID]
	if !ok {
		return ""
	}
	switch field {
	case plotItemConfigFieldTfrDefaultBaselineWindow:
		return cfg.TfrDefaultBaselineWindowSpec
	case plotItemConfigFieldComparisonWindows:
		return cfg.ComparisonWindowsSpec
	case plotItemConfigFieldComparisonSegment:
		return cfg.ComparisonSegment
	case plotItemConfigFieldComparisonColumn:
		return cfg.ComparisonColumn
	case plotItemConfigFieldComparisonValues:
		return cfg.ComparisonValuesSpec
	case plotItemConfigFieldComparisonLabels:
		return cfg.ComparisonLabelsSpec
	case plotItemConfigFieldComparisonROIs:
		return cfg.ComparisonROIsSpec
	case plotItemConfigFieldTopomapWindow:
		return cfg.TopomapWindowsSpec
	case plotItemConfigFieldTfrTopomapActiveWindow:
		return cfg.TfrTopomapActiveWindow
	case plotItemConfigFieldTfrTopomapWindowSizeMs:
		return cfg.TfrTopomapWindowSizeMs
	case plotItemConfigFieldTfrTopomapWindowCount:
		return cfg.TfrTopomapWindowCount
	case plotItemConfigFieldTfrTopomapLabelXPosition:
		return cfg.TfrTopomapLabelXPosition
	case plotItemConfigFieldTfrTopomapLabelYPositionBottom:
		return cfg.TfrTopomapLabelYPositionBottom
	case plotItemConfigFieldTfrTopomapLabelYPosition:
		return cfg.TfrTopomapLabelYPosition
	case plotItemConfigFieldTfrTopomapTitleY:
		return cfg.TfrTopomapTitleY
	case plotItemConfigFieldTfrTopomapTitlePad:
		return cfg.TfrTopomapTitlePad
	case plotItemConfigFieldTfrTopomapSubplotsRight:
		return cfg.TfrTopomapSubplotsRight
	case plotItemConfigFieldTfrTopomapTemporalHspace:
		return cfg.TfrTopomapTemporalHspace
	case plotItemConfigFieldTfrTopomapTemporalWspace:
		return cfg.TfrTopomapTemporalWspace
	case plotItemConfigFieldConnectivityCircleTopFraction:
		return cfg.ConnectivityCircleTopFraction
	case plotItemConfigFieldConnectivityCircleMinLines:
		return cfg.ConnectivityCircleMinLines
	case plotItemConfigFieldConnectivityNetworkTopFraction:
		return cfg.ConnectivityNetworkTopFraction
	case plotItemConfigFieldBehaviorTemporalStatsFeatureFolder:
		return cfg.BehaviorTemporalStatsFeatureFolder
	case plotItemConfigFieldDoseResponseDoseColumn:
		return cfg.DoseResponseDoseColumn
	case plotItemConfigFieldDoseResponseResponseColumn:
		return cfg.DoseResponseResponseColumn
	case plotItemConfigFieldDoseResponsePainColumn:
		return cfg.DoseResponsePainColumn
	case plotItemConfigFieldDoseResponseSegment:
		return cfg.DoseResponseSegment
	default:
		return ""
	}
}

func (m *Model) ensurePlotItemConfig(plotID string) PlotItemConfig {
	if m.plotItemConfigs == nil {
		m.plotItemConfigs = make(map[string]PlotItemConfig)
	}
	cfg, ok := m.plotItemConfigs[plotID]
	if !ok {
		cfg = PlotItemConfig{}
		m.plotItemConfigs[plotID] = cfg
	}
	return cfg
}

func (m *Model) setPlotItemTextFieldValue(plotID string, field plotItemConfigField, value string) {
	cfg := m.ensurePlotItemConfig(plotID)
	switch field {
	case plotItemConfigFieldTfrDefaultBaselineWindow:
		cfg.TfrDefaultBaselineWindowSpec = strings.Join(strings.Fields(value), " ")
	case plotItemConfigFieldComparisonWindows:
		cfg.ComparisonWindowsSpec = strings.Join(strings.Fields(value), " ")
		featureGroup := m.getFeatureGroupForPlot(plotID)
		windows := m.GetPlottingComparisonWindows(featureGroup)
		if len(windows) > 0 {
			unknown := unknownFromList(strings.Fields(cfg.ComparisonWindowsSpec), windows)
			if len(unknown) > 0 {
				m.ShowToast("Unknown window(s): "+strings.Join(unknown, ", "), "warning")
			}
		}
	case plotItemConfigFieldComparisonSegment:
		cfg.ComparisonSegment = strings.TrimSpace(value)
		featureGroup := m.getFeatureGroupForPlot(plotID)
		windows := m.GetPlottingComparisonWindows(featureGroup)
		if cfg.ComparisonSegment != "" && len(windows) > 0 {
			unknown := unknownFromList([]string{cfg.ComparisonSegment}, windows)
			if len(unknown) > 0 {
				m.ShowToast("Unknown segment: "+unknown[0], "warning")
			}
		}
	case plotItemConfigFieldComparisonColumn:
		cfg.ComparisonColumn = strings.TrimSpace(value)
		if cfg.ComparisonColumn != "" && len(m.availableColumns) > 0 {
			unknown := unknownFromList([]string{cfg.ComparisonColumn}, m.availableColumns)
			if len(unknown) > 0 {
				m.ShowToast("Unknown events.tsv column: "+unknown[0], "warning")
			}
		}
	case plotItemConfigFieldComparisonValues:
		cfg.ComparisonValuesSpec = strings.Join(strings.Fields(value), " ")
	case plotItemConfigFieldComparisonLabels:
		cfg.ComparisonLabelsSpec = strings.TrimSpace(value)
	case plotItemConfigFieldComparisonROIs:
		cfg.ComparisonROIsSpec = strings.Join(strings.Fields(value), " ")
	case plotItemConfigFieldTopomapWindow:
		cfg.TopomapWindowsSpec = strings.Join(strings.Fields(value), " ")
		featureGroup := m.getFeatureGroupForPlot(plotID)
		windows := m.GetPlottingComparisonWindows(featureGroup)
		if cfg.TopomapWindowsSpec != "" && len(windows) > 0 {
			unknown := unknownFromList(strings.Fields(cfg.TopomapWindowsSpec), windows)
			if len(unknown) > 0 {
				m.ShowToast("Unknown window: "+unknown[0], "warning")
			}
		}
	case plotItemConfigFieldTfrTopomapActiveWindow:
		cfg.TfrTopomapActiveWindow = strings.Join(strings.Fields(value), " ")
	case plotItemConfigFieldTfrTopomapWindowSizeMs:
		cfg.TfrTopomapWindowSizeMs = strings.TrimSpace(value)
	case plotItemConfigFieldTfrTopomapWindowCount:
		cfg.TfrTopomapWindowCount = strings.TrimSpace(value)
	case plotItemConfigFieldTfrTopomapLabelXPosition:
		cfg.TfrTopomapLabelXPosition = strings.TrimSpace(value)
	case plotItemConfigFieldTfrTopomapLabelYPositionBottom:
		cfg.TfrTopomapLabelYPositionBottom = strings.TrimSpace(value)
	case plotItemConfigFieldTfrTopomapLabelYPosition:
		cfg.TfrTopomapLabelYPosition = strings.TrimSpace(value)
	case plotItemConfigFieldTfrTopomapTitleY:
		cfg.TfrTopomapTitleY = strings.TrimSpace(value)
	case plotItemConfigFieldTfrTopomapTitlePad:
		cfg.TfrTopomapTitlePad = strings.TrimSpace(value)
	case plotItemConfigFieldTfrTopomapSubplotsRight:
		cfg.TfrTopomapSubplotsRight = strings.TrimSpace(value)
	case plotItemConfigFieldTfrTopomapTemporalHspace:
		cfg.TfrTopomapTemporalHspace = strings.TrimSpace(value)
	case plotItemConfigFieldTfrTopomapTemporalWspace:
		cfg.TfrTopomapTemporalWspace = strings.TrimSpace(value)
	case plotItemConfigFieldConnectivityCircleTopFraction:
		cfg.ConnectivityCircleTopFraction = strings.TrimSpace(value)
	case plotItemConfigFieldConnectivityCircleMinLines:
		cfg.ConnectivityCircleMinLines = strings.TrimSpace(value)
	case plotItemConfigFieldConnectivityNetworkTopFraction:
		cfg.ConnectivityNetworkTopFraction = strings.TrimSpace(value)
	case plotItemConfigFieldBehaviorTemporalStatsFeatureFolder:
		cfg.BehaviorTemporalStatsFeatureFolder = strings.TrimSpace(value)
	case plotItemConfigFieldDoseResponseDoseColumn:
		cfg.DoseResponseDoseColumn = strings.TrimSpace(value)
		if cfg.DoseResponseDoseColumn != "" && len(m.availableColumns) > 0 {
			unknown := unknownFromList([]string{cfg.DoseResponseDoseColumn}, m.availableColumns)
			if len(unknown) > 0 {
				m.ShowToast("Unknown events.tsv column: "+unknown[0], "warning")
			}
		}
	case plotItemConfigFieldDoseResponseResponseColumn:
		cfg.DoseResponseResponseColumn = strings.Join(strings.Fields(value), " ")
		if cfg.DoseResponseResponseColumn != "" && len(m.trialTableFeatureCategories) > 0 {
			unknown := unknownFromList(strings.Fields(cfg.DoseResponseResponseColumn), m.trialTableFeatureCategories)
			if len(unknown) > 0 {
				m.ShowToast("Unknown feature category: "+unknown[0], "warning")
			}
		}
	case plotItemConfigFieldDoseResponsePainColumn:
		cfg.DoseResponsePainColumn = strings.TrimSpace(value)
		if cfg.DoseResponsePainColumn != "" && len(m.availableColumns) > 0 {
			unknown := unknownFromList([]string{cfg.DoseResponsePainColumn}, m.availableColumns)
			if len(unknown) > 0 {
				m.ShowToast("Unknown events.tsv column: "+unknown[0], "warning")
			}
		}
	case plotItemConfigFieldDoseResponseSegment:
		cfg.DoseResponseSegment = strings.TrimSpace(value)
		// Segment is a feature window label, not necessarily an events.tsv column.
	default:
		return
	}
	m.plotItemConfigs[plotID] = cfg
}

func unknownFromList(requested []string, available []string) []string {
	availableSet := make(map[string]struct{}, len(available))
	for _, item := range available {
		availableSet[strings.ToLower(strings.TrimSpace(item))] = struct{}{}
	}
	var unknown []string
	for _, item := range requested {
		item = strings.TrimSpace(item)
		if item == "" {
			continue
		}
		if _, ok := availableSet[strings.ToLower(item)]; !ok {
			unknown = append(unknown, item)
		}
	}
	return unknown
}
