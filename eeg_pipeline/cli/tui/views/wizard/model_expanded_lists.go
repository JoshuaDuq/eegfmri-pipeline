package wizard

import "strings"

// Expanded-list selection and option helpers.

func (m Model) getExpandedListLength() int {
	switch m.expandedOption {
	case expandedConnectivityMeasures:
		return len(connectivityMeasures)
	case expandedDirectedConnMeasures:
		return len(directedConnectivityMeasures)
	case expandedConditionCompareColumn, expandedTemporalConditionColumn, expandedClusterConditionColumn:
		return len(m.GetAvailableColumns())
	case expandedConnConditionColumn:
		return len(m.GetAvailableColumns())
	case expandedConditionCompareValues:
		if m.conditionCompareColumn == "" {
			return 0
		}
		return len(m.GetDiscoveredColumnValues(m.conditionCompareColumn))
	case expandedTemporalConditionValues:
		if m.temporalConditionColumn == "" {
			return 0
		}
		return len(m.GetDiscoveredColumnValues(m.temporalConditionColumn))
	case expandedClusterConditionValues:
		if m.clusterConditionColumn == "" {
			return 0
		}
		return len(m.GetDiscoveredColumnValues(m.clusterConditionColumn))
	case expandedPlotComparisonColumn:
		if m.editingPlotField == plotItemConfigFieldDoseResponseResponseColumn {
			return len(m.GetTrialTableFeatureCategories())
		}
		return len(m.GetPlottingComparisonColumns())
	case expandedPlotComparisonValues:
		// Check plot-specific column first, then fall back to global
		col := ""
		if m.editingPlotID != "" {
			if cfg, ok := m.plotItemConfigs[m.editingPlotID]; ok && cfg.ComparisonColumn != "" {
				col = cfg.ComparisonColumn
			}
		}
		if col == "" {
			col = m.plotComparisonColumn
		}
		if col == "" {
			return 0
		}
		return len(m.GetPlottingComparisonColumnValues(col))
	case expandedConditionCompareWindows:
		return len(m.availableWindows)
	case expandedPlotComparisonWindows:
		if m.editingPlotID != "" {
			if m.editingPlotField == plotItemConfigFieldDoseResponseSegment {
				// Dose-response segment uses global windows list (same as renderer),
				// not feature-group-specific windows.
				return len(m.GetPlottingComparisonWindows())
			}
			featureGroup := m.getFeatureGroupForPlot(m.editingPlotID)
			return len(m.GetPlottingComparisonWindows(featureGroup))
		}
		return len(m.GetPlottingComparisonWindows())
	case expandedPlotComparisonROIs:
		return len(m.discoveredROIs)
	case expandedDoseResponseBands:
		return len(m.GetDoseResponseBands(m.getDoseResponseCategoriesForEditingPlot()))
	case expandedDoseResponseROIs:
		return len(m.GetDoseResponseROIs(m.getDoseResponseCategoriesForEditingPlot()))
	case expandedDoseResponseScopes:
		return len(m.GetDoseResponseScopes(m.getDoseResponseCategoriesForEditingPlot()))
	case expandedDoseResponseStat:
		return len(m.GetDoseResponseStats(m.getDoseResponseCategoriesForEditingPlot()))
	case expandedRunAdjustmentColumn:
		return len(m.GetAvailableColumns())
	case expandedCorrelationsTargetColumn:
		return len(m.GetAvailableColumns()) + 1 // +1 for "(none)" option
	case expandedTemporalTargetColumn:
		return len(m.GetAvailableColumns()) + 1 // +1 for "(default)" option
	case expandedMLTargetColumn:
		return len(m.GetAvailableColumns()) + 1 // +1 for "(stage default)" option
	case expandedMLFeatureFamilies:
		return len(m.mlFeatureFamiliesOptions())
	case expandedMLFeatureBands:
		return len(m.mlFeatureBandsOptions())
	case expandedMLFeatureSegments:
		return len(m.mlFeatureSegmentsOptions())
	case expandedMLFeatureScopes:
		return len(m.mlFeatureScopesOptions())
	case expandedMLFeatureStats:
		return len(m.mlFeatureStatsOptions())
	case expandedItpcConditionColumn:
		return len(m.GetAvailableColumns())
	case expandedConnConditionValues:
		if m.connConditionColumn == "" {
			return 0
		}
		return len(m.GetDiscoveredColumnValues(m.connConditionColumn))
	case expandedFmriCondAColumn, expandedFmriCondBColumn:
		return len(m.fmriDiscoveredColumns)
	case expandedFmriCondAValue:
		return len(m.GetFmriDiscoveredColumnValues(m.sourceLocFmriCondAColumn))
	case expandedFmriCondBValue:
		return len(m.GetFmriDiscoveredColumnValues(m.sourceLocFmriCondBColumn))
	case expandedFmriAnalysisCondAColumn, expandedFmriAnalysisCondBColumn:
		n := len(m.fmriDiscoveredColumns)
		if n == 0 {
			return 1
		}
		return n
	case expandedFmriAnalysisCondAValue:
		n := len(m.GetFmriDiscoveredColumnValues(m.fmriAnalysisCondAColumn))
		if n == 0 {
			return 1
		}
		return n
	case expandedFmriAnalysisCondBValue:
		n := len(m.GetFmriDiscoveredColumnValues(m.fmriAnalysisCondBColumn))
		if n == 0 {
			return 1
		}
		return n
	case expandedFmriAnalysisStimPhases:
		return len(m.getExpandedListItems())
	case expandedFmriAnalysisScopeTrialTypes:
		return len(m.getExpandedListItems())
	case expandedFmriTrialSigGroupColumn:
		n := len(m.fmriDiscoveredColumns)
		if n == 0 {
			return 1
		}
		return n
	case expandedFmriTrialSigGroupValues:
		n := len(m.GetFmriDiscoveredColumnValues(m.fmriTrialSigGroupColumn))
		if n == 0 {
			return 1
		}
		return n
	case expandedFmriTrialSigStimPhases:
		return len(m.getExpandedListItems())
	case expandedFmriTrialSigScopeTrialTypes:
		return len(m.getExpandedListItems())
	case expandedSourceLocFmriStimPhases:
		return len(m.getExpandedListItems())
	case expandedSourceLocFmriScopeTrialTypes:
		return len(m.getExpandedListItems())
	case expandedIAFRois:
		return len(m.rois)
	case expandedItpcConditionValues:
		if m.itpcConditionColumn == "" {
			return 0
		}
		return len(m.GetDiscoveredColumnValues(m.itpcConditionColumn))
	case expandedBehaviorScatterFeatures:
		return len(behaviorScatterFeatureTypes)
	case expandedBehaviorScatterColumns:
		return len(m.GetPlottingComparisonColumns())
	case expandedBehaviorScatterAggregation:
		return len(behaviorScatterAggregationModes)
	case expandedBehaviorScatterSegment:
		return len(m.GetPlottingComparisonWindows())
	case expandedTemporalTopomapsFeatureDir:
		return len(m.temporalTopomapsStatsFeatureFolders)
	case expandedPainResidualCrossfitGroupColumn:
		if len(m.GetAvailableColumns()) == 0 {
			return 2
		}
		return len(m.GetAvailableColumns()) + 1
	case expandedStabilityGroupColumn:
		return 3
	case expandedGroupLevelTarget:
		targets := m.availableGroupLevelTargets()
		if len(targets) == 0 {
			return 2
		}
		return len(targets) + 1
	}
	return 0
}

// getExpandedListItems returns the items in the currently expanded list
func (m Model) getExpandedListItems() []string {
	switch m.expandedOption {
	case expandedConnectivityMeasures:
		items := make([]string, len(connectivityMeasures))
		for i, measure := range connectivityMeasures {
			items[i] = measure.Key
		}
		return items
	case expandedDirectedConnMeasures:
		items := make([]string, len(directedConnectivityMeasures))
		for i, measure := range directedConnectivityMeasures {
			items[i] = measure.Key
		}
		return items
	case expandedConditionCompareColumn, expandedTemporalConditionColumn, expandedClusterConditionColumn:
		return m.GetAvailableColumns()
	case expandedConnConditionColumn:
		return m.GetAvailableColumns()
	case expandedConditionCompareValues:
		if m.conditionCompareColumn == "" {
			return nil
		}
		return m.GetDiscoveredColumnValues(m.conditionCompareColumn)
	case expandedTemporalConditionValues:
		if m.temporalConditionColumn == "" {
			return nil
		}
		return m.GetDiscoveredColumnValues(m.temporalConditionColumn)
	case expandedClusterConditionValues:
		if m.clusterConditionColumn == "" {
			return nil
		}
		return m.GetDiscoveredColumnValues(m.clusterConditionColumn)
	case expandedPlotComparisonColumn:
		if m.editingPlotField == plotItemConfigFieldDoseResponseResponseColumn {
			return m.GetTrialTableFeatureCategories()
		}
		return m.GetPlottingComparisonColumns()
	case expandedPlotComparisonValues:
		// Check plot-specific column first, then fall back to global
		col := ""
		if m.editingPlotID != "" {
			if cfg, ok := m.plotItemConfigs[m.editingPlotID]; ok && cfg.ComparisonColumn != "" {
				col = cfg.ComparisonColumn
			}
		}
		if col == "" {
			col = m.plotComparisonColumn
		}
		if col == "" {
			return nil
		}
		return m.GetPlottingComparisonColumnValues(col)
	case expandedConditionCompareWindows:
		return m.availableWindows
	case expandedPlotComparisonWindows:
		if m.editingPlotID != "" {
			if m.editingPlotField == plotItemConfigFieldDoseResponseSegment {
				// Dose-response segment uses global windows list (same as renderer),
				// not feature-group-specific windows.
				return m.GetPlottingComparisonWindows()
			}
			featureGroup := m.getFeatureGroupForPlot(m.editingPlotID)
			return m.GetPlottingComparisonWindows(featureGroup)
		}
		return m.GetPlottingComparisonWindows()
	case expandedPlotComparisonROIs:
		return m.discoveredROIs
	case expandedDoseResponseBands:
		return m.GetDoseResponseBands(m.getDoseResponseCategoriesForEditingPlot())
	case expandedDoseResponseROIs:
		return m.GetDoseResponseROIs(m.getDoseResponseCategoriesForEditingPlot())
	case expandedDoseResponseScopes:
		return m.GetDoseResponseScopes(m.getDoseResponseCategoriesForEditingPlot())
	case expandedDoseResponseStat:
		return m.GetDoseResponseStats(m.getDoseResponseCategoriesForEditingPlot())
	case expandedRunAdjustmentColumn:
		return m.GetAvailableColumns()
	case expandedCorrelationsTargetColumn:
		return append([]string{"(none)"}, m.GetAvailableColumns()...)
	case expandedTemporalTargetColumn:
		return append([]string{"(default)"}, m.GetAvailableColumns()...)
	case expandedMLTargetColumn:
		return append([]string{"(stage default)"}, m.GetAvailableColumns()...)
	case expandedMLFeatureFamilies:
		return m.mlFeatureFamiliesOptions()
	case expandedMLFeatureBands:
		return m.mlFeatureBandsOptions()
	case expandedMLFeatureSegments:
		return m.mlFeatureSegmentsOptions()
	case expandedMLFeatureScopes:
		return m.mlFeatureScopesOptions()
	case expandedMLFeatureStats:
		return m.mlFeatureStatsOptions()
	case expandedItpcConditionColumn:
		return m.GetAvailableColumns()
	case expandedFmriCondAColumn, expandedFmriCondBColumn:
		return m.fmriDiscoveredColumns
	case expandedFmriCondAValue:
		return m.GetFmriDiscoveredColumnValues(m.sourceLocFmriCondAColumn)
	case expandedFmriCondBValue:
		return m.GetFmriDiscoveredColumnValues(m.sourceLocFmriCondBColumn)
	case expandedFmriAnalysisCondAColumn, expandedFmriAnalysisCondBColumn:
		if len(m.fmriDiscoveredColumns) == 0 {
			return []string{"(type manually)"}
		}
		return m.fmriDiscoveredColumns
	case expandedFmriAnalysisCondAValue:
		vals := m.GetFmriDiscoveredColumnValues(m.fmriAnalysisCondAColumn)
		if len(vals) == 0 {
			return []string{"(type manually)"}
		}
		return vals
	case expandedFmriAnalysisCondBValue:
		vals := m.GetFmriDiscoveredColumnValues(m.fmriAnalysisCondBColumn)
		if len(vals) == 0 {
			return []string{"(type manually)"}
		}
		return vals
	case expandedFmriAnalysisStimPhases:
		items := []string{"(none)", "(all)"}
		vals := m.GetFmriDiscoveredColumnValues("stim_phase")
		if len(vals) == 0 {
			return append(items, "(type manually)")
		}
		return append(items, vals...)
	case expandedFmriAnalysisScopeTrialTypes:
		items := []string{"(none)"}
		vals := m.GetFmriDiscoveredColumnValues("trial_type")
		if len(vals) == 0 {
			return append(items, "(type manually)")
		}
		return append(items, vals...)
	case expandedFmriTrialSigGroupColumn:
		if len(m.fmriDiscoveredColumns) == 0 {
			return []string{"(type manually)"}
		}
		return m.fmriDiscoveredColumns
	case expandedFmriTrialSigGroupValues:
		if m.fmriTrialSigGroupColumn == "" {
			return nil
		}
		vals := m.GetFmriDiscoveredColumnValues(m.fmriTrialSigGroupColumn)
		if len(vals) == 0 {
			return []string{"(type manually)"}
		}
		return vals
	case expandedFmriTrialSigStimPhases:
		items := []string{"(none)", "(all)"}
		vals := m.GetFmriDiscoveredColumnValues("stim_phase")
		if len(vals) == 0 {
			return append(items, "(type manually)")
		}
		return append(items, vals...)
	case expandedFmriTrialSigScopeTrialTypes:
		items := []string{"(none)"}
		vals := m.GetFmriDiscoveredColumnValues("trial_type")
		if len(vals) == 0 {
			return append(items, "(type manually)")
		}
		return append(items, vals...)
	case expandedSourceLocFmriStimPhases:
		items := []string{"(none)", "(all)"}
		vals := m.GetFmriDiscoveredColumnValues("stim_phase")
		if len(vals) == 0 {
			return append(items, "(type manually)")
		}
		return append(items, vals...)
	case expandedSourceLocFmriScopeTrialTypes:
		items := []string{"(none)"}
		vals := m.GetFmriDiscoveredColumnValues("trial_type")
		if len(vals) == 0 {
			return append(items, "(type manually)")
		}
		return append(items, vals...)
	case expandedIAFRois:
		items := make([]string, 0, len(m.rois))
		for _, roi := range m.rois {
			key := strings.TrimSpace(roi.Key)
			if key != "" {
				items = append(items, key)
			}
		}
		return items
	case expandedItpcConditionValues:
		if m.itpcConditionColumn == "" {
			return nil
		}
		return m.GetDiscoveredColumnValues(m.itpcConditionColumn)
	case expandedConnConditionValues:
		if m.connConditionColumn == "" {
			return nil
		}
		return m.GetDiscoveredColumnValues(m.connConditionColumn)
	case expandedBehaviorScatterFeatures:
		return behaviorScatterFeatureTypes
	case expandedBehaviorScatterColumns:
		return m.GetPlottingComparisonColumns()
	case expandedBehaviorScatterAggregation:
		return behaviorScatterAggregationModes
	case expandedBehaviorScatterSegment:
		return m.GetPlottingComparisonWindows()
	case expandedTemporalTopomapsFeatureDir:
		return m.temporalTopomapsStatsFeatureFolders
	case expandedPainResidualCrossfitGroupColumn:
		items := []string{"(default: run column)"}
		cols := m.GetAvailableColumns()
		if len(cols) == 0 {
			return append(items, "(type manually)")
		}
		return append(items, cols...)
	case expandedStabilityGroupColumn:
		return []string{"(auto)", "run", "block"}
	case expandedGroupLevelTarget:
		targets := m.availableGroupLevelTargets()
		if len(targets) == 0 {
			return []string{"(default)", "(type manually)"}
		}
		return append([]string{"(default)"}, targets...)
	}
	return nil
}

func (m Model) availableGroupLevelTargets() []string {
	cols := m.GetAvailableColumns()
	if len(cols) == 0 {
		return nil
	}
	out := make([]string, 0, len(cols))
	seen := make(map[string]struct{}, len(cols))
	for _, c := range cols {
		val := strings.TrimSpace(c)
		key := strings.ToLower(val)
		if val == "" {
			continue
		}
		if _, ok := seen[key]; ok {
			continue
		}
		seen[key] = struct{}{}
		out = append(out, val)
	}
	return out
}

// isColumnValueSelected checks if a value is selected for the current column context
func (m Model) isColumnValueSelected(value string) bool {
	var selectedValues string
	switch m.expandedOption {
	case expandedConditionCompareValues:
		selectedValues = m.conditionCompareValues
	case expandedTemporalConditionValues:
		selectedValues = m.temporalConditionValues
	case expandedClusterConditionValues:
		selectedValues = m.clusterConditionValues
	case expandedPlotComparisonValues:
		// Check plot-specific config first
		if m.editingPlotID != "" {
			if cfg, ok := m.plotItemConfigs[m.editingPlotID]; ok {
				selectedValues = cfg.ComparisonValuesSpec
			}
		}
		if selectedValues == "" {
			selectedValues = m.plotComparisonValuesSpec
		}
	case expandedConditionCompareWindows:
		selectedValues = m.conditionCompareWindows
	case expandedPlotComparisonWindows:
		// Check plot-specific config first
		if m.editingPlotID != "" {
			if cfg, ok := m.plotItemConfigs[m.editingPlotID]; ok {
				switch m.editingPlotField {
				case plotItemConfigFieldComparisonSegment:
					selectedValues = cfg.ComparisonSegment
				case plotItemConfigFieldDoseResponseSegment:
					selectedValues = cfg.DoseResponseSegment
				case plotItemConfigFieldTopomapWindow:
					selectedValues = cfg.TopomapWindowsSpec
				default:
					selectedValues = cfg.ComparisonWindowsSpec
				}
			}
		}
		if selectedValues == "" {
			selectedValues = m.plotComparisonWindowsSpec
		}
	case expandedPlotComparisonROIs:
		// Check plot-specific config first
		if m.editingPlotID != "" {
			if cfg, ok := m.plotItemConfigs[m.editingPlotID]; ok {
				selectedValues = cfg.ComparisonROIsSpec
			}
		}
	case expandedItpcConditionValues:
		selectedValues = m.itpcConditionValues
	case expandedConnConditionValues:
		selectedValues = m.connConditionValues
	case expandedFmriTrialSigGroupValues:
		selectedValues = m.fmriTrialSigGroupValuesSpec
	case expandedBehaviorScatterFeatures:
		if m.editingPlotID != "" {
			if cfg, ok := m.plotItemConfigs[m.editingPlotID]; ok {
				selectedValues = cfg.BehaviorScatterFeaturesSpec
			}
		}
	case expandedBehaviorScatterColumns:
		if m.editingPlotID != "" {
			if cfg, ok := m.plotItemConfigs[m.editingPlotID]; ok {
				selectedValues = cfg.BehaviorScatterColumnsSpec
			}
		}
	case expandedBehaviorScatterAggregation:
		if m.editingPlotID != "" {
			if cfg, ok := m.plotItemConfigs[m.editingPlotID]; ok {
				selectedValues = cfg.BehaviorScatterAggregationModesSpec
			}
		}
	case expandedBehaviorScatterSegment:
		if m.editingPlotID != "" {
			if cfg, ok := m.plotItemConfigs[m.editingPlotID]; ok {
				selectedValues = cfg.BehaviorScatterSegmentSpec
			}
		}
	case expandedDoseResponseBands:
		if m.editingPlotID != "" {
			if cfg, ok := m.plotItemConfigs[m.editingPlotID]; ok {
				selectedValues = cfg.DoseResponseBandsSpec
			}
		}
	case expandedDoseResponseROIs:
		if m.editingPlotID != "" {
			if cfg, ok := m.plotItemConfigs[m.editingPlotID]; ok {
				selectedValues = cfg.DoseResponseROIsSpec
			}
		}
	case expandedDoseResponseScopes:
		if m.editingPlotID != "" {
			if cfg, ok := m.plotItemConfigs[m.editingPlotID]; ok {
				selectedValues = cfg.DoseResponseScopesSpec
			}
		}
	case expandedDoseResponseStat:
		if m.editingPlotID != "" {
			if cfg, ok := m.plotItemConfigs[m.editingPlotID]; ok {
				selectedValues = cfg.DoseResponseStat
			}
		}
	case expandedMLFeatureFamilies:
		selectedValues = m.mlFeatureFamiliesSpec
	case expandedMLFeatureBands:
		selectedValues = m.mlFeatureBandsSpec
	case expandedMLFeatureSegments:
		selectedValues = m.mlFeatureSegmentsSpec
	case expandedMLFeatureScopes:
		selectedValues = m.mlFeatureScopesSpec
	case expandedMLFeatureStats:
		selectedValues = m.mlFeatureStatsSpec
	case expandedPlotComparisonColumn:
		if m.editingPlotID != "" && m.editingPlotField == plotItemConfigFieldDoseResponseResponseColumn {
			if cfg, ok := m.plotItemConfigs[m.editingPlotID]; ok {
				selectedValues = cfg.DoseResponseResponseColumn
			}
		}
	default:
		return false
	}
	if selectedValues == "" {
		return false
	}
	// Check if value is in space or comma-separated list
	for _, v := range strings.Fields(selectedValues) {
		if v == value {
			return true
		}
	}
	for _, v := range strings.Split(selectedValues, ",") {
		if strings.TrimSpace(v) == value {
			return true
		}
	}
	return false
}

// handleExpandedListToggle handles toggling items in expanded column/value lists
func (m *Model) handleExpandedListToggle() {
	items := m.getExpandedListItems()
	if m.subCursor < 0 || m.subCursor >= len(items) {
		return
	}

	selectedItem := items[m.subCursor]

	switch m.expandedOption {
	case expandedConnectivityMeasures:
		m.connectivityMeasures[m.subCursor] = !m.connectivityMeasures[m.subCursor]

	case expandedDirectedConnMeasures:
		m.directedConnMeasures[m.subCursor] = !m.directedConnMeasures[m.subCursor]

	case expandedConditionCompareColumn:
		m.conditionCompareColumn = selectedItem
		m.conditionCompareValues = "" // Reset values when column changes
		m.expandedOption = expandedNone
		m.subCursor = 0

	case expandedTemporalConditionColumn:
		m.temporalConditionColumn = selectedItem
		m.temporalConditionValues = "" // Reset values when column changes
		m.expandedOption = expandedNone
		m.subCursor = 0

	case expandedClusterConditionColumn:
		m.clusterConditionColumn = selectedItem
		m.clusterConditionValues = "" // Reset values when column changes
		m.expandedOption = expandedNone
		m.subCursor = 0

	case expandedConditionCompareValues:
		m.toggleColumnValue(selectedItem, &m.conditionCompareValues)

	case expandedTemporalConditionValues:
		m.toggleColumnValue(selectedItem, &m.temporalConditionValues)

	case expandedClusterConditionValues:
		m.toggleColumnValue(selectedItem, &m.clusterConditionValues)

	case expandedPlotComparisonColumn:
		// Update plot-specific config if editing a plot field, otherwise global
		if m.editingPlotID != "" {
			plotID := m.editingPlotID
			cfg := m.ensurePlotItemConfig(plotID)

			switch m.editingPlotField {
			case plotItemConfigFieldDoseResponseDoseColumn:
				cfg.DoseResponseDoseColumn = selectedItem
			case plotItemConfigFieldDoseResponseResponseColumn:
				// Multi-select feature categories (space-separated)
				m.toggleSpaceValue(selectedItem, &cfg.DoseResponseResponseColumn)
				m.plotItemConfigs[plotID] = cfg
				return
			case plotItemConfigFieldDoseResponsePainColumn:
				cfg.DoseResponsePainColumn = selectedItem
			default:
				cfg.ComparisonColumn = selectedItem
				cfg.ComparisonValuesSpec = "" // Reset values when column changes
			}

			m.plotItemConfigs[plotID] = cfg
			m.editingPlotID = ""
			m.editingPlotField = plotItemConfigFieldNone
		} else {
			m.plotComparisonColumn = selectedItem
			m.plotComparisonValuesSpec = "" // Reset values when column changes
		}
		m.expandedOption = expandedNone
		m.subCursor = 0

	case expandedPlotComparisonValues:
		// Update plot-specific config if editing a plot field, otherwise global
		if m.editingPlotID != "" {
			cfg := m.ensurePlotItemConfig(m.editingPlotID)
			m.toggleSpaceValue(selectedItem, &cfg.ComparisonValuesSpec)
			m.plotItemConfigs[m.editingPlotID] = cfg
		} else {
			m.toggleSpaceValue(selectedItem, &m.plotComparisonValuesSpec)
		}

	case expandedConditionCompareWindows:
		m.toggleSpaceValue(selectedItem, &m.conditionCompareWindows)

	case expandedPlotComparisonWindows:
		// Update plot-specific config if editing a plot field, otherwise global
		if m.editingPlotID != "" {
			plotID := m.editingPlotID
			cfg := m.ensurePlotItemConfig(plotID)
			switch m.editingPlotField {
			case plotItemConfigFieldComparisonSegment:
				// Segment is single-select
				cfg.ComparisonSegment = selectedItem
				m.plotItemConfigs[plotID] = cfg
				m.editingPlotID = ""
				m.editingPlotField = plotItemConfigFieldNone
				m.expandedOption = expandedNone
				m.subCursor = 0
			case plotItemConfigFieldDoseResponseSegment:
				// Dose-response segment is single-select
				cfg.DoseResponseSegment = selectedItem
				m.plotItemConfigs[plotID] = cfg
				m.editingPlotID = ""
				m.editingPlotField = plotItemConfigFieldNone
				m.expandedOption = expandedNone
				m.subCursor = 0
			case plotItemConfigFieldTopomapWindow:
				// TopomapWindowsSpec is multi-select
				m.toggleSpaceValue(selectedItem, &cfg.TopomapWindowsSpec)
				m.plotItemConfigs[plotID] = cfg
			default:
				// Windows is multi-select
				m.toggleSpaceValue(selectedItem, &cfg.ComparisonWindowsSpec)
				m.plotItemConfigs[plotID] = cfg
			}
		} else {
			m.toggleSpaceValue(selectedItem, &m.plotComparisonWindowsSpec)
		}

	case expandedPlotComparisonROIs:
		// Update plot-specific config - ROIs is multi-select
		if m.editingPlotID != "" {
			cfg := m.ensurePlotItemConfig(m.editingPlotID)
			m.toggleSpaceValue(selectedItem, &cfg.ComparisonROIsSpec)
			m.plotItemConfigs[m.editingPlotID] = cfg
		}

	case expandedRunAdjustmentColumn:
		m.runAdjustmentColumn = selectedItem
		m.expandedOption = expandedNone
		m.subCursor = 0

	case expandedCorrelationsTargetColumn:
		if selectedItem == "(none)" {
			m.correlationsTargetColumn = ""
		} else {
			m.correlationsTargetColumn = selectedItem
		}
		m.expandedOption = expandedNone
		m.subCursor = 0

	case expandedTemporalTargetColumn:
		if selectedItem == "(default)" {
			m.temporalTargetColumn = ""
		} else {
			m.temporalTargetColumn = selectedItem
		}
		m.expandedOption = expandedNone
		m.subCursor = 0

	case expandedMLTargetColumn:
		if selectedItem == "(stage default)" {
			m.mlTarget = ""
		} else {
			m.mlTarget = selectedItem
		}
		m.expandedOption = expandedNone
		m.subCursor = 0
	case expandedMLFeatureFamilies:
		if selectedItem == "(config default)" {
			m.mlFeatureFamiliesSpec = ""
		} else {
			m.toggleSpaceValue(selectedItem, &m.mlFeatureFamiliesSpec)
		}
	case expandedMLFeatureBands:
		if selectedItem == "(none)" {
			m.mlFeatureBandsSpec = ""
		} else {
			m.toggleSpaceValue(selectedItem, &m.mlFeatureBandsSpec)
		}
	case expandedMLFeatureSegments:
		if selectedItem == "(none)" {
			m.mlFeatureSegmentsSpec = ""
		} else {
			m.toggleSpaceValue(selectedItem, &m.mlFeatureSegmentsSpec)
		}
	case expandedMLFeatureScopes:
		if selectedItem == "(none)" {
			m.mlFeatureScopesSpec = ""
		} else {
			m.toggleSpaceValue(selectedItem, &m.mlFeatureScopesSpec)
		}
	case expandedMLFeatureStats:
		if selectedItem == "(none)" {
			m.mlFeatureStatsSpec = ""
		} else {
			m.toggleSpaceValue(selectedItem, &m.mlFeatureStatsSpec)
		}

	case expandedItpcConditionColumn:
		m.itpcConditionColumn = selectedItem
		m.itpcConditionValues = "" // Reset values when column changes
		m.expandedOption = expandedNone
		m.subCursor = 0
	case expandedConnConditionColumn:
		m.connConditionColumn = selectedItem
		m.connConditionValues = "" // Reset values when column changes
		m.expandedOption = expandedNone
		m.subCursor = 0
	case expandedFmriCondAColumn:
		m.sourceLocFmriCondAColumn = selectedItem
		m.sourceLocFmriCondAValue = "" // Reset value when column changes
		m.expandedOption = expandedNone
		m.subCursor = 0
	case expandedFmriCondAValue:
		m.sourceLocFmriCondAValue = selectedItem
		m.expandedOption = expandedNone
		m.subCursor = 0
	case expandedFmriCondBColumn:
		m.sourceLocFmriCondBColumn = selectedItem
		m.sourceLocFmriCondBValue = "" // Reset value when column changes
		m.expandedOption = expandedNone
		m.subCursor = 0
	case expandedFmriCondBValue:
		m.sourceLocFmriCondBValue = selectedItem
		m.expandedOption = expandedNone
		m.subCursor = 0
	case expandedSourceLocFmriStimPhases:
		switch selectedItem {
		case "(none)":
			m.sourceLocFmriStimPhasesToModel = ""
			m.expandedOption = expandedNone
			m.subCursor = 0
		case "(all)":
			m.sourceLocFmriStimPhasesToModel = "all"
			m.expandedOption = expandedNone
			m.subCursor = 0
		case "(type manually)":
			m.expandedOption = expandedNone
			m.subCursor = 0
			m.startTextEdit(textFieldSourceLocFmriStimPhasesToModel)
		default:
			m.sourceLocFmriStimPhasesToModel = selectedItem
			m.expandedOption = expandedNone
			m.subCursor = 0
		}
	case expandedSourceLocFmriScopeTrialTypes:
		switch selectedItem {
		case "(none)":
			m.sourceLocFmriConditionScopeTrialTypes = ""
			m.expandedOption = expandedNone
			m.subCursor = 0
		case "(type manually)":
			m.expandedOption = expandedNone
			m.subCursor = 0
			m.startTextEdit(textFieldSourceLocFmriConditionScopeTrialTypes)
		default:
			m.toggleSpaceValue(selectedItem, &m.sourceLocFmriConditionScopeTrialTypes)
		}
	case expandedIAFRois:
		m.toggleColumnValue(selectedItem, &m.iafRoisSpec)
	case expandedFmriAnalysisCondAColumn:
		if selectedItem == "(type manually)" {
			m.expandedOption = expandedNone
			m.subCursor = 0
			m.startTextEdit(textFieldFmriAnalysisCondAColumn)
		} else {
			m.fmriAnalysisCondAColumn = selectedItem
			m.fmriAnalysisCondAValue = ""
			m.expandedOption = expandedNone
			m.subCursor = 0
		}
	case expandedFmriAnalysisCondAValue:
		if selectedItem == "(type manually)" {
			m.expandedOption = expandedNone
			m.subCursor = 0
			m.startTextEdit(textFieldFmriAnalysisCondAValue)
		} else {
			m.fmriAnalysisCondAValue = selectedItem
			m.expandedOption = expandedNone
			m.subCursor = 0
		}
	case expandedFmriAnalysisCondBColumn:
		if selectedItem == "(type manually)" {
			m.expandedOption = expandedNone
			m.subCursor = 0
			m.startTextEdit(textFieldFmriAnalysisCondBColumn)
		} else {
			m.fmriAnalysisCondBColumn = selectedItem
			m.fmriAnalysisCondBValue = ""
			m.expandedOption = expandedNone
			m.subCursor = 0
		}
	case expandedFmriAnalysisCondBValue:
		if selectedItem == "(type manually)" {
			m.expandedOption = expandedNone
			m.subCursor = 0
			m.startTextEdit(textFieldFmriAnalysisCondBValue)
		} else {
			m.fmriAnalysisCondBValue = selectedItem
			m.expandedOption = expandedNone
			m.subCursor = 0
		}
	case expandedFmriAnalysisStimPhases:
		switch selectedItem {
		case "(none)":
			m.fmriAnalysisStimPhasesToModel = ""
			m.expandedOption = expandedNone
			m.subCursor = 0
		case "(all)":
			m.fmriAnalysisStimPhasesToModel = "all"
			m.expandedOption = expandedNone
			m.subCursor = 0
		case "(type manually)":
			m.expandedOption = expandedNone
			m.subCursor = 0
			m.startTextEdit(textFieldFmriAnalysisStimPhasesToModel)
		default:
			m.fmriAnalysisStimPhasesToModel = selectedItem
			m.expandedOption = expandedNone
			m.subCursor = 0
		}
	case expandedFmriAnalysisScopeTrialTypes:
		switch selectedItem {
		case "(none)":
			m.fmriAnalysisScopeTrialTypes = ""
			m.expandedOption = expandedNone
			m.subCursor = 0
		case "(type manually)":
			m.expandedOption = expandedNone
			m.subCursor = 0
			m.startTextEdit(textFieldFmriAnalysisScopeTrialTypes)
		default:
			m.toggleSpaceValue(selectedItem, &m.fmriAnalysisScopeTrialTypes)
		}
	case expandedFmriTrialSigGroupColumn:
		if selectedItem == "(type manually)" {
			m.expandedOption = expandedNone
			m.subCursor = 0
			m.startTextEdit(textFieldFmriTrialSigGroupColumn)
		} else {
			m.fmriTrialSigGroupColumn = selectedItem
			m.fmriTrialSigGroupValuesSpec = ""
			m.expandedOption = expandedNone
			m.subCursor = 0
		}
	case expandedFmriTrialSigGroupValues:
		if selectedItem == "(type manually)" {
			m.expandedOption = expandedNone
			m.subCursor = 0
			m.startTextEdit(textFieldFmriTrialSigGroupValues)
		} else {
			m.toggleSpaceValue(selectedItem, &m.fmriTrialSigGroupValuesSpec)
		}
	case expandedFmriTrialSigStimPhases:
		switch selectedItem {
		case "(none)":
			m.fmriTrialSigScopeStimPhases = ""
			m.expandedOption = expandedNone
			m.subCursor = 0
		case "(all)":
			m.fmriTrialSigScopeStimPhases = "all"
			m.expandedOption = expandedNone
			m.subCursor = 0
		case "(type manually)":
			m.expandedOption = expandedNone
			m.subCursor = 0
			m.startTextEdit(textFieldFmriTrialSigScopeStimPhases)
		default:
			m.fmriTrialSigScopeStimPhases = selectedItem
			m.expandedOption = expandedNone
			m.subCursor = 0
		}
	case expandedFmriTrialSigScopeTrialTypes:
		switch selectedItem {
		case "(none)":
			m.fmriTrialSigScopeTrialTypes = ""
			m.expandedOption = expandedNone
			m.subCursor = 0
		case "(type manually)":
			m.expandedOption = expandedNone
			m.subCursor = 0
			m.startTextEdit(textFieldFmriTrialSigScopeTrialTypes)
		default:
			m.toggleSpaceValue(selectedItem, &m.fmriTrialSigScopeTrialTypes)
		}

	case expandedItpcConditionValues:
		m.toggleColumnValue(selectedItem, &m.itpcConditionValues)
	case expandedConnConditionValues:
		m.toggleColumnValue(selectedItem, &m.connConditionValues)

	case expandedBehaviorScatterFeatures:
		if m.editingPlotID != "" {
			cfg := m.ensurePlotItemConfig(m.editingPlotID)
			m.toggleSpaceValue(selectedItem, &cfg.BehaviorScatterFeaturesSpec)
			m.plotItemConfigs[m.editingPlotID] = cfg
		}

	case expandedBehaviorScatterColumns:
		if m.editingPlotID != "" {
			cfg := m.ensurePlotItemConfig(m.editingPlotID)
			m.toggleSpaceValue(selectedItem, &cfg.BehaviorScatterColumnsSpec)
			m.plotItemConfigs[m.editingPlotID] = cfg
		}

	case expandedBehaviorScatterAggregation:
		if m.editingPlotID != "" {
			cfg := m.ensurePlotItemConfig(m.editingPlotID)
			m.toggleSpaceValue(selectedItem, &cfg.BehaviorScatterAggregationModesSpec)
			m.plotItemConfigs[m.editingPlotID] = cfg
		}

	case expandedBehaviorScatterSegment:
		if m.editingPlotID != "" {
			cfg := m.ensurePlotItemConfig(m.editingPlotID)
			// Segment is single-select
			cfg.BehaviorScatterSegmentSpec = selectedItem
			m.plotItemConfigs[m.editingPlotID] = cfg
			m.editingPlotID = ""
			m.editingPlotField = plotItemConfigFieldNone
			m.expandedOption = expandedNone
			m.subCursor = 0
		}
	case expandedTemporalTopomapsFeatureDir:
		if m.editingPlotID != "" {
			cfg := m.ensurePlotItemConfig(m.editingPlotID)
			cfg.BehaviorTemporalStatsFeatureFolder = selectedItem
			m.plotItemConfigs[m.editingPlotID] = cfg
			m.editingPlotID = ""
			m.editingPlotField = plotItemConfigFieldNone
			m.expandedOption = expandedNone
			m.subCursor = 0
		}
	case expandedDoseResponseBands:
		if m.editingPlotID != "" {
			cfg := m.ensurePlotItemConfig(m.editingPlotID)
			m.toggleSpaceValue(selectedItem, &cfg.DoseResponseBandsSpec)
			m.plotItemConfigs[m.editingPlotID] = cfg
		}
	case expandedDoseResponseROIs:
		if m.editingPlotID != "" {
			cfg := m.ensurePlotItemConfig(m.editingPlotID)
			m.toggleSpaceValue(selectedItem, &cfg.DoseResponseROIsSpec)
			m.plotItemConfigs[m.editingPlotID] = cfg
		}
	case expandedDoseResponseScopes:
		if m.editingPlotID != "" {
			cfg := m.ensurePlotItemConfig(m.editingPlotID)
			m.toggleSpaceValue(selectedItem, &cfg.DoseResponseScopesSpec)
			m.plotItemConfigs[m.editingPlotID] = cfg
		}
	case expandedDoseResponseStat:
		if m.editingPlotID != "" {
			cfg := m.ensurePlotItemConfig(m.editingPlotID)
			cfg.DoseResponseStat = selectedItem
			m.plotItemConfigs[m.editingPlotID] = cfg
			m.editingPlotID = ""
			m.editingPlotField = plotItemConfigFieldNone
			m.expandedOption = expandedNone
			m.subCursor = 0
		}
	case expandedPainResidualCrossfitGroupColumn:
		switch selectedItem {
		case "(default: run column)":
			m.painResidualCrossfitGroupColumn = ""
			m.expandedOption = expandedNone
			m.subCursor = 0
		case "(type manually)":
			m.expandedOption = expandedNone
			m.subCursor = 0
			m.startTextEdit(textFieldPainResidualCrossfitGroupColumn)
		default:
			m.painResidualCrossfitGroupColumn = selectedItem
			m.expandedOption = expandedNone
			m.subCursor = 0
		}
	case expandedStabilityGroupColumn:
		switch selectedItem {
		case "run":
			m.stabilityGroupColumn = 1
		case "block":
			m.stabilityGroupColumn = 2
		default:
			m.stabilityGroupColumn = 0
		}
		m.expandedOption = expandedNone
		m.subCursor = 0
	case expandedGroupLevelTarget:
		switch selectedItem {
		case "(default)":
			m.groupLevelTarget = ""
		case "(type manually)":
			m.expandedOption = expandedNone
			m.subCursor = 0
			m.startTextEdit(textFieldGroupLevelTarget)
			m.useDefaultAdvanced = false
			return
		default:
			m.groupLevelTarget = selectedItem
		}
		m.expandedOption = expandedNone
		m.subCursor = 0
	}

	m.useDefaultAdvanced = false
}

// shouldRenderExpandedListAfterOption checks if we should render an expanded list after the given option
func (m Model) shouldRenderExpandedListAfterOption(opt optionType) bool {
	switch m.expandedOption {
	case expandedConditionCompareColumn:
		return opt == optConditionCompareColumn
	case expandedConditionCompareValues:
		return opt == optConditionCompareValues
	case expandedConditionCompareWindows:
		return opt == optConditionCompareWindows
	case expandedTemporalConditionColumn:
		return opt == optTemporalConditionColumn
	case expandedTemporalConditionValues:
		return opt == optTemporalConditionValues
	case expandedClusterConditionColumn:
		return opt == optClusterConditionColumn
	case expandedClusterConditionValues:
		return opt == optClusterConditionValues
	case expandedPlotComparisonColumn:
		return opt == optPlotComparisonColumn
	case expandedPlotComparisonValues:
		return opt == optPlotComparisonValues
	case expandedPlotComparisonWindows:
		return opt == optPlotComparisonWindows
	case expandedRunAdjustmentColumn:
		return opt == optRunAdjustmentColumn
	case expandedCorrelationsTargetColumn:
		return opt == optCorrelationsTargetColumn
	case expandedTemporalTargetColumn:
		return opt == optTemporalTargetColumn
	case expandedMLTargetColumn:
		return opt == optMLTarget
	case expandedMLFeatureFamilies:
		return opt == optMLFeatureFamilies
	case expandedMLFeatureBands:
		return opt == optMLFeatureBands
	case expandedMLFeatureSegments:
		return opt == optMLFeatureSegments
	case expandedMLFeatureScopes:
		return opt == optMLFeatureScopes
	case expandedMLFeatureStats:
		return opt == optMLFeatureStats
	case expandedItpcConditionColumn:
		return opt == optItpcConditionColumn
	case expandedConnConditionColumn:
		return opt == optConnConditionColumn
	case expandedFmriCondAColumn:
		return opt == optSourceLocFmriCondAColumn
	case expandedFmriCondAValue:
		return opt == optSourceLocFmriCondAValue
	case expandedFmriCondBColumn:
		return opt == optSourceLocFmriCondBColumn
	case expandedFmriCondBValue:
		return opt == optSourceLocFmriCondBValue
	case expandedFmriAnalysisCondAColumn:
		return opt == optFmriAnalysisCondAColumn
	case expandedFmriAnalysisCondAValue:
		return opt == optFmriAnalysisCondAValue
	case expandedFmriAnalysisCondBColumn:
		return opt == optFmriAnalysisCondBColumn
	case expandedFmriAnalysisCondBValue:
		return opt == optFmriAnalysisCondBValue
	case expandedFmriAnalysisStimPhases:
		return opt == optFmriAnalysisStimPhasesToModel
	case expandedFmriAnalysisScopeTrialTypes:
		return opt == optFmriAnalysisScopeTrialTypes
	case expandedFmriTrialSigGroupColumn:
		return opt == optFmriTrialSigGroupColumn
	case expandedFmriTrialSigGroupValues:
		return opt == optFmriTrialSigGroupValues
	case expandedFmriTrialSigStimPhases:
		return opt == optFmriTrialSigScopeStimPhases
	case expandedFmriTrialSigScopeTrialTypes:
		return opt == optFmriTrialSigScopeTrialTypes
	case expandedItpcConditionValues:
		return opt == optItpcConditionValues
	case expandedConnConditionValues:
		return opt == optConnConditionValues
	case expandedSourceLocFmriStimPhases:
		return opt == optSourceLocFmriStimPhasesToModel
	case expandedSourceLocFmriScopeTrialTypes:
		return opt == optSourceLocFmriConditionScopeTrialTypes
	case expandedIAFRois:
		return opt == optIAFRois
	case expandedPainResidualCrossfitGroupColumn:
		return opt == optPainResidualCrossfitGroupColumn
	case expandedStabilityGroupColumn:
		return opt == optStabilityGroupColumn
	case expandedGroupLevelTarget:
		return opt == optGroupLevelTarget
	}
	return false
}

// isExpandedItemSelected checks if an item at the given index is selected in the expanded list
func (m Model) isExpandedItemSelected(_ int, item string) bool {
	switch m.expandedOption {
	case expandedConditionCompareColumn:
		return m.conditionCompareColumn == item
	case expandedTemporalConditionColumn:
		return m.temporalConditionColumn == item
	case expandedClusterConditionColumn:
		return m.clusterConditionColumn == item
	case expandedPlotComparisonColumn:
		// Check plot-specific config first
		if m.editingPlotID != "" {
			if cfg, ok := m.plotItemConfigs[m.editingPlotID]; ok && cfg.ComparisonColumn != "" {
				return cfg.ComparisonColumn == item
			}
		}
		return m.plotComparisonColumn == item
	case expandedRunAdjustmentColumn:
		return m.runAdjustmentColumn == item
	case expandedCorrelationsTargetColumn:
		if item == "(none)" {
			return m.correlationsTargetColumn == ""
		}
		return m.correlationsTargetColumn == item
	case expandedTemporalTargetColumn:
		if item == "(default)" {
			return m.temporalTargetColumn == ""
		}
		return m.temporalTargetColumn == item
	case expandedMLTargetColumn:
		if item == "(stage default)" {
			return m.mlTarget == ""
		}
		return m.mlTarget == item
	case expandedMLFeatureFamilies:
		if item == "(config default)" {
			return strings.TrimSpace(m.mlFeatureFamiliesSpec) == ""
		}
		return m.isColumnValueSelected(item)
	case expandedMLFeatureBands:
		if item == "(none)" {
			return strings.TrimSpace(m.mlFeatureBandsSpec) == ""
		}
		return m.isColumnValueSelected(item)
	case expandedMLFeatureSegments:
		if item == "(none)" {
			return strings.TrimSpace(m.mlFeatureSegmentsSpec) == ""
		}
		return m.isColumnValueSelected(item)
	case expandedMLFeatureScopes:
		if item == "(none)" {
			return strings.TrimSpace(m.mlFeatureScopesSpec) == ""
		}
		return m.isColumnValueSelected(item)
	case expandedMLFeatureStats:
		if item == "(none)" {
			return strings.TrimSpace(m.mlFeatureStatsSpec) == ""
		}
		return m.isColumnValueSelected(item)
	case expandedItpcConditionColumn:
		return m.itpcConditionColumn == item
	case expandedConnConditionColumn:
		return m.connConditionColumn == item
	case expandedFmriCondAColumn:
		return m.sourceLocFmriCondAColumn == item
	case expandedFmriCondAValue:
		return m.sourceLocFmriCondAValue == item
	case expandedFmriCondBColumn:
		return m.sourceLocFmriCondBColumn == item
	case expandedFmriCondBValue:
		return m.sourceLocFmriCondBValue == item
	case expandedSourceLocFmriStimPhases:
		if item == "(none)" {
			return strings.TrimSpace(m.sourceLocFmriStimPhasesToModel) == ""
		}
		if item == "(all)" {
			return strings.TrimSpace(m.sourceLocFmriStimPhasesToModel) == "all"
		}
		for _, p := range splitSpaceList(m.sourceLocFmriStimPhasesToModel) {
			if p == item {
				return true
			}
		}
		return false
	case expandedSourceLocFmriScopeTrialTypes:
		if item == "(none)" {
			return strings.TrimSpace(m.sourceLocFmriConditionScopeTrialTypes) == ""
		}
		for _, p := range splitSpaceList(m.sourceLocFmriConditionScopeTrialTypes) {
			if p == item {
				return true
			}
		}
		return false
	case expandedFmriAnalysisCondAColumn:
		return m.fmriAnalysisCondAColumn == item
	case expandedFmriAnalysisCondAValue:
		return m.fmriAnalysisCondAValue == item
	case expandedFmriAnalysisCondBColumn:
		return m.fmriAnalysisCondBColumn == item
	case expandedFmriAnalysisCondBValue:
		return m.fmriAnalysisCondBValue == item
	case expandedFmriAnalysisStimPhases:
		if item == "(none)" {
			return strings.TrimSpace(m.fmriAnalysisStimPhasesToModel) == ""
		}
		if item == "(all)" {
			return strings.TrimSpace(m.fmriAnalysisStimPhasesToModel) == "all"
		}
		for _, p := range splitSpaceList(m.fmriAnalysisStimPhasesToModel) {
			if p == item {
				return true
			}
		}
		return false
	case expandedFmriAnalysisScopeTrialTypes:
		if item == "(none)" {
			return strings.TrimSpace(m.fmriAnalysisScopeTrialTypes) == ""
		}
		for _, p := range splitSpaceList(m.fmriAnalysisScopeTrialTypes) {
			if p == item {
				return true
			}
		}
		return false
	case expandedFmriTrialSigGroupColumn:
		return m.fmriTrialSigGroupColumn == item
	case expandedFmriTrialSigStimPhases:
		if item == "(none)" {
			return strings.TrimSpace(m.fmriTrialSigScopeStimPhases) == ""
		}
		if item == "(all)" {
			return strings.TrimSpace(m.fmriTrialSigScopeStimPhases) == "all"
		}
		for _, p := range splitSpaceList(m.fmriTrialSigScopeStimPhases) {
			if p == item {
				return true
			}
		}
		return false
	case expandedFmriTrialSigScopeTrialTypes:
		if item == "(none)" {
			return strings.TrimSpace(m.fmriTrialSigScopeTrialTypes) == ""
		}
		for _, p := range splitSpaceList(m.fmriTrialSigScopeTrialTypes) {
			if p == item {
				return true
			}
		}
		return false
	case expandedIAFRois:
		for _, roi := range splitCSVList(m.iafRoisSpec) {
			if roi == item {
				return true
			}
		}
		return false
	case expandedConditionCompareValues, expandedTemporalConditionValues, expandedClusterConditionValues, expandedPlotComparisonValues,
		expandedConditionCompareWindows, expandedPlotComparisonWindows, expandedItpcConditionValues, expandedConnConditionValues,
		expandedFmriTrialSigGroupValues, expandedDoseResponseBands, expandedDoseResponseROIs, expandedDoseResponseScopes, expandedDoseResponseStat:
		return m.isColumnValueSelected(item)
	case expandedBehaviorScatterSegment:
		// Check plot-specific config
		if m.editingPlotID != "" {
			if cfg, ok := m.plotItemConfigs[m.editingPlotID]; ok {
				return cfg.BehaviorScatterSegmentSpec == item
			}
		}
		return false
	case expandedPainResidualCrossfitGroupColumn:
		if item == "(default: run column)" {
			return strings.TrimSpace(m.painResidualCrossfitGroupColumn) == ""
		}
		if item == "(type manually)" {
			return false
		}
		return m.painResidualCrossfitGroupColumn == item
	case expandedStabilityGroupColumn:
		switch item {
		case "(auto)":
			return m.stabilityGroupColumn == 0
		case "run":
			return m.stabilityGroupColumn == 1
		case "block":
			return m.stabilityGroupColumn == 2
		}
		return false
	case expandedGroupLevelTarget:
		if item == "(default)" {
			return strings.TrimSpace(m.groupLevelTarget) == ""
		}
		if item == "(type manually)" {
			return false
		}
		return strings.EqualFold(strings.TrimSpace(m.groupLevelTarget), strings.TrimSpace(item))
	}
	return false
}

// toggleColumnValue toggles a value in a comma-separated list
func (m *Model) toggleColumnValue(value string, target *string) {
	if *target == "" {
		*target = value
		return
	}

	// Parse existing values
	existing := strings.Split(*target, ",")
	var newValues []string
	found := false

	for _, v := range existing {
		v = strings.TrimSpace(v)
		if v == value {
			found = true
		} else if v != "" {
			newValues = append(newValues, v)
		}
	}

	if !found {
		newValues = append(newValues, value)
	}

	*target = strings.Join(newValues, ",")
}

// toggleSpaceValue toggles a value in a space-separated list
func (m *Model) toggleSpaceValue(value string, target *string) {
	if *target == "" {
		*target = value
		return
	}

	existing := strings.Fields(*target)
	var newValues []string
	found := false

	for _, v := range existing {
		if v == value {
			found = true
		} else if v != "" {
			newValues = append(newValues, v)
		}
	}

	if !found {
		newValues = append(newValues, value)
	}

	*target = strings.Join(newValues, " ")
}

func (m Model) getDoseResponseCategoriesForEditingPlot() []string {
	if m.editingPlotID != "" {
		if cfg, ok := m.plotItemConfigs[m.editingPlotID]; ok {
			cats := splitSpaceList(cfg.DoseResponseResponseColumn)
			if len(cats) > 0 {
				return cats
			}
		}
	}
	return m.GetTrialTableFeatureCategories()
}

///////////////////////////////////////////////////////////////////
