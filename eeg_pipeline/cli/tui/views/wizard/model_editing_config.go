package wizard

import (
	"fmt"
	"sort"
	"strings"

	"github.com/eeg-pipeline/tui/messages"
)

// Expanded-list selection and config/text-edit mutation helpers.

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
	case expandedRunAdjustmentColumn:
		return len(m.GetAvailableColumns())
	case expandedCorrelationsTargetColumn:
		return len(m.GetAvailableColumns()) + 1 // +1 for "(none)" option
	case expandedTemporalTargetColumn:
		return len(m.GetAvailableColumns()) + 1 // +1 for "(default)" option
	case expandedMLTargetColumn:
		return len(m.GetAvailableColumns()) + 1 // +1 for "(stage default)" option
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
	case expandedRunAdjustmentColumn:
		return m.GetAvailableColumns()
	case expandedCorrelationsTargetColumn:
		return append([]string{"(none)"}, m.GetAvailableColumns()...)
	case expandedTemporalTargetColumn:
		return append([]string{"(default)"}, m.GetAvailableColumns()...)
	case expandedMLTargetColumn:
		return append([]string{"(stage default)"}, m.GetAvailableColumns()...)
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
	}
	return nil
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
				if m.editingPlotField == plotItemConfigFieldComparisonSegment {
					selectedValues = cfg.ComparisonSegment
				} else if m.editingPlotField == plotItemConfigFieldDoseResponseSegment {
					selectedValues = cfg.DoseResponseSegment
				} else if m.editingPlotField == plotItemConfigFieldTopomapWindow {
					selectedValues = cfg.TopomapWindowsSpec
				} else {
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
			if m.editingPlotField == plotItemConfigFieldComparisonSegment {
				// Segment is single-select
				cfg.ComparisonSegment = selectedItem
				m.plotItemConfigs[plotID] = cfg
				m.editingPlotID = ""
				m.editingPlotField = plotItemConfigFieldNone
				m.expandedOption = expandedNone
				m.subCursor = 0
			} else if m.editingPlotField == plotItemConfigFieldDoseResponseSegment {
				// Dose-response segment is single-select
				cfg.DoseResponseSegment = selectedItem
				m.plotItemConfigs[plotID] = cfg
				m.editingPlotID = ""
				m.editingPlotField = plotItemConfigFieldNone
				m.expandedOption = expandedNone
				m.subCursor = 0
			} else if m.editingPlotField == plotItemConfigFieldTopomapWindow {
				// TopomapWindowsSpec is multi-select
				m.toggleSpaceValue(selectedItem, &cfg.TopomapWindowsSpec)
				m.plotItemConfigs[plotID] = cfg
			} else {
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
	case expandedPainResidualCrossfitGroupColumn:
		return opt == optPainResidualCrossfitGroupColumn
	case expandedStabilityGroupColumn:
		return opt == optStabilityGroupColumn
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
	case expandedConditionCompareValues, expandedTemporalConditionValues, expandedClusterConditionValues, expandedPlotComparisonValues,
		expandedConditionCompareWindows, expandedPlotComparisonWindows, expandedItpcConditionValues, expandedConnConditionValues,
		expandedFmriTrialSigGroupValues:
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

func (m *Model) SetConfigSummary(summary messages.ConfigSummary) {
	if m.task == "" && summary.Task != "" {
		m.task = summary.Task
	}
	if m.bidsRoot == "" && summary.BidsRoot != "" {
		m.bidsRoot = summary.BidsRoot
	}
	if m.bidsFmriRoot == "" && summary.BidsFmriRoot != "" {
		m.bidsFmriRoot = summary.BidsFmriRoot
	}
	if m.derivRoot == "" && summary.DerivRoot != "" {
		m.derivRoot = summary.DerivRoot
	}
	if m.sourceRoot == "" && summary.SourceRoot != "" {
		m.sourceRoot = summary.SourceRoot
	}
	if summary.PreprocessingNJobs > 0 {
		m.prepNJobs = summary.PreprocessingNJobs
	}
}

func (m *Model) SetRepoRoot(repoRoot string) {
	m.repoRoot = repoRoot
}

func (m *Model) startTextEdit(field textField) {
	m.editingTextField = field
	m.textBuffer = m.getTextFieldValue(field)
	m.editingPlotID = ""
	m.editingPlotField = plotItemConfigFieldNone
	m.editingText = true
}

func (m *Model) commitTextInput() {
	if m.editingPlotID != "" && m.editingPlotField != plotItemConfigFieldNone {
		m.setPlotItemTextFieldValue(m.editingPlotID, m.editingPlotField, m.textBuffer)
		return
	}
	m.setTextFieldValue(m.editingTextField, m.textBuffer)
}

func (m *Model) startPlotTextEdit(plotID string, field plotItemConfigField) {
	m.editingTextField = textFieldNone
	m.editingPlotID = plotID
	m.editingPlotField = field
	m.textBuffer = m.getPlotItemTextFieldValue(plotID, field)
	m.editingText = true
}

func (m Model) getTextFieldValue(field textField) string {
	switch field {
	case textFieldTask:
		return m.task
	case textFieldBidsRoot:
		return m.bidsRoot
	case textFieldBidsFmriRoot:
		return m.bidsFmriRoot
	case textFieldDerivRoot:
		return m.derivRoot
	case textFieldSourceRoot:
		return m.sourceRoot
	case textFieldFmriFmriprepImage:
		return m.fmriFmriprepImage
	case textFieldFmriFmriprepOutputDir:
		return m.fmriFmriprepOutputDir
	case textFieldFmriFmriprepWorkDir:
		return m.fmriFmriprepWorkDir
	case textFieldFmriFreesurferLicenseFile:
		return m.fmriFreesurferLicenseFile
	case textFieldFmriFreesurferSubjectsDir:
		return m.fmriFreesurferSubjectsDir
	case textFieldFmriOutputSpaces:
		return m.fmriOutputSpacesSpec
	case textFieldFmriIgnore:
		return m.fmriIgnoreSpec
	case textFieldFmriBidsFilterFile:
		return m.fmriBidsFilterFile
	case textFieldFmriExtraArgs:
		return m.fmriExtraArgs
	case textFieldFmriSkullStripTemplate:
		return m.fmriSkullStripTemplate
	case textFieldFmriTaskId:
		return m.fmriTaskId
	case textFieldFmriAnalysisFmriprepSpace:
		return m.fmriAnalysisFmriprepSpace
	case textFieldFmriAnalysisRuns:
		return m.fmriAnalysisRunsSpec
	case textFieldFmriAnalysisCondAColumn:
		return m.fmriAnalysisCondAColumn
	case textFieldFmriAnalysisCondAValue:
		return m.fmriAnalysisCondAValue
	case textFieldFmriAnalysisCondBColumn:
		return m.fmriAnalysisCondBColumn
	case textFieldFmriAnalysisCondBValue:
		return m.fmriAnalysisCondBValue
	case textFieldFmriAnalysisContrastName:
		return m.fmriAnalysisContrastName
	case textFieldFmriAnalysisFormula:
		return m.fmriAnalysisFormula
	case textFieldFmriAnalysisEventsToModel:
		return m.fmriAnalysisEventsToModel
	case textFieldFmriAnalysisScopeTrialTypes:
		return m.fmriAnalysisScopeTrialTypes
	case textFieldFmriAnalysisStimPhasesToModel:
		return m.fmriAnalysisStimPhasesToModel
	case textFieldFmriAnalysisOutputDir:
		return m.fmriAnalysisOutputDir
	case textFieldFmriAnalysisFreesurferDir:
		return m.fmriAnalysisFreesurferDir
	case textFieldFmriAnalysisSignatureDir:
		return m.fmriAnalysisSignatureDir
	case textFieldFmriTrialSigGroupColumn:
		return m.fmriTrialSigGroupColumn
	case textFieldFmriTrialSigGroupValues:
		return m.fmriTrialSigGroupValuesSpec
	case textFieldFmriTrialSigScopeTrialTypes:
		return m.fmriTrialSigScopeTrialTypes
	case textFieldFmriTrialSigScopeStimPhases:
		return m.fmriTrialSigScopeStimPhases
	case textFieldRawMontage:
		return m.rawMontage
	case textFieldPrepMontage:
		return m.prepMontage
	case textFieldPrepChTypes:
		return m.prepChTypes
	case textFieldPrepEegReference:
		return m.prepEegReference
	case textFieldPrepEogChannels:
		return m.prepEogChannels
	case textFieldPrepConditions:
		return m.prepConditions
	case textFieldPrepFileExtension:
		return m.prepFileExtension
	case textFieldPrepRenameAnotDict:
		return m.prepRenameAnotDict
	case textFieldPrepCustomBadDict:
		return m.prepCustomBadDict
	case textFieldRawEventPrefixes:
		return m.rawEventPrefixes
	case textFieldMergeEventPrefixes:
		return m.mergeEventPrefixes
	case textFieldMergeEventTypes:
		return m.mergeEventTypes
	case textFieldMergeQCColumns:
		return m.mergeQCColumns
	case textFieldFmriRawSession:
		return m.fmriRawSession
	case textFieldFmriRawRestTask:
		return m.fmriRawRestTask
	case textFieldFmriRawDcm2niixPath:
		return m.fmriRawDcm2niixPath
	case textFieldFmriRawDcm2niixArgs:
		return m.fmriRawDcm2niixArgs
	case textFieldConditionCompareColumn:
		return m.conditionCompareColumn
	case textFieldConditionCompareWindows:
		return m.conditionCompareWindows
	case textFieldConditionCompareValues:
		return m.conditionCompareValues
	case textFieldTemporalConditionColumn:
		return m.temporalConditionColumn
	case textFieldTemporalConditionValues:
		return m.temporalConditionValues
	case textFieldTemporalTargetColumn:
		return m.temporalTargetColumn
	case textFieldTfHeatmapFreqs:
		return m.tfHeatmapFreqsSpec
	case textFieldRunAdjustmentColumn:
		return m.runAdjustmentColumn
	case textFieldPainResidualCrossfitGroupColumn:
		return m.painResidualCrossfitGroupColumn
	case textFieldPainResidualSplineDfCandidates:
		return m.painResidualSplineDfCandidates
	case textFieldPainResidualModelComparePolyDegrees:
		return m.painResidualModelComparePolyDegrees
	case textFieldClusterConditionColumn:
		return m.clusterConditionColumn
	case textFieldClusterConditionValues:
		return m.clusterConditionValues
	case textFieldCorrelationsTargetColumn:
		return m.correlationsTargetColumn
	case textFieldCorrelationsTypes:
		return m.correlationsTypesSpec
	case textFieldItpcConditionColumn:
		return m.itpcConditionColumn
	case textFieldItpcConditionValues:
		return m.itpcConditionValues
	case textFieldConnConditionColumn:
		return m.connConditionColumn
	case textFieldConnConditionValues:
		return m.connConditionValues
	case textFieldPACPairs:
		return m.pacPairsSpec
	case textFieldBurstBands:
		return m.burstBandsSpec
	case textFieldERDSBands:
		return m.erdsBandsSpec
	case textFieldSpectralRatioPairs:
		return m.spectralRatioPairsSpec
	case textFieldAsymmetryChannelPairs:
		return m.asymmetryChannelPairsSpec
	case textFieldAsymmetryActivationBands:
		return m.asymmetryActivationBandsSpec
	case textFieldIAFRois:
		return m.iafRoisSpec
	case textFieldERPComponents:
		return m.erpComponentsSpec
	case textFieldSourceLocSubject:
		return m.sourceLocSubject
	case textFieldSourceLocTrans:
		return m.sourceLocTrans
	case textFieldSourceLocBem:
		return m.sourceLocBem
	case textFieldSourceLocFmriStatsMap:
		return m.sourceLocFmriStatsMap
	case textFieldSourceLocFmriCondAColumn:
		return m.sourceLocFmriCondAColumn
	case textFieldSourceLocFmriCondAValue:
		return m.sourceLocFmriCondAValue
	case textFieldSourceLocFmriCondBColumn:
		return m.sourceLocFmriCondBColumn
	case textFieldSourceLocFmriCondBValue:
		return m.sourceLocFmriCondBValue
	case textFieldSourceLocFmriContrastFormula:
		return m.sourceLocFmriContrastFormula
	case textFieldSourceLocFmriContrastName:
		return m.sourceLocFmriContrastName
	case textFieldSourceLocFmriRunsToInclude:
		return m.sourceLocFmriRunsToInclude
	case textFieldSourceLocFmriConditionScopeTrialTypes:
		return m.sourceLocFmriConditionScopeTrialTypes
	case textFieldSourceLocFmriStimPhasesToModel:
		return m.sourceLocFmriStimPhasesToModel
	case textFieldSourceLocFmriWindowAName:
		return m.sourceLocFmriWindowAName
	case textFieldSourceLocFmriWindowBName:
		return m.sourceLocFmriWindowBName
	case textFieldPlotBboxInches:
		return m.plotBboxInches
	case textFieldPlotFontFamily:
		return m.plotFontFamily
	case textFieldPlotFontWeight:
		return m.plotFontWeight
	case textFieldPlotLayoutTightRect:
		return m.plotLayoutTightRectSpec
	case textFieldPlotLayoutTightRectMicrostate:
		return m.plotLayoutTightRectMicrostateSpec
	case textFieldPlotGridSpecWidthRatios:
		return m.plotGridSpecWidthRatiosSpec
	case textFieldPlotGridSpecHeightRatios:
		return m.plotGridSpecHeightRatiosSpec
	case textFieldPlotFigureSizeStandard:
		return m.plotFigureSizeStandardSpec
	case textFieldPlotFigureSizeMedium:
		return m.plotFigureSizeMediumSpec
	case textFieldPlotFigureSizeSmall:
		return m.plotFigureSizeSmallSpec
	case textFieldPlotFigureSizeSquare:
		return m.plotFigureSizeSquareSpec
	case textFieldPlotFigureSizeWide:
		return m.plotFigureSizeWideSpec
	case textFieldPlotFigureSizeTFR:
		return m.plotFigureSizeTFRSpec
	case textFieldPlotFigureSizeTopomap:
		return m.plotFigureSizeTopomapSpec
	case textFieldPlotColorPain:
		return m.plotColorPain
	case textFieldPlotColorNonpain:
		return m.plotColorNonpain
	case textFieldPlotColorSignificant:
		return m.plotColorSignificant
	case textFieldPlotColorNonsignificant:
		return m.plotColorNonsignificant
	case textFieldPlotColorGray:
		return m.plotColorGray
	case textFieldPlotColorLightGray:
		return m.plotColorLightGray
	case textFieldPlotColorBlack:
		return m.plotColorBlack
	case textFieldPlotColorBlue:
		return m.plotColorBlue
	case textFieldPlotColorRed:
		return m.plotColorRed
	case textFieldPlotColorNetworkNode:
		return m.plotColorNetworkNode
	case textFieldPlotScatterEdgecolor:
		return m.plotScatterEdgeColor
	case textFieldPlotHistEdgecolor:
		return m.plotHistEdgeColor
	case textFieldPlotKdeColor:
		return m.plotKdeColor
	case textFieldPlotTopomapColormap:
		return m.plotTopomapColormap
	case textFieldPlotTopomapSigMaskMarker:
		return m.plotTopomapSigMaskMarker
	case textFieldPlotTopomapSigMaskMarkerFaceColor:
		return m.plotTopomapSigMaskMarkerFaceColor
	case textFieldPlotTopomapSigMaskMarkerEdgeColor:
		return m.plotTopomapSigMaskMarkerEdgeColor
	case textFieldPlotTfrDefaultBaselineWindow:
		return m.plotTfrDefaultBaselineWindowSpec
	case textFieldPlotPacCmap:
		return m.plotPacCmap
	case textFieldPlotPacPairs:
		return m.plotPacPairsSpec
	case textFieldPlotConnectivityMeasures:
		return strings.Join(m.selectedConnectivityMeasures(), " ")
	case textFieldPlotSpectralMetrics:
		return m.plotSpectralMetricsSpec
	case textFieldPlotBurstsMetrics:
		return m.plotBurstsMetricsSpec
	case textFieldPlotTemporalTimeBins:
		return m.plotTemporalTimeBinsSpec
	case textFieldPlotTemporalTimeLabels:
		return m.plotTemporalTimeLabelsSpec
	case textFieldPlotAsymmetryStat:
		return m.plotAsymmetryStatSpec
	case textFieldPlotComparisonWindows:
		return m.plotComparisonWindowsSpec
	case textFieldPlotComparisonSegment:
		return m.plotComparisonSegment
	case textFieldPlotComparisonColumn:
		return m.plotComparisonColumn
	case textFieldPlotComparisonValues:
		return m.plotComparisonValuesSpec
	case textFieldPlotComparisonLabels:
		return m.plotComparisonLabelsSpec
	case textFieldPlotComparisonROIs:
		return m.plotComparisonROIsSpec
	// Machine Learning advanced config text fields
	case textFieldMLTarget:
		return m.mlTarget
	case textFieldMLFmriSigContrastName:
		return m.mlFmriSigContrastName
	case textFieldMLFeatureFamilies:
		return m.mlFeatureFamiliesSpec
	case textFieldMLFeatureBands:
		return m.mlFeatureBandsSpec
	case textFieldMLFeatureSegments:
		return m.mlFeatureSegmentsSpec
	case textFieldMLFeatureScopes:
		return m.mlFeatureScopesSpec
	case textFieldMLFeatureStats:
		return m.mlFeatureStatsSpec
	case textFieldMLCovariates:
		return m.mlCovariatesSpec
	case textFieldMLBaselinePredictors:
		return m.mlBaselinePredictorsSpec
	// Machine Learning hyperparameter text fields
	case textFieldElasticNetAlphaGrid:
		return m.elasticNetAlphaGrid
	case textFieldElasticNetL1RatioGrid:
		return m.elasticNetL1RatioGrid
	case textFieldRidgeAlphaGrid:
		return m.ridgeAlphaGrid
	case textFieldRfMaxDepthGrid:
		return m.rfMaxDepthGrid
	case textFieldVarianceThresholdGrid:
		return m.varianceThresholdGrid
	// Preprocessing text fields
	case textFieldIcaLabelsToKeep:
		return m.icaLabelsToKeep
	default:
		return ""
	}
}

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

func (m *Model) ApplyConfigKeys(values map[string]interface{}) {
	asString := func(v interface{}) (string, bool) {
		s, ok := v.(string)
		if !ok {
			return "", false
		}
		return strings.TrimSpace(s), true
	}
	asBool := func(v interface{}) (bool, bool) {
		b, ok := v.(bool)
		return b, ok
	}
	asInt := func(v interface{}) (int, bool) {
		switch n := v.(type) {
		case float64:
			return int(n), true
		case int:
			return n, true
		default:
			return 0, false
		}
	}
	asStringList := func(v interface{}) ([]string, bool) {
		raw, ok := v.([]interface{})
		if !ok {
			return nil, false
		}
		var out []string
		for _, item := range raw {
			s, ok := item.(string)
			if !ok {
				continue
			}
			s = strings.TrimSpace(s)
			if s != "" {
				out = append(out, s)
			}
		}
		return out, true
	}
	asListSpec := func(v interface{}) (string, bool) {
		switch vals := v.(type) {
		case []interface{}:
			out := make([]string, 0, len(vals))
			for _, item := range vals {
				s := strings.TrimSpace(fmt.Sprintf("%v", item))
				if s != "" && s != "<nil>" {
					out = append(out, s)
				}
			}
			return strings.Join(out, " "), true
		case []string:
			out := make([]string, 0, len(vals))
			for _, item := range vals {
				s := strings.TrimSpace(item)
				if s != "" {
					out = append(out, s)
				}
			}
			return strings.Join(out, " "), true
		default:
			return "", false
		}
	}

	if rawBands, ok := values["time_frequency_analysis.bands"]; ok {
		bands := parseConfigBands(rawBands)
		if len(bands) > 0 {
			m.bands = bands
			m.bandSelected = make(map[int]bool)
			for i := range m.bands {
				m.bandSelected[i] = true
			}
			m.bandCursor = 0
		}
	}

	// fMRI preprocessing defaults (used by the fMRI pipeline wizard)
	if v, ok := values["paths.bids_fmri_root"]; ok {
		if s, ok := asString(v); ok && s != "" {
			m.bidsFmriRoot = s
		}
	}
	if v, ok := values["fmri_preprocessing.engine"]; ok {
		if s, ok := asString(v); ok {
			switch s {
			case "apptainer":
				m.fmriEngineIndex = 1
			case "docker":
				m.fmriEngineIndex = 0
			}
		}
	}
	if v, ok := values["fmri_preprocessing.fmriprep.image"]; ok {
		if s, ok := asString(v); ok && s != "" {
			m.fmriFmriprepImage = s
		}
	}
	if v, ok := values["fmri_preprocessing.fmriprep.output_dir"]; ok {
		if s, ok := asString(v); ok {
			m.fmriFmriprepOutputDir = s
		}
	}
	if v, ok := values["fmri_preprocessing.fmriprep.work_dir"]; ok {
		if s, ok := asString(v); ok {
			m.fmriFmriprepWorkDir = s
		}
	}
	if v, ok := values["fmri_preprocessing.fmriprep.fs_license_file"]; ok {
		if s, ok := asString(v); ok {
			m.fmriFreesurferLicenseFile = s
		}
	}
	if v, ok := values["paths.freesurfer_license"]; ok {
		if strings.TrimSpace(m.fmriFreesurferLicenseFile) == "" {
			if s, ok := asString(v); ok {
				m.fmriFreesurferLicenseFile = s
			}
		}
	}
	if v, ok := values["fmri_preprocessing.fmriprep.fs_subjects_dir"]; ok {
		if s, ok := asString(v); ok {
			m.fmriFreesurferSubjectsDir = s
		}
	}
	if v, ok := values["fmri_preprocessing.fmriprep.output_spaces"]; ok {
		if list, ok := asStringList(v); ok && len(list) > 0 {
			m.fmriOutputSpacesSpec = strings.Join(list, " ")
		}
	}
	if v, ok := values["fmri_preprocessing.fmriprep.ignore"]; ok {
		if list, ok := asStringList(v); ok && len(list) > 0 {
			m.fmriIgnoreSpec = strings.Join(list, " ")
		}
	}
	if v, ok := values["fmri_preprocessing.fmriprep.bids_filter_file"]; ok {
		if s, ok := asString(v); ok {
			m.fmriBidsFilterFile = s
		}
	}
	if v, ok := values["fmri_preprocessing.fmriprep.extra_args"]; ok {
		if s, ok := asString(v); ok {
			m.fmriExtraArgs = s
		}
	}
	if v, ok := values["fmri_preprocessing.fmriprep.use_aroma"]; ok {
		if b, ok := asBool(v); ok {
			m.fmriUseAroma = b
		}
	}
	if v, ok := values["fmri_preprocessing.fmriprep.skip_bids_validation"]; ok {
		if b, ok := asBool(v); ok {
			m.fmriSkipBidsValidation = b
		}
	}
	if v, ok := values["fmri_preprocessing.fmriprep.stop_on_first_crash"]; ok {
		if b, ok := asBool(v); ok {
			m.fmriStopOnFirstCrash = b
		}
	}
	if v, ok := values["fmri_preprocessing.fmriprep.clean_workdir"]; ok {
		if b, ok := asBool(v); ok {
			m.fmriCleanWorkdir = b
		}
	}
	if v, ok := values["fmri_preprocessing.fmriprep.fs_no_reconall"]; ok {
		if b, ok := asBool(v); ok {
			m.fmriSkipReconstruction = b
		}
	}
	if v, ok := values["fmri_preprocessing.fmriprep.mem_mb"]; ok {
		if n, ok := asInt(v); ok {
			m.fmriMemMb = n
		}
	}

	// ML defaults from eeg_config.yaml for TUI hydration.
	if v, ok := values["machine_learning.targets.regression"]; ok {
		if s, ok := asString(v); ok && strings.TrimSpace(m.mlTarget) == "" {
			m.mlTarget = s
		}
	}
	if v, ok := values["machine_learning.targets.classification"]; ok {
		if s, ok := asString(v); ok && strings.TrimSpace(s) != "" {
			m.mlTarget = s
		}
	}
	if v, ok := values["machine_learning.targets.binary_threshold"]; ok && v != nil {
		switch n := v.(type) {
		case float64:
			m.mlBinaryThresholdEnabled = true
			m.mlBinaryThreshold = n
		case int:
			m.mlBinaryThresholdEnabled = true
			m.mlBinaryThreshold = float64(n)
		}
	}
	if v, ok := values["machine_learning.data.feature_families"]; ok {
		if spec, ok := asListSpec(v); ok {
			m.mlFeatureFamiliesSpec = spec
		}
	}
	if v, ok := values["machine_learning.data.feature_bands"]; ok {
		if spec, ok := asListSpec(v); ok {
			m.mlFeatureBandsSpec = spec
		}
	}
	if v, ok := values["machine_learning.data.feature_segments"]; ok {
		if spec, ok := asListSpec(v); ok {
			m.mlFeatureSegmentsSpec = spec
		}
	}
	if v, ok := values["machine_learning.data.feature_scopes"]; ok {
		if spec, ok := asListSpec(v); ok {
			m.mlFeatureScopesSpec = spec
		}
	}
	if v, ok := values["machine_learning.data.feature_stats"]; ok {
		if spec, ok := asListSpec(v); ok {
			m.mlFeatureStatsSpec = spec
		}
	}
	if v, ok := values["machine_learning.data.feature_harmonization"]; ok {
		if s, ok := asString(v); ok {
			switch strings.ToLower(strings.TrimSpace(s)) {
			case "intersection":
				m.mlFeatureHarmonization = MLFeatureHarmonizationIntersection
			case "union_impute":
				m.mlFeatureHarmonization = MLFeatureHarmonizationUnionImpute
			default:
				m.mlFeatureHarmonization = MLFeatureHarmonizationDefault
			}
		}
	}
	if v, ok := values["machine_learning.data.covariates"]; ok {
		if spec, ok := asListSpec(v); ok {
			m.mlCovariatesSpec = spec
		}
	}
	if v, ok := values["machine_learning.data.require_trial_ml_safe"]; ok {
		if b, ok := asBool(v); ok {
			m.mlRequireTrialMlSafe = b
		}
	}
	if v, ok := values["machine_learning.classification.model"]; ok {
		if s, ok := asString(v); ok {
			switch strings.ToLower(strings.TrimSpace(s)) {
			case "svm":
				m.mlClassificationModel = MLClassificationSVM
			case "lr":
				m.mlClassificationModel = MLClassificationLR
			case "rf":
				m.mlClassificationModel = MLClassificationRF
			case "cnn":
				m.mlClassificationModel = MLClassificationCNN
			default:
				m.mlClassificationModel = MLClassificationDefault
			}
		}
	}
	if v, ok := values["machine_learning.cv.inner_splits"]; ok {
		if n, ok := asInt(v); ok {
			m.innerSplits = n
		}
	}
	if v, ok := values["machine_learning.models.elasticnet.alpha_grid"]; ok {
		if spec, ok := asListSpec(v); ok && strings.TrimSpace(spec) != "" {
			m.elasticNetAlphaGrid = strings.Join(splitLooseList(spec), ",")
		}
	}
	if v, ok := values["machine_learning.models.elasticnet.l1_ratio_grid"]; ok {
		if spec, ok := asListSpec(v); ok && strings.TrimSpace(spec) != "" {
			m.elasticNetL1RatioGrid = strings.Join(splitLooseList(spec), ",")
		}
	}
	if v, ok := values["machine_learning.models.ridge.alpha_grid"]; ok {
		if spec, ok := asListSpec(v); ok && strings.TrimSpace(spec) != "" {
			m.ridgeAlphaGrid = strings.Join(splitLooseList(spec), ",")
		}
	}
	if v, ok := values["machine_learning.models.random_forest.n_estimators"]; ok {
		if n, ok := asInt(v); ok {
			m.rfNEstimators = n
		}
	}
	if v, ok := values["machine_learning.models.random_forest.max_depth_grid"]; ok {
		if spec, ok := asListSpec(v); ok && strings.TrimSpace(spec) != "" {
			m.rfMaxDepthGrid = strings.Join(splitLooseList(spec), ",")
		}
	}
	if v, ok := values["machine_learning.preprocessing.variance_threshold_grid"]; ok {
		if spec, ok := asListSpec(v); ok && strings.TrimSpace(spec) != "" {
			m.varianceThresholdGrid = strings.Join(splitLooseList(spec), ",")
		}
	}
}

func parseConfigBands(value interface{}) []FrequencyBand {
	raw, ok := value.(map[string]interface{})
	if !ok {
		return nil
	}
	names := make([]string, 0, len(raw))
	for name := range raw {
		names = append(names, name)
	}
	sort.Strings(names)

	var bands []FrequencyBand
	for _, name := range names {
		entry, ok := raw[name].([]interface{})
		if !ok || len(entry) < 2 {
			continue
		}
		var low, high float64
		var okLow, okHigh bool

		// Try to parse as float64 first
		if f, ok := entry[0].(float64); ok {
			low = f
			okLow = true
		} else if i, ok := entry[0].(int); ok {
			low = float64(i)
			okLow = true
		}

		if f, ok := entry[1].(float64); ok {
			high = f
			okHigh = true
		} else if i, ok := entry[1].(int); ok {
			high = float64(i)
			okHigh = true
		}

		if !okLow || !okHigh {
			continue
		}
		bands = append(bands, FrequencyBand{
			Key:         name,
			Name:        titleCase(name),
			Description: fmt.Sprintf("%.1f-%.1f Hz", low, high),
			LowHz:       low,
			HighHz:      high,
		})
	}
	return bands
}

func titleCase(value string) string {
	if value == "" {
		return value
	}
	return strings.ToUpper(value[:1]) + value[1:]
}

func (m Model) behaviorSections() []behaviorSection {
	return []behaviorSection{
		{Key: "general", Label: "General", Enabled: true},
		{Key: "trial_table", Label: "Trial Table", Enabled: m.isComputationSelected("trial_table")},
		{Key: "correlations", Label: "Correlations", Enabled: m.isComputationSelected("correlations")},
		{Key: "pain_sensitivity", Label: "Pain Sensitivity", Enabled: m.isComputationSelected("pain_sensitivity")},
		{Key: "regression", Label: "Regression", Enabled: m.isComputationSelected("regression")},
		{Key: "stability", Label: "Stability", Enabled: m.isComputationSelected("stability")},
		{Key: "consistency", Label: "Consistency", Enabled: m.isComputationSelected("consistency")},
		{Key: "influence", Label: "Influence", Enabled: m.isComputationSelected("influence")},
		{Key: "condition", Label: "Condition", Enabled: m.isComputationSelected("condition")},
		{Key: "temporal", Label: "Temporal", Enabled: m.isComputationSelected("temporal")},
		{Key: "cluster", Label: "Cluster", Enabled: m.isComputationSelected("cluster")},
		{Key: "mediation", Label: "Mediation", Enabled: m.isComputationSelected("mediation")},
		{Key: "moderation", Label: "Moderation", Enabled: m.isComputationSelected("moderation")},
		{Key: "mixed_effects", Label: "Mixed Effects", Enabled: m.isComputationSelected("mixed_effects")},
	}
}

func (m *Model) setTextFieldValue(field textField, value string) {
	value = strings.TrimSpace(value)
	switch field {
	case textFieldTask:
		m.task = value
	case textFieldBidsRoot:
		m.bidsRoot = value
	case textFieldBidsFmriRoot:
		m.bidsFmriRoot = value
	case textFieldDerivRoot:
		m.derivRoot = value
	case textFieldSourceRoot:
		m.sourceRoot = value
	case textFieldFmriFmriprepImage:
		m.fmriFmriprepImage = value
	case textFieldFmriFmriprepOutputDir:
		m.fmriFmriprepOutputDir = value
	case textFieldFmriFmriprepWorkDir:
		m.fmriFmriprepWorkDir = value
	case textFieldFmriFreesurferLicenseFile:
		m.fmriFreesurferLicenseFile = value
	case textFieldFmriFreesurferSubjectsDir:
		m.fmriFreesurferSubjectsDir = value
	case textFieldFmriOutputSpaces:
		m.fmriOutputSpacesSpec = strings.Join(strings.Fields(value), " ")
	case textFieldFmriIgnore:
		m.fmriIgnoreSpec = strings.Join(strings.Fields(value), " ")
	case textFieldFmriBidsFilterFile:
		m.fmriBidsFilterFile = value
	case textFieldFmriExtraArgs:
		m.fmriExtraArgs = value
	case textFieldFmriSkullStripTemplate:
		m.fmriSkullStripTemplate = strings.TrimSpace(value)
	case textFieldFmriTaskId:
		m.fmriTaskId = strings.TrimSpace(value)
	case textFieldFmriAnalysisFmriprepSpace:
		m.fmriAnalysisFmriprepSpace = strings.TrimSpace(value)
	case textFieldFmriAnalysisRuns:
		m.fmriAnalysisRunsSpec = strings.Join(strings.Fields(value), " ")
	case textFieldFmriAnalysisCondAColumn:
		m.fmriAnalysisCondAColumn = strings.TrimSpace(value)
	case textFieldFmriAnalysisCondAValue:
		m.fmriAnalysisCondAValue = strings.TrimSpace(value)
	case textFieldFmriAnalysisCondBColumn:
		m.fmriAnalysisCondBColumn = strings.TrimSpace(value)
	case textFieldFmriAnalysisCondBValue:
		m.fmriAnalysisCondBValue = strings.TrimSpace(value)
	case textFieldFmriAnalysisContrastName:
		m.fmriAnalysisContrastName = strings.TrimSpace(value)
	case textFieldFmriAnalysisFormula:
		m.fmriAnalysisFormula = strings.TrimSpace(value)
	case textFieldFmriAnalysisEventsToModel:
		m.fmriAnalysisEventsToModel = strings.TrimSpace(value)
	case textFieldFmriAnalysisScopeTrialTypes:
		m.fmriAnalysisScopeTrialTypes = strings.Join(strings.Fields(value), " ")
	case textFieldFmriAnalysisStimPhasesToModel:
		m.fmriAnalysisStimPhasesToModel = strings.TrimSpace(value)
	case textFieldFmriAnalysisOutputDir:
		m.fmriAnalysisOutputDir = strings.TrimSpace(value)
	case textFieldFmriAnalysisFreesurferDir:
		m.fmriAnalysisFreesurferDir = strings.TrimSpace(value)
	case textFieldFmriAnalysisSignatureDir:
		m.fmriAnalysisSignatureDir = strings.TrimSpace(value)
	case textFieldFmriTrialSigGroupColumn:
		m.fmriTrialSigGroupColumn = strings.TrimSpace(value)
		m.fmriTrialSigGroupValuesSpec = "" // Reset values when column changes
	case textFieldFmriTrialSigGroupValues:
		m.fmriTrialSigGroupValuesSpec = strings.Join(strings.Fields(value), " ")
	case textFieldFmriTrialSigScopeTrialTypes:
		m.fmriTrialSigScopeTrialTypes = strings.Join(strings.Fields(value), " ")
	case textFieldFmriTrialSigScopeStimPhases:
		m.fmriTrialSigScopeStimPhases = strings.TrimSpace(value)
	case textFieldRawMontage:
		m.rawMontage = value
	case textFieldPrepMontage:
		m.prepMontage = value
	case textFieldPrepChTypes:
		m.prepChTypes = value
	case textFieldPrepEegReference:
		m.prepEegReference = value
	case textFieldPrepEogChannels:
		m.prepEogChannels = value
	case textFieldPrepConditions:
		m.prepConditions = value
	case textFieldPrepFileExtension:
		m.prepFileExtension = strings.TrimSpace(value)
	case textFieldPrepRenameAnotDict:
		m.prepRenameAnotDict = value
	case textFieldPrepCustomBadDict:
		m.prepCustomBadDict = value
	case textFieldRawEventPrefixes:
		m.rawEventPrefixes = value
	case textFieldMergeEventPrefixes:
		m.mergeEventPrefixes = value
	case textFieldMergeEventTypes:
		m.mergeEventTypes = value
	case textFieldMergeQCColumns:
		m.mergeQCColumns = value
	case textFieldFmriRawSession:
		m.fmriRawSession = value
	case textFieldFmriRawRestTask:
		m.fmriRawRestTask = value
	case textFieldFmriRawDcm2niixPath:
		m.fmriRawDcm2niixPath = value
	case textFieldFmriRawDcm2niixArgs:
		m.fmriRawDcm2niixArgs = value
	case textFieldConditionCompareColumn:
		m.conditionCompareColumn = strings.TrimSpace(value)
	case textFieldConditionCompareWindows:
		m.conditionCompareWindows = strings.TrimSpace(value)
	case textFieldConditionCompareValues:
		m.conditionCompareValues = strings.TrimSpace(value)
	case textFieldTemporalConditionColumn:
		m.temporalConditionColumn = strings.TrimSpace(value)
	case textFieldTemporalConditionValues:
		m.temporalConditionValues = strings.TrimSpace(value)
	case textFieldTemporalTargetColumn:
		m.temporalTargetColumn = strings.TrimSpace(value)
	case textFieldTfHeatmapFreqs:
		m.tfHeatmapFreqsSpec = strings.Join(strings.Fields(value), "")
	case textFieldRunAdjustmentColumn:
		m.runAdjustmentColumn = strings.TrimSpace(value)
	case textFieldPainResidualCrossfitGroupColumn:
		m.painResidualCrossfitGroupColumn = strings.TrimSpace(value)
	case textFieldPainResidualSplineDfCandidates:
		m.painResidualSplineDfCandidates = strings.Join(strings.Fields(value), "")
	case textFieldPainResidualModelComparePolyDegrees:
		m.painResidualModelComparePolyDegrees = strings.Join(strings.Fields(value), "")
	case textFieldClusterConditionColumn:
		m.clusterConditionColumn = strings.TrimSpace(value)
	case textFieldClusterConditionValues:
		m.clusterConditionValues = strings.TrimSpace(value)
	case textFieldCorrelationsTargetColumn:
		m.correlationsTargetColumn = strings.TrimSpace(value)
	case textFieldCorrelationsTypes:
		m.correlationsTypesSpec = strings.Join(strings.Fields(value), "")
	case textFieldItpcConditionColumn:
		m.itpcConditionColumn = strings.TrimSpace(value)
	case textFieldItpcConditionValues:
		m.itpcConditionValues = strings.TrimSpace(value)
	case textFieldConnConditionColumn:
		m.connConditionColumn = strings.TrimSpace(value)
	case textFieldConnConditionValues:
		m.connConditionValues = strings.TrimSpace(value)
	case textFieldPACPairs:
		m.pacPairsSpec = strings.Join(strings.Fields(value), "")
	case textFieldBurstBands:
		m.burstBandsSpec = strings.Join(strings.Fields(value), "")
	case textFieldERDSBands:
		m.erdsBandsSpec = strings.Join(strings.Fields(value), "")
	case textFieldSpectralRatioPairs:
		m.spectralRatioPairsSpec = strings.Join(strings.Fields(value), "")
	case textFieldAsymmetryChannelPairs:
		m.asymmetryChannelPairsSpec = strings.Join(strings.Fields(value), "")
	case textFieldAsymmetryActivationBands:
		m.asymmetryActivationBandsSpec = strings.Join(strings.Fields(value), "")
	case textFieldIAFRois:
		m.iafRoisSpec = strings.Join(strings.Fields(value), "")
	case textFieldERPComponents:
		m.erpComponentsSpec = strings.Join(strings.Fields(value), "")
	case textFieldSourceLocSubject:
		m.sourceLocSubject = value
	case textFieldSourceLocTrans:
		m.sourceLocTrans = value
	case textFieldSourceLocBem:
		m.sourceLocBem = value
	case textFieldSourceLocFmriStatsMap:
		m.sourceLocFmriStatsMap = value
	case textFieldSourceLocFmriCondAColumn:
		m.sourceLocFmriCondAColumn = value
	case textFieldSourceLocFmriCondAValue:
		m.sourceLocFmriCondAValue = value
	case textFieldSourceLocFmriCondBColumn:
		m.sourceLocFmriCondBColumn = value
	case textFieldSourceLocFmriCondBValue:
		m.sourceLocFmriCondBValue = value
	case textFieldSourceLocFmriContrastFormula:
		m.sourceLocFmriContrastFormula = value
	case textFieldSourceLocFmriContrastName:
		m.sourceLocFmriContrastName = value
	case textFieldSourceLocFmriRunsToInclude:
		m.sourceLocFmriRunsToInclude = value
	case textFieldSourceLocFmriConditionScopeTrialTypes:
		m.sourceLocFmriConditionScopeTrialTypes = strings.Join(strings.Fields(value), " ")
	case textFieldSourceLocFmriStimPhasesToModel:
		m.sourceLocFmriStimPhasesToModel = strings.TrimSpace(value)
	case textFieldSourceLocFmriWindowAName:
		m.sourceLocFmriWindowAName = value
	case textFieldSourceLocFmriWindowBName:
		m.sourceLocFmriWindowBName = value
	case textFieldPlotBboxInches:
		m.plotBboxInches = value
	case textFieldPlotFontFamily:
		m.plotFontFamily = value
	case textFieldPlotFontWeight:
		m.plotFontWeight = value
	case textFieldPlotLayoutTightRect:
		m.plotLayoutTightRectSpec = strings.Join(strings.Fields(value), " ")
	case textFieldPlotLayoutTightRectMicrostate:
		m.plotLayoutTightRectMicrostateSpec = strings.Join(strings.Fields(value), " ")
	case textFieldPlotGridSpecWidthRatios:
		m.plotGridSpecWidthRatiosSpec = strings.Join(strings.Fields(value), " ")
	case textFieldPlotGridSpecHeightRatios:
		m.plotGridSpecHeightRatiosSpec = strings.Join(strings.Fields(value), " ")
	case textFieldPlotFigureSizeStandard:
		m.plotFigureSizeStandardSpec = strings.Join(strings.Fields(value), " ")
	case textFieldPlotFigureSizeMedium:
		m.plotFigureSizeMediumSpec = strings.Join(strings.Fields(value), " ")
	case textFieldPlotFigureSizeSmall:
		m.plotFigureSizeSmallSpec = strings.Join(strings.Fields(value), " ")
	case textFieldPlotFigureSizeSquare:
		m.plotFigureSizeSquareSpec = strings.Join(strings.Fields(value), " ")
	case textFieldPlotFigureSizeWide:
		m.plotFigureSizeWideSpec = strings.Join(strings.Fields(value), " ")
	case textFieldPlotFigureSizeTFR:
		m.plotFigureSizeTFRSpec = strings.Join(strings.Fields(value), " ")
	case textFieldPlotFigureSizeTopomap:
		m.plotFigureSizeTopomapSpec = strings.Join(strings.Fields(value), " ")
	case textFieldPlotColorPain:
		m.plotColorPain = value
	case textFieldPlotColorNonpain:
		m.plotColorNonpain = value
	case textFieldPlotColorSignificant:
		m.plotColorSignificant = value
	case textFieldPlotColorNonsignificant:
		m.plotColorNonsignificant = value
	case textFieldPlotColorGray:
		m.plotColorGray = value
	case textFieldPlotColorLightGray:
		m.plotColorLightGray = value
	case textFieldPlotColorBlack:
		m.plotColorBlack = value
	case textFieldPlotColorBlue:
		m.plotColorBlue = value
	case textFieldPlotColorRed:
		m.plotColorRed = value
	case textFieldPlotColorNetworkNode:
		m.plotColorNetworkNode = value
	case textFieldPlotScatterEdgecolor:
		m.plotScatterEdgeColor = value
	case textFieldPlotHistEdgecolor:
		m.plotHistEdgeColor = value
	case textFieldPlotKdeColor:
		m.plotKdeColor = value
	case textFieldPlotTopomapColormap:
		m.plotTopomapColormap = value
	case textFieldPlotTopomapSigMaskMarker:
		m.plotTopomapSigMaskMarker = value
	case textFieldPlotTopomapSigMaskMarkerFaceColor:
		m.plotTopomapSigMaskMarkerFaceColor = value
	case textFieldPlotTopomapSigMaskMarkerEdgeColor:
		m.plotTopomapSigMaskMarkerEdgeColor = value
	case textFieldPlotTfrDefaultBaselineWindow:
		m.plotTfrDefaultBaselineWindowSpec = strings.Join(strings.Fields(value), " ")
	case textFieldPlotPacCmap:
		m.plotPacCmap = value
	case textFieldPlotPacPairs:
		m.plotPacPairsSpec = strings.Join(strings.Fields(value), " ")
	case textFieldPlotConnectivityMeasures:
		// Parse space-separated measure keys (e.g. "aec wpli") and map to the
		// checkbox model used elsewhere in the wizard.
		for i := range connectivityMeasures {
			m.connectivityMeasures[i] = false
		}
		for _, token := range strings.Fields(value) {
			for i, measure := range connectivityMeasures {
				if strings.EqualFold(token, measure.Key) || strings.EqualFold(token, measure.Name) {
					m.connectivityMeasures[i] = true
				}
			}
		}
	case textFieldPlotSpectralMetrics:
		m.plotSpectralMetricsSpec = strings.Join(strings.Fields(value), " ")
	case textFieldPlotBurstsMetrics:
		m.plotBurstsMetricsSpec = strings.Join(strings.Fields(value), " ")
	case textFieldPlotTemporalTimeBins:
		m.plotTemporalTimeBinsSpec = strings.Join(strings.Fields(value), " ")
	case textFieldPlotTemporalTimeLabels:
		m.plotTemporalTimeLabelsSpec = strings.Join(strings.Fields(value), " ")
	case textFieldPlotAsymmetryStat:
		m.plotAsymmetryStatSpec = value
	case textFieldPlotComparisonWindows:
		m.plotComparisonWindowsSpec = strings.Join(strings.Fields(value), " ")
	case textFieldPlotComparisonSegment:
		m.plotComparisonSegment = strings.TrimSpace(value)
	case textFieldPlotComparisonColumn:
		m.plotComparisonColumn = strings.TrimSpace(value)
	case textFieldPlotComparisonValues:
		m.plotComparisonValuesSpec = strings.Join(strings.Fields(value), " ")
	case textFieldPlotComparisonLabels:
		m.plotComparisonLabelsSpec = strings.TrimSpace(value)
	case textFieldPlotComparisonROIs:
		m.plotComparisonROIsSpec = strings.Join(strings.Fields(value), " ")
	// Machine Learning advanced config text fields
	case textFieldMLTarget:
		m.mlTarget = strings.TrimSpace(value)
	case textFieldMLFmriSigContrastName:
		m.mlFmriSigContrastName = strings.TrimSpace(value)
	case textFieldMLFeatureFamilies:
		m.mlFeatureFamiliesSpec = strings.TrimSpace(value)
	case textFieldMLFeatureBands:
		m.mlFeatureBandsSpec = strings.TrimSpace(value)
	case textFieldMLFeatureSegments:
		m.mlFeatureSegmentsSpec = strings.TrimSpace(value)
	case textFieldMLFeatureScopes:
		m.mlFeatureScopesSpec = strings.TrimSpace(value)
	case textFieldMLFeatureStats:
		m.mlFeatureStatsSpec = strings.TrimSpace(value)
	case textFieldMLCovariates:
		m.mlCovariatesSpec = strings.TrimSpace(value)
	case textFieldMLBaselinePredictors:
		m.mlBaselinePredictorsSpec = strings.TrimSpace(value)
	// Machine Learning hyperparameter text fields
	case textFieldElasticNetAlphaGrid:
		m.elasticNetAlphaGrid = strings.Join(splitLooseList(value), ",")
	case textFieldElasticNetL1RatioGrid:
		m.elasticNetL1RatioGrid = strings.Join(splitLooseList(value), ",")
	case textFieldRidgeAlphaGrid:
		m.ridgeAlphaGrid = strings.Join(splitLooseList(value), ",")
	case textFieldRfMaxDepthGrid:
		m.rfMaxDepthGrid = strings.Join(splitLooseList(value), ",")
	case textFieldVarianceThresholdGrid:
		m.varianceThresholdGrid = strings.Join(splitLooseList(value), ",")
	// Preprocessing text fields
	case textFieldIcaLabelsToKeep:
		m.icaLabelsToKeep = strings.Join(strings.Fields(value), "")
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

///////////////////////////////////////////////////////////////////
// Advanced Options Helpers
///////////////////////////////////////////////////////////////////

type optionType int

const (
	// Feature Pipeline Advanced Options
	optUseDefaults optionType = iota
	// Features section headers (expand/collapse)
	optFeatGroupConnectivity
	optFeatGroupDirectedConnectivity
	optFeatGroupPAC
	optFeatGroupAperiodic
	optFeatGroupComplexity
	optFeatGroupBursts
	optFeatGroupPower
	optFeatGroupSpectral
	optFeatGroupERP
	optFeatGroupRatios
	optFeatGroupAsymmetry
	optFeatGroupQuality
	optFeatGroupERDS
	optFeatGroupSpatialTransform
	optFeatGroupStorage
	optFeatGroupExecution
	optFeatGroupSourceLoc
	optFeatGroupITPC
	// Behavior section headers (expand/collapse)
	optBehaviorGroupGeneral
	optBehaviorGroupTrialTable
	optBehaviorGroupPainResidual
	optBehaviorGroupCorrelations
	optBehaviorGroupPainSens
	optBehaviorGroupRegression
	optBehaviorGroupModels
	optBehaviorGroupStability
	optBehaviorGroupConsistency
	optBehaviorGroupInfluence
	optBehaviorGroupReport
	optBehaviorGroupCondition
	optBehaviorGroupTemporal
	optBehaviorGroupCluster
	optBehaviorGroupMediation
	optBehaviorGroupModeration
	optBehaviorGroupMixedEffects
	optConnectivity
	optPACPhaseRange
	optPACAmpRange
	optPACMethod
	optPACMinEpochs
	optPACPairs
	optPACSource
	optPACNormalize
	optPACNSurrogates
	optPACAllowHarmonicOverlap
	optPACMaxHarmonic
	optPACHarmonicToleranceHz
	optPACRandomSeed
	optPACComputeWaveformQC
	optPACWaveformOffsetMs
	optAperiodicFmin
	optAperiodicFmax
	optAperiodicPeakZ
	optAperiodicMinR2
	optAperiodicMinPoints
	optAperiodicPsdBandwidth
	optAperiodicMaxRms
	optAperiodicLineNoiseFreq
	optAperiodicLineNoiseWidthHz
	optAperiodicLineNoiseHarmonics
	optAperiodicMinSegmentSec
	optAperiodicModel
	optAperiodicPsdMethod
	optAperiodicExcludeLineNoise
	optPEOrder
	optPEDelay
	optComplexitySignalBasis
	optComplexityMinSegmentSec
	optComplexityMinSamples
	optComplexityZscore
	optBurstThresholdMethod
	optBurstThresholdPercentile
	optBurstThreshold
	optBurstThresholdReference
	optBurstMinTrialsPerCondition
	optBurstMinSegmentSec
	optBurstSkipInvalidSegments
	optBurstMinDuration
	optBurstMinCycles
	optBurstBands
	optERPBaseline
	optERPAllowNoBaseline
	optERPComponents
	optERPSmoothMs
	optERPPeakProminenceUv
	optERPLowpassHz
	optPowerBaselineMode
	optPowerRequireBaseline
	optPowerSubtractEvoked
	optPowerMinTrialsPerCondition
	optPowerExcludeLineNoise
	optPowerLineNoiseFreq
	optPowerLineNoiseWidthHz
	optPowerLineNoiseHarmonics
	optPowerEmitDb
	optSpectralEdge
	optSpectralRatioPairs
	optSpectralIncludeLogRatios
	optSpectralExcludeLineNoise
	optSpectralLineNoiseFreq
	optSpectralLineNoiseWidthHz
	optSpectralLineNoiseHarmonics
	optSpectralPsdMethod
	optSpectralPsdAdaptive
	optSpectralMultitaperAdaptive
	optSpectralFmin
	optSpectralFmax
	optSpectralMinSegmentSec
	optSpectralMinCyclesAtFmin
	optAperiodicSubtractEvoked
	optAsymmetryChannelPairs
	optAsymmetryMinSegmentSec
	optAsymmetryMinCyclesAtFmin
	optAsymmetrySkipInvalidSegments
	optAsymmetryEmitActivationConvention
	optAsymmetryActivationBands
	// Ratios options
	optRatiosMinSegmentSec
	optRatiosMinCyclesAtFmin
	optRatiosSkipInvalidSegments
	// Quality options
	optQualityPsdMethod
	optQualityFmin
	optQualityFmax
	optQualityNFft
	optQualityExcludeLineNoise
	optQualityLineNoiseFreq
	optQualityLineNoiseWidthHz
	optQualityLineNoiseHarmonics
	optQualitySnrSignalBandMin
	optQualitySnrSignalBandMax
	optQualitySnrNoiseBandMin
	optQualitySnrNoiseBandMax
	optQualityMuscleBandMin
	optQualityMuscleBandMax
	// ERDS options
	optERDSUseLogRatio
	optERDSMinBaselinePower
	optERDSMinActivePower
	optERDSMinSegmentSec
	optERDSBands
	optERDSOnsetThresholdSigma
	optERDSOnsetMinDurationMs
	optERDSReboundMinLatencyMs
	optERDSInferContralateral
	optConnOutputLevel
	optConnGraphMetrics
	optConnGraphProp
	optConnWindowLen
	optConnWindowStep
	optConnAECMode
	optConnAECOutput
	optConnForceWithinEpochML
	optConnGranularity
	optConnConditionColumn
	optConnConditionValues
	optConnMinEpochsPerGroup
	optConnMinCyclesPerBand
	optConnWarnNoSpatialTransform
	optConnPhaseEstimator
	optConnMinSegmentSec
	// Directed connectivity options (PSI, DTF, PDC)
	optDirectedConnMeasures
	optDirectedConnOutputLevel
	optDirectedConnMvarOrder
	optDirectedConnNFreqs
	optDirectedConnMinSegSamples
	// Spatial transform options
	optSpatialTransform
	optSpatialTransformLambda2
	optSpatialTransformStiffness
	optMinEpochs
	optFeatAnalysisMode
	optFeatComputeChangeScores
	optFeatSaveTfrWithSidecar
	optFeatNJobsBands
	optFeatNJobsConnectivity
	optFeatNJobsAperiodic
	optFeatNJobsComplexity
	// Source localization options (LCMV, eLORETA)
	optSourceLocMode
	optSourceLocMethod
	optSourceLocSpacing
	optSourceLocParc
	optSourceLocReg
	optSourceLocSnr
	optSourceLocLoose
	optSourceLocDepth
	optSourceLocConnMethod
	optSourceLocSubject
	optSourceLocTrans
	optSourceLocBem
	optSourceLocMindistMm
	optSourceLocFmriEnabled
	optSourceLocFmriStatsMap
	optSourceLocFmriProvenance
	optSourceLocFmriRequireProvenance
	optSourceLocFmriThreshold
	optSourceLocFmriTail
	optSourceLocFmriMinClusterVox
	optSourceLocFmriMinClusterMM3
	optSourceLocFmriMaxClusters
	optSourceLocFmriMaxVoxPerClus
	optSourceLocFmriMaxTotalVox
	optSourceLocFmriRandomSeed
	// BEM/Trans generation options (Docker-based)
	optSourceLocCreateTrans
	optSourceLocCreateBemModel
	optSourceLocCreateBemSolution
	// fMRI GLM contrast builder options
	optSourceLocFmriContrastEnabled
	optSourceLocFmriContrastType
	optSourceLocFmriCondAColumn
	optSourceLocFmriCondAValue
	optSourceLocFmriCondBColumn
	optSourceLocFmriCondBValue
	optSourceLocFmriContrastFormula
	optSourceLocFmriContrastName
	optSourceLocFmriRunsToInclude
	optSourceLocFmriAutoDetectRuns
	optSourceLocFmriHrfModel
	optSourceLocFmriDriftModel
	optSourceLocFmriHighPassHz
	optSourceLocFmriLowPassHz
	optSourceLocFmriConditionScopeTrialTypes
	optSourceLocFmriStimPhasesToModel
	optSourceLocFmriClusterCorrection
	optSourceLocFmriClusterPThreshold
	optSourceLocFmriOutputType
	optSourceLocFmriResampleToFS
	optSourceLocFmriWindowAName
	optSourceLocFmriWindowATmin
	optSourceLocFmriWindowATmax
	optSourceLocFmriWindowBName
	optSourceLocFmriWindowBTmin
	optSourceLocFmriWindowBTmax
	// ITPC options (condition-based)
	optItpcMethod
	optItpcAllowUnsafeLoo
	optItpcBaselineCorrection
	optItpcConditionColumn
	optItpcConditionValues
	optItpcMinTrialsPerCondition
	optItpcNJobs

	// Storage options
	optSaveSubjectLevelFeatures
	optFeatAlsoSaveCsv
	// Behavior options - General
	optCorrMethod
	optBootstrap
	optNPerm
	optRNGSeed
	optControlTemp
	optControlOrder
	optRunAdjustmentEnabled
	optRunAdjustmentColumn
	optRunAdjustmentIncludeInCorrelations
	optRunAdjustmentMaxDummies
	optFDRAlpha
	// Behavior options - Cluster
	optClusterThreshold
	optClusterMinSize
	optClusterTail
	optClusterConditionColumn
	optClusterConditionValues
	// Behavior options - Mediation
	optMediationBootstrap
	optMediationMaxMediatorsEnabled
	optMediationMaxMediators
	optMediationPermutations
	// Behavior options - Moderation
	optModerationMaxFeaturesEnabled
	optModerationMaxFeatures
	optModerationPermutations
	// Behavior options - Mixed Effects
	optMixedMaxFeatures
	// Behavior options - Condition
	optConditionEffectThreshold
	optConditionFailFast
	optConditionOverwrite
	optConditionCompareColumn
	optConditionCompareWindows
	optConditionCompareValues
	optConditionWindowPrimaryUnit
	optConditionPermutationPrimary
	// Behavior options - Trial table / residual
	optTrialTableFormat
	optTrialTableAddLagFeatures
	optTrialOrderMaxMissingFraction
	optFeatureSummariesEnabled
	optFeatureQCEnabled
	optFeatureQCMaxMissingPct
	optFeatureQCMinVariance
	optFeatureQCCheckWithinRunVariance
	optPainResidualEnabled
	optPainResidualMethod
	optPainResidualPolyDegree
	optPainResidualSplineDfCandidates
	optPainResidualModelCompare
	optPainResidualModelComparePolyDegrees
	optPainResidualBreakpoint
	optPainResidualBreakpointCandidates
	optPainResidualBreakpointQlow
	optPainResidualBreakpointQhigh
	optPainResidualCrossfitEnabled
	optPainResidualCrossfitGroupColumn
	optPainResidualCrossfitNSplits
	optPainResidualCrossfitMethod
	optPainResidualCrossfitSplineKnots
	// Behavior options - General extra
	optRobustCorrelation
	optBehaviorNJobs
	optComputeChangeScores
	optComputeBayesFactors
	optComputeLosoStability
	// Behavior options - Regression
	optRegressionOutcome
	optRegressionIncludeTemperature
	optRegressionTempControl
	optRegressionTempSplineKnots
	optRegressionTempSplineQlow
	optRegressionTempSplineQhigh
	optRegressionIncludeTrialOrder
	optRegressionIncludePrev
	optRegressionIncludeRunBlock
	optRegressionIncludeInteraction
	optRegressionStandardize
	optRegressionPermutations
	optRegressionMaxFeatures
	// Behavior options - Models
	optModelsIncludeTemperature
	optModelsTempControl
	optModelsTempSplineKnots
	optModelsTempSplineQlow
	optModelsTempSplineQhigh
	optModelsIncludeTrialOrder
	optModelsIncludePrev
	optModelsIncludeRunBlock
	optModelsIncludeInteraction
	optModelsStandardize
	optModelsMaxFeatures
	optModelsOutcomeRating
	optModelsOutcomePainResidual
	optModelsOutcomeTemperature
	optModelsOutcomePainBinary
	optModelsFamilyOLS
	optModelsFamilyRobust
	optModelsFamilyQuantile
	optModelsFamilyLogit
	optModelsBinaryOutcome
	// Behavior options - Stability
	optStabilityMethod
	optStabilityOutcome
	optStabilityGroupColumn
	optStabilityPartialTemp
	optStabilityMaxFeatures
	optStabilityAlpha
	// Behavior options - Consistency / Influence
	optConsistencyEnabled
	optInfluenceOutcomeRating
	optInfluenceOutcomePainResidual
	optInfluenceOutcomeTemperature
	optInfluenceMaxFeatures
	optInfluenceIncludeTemperature
	optInfluenceTempControl
	optInfluenceTempSplineKnots
	optInfluenceTempSplineQlow
	optInfluenceTempSplineQhigh
	optInfluenceIncludeTrialOrder
	optInfluenceIncludeRunBlock
	optInfluenceIncludeInteraction
	optInfluenceStandardize
	optInfluenceCooksThreshold
	optInfluenceLeverageThreshold
	// Behavior options - Report
	optReportTopN
	// Behavior options - Correlations / pain sensitivity
	optCorrelationsTargetRating
	optCorrelationsTargetTemperature
	optCorrelationsTargetPainResidual
	optCorrelationsTypes
	optCorrelationsPreferPainResidual
	optCorrelationsUseCrossfitPainResidual
	optCorrelationsMultilevel
	optCorrelationsPrimaryUnit
	optCorrelationsPermutationPrimary
	optCorrelationsTargetColumn
	optGroupLevelBlockPermutation
	// Behavior options - Pain sensitivity / temporal
	optTemporalResolutionMs
	optTemporalTimeMinMs
	optTemporalTimeMaxMs
	optTemporalSmoothMs
	optTemporalTargetColumn
	optTemporalSplitByCondition
	optTemporalConditionColumn
	optTemporalConditionValues
	optTemporalIncludeROIAverages
	optTemporalIncludeTFGrid
	// Temporal feature selection
	optTemporalFeaturePower
	optTemporalFeatureITPC
	optTemporalFeatureERDS
	// ITPC-specific options
	optTemporalITPCBaselineCorrection
	optTemporalITPCBaselineMin
	optTemporalITPCBaselineMax
	// ERDS-specific options
	optTemporalERDSBaselineMin
	optTemporalERDSBaselineMax
	optTemporalERDSMethod
	// TF Heatmap options
	optTemporalTfHeatmapEnabled
	optTemporalTfHeatmapFreqs
	optTemporalTfHeatmapTimeResMs
	// Behavior options - Mixed effects / mediation
	optMixedEffectsType
	optMediationMinEffect
	// Behavior options - Output
	optBehaviorGroupOutput
	optAlsoSaveCsv
	optBehaviorOverwrite
	// Plotting options
	optPlotPNG
	optPlotSVG
	optPlotPDF
	optPlotDPI
	optPlotSaveDPI
	optPlotSharedColorbar
	// Machine Learning options
	optMLNPerm
	optMLInnerSplits
	optMLOuterJobs
	optMLTarget
	optMLFmriSigGroup
	optMLFmriSigMethod
	optMLFmriSigContrastName
	optMLFmriSigSignature
	optMLFmriSigMetric
	optMLFmriSigNormalization
	optMLFmriSigRoundDecimals
	optMLRegressionModel
	optMLClassificationModel
	optMLBinaryThresholdEnabled
	optMLBinaryThreshold
	optMLFeatureFamilies
	optMLFeatureBands
	optMLFeatureSegments
	optMLFeatureScopes
	optMLFeatureStats
	optMLFeatureHarmonization
	optMLCovariates
	optMLBaselinePredictors
	optMLRequireTrialMlSafe
	optMLUncertaintyAlpha
	optMLPermNRepeats
	// Preprocessing group headers (collapsible sections)
	optPrepGroupStages
	optPrepGroupGeneral
	optPrepGroupFiltering
	optPrepGroupPyprep
	optPrepGroupICA
	optPrepGroupEpoching
	// Preprocessing options
	optPrepStageBadChannels
	optPrepStageFiltering
	optPrepStageICA
	optPrepStageEpoching
	optPrepUsePyprep
	optPrepUseIcalabel
	optPrepNJobs
	optPrepMontage
	optPrepResample
	optPrepLFreq
	optPrepHFreq
	optPrepNotch
	optPrepLineFreq
	optPrepChTypes
	optPrepEegReference
	optPrepEogChannels
	optPrepRandomState
	optPrepTaskIsRest
	optPrepZaplineFline
	optPrepFindBreaks
	// PyPREP advanced options
	optPrepRansac
	optPrepRepeats
	optPrepAverageReref
	optPrepFileExtension
	optPrepConsiderPreviousBads
	optPrepOverwriteChansTsv
	optPrepDeleteBreaks
	optPrepBreaksMinLength
	optPrepTStartAfterPrevious
	optPrepTStopBeforeNext
	optPrepRenameAnotDict
	optPrepCustomBadDict
	// ICA options
	optPrepSpatialFilter
	optPrepICAAlgorithm
	optPrepICAComp
	optPrepICALFreq
	optPrepICARejThresh
	optPrepProbThresh
	optPrepKeepMnebidsBads
	optIcaLabelsToKeep
	// Epoching options
	optPrepConditions
	optPrepRejectMethod
	optPrepRunSourceEstimation
	optPrepEpochsTmin
	optPrepEpochsTmax
	optPrepEpochsBaseline
	optPrepEpochsNoBaseline
	optPrepEpochsReject
	optPrepWriteCleanEvents
	optPrepOverwriteCleanEvents
	optPrepCleanEventsStrict
	// Raw-to-BIDS options
	optRawMontage
	optRawLineFreq
	optRawOverwrite
	optRawTrimToFirstVolume
	optRawEventPrefixes
	optRawKeepAnnotations
	// Merge-behavior options
	optMergeEventPrefixes
	optMergeEventTypes
	optMergeQCColumns
	// fMRI Raw-to-BIDS options
	optFmriRawSession
	optFmriRawRestTask
	optFmriRawIncludeRest
	optFmriRawIncludeFieldmaps
	optFmriRawDicomMode
	optFmriRawOverwrite
	optFmriRawCreateEvents
	optFmriRawEventGranularity
	optFmriRawOnsetReference
	optFmriRawOnsetOffsetS
	optFmriRawDcm2niixPath
	optFmriRawDcm2niixArgs
	// Plotting advanced options (wizard-only)
	optPlotGroupDefaults
	optPlotGroupFonts
	optPlotGroupLayout
	optPlotGroupFigureSizes
	optPlotGroupColors
	optPlotGroupAlpha
	optPlotGroupScatter
	optPlotGroupBar
	optPlotGroupLine
	optPlotGroupHistogram
	optPlotGroupKDE
	optPlotGroupErrorbar
	optPlotGroupText
	optPlotGroupValidation
	optPlotGroupTFRMisc
	optPlotGroupTopomap
	optPlotGroupTFR
	optPlotGroupSizing
	optPlotGroupSelection
	optPlotGroupComparisons
	optPlotBboxInches
	optPlotPadInches
	optPlotFontFamily
	optPlotFontWeight
	optPlotFontSizeSmall
	optPlotFontSizeMedium
	optPlotFontSizeLarge
	optPlotFontSizeTitle
	optPlotFontSizeAnnotation
	optPlotFontSizeLabel
	optPlotFontSizeYLabel
	optPlotFontSizeSuptitle
	optPlotFontSizeFigureTitle
	optPlotLayoutTightRect
	optPlotLayoutTightRectMicrostate
	optPlotGridSpecWidthRatios
	optPlotGridSpecHeightRatios
	optPlotGridSpecHspace
	optPlotGridSpecWspace
	optPlotGridSpecLeft
	optPlotGridSpecRight
	optPlotGridSpecTop
	optPlotGridSpecBottom
	optPlotFigureSizeStandard
	optPlotFigureSizeMedium
	optPlotFigureSizeSmall
	optPlotFigureSizeSquare
	optPlotFigureSizeWide
	optPlotFigureSizeTFR
	optPlotFigureSizeTopomap
	optPlotColorPain
	optPlotColorNonpain
	optPlotColorSignificant
	optPlotColorNonsignificant
	optPlotColorGray
	optPlotColorLightGray
	optPlotColorBlack
	optPlotColorBlue
	optPlotColorRed
	optPlotColorNetworkNode
	optPlotAlphaGrid
	optPlotAlphaFill
	optPlotAlphaCI
	optPlotAlphaCILine
	optPlotAlphaTextBox
	optPlotAlphaViolinBody
	optPlotAlphaRidgeFill
	optPlotScatterMarkerSizeSmall
	optPlotScatterMarkerSizeLarge
	optPlotScatterMarkerSizeDefault
	optPlotScatterAlpha
	optPlotScatterEdgecolor
	optPlotScatterEdgewidth
	optPlotBarAlpha
	optPlotBarWidth
	optPlotBarCapsize
	optPlotBarCapsizeLarge
	optPlotLineWidthThin
	optPlotLineWidthStandard
	optPlotLineWidthThick
	optPlotLineWidthBold
	optPlotLineAlphaStandard
	optPlotLineAlphaDim
	optPlotLineAlphaZeroLine
	optPlotLineAlphaFitLine
	optPlotLineAlphaDiagonal
	optPlotLineAlphaReference
	optPlotLineRegressionWidth
	optPlotLineResidualWidth
	optPlotLineQQWidth
	optPlotHistBins
	optPlotHistBinsBehavioral
	optPlotHistBinsResidual
	optPlotHistBinsTFR
	optPlotHistEdgecolor
	optPlotHistEdgewidth
	optPlotHistAlpha
	optPlotHistAlphaResidual
	optPlotHistAlphaTFR
	optPlotKdePoints
	optPlotKdeColor
	optPlotKdeLinewidth
	optPlotKdeAlpha
	optPlotErrorbarMarkersize
	optPlotErrorbarCapsize
	optPlotErrorbarCapsizeLarge
	optPlotTextStatsX
	optPlotTextStatsY
	optPlotTextPvalueX
	optPlotTextPvalueY
	optPlotTextBootstrapX
	optPlotTextBootstrapY
	optPlotTextChannelAnnotationX
	optPlotTextChannelAnnotationY
	optPlotTextTitleY
	optPlotTextResidualQcTitleY
	optPlotValidationMinBinsForCalibration
	optPlotValidationMaxBinsForCalibration
	optPlotValidationSamplesPerBin
	optPlotValidationMinRoisForFDR
	optPlotValidationMinPvaluesForFDR
	optPlotTfrDefaultBaselineWindow
	optPlotTopomapContours
	optPlotTopomapColormap
	optPlotTopomapColorbarFraction
	optPlotTopomapColorbarPad
	optPlotTopomapDiffAnnotation
	optPlotTopomapAnnotateDescriptive
	optPlotTopomapSigMaskMarker
	optPlotTopomapSigMaskMarkerFaceColor
	optPlotTopomapSigMaskMarkerEdgeColor
	optPlotTopomapSigMaskLinewidth
	optPlotTopomapSigMaskMarkersize
	optPlotTFRLogBase
	optPlotTFRPercentageMultiplier
	optPlotTFRTopomapWindowSizeMs
	optPlotTFRTopomapWindowCount
	optPlotTFRTopomapLabelXPosition
	optPlotTFRTopomapLabelYPositionBottom
	optPlotTFRTopomapLabelYPosition
	optPlotTFRTopomapTitleY
	optPlotTFRTopomapTitlePad
	optPlotTFRTopomapSubplotsRight
	optPlotTFRTopomapTemporalHspace
	optPlotTFRTopomapTemporalWspace
	optPlotRoiWidthPerBand
	optPlotRoiWidthPerMetric
	optPlotRoiHeightPerRoi
	optPlotPowerWidthPerBand
	optPlotPowerHeightPerSegment
	optPlotItpcWidthPerBin
	optPlotItpcHeightPerBand
	optPlotItpcWidthPerBandBox
	optPlotItpcHeightBox
	optPlotPacCmap
	optPlotPacWidthPerRoi
	optPlotPacHeightBox
	optPlotAperiodicWidthPerColumn
	optPlotAperiodicHeightPerRow
	optPlotAperiodicNPerm
	optPlotComplexityWidthPerMeasure
	optPlotComplexityHeightPerSegment
	optPlotConnectivityWidthPerCircle
	optPlotConnectivityWidthPerBand
	optPlotConnectivityHeightPerMeasure
	optPlotConnectivityCircleTopFraction
	optPlotConnectivityCircleMinLines
	optPlotConnectivityNetworkTopFraction
	optPlotPacPairs
	optPlotConnectivityMeasures
	optPlotSpectralMetrics
	optPlotBurstsMetrics
	optPlotAsymmetryStat
	optPlotTemporalTimeBins
	optPlotTemporalTimeLabels
	// Plotting comparisons (global)
	optPlotCompareWindows
	optPlotComparisonWindows
	optPlotCompareColumns
	optPlotComparisonSegment
	optPlotComparisonColumn
	optPlotComparisonValues
	optPlotComparisonLabels
	optPlotComparisonROIs
	optPlotOverwrite
	// TFR parameters (features pipeline)
	optFeatGroupTFR
	optTfrFreqMin
	optTfrFreqMax
	optTfrNFreqs
	optTfrMinCycles
	optTfrNCyclesFactor
	optTfrWorkers
	// Band envelope + IAF (features pipeline)
	optBandEnvelopePadSec
	optBandEnvelopePadCycles
	optIAFEnabled
	optIAFAlphaWidthHz
	optIAFSearchRangeMin
	optIAFSearchRangeMax
	optIAFMinProminence
	optIAFRois
	optIAFMinCyclesAtFmin
	optIAFMinBaselineSec
	optIAFAllowFullFallback
	optIAFAllowAllChannelsFallback
	// Machine Learning model hyperparameters
	optElasticNetAlphaGrid
	optElasticNetL1RatioGrid
	optRidgeAlphaGrid
	optRfNEstimators
	optRfMaxDepthGrid
	optVarianceThresholdGrid
	// fMRI preprocessing group headers (collapsible sections)
	optFmriGroupRuntime
	optFmriGroupOutput
	optFmriGroupPerformance
	optFmriGroupAnatomical
	optFmriGroupBold
	optFmriGroupQc
	optFmriGroupDenoising
	optFmriGroupSurface
	optFmriGroupMultiecho
	optFmriGroupRepro
	optFmriGroupValidation
	optFmriGroupAdvanced
	// fMRI preprocessing (fMRIPrep-style)
	optFmriEngine
	optFmriFmriprepImage
	optFmriFmriprepOutputDir
	optFmriFmriprepWorkDir
	optFmriFreesurferLicenseFile
	optFmriFreesurferSubjectsDir
	optFmriOutputSpaces
	optFmriIgnore
	optFmriBidsFilterFile
	optFmriUseAroma
	optFmriSkipBidsValidation
	optFmriStopOnFirstCrash
	optFmriCleanWorkdir
	optFmriSkipReconstruction
	optFmriMemMb
	optFmriExtraArgs
	// Additional fMRIPrep options
	optFmriNThreads
	optFmriOmpNThreads
	optFmriLowMem
	optFmriLongitudinal
	optFmriCiftiOutput
	optFmriSkullStripTemplate
	optFmriSkullStripFixedSeed
	optFmriRandomSeed
	optFmriDummyScans
	optFmriBold2T1wInit
	optFmriBold2T1wDof
	optFmriSliceTimeRef
	optFmriFdSpikeThreshold
	optFmriDvarsSpikeThreshold
	optFmriMeOutputEchos
	optFmriMedialSurfaceNan
	optFmriNoMsm
	optFmriLevel
	optFmriTaskId
	// fMRI analysis (first-level GLM + contrasts) group headers
	optFmriAnalysisGroupInput
	optFmriAnalysisGroupContrast
	optFmriAnalysisGroupGLM
	optFmriAnalysisGroupConfounds
	optFmriAnalysisGroupOutput
	optFmriAnalysisGroupPlotting
	// fMRI analysis options
	optFmriAnalysisInputSource
	optFmriAnalysisFmriprepSpace
	optFmriAnalysisRequireFmriprep
	optFmriAnalysisRuns
	optFmriAnalysisContrastType
	optFmriAnalysisCondAColumn
	optFmriAnalysisCondAValue
	optFmriAnalysisCondBColumn
	optFmriAnalysisCondBValue
	optFmriAnalysisContrastName
	optFmriAnalysisFormula
	optFmriAnalysisHrfModel
	optFmriAnalysisDriftModel
	optFmriAnalysisHighPassHz
	optFmriAnalysisLowPassHz
	optFmriAnalysisSmoothingFwhm
	optFmriAnalysisEventsToModel
	optFmriAnalysisScopeTrialTypes
	optFmriAnalysisStimPhasesToModel
	optFmriAnalysisConfoundsStrategy
	optFmriAnalysisWriteDesignMatrix
	optFmriAnalysisOutputType
	optFmriAnalysisOutputDir
	optFmriAnalysisResampleToFS
	optFmriAnalysisFreesurferDir
	// fMRI analysis plotting/report options
	optFmriAnalysisPlotsEnabled
	optFmriAnalysisPlotHTML
	optFmriAnalysisPlotSpace
	optFmriAnalysisPlotThresholdMode
	optFmriAnalysisPlotZThreshold
	optFmriAnalysisPlotFdrQ
	optFmriAnalysisPlotClusterMinVoxels
	optFmriAnalysisPlotVmaxMode
	optFmriAnalysisPlotVmaxManual
	optFmriAnalysisPlotIncludeUnthresholded
	optFmriAnalysisPlotFormatPNG
	optFmriAnalysisPlotFormatSVG
	optFmriAnalysisPlotTypeSlices
	optFmriAnalysisPlotTypeGlass
	optFmriAnalysisPlotTypeHist
	optFmriAnalysisPlotTypeClusters
	optFmriAnalysisPlotEffectSize
	optFmriAnalysisPlotStandardError
	optFmriAnalysisPlotMotionQC
	optFmriAnalysisPlotCarpetQC
	optFmriAnalysisPlotTSNRQC
	optFmriAnalysisPlotDesignQC
	optFmriAnalysisPlotEmbedImages
	optFmriAnalysisPlotSignatures
	optFmriAnalysisSignatureDir
	// fMRI analysis trial-wise signatures options
	optFmriTrialSigGroup
	optFmriTrialSigMethod
	optFmriTrialSigIncludeOtherEvents
	optFmriTrialSigMaxTrialsPerRun
	optFmriTrialSigFixedEffectsWeighting
	optFmriTrialSigWriteTrialBetas
	optFmriTrialSigWriteTrialVariances
	optFmriTrialSigWriteConditionBetas
	optFmriTrialSigSignatureNPS
	optFmriTrialSigSignatureSIIPS1
	optFmriTrialSigLssOtherRegressors
	optFmriTrialSigGroupColumn
	optFmriTrialSigGroupValues
	optFmriTrialSigScopeTrialTypes
	optFmriTrialSigScopeStimPhases
	optFmriTrialSigGroupScope
	// System/global settings
	optSystemNJobs
	optSystemStrictMode
	optLoggingLevel
)

const (
	expandedNone                            = -1
	expandedConnectivityMeasures            = 0
	expandedConditionCompareColumn          = 1
	expandedConditionCompareValues          = 2
	expandedConditionCompareWindows         = 3
	expandedTemporalConditionColumn         = 4
	expandedTemporalConditionValues         = 5
	expandedClusterConditionColumn          = 6
	expandedClusterConditionValues          = 7
	expandedPlotComparisonColumn            = 8
	expandedPlotComparisonValues            = 9
	expandedPlotComparisonWindows           = 10
	expandedPlotComparisonROIs              = 11
	expandedDirectedConnMeasures            = 12
	expandedRunAdjustmentColumn             = 13
	expandedCorrelationsTargetColumn        = 14
	expandedItpcConditionColumn             = 15
	expandedItpcConditionValues             = 16
	expandedFmriCondAColumn                 = 17
	expandedFmriCondAValue                  = 18
	expandedFmriCondBColumn                 = 19
	expandedFmriCondBValue                  = 20
	expandedBehaviorScatterFeatures         = 21
	expandedBehaviorScatterColumns          = 22
	expandedBehaviorScatterAggregation      = 23
	expandedBehaviorScatterSegment          = 24
	expandedTemporalTargetColumn            = 25
	expandedTemporalTopomapsFeatureDir      = 26
	expandedConnConditionColumn             = 27
	expandedConnConditionValues             = 28
	expandedMLTargetColumn                  = 29
	expandedFmriAnalysisCondAColumn         = 30
	expandedFmriAnalysisCondAValue          = 31
	expandedFmriAnalysisCondBColumn         = 32
	expandedFmriAnalysisCondBValue          = 33
	expandedFmriAnalysisStimPhases          = 34
	expandedFmriTrialSigGroupColumn         = 35
	expandedFmriTrialSigGroupValues         = 36
	expandedSourceLocFmriStimPhases         = 37
	expandedPainResidualCrossfitGroupColumn = 38
	expandedStabilityGroupColumn            = 39
	expandedFmriTrialSigStimPhases          = 40
	expandedFmriAnalysisScopeTrialTypes     = 41
	expandedFmriTrialSigScopeTrialTypes     = 42
	expandedSourceLocFmriScopeTrialTypes    = 43
)

// getFeaturesOptions returns the active advanced options for the features pipeline
