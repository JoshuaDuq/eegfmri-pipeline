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

	case expandedRunAdjustmentColumn:
		return len(m.GetAvailableColumns())
	case expandedBehaviorOutcomeColumn:
		return len(m.GetAvailableColumns()) + 1 // +1 for "(default)" option
	case expandedBehaviorPredictorColumn:
		return len(m.GetAvailableColumns()) + 1 // +1 for "(default)" option
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
	case expandedSourceLocContrastColumn:
		return len(m.GetAvailableColumns())
	case expandedSourceLocContrastValueA, expandedSourceLocContrastValueB:
		if strings.TrimSpace(m.sourceLocContrastCondition) == "" {
			return 0
		}
		return len(m.GetDiscoveredColumnValues(m.sourceLocContrastCondition))
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
	case expandedFmriAnalysisScopeColumn:
		n := len(m.fmriDiscoveredColumns)
		if n == 0 {
			return 1
		}
		return n
	case expandedFmriAnalysisStimPhases:
		return len(m.getExpandedListItems())
	case expandedFmriAnalysisScopeTrialTypes:
		return len(m.getExpandedListItems())
	case expandedFmriAnalysisPhaseColumn:
		n := len(m.fmriDiscoveredColumns)
		if n == 0 {
			return 1
		}
		return n
	case expandedFmriAnalysisPhaseScopeColumn:
		n := len(m.fmriDiscoveredColumns)
		if n == 0 {
			return 1
		}
		return n
	case expandedFmriAnalysisPhaseScopeValue:
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
	case expandedFmriTrialSigScopeTrialTypeColumn:
		n := len(m.fmriDiscoveredColumns)
		if n == 0 {
			return 1
		}
		return n
	case expandedFmriTrialSigScopePhaseColumn:
		n := len(m.fmriDiscoveredColumns)
		if n == 0 {
			return 1
		}
		return n
	case expandedFmriTrialSigScopeTrialTypes:
		return len(m.getExpandedListItems())
	case expandedSourceLocFmriStimPhases:
		return len(m.getExpandedListItems())
	case expandedSourceLocFmriScopeTrialTypes:
		return len(m.getExpandedListItems())
	case expandedSourceLocFmriScopeTrialTypeColumn:
		n := len(m.fmriDiscoveredColumns)
		if n == 0 {
			return 1
		}
		return n
	case expandedSourceLocFmriPhaseColumn:
		n := len(m.fmriDiscoveredColumns)
		if n == 0 {
			return 1
		}
		return n
	case expandedSourceLocFmriPhaseScopeColumn:
		n := len(m.fmriDiscoveredColumns)
		if n == 0 {
			return 1
		}
		return n
	case expandedSourceLocFmriPhaseScopeValue:
		return len(m.getExpandedListItems())
	case expandedIAFRois:
		return len(m.rois)
	case expandedItpcConditionValues:
		if m.itpcConditionColumn == "" {
			return 0
		}
		return len(m.GetDiscoveredColumnValues(m.itpcConditionColumn))

	case expandedFmriSecondLevelContrastNames:
		n := len(m.GetFmriSecondLevelDiscoveredContrastNames())
		if n == 0 {
			return 1
		}
		return n
	case expandedFmriSecondLevelSubjectColumn, expandedFmriSecondLevelGroupColumn:
		n := len(m.currentFmriSecondLevelCovariatesColumns())
		if n == 0 {
			return 1
		}
		return n
	case expandedFmriSecondLevelCovariateColumns:
		return len(m.getExpandedListItems())
	case expandedFmriSecondLevelGroupAValue, expandedFmriSecondLevelGroupBValue:
		n := len(m.GetFmriSecondLevelDiscoveredCovariateValues(m.fmriSecondLevelGroupColumn))
		if n == 0 {
			return 1
		}
		return n

	case expandedPredictorResidualCrossfitGroupColumn:
		if len(m.GetAvailableColumns()) == 0 {
			return 2
		}
		return len(m.GetAvailableColumns()) + 1
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

	case expandedRunAdjustmentColumn:
		return m.GetAvailableColumns()
	case expandedBehaviorOutcomeColumn:
		return append([]string{"(default)"}, m.GetAvailableColumns()...)
	case expandedBehaviorPredictorColumn:
		return append([]string{"(default)"}, m.GetAvailableColumns()...)
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
	case expandedSourceLocContrastColumn:
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
	case expandedFmriAnalysisScopeColumn:
		if len(m.fmriDiscoveredColumns) == 0 {
			return []string{"(type manually)"}
		}
		return m.fmriDiscoveredColumns
	case expandedFmriAnalysisPhaseColumn:
		if len(m.fmriDiscoveredColumns) == 0 {
			return []string{"(type manually)"}
		}
		return m.fmriDiscoveredColumns
	case expandedFmriAnalysisPhaseScopeColumn:
		if len(m.fmriDiscoveredColumns) == 0 {
			return []string{"(type manually)"}
		}
		return m.fmriDiscoveredColumns
	case expandedFmriAnalysisPhaseScopeValue:
		items := []string{"(none)"}
		scopeCol := m.resolveFmriConditionColumn(m.fmriAnalysisPhaseScopeColumn)
		vals := m.GetFmriDiscoveredColumnValues(scopeCol)
		if len(vals) == 0 {
			return append(items, "(type manually)")
		}
		return append(items, vals...)
	case expandedFmriAnalysisStimPhases:
		items := []string{"(none)", "(all)"}
		phaseCol := m.resolveFmriPhaseColumn(m.fmriAnalysisPhaseColumn)
		vals := m.GetFmriDiscoveredColumnValues(phaseCol)
		if len(vals) == 0 {
			return append(items, "(type manually)")
		}
		return append(items, vals...)
	case expandedFmriAnalysisScopeTrialTypes:
		items := []string{"(none)"}
		scopeCol := m.resolveFmriConditionColumn(m.fmriAnalysisScopeColumn)
		vals := m.GetFmriDiscoveredColumnValues(scopeCol)
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
		phaseCol := m.resolveFmriPhaseColumn(m.fmriTrialSigScopePhaseColumn)
		vals := m.GetFmriDiscoveredColumnValues(phaseCol)
		if len(vals) == 0 {
			return append(items, "(type manually)")
		}
		return append(items, vals...)
	case expandedFmriTrialSigScopeTrialTypeColumn:
		if len(m.fmriDiscoveredColumns) == 0 {
			return []string{"(type manually)"}
		}
		return m.fmriDiscoveredColumns
	case expandedFmriTrialSigScopePhaseColumn:
		if len(m.fmriDiscoveredColumns) == 0 {
			return []string{"(type manually)"}
		}
		return m.fmriDiscoveredColumns
	case expandedFmriTrialSigScopeTrialTypes:
		items := []string{"(none)"}
		trialScopeCol := m.resolveFmriConditionColumn(m.fmriTrialSigScopeTrialTypeColumn)
		vals := m.GetFmriDiscoveredColumnValues(trialScopeCol)
		if len(vals) == 0 {
			return append(items, "(type manually)")
		}
		return append(items, vals...)
	case expandedSourceLocFmriStimPhases:
		items := []string{"(none)", "(all)"}
		phaseCol := m.resolveFmriPhaseColumn(m.sourceLocFmriPhaseColumn)
		vals := m.GetFmriDiscoveredColumnValues(phaseCol)
		if len(vals) == 0 {
			return append(items, "(type manually)")
		}
		return append(items, vals...)
	case expandedSourceLocFmriScopeTrialTypeColumn:
		if len(m.fmriDiscoveredColumns) == 0 {
			return []string{"(type manually)"}
		}
		return m.fmriDiscoveredColumns
	case expandedSourceLocFmriPhaseColumn:
		if len(m.fmriDiscoveredColumns) == 0 {
			return []string{"(type manually)"}
		}
		return m.fmriDiscoveredColumns
	case expandedSourceLocFmriPhaseScopeColumn:
		if len(m.fmriDiscoveredColumns) == 0 {
			return []string{"(type manually)"}
		}
		return m.fmriDiscoveredColumns
	case expandedSourceLocFmriPhaseScopeValue:
		items := []string{"(none)"}
		scopeCol := m.resolveFmriConditionColumn(m.sourceLocFmriPhaseScopeColumn)
		vals := m.GetFmriDiscoveredColumnValues(scopeCol)
		if len(vals) == 0 {
			return append(items, "(type manually)")
		}
		return append(items, vals...)
	case expandedSourceLocFmriScopeTrialTypes:
		items := []string{"(none)"}
		scopeCol := m.resolveFmriConditionColumn(m.sourceLocFmriConditionScopeColumn)
		vals := m.GetFmriDiscoveredColumnValues(scopeCol)
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
	case expandedSourceLocContrastValueA, expandedSourceLocContrastValueB:
		if strings.TrimSpace(m.sourceLocContrastCondition) == "" {
			return nil
		}
		return m.GetDiscoveredColumnValues(m.sourceLocContrastCondition)

	case expandedFmriSecondLevelContrastNames:
		contrasts := m.GetFmriSecondLevelDiscoveredContrastNames()
		if len(contrasts) == 0 {
			return []string{"(type manually)"}
		}
		return contrasts
	case expandedFmriSecondLevelSubjectColumn, expandedFmriSecondLevelGroupColumn:
		columns := m.currentFmriSecondLevelCovariatesColumns()
		if len(columns) == 0 {
			return []string{"(type manually)"}
		}
		return columns
	case expandedFmriSecondLevelCovariateColumns:
		columns := m.currentFmriSecondLevelCovariatesColumns()
		if len(columns) == 0 {
			return []string{"(none)", "(type manually)"}
		}
		return append([]string{"(none)"}, columns...)
	case expandedFmriSecondLevelGroupAValue, expandedFmriSecondLevelGroupBValue:
		values := m.GetFmriSecondLevelDiscoveredCovariateValues(m.fmriSecondLevelGroupColumn)
		if len(values) == 0 {
			return []string{"(type manually)"}
		}
		return values

	case expandedPredictorResidualCrossfitGroupColumn:
		items := []string{"(default: run column)"}
		cols := m.GetAvailableColumns()
		if len(cols) == 0 {
			return append(items, "(type manually)")
		}
		return append(items, cols...)
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

func (m Model) currentFmriSecondLevelCovariatesColumns() []string {
	columns, _, _ := m.currentFmriSecondLevelCovariatesDiscovery()
	return columns
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

	case expandedItpcConditionValues:
		selectedValues = m.itpcConditionValues
	case expandedConnConditionValues:
		selectedValues = m.connConditionValues
	case expandedFmriTrialSigGroupValues:
		selectedValues = m.fmriTrialSigGroupValuesSpec

	case expandedFmriSecondLevelContrastNames:
		selectedValues = m.fmriSecondLevelContrastNames
	case expandedFmriSecondLevelCovariateColumns:
		selectedValues = m.fmriSecondLevelCovariateColumns

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

	case expandedRunAdjustmentColumn:
		m.runAdjustmentColumn = selectedItem
		m.expandedOption = expandedNone
		m.subCursor = 0

	case expandedBehaviorOutcomeColumn:
		if selectedItem == "(default)" {
			m.behaviorOutcomeColumn = ""
		} else {
			m.behaviorOutcomeColumn = selectedItem
		}
		m.expandedOption = expandedNone
		m.subCursor = 0

	case expandedBehaviorPredictorColumn:
		if selectedItem == "(default)" {
			m.behaviorPredictorColumn = ""
		} else {
			m.behaviorPredictorColumn = selectedItem
		}
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
	case expandedSourceLocContrastColumn:
		m.sourceLocContrastCondition = selectedItem
		m.sourceLocContrastA = ""
		m.sourceLocContrastB = ""
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
	case expandedSourceLocFmriPhaseColumn:
		if selectedItem == "(type manually)" {
			m.expandedOption = expandedNone
			m.subCursor = 0
			m.startTextEdit(textFieldSourceLocFmriPhaseColumn)
		} else {
			m.sourceLocFmriPhaseColumn = selectedItem
			m.sourceLocFmriStimPhasesToModel = ""
			m.expandedOption = expandedNone
			m.subCursor = 0
		}
	case expandedSourceLocFmriPhaseScopeColumn:
		if selectedItem == "(type manually)" {
			m.expandedOption = expandedNone
			m.subCursor = 0
			m.startTextEdit(textFieldSourceLocFmriPhaseScopeColumn)
		} else {
			m.sourceLocFmriPhaseScopeColumn = selectedItem
			m.sourceLocFmriPhaseScopeValue = ""
			m.expandedOption = expandedNone
			m.subCursor = 0
		}
	case expandedSourceLocFmriPhaseScopeValue:
		switch selectedItem {
		case "(none)":
			m.sourceLocFmriPhaseScopeValue = ""
			m.expandedOption = expandedNone
			m.subCursor = 0
		case "(type manually)":
			m.expandedOption = expandedNone
			m.subCursor = 0
			m.startTextEdit(textFieldSourceLocFmriPhaseScopeValue)
		default:
			m.sourceLocFmriPhaseScopeValue = selectedItem
			m.expandedOption = expandedNone
			m.subCursor = 0
		}
	case expandedSourceLocFmriScopeTrialTypeColumn:
		if selectedItem == "(type manually)" {
			m.expandedOption = expandedNone
			m.subCursor = 0
			m.startTextEdit(textFieldSourceLocFmriConditionScopeColumn)
		} else {
			m.sourceLocFmriConditionScopeColumn = selectedItem
			m.sourceLocFmriConditionScopeTrialTypes = ""
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
	case expandedFmriAnalysisScopeColumn:
		if selectedItem == "(type manually)" {
			m.expandedOption = expandedNone
			m.subCursor = 0
			m.startTextEdit(textFieldFmriAnalysisScopeColumn)
		} else {
			m.fmriAnalysisScopeColumn = selectedItem
			m.fmriAnalysisScopeTrialTypes = ""
			m.expandedOption = expandedNone
			m.subCursor = 0
		}
	case expandedFmriAnalysisPhaseColumn:
		if selectedItem == "(type manually)" {
			m.expandedOption = expandedNone
			m.subCursor = 0
			m.startTextEdit(textFieldFmriAnalysisPhaseColumn)
		} else {
			m.fmriAnalysisPhaseColumn = selectedItem
			m.fmriAnalysisStimPhasesToModel = ""
			m.expandedOption = expandedNone
			m.subCursor = 0
		}
	case expandedFmriAnalysisPhaseScopeColumn:
		if selectedItem == "(type manually)" {
			m.expandedOption = expandedNone
			m.subCursor = 0
			m.startTextEdit(textFieldFmriAnalysisPhaseScopeColumn)
		} else {
			m.fmriAnalysisPhaseScopeColumn = selectedItem
			m.fmriAnalysisPhaseScopeValue = ""
			m.expandedOption = expandedNone
			m.subCursor = 0
		}
	case expandedFmriAnalysisPhaseScopeValue:
		switch selectedItem {
		case "(none)":
			m.fmriAnalysisPhaseScopeValue = ""
			m.expandedOption = expandedNone
			m.subCursor = 0
		case "(type manually)":
			m.expandedOption = expandedNone
			m.subCursor = 0
			m.startTextEdit(textFieldFmriAnalysisPhaseScopeValue)
		default:
			m.fmriAnalysisPhaseScopeValue = selectedItem
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
	case expandedFmriTrialSigScopeTrialTypeColumn:
		if selectedItem == "(type manually)" {
			m.expandedOption = expandedNone
			m.subCursor = 0
			m.startTextEdit(textFieldFmriTrialSigScopeTrialTypeColumn)
		} else {
			m.fmriTrialSigScopeTrialTypeColumn = selectedItem
			m.fmriTrialSigScopeTrialTypes = ""
			m.expandedOption = expandedNone
			m.subCursor = 0
		}
	case expandedFmriTrialSigScopePhaseColumn:
		if selectedItem == "(type manually)" {
			m.expandedOption = expandedNone
			m.subCursor = 0
			m.startTextEdit(textFieldFmriTrialSigScopePhaseColumn)
		} else {
			m.fmriTrialSigScopePhaseColumn = selectedItem
			m.fmriTrialSigScopeStimPhases = ""
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
	case expandedSourceLocContrastValueA:
		m.sourceLocContrastA = selectedItem
		m.expandedOption = expandedNone
		m.subCursor = 0
	case expandedSourceLocContrastValueB:
		m.sourceLocContrastB = selectedItem
		m.expandedOption = expandedNone
		m.subCursor = 0

	case expandedFmriSecondLevelContrastNames:
		if selectedItem == "(type manually)" {
			m.expandedOption = expandedNone
			m.subCursor = 0
			m.startTextEdit(textFieldFmriSecondLevelContrastNames)
		} else {
			m.toggleSpaceValue(selectedItem, &m.fmriSecondLevelContrastNames)
		}
	case expandedFmriSecondLevelSubjectColumn:
		if selectedItem == "(type manually)" {
			m.expandedOption = expandedNone
			m.subCursor = 0
			m.startTextEdit(textFieldFmriSecondLevelSubjectColumn)
		} else {
			m.fmriSecondLevelSubjectColumn = selectedItem
			m.expandedOption = expandedNone
			m.subCursor = 0
		}
	case expandedFmriSecondLevelCovariateColumns:
		switch selectedItem {
		case "(none)":
			m.fmriSecondLevelCovariateColumns = ""
			m.expandedOption = expandedNone
			m.subCursor = 0
		case "(type manually)":
			m.expandedOption = expandedNone
			m.subCursor = 0
			m.startTextEdit(textFieldFmriSecondLevelCovariateColumns)
		default:
			m.toggleSpaceValue(selectedItem, &m.fmriSecondLevelCovariateColumns)
		}
	case expandedFmriSecondLevelGroupColumn:
		if selectedItem == "(type manually)" {
			m.expandedOption = expandedNone
			m.subCursor = 0
			m.startTextEdit(textFieldFmriSecondLevelGroupColumn)
		} else {
			m.fmriSecondLevelGroupColumn = selectedItem
			m.fmriSecondLevelGroupAValue = ""
			m.fmriSecondLevelGroupBValue = ""
			m.expandedOption = expandedNone
			m.subCursor = 0
		}
	case expandedFmriSecondLevelGroupAValue:
		if selectedItem == "(type manually)" {
			m.expandedOption = expandedNone
			m.subCursor = 0
			m.startTextEdit(textFieldFmriSecondLevelGroupAValue)
		} else if strings.TrimSpace(m.fmriSecondLevelGroupBValue) == selectedItem {
			m.ShowToast("Group A and Group B must be different", "warning")
		} else {
			m.fmriSecondLevelGroupAValue = selectedItem
			m.expandedOption = expandedNone
			m.subCursor = 0
		}
	case expandedFmriSecondLevelGroupBValue:
		if selectedItem == "(type manually)" {
			m.expandedOption = expandedNone
			m.subCursor = 0
			m.startTextEdit(textFieldFmriSecondLevelGroupBValue)
		} else if strings.TrimSpace(m.fmriSecondLevelGroupAValue) == selectedItem {
			m.ShowToast("Group A and Group B must be different", "warning")
		} else {
			m.fmriSecondLevelGroupBValue = selectedItem
			m.expandedOption = expandedNone
			m.subCursor = 0
		}

	case expandedPredictorResidualCrossfitGroupColumn:
		switch selectedItem {
		case "(default: run column)":
			m.predictorResidualCrossfitGroupColumn = ""
			m.expandedOption = expandedNone
			m.subCursor = 0
		case "(type manually)":
			m.expandedOption = expandedNone
			m.subCursor = 0
			m.startTextEdit(textFieldPredictorResidualCrossfitGroupColumn)
		default:
			m.predictorResidualCrossfitGroupColumn = selectedItem
			m.expandedOption = expandedNone
			m.subCursor = 0
		}
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
	case expandedTemporalConditionColumn:
		return opt == optTemporalConditionColumn
	case expandedTemporalConditionValues:
		return opt == optTemporalConditionValues
	case expandedClusterConditionColumn:
		return opt == optClusterConditionColumn
	case expandedClusterConditionValues:
		return opt == optClusterConditionValues

	case expandedRunAdjustmentColumn:
		return opt == optRunAdjustmentColumn
	case expandedBehaviorOutcomeColumn:
		return opt == optBehaviorOutcomeColumn
	case expandedBehaviorPredictorColumn:
		return opt == optBehaviorPredictorColumn
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
	case expandedSourceLocContrastColumn:
		return opt == optSourceLocContrastConditionColumn
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
	case expandedFmriAnalysisScopeColumn:
		return opt == optFmriAnalysisScopeColumn
	case expandedFmriAnalysisPhaseColumn:
		return opt == optFmriAnalysisPhaseColumn
	case expandedFmriAnalysisPhaseScopeColumn:
		return opt == optFmriAnalysisPhaseScopeColumn
	case expandedFmriAnalysisPhaseScopeValue:
		return opt == optFmriAnalysisPhaseScopeValue
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
	case expandedFmriTrialSigScopeTrialTypeColumn:
		return opt == optFmriTrialSigScopeTrialTypeColumn
	case expandedFmriTrialSigScopePhaseColumn:
		return opt == optFmriTrialSigScopePhaseColumn
	case expandedFmriTrialSigScopeTrialTypes:
		return opt == optFmriTrialSigScopeTrialTypes
	case expandedItpcConditionValues:
		return opt == optItpcConditionValues
	case expandedConnConditionValues:
		return opt == optConnConditionValues
	case expandedSourceLocContrastValueA:
		return opt == optSourceLocContrastConditionA
	case expandedSourceLocContrastValueB:
		return opt == optSourceLocContrastConditionB
	case expandedSourceLocFmriStimPhases:
		return opt == optSourceLocFmriStimPhasesToModel
	case expandedSourceLocFmriPhaseColumn:
		return opt == optSourceLocFmriPhaseColumn
	case expandedSourceLocFmriPhaseScopeColumn:
		return opt == optSourceLocFmriPhaseScopeColumn
	case expandedSourceLocFmriPhaseScopeValue:
		return opt == optSourceLocFmriPhaseScopeValue
	case expandedSourceLocFmriScopeTrialTypeColumn:
		return opt == optSourceLocFmriConditionScopeColumn
	case expandedSourceLocFmriScopeTrialTypes:
		return opt == optSourceLocFmriConditionScopeTrialTypes
	case expandedIAFRois:
		return opt == optIAFRois
	case expandedFmriSecondLevelContrastNames:
		return opt == optFmriSecondLevelContrastNames
	case expandedFmriSecondLevelSubjectColumn:
		return opt == optFmriSecondLevelSubjectColumn
	case expandedFmriSecondLevelCovariateColumns:
		return opt == optFmriSecondLevelCovariateColumns
	case expandedFmriSecondLevelGroupColumn:
		return opt == optFmriSecondLevelGroupColumn
	case expandedFmriSecondLevelGroupAValue:
		return opt == optFmriSecondLevelGroupAValue
	case expandedFmriSecondLevelGroupBValue:
		return opt == optFmriSecondLevelGroupBValue
	case expandedPredictorResidualCrossfitGroupColumn:
		return opt == optPredictorResidualCrossfitGroupColumn
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

	case expandedRunAdjustmentColumn:
		return m.runAdjustmentColumn == item
	case expandedBehaviorOutcomeColumn:
		if item == "(default)" {
			return strings.TrimSpace(m.behaviorOutcomeColumn) == ""
		}
		return m.behaviorOutcomeColumn == item
	case expandedBehaviorPredictorColumn:
		if item == "(default)" {
			return strings.TrimSpace(m.behaviorPredictorColumn) == ""
		}
		return m.behaviorPredictorColumn == item
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
	case expandedSourceLocContrastColumn:
		return m.sourceLocContrastCondition == item
	case expandedFmriCondAColumn:
		return m.sourceLocFmriCondAColumn == item
	case expandedFmriCondAValue:
		return m.sourceLocFmriCondAValue == item
	case expandedFmriCondBColumn:
		return m.sourceLocFmriCondBColumn == item
	case expandedFmriCondBValue:
		return m.sourceLocFmriCondBValue == item
	case expandedSourceLocContrastValueA:
		return m.sourceLocContrastA == item
	case expandedSourceLocContrastValueB:
		return m.sourceLocContrastB == item
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
	case expandedSourceLocFmriPhaseColumn:
		return m.sourceLocFmriPhaseColumn == item
	case expandedSourceLocFmriPhaseScopeColumn:
		return m.sourceLocFmriPhaseScopeColumn == item
	case expandedSourceLocFmriPhaseScopeValue:
		if item == "(none)" {
			return strings.TrimSpace(m.sourceLocFmriPhaseScopeValue) == ""
		}
		return strings.TrimSpace(m.sourceLocFmriPhaseScopeValue) == item
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
	case expandedSourceLocFmriScopeTrialTypeColumn:
		return m.sourceLocFmriConditionScopeColumn == item
	case expandedFmriAnalysisCondAColumn:
		return m.fmriAnalysisCondAColumn == item
	case expandedFmriAnalysisCondAValue:
		return m.fmriAnalysisCondAValue == item
	case expandedFmriAnalysisCondBColumn:
		return m.fmriAnalysisCondBColumn == item
	case expandedFmriAnalysisCondBValue:
		return m.fmriAnalysisCondBValue == item
	case expandedFmriAnalysisScopeColumn:
		return m.fmriAnalysisScopeColumn == item
	case expandedFmriAnalysisPhaseColumn:
		return m.fmriAnalysisPhaseColumn == item
	case expandedFmriAnalysisPhaseScopeColumn:
		return m.fmriAnalysisPhaseScopeColumn == item
	case expandedFmriAnalysisPhaseScopeValue:
		if item == "(none)" {
			return strings.TrimSpace(m.fmriAnalysisPhaseScopeValue) == ""
		}
		return strings.TrimSpace(m.fmriAnalysisPhaseScopeValue) == item
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
	case expandedFmriTrialSigScopeTrialTypeColumn:
		return m.fmriTrialSigScopeTrialTypeColumn == item
	case expandedFmriTrialSigScopePhaseColumn:
		return m.fmriTrialSigScopePhaseColumn == item
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
		expandedPlotComparisonWindows, expandedItpcConditionValues, expandedConnConditionValues,
		expandedFmriTrialSigGroupValues, expandedDoseResponseBands, expandedDoseResponseROIs, expandedDoseResponseScopes, expandedDoseResponseStat:
		return m.isColumnValueSelected(item)

	case expandedFmriSecondLevelContrastNames:
		for _, contrast := range splitSpaceList(m.fmriSecondLevelContrastNames) {
			if contrast == item {
				return true
			}
		}
		return false
	case expandedFmriSecondLevelSubjectColumn:
		return m.fmriSecondLevelSubjectColumn == item
	case expandedFmriSecondLevelCovariateColumns:
		if item == "(none)" {
			return strings.TrimSpace(m.fmriSecondLevelCovariateColumns) == ""
		}
		for _, column := range splitSpaceList(m.fmriSecondLevelCovariateColumns) {
			if column == item {
				return true
			}
		}
		return false
	case expandedFmriSecondLevelGroupColumn:
		return m.fmriSecondLevelGroupColumn == item
	case expandedFmriSecondLevelGroupAValue:
		return m.fmriSecondLevelGroupAValue == item
	case expandedFmriSecondLevelGroupBValue:
		return m.fmriSecondLevelGroupBValue == item
	case expandedPredictorResidualCrossfitGroupColumn:
		if item == "(default: run column)" {
			return strings.TrimSpace(m.predictorResidualCrossfitGroupColumn) == ""
		}
		if item == "(type manually)" {
			return false
		}
		return m.predictorResidualCrossfitGroupColumn == item
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

///////////////////////////////////////////////////////////////////
