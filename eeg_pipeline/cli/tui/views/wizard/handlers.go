package wizard

import (
	"fmt"
	"strconv"

	"github.com/eeg-pipeline/tui/styles"
	"github.com/eeg-pipeline/tui/types"

	tea "github.com/charmbracelet/bubbletea"
)

///////////////////////////////////////////////////////////////////
// Cursor Reset Helper
///////////////////////////////////////////////////////////////////

// resetCursorsForStep resets all cursor positions when entering a new step
// to prevent UI state from persisting incorrectly between steps
func (m *Model) resetCursorsForStep() {
	// Reset all step-specific cursors to 0
	m.categoryIndex = 0
	m.subjectCursor = 0
	m.computationCursor = 0
	m.bandCursor = 0
	m.spatialCursor = 0
	m.featureFileCursor = 0
	m.tfrROICursor = 0
	m.advancedCursor = 0
	m.subCursor = 0
	m.expandedOption = -1

	// Reset any editing states
	m.filteringSubject = false
	m.subjectFilter = ""
	m.editingNumber = false
	m.numberBuffer = ""
	m.editingRangeIdx = -1
	m.editingField = 0
	m.editingTfrChans = false
}

///////////////////////////////////////////////////////////////////
// Navigation Handlers
///////////////////////////////////////////////////////////////////

func (m *Model) handleUp() {
	switch m.CurrentStep {
	case types.StepSelectMode:
		if m.modeIndex > 0 {
			m.modeIndex--
		} else {
			m.modeIndex = len(m.modeOptions) - 1
		}
	case types.StepSelectComputations:
		if m.computationCursor > 0 {
			m.computationCursor--
		} else {
			m.computationCursor = len(m.computations) - 1
		}
	case types.StepTFRVizType:
		if m.tfrVizType > 0 {
			m.tfrVizType--
		} else {
			m.tfrVizType = len(m.tfrVizTypes) - 1
		}
	case types.StepConfigureOptions:
		// Features pipeline category selection
		if m.categoryIndex > 0 {
			m.categoryIndex--
		} else {
			m.categoryIndex = len(m.categories) - 1
		}
	case types.StepSelectSubjects:
		if m.subjectCursor > 0 {
			m.subjectCursor--
		} else if len(m.subjects) > 0 {
			m.subjectCursor = len(m.subjects) - 1
		}
	case types.StepSelectBands:
		if m.bandCursor > 0 {
			m.bandCursor--
		} else {
			m.bandCursor = len(m.bands) - 1
		}
	case types.StepSelectFeatureFiles:
		if m.featureFileCursor > 0 {
			m.featureFileCursor--
		} else if len(m.featureFiles) > 0 {
			m.featureFileCursor = len(m.featureFiles) - 1
		}
	case types.StepSelectSpatial:
		if m.spatialCursor > 0 {
			m.spatialCursor--
		} else {
			m.spatialCursor = len(spatialModes) - 1
		}
	case types.StepTFRChannels:
		if !m.editingTfrChans {
			// If in ROI mode, navigate ROIs; otherwise navigate modes
			if m.tfrChannelMode == 0 && len(m.tfrROIs) > 0 {
				// Navigate within ROI list
				if m.tfrROICursor > 0 {
					m.tfrROICursor--
				} else {
					m.tfrROICursor = len(m.tfrROIs) - 1
				}
			} else {
				// Navigate channel modes
				if m.tfrChannelMode > 0 {
					m.tfrChannelMode--
				} else {
					m.tfrChannelMode = len(m.tfrChannelModes) - 1
				}
			}
		}
	case types.StepTimeRange:
		if m.editingRangeIdx >= 0 {
			if m.editingField > 0 {
				m.editingField--
			} else {
				m.editingField = 2
			}
		} else {
			if m.timeRangeCursor > 0 {
				m.timeRangeCursor--
			} else if len(m.TimeRanges) > 0 {
				m.timeRangeCursor = len(m.TimeRanges) - 1
			}
		}
	case types.StepAdvancedConfig:
		// Check if an expandable option is open
		if m.expandedOption >= 0 {
			// Navigate within the expanded list (connectivity measures)
			if m.expandedOption == 4 { // Connectivity measures
				if m.subCursor > 0 {
					m.subCursor--
				} else {
					m.subCursor = len(connectivityMeasures) - 1
				}
			}
		} else {
			// Navigate between main options
			optCount := m.getAdvancedOptionCount()
			if m.advancedCursor > 0 {
				m.advancedCursor--
			} else {
				m.advancedCursor = optCount - 1
			}
		}
	}
}

func (m *Model) handleDown() {
	switch m.CurrentStep {
	case types.StepSelectMode:
		if m.modeIndex < len(m.modeOptions)-1 {
			m.modeIndex++
		} else {
			m.modeIndex = 0
		}
	case types.StepSelectComputations:
		if m.computationCursor < len(m.computations)-1 {
			m.computationCursor++
		} else {
			m.computationCursor = 0
		}
	case types.StepTFRVizType:
		if m.tfrVizType < len(m.tfrVizTypes)-1 {
			m.tfrVizType++
		} else {
			m.tfrVizType = 0
		}
	case types.StepConfigureOptions:
		// Features pipeline category selection
		if m.categoryIndex < len(m.categories)-1 {
			m.categoryIndex++
		} else {
			m.categoryIndex = 0
		}
	case types.StepSelectSubjects:
		if m.subjectCursor < len(m.subjects)-1 {
			m.subjectCursor++
		} else {
			m.subjectCursor = 0
		}
	case types.StepSelectBands:
		if m.bandCursor < len(m.bands)-1 {
			m.bandCursor++
		} else {
			m.bandCursor = 0
		}
	case types.StepSelectFeatureFiles:
		if m.featureFileCursor < len(m.featureFiles)-1 {
			m.featureFileCursor++
		} else {
			m.featureFileCursor = 0
		}
	case types.StepSelectSpatial:
		if m.spatialCursor < len(spatialModes)-1 {
			m.spatialCursor++
		} else {
			m.spatialCursor = 0
		}
	case types.StepTFRChannels:
		if !m.editingTfrChans {
			// If in ROI mode, navigate ROIs; otherwise navigate modes
			if m.tfrChannelMode == 0 && len(m.tfrROIs) > 0 {
				// Navigate within ROI list
				if m.tfrROICursor < len(m.tfrROIs)-1 {
					m.tfrROICursor++
				} else {
					m.tfrROICursor = 0
				}
			} else {
				// Navigate channel modes
				if m.tfrChannelMode < len(m.tfrChannelModes)-1 {
					m.tfrChannelMode++
				} else {
					m.tfrChannelMode = 0
				}
			}
		}
	case types.StepTimeRange:
		if m.editingRangeIdx >= 0 {
			if m.editingField < 2 {
				m.editingField++
			} else {
				m.editingField = 0
			}
		} else {
			if m.timeRangeCursor < len(m.TimeRanges)-1 {
				m.timeRangeCursor++
			} else {
				m.timeRangeCursor = 0
			}
		}
	case types.StepAdvancedConfig:
		// Check if an expandable option is open
		if m.expandedOption >= 0 {
			// Navigate within the expanded list (connectivity measures)
			if m.expandedOption == 4 { // Connectivity measures
				if m.subCursor < len(connectivityMeasures)-1 {
					m.subCursor++
				} else {
					m.subCursor = 0
				}
			}
		} else {
			// Navigate between main options
			optCount := m.getAdvancedOptionCount()
			if m.advancedCursor < optCount-1 {
				m.advancedCursor++
			} else {
				m.advancedCursor = 0
			}
		}
	}
}

func (m *Model) handleLeft() {
	// TFR: switch to previous channel mode
	if m.CurrentStep == types.StepTFRChannels && !m.editingTfrChans {
		if m.tfrChannelMode > 0 {
			m.tfrChannelMode--
		} else {
			m.tfrChannelMode = len(m.tfrChannelModes) - 1
		}
	}
}

func (m *Model) handleRight() {
	// TFR: switch to next channel mode
	if m.CurrentStep == types.StepTFRChannels && !m.editingTfrChans {
		if m.tfrChannelMode < len(m.tfrChannelModes)-1 {
			m.tfrChannelMode++
		} else {
			m.tfrChannelMode = 0
		}
	}
}

func (m Model) handleEnter() (tea.Model, tea.Cmd) {
	if m.CurrentStep == types.StepReviewExecute {
		if len(m.validationErrors) > 0 {
			return m, nil
		}
		m.ConfirmingExecute = true
		return m, nil
	}

	// For visualize mode, skip directly to subjects (most compute-only steps are irrelevant)
	if m.CurrentStep == types.StepSelectMode {
		isVisualize := len(m.modeOptions) > 0 && m.modeOptions[m.modeIndex] == styles.ModeVisualize
		if isVisualize && (m.Pipeline == types.PipelineBehavior || m.Pipeline == types.PipelineFeatures) {
			for i, step := range m.steps {
				if step == types.StepSelectSubjects {
					m.stepIndex = i
					m.CurrentStep = step
					m.resetCursorsForStep() // Reset cursors when jumping to step
					return m, tea.ClearScreen
				}
			}
		}
	}

	if m.stepIndex < len(m.steps)-1 {
		m.stepIndex++
		m.CurrentStep = m.steps[m.stepIndex]
		m.resetCursorsForStep() // Reset cursors when entering new step

		// Skip advanced config step if in visualize mode (compute-only options)
		if m.CurrentStep == types.StepAdvancedConfig {
			isVisualize := len(m.modeOptions) > 0 && m.modeOptions[m.modeIndex] == styles.ModeVisualize
			if isVisualize {
				// Skip to next step
				if m.stepIndex < len(m.steps)-1 {
					m.stepIndex++
					m.CurrentStep = m.steps[m.stepIndex]
					m.resetCursorsForStep() // Reset again for the skipped-to step
				}
			}
		}

		if m.CurrentStep == types.StepReviewExecute {
			m.validationErrors = m.validate()
		}
	}
	return m, tea.ClearScreen
}

func (m *Model) handleSpace() {
	switch m.CurrentStep {
	case types.StepSelectComputations:
		m.computationSelected[m.computationCursor] = !m.computationSelected[m.computationCursor]
	case types.StepConfigureOptions:
		// Features pipeline category selection
		m.selected[m.categoryIndex] = !m.selected[m.categoryIndex]
	case types.StepSelectSubjects:
		if m.subjectCursor < len(m.subjects) {
			subj := m.subjects[m.subjectCursor].ID
			m.subjectSelected[subj] = !m.subjectSelected[subj]
			m.updateFeatureAvailability() // Recalculate based on new selection
		}
	case types.StepSelectBands:
		m.bandSelected[m.bandCursor] = !m.bandSelected[m.bandCursor]
	case types.StepSelectFeatureFiles:
		if m.featureFileCursor < len(m.featureFiles) {
			key := m.featureFiles[m.featureFileCursor].Key
			m.featureFileSelected[key] = !m.featureFileSelected[key]
		}
	case types.StepSelectSpatial:
		m.spatialSelected[m.spatialCursor] = !m.spatialSelected[m.spatialCursor]
	case types.StepTFRChannels:
		// In ROI mode, toggle selected ROI
		if m.tfrChannelMode == 0 && m.tfrROICursor < len(m.tfrROIs) {
			roi := m.tfrROIs[m.tfrROICursor]
			m.tfrROISelected[roi] = !m.tfrROISelected[roi]
		} else if m.tfrChannelMode == 3 {
			// Specific channels mode: toggle editing
			m.editingTfrChans = !m.editingTfrChans
		}
	case types.StepTimeRange:
		if m.editingRangeIdx >= 0 {
			// Commit and move to next field, or exit if at end
			if m.editingField < 2 {
				m.editingField++
			} else {
				m.editingRangeIdx = -1
				m.editingField = 0
			}
		} else {
			m.editingRangeIdx = m.timeRangeCursor
			m.editingField = 1 // Start at Tmin
		}
	case types.StepAdvancedConfig:
		m.toggleAdvancedOption()
	}
}

func (m *Model) selectAll() {
	switch m.CurrentStep {
	case types.StepSelectComputations:
		for i := range m.computations {
			m.computationSelected[i] = true
		}
	case types.StepConfigureOptions:
		// Features pipeline category selection
		for i := range m.categories {
			m.selected[i] = true
		}
	case types.StepSelectSubjects:
		for _, s := range m.subjects {
			m.subjectSelected[s.ID] = true
		}
		m.updateFeatureAvailability()
	case types.StepSelectBands:
		for i := range m.bands {
			m.bandSelected[i] = true
		}
	case types.StepSelectSpatial:
		for i := range spatialModes {
			m.spatialSelected[i] = true
		}
	case types.StepSelectFeatureFiles:
		for _, f := range m.featureFiles {
			m.featureFileSelected[f.Key] = true
		}
	case types.StepTFRChannels:
		// Select all ROIs when in ROI mode
		if m.tfrChannelMode == 0 {
			for _, roi := range m.tfrROIs {
				m.tfrROISelected[roi] = true
			}
		}
	}
}

func (m *Model) selectNone() {
	switch m.CurrentStep {
	case types.StepSelectComputations:
		m.computationSelected = make(map[int]bool)
	case types.StepConfigureOptions:
		// Features pipeline category selection
		m.selected = make(map[int]bool)
	case types.StepSelectSubjects:
		m.subjectSelected = make(map[string]bool)
		m.updateFeatureAvailability()
	case types.StepSelectBands:
		m.bandSelected = make(map[int]bool)
	case types.StepSelectSpatial:
		m.spatialSelected = make(map[int]bool)
	case types.StepSelectFeatureFiles:
		m.featureFileSelected = make(map[string]bool)
	case types.StepTFRChannels:
		// Clear all ROIs when in ROI mode
		if m.tfrChannelMode == 0 {
			m.tfrROISelected = make(map[string]bool)
		}
	}
}

func (m *Model) GoBack() bool {
	if m.ConfirmingExecute {
		m.ConfirmingExecute = false
		return true
	}

	// If in advanced config with an expanded option, collapse it first
	if m.CurrentStep == types.StepAdvancedConfig && m.expandedOption >= 0 {
		m.expandedOption = -1
		m.subCursor = 0
		return true
	}

	if m.stepIndex > 0 {
		if m.CurrentStep == types.StepReviewExecute {
			m.validationErrors = nil
		}

		// Clear subject filter when leaving subject step
		if m.CurrentStep == types.StepSelectSubjects {
			m.subjectFilter = ""
			m.filteringSubject = false
		}

		// Check if we should skip compute-only steps when going back
		isVisualize := len(m.modeOptions) > 0 && m.modeOptions[m.modeIndex] == styles.ModeVisualize
		if isVisualize && (m.Pipeline == types.PipelineBehavior || m.Pipeline == types.PipelineFeatures) {
			// In visualize mode, skip compute-only steps when going back
			if m.CurrentStep == types.StepSelectSubjects {
				// Go directly back to mode selection
				for i, step := range m.steps {
					if step == types.StepSelectMode {
						m.stepIndex = i
						m.CurrentStep = step
						m.resetCursorsForStep() // Reset cursors when jumping back
						return true
					}
				}
			}
		}

		m.stepIndex--
		m.CurrentStep = m.steps[m.stepIndex]
		m.resetCursorsForStep() // Reset cursors when going back

		// Skip advanced config step when going back in visualize mode
		if m.CurrentStep == types.StepAdvancedConfig && isVisualize {
			if m.stepIndex > 0 {
				m.stepIndex--
				m.CurrentStep = m.steps[m.stepIndex]
				m.resetCursorsForStep() // Reset again for the skipped-to step
			}
		}

		return true
	}
	return false
}

///////////////////////////////////////////////////////////////////
// Validation
///////////////////////////////////////////////////////////////////

func (m *Model) validate() []string {
	var errors []string

	selectedCount := 0
	validCount := 0
	for subjID, selected := range m.subjectSelected {
		if selected {
			selectedCount++
			for _, s := range m.subjects {
				if s.ID == subjID {
					if valid, reason := m.Pipeline.ValidateSubject(s); !valid {
						errors = append(errors, fmt.Sprintf("Subject %s: %s", subjID, reason))
					} else {
						validCount++
					}
					break
				}
			}
		}
	}

	if selectedCount == 0 {
		errors = append(errors, "No subjects selected")
	} else if validCount == 0 {
		errors = append(errors, "No valid subjects selected for this pipeline")
	}

	if m.Pipeline == types.PipelineFeatures && m.modeOptions[m.modeIndex] == styles.ModeCompute {
		categoryCount := 0
		for _, selected := range m.selected {
			if selected {
				categoryCount++
			}
		}
		if categoryCount == 0 {
			errors = append(errors, "No feature categories selected")
		}

		bandCount := 0
		for _, selected := range m.bandSelected {
			if selected {
				bandCount++
			}
		}
		if bandCount == 0 {
			errors = append(errors, "No frequency bands selected")
		}

		// Validate time range selection
		if len(m.TimeRanges) == 0 {
			errors = append(errors, "No time ranges defined")
		} else {
			names := make(map[string]bool)
			for _, tr := range m.TimeRanges {
				if tr.Name == "" {
					errors = append(errors, "All time ranges must have a name")
					break
				}
				if names[tr.Name] {
					errors = append(errors, fmt.Sprintf("Duplicate time range name: %s", tr.Name))
					break
				}
				names[tr.Name] = true

				// Check if numeric values are valid (start < end)
				if tr.Tmin != "" && tr.Tmax != "" {
					tmin, errMin := strconv.ParseFloat(tr.Tmin, 64)
					tmax, errMax := strconv.ParseFloat(tr.Tmax, 64)
					if errMin == nil && errMax == nil && tmin > tmax {
						errors = append(errors, fmt.Sprintf("Range '%s': Start time (%.3f) must be less than end time (%.3f)", tr.Name, tmin, tmax))
					}
				}
			}

			// Check for required 'baseline' if baseline-dependent features are selected
			hasBaseline := names["baseline"]
			needsBaseline := false
			for i, cat := range m.categories {
				if m.selected[i] {
					if cat == "erds" || cat == "erp" || cat == "bursts" {
						needsBaseline = true
						break
					}
				}
			}
			if needsBaseline && !hasBaseline {
				errors = append(errors, "Time range 'baseline' is required for baseline-normalized features (ERDS, ERP, bursts)")
			}
		}
	}

	if m.Pipeline == types.PipelineBehavior && m.modeOptions[m.modeIndex] == styles.ModeCompute {
		// Validate computations selection
		computationCount := 0
		for _, selected := range m.computationSelected {
			if selected {
				computationCount++
			}
		}
		if computationCount == 0 {
			errors = append(errors, "No behavior computations selected")
		}

		// Validate feature files selection (consolidated)
		featureFileCount := 0
		for _, selected := range m.featureFileSelected {
			if selected {
				featureFileCount++
			}
		}
		if featureFileCount == 0 {
			errors = append(errors, "No feature files selected")
		}
	}

	return errors
}

///////////////////////////////////////////////////////////////////
// Advanced Configuration Helpers
///////////////////////////////////////////////////////////////////

// getAdvancedOptionCount returns the number of options for the current pipeline
func (m *Model) getAdvancedOptionCount() int {
	switch m.Pipeline {
	case types.PipelineFeatures:
		// Dynamic count based on selected categories
		count := 1 // Always have "Use defaults"
		if m.isCategorySelected("connectivity") {
			count += 1 // Connectivity measures
		}
		if m.isCategorySelected("pac") {
			count += 2 // Phase range, Amp range
		}
		if m.isCategorySelected("aperiodic") {
			count += 1 // Aperiodic range
		}
		if m.isCategorySelected("complexity") {
			count += 1 // PE order
		}
		if count == 1 {
			count = 2 // Ensure at least "Use defaults" shows
		}
		return count
	case types.PipelineBehavior:
		// Use defaults, Correlation method, Bootstrap, N permutations, RNG seed, Control temperature, Control trial order, FDR alpha
		return 8
	case types.PipelineDecoding:
		// Use defaults, N permutations, Inner splits, RNG seed, Skip time-gen
		return 5
	default:
		return 1
	}
}

// toggleAdvancedOption handles Space key for advanced config options
func (m *Model) toggleAdvancedOption() {
	switch m.Pipeline {
	case types.PipelineFeatures:
		m.toggleFeaturesAdvancedOption()
	case types.PipelineBehavior:
		m.toggleBehaviorAdvancedOption()
	case types.PipelineDecoding:
		m.toggleDecodingAdvancedOption()
	}
}

func (m *Model) toggleFeaturesAdvancedOption() {
	// If an option is expanded, toggle items within it
	if m.expandedOption >= 0 {
		if m.expandedOption == 4 { // Connectivity measures
			m.connectivityMeasures[m.subCursor] = !m.connectivityMeasures[m.subCursor]
		}
		m.useDefaultAdvanced = false
		return
	}

	// Build dynamic option list to match cursor position to option type
	type optionType int
	const (
		optUseDefaults optionType = iota
		optMicrostateStates
		optGroupTemplates
		optFixedTemplates
		optConnectivity
		optPACPhaseRange
		optPACAmpRange
		optAperiodicRange
		optPEOrder
		optBurstPercentile
	)

	var options []optionType
	options = append(options, optUseDefaults)

	if m.isCategorySelected("connectivity") {
		options = append(options, optConnectivity)
	}
	if m.isCategorySelected("pac") {
		options = append(options, optPACPhaseRange, optPACAmpRange)
	}
	if m.isCategorySelected("aperiodic") {
		options = append(options, optAperiodicRange)
	}
	if m.isCategorySelected("complexity") {
		options = append(options, optPEOrder)
	}

	if m.advancedCursor >= len(options) {
		return
	}

	opt := options[m.advancedCursor]
	switch opt {
	case optUseDefaults:
		m.useDefaultAdvanced = !m.useDefaultAdvanced
	case optConnectivity:
		m.expandedOption = 4
		m.subCursor = 0
		m.useDefaultAdvanced = false
	case optPACPhaseRange:
		// Cycle through common phase ranges: theta(4-8) -> delta(2-4) -> alpha(8-13) -> theta
		if m.pacPhaseMin == 4.0 && m.pacPhaseMax == 8.0 {
			m.pacPhaseMin, m.pacPhaseMax = 2.0, 4.0 // delta
		} else if m.pacPhaseMin == 2.0 && m.pacPhaseMax == 4.0 {
			m.pacPhaseMin, m.pacPhaseMax = 8.0, 13.0 // alpha
		} else {
			m.pacPhaseMin, m.pacPhaseMax = 4.0, 8.0 // theta (default)
		}
		m.useDefaultAdvanced = false
	case optPACAmpRange:
		// Cycle through common amplitude ranges: gamma(30-80) -> broad-gamma(40-100) -> high-gamma(60-120) -> gamma
		if m.pacAmpMin == 30.0 && m.pacAmpMax == 80.0 {
			m.pacAmpMin, m.pacAmpMax = 40.0, 100.0 // broader gamma
		} else if m.pacAmpMin == 40.0 && m.pacAmpMax == 100.0 {
			m.pacAmpMin, m.pacAmpMax = 60.0, 120.0 // high gamma
		} else {
			m.pacAmpMin, m.pacAmpMax = 30.0, 80.0 // default gamma
		}
		m.useDefaultAdvanced = false
	case optAperiodicRange:
		// Cycle through common aperiodic ranges: standard(2-40) -> narrow(3-30) -> broad(1-50) -> standard
		if m.aperiodicFmin == 2.0 && m.aperiodicFmax == 40.0 {
			m.aperiodicFmin, m.aperiodicFmax = 3.0, 30.0 // narrow
		} else if m.aperiodicFmin == 3.0 && m.aperiodicFmax == 30.0 {
			m.aperiodicFmin, m.aperiodicFmax = 1.0, 50.0 // broad
		} else {
			m.aperiodicFmin, m.aperiodicFmax = 2.0, 40.0 // standard
		}
		m.useDefaultAdvanced = false
	case optPEOrder:
		m.complexityPEOrder++
		if m.complexityPEOrder > 7 {
			m.complexityPEOrder = 3
		}
		m.useDefaultAdvanced = false
	}
}

func (m *Model) toggleBehaviorAdvancedOption() {
	switch m.advancedCursor {
	case 0: // Use defaults
		m.useDefaultAdvanced = !m.useDefaultAdvanced
	case 1: // Correlation method
		if m.correlationMethod == "spearman" {
			m.correlationMethod = "pearson"
		} else {
			m.correlationMethod = "spearman"
		}
		m.useDefaultAdvanced = false
	case 2: // Bootstrap samples - enter edit mode to type number
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case 3: // N permutations - enter edit mode to type number
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case 4: // RNG seed - enter edit mode to type number
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case 5: // Control temperature
		m.controlTemperature = !m.controlTemperature
		m.useDefaultAdvanced = false
	case 6: // Control trial order
		m.controlTrialOrder = !m.controlTrialOrder
		m.useDefaultAdvanced = false
	case 7: // FDR alpha - cycle through common values
		switch m.fdrAlpha {
		case 0.05:
			m.fdrAlpha = 0.01
		case 0.01:
			m.fdrAlpha = 0.10
		case 0.10:
			m.fdrAlpha = 0.05
		default:
			m.fdrAlpha = 0.05
		}
		m.useDefaultAdvanced = false
	}
}

func (m *Model) toggleDecodingAdvancedOption() {
	switch m.advancedCursor {
	case 0: // Use defaults
		m.useDefaultAdvanced = !m.useDefaultAdvanced
	case 1: // N permutations - enter edit mode to type number
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case 2: // Inner splits - enter edit mode to type number
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case 3: // RNG seed - enter edit mode to type number
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case 4: // Skip time-gen
		m.skipTimeGen = !m.skipTimeGen
		m.useDefaultAdvanced = false
	}
}

///////////////////////////////////////////////////////////////////
// Number Input Helpers
///////////////////////////////////////////////////////////////////

// commitNumberInput parses the number buffer and applies it to the current field
func (m *Model) commitNumberInput() {
	if m.numberBuffer == "" {
		return
	}

	val, err := strconv.Atoi(m.numberBuffer)
	if err != nil {
		return // Invalid number, ignore
	}

	switch m.Pipeline {
	case types.PipelineFeatures:
		m.commitFeaturesNumber(val)
	case types.PipelineBehavior:
		m.commitBehaviorNumber(val)
	case types.PipelineDecoding:
		m.commitDecodingNumber(val)
	}
	m.useDefaultAdvanced = false
}

func (m *Model) commitFeaturesNumber(val int) {
	// No numeric fields for current features
}

func (m *Model) commitBehaviorNumber(val int) {
	switch m.advancedCursor {
	case 2: // Bootstrap samples
		if val >= 0 {
			m.bootstrapSamples = val
		}
	case 3: // N permutations
		if val >= 0 {
			m.nPermutations = val
		}
	case 4: // RNG seed
		if val >= 0 {
			m.rngSeed = val
		}
	}
}

func (m *Model) commitDecodingNumber(val int) {
	switch m.advancedCursor {
	case 1: // N permutations
		if val >= 0 {
			m.decodingNPerm = val
		}
	case 2: // Inner splits
		if val >= 2 {
			m.innerSplits = val
		}
	case 3: // RNG seed
		if val >= 0 {
			m.rngSeed = val
		}
	}
}

// startNumberEdit enters editing mode for the current field
func (m *Model) startNumberEdit() {
	m.editingNumber = true
	m.numberBuffer = ""
}
