package wizard

import (
	"fmt"
	"strconv"
	"strings"

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
	m.advancedCursor = 0
	m.subCursor = 0
	m.expandedOption = -1
	m.plotCursor = 0
	m.plotOffset = 0
	if m.CurrentStep == types.StepSelectPlots {
		m.plotCursor = m.findNextVisiblePlot(-1, 1) // Start at first visible if possible
	}
	m.plotConfigCursor = 0

	// Reset any editing states
	m.filteringSubject = false
	m.subjectFilter = ""
	m.editingNumber = false
	m.numberBuffer = ""
	m.editingRangeIdx = -1
	m.editingField = 0
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
	case types.StepConfigureOptions, types.StepSelectPlotCategories:
		// Features pipeline category selection or Plotting pipeline category selection
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
	case types.StepSelectPlots:
		m.plotCursor = m.findNextVisiblePlot(m.plotCursor, -1)
	case types.StepPlotConfig:
		options := m.getPlotConfigOptions()
		if len(options) == 0 {
			break
		}
		if m.plotConfigCursor > 0 {
			m.plotConfigCursor--
		} else {
			m.plotConfigCursor = len(options) - 1
		}
	case types.StepSelectSpatial:
		if m.spatialCursor > 0 {
			m.spatialCursor--
		} else {
			m.spatialCursor = len(spatialModes) - 1
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
	case types.StepConfigureOptions, types.StepSelectPlotCategories:
		// Features pipeline category selection or Plotting pipeline category selection
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
	case types.StepSelectPlots:
		m.plotCursor = m.findNextVisiblePlot(m.plotCursor, 1)
	case types.StepPlotConfig:
		options := m.getPlotConfigOptions()
		if len(options) == 0 {
			break
		}
		if m.plotConfigCursor < len(options)-1 {
			m.plotConfigCursor++
		} else {
			m.plotConfigCursor = 0
		}
	case types.StepSelectSpatial:
		if m.spatialCursor < len(spatialModes)-1 {
			m.spatialCursor++
		} else {
			m.spatialCursor = 0
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
}

func (m *Model) handleRight() {
}

func (m Model) handleEnter() (tea.Model, tea.Cmd) {
	if m.CurrentStep == types.StepReviewExecute {
		if len(m.validationErrors) > 0 {
			return m, nil
		}
		m.ConfirmingExecute = true
		return m, nil
	}

	// Per-step validation
	errors := m.validateStep()
	if len(errors) > 0 {
		m.validationErrors = errors
		return m, nil
	}
	m.validationErrors = nil // Clear if valid

	if m.stepIndex < len(m.steps)-1 {
		m.stepIndex++
		m.CurrentStep = m.steps[m.stepIndex]
		m.resetCursorsForStep()

		if m.CurrentStep == types.StepReviewExecute {
			m.validationErrors = m.validate()
		}
	}
	return m, tea.ClearScreen
}

func (m *Model) validateStep() []string {
	var errors []string
	switch m.CurrentStep {
	case types.StepSelectSubjects:
		count := 0
		for _, sel := range m.subjectSelected {
			if sel {
				count++
			}
		}
		if count == 0 {
			errors = append(errors, "Select at least one subject")
		}
	case types.StepSelectComputations:
		count := 0
		for _, sel := range m.computationSelected {
			if sel {
				count++
			}
		}
		if count == 0 {
			errors = append(errors, "Select at least one analysis to run")
		}
	case types.StepSelectFeatureFiles:
		count := 0
		for _, sel := range m.featureFileSelected {
			if sel {
				count++
			}
		}
		if count == 0 {
			errors = append(errors, "Select at least one feature file to load")
		}
	case types.StepConfigureOptions, types.StepSelectPlotCategories:
		count := 0
		for _, sel := range m.selected {
			if sel {
				count++
			}
		}
		if count == 0 {
			errors = append(errors, "Select at least one category")
		}
	case types.StepSelectBands:
		count := 0
		for _, sel := range m.bandSelected {
			if sel {
				count++
			}
		}
		if count == 0 {
			errors = append(errors, "Select at least one frequency band")
		}
	case types.StepSelectPlots:
		count := 0
		for _, sel := range m.plotSelected {
			if sel {
				count++
			}
		}
		if count == 0 {
			errors = append(errors, "Select at least one plot to generate")
		}
	case types.StepSelectSpatial:
		count := 0
		for _, sel := range m.spatialSelected {
			if sel {
				count++
			}
		}
		if count == 0 {
			errors = append(errors, "Select at least one spatial mode")
		}
	case types.StepTimeRange:
		errors = m.validateTimeRanges()
	case types.StepPlotConfig:
		count := 0
		for _, sel := range m.plotFormatSelected {
			if sel {
				count++
			}
		}
		if count == 0 {
			errors = append(errors, "Select at least one output format (PNG, SVG, or PDF)")
		}
	}
	return errors
}

func (m *Model) handleSpace() {
	switch m.CurrentStep {
	case types.StepSelectComputations:
		m.computationSelected[m.computationCursor] = !m.computationSelected[m.computationCursor]
	case types.StepConfigureOptions, types.StepSelectPlotCategories:
		// Features pipeline category selection or Plotting pipeline category selection
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
	case types.StepSelectPlots:
		if m.plotCursor < len(m.plotItems) {
			m.plotSelected[m.plotCursor] = !m.plotSelected[m.plotCursor]
		}
	case types.StepPlotConfig:
		options := m.getPlotConfigOptions()
		if m.plotConfigCursor < 0 || m.plotConfigCursor >= len(options) {
			break
		}
		opt := options[m.plotConfigCursor]
		switch opt {
		case optPlotPNG:
			m.plotFormatSelected["png"] = !m.plotFormatSelected["png"]
		case optPlotSVG:
			m.plotFormatSelected["svg"] = !m.plotFormatSelected["svg"]
		case optPlotPDF:
			m.plotFormatSelected["pdf"] = !m.plotFormatSelected["pdf"]
		case optPlotDPI:
			if len(m.plotDpiOptions) > 0 {
				m.plotDpiIndex = (m.plotDpiIndex + 1) % len(m.plotDpiOptions)
			}
		case optPlotSaveDPI:
			if len(m.plotDpiOptions) > 0 {
				m.plotSavefigDpiIndex = (m.plotSavefigDpiIndex + 1) % len(m.plotDpiOptions)
			}
		case optPlotSharedColorbar:
			m.plotSharedColorbar = !m.plotSharedColorbar
		}

	case types.StepSelectSpatial:
		m.spatialSelected[m.spatialCursor] = !m.spatialSelected[m.spatialCursor]
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
	case types.StepConfigureOptions, types.StepSelectPlotCategories:
		// Features pipeline or Plotting categories selection
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
	case types.StepSelectPlots:
		for i, plot := range m.plotItems {
			if m.IsPlotCategorySelected(plot.Group) {
				m.plotSelected[i] = true
			}
		}
	}
}

func (m *Model) selectNone() {
	switch m.CurrentStep {
	case types.StepSelectComputations:
		m.computationSelected = make(map[int]bool)
	case types.StepConfigureOptions, types.StepSelectPlotCategories:
		// Features pipeline or Plotting categories selection
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
	case types.StepSelectPlots:
		for i, plot := range m.plotItems {
			if m.IsPlotCategorySelected(plot.Group) {
				m.plotSelected[i] = false
			}
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

		m.stepIndex--
		m.CurrentStep = m.steps[m.stepIndex]
		m.resetCursorsForStep() // Reset cursors when going back
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
					valid, reason := m.Pipeline.ValidateSubject(s)
					if m.Pipeline == types.PipelinePlotting {
						valid, reason = m.validatePlottingSubject(s)
					}
					if !valid {
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
		errors = append(errors, m.validateTimeRanges()...)
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

	if m.Pipeline == types.PipelinePlotting {
		plotCount := 0
		for _, selected := range m.plotSelected {
			if selected {
				plotCount++
			}
		}
		if plotCount == 0 {
			errors = append(errors, "No plots selected")
		}
		formatCount := 0
		for _, selected := range m.plotFormatSelected {
			if selected {
				formatCount++
			}
		}
		if formatCount == 0 {
			errors = append(errors, "No output formats selected")
		}
	}

	return errors
}

func (m *Model) validateTimeRanges() []string {
	var errors []string

	if len(m.TimeRanges) == 0 {
		errors = append(errors, "No time ranges defined")
		return errors
	}

	names := make(map[string]bool)
	for _, tr := range m.TimeRanges {
		if tr.Name == "" {
			errors = append(errors, "All time ranges must have a name")
			continue
		}
		if names[tr.Name] {
			errors = append(errors, fmt.Sprintf("Duplicate time range name: %s", tr.Name))
			continue
		}
		names[tr.Name] = true

		// Check if numeric values are valid (start < end)
		if tr.Tmin != "" && tr.Tmax != "" {
			tmin, errMin := strconv.ParseFloat(tr.Tmin, 64)
			tmax, errMax := strconv.ParseFloat(tr.Tmax, 64)
			if errMin == nil && errMax == nil && tmin >= tmax {
				errors = append(errors, fmt.Sprintf("Range '%s': Start time (%.3f) must be less than end time (%.3f)", tr.Name, tmin, tmax))
			}
		}
	}

	// Check for required 'baseline' and 'active'
	hasBaseline := names["baseline"]
	hasActive := names["active"]

	if !hasBaseline {
		errors = append(errors, "Time range 'baseline' is required")
	}
	if !hasActive {
		errors = append(errors, "Time range 'active' is required")
	}

	// Validate that baseline and active have values
	for _, tr := range m.TimeRanges {
		if (tr.Name == "baseline" || tr.Name == "active") && (tr.Tmin == "" || tr.Tmax == "") {
			errors = append(errors, fmt.Sprintf("Time range '%s' must have both start and end times defined", tr.Name))
		}
	}

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

	return errors
}

func (m Model) plotRequirements() (requiresEpochs bool, requiresFeatures bool) {
	for i, plot := range m.plotItems {
		if !m.plotSelected[i] {
			continue
		}
		if plot.RequiresEpochs {
			requiresEpochs = true
		}
		if plot.RequiresFeatures {
			requiresFeatures = true
		}
	}
	return requiresEpochs, requiresFeatures
}

func (m Model) validatePlottingSubject(s types.SubjectStatus) (bool, string) {
	requiresEpochs, requiresFeatures := m.plotRequirements()
	if requiresEpochs && !s.HasEpochs {
		return false, "missing epochs"
	}
	if requiresFeatures && !s.HasFeatures {
		return false, "missing features"
	}
	return true, ""
}

///////////////////////////////////////////////////////////////////
// Advanced Configuration Helpers
///////////////////////////////////////////////////////////////////

// getAdvancedOptionCount returns the number of options for the current pipeline
func (m *Model) getAdvancedOptionCount() int {
	switch m.Pipeline {
	case types.PipelineFeatures:
		return len(m.getFeaturesOptions())
	case types.PipelineBehavior:
		return len(m.getBehaviorOptions())

	case types.PipelineDecoding:
		return len(m.getDecodingOptions())
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
	options := m.getFeaturesOptions()

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
	case optERPBaseline:
		m.erpBaselineCorrection = !m.erpBaselineCorrection
		m.useDefaultAdvanced = false
	case optBurstThreshold:
		if m.burstThresholdZ == 1.5 {
			m.burstThresholdZ = 2.0
		} else if m.burstThresholdZ == 2.0 {
			m.burstThresholdZ = 2.5
		} else if m.burstThresholdZ == 2.5 {
			m.burstThresholdZ = 3.0
		} else {
			m.burstThresholdZ = 1.5
		}
		m.useDefaultAdvanced = false
	case optPowerBaselineMode:
		m.powerBaselineMode = (m.powerBaselineMode + 1) % 5
		m.useDefaultAdvanced = false
	case optSpectralEdge:
		if m.spectralEdgePercentile == 0.90 {
			m.spectralEdgePercentile = 0.95
		} else if m.spectralEdgePercentile == 0.95 {
			m.spectralEdgePercentile = 0.99
		} else {
			m.spectralEdgePercentile = 0.90
		}
		m.useDefaultAdvanced = false
	case optConnOutputLevel:
		m.connOutputLevel = (m.connOutputLevel + 1) % 2
		m.useDefaultAdvanced = false
	case optConnGraphMetrics:
		m.connGraphMetrics = !m.connGraphMetrics
		m.useDefaultAdvanced = false
	case optConnAECMode:
		m.connAECMode = (m.connAECMode + 1) % 3
		m.useDefaultAdvanced = false
	}

}

func (m *Model) toggleBehaviorAdvancedOption() {
	options := m.getBehaviorOptions()
	if m.advancedCursor < 0 || m.advancedCursor >= len(options) {
		return
	}

	opt := options[m.advancedCursor]
	switch opt {
	case optUseDefaults:
		m.useDefaultAdvanced = !m.useDefaultAdvanced
	case optCorrMethod:
		if m.correlationMethod == "spearman" {
			m.correlationMethod = "pearson"
		} else {
			m.correlationMethod = "spearman"
		}
		m.useDefaultAdvanced = false
	case optBootstrap:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optNPerm:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optRNGSeed:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optControlTemp:
		m.controlTemperature = !m.controlTemperature
		m.useDefaultAdvanced = false
	case optControlOrder:
		m.controlTrialOrder = !m.controlTrialOrder
		m.useDefaultAdvanced = false
	case optFDRAlpha:
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
	options := m.getDecodingOptions()
	if m.advancedCursor < 0 || m.advancedCursor >= len(options) {
		return
	}

	opt := options[m.advancedCursor]
	switch opt {
	case optUseDefaults:
		m.useDefaultAdvanced = !m.useDefaultAdvanced
	case optDecodingNPerm, optDecodingInnerSplits, optRNGSeed:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optDecodingSkipTimeGen:
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
	// Re-build the same options slice as toggleFeaturesAdvancedOption to find current opt
	options := m.getFeaturesOptions()

	if m.advancedCursor >= len(options) {
		return
	}

	opt := options[m.advancedCursor]
	_ = opt // No numeric fields currently
}

func (m *Model) commitBehaviorNumber(val int) {
	options := m.getBehaviorOptions()
	if m.advancedCursor < 0 || m.advancedCursor >= len(options) {
		return
	}

	opt := options[m.advancedCursor]
	switch opt {
	case optBootstrap:
		if val >= 0 {
			m.bootstrapSamples = val
		}
	case optNPerm:
		if val >= 0 {
			m.nPermutations = val
		}
	case optRNGSeed:
		if val >= 0 {
			m.rngSeed = val
		}
	}
}

func (m *Model) commitDecodingNumber(val int) {
	options := m.getDecodingOptions()
	if m.advancedCursor < 0 || m.advancedCursor >= len(options) {
		return
	}

	opt := options[m.advancedCursor]
	switch opt {
	case optDecodingNPerm:
		if val >= 0 {
			m.decodingNPerm = val
		}
	case optDecodingInnerSplits:
		if val >= 2 {
			m.innerSplits = val
		}
	case optRNGSeed:
		if val >= 0 {
			m.rngSeed = val
		}
	}
}

// findNextVisiblePlot finds the next plot index in a visible category
func (m Model) findNextVisiblePlot(current int, delta int) int {
	if len(m.plotItems) == 0 {
		return 0
	}

	// Try to find one
	next := current
	if next < 0 {
		next = len(m.plotItems) - 1
	}

	for i := 0; i < len(m.plotItems); i++ {
		next = (next + delta + len(m.plotItems)) % len(m.plotItems)
		if m.IsPlotCategorySelected(m.plotItems[next].Group) {
			return next
		}
	}
	return current
}

// startNumberEdit enters editing mode for the current field
func (m *Model) startNumberEdit() {
	m.editingNumber = true
	m.numberBuffer = ""
}

// IsPlotCategorySelected checks if a plot group/category is selected
func (m Model) IsPlotCategorySelected(group string) bool {
	// If not in Plotting pipeline or no Categories step, assume all selected
	hasCategoryStep := false
	for _, s := range m.steps {
		if s == types.StepSelectPlotCategories {
			hasCategoryStep = true
			break
		}
	}

	if m.Pipeline != types.PipelinePlotting || !hasCategoryStep {
		return true
	}

	// plotCategories are defined in model.go
	for i, cat := range plotCategories {
		if strings.EqualFold(cat.Key, group) {
			return m.selected[i]
		}
	}
	return false
}

// SelectedPlotCategoryKeys returns the keys of selected plot categories
func (m Model) SelectedPlotCategoryKeys() []string {
	var keys []string
	for i, cat := range plotCategories {
		if m.selected[i] {
			keys = append(keys, cat.Key)
		}
	}
	return keys
}
