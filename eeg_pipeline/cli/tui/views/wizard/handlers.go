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
	m.advancedOffset = 0
	m.subCursor = 0
	m.expandedOption = expandedNone
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
	m.editingText = false
	m.textBuffer = ""
	m.editingTextField = textFieldNone
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
			if m.expandedOption == expandedConnectivityMeasures {
				if m.subCursor > 0 {
					m.subCursor--
				} else {
					m.subCursor = len(connectivityMeasures) - 1
				}
			}
			m.UpdateAdvancedOffset()
		} else {
			// Navigate between main options
			optCount := m.getAdvancedOptionCount()
			if m.advancedCursor > 0 {
				m.advancedCursor--
			} else {
				m.advancedCursor = optCount - 1
			}
			m.UpdateAdvancedOffset()
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
			if m.expandedOption == expandedConnectivityMeasures {
				if m.subCursor < len(connectivityMeasures)-1 {
					m.subCursor++
				} else {
					m.subCursor = 0
				}
			}
			m.UpdateAdvancedOffset()
		} else {
			// Navigate between main options
			optCount := m.getAdvancedOptionCount()
			if m.advancedCursor < optCount-1 {
				m.advancedCursor++
			} else {
				m.advancedCursor = 0
			}
			m.UpdateAdvancedOffset()
		}
	}
}

func (m *Model) handleTab() {
	switch m.CurrentStep {
	case types.StepAdvancedConfig:
		if m.expandedOption >= 0 {
			// Collapse and move to next primary option
			m.expandedOption = expandedNone
			m.subCursor = 0
			optCount := m.getAdvancedOptionCount()
			if m.advancedCursor < optCount-1 {
				m.advancedCursor++
			} else {
				m.advancedCursor = 0
			}
			m.UpdateAdvancedOffset()
		} else {
			m.handleDown()
		}
	default:
		m.handleDown()
	}
}

func (m *Model) handleLeft() {
	// Left/right navigation is no longer used for behavior advanced config
	// since sections are now collapsible like the features pipeline
}

func (m *Model) handleRight() {
	// Left/right navigation is no longer used for behavior advanced config
	// since sections are now collapsible like the features pipeline
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

		// Dynamic step skipping based on pipeline and mode
		for m.stepIndex < len(m.steps)-1 {
			skip := false
			switch m.Pipeline {
			case types.PipelineFeatures:
				mode := m.modeOptions[m.modeIndex]
				switch mode {
				case "combine":
					// For combine, only Subjects and Execute are needed
					if m.CurrentStep != types.StepReviewExecute && m.CurrentStep != types.StepSelectSubjects && m.CurrentStep != types.StepSelectMode {
						skip = true
					}
				case styles.ModeVisualize:
					// For visualize, skip bands, spatial, time, and advanced config (handled by specialized plotting pipeline)
					if m.CurrentStep == types.StepSelectBands || m.CurrentStep == types.StepSelectSpatial || m.CurrentStep == types.StepTimeRange || m.CurrentStep == types.StepAdvancedConfig {
						skip = true
					}
				}
			case types.PipelineBehavior:
				mode := m.modeOptions[m.modeIndex]
				if mode == styles.ModeVisualize {
					// For visualize, skip computations selection, features selection, bands, and advanced config
					if m.CurrentStep == types.StepSelectComputations || m.CurrentStep == types.StepSelectFeatureFiles || m.CurrentStep == types.StepSelectBands || m.CurrentStep == types.StepAdvancedConfig {
						skip = true
					}
				}
			}

			if !skip {
				break
			}
			m.stepIndex++
			m.CurrentStep = m.steps[m.stepIndex]
		}

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
		if m.CurrentStep == types.StepSelectPlotCategories && m.Pipeline == types.PipelinePlotting {
			m.togglePlotCategory(m.categoryIndex)
		} else {
			m.selected[m.categoryIndex] = !m.selected[m.categoryIndex]
		}
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
		if m.CurrentStep == types.StepSelectPlotCategories && m.Pipeline == types.PipelinePlotting {
			for i := range m.plotItems {
				m.plotSelected[i] = true
			}
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
		if m.CurrentStep == types.StepSelectPlotCategories && m.Pipeline == types.PipelinePlotting {
			for i := range m.plotItems {
				m.plotSelected[i] = false
			}
		}
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
		m.expandedOption = expandedNone
		m.subCursor = 0
		m.UpdateAdvancedOffset()
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

		// Dynamic step skipping (reverse)
		for m.stepIndex > 0 {
			skip := false
			switch m.Pipeline {
			case types.PipelineFeatures:
				mode := m.modeOptions[m.modeIndex]
				switch mode {
				case "combine":
					if m.CurrentStep != types.StepReviewExecute && m.CurrentStep != types.StepSelectSubjects && m.CurrentStep != types.StepSelectMode {
						skip = true
					}
				case styles.ModeVisualize:
					if m.CurrentStep == types.StepSelectBands || m.CurrentStep == types.StepSelectSpatial || m.CurrentStep == types.StepTimeRange || m.CurrentStep == types.StepAdvancedConfig {
						skip = true
					}
				}
			case types.PipelineBehavior:
				mode := m.modeOptions[m.modeIndex]
				if mode == styles.ModeVisualize {
					if m.CurrentStep == types.StepSelectComputations || m.CurrentStep == types.StepSelectFeatureFiles || m.CurrentStep == types.StepSelectBands || m.CurrentStep == types.StepAdvancedConfig {
						skip = true
					}
				}
			}

			if !skip {
				break
			}
			m.stepIndex--
			m.CurrentStep = m.steps[m.stepIndex]
		}

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
		if len(m.SelectedPlotIDs()) == 0 {
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

func (m Model) plotRequirements() (requiresEpochs bool, requiresFeatures bool, requiresStats bool) {
	for i, plot := range m.plotItems {
		if !m.plotSelected[i] || !m.IsPlotCategorySelected(plot.Group) {
			continue
		}
		if plot.RequiresEpochs {
			requiresEpochs = true
		}
		if plot.RequiresFeatures {
			requiresFeatures = true
		}
		if plot.RequiresStats {
			requiresStats = true
		}
	}
	return requiresEpochs, requiresFeatures, requiresStats
}

func (m Model) validatePlottingSubject(s types.SubjectStatus) (bool, string) {
	requiresEpochs, requiresFeatures, requiresStats := m.plotRequirements()
	if requiresEpochs && !s.HasEpochs {
		return false, "missing epochs"
	}
	if requiresFeatures && !s.HasFeatures {
		return false, "missing features"
	}
	if requiresStats && !s.HasStats {
		return false, "missing stats"
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
	case types.PipelinePreprocessing:
		return len(m.getPreprocessingOptions())
	case types.PipelineRawToBIDS:
		return len(m.getRawToBidsOptions())
	case types.PipelineMergePsychoPyData:
		return len(m.getMergeBehaviorOptions())
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
	case types.PipelinePreprocessing:
		m.togglePreprocessingAdvancedOption()
	case types.PipelineRawToBIDS:
		m.toggleRawToBidsAdvancedOption()
	case types.PipelineMergePsychoPyData:
		m.toggleMergeBehaviorAdvancedOption()
	}
}

func (m *Model) toggleFeaturesAdvancedOption() {
	// If an option is expanded, toggle items within it
	if m.expandedOption >= 0 {
		if m.expandedOption == expandedConnectivityMeasures {
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
	case optFeatGroupConnectivity:
		m.featGroupConnectivityExpanded = !m.featGroupConnectivityExpanded
		if !m.featGroupConnectivityExpanded && m.expandedOption == expandedConnectivityMeasures {
			m.expandedOption = expandedNone
			m.subCursor = 0
		}
		m.useDefaultAdvanced = false
	case optFeatGroupPAC:
		m.featGroupPACExpanded = !m.featGroupPACExpanded
		m.useDefaultAdvanced = false
	case optFeatGroupAperiodic:
		m.featGroupAperiodicExpanded = !m.featGroupAperiodicExpanded
		m.useDefaultAdvanced = false
	case optFeatGroupComplexity:
		m.featGroupComplexityExpanded = !m.featGroupComplexityExpanded
		m.useDefaultAdvanced = false
	case optFeatGroupBursts:
		m.featGroupBurstsExpanded = !m.featGroupBurstsExpanded
		m.useDefaultAdvanced = false
	case optFeatGroupPower:
		m.featGroupPowerExpanded = !m.featGroupPowerExpanded
		m.useDefaultAdvanced = false
	case optFeatGroupSpectral:
		m.featGroupSpectralExpanded = !m.featGroupSpectralExpanded
		m.useDefaultAdvanced = false
	case optFeatGroupERP:
		m.featGroupERPExpanded = !m.featGroupERPExpanded
		m.useDefaultAdvanced = false
	case optFeatGroupRatios:
		m.featGroupRatiosExpanded = !m.featGroupRatiosExpanded
		m.useDefaultAdvanced = false
	case optFeatGroupAsymmetry:
		m.featGroupAsymmetryExpanded = !m.featGroupAsymmetryExpanded
		m.useDefaultAdvanced = false
	case optFeatGroupStorage:
		m.featGroupStorageExpanded = !m.featGroupStorageExpanded
		m.useDefaultAdvanced = false
	case optFeatGroupExecution:
		m.featGroupExecutionExpanded = !m.featGroupExecutionExpanded
		m.useDefaultAdvanced = false
	case optFeatGroupValidation:
		m.featGroupValidationExpanded = !m.featGroupValidationExpanded
		m.useDefaultAdvanced = false
	case optConnectivity:
		m.expandedOption = expandedConnectivityMeasures
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
	case optPACMethod:
		m.pacMethod = (m.pacMethod + 1) % 4
		m.useDefaultAdvanced = false
	case optPACMinEpochs:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optPACPairs:
		m.startTextEdit(textFieldPACPairs)
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
	case optAperiodicPeakZ, optAperiodicMinR2, optAperiodicMinPoints:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optPEOrder:
		m.complexityPEOrder++
		if m.complexityPEOrder > 7 {
			m.complexityPEOrder = 3
		}
		m.useDefaultAdvanced = false
	case optPEDelay:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optERPBaseline:
		m.erpBaselineCorrection = !m.erpBaselineCorrection
		m.useDefaultAdvanced = false
	case optERPAllowNoBaseline:
		m.erpAllowNoBaseline = !m.erpAllowNoBaseline
		m.useDefaultAdvanced = false
	case optERPComponents:
		m.startTextEdit(textFieldERPComponents)
		m.useDefaultAdvanced = false
	case optBurstThreshold:
		switch m.burstThresholdZ {
		case 1.5:
			m.burstThresholdZ = 2.0
		case 2.0:
			m.burstThresholdZ = 2.5
		case 2.5:
			m.burstThresholdZ = 3.0
		default:
			m.burstThresholdZ = 1.5
		}
		m.useDefaultAdvanced = false
	case optBurstMinDuration:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optBurstBands:
		m.startTextEdit(textFieldBurstBands)
		m.useDefaultAdvanced = false
	case optPowerBaselineMode:
		m.powerBaselineMode = (m.powerBaselineMode + 1) % 5
		m.useDefaultAdvanced = false
	case optPowerRequireBaseline:
		m.powerRequireBaseline = !m.powerRequireBaseline
		m.useDefaultAdvanced = false
	case optSpectralEdge:
		switch m.spectralEdgePercentile {
		case 0.90:
			m.spectralEdgePercentile = 0.95
		case 0.95:
			m.spectralEdgePercentile = 0.99
		default:
			m.spectralEdgePercentile = 0.90
		}
		m.useDefaultAdvanced = false
	case optConnOutputLevel:
		m.connOutputLevel = (m.connOutputLevel + 1) % 2
		m.useDefaultAdvanced = false
	case optConnGraphMetrics:
		m.connGraphMetrics = !m.connGraphMetrics
		m.useDefaultAdvanced = false
	case optConnGraphProp, optConnWindowLen, optConnWindowStep:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optConnAECMode:
		m.connAECMode = (m.connAECMode + 1) % 3
		m.useDefaultAdvanced = false
	case optExportAll:
		m.exportAllFeatures = !m.exportAllFeatures
		m.useDefaultAdvanced = false
	case optMinEpochs:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optSpectralRatioPairs:
		m.startTextEdit(textFieldSpectralRatioPairs)
		m.useDefaultAdvanced = false
	case optAsymmetryChannelPairs:
		m.startTextEdit(textFieldAsymmetryChannelPairs)
		m.useDefaultAdvanced = false
	case optFailOnMissingWindows:
		m.failOnMissingWindows = !m.failOnMissingWindows
		m.useDefaultAdvanced = false
	case optFailOnMissingNamedWindow:
		m.failOnMissingNamedWindow = !m.failOnMissingNamedWindow
		m.useDefaultAdvanced = false
	}

	// Clamp cursor after expand/collapse changes.
	options = m.getFeaturesOptions()
	if m.advancedCursor >= len(options) {
		m.advancedCursor = len(options) - 1
	}
	if m.advancedCursor < 0 {
		m.advancedCursor = 0
	}
	m.UpdateAdvancedOffset()
}

func (m *Model) toggleBehaviorAdvancedOption() {
	options := m.getBehaviorOptions()
	if m.advancedCursor < 0 || m.advancedCursor >= len(options) {
		return
	}

	sections := m.behaviorSections()
	sectionEnabled := true
	if len(sections) > 0 {
		idx := m.behaviorConfigSection
		if idx < 0 {
			idx = 0
		}
		if idx >= len(sections) {
			idx = len(sections) - 1
		}
		sectionEnabled = sections[idx].Enabled
	}

	opt := options[m.advancedCursor]
	if opt != optUseDefaults && !sectionEnabled {
		return
	}

	switch opt {
	case optUseDefaults:
		m.useDefaultAdvanced = !m.useDefaultAdvanced
	// Behavior section header toggles
	case optBehaviorGroupGeneral:
		m.behaviorGroupGeneralExpanded = !m.behaviorGroupGeneralExpanded
		m.useDefaultAdvanced = false
	case optBehaviorGroupTrialTable:
		m.behaviorGroupTrialTableExpanded = !m.behaviorGroupTrialTableExpanded
		m.useDefaultAdvanced = false
	case optBehaviorGroupCorrelations:
		m.behaviorGroupCorrelationsExpanded = !m.behaviorGroupCorrelationsExpanded
		m.useDefaultAdvanced = false
	case optBehaviorGroupPainSens:
		m.behaviorGroupPainSensExpanded = !m.behaviorGroupPainSensExpanded
		m.useDefaultAdvanced = false
	case optBehaviorGroupConfounds:
		m.behaviorGroupConfoundsExpanded = !m.behaviorGroupConfoundsExpanded
		m.useDefaultAdvanced = false
	case optBehaviorGroupRegression:
		m.behaviorGroupRegressionExpanded = !m.behaviorGroupRegressionExpanded
		m.useDefaultAdvanced = false
	case optBehaviorGroupModels:
		m.behaviorGroupModelsExpanded = !m.behaviorGroupModelsExpanded
		m.useDefaultAdvanced = false
	case optBehaviorGroupStability:
		m.behaviorGroupStabilityExpanded = !m.behaviorGroupStabilityExpanded
		m.useDefaultAdvanced = false
	case optBehaviorGroupConsistency:
		m.behaviorGroupConsistencyExpanded = !m.behaviorGroupConsistencyExpanded
		m.useDefaultAdvanced = false
	case optBehaviorGroupInfluence:
		m.behaviorGroupInfluenceExpanded = !m.behaviorGroupInfluenceExpanded
		m.useDefaultAdvanced = false
	case optBehaviorGroupCondition:
		m.behaviorGroupConditionExpanded = !m.behaviorGroupConditionExpanded
		m.useDefaultAdvanced = false
	case optBehaviorGroupTemporal:
		m.behaviorGroupTemporalExpanded = !m.behaviorGroupTemporalExpanded
		m.useDefaultAdvanced = false
	case optBehaviorGroupCluster:
		m.behaviorGroupClusterExpanded = !m.behaviorGroupClusterExpanded
		m.useDefaultAdvanced = false
	case optBehaviorGroupMediation:
		m.behaviorGroupMediationExpanded = !m.behaviorGroupMediationExpanded
		m.useDefaultAdvanced = false
	case optBehaviorGroupMixedEffects:
		m.behaviorGroupMixedEffectsExpanded = !m.behaviorGroupMixedEffectsExpanded
		m.useDefaultAdvanced = false
	case optCorrMethod:
		if m.correlationMethod == "spearman" {
			m.correlationMethod = "pearson"
		} else {
			m.correlationMethod = "spearman"
		}
		m.useDefaultAdvanced = false
	case optRobustCorrelation:
		m.robustCorrelation = (m.robustCorrelation + 1) % 4
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
	case optBehaviorNJobs, optBehaviorMinSamples:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optControlTemp:
		m.controlTemperature = !m.controlTemperature
		m.useDefaultAdvanced = false
	case optControlOrder:
		m.controlTrialOrder = !m.controlTrialOrder
		m.useDefaultAdvanced = false
	case optFDRAlpha:
		m.startNumberEdit()
		m.useDefaultAdvanced = false

	case optComputeChangeScores:
		m.behaviorComputeChangeScores = !m.behaviorComputeChangeScores
		m.useDefaultAdvanced = false
	case optComputeLosoStability:
		m.behaviorComputeLosoStability = !m.behaviorComputeLosoStability
		m.useDefaultAdvanced = false
	case optComputeBayesFactors:
		m.behaviorComputeBayesFactors = !m.behaviorComputeBayesFactors
		m.useDefaultAdvanced = false

	// Trial table / residual options
	case optTrialTableFormat:
		m.trialTableFormat = (m.trialTableFormat + 1) % 2
		m.useDefaultAdvanced = false
	case optTrialTableIncludeFeatures:
		m.trialTableIncludeFeatures = !m.trialTableIncludeFeatures
		m.useDefaultAdvanced = false
	case optTrialTableIncludeCovars:
		m.trialTableIncludeCovars = !m.trialTableIncludeCovars
		m.useDefaultAdvanced = false
	case optTrialTableIncludeEvents:
		m.trialTableIncludeEvents = !m.trialTableIncludeEvents
		m.useDefaultAdvanced = false
	case optTrialTableAddLagFeatures:
		m.trialTableAddLagFeatures = !m.trialTableAddLagFeatures
		m.useDefaultAdvanced = false
	case optTrialTableExtraEventCols:
		m.startTextEdit(textFieldTrialTableExtraEventColumns)
		m.useDefaultAdvanced = false
	case optTrialTableValidate:
		m.trialTableValidateEnabled = !m.trialTableValidateEnabled
		m.useDefaultAdvanced = false
	case optTrialTableRatingMin, optTrialTableRatingMax, optTrialTableTempMin, optTrialTableTempMax, optTrialTableHighMissingFrac:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optFeatureSummariesEnabled:
		m.featureSummariesEnabled = !m.featureSummariesEnabled
		m.useDefaultAdvanced = false
	case optPainResidualEnabled:
		m.painResidualEnabled = !m.painResidualEnabled
		m.useDefaultAdvanced = false
	case optPainResidualMethod:
		m.painResidualMethod = (m.painResidualMethod + 1) % 2
		m.useDefaultAdvanced = false
	case optPainResidualMinSamples, optPainResidualPolyDegree:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optPainResidualModelCompare:
		m.painResidualModelCompareEnabled = !m.painResidualModelCompareEnabled
		m.useDefaultAdvanced = false
	case optPainResidualModelCompareMinSamples:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optPainResidualBreakpoint:
		m.painResidualBreakpointEnabled = !m.painResidualBreakpointEnabled
		m.useDefaultAdvanced = false
	case optPainResidualBreakpointMinSamples, optPainResidualBreakpointCandidates, optPainResidualBreakpointQlow, optPainResidualBreakpointQhigh:
		m.startNumberEdit()
		m.useDefaultAdvanced = false

	// Confounds
	case optConfoundsAddAsCovariates:
		m.confoundsAddAsCovariates = !m.confoundsAddAsCovariates
		m.useDefaultAdvanced = false
	case optConfoundsMaxCovariates:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optConfoundsQCColumnPatterns:
		m.startTextEdit(textFieldConfoundsQCColumnPatterns)
		m.useDefaultAdvanced = false

	// Regression
	case optRegressionFeatureSet:
		m.regressionFeatureSet = (m.regressionFeatureSet + 1) % 2
		m.useDefaultAdvanced = false
	case optRegressionOutcome:
		m.regressionOutcome = (m.regressionOutcome + 1) % 2
		m.useDefaultAdvanced = false
	case optRegressionIncludeTemperature:
		m.regressionIncludeTemperature = !m.regressionIncludeTemperature
		m.useDefaultAdvanced = false
	case optRegressionTempControl:
		m.regressionTempControl = (m.regressionTempControl + 1) % 2
		m.useDefaultAdvanced = false
	case optRegressionIncludeTrialOrder:
		m.regressionIncludeTrialOrder = !m.regressionIncludeTrialOrder
		m.useDefaultAdvanced = false
	case optRegressionIncludePrev:
		m.regressionIncludePrev = !m.regressionIncludePrev
		m.useDefaultAdvanced = false
	case optRegressionIncludeRunBlock:
		m.regressionIncludeRunBlock = !m.regressionIncludeRunBlock
		m.useDefaultAdvanced = false
	case optRegressionIncludeInteraction:
		m.regressionIncludeInteraction = !m.regressionIncludeInteraction
		m.useDefaultAdvanced = false
	case optRegressionStandardize:
		m.regressionStandardize = !m.regressionStandardize
		m.useDefaultAdvanced = false
	case optRegressionMinSamples, optRegressionPermutations, optRegressionMaxFeatures:
		m.startNumberEdit()
		m.useDefaultAdvanced = false

	// Models
	case optModelsFeatureSet:
		m.modelsFeatureSet = (m.modelsFeatureSet + 1) % 2
		m.useDefaultAdvanced = false
	case optModelsIncludeTemperature:
		m.modelsIncludeTemperature = !m.modelsIncludeTemperature
		m.useDefaultAdvanced = false
	case optModelsTempControl:
		m.modelsTempControl = (m.modelsTempControl + 1) % 2
		m.useDefaultAdvanced = false
	case optModelsIncludeTrialOrder:
		m.modelsIncludeTrialOrder = !m.modelsIncludeTrialOrder
		m.useDefaultAdvanced = false
	case optModelsIncludePrev:
		m.modelsIncludePrev = !m.modelsIncludePrev
		m.useDefaultAdvanced = false
	case optModelsIncludeRunBlock:
		m.modelsIncludeRunBlock = !m.modelsIncludeRunBlock
		m.useDefaultAdvanced = false
	case optModelsIncludeInteraction:
		m.modelsIncludeInteraction = !m.modelsIncludeInteraction
		m.useDefaultAdvanced = false
	case optModelsStandardize:
		m.modelsStandardize = !m.modelsStandardize
		m.useDefaultAdvanced = false
	case optModelsMinSamples, optModelsMaxFeatures:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optModelsOutcomeRating:
		m.modelsOutcomeRating = !m.modelsOutcomeRating
		if !m.modelsOutcomeRating && !m.modelsOutcomePainResidual && !m.modelsOutcomePainBinary {
			m.modelsOutcomeRating = true
		}
		m.useDefaultAdvanced = false
	case optModelsOutcomePainResidual:
		m.modelsOutcomePainResidual = !m.modelsOutcomePainResidual
		if !m.modelsOutcomeRating && !m.modelsOutcomePainResidual && !m.modelsOutcomePainBinary {
			m.modelsOutcomePainResidual = true
		}
		m.useDefaultAdvanced = false
	case optModelsOutcomePainBinary:
		m.modelsOutcomePainBinary = !m.modelsOutcomePainBinary
		if !m.modelsOutcomeRating && !m.modelsOutcomePainResidual && !m.modelsOutcomePainBinary {
			m.modelsOutcomePainBinary = true
		}
		m.useDefaultAdvanced = false
	case optModelsFamilyOLS:
		m.modelsFamilyOLS = !m.modelsFamilyOLS
		if !m.modelsFamilyOLS && !m.modelsFamilyRobust && !m.modelsFamilyQuantile && !m.modelsFamilyLogit {
			m.modelsFamilyOLS = true
		}
		m.useDefaultAdvanced = false
	case optModelsFamilyRobust:
		m.modelsFamilyRobust = !m.modelsFamilyRobust
		if !m.modelsFamilyOLS && !m.modelsFamilyRobust && !m.modelsFamilyQuantile && !m.modelsFamilyLogit {
			m.modelsFamilyRobust = true
		}
		m.useDefaultAdvanced = false
	case optModelsFamilyQuantile:
		m.modelsFamilyQuantile = !m.modelsFamilyQuantile
		if !m.modelsFamilyOLS && !m.modelsFamilyRobust && !m.modelsFamilyQuantile && !m.modelsFamilyLogit {
			m.modelsFamilyQuantile = true
		}
		m.useDefaultAdvanced = false
	case optModelsFamilyLogit:
		m.modelsFamilyLogit = !m.modelsFamilyLogit
		if !m.modelsFamilyOLS && !m.modelsFamilyRobust && !m.modelsFamilyQuantile && !m.modelsFamilyLogit {
			m.modelsFamilyLogit = true
		}
		m.useDefaultAdvanced = false
	case optModelsBinaryOutcome:
		m.modelsBinaryOutcome = (m.modelsBinaryOutcome + 1) % 2
		m.useDefaultAdvanced = false

	// Stability
	case optStabilityFeatureSet:
		m.stabilityFeatureSet = (m.stabilityFeatureSet + 1) % 2
		m.useDefaultAdvanced = false
	case optStabilityMethod:
		m.stabilityMethod = (m.stabilityMethod + 1) % 2
		m.useDefaultAdvanced = false
	case optStabilityOutcome:
		m.stabilityOutcome = (m.stabilityOutcome + 1) % 3
		m.useDefaultAdvanced = false
	case optStabilityGroupColumn:
		m.stabilityGroupColumn = (m.stabilityGroupColumn + 1) % 3
		m.useDefaultAdvanced = false
	case optStabilityPartialTemp:
		m.stabilityPartialTemp = !m.stabilityPartialTemp
		m.useDefaultAdvanced = false
	case optStabilityMinGroupTrials, optStabilityMaxFeatures, optStabilityAlpha:
		m.startNumberEdit()
		m.useDefaultAdvanced = false

	// Consistency
	case optConsistencyEnabled:
		m.consistencyEnabled = !m.consistencyEnabled
		m.useDefaultAdvanced = false

	// Influence
	case optInfluenceFeatureSet:
		m.influenceFeatureSet = (m.influenceFeatureSet + 1) % 2
		m.useDefaultAdvanced = false
	case optInfluenceOutcomeRating:
		m.influenceOutcomeRating = !m.influenceOutcomeRating
		if !m.influenceOutcomeRating && !m.influenceOutcomePainResidual {
			m.influenceOutcomeRating = true
		}
		m.useDefaultAdvanced = false
	case optInfluenceOutcomePainResidual:
		m.influenceOutcomePainResidual = !m.influenceOutcomePainResidual
		if !m.influenceOutcomeRating && !m.influenceOutcomePainResidual {
			m.influenceOutcomePainResidual = true
		}
		m.useDefaultAdvanced = false
	case optInfluenceMaxFeatures:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optInfluenceIncludeTemperature:
		m.influenceIncludeTemperature = !m.influenceIncludeTemperature
		m.useDefaultAdvanced = false
	case optInfluenceTempControl:
		m.influenceTempControl = (m.influenceTempControl + 1) % 2
		m.useDefaultAdvanced = false
	case optInfluenceIncludeTrialOrder:
		m.influenceIncludeTrialOrder = !m.influenceIncludeTrialOrder
		m.useDefaultAdvanced = false
	case optInfluenceIncludeRunBlock:
		m.influenceIncludeRunBlock = !m.influenceIncludeRunBlock
		m.useDefaultAdvanced = false
	case optInfluenceIncludeInteraction:
		m.influenceIncludeInteraction = !m.influenceIncludeInteraction
		m.useDefaultAdvanced = false
	case optInfluenceStandardize:
		m.influenceStandardize = !m.influenceStandardize
		m.useDefaultAdvanced = false
	case optInfluenceCooksThreshold, optInfluenceLeverageThreshold:
		m.startNumberEdit()
		m.useDefaultAdvanced = false

	// Pain sensitivity
	case optPainSensitivityMinTrials:
		m.startNumberEdit()
		m.useDefaultAdvanced = false

	// Temporal
	case optTemporalResolutionMs, optTemporalTimeMinMs, optTemporalTimeMaxMs, optTemporalSmoothMs:
		m.startNumberEdit()
		m.useDefaultAdvanced = false

	// Cluster options
	case optClusterThreshold:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optClusterMinSize:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optClusterTail:
		switch m.clusterTail {
		case 0:
			m.clusterTail = 1
		case 1:
			m.clusterTail = -1
		case -1:
			m.clusterTail = 0
		default:
			m.clusterTail = 0
		}
		m.useDefaultAdvanced = false
	// Mediation options
	case optMediationBootstrap, optMediationMinEffect, optMediationMaxMediators:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	// Mixed effects options
	case optMixedMaxFeatures:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optMixedEffectsType:
		m.mixedEffectsType = (m.mixedEffectsType + 1) % 2
		m.useDefaultAdvanced = false
	// Condition options
	case optConditionEffectThreshold:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optConditionMinTrials:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optConditionFailFast:
		m.conditionFailFast = !m.conditionFailFast
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

func (m *Model) togglePreprocessingAdvancedOption() {
	options := m.getPreprocessingOptions()
	if m.advancedCursor < 0 || m.advancedCursor >= len(options) {
		return
	}

	opt := options[m.advancedCursor]
	switch opt {
	case optUseDefaults:
		m.useDefaultAdvanced = !m.useDefaultAdvanced
	case optPrepUsePyprep:
		m.prepUsePyprep = !m.prepUsePyprep
		m.useDefaultAdvanced = false
	case optPrepUseIcalabel:
		m.prepUseIcalabel = !m.prepUseIcalabel
		m.useDefaultAdvanced = false
	case optPrepNJobs, optPrepResample, optPrepLFreq, optPrepHFreq, optPrepNotch, optPrepICAComp, optPrepProbThresh, optPrepEpochsTmin, optPrepEpochsTmax:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optPrepICAMethod:
		m.prepICAMethod = (m.prepICAMethod + 1) % 3
		m.useDefaultAdvanced = false
	}
}

func (m *Model) toggleRawToBidsAdvancedOption() {
	options := m.getRawToBidsOptions()
	if m.advancedCursor < 0 || m.advancedCursor >= len(options) {
		return
	}

	opt := options[m.advancedCursor]
	switch opt {
	case optUseDefaults:
		m.useDefaultAdvanced = !m.useDefaultAdvanced
	case optRawMontage:
		m.startTextEdit(textFieldRawMontage)
		m.useDefaultAdvanced = false
	case optRawLineFreq:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optRawOverwrite:
		m.rawOverwrite = !m.rawOverwrite
		m.useDefaultAdvanced = false
	case optRawZeroBaseOnsets:
		m.rawZeroBaseOnsets = !m.rawZeroBaseOnsets
		m.useDefaultAdvanced = false
	case optRawTrimToFirstVolume:
		m.rawTrimToFirstVolume = !m.rawTrimToFirstVolume
		m.useDefaultAdvanced = false
	case optRawEventPrefixes:
		m.startTextEdit(textFieldRawEventPrefixes)
		m.useDefaultAdvanced = false
	case optRawKeepAnnotations:
		m.rawKeepAnnotations = !m.rawKeepAnnotations
		m.useDefaultAdvanced = false
	}
}

func (m *Model) toggleMergeBehaviorAdvancedOption() {
	options := m.getMergeBehaviorOptions()
	if m.advancedCursor < 0 || m.advancedCursor >= len(options) {
		return
	}

	opt := options[m.advancedCursor]
	switch opt {
	case optUseDefaults:
		m.useDefaultAdvanced = !m.useDefaultAdvanced
	case optMergeEventPrefixes:
		m.startTextEdit(textFieldMergeEventPrefixes)
		m.useDefaultAdvanced = false
	case optMergeEventTypes:
		m.startTextEdit(textFieldMergeEventTypes)
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

	val, err := strconv.ParseFloat(m.numberBuffer, 64)
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
	case types.PipelinePreprocessing:
		m.commitPreprocessingNumber(val)
	case types.PipelineRawToBIDS:
		m.commitRawToBidsNumber(val)
	}
	m.useDefaultAdvanced = false
}

func (m *Model) commitFeaturesNumber(val float64) {
	// Re-build the same options slice as toggleFeaturesAdvancedOption to find current opt
	options := m.getFeaturesOptions()

	if m.advancedCursor >= len(options) {
		return
	}

	opt := options[m.advancedCursor]
	switch opt {
	case optAperiodicPeakZ:
		m.aperiodicPeakZ = val
	case optAperiodicMinR2:
		m.aperiodicMinR2 = val
	case optAperiodicMinPoints:
		m.aperiodicMinPoints = int(val)
	case optPACMinEpochs:
		m.pacMinEpochs = int(val)
	case optPEDelay:
		m.complexityPEDelay = int(val)
	case optBurstMinDuration:
		m.burstMinDuration = int(val)
	case optMinEpochs:
		m.minEpochsForFeatures = int(val)
	case optConnGraphProp:
		m.connGraphProp = val
	case optConnWindowLen:
		m.connWindowLen = val
	case optConnWindowStep:
		m.connWindowStep = val
	}
}

func (m *Model) commitBehaviorNumber(val float64) {
	options := m.getBehaviorOptions()
	if m.advancedCursor < 0 || m.advancedCursor >= len(options) {
		return
	}

	opt := options[m.advancedCursor]
	switch opt {
	case optBootstrap:
		if val >= 0 {
			m.bootstrapSamples = int(val)
		}
	case optNPerm:
		if val >= 0 {
			m.nPermutations = int(val)
		}
	case optRNGSeed:
		if val >= 0 {
			m.rngSeed = int(val)
		}
	case optBehaviorNJobs:
		m.behaviorNJobs = int(val)
	case optBehaviorMinSamples:
		if val >= 0 {
			m.behaviorMinSamples = int(val)
		}
	case optFDRAlpha:
		if val > 0 && val <= 1 {
			m.fdrAlpha = val
		}

	// Trial table thresholds
	case optTrialTableRatingMin:
		m.trialTableRatingMin = val
	case optTrialTableRatingMax:
		m.trialTableRatingMax = val
	case optTrialTableTempMin:
		m.trialTableTempMin = val
	case optTrialTableTempMax:
		m.trialTableTempMax = val
	case optTrialTableHighMissingFrac:
		if val >= 0 && val <= 1 {
			m.trialTableHighMissingFrac = val
		}

	// Pain residual + diagnostics
	case optPainResidualMinSamples:
		if val >= 0 {
			m.painResidualMinSamples = int(val)
		}
	case optPainResidualPolyDegree:
		if val >= 1 {
			m.painResidualPolyDegree = int(val)
		}
	case optPainResidualModelCompareMinSamples:
		if val >= 0 {
			m.painResidualModelCompareMinSamples = int(val)
		}
	case optPainResidualBreakpointMinSamples:
		if val >= 0 {
			m.painResidualBreakpointMinSamples = int(val)
		}
	case optPainResidualBreakpointCandidates:
		if val >= 5 {
			m.painResidualBreakpointCandidates = int(val)
		}
	case optPainResidualBreakpointQlow:
		if val > 0 && val < 1 {
			m.painResidualBreakpointQlow = val
		}
	case optPainResidualBreakpointQhigh:
		if val > 0 && val < 1 {
			m.painResidualBreakpointQhigh = val
		}

	// Confounds
	case optConfoundsMaxCovariates:
		if val >= 0 {
			m.confoundsMaxCovariates = int(val)
		}

	// Regression
	case optRegressionMinSamples:
		if val >= 0 {
			m.regressionMinSamples = int(val)
		}
	case optRegressionPermutations:
		if val >= 0 {
			m.regressionPermutations = int(val)
		}
	case optRegressionMaxFeatures:
		if val >= 0 {
			m.regressionMaxFeatures = int(val)
		}

	// Models
	case optModelsMinSamples:
		if val >= 0 {
			m.modelsMinSamples = int(val)
		}
	case optModelsMaxFeatures:
		if val >= 0 {
			m.modelsMaxFeatures = int(val)
		}

	// Stability
	case optStabilityMinGroupTrials:
		if val >= 0 {
			m.stabilityMinGroupTrials = int(val)
		}
	case optStabilityMaxFeatures:
		if val >= 0 {
			m.stabilityMaxFeatures = int(val)
		}
	case optStabilityAlpha:
		if val > 0 && val <= 1 {
			m.stabilityAlpha = val
		}

	// Influence
	case optInfluenceMaxFeatures:
		if val >= 0 {
			m.influenceMaxFeatures = int(val)
		}
	case optInfluenceCooksThreshold:
		if val >= 0 {
			m.influenceCooksThreshold = val
		}
	case optInfluenceLeverageThreshold:
		if val >= 0 {
			m.influenceLeverageThreshold = val
		}

	// Pain sensitivity / temporal
	case optPainSensitivityMinTrials:
		if val >= 0 {
			m.painSensitivityMinTrials = int(val)
		}
	case optTemporalResolutionMs:
		if val >= 1 {
			m.temporalResolutionMs = int(val)
		}
	case optTemporalTimeMinMs:
		m.temporalTimeMinMs = int(val)
	case optTemporalTimeMaxMs:
		m.temporalTimeMaxMs = int(val)
	case optTemporalSmoothMs:
		if val >= 0 {
			m.temporalSmoothMs = int(val)
		}

	case optClusterMinSize:
		if val >= 1 {
			m.clusterMinSize = int(val)
		}
	case optClusterThreshold:
		if val > 0 && val <= 1 {
			m.clusterThreshold = val
		}
	case optMediationBootstrap:
		if val >= 0 {
			m.mediationBootstrap = int(val)
		}
	case optMediationMinEffect:
		if val >= 0 {
			m.mediationMinEffect = val
		}
	case optMediationMaxMediators:
		if val >= 1 {
			m.mediationMaxMediators = int(val)
		}
	case optMixedMaxFeatures:
		if val >= 1 {
			m.mixedMaxFeatures = int(val)
		}
	case optConditionEffectThreshold:
		m.conditionEffectThreshold = val
	case optConditionMinTrials:
		if val >= 0 {
			m.conditionMinTrials = int(val)
		}
	}
}

func (m *Model) commitDecodingNumber(val float64) {
	options := m.getDecodingOptions()
	if m.advancedCursor < 0 || m.advancedCursor >= len(options) {
		return
	}

	opt := options[m.advancedCursor]
	switch opt {
	case optDecodingNPerm:
		if val >= 0 {
			m.decodingNPerm = int(val)
		}
	case optDecodingInnerSplits:
		if val >= 2 {
			m.innerSplits = int(val)
		}
	case optRNGSeed:
		if val >= 0 {
			m.rngSeed = int(val)
		}
	}
}

func (m *Model) commitPreprocessingNumber(val float64) {
	options := m.getPreprocessingOptions()
	if m.advancedCursor < 0 || m.advancedCursor >= len(options) {
		return
	}

	opt := options[m.advancedCursor]
	switch opt {
	case optPrepNJobs:
		if val >= 1 {
			m.prepNJobs = int(val)
		}
	case optPrepResample:
		if val > 0 {
			m.prepResample = int(val)
		}
	case optPrepLFreq:
		m.prepLFreq = val
	case optPrepHFreq:
		m.prepHFreq = val
	case optPrepNotch:
		if val > 0 {
			m.prepNotch = int(val)
		}
	case optPrepICAComp:
		if val > 0 {
			m.prepICAComp = val
		}
	case optPrepProbThresh:
		if val >= 0 && val <= 1 {
			m.prepProbThresh = val
		}
	case optPrepEpochsTmin:
		m.prepEpochsTmin = val
	case optPrepEpochsTmax:
		m.prepEpochsTmax = val
	}
}

func (m *Model) commitRawToBidsNumber(val float64) {
	options := m.getRawToBidsOptions()
	if m.advancedCursor < 0 || m.advancedCursor >= len(options) {
		return
	}

	opt := options[m.advancedCursor]
	switch opt {
	case optRawLineFreq:
		if val > 0 {
			m.rawLineFreq = int(val)
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

func (m *Model) togglePlotCategory(idx int) {
	if idx < 0 || idx >= len(m.categories) {
		return
	}
	m.selected[idx] = !m.selected[idx]

	categories := m.plotCategories
	if len(categories) == 0 {
		categories = defaultPlotCategories
	}
	if idx >= len(categories) {
		return
	}
	key := categories[idx].Key
	for i, plot := range m.plotItems {
		if strings.EqualFold(plot.Group, key) {
			m.plotSelected[i] = m.selected[idx]
		}
	}
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

	categories := m.plotCategories
	if len(categories) == 0 {
		categories = defaultPlotCategories
	}
	for i, cat := range categories {
		if strings.EqualFold(cat.Key, group) {
			return m.selected[i]
		}
	}
	return false
}

// SelectedPlotCategoryKeys returns the keys of selected plot categories
func (m Model) SelectedPlotCategoryKeys() []string {
	var keys []string
	categories := m.plotCategories
	if len(categories) == 0 {
		categories = defaultPlotCategories
	}
	for i, cat := range categories {
		if m.selected[i] {
			keys = append(keys, cat.Key)
		}
	}
	return keys
}
