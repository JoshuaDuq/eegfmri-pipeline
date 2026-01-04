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
	m.featurePlotterCursor = 0
	m.featurePlotterOffset = 0
	if m.CurrentStep == types.StepSelectFeaturePlotters {
		m.featurePlotterCursor = m.findNextFeaturePlotter(-1, 1)
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
	m.editingPlotID = ""
	m.editingPlotField = plotItemConfigFieldNone
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
	case types.StepSelectFeaturePlotters:
		m.featurePlotterCursor = m.findNextFeaturePlotter(m.featurePlotterCursor, -1)
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
			if m.Pipeline == types.PipelinePlotting {
				m.advancedCursor = m.findNextPlottingAdvancedRow(m.advancedCursor, -1)
			} else {
				optCount := m.getAdvancedOptionCount()
				if m.advancedCursor > 0 {
					m.advancedCursor--
				} else {
					m.advancedCursor = optCount - 1
				}
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
	case types.StepSelectFeaturePlotters:
		m.featurePlotterCursor = m.findNextFeaturePlotter(m.featurePlotterCursor, 1)
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
			if m.Pipeline == types.PipelinePlotting {
				m.advancedCursor = m.findNextPlottingAdvancedRow(m.advancedCursor, 1)
			} else {
				optCount := m.getAdvancedOptionCount()
				if m.advancedCursor < optCount-1 {
					m.advancedCursor++
				} else {
					m.advancedCursor = 0
				}
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
			case types.PipelinePlotting:
				if m.CurrentStep == types.StepSelectFeaturePlotters && len(m.selectedFeaturePlotterCategories()) == 0 {
					skip = true
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
	case types.StepSelectFeaturePlotters:
		if len(m.selectedFeaturePlotterCategories()) == 0 {
			break
		}
		if m.featurePlotters == nil && strings.TrimSpace(m.featurePlotterError) == "" {
			errors = append(errors, "Feature plot list is still loading")
			break
		}
		if m.featurePlotters == nil && strings.TrimSpace(m.featurePlotterError) != "" {
			// If discovery failed, let the user proceed (defaults to running all plotters).
			break
		}
		items := m.featurePlotterItems()
		count := 0
		for _, p := range items {
			if m.featurePlotterSelected[p.ID] {
				count++
			}
		}
		if count == 0 {
			errors = append(errors, "Select at least one feature plot")
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
			plotID := m.plotItems[m.plotCursor].ID
			if m.plotSelected[m.plotCursor] {
				_ = m.ensurePlotItemConfig(plotID)
				if m.plotItemConfigExpanded == nil {
					m.plotItemConfigExpanded = make(map[string]bool)
				}
				if _, ok := m.plotItemConfigExpanded[plotID]; !ok {
					m.plotItemConfigExpanded[plotID] = false
				}
			} else {
				delete(m.plotItemConfigs, plotID)
				delete(m.plotItemConfigExpanded, plotID)
			}
		}
	case types.StepSelectFeaturePlotters:
		items := m.featurePlotterItems()
		if len(items) == 0 {
			break
		}
		if m.featurePlotterCursor < 0 || m.featurePlotterCursor >= len(items) {
			break
		}
		id := items[m.featurePlotterCursor].ID
		m.featurePlotterSelected[id] = !m.featurePlotterSelected[id]

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
	case types.StepSelectFeaturePlotters:
		for _, p := range m.featurePlotterItems() {
			m.featurePlotterSelected[p.ID] = true
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
	case types.StepSelectFeaturePlotters:
		for _, p := range m.featurePlotterItems() {
			m.featurePlotterSelected[p.ID] = false
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
	case types.PipelinePlotting:
		return len(m.getPlottingAdvancedRows())

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

func (m *Model) findNextPlottingAdvancedRow(current int, delta int) int {
	rows := m.getPlottingAdvancedRows()
	if len(rows) == 0 {
		return 0
	}
	next := current
	for i := 0; i < len(rows); i++ {
		next = (next + delta + len(rows)) % len(rows)
		switch rows[next].kind {
		case plottingRowSection, plottingRowPlotInfo:
			continue
		default:
			return next
		}
	}
	return current
}

// toggleAdvancedOption handles Space key for advanced config options
func (m *Model) toggleAdvancedOption() {
	switch m.Pipeline {
	case types.PipelineFeatures:
		m.toggleFeaturesAdvancedOption()
	case types.PipelineBehavior:
		m.toggleBehaviorAdvancedOption()
	case types.PipelinePlotting:
		m.togglePlottingAdvancedOption()
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
	// TFR section
	case optFeatGroupTFR:
		m.featGroupTFRExpanded = !m.featGroupTFRExpanded
	case optTfrFreqMin, optTfrFreqMax, optTfrNFreqs, optTfrMinCycles, optTfrNCyclesFactor, optTfrDecim, optTfrWorkers:
		m.startNumberEdit()
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

func (m *Model) togglePlottingAdvancedOption() {
	rows := m.getPlottingAdvancedRows()
	if m.advancedCursor < 0 || m.advancedCursor >= len(rows) {
		return
	}

	cycleTriState := func(v *bool) *bool {
		if v == nil {
			t := true
			return &t
		}
		if *v {
			f := false
			return &f
		}
		return nil
	}

	row := rows[m.advancedCursor]
	switch row.kind {
	case plottingRowPlotHeader:
		m.plotItemConfigExpanded[row.plotID] = !m.plotItemConfigExpanded[row.plotID]
		m.UpdateAdvancedOffset()
		return
	case plottingRowPlotField:
		cfg := m.ensurePlotItemConfig(row.plotID)
		switch row.plotField {
		case plotItemConfigFieldCompareWindows:
			cfg.CompareWindows = cycleTriState(cfg.CompareWindows)
			m.plotItemConfigs[row.plotID] = cfg
			m.useDefaultAdvanced = false
		case plotItemConfigFieldCompareColumns:
			cfg.CompareColumns = cycleTriState(cfg.CompareColumns)
			m.plotItemConfigs[row.plotID] = cfg
			m.useDefaultAdvanced = false
		case plotItemConfigFieldTfrDefaultBaselineWindow,
			plotItemConfigFieldComparisonWindows,
			plotItemConfigFieldComparisonSegment,
			plotItemConfigFieldComparisonColumn,
			plotItemConfigFieldComparisonValues,
			plotItemConfigFieldComparisonROIs:
			m.startPlotTextEdit(row.plotID, row.plotField)
			m.useDefaultAdvanced = false
		}
		m.UpdateAdvancedOffset()
		return
	case plottingRowSection, plottingRowPlotInfo:
		return
	}

	opt := row.opt
	switch opt {
	case optUseDefaults:
		m.useDefaultAdvanced = !m.useDefaultAdvanced
		if m.useDefaultAdvanced {
			m.expandedOption = expandedNone
			m.subCursor = 0
		}
		m.UpdateAdvancedOffset()
		return

	case optPlotGroupDefaults:
		m.plotGroupDefaultsExpanded = !m.plotGroupDefaultsExpanded
		m.useDefaultAdvanced = false
	case optPlotGroupFonts:
		m.plotGroupFontsExpanded = !m.plotGroupFontsExpanded
		m.useDefaultAdvanced = false
	case optPlotGroupLayout:
		m.plotGroupLayoutExpanded = !m.plotGroupLayoutExpanded
		m.useDefaultAdvanced = false
	case optPlotGroupFigureSizes:
		m.plotGroupFigureSizesExpanded = !m.plotGroupFigureSizesExpanded
		m.useDefaultAdvanced = false
	case optPlotGroupColors:
		m.plotGroupColorsExpanded = !m.plotGroupColorsExpanded
		m.useDefaultAdvanced = false
	case optPlotGroupAlpha:
		m.plotGroupAlphaExpanded = !m.plotGroupAlphaExpanded
		m.useDefaultAdvanced = false
	case optPlotGroupScatter:
		m.plotGroupScatterExpanded = !m.plotGroupScatterExpanded
		m.useDefaultAdvanced = false
	case optPlotGroupBar:
		m.plotGroupBarExpanded = !m.plotGroupBarExpanded
		m.useDefaultAdvanced = false
	case optPlotGroupLine:
		m.plotGroupLineExpanded = !m.plotGroupLineExpanded
		m.useDefaultAdvanced = false
	case optPlotGroupHistogram:
		m.plotGroupHistogramExpanded = !m.plotGroupHistogramExpanded
		m.useDefaultAdvanced = false
	case optPlotGroupKDE:
		m.plotGroupKDEExpanded = !m.plotGroupKDEExpanded
		m.useDefaultAdvanced = false
	case optPlotGroupErrorbar:
		m.plotGroupErrorbarExpanded = !m.plotGroupErrorbarExpanded
		m.useDefaultAdvanced = false
	case optPlotGroupText:
		m.plotGroupTextExpanded = !m.plotGroupTextExpanded
		m.useDefaultAdvanced = false
	case optPlotGroupValidation:
		m.plotGroupValidationExpanded = !m.plotGroupValidationExpanded
		m.useDefaultAdvanced = false
	case optPlotGroupTopomap:
		m.plotGroupTopomapExpanded = !m.plotGroupTopomapExpanded
		m.useDefaultAdvanced = false
	case optPlotGroupTFR:
		m.plotGroupTFRExpanded = !m.plotGroupTFRExpanded
		m.useDefaultAdvanced = false
	case optPlotGroupSizing:
		m.plotGroupSizingExpanded = !m.plotGroupSizingExpanded
		m.useDefaultAdvanced = false
	case optPlotGroupSelection:
		m.plotGroupSelectionExpanded = !m.plotGroupSelectionExpanded
		m.useDefaultAdvanced = false

	case optPlotBboxInches:
		m.startTextEdit(textFieldPlotBboxInches)
		m.useDefaultAdvanced = false
	case optPlotFontFamily:
		m.startTextEdit(textFieldPlotFontFamily)
		m.useDefaultAdvanced = false
	case optPlotFontWeight:
		m.startTextEdit(textFieldPlotFontWeight)
		m.useDefaultAdvanced = false
	case optPlotLayoutTightRect:
		m.startTextEdit(textFieldPlotLayoutTightRect)
		m.useDefaultAdvanced = false
	case optPlotLayoutTightRectMicrostate:
		m.startTextEdit(textFieldPlotLayoutTightRectMicrostate)
		m.useDefaultAdvanced = false
	case optPlotGridSpecWidthRatios:
		m.startTextEdit(textFieldPlotGridSpecWidthRatios)
		m.useDefaultAdvanced = false
	case optPlotGridSpecHeightRatios:
		m.startTextEdit(textFieldPlotGridSpecHeightRatios)
		m.useDefaultAdvanced = false
	case optPlotFigureSizeStandard:
		m.startTextEdit(textFieldPlotFigureSizeStandard)
		m.useDefaultAdvanced = false
	case optPlotFigureSizeMedium:
		m.startTextEdit(textFieldPlotFigureSizeMedium)
		m.useDefaultAdvanced = false
	case optPlotFigureSizeSmall:
		m.startTextEdit(textFieldPlotFigureSizeSmall)
		m.useDefaultAdvanced = false
	case optPlotFigureSizeSquare:
		m.startTextEdit(textFieldPlotFigureSizeSquare)
		m.useDefaultAdvanced = false
	case optPlotFigureSizeWide:
		m.startTextEdit(textFieldPlotFigureSizeWide)
		m.useDefaultAdvanced = false
	case optPlotFigureSizeTFR:
		m.startTextEdit(textFieldPlotFigureSizeTFR)
		m.useDefaultAdvanced = false
	case optPlotFigureSizeTopomap:
		m.startTextEdit(textFieldPlotFigureSizeTopomap)
		m.useDefaultAdvanced = false

	case optPlotColorPain:
		m.startTextEdit(textFieldPlotColorPain)
		m.useDefaultAdvanced = false
	case optPlotColorNonpain:
		m.startTextEdit(textFieldPlotColorNonpain)
		m.useDefaultAdvanced = false
	case optPlotColorSignificant:
		m.startTextEdit(textFieldPlotColorSignificant)
		m.useDefaultAdvanced = false
	case optPlotColorNonsignificant:
		m.startTextEdit(textFieldPlotColorNonsignificant)
		m.useDefaultAdvanced = false
	case optPlotColorGray:
		m.startTextEdit(textFieldPlotColorGray)
		m.useDefaultAdvanced = false
	case optPlotColorLightGray:
		m.startTextEdit(textFieldPlotColorLightGray)
		m.useDefaultAdvanced = false
	case optPlotColorBlack:
		m.startTextEdit(textFieldPlotColorBlack)
		m.useDefaultAdvanced = false
	case optPlotColorBlue:
		m.startTextEdit(textFieldPlotColorBlue)
		m.useDefaultAdvanced = false
	case optPlotColorRed:
		m.startTextEdit(textFieldPlotColorRed)
		m.useDefaultAdvanced = false
	case optPlotColorNetworkNode:
		m.startTextEdit(textFieldPlotColorNetworkNode)
		m.useDefaultAdvanced = false

	case optPlotScatterEdgecolor:
		m.startTextEdit(textFieldPlotScatterEdgecolor)
		m.useDefaultAdvanced = false
	case optPlotHistEdgecolor:
		m.startTextEdit(textFieldPlotHistEdgecolor)
		m.useDefaultAdvanced = false
	case optPlotKdeColor:
		m.startTextEdit(textFieldPlotKdeColor)
		m.useDefaultAdvanced = false

	case optPlotTopomapContours,
		optPlotPadInches,
		optPlotFontSizeSmall,
		optPlotFontSizeMedium,
		optPlotFontSizeLarge,
		optPlotFontSizeTitle,
		optPlotFontSizeAnnotation,
		optPlotFontSizeLabel,
		optPlotFontSizeYLabel,
		optPlotFontSizeSuptitle,
		optPlotFontSizeFigureTitle,
		optPlotGridSpecHspace,
		optPlotGridSpecWspace,
		optPlotGridSpecLeft,
		optPlotGridSpecRight,
		optPlotGridSpecTop,
		optPlotGridSpecBottom,
		optPlotAlphaGrid,
		optPlotAlphaFill,
		optPlotAlphaCI,
		optPlotAlphaCILine,
		optPlotAlphaTextBox,
		optPlotAlphaViolinBody,
		optPlotAlphaRidgeFill,
		optPlotScatterMarkerSizeSmall,
		optPlotScatterMarkerSizeLarge,
		optPlotScatterMarkerSizeDefault,
		optPlotScatterAlpha,
		optPlotScatterEdgewidth,
		optPlotBarAlpha,
		optPlotBarWidth,
		optPlotBarCapsize,
		optPlotBarCapsizeLarge,
		optPlotLineWidthThin,
		optPlotLineWidthStandard,
		optPlotLineWidthThick,
		optPlotLineWidthBold,
		optPlotLineAlphaStandard,
		optPlotLineAlphaDim,
		optPlotLineAlphaZeroLine,
		optPlotLineAlphaFitLine,
		optPlotLineAlphaDiagonal,
		optPlotLineAlphaReference,
		optPlotLineRegressionWidth,
		optPlotLineResidualWidth,
		optPlotLineQQWidth,
		optPlotHistBins,
		optPlotHistBinsBehavioral,
		optPlotHistBinsResidual,
		optPlotHistBinsTFR,
		optPlotHistEdgewidth,
		optPlotHistAlpha,
		optPlotHistAlphaResidual,
		optPlotHistAlphaTFR,
		optPlotKdePoints,
		optPlotKdeLinewidth,
		optPlotKdeAlpha,
		optPlotErrorbarMarkersize,
		optPlotErrorbarCapsize,
		optPlotErrorbarCapsizeLarge,
		optPlotTextStatsX,
		optPlotTextStatsY,
		optPlotTextPvalueX,
		optPlotTextPvalueY,
		optPlotTextBootstrapX,
		optPlotTextBootstrapY,
		optPlotTextChannelAnnotationX,
		optPlotTextChannelAnnotationY,
		optPlotTextTitleY,
		optPlotTextResidualQcTitleY,
		optPlotValidationMinSamplesForPlot,
		optPlotValidationMinSamplesForKDE,
		optPlotValidationMinSamplesForFit,
		optPlotValidationMinSamplesForCalibration,
		optPlotValidationMinBinsForCalibration,
		optPlotValidationMaxBinsForCalibration,
		optPlotValidationSamplesPerBin,
		optPlotValidationMinRoisForFDR,
		optPlotValidationMinPvaluesForFDR,
		optPlotTopomapColorbarFraction,
		optPlotTopomapColorbarPad,
		optPlotTopomapSigMaskLinewidth,
		optPlotTopomapSigMaskMarkersize,
		optPlotTFRLogBase,
		optPlotTFRPercentageMultiplier,
		optPlotRoiWidthPerBand,
		optPlotRoiWidthPerMetric,
		optPlotRoiHeightPerRoi,
		optPlotPowerWidthPerBand,
		optPlotPowerHeightPerSegment,
		optPlotItpcWidthPerBin,
		optPlotItpcHeightPerBand,
		optPlotItpcWidthPerBandBox,
		optPlotItpcHeightBox,
		optPlotPacWidthPerRoi,
		optPlotPacHeightBox,
		optPlotAperiodicWidthPerColumn,
		optPlotAperiodicHeightPerRow,
		optPlotAperiodicNPerm,
		optPlotQualityWidthPerPlot,
		optPlotQualityHeightPerPlot,
		optPlotQualityDistributionNCols,
		optPlotQualityDistributionMaxFeatures,
		optPlotQualityOutlierZThreshold,
		optPlotQualityOutlierMaxFeatures,
		optPlotQualityOutlierMaxTrials,
		optPlotQualitySnrThresholdDb,
		optPlotComplexityWidthPerMeasure,
		optPlotComplexityHeightPerSegment,
		optPlotConnectivityWidthPerCircle,
		optPlotConnectivityWidthPerBand,
		optPlotConnectivityHeightPerMeasure,
		optPlotConnectivityCircleTopFraction,
		optPlotConnectivityCircleMinLines:
		m.startNumberEdit()
		m.useDefaultAdvanced = false

	case optPlotTopomapColormap:
		m.startTextEdit(textFieldPlotTopomapColormap)
		m.useDefaultAdvanced = false
	case optPlotTopomapSigMaskMarker:
		m.startTextEdit(textFieldPlotTopomapSigMaskMarker)
		m.useDefaultAdvanced = false
	case optPlotTopomapSigMaskMarkerFaceColor:
		m.startTextEdit(textFieldPlotTopomapSigMaskMarkerFaceColor)
		m.useDefaultAdvanced = false
	case optPlotTopomapSigMaskMarkerEdgeColor:
		m.startTextEdit(textFieldPlotTopomapSigMaskMarkerEdgeColor)
		m.useDefaultAdvanced = false
	case optPlotPacCmap:
		m.startTextEdit(textFieldPlotPacCmap)
		m.useDefaultAdvanced = false

	case optPlotTopomapDiffAnnotation:
		m.plotTopomapDiffAnnotation = cycleTriState(m.plotTopomapDiffAnnotation)
		m.useDefaultAdvanced = false
	case optPlotTopomapAnnotateDescriptive:
		m.plotTopomapAnnotateDesc = cycleTriState(m.plotTopomapAnnotateDesc)
		m.useDefaultAdvanced = false

	case optPlotPacPairs:
		m.startTextEdit(textFieldPlotPacPairs)
		m.useDefaultAdvanced = false
	case optPlotConnectivityMeasures:
		m.startTextEdit(textFieldPlotConnectivityMeasures)
		m.useDefaultAdvanced = false
	case optPlotSpectralMetrics:
		m.startTextEdit(textFieldPlotSpectralMetrics)
		m.useDefaultAdvanced = false
	case optPlotBurstsMetrics:
		m.startTextEdit(textFieldPlotBurstsMetrics)
		m.useDefaultAdvanced = false
	case optPlotAsymmetryStat:
		m.startTextEdit(textFieldPlotAsymmetryStat)
		m.useDefaultAdvanced = false
	case optPlotTemporalTimeBins:
		m.startTextEdit(textFieldPlotTemporalTimeBins)
		m.useDefaultAdvanced = false
	case optPlotTemporalTimeLabels:
		m.startTextEdit(textFieldPlotTemporalTimeLabels)
		m.useDefaultAdvanced = false
	}

	rows = m.getPlottingAdvancedRows()
	if len(rows) == 0 {
		m.advancedCursor = 0
		m.UpdateAdvancedOffset()
		return
	}
	if m.advancedCursor >= len(rows) {
		m.advancedCursor = len(rows) - 1
	}
	if m.advancedCursor < 0 {
		m.advancedCursor = 0
	}
	if rows[m.advancedCursor].kind == plottingRowSection || rows[m.advancedCursor].kind == plottingRowPlotInfo {
		m.advancedCursor = m.findNextPlottingAdvancedRow(m.advancedCursor, 1)
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
	case optBehaviorGroupReport:
		m.behaviorGroupReportExpanded = !m.behaviorGroupReportExpanded
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
	case optTrialTableOnlyMode:
		m.trialTableOnly = !m.trialTableOnly
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
		m.regressionOutcome = (m.regressionOutcome + 1) % 3
		m.useDefaultAdvanced = false
	case optRegressionIncludeTemperature:
		m.regressionIncludeTemperature = !m.regressionIncludeTemperature
		m.useDefaultAdvanced = false
	case optRegressionTempControl:
		m.regressionTempControl = (m.regressionTempControl + 1) % 3
		m.useDefaultAdvanced = false
	case optRegressionTempSplineKnots, optRegressionTempSplineQlow, optRegressionTempSplineQhigh, optRegressionTempSplineMinSamples:
		m.startNumberEdit()
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
		m.modelsTempControl = (m.modelsTempControl + 1) % 3
		m.useDefaultAdvanced = false
	case optModelsTempSplineKnots, optModelsTempSplineQlow, optModelsTempSplineQhigh, optModelsTempSplineMinSamples:
		m.startNumberEdit()
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
		if !m.modelsOutcomeRating && !m.modelsOutcomePainResidual && !m.modelsOutcomeTemperature && !m.modelsOutcomePainBinary {
			m.modelsOutcomeRating = true
		}
		m.useDefaultAdvanced = false
	case optModelsOutcomePainResidual:
		m.modelsOutcomePainResidual = !m.modelsOutcomePainResidual
		if !m.modelsOutcomeRating && !m.modelsOutcomePainResidual && !m.modelsOutcomeTemperature && !m.modelsOutcomePainBinary {
			m.modelsOutcomePainResidual = true
		}
		m.useDefaultAdvanced = false
	case optModelsOutcomeTemperature:
		m.modelsOutcomeTemperature = !m.modelsOutcomeTemperature
		if !m.modelsOutcomeRating && !m.modelsOutcomePainResidual && !m.modelsOutcomeTemperature && !m.modelsOutcomePainBinary {
			m.modelsOutcomeTemperature = true
		}
		m.useDefaultAdvanced = false
	case optModelsOutcomePainBinary:
		m.modelsOutcomePainBinary = !m.modelsOutcomePainBinary
		if !m.modelsOutcomeRating && !m.modelsOutcomePainResidual && !m.modelsOutcomeTemperature && !m.modelsOutcomePainBinary {
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
		if !m.influenceOutcomeRating && !m.influenceOutcomePainResidual && !m.influenceOutcomeTemperature {
			m.influenceOutcomeRating = true
		}
		m.useDefaultAdvanced = false
	case optInfluenceOutcomePainResidual:
		m.influenceOutcomePainResidual = !m.influenceOutcomePainResidual
		if !m.influenceOutcomeRating && !m.influenceOutcomePainResidual && !m.influenceOutcomeTemperature {
			m.influenceOutcomePainResidual = true
		}
		m.useDefaultAdvanced = false
	case optInfluenceOutcomeTemperature:
		m.influenceOutcomeTemperature = !m.influenceOutcomeTemperature
		if !m.influenceOutcomeRating && !m.influenceOutcomePainResidual && !m.influenceOutcomeTemperature {
			m.influenceOutcomeTemperature = true
		}
		m.useDefaultAdvanced = false
	case optInfluenceMaxFeatures:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optInfluenceIncludeTemperature:
		m.influenceIncludeTemperature = !m.influenceIncludeTemperature
		m.useDefaultAdvanced = false
	case optInfluenceTempControl:
		m.influenceTempControl = (m.influenceTempControl + 1) % 3
		m.useDefaultAdvanced = false
	case optInfluenceTempSplineKnots, optInfluenceTempSplineQlow, optInfluenceTempSplineQhigh, optInfluenceTempSplineMinSamples:
		m.startNumberEdit()
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

	// Report
	case optReportTopN:
		m.startNumberEdit()
		m.useDefaultAdvanced = false

	// Correlations
	case optCorrelationsFeatureSet:
		m.correlationsFeatureSet = (m.correlationsFeatureSet + 1) % 2
		m.useDefaultAdvanced = false
	case optCorrelationsTargetRating:
		m.correlationsTargetRating = !m.correlationsTargetRating
		if !m.correlationsTargetRating && !m.correlationsTargetTemperature && !m.correlationsTargetPainResidual {
			m.correlationsTargetRating = true
		}
		m.useDefaultAdvanced = false
	case optCorrelationsTargetTemperature:
		m.correlationsTargetTemperature = !m.correlationsTargetTemperature
		if !m.correlationsTargetRating && !m.correlationsTargetTemperature && !m.correlationsTargetPainResidual {
			m.correlationsTargetTemperature = true
		}
		m.useDefaultAdvanced = false
	case optCorrelationsTargetPainResidual:
		m.correlationsTargetPainResidual = !m.correlationsTargetPainResidual
		if !m.correlationsTargetRating && !m.correlationsTargetTemperature && !m.correlationsTargetPainResidual {
			m.correlationsTargetPainResidual = true
		}
		m.useDefaultAdvanced = false

	// Pain sensitivity
	case optPainSensitivityFeatureSet:
		m.painSensitivityFeatureSet = (m.painSensitivityFeatureSet + 1) % 2
		m.useDefaultAdvanced = false
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
	case optDecodingNPerm, optDecodingInnerSplits, optDecodingMinTrialsInner, optRNGSeed, optRfNEstimators:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optDecodingSkipTimeGen:
		m.skipTimeGen = !m.skipTimeGen
		m.useDefaultAdvanced = false
	case optElasticNetAlphaGrid:
		m.startTextEdit(textFieldElasticNetAlphaGrid)
		m.useDefaultAdvanced = false
	case optElasticNetL1RatioGrid:
		m.startTextEdit(textFieldElasticNetL1RatioGrid)
		m.useDefaultAdvanced = false
	case optRfMaxDepthGrid:
		m.startTextEdit(textFieldRfMaxDepthGrid)
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
	case optIcaLabelsToKeep:
		m.startTextEdit(textFieldIcaLabelsToKeep)
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
	case types.PipelinePlotting:
		m.commitPlottingNumber(val)
	case types.PipelineDecoding:
		m.commitDecodingNumber(val)
	case types.PipelinePreprocessing:
		m.commitPreprocessingNumber(val)
	case types.PipelineRawToBIDS:
		m.commitRawToBidsNumber(val)
	}
	m.useDefaultAdvanced = false
}

func (m *Model) commitPlottingNumber(val float64) {
	rows := m.getPlottingAdvancedRows()
	if m.advancedCursor < 0 || m.advancedCursor >= len(rows) {
		return
	}
	if rows[m.advancedCursor].kind != plottingRowOption {
		return
	}

	opt := rows[m.advancedCursor].opt
	switch opt {
	case optPlotPadInches:
		if val >= 0 {
			m.plotPadInches = val
		}

	case optPlotFontSizeSmall:
		if val >= 0 {
			m.plotFontSizeSmall = int(val)
		}
	case optPlotFontSizeMedium:
		if val >= 0 {
			m.plotFontSizeMedium = int(val)
		}
	case optPlotFontSizeLarge:
		if val >= 0 {
			m.plotFontSizeLarge = int(val)
		}
	case optPlotFontSizeTitle:
		if val >= 0 {
			m.plotFontSizeTitle = int(val)
		}
	case optPlotFontSizeAnnotation:
		if val >= 0 {
			m.plotFontSizeAnnotation = int(val)
		}
	case optPlotFontSizeLabel:
		if val >= 0 {
			m.plotFontSizeLabel = int(val)
		}
	case optPlotFontSizeYLabel:
		if val >= 0 {
			m.plotFontSizeYLabel = int(val)
		}
	case optPlotFontSizeSuptitle:
		if val >= 0 {
			m.plotFontSizeSuptitle = int(val)
		}
	case optPlotFontSizeFigureTitle:
		if val >= 0 {
			m.plotFontSizeFigureTitle = int(val)
		}

	case optPlotGridSpecHspace:
		if val >= 0 {
			m.plotGridSpecHspace = val
		}
	case optPlotGridSpecWspace:
		if val >= 0 {
			m.plotGridSpecWspace = val
		}
	case optPlotGridSpecLeft:
		if val == 0 || (val > 0 && val <= 1) {
			m.plotGridSpecLeft = val
		}
	case optPlotGridSpecRight:
		if val == 0 || (val > 0 && val <= 1) {
			m.plotGridSpecRight = val
		}
	case optPlotGridSpecTop:
		if val == 0 || (val > 0 && val <= 1) {
			m.plotGridSpecTop = val
		}
	case optPlotGridSpecBottom:
		if val == 0 || (val > 0 && val <= 1) {
			m.plotGridSpecBottom = val
		}

	case optPlotAlphaGrid:
		if val == 0 || (val > 0 && val <= 1) {
			m.plotAlphaGrid = val
		}
	case optPlotAlphaFill:
		if val == 0 || (val > 0 && val <= 1) {
			m.plotAlphaFill = val
		}
	case optPlotAlphaCI:
		if val == 0 || (val > 0 && val <= 1) {
			m.plotAlphaCI = val
		}
	case optPlotAlphaCILine:
		if val == 0 || (val > 0 && val <= 1) {
			m.plotAlphaCILine = val
		}
	case optPlotAlphaTextBox:
		if val == 0 || (val > 0 && val <= 1) {
			m.plotAlphaTextBox = val
		}
	case optPlotAlphaViolinBody:
		if val == 0 || (val > 0 && val <= 1) {
			m.plotAlphaViolinBody = val
		}
	case optPlotAlphaRidgeFill:
		if val == 0 || (val > 0 && val <= 1) {
			m.plotAlphaRidgeFill = val
		}

	case optPlotScatterMarkerSizeSmall:
		if val >= 0 {
			m.plotScatterMarkerSizeSmall = int(val)
		}
	case optPlotScatterMarkerSizeLarge:
		if val >= 0 {
			m.plotScatterMarkerSizeLarge = int(val)
		}
	case optPlotScatterMarkerSizeDefault:
		if val >= 0 {
			m.plotScatterMarkerSizeDefault = int(val)
		}
	case optPlotScatterAlpha:
		if val == 0 || (val > 0 && val <= 1) {
			m.plotScatterAlpha = val
		}
	case optPlotScatterEdgewidth:
		if val >= 0 {
			m.plotScatterEdgeWidth = val
		}

	case optPlotBarAlpha:
		if val == 0 || (val > 0 && val <= 1) {
			m.plotBarAlpha = val
		}
	case optPlotBarWidth:
		if val >= 0 {
			m.plotBarWidth = val
		}
	case optPlotBarCapsize:
		if val >= 0 {
			m.plotBarCapsize = int(val)
		}
	case optPlotBarCapsizeLarge:
		if val >= 0 {
			m.plotBarCapsizeLarge = int(val)
		}

	case optPlotLineWidthThin:
		if val >= 0 {
			m.plotLineWidthThin = val
		}
	case optPlotLineWidthStandard:
		if val >= 0 {
			m.plotLineWidthStandard = val
		}
	case optPlotLineWidthThick:
		if val >= 0 {
			m.plotLineWidthThick = val
		}
	case optPlotLineWidthBold:
		if val >= 0 {
			m.plotLineWidthBold = val
		}
	case optPlotLineAlphaStandard:
		if val == 0 || (val > 0 && val <= 1) {
			m.plotLineAlphaStandard = val
		}
	case optPlotLineAlphaDim:
		if val == 0 || (val > 0 && val <= 1) {
			m.plotLineAlphaDim = val
		}
	case optPlotLineAlphaZeroLine:
		if val == 0 || (val > 0 && val <= 1) {
			m.plotLineAlphaZeroLine = val
		}
	case optPlotLineAlphaFitLine:
		if val == 0 || (val > 0 && val <= 1) {
			m.plotLineAlphaFitLine = val
		}
	case optPlotLineAlphaDiagonal:
		if val == 0 || (val > 0 && val <= 1) {
			m.plotLineAlphaDiagonal = val
		}
	case optPlotLineAlphaReference:
		if val == 0 || (val > 0 && val <= 1) {
			m.plotLineAlphaReference = val
		}
	case optPlotLineRegressionWidth:
		if val >= 0 {
			m.plotLineRegressionWidth = val
		}
	case optPlotLineResidualWidth:
		if val >= 0 {
			m.plotLineResidualWidth = val
		}
	case optPlotLineQQWidth:
		if val >= 0 {
			m.plotLineQQWidth = val
		}

	case optPlotHistBins:
		if val >= 0 {
			m.plotHistBins = int(val)
		}
	case optPlotHistBinsBehavioral:
		if val >= 0 {
			m.plotHistBinsBehavioral = int(val)
		}
	case optPlotHistBinsResidual:
		if val >= 0 {
			m.plotHistBinsResidual = int(val)
		}
	case optPlotHistBinsTFR:
		if val >= 0 {
			m.plotHistBinsTFR = int(val)
		}
	case optPlotHistEdgewidth:
		if val >= 0 {
			m.plotHistEdgeWidth = val
		}
	case optPlotHistAlpha:
		if val == 0 || (val > 0 && val <= 1) {
			m.plotHistAlpha = val
		}
	case optPlotHistAlphaResidual:
		if val == 0 || (val > 0 && val <= 1) {
			m.plotHistAlphaResidual = val
		}
	case optPlotHistAlphaTFR:
		if val == 0 || (val > 0 && val <= 1) {
			m.plotHistAlphaTFR = val
		}

	case optPlotKdePoints:
		if val >= 0 {
			m.plotKdePoints = int(val)
		}
	case optPlotKdeLinewidth:
		if val >= 0 {
			m.plotKdeLinewidth = val
		}
	case optPlotKdeAlpha:
		if val == 0 || (val > 0 && val <= 1) {
			m.plotKdeAlpha = val
		}

	case optPlotErrorbarMarkersize:
		if val >= 0 {
			m.plotErrorbarMarkerSize = int(val)
		}
	case optPlotErrorbarCapsize:
		if val >= 0 {
			m.plotErrorbarCapsize = int(val)
		}
	case optPlotErrorbarCapsizeLarge:
		if val >= 0 {
			m.plotErrorbarCapsizeLarge = int(val)
		}

	case optPlotTextStatsX:
		m.plotTextStatsX = val
	case optPlotTextStatsY:
		m.plotTextStatsY = val
	case optPlotTextPvalueX:
		m.plotTextPvalueX = val
	case optPlotTextPvalueY:
		m.plotTextPvalueY = val
	case optPlotTextBootstrapX:
		m.plotTextBootstrapX = val
	case optPlotTextBootstrapY:
		m.plotTextBootstrapY = val
	case optPlotTextChannelAnnotationX:
		m.plotTextChannelAnnotationX = val
	case optPlotTextChannelAnnotationY:
		m.plotTextChannelAnnotationY = val
	case optPlotTextTitleY:
		m.plotTextTitleY = val
	case optPlotTextResidualQcTitleY:
		m.plotTextResidualQcTitleY = val

	case optPlotValidationMinSamplesForPlot:
		if val >= 0 {
			m.plotValidationMinSamplesForPlot = int(val)
		}
	case optPlotValidationMinSamplesForKDE:
		if val >= 0 {
			m.plotValidationMinSamplesForKDE = int(val)
		}
	case optPlotValidationMinSamplesForFit:
		if val >= 0 {
			m.plotValidationMinSamplesForFit = int(val)
		}
	case optPlotValidationMinSamplesForCalibration:
		if val >= 0 {
			m.plotValidationMinSamplesForCalibration = int(val)
		}
	case optPlotValidationMinBinsForCalibration:
		if val >= 0 {
			m.plotValidationMinBinsForCalibration = int(val)
		}
	case optPlotValidationMaxBinsForCalibration:
		if val >= 0 {
			m.plotValidationMaxBinsForCalibration = int(val)
		}
	case optPlotValidationSamplesPerBin:
		if val >= 0 {
			m.plotValidationSamplesPerBin = int(val)
		}
	case optPlotValidationMinRoisForFDR:
		if val >= 0 {
			m.plotValidationMinRoisForFDR = int(val)
		}
	case optPlotValidationMinPvaluesForFDR:
		if val >= 0 {
			m.plotValidationMinPvaluesForFDR = int(val)
		}

	case optPlotTopomapSigMaskLinewidth:
		if val >= 0 {
			m.plotTopomapSigMaskLinewidth = val
		}
	case optPlotTopomapSigMaskMarkersize:
		if val >= 0 {
			m.plotTopomapSigMaskMarkerSize = val
		}

	case optPlotTopomapContours:
		if val >= 0 {
			m.plotTopomapContours = int(val)
		}
	case optPlotTopomapColorbarFraction:
		// Allow 0 to reset to default; otherwise constrain to [0,1].
		if val == 0 || (val > 0 && val <= 1) {
			m.plotTopomapColorbarFraction = val
		}
	case optPlotTopomapColorbarPad:
		if val == 0 || (val > 0 && val <= 1) {
			m.plotTopomapColorbarPad = val
		}

	case optPlotTFRLogBase:
		if val <= 0 {
			m.plotTFRLogBase = 0
		} else {
			m.plotTFRLogBase = val
		}
	case optPlotTFRPercentageMultiplier:
		if val <= 0 {
			m.plotTFRPercentageMultiplier = 0
		} else {
			m.plotTFRPercentageMultiplier = val
		}

	case optPlotRoiWidthPerBand:
		if val >= 0 {
			m.plotRoiWidthPerBand = val
		}
	case optPlotRoiWidthPerMetric:
		if val >= 0 {
			m.plotRoiWidthPerMetric = val
		}
	case optPlotRoiHeightPerRoi:
		if val >= 0 {
			m.plotRoiHeightPerRoi = val
		}

	case optPlotPowerWidthPerBand:
		if val >= 0 {
			m.plotPowerWidthPerBand = val
		}
	case optPlotPowerHeightPerSegment:
		if val >= 0 {
			m.plotPowerHeightPerSegment = val
		}

	case optPlotItpcWidthPerBin:
		if val >= 0 {
			m.plotItpcWidthPerBin = val
		}
	case optPlotItpcHeightPerBand:
		if val >= 0 {
			m.plotItpcHeightPerBand = val
		}
	case optPlotItpcWidthPerBandBox:
		if val >= 0 {
			m.plotItpcWidthPerBandBox = val
		}
	case optPlotItpcHeightBox:
		if val >= 0 {
			m.plotItpcHeightBox = val
		}

	case optPlotPacWidthPerRoi:
		if val >= 0 {
			m.plotPacWidthPerRoi = val
		}
	case optPlotPacHeightBox:
		if val >= 0 {
			m.plotPacHeightBox = val
		}

	case optPlotAperiodicWidthPerColumn:
		if val >= 0 {
			m.plotAperiodicWidthPerColumn = val
		}
	case optPlotAperiodicHeightPerRow:
		if val >= 0 {
			m.plotAperiodicHeightPerRow = val
		}
	case optPlotAperiodicNPerm:
		if val >= 0 {
			m.plotAperiodicNPerm = int(val)
		}

	case optPlotQualityWidthPerPlot:
		if val >= 0 {
			m.plotQualityWidthPerPlot = val
		}
	case optPlotQualityHeightPerPlot:
		if val >= 0 {
			m.plotQualityHeightPerPlot = val
		}
	case optPlotQualityDistributionNCols:
		if val >= 0 {
			m.plotQualityDistributionNCols = int(val)
		}
	case optPlotQualityDistributionMaxFeatures:
		if val >= 0 {
			m.plotQualityDistributionMaxFeatures = int(val)
		}
	case optPlotQualityOutlierZThreshold:
		if val >= 0 {
			m.plotQualityOutlierZThreshold = val
		}
	case optPlotQualityOutlierMaxFeatures:
		if val >= 0 {
			m.plotQualityOutlierMaxFeatures = int(val)
		}
	case optPlotQualityOutlierMaxTrials:
		if val >= 0 {
			m.plotQualityOutlierMaxTrials = int(val)
		}
	case optPlotQualitySnrThresholdDb:
		m.plotQualitySnrThresholdDb = val

	case optPlotComplexityWidthPerMeasure:
		if val >= 0 {
			m.plotComplexityWidthPerMeasure = val
		}
	case optPlotComplexityHeightPerSegment:
		if val >= 0 {
			m.plotComplexityHeightPerSegment = val
		}

	case optPlotConnectivityWidthPerCircle:
		if val >= 0 {
			m.plotConnectivityWidthPerCircle = val
		}
	case optPlotConnectivityWidthPerBand:
		if val >= 0 {
			m.plotConnectivityWidthPerBand = val
		}
	case optPlotConnectivityHeightPerMeasure:
		if val >= 0 {
			m.plotConnectivityHeightPerMeasure = val
		}
	case optPlotConnectivityCircleTopFraction:
		if val == 0 || (val > 0 && val <= 1) {
			m.plotConnectivityCircleTopFraction = val
		}
	case optPlotConnectivityCircleMinLines:
		if val >= 0 {
			m.plotConnectivityCircleMinLines = int(val)
		}
	}
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
	// TFR options
	case optTfrFreqMin:
		if val >= 0 {
			m.tfrFreqMin = val
		}
	case optTfrFreqMax:
		if val > 0 {
			m.tfrFreqMax = val
		}
	case optTfrNFreqs:
		if val >= 1 {
			m.tfrNFreqs = int(val)
		}
	case optTfrMinCycles:
		if val >= 1 {
			m.tfrMinCycles = val
		}
	case optTfrNCyclesFactor:
		if val >= 0.5 {
			m.tfrNCyclesFactor = val
		}
	case optTfrDecim:
		if val >= 1 {
			m.tfrDecim = int(val)
		}
	case optTfrWorkers:
		m.tfrWorkers = int(val)
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
	case optRegressionTempSplineKnots:
		if val >= 4 {
			m.regressionTempSplineKnots = int(val)
		}
	case optRegressionTempSplineQlow:
		if val > 0 && val < 1 {
			m.regressionTempSplineQlow = val
		}
	case optRegressionTempSplineQhigh:
		if val > 0 && val < 1 {
			m.regressionTempSplineQhigh = val
		}
	case optRegressionTempSplineMinSamples:
		if val >= 0 {
			m.regressionTempSplineMinN = int(val)
		}
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
	case optModelsTempSplineKnots:
		if val >= 4 {
			m.modelsTempSplineKnots = int(val)
		}
	case optModelsTempSplineQlow:
		if val > 0 && val < 1 {
			m.modelsTempSplineQlow = val
		}
	case optModelsTempSplineQhigh:
		if val > 0 && val < 1 {
			m.modelsTempSplineQhigh = val
		}
	case optModelsTempSplineMinSamples:
		if val >= 0 {
			m.modelsTempSplineMinN = int(val)
		}
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
	case optInfluenceTempSplineKnots:
		if val >= 4 {
			m.influenceTempSplineKnots = int(val)
		}
	case optInfluenceTempSplineQlow:
		if val > 0 && val < 1 {
			m.influenceTempSplineQlow = val
		}
	case optInfluenceTempSplineQhigh:
		if val > 0 && val < 1 {
			m.influenceTempSplineQhigh = val
		}
	case optInfluenceTempSplineMinSamples:
		if val >= 0 {
			m.influenceTempSplineMinN = int(val)
		}
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
	case optReportTopN:
		if val >= 1 {
			m.reportTopN = int(val)
		}
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
	case optDecodingMinTrialsInner:
		if val >= 1 {
			m.decodingMinTrialsInner = int(val)
		}
	case optRNGSeed:
		if val >= 0 {
			m.rngSeed = int(val)
		}
	case optRfNEstimators:
		if val >= 1 {
			m.rfNEstimators = int(val)
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

func (m Model) findNextFeaturePlotter(current int, delta int) int {
	items := m.featurePlotterItems()
	if len(items) == 0 {
		return 0
	}
	next := current
	if next < 0 {
		next = len(items) - 1
	}
	return (next + delta + len(items)) % len(items)
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
