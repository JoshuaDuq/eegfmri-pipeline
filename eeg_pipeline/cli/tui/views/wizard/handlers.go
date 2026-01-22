package wizard

import (
	"fmt"
	"strconv"
	"strings"

	"github.com/eeg-pipeline/tui/executor"
	"github.com/eeg-pipeline/tui/styles"
	"github.com/eeg-pipeline/tui/types"

	tea "github.com/charmbracelet/bubbletea"
)

// browseForFile opens a file picker dialog for the specified field
func (m *Model) browseForFile(prompt, field string, fileTypeDesc, extensions string) tea.Cmd {
	return executor.PickFile(prompt, field, fileTypeDesc, extensions)
}

const (
	minSubjectsRequired   = 1
	minSubjectsForGroupCV = 2
	timeRangeFieldCount   = 3
)

///////////////////////////////////////////////////////////////////
// Cursor Reset Helper
///////////////////////////////////////////////////////////////////

// resetCursorsForStep resets all cursor positions when entering a new step
// to prevent UI state from persisting incorrectly between steps
func (m *Model) resetCursorsForStep() {
	m.categoryIndex = 0
	m.subjectCursor = 0
	m.computationCursor = 0
	m.bandCursor = 0
	m.roiCursor = 0
	m.spatialCursor = 0
	m.featureFileCursor = 0
	m.advancedCursor = 0
	m.advancedOffset = 0
	m.subCursor = 0
	m.expandedOption = expandedNone
	m.plotCursor = 0
	m.plotOffset = 0
	if m.CurrentStep == types.StepSelectPlots {
		m.plotCursor = m.findNextVisiblePlot(-1, 1)
	}
	m.featurePlotterCursor = 0
	m.featurePlotterOffset = 0
	if m.CurrentStep == types.StepSelectFeaturePlotters {
		m.featurePlotterCursor = m.findNextFeaturePlotter(-1, 1)
	}
	m.plotConfigCursor = 0

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

// moveCursorInList moves cursor within a list with wraparound
func moveCursorInList(current int, delta int, listLength int) int {
	if listLength == 0 {
		return 0
	}
	return (current + delta + listLength) % listLength
}

// clampCursor ensures cursor is within valid bounds
func clampCursor(cursor int, maxIndex int) int {
	if cursor < 0 {
		return 0
	}
	if cursor > maxIndex {
		return maxIndex
	}
	return cursor
}

// shouldSkipStep determines if a step should be skipped based on pipeline and mode
func (m *Model) shouldSkipStep(step types.WizardStep) bool {
	switch m.Pipeline {
	case types.PipelineFeatures:
		mode := m.modeOptions[m.modeIndex]
		switch mode {
		case "combine":
			// For combine, only Subjects are needed
			return step != types.StepSelectSubjects && step != types.StepSelectMode
		case styles.ModeVisualize:
			// For visualize, skip bands, ROIs, spatial, time, and advanced config
			return step == types.StepSelectBands || step == types.StepSelectROIs || step == types.StepSelectSpatial || step == types.StepTimeRange || step == types.StepAdvancedConfig
		}
	case types.PipelineBehavior:
		mode := m.modeOptions[m.modeIndex]
		if mode == styles.ModeVisualize {
			// For visualize, skip computations selection, features selection, and advanced config
			return step == types.StepSelectComputations || step == types.StepSelectFeatureFiles || step == types.StepAdvancedConfig
		}
	case types.PipelinePlotting:
		if step == types.StepSelectFeaturePlotters && len(m.selectedFeaturePlotterCategories()) == 0 {
			return true
		}
	}
	return false
}

func (m *Model) handleUp() {
	switch m.CurrentStep {
	case types.StepSelectMode:
		m.modeIndex = moveCursorInList(m.modeIndex, -1, len(m.modeOptions))

	case types.StepSelectComputations:
		m.computationCursor = moveCursorInList(m.computationCursor, -1, len(m.computations))
	case types.StepConfigureOptions, types.StepSelectPlotCategories:
		if m.showGlobalStyling && m.CurrentStep == types.StepSelectPlotCategories {
			options := m.getGlobalStylingOptions()
			if len(options) > 0 {
				m.globalStylingCursor = moveCursorInList(m.globalStylingCursor, -1, len(options))
			}
		} else {
			m.categoryIndex = moveCursorInList(m.categoryIndex, -1, len(m.categories))
		}
	case types.StepSelectSubjects:
		if len(m.subjects) > 0 {
			m.subjectCursor = moveCursorInList(m.subjectCursor, -1, len(m.subjects))
		}
	case types.StepSelectBands:
		m.bandCursor = moveCursorInList(m.bandCursor, -1, len(m.bands))
	case types.StepSelectROIs:
		m.roiCursor = moveCursorInList(m.roiCursor, -1, len(m.rois))
	case types.StepSelectFeatureFiles:
		applicable := m.GetApplicableFeatureFiles()
		if len(applicable) > 0 {
			m.featureFileCursor = moveCursorInList(m.featureFileCursor, -1, len(applicable))
		}
	case types.StepSelectPlots:
		m.plotCursor = m.findNextVisiblePlot(m.plotCursor, -1)
	case types.StepSelectFeaturePlotters:
		m.featurePlotterCursor = m.findNextFeaturePlotter(m.featurePlotterCursor, -1)
	case types.StepPlotConfig:
		options := m.getPlotConfigOptions()
		if len(options) > 0 {
			m.plotConfigCursor = moveCursorInList(m.plotConfigCursor, -1, len(options))
		}
	case types.StepSelectSpatial:
		m.spatialCursor = moveCursorInList(m.spatialCursor, -1, len(spatialModes))
	case types.StepTimeRange:
		if m.editingRangeIdx >= 0 {
			m.editingField = moveCursorInList(m.editingField, -1, timeRangeFieldCount)
		} else if len(m.TimeRanges) > 0 {
			m.timeRangeCursor = moveCursorInList(m.timeRangeCursor, -1, len(m.TimeRanges))
		}
	case types.StepSelectPreprocessingStages:
		m.prepStageCursor = moveCursorInList(m.prepStageCursor, -1, len(m.prepStages))
	case types.StepPreprocessingFiltering, types.StepPreprocessingICA, types.StepPreprocessingEpochs:
		m.advancedCursor = moveCursorInList(m.advancedCursor, -1, 5)
	case types.StepAdvancedConfig:
		if m.expandedOption >= 0 {
			listLen := m.getExpandedListLength()
			if listLen > 0 {
				m.subCursor = moveCursorInList(m.subCursor, -1, listLen)
			}
			m.UpdateAdvancedOffset()
		} else {
			if m.Pipeline == types.PipelinePlotting {
				m.advancedCursor = m.findNextPlottingAdvancedRow(m.advancedCursor, -1)
			} else {
				optCount := m.getAdvancedOptionCount()
				m.advancedCursor = moveCursorInList(m.advancedCursor, -1, optCount)
			}
			m.UpdateAdvancedOffset()
		}
	}
}

func (m *Model) handleDown() {
	switch m.CurrentStep {
	case types.StepSelectMode:
		m.modeIndex = moveCursorInList(m.modeIndex, 1, len(m.modeOptions))

	case types.StepSelectComputations:
		m.computationCursor = moveCursorInList(m.computationCursor, 1, len(m.computations))
	case types.StepConfigureOptions, types.StepSelectPlotCategories:
		if m.showGlobalStyling && m.CurrentStep == types.StepSelectPlotCategories {
			options := m.getGlobalStylingOptions()
			if len(options) > 0 {
				m.globalStylingCursor = moveCursorInList(m.globalStylingCursor, 1, len(options))
			}
		} else {
			m.categoryIndex = moveCursorInList(m.categoryIndex, 1, len(m.categories))
		}
	case types.StepSelectSubjects:
		m.subjectCursor = moveCursorInList(m.subjectCursor, 1, len(m.subjects))
	case types.StepSelectBands:
		m.bandCursor = moveCursorInList(m.bandCursor, 1, len(m.bands))
	case types.StepSelectROIs:
		m.roiCursor = moveCursorInList(m.roiCursor, 1, len(m.rois))
	case types.StepSelectFeatureFiles:
		applicable := m.GetApplicableFeatureFiles()
		m.featureFileCursor = moveCursorInList(m.featureFileCursor, 1, len(applicable))
	case types.StepSelectPlots:
		m.plotCursor = m.findNextVisiblePlot(m.plotCursor, 1)
	case types.StepSelectFeaturePlotters:
		m.featurePlotterCursor = m.findNextFeaturePlotter(m.featurePlotterCursor, 1)
	case types.StepPlotConfig:
		options := m.getPlotConfigOptions()
		if len(options) > 0 {
			m.plotConfigCursor = moveCursorInList(m.plotConfigCursor, 1, len(options))
		}
	case types.StepSelectSpatial:
		m.spatialCursor = moveCursorInList(m.spatialCursor, 1, len(spatialModes))
	case types.StepTimeRange:
		if m.editingRangeIdx >= 0 {
			m.editingField = moveCursorInList(m.editingField, 1, timeRangeFieldCount)
		} else if len(m.TimeRanges) > 0 {
			m.timeRangeCursor = moveCursorInList(m.timeRangeCursor, 1, len(m.TimeRanges))
		}
	case types.StepSelectPreprocessingStages:
		m.prepStageCursor = moveCursorInList(m.prepStageCursor, 1, len(m.prepStages))
	case types.StepPreprocessingFiltering, types.StepPreprocessingICA, types.StepPreprocessingEpochs:
		m.advancedCursor = moveCursorInList(m.advancedCursor, 1, 5)
	case types.StepAdvancedConfig:
		if m.expandedOption >= 0 {
			listLen := m.getExpandedListLength()
			if listLen > 0 {
				m.subCursor = moveCursorInList(m.subCursor, 1, listLen)
			}
			m.UpdateAdvancedOffset()
		} else {
			if m.Pipeline == types.PipelinePlotting {
				m.advancedCursor = m.findNextPlottingAdvancedRow(m.advancedCursor, 1)
			} else {
				optCount := m.getAdvancedOptionCount()
				m.advancedCursor = moveCursorInList(m.advancedCursor, 1, optCount)
			}
			m.UpdateAdvancedOffset()
		}
	}
}

func (m *Model) handleTab() {
	switch m.CurrentStep {
	case types.StepSelectSubjects:
		if m.Pipeline == types.PipelineML {
			if m.mlScope == MLCVScopeGroup {
				m.mlScope = MLCVScopeSubject
			} else {
				m.mlScope = MLCVScopeGroup
			}
			return
		}
	case types.StepAdvancedConfig:
		if m.expandedOption >= 0 {
			m.expandedOption = expandedNone
			m.subCursor = 0
			m.editingPlotID = ""
			m.editingPlotField = plotItemConfigFieldNone
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

func (m Model) handleEnter() (tea.Model, tea.Cmd) {
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

		for m.stepIndex < len(m.steps)-1 {
			if !m.shouldSkipStep(m.CurrentStep) {
				break
			}
			m.stepIndex++
			m.CurrentStep = m.steps[m.stepIndex]
		}

		if m.CurrentStep == types.StepConfigureOptions {
			m.updateFeatureAvailability()
		}

		m.resetCursorsForStep()
	} else {
		m.validationErrors = m.validate()
		if len(m.validationErrors) == 0 {
			m.ConfirmingExecute = true
		}
	}
	return m, tea.ClearScreen
}

// countSelectedItems counts the number of selected items in a map
func countSelectedItems(selected map[int]bool) int {
	count := 0
	for _, sel := range selected {
		if sel {
			count++
		}
	}
	return count
}

// countSelectedStringItems counts the number of selected items in a string-keyed map
func countSelectedStringItems(selected map[string]bool) int {
	count := 0
	for _, sel := range selected {
		if sel {
			count++
		}
	}
	return count
}

func (m *Model) validateStep() []string {
	var errors []string
	switch m.CurrentStep {

	case types.StepSelectSubjects:
		selectedCount := countSelectedStringItems(m.subjectSelected)
		validCount := 0
		for subjID, selected := range m.subjectSelected {
			if !selected {
				continue
			}
			for _, s := range m.subjects {
				if s.ID != subjID {
					continue
				}
				valid, _ := m.Pipeline.ValidateSubject(s)
				if m.Pipeline == types.PipelinePlotting {
					valid, _ = m.validatePlottingSubject(s)
				}
				if valid {
					validCount++
				}
				break
			}
		}
		minRequired := minSubjectsRequired
		if m.Pipeline == types.PipelineML && m.mlScope == MLCVScopeGroup {
			minRequired = minSubjectsForGroupCV
		}
		if selectedCount < minRequired {
			errors = append(errors, fmt.Sprintf("Select at least %d subject(s)", minRequired))
		}
		if validCount < minRequired {
			errors = append(errors, fmt.Sprintf("Select at least %d valid subject(s)", minRequired))
		}
	case types.StepSelectComputations:
		totalCount := countSelectedItems(m.computationSelected)
		if totalCount == 0 {
			errors = append(errors, "Select at least one analysis to run")
		}
	case types.StepSelectFeatureFiles:
		count := countSelectedStringItems(m.featureFileSelected)
		if count == 0 {
			errors = append(errors, "Select at least one feature file to load")
		}
	case types.StepConfigureOptions, types.StepSelectPlotCategories:
		count := countSelectedItems(m.selected)
		if count == 0 {
			errors = append(errors, "Select at least one category")
		}
	case types.StepSelectBands:
		count := countSelectedItems(m.bandSelected)
		if count == 0 {
			errors = append(errors, "Select at least one frequency band")
		}
	case types.StepSelectPlots:
		count := countSelectedItems(m.plotSelected)
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
		count := countSelectedItems(m.spatialSelected)
		if count == 0 {
			errors = append(errors, "Select at least one spatial mode")
		}
	case types.StepSelectPreprocessingStages:
		count := countSelectedItems(m.prepStageSelected)
		if count == 0 {
			errors = append(errors, "Select at least one preprocessing stage")
		}
	case types.StepTimeRange:
		errors = m.validateTimeRanges()
	case types.StepPlotConfig:
		count := countSelectedStringItems(m.plotFormatSelected)
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
		if m.showGlobalStyling && m.CurrentStep == types.StepSelectPlotCategories {
			// Handle space in global styling panel - toggle group expansion
			options := m.getGlobalStylingOptions()
			if m.globalStylingCursor < len(options) {
				opt := options[m.globalStylingCursor]
				m.togglePlotGroupExpansion(opt)
				m.globalStylingOptions = m.getGlobalStylingOptions()
			}
		} else if m.CurrentStep == types.StepSelectPlotCategories && m.Pipeline == types.PipelinePlotting {
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
	case types.StepSelectROIs:
		m.roiSelected[m.roiCursor] = !m.roiSelected[m.roiCursor]
	case types.StepSelectFeatureFiles:
		applicable := m.GetApplicableFeatureFiles()
		if m.featureFileCursor < len(applicable) {
			key := applicable[m.featureFileCursor].Key
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
		case optPlotOverwrite:
			if m.plotOverwrite == nil {
				val := true
				m.plotOverwrite = &val
			} else {
				val := !*m.plotOverwrite
				m.plotOverwrite = &val
			}
		}

	case types.StepSelectSpatial:
		m.spatialSelected[m.spatialCursor] = !m.spatialSelected[m.spatialCursor]
	case types.StepSelectPreprocessingStages:
		if m.prepStageCursor < len(m.prepStages) {
			m.prepStageSelected[m.prepStageCursor] = !m.prepStageSelected[m.prepStageCursor]
		}
	case types.StepTimeRange:
		if m.editingRangeIdx >= 0 {
			if m.editingField < timeRangeFieldCount-1 {
				m.editingField++
			} else {
				m.editingRangeIdx = -1
				m.editingField = 0
			}
		} else {
			m.editingRangeIdx = m.timeRangeCursor
			m.editingField = 1
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
	case types.StepSelectROIs:
		for i := range m.rois {
			m.roiSelected[i] = true
		}
	case types.StepSelectSpatial:
		for i := range spatialModes {
			m.spatialSelected[i] = true
		}
	case types.StepSelectPreprocessingStages:
		for i := range m.prepStages {
			m.prepStageSelected[i] = true
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
	case types.StepSelectROIs:
		m.roiSelected = make(map[int]bool)
	case types.StepSelectSpatial:
		m.spatialSelected = make(map[int]bool)
	case types.StepSelectPreprocessingStages:
		m.prepStageSelected = make(map[int]bool)
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
		m.editingPlotID = ""
		m.editingPlotField = plotItemConfigFieldNone
		m.UpdateAdvancedOffset()
		return true
	}

	if m.stepIndex > 0 {
		if m.CurrentStep == types.StepSelectSubjects {
			m.subjectFilter = ""
			m.filteringSubject = false
		}

		m.stepIndex--
		m.CurrentStep = m.steps[m.stepIndex]

		for m.stepIndex > 0 {
			if !m.shouldSkipStep(m.CurrentStep) {
				break
			}
			m.stepIndex--
			m.CurrentStep = m.steps[m.stepIndex]
		}

		m.resetCursorsForStep()
		return true
	}
	return false
}

///////////////////////////////////////////////////////////////////
// Validation
///////////////////////////////////////////////////////////////////

func (m *Model) validate() []string {
	var errors []string

	selectedCount := countSelectedStringItems(m.subjectSelected)
	validCount := 0
	for subjID, selected := range m.subjectSelected {
		if !selected {
			continue
		}
		for _, s := range m.subjects {
			if s.ID != subjID {
				continue
			}
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

	minRequired := minSubjectsRequired
	if m.Pipeline == types.PipelineML && m.mlScope == MLCVScopeGroup {
		minRequired = minSubjectsForGroupCV
	}

	if selectedCount < minRequired {
		errors = append(errors, fmt.Sprintf("Select at least %d subject(s)", minRequired))
	} else if validCount == 0 {
		errors = append(errors, "No valid subjects selected for this pipeline")
	}

	if m.Pipeline == types.PipelineFeatures && m.modeOptions[m.modeIndex] == styles.ModeCompute {
		categoryCount := countSelectedItems(m.selected)
		if categoryCount == 0 {
			errors = append(errors, "No feature categories selected")
		}

		bandCount := countSelectedItems(m.bandSelected)
		if bandCount == 0 {
			errors = append(errors, "No frequency bands selected")
		}

		errors = append(errors, m.validateTimeRanges()...)
	}

	if m.Pipeline == types.PipelineBehavior && m.modeOptions[m.modeIndex] == styles.ModeCompute {
		computationCount := countSelectedItems(m.computationSelected)
		if computationCount == 0 {
			errors = append(errors, "No behavior computations selected")
		}

		featureFileCount := countSelectedStringItems(m.featureFileSelected)
		if featureFileCount == 0 {
			errors = append(errors, "No feature files selected")
		}
	}

	if m.Pipeline == types.PipelinePlotting {
		if len(m.SelectedPlotIDs()) == 0 {
			errors = append(errors, "No plots selected")
		}
		formatCount := countSelectedStringItems(m.plotFormatSelected)
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

		if tr.Tmin != "" && tr.Tmax != "" {
			tmin, errMin := strconv.ParseFloat(tr.Tmin, 64)
			tmax, errMax := strconv.ParseFloat(tr.Tmax, 64)
			bothValid := errMin == nil && errMax == nil
			startNotLessThanEnd := tmin >= tmax
			if bothValid && startNotLessThanEnd {
				errors = append(errors, fmt.Sprintf("Range '%s': Start time (%.3f) must be less than end time (%.3f)", tr.Name, tmin, tmax))
			}
		}
	}

	hasBaseline := names["baseline"]
	hasActive := names["active"]

	baselineRequiredCategories := []string{"erds", "erp", "bursts"}
	needsBaseline := false
	for i, cat := range m.categories {
		if !m.selected[i] {
			continue
		}
		for _, requiredCat := range baselineRequiredCategories {
			if cat == requiredCat {
				needsBaseline = true
				break
			}
		}
		if needsBaseline {
			break
		}
	}

	powerNeedsBaseline := m.isCategorySelected("power") && m.powerRequireBaseline
	if powerNeedsBaseline && !hasBaseline {
		errors = append(errors, "Time range 'baseline' is required for power normalization (power.require_baseline=true)")
	}

	if needsBaseline && !hasBaseline {
		errors = append(errors, "Time range 'baseline' is required for baseline-normalized features (ERDS, ERP, bursts)")
	}

	if m.isCategorySelected("power") && !hasActive {
		for _, tr := range m.TimeRanges {
			if tr.Name != "baseline" && tr.Tmin != "" && tr.Tmax != "" {
				hasActive = true
				break
			}
		}
		if !hasActive {
			errors = append(errors, "At least one non-baseline time range with valid times is required for power extraction")
		}
	}

	for _, tr := range m.TimeRanges {
		if tr.Name == "" {
			continue
		}
		if tr.Tmin == "" || tr.Tmax == "" {
			errors = append(errors, fmt.Sprintf("Time range '%s' has missing tmin or tmax values", tr.Name))
		}
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
	// During subject selection (the first plotting step), keep validation permissive
	// so users can choose subjects before narrowing the plot set.
	if m.CurrentStep == types.StepSelectSubjects {
		if !s.HasEpochs && !s.HasFeatures && !s.HasStats {
			return false, "no derivatives"
		}
		return true, ""
	}

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

	case types.PipelineML:
		return len(m.getMLOptions())
	case types.PipelinePreprocessing:
		return len(m.getPreprocessingOptions())
	case types.PipelineFmri:
		return len(m.getFmriPreprocessingOptions())
	case types.PipelineRawToBIDS:
		return len(m.getRawToBidsOptions())
	case types.PipelineFmriRawToBIDS:
		return len(m.getFmriRawToBidsOptions())
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
	maxIterations := len(rows)
	for i := 0; i < maxIterations; i++ {
		next = moveCursorInList(next, delta, len(rows))
		rowKind := rows[next].kind
		isNonSelectableRow := rowKind == plottingRowSection || rowKind == plottingRowPlotInfo
		if !isNonSelectableRow {
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
	case types.PipelineML:
		m.toggleMLAdvancedOption()
	case types.PipelinePreprocessing:
		m.togglePreprocessingAdvancedOption()
	case types.PipelineFmri:
		m.toggleFmriAdvancedOption()
	case types.PipelineRawToBIDS:
		m.toggleRawToBidsAdvancedOption()
	case types.PipelineFmriRawToBIDS:
		m.toggleFmriRawToBidsAdvancedOption()
	case types.PipelineMergePsychoPyData:
		m.toggleMergeBehaviorAdvancedOption()
	}
}

func (m *Model) toggleFeaturesAdvancedOption() {
	if m.expandedOption >= 0 {
		m.handleExpandedListToggle()
		return
	}

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
	case optFeatGroupDirectedConnectivity:
		m.featGroupDirectedConnExpanded = !m.featGroupDirectedConnExpanded
		if !m.featGroupDirectedConnExpanded && m.expandedOption == expandedDirectedConnMeasures {
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
	case optFeatGroupQuality:
		m.featGroupQualityExpanded = !m.featGroupQualityExpanded
		m.useDefaultAdvanced = false
	case optFeatGroupERDS:
		m.featGroupERDSExpanded = !m.featGroupERDSExpanded
		m.useDefaultAdvanced = false
	// Asymmetry advanced options
	case optAsymmetryMinSegmentSec:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optAsymmetryMinCyclesAtFmin:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optAsymmetrySkipInvalidSegments:
		m.asymmetrySkipInvalidSegments = !m.asymmetrySkipInvalidSegments
		m.useDefaultAdvanced = false
	// Ratios advanced options
	case optRatiosMinSegmentSec:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optRatiosMinCyclesAtFmin:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optRatiosSkipInvalidSegments:
		m.ratiosSkipInvalidSegments = !m.ratiosSkipInvalidSegments
		m.useDefaultAdvanced = false
	// Spectral advanced options
	case optSpectralPsdMethod:
		m.spectralPsdMethod = (m.spectralPsdMethod + 1) % 2 // 0: multitaper, 1: welch
		m.useDefaultAdvanced = false
	case optSpectralFmin:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optSpectralFmax:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optSpectralMinSegmentSec:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optSpectralMinCyclesAtFmin:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	// Quality advanced options
	case optQualityPsdMethod:
		m.qualityPsdMethod = (m.qualityPsdMethod + 1) % 2 // 0: welch, 1: multitaper
		m.useDefaultAdvanced = false
	case optQualityFmin:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optQualityFmax:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optQualityNFft:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optQualityExcludeLineNoise:
		m.qualityExcludeLineNoise = !m.qualityExcludeLineNoise
		m.useDefaultAdvanced = false
	case optQualityLineNoiseFreq:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optQualityLineNoiseWidthHz:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optQualityLineNoiseHarmonics:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optQualitySnrSignalBandMin:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optQualitySnrSignalBandMax:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optQualitySnrNoiseBandMin:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optQualitySnrNoiseBandMax:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optQualityMuscleBandMin:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optQualityMuscleBandMax:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	// ERDS advanced options
	case optERDSUseLogRatio:
		m.erdsUseLogRatio = !m.erdsUseLogRatio
		m.useDefaultAdvanced = false
	case optERDSMinBaselinePower:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optERDSMinActivePower:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optERDSMinSegmentSec:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optERDSBands:
		m.startTextEdit(textFieldERDSBands)
		m.useDefaultAdvanced = false
	case optFeatGroupStorage:
		m.featGroupStorageExpanded = !m.featGroupStorageExpanded
	case optSaveSubjectLevelFeatures:
		m.saveSubjectLevelFeatures = !m.saveSubjectLevelFeatures
		m.useDefaultAdvanced = false
	case optFeatAlsoSaveCsv:
		m.featAlsoSaveCsv = !m.featAlsoSaveCsv
		m.useDefaultAdvanced = false
	case optFeatGroupExecution:
		m.featGroupExecutionExpanded = !m.featGroupExecutionExpanded
		m.useDefaultAdvanced = false
	case optFeatGroupSourceLoc:
		m.featGroupSourceLocExpanded = !m.featGroupSourceLocExpanded
		m.useDefaultAdvanced = false
	case optConnectivity:
		m.expandedOption = expandedConnectivityMeasures
		m.subCursor = 0
		m.useDefaultAdvanced = false
	case optDirectedConnMeasures:
		m.expandedOption = expandedDirectedConnMeasures
		m.subCursor = 0
		m.useDefaultAdvanced = false
	case optDirectedConnOutputLevel:
		m.directedConnOutputLevel = (m.directedConnOutputLevel + 1) % 2 // 0: full, 1: global_only
		m.useDefaultAdvanced = false
	case optDirectedConnMvarOrder:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optDirectedConnNFreqs:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optDirectedConnMinSegSamples:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	// Source localization options
	case optSourceLocMode:
		m.sourceLocMode = (m.sourceLocMode + 1) % 2 // 0: EEG-only, 1: fMRI-informed
		m.useDefaultAdvanced = false
	case optSourceLocMethod:
		m.sourceLocMethod = (m.sourceLocMethod + 1) % 2 // 0: lcmv, 1: eloreta
		m.useDefaultAdvanced = false
	case optSourceLocSpacing:
		m.sourceLocSpacing = (m.sourceLocSpacing + 1) % 4 // 0: oct5, 1: oct6, 2: ico4, 3: ico5
		m.useDefaultAdvanced = false
	case optSourceLocParc:
		m.sourceLocParc = (m.sourceLocParc + 1) % 3 // 0: aparc, 1: aparc.a2009s, 2: HCPMMP1
		m.useDefaultAdvanced = false
	case optSourceLocReg:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optSourceLocSnr:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optSourceLocLoose:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optSourceLocDepth:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optSourceLocConnMethod:
		m.sourceLocConnMethod = (m.sourceLocConnMethod + 1) % 3 // 0: aec, 1: wpli, 2: plv
		m.useDefaultAdvanced = false
	case optSourceLocSubject:
		m.startTextEdit(textFieldSourceLocSubject)
		m.useDefaultAdvanced = false
	case optSourceLocTrans:
		m.browsingField = "sourceLocTrans"
		m.pendingFileCmd = m.browseForFile("Select coregistration transform file", "sourceLocTrans", "FIF files", "fif")
		m.useDefaultAdvanced = false
	case optSourceLocBem:
		m.browsingField = "sourceLocBem"
		m.pendingFileCmd = m.browseForFile("Select BEM solution file", "sourceLocBem", "FIF files", "fif")
		m.useDefaultAdvanced = false
	case optSourceLocMindistMm:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optSourceLocFmriEnabled:
		m.sourceLocFmriEnabled = !m.sourceLocFmriEnabled
		m.useDefaultAdvanced = false
	case optSourceLocFmriStatsMap:
		m.browsingField = "sourceLocFmriStatsMap"
		m.pendingFileCmd = m.browseForFile("Select fMRI statistical map", "sourceLocFmriStatsMap", "NIfTI files", "nii,nii.gz")
		m.useDefaultAdvanced = false
	case optSourceLocFmriProvenance:
		m.sourceLocFmriProvenance = (m.sourceLocFmriProvenance + 1) % 2 // 0: independent, 1: same_dataset
		m.useDefaultAdvanced = false
	case optSourceLocFmriRequireProvenance:
		m.sourceLocFmriRequireProv = !m.sourceLocFmriRequireProv
		m.useDefaultAdvanced = false
	case optSourceLocFmriThreshold:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optSourceLocFmriTail:
		m.sourceLocFmriTail = (m.sourceLocFmriTail + 1) % 2 // 0: pos, 1: abs
		m.useDefaultAdvanced = false
	case optSourceLocFmriMinClusterVox:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optSourceLocFmriMaxClusters:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optSourceLocFmriMaxVoxPerClus:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optSourceLocFmriMaxTotalVox:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optSourceLocFmriRandomSeed:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	// BEM/Trans generation options (Docker-based)
	case optSourceLocCreateTrans:
		m.sourceLocCreateTrans = !m.sourceLocCreateTrans
		m.useDefaultAdvanced = false
	case optSourceLocCreateBemModel:
		m.sourceLocCreateBemModel = !m.sourceLocCreateBemModel
		m.useDefaultAdvanced = false
	case optSourceLocCreateBemSolution:
		m.sourceLocCreateBemSolution = !m.sourceLocCreateBemSolution
		m.useDefaultAdvanced = false
	// fMRI contrast builder options
	case optSourceLocFmriContrastEnabled:
		m.sourceLocFmriContrastEnabled = !m.sourceLocFmriContrastEnabled
		// Trigger condition discovery when enabling contrast builder
		if m.sourceLocFmriContrastEnabled && len(m.sourceLocFmriConditions) == 0 {
			subject := ""
			for _, s := range m.subjects {
				if m.subjectSelected[s.ID] {
					subject = s.ID
					break
				}
			}
			m.pendingFmriConditionsCmd = executor.DiscoverFmriConditions(m.repoRoot, subject, m.task)
		}
		m.useDefaultAdvanced = false
	case optSourceLocFmriContrastType:
		m.sourceLocFmriContrastType = (m.sourceLocFmriContrastType + 1) % 4 // 0: t-test, 1: paired, 2: F-test, 3: custom
		m.useDefaultAdvanced = false
	case optSourceLocFmriCondAColumn:
		if len(m.fmriDiscoveredColumns) > 0 {
			m.expandedOption = expandedFmriCondAColumn
			m.subCursor = 0
		} else {
			m.startTextEdit(textFieldSourceLocFmriCondAColumn)
		}
		m.useDefaultAdvanced = false
	case optSourceLocFmriCondAValue:
		colVals := m.GetFmriDiscoveredColumnValues(m.sourceLocFmriCondAColumn)
		if len(colVals) > 0 {
			m.expandedOption = expandedFmriCondAValue
			m.subCursor = 0
		} else {
			m.startTextEdit(textFieldSourceLocFmriCondAValue)
		}
		m.useDefaultAdvanced = false
	case optSourceLocFmriCondBColumn:
		if len(m.fmriDiscoveredColumns) > 0 {
			m.expandedOption = expandedFmriCondBColumn
			m.subCursor = 0
		} else {
			m.startTextEdit(textFieldSourceLocFmriCondBColumn)
		}
		m.useDefaultAdvanced = false
	case optSourceLocFmriCondBValue:
		colVals := m.GetFmriDiscoveredColumnValues(m.sourceLocFmriCondBColumn)
		if len(colVals) > 0 {
			m.expandedOption = expandedFmriCondBValue
			m.subCursor = 0
		} else {
			m.startTextEdit(textFieldSourceLocFmriCondBValue)
		}
		m.useDefaultAdvanced = false
	case optSourceLocFmriContrastFormula:
		m.startTextEdit(textFieldSourceLocFmriContrastFormula)
		m.useDefaultAdvanced = false
	case optSourceLocFmriContrastName:
		m.startTextEdit(textFieldSourceLocFmriContrastName)
		m.useDefaultAdvanced = false
	case optSourceLocFmriRunsToInclude:
		m.startTextEdit(textFieldSourceLocFmriRunsToInclude)
		m.useDefaultAdvanced = false
	case optSourceLocFmriAutoDetectRuns:
		m.sourceLocFmriAutoDetectRuns = !m.sourceLocFmriAutoDetectRuns
		m.useDefaultAdvanced = false
	case optSourceLocFmriHrfModel:
		m.sourceLocFmriHrfModel = (m.sourceLocFmriHrfModel + 1) % 3 // 0: SPM, 1: FLOBS, 2: FIR
		m.useDefaultAdvanced = false
	case optSourceLocFmriDriftModel:
		m.sourceLocFmriDriftModel = (m.sourceLocFmriDriftModel + 1) % 3 // 0: none, 1: cosine, 2: polynomial
		m.useDefaultAdvanced = false
	case optSourceLocFmriHighPassHz:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optSourceLocFmriLowPassHz:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optSourceLocFmriClusterCorrection:
		m.sourceLocFmriClusterCorrection = !m.sourceLocFmriClusterCorrection
		m.useDefaultAdvanced = false
	case optSourceLocFmriClusterPThreshold:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optSourceLocFmriOutputType:
		m.sourceLocFmriOutputType = (m.sourceLocFmriOutputType + 1) % 4 // 0: z-score, 1: t-stat, 2: cope, 3: beta
		m.useDefaultAdvanced = false
	case optSourceLocFmriResampleToFS:
		m.sourceLocFmriResampleToFS = !m.sourceLocFmriResampleToFS
		m.useDefaultAdvanced = false
	case optSourceLocFmriWindowAName:
		m.startTextEdit(textFieldSourceLocFmriWindowAName)
		m.useDefaultAdvanced = false
	case optSourceLocFmriWindowATmin:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optSourceLocFmriWindowATmax:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optSourceLocFmriWindowBName:
		m.startTextEdit(textFieldSourceLocFmriWindowBName)
		m.useDefaultAdvanced = false
	case optSourceLocFmriWindowBTmin:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optSourceLocFmriWindowBTmax:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	// ITPC options
	case optFeatGroupITPC:
		m.featGroupITPCExpanded = !m.featGroupITPCExpanded
		m.useDefaultAdvanced = false
	case optItpcMethod:
		m.itpcMethod = (m.itpcMethod + 1) % 4 // 0: global, 1: fold_global, 2: loo, 3: condition
		m.useDefaultAdvanced = false
	case optItpcConditionColumn:
		if len(m.availableColumns) > 0 {
			m.expandedOption = expandedItpcConditionColumn
			m.subCursor = 0
		} else {
			m.startTextEdit(textFieldItpcConditionColumn)
		}
		m.useDefaultAdvanced = false
	case optItpcConditionValues:
		if m.itpcConditionColumn == "" {
			m.ShowToast("Select a condition column first", "warning")
			return
		}
		if vals := m.GetDiscoveredColumnValues(m.itpcConditionColumn); len(vals) > 0 {
			m.expandedOption = expandedItpcConditionValues
			m.subCursor = 0
		} else {
			m.startTextEdit(textFieldItpcConditionValues)
		}
		m.useDefaultAdvanced = false
	case optItpcMinTrialsPerCondition:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optPACPhaseRange:
		if m.pacPhaseMin == 4.0 && m.pacPhaseMax == 8.0 {
			m.pacPhaseMin, m.pacPhaseMax = 2.0, 4.0 // delta
		} else if m.pacPhaseMin == 2.0 && m.pacPhaseMax == 4.0 {
			m.pacPhaseMin, m.pacPhaseMax = 8.0, 13.0 // alpha
		} else {
			m.pacPhaseMin, m.pacPhaseMax = 4.0, 8.0 // theta (default)
		}
		m.useDefaultAdvanced = false
	case optPACAmpRange:
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
	case optPACSource:
		m.pacSource = (m.pacSource + 1) % 2 // 0: precomputed, 1: tfr
		m.useDefaultAdvanced = false
	case optPACNormalize:
		m.pacNormalize = !m.pacNormalize
		m.useDefaultAdvanced = false
	case optPACNSurrogates:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optPACAllowHarmonicOverlap:
		m.pacAllowHarmonicOvrlap = !m.pacAllowHarmonicOvrlap
		m.useDefaultAdvanced = false
	case optPACMaxHarmonic:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optPACHarmonicToleranceHz:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optPACRandomSeed:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optPACComputeWaveformQC:
		m.pacComputeWaveformQC = !m.pacComputeWaveformQC
		m.useDefaultAdvanced = false
	case optPACWaveformOffsetMs:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optAperiodicFmin, optAperiodicFmax:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optAperiodicPeakZ, optAperiodicMinR2, optAperiodicMinPoints, optAperiodicPsdBandwidth, optAperiodicMaxRms, optAperiodicLineNoiseFreq, optAperiodicLineNoiseWidthHz, optAperiodicLineNoiseHarmonics:
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
	case optComplexitySignalBasis:
		m.complexitySignalBasis++
		if m.complexitySignalBasis > 1 {
			m.complexitySignalBasis = 0
		}
		m.useDefaultAdvanced = false
	case optComplexityMinSegmentSec, optComplexityMinSamples:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optComplexityZscore:
		m.complexityZscore = !m.complexityZscore
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
	case optERPSmoothMs, optERPPeakProminenceUv, optERPLowpassHz:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optBurstThresholdMethod:
		m.burstThresholdMethod++
		if m.burstThresholdMethod > 2 {
			m.burstThresholdMethod = 0
		}
		m.useDefaultAdvanced = false
	case optBurstThresholdPercentile:
		m.startNumberEdit()
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

	case optMinEpochs:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optSpectralRatioPairs:
		m.startTextEdit(textFieldSpectralRatioPairs)
		m.useDefaultAdvanced = false
	case optAperiodicSubtractEvoked:
		m.aperiodicSubtractEvoked = !m.aperiodicSubtractEvoked
		m.useDefaultAdvanced = false
	case optAsymmetryChannelPairs:
		m.startTextEdit(textFieldAsymmetryChannelPairs)
		m.useDefaultAdvanced = false
	// Spatial transform section
	case optFeatGroupSpatialTransform:
		m.featGroupSpatialTransformExpanded = !m.featGroupSpatialTransformExpanded
		m.useDefaultAdvanced = false
	case optSpatialTransform:
		m.spatialTransform = (m.spatialTransform + 1) % 3 // 0=none, 1=csd, 2=laplacian
		m.useDefaultAdvanced = false
	case optSpatialTransformLambda2, optSpatialTransformStiffness:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	// TFR section
	case optFeatGroupTFR:
		m.featGroupTFRExpanded = !m.featGroupTFRExpanded
	case optTfrFreqMin, optTfrFreqMax, optTfrNFreqs, optTfrMinCycles, optTfrNCyclesFactor, optTfrWorkers:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	}

	options = m.getFeaturesOptions()
	if len(options) > 0 {
		m.advancedCursor = clampCursor(m.advancedCursor, len(options)-1)
	}
	m.UpdateAdvancedOffset()
}

func (m *Model) togglePlottingAdvancedOption() {
	if m.expandedOption >= 0 {
		m.handleExpandedListToggle()
		return
	}

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
		case plotItemConfigFieldItpcSharedColorbar:
			cfg.ItpcSharedColorbar = cycleTriState(cfg.ItpcSharedColorbar)
			m.plotItemConfigs[row.plotID] = cfg
			m.useDefaultAdvanced = false
		case plotItemConfigFieldComparisonWindows, plotItemConfigFieldComparisonSegment, plotItemConfigFieldTopomapWindow:
			// Open dropdown if windows available, otherwise text edit
			if len(m.availableWindows) > 0 {
				m.expandedOption = expandedPlotComparisonWindows
				m.subCursor = 0
				m.editingPlotID = row.plotID
				m.editingPlotField = row.plotField
			} else {
				m.startPlotTextEdit(row.plotID, row.plotField)
			}
			m.useDefaultAdvanced = false
		case plotItemConfigFieldComparisonColumn:
			// Open dropdown if columns available, otherwise text edit
			plotCols := m.GetPlottingComparisonColumns()
			if len(plotCols) > 0 {
				m.expandedOption = expandedPlotComparisonColumn
				m.subCursor = 0
				m.editingPlotID = row.plotID
				m.editingPlotField = row.plotField
			} else {
				m.startPlotTextEdit(row.plotID, row.plotField)
			}
			m.useDefaultAdvanced = false
		case plotItemConfigFieldConnectivityCircleTopFraction, plotItemConfigFieldConnectivityCircleMinLines, plotItemConfigFieldConnectivityNetworkTopFraction:
			m.startPlotTextEdit(row.plotID, row.plotField)
			m.useDefaultAdvanced = false
		case plotItemConfigFieldTfrTopomapActiveWindow, plotItemConfigFieldTfrTopomapWindowSizeMs, plotItemConfigFieldTfrTopomapWindowCount,
			plotItemConfigFieldTfrTopomapLabelXPosition, plotItemConfigFieldTfrTopomapLabelYPositionBottom,
			plotItemConfigFieldTfrTopomapLabelYPosition, plotItemConfigFieldTfrTopomapTitleY,
			plotItemConfigFieldTfrTopomapTitlePad, plotItemConfigFieldTfrTopomapSubplotsRight,
			plotItemConfigFieldTfrTopomapTemporalHspace, plotItemConfigFieldTfrTopomapTemporalWspace:
			m.startPlotTextEdit(row.plotID, row.plotField)
			m.useDefaultAdvanced = false
		case plotItemConfigFieldComparisonValues:
			// Open dropdown if column selected and values available
			cfg := m.plotItemConfigs[row.plotID]
			col := cfg.ComparisonColumn
			if col == "" {
				col = m.plotComparisonColumn // fallback to global
			}
			if vals := m.GetPlottingComparisonColumnValues(col); len(vals) > 0 {
				m.expandedOption = expandedPlotComparisonValues
				m.subCursor = 0
				m.editingPlotID = row.plotID
				m.editingPlotField = row.plotField
			} else {
				m.startPlotTextEdit(row.plotID, row.plotField)
			}
			m.useDefaultAdvanced = false
		case plotItemConfigFieldComparisonROIs:
			// Open dropdown if ROIs available, otherwise text edit
			if len(m.discoveredROIs) > 0 {
				m.expandedOption = expandedPlotComparisonROIs
				m.subCursor = 0
				m.editingPlotID = row.plotID
				m.editingPlotField = row.plotField
			} else {
				m.startPlotTextEdit(row.plotID, row.plotField)
			}
			m.useDefaultAdvanced = false
		case plotItemConfigFieldComparisonLabels:
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

	case optPlotGroupDefaults, optPlotGroupFonts, optPlotGroupLayout, optPlotGroupFigureSizes,
		optPlotGroupColors, optPlotGroupAlpha, optPlotGroupScatter, optPlotGroupBar,
		optPlotGroupLine, optPlotGroupHistogram, optPlotGroupKDE, optPlotGroupErrorbar,
		optPlotGroupText, optPlotGroupValidation, optPlotGroupTopomap, optPlotGroupTFR,
		optPlotGroupSizing, optPlotGroupSelection, optPlotGroupComparisons, optPlotGroupTFRMisc:
		m.togglePlotGroupExpansion(opt)
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
		optPlotTFRTopomapWindowSizeMs,
		optPlotTFRTopomapWindowCount,
		optPlotTFRTopomapLabelXPosition,
		optPlotTFRTopomapLabelYPositionBottom,
		optPlotTFRTopomapLabelYPosition,
		optPlotTFRTopomapTitleY,
		optPlotTFRTopomapTitlePad,
		optPlotTFRTopomapSubplotsRight,
		optPlotTFRTopomapTemporalHspace,
		optPlotTFRTopomapTemporalWspace,
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
		optPlotConnectivityCircleMinLines,
		optPlotConnectivityNetworkTopFraction:
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

	case optPlotCompareWindows:
		m.plotCompareWindows = cycleTriState(m.plotCompareWindows)
		m.useDefaultAdvanced = false
	case optPlotComparisonWindows:
		if len(m.availableWindows) > 0 {
			m.expandedOption = expandedPlotComparisonWindows
			m.subCursor = 0
		} else {
			m.startTextEdit(textFieldPlotComparisonWindows)
		}
		m.useDefaultAdvanced = false
	case optPlotCompareColumns:
		m.plotCompareColumns = cycleTriState(m.plotCompareColumns)
		m.useDefaultAdvanced = false
	case optPlotComparisonSegment:
		m.startTextEdit(textFieldPlotComparisonSegment)
		m.useDefaultAdvanced = false
	case optPlotComparisonColumn:
		if len(m.discoveredColumns) > 0 {
			m.expandedOption = expandedPlotComparisonColumn
			m.subCursor = 0
		} else {
			m.startTextEdit(textFieldPlotComparisonColumn)
		}
		m.useDefaultAdvanced = false
	case optPlotComparisonValues:
		if m.plotComparisonColumn == "" {
			m.ShowToast("Select a column first", "warning")
			return
		}
		if vals := m.GetPlottingComparisonColumnValues(m.plotComparisonColumn); len(vals) > 0 {
			m.expandedOption = expandedPlotComparisonValues
			m.subCursor = 0
		} else {
			m.startTextEdit(textFieldPlotComparisonValues)
		}
		m.useDefaultAdvanced = false
	case optPlotComparisonLabels:
		m.startTextEdit(textFieldPlotComparisonLabels)
		m.useDefaultAdvanced = false
	case optPlotComparisonROIs:
		m.startTextEdit(textFieldPlotComparisonROIs)
		m.useDefaultAdvanced = false
	case optPlotOverwrite:
		if m.plotOverwrite == nil {
			val := true
			m.plotOverwrite = &val
		} else {
			val := !*m.plotOverwrite
			m.plotOverwrite = &val
		}
		m.useDefaultAdvanced = false
	}

	rows = m.getPlottingAdvancedRows()
	if len(rows) == 0 {
		m.advancedCursor = 0
		m.UpdateAdvancedOffset()
		return
	}
	m.advancedCursor = clampCursor(m.advancedCursor, len(rows)-1)
	if rows[m.advancedCursor].kind == plottingRowSection || rows[m.advancedCursor].kind == plottingRowPlotInfo {
		m.advancedCursor = m.findNextPlottingAdvancedRow(m.advancedCursor, 1)
	}
	m.UpdateAdvancedOffset()
}

func (m *Model) toggleBehaviorAdvancedOption() {
	if m.expandedOption >= 0 {
		m.handleExpandedListToggle()
		return
	}

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
	case optBehaviorGroupPainResidual:
		m.behaviorGroupPainResidualExpanded = !m.behaviorGroupPainResidualExpanded
		m.useDefaultAdvanced = false
	case optBehaviorGroupCorrelations:
		m.behaviorGroupCorrelationsExpanded = !m.behaviorGroupCorrelationsExpanded
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
	case optBehaviorGroupModeration:
		m.behaviorGroupModerationExpanded = !m.behaviorGroupModerationExpanded
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
	case optBehaviorNJobs:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optControlTemp:
		m.controlTemperature = !m.controlTemperature
		m.useDefaultAdvanced = false
	case optControlOrder:
		m.controlTrialOrder = !m.controlTrialOrder
		m.useDefaultAdvanced = false
	case optRunAdjustmentEnabled:
		m.runAdjustmentEnabled = !m.runAdjustmentEnabled
		m.useDefaultAdvanced = false
	case optRunAdjustmentColumn:
		if len(m.availableColumns) > 0 {
			m.expandedOption = expandedRunAdjustmentColumn
			m.subCursor = 0
		} else {
			m.startTextEdit(textFieldRunAdjustmentColumn)
		}
		m.useDefaultAdvanced = false
	case optRunAdjustmentIncludeInCorrelations:
		m.runAdjustmentIncludeInCorrelations = !m.runAdjustmentIncludeInCorrelations
		m.useDefaultAdvanced = false
	case optRunAdjustmentMaxDummies:
		m.startNumberEdit()
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
	case optTrialTableHighMissingFrac:
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
	case optPainResidualPolyDegree:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optPainResidualSplineDfCandidates:
		m.startTextEdit(textFieldPainResidualSplineDfCandidates)
		m.useDefaultAdvanced = false
	case optPainResidualModelCompare:
		m.painResidualModelCompareEnabled = !m.painResidualModelCompareEnabled
		m.useDefaultAdvanced = false
	case optPainResidualModelComparePolyDegrees:
		m.startTextEdit(textFieldPainResidualModelComparePolyDegrees)
		m.useDefaultAdvanced = false
	case optPainResidualBreakpoint:
		m.painResidualBreakpointEnabled = !m.painResidualBreakpointEnabled
		m.useDefaultAdvanced = false
	case optPainResidualBreakpointCandidates, optPainResidualBreakpointQlow, optPainResidualBreakpointQhigh:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optPainResidualCrossfitEnabled:
		m.painResidualCrossfitEnabled = !m.painResidualCrossfitEnabled
		m.useDefaultAdvanced = false
	case optPainResidualCrossfitGroupColumn:
		m.startTextEdit(textFieldPainResidualCrossfitGroupColumn)
		m.useDefaultAdvanced = false
	case optPainResidualCrossfitNSplits:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optPainResidualCrossfitMethod:
		m.painResidualCrossfitMethod = (m.painResidualCrossfitMethod + 1) % 2
		m.useDefaultAdvanced = false
	case optPainResidualCrossfitSplineKnots:
		m.startNumberEdit()
		m.useDefaultAdvanced = false

	// Regression
	case optRegressionOutcome:
		m.regressionOutcome = (m.regressionOutcome + 1) % 3
		m.useDefaultAdvanced = false
	case optRegressionIncludeTemperature:
		m.regressionIncludeTemperature = !m.regressionIncludeTemperature
		m.useDefaultAdvanced = false
	case optRegressionTempControl:
		m.regressionTempControl = (m.regressionTempControl + 1) % 3
		m.useDefaultAdvanced = false
	case optRegressionTempSplineKnots, optRegressionTempSplineQlow, optRegressionTempSplineQhigh:
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
	case optRegressionPermutations, optRegressionMaxFeatures:
		m.startNumberEdit()
		m.useDefaultAdvanced = false

	// Models
	case optModelsIncludeTemperature:
		m.modelsIncludeTemperature = !m.modelsIncludeTemperature
		m.useDefaultAdvanced = false
	case optModelsTempControl:
		m.modelsTempControl = (m.modelsTempControl + 1) % 3
		m.useDefaultAdvanced = false
	case optModelsTempSplineKnots, optModelsTempSplineQlow, optModelsTempSplineQhigh:
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
	case optModelsMaxFeatures:
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
	case optStabilityMaxFeatures, optStabilityAlpha:
		m.startNumberEdit()
		m.useDefaultAdvanced = false

	// Consistency
	case optConsistencyEnabled:
		m.consistencyEnabled = !m.consistencyEnabled
		m.useDefaultAdvanced = false

	// Influence
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
	case optInfluenceTempSplineKnots, optInfluenceTempSplineQlow, optInfluenceTempSplineQhigh:
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
	case optCorrelationsPreferPainResidual:
		m.correlationsPreferPainResidual = !m.correlationsPreferPainResidual
		m.useDefaultAdvanced = false
	case optCorrelationsUseCrossfitPainResidual:
		m.correlationsUseCrossfitResidual = !m.correlationsUseCrossfitResidual
		m.useDefaultAdvanced = false
	case optCorrelationsPrimaryUnit:
		m.correlationsPrimaryUnit = (m.correlationsPrimaryUnit + 1) % 2
		m.useDefaultAdvanced = false
	case optCorrelationsPermutationPrimary:
		m.correlationsPermutationPrimary = !m.correlationsPermutationPrimary
		m.useDefaultAdvanced = false
	case optCorrelationsTargetColumn:
		if len(m.availableColumns) > 0 {
			m.expandedOption = expandedCorrelationsTargetColumn
			m.subCursor = 0
		} else {
			m.startTextEdit(textFieldCorrelationsTargetColumn)
		}
		m.useDefaultAdvanced = false
	case optCorrelationsMultilevel:
		// Toggle multilevel_correlations computation
		for i, comp := range m.computations {
			if comp.Key == "multilevel_correlations" {
				m.computationSelected[i] = !m.computationSelected[i]
				break
			}
		}
		m.useDefaultAdvanced = false

	// Temporal
	case optTemporalResolutionMs, optTemporalTimeMinMs, optTemporalTimeMaxMs, optTemporalSmoothMs:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optTemporalSplitByCondition:
		m.temporalSplitByCondition = !m.temporalSplitByCondition
		m.useDefaultAdvanced = false
	case optTemporalConditionColumn:
		if len(m.availableColumns) > 0 {
			m.expandedOption = expandedTemporalConditionColumn
			m.subCursor = 0
		} else {
			m.startTextEdit(textFieldTemporalConditionColumn)
		}
		m.useDefaultAdvanced = false
	case optTemporalConditionValues:
		if m.temporalConditionColumn == "" {
			m.ShowToast("Select a column first", "warning")
			return
		}
		if vals := m.GetDiscoveredColumnValues(m.temporalConditionColumn); len(vals) > 0 {
			m.expandedOption = expandedTemporalConditionValues
			m.subCursor = 0
		} else {
			m.startTextEdit(textFieldTemporalConditionValues)
		}
		m.useDefaultAdvanced = false
	case optTemporalIncludeROIAverages:
		m.temporalIncludeROIAverages = !m.temporalIncludeROIAverages
		m.useDefaultAdvanced = false
	case optTemporalIncludeTFGrid:
		m.temporalIncludeTFGrid = !m.temporalIncludeTFGrid
		m.useDefaultAdvanced = false
	// Temporal feature selection
	case optTemporalFeaturePower:
		m.temporalFeaturePowerEnabled = !m.temporalFeaturePowerEnabled
		if !m.temporalFeaturePowerEnabled && !m.temporalFeatureITPCEnabled && !m.temporalFeatureERDSEnabled {
			m.temporalFeaturePowerEnabled = true
		}
		m.useDefaultAdvanced = false
	case optTemporalFeatureITPC:
		m.temporalFeatureITPCEnabled = !m.temporalFeatureITPCEnabled
		if !m.temporalFeaturePowerEnabled && !m.temporalFeatureITPCEnabled && !m.temporalFeatureERDSEnabled {
			m.temporalFeatureITPCEnabled = true
		}
		m.useDefaultAdvanced = false
	case optTemporalFeatureERDS:
		m.temporalFeatureERDSEnabled = !m.temporalFeatureERDSEnabled
		if !m.temporalFeaturePowerEnabled && !m.temporalFeatureITPCEnabled && !m.temporalFeatureERDSEnabled {
			m.temporalFeatureERDSEnabled = true
		}
		m.useDefaultAdvanced = false
	// ITPC-specific options
	case optTemporalITPCBaselineCorrection:
		m.temporalITPCBaselineCorrection = !m.temporalITPCBaselineCorrection
		m.useDefaultAdvanced = false
	case optTemporalITPCBaselineMin, optTemporalITPCBaselineMax:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	// ERDS-specific options
	case optTemporalERDSBaselineMin, optTemporalERDSBaselineMax:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optTemporalERDSMethod:
		m.temporalERDSMethod = (m.temporalERDSMethod + 1) % 2 // Toggle between percent and zscore
		m.useDefaultAdvanced = false
	// TF Heatmap options
	case optTemporalTfHeatmapEnabled:
		m.tfHeatmapEnabled = !m.tfHeatmapEnabled
		m.useDefaultAdvanced = false
	case optTemporalTfHeatmapFreqs:
		m.startTextEdit(textFieldTfHeatmapFreqs)
		m.useDefaultAdvanced = false
	case optTemporalTfHeatmapTimeResMs:
		m.startNumberEdit()
		m.useDefaultAdvanced = false

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
	case optClusterConditionColumn:
		if len(m.availableColumns) > 0 {
			m.expandedOption = expandedClusterConditionColumn
			m.subCursor = 0
		} else {
			m.startTextEdit(textFieldClusterConditionColumn)
		}
		m.useDefaultAdvanced = false
	case optClusterConditionValues:
		if m.clusterConditionColumn == "" {
			m.ShowToast("Select a column first", "warning")
			return
		}
		if vals := m.GetDiscoveredColumnValues(m.clusterConditionColumn); len(vals) > 0 {
			m.expandedOption = expandedClusterConditionValues
			m.subCursor = 0
		} else {
			m.startTextEdit(textFieldClusterConditionValues)
		}
		m.useDefaultAdvanced = false
	// Mediation options
	case optMediationBootstrap, optMediationMinEffect, optMediationPermutations:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optMediationMaxMediatorsEnabled:
		m.mediationMaxMediatorsEnabled = !m.mediationMaxMediatorsEnabled
		m.useDefaultAdvanced = false
	case optMediationMaxMediators:
		if m.mediationMaxMediatorsEnabled {
			m.startNumberEdit()
			m.useDefaultAdvanced = false
		}
	// Moderation options
	case optModerationMaxFeaturesEnabled:
		m.moderationMaxFeaturesEnabled = !m.moderationMaxFeaturesEnabled
		m.useDefaultAdvanced = false
	case optModerationMaxFeatures:
		if m.moderationMaxFeaturesEnabled {
			m.startNumberEdit()
			m.useDefaultAdvanced = false
		}
	case optModerationPermutations:
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
	case optConditionCompareColumn:
		if len(m.availableColumns) > 0 {
			m.expandedOption = expandedConditionCompareColumn
			m.subCursor = 0
		} else {
			m.startTextEdit(textFieldConditionCompareColumn)
		}
		m.useDefaultAdvanced = false
	case optConditionCompareWindows:
		if len(m.availableWindows) > 0 {
			m.expandedOption = expandedConditionCompareWindows
			m.subCursor = 0
		} else {
			m.startTextEdit(textFieldConditionCompareWindows)
		}
		m.useDefaultAdvanced = false
	case optConditionCompareValues:
		if m.conditionCompareColumn == "" {
			m.ShowToast("Select a column first", "warning")
			return
		}
		if vals := m.GetDiscoveredColumnValues(m.conditionCompareColumn); len(vals) > 0 {
			m.expandedOption = expandedConditionCompareValues
			m.subCursor = 0
		} else {
			m.startTextEdit(textFieldConditionCompareValues)
		}
		m.useDefaultAdvanced = false
	case optConditionWindowPrimaryUnit:
		m.conditionWindowPrimaryUnit = (m.conditionWindowPrimaryUnit + 1) % 2
		m.useDefaultAdvanced = false
	case optConditionPermutationPrimary:
		m.conditionPermutationPrimary = !m.conditionPermutationPrimary
		m.useDefaultAdvanced = false
	case optConditionEffectThreshold:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optConditionFailFast:
		m.conditionFailFast = !m.conditionFailFast
		m.useDefaultAdvanced = false
	case optConditionOverwrite:
		m.conditionOverwrite = !m.conditionOverwrite
		m.useDefaultAdvanced = false

	// Output section
	case optBehaviorGroupOutput:
		m.behaviorGroupOutputExpanded = !m.behaviorGroupOutputExpanded
		m.useDefaultAdvanced = false
	case optAlsoSaveCsv:
		m.alsoSaveCsv = !m.alsoSaveCsv
		m.useDefaultAdvanced = false
	case optBehaviorOverwrite:
		m.behaviorOverwrite = !m.behaviorOverwrite
		m.useDefaultAdvanced = false
	}

	options = m.getBehaviorOptions()
	if len(options) > 0 {
		m.advancedCursor = clampCursor(m.advancedCursor, len(options)-1)
	}
}

func (m *Model) toggleMLAdvancedOption() {
	options := m.getMLOptions()
	if m.advancedCursor < 0 || m.advancedCursor >= len(options) {
		return
	}

	opt := options[m.advancedCursor]
	switch opt {
	case optUseDefaults:
		m.useDefaultAdvanced = !m.useDefaultAdvanced
	case optMLNPerm, optMLInnerSplits, optRNGSeed, optRfNEstimators:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optMLSkipTimeGen:
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
	// Group expansion toggles
	case optPrepGroupStages:
		m.prepGroupStagesExpanded = !m.prepGroupStagesExpanded
	case optPrepGroupGeneral:
		m.prepGroupGeneralExpanded = !m.prepGroupGeneralExpanded
	case optPrepGroupFiltering:
		m.prepGroupFilteringExpanded = !m.prepGroupFilteringExpanded
	case optPrepGroupPyprep:
		m.prepGroupPyprepExpanded = !m.prepGroupPyprepExpanded
	case optPrepGroupICA:
		m.prepGroupICAExpanded = !m.prepGroupICAExpanded
	case optPrepGroupEpoching:
		m.prepGroupEpochingExpanded = !m.prepGroupEpochingExpanded
	// Stage toggles
	case optPrepStageBadChannels:
		m.prepStageSelected[0] = !m.prepStageSelected[0]
		m.useDefaultAdvanced = false
	case optPrepStageFiltering:
		m.prepStageSelected[1] = !m.prepStageSelected[1]
		m.useDefaultAdvanced = false
	case optPrepStageICA:
		m.prepStageSelected[2] = !m.prepStageSelected[2]
		m.useDefaultAdvanced = false
	case optPrepStageEpoching:
		m.prepStageSelected[3] = !m.prepStageSelected[3]
		m.useDefaultAdvanced = false
	case optPrepUsePyprep:
		m.prepUsePyprep = !m.prepUsePyprep
		m.useDefaultAdvanced = false
	case optPrepUseIcalabel:
		m.prepUseIcalabel = !m.prepUseIcalabel
		m.useDefaultAdvanced = false
	case optPrepMontage:
		m.startTextEdit(textFieldPrepMontage)
		m.useDefaultAdvanced = false
	case optPrepChTypes:
		m.startTextEdit(textFieldPrepChTypes)
		m.useDefaultAdvanced = false
	case optPrepEegReference:
		m.startTextEdit(textFieldPrepEegReference)
		m.useDefaultAdvanced = false
	case optPrepEogChannels:
		m.startTextEdit(textFieldPrepEogChannels)
		m.useDefaultAdvanced = false
	case optPrepRandomState:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optPrepTaskIsRest:
		m.prepTaskIsRest = !m.prepTaskIsRest
		m.useDefaultAdvanced = false
	case optPrepNJobs, optPrepResample, optPrepLFreq, optPrepHFreq, optPrepNotch, optPrepLineFreq, optPrepZaplineFline, optPrepICAComp, optPrepICALFreq, optPrepICARejThresh, optPrepProbThresh, optPrepEpochsTmin, optPrepEpochsTmax, optPrepEpochsBaseline, optPrepEpochsReject:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optPrepFindBreaks:
		m.prepFindBreaks = !m.prepFindBreaks
		m.useDefaultAdvanced = false
	case optPrepRansac:
		m.prepRansac = !m.prepRansac
		m.useDefaultAdvanced = false
	case optPrepRepeats, optPrepBreaksMinLength, optPrepTStartAfterPrevious, optPrepTStopBeforeNext:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optPrepAverageReref:
		m.prepAverageReref = !m.prepAverageReref
		m.useDefaultAdvanced = false
	case optPrepFileExtension:
		m.startTextEdit(textFieldPrepFileExtension)
		m.useDefaultAdvanced = false
	case optPrepConsiderPreviousBads:
		m.prepConsiderPreviousBads = !m.prepConsiderPreviousBads
		m.useDefaultAdvanced = false
	case optPrepOverwriteChansTsv:
		m.prepOverwriteChansTsv = !m.prepOverwriteChansTsv
		m.useDefaultAdvanced = false
	case optPrepDeleteBreaks:
		m.prepDeleteBreaks = !m.prepDeleteBreaks
		m.useDefaultAdvanced = false
	case optPrepRenameAnotDict:
		m.startTextEdit(textFieldPrepRenameAnotDict)
		m.useDefaultAdvanced = false
	case optPrepCustomBadDict:
		m.startTextEdit(textFieldPrepCustomBadDict)
		m.useDefaultAdvanced = false
	case optPrepSpatialFilter:
		m.prepSpatialFilter = (m.prepSpatialFilter + 1) % 2
		m.useDefaultAdvanced = false
	case optPrepICAAlgorithm:
		m.prepICAAlgorithm = (m.prepICAAlgorithm + 1) % 4
		m.useDefaultAdvanced = false
	case optPrepKeepMnebidsBads:
		m.prepKeepMnebidsBads = !m.prepKeepMnebidsBads
		m.useDefaultAdvanced = false
	case optIcaLabelsToKeep:
		m.startTextEdit(textFieldIcaLabelsToKeep)
		m.useDefaultAdvanced = false
	case optPrepEpochsNoBaseline:
		m.prepEpochsNoBaseline = !m.prepEpochsNoBaseline
		m.useDefaultAdvanced = false
	case optPrepConditions:
		m.startTextEdit(textFieldPrepConditions)
		m.useDefaultAdvanced = false
	case optPrepRejectMethod:
		m.prepRejectMethod = (m.prepRejectMethod + 1) % 3
		m.useDefaultAdvanced = false
	case optPrepRunSourceEstimation:
		m.prepRunSourceEstimation = !m.prepRunSourceEstimation
	case optPrepWriteCleanEvents:
		m.prepWriteCleanEvents = !m.prepWriteCleanEvents
	case optPrepOverwriteCleanEvents:
		m.prepOverwriteCleanEvents = !m.prepOverwriteCleanEvents
	case optPrepCleanEventsStrict:
		m.prepCleanEventsStrict = !m.prepCleanEventsStrict
		m.useDefaultAdvanced = false
	}

	// Clamp cursor after expand/collapse changes
	options = m.getPreprocessingOptions()
	if len(options) > 0 {
		m.advancedCursor = clampCursor(m.advancedCursor, len(options)-1)
	}
	m.UpdateAdvancedOffset()
}

func (m *Model) toggleFmriAdvancedOption() {
	options := m.getFmriPreprocessingOptions()
	if m.advancedCursor < 0 || m.advancedCursor >= len(options) {
		return
	}

	opt := options[m.advancedCursor]
	switch opt {
	case optUseDefaults:
		m.useDefaultAdvanced = !m.useDefaultAdvanced

	// Group expansion toggles
	case optFmriGroupRuntime:
		m.fmriGroupRuntimeExpanded = !m.fmriGroupRuntimeExpanded
	case optFmriGroupOutput:
		m.fmriGroupOutputExpanded = !m.fmriGroupOutputExpanded
	case optFmriGroupPerformance:
		m.fmriGroupPerformanceExpanded = !m.fmriGroupPerformanceExpanded
	case optFmriGroupAnatomical:
		m.fmriGroupAnatomicalExpanded = !m.fmriGroupAnatomicalExpanded
	case optFmriGroupBold:
		m.fmriGroupBoldExpanded = !m.fmriGroupBoldExpanded
	case optFmriGroupQc:
		m.fmriGroupQcExpanded = !m.fmriGroupQcExpanded
	case optFmriGroupDenoising:
		m.fmriGroupDenoisingExpanded = !m.fmriGroupDenoisingExpanded
	case optFmriGroupSurface:
		m.fmriGroupSurfaceExpanded = !m.fmriGroupSurfaceExpanded
	case optFmriGroupMultiecho:
		m.fmriGroupMultiechoExpanded = !m.fmriGroupMultiechoExpanded
	case optFmriGroupRepro:
		m.fmriGroupReproExpanded = !m.fmriGroupReproExpanded
	case optFmriGroupValidation:
		m.fmriGroupValidationExpanded = !m.fmriGroupValidationExpanded
	case optFmriGroupAdvanced:
		m.fmriGroupAdvancedExpanded = !m.fmriGroupAdvancedExpanded

	// Runtime
	case optFmriEngine:
		m.fmriEngineIndex = (m.fmriEngineIndex + 1) % 2
		m.useDefaultAdvanced = false
	case optFmriFmriprepImage:
		m.startTextEdit(textFieldFmriFmriprepImage)
		m.useDefaultAdvanced = false

	// Output
	case optFmriOutputSpaces:
		m.startTextEdit(textFieldFmriOutputSpaces)
		m.useDefaultAdvanced = false
	case optFmriIgnore:
		m.startTextEdit(textFieldFmriIgnore)
		m.useDefaultAdvanced = false
	case optFmriLevel:
		m.fmriLevelIndex = (m.fmriLevelIndex + 1) % 3
		m.useDefaultAdvanced = false
	case optFmriCiftiOutput:
		m.fmriCiftiOutputIndex = (m.fmriCiftiOutputIndex + 1) % 3
		m.useDefaultAdvanced = false
	case optFmriTaskId:
		m.startTextEdit(textFieldFmriTaskId)
		m.useDefaultAdvanced = false

	// Performance
	case optFmriNThreads:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optFmriOmpNThreads:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optFmriMemMb:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optFmriLowMem:
		m.fmriLowMem = !m.fmriLowMem
		m.useDefaultAdvanced = false

	// Anatomical
	case optFmriSkipReconstruction:
		m.fmriSkipReconstruction = !m.fmriSkipReconstruction
		m.useDefaultAdvanced = false
	case optFmriLongitudinal:
		m.fmriLongitudinal = !m.fmriLongitudinal
		m.useDefaultAdvanced = false
	case optFmriSkullStripTemplate:
		m.startTextEdit(textFieldFmriSkullStripTemplate)
		m.useDefaultAdvanced = false
	case optFmriSkullStripFixedSeed:
		m.fmriSkullStripFixedSeed = !m.fmriSkullStripFixedSeed
		m.useDefaultAdvanced = false

	// BOLD processing
	case optFmriBold2T1wInit:
		m.fmriBold2T1wInitIndex = (m.fmriBold2T1wInitIndex + 1) % 2
		m.useDefaultAdvanced = false
	case optFmriBold2T1wDof:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optFmriSliceTimeRef:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optFmriDummyScans:
		m.startNumberEdit()
		m.useDefaultAdvanced = false

	// Quality control
	case optFmriFdSpikeThreshold:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optFmriDvarsSpikeThreshold:
		m.startNumberEdit()
		m.useDefaultAdvanced = false

	// Denoising
	case optFmriUseAroma:
		m.fmriUseAroma = !m.fmriUseAroma
		m.useDefaultAdvanced = false

	// Surface
	case optFmriMedialSurfaceNan:
		m.fmriMedialSurfaceNan = !m.fmriMedialSurfaceNan
		m.useDefaultAdvanced = false
	case optFmriNoMsm:
		m.fmriNoMsm = !m.fmriNoMsm
		m.useDefaultAdvanced = false

	// Multi-echo
	case optFmriMeOutputEchos:
		m.fmriMeOutputEchos = !m.fmriMeOutputEchos
		m.useDefaultAdvanced = false

	// Reproducibility
	case optFmriRandomSeed:
		m.startNumberEdit()
		m.useDefaultAdvanced = false

	// Validation
	case optFmriSkipBidsValidation:
		m.fmriSkipBidsValidation = !m.fmriSkipBidsValidation
		m.useDefaultAdvanced = false
	case optFmriStopOnFirstCrash:
		m.fmriStopOnFirstCrash = !m.fmriStopOnFirstCrash
		m.useDefaultAdvanced = false
	case optFmriCleanWorkdir:
		m.fmriCleanWorkdir = !m.fmriCleanWorkdir
		m.useDefaultAdvanced = false

	// Advanced
	case optFmriExtraArgs:
		m.startTextEdit(textFieldFmriExtraArgs)
		m.useDefaultAdvanced = false
	}

	// Clamp cursor after expand/collapse changes
	options = m.getFmriPreprocessingOptions()
	if len(options) > 0 {
		m.advancedCursor = clampCursor(m.advancedCursor, len(options)-1)
	}
	m.UpdateAdvancedOffset()
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

func (m *Model) toggleFmriRawToBidsAdvancedOption() {
	options := m.getFmriRawToBidsOptions()
	if m.advancedCursor < 0 || m.advancedCursor >= len(options) {
		return
	}

	opt := options[m.advancedCursor]
	switch opt {
	case optUseDefaults:
		m.useDefaultAdvanced = !m.useDefaultAdvanced
	case optFmriRawSession:
		m.startTextEdit(textFieldFmriRawSession)
		m.useDefaultAdvanced = false
	case optFmriRawRestTask:
		m.startTextEdit(textFieldFmriRawRestTask)
		m.useDefaultAdvanced = false
	case optFmriRawIncludeRest:
		m.fmriRawIncludeRest = !m.fmriRawIncludeRest
		m.useDefaultAdvanced = false
	case optFmriRawIncludeFieldmaps:
		m.fmriRawIncludeFieldmaps = !m.fmriRawIncludeFieldmaps
		m.useDefaultAdvanced = false
	case optFmriRawDicomMode:
		m.fmriRawDicomModeIndex = (m.fmriRawDicomModeIndex + 1) % 3
		m.useDefaultAdvanced = false
	case optFmriRawOverwrite:
		m.fmriRawOverwrite = !m.fmriRawOverwrite
		m.useDefaultAdvanced = false
	case optFmriRawCreateEvents:
		m.fmriRawCreateEvents = !m.fmriRawCreateEvents
		m.useDefaultAdvanced = false
	case optFmriRawEventGranularity:
		m.fmriRawEventGranularity = (m.fmriRawEventGranularity + 1) % 2
		m.useDefaultAdvanced = false
	case optFmriRawOnsetReference:
		m.fmriRawOnsetRefIndex = (m.fmriRawOnsetRefIndex + 1) % 3
		m.useDefaultAdvanced = false
	case optFmriRawOnsetOffsetS:
		m.startNumberEdit()
		m.useDefaultAdvanced = false
	case optFmriRawDcm2niixPath:
		m.startTextEdit(textFieldFmriRawDcm2niixPath)
		m.useDefaultAdvanced = false
	case optFmriRawDcm2niixArgs:
		m.startTextEdit(textFieldFmriRawDcm2niixArgs)
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

func (m *Model) commitFmriRawToBidsNumber(val float64) {
	options := m.getFmriRawToBidsOptions()
	if m.advancedCursor < 0 || m.advancedCursor >= len(options) {
		return
	}
	opt := options[m.advancedCursor]
	switch opt {
	case optFmriRawOnsetOffsetS:
		m.fmriRawOnsetOffsetS = val
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
	case types.PipelineML:
		m.commitMLNumber(val)
	case types.PipelinePreprocessing:
		m.commitPreprocessingNumber(val)
	case types.PipelineFmri:
		m.commitFmriNumber(val)
	case types.PipelineRawToBIDS:
		m.commitRawToBidsNumber(val)
	case types.PipelineFmriRawToBIDS:
		m.commitFmriRawToBidsNumber(val)
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
	case optPlotTFRTopomapWindowSizeMs:
		if val <= 0 {
			m.plotTFRTopomapWindowSizeMs = 0
		} else {
			m.plotTFRTopomapWindowSizeMs = val
		}
	case optPlotTFRTopomapWindowCount:
		if val <= 0 {
			m.plotTFRTopomapWindowCount = 0
		} else {
			m.plotTFRTopomapWindowCount = int(val)
		}
	case optPlotTFRTopomapLabelXPosition:
		m.plotTFRTopomapLabelXPosition = val
	case optPlotTFRTopomapLabelYPositionBottom:
		m.plotTFRTopomapLabelYPositionBottom = val
	case optPlotTFRTopomapLabelYPosition:
		m.plotTFRTopomapLabelYPosition = val
	case optPlotTFRTopomapTitleY:
		m.plotTFRTopomapTitleY = val
	case optPlotTFRTopomapTitlePad:
		if val <= 0 {
			m.plotTFRTopomapTitlePad = 0
		} else {
			m.plotTFRTopomapTitlePad = int(val)
		}
	case optPlotTFRTopomapSubplotsRight:
		if val <= 0 || val > 1 {
			m.plotTFRTopomapSubplotsRight = 0.75
		} else {
			m.plotTFRTopomapSubplotsRight = val
		}
	case optPlotTFRTopomapTemporalHspace:
		if val < 0 {
			m.plotTFRTopomapTemporalHspace = 0
		} else {
			m.plotTFRTopomapTemporalHspace = val
		}
	case optPlotTFRTopomapTemporalWspace:
		if val < 0 {
			m.plotTFRTopomapTemporalWspace = 0
		} else {
			m.plotTFRTopomapTemporalWspace = val
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
	case optPlotConnectivityNetworkTopFraction:
		if val == 0 || (val > 0 && val <= 1) {
			m.plotConnectivityNetworkTopFraction = val
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
	case optAperiodicFmin:
		if val > 0 && val < m.aperiodicFmax {
			m.aperiodicFmin = val
		}
	case optAperiodicFmax:
		if val > 0 && val > m.aperiodicFmin {
			m.aperiodicFmax = val
		}
	case optAperiodicPeakZ:
		m.aperiodicPeakZ = val
	case optAperiodicMinR2:
		m.aperiodicMinR2 = val
	case optAperiodicMinPoints:
		m.aperiodicMinPoints = int(val)
	case optAperiodicPsdBandwidth:
		if val >= 0 {
			m.aperiodicPsdBandwidth = val
		}
	case optAperiodicMaxRms:
		if val >= 0 {
			m.aperiodicMaxRms = val
		}
	case optAperiodicLineNoiseFreq:
		if val > 0 {
			m.aperiodicLineNoiseFreq = val
		}
	case optAperiodicLineNoiseWidthHz:
		if val > 0 {
			m.aperiodicLineNoiseWidthHz = val
		}
	case optAperiodicLineNoiseHarmonics:
		if val >= 0 {
			m.aperiodicLineNoiseHarmonics = int(val)
		}
	case optPACMinEpochs:
		m.pacMinEpochs = int(val)
	case optPACNSurrogates:
		m.pacNSurrogates = int(val)
	case optPACMaxHarmonic:
		m.pacMaxHarmonic = int(val)
	case optPACHarmonicToleranceHz:
		m.pacHarmonicToleranceHz = val
	case optPACRandomSeed:
		m.pacRandomSeed = int(val)
	case optPACWaveformOffsetMs:
		m.pacWaveformOffsetMs = val
	case optPEDelay:
		m.complexityPEDelay = int(val)
	case optComplexityMinSegmentSec:
		if val > 0 {
			m.complexityMinSegmentSec = val
		}
	case optComplexityMinSamples:
		if val >= 0 {
			m.complexityMinSamples = int(val)
		}
	case optBurstThresholdPercentile:
		if val >= 0 && val <= 100 {
			m.burstThresholdPercentile = val
		}
	case optBurstMinDuration:
		m.burstMinDuration = int(val)
	case optERPSmoothMs:
		if val >= 0 {
			m.erpSmoothMs = val
		}
	case optERPPeakProminenceUv:
		if val >= 0 {
			m.erpPeakProminenceUv = val
		}
	case optERPLowpassHz:
		if val > 0 {
			m.erpLowpassHz = val
		}
	case optMinEpochs:
		m.minEpochsForFeatures = int(val)
	case optConnGraphProp:
		m.connGraphProp = val
	case optConnWindowLen:
		m.connWindowLen = val
	case optConnWindowStep:
		m.connWindowStep = val

	// Source localization numeric options
	case optSourceLocReg:
		if val >= 0 {
			m.sourceLocReg = val
		}
	case optSourceLocSnr:
		if val > 0 {
			m.sourceLocSnr = val
		}
	case optSourceLocLoose:
		if val >= 0 && val <= 1 {
			m.sourceLocLoose = val
		}
	case optSourceLocDepth:
		if val >= 0 && val <= 1 {
			m.sourceLocDepth = val
		}
	case optSourceLocMindistMm:
		if val >= 0 {
			m.sourceLocMindistMm = val
		}

	case optSourceLocFmriThreshold:
		if val > 0 {
			m.sourceLocFmriThreshold = val
		}
	case optSourceLocFmriMinClusterVox:
		if val >= 0 {
			m.sourceLocFmriMinClusterVox = int(val)
		}
	case optSourceLocFmriMaxClusters:
		if val >= 1 {
			m.sourceLocFmriMaxClusters = int(val)
		}
	case optSourceLocFmriMaxVoxPerClus:
		if val >= 0 {
			m.sourceLocFmriMaxVoxPerClus = int(val)
		}
	case optSourceLocFmriMaxTotalVox:
		if val >= 0 {
			m.sourceLocFmriMaxTotalVox = int(val)
		}
	case optSourceLocFmriRandomSeed:
		if val >= 0 {
			m.sourceLocFmriRandomSeed = int(val)
		}
	case optSourceLocFmriHighPassHz:
		if val >= 0 {
			m.sourceLocFmriHighPassHz = val
		}
	case optSourceLocFmriLowPassHz:
		if val > 0 {
			m.sourceLocFmriLowPassHz = val
		}
	case optSourceLocFmriClusterPThreshold:
		if val >= 0 && val <= 1 {
			m.sourceLocFmriClusterPThreshold = val
		}
	case optSourceLocFmriWindowATmin:
		m.sourceLocFmriWindowATmin = val
	case optSourceLocFmriWindowATmax:
		m.sourceLocFmriWindowATmax = val
	case optSourceLocFmriWindowBTmin:
		m.sourceLocFmriWindowBTmin = val
	case optSourceLocFmriWindowBTmax:
		m.sourceLocFmriWindowBTmax = val
	// ITPC options
	case optItpcMinTrialsPerCondition:
		if val >= 1 {
			m.itpcMinTrialsPerCondition = int(val)
		}
	// Spatial transform options
	case optSpatialTransformLambda2:
		if val > 0 {
			m.spatialTransformLambda2 = val
		}
	case optSpatialTransformStiffness:
		if val >= 0 {
			m.spatialTransformStiffness = val
		}
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
	case optTfrWorkers:
		m.tfrWorkers = int(val)
	// Asymmetry options
	case optAsymmetryMinSegmentSec:
		if val >= 0 {
			m.asymmetryMinSegmentSec = val
		}
	case optAsymmetryMinCyclesAtFmin:
		if val >= 0 {
			m.asymmetryMinCyclesAtFmin = val
		}
	// Ratios options
	case optRatiosMinSegmentSec:
		if val >= 0 {
			m.ratiosMinSegmentSec = val
		}
	case optRatiosMinCyclesAtFmin:
		if val >= 0 {
			m.ratiosMinCyclesAtFmin = val
		}
	// Spectral options
	case optSpectralFmin:
		if val >= 0 {
			m.spectralFmin = val
		}
	case optSpectralFmax:
		if val > 0 {
			m.spectralFmax = val
		}
	case optSpectralMinSegmentSec:
		if val >= 0 {
			m.spectralMinSegmentSec = val
		}
	case optSpectralMinCyclesAtFmin:
		if val >= 0 {
			m.spectralMinCyclesAtFmin = val
		}
	// Quality options
	case optQualityFmin:
		if val >= 0 {
			m.qualityFmin = val
		}
	case optQualityFmax:
		if val > 0 {
			m.qualityFmax = val
		}
	case optQualityNFft:
		if val >= 1 {
			m.qualityNfft = int(val)
		}
	case optQualityLineNoiseFreq:
		if val > 0 {
			m.qualityLineNoiseFreq = val
		}
	case optQualityLineNoiseWidthHz:
		if val >= 0 {
			m.qualityLineNoiseWidthHz = val
		}
	case optQualityLineNoiseHarmonics:
		if val >= 0 {
			m.qualityLineNoiseHarmonics = int(val)
		}
	case optQualitySnrSignalBandMin:
		if val >= 0 {
			m.qualitySnrSignalBandMin = val
		}
	case optQualitySnrSignalBandMax:
		if val > 0 {
			m.qualitySnrSignalBandMax = val
		}
	case optQualitySnrNoiseBandMin:
		if val >= 0 {
			m.qualitySnrNoiseBandMin = val
		}
	case optQualitySnrNoiseBandMax:
		if val > 0 {
			m.qualitySnrNoiseBandMax = val
		}
	case optQualityMuscleBandMin:
		if val >= 0 {
			m.qualityMuscleBandMin = val
		}
	case optQualityMuscleBandMax:
		if val > 0 {
			m.qualityMuscleBandMax = val
		}
	// ERDS options
	case optERDSMinBaselinePower:
		if val > 0 {
			m.erdsMinBaselinePower = val
		}
	case optERDSMinActivePower:
		if val > 0 {
			m.erdsMinActivePower = val
		}
	case optERDSMinSegmentSec:
		if val >= 0 {
			m.erdsMinSegmentSec = val
		}
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
	case optFDRAlpha:
		if val > 0 && val <= 1 {
			m.fdrAlpha = val
		}
	case optRunAdjustmentMaxDummies:
		if val >= 1 {
			m.runAdjustmentMaxDummies = int(val)
		}

	case optTrialTableHighMissingFrac:
		if val >= 0 && val <= 1 {
			m.trialTableHighMissingFrac = val
		}

	// Pain residual + diagnostics
	case optPainResidualPolyDegree:
		if val >= 1 {
			m.painResidualPolyDegree = int(val)
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
	case optPainResidualCrossfitNSplits:
		if val >= 2 {
			m.painResidualCrossfitNSplits = int(val)
		}
	case optPainResidualCrossfitSplineKnots:
		if val >= 3 {
			m.painResidualCrossfitSplineKnots = int(val)
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
	case optModelsMaxFeatures:
		if val >= 0 {
			m.modelsMaxFeatures = int(val)
		}

	// Stability
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
	// ITPC temporal options
	case optTemporalITPCBaselineMin:
		m.temporalITPCBaselineMin = val
	case optTemporalITPCBaselineMax:
		m.temporalITPCBaselineMax = val
	// ERDS temporal options
	case optTemporalERDSBaselineMin:
		m.temporalERDSBaselineMin = val
	case optTemporalERDSBaselineMax:
		m.temporalERDSBaselineMax = val
	// TF Heatmap options
	case optTemporalTfHeatmapTimeResMs:
		if val >= 1 {
			m.tfHeatmapTimeResMs = int(val)
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
	case optMediationPermutations:
		if val >= 0 {
			m.mediationPermutations = int(val)
		}
	case optMediationMinEffect:
		if val >= 0 {
			m.mediationMinEffect = val
		}
	case optMediationMaxMediators:
		if val >= 1 {
			m.mediationMaxMediators = int(val)
		}
	case optModerationMaxFeatures:
		if val >= 1 {
			m.moderationMaxFeatures = int(val)
		}
	case optModerationPermutations:
		if val >= 0 {
			m.moderationPermutations = int(val)
		}
	case optMixedMaxFeatures:
		if val >= 1 {
			m.mixedMaxFeatures = int(val)
		}
	case optConditionEffectThreshold:
		m.conditionEffectThreshold = val
	}
}

func (m *Model) commitMLNumber(val float64) {
	options := m.getMLOptions()
	if m.advancedCursor < 0 || m.advancedCursor >= len(options) {
		return
	}

	opt := options[m.advancedCursor]
	switch opt {
	case optMLNPerm:
		if val >= 0 {
			m.mlNPerm = int(val)
		}
	case optMLInnerSplits:
		if val >= 2 {
			m.innerSplits = int(val)
		}
	case optMLOuterJobs:
		if val >= 1 {
			m.outerJobs = int(val)
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
		if val >= -1 {
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
	case optPrepLineFreq:
		if val > 0 {
			m.prepLineFreq = int(val)
		}
	case optPrepZaplineFline:
		if val > 0 {
			m.prepZaplineFline = val
		}
	case optPrepICAComp:
		if val > 0 {
			m.prepICAComp = val
		}
	case optPrepICALFreq:
		if val > 0 {
			m.prepICALFreq = val
		}
	case optPrepICARejThresh:
		if val >= 0 {
			m.prepICARejThresh = val
		}
	case optPrepProbThresh:
		if val >= 0 && val <= 1 {
			m.prepProbThresh = val
		}
	case optPrepRepeats:
		if val >= 1 {
			m.prepRepeats = int(val)
		}
	case optPrepBreaksMinLength:
		if val > 0 {
			m.prepBreaksMinLength = int(val)
		}
	case optPrepTStartAfterPrevious:
		if val >= 0 {
			m.prepTStartAfterPrevious = int(val)
		}
	case optPrepTStopBeforeNext:
		if val >= 0 {
			m.prepTStopBeforeNext = int(val)
		}
	case optPrepEpochsTmin:
		m.prepEpochsTmin = val
	case optPrepEpochsTmax:
		m.prepEpochsTmax = val
	case optPrepEpochsBaseline:
		// For baseline, user enters a single number (start), and we assume end = 0
		// A more sophisticated approach would use text field for "start end" format
		m.prepEpochsBaselineStart = val
		m.prepEpochsBaselineEnd = 0
	case optPrepEpochsReject:
		if val >= 0 {
			m.prepEpochsReject = val
		}
	}
}

func (m *Model) commitFmriNumber(val float64) {
	options := m.getFmriPreprocessingOptions()
	if m.advancedCursor < 0 || m.advancedCursor >= len(options) {
		return
	}

	opt := options[m.advancedCursor]
	switch opt {
	case optFmriNThreads:
		if val >= 0 {
			m.fmriNThreads = int(val)
		}
	case optFmriOmpNThreads:
		if val >= 0 {
			m.fmriOmpNThreads = int(val)
		}
	case optFmriMemMb:
		if val >= 0 {
			m.fmriMemMb = int(val)
		}
	case optFmriBold2T1wDof:
		if val >= 0 {
			m.fmriBold2T1wDof = int(val)
		}
	case optFmriSliceTimeRef:
		if val >= 0 && val <= 1 {
			m.fmriSliceTimeRef = val
		}
	case optFmriDummyScans:
		if val >= 0 {
			m.fmriDummyScans = int(val)
		}
	case optFmriFdSpikeThreshold:
		if val >= 0 {
			m.fmriFdSpikeThreshold = val
		}
	case optFmriDvarsSpikeThreshold:
		if val >= 0 {
			m.fmriDvarsSpikeThreshold = val
		}
	case optFmriRandomSeed:
		if val >= 0 {
			m.fmriRandomSeed = int(val)
		}
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

	next := current
	if next < 0 {
		next = len(m.plotItems) - 1
	}

	maxIterations := len(m.plotItems)
	for i := 0; i < maxIterations; i++ {
		next = moveCursorInList(next, delta, len(m.plotItems))
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
	return moveCursorInList(next, delta, len(items))
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
	categoryKey := categories[idx].Key
	for i, plot := range m.plotItems {
		if strings.EqualFold(plot.Group, categoryKey) {
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

// initBandEditBuffer initializes the edit buffer with the current field value
func (m *Model) initBandEditBuffer() {
	if m.editingBandIdx < 0 || m.editingBandIdx >= len(m.bands) {
		return
	}
	band := m.bands[m.editingBandIdx]
	switch m.editingBandField {
	case 0:
		m.bandEditBuffer = band.Name
	case 1:
		m.bandEditBuffer = fmt.Sprintf("%.1f", band.LowHz)
	case 2:
		m.bandEditBuffer = fmt.Sprintf("%.1f", band.HighHz)
	}
}

// commitBandEdit commits the current edit buffer to the band field
func (m *Model) commitBandEdit() {
	if m.editingBandIdx < 0 || m.editingBandIdx >= len(m.bands) {
		return
	}
	switch m.editingBandField {
	case 0:
		// Name field
		if m.bandEditBuffer != "" {
			m.bands[m.editingBandIdx].Name = m.bandEditBuffer
			m.bands[m.editingBandIdx].Key = strings.ToLower(m.bandEditBuffer)
		}
	case 1:
		// LowHz field
		if val, err := strconv.ParseFloat(m.bandEditBuffer, 64); err == nil && val >= 0 {
			m.bands[m.editingBandIdx].LowHz = val
		}
	case 2:
		// HighHz field
		if val, err := strconv.ParseFloat(m.bandEditBuffer, 64); err == nil && val >= 0 {
			m.bands[m.editingBandIdx].HighHz = val
		}
	}
}

// startBandEdit starts editing the current band's frequencies
func (m *Model) startBandEdit() {
	if m.bandCursor >= 0 && m.bandCursor < len(m.bands) {
		m.editingBandIdx = m.bandCursor
		m.editingBandField = 1 // Start with LowHz
		m.initBandEditBuffer()
	}
}

// addNewBand adds a new custom frequency band
func (m *Model) addNewBand() {
	newKey := fmt.Sprintf("custom%d", len(m.bands)+1)
	newBand := FrequencyBand{
		Key:         newKey,
		Name:        strings.Title(newKey),
		Description: "Custom frequency band",
		LowHz:       1.0,
		HighHz:      10.0,
	}
	m.bands = append(m.bands, newBand)
	m.bandCursor = len(m.bands) - 1
	m.editingBandIdx = m.bandCursor
	m.editingBandField = 0 // Start with name
	m.initBandEditBuffer()
}

// removeBand removes the currently selected band
func (m *Model) removeBand() {
	if len(m.bands) <= 1 {
		return // Keep at least one band
	}
	if m.bandCursor >= 0 && m.bandCursor < len(m.bands) {
		m.bands = append(m.bands[:m.bandCursor], m.bands[m.bandCursor+1:]...)
		delete(m.bandSelected, m.bandCursor)
		newSelected := make(map[int]bool)
		for i, sel := range m.bandSelected {
			if i > m.bandCursor {
				newSelected[i-1] = sel
			} else {
				newSelected[i] = sel
			}
		}
		m.bandSelected = newSelected
		if m.bandCursor >= len(m.bands) {
			m.bandCursor = len(m.bands) - 1
		}
		if m.bandCursor < 0 {
			m.bandCursor = 0
		}
	}
}

// initROIEditBuffer initializes the edit buffer with the current ROI field value
func (m *Model) initROIEditBuffer() {
	if m.editingROIIdx < 0 || m.editingROIIdx >= len(m.rois) {
		return
	}
	roi := m.rois[m.editingROIIdx]
	switch m.editingROIField {
	case 0:
		m.roiEditBuffer = roi.Name
	case 1:
		m.roiEditBuffer = roi.Channels
	}
}

// commitROIEdit commits the current edit buffer to the ROI field
func (m *Model) commitROIEdit() {
	if m.editingROIIdx < 0 || m.editingROIIdx >= len(m.rois) {
		return
	}
	switch m.editingROIField {
	case 0:
		// Name field
		if m.roiEditBuffer != "" {
			m.rois[m.editingROIIdx].Name = m.roiEditBuffer
			m.rois[m.editingROIIdx].Key = strings.ReplaceAll(m.roiEditBuffer, " ", "_")
		}
	case 1:
		// Channels field
		if m.roiEditBuffer != "" {
			m.rois[m.editingROIIdx].Channels = m.roiEditBuffer
		}
	}
}

// startROIEdit starts editing the current ROI's channels
func (m *Model) startROIEdit() {
	if m.roiCursor >= 0 && m.roiCursor < len(m.rois) {
		m.editingROIIdx = m.roiCursor
		m.editingROIField = 1 // Start with channels
		m.initROIEditBuffer()
	}
}

// addNewROI adds a new custom ROI
func (m *Model) addNewROI() {
	newKey := fmt.Sprintf("Custom_%d", len(m.rois)+1)
	newROI := ROIDefinition{
		Key:      newKey,
		Name:     fmt.Sprintf("Custom %d", len(m.rois)+1),
		Channels: "Cz,Pz",
	}
	m.rois = append(m.rois, newROI)
	m.roiCursor = len(m.rois) - 1
	m.editingROIIdx = m.roiCursor
	m.editingROIField = 0 // Start with name
	m.initROIEditBuffer()
}

// removeROI removes the currently selected ROI
func (m *Model) removeROI() {
	if len(m.rois) <= 1 {
		return // Keep at least one ROI
	}
	if m.roiCursor >= 0 && m.roiCursor < len(m.rois) {
		m.rois = append(m.rois[:m.roiCursor], m.rois[m.roiCursor+1:]...)
		delete(m.roiSelected, m.roiCursor)
		newSelected := make(map[int]bool)
		for i, sel := range m.roiSelected {
			if i > m.roiCursor {
				newSelected[i-1] = sel
			} else {
				newSelected[i] = sel
			}
		}
		m.roiSelected = newSelected
		if m.roiCursor >= len(m.rois) {
			m.roiCursor = len(m.rois) - 1
		}
		if m.roiCursor < 0 {
			m.roiCursor = 0
		}
	}
}

// togglePlotGroupExpansion toggles the expansion state of a plot group option
func (m *Model) togglePlotGroupExpansion(opt optionType) {
	switch opt {
	case optPlotGroupDefaults:
		m.plotGroupDefaultsExpanded = !m.plotGroupDefaultsExpanded
	case optPlotGroupFonts:
		m.plotGroupFontsExpanded = !m.plotGroupFontsExpanded
	case optPlotGroupLayout:
		m.plotGroupLayoutExpanded = !m.plotGroupLayoutExpanded
	case optPlotGroupFigureSizes:
		m.plotGroupFigureSizesExpanded = !m.plotGroupFigureSizesExpanded
	case optPlotGroupColors:
		m.plotGroupColorsExpanded = !m.plotGroupColorsExpanded
	case optPlotGroupAlpha:
		m.plotGroupAlphaExpanded = !m.plotGroupAlphaExpanded
	case optPlotGroupScatter:
		m.plotGroupScatterExpanded = !m.plotGroupScatterExpanded
	case optPlotGroupBar:
		m.plotGroupBarExpanded = !m.plotGroupBarExpanded
	case optPlotGroupLine:
		m.plotGroupLineExpanded = !m.plotGroupLineExpanded
	case optPlotGroupHistogram:
		m.plotGroupHistogramExpanded = !m.plotGroupHistogramExpanded
	case optPlotGroupKDE:
		m.plotGroupKDEExpanded = !m.plotGroupKDEExpanded
	case optPlotGroupErrorbar:
		m.plotGroupErrorbarExpanded = !m.plotGroupErrorbarExpanded
	case optPlotGroupText:
		m.plotGroupTextExpanded = !m.plotGroupTextExpanded
	case optPlotGroupValidation:
		m.plotGroupValidationExpanded = !m.plotGroupValidationExpanded
	case optPlotGroupTopomap:
		m.plotGroupTopomapExpanded = !m.plotGroupTopomapExpanded
	case optPlotGroupTFR:
		m.plotGroupTFRExpanded = !m.plotGroupTFRExpanded
	case optPlotGroupTFRMisc:
		m.plotGroupTFRMiscExpanded = !m.plotGroupTFRMiscExpanded
	case optPlotGroupSizing:
		m.plotGroupSizingExpanded = !m.plotGroupSizingExpanded
	case optPlotGroupSelection:
		m.plotGroupSelectionExpanded = !m.plotGroupSelectionExpanded
	case optPlotGroupComparisons:
		m.plotGroupComparisonsExpanded = !m.plotGroupComparisonsExpanded
	}
}
