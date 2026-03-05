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

// File layout notes:
// - `handlers.go`: navigation, validation, step transitions, shared edit helpers.
// - `handlers_toggles_*.go`: advanced option toggle handlers per pipeline groups.
// - `handlers_commit_numbers.go`: numeric commit logic by pipeline.

// browseForFile opens a file picker dialog for the specified field
func (m *Model) browseForFile(prompt, field string, fileTypeDesc, extensions string) tea.Cmd {
	return executor.PickFile(prompt, field, fileTypeDesc, extensions)
}

const (
	minSubjectsRequired   = 1
	minSubjectsForGroupCV = 2
	timeRangeFieldCount   = 3
)

var groupSupportedPlotIDs = map[string]struct{}{
	"band_power_topomaps":    {},
	"power_by_condition":     {},
	"power_spectral_density": {},
}

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
		if m.Pipeline == types.PipelinePlotting {
			if m.plottingScope == PlottingScopeGroup {
				m.plottingScope = PlottingScopeSubject
			} else {
				m.plottingScope = PlottingScopeGroup
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
	prevStep := m.CurrentStep

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

		// Mode-dependent defaults (fMRI analysis)
		if prevStep == types.StepSelectMode && m.Pipeline == types.PipelineFmriAnalysis {
			mode := ""
			if m.modeIndex >= 0 && m.modeIndex < len(m.modeOptions) {
				mode = m.modeOptions[m.modeIndex]
			}
			if mode == "trial-signatures" {
				space := strings.TrimSpace(m.fmriAnalysisFmriprepSpace)
				if space == "" || strings.EqualFold(space, "T1w") {
					m.fmriAnalysisFmriprepSpace = "MNI152NLin2009cAsym"
				}
			}
		}

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
			m.ReadyToExecute = true
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
		if m.Pipeline == types.PipelinePlotting && m.plottingScope == PlottingScopeGroup {
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
		count := m.countSelectedVisiblePlots()
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
			if !m.IsPlotVisibleForSelection(m.plotItems[m.plotCursor]) {
				m.plotCursor = m.findNextVisiblePlot(m.plotCursor, 1)
				if m.plotCursor < 0 || m.plotCursor >= len(m.plotItems) {
					break
				}
				if !m.IsPlotVisibleForSelection(m.plotItems[m.plotCursor]) {
					break
				}
			}
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
			if m.IsPlotVisibleForSelection(plot) {
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
			if m.IsPlotVisibleForSelection(plot) {
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
	if m.Pipeline == types.PipelinePlotting && m.plottingScope == PlottingScopeGroup {
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

	if m.Pipeline == types.PipelineFmriAnalysis {
		mode := ""
		if m.modeIndex >= 0 && m.modeIndex < len(m.modeOptions) {
			mode = m.modeOptions[m.modeIndex]
		}
		groupingEnabled := strings.TrimSpace(m.fmriTrialSigGroupColumn) != "" && strings.TrimSpace(m.fmriTrialSigGroupValuesSpec) != ""

		if mode == "trial-signatures" {
			if groupingEnabled {
				// Grouping mode uses Group Column/Values for trial selection; Cond A/B may be left empty.
				if strings.TrimSpace(m.fmriTrialSigGroupColumn) == "" || strings.TrimSpace(m.fmriTrialSigGroupValuesSpec) == "" {
					errors = append(errors, "fMRI trial signatures: Group Column and Group Values are required when grouping is enabled")
				}
			} else {
				if strings.TrimSpace(m.fmriAnalysisCondAValue) == "" {
					errors = append(errors, "fMRI trial signatures: Cond A Value is required (or enable grouping)")
				}
				if strings.TrimSpace(m.fmriAnalysisCondBValue) == "" {
					errors = append(errors, "fMRI trial signatures: Cond B Value is required (or enable grouping)")
				}
			}
		} else {
			// First-level
			if strings.TrimSpace(m.fmriAnalysisCondAValue) == "" {
				errors = append(errors, "fMRI analysis: Cond A Value is required")
			}
		}
		if mode == "first-level" && m.fmriAnalysisContrastType%2 == 1 && strings.TrimSpace(m.fmriAnalysisFormula) == "" {
			errors = append(errors, "fMRI first-level: custom contrast requires a formula")
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
		if !m.plotSelected[i] || !m.IsPlotVisibleForSelection(plot) {
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
	case types.PipelineFmriAnalysis:
		return len(m.getFmriAnalysisOptions())
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
		if m.IsPlotVisibleForSelection(m.plotItems[next]) {
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

func (m Model) isPlotSupportedForScope(plot PlotItem) bool {
	if m.Pipeline != types.PipelinePlotting || m.plottingScope != PlottingScopeGroup {
		return true
	}
	_, ok := groupSupportedPlotIDs[strings.TrimSpace(plot.ID)]
	return ok
}

func (m Model) IsPlotVisibleForSelection(plot PlotItem) bool {
	if !m.IsPlotCategorySelected(plot.Group) {
		return false
	}
	return m.isPlotSupportedForScope(plot)
}

func (m Model) countSelectedVisiblePlots() int {
	count := 0
	for i, plot := range m.plotItems {
		if !m.IsPlotVisibleForSelection(plot) {
			continue
		}
		if m.plotSelected[i] {
			count++
		}
	}
	return count
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
	m.roiEditCursorPos = len(m.roiEditBuffer)
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

func (m *Model) moveROIEditCursorLeft() {
	if m.roiEditCursorPos > 0 {
		m.roiEditCursorPos--
	}
}

func (m *Model) moveROIEditCursorRight() {
	if m.roiEditCursorPos < len(m.roiEditBuffer) {
		m.roiEditCursorPos++
	}
}

func (m *Model) backspaceROIEditBuffer() {
	if m.roiEditCursorPos <= 0 || len(m.roiEditBuffer) == 0 {
		return
	}
	before := m.roiEditBuffer[:m.roiEditCursorPos-1]
	after := m.roiEditBuffer[m.roiEditCursorPos:]
	m.roiEditBuffer = before + after
	m.roiEditCursorPos--
}

func (m *Model) insertROIEditChar(char string) {
	if len(char) == 0 {
		return
	}
	before := m.roiEditBuffer[:m.roiEditCursorPos]
	after := m.roiEditBuffer[m.roiEditCursorPos:]
	m.roiEditBuffer = before + char + after
	m.roiEditCursorPos += len(char)
}

func (m Model) roiEditDisplayValue() string {
	cursor := m.roiEditCursorPos
	if cursor < 0 {
		cursor = 0
	}
	if cursor > len(m.roiEditBuffer) {
		cursor = len(m.roiEditBuffer)
	}
	return m.roiEditBuffer[:cursor] + "\u258c" + m.roiEditBuffer[cursor:]
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
	case optPlotGroupSourceLoc:
		m.plotGroupSourceLocExpanded = !m.plotGroupSourceLocExpanded
	case optPlotGroupSelection:
		m.plotGroupSelectionExpanded = !m.plotGroupSelectionExpanded
	case optPlotGroupComparisons:
		m.plotGroupComparisonsExpanded = !m.plotGroupComparisonsExpanded
	}
}
