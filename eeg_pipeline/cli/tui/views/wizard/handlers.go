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
	m.cmdScrollOffset = 0
	m.subCursor = 0
	m.expandedOption = expandedNone
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
	}
	return false
}

func (m *Model) handleUp() {
	switch m.CurrentStep {
	case types.StepSelectMode:
		m.modeIndex = moveCursorInList(m.modeIndex, -1, len(m.modeOptions))

	case types.StepSelectComputations:
		m.computationCursor = moveCursorInList(m.computationCursor, -1, len(m.computations))
	case types.StepConfigureOptions:
		m.categoryIndex = moveCursorInList(m.categoryIndex, -1, len(m.categories))
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
	case types.StepSelectSpatial:
		m.spatialCursor = moveCursorInList(m.spatialCursor, -1, len(spatialModes))
	case types.StepTimeRange:
		if m.editingRangeIdx >= 0 {
			m.editingField = moveCursorInList(m.editingField, -1, timeRangeFieldCount)
		} else if m.timeRangeSelectableCount() > 0 {
			m.timeRangeCursor = moveCursorInList(m.timeRangeCursor, -1, m.timeRangeSelectableCount())
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
			optCount := m.getAdvancedOptionCount()
			m.advancedCursor = moveCursorInList(m.advancedCursor, -1, optCount)
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
	case types.StepConfigureOptions:
		m.categoryIndex = moveCursorInList(m.categoryIndex, 1, len(m.categories))
	case types.StepSelectSubjects:
		m.subjectCursor = moveCursorInList(m.subjectCursor, 1, len(m.subjects))
	case types.StepSelectBands:
		m.bandCursor = moveCursorInList(m.bandCursor, 1, len(m.bands))
	case types.StepSelectROIs:
		m.roiCursor = moveCursorInList(m.roiCursor, 1, len(m.rois))
	case types.StepSelectFeatureFiles:
		applicable := m.GetApplicableFeatureFiles()
		m.featureFileCursor = moveCursorInList(m.featureFileCursor, 1, len(applicable))
	case types.StepSelectSpatial:
		m.spatialCursor = moveCursorInList(m.spatialCursor, 1, len(spatialModes))
	case types.StepTimeRange:
		if m.editingRangeIdx >= 0 {
			m.editingField = moveCursorInList(m.editingField, 1, timeRangeFieldCount)
		} else if m.timeRangeSelectableCount() > 0 {
			m.timeRangeCursor = moveCursorInList(m.timeRangeCursor, 1, m.timeRangeSelectableCount())
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
			optCount := m.getAdvancedOptionCount()
			m.advancedCursor = moveCursorInList(m.advancedCursor, 1, optCount)
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
	errors := m.validateCurrentStep()
	if len(errors) > 0 {
		m.validationErrors = errors
		return m, nil
	}
	m.validationErrors = nil // Clear if valid

	if m.stepIndex < len(m.steps)-1 {
		m.runStepExitHook(prevStep)
		m.stepIndex++
		m.CurrentStep = m.steps[m.stepIndex]

		for m.stepIndex < len(m.steps)-1 {
			if !m.shouldSkipStep(m.CurrentStep) {
				break
			}
			m.stepIndex++
			m.CurrentStep = m.steps[m.stepIndex]
		}

		m.runStepEnterHook(m.CurrentStep)
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

func (m *Model) handleSpace() {
	switch m.CurrentStep {
	case types.StepSelectComputations:
		m.computationSelected[m.computationCursor] = !m.computationSelected[m.computationCursor]
	case types.StepConfigureOptions:
		m.selected[m.categoryIndex] = !m.selected[m.categoryIndex]
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
		} else if m.timeRangeCursorOnRestToggle() {
			m.prepTaskIsRest = !m.prepTaskIsRest
			m.applyFeatureRestConstraints()
		} else {
			rangeIndex := m.selectedTimeRangeIndex()
			if rangeIndex < 0 || rangeIndex >= len(m.TimeRanges) {
				break
			}
			m.editingRangeIdx = rangeIndex
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
	case types.StepConfigureOptions:
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
	}
}

func (m *Model) selectNone() {
	switch m.CurrentStep {
	case types.StepSelectComputations:
		m.computationSelected = make(map[int]bool)
	case types.StepConfigureOptions:
		m.selected = make(map[int]bool)
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
	}
}

func (m *Model) GoBack() bool {
	// If in advanced config with an expanded option, collapse it first
	if m.CurrentStep == types.StepAdvancedConfig && m.expandedOption >= 0 {
		m.expandedOption = expandedNone
		m.subCursor = 0
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
	if m.Pipeline == types.PipelineFmriAnalysis {
		mode := ""
		if m.modeIndex >= 0 && m.modeIndex < len(m.modeOptions) {
			mode = m.modeOptions[m.modeIndex]
		}
		if mode == "second-level" {
			minRequired = minSubjectsForGroupCV
		}
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

		if invalid := invalidCorrelationTypes(m.correlationsTypesSpec); len(invalid) > 0 {
			errors = append(
				errors,
				fmt.Sprintf(
					"Invalid correlation type selection: %s",
					strings.Join(invalid, ", "),
				),
			)
		}
	}

	if m.Pipeline == types.PipelineFmriAnalysis {
		mode := ""
		if m.modeIndex >= 0 && m.modeIndex < len(m.modeOptions) {
			mode = m.modeOptions[m.modeIndex]
		}
		if mode == "second-level" {
			contrastNames := strings.Fields(strings.TrimSpace(m.fmriSecondLevelContrastNames))
			conditionLabels := strings.Fields(strings.TrimSpace(m.fmriSecondLevelConditionLabels))
			modelIndex := m.fmriSecondLevelModelIndex % 4
			model := []string{"one-sample", "two-sample", "paired", "repeated-measures"}[modelIndex]

			if len(contrastNames) == 0 {
				errors = append(errors, "fMRI second-level: at least one input contrast name is required")
			}
			if len(conditionLabels) > 0 && len(conditionLabels) != len(contrastNames) {
				errors = append(errors, "fMRI second-level: condition labels must match the number of input contrast names")
			}
			switch model {
			case "one-sample":
				if len(contrastNames) != 1 {
					errors = append(errors, "fMRI second-level: one-sample model requires exactly one input contrast")
				}
			case "two-sample":
				if len(contrastNames) != 1 {
					errors = append(errors, "fMRI second-level: two-sample model requires exactly one input contrast")
				}
				if strings.TrimSpace(m.fmriSecondLevelCovariatesFile) == "" {
					errors = append(errors, "fMRI second-level: two-sample model requires a covariates file")
				}
				if strings.TrimSpace(m.fmriSecondLevelGroupColumn) == "" {
					errors = append(errors, "fMRI second-level: two-sample model requires a group column")
				}
				if strings.TrimSpace(m.fmriSecondLevelGroupAValue) == "" || strings.TrimSpace(m.fmriSecondLevelGroupBValue) == "" {
					errors = append(errors, "fMRI second-level: two-sample model requires both group A and group B values")
				}
				if strings.TrimSpace(m.fmriSecondLevelGroupAValue) != "" &&
					strings.TrimSpace(m.fmriSecondLevelGroupAValue) == strings.TrimSpace(m.fmriSecondLevelGroupBValue) {
					errors = append(errors, "fMRI second-level: group A and group B values must be different")
				}
			case "paired":
				if len(contrastNames) != 2 {
					errors = append(errors, "fMRI second-level: paired model requires exactly two input contrasts ordered as A B")
				}
			case "repeated-measures":
				if len(contrastNames) < 2 {
					errors = append(errors, "fMRI second-level: repeated-measures model requires at least two input contrasts")
				}
				if strings.TrimSpace(m.fmriSecondLevelCovariateColumns) != "" {
					errors = append(errors, "fMRI second-level: repeated-measures model does not support subject-level covariates")
				}
			}
			if strings.TrimSpace(m.fmriSecondLevelSubjectColumn) == "" {
				errors = append(errors, "fMRI second-level: subject column must not be empty")
			}
			if m.fmriSecondLevelPermutationEnabled && m.fmriSecondLevelPermutationCount <= 0 {
				errors = append(errors, "fMRI second-level: permutation count must be > 0 when permutation inference is enabled")
			}
			return errors
		}
		if mode == "rest" {
			if strings.TrimSpace(m.bidsRestRoot) == "" {
				errors = append(errors, "fMRI resting-state: bids_rest_root is required")
			}
			if strings.TrimSpace(m.derivRestRoot) == "" {
				errors = append(errors, "fMRI resting-state: deriv_rest_root is required")
			}
			if strings.TrimSpace(m.fmriAnalysisAtlasLabelsImg) == "" {
				errors = append(errors, "fMRI resting-state: atlas labels image is required")
			}
			return errors
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

	if m.Pipeline == types.PipelineFmri && m.fmriTaskIsRest {
		if strings.TrimSpace(m.bidsRestRoot) == "" {
			errors = append(errors, "fMRI preprocessing: bids_rest_root is required when Rest Roots is enabled")
		}
		if strings.TrimSpace(m.derivRestRoot) == "" {
			errors = append(errors, "fMRI preprocessing: deriv_rest_root is required when Rest Roots is enabled")
		}
	}

	return errors
}

func (m *Model) validateTimeRanges() []string {
	var errors []string

	if len(m.TimeRanges) == 0 {
		if m.prepTaskIsRest {
			return nil
		}
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

	powerNeedsBaseline := m.isCategorySelected("power") && m.powerRequireBaseline && !m.prepTaskIsRest
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

// Advanced Configuration Helpers
///////////////////////////////////////////////////////////////////

// getAdvancedOptionCount returns the number of options for the current pipeline
func (m *Model) getAdvancedOptionCount() int {
	switch m.Pipeline {
	case types.PipelineFeatures:
		return len(m.getFeaturesOptions())
	case types.PipelineBehavior:
		return len(m.getBehaviorOptions())
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

// startNumberEdit enters editing mode for the current field
func (m *Model) startNumberEdit() {
	m.editingNumber = true
	m.numberBuffer = ""
}

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
