package wizard

import (
	"strings"

	"github.com/eeg-pipeline/tui/types"
)

type stepRenderer func(Model) string
type stepValidator func(*Model) []string
type stepHook func(*Model)

type stepDefinition struct {
	render   stepRenderer
	validate stepValidator
	onEnter  stepHook
	onExit   stepHook
}

var stepDefinitions = map[types.WizardStep]stepDefinition{
	types.StepSelectMode: {
		render: Model.renderModeSelection,
		onExit: (*Model).applyPostModeSelectionDefaults,
	},
	types.StepSelectComputations: {
		render:   Model.renderComputationSelection,
		validate: (*Model).validateComputationSelectionStep,
	},
	types.StepSelectFeatureFiles: {
		render:   Model.renderFeatureFileSelection,
		validate: (*Model).validateFeatureFileSelectionStep,
	},
	types.StepConfigureOptions: {
		render:   Model.renderCategorySelection,
		validate: (*Model).validateCategorySelectionStep,
		onEnter:  (*Model).prepareConfigureOptionsStep,
	},
	types.StepSelectBands: {
		render:   Model.renderBandSelection,
		validate: (*Model).validateBandSelectionStep,
	},
	types.StepSelectROIs: {
		render: Model.renderROISelection,
	},
	types.StepSelectSpatial: {
		render:   Model.renderSpatialSelection,
		validate: (*Model).validateSpatialSelectionStep,
	},
	types.StepTimeRange: {
		render:   Model.renderTimeRange,
		validate: (*Model).validateTimeRangeStep,
	},
	types.StepAdvancedConfig: {
		render: Model.renderAdvancedConfig,
	},
	types.StepSelectPlots: {
		render:   Model.renderPlotSelection,
		validate: (*Model).validatePlotSelectionStep,
	},
	types.StepSelectFeaturePlotters: {
		render:   Model.renderFeaturePlotterSelection,
		validate: (*Model).validateFeaturePlotterSelectionStep,
	},
	types.StepSelectPlotCategories: {
		render:   Model.renderCategorySelection,
		validate: (*Model).validateCategorySelectionStep,
	},
	types.StepPlotConfig: {
		render:   Model.renderPlotConfig,
		validate: (*Model).validatePlotConfigStep,
	},
	types.StepSelectSubjects: {
		render:   Model.renderSubjectSelection,
		validate: (*Model).validateSubjectSelectionStep,
	},
	types.StepSelectPreprocessingStages: {
		render:   Model.renderPreprocessingStageSelection,
		validate: (*Model).validatePreprocessingStageSelectionStep,
	},
	types.StepPreprocessingFiltering: {
		render: Model.renderPreprocessingFiltering,
	},
	types.StepPreprocessingICA: {
		render: Model.renderPreprocessingICA,
	},
	types.StepPreprocessingEpochs: {
		render: Model.renderPreprocessingEpochs,
	},
}

func (m Model) stepDefinition(step types.WizardStep) stepDefinition {
	if definition, ok := stepDefinitions[step]; ok {
		return definition
	}
	return stepDefinition{}
}

func (m *Model) runStepEnterHook(step types.WizardStep) {
	hook := m.stepDefinition(step).onEnter
	if hook == nil {
		return
	}
	hook(m)
}

func (m *Model) runStepExitHook(step types.WizardStep) {
	hook := m.stepDefinition(step).onExit
	if hook == nil {
		return
	}
	hook(m)
}

func (m *Model) prepareConfigureOptionsStep() {
	m.updateFeatureAvailability()
}

func (m *Model) applyPostModeSelectionDefaults() {
	if m.Pipeline != types.PipelineFmriAnalysis {
		return
	}

	mode := ""
	if m.modeIndex >= 0 && m.modeIndex < len(m.modeOptions) {
		mode = m.modeOptions[m.modeIndex]
	}
	if mode != "trial-signatures" {
		return
	}

	space := strings.TrimSpace(m.fmriAnalysisFmriprepSpace)
	if space == "" || strings.EqualFold(space, "T1w") {
		m.fmriAnalysisFmriprepSpace = "MNI152NLin2009cAsym"
	}
}
