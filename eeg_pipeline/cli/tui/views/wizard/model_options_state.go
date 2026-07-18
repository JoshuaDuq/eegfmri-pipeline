package wizard

import "github.com/eeg-pipeline/tui/types"

func (m Model) isCurrentlyEditing(opt optionType) bool {
	if !m.editingNumber {
		return false
	}

	var options []optionType
	switch m.Pipeline {
	case types.PipelineFeatures:
		options = m.getFeaturesOptions()
	case types.PipelineBehavior:
		options = m.getBehaviorOptions()
	case types.PipelineML:
		options = m.getMLOptions()
	case types.PipelinePreprocessing:
		options = m.getPreprocessingOptions()
	case types.PipelineFmri:
		options = m.getFmriPreprocessingOptions()
	case types.PipelineFmriAnalysis:
		options = m.getFmriAnalysisOptions()
	default:
		return false
	}
	if m.advancedCursor < 0 || m.advancedCursor >= len(options) {
		return false
	}
	return options[m.advancedCursor] == opt
}
