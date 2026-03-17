package wizard

import "github.com/eeg-pipeline/tui/types"

func (m Model) featureRestModeEnabled() bool {
	return m.Pipeline == types.PipelineFeatures && m.prepTaskIsRest
}

func (m Model) restIncompatibleFeatureCategory(category string) bool {
	if !m.featureRestModeEnabled() {
		return false
	}
	switch category {
	case "erp", "erds", "itpc":
		return true
	default:
		return false
	}
}

func (m *Model) applyFeatureRestConstraints() {
	if m.Pipeline != types.PipelineFeatures || !m.prepTaskIsRest {
		return
	}

	for idx, category := range m.categories {
		if m.restIncompatibleFeatureCategory(category) {
			m.selected[idx] = false
		}
	}

	m.powerRequireBaseline = false
	m.powerSubtractEvoked = false
	m.powerMinTrialsPerCondition = 2

	m.burstThresholdReference = 1
	m.burstMinTrialsPerCondition = 10

	m.connGranularity = 2
	m.connConditionColumn = ""
	m.connConditionValues = ""
	m.connForceWithinEpochML = false

	m.sourceLocContrastEnabled = false
	m.sourceLocContrastCondition = ""
	m.sourceLocContrastA = ""
	m.sourceLocContrastB = ""
	m.sourceLocContrastMinTrials = 5

	m.featAnalysisMode = 0
}
