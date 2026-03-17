package wizard

import (
	"strings"

	"github.com/eeg-pipeline/tui/messages"
)

// Wizard text-editing state transitions and summary injection.

func (m *Model) SetConfigSummary(summary messages.ConfigSummary) {
	if (m.task == "" || m.task == "task") && summary.Task != "" {
		m.task = summary.Task
	}
	if m.bidsRoot == "" && summary.BidsRoot != "" {
		m.bidsRoot = summary.BidsRoot
	}
	if m.bidsRestRoot == "" && summary.BidsRestRoot != "" {
		m.bidsRestRoot = summary.BidsRestRoot
	}
	if m.bidsFmriRoot == "" && summary.BidsFmriRoot != "" {
		m.bidsFmriRoot = summary.BidsFmriRoot
	}
	if m.derivRoot == "" && summary.DerivRoot != "" {
		m.derivRoot = summary.DerivRoot
		m.fmriSecondLevelContrastDiscoveryKey = ""
	}
	if m.derivRestRoot == "" && summary.DerivRestRoot != "" {
		m.derivRestRoot = summary.DerivRestRoot
	}
	if m.sourceRoot == "" && summary.SourceRoot != "" {
		m.sourceRoot = summary.SourceRoot
	}
	if summary.PreprocessingNJobs > 0 {
		m.prepNJobs = summary.PreprocessingNJobs
	}
}

func (m *Model) SetTask(task string) {
	task = strings.TrimSpace(task)
	if task == "" {
		return
	}
	m.task = task
	m.fmriSecondLevelContrastDiscoveryKey = ""
}

func (m *Model) SetRepoRoot(repoRoot string) {
	m.repoRoot = repoRoot
}

func (m *Model) startTextEdit(field textField) {
	m.editingTextField = field
	m.textBuffer = m.getTextFieldValue(field)
	m.editingPlotID = ""
	m.editingPlotField = plotItemConfigFieldNone
	m.editingText = true
}

func (m *Model) commitTextInput() {
	if m.editingPlotID != "" && m.editingPlotField != plotItemConfigFieldNone {
		m.setPlotItemTextFieldValue(m.editingPlotID, m.editingPlotField, m.textBuffer)
		return
	}
	m.setTextFieldValue(m.editingTextField, m.textBuffer)
}

func (m *Model) startPlotTextEdit(plotID string, field plotItemConfigField) {
	m.editingTextField = textFieldNone
	m.editingPlotID = plotID
	m.editingPlotField = field
	m.textBuffer = m.getPlotItemTextFieldValue(plotID, field)
	m.editingText = true
}
